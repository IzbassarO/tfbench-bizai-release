"""Task 1 third member: 2D U-Net on the time x link grid (see report, Sec. III-A).

Producer for the all-in queue (`tfb.t1.unet:run`, src/tfb/allin/README.md):
  gate : 3 seeds x {last30, autumn}; blend (1-b)/2 LGBM + (1-b)/2 MLP random_12 + b U-Net -> pp2.
         b is fitted on one holdout and scored on the other, vs the allin reference (tfb.allin.baseline).
         tfb.t1.unet_seeds reuses this with more seeds and another b grid.
  final: 3 seeds on all train days (+ validation/private observed cells as transductive
         re-mask targets) -> state_raw.csv with the gate's b.
Inputs are only the released masked layer, train-truth profiles (holdout days excluded)
and eligible ramps. Weights, train seconds (device) and CPU inference seconds from the
saved weights are kept in <checkpoint>/runs/<hold>/seed<S>/.

Mac smoke (one small panel, minutes, <2 GB):
  PYTHONPATH=src .venv/bin/python -m tfb.t1.unet
"""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import gc
import json
from functools import lru_cache
import platform
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from tfb import cache, io
from tfb.t1.base import day_mask, fit_profiles, slot_of, write_state
from tfb.t1.train import SPEED_CAP, holdout_days
from tfb.t2b import data as T2
from tfb.t1.mlp_colab.runner import atomic_json, atomic_torch, seed_all

DEFAULTS = dict(epochs=30, batch=8, width=32, lr=2e-3, weight_decay=1e-4, panels=None, steps=None)
WIN, CTX = 432, 72                  # 1.5-day window; predictions keep the central day
S_S, S_F = 8.0, 100.0               # residual scales: speed km/h, flow vph per lane
B_GRID = (0.15, 0.25, 0.33)         # U-Net blend weight grid (report 08 §3)
STEP = io.STEPS_PER_DAY


# ---------------------------------------------------------------- data
def load_panel(panel, split, pn_train, exclude_days):
    """Arrays in downstream link order. ys/yf = train truth on scored target cells outside the excluded days."""
    d = T2.load(panel, split); T2.load.cache_clear()
    pn = pn_train if split == "train" else cache.load(panel, split)
    o = T2.downstream_order(panel, pn.links)
    ps, pf = fit_profiles(pn_train, exclude_days)
    out = dict(panel=panel, order=o, slot=slot_of(pn), prof_s=ps[:, o].astype(np.float32), prof_f=pf[:, o].astype(np.float32),
               ms=d["mspeed"], mf=d["mflow"], mo=pn.mocc[:, o], mpct=pn.mpct[:, o], melig=d["melig"],
               ron=d["ron"], roff=d["roff"], target=pn.target[:, o],
               **{k: d[k].astype(np.float32) for k in ("v_f", "lanes", "cap", "length")})
    if split == "train":
        ok = d["elig"] & np.isfinite(d["speed"]) & np.isfinite(d["flow"]) & out["target"]
        if exclude_days is not None: ok &= ~day_mask(pn, exclude_days)[:, None]
        out["ys"] = np.where(ok, d["speed"], np.nan).astype(np.float32)
        out["yf"] = np.where(ok, d["flow"], np.nan).astype(np.float32)
    return out


def fill_time(x, v):
    """Linear interpolation along axis 1 over valid cells (hold at the ends, 0 where a column has none)."""
    T = x.shape[1]; idx = np.broadcast_to(np.arange(T)[None, :, None], x.shape)
    prev = np.maximum.accumulate(np.where(v, idx, -1), axis=1)
    nxt = np.minimum.accumulate(np.where(v, idx, T)[:, ::-1], axis=1)[:, ::-1]
    xp = np.take_along_axis(x, prev.clip(0), 1); xn = np.take_along_axis(x, nxt.clip(None, T - 1), 1)
    hp, hn = prev >= 0, nxt < T
    w = (idx - prev) / np.maximum(nxt - prev, 1)
    return np.where(hp & hn, xp + (xn - xp) * w, np.where(hp, xp, np.where(hn, xn, 0))).astype(np.float32)


def remask(rng, obs, B, L):
    """Extra hidden cells like the release: share 0.2/0.3/0.5, half random cells, half time x link blocks,
    plus a Task 2-style 18-step full-corridor blackout on 30% of windows."""
    p = rng.choice([0.2, 0.3, 0.5], size=(B, 1, 1)); W = obs.shape[1]
    blocks = rng.random((B, W // 12 + 1, L // 4 + 1)) < p / 2
    hide = (rng.random((B, W, L)) < p / 2) | np.repeat(np.repeat(blocks, 12, 1), 4, 2)[:, :W, :L]
    for b in np.flatnonzero(rng.random(B) < 0.3):
        t = rng.integers(0, W - 18); hide[b, t:t + 18] = True
    return hide & obs


def batch(d, starts, rng=None, win=WIN):
    """Input [B,C,win,Lp], base/labels/loss mask [B,win,L]. rng=None: no augmentation (inference)."""
    rows = np.asarray(starts)[:, None] + np.arange(win)[None, :]
    g = lambda k: d[k][rows]
    ms, mf, mo = g("ms"), g("mf"), g("mo")
    B, _, L = ms.shape
    obs = np.isfinite(ms) & np.isfinite(mf)
    hide = remask(rng, obs & g("melig"), B, L) if rng is not None else np.zeros_like(obs)
    vis = obs & ~hide
    slot = d["slot"][rows]; ps, pf = d["prof_s"][slot], d["prof_f"][slot]
    base_s = ps + fill_time(np.where(vis, ms - ps, 0), vis)
    base_f = pf + fill_time(np.where(vis, mf - pf, 0), vis)
    vf, cap = d["v_f"], d["cap"]
    tod = 2 * np.pi * (slot % STEP) / STEP
    bc = lambda a: np.broadcast_to(a, (B, win, L))
    ch = [np.where(vis, ms, 0) / vf, np.where(vis, mf, 0) / cap, np.where(vis, np.nan_to_num(mo), 0), vis,
          np.where(vis, g("mpct"), 0) / 100, base_s / vf, base_f / cap, ps / vf, pf / cap]
    for k in ("ron", "roff"):
        r = g(k); ch += [np.nan_to_num(r) / cap, np.isfinite(r)]
    ch += [bc(vf / 100), bc(d["lanes"] / 6), bc(cap / d["lanes"] / 2000), bc(d["length"]),
           bc(np.sin(tod)[..., None]), bc(np.cos(tod)[..., None]), bc((slot // STEP >= 5)[..., None]), bc(np.ones(L))]
    Lp = -(-L // 8) * 8
    x = np.zeros((B, len(ch), win, Lp), np.float32)
    for i, c in enumerate(ch): x[:, i, :, :L] = c
    out = dict(x=x, base_s=base_s, base_f=base_f)
    if rng is not None:
        ys, yf = (g("ys"), g("yf")) if "ys" in d else (np.full_like(ms, np.nan),) * 2   # validation/private: re-mask only
        truth = np.isfinite(ys)
        ys = np.where(truth, ys, ms); yf = np.where(truth, yf, mf)        # re-masked cells: released = truth
        out["m"] = truth | hide
        out["y"] = np.stack([np.where(out["m"], (ys - base_s) / S_S, 0),
                             np.where(out["m"], (yf - base_f) / d["lanes"] / S_F, 0)], 1).astype(np.float32)
    return out


# ---------------------------------------------------------------- model
def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
                         nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU())


class UNet(nn.Module):
    """4 levels (w, 2w, 4w, 8w); strided-conv down, transposed-conv up (deterministic on CUDA)."""
    def __init__(self, c_in, w=32):
        super().__init__()
        ws = [w, 2 * w, 4 * w, 8 * w]
        self.inc = block(c_in, w)
        self.down = nn.ModuleList(nn.Sequential(nn.Conv2d(ws[i], ws[i + 1], 3, 2, 1), block(ws[i + 1], ws[i + 1])) for i in range(3))
        self.up = nn.ModuleList(nn.ConvTranspose2d(ws[i + 1], ws[i], 2, 2) for i in range(3))
        self.dec = nn.ModuleList(block(2 * ws[i], ws[i]) for i in range(3))
        self.head = nn.Conv2d(w, 2, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)   # starts at the interpolated base

    def forward(self, x):
        skips = [self.inc(x)]
        for d in self.down: skips.append(d(skips[-1]))
        h = skips.pop()
        for i in reversed(range(3)): h = self.dec[i](torch.cat([self.up[i](h), skips.pop()], 1))
        return self.head(h)


C_IN = 21


# ---------------------------------------------------------------- train / predict
def fit(panels, cfg, seed, device, directory, beat):
    """Fixed-epoch OneCycle fit with per-epoch resume. Returns (net, record)."""
    directory.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    net = UNet(C_IN, cfg["width"]).to(device)
    days = np.array([len(d["ms"]) // STEP for d in panels])
    steps = cfg["steps"] or int(days.sum()) // cfg["batch"]
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * steps)
    state = dict(epoch=0, history=[], train_seconds=0.0); win = cfg.get("win", WIN)
    resume = directory / "resume.pt"
    if resume.exists():
        z = torch.load(resume, map_location=device, weights_only=False)
        net.load_state_dict(z["net"]); opt.load_state_dict(z["opt"]); sch.load_state_dict(z["sch"]); state = z["state"]
    for epoch in range(state["epoch"], cfg["epochs"]):
        rng = np.random.default_rng([seed, epoch])          # window order/masks reproducible per epoch on resume
        net.train(); t0 = time.perf_counter(); total = 0.0
        for _ in range(steps):
            d = panels[rng.choice(len(panels), p=days / days.sum())]
            b = batch(d, rng.integers(0, len(d["ms"]) - win + 1, cfg["batch"]), rng, win)
            L = b["m"].shape[-1]
            y = torch.from_numpy(b["y"]).to(device); m = torch.from_numpy(b["m"]).to(device)[:, None].float()
            pred = net(torch.from_numpy(b["x"]).to(device))[..., :L]
            loss = ((pred - y) ** 2 * m).sum() / (2 * m.sum()).clamp(min=1)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
            total += loss.item()
        if not np.isfinite(total): raise FloatingPointError("U-Net nonfinite loss")
        if device.type == "cuda": torch.cuda.synchronize()
        state["train_seconds"] += time.perf_counter() - t0
        state["history"].append(dict(epoch=epoch + 1, loss=total / steps, seconds=time.perf_counter() - t0))
        state["epoch"] = epoch + 1
        atomic_torch(resume, dict(net=net.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(), state=state))
        print(f"unet seed{seed} epoch {epoch + 1}/{cfg['epochs']} loss {total / steps:.4f} ({state['history'][-1]['seconds']:.0f}s)", flush=True)
        beat(dict(unet=str(directory), **state["history"][-1]))   # may pause the worker; checkpoint is durable
    return net, state


@torch.no_grad()
def predict(net, d, days, device, bs=8, win=WIN, ctx=CTX):
    """Speed/flow [len(days)*288, L] in the cache's original link order, for whole days of panel d."""
    net.eval(); T = len(d["ms"]); out_s, out_f = [], []
    for i in range(0, len(days), bs):
        dd = np.asarray(days[i:i + bs]); starts = np.clip(dd * STEP - ctx, 0, T - win)
        b = batch(d, starts, win=win); L = b["base_s"].shape[-1]
        r = net(torch.from_numpy(b["x"]).to(device))[..., :L].float().cpu().numpy()
        sp = np.clip(b["base_s"] + r[:, 0] * S_S, 3.0, SPEED_CAP)
        fl = np.maximum(b["base_f"] + r[:, 1] * S_F * d["lanes"], 100.0)
        k = (dd * STEP - starts)[:, None] + np.arange(STEP)[None, :]
        out_s += list(np.take_along_axis(sp, k[..., None], 1)); out_f += list(np.take_along_axis(fl, k[..., None], 1))
    inv = np.argsort(d["order"])
    sp, fl = np.concatenate(out_s)[:, inv], np.concatenate(out_f)[:, inv]
    if not (np.isfinite(sp).all() and np.isfinite(fl).all()): raise FloatingPointError(d["panel"])
    return sp.astype(np.float32), fl.astype(np.float32)


@lru_cache(maxsize=1)   # the three seeds of one holdout share the loaded arrays
def load_all(hold, names):
    days = None if hold == "final" else holdout_days(hold)
    panels, jobs = [], []
    for panel in names:
        pn = cache.load(panel, "train")
        panels.append(load_panel(panel, "train", pn, days))
        if hold == "final":
            for split in ("validation", "private"):
                panels.append(load_panel(panel, split, pn, None)); jobs.append((len(panels) - 1, split))
        else: jobs.append((len(panels) - 1, "train"))
        del pn; gc.collect()
    return panels, jobs


def train_run(hold, cfg, seed, device, root, beat):
    """One seed on one holdout ('final' = all train days + validation/private transductive). Predictions per
    (panel, split) as full [T,L] arrays in <root>/pred/; weights + timings in <root>."""
    done = root / "done.json"
    if done.exists(): return json.loads(done.read_text())
    win, ctx = cfg.get("win", WIN), cfg.get("ctx", CTX)   # optional cfg keys: window length / leading context (steps)
    if win % 8 or not 0 <= ctx <= win - STEP: raise ValueError(f"U-Net window {win}/{ctx}: win % 8 == 0, ctx + 288 <= win")
    days = None if hold == "final" else holdout_days(hold)
    panels, jobs = load_all(hold, tuple(cfg["panels"] or io.panels()))
    net, state = fit(panels, cfg, seed, device, root, beat)
    torch.save(dict(net=net.state_dict(), cfg=cfg, c_in=C_IN, seed=seed, hold=hold), root / "weights.pt")
    pred_days = lambda d: np.arange(len(d["ms"]) // STEP) if days is None else days
    (root / "pred").mkdir(exist_ok=True)
    t0 = time.perf_counter()
    for i, split in jobs:
        sp, fl = predict(net, panels[i], pred_days(panels[i]), device, win=win, ctx=ctx)
        np.savez(root / "pred" / f"{panels[i]['panel']}_{split}.npz", sp=sp, fl=fl)
    infer = time.perf_counter() - t0
    record = dict(hold=hold, seed=seed, cfg=cfg, epochs=state["epoch"], history=state["history"],
                  train_seconds=state["train_seconds"], train_device=device_name(device), infer_seconds_device=infer,
                  torch=torch.__version__, weights=str(root / "weights.pt"))
    if seed == 42:   # inference from the saved weights on CPU (timing for the report) + agreement with the device run
        cpu = UNet(C_IN, cfg["width"]); cpu.load_state_dict(torch.load(root / "weights.pt", map_location="cpu")["net"])
        t0 = time.perf_counter(); diff = 0.0
        for i, split in jobs:
            sp, fl = predict(cpu, panels[i], pred_days(panels[i]), torch.device("cpu"), win=win, ctx=ctx)
            with np.load(root / "pred" / f"{panels[i]['panel']}_{split}.npz") as z:
                diff = max(diff, float(np.abs(z["sp"] - sp).max()), float(np.abs(z["fl"] - fl).max()))
        record.update(infer_seconds_cpu=time.perf_counter() - t0, cpu=platform.processor() or platform.machine(),
                      cpu_threads=torch.get_num_threads(), cpu_vs_device_maxabs=diff)
    atomic_json(done, record)
    (root / "resume.pt").unlink(missing_ok=True)
    del net; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    return record


def device_name(device):
    return torch.cuda.get_device_name() if device.type == "cuda" else platform.processor() or platform.machine()


# ---------------------------------------------------------------- gate / final (allin producer)
def holdout_dirs(bundle, hold, run_roots, dest):
    """Seed-mean U-Net at the bundle's target cells, in the scorer's npz format (row/col/sp/fl)."""
    manifest = json.loads((bundle / "manifest.json").read_text())
    dest.mkdir(parents=True, exist_ok=True)
    for panel in manifest["holdouts"][hold]["panels"]:
        with np.load(bundle / hold / "score" / f"{panel}.npz") as c:
            rows, tr, tc = c["rows"], c["target_row"], c["target_col"]
        sp = fl = 0
        for r in run_roots:
            with np.load(r / "pred" / f"{panel}_train.npz") as z:
                sp = sp + z["sp"][tr, tc] / len(run_roots); fl = fl + z["fl"][tr, tc] / len(run_roots)
        np.savez_compressed(dest / f"{panel}.npz", row=rows[tr], col=tc, sp=sp.astype(np.float32), fl=fl.astype(np.float32))
    return dest


def mix(lgb, mlp, unet, b):
    """Scorer input P with 0.5 lgb + 0.5 P = (1-b)/2 lgb + (1-b)/2 mlp + b unet (scoring.evaluate fixes 0.5 LGBM)."""
    return ((1 - b) * mlp + 2 * b * unet - b * lgb).astype(np.float32)


def blend_dir(bundle, hold, mlp_dir, unet_dir, b, dest):
    manifest = json.loads((bundle / "manifest.json").read_text())
    dest.mkdir(parents=True, exist_ok=True)
    for panel in manifest["holdouts"][hold]["panels"]:
        with np.load(bundle / hold / "score" / f"{panel}.npz") as c, np.load(mlp_dir / f"{panel}.npz") as m, \
             np.load(unet_dir / f"{panel}.npz") as u:
            tr, tc = c["target_row"], c["target_col"]
            np.savez_compressed(dest / f"{panel}.npz", row=m["row"], col=m["col"],
                                sp=mix(c["lgb_sp"][tr, tc], m["sp"], u["sp"], b), fl=mix(c["lgb_fl"][tr, tc], m["fl"], u["fl"], b))
    return dest


def final_component(cfg, device, runs, component, beat=lambda *a, **k: None):
    """Train (or reuse) each (seed, dir) on all train days + validation/private re-mask targets; the seed mean of the
    validation/private predictions -> component csv (template order). Also scripts/reproduce.sh stage 7."""
    records = {str(s): train_run("final", cfg, s, device, r, beat) for s, r in runs}
    preds = {}
    for panel in cfg["panels"] or io.panels():
        for split in ("validation", "private"):
            arrays = [dict(np.load(r / "pred" / f"{panel}_{split}.npz")) for _, r in runs]
            preds[panel, split] = (np.mean([z["sp"] for z in arrays], 0), np.mean([z["fl"] for z in arrays], 0))
    write_state(preds, component, meta=dict(model="U-Net", seeds=[s for s, _ in runs], cfg=cfg))
    return records


# ---------------------------------------------------------------- Mac smoke
def smoke():
    """One small panel, last30, a few steps: shapes, finiteness, resume, CPU timing, blend algebra."""
    import shutil
    from tfb.eval.score_t1 import score_arrays
    rng = np.random.default_rng(0); lgb, mlp, un = rng.random((3, 5)) * 100
    for b in B_GRID:
        assert np.allclose(0.5 * lgb + 0.5 * mix(lgb, mlp, un, b), (1 - b) / 2 * lgb + (1 - b) / 2 * mlp + b * un, atol=1e-3)
    root = io.PROJECT / "outputs/_scratch/t1_unet"; shutil.rmtree(root, ignore_errors=True)
    cfg = dict(DEFAULTS, panels=["D12_I405_S"], epochs=2, steps=6, batch=4, width=16)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rec = train_run("last30", cfg, 42, dev, root / "s42", lambda *a, **k: None)
    pn = cache.load("D12_I405_S", "train"); days = holdout_days("last30"); rows = day_mask(pn, days)
    with np.load(root / "s42" / "pred" / "D12_I405_S_train.npz") as z:
        sp, fl = pn.speed.copy(), pn.flow.copy(); sp[rows], fl[rows] = z["sp"], z["fl"]
    S = score_arrays(pn, sp, fl, rows[:, None]); print(S)
    assert np.isfinite(S.S_state).all() and rec["cpu_vs_device_maxabs"] < 1e-2
    print({k: rec[k] for k in ("train_seconds", "infer_seconds_device", "infer_seconds_cpu", "cpu_vs_device_maxabs")})
    shutil.rmtree(root)


if __name__ == "__main__":
    smoke()
