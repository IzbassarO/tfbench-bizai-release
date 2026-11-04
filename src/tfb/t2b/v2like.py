"""Ongoing gap (see report, Sec. III-C): what family A v2 has that the family B noloc ongoing member lacks, ported location-free.

On the 292 harness windows where v2 exists the family B noloc member loses 0.029 IoU, almost all of it on growing queues
(+0.061 on 109 windows) and on 1-2-link queues at T (+0.106): it predicts too few cells (FN 1050 vs 662). family A's
deployed ongoing model (0.5 full + 0.5 no_location, 230 columns) takes 55 % of its gain from `extra_slow80 / slow70`
(distance to the nearest link at <= 0.8 / 0.7 v_f: the slowdown that precedes the 0.6 v_f queue), 18 % from the
full-day masked prefix (`day_*`) and only 4 % from location priors. Ported here, measured on the Q0 harness.

Factors (one at a time, then combined; each refit of the ongoing noloc member per cut, 3 seeds):
  slow  FT.V2L["slow"]: slow70 / slow80 count, nearest signed / absolute km, upstream / downstream km, corridor min and
        lower-quartile speed ratio, min ratio over 3 links up / down (from the forward-filled speed ratio at T)
  day   FT.V2L["day"]: day context from the masked layer of the window's UTC day before T-60 plus the window history:
        queue run minutes, cumulative queued minutes, minutes since the first queue (link), corridor queued count, its
        peak, minutes since the peak, change since the peak, change over 30 min, minutes since the first corridor queue,
        known fraction (released validation / private: the masked prefix is 60 % finite, as in train)
  xtr   training on the cache's 400 extra ongoing windows per panel too (recipe: scanned windows only)
  hw    training windows = the harness's own admissible windows of the fit months (official selector) + the cache
  raw   fit on raw labels Y instead of the denoised YD
  loc   location priors kept (attribution only, never a candidate)

  PYTHONPATH=src python -m tfb.t2b.v2like [--variants ...]   -> outputs/t2b/harness/gap_<variant>.parquet, gap.csv
"""
from __future__ import annotations
import argparse, pickle, time
import numpy as np
import pandas as pd
from .. import io
from . import cv, data, evaluate as ev, features as FT, harness as HS, hybrid as H, model as M
from .q2feat import Q2_BALLAST

OG = HS.OG
VARIANTS = {"base": {}, "slow": dict(slow=True), "day": dict(day=True), "xtr": dict(extra=True), "hw": dict(cache="cache_hw"),
            "raw": dict(raw=True), "loc": dict(drop=()),
            "slow_day": dict(slow=True, day=True), "slow_xtr": dict(slow=True, extra=True), "slow_hw": dict(slow=True, cache="cache_hw"),
            "slow_day_hw": dict(slow=True, day=True, cache="cache_hw"), "slow_day_xtr": dict(slow=True, day=True, extra=True),
            "slow_day_hw_raw": dict(slow=True, day=True, cache="cache_hw", raw=True),
            # see report, Sec. III-C: v2l = slow_hw refit with its probabilities kept (blend weights 0.4 / 0.5 / 0.6 vs E25)
            "v2l": dict(slow=True, cache="cache_hw"), "v2l_s5": dict(slow=True, cache="cache_hw", seeds=(42, 7, 2031, 11, 23)),
            "v2l_q2t": dict(slow=True, cache="cache_hw", q2=True), "v2l_alive": dict(slow=True, cache="cache_hw", alive=True),
            "v2l_q2t_alive": dict(slow=True, cache="cache_hw", q2=True, alive=True),
            "v2l_s5_q2t_alive": dict(slow=True, cache="cache_hw", q2=True, alive=True, seeds=(42, 7, 2031, 11, 23))}
PLUS = ("v2l", "v2l_s5", "v2l_q2t", "v2l_alive")
BLENDS = {"E25": 0.75, "E40": 0.4, "E50": 0.5, "E60": 0.6}   # weight of v2 in decode(w v2 + (1 - w) member)


def log(s):
    print(f"[v2like {time.strftime('%H:%M:%S')}] {s}", flush=True)


def _prev_next(mask: np.ndarray):
    """[n, L] bool -> index of the nearest True at or before / at or after each position (-1 / L when none)."""
    n, L = mask.shape; j = np.arange(L)[None]
    prev = np.maximum.accumulate(np.where(mask, j, -1), axis=1)
    nxt = np.minimum.accumulate(np.where(mask, j, L)[:, ::-1], axis=1)[:, ::-1]
    return prev, nxt


def slow_features(st: dict, r: np.ndarray) -> dict:
    """r: [n, 13, L] forward-filled speed ratio (NaN = never observed in the history). family A extras, location-free."""
    x = st["x"]; rT = r[:, 12]; n, L = rT.shape; f = {}
    with np.errstate(all="ignore"):
        for thr, nm in ((0.7, "70"), (0.8, "80")):
            slow = np.isfinite(rT) & (rT <= thr)
            prev, nxt = _prev_next(slow)
            up = np.where(prev >= 0, x[None] - x[np.maximum(prev, 0)], np.nan)
            down = np.where(nxt < L, x[np.minimum(nxt, L - 1)] - x[None], np.nan)
            f[f"slow{nm}_count"] = np.broadcast_to(slow.sum(1)[:, None], (n, L)).astype(np.float32)
            f[f"slow{nm}_up_km"] = up; f[f"slow{nm}_down_km"] = down
            f[f"slow{nm}_abs_km"] = np.fmin(up, down)
            f[f"slow{nm}_signed_km"] = np.where(np.isnan(up) | (np.nan_to_num(down, nan=np.inf) < np.nan_to_num(up, nan=np.inf)), down, -up)
        f["cor_min_r"] = np.broadcast_to(np.nanmin(rT, 1)[:, None], (n, L))
        f["cor_q25_r"] = np.broadcast_to(np.nanquantile(rT, 0.25, axis=1)[:, None], (n, L))
        for side, offs in (("up", (1, 2, 3)), ("down", (-1, -2, -3))):
            a = np.stack([np.roll(rT, o, 1) for o in offs], 0)
            for o, arr in zip(offs, a):
                if o > 0: arr[:, :o] = np.nan
                else: arr[:, o:] = np.nan
            f[f"{side}_min_r3"] = np.nanmin(a, 0)
    return f


def day_context(meta: pd.DataFrame, S: np.ndarray, net: dict) -> np.ndarray:
    """[n, L, len(FT.DAY_COLS)] from the masked layer of the window's UTC day before T-60 plus the window history S (<= T)."""
    panel = str(meta.panel.iloc[0]); split = "train"
    if "window_id" in meta:
        split = "validation" if "_validation_" in str(meta.window_id.iloc[0]) else "private"
    ms = data.load(panel, split)["mspeed"]; vcut = net["vcut"]; ts = meta.t.to_numpy()
    n, L = len(ts), S.shape[-1]; out = np.full((n, L, len(FT.DAY_COLS)), np.nan, np.float32)
    for i, t in enumerate(ts):
        k = int(t % 288)
        full = np.concatenate([ms[t - k:t - 12], S[i]]) if k >= 12 else S[i, 12 - k:]
        m = len(full); slot = np.arange(m)[:, None]
        obs = np.isfinite(full); last = np.maximum.accumulate(np.where(obs, slot, -1), axis=0); known = last >= 0
        filled = full[np.maximum(last, 0), np.arange(L)[None]]
        queued = known & (np.nan_to_num(filled, nan=999) <= vcut)
        last_free = np.maximum.accumulate(np.where(~queued, slot, -1), axis=0)
        run = np.where(queued[-1], (m - 1 - last_free[-1]) * 5.0, 0.0)
        anyq = queued.any(0); first = queued.argmax(0)
        cnt = np.where(known.any(1), queued.sum(1).astype(float), np.nan)
        col = [np.where(known[-1], run, np.nan), np.where(known[-1], queued.sum(0) * 5.0, np.nan),
               np.where(anyq, (m - 1 - first) * 5.0, np.nan)]
        if np.isfinite(cnt[-1]):
            cur = cnt[-1]; peak = np.nanmax(cnt); ps = np.flatnonzero(cnt == peak)[-1]; fc = np.flatnonzero(cnt > 0)
            g = [cur, peak, (m - 1 - ps) * 5.0, cur - peak, cur - cnt[max(0, m - 7)], (m - 1 - fc[0]) * 5.0 if fc.size else np.nan]
        else:
            g = [np.nan] * 6
        col += [np.full(L, v) for v in g] + [np.full(L, known[-1].mean())]
        out[i] = np.stack(col, -1)
    return out


def build_cache_hw(log=log, windows=None, name="cache_hw"):
    """outputs/t2b/<name>: the recipe cache's windows plus the admissible ongoing windows of <windows>/windows_<p>.parquet
    (default the harness's, HS.OUT) (dedup by t)."""
    windows = windows or HS.OUT; dst = io.OUTPUTS / "t2b" / name; dst.mkdir(parents=True, exist_ok=True)
    for p in io.T2_PANELS:
        if (dst / f"{p}_train.npz").exists():
            continue
        old = ev.load_train(p); W = pd.read_parquet(windows / f"windows_{p}.parquet"); W = W[W.condition == OG]
        W = W[~W.t.isin(set(old["meta"].t))]; B = HS.bundle(p, W)
        meta = pd.concat([old["meta"], B["meta"][["t", "condition", "coverage", "pers_iou", "panel", "date", "month", "block_day"]]], ignore_index=True)
        arrs = {k: np.concatenate([old[k], B[k]]) for k in ("S", "Q", "RON", "ROFF", "Y", "YD", "E")}
        o = np.argsort(meta.t.to_numpy(), kind="stable"); meta = meta.iloc[o].reset_index(drop=True)
        meta.to_parquet(dst / f"{p}_train_meta.parquet"); np.savez_compressed(dst / f"{p}_train.npz", **{k: v[o] for k, v in arrs.items()})
        log(f"cache_hw {p}: {len(old['meta'])} recipe + {len(W)} harness ongoing windows")


def run(variants: dict, probs: dict, Ws: dict, Bs: dict) -> pd.DataFrame:
    rows = []
    for name, cfg in variants.items():
        f = HS.OUT / f"gap_{name}.parquet"
        if f.exists():
            rows.append(pd.read_parquet(f)); log(f"{name}: cached"); continue
        FT.V2L["slow"] = cfg.get("slow", False); FT.V2L["day"] = cfg.get("day", False); cv.LAB["raw"] = cfg.get("raw", False)
        FT.Q2["on"] = cfg.get("q2", False); FT.ALIVE["on"] = cfg.get("alive", False)
        drop = cfg.get("drop", tuple(M.LOC) + (Q2_BALLAST if cfg.get("q2") else ()))
        part = []; P = {}
        try:
            for cut, (fit_m, ev_m) in HS.CUTS.items():
                ms = H.noloc_fit(fit_m, cfg.get("cache", "cache"), log=log, cond=OG, extra=cfg.get("extra", False), drop=drop, seeds=cfg.get("seeds", H.SEEDS))
                for p in io.T2_PANELS:
                    B = Bs[p]; d = probs[(cut, p, OG)]; mk = d["mk"]; W = Ws[p]
                    G, wpos = HS.noloc_grid(ms, p, B, OG, cut); assert (wpos == mk).all(); P[(cut, p)] = G.astype(np.float32)
                    Y, E = B["Y"][mk], B["E"][mk]; ts = W.t.to_numpy()[mk]; et = W.elig_truth.to_numpy()[mk]
                    base = dict(variant=name, cut=cut, panel=p, t=ts, elig_truth=et, iou=ev.window_iou(HS.dec6(G), Y, E))
                    r = pd.DataFrame(base)
                    for k in ("v2", *BLENDS):
                        r[k] = np.nan
                    if "v2" in d:
                        has, P2 = d["v2"]; idx = np.flatnonzero(has)
                        r.loc[idx, "v2"] = ev.window_iou(HS.dec6(P2), Y[idx], E[idx])
                        for k, w in BLENDS.items():
                            r.loc[idx, k] = ev.window_iou(HS.dec6(w * P2 + (1 - w) * G[idx]), Y[idx], E[idx])
                    part.append(r)
                log(f"{name} {cut}: " + " ".join(f"{p} {g.iou.mean():.3f}" for p, g in pd.concat(part[-8:]).groupby('panel')))
        finally:
            FT.V2L["slow"] = FT.V2L["day"] = False; cv.LAB["raw"] = False; FT.Q2["on"] = FT.ALIVE["on"] = False
        df = pd.concat(part, ignore_index=True); df.to_parquet(f); rows.append(df)
        pickle.dump(P, open(HS.OUT / f"gapP_{name}.pkl", "wb"))
    return pd.concat(rows, ignore_index=True)


def table(df: pd.DataFrame, ref: str = "base", ref_blend: str = "E25") -> pd.DataFrame:
    """Per variant: all windows (mean, paired delta vs `ref`) and the v2 windows (alone; every blend vs `ref_blend` of `ref`)."""
    df = df[df.elig_truth]; key = ["cut", "panel", "t"]
    b = df[df.variant == ref].set_index(key); out = []
    for v, g in df.groupby("variant"):
        g = g.set_index(key); d = g.iou - b.iou.reindex(g.index); gv = g.dropna(subset=["v2"]); dv = gv.iou - b.iou.reindex(gv.index)
        r = dict(variant=v, n=len(g), iou=g.iou.mean(), d_all=d.mean(), se_all=d.std(ddof=1) / np.sqrt(len(d)),
                 janfeb=d.xs("janfeb").mean(), octnov=d.xs("octnov").mean(), changed=int((d != 0).sum()),
                 n_v2=len(gv), iou_v2w=gv.iou.mean(), v2=gv.v2.mean(), d_v2w=dv.mean(), se_v2w=dv.std(ddof=1) / np.sqrt(len(dv)))
        for k in BLENDS:
            if k in gv and gv[k].notna().any():
                e = gv[k] - b[ref_blend].reindex(gv.index); r[k] = gv[k].mean(); r[f"d{k}"] = e.mean(); r[f"se{k}"] = e.std(ddof=1) / np.sqrt(len(e))
        out.append(r)
    return pd.DataFrame(out)


if __name__ == "__main__":
    pd.set_option("display.width", 300)
    ap = argparse.ArgumentParser(); ap.add_argument("--variants", nargs="*", default=list(VARIANTS)); ap.add_argument("--ref", default="base")
    ap.add_argument("--out", default="gap.csv"); a = ap.parse_args()
    probs = pickle.load(open(HS.OUT / "probs.pkl", "rb"))
    Ws = {p: pd.read_parquet(HS.OUT / f"windows_{p}.parquet") for p in io.T2_PANELS}
    Bs = {p: HS.bundle(p, Ws[p]) for p in io.T2_PANELS}
    if any(VARIANTS[v].get("cache") == "cache_hw" for v in a.variants):
        build_cache_hw()
    df = run({v: VARIANTS[v] for v in dict.fromkeys([a.ref, *a.variants])}, probs, Ws, Bs)
    t = table(df, a.ref); t.to_csv(HS.OUT / a.out, index=False); print(t.round(4).to_string(index=False))
    log("V2LIKE DONE")
