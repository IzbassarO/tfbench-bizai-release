"""Pooled LightGBM for Task 1 (speed and per-lane flow residuals vs profile-anchored interpolation).

usage:
  python -m tfb.t1.train holdout [last30|autumn]    -> trains on other train days, scores holdout, logs; saves models +
                                                        holdout predictions to cache/t1_runs/<TFB_TAG>_<holdout>/
  python -m tfb.t1.train bo [last30|autumn] <run>   -> holdout with simulated Task 2 blackouts: main models of run
                                                        <run>, scored with and without the dedicated blackout model
  python -m tfb.t1.train bo_patch <in> <out>       -> dedicated blackout model (all train days) overwrites the fully dark
                                                        target cells of an existing validation+private state file
  python -m tfb.t1.train final                       -> trains on all train days, predicts validation+private,
                                                        writes outputs/t1/state_submission[_seed<S>].csv
env (defaults = v1): TFB_SEED=42, TFB_TAG=v1, TFB_ROUNDS=3000, TFB_LR=0.05,
  TFB_TRANS=0   number of pseudo-holdout masks of the test period's observed cells added as training rows
                (each hides TRANS_P of them; masks 0-4 are disjoint, every further 5 use a fresh hash),
  TFB_TW=1      sample weight of those transductive (test-period) rows vs train rows,
  TFB_CW=1      sample weight of congested rows (label speed < 0.6 v_f), multiplied with TFB_TW,
  TFB_DENSITY=0 also fit a per-lane density residual model (for gated reconciliation, tfb.t1.posthoc),
  TFB_BO=0      final: dedicated blackout model for target cells inside Task 2 blackouts.
"""
from __future__ import annotations
import json, sys, time, zlib
import numpy as np
import pandas as pd
import lightgbm as lgb
from .. import io, cache, final_config as FC
from ..eval.score_t1 import score_arrays, overall
from ..t3.proxy import evaluate as phys_eval
from . import features as FE
from pathlib import Path
from .base import HOLDOUT_LAST30, HOLDOUT_AUTUMN, day_mask, write_state, read_state, log_experiment

import os
SEED = int(os.environ.get("TFB_SEED", 42))
TAG = os.environ.get("TFB_TAG", "v1")
TRANS = int(os.environ.get("TFB_TRANS", 0))
TW = float(os.environ.get("TFB_TW", 1))
CW = float(os.environ.get("TFB_CW", 1))
DENSITY = os.environ.get("TFB_DENSITY", "0") == "1"
BO = os.environ.get("TFB_BO", "0") == "1"
SPEED_CAP = 120.75   # generator hard cap: max eligible train speed is exactly 120.75 on every panel
ROWS_PER_PANEL = 400_000          # training cells sampled per panel (pooled -> ~4M rows)
TRANS_P, TRANS_ROWS = 0.2, 100_000   # share of observed cells hidden per pseudo mask; rows kept per (panel, mask)
BO_LEN, BO_GAP = 18, 72           # Task 2 blackout: rows T+1..T+18 dark on every link; origins >= 6 h apart
BO_ROWS = 150_000                 # blackout-model training cells per panel
PARAMS = dict(objective="regression", learning_rate=float(os.environ.get("TFB_LR", 0.05)), num_leaves=127, min_data_in_leaf=200,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, num_threads=8, seed=SEED,
              deterministic=True, force_row_wise=True, verbose=-1)
ROUNDS = int(os.environ.get("TFB_ROUNDS", 3000))
EARLY_STOP = 100
_cfg = FC.get("task1.lgbm")   # configs/final.yaml when TFB_CONFIG is set (scripts/reproduce.sh); else the defaults above
if _cfg:
    assert set(_cfg["params"]) == set(PARAMS) - {"seed"}, "configs/final.yaml task1.lgbm.params must list every LightGBM param"
    PARAMS.update(_cfg["params"])
    ROWS_PER_PANEL, TRANS_P, TRANS_ROWS, EARLY_STOP = _cfg["rows_per_panel"], _cfg["trans_p"], _cfg["trans_rows"], _cfg["early_stopping"]


def sample_cells(pn, rows_ok: np.ndarray, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    idx = np.flatnonzero(pn.target & rows_ok[:, None] & np.isfinite(pn.speed) & np.isfinite(pn.flow))
    if len(idx) > n:
        idx = rng.choice(idx, n, replace=False)
    idx.sort()
    return np.unravel_index(idx, pn.target.shape)


def labels(df, sp, fl, lanes):
    """Residual targets: speed, per-lane flow, per-lane density (vs the interpolated base)."""
    return (sp - df.base_s.to_numpy(), (fl - df.base_f.to_numpy()) / lanes,
            fl / np.maximum(sp, 1) / lanes - df.base_k_lane.to_numpy())


def trans_masks(obs, key: str, n: int):
    """n hide masks of share TRANS_P: buckets of a per-(panel, split) hash; each block of 1/TRANS_P masks is disjoint,
    the next block re-hashes (key|block), so masks 0-4 are unchanged for any n."""
    per = round(1 / TRANS_P); u = None
    for k in range(n):
        b, i = divmod(k, per)
        if i == 0:
            u = np.random.default_rng(zlib.crc32((key + (f"|{b}" if b else "")).encode())).random(obs.shape, dtype=np.float32)
        yield obs & (u >= i * TRANS_P) & (u < (i + 1) * TRANS_P)


def trans_rows(pn, pn_train, rows_ok, exclude_days, rng, cap):
    """Transductive rows: TRANS hash-bucket masks (share TRANS_P each) of the test period's observed eligible cells,
    hidden like targets; label = released value (equals truth on observed cells)."""
    obs = np.isfinite(pn.mspeed) & np.isfinite(pn.mflow) & pn.melig & rows_ok[:, None]
    out = []
    for hide in trans_masks(obs, f"{pn.panel}|{pn.split}", TRANS):
        q = cache.Panel(pn); q["mspeed"] = np.where(hide, np.nan, pn.mspeed); q["mflow"] = np.where(hide, np.nan, pn.mflow)
        idx = np.flatnonzero(hide)
        if len(idx) > cap:
            idx = np.sort(rng.choice(idx, cap, replace=False))
        r, c = np.unravel_index(idx, hide.shape)
        df = FE.build(q, FE.panel_fields(q, pn_train, exclude_days), r, c)
        out.append((df, *labels(df, pn.mspeed[r, c], pn.mflow[r, c], pn.lanes[c])))
    return out


def make_sets(X, y, rng, w=None):
    """Binned train/valid Datasets built once (raw frames freed after construction); bins do not depend on the label,
    so fit_sets reuses them for every target via set_label -> same boosters as one fresh Dataset per target."""
    hold = rng.random(len(X)) < 0.05
    kw = lambda m: dict(weight=None if w is None else w[m], categorical_feature=FE.CAT, params=PARAMS)
    dtr = lgb.Dataset(X[~hold], y[~hold], **kw(~hold)).construct()
    dva = lgb.Dataset(X[hold], y[hold], reference=dtr, **kw(hold)).construct()
    return dtr, dva, hold


def fit_sets(dtr, dva, hold, ys: dict, rounds=None):
    models = {}
    for name, y in ys.items():
        dtr.set_label(y[~hold]); dva.set_label(y[hold])
        t0 = time.time()
        m = lgb.train(PARAMS, dtr, rounds or ROUNDS, valid_sets=[dva], callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(0)])
        print(f"  {name}: best_iter={m.best_iteration} rmse={m.best_score['valid_0']['l2'] ** 0.5:.3f} ({time.time() - t0:.0f}s)", flush=True)
        models[name] = m
    return models


def fit(X, ys: dict, rng, rounds=None, w=None):
    return fit_sets(*make_sets(X, next(iter(ys.values())), rng, w), ys, rounds)


def build_training(train_days_excluded, panels, test_splits=()):
    """Training frame X, targets ys, weights w and the rng (its next draw is the early-stopping hold mask).
    test_splits: splits whose observed cells feed transductive rows (holdout: 'train' restricted to the holdout days)."""
    rng = np.random.default_rng(SEED); rng_t = np.random.default_rng(SEED + 1)
    parts, is_t = [], []
    for panel in panels:
        pn = cache.load(panel, "train")
        rows_ok = ~day_mask(pn, train_days_excluded) if train_days_excluded is not None else np.ones(len(pn.mspeed), bool)
        F = FE.panel_fields(pn, pn, exclude_days=train_days_excluded)
        r, c = sample_cells(pn, rows_ok, ROWS_PER_PANEL, rng)
        df = FE.build(pn, F, r, c)
        parts.append((df, *labels(df, pn.speed[r, c], pn.flow[r, c], pn.lanes[c]))); is_t.append(False); del F
        for split in (test_splits if TRANS else ()):
            pt = pn if split == "train" else cache.load(panel, split)
            tr = trans_rows(pt, pn, ~rows_ok if split == "train" else np.ones(len(pt.mspeed), bool), train_days_excluded, rng_t,
                            TRANS_ROWS // len(test_splits))   # same transductive share in holdout (1 month) and final (2)
            parts += tr; is_t += [True] * len(tr)
        print(f"  features {panel}: {sum(len(p[0]) for p in parts)} rows cumulative", flush=True); del pn
    X = pd.concat([p[0] for p in parts], ignore_index=True)
    ys = {"speed": np.concatenate([p[1] for p in parts]), "flow_lane": np.concatenate([p[2] for p in parts])}
    if DENSITY:
        ys["k_lane"] = np.concatenate([p[3] for p in parts])
    w = None if TW == 1 and CW == 1 else np.concatenate([np.full(len(p[0]), TW if t else 1.0, np.float32)
                                                         * np.where(p[1] + p[0].base_s.to_numpy() < 0.6 * p[0].v_f.to_numpy(), CW, 1).astype(np.float32)
                                                         for p, t in zip(parts, is_t)])
    parts.clear()   # memory: per-panel frames go before the Datasets are built, X right after (train_models)
    return X, ys, w, rng


def train_models(train_days_excluded, panels, test_splits=()):
    X, ys, w, rng = build_training(train_days_excluded, panels, test_splits)
    cols = list(X.columns); sets = make_sets(X, ys["speed"], rng, w); del X
    return fit_sets(*sets, ys), cols


def bo_origins(pn, rows_ok, n, rng) -> np.ndarray:
    """Up to n simulated Task 2 origins T: a truth queue (v <= 0.6 v_f) somewhere in T+1..T+6, dark span inside one
    allowed day, origins >= BO_GAP rows apart (organizer windows sit at queue times with 6 h shared spacing)."""
    q = ((pn.speed <= 0.6 * pn.v_f[None, :]) & pn.elig).any(1)
    hz = np.zeros(len(q), bool)
    for h in range(1, 7):
        hz[:-h] |= q[h:]
    ok = np.flatnonzero(hz & rows_ok & (np.arange(len(q)) % io.STEPS_PER_DAY + BO_LEN < io.STEPS_PER_DAY))
    blocked = np.zeros(len(q), bool); picked = []
    for t in rng.permutation(ok):
        if blocked[t]:
            continue
        picked.append(t); blocked[max(t - BO_GAP + 1, 0):t + BO_GAP] = True
        if len(picked) == n:
            break
    return np.sort(np.array(picked, int))


def blackout(pn, origins):
    """Copy of pn with every link dark on rows T+1..T+BO_LEN (speed/flow/occupancy; pct and eligibility stay, as released)."""
    dark = np.zeros(len(pn.mspeed), bool)
    for t in origins:
        dark[t + 1:t + 1 + BO_LEN] = True
    q = cache.Panel(pn)
    for k in ("mspeed", "mflow", "mocc"):
        q[k] = np.where(dark[:, None], np.nan, pn[k])
    return q, dark


def train_bo(train_days_excluded, panels):
    """Dedicated blackout model: dense simulated blackouts (~1.5 per train day) on Task 2 panels; rows = every eligible
    cell inside them (all are dark, exactly like a blackout target)."""
    rng = np.random.default_rng(SEED + 2); parts = []
    for panel in [p for p in panels if p in io.T2_PANELS]:
        pn = cache.load(panel, "train")
        rows_ok = ~day_mask(pn, train_days_excluded) if train_days_excluded is not None else np.ones(len(pn.mspeed), bool)
        q, dark = blackout(pn, bo_origins(pn, rows_ok, int(1.5 * rows_ok.sum() / io.STEPS_PER_DAY), rng))
        idx = np.flatnonzero(dark[:, None] & pn.elig & np.isfinite(pn.speed) & np.isfinite(pn.flow))
        if len(idx) > BO_ROWS:
            idx = np.sort(rng.choice(idx, BO_ROWS, replace=False))
        r, c = np.unravel_index(idx, pn.speed.shape)
        df = FE.build_bo(q, FE.panel_fields(q, pn, train_days_excluded), r, c)
        parts.append((df, *labels(df, pn.speed[r, c], pn.flow[r, c], pn.lanes[c])))
        print(f"  blackout rows {panel}: {len(df)}", flush=True); del pn, q
    X = pd.concat([p[0] for p in parts], ignore_index=True)
    return fit(X, {"speed": np.concatenate([p[1] for p in parts]), "flow_lane": np.concatenate([p[2] for p in parts])}, rng), list(X.columns)


def dark_rows(pn) -> np.ndarray:
    """[T] rows where every link is blank in the masked layer (Task 2 blackouts in validation/private)."""
    return ~np.isfinite(pn.mspeed).any(1)


def predict_panel(models, cols, pn, pn_train, exclude_days, rows_mask, bo=None):
    """Full [T,L] prediction arrays (base + residual) for target cells in rows_mask; base elsewhere.
    bo = (models, cols): target cells on fully dark rows use the dedicated blackout model.
    Returns speed, flow, per-lane density (None unless a k_lane model exists)."""
    F = FE.panel_fields(pn, pn_train, exclude_days)
    sp = F["base_s"].copy(); fl = F["base_f"].copy()
    kd = (fl / np.maximum(sp, 1) / pn.lanes).astype(np.float32) if "k_lane" in models else None
    tgt = pn.target & rows_mask[:, None]
    dark = dark_rows(pn)[:, None] & tgt if bo is not None else np.zeros_like(tgt)
    for ms, cs, sel, build in ((models, cols, tgt & ~dark, FE.build), (bo[0] if bo else None, bo[1] if bo else None, dark, FE.build_bo)):
        r, c = np.nonzero(sel)
        for s in range(0, len(r), 1_000_000):
            rr, cc = r[s:s + 1_000_000], c[s:s + 1_000_000]
            df = build(pn, F, rr, cc)[cs]
            sp[rr, cc] = df.base_s.to_numpy() + ms["speed"].predict(df, num_threads=8)
            fl[rr, cc] = df.base_f.to_numpy() + ms["flow_lane"].predict(df, num_threads=8) * pn.lanes[cc]
            if kd is not None and "k_lane" in ms:
                kd[rr, cc] = df.base_k_lane.to_numpy() + ms["k_lane"].predict(df, num_threads=8)
    sp = np.clip(np.nan_to_num(sp, nan=0), 3.0, SPEED_CAP); fl = np.clip(np.nan_to_num(fl, nan=0), 100.0, None)
    if kd is not None:
        kd = np.clip(np.nan_to_num(kd, nan=0), 0.5, None).astype(np.float32)
    return sp.astype(np.float32), fl.astype(np.float32), kd


def holdout_days(name):
    return HOLDOUT_LAST30 if name == "last30" else HOLDOUT_AUTUMN


def run_dir(tag, name):
    return io.CACHE / "t1_runs" / f"{tag}_{name}"


def save_models(models, cols, d, prefix=""):
    d.mkdir(parents=True, exist_ok=True)
    for k, m in models.items():
        m.save_model(str(d / f"{prefix}lgbm_{k}.txt"))
    (d / f"{prefix}cols.json").write_text(json.dumps(cols))


def load_models(d, prefix=""):
    names = [p.stem.split("lgbm_", 1)[1] for p in sorted(d.glob(f"{prefix}lgbm_*.txt"))]
    return {k: lgb.Booster(model_file=str(d / f"{prefix}lgbm_{k}.txt")) for k in names}, json.loads((d / f"{prefix}cols.json").read_text())


def score_preds(preds: dict, days, label: str):
    """preds[panel] = (speed, flow) [T,L] arrays (train panels). Exact S_state (template targets) + physics proxies."""
    tabs, phys = [], []
    for panel, (sp, fl) in preds.items():
        pn = cache.load(panel, "train"); rows = day_mask(pn, days)
        tabs.append(score_arrays(pn, sp, fl, mask=np.broadcast_to(rows[:, None], pn.target.shape)))
        ph = phys_eval(pn, sp, fl, rows); ph["panel"] = panel; phys.append(ph)
        print(label, panel, tabs[-1][tabs[-1].regime == "mean"].S_state.iloc[0].round(4), {k: round(v, 4) for k, v in ph.items() if k != "panel"}, flush=True)
    d = pd.concat(tabs); s = overall(tabs) if len(preds) == 10 else float(d[d.regime == "mean"].S_state.mean())
    return s, d, pd.DataFrame(phys)


def run_holdout(name: str, panels):
    days = holdout_days(name)
    models, cols = train_models(days, panels, test_splits=("train",))
    out = run_dir(TAG, name); save_models(models, cols, out)
    preds = {}
    for panel in panels:
        pn = cache.load(panel, "train"); rows = day_mask(pn, days)
        sp, fl, kd = predict_panel(models, cols, pn, pn, days, rows)
        np.savez(out / f"{panel}.npz", rows=np.flatnonzero(rows), sp=sp[rows], fl=fl[rows], **({"kd": kd[rows]} if kd is not None else {}))
        preds[panel] = (sp, fl); del pn
    s, d, phys = score_preds(preds, days, name)
    print(name, f"S_state={s:.5f}"); print(d.to_string())
    rep = io.PROJECT / "reports" / "t1_runs"; rep.mkdir(exist_ok=True)
    phys.to_csv(rep / f"{TAG}_{name}_phys.csv", index=False); d.to_csv(rep / f"{TAG}_{name}_scores.csv", index=False)
    imp = pd.DataFrame({"feature": cols, "speed": models["speed"].feature_importance("gain"), "flow": models["flow_lane"].feature_importance("gain")})
    imp.to_csv(rep / f"{TAG}_{name}_importance.csv", index=False)
    iters = {k: m.best_iteration for k, m in models.items()}
    log_experiment(task="t1", change=f"{TAG}: rounds={ROUNDS} lr={PARAMS['learning_rate']} trans={TRANS} seed={SEED} ({'all' if len(panels) == 10 else ','.join(panels)})",
                   local_metric=f"S_state_{name}", local_value=round(s, 5),
                   notes=f"S_phys_proxy={phys.S_phys_proxy.mean():.4f}; best_iter={iters}; run={out.name}")
    return s


def run_bo(name: str, run: str, panels):
    """Same holdout, plus 10 simulated blackouts per Task 2 panel (validation rate); main models from run <run>."""
    days = holdout_days(name)
    main, cols = load_models(run_dir(run, name))
    bo = train_bo(days, panels)
    save_models(*bo, run_dir(run, name), prefix="bo_")
    res = {"main_only": {}, "with_bo": {}}; dark_err = {"main_only": [], "with_bo": []}
    for panel in panels:
        pn = cache.load(panel, "train"); rows = day_mask(pn, days)
        origins = bo_origins(pn, rows, 10, np.random.default_rng(zlib.crc32(panel.encode()))) if panel in io.T2_PANELS else np.array([], int)
        q, _ = blackout(pn, origins)
        dk = dark_rows(q)[:, None] & q.target & rows[:, None]
        for key, b in (("main_only", None), ("with_bo", bo)):
            sp, fl, _ = predict_panel(main, cols, q, pn, days, rows, bo=b)
            res[key][panel] = (sp, fl)
            if dk.any():
                dark_err[key].append(np.c_[sp[dk] - pn.speed[dk], (fl[dk] - pn.flow[dk]) / np.broadcast_to(pn.lanes, dk.shape)[dk]])
        del pn, q
    out = {}
    for key in res:
        s, d, phys = score_preds(res[key], days, f"{name}/{key}")
        e = np.concatenate(dark_err[key]); out[key] = s
        rm = np.sqrt(np.mean(e.astype(np.float64) ** 2, axis=0))
        print(f"{name} {key}: S_state={s:.5f} dark cells n={len(e)} rmse speed={rm[0]:.3f} flow/lane={rm[1]:.2f}", flush=True)
        log_experiment(task="t1", change=f"{run}+blackout-sim holdout: {key}", local_metric=f"S_state_{name}_bo", local_value=round(s, 5),
                       notes=f"dark target cells n={len(e)} rmse_speed={rm[0]:.3f} rmse_flow_lane={rm[1]:.2f}; S_phys_proxy={phys.S_phys_proxy.mean():.4f}")
    return out


def run_final(panels):
    models, cols = train_models(None, panels, test_splits=("validation", "private"))
    bo = train_bo(None, panels) if BO else None
    preds = {}
    for panel in panels:
        pn_train = cache.load(panel, "train")
        for split in ("validation", "private"):
            pn = cache.load(panel, split)
            sp, fl, _ = predict_panel(models, cols, pn, pn_train, None, np.ones(len(pn.mspeed), bool), bo=bo)
            preds[(panel, split)] = (sp, fl)
            lt50 = float((fl[pn.target] < 50).mean())
            print(panel, split, "done; share<50vph", lt50, flush=True); assert lt50 < 0.01
        del pn_train
    suffix = "" if (SEED == 42 and TAG == "v1") else f"_{TAG}" * (TAG != "v1") + f"_seed{SEED}" * (SEED != 42)
    out = io.OUTPUTS / "t1" / f"state_submission{suffix}.csv"; out.parent.mkdir(parents=True, exist_ok=True)
    write_state(preds, out, meta={"model": "pooled LGBM speed+flow_lane residual on profile-anchored interpolation", "tag": TAG,
                                  "rounds": {k: m.best_iteration for k, m in models.items()}, "params": PARAMS, "rows_per_panel": ROWS_PER_PANEL,
                                  "seed": SEED, "trans": TRANS, "blackout_model": BO})
    for k, m in models.items():
        m.save_model(str(io.OUTPUTS / "t1" / f"lgbm_{k}{suffix}.txt"))
    if bo:
        save_models(*bo, io.OUTPUTS / "t1", prefix=f"bo{suffix}_")
    print("wrote", out)


def run_bo_patch(src, dst, panels):
    bo, bo_cols = train_bo(None, panels)
    save_models(bo, bo_cols, io.OUTPUTS / "t1", prefix="bo_")
    preds, n = {}, 0
    for (panel, split), (sp, fl, tgt) in read_state(src).items():
        pn = cache.load(panel, split); dk = dark_rows(pn)[:, None] & tgt
        if dk.any():
            r, c = np.nonzero(dk)
            df = FE.build_bo(pn, FE.panel_fields(pn, cache.load(panel, "train")), r, c)[bo_cols]
            sp[r, c] = np.clip(np.nan_to_num(df.base_s.to_numpy() + bo["speed"].predict(df, num_threads=8), nan=0), 3.0, SPEED_CAP)
            fl[r, c] = np.clip(np.nan_to_num(df.base_f.to_numpy() + bo["flow_lane"].predict(df, num_threads=8) * pn.lanes[c], nan=0), 100.0, None)
            n += len(r)
        preds[(panel, split)] = (sp, fl)
    write_state(preds, Path(dst), meta={"post": "dedicated blackout model on fully dark target cells", "input": str(src), "cells": n,
                                         "rounds": {k: m.best_iteration for k, m in bo.items()}, "seed": SEED})
    print("wrote", dst, "dark target cells replaced:", n)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "holdout":
        run_holdout(sys.argv[2], sys.argv[3:] or io.panels())
    elif mode == "bo_patch":
        run_bo_patch(sys.argv[2], sys.argv[3], io.panels())
    elif mode == "bo":
        run_bo(sys.argv[2], sys.argv[3], sys.argv[4:] or io.panels())
    else:
        run_final(sys.argv[2:] or io.panels())
