"""Pooled LightGBM for Task 1 (speed and per-lane flow residuals vs profile-anchored interpolation).

  python -m tfb.t1.train final [panel ...]   -> trains on all train days, predicts validation+private,
                                                writes outputs/t1/state_submission[_<TAG>][_seed<S>].csv
env: TFB_SEED=42, TFB_TAG=v1, TFB_ROUNDS=3000, TFB_LR=0.05,
  TFB_TRANS=0   number of pseudo-holdout masks of the test period's observed cells added as training rows
                (each hides TRANS_P of them; masks 0-4 are disjoint, every further 5 use a fresh hash),
  TFB_TW=1      sample weight of those transductive (test-period) rows vs train rows,
  TFB_CW=1      sample weight of congested rows (label speed < 0.6 v_f), multiplied with TFB_TW.
"""
from __future__ import annotations
import sys, time, zlib
import numpy as np
import pandas as pd
import lightgbm as lgb
from .. import io, cache
from . import features as FE
from .base import day_mask, write_state

import os
SEED = int(os.environ.get("TFB_SEED", 42))
TAG = os.environ.get("TFB_TAG", "v1")
TRANS = int(os.environ.get("TFB_TRANS", 0))
TW = float(os.environ.get("TFB_TW", 1))
CW = float(os.environ.get("TFB_CW", 1))
SPEED_CAP = 120.75   # generator hard cap: max eligible train speed is exactly 120.75 on every panel
ROWS_PER_PANEL = 400_000          # training cells sampled per panel (pooled -> ~4M rows)
TRANS_P, TRANS_ROWS = 0.2, 100_000   # share of observed cells hidden per pseudo mask; rows kept per (panel, mask)
PARAMS = dict(objective="regression", learning_rate=float(os.environ.get("TFB_LR", 0.05)), num_leaves=127, min_data_in_leaf=200,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, num_threads=8, seed=SEED,
              deterministic=True, force_row_wise=True, verbose=-1)
ROUNDS = int(os.environ.get("TFB_ROUNDS", 3000))
EARLY_STOP = 100


def sample_cells(pn, rows_ok: np.ndarray, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    idx = np.flatnonzero(pn.target & rows_ok[:, None] & np.isfinite(pn.speed) & np.isfinite(pn.flow))
    if len(idx) > n:
        idx = rng.choice(idx, n, replace=False)
    idx.sort()
    return np.unravel_index(idx, pn.target.shape)


def labels(df, sp, fl, lanes):
    """Residual targets: speed, per-lane flow (vs the interpolated base)."""
    return sp - df.base_s.to_numpy(), (fl - df.base_f.to_numpy()) / lanes


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


def build_training(train_days_excluded, panels, test_splits=()):
    """Training frame X, targets ys, weights w and the rng (its next draw is the early-stopping hold mask).
    test_splits: splits whose observed cells feed transductive rows."""
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
                            TRANS_ROWS // len(test_splits))
            parts += tr; is_t += [True] * len(tr)
        print(f"  features {panel}: {sum(len(p[0]) for p in parts)} rows cumulative", flush=True); del pn
    X = pd.concat([p[0] for p in parts], ignore_index=True)
    ys = {"speed": np.concatenate([p[1] for p in parts]), "flow_lane": np.concatenate([p[2] for p in parts])}
    w = None if TW == 1 and CW == 1 else np.concatenate([np.full(len(p[0]), TW if t else 1.0, np.float32)
                                                         * np.where(p[1] + p[0].base_s.to_numpy() < 0.6 * p[0].v_f.to_numpy(), CW, 1).astype(np.float32)
                                                         for p, t in zip(parts, is_t)])
    parts.clear()   # memory: per-panel frames go before the Datasets are built, X right after (train_models)
    return X, ys, w, rng


def train_models(train_days_excluded, panels, test_splits=()):
    X, ys, w, rng = build_training(train_days_excluded, panels, test_splits)
    cols = list(X.columns); sets = make_sets(X, ys["speed"], rng, w); del X
    return fit_sets(*sets, ys), cols


def predict_panel(models, cols, pn, pn_train, exclude_days, rows_mask):
    """Full [T,L] prediction arrays (base + residual) for target cells in rows_mask; base elsewhere. Returns speed, flow."""
    F = FE.panel_fields(pn, pn_train, exclude_days)
    sp = F["base_s"].copy(); fl = F["base_f"].copy()
    r, c = np.nonzero(pn.target & rows_mask[:, None])
    for s in range(0, len(r), 1_000_000):
        rr, cc = r[s:s + 1_000_000], c[s:s + 1_000_000]
        df = FE.build(pn, F, rr, cc)[cols]
        sp[rr, cc] = df.base_s.to_numpy() + models["speed"].predict(df, num_threads=8)
        fl[rr, cc] = df.base_f.to_numpy() + models["flow_lane"].predict(df, num_threads=8) * pn.lanes[cc]
    sp = np.clip(np.nan_to_num(sp, nan=0), 3.0, SPEED_CAP); fl = np.clip(np.nan_to_num(fl, nan=0), 100.0, None)
    return sp.astype(np.float32), fl.astype(np.float32)


def run_final(panels):
    models, cols = train_models(None, panels, test_splits=("validation", "private"))
    preds = {}
    for panel in panels:
        pn_train = cache.load(panel, "train")
        for split in ("validation", "private"):
            pn = cache.load(panel, split)
            sp, fl = predict_panel(models, cols, pn, pn_train, None, np.ones(len(pn.mspeed), bool))
            preds[(panel, split)] = (sp, fl)
            lt50 = float((fl[pn.target] < 50).mean())
            print(panel, split, "done; share<50vph", lt50, flush=True); assert lt50 < 0.01
        del pn_train
    suffix = "" if (SEED == 42 and TAG == "v1") else f"_{TAG}" * (TAG != "v1") + f"_seed{SEED}" * (SEED != 42)
    out = io.OUTPUTS / "t1" / f"state_submission{suffix}.csv"; out.parent.mkdir(parents=True, exist_ok=True)
    write_state(preds, out, meta={"model": "pooled LGBM speed+flow_lane residual on profile-anchored interpolation", "tag": TAG,
                                  "rounds": {k: m.best_iteration for k, m in models.items()}, "params": PARAMS, "rows_per_panel": ROWS_PER_PANEL,
                                  "seed": SEED, "trans": TRANS})
    for k, m in models.items():
        m.save_model(str(io.OUTPUTS / "t1" / f"lgbm_{k}{suffix}.txt"))
    print("wrote", out)


if __name__ == "__main__":
    run_final(sys.argv[2:] or io.panels())
