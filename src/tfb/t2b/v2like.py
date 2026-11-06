"""Slow-link ongoing features, location-free (member v2l), and its training cache.

  slow  FT.V2L["slow"]: slow70 / slow80 count, nearest signed / absolute km, upstream / downstream km, corridor min and
        lower-quartile speed ratio, min ratio over 3 links up / down (from the forward-filled speed ratio at T)
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from .. import io
from . import evaluate as ev, harness as HS

OG = HS.OG


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


def build_cache_hw(log=log, windows=None, name="cache_hw"):
    """outputs/t2b/<name>: the base cache's windows plus the admissible ongoing windows of <windows>/windows_<p>.parquet
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
        log(f"cache_hw {p}: {len(old['meta'])} base + {len(W)} selector ongoing windows")
