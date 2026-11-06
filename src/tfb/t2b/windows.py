"""Release-like Task 2 window selector on train (published organizer policy).

Chronological scan from a block start; history T-60..T-5; ongoing iff the history holds
>=2 queued obs on one link, onset iff it holds none; coverage >= .7 eligible history cells;
a queue somewhere in T+5..T+30; persistence (state at T-5) eligible IoU <= .9; one shared
360-min clock; 5 windows per condition. Val/private windows sit in the first days of their
month, so a block starting on day 1 of a month is the release-like sample.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

H, F, DAY = 12, 6, 288


def ffill_day(x: np.ndarray) -> np.ndarray:
    """Same-day forward fill along time of a [T, L] array."""
    out = x.copy()
    for s in range(0, len(x), DAY):
        blk = out[s:s + DAY]
        for t in range(1, len(blk)):
            m = np.isnan(blk[t]); blk[t, m] = blk[t - 1, m]
    return out


def iou(pred: np.ndarray, true: np.ndarray, elig: np.ndarray) -> float:
    p, t = pred[elig], true[elig]
    if elig.sum() == 0:
        return 0.0
    u = (p | t).sum()
    return 1.0 if u == 0 else float((p & t).sum() / u)


def candidates(lab: np.ndarray, elig: np.ndarray) -> pd.DataFrame:
    """Per-origin selector predicates for every in-day origin."""
    T = len(lab); rows = []
    for t in range(H, T - F):
        if (t % DAY) < H or (t % DAY) > DAY - 1 - F:
            continue
        h = lab[t - H:t]
        tot = h.sum()
        cond = "queue_ongoing" if h.sum(0).max() >= 2 else "queue_onset"
        if cond == "queue_onset" and tot > 0:
            cond = "queue_onset_dirty"
        fut = lab[t + 1:t + 1 + F]
        if not fut.any():
            continue
        cov = elig[t - H:t].mean()
        pers = iou(np.repeat(lab[t - 1:t], F, 0), fut, elig[t + 1:t + 1 + F])
        rows.append((t, cond, cov, pers))
    return pd.DataFrame(rows, columns=["t", "condition", "coverage", "pers_iou"])


def scan(cand: pd.DataFrame, start: int, stop: int, quota: int = 5, spacing: int = 72,
         onset_dirty_ok: bool = True) -> pd.DataFrame:
    c = cand[(cand.t >= start) & (cand.t < stop) & (cand.coverage >= 0.7) & (cand.pers_iou <= 0.9)]
    last = -10 ** 9; n = {"queue_onset": 0, "queue_ongoing": 0}; keep = []
    for r in c.itertuples():
        cond = "queue_onset" if r.condition == "queue_onset_dirty" and onset_dirty_ok else r.condition
        if cond not in n or n[cond] >= quota or r.t - last < spacing:
            continue
        keep.append(r.Index); n[cond] += 1; last = r.t
        if min(n.values()) >= quota:
            break
    out = cand.loc[keep].copy()
    out["condition"] = out.condition.replace({"queue_onset_dirty": "queue_onset"})
    return out
