"""Training rows on release-like windows: location priors computed without the window's own month (no in-sample prior).

Labels for fitting: conservative denoised truth (build.den_labels). block_day == -1 marks training-only extras.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .. import io
from . import model as M


def _days(B, months):
    d = pd.to_datetime(pd.Series(io.split_dates("train")))
    return np.flatnonzero(d.dt.strftime("%Y-%m").isin(months).to_numpy())


def _sub(B, mask):
    return {k: B[k][mask] for k in ("S", "Q", "RON", "ROFF", "Y", "YD", "E")}


CONDS = ("queue_onset", "queue_ongoing")


def rows(panel, B, cond, months, prior_months, loo=False, extra=False):
    """Rows of windows in `months`; priors from prior_months (minus own month if loo)."""
    meta = B["meta"]; out = []
    for mon in months:
        mk = ((meta.month == mon) & (meta.condition == cond) & ((meta.block_day != -1) | extra)).to_numpy()
        if not mk.any():
            continue
        pm = [m for m in prior_months if not (loo and m == mon)]
        pr = M.priors(panel, _days(B, pm))
        X, names, idx = M.panel_rows(_sub(B, mk), B["net"], meta[mk], pr, cond)
        sb = _sub(B, mk)
        y = sb["YD"][idx[:, 0], idx[:, 1] - 1, idx[:, 2]]; e = sb["E"][idx[:, 0], idx[:, 1] - 1, idx[:, 2]]
        out.append((X, names, idx, y, e, np.flatnonzero(mk)))
    return out
