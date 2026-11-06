"""Family A probabilities on the t2b link grid (same key alignment as tfb.t2b.codex.grid).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _prob_grid(sub, panel, split, wi):
    """codex.grid for float probabilities."""
    from . import data
    d = data.load(panel, split)
    pos = {l: i for i, l in enumerate(d["links"])}; wpos = {w: i for i, w in enumerate(wi.window_id)}
    s = sub[sub.window_id.isin(wpos)]
    G = np.zeros((len(wi), 6, len(pos)))
    start = pd.to_datetime(wi.forecast_origin, utc=True).to_numpy()
    r = s.window_id.map(wpos).to_numpy()
    k = ((pd.to_datetime(s.timestamp, utc=True).to_numpy() - start[r]) / np.timedelta64(5, "m")).astype(int) - 1
    G[r, k, s.link_id.map(pos).to_numpy()] = s.queue_pred.to_numpy()
    assert len(s) == G.size
    return G
