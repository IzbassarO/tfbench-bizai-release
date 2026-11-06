"""Imputation of hidden train cells with a speed booster, for the Task 2 hybrid labels.

Imputer input = a released-like masked layer built from the truth: eligible cells observed, every imputed cell hidden
(pct stays, as released). Speed model only; inference only.
"""
from __future__ import annotations
import numpy as np
from .. import io, cache
from . import features as FE
from .train import SPEED_CAP

PANELS = io.T2_PANELS        # Task 2 labels only exist here


def hidden_layer(pn, hide):
    q = cache.Panel(pn); obs = pn.elig & ~hide
    for k, v in (("mspeed", pn.speed), ("mflow", pn.flow), ("mocc", pn.occ)):
        q[k] = np.where(obs, v, np.nan).astype(np.float32)
    q["mpct"] = pn.pct; q["melig"] = obs
    return q


def impute(models, q, pn_train, exclude_days, sel, levels=(0,)):
    """{n_trees: [T,L] mean over speed models of base + residual of the first n_trees trees (0 = all)} at the cells sel;
    NaN elsewhere. Levels are cumulative, so all of them cost one pass of the largest."""
    F = FE.panel_fields(q, pn_train, exclude_days); cols = models[0].feature_name()
    out = {n: np.full(sel.shape, np.nan, np.float32) for n in levels}; r, c = np.nonzero(sel)
    for s in range(0, len(r), 1_000_000):
        rr, cc = r[s:s + 1_000_000], c[s:s + 1_000_000]
        df = FE.build(q, F, rr, cc)[cols]; res = np.zeros(len(rr)); start = 0
        for n in sorted(levels, key=lambda n: n or 10 ** 9):
            end = min(n or 10 ** 9, min(m.num_trees() for m in models))
            res += np.mean([m.predict(df, start_iteration=start, num_iteration=end - start, num_threads=8) for m in models], axis=0) if end > start else 0
            out[n][rr, cc] = np.clip(df.base_s.to_numpy() + res, 3.0, SPEED_CAP); start = end
    return out
