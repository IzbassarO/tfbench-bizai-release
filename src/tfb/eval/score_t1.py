"""Exact Task 1 scorer on train (template targets, per-lane flow, regime macro-mean)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .. import io


def score_arrays(pn, pred_speed: np.ndarray, pred_flow: np.ndarray, mask: np.ndarray | None = None) -> pd.DataFrame:
    """pn: cached train Panel; pred arrays [T, L]; mask: subset of target cells to score (default all targets)."""
    tgt = pn.target if mask is None else (pn.target & mask)
    day_reg = np.repeat(pn.regime, io.STEPS_PER_DAY)
    rows = []
    for r, name in enumerate(io.REGIMES):
        m = tgt & (day_reg == r)[:, None]
        n = int(m.sum())
        if n == 0:
            rows.append(dict(panel=pn.panel, regime=name, n=0, rmse_speed=np.nan, rmse_flow_lane=np.nan, S_speed=0, S_flow=0, S_state=0.0)); continue
        ds = pred_speed[m] - pn.speed[m]
        lanes = np.broadcast_to(pn.lanes, m.shape)[m]
        dq = (pred_flow[m] - pn.flow[m]) / lanes
        rs = float(np.sqrt(np.mean(ds.astype(np.float64) ** 2))); rq = float(np.sqrt(np.mean(dq.astype(np.float64) ** 2)))
        ss = max(0.0, 1 - rs / 25); sq = max(0.0, 1 - rq / 600)
        rows.append(dict(panel=pn.panel, regime=name, n=n, rmse_speed=rs, rmse_flow_lane=rq, S_speed=ss, S_flow=sq, S_state=0.54 * ss + 0.46 * sq))
    d = pd.DataFrame(rows)
    d.loc[len(d)] = dict(panel=pn.panel, regime="mean", n=int(d.n.sum()), rmse_speed=np.nan, rmse_flow_lane=np.nan, S_speed=np.nan, S_flow=np.nan, S_state=float(d.S_state.mean()))
    return d


def overall(panel_tables: list[pd.DataFrame]) -> float:
    fam = io.family_of()
    d = pd.concat(panel_tables); d = d[d.regime == "mean"].copy(); d["family"] = d.panel.map(fam)
    return float(d.groupby("family").S_state.mean().mean())
