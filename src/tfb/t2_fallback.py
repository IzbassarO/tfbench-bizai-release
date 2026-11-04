"""Rule-based Task 2 fallback used only when family A's outputs/t2/queue_submission.csv is absent.

onset: queue_pred = 0 for T+5..T+25; at T+30 mark eligible links whose released speed at T is
       below 0.8*v_f (closest to tipping); if none, the single lowest speed/v_f link.
ongoing: persistence of the queue at T (speed_T <= v_cut) through all six steps.
Uses only data at or before T.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from . import io, cache


def build(out: Path) -> Path:
    rows = []
    for panel in io.T2_PANELS:
        net = io.network(panel).set_index("link_id")
        for split in ("validation", "private"):
            d = io.t2_dir(panel, split)
            if not d.exists():
                continue
            widx = pd.read_csv(d / "window_index.csv", dtype=str)
            tpl = pd.read_csv(d / "sample_submission_queue.csv", dtype=str)
            pn = cache.load(panel, split)
            dpos = {x: i for i, x in enumerate(pn.dates)}; lpos = {l: i for i, l in enumerate(pn.links)}
            for w in widx.itertuples(index=False):
                tT = io.ts_index(pd.Series([w.forecast_origin]), dpos)[0]
                sp = pn.mspeed[tT]; ok = pn.melig[tT] & np.isfinite(sp)
                ratio = np.where(ok, sp / pn.v_f, np.inf)
                t = tpl[tpl.window_id == w.window_id].copy()
                li = t.link_id.map(lpos).to_numpy()
                if w.condition == "queue_ongoing":
                    q = (ratio <= 0.6)
                    t["queue_pred"] = q[li].astype(int)
                else:
                    cand = ratio < 0.8
                    if not cand.any():
                        cand = np.zeros_like(cand); cand[np.argmin(ratio)] = True
                    last = t.timestamp == w.forecast_end
                    t["queue_pred"] = (cand[li] & last.to_numpy()).astype(int)
                rows.append(t[["window_id", "timestamp", "link_id", "queue_pred"]])
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(out, index=False)
    return out
