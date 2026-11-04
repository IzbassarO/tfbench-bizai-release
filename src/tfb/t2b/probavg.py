"""Probability-average ensemble with family A v2 (outputs/t2/probs_v2.parquet, pre-calibration probabilities):
P = a_c * t2b + (1 - a_c) * family A per condition c, decoded with our expected-IoU top-m (onset T+5..T+25 = 0).

python -m tfb.t2b.probavg  -> CV grid on family A's official-like windows (both folds) + outputs/t2b/queue_submission_probavg.csv
"""
from __future__ import annotations
import pickle
import numpy as np
import pandas as pd
from .. import io
from . import cv, evaluate as ev, model as M, run
from .report import _grids, load_fold

OUT = io.OUTPUTS / "t2b"
A = (0.0, 0.25, 0.5, 0.75, 1.0)


def cv_grid() -> pd.DataFrame:
    rows = []
    for fold in cv.FOLDS:
        probs, cb = load_fold(fold, "hybrid")
        for (pk, cond, _), (Pv, wpos) in _grids(probs).items():
            if "@" not in pk:
                continue
            b = cb[pk]; mine = run.blend(Pv, run.CONFIG["blend"]); cx = b["codex"]
            Y, E = b["Y"][wpos], b["E"][wpos]
            cands = {f"a{a}": a * mine + (1 - a) * cx["probability_v2"][wpos] for a in A}
            dec = {k: M.decode(P) for k, P in cands.items()}
            dec["codex_v2_bin"] = cx["prediction_v2"][wpos].astype(bool)
            for k, D in dec.items():
                if cond == "queue_onset":
                    D[:, :5] = False
                rows.append(pd.DataFrame({"fold": fold, "panel": pk.split("@")[0], "condition": cond, "model": k,
                                          "iou": ev.window_iou(D, Y, E)}))
    df = pd.concat(rows, ignore_index=True)
    return pd.DataFrame([dict(fold=f, condition=c, model=m, iou=_fam(g)) for (f, c, m), g in df.groupby(["fold", "condition", "model"])])


def _fam(g):
    p = g.groupby("panel").iou.mean()
    return p.groupby(p.index.str.rsplit("_", n=1).str[0]).mean().mean()


def write(a_on: float, a_og: float, f="queue_submission_probavg.csv"):
    P = pickle.load(open(OUT / "final_probs.pkl", "rb"))
    cp = pd.read_parquet(io.OUTPUTS / "t2" / "probs_v2.parquet").rename(columns={"queue_prob": "queue_pred"})
    parts = []
    for p in io.T2_PANELS:
        for s in ("validation", "private"):
            wi = pd.read_csv(io.t2_dir(p, s) / "window_index.csv", dtype=str)
            C = _prob_grid(cp, p, s, wi); mine = run.blend(P[(p, s)], run.CONFIG["blend"])
            on = (wi.condition == "queue_onset").to_numpy()
            a = np.where(on, a_on, a_og)[:, None, None]
            D = M.decode(a * mine + (1 - a) * C); D[on, :5] = False
            parts.append(run.to_rows(D, p, s, wi))
    df = pd.concat(parts, ignore_index=True); df.to_csv(OUT / f, index=False)
    return run.verify([f])


def _prob_grid(sub, panel, split, wi):
    """codex.grid for float probabilities (same key alignment)."""
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


if __name__ == "__main__":
    import json
    g = cv_grid(); g.to_csv(OUT / "probavg_cv.csv", index=False)
    print(g.pivot_table(index="model", columns=["condition", "fold"], values="iou").round(4).to_string())
