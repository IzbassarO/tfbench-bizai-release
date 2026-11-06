"""Location-free onset on top of family A v2 ongoing.

Onset T+30 probability per window = 0.5 family A v4 + 0.5 family B noloc, decoded with family A's exact expected-IoU top-m
(force non-empty); T+5..T+25 = 0. Ongoing rows = family A v2 binaries, replaced downstream by the ongoing decode.
"""
from __future__ import annotations
import pickle
import numpy as np
import pandas as pd
from .. import io
from ..t2.decode import expected_iou_decode
from . import codex, run
from .probavg import _prob_grid

T2 = io.OUTPUTS / "t2"
LADDER = {"A": (0.5, 0.5, 0)}  # name: (w_v4, w_t2b_noloc, w_v2)
DELIVER = ("A",)


def decode30(p30: np.ndarray) -> np.ndarray:
    """[n, L] T+30 probabilities -> [n, 6, L] binaries (family A v4 onset decoding)."""
    D = np.zeros((len(p30), 6, p30.shape[1]), bool)
    for i, p in enumerate(p30):
        D[i, 5] = expected_iou_decode(p, force_nonempty=True).astype(bool)
    return D


def mix(w, v4, nl, v2, ph=None):
    return w[0] * v4 + w[1] * nl + w[2] * v2 + (w[3] * ph if len(w) > 3 else 0)


def write():
    P = pickle.load(open(io.OUTPUTS / "t2b" / "final_probs.pkl", "rb"))
    pv2 = pd.read_parquet(T2 / "probs_v2.parquet").rename(columns={"queue_prob": "queue_pred"})
    pv4 = pd.read_parquet(T2 / "probs_v4_onset.parquet").rename(columns={"queue_prob": "queue_pred"})
    csub = codex.submission(T2 / "queue_submission_v2.csv")
    parts = {k: [] for k in DELIVER}; changed = []
    for p in io.T2_PANELS:
        for s in ("validation", "private"):
            wi = pd.read_csv(io.t2_dir(p, s) / "window_index.csv", dtype=str)
            on = (wi.condition == "queue_onset").to_numpy()
            G = codex.grid(csub, p, s, wi)
            v2 = _prob_grid(pv2, p, s, wi)[on, 5]; v4 = _prob_grid(pv4, p, s, wi)[on, 5]
            nl = P[(p, s)]["noloc"][on, 5]
            for k in DELIVER:
                D = G.copy(); D[on] = decode30(mix(LADDER[k], v4, nl, v2)); parts[k].append(run.to_rows(D, p, s, wi))
                d = (D[on] != G[on]).any((1, 2))
                changed.append(dict(variant=k, panel=p, split=s, windows_changed=int(d.sum()), cells_changed=int((D[on] != G[on]).sum())))
    files = []
    for k, ps in parts.items():
        f = f"../t2/queue_onset_{k}.csv"
        pd.concat(ps, ignore_index=True).to_csv(io.OUTPUTS / "t2b" / f, index=False); files.append(f)
    return run.verify(files), pd.DataFrame(changed)


def write_e25(f="../t2/queue_onset_E25.csv", w=0.25):
    """Onset A = decode30(0.5 v4 + 0.5 t2b noloc), ongoing = decode((1 - w) v2 raw + w family B noloc) over all 6 horizons.
    Inputs: t2b/final_probs.pkl,
    t2/probs_v2.parquet, t2/probs_v4_onset.parquet, t2/queue_submission_v2.csv."""
    P = pickle.load(open(io.OUTPUTS / "t2b" / "final_probs.pkl", "rb"))
    pv2 = pd.read_parquet(T2 / "probs_v2.parquet").rename(columns={"queue_prob": "queue_pred"})
    pv4 = pd.read_parquet(T2 / "probs_v4_onset.parquet").rename(columns={"queue_prob": "queue_pred"})
    csub = codex.submission(T2 / "queue_submission_v2.csv"); parts = []
    for p in io.T2_PANELS:
        for s in ("validation", "private"):
            wi = pd.read_csv(io.t2_dir(p, s) / "window_index.csv", dtype=str)
            on = (wi.condition == "queue_onset").to_numpy(); D = codex.grid(csub, p, s, wi)
            C2 = _prob_grid(pv2, p, s, wi); v4 = _prob_grid(pv4, p, s, wi)[on, 5]; nl = P[(p, s)]["noloc"]
            D[on] = decode30(0.5 * v4 + 0.5 * nl[on, 5])
            D[~on] = np.stack([expected_iou_decode(x, force_nonempty=True) for x in (1 - w) * C2[~on] + w * nl[~on]]).astype(bool)
            parts.append(run.to_rows(D, p, s, wi))
    pd.concat(parts, ignore_index=True).to_csv(io.OUTPUTS / "t2b" / f, index=False)
    return run.verify([f])
