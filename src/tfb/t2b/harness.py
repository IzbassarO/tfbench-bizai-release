"""Every admissible Task 2 window on the train months (official selector, no 5-per-panel quota), training windows of v2l.

Selector (published synthetic-release configuration and scoring spec): origins every 5 min inside the day; latent label =
hybrid train speed (released truth on eligible cells, clean imputer elsewhere) <= 0.6 v_f; history T-60..T-5 (12 slots),
horizon T+5..T+30; onset = no queued cell in the history and a queue in the horizon (the first such T, i.e. the queue is
born at T+30); ongoing = an established queue in the history (>= 2 queued slots on one link) and a queue in the horizon,
persistence IoU (state at T-5 repeated, all cells) <= 0.9; history coverage (eligible cells) >= 0.7; spacing >= 360 min
within (panel, condition); every window kept.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .. import io
from . import build, data, evaluate as ev, hybrid as H, windows as W

OUT = io.OUTPUTS / "t2b" / "harness"
DAY, SPACING = 288, 72
ON, OG = "queue_onset", "queue_ongoing"


def select(panel: str, mode: str = "mix_clean") -> pd.DataFrame:
    """Every admissible window of a train panel, per-condition 360-min clock, no quota; latent queue state = clean-imputer labels."""
    d = data.load(panel, "train")
    ls = H.label_speed(panel, mode); ls = np.where(np.isfinite(ls), ls, d["speed"])
    lab = np.nan_to_num(ls, nan=999) <= d["vcut"]
    cand = W.candidates(lab, d["elig"])
    cand["pers_iou"] = W.candidates(lab, np.ones_like(d["elig"])).pers_iou.to_numpy()   # all-cell persistence
    cand["birth"] = [int(lab[t + 1:t + 7].any(1).argmax()) for t in cand.t]      # first queued horizon slot (latent)
    cand["elig_truth"] = [bool((np.nan_to_num(d["speed"][t + 1:t + 7], nan=999) <= d["vcut"])[d["elig"][t + 1:t + 7]].any()) for t in cand.t]
    ok = cand[(cand.coverage >= 0.7) & (cand.pers_iou <= 0.9) & ((cand.condition != ON) | (cand.birth == 5))]
    keep = []
    for cond in (ON, OG):
        last = -10 ** 9
        for r in ok[ok.condition == cond].itertuples():
            if r.t - last >= SPACING:
                keep.append(r.Index); last = r.t
    w = cand.loc[sorted(keep)].copy()
    w["panel"] = panel; w["date"] = [d["dates"][t // DAY] for t in w.t]; w["month"] = w.date.astype(str).str[:7]
    w["block_day"] = 0; w["dirty_skipped"] = int((ok.condition == "queue_onset_dirty").sum())
    return w.reset_index(drop=True)


def bundle(panel: str, w: pd.DataFrame) -> dict:
    d = data.load(panel, "train"); ts = w.t.to_numpy()
    S, Q, RON, ROFF = build._pack(d, ts)
    raw = np.nan_to_num(d["speed"], nan=999) <= d["vcut"]; den = build.den_labels(d["speed"], d["vcut"])
    B = dict(meta=w.reset_index(drop=True), S=S, Q=Q, RON=RON, ROFF=ROFF, Y=np.stack([raw[t + 1:t + 7] for t in ts]),
             YD=np.stack([den[t + 1:t + 7] for t in ts]), E=np.stack([d["elig"][t + 1:t + 7] for t in ts]))
    B["net"] = ev.load_train(panel)["net"]
    return B
