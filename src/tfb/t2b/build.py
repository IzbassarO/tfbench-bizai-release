"""Window tensors for the train windows (release-like selector) and for released validation/private windows.

Per window: S/Q [13, L] speed/flow at T-60..T (12 raw history slots + the masked row at T,
exactly what a released window provides), RON/ROFF [13, L] ramp flows, and for train the
truth [6, L] (raw and median-denoised) plus eligibility.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import data, windows as W

DAY = 288


def den_labels(sp: np.ndarray, vcut: np.ndarray, band: float = 4.0) -> np.ndarray:
    """Conservative denoised queue labels: raw threshold, flipping only cells within +-band km/h of v_cut
    whose observed temporal neighbours (same day) both sit on the other side (isolated noise flips)."""
    raw = np.nan_to_num(sp, nan=999) <= vcut
    obs = np.isfinite(sp)
    prev = np.roll(raw, 1, 0); nxt = np.roll(raw, -1, 0)
    po = np.roll(obs, 1, 0); no = np.roll(obs, -1, 0)
    po[::DAY] = False; no[DAY - 1::DAY] = False
    near = obs & (np.abs(np.nan_to_num(sp, nan=999) - vcut) < band)
    flip = near & po & no & (prev != raw) & (nxt != raw)
    return raw ^ flip


def _pack(d, ts, hs=None, hf=None):
    """Inputs at origins ts. hs/hf given (released windows) or taken from raw train arrays."""
    n, L = len(ts), len(d["links"])
    S = np.full((n, 13, L), np.nan, np.float32); Q = S.copy()
    RON = S.copy(); ROFF = S.copy()
    for i, t in enumerate(ts):
        if hs is None:
            S[i, :12] = d["speed"][t - 12:t]; Q[i, :12] = d["flow"][t - 12:t]
        else:
            S[i, :12] = hs[i]; Q[i, :12] = hf[i]
        S[i, 12] = d["mspeed"][t]; Q[i, 12] = d["mflow"][t]
        RON[i] = d["ron"][t - 12:t + 1]; ROFF[i] = d["roff"][t - 12:t + 1]
    return S, Q, RON, ROFF


def train_windows(panel: str, starts=(1, 8, 15, 22), n_extra_ongoing: int = 400, lab_speed=None) -> dict:
    """lab_speed: alternative [T, L] label speed (hybrid labels) for window selection and targets; training cells E = every
    finite label cell. Inputs stay raw."""
    d = data.load(panel, "train"); ls = d["speed"] if lab_speed is None else lab_speed
    sel = W.ffill_day(ls)
    lab = np.nan_to_num(sel, nan=999) <= d["vcut"]
    cand = W.candidates(lab, d["elig"])
    cand_all = W.candidates(lab, np.ones_like(d["elig"]))  # all-cell persistence
    cand["pers_iou"] = cand_all.pers_iou.to_numpy()
    wins = []
    dts = pd.to_datetime(pd.Series(d["dates"]))
    for (y, m), g in dts.groupby([dts.dt.year, dts.dt.month]):
        days = g.dt.day.to_numpy(); idx = g.index.to_numpy()
        for k, s in enumerate(starts):
            nxt = [x for x in starts[k + 1:] if x in days]
            i0 = idx[days == s][0]; i1 = idx[days == nxt[0]][0] if nxt else idx[-1] + 1
            w = W.scan(cand, i0 * DAY, i1 * DAY)
            w["block_day"] = s; w["month"] = f"{y}-{m:02d}"
            wins.append(w)
    w = pd.concat(wins, ignore_index=True).drop_duplicates("t")
    # training-only extras (block_day=-1): every clean onset event at T=e-30, and valid ongoing origins
    # on a 30-min grid (capped per panel); never scored
    ok = (cand.coverage >= 0.7) & (cand.pers_iou <= 0.9) & ~cand.t.isin(w.t)
    first = np.array([lab[t + 1:t + 7].any(1).argmax() for t in cand.t])
    on = cand[ok & (cand.condition == "queue_onset") & (first == 5)]
    og = cand[ok & (cand.condition == "queue_ongoing") & (cand.t % 6 == 0)]
    if len(og) > n_extra_ongoing:
        og = og.iloc[np.sort(np.random.default_rng(0).choice(len(og), n_extra_ongoing, replace=False))]
    ex = pd.concat([on, og]).assign(block_day=-1)
    ex["month"] = [str(d["dates"][t // DAY])[:7] for t in ex.t]
    w = pd.concat([w, ex], ignore_index=True).sort_values("t").reset_index(drop=True)
    w["panel"] = panel; w["date"] = [d["dates"][t // DAY] for t in w.t]
    ts = w.t.to_numpy()
    S, Q, RON, ROFF = _pack(d, ts)
    raw = np.nan_to_num(ls, nan=999) <= d["vcut"]
    den = den_labels(ls, d["vcut"])
    Y = np.stack([raw[t + 1:t + 7] for t in ts]); YD = np.stack([den[t + 1:t + 7] for t in ts])
    el = d["elig"] if lab_speed is None else np.isfinite(ls)
    E = np.stack([el[t + 1:t + 7] for t in ts])
    return dict(meta=w, S=S, Q=Q, RON=RON, ROFF=ROFF, Y=Y, YD=YD, E=E)


def released(panel: str, split: str) -> dict:
    d = data.load(panel, split)
    wi, hs, hf = data.released_windows(panel, split)
    ts = np.array([data.t_index(d, s) for s in wi.forecast_origin])
    S, Q, RON, ROFF = _pack(d, ts, hs, hf)
    meta = pd.DataFrame({"window_id": wi.window_id, "t": ts, "condition": wi.condition,
                         "panel": panel, "date": wi.date, "forecast_origin": wi.forecast_origin})
    return dict(meta=meta, S=S, Q=Q, RON=RON, ROFF=ROFF)
