"""Task 1 shared pieces: weekday x tod profile, temporal interpolation, state writer.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .. import io, cache

SLOTS = 7 * io.STEPS_PER_DAY


def day_mask(pn, days: np.ndarray) -> np.ndarray:
    """[T] bool for the given day indices."""
    m = np.zeros(len(pn.dates), bool); m[days] = True
    return np.repeat(m, io.STEPS_PER_DAY)


def slot_of(pn) -> np.ndarray:
    """weekday*288 + tod for each row of the time axis."""
    wd = pd.to_datetime(pd.Series(pn.dates)).dt.weekday.to_numpy()
    return (np.repeat(wd, io.STEPS_PER_DAY) * io.STEPS_PER_DAY + np.tile(np.arange(io.STEPS_PER_DAY), len(pn.dates))).astype(int)


def profile(values: np.ndarray, valid: np.ndarray, slot: np.ndarray) -> np.ndarray:
    """[SLOTS, L] mean of values over valid rows, link-mean fallback for empty slots."""
    L = values.shape[1]
    s = np.zeros((SLOTS, L)); c = np.zeros((SLOTS, L))
    v = np.where(valid, values, 0.0)
    np.add.at(s, slot, v); np.add.at(c, slot, valid.astype(float))
    link_mean = s.sum(0) / np.maximum(c.sum(0), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(c > 0, s / np.maximum(c, 1), link_mean[None, :])
    return p


def fit_profiles(pn_train, exclude_days: np.ndarray | None = None):
    """Profiles of truth speed/flow (eligible cells) over train days not excluded."""
    slot = slot_of(pn_train)
    valid = pn_train.elig & np.isfinite(pn_train.speed) & np.isfinite(pn_train.flow)
    if exclude_days is not None and len(exclude_days):
        valid &= ~day_mask(pn_train, exclude_days)[:, None]
    return profile(pn_train.speed, valid, slot), profile(pn_train.flow, valid, slot)


def interp_time(obs: np.ndarray, valid: np.ndarray):
    """Linear interpolation along time per link; returns (interp[T,L], gap_prev[T,L], gap_next[T,L]) in steps.
    Outside the observed range the nearest value is held; gaps there are large."""
    T, L = obs.shape
    out = np.empty((T, L), np.float32); gp = np.empty((T, L), np.int32); gn = np.empty((T, L), np.int32)
    t = np.arange(T)
    for j in range(L):
        idx = np.flatnonzero(valid[:, j])
        if len(idx) == 0:
            out[:, j] = np.nan; gp[:, j] = T; gn[:, j] = T; continue
        out[:, j] = np.interp(t, idx, obs[idx, j])
        prev = np.searchsorted(idx, t, side="right") - 1
        nxt = np.searchsorted(idx, t, side="left")
        gp[:, j] = np.where(prev >= 0, t - idx[np.clip(prev, 0, None)], T)
        gn[:, j] = np.where(nxt < len(idx), idx[np.clip(nxt, None, len(idx) - 1)] - t, T)
    return out, gp, gn


def write_state(preds: dict, out, meta: dict | None = None):
    """preds[(panel, split)] = (speed[T,L], flow[T,L]); writes template-ordered CSV with released timestamp text."""
    first = True
    for (panel, split), (sp, fl) in preds.items():
        pn = cache.load(panel, split)
        tpl = io.t1_template(panel, split)
        dpos = {d: i for i, d in enumerate(pn.dates)}; lpos = {l: i for i, l in enumerate(pn.links)}
        r = io.ts_index(tpl.timestamp, dpos); c = tpl.link_id.map(lpos).to_numpy()
        tpl["speed_kmh"] = np.round(sp[r, c].astype(np.float64), 3); tpl["flow_vph"] = np.round(fl[r, c].astype(np.float64), 2)
        assert np.isfinite(tpl.speed_kmh).all() and np.isfinite(tpl.flow_vph).all(), f"{panel}/{split}: non-finite predictions"
        tpl.to_csv(out, mode="w" if first else "a", header=first, index=False); first = False
    if meta is not None:
        out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


def read_state(path) -> dict:
    """State file (template rows) -> {(panel, split): (speed[T,L], flow[T,L], target[T,L])}; non-target cells are filler."""
    st = pd.read_csv(path, dtype={"panel": str, "timestamp": str, "station_id": str, "link_id": str, "mask_regime": str})
    out = {}
    for panel in io.panels():
        for split in ("validation", "private"):
            pn = cache.load(panel, split)
            dpos = {d: i for i, d in enumerate(pn.dates)}; lpos = {l: i for i, l in enumerate(pn.links)}
            rows = st[(st.panel == panel) & st.timestamp.str[:10].isin(dpos.keys())]
            assert len(rows) == len(io.t1_template(panel, split)), (panel, split)
            r = io.ts_index(rows.timestamp, dpos); c = rows.link_id.map(lpos).to_numpy()
            sp = np.full(pn.mspeed.shape, 100.0, np.float32); fl = np.full(pn.mspeed.shape, 1000.0, np.float32)
            sp[r, c] = rows.speed_kmh.to_numpy(); fl[r, c] = rows.flow_vph.to_numpy()
            tgt = np.zeros(sp.shape, bool); tgt[r, c] = True
            out[(panel, split)] = (sp, fl, tgt)
    return out
