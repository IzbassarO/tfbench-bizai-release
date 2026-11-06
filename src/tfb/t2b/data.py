"""Panel arrays for the physics queue model, links ordered upstream -> downstream.

Reads the shared npz cache (tfb.cache, read-only). E/N panels: topology
order_index increases downstream; W/S panels: it decreases (verified on all
8 panels), so their link axis is flipped. Index j+1 is always downstream of j.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
import pandas as pd
from .. import cache, io


def downstream_order(panel: str, links: np.ndarray) -> np.ndarray:
    top = io.topology(panel).set_index("mainline_link_id")
    oi = top.order_index.astype(int).reindex(links).to_numpy()
    o = np.argsort(oi, kind="stable")
    return o[::-1] if panel.endswith(("_W", "_S")) else o


@lru_cache(maxsize=3)
def load(panel: str, split: str) -> dict:
    p = cache.load(panel, split)
    o = downstream_order(panel, p["links"])
    d = {"panel": panel, "split": split, "dates": p["dates"], "links": p["links"][o]}
    for k in ("lanes", "v_f", "cap", "k_crit", "k_jam", "length", "milepost"):
        d[k] = p[k][o].astype(np.float64)
    for k in ("mspeed", "mflow", "melig") + (("speed", "flow", "elig", "pct") if split == "train" else ()):
        d[k] = p[k][:, o]
    # ramp flows per mainline link (eligible only; NaN = no evidence), on and off separately
    T, L = d["mspeed"].shape
    inv = np.empty_like(o); inv[o] = np.arange(L)
    rl = p["ramp_link"]; ok = rl >= 0
    for name, sel in (("ron", p["ramp_on"] & ok), ("roff", ~p["ramp_on"] & ok)):
        a = np.full((T, L), np.nan, np.float32)
        cols = inv[rl[sel]]; rf = p["rflow"][:, sel]
        for c in np.unique(cols):
            m = cols == c
            a[:, c] = np.where(np.isnan(rf[:, m]).all(1), np.nan, np.nansum(rf[:, m], 1))
        d[name] = a
        d[name + "_has"] = np.isin(np.arange(L), cols)
    d["vcut"] = 0.6 * d["v_f"]
    return d


def t_index(d: dict, ts: str) -> int:
    day = list(d["dates"]).index(ts[:10])
    return day * io.STEPS_PER_DAY + int(ts[11:13]) * 12 + int(ts[14:16]) // 5


def released_windows(panel: str, split: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """window_index plus history [W, 12, L] speed/flow (downstream order) from window_history.parquet."""
    d = load(panel, split)
    wi = pd.read_csv(io.t2_dir(panel, split) / "window_index.csv", dtype=str)
    h = pd.read_parquet(io.t2_dir(panel, split) / "window_history.parquet")
    pos = {l: i for i, l in enumerate(d["links"])}
    W, L = len(wi), len(d["links"])
    hs = np.full((W, 12, L), np.nan, np.float32); hf = hs.copy()
    wpos = {w: i for i, w in enumerate(wi.window_id)}
    start = pd.to_datetime(wi.history_start, utc=True)
    r = h.window_id.map(wpos).to_numpy()
    k = ((pd.to_datetime(h.timestamp, utc=True).to_numpy() - start.to_numpy()[r]) / np.timedelta64(5, "m")).astype(int)
    c = h.link_id.map(pos).to_numpy()
    hs[r, k, c] = h.speed_kmh.to_numpy(np.float32); hf[r, k, c] = h.flow_vph.to_numpy(np.float32)
    return wi, hs, hf
