"""Pooled physics-feature queue models (no link/panel identity) + expected-IoU decoding.

Onset: one row per (window, link), target = queued at T+30; T+5..T+25 predicted 0
(organizer-confirmed selection artifact). Ongoing: one row per (window, link, h).
Optional location priors (fit-period queue frequency) are separate columns so the
"noloc" variant can drop them.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
import pandas as pd
import lightgbm as lgb
from . import data, features as FT
from .build import den_labels

DAY = 288
LOC = ("p_q_hour", "p_q_hour30", "p_onset")


@lru_cache(maxsize=8)
def day_stats(panel: str):
    """Per train day: queue frequency by hour [D, L, 24] and onset first-slot counts [D, L] (denoised labels)."""
    d = data.load(panel, "train"); L = len(d["links"])
    q = den_labels(d["speed"], d["vcut"]).reshape(-1, DAY, L)
    ph = q.reshape(len(q), 24, 12, L).mean(2).transpose(0, 2, 1)
    anyq = q.any(2); ons = np.zeros((len(q), L))
    for dd in range(len(q)):
        for t in range(18, DAY):
            if anyq[dd, t] and not anyq[dd, t - 18:t].any():
                ons[dd] += q[dd, t]
    return ph, ons


@lru_cache(maxsize=8)
def day_flow_q(panel: str):
    """Per train day and link: 99.5th percentile of observed 5-min flow [D, L]."""
    d = data.load(panel, "train"); L = len(d["links"])
    f = d["flow"].reshape(-1, DAY, L)
    return np.nanpercentile(f, 99.5, axis=1)


def priors(panel: str, days) -> dict:
    """Location priors (scenario-dependent) + empirical capacity cemp (scenario-invariant physics) from given days."""
    ph, ons = day_stats(panel)
    days = np.asarray(days); o = ons[days].sum(0)
    cemp = np.nanmedian(day_flow_q(panel)[days], 0)
    return {"ph": ph[days].mean(0), "onset": o / max(o.sum(), 1), "cemp": cemp}


def prior_cols(pr: dict, tod_h: np.ndarray) -> dict:
    """pr['ph'] is [L, 24] (one prior) or [n, L, 24] (per-window test-month prior); onset [L] or [n, L]."""
    if "cols" in pr:  # precomputed per-window columns (onset_tm)
        return pr["cols"]
    h = np.clip(tod_h.astype(int), 0, 23); h30 = np.clip((tod_h + 0.5).astype(int), 0, 23)
    ph, on = pr["ph"], pr["onset"]
    if ph.ndim == 3:
        i = np.arange(len(h))
        return {"p_q_hour": ph[i, :, h], "p_q_hour30": ph[i, :, h30], "p_onset": on}
    return {"p_q_hour": ph[:, h].T, "p_q_hour30": ph[:, h30].T,
            "p_onset": np.broadcast_to(on, (len(tod_h), len(on)))}


def panel_rows(B: dict, net: dict, meta: pd.DataFrame, pr: dict, cond: str):
    """Rows for one panel/condition. Returns X, names, index (window_pos, h, link)."""
    t = meta.t.to_numpy(); tod = (t % DAY) / 12.0
    dow = pd.to_datetime(meta.date).dt.dayofweek.to_numpy()
    st = FT.static(net, pr.get("cemp")); wt = FT.window_tensors(B, net)
    if FT.V2L["day"] and "panel" in meta:
        from .v2like import day_context
        B = dict(B, DC=day_context(meta, B["S"], net))
    Xl, names, geo = FT.link_features(B, net, st, wt, tod, dow, prior_cols(pr, tod))
    n, L, _ = Xl.shape
    hs = [6] if cond == "queue_onset" else [1, 2, 3, 4, 5, 6]
    Xs, idx = [], []
    for h in hs:
        parts = [Xl, FT.horizon_features(geo, st["x"], h)]
        if FT.ALIVE["on"]:
            parts.append(FT.alive_h(geo["alive"], h, L))
        Xs.append(np.concatenate(parts + [np.full((n, L, 1), h, np.float32)], -1).reshape(n * L, -1))
        w, j = np.divmod(np.arange(n * L), L)
        idx.append(np.stack([w, np.full(n * L, h), j], 1))
    return np.concatenate(Xs), names + ["e_shock", "e_kin", "e_pers"] + (list(FT.ALIVE_H) if FT.ALIVE["on"] else []) + ["h"], np.concatenate(idx)


PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
              deterministic=True, force_row_wise=True, num_threads=4, verbose=-1, seed=42)


def fit(X, y, names, drop=(), rounds=400, seed=42):
    keep = [i for i, nm in enumerate(names) if nm not in drop]
    p = dict(PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
    m = lgb.train(p, lgb.Dataset(X[:, keep], y.astype(np.float32), free_raw_data=True), rounds)
    return m, keep


def predict(model_keep, X):
    m, keep = model_keep
    return m.predict(X[:, keep])


def to_grid(p: np.ndarray, idx: np.ndarray, n: int, L: int) -> np.ndarray:
    P = np.zeros((n, 6, L), np.float64)
    P[idx[:, 0], idx[:, 1] - 1, idx[:, 2]] = p
    return P


def decode(P: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Expected-IoU top-m per window (ratio-of-expectations; m=0 scored by P(no positive))."""
    n = len(P); out = np.zeros(P.shape, bool)
    for i in range(n):
        p = P[i].ravel(); o = np.argsort(-p); ps = p[o]
        tot = ps.sum(); cs = np.cumsum(ps); m = np.arange(1, len(ps) + 1)
        e = cs / (m + tot - cs)
        e0 = np.exp(np.sum(np.log1p(-np.clip(ps, 0, 1 - 1e-9))))
        k = int(np.argmax(e)) + 1
        if e0 >= e[k - 1]:
            continue
        k = max(1, int(round(k * scale)))
        out[i].ravel()[o[:k]] = True
    return out
