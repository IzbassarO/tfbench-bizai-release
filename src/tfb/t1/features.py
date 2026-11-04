"""Feature matrix for Task 1 cells. All features use the released masked layer + train profiles + ramps."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from .. import io
from .base import fit_profiles, slot_of, interp_time

PANEL_ID = {p: i for i, p in enumerate(io.panels())}
FDF = os.environ.get("TFB_FDF", "0") == "1"   # FD-derived features (tfb.t1.cand candidate c); off = unchanged columns


def shift_t(a, k):  # value at t+k
    out = np.full_like(a, np.nan); 
    if k > 0: out[:-k] = a[k:]
    elif k < 0: out[-k:] = a[:k]
    else: out[:] = a
    return out


def shift_l(a, k):  # value at link j+k (downstream = +)
    out = np.full_like(a, np.nan)
    if k > 0: out[:, :-k] = a[:, k:]
    elif k < 0: out[:, -k:] = a[:, :k]
    else: out[:] = a
    return out


def panel_fields(pn, pn_train, exclude_days=None):
    """Dense [T,L] intermediate fields for one (panel, split)."""
    ps, pf = fit_profiles(pn_train, exclude_days)
    slot = slot_of(pn); prof_s, prof_f = ps[slot].astype(np.float32), pf[slot].astype(np.float32)
    valid = np.isfinite(pn.mspeed) & np.isfinite(pn.mflow)
    rs, gp, gn = interp_time(pn.mspeed - prof_s, valid); rf, _, _ = interp_time(pn.mflow - prof_f, valid)
    base_s = prof_s + rs; base_f = prof_f + rf
    # ramps attached to each link: on / off sums (NaN if no eligible reading; 0 if no ramp attached)
    T, L = pn.mspeed.shape
    ron = np.zeros((T, L), np.float32); roff = np.zeros((T, L), np.float32)
    has_on = np.zeros(L, bool); has_off = np.zeros(L, bool)
    for r, j in enumerate(pn.ramp_link):
        if j < 0: continue
        v = pn.rflow[:, r]
        if pn.ramp_on[r]: has_on[j] = True; ron[:, j] = np.where(np.isfinite(v), ron[:, j] + np.nan_to_num(v), np.nan)
        else: has_off[j] = True; roff[:, j] = np.where(np.isfinite(v), roff[:, j] + np.nan_to_num(v), np.nan)
    # day-level residual of observed cells vs profile (per link-day and per panel-day)
    D = len(pn.dates)
    res_s = np.where(valid, pn.mspeed - prof_s, np.nan).reshape(D, io.STEPS_PER_DAY, L)
    res_f = np.where(valid, pn.mflow - prof_f, np.nan).reshape(D, io.STEPS_PER_DAY, L)
    with np.errstate(all="ignore"):
        day_s = np.repeat(np.nanmean(res_s, axis=1), io.STEPS_PER_DAY, axis=0)
        day_f = np.repeat(np.nanmean(res_f, axis=1), io.STEPS_PER_DAY, axis=0)
        pday_s = np.repeat(np.nanmean(res_s, axis=(1, 2)), io.STEPS_PER_DAY)
        pday_f = np.repeat(np.nanmean(res_f, axis=(1, 2)), io.STEPS_PER_DAY)
    return dict(prof_s=prof_s, prof_f=prof_f, base_s=base_s, base_f=base_f, gp=gp, gn=gn, valid=valid,
                ron=ron, roff=roff, has_on=has_on, has_off=has_off, day_s=day_s, day_f=day_f, pday_s=pday_s, pday_f=pday_f, slot=slot)


def build(pn, F, rows: np.ndarray, cols: np.ndarray) -> pd.DataFrame:
    """Feature frame for the cells (rows[i], cols[i])."""
    T, L = pn.mspeed.shape
    f = {}
    g = lambda a: a[rows, cols]
    f["panel"] = np.full(len(rows), PANEL_ID[pn.panel], np.int16)
    f["link_rank"] = cols.astype(np.int16); f["link_rel"] = (cols / L).astype(np.float32)
    for k in ("lanes", "v_f", "cap", "k_crit", "length", "milepost"):
        f[k] = pn[k][cols]
    f["cap_lane"] = pn.cap[cols] / pn.lanes[cols]
    f["regime"] = np.repeat(pn.regime, io.STEPS_PER_DAY)[rows].astype(np.int8)
    f["tod"] = (rows % io.STEPS_PER_DAY).astype(np.int16); f["wd"] = (F["slot"][rows] // io.STEPS_PER_DAY).astype(np.int8)
    for k in ("prof_s", "prof_f", "base_s", "base_f", "gp", "gn", "day_s", "day_f"):
        f[k] = g(F[k])
    f["pday_s"] = F["pday_s"][rows]; f["pday_f"] = F["pday_f"][rows]
    f["gp"] = np.minimum(f["gp"], 288); f["gn"] = np.minimum(f["gn"], 288)
    f["gap_min"] = np.minimum(f["gp"], f["gn"]); f["gap_sum"] = f["gp"] + f["gn"]
    f["base_f_lane"] = f["base_f"] / f["lanes"]
    f["base_k_lane"] = f["base_f"] / np.maximum(f["base_s"], 1) / f["lanes"]
    f["base_k_rel"] = f["base_k_lane"] * f["lanes"] / f["k_crit"]
    f["base_v_rel"] = f["base_s"] / f["v_f"]
    f["prof_v_rel"] = f["prof_s"] / f["v_f"]
    ms, mf = pn.mspeed, pn.mflow
    # own link raw observations around t
    for k in (-3, -2, -1, 1, 2, 3):
        f[f"s_t{k:+d}"] = g(shift_t(ms, k)); f[f"f_t{k:+d}"] = g(shift_t(mf, k))
    # neighbours: interpolated base at t and raw at t-1,t,t+1
    for dj in (-3, -2, -1, 1, 2, 3):
        f[f"bs_l{dj:+d}"] = g(shift_l(F["base_s"], dj)); f[f"bf_l{dj:+d}"] = g(shift_l(F["base_f"], dj))
        f[f"bs_l{dj:+d}_rel"] = f[f"bs_l{dj:+d}"] / shift_l(pn.v_f[None, :], dj)[0][cols]
        f[f"gap_l{dj:+d}"] = g(shift_l(np.minimum(F["gp"], F["gn"]).astype(np.float32), dj))
    for dj in (-1, 1):
        for k in (-1, 0, 1):
            f[f"s_l{dj:+d}_t{k:+d}"] = g(shift_l(shift_t(ms, k), dj)); f[f"f_l{dj:+d}_t{k:+d}"] = g(shift_l(shift_t(mf, k), dj))
        f[f"pct_l{dj:+d}"] = g(shift_l(pn.mpct.astype(np.float32), dj))
    # ramps at t-1, t, t+1 (own link) and on-ramp of upstream link
    for k in (-1, 0, 1):
        f[f"ron_t{k:+d}"] = g(shift_t(F["ron"], k)); f[f"roff_t{k:+d}"] = g(shift_t(F["roff"], k))
    f["ron_up"] = g(shift_l(F["ron"], -1)); f["roff_down"] = g(shift_l(F["roff"], 1))
    f["has_on"] = F["has_on"][cols].astype(np.int8); f["has_off"] = F["has_off"][cols].astype(np.int8)
    # neighbour-based spatial interpolation at t (mean of nearest observed up/down base values)
    f["sp_mean_s"] = np.nanmean(np.stack([f["bs_l-1"], f["bs_l+1"]]), axis=0)
    f["sp_mean_f"] = np.nanmean(np.stack([f["bf_l-1"], f["bf_l+1"]]), axis=0)
    if FDF:
        fd_features(f, pn, cols)
    df = pd.DataFrame({k: np.asarray(v, dtype=np.float32) if np.asarray(v).dtype.kind == "f" else v for k, v in f.items()})
    return df


def fd_features(f: dict, pn, cols):
    """Triangular-FD views of the interpolated base: FD flow / speed at the base density and the base's residuals to
    them, congestion / over-critical flags, neighbour relative densities and their gradient, congested-neighbour count."""
    kj, kc, cap, vf = pn.k_jam[cols], f["k_crit"], f["cap"], f["v_f"]
    with np.errstate(all="ignore"):
        k = f["base_f"] / np.maximum(f["base_s"], 1)
        qfd = np.where(k <= kc, vf * k, cap / np.maximum(kj - kc, 1e-9) * np.maximum(kj - k, 0))
        f["fd_q_lane"] = qfd / f["lanes"]; f["fd_res_q_lane"] = (f["base_f"] - qfd) / f["lanes"]
        f["fd_v"] = qfd / np.maximum(k, 1e-3); f["fd_res_v"] = f["base_s"] - f["fd_v"]
        f["fd_cong"] = (f["base_v_rel"] < 0.6).astype(np.float32); f["fd_overcrit"] = (f["base_k_rel"] > 1).astype(np.float32)
        for dj in (-1, 1):
            f[f"bk_l{dj:+d}_rel"] = f[f"bf_l{dj:+d}"] / np.maximum(f[f"bs_l{dj:+d}"], 1) / shift_l(pn.k_crit[None, :], dj)[0][cols]
        f["dk_rel"] = f["bk_l+1_rel"] - f["bk_l-1_rel"]
        f["n_cong_nb"] = sum((f[f"bs_l{dj:+d}_rel"] < 0.6).astype(np.float32) for dj in (-3, -2, -1, 1, 2, 3))


CAT = ["panel", "regime"]


def build_bo(pn, F, rows: np.ndarray, cols: np.ndarray) -> pd.DataFrame:
    """build() + ramp mass-balance features for cells inside a whole-panel blackout (ramps stay released there)."""
    df = build(pn, F, rows, cols)
    net = np.nan_to_num(F["ron"]) - np.nan_to_num(F["roff"])
    cum = np.cumsum(net, axis=1)                                   # net ramp inflow from panel start through link j
    miss = np.cumsum(~np.isfinite(F["ron"]) | ~np.isfinite(F["roff"]), axis=1).astype(np.float32)
    prev = np.clip(rows - F["gp"][rows, cols], 0, None)            # own link's last observed row
    df["cum_net"] = cum[rows, cols]; df["cum_net_prev"] = cum[prev, cols]
    df["d_cum_net"] = df["cum_net"] - df["cum_net_prev"]; df["ramp_miss_up"] = miss[rows, cols]
    for k in (-3, -2, 2, 3):
        df[f"ron_t{k:+d}"] = shift_t(F["ron"], k)[rows, cols]; df[f"roff_t{k:+d}"] = shift_t(F["roff"], k)[rows, cols]
    return df
