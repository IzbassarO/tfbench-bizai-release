"""Physics features per (window, link[, horizon]) from the 13 slots T-60..T only (causal by construction).

Links are ordered upstream -> downstream; x = position (km) of link centre along the corridor.
Queue clusters at T: runs of queued links (gap <= 1 link). For each link we describe its
relation to the nearest cluster (inside / upstream of tail / downstream of head, in km), the
cluster's age, length, head recovery, kinematic tail velocity (history) and LWR shockwave
speed at the tail w = (q_up - q_in)/(k_up - k_in), and the predicted tail at each horizon.
Bottleneck stress = flow / capacity of the tightest link 0..3 downstream.
"""
from __future__ import annotations
import warnings
import numpy as np

warnings.filterwarnings("ignore", message="(Mean of empty slice|All-NaN slice encountered)")

NSLOT = 13
PHYS = {"on": False}  # extra location-free physics features (G member, tfb.t2b.phys)
ASM = {"on": False}   # ASM propagation features (tfb.t2b.asm); off = recipe features unchanged
Q2 = {"on": False}    # structure / N-balance / inflow features (tfb.t2b.q2feat, 2026-11-01); off = recipe unchanged
V2L = {"slow": False, "day": False}  # v2-like ongoing features (tfb.t2b.v2like, see report, Sec. III-C): slow-link distances, day context
ALIVE = {"on": False}  # queue trajectory over the history + alive / dissolved projection per step (see report, Sec. III-C)
ALIVE_H = ("al_proj_h", "al_alive_h", "al_frac_h")
K_EXT = 12  # downstream links scanned for the onset head (extent features)
DAY_COLS = ("run_min", "cum_min", "since_first", "cor_count", "cor_peak", "cor_since_peak", "cor_dpeak", "cor_d30", "cor_since_first", "known_frac")


def ffill_slots(a: np.ndarray) -> np.ndarray:
    a = a.copy()
    for k in range(1, a.shape[1]):
        m = np.isnan(a[:, k]); a[:, k][m] = a[:, k - 1][m]
    return a


def _runs(mask: np.ndarray, gap: int = 1):
    js = np.flatnonzero(mask)
    if len(js) == 0:
        return []
    cuts = np.flatnonzero(np.diff(js) > gap + 1) + 1
    return [(g[0], g[-1]) for g in np.split(js, cuts)]


def static(net: dict, cemp: np.ndarray | None = None) -> dict:
    """cemp: empirical per-link capacity (train p99.5 flow); bottleneck capacity = min over downstream links."""
    L = len(net["v_f"]); length = net["length"]
    x = np.cumsum(length) - length / 2
    capd = np.array([net["cap"][j:j + 4].min() for j in range(L)])
    out = dict(x=x, capd=capd, cap_drop=capd / net["cap"], L=L)
    if cemp is not None:
        c = np.where(np.isfinite(cemp), cemp, net["cap"]); out["c"] = c
        out["ce4"] = np.array([c[j:j + 4].min() for j in range(L)])
        out["ce10"] = np.array([c[j:j + 10].min() for j in range(L)])
        out["ce_drop"] = out["ce10"] / c
    return out


def window_tensors(B: dict, net: dict) -> dict:
    """Shared per-window arrays: ffilled speed/flow, queued masks per slot, ratio."""
    s = ffill_slots(B["S"]); q = ffill_slots(B["Q"])
    qm = np.nan_to_num(s, nan=999) <= net["vcut"]
    return dict(s=s, q=q, qm=qm, r=s / net["v_f"])


def link_features(B: dict, net: dict, st: dict, wt: dict, tod: np.ndarray, dow: np.ndarray,
                  prior: dict | None = None) -> tuple[np.ndarray, list[str], dict]:
    """[n, L, F] link features + per-window cluster geometry for horizon features."""
    s, q, qm, r = wt["s"], wt["q"], wt["qm"], wt["r"]
    n, _, L = s.shape; x = st["x"]; cap = net["cap"]; capd = st["capd"]
    f = {}
    with np.errstate(all="ignore"):
        f["r_T"] = r[:, 12]; f["r_T3"] = r[:, 9]; f["r_T6"] = r[:, 6]; f["r_T12"] = r[:, 0]
        rn = np.stack([np.roll(r[:, 12], k, 1) for k in (-2, -1, 1, 2)], 0)
        f["r_nb_min"] = np.nanmin(np.concatenate([rn, r[None, :, 12]]), 0)
        f["stress_T"] = q[:, 12] / capd; f["stress_T6"] = q[:, 6] / capd; f["stress_T12"] = q[:, 0] / capd
        f["stress_slope"] = (np.nanmean(q[:, 10:13], 1) - np.nanmean(q[:, 0:3], 1)) / capd
        dn = np.stack([np.roll(q[:, 12] / capd, -k, 1) for k in range(1, 4)], 0); dn[:, :, -3:] = np.nan
        f["stress_down_max"] = np.nanmax(dn, 0)
        up = np.stack([np.roll(q[:, 12], k, 1) for k in range(1, 6)], 0); up[:, :, :5] = np.nan
        f["upflow_stress"] = np.nanmean(up, 0) / capd
        f["qT"] = qm[:, 12].astype(np.float32); f["q_T1"] = qm[:, 11]; f["q_T3"] = qm[:, 9]; f["q_T6"] = qm[:, 6]
        f["q_hist_frac"] = qm.mean(1)
        f["occ_proxy"] = q[:, 12] / np.maximum(s[:, 12], 1) / net["k_crit"]  # density / k_crit
    if "ce4" in st:
        with np.errstate(all="ignore"):
            q3 = np.nanmean(q[:, 10:13], 1); q0 = np.nanmean(q[:, 0:3], 1)
            for k in ("ce4", "ce10"):
                f[f"se_{k}"] = q[:, 12] / st[k]
                f[f"se_{k}_T6"] = q[:, 6] / st[k]
                f[f"se_{k}_slope"] = (q3 - q0) / st[k]
                f[f"ttb_{k}"] = np.clip((st[k] - q3) / np.maximum((q3 - q0) / 10, 1.0), -50, 200)  # slots to breakdown
            f["se_up_ce10"] = f["upflow_stress"] * capd / st["ce10"]
            f["ce_drop"] = np.broadcast_to(st["ce_drop"], (n, L))
    f["cap_drop"] = np.broadcast_to(st["cap_drop"], (n, L)); f["lanes"] = np.broadcast_to(net["lanes"], (n, L))
    f["vf"] = np.broadcast_to(net["v_f"], (n, L)); f["len"] = np.broadcast_to(net["length"], (n, L))
    f["ron"] = np.nan_to_num(np.nanmean(B["RON"][:, 10:13], 1), nan=-1)
    f["tod"] = np.broadcast_to(tod[:, None], (n, L)); f["weekend"] = np.broadcast_to((dow >= 5)[:, None], (n, L))
    # cluster geometry at T
    keys = ["inside", "d_up", "d_down", "c_len", "c_age", "c_head_r", "c_head_dr", "c_vtail", "c_w", "c_tail_x",
            "c_head_x", "n_clusters", "c_qup_qdis", "c_mean_r"]
    for k in keys:
        f[k] = np.full((n, L), np.nan, np.float32)
    geo = {"tail_x": np.full((n, L), np.nan), "head_x": np.full((n, L), np.nan),
           "w": np.full((n, L), np.nan), "vt": np.full((n, L), np.nan)}
    for i in range(n):
        cl = _runs(qm[i, 12])
        f["n_clusters"][i] = len(cl)
        if not cl:
            continue
        rows = []
        for (a, b) in cl:
            # age: slots back the cluster's head neighbourhood was continuously queued
            age = 0
            for k in range(11, -1, -1):
                if qm[i, k, max(a, b - 2):b + 1].any():
                    age += 1
                else:
                    break
            # kinematic tail: tail of the run touching the head 3 slots ago
            def tail_at(k):
                for (a2, b2) in _runs(qm[i, k]):
                    if b2 >= a - 1 and a2 <= b + 1:
                        return x[a2]
                return np.nan
            t3 = tail_at(9)
            vt = (x[a] - t3) / 3 if np.isfinite(t3) else 0.0  # km per slot (+ = tail moves downstream)
            ua = slice(max(a - 3, 0), a); ia = slice(a, min(a + 3, b + 1))
            with np.errstate(all="ignore"):
                qu, vu = np.nanmean(q[i, 12, ua]), np.nanmean(s[i, 12, ua])
                qi, vi = np.nanmean(q[i, 12, ia]), np.nanmean(s[i, 12, ia])
                ku, ki = qu / max(vu, 1), qi / max(vi, 1)
                w = (qu - qi) / (ku - ki) if np.isfinite(ku - ki) and abs(ku - ki) > 1e-3 else 0.0
                w = float(np.clip(np.nan_to_num(w), -40, 40)) / 12  # km per slot
                qdis = np.nanmean(q[i, 12, b + 1:b + 3]) if b + 1 < L else np.nan
                hr = r[i, 12, b]; hdr = r[i, 12, b] - r[i, 9, b]
            rows.append((a, b, age, vt, w, qu / qdis if qdis and np.isfinite(qdis) else np.nan, hr, hdr,
                         np.nanmean(r[i, 12, a:b + 1])))
        heads = np.array([x[b] for (a, b, *_) in rows]); tails = np.array([x[a] for (a, b, *_) in rows])
        # nearest cluster per link by distance to its interval
        dist = np.maximum(tails[None, :] - x[:, None], 0) + np.maximum(x[:, None] - heads[None, :], 0)
        c = dist.argmin(1)
        for k, (a, b, age, vt, w, qr, hr, hdr, mr) in enumerate(rows):
            j = c == k
            f["c_len"][i, j] = x[b] - x[a] + net["length"][b]; f["c_age"][i, j] = age
            f["c_head_r"][i, j] = hr; f["c_head_dr"][i, j] = hdr; f["c_vtail"][i, j] = vt; f["c_w"][i, j] = w
            f["c_tail_x"][i, j] = x[a]; f["c_head_x"][i, j] = x[b]; f["c_qup_qdis"][i, j] = qr; f["c_mean_r"][i, j] = mr
            geo["tail_x"][i, j] = x[a]; geo["head_x"][i, j] = x[b]; geo["w"][i, j] = w; geo["vt"][i, j] = vt
        f["inside"][i] = (dist[np.arange(L), c] == 0)
        f["d_up"][i] = np.where(x < tails[c], tails[c] - x, 0)
        f["d_down"][i] = np.where(x > heads[c], x - heads[c], 0)
    if PHYS["on"] and "c" in st:
        f.update(phys_features(B, net, st, s, q, r))
    if ASM["on"]:
        from .asm import features as asm_features
        f.update(asm_features(B, net, st))
    if Q2["on"]:
        f.update(q2_features(net, st, s, q, qm, geo))
    if V2L["slow"]:
        from .v2like import slow_features
        f.update(slow_features(st, r))
    if V2L["day"] and "DC" in B:
        f.update({f"day_{k}": B["DC"][..., i] for i, k in enumerate(DAY_COLS)})
    if ALIVE["on"]:
        a, geo["alive"] = alive_features(qm); f.update(a)
    if prior is not None:
        for k, v in prior.items():
            f[k] = v
    names = list(f)
    X = np.stack([np.asarray(f[k], np.float32) for k in names], -1)
    return X, names, geo


def _down(a: np.ndarray, K: int) -> np.ndarray:
    """[n, L] -> [n, L, K]: value at link j+o (o = 0..K-1), NaN beyond the corridor end."""
    L = a.shape[1]; J = np.arange(L)[:, None] + np.arange(K)[None]
    out = a[:, np.minimum(J, L - 1)].astype(np.float64); out[:, J >= L] = np.nan
    return out


def phys_features(B, net, st, s, q, r) -> dict:
    """Onset physics without location identity: density/k_crit level and trend, speed-drop slopes, upstream
    arrival and on-ramp growth vs observed discharge capacity c (train p99.5 flow, scenario-invariant), lane drop,
    and the onset extent: strongest downstream bottleneck within K_EXT links (stress, slots to breakdown, km away)
    and the tail speed it would need to reach this link by T+30."""
    x, c = st["x"], st["c"]; f = {}
    with np.errstate(all="ignore"):
        k = q / np.maximum(s, 1) / net["k_crit"]
        f["k_T"] = k[:, 12]; f["k_slope"] = np.nanmean(k[:, 10:13], 1) - np.nanmean(k[:, 0:3], 1)
        f["k_down_max"] = np.nanmax(_down(k[:, 12], 4), 2)
        f["dr3"] = r[:, 12] - r[:, 9]; f["dr6"] = r[:, 12] - r[:, 6]; f["dr12"] = r[:, 12] - r[:, 0]
        q3, q0 = np.nanmean(q[:, 10:13], 1), np.nanmean(q[:, 0:3], 1)
        u = q3 / c; tt = np.clip((c - q3) / np.maximum((q3 - q0) / 10, 1.0), -50, 200)
        up = np.stack([np.roll(q3 - q0, j, 1) for j in range(1, 6)], 0); up[:, :, :5] = np.nan
        f["up_growth"] = np.nanmean(up, 0) / st["ce4"]
        ron = np.nan_to_num(B["RON"]); r3, r0 = ron[:, 10:13].mean(1), ron[:, 0:3].mean(1)
        f["ron_down"] = np.nansum(_down(r3, 4), 2) / st["ce4"]; f["ron_growth"] = np.nansum(_down(r3 - r0, 4), 2) / st["ce4"]
        f["demand_ce4"] = (np.nanmean(np.stack([np.roll(q3, j, 1) for j in (1, 2)]), 0) + f["ron_down"] * st["ce4"]) / st["ce4"]
        f["lane_drop"] = np.broadcast_to(np.nanmin(_down(net["lanes"][None].astype(float), 4), 2)[0] / net["lanes"], u.shape)
        U = _down(u, K_EXT); TT = _down(tt, K_EXT); D = _down(x[None], K_EXT)[0] - x[:, None]
        b = np.nanargmax(np.where(np.isnan(U), -np.inf, U), 2)
        ii = np.arange(len(u))[:, None]; jj = np.arange(u.shape[1])[None]
        f["ext_u"] = U[ii, jj, b]; f["ext_ttb"] = TT[ii, jj, b]; f["ext_d"] = D[jj, b]
        f["ext_req_w"] = f["ext_d"] / np.maximum(6 - f["ext_ttb"], 0.5)  # km per slot the tail must travel
        f["ext_n_hot"] = np.nansum(U > 0.9, 2)
    return f


def q2_features(net, st, s, q, qm, geo) -> dict:
    """Q2 (see report, Sec. III-C): structure of the next links, vehicle balance dN = sum(q / v * len) over 15 / 30 / 60 min on the
    corridor and on the 6 links ending at j (storage-normalised), inflow to the tail vs 0.69 * capacity of the head,
    upstream flow / capacity, density / k_crit downstream, and the tail movement over 6 slots. Observed cells <= T only
    (s, q are the forward-filled history; a missing cell carries the last observed value)."""
    x, L = st["x"], st["L"]; cap, lanes, kc, kj, ln = net["cap"], net["lanes"], net["k_crit"], net["k_jam"], net["length"]
    n = s.shape[0]; f = {}
    nxt = np.minimum(np.arange(L) + 1, L - 1)
    f["cap_next_r"] = np.broadcast_to(cap[nxt] / cap, (n, L)); f["lanes_next_d"] = np.broadcast_to(lanes[nxt] - lanes, (n, L))
    f["kcrit_next_r"] = np.broadcast_to(kc[nxt] / kc, (n, L))
    K = 12; J = np.arange(L)[:, None] + np.arange(1, K + 1)[None]; Jc = np.minimum(J, L - 1)
    drop = (cap[Jc] < 0.9 * cap[:, None]) & (J < L)
    d = np.where(drop.any(1), x[Jc[np.arange(L), drop.argmax(1)]] - x, 30.0)
    f["d_capdrop"] = np.broadcast_to(d, (n, L))
    with np.errstate(all="ignore"):
        N = q / np.maximum(s, 1.0) * ln                       # [n, 13, L] vehicles on the link
        store = ln * kj; cstore = store.sum()
        for k, nm in ((3, "15"), (6, "30"), (12, "60")):
            dN = N[:, 12] - N[:, 12 - k]                        # [n, L]
            f[f"dN_cor_{nm}"] = np.broadcast_to(np.nansum(dN, 1)[:, None] / cstore, (n, L))
            cs = np.concatenate([np.zeros((n, 1)), np.nancumsum(dN, 1)], 1); ss = np.concatenate([[0.0], np.cumsum(store)])
            j = np.arange(L); a = np.maximum(j - 5, 0)
            f[f"dN_seg_{nm}"] = (cs[:, j + 1] - cs[:, a]) / (ss[j + 1] - ss[a])
        qT = q[:, 12]; up1 = np.roll(qT, 1, 1); up1[:, 0] = np.nan
        f["q_up_cap"] = up1 / cap
        # inflow to the tail of the nearest cluster (or to this link when none) vs 0.69 * capacity of its head
        tail_j = np.where(np.isfinite(geo["tail_x"]), np.searchsorted(x, np.nan_to_num(geo["tail_x"], nan=0.0)), np.arange(L)[None])
        head_j = np.where(np.isfinite(geo["head_x"]), np.searchsorted(x, np.nan_to_num(geo["head_x"], nan=0.0)), np.arange(L)[None])
        tail_j = np.clip(tail_j, 0, L - 1).astype(int); head_j = np.clip(head_j, 0, L - 1).astype(int)
        ii = np.arange(n)[:, None]
        qin = np.where(tail_j > 0, qT[ii, np.maximum(tail_j - 1, 0)], np.nan)
        f["qin_tail_r"] = qin / (0.69 * cap[head_j])
        kk = q / np.maximum(s, 1.0) / kc
        f["k_down_max3"] = np.nanmax(_down(kk[:, 12], 4)[:, :, 1:], 2)
        f["k_T_r"] = kk[:, 12]
        # tail movement over 6 slots (km per slot, + = downstream): tail at T vs tail of the run touching it at T-30
        vt6 = np.full((n, L), np.nan)
        for i in range(n):
            if not np.isfinite(geo["tail_x"][i]).any():
                continue
            runs6 = _runs(qm[i, 6])
            for tx in np.unique(geo["tail_x"][i][np.isfinite(geo["tail_x"][i])]):
                j = geo["tail_x"][i] == tx; a = int(np.searchsorted(x, tx)); b = int(np.searchsorted(x, geo["head_x"][i][j][0]))
                for (a2, b2) in runs6:
                    if b2 >= a - 1 and a2 <= b + 1:
                        vt6[i, j] = (tx - x[a2]) / 6; break
        f["c_vtail6"] = vt6
    return f


def alive_features(qm: np.ndarray) -> tuple[dict, dict]:
    """Queue trajectory over the history (qm [n, 13, L]: forward-filled queued masks of the slots <= T): corridor
    queued-link count at T and its change over 15 / 30 / 60 min, links newly queued / freed over 15 and 30 min, peak
    count and distance to it; per link the queued / free run length up to T and the 15-min flip. Also returns the
    per-window count and 15-min slope for the step columns (alive_h)."""
    n, _, L = qm.shape; cnt = qm.sum(2).astype(np.float32)
    bc = lambda v: np.broadcast_to(np.asarray(v, np.float32)[:, None], (n, L))
    f = {"al_cnt_T": bc(cnt[:, 12]), "al_d15": bc(cnt[:, 12] - cnt[:, 9]), "al_d30": bc(cnt[:, 12] - cnt[:, 6]),
         "al_d60": bc(cnt[:, 12] - cnt[:, 0]), "al_peak": bc(cnt.max(1)), "al_dpeak": bc(cnt[:, 12] - cnt.max(1)),
         "al_new15": bc((qm[:, 12] & ~qm[:, 9]).sum(1)), "al_freed15": bc((~qm[:, 12] & qm[:, 9]).sum(1)),
         "al_new30": bc((qm[:, 12] & ~qm[:, 6]).sum(1)), "al_freed30": bc((~qm[:, 12] & qm[:, 6]).sum(1))}
    qrun = np.zeros((n, L), np.float32); frun = np.zeros((n, L), np.float32); aq = np.ones((n, L), bool); af = aq.copy()
    for k in range(12, -1, -1):
        aq &= qm[:, k]; af &= ~qm[:, k]; qrun += aq; frun += af
    f["al_qrun"] = qrun; f["al_frun"] = frun; f["al_flip15"] = qm[:, 12].astype(np.float32) - qm[:, 9]
    return f, dict(cnt=cnt[:, 12], slope=(cnt[:, 12] - cnt[:, 9]) / 3)


def alive_h(aw: dict, h: int, L: int) -> np.ndarray:
    """[n, L, 3] step columns: projected queued-link count at step h (count at T + 15-min slope x h, floor 0), the
    alive flag of that projection, and its fraction of the count at T ('alive / dissolved at the step')."""
    proj = np.maximum(aw["cnt"] + aw["slope"] * h, 0)
    cols = np.stack([proj, proj > 0.5, proj / np.maximum(aw["cnt"], 1)], -1).astype(np.float32)
    return np.broadcast_to(cols[:, None], (len(proj), L, 3))


def horizon_features(geo: dict, x: np.ndarray, h: int) -> np.ndarray:
    """Signed distance (km) of each link inside the predicted cluster at step h (1..6): >0 inside."""
    tail_s = geo["tail_x"] + geo["w"] * h           # shockwave tail
    tail_k = geo["tail_x"] + geo["vt"] * h          # kinematic tail
    hx = geo["head_x"]
    e_s = np.minimum(x[None] - tail_s, hx - x[None] + 0.05)
    e_k = np.minimum(x[None] - tail_k, hx - x[None] + 0.05)
    e_p = np.minimum(x[None] - geo["tail_x"], hx - x[None] + 0.05)
    return np.stack([e_s, e_k, e_p], -1).astype(np.float32)
