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

V2L = {"slow": False}  # slow-link ongoing features (tfb.t2b.v2like)


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
    if V2L["slow"]:
        from .v2like import slow_features
        f.update(slow_features(st, r))
    if prior is not None:
        for k, v in prior.items():
            f[k] = v
    names = list(f)
    X = np.stack([np.asarray(f[k], np.float32) for k in names], -1)
    return X, names, geo


def horizon_features(geo: dict, x: np.ndarray, h: int) -> np.ndarray:
    """Signed distance (km) of each link inside the predicted cluster at step h (1..6): >0 inside."""
    tail_s = geo["tail_x"] + geo["w"] * h           # shockwave tail
    tail_k = geo["tail_x"] + geo["vt"] * h          # kinematic tail
    hx = geo["head_x"]
    e_s = np.minimum(x[None] - tail_s, hx - x[None] + 0.05)
    e_k = np.minimum(x[None] - tail_k, hx - x[None] + 0.05)
    e_p = np.minimum(x[None] - geo["tail_x"], hx - x[None] + 0.05)
    return np.stack([e_s, e_k, e_p], -1).astype(np.float32)
