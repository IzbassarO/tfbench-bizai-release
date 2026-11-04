"""Local Task 3 proxy on train, faithful to src/task3/score_task3.py (Mode A, all ten panels):

per panel x regime (the regime's days only): field = released truth on eligible cells, submission on template targets;
  S_FD  exactly as the scorer (per lane, whole field, 50-vph trap on submitted cells);
  S_LWR = max(0, 1 - min(1, sum|dN - rhs| / sum|rhs|)) over forward 5-min transitions t->t+1 of eligible cells whose start
          row has valid attached ramps (no attached ramp of a type = valid), N = q/max(v,1)*length.
  rhs: the organizer's boundary flows are "projected onto the released observations, so a participant who reproduces
  what was published satisfies the conservation identity exactly" -> rhs := dN of the truth field (exact up to the
  organizer's own residual; the organizer's perfect answer scores S_phys 0.96, so the LB sits a little below this proxy).
S_phys = (S_FD + 2 S_LWR)/3, mean over regimes (callers average panels).
"""
from __future__ import annotations
import numpy as np
from .. import io


def fd_score(pn, speed: np.ndarray, flow: np.ndarray, rows: np.ndarray, submitted: np.ndarray) -> tuple[float, float]:
    """(S_FD, share of submitted cells < 50 vph) on eligible cells of `rows` (time mask)."""
    el = pn.elig[rows] & np.isfinite(speed[rows]) & np.isfinite(flow[rows])
    v = speed[rows][el].astype(np.float64); q = flow[rows][el].astype(np.float64)
    L = np.broadcast_to(np.arange(len(pn.links)), el.shape)[el]
    lanes, vf, cap, kc, kj = (a[L].astype(np.float64) for a in (pn.lanes, pn.v_f, pn.cap, pn.k_crit, pn.k_jam))
    k = q / np.maximum(v, 1.0)
    ql, kl, capl, kcl, kjl = q / lanes, k / lanes, cap / lanes, kc / lanes, kj / lanes
    qfd = np.where(kl <= kcl, vf * kl, capl / np.maximum(kjl - kcl, 1e-9) * np.maximum(kjl - kl, 0))
    s_fd = max(0.0, 1 - np.abs(ql - qfd).sum() / (np.abs(ql).sum() + 1e-9))
    sub = submitted[rows][el]
    lt50 = float((q[sub] < 50).mean()) if sub.any() else 0.0
    return (0.0 if lt50 > 0.2 else float(s_fd)), lt50


def ramp_ok(pn) -> np.ndarray:
    """[T,L] Mode A validity: every ramp type attached in the topology has a valid released reading on that row."""
    topo = io.topology(pn.panel).drop_duplicates("link_id").set_index("link_id").reindex(pn.links)
    ok = np.ones(pn.mspeed.shape, bool)
    for col, is_on in (("on_ramp_link_ids", True), ("off_ramp_link_ids", False)):
        attached = topo[col].fillna("").astype(str).str.len().to_numpy() > 0
        have = np.zeros(pn.mspeed.shape, bool)
        for r, j in enumerate(pn.ramp_link):
            if j >= 0 and pn.ramp_on[r] == is_on:
                have[:, j] |= np.isfinite(pn.rflow[:, r])
        ok &= have | ~attached[None, :]
    return ok


def lwr_score(pn, speed, flow, rows, rok=None) -> float:
    if rok is None:
        rok = ramp_ok(pn)
    N = lambda v, q: (q / np.maximum(v, 1.0)) * pn.length[None, :]
    valid = pn.elig & rows[:, None] & np.isfinite(speed) & np.isfinite(flow)
    t = np.nonzero(rows[:-1])[0]
    keep = valid[t] & valid[t + 1] & rok[t]
    dn = N(speed[t + 1], flow[t + 1]) - N(speed[t], flow[t])
    rhs = N(pn.speed[t + 1], pn.flow[t + 1]) - N(pn.speed[t], pn.flow[t])
    e = np.abs(dn - rhs)[keep].sum() / (np.abs(rhs)[keep].sum() + 1e-9)
    return float(max(0.0, 1 - min(1.0, e)))


def evaluate(pn, pred_speed, pred_flow, rows: np.ndarray) -> dict:
    """Reconstructed field over time-rows `rows` (bool [T]), scored per regime then averaged like the organizer."""
    day_reg = np.repeat(pn.regime, io.STEPS_PER_DAY); rok = ramp_ok(pn); out = []
    for r in range(len(io.REGIMES)):
        rr = rows & (day_reg == r)
        if not rr.any():
            continue
        sub = pn.target & rr[:, None]
        sp = np.where(sub, pred_speed, pn.speed); fl = np.where(sub, pred_flow, pn.flow)
        s_fd, lt50 = fd_score(pn, sp, fl, rr, sub)
        s_lwr = lwr_score(pn, sp, fl, rr, rok)
        out.append((s_fd, lt50, s_lwr, (s_fd + 2 * s_lwr) / 3))
    m = np.mean(out, axis=0)
    return {"S_FD": m[0], "lt50": max(o[1] for o in out), "S_LWR_proxy": m[2], "S_phys_proxy": m[3]}
