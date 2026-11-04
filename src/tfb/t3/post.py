"""Physics post-processing of Task 1 target cells (observed cells never change).

Every transform keeps the congestion class (v < 0.6 v_f) of each cell and keeps flow >= 100 vph.
  fd_soft(alpha, keep)     q' = q + alpha (q_fd(k) - q), k = q/v; keep='k': v' = q'/k (N unchanged), keep='v': v unchanged
  conserve(alpha, passes)  N <- N + alpha (N_cons - N), N_cons = mean of N(t-1)+rhs(t-1) and N(t+1)-rhs(t), rhs from
                           sensored topology neighbours + valid ramps (links with an unsensored neighbour or an invalid
                           attached ramp are skipped); free cells move flow, congested cells move speed; never across a
                           free/congested boundary in t-1..t+1
  reconcile(kd, a, gated)  v' = a v + (1-a) q/k, q' = a q + (1-a) k v with k from the density model; gated: v < 0.6 v_f only
"""
from __future__ import annotations
import numpy as np
from .. import io

DT_H = 5 / 60


def _cong(sp, pn):
    return sp < 0.6 * pn.v_f[None, :]


def _guard(sp, fl, sp2, fl2, tgt, pn):
    ok = tgt & np.isfinite(sp2) & np.isfinite(fl2) & (_cong(sp, pn) == _cong(sp2, pn)) & (fl2 >= 100) & (sp2 >= 3)
    return np.where(ok, np.minimum(sp2, 120.75), sp).astype(np.float32), np.where(ok, fl2, fl).astype(np.float32)


def edge_cells(obs):
    """[T, L]: cells next to an observed one in time (t - 1 or t + 1)."""
    e = np.zeros(obs.shape, bool); e[1:] |= obs[:-1]; e[:-1] |= obs[1:]
    return e


def alpha_cells(sp, pn, a, obs):
    """pp2 alpha per cell from a = {cong, free[, cong_e, free_e]} (congested: v < 0.6 v_f); the *_e alphas apply at
    cells next to an observed one in time (obs [T, L], tfb.t1.cellkind)."""
    cong = _cong(sp, pn); al = np.where(cong, a["cong"], a["free"])
    if "cong_e" in a: al = np.where(edge_cells(obs), np.where(cong, a["cong_e"], a["free_e"]), al)
    return al


def fd_soft(sp, fl, tgt, pn, alpha, keep="k"):
    k = fl / np.maximum(sp, 1)
    kc, kj, cap, vf = pn.k_crit[None, :], pn.k_jam[None, :], pn.cap[None, :], pn.v_f[None, :]
    qfd = np.where(k <= kc, vf * k, cap / np.maximum(kj - kc, 1e-9) * np.maximum(kj - k, 0))
    fl2 = fl + alpha * (qfd - fl)
    sp2 = fl2 / np.maximum(k, 1e-6) if keep == "k" else sp
    return _guard(sp, fl, sp2, fl2, tgt, pn)


def reconcile(sp, fl, kd, tgt, pn, a=0.75, gated=True):
    k = kd * pn.lanes[None, :]
    sel = tgt & (_cong(sp, pn) if gated else True)
    return _guard(sp, fl, a * sp + (1 - a) * fl / k, a * fl + (1 - a) * k * sp, sel, pn)


def conserve(sp, fl, tgt, pn, rok, alpha, passes):
    sp, fl = sp.astype(np.float64).copy(), fl.astype(np.float64).copy()
    topo = io.topology(pn.panel).drop_duplicates("link_id").set_index("link_id").reindex(pn.links)
    pos = {l: i for i, l in enumerate(pn.links)}
    ids = lambda s: [x for x in str(s).split(";") if x and x != "nan"]
    nb = [(ids(a), ids(b)) for a, b in zip(topo.incoming_link_ids, topo.outgoing_link_ids)]
    ramp = np.zeros(sp.shape)
    for r, j in enumerate(pn.ramp_link):
        if j >= 0:
            ramp[:, j] += np.nan_to_num(pn.rflow[:, r]) * (1 if pn.ramp_on[r] else -1)
    cong = _cong(sp, pn); length = pn.length
    stable = np.zeros(sp.shape, bool); stable[1:-1] = (cong[:-2] == cong[1:-1]) & (cong[1:-1] == cong[2:])
    for _ in range(passes):
        for j, (inc, out) in enumerate(nb):
            if not all(x in pos for x in inc + out) or not tgt[:, j].any():
                continue
            rhs = DT_H * (sum((fl[:, pos[x]] for x in inc), np.zeros(len(fl))) - sum((fl[:, pos[x]] for x in out), np.zeros(len(fl))) + ramp[:, j])
            rhs = np.where(rok[:, j], rhs, np.nan)
            N = fl[:, j] / np.maximum(sp[:, j], 1) * length[j]
            fwd = np.full(len(N), np.nan); bwd = fwd.copy()
            fwd[1:] = N[:-1] + rhs[:-1]; bwd[:-1] = N[1:] - rhs[:-1]
            with np.errstate(all="ignore"):
                Nc = np.nanmean(np.c_[fwd, bwd], axis=1)
            upd = tgt[:, j] & stable[:, j] & np.isfinite(Nc)
            Nn = np.maximum(N + alpha * (Nc - N), 1e-3)
            k = Nn / length[j]
            fl[:, j] = np.where(upd & ~cong[:, j], k * sp[:, j], fl[:, j])
            sp[:, j] = np.where(upd & cong[:, j], np.minimum(fl[:, j] / k, 0.6 * pn.v_f[j] - 1e-3), sp[:, j])
    return np.clip(sp, 3, 120.75).astype(np.float32), np.maximum(fl, 100).astype(np.float32)
