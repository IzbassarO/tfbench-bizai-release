"""Physics post-processing of Task 1 target cells (observed cells never change).

Every transform keeps the congestion class (v < 0.6 v_f) of each cell and keeps flow >= 100 vph.
  fd_soft(alpha, keep)     q' = q + alpha (q_fd(k) - q), k = q/v; keep='k': v' = q'/k (N unchanged), keep='v': v unchanged
"""
from __future__ import annotations
import numpy as np


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
    cells next to an observed one in time (obs [T, L])."""
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
