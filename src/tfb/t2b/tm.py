"""Test-month (transductive, causal) location priors from a split's released MASKED mainline layer.

For a window on day d at origin t: per-link x hour queue frequency and onset-site counts over
the same calendar month's days strictly before d (mode "days"), optionally plus same-day hours
that end at or before T (mode "sameday"); "days_ph" adapts only the hourly frequency. Shrunk toward the train prior:
  ph = (n_queued + kappa * ph_train) / (n_observed + kappa)
  onset = (counts + kappa_o * onset_train) / (sum(counts) + kappa_o)
Masked cells (Task 1 targets, blackouts) are simply unobserved. Nothing after T is read
(tests/t2b/test_causal.py::test_test_month_priors_ignore_future). Empirical capacity stays the train value (scenario-invariant).
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
import pandas as pd
from . import data

DAY = 288


@lru_cache(maxsize=24)
def stats(panel: str, split: str):
    """[D, L, 24] queued / observed counts and onset first-slot counts from the masked layer."""
    d = data.load(panel, split); sp = d["mspeed"]; L = sp.shape[1]; D = len(sp) // DAY
    obs = np.isfinite(sp); q = obs & (np.nan_to_num(sp, nan=999) <= d["vcut"])
    num = q.reshape(D, 24, 12, L).sum(2).transpose(0, 2, 1).astype(np.float32)
    den = obs.reshape(D, 24, 12, L).sum(2).transpose(0, 2, 1).astype(np.float32)
    ons = np.zeros((D, L, 24), np.float32)
    qd = q.reshape(D, DAY, L); anyq = qd.any(2)
    for dd in range(D):
        for t in range(18, DAY):
            if anyq[dd, t] and not anyq[dd, t - 18:t].any():
                ons[dd, :, t // 12] += qd[dd, t]
    months = pd.to_datetime(pd.Series(d["dates"])).dt.strftime("%Y-%m").to_numpy()
    return num, den, ons, months


def window_priors(panel: str, split: str, ts: np.ndarray, base: dict, mode: str = "days",
                  kappa: float = 24.0, kappa_o: float = 20.0) -> dict:
    """Per-window priors {ph [n, L, 24], onset [n, L], cemp} blending test-month evidence into `base`."""
    num, den, ons, months = stats(panel, split)
    n = len(ts); L = num.shape[1]
    ph = np.empty((n, L, 24), np.float32); on = np.empty((n, L), np.float32)
    for i, t in enumerate(ts):
        day, hr = t // DAY, (t % DAY) // 12
        days = np.flatnonzero((months == months[day]) & (np.arange(len(months)) < day))
        nq, no, oc = num[days].sum(0), den[days].sum(0), ons[days].sum((0, 2))
        if mode.startswith("sameday") and hr > 0:  # completed hours of the same day only (all end <= T)
            nq = nq.copy(); no = no.copy(); oc = oc.copy()
            nq[:, :hr] += num[day, :, :hr]; no[:, :hr] += den[day, :, :hr]; oc += ons[day, :, :hr].sum(1)
        ph[i] = (nq + kappa * base["ph"]) / (no + kappa)
        # "*_ph" modes: only the hourly queue frequency adapts; the onset-site prior stays train
        on[i] = base["onset"] if mode.endswith("_ph") else (oc + kappa_o * base["onset"]) / (oc.sum() + kappa_o)
    return {"ph": ph, "onset": on, "cemp": base["cemp"]}
