"""Leakage test: overwriting every slot after T (mainline, ramps, quality) must not change features."""
import numpy as np
import pandas as pd
from tfb.t2b import build, data, model as M
from tfb.t2b.evaluate import load_train

import pytest
from tfb import io
NEEDS_CACHE = pytest.mark.skipif(not (io.OUTPUTS / "t2b" / "cache").exists(),
                                 reason="needs the t2b training cache (run.py stage t2b_base)")


@NEEDS_CACHE
def test_features_ignore_future():
    panel = "D7_I210_W"
    d = dict(data.load(panel, "train"))
    t = 288 * 40 + 17 * 12  # an afternoon origin
    B0 = build._pack(d, np.array([t]))
    rng = np.random.default_rng(0)
    for k in ("speed", "flow", "mspeed", "mflow", "ron", "roff"):
        a = d[k].copy(); a[t + 1:] = rng.uniform(0, 200, a[t + 1:].shape); d[k] = a
    B1 = build._pack(d, np.array([t]))
    net = load_train(panel)["net"]
    meta = pd.DataFrame({"t": [t], "date": [d["dates"][t // 288]]})
    pr = M.priors(panel, np.arange(30))
    for cond in ("queue_onset", "queue_ongoing"):
        X0, _, _ = M.panel_rows(dict(zip(("S", "Q", "RON", "ROFF"), B0)), net, meta, pr, cond)
        X1, _, _ = M.panel_rows(dict(zip(("S", "Q", "RON", "ROFF"), B1)), net, meta, pr, cond)
        np.testing.assert_array_equal(X0, X1)


@NEEDS_CACHE
def test_v2like_features_ignore_future():
    """The slow-link and day-context columns (tfb.t2b.v2like) read nothing after T either."""
    from tfb.t2b import features as FT
    FT.V2L["slow"] = True; FT.V2L["day"] = True
    try:
        test_features_ignore_future()
    finally:
        FT.V2L["slow"] = False; FT.V2L["day"] = False




def test_history_matches_release():
    """Train history packed from raw arrays equals the released window_history.parquet."""
    panel = "D7_I405_S"
    d = data.load(panel, "train")
    wi, hs, _ = data.released_windows(panel, "train")
    ts = np.array([data.t_index(d, s) for s in wi.forecast_origin])
    S = build._pack(d, ts)[0]
    np.testing.assert_array_equal(np.isnan(S[:, :12]), np.isnan(hs))
    assert np.nanmax(np.abs(S[:, :12] - hs)) == 0


def test_iou_and_decode():
    from tfb.t2b.evaluate import window_iou
    Y = np.zeros((3, 6, 4), bool); E = np.ones_like(Y)
    Y[0, 5, 1] = True; Y[1, 5, :2] = True              # window 2: empty truth
    P = np.zeros_like(Y); P[0, 5, 1:3] = True
    np.testing.assert_allclose(window_iou(P, Y, E), [0.5, 0.0, 1.0])
    E2 = E.copy(); E2[2] = False
    assert window_iou(P, Y, E2)[2] == 0.0               # no eligible cells -> 0 (official scorer)
    pr = np.zeros((2, 6, 4)); pr[0, 5, :2] = [0.9, 0.8]; pr[1, 5, 0] = 0.01
    D = M.decode(pr)
    assert D[0, 5, :2].all() and D[0].sum() == 2 and D[1].sum() == 0


def test_test_month_priors_ignore_future():
    """Test-month priors for a window must not change when masked data at/after its day (days mode)
    or after T (sameday mode) is overwritten."""
    from tfb.t2b import tm
    panel, split = "D7_I210_W", "validation"
    d = data.load(panel, split)
    t = 288 * 3 + 15 * 12 + 7
    base = M.priors(panel, np.arange(30))
    a = {m: tm.window_priors(panel, split, np.array([t]), base, m) for m in ("days", "sameday")}
    sp = d["mspeed"].copy()
    try:
        d["mspeed"][t + 1:] = 5.0                     # everything after T becomes queued
        tm.stats.cache_clear()
        b = {m: tm.window_priors(panel, split, np.array([t]), base, m) for m in ("days", "sameday")}
    finally:
        d["mspeed"][:] = sp; tm.stats.cache_clear()
    for m in a:
        np.testing.assert_array_equal(a[m]["ph"], b[m]["ph"]); np.testing.assert_array_equal(a[m]["onset"], b[m]["onset"])
