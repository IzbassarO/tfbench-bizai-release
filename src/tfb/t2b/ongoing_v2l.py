"""Ongoing member "v2l" of family B, blended with family A v2 on the ongoing rows.

Member: noloc ongoing features + FT.V2L slow columns (tfb.t2b.v2like), training windows = the base cache + every admissible
ongoing window of the given selector (tfb.t2b.harness), all train months, 3 seeds, denoised labels, no location priors.
Ongoing decode = exact expected-IoU top-m of W * v2 + (1 - W) * v2l over the 6 horizons (W = 0.5).
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from .. import io
from ..t2.decode import expected_iou_decode
from . import codex, cv, evaluate as ev, features as FT, harness as HS, hybrid as H, model as M, run as R
from .ladder import T2, _prob_grid
from .v2like import build_cache_hw

OUT = io.OUTPUTS / "t2b" / "v2l"
DEPLOYED = "queue_onset_Ahyb_mix_clean_E25.csv"


def log(s):
    print(f"[v2l {time.strftime('%H:%M:%S')}] {s}", flush=True)


def fit_final(seeds=H.SEEDS, cache="cache_hw", windows=None):
    """v2l member on all train months; cache / windows: the training cache and the admissible windows it adds."""
    build_cache_hw(log, windows, cache)
    FT.V2L["slow"] = True
    try:
        ms = H.noloc_fit(R.ALL_MONTHS, cache, log=log, cond=HS.OG, extra=False, seeds=seeds, drop=tuple(M.LOC))
        P = H.released_noloc(ms, cond=HS.OG, h=None)
        p0 = io.T2_PANELS[0]; B = ev.load_train(p0, cache)
        names = M.panel_rows(cv._sub(B, np.array([0])), B["net"], B["meta"].iloc[[0]], M.priors(p0, cv._days(None, R.ALL_MONTHS)), HS.OG)[1]
    finally:
        FT.V2L["slow"] = False
    keep = [names[i] for i in ms[0][1]]
    return ms, P, keep


def write_queue(P, w: float, name: str) -> tuple[dict, pd.DataFrame]:
    """outputs/t2/<name>: onset rows of the deployed file, ongoing rows = decode(w v2 + (1 - w) v2l)."""
    dep = codex.submission(T2 / DEPLOYED); pv2 = pd.read_parquet(T2 / "probs_v2.parquet").rename(columns={"queue_prob": "queue_pred"})
    parts, ch = [], []
    for p in io.T2_PANELS:
        for s in ("validation", "private"):
            wi = pd.read_csv(io.t2_dir(p, s) / "window_index.csv", dtype=str); on = (wi.condition == HS.ON).to_numpy()
            G = codex.grid(dep, p, s, wi); D = G.copy()
            C2 = _prob_grid(pv2, p, s, wi)[~on]; nl = P[(p, s)]; assert nl.shape == C2.shape, (nl.shape, C2.shape)
            D[~on] = np.stack([expected_iou_decode(x, force_nonempty=True) for x in w * C2 + (1 - w) * nl]).astype(bool)
            parts.append(R.to_rows(D, p, s, wi))
            ch.append(dict(panel=p, split=s, ongoing_windows=int((~on).sum()), ongoing_windows_changed=int((D[~on] != G[~on]).any((1, 2)).sum()),
                           ongoing_cells_changed=int((D[~on] != G[~on]).sum()), onset_cells_changed=int((D[on] != G[on]).sum()),
                           positives_new=int(D[~on].sum()), positives_old=int(G[~on].sum())))
    pd.concat(parts, ignore_index=True).to_csv(T2 / name, index=False)
    return R.verify([f"../t2/{name}"]), pd.DataFrame(ch)
