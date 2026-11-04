"""Task 2 online check: no recipe member may read a released value after the forecast origin T.

For every released validation/private Task 2 window, rebuilds each member's deployed per-window model inputs
twice: from the released data, and from a copy in which every row after T is blanked (--fill nan) or replaced
by random values (--fill random). The trained models are fixed and deterministic, so identical inputs mean
identical probabilities and identical queue_pred. Members (configs/finals.yaml):
  v2   family A v2 features (tfb.t2.v2_predict path), ongoing member (weight 0.5 with v2l)
  v4   family A v4 onset features (tfb.t2.v4_hyb.released_features path), onset A_hyb member (0.5)
  t2b  family B noloc rows (tfb.t2b.hybrid.released_noloc path), onset A_hyb member (0.5, on hybrid labels)
  t2b_v2l  the same rows with the slow-link features on (tfb.t2b.v2like): ongoing member v2l (0.5 with v2)
--before60 also blanks everything before T-60 (the released window history stays). It reports which members
change; it is not a failure, because the rule only constrains data after T.
Training-side inputs (train-split models, train priors, hybrid labels) are outside this check (README, "Online rule").

usage: PYTHONPATH=src python scripts/check_online.py [--control] [--fill nan|random] [--before60] [--panels P ...]
exit 1 if any member's inputs change when data after T change.
"""
from __future__ import annotations
import argparse
import dataclasses
import sys
import numpy as np
from tfb import io
from tfb.t2 import io as CIO
from tfb.t2.context import build_day_context
from tfb.t2.v2_data import extra_features
from tfb.t2.features import build_features
from tfb.t2.run import enrich_features
from tfb.t2.v4_predict import build_window_features
from contextlib import contextmanager
from tfb.t2b import build, cv, data as TD, evaluate as ev, features as FT, model as M, run as TR

OUT = io.OUTPUTS / "t2"
MEMBERS = ("v2", "v4", "t2b", "t2b_v2l")   # the recipe members of both finals (configs/finals.yaml)


@contextmanager
def v2l_enabled():
    """Slow-link features of the v2l ongoing member (tfb.t2b.v2like; see report, Sec. III-C, candidate cand_a2692183_og50v2l)."""
    FT.V2L["slow"] = True
    try:
        yield
    finally:
        FT.V2L["slow"] = False


FIVE = np.timedelta64(5, "m")


def noise(rng, a, lo, hi):
    return rng.uniform(lo, hi, a.shape).astype(a.dtype) if a.dtype.kind == "f" else rng.random(a.shape) < 0.5


def blank(a, rows, fill, rng, lo=0.0, hi=1.0):
    """Copy of a with rows (bool [T]) replaced: NaN / False, or random values in [lo, hi)."""
    a = np.array(a, copy=True)
    if fill == "nan": a[rows] = np.nan if a.dtype.kind == "f" else False
    else: a[rows] = noise(rng, a[rows], lo, hi)
    return a


def codex_inputs(cur, hist, w, net, pid, prior, onset_prior, capacity):
    local = CIO.history_arrays(hist, w.forecast_origin, net, cur)
    oi = int((local.timestamps[-1] - cur.timestamps[0]) / FIVE)
    x, names = build_features(local.speed, local.flow, local.occupancy, origin=12, timestamps=local.timestamps,
                              ramp_on=local.ramp_on, location_prior=prior, panel_id=pid, **net.feature_kwargs())
    ctx, ctx_names = build_day_context(cur.speed, cur.flow, cur.timestamps, origin=oi, vf=net.vf)
    x, names = enrich_features(x, names, local, net, onset_prior=onset_prior, day_context=ctx, day_names=ctx_names)
    extra, extra_names = extra_features(local, net)
    v2 = (np.column_stack([x, extra]), names + extra_names)
    v4 = build_window_features(local, cur, net, location_prior=prior, onset_prior=onset_prior,
                               capacity=capacity, panel_id=pid, origin_idx=oi)
    return {"v2": v2, "v4": v4}


def t2b_inputs(d, t, hs, hf, meta, net, pr, cond):
    S, Q, RON, ROFF = build._pack(d, np.array([t]), hs[None], hf[None]); B = dict(S=S, Q=Q, RON=RON, ROFF=ROFF)
    out = {"t2b": M.panel_rows(B, net, meta, pr, cond)[:2]}
    with v2l_enabled():
        out["t2b_v2l"] = M.panel_rows(B, net, meta, pr, cond)[:2]
    return out


def diff(a, b):
    """First differing feature name, or None (bitwise, NaN == NaN)."""
    (xa, na), (xb, nb) = a, b
    if list(na) != list(nb) or xa.shape != xb.shape: return "schema"
    same = (xa == xb) | (np.isnan(xa) & np.isnan(xb))
    bad = np.flatnonzero(~same.all(0))
    return None if not len(bad) else str(na[bad[0]]) if bad[0] < len(na) else f"col{bad[0]}"


def check(panels, fill, before60, seed=0, control=False):
    rng = np.random.default_rng(seed); root = CIO.resolve_root(io.RELEASE)
    codex_panels = CIO.panel_names(root); rows, failures = [], 0
    base = OUT / "datasets/final_enhanced_release"
    for panel in panels:
        net = CIO.read_network(root, panel); pid = codex_panels.index(panel)
        prior, onset_prior = np.load(base / f"{panel}_prior.npy"), np.load(base / f"{panel}_onset_prior.npy")
        with np.load(OUT / "v4/data/final" / f"{panel}_capacity.npz") as z: capacity = {k: z[k] for k in z.files}
        pr = M.priors(panel, cv._days(None, TR.ALL_MONTHS)); bnet = ev.load_train(panel)["net"]
        for split in ("validation", "private"):
            index, history, _ = CIO.read_window_assets(root, panel, split)
            days = set(index.date.astype(str)) | {str(np.datetime64(x) - np.timedelta64(1, "D")) for x in index.date.astype(str)}
            cur = CIO.read_panel_arrays(root, net, split, OUT / "cache", masked=True, dates=days)
            d = TD.load(panel, split); R = build.released(panel, split)
            _, hs, hf = TD.released_windows(panel, split)
            for i, w in enumerate(index.itertuples()):
                T = np.datetime64(str(w.forecast_origin).replace("Z", ""), "m"); t = int(R["meta"].t.iloc[i])
                assert R["meta"].window_id.iloc[i] == w.window_id
                hist = history[history.window_id.eq(w.window_id)]
                stamps = cur.timestamps.astype("datetime64[m]")
                cut_c = ((stamps >= T) if control else (stamps > T)) | ((stamps < T - 12 * FIVE) if before60 else False)
                cut_t = np.arange(len(d["mspeed"])) >= t if control else np.arange(len(d["mspeed"])) > t
                if before60: cut_t |= np.arange(len(d["mspeed"])) < t - 12
                cur2 = dataclasses.replace(cur, speed=blank(cur.speed, cut_c, fill, rng, 5, 130), flow=blank(cur.flow, cut_c, fill, rng, 0, 9000),
                                           occupancy=blank(cur.occupancy, cut_c, fill, rng), eligible=blank(cur.eligible, cut_c, fill, rng),
                                           ramp_on=blank(cur.ramp_on, cut_c, fill, rng, 0, 3000))
                d2 = dict(d, mspeed=blank(d["mspeed"], cut_t, fill, rng, 5, 130), mflow=blank(d["mflow"], cut_t, fill, rng, 0, 9000),
                          ron=blank(d["ron"], cut_t, fill, rng, 0, 3000), roff=blank(d["roff"], cut_t, fill, rng, 0, 3000))
                meta = R["meta"].iloc[[i]]
                a = codex_inputs(cur, hist, w, net, pid, prior, onset_prior, capacity) | t2b_inputs(d, t, hs[i], hf[i], meta, bnet, pr, w.condition)
                b = codex_inputs(cur2, hist, w, net, pid, prior, onset_prior, capacity) | t2b_inputs(d2, t, hs[i], hf[i], meta, bnet, pr, w.condition)
                for member in MEMBERS:
                    first = diff(a[member], b[member])
                    rows.append((panel, split, w.condition, member, first))
                    if control:                       # positive control: the row at T is replaced too -> every member must change
                        if first is None:
                            failures += 1; print(f"UNCHANGED {member} {panel}/{split} {w.window_id} T={T}", flush=True)
                    elif first is not None and not before60:
                        failures += 1
                        print(f"CHANGED {member} {panel}/{split} {w.window_id} T={T}: first feature {first}", flush=True)
            TD.load.cache_clear(); print(f"{panel}/{split}: done", flush=True)
    return rows, failures


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fill", choices=["nan", "random"], default="random")
    p.add_argument("--before60", action="store_true")
    p.add_argument("--panels", nargs="*", default=list(io.T2_PANELS))
    p.add_argument("--control", action="store_true", help="positive control: replace rows >= T (the row at T too); every member must change")
    a = p.parse_args(argv)
    rows, failures = check(a.panels, a.fill, a.before60, control=a.control)
    import pandas as pd
    tab = pd.DataFrame(rows, columns=["panel", "split", "condition", "member", "first_diff"])
    tab["changed"] = tab.first_diff.notna()
    print(tab.groupby(["member", "condition"]).agg(windows=("changed", "size"), changed=("changed", "sum")).to_string())
    if a.before60:
        print("before-T-60 inputs used (first changed feature):", tab[tab.changed].groupby("member").first_diff.first().to_dict())
    mode = "POSITIVE CONTROL: data at and after T replaced" if a.control else "data before T-60 and after T replaced" if a.before60 else "data after T replaced"
    what = "unchanged member-windows" if a.control else "after-T changes"
    print(f"{mode} ({a.fill}): {len(tab) // len(MEMBERS)} windows, {failures} {what} -> {'FAIL' if failures else 'OK'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
