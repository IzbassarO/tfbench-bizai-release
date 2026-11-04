"""Location-free onset ladder on top of family A v2 ongoing (LB 2026-10-25: v4 location-free onset +0.0008 total).

Onset T+30 probability per window = blend of
  free : location-free members  (family A v4 normal mean3; optionally family B noloc)
  loc  : location-prior member  (family A v2 raw probabilities, probs_v2.parquet)
decoded with family A's exact expected-IoU top-m (force non-empty, as v4 deployment); T+5..T+25 = 0.
Ongoing rows = family A v2 binaries, unchanged.

  A : 0.5 v4 + 0.5 family B noloc   (fully location-free, two independent models; v4 alone = queue_submission_v4_onset.csv)
  B : 0.75 v4 + 0.25 v2  (dropped 2026-10-25)
  C : 0.5 v4 + 0.5 v2  (dropped 2026-10-25: blends toward v2, which lost to v4 on LB)
  E : onset A; ongoing = decode(0.5 v2 raw + 0.5 family B noloc ongoing) over all 6 horizons (location-free ongoing blend)
  F : A with family B noloc onset averaged over 6 seeds (frozen 3 + 3 new); v4 stays its frozen 3-seed mean

python -m tfb.t2b.ladder [--ef] -> CV sanity tables (family A official-like windows, both folds) + outputs/t2/queue_onset_*.csv
"""
from __future__ import annotations
import pickle
import numpy as np
import pandas as pd
from .. import cache, io
from ..t2.decode import expected_iou_decode
from . import codex, cv, data, evaluate as ev, run
from .probavg import _fam, _prob_grid
from .report import _grids, load_fold

T2 = io.OUTPUTS / "t2"
LADDER = {  # name: (w_v4, w_t2b_noloc, w_v2)
    "v2_bin": None, "v2_raw": (0, 0, 1), "v4": (1, 0, 0),
    "A": (0.5, 0.5, 0), "B": (0.75, 0, 0.25), "C": (0.5, 0, 0.5), "t2b_noloc": (0, 1, 0),
}
DELIVER = ("A",)  # B dropped 2026-10-25 (LB: A 0.87309)
NEW_SEEDS = (11, 23, 57)


def decode30(p30: np.ndarray) -> np.ndarray:
    """[n, L] T+30 probabilities -> [n, 6, L] binaries (family A v4 onset decoding)."""
    D = np.zeros((len(p30), 6, p30.shape[1]), bool)
    for i, p in enumerate(p30):
        D[i, 5] = expected_iou_decode(p, force_nonempty=True).astype(bool)
    return D


def mix(w, v4, nl, v2, ph=None):
    return w[0] * v4 + w[1] * nl + w[2] * v2 + (w[3] * ph if len(w) > 3 else 0)


def cv_table(ladder=LADDER, extra=None) -> pd.DataFrame:
    """extra: {fold: {panel: (P30, wpos)}} 4th member (weight w[3]), e.g. phys.stage_cv output."""
    rows = []
    for fold in cv.FOLDS:
        probs, _ = load_fold(fold, "hybrid")
        nl = {}
        for (pk, cond, _), (Pv, wpos) in _grids(probs).items():
            if "@" in pk and cond == "queue_onset":
                nl.setdefault(pk.split("@")[0], []).append((Pv["noloc"], wpos))
        nl = {k: (np.concatenate([a for a, _ in sorted(v, key=lambda x: x[1][0])]),
                  np.concatenate([b for _, b in sorted(v, key=lambda x: x[1][0])])) for k, v in nl.items()}
        for p in io.T2_PANELS:
            B = codex.windows(fold, p); mk = (B["meta"].condition == "queue_onset").to_numpy()
            z = np.load(T2 / "v4" / "probabilities" / fold / f"{p}_normal.npz")
            pos = {w: i for i, w in enumerate(z["window_id"])}; ii = [pos[w] for w in B["meta"].window_id[mk]]
            o = data.downstream_order(p, cache.load(p, "train")["links"])
            assert (z["truth"][ii][..., o] == B["Y"][mk]).all()
            v4 = z["probability"][ii][:, o]; v2 = B["codex"]["probability_v2"][mk][:, 5]
            Pn, wpos = nl[p]; assert (np.flatnonzero(mk) == wpos).all()
            t2 = Pn[:, 5]; ph = None
            if extra:
                ph, pw = extra[fold][p]; assert (pw == wpos).all()
            Y, E = B["Y"][mk], B["E"][mk]
            for name, w in ladder.items():
                D = B["codex"]["prediction_v2"][mk].astype(bool) if w is None else decode30(mix(w, v4, t2, v2, ph))
                rows.append(pd.DataFrame({"fold": fold, "panel": p, "model": name, "iou": ev.window_iou(D, Y, E),
                                          "n_pred": D.sum((1, 2))}))
    df = pd.concat(rows, ignore_index=True)
    return pd.DataFrame([dict(fold=f, model=m, onset=_fam(g), n_pred=g.n_pred.mean()) for (f, m), g in df.groupby(["fold", "model"])])


def write():
    P = pickle.load(open(io.OUTPUTS / "t2b" / "final_probs.pkl", "rb"))
    pv2 = pd.read_parquet(T2 / "probs_v2.parquet").rename(columns={"queue_prob": "queue_pred"})
    pv4 = pd.read_parquet(T2 / "probs_v4_onset.parquet").rename(columns={"queue_prob": "queue_pred"})
    csub = codex.submission(T2 / "queue_submission_v2.csv")
    parts = {k: [] for k in DELIVER}; changed = []
    for p in io.T2_PANELS:
        for s in ("validation", "private"):
            wi = pd.read_csv(io.t2_dir(p, s) / "window_index.csv", dtype=str)
            on = (wi.condition == "queue_onset").to_numpy()
            G = codex.grid(csub, p, s, wi)
            v2 = _prob_grid(pv2, p, s, wi)[on, 5]; v4 = _prob_grid(pv4, p, s, wi)[on, 5]
            nl = P[(p, s)]["noloc"][on, 5]
            for k in DELIVER:
                D = G.copy(); D[on] = decode30(mix(LADDER[k], v4, nl, v2)); parts[k].append(run.to_rows(D, p, s, wi))
                d = (D[on] != G[on]).any((1, 2))
                changed.append(dict(variant=k, panel=p, split=s, windows_changed=int(d.sum()), cells_changed=int((D[on] != G[on]).sum())))
    files = []
    for k, ps in parts.items():
        f = f"../t2/queue_onset_{k}.csv"
        pd.concat(ps, ignore_index=True).to_csv(io.OUTPUTS / "t2b" / f, index=False); files.append(f)
    return run.verify(files), pd.DataFrame(changed)


def cv_ongoing() -> pd.DataFrame:
    """Ongoing on family A's official-like windows: v2 binaries vs raw-probability blends with family B noloc (family A decoder)."""
    rows = []
    for fold in cv.FOLDS:
        probs, cb = load_fold(fold, "hybrid")
        for (pk, cond, _), (Pv, wpos) in _grids(probs).items():
            if "@" not in pk or cond != "queue_ongoing":
                continue
            b = cb[pk]; v2 = b["codex"]["probability_v2"][wpos]; Y, E = b["Y"][wpos], b["E"][wpos]
            cands = {"v2_bin": b["codex"]["prediction_v2"][wpos].astype(bool)}
            for a in (0.0, 0.25, 0.5, 1.0):
                P = a * Pv["noloc"] + (1 - a) * v2
                cands[f"noloc{a}"] = np.stack([expected_iou_decode(x, force_nonempty=True) for x in P]).astype(bool)
            for k, D in cands.items():
                rows.append(pd.DataFrame({"fold": fold, "panel": pk.split("@")[0], "model": k, "iou": ev.window_iou(D, Y, E)}))
    df = pd.concat(rows, ignore_index=True)
    return pd.DataFrame([dict(fold=f, model=m, ongoing=_fam(g)) for (f, m), g in df.groupby(["fold", "model"])])


def noloc_onset_seeds(seeds=NEW_SEEDS) -> dict:
    """family B noloc onset T+30 probabilities on released windows for extra seeds (same rows/config as run.stage_final)."""
    from . import build, model as M
    Bs = {p: ev.load_train(p) for p in io.T2_PANELS}; cond = "queue_onset"
    tr = {p: cv.rows(p, Bs[p], cond, run.ALL_MONTHS, run.ALL_MONTHS, loo=True, extra=True) for p in io.T2_PANELS}
    names = next(r[1] for p in io.T2_PANELS for r in tr[p])
    X = np.concatenate([r[0] for p in io.T2_PANELS for r in tr[p]]); y = np.concatenate([r[3] for p in io.T2_PANELS for r in tr[p]])
    e = np.concatenate([r[4] for p in io.T2_PANELS for r in tr[p]]); X, y = X[e], y[e]; del tr
    ms = [M.fit(X, y, names, drop=M.LOC, seed=s) for s in seeds]; del X, y
    out = {}
    for p in io.T2_PANELS:
        pr = M.priors(p, cv._days(None, run.ALL_MONTHS))
        for s in ("validation", "private"):
            R = build.released(p, s); mk = (R["meta"].condition == cond).to_numpy()
            Xr, _, idx = M.panel_rows({k: R[k][mk] for k in ("S", "Q", "RON", "ROFF")}, Bs[p]["net"], R["meta"][mk], pr, cond)
            out[(p, s)] = M.to_grid(np.mean([M.predict(m, Xr) for m in ms], 0), idx, mk.sum(), len(Bs[p]["net"]["v_f"]))[:, 5]
    return out


def write_ef():
    P = pickle.load(open(io.OUTPUTS / "t2b" / "final_probs.pkl", "rb"))
    pv2 = pd.read_parquet(T2 / "probs_v2.parquet").rename(columns={"queue_prob": "queue_pred"})
    pv4 = pd.read_parquet(T2 / "probs_v4_onset.parquet").rename(columns={"queue_prob": "queue_pred"})
    csub = codex.submission(T2 / "queue_submission_v2.csv"); extra = noloc_onset_seeds()
    parts = {"E": [], "F": []}; changed = []
    for p in io.T2_PANELS:
        for s in ("validation", "private"):
            wi = pd.read_csv(io.t2_dir(p, s) / "window_index.csv", dtype=str)
            on = (wi.condition == "queue_onset").to_numpy(); G = codex.grid(csub, p, s, wi)
            C2 = _prob_grid(pv2, p, s, wi); v4 = _prob_grid(pv4, p, s, wi)[on, 5]; nl = P[(p, s)]["noloc"]
            A = decode30(0.5 * v4 + 0.5 * nl[on, 5])
            F = decode30(0.5 * v4 + 0.5 * (0.5 * nl[on, 5] + 0.5 * extra[(p, s)]))  # 3 frozen + 3 new seeds, equal weight
            og = 0.5 * C2[~on] + 0.5 * nl[~on]
            E = G.copy(); E[on] = A; E[~on] = np.stack([expected_iou_decode(x, force_nonempty=True) for x in og]).astype(bool)
            Fd = G.copy(); Fd[on] = F
            for k, D in (("E", E), ("F", Fd)):
                parts[k].append(run.to_rows(D, p, s, wi))
                changed.append(dict(variant=k, panel=p, split=s,
                                    onset_windows=int((D[on] != G[on]).any((1, 2)).sum()), onset_cells=int((D[on] != G[on]).sum()),
                                    ongoing_windows=int((D[~on] != G[~on]).any((1, 2)).sum()), ongoing_cells=int((D[~on] != G[~on]).sum()),
                                    f_vs_a_cells=int((F != A).sum()) if k == "F" else 0))
    files = []
    for k, ps in parts.items():
        f = f"../t2/queue_onset_{k}.csv"
        pd.concat(ps, ignore_index=True).to_csv(io.OUTPUTS / "t2b" / f, index=False); files.append(f)
    return run.verify(files), pd.DataFrame(changed)


def write_e25(f="../t2/queue_onset_E25.csv", w=0.25):
    """E25 without the phys member (same arithmetic as tfb.t2b.phys.write's E25): onset A = decode30(0.5 v4 + 0.5 t2b
    noloc), ongoing = decode((1 - w) v2 raw + w family B noloc) over all 6 horizons (w = 0.25: E25; --e50: w = 0.5). Inputs: t2b/final_probs.pkl,
    t2/probs_v2.parquet, t2/probs_v4_onset.parquet, t2/queue_submission_v2.csv."""
    P = pickle.load(open(io.OUTPUTS / "t2b" / "final_probs.pkl", "rb"))
    pv2 = pd.read_parquet(T2 / "probs_v2.parquet").rename(columns={"queue_prob": "queue_pred"})
    pv4 = pd.read_parquet(T2 / "probs_v4_onset.parquet").rename(columns={"queue_prob": "queue_pred"})
    csub = codex.submission(T2 / "queue_submission_v2.csv"); parts = []
    for p in io.T2_PANELS:
        for s in ("validation", "private"):
            wi = pd.read_csv(io.t2_dir(p, s) / "window_index.csv", dtype=str)
            on = (wi.condition == "queue_onset").to_numpy(); D = codex.grid(csub, p, s, wi)
            C2 = _prob_grid(pv2, p, s, wi); v4 = _prob_grid(pv4, p, s, wi)[on, 5]; nl = P[(p, s)]["noloc"]
            D[on] = decode30(0.5 * v4 + 0.5 * nl[on, 5])
            D[~on] = np.stack([expected_iou_decode(x, force_nonempty=True) for x in (1 - w) * C2[~on] + w * nl[~on]]).astype(bool)
            parts.append(run.to_rows(D, p, s, wi))
    pd.concat(parts, ignore_index=True).to_csv(io.OUTPUTS / "t2b" / f, index=False)
    return run.verify([f])


if __name__ == "__main__":
    import json, sys
    out = io.OUTPUTS / "t2b" / "ladder"; out.mkdir(parents=True, exist_ok=True)
    if "--e25" in sys.argv:  # light writer for reproduce.sh (no training)
        write_e25(); sys.exit()
    if "--e50" in sys.argv:  # diagnostic: ongoing weight 0.5 (E25 default untouched)
        write_e25("../t2/queue_onset_E50.csv", 0.5); sys.exit()
    if "--ef" in sys.argv:
        t = cv_ongoing(); t.to_csv(out / "cv_ongoing.csv", index=False)
        print(t.pivot(index="model", columns="fold", values="ongoing").round(4).to_string())
        ver, ch = write_ef(); json.dump(ver, open(out / "verification_ef.json", "w"), indent=1); ch.to_csv(out / "changes_ef.csv", index=False)
        print(ch.groupby(["variant", "split"]).sum(numeric_only=True).to_string())
        sys.exit()
    t = cv_table(); t.to_csv(out / "cv.csv", index=False)
    print(t.pivot(index="model", columns="fold", values=["onset", "n_pred"]).round(4).to_string())
    ver, ch = write(); json.dump(ver, open(out / "verification.json", "w"), indent=1); ch.to_csv(out / "changes.csv", index=False)
    print(ch.groupby(["variant", "split"])[["windows_changed", "cells_changed"]].sum().to_string())
