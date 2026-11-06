"""A_hyb: both location-free onset members retrained on hybrid T+30 labels (ongoing = family A v2).

Label speed (train only, labels never features): outputs/t1/train_hybrid_speed_clean = released truth on eligible cells,
clean-imputer speed on every other cell, including pct<75 cells whose truth is generator time interpolation.
queued = label speed <= 0.6 v_f.
  family B noloc : windows re-selected on hybrid labels (outputs/t2b/cache_hyb_<mode>), trained on every labelled T+30 cell.
  v4             : V4's rows/features (eligible T+30 cells), train_y replaced (tfb.t2.v4_hyb).
"""
from __future__ import annotations
import pickle
import numpy as np
import pandas as pd
from .. import io
from ..t2 import v4_hyb as V
from . import build, codex, cv, data, evaluate as ev, ladder as LD, model as M, run

HYB_MIX = io.OUTPUTS / "t1" / "train_hybrid_speed_clean"
OUT = io.OUTPUTS / "t2b" / "hybrid"
SEEDS = (42, 7, 2031)
V4_SEEDS = (42, 137, 271)
COND = "queue_onset"


def hybrid_speed(panel: str, src=None) -> np.ndarray:
    """[T, L] hybrid train speed in t2b link order (released speed on eligible cells, clean LightGBM imputation on
    every other cell, including pct<75 cells that the generator time-interpolated)."""
    df = pd.read_parquet(src or HYB_MIX, columns=["timestamp", "link_id", "speed_hybrid"], filters=[("panel", "=", panel)])
    d = data.load(panel, "train"); pos = {l: i for i, l in enumerate(d["links"])}
    day = {str(x)[:10]: i for i, x in enumerate(d["dates"])}
    tc = df.timestamp.astype("category"); cats = tc.cat.categories.astype(str)
    tix = np.array([day[c[:10]] * 288 + int(c[11:13]) * 12 + int(c[14:16]) // 5 for c in cats])[tc.cat.codes.to_numpy()]
    lc = df.link_id.astype("category"); lix = np.array([pos.get(str(c), -1) for c in lc.cat.categories])[lc.cat.codes.to_numpy()]
    out = np.full(d["speed"].shape, np.nan, np.float32); ok = lix >= 0
    out[tix[ok], lix[ok]] = df.speed_hybrid.to_numpy(np.float32)[ok]
    assert np.isfinite(out).mean() > 0.99, f"{panel}: hybrid grid coverage {np.isfinite(out).mean():.3f}"
    return out


def label_speed(panel: str, mode: str = "mix_clean") -> np.ndarray:
    """[T, L] label speed: the hybrid speed where available, else the released speed. mode only names the outputs."""
    raw = data.load(panel, "train")["speed"]; h = hybrid_speed(panel)
    return np.where(np.isfinite(h), h, raw)


def build_cache(mode="mix_clean"):
    """Hybrid-label train windows -> outputs/t2b/cache_hyb_<mode>/."""
    dst = io.OUTPUTS / "t2b" / f"cache_hyb_{mode}"; dst.mkdir(parents=True, exist_ok=True)
    for p in io.T2_PANELS:
        b = build.train_windows(p, lab_speed=label_speed(p, mode))
        b["meta"].to_parquet(dst / f"{p}_train_meta.parquet")
        np.savez_compressed(dst / f"{p}_train.npz", **{k: v for k, v in b.items() if k != "meta"})


def v4_label_fn(mode="mix_clean"):
    """Labels at family A (time, link id) from the label speed (t2b index == family A index: verified by truth replay)."""
    memo = {}

    def fn(panel, t30, link_ids):
        if panel not in memo:
            d = data.load(panel, "train"); pos = {l: i for i, l in enumerate(d["links"])}
            ls = label_speed(panel, mode)
            memo.clear(); memo[panel] = (np.nan_to_num(ls, nan=999) <= d["vcut"], pos)
        q, pos = memo[panel]
        return q[t30, np.array([pos[l] for l in link_ids])]
    return fn


def noloc_fit(months, cache_dir, seeds=SEEDS, log=print, cond=COND, extra=True, drop=M.LOC):
    Bs = {p: ev.load_train(p, cache_dir) for p in io.T2_PANELS}
    tr = [r for p in io.T2_PANELS for r in cv.rows(p, Bs[p], cond, months, months, loo=True, extra=extra)]
    names = tr[0][1]; e = np.concatenate([r[4] for r in tr])
    X = np.concatenate([r[0] for r in tr])[e]; y = np.concatenate([r[3] for r in tr])[e]; del tr
    log(f"noloc fit [{cache_dir}] {months[0]}..{months[-1]}: {len(y):,} rows, pos {y.mean():.4f}")
    return [M.fit(X, y, names, drop=drop, seed=s) for s in seeds]


def dec30(P: np.ndarray) -> np.ndarray:
    """[n, L] T+30 probabilities (expected-IoU decode) or binaries (used as is) -> [n, 6, L] binaries."""
    if P.dtype != bool:
        return LD.decode30(P)
    D = np.zeros((len(P), 6, P.shape[1]), bool); D[:, 5] = P
    return D


def stage_final(mode="mix_clean", log=print):
    """Final members on hybrid labels -> released T+30 probabilities (pickle) for A_hyb."""
    OUT.mkdir(parents=True, exist_ok=True)
    y, st = V.relabel("final", v4_label_fn(mode)); st.to_csv(OUT / f"v4_relabel_final_{mode}.csv", index=False)
    ms = V.frozen_models() if not (st.flip_to_pos.sum() + st.flip_to_neg.sum()) else V.fit("final", f"hyb_{mode}", y, V4_SEEDS)[0]
    pv4h = V.predict_released(ms)
    ms = noloc_fit(run.ALL_MONTHS, f"cache_hyb_{mode}", log=log)
    nlh = released_noloc(ms)
    pickle.dump(dict(v4h=pv4h, nlh=nlh), open(OUT / f"final_{mode}.pkl", "wb"))


def released_noloc(ms, cond=COND, h=5) -> dict:
    out = {}
    for p in io.T2_PANELS:
        pr = M.priors(p, cv._days(None, run.ALL_MONTHS)); net = ev.load_train(p)["net"]
        for s in ("validation", "private"):
            R = build.released(p, s); mk = (R["meta"].condition == cond).to_numpy()
            X, _, idx = M.panel_rows({k: R[k][mk] for k in ("S", "Q", "RON", "ROFF")}, net, R["meta"][mk], pr, cond)
            g = M.to_grid(np.mean([M.predict(m, X) for m in ms], 0), idx, mk.sum(), len(net["v_f"]))
            out[(p, s)] = g if h is None else g[:, h]
    return out


def long_to_grid(df, panel, split, wi_on) -> np.ndarray:
    """(window_id, link_id, p) -> [n_onset, L] in t2b link order, rows = wi_on order."""
    links = data.load(panel, split)["links"]
    s = df[df.window_id.isin(set(wi_on.window_id))].pivot(index="window_id", columns="link_id", values="p")
    return s.reindex(index=wi_on.window_id, columns=links).to_numpy()


def write(onset: dict, name: str, ref="queue_onset_A.csv") -> tuple:
    """onset: {(panel, split): P30 or bool D30 [n_onset, L]} -> outputs/t2b/<name> (ongoing = v2 binaries)."""
    csub = codex.submission(LD.T2 / "queue_submission_v2.csv"); A = codex.submission(LD.T2 / ref)
    parts, ch = [], []
    for p in io.T2_PANELS:
        for s in ("validation", "private"):
            wi = pd.read_csv(io.t2_dir(p, s) / "window_index.csv", dtype=str); on = (wi.condition == COND).to_numpy()
            G = codex.grid(csub, p, s, wi); Ga = codex.grid(A, p, s, wi)
            D = G.copy(); D[on] = dec30(onset[(p, s)]); parts.append(run.to_rows(D, p, s, wi))
            ch.append(dict(panel=p, split=s, onset_windows_vs_ref=int((D[on] != Ga[on]).any((1, 2)).sum()),
                           onset_cells_vs_ref=int((D[on] != Ga[on]).sum()), ongoing_cells_vs_v2=int((D[~on] != G[~on]).sum())))
    f = name; (io.OUTPUTS / "t2b" / f).parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_csv(io.OUTPUTS / "t2b" / f, index=False)
    return run.verify([f]), pd.DataFrame(ch)


def write_a_hyb(mode="mix_clean"):
    """A_hyb = decode30(0.5 v4 + 0.5 family B noloc) on the final members of `mode`."""
    F = pickle.load(open(OUT / f"final_{mode}.pkl", "rb")); on = {}
    for p in io.T2_PANELS:
        for s in ("validation", "private"):
            wi = pd.read_csv(io.t2_dir(p, s) / "window_index.csv", dtype=str); wo = wi[wi.condition == COND]
            on[(p, s)] = 0.5 * long_to_grid(F["v4h"], p, s, wo) + 0.5 * F["nlh"][(p, s)]
    return write(on, f"../t2/queue_onset_A_hyb_{mode}.csv")


def splice(onset_file: str, ongoing_file: str, out: str, ref="queue_onset_A_hyb.csv") -> tuple:
    """outputs/t2/<out> = onset-window rows of <onset_file> + ongoing-window rows of <ongoing_file> (both outputs/t2)."""
    on = {w for p in io.T2_PANELS for s in ("validation", "private")
          for w in pd.read_csv(io.t2_dir(p, s) / "window_index.csv", dtype=str).query("condition == @COND").window_id}
    a, b, r = (codex.submission(LD.T2 / f) for f in (onset_file, ongoing_file, ref))
    df = pd.concat([a[a.window_id.isin(on)], b[~b.window_id.isin(on)]]).set_index(["window_id", "timestamp", "link_id"])
    df = df.reindex(r.set_index(["window_id", "timestamp", "link_id"]).index).reset_index()
    df.to_csv(LD.T2 / out, index=False)
    m = df.merge(r, on=["window_id", "timestamp", "link_id"], suffixes=("", "_ref"))
    d = m[m.queue_pred != m.queue_pred_ref]; split = d.window_id.str.contains("_validation_").map({True: "validation", False: "private"})
    ch = d.assign(split=split, onset=d.window_id.isin(on)).groupby(["split", "onset"]).agg(cells=("link_id", "size"), windows=("window_id", "nunique"))
    return run.verify([f"../t2/{out}"]), ch
