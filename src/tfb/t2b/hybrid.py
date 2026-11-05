"""A_hyb: both location-free onset members of A retrained on hybrid T+30 labels (ongoing = family A v2).

Hisernberg (rebuilt data, single-factor LB): onset retrained on hybrid labels (missing cells filled by an LGB
imputer, queued recall 0.27 -> 0.95) = +0.0045 total; ongoing on hybrid labels lost, so ongoing is untouched.

Label speed (train only, labels never features), default mode "pure": outputs/t1/train_hybrid_speed.parquet as is
(= released truth on eligible cells, Task 1 imputation on every other cell, including pct<75 cells whose truth is
generator time interpolation). "fill" = truth wherever finite (keeps those interpolated cells), hybrid only where NaN.
queued = label speed <= 0.6 v_f.
  family B noloc : windows re-selected on hybrid labels (outputs/t2b/cache_hyb), trained on every labelled T+30 cell.
  v4        : V4's frozen rows/features (eligible T+30 cells), train_y replaced (tfb.t2.v4_hyb).
Evaluation always uses the raw truth on eligible cells (family A official-like windows, both folds).

python -m tfb.t2b.hybrid --stage audit|cv|final|all [--mode pure|fill|interp|raw[_elig][_keepwin]]
"""
from __future__ import annotations
import pickle
import numpy as np
import pandas as pd
from .. import cache, io
from ..t2 import v4_hyb as V
from . import build, codex, cv, data, evaluate as ev, ladder as LD, model as M, run
from .probavg import _fam

HYB = io.OUTPUTS / "t1" / "train_hybrid_speed.parquet"
HYB_MIX = io.OUTPUTS / "t1" / "train_hybrid_speed_mix.parquet"  # mode "mix": 0.5 LGBM 8x + 0.5 MLP imputer (tfb.t2b.hyb_relabel)
OUT = io.OUTPUTS / "t2b" / "hybrid"
SEEDS = (42, 7, 2031)
V4_SEEDS = (42, 137, 271)
COND = "queue_onset"
_SPEED = ("speed_hybrid", "speed_kmh_hybrid", "hybrid_speed", "speed_kmh", "speed")


def hybrid_speed(panel: str, src=HYB) -> np.ndarray:
    """[T, L] hybrid train speed in t2b link order (tfb.t1.hybrid: released speed on eligible cells, Task 1 LGBM
    imputation on every other cell, incl. pct<75 cells that the generator time-interpolated)."""
    df = pd.read_parquet(src, columns=["timestamp", "link_id", "speed_hybrid"], filters=[("panel", "=", panel)])
    d = data.load(panel, "train"); pos = {l: i for i, l in enumerate(d["links"])}
    day = {str(x)[:10]: i for i, x in enumerate(d["dates"])}
    tc = df.timestamp.astype("category"); cats = tc.cat.categories.astype(str)
    tix = np.array([day[c[:10]] * 288 + int(c[11:13]) * 12 + int(c[14:16]) // 5 for c in cats])[tc.cat.codes.to_numpy()]
    lc = df.link_id.astype("category"); lix = np.array([pos.get(str(c), -1) for c in lc.cat.categories])[lc.cat.codes.to_numpy()]
    out = np.full(d["speed"].shape, np.nan, np.float32); ok = lix >= 0
    out[tix[ok], lix[ok]] = df.speed_hybrid.to_numpy(np.float32)[ok]
    assert np.isfinite(out).mean() > 0.99, f"{panel}: hybrid grid coverage {np.isfinite(out).mean():.3f}"
    return out


def interp_speed(panel: str) -> np.ndarray:
    """Control: missing truth filled by within-day linear time interpolation (the weaker fill Hisernberg replaced)."""
    raw = data.load(panel, "train")["speed"]; T, L = raw.shape
    a = pd.DataFrame(raw.reshape(-1, 288, L).transpose(1, 0, 2).reshape(288, -1))
    return a.interpolate(axis=0, limit_area="inside").to_numpy(np.float32).reshape(288, -1, L).transpose(1, 0, 2).reshape(T, L)


def label_speed(panel: str, mode: str = "pure") -> np.ndarray:
    """mode = <fill|pure|interp|raw>[_elig][_keepwin]; suffixes only change build_cache (eligible-only training
    cells / windows selected on the raw truth as now)."""
    raw = data.load(panel, "train")["speed"]; mode = mode.split("_")[0]
    if mode == "raw":
        return raw
    if mode == "interp":
        return interp_speed(panel)
    if mode == "mix":  # as pure, other imputer
        h = hybrid_speed(panel, HYB_MIX); return np.where(np.isfinite(h), h, raw)
    h = hybrid_speed(panel)
    return np.where(np.isfinite(raw), raw, h) if mode == "fill" else np.where(np.isfinite(h), h, raw)


def build_cache(mode="pure", log=print) -> pd.DataFrame:
    """Hybrid-label train windows -> outputs/t2b/cache_hyb_<mode>/ + label-change audit vs the current cache."""
    dst = io.OUTPUTS / "t2b" / f"cache_hyb_{mode}"; dst.mkdir(parents=True, exist_ok=True); rows = []
    for p in io.T2_PANELS:
        d = data.load(p, "train"); ls = label_speed(p, mode)
        b = build.train_windows(p, lab_speed=ls, train_cells="elig" if "_elig" in mode else "finite",
                                select_speed=d["speed"] if "_keepwin" in mode else None)
        b["meta"].to_parquet(dst / f"{p}_train_meta.parquet")
        np.savez_compressed(dst / f"{p}_train.npz", **{k: v for k, v in b.items() if k != "meta"})
        old = ev.load_train(p)
        qr, qh = np.nan_to_num(d["speed"], nan=999) <= d["vcut"], np.nan_to_num(ls, nan=999) <= d["vcut"]
        fin = np.isfinite(d["speed"])
        on_o, on_n = old["meta"].condition.eq(COND).to_numpy(), b["meta"].condition.eq(COND).to_numpy()
        t_o, t_n = old["meta"].t.to_numpy()[on_o], b["meta"].t.to_numpy()[on_n]
        # T+30 label cells of the CURRENT onset windows: training label old (YD on E) vs hybrid
        Eo, Yo = old["E"][on_o][:, 5], old["YD"][on_o][:, 5]
        Yh = np.stack([qh[t + 6] for t in t_o]); Fh = np.stack([np.isfinite(ls[t + 6]) for t in t_o])
        rows.append(dict(panel=p, mode=mode,
                         all_cells_missing_truth=int((~fin).sum()), filled_by_hybrid=int((~fin & np.isfinite(ls)).sum()),
                         filled_queued=int((~fin & qh).sum()), observed_changed=int((fin & (qr != qh)).sum()),
                         onset_windows_old=int(on_o.sum()), onset_windows_new=int(on_n.sum()),
                         onset_windows_kept=len(set(t_o) & set(t_n)),
                         t30_cells_old_trained=int(Eo.sum()), t30_cells_new_trained=int(Fh.sum()),
                         t30_queued_old=int((Yo & Eo).sum()), t30_queued_new=int((Yh & Fh).sum()),
                         t30_label_changed=int(((Yh & Fh) != (Yo & Eo)).sum()),
                         t30_train_rows_old_total=int(old["E"][on_o][:, 5].sum()), t30_train_rows_new_total=int(b["E"][on_n][:, 5].sum()),
                         t30_pos_new_total=int((b["YD"][on_n][:, 5] & b["E"][on_n][:, 5]).sum())))
        log(f"cache_hyb_{mode} {p}: {rows[-1]}")
    return pd.DataFrame(rows)


def v4_label_fn(mode="pure"):
    """Labels at family A (time, link id) from the label speed (t2b index == family A index: verified by truth replay)."""
    memo = {}

    def fn(panel, t30, link_ids):
        if panel not in memo:
            d = data.load(panel, "train"); pos = {l: i for i, l in enumerate(d["links"])}
            ls = d["speed"] if mode == "truth" else label_speed(panel, mode)  # suffix-free: V4 rows are eligible
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


def noloc_codex(fold, ms, cond=COND, h=5) -> dict:
    """{panel: P30 [n_onset_codex_windows, L]} (t2b order, aligned to codex.windows onset rows); h=None: all 6 horizons."""
    fit_m, ev_m = cv.FOLDS[fold]; out = {}
    for p in io.T2_PANELS:
        C = codex.windows(fold, p); C["net"] = ev.load_train(p)["net"]; parts = []
        for (X, _, idx, _, _, wpos) in cv.rows(p, C, cond, ev_m, fit_m):
            g = M.to_grid(np.mean([M.predict(m, X) for m in ms], 0), idx, len(wpos), len(C["net"]["v_f"]))
            parts.append((g if h is None else g[:, h], wpos))
        parts.sort(key=lambda t: t[1][0]); out[p] = np.concatenate([a for a, _ in parts])
    return out


def v4_codex(fold, probs: dict | None = None) -> dict:
    """V4 T+30 probabilities aligned like noloc_codex; probs from v4_hyb.predict_cv or the frozen npz."""
    out = {}
    for p in io.T2_PANELS:
        B = codex.windows(fold, p); mk = (B["meta"].condition == COND).to_numpy()
        if probs is None:
            z = np.load(LD.T2 / "v4" / "probabilities" / fold / f"{p}_normal.npz"); P, wid = z["probability"], z["window_id"]
        else:
            P, wid = probs[p]
        pos = {w: i for i, w in enumerate(wid)}; ii = [pos[w] for w in B["meta"].window_id[mk]]
        out[p] = P[ii][:, data.downstream_order(p, cache.load(p, "train")["links"])]
    return out


def dec30(P: np.ndarray) -> np.ndarray:
    """[n, L] T+30 probabilities (expected-IoU decode) or binaries (used as is) -> [n, 6, L] binaries."""
    if P.dtype != bool:
        return LD.decode30(P)
    D = np.zeros((len(P), 6, P.shape[1]), bool); D[:, 5] = P
    return D


def score(fold, cands: dict) -> pd.DataFrame:
    """cands: {name: {panel: P30 or bool D30}} -> family-macro onset IoU on family A windows (raw truth, eligible cells)."""
    rows = []
    for p in io.T2_PANELS:
        B = codex.windows(fold, p); mk = (B["meta"].condition == COND).to_numpy(); Y, E = B["Y"][mk], B["E"][mk]
        for name, P in cands.items():
            D = B["codex"]["prediction_v2"][mk].astype(bool) if P is None else dec30(P[p])
            rows.append(pd.DataFrame({"fold": fold, "panel": p, "model": name, "iou": ev.window_iou(D, Y, E)}))
    df = pd.concat(rows, ignore_index=True)
    return pd.DataFrame([dict(fold=f, model=m, onset=_fam(g)) for (f, m), g in df.groupby(["fold", "model"])])


def stage_cv(mode="pure", log=print) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True); res = []
    for fold, (fit_m, _) in cv.FOLDS.items():
        y, st = V.relabel(fold, v4_label_fn(mode)); st.to_csv(OUT / f"v4_relabel_{fold}_{mode}.csv", index=False)
        log(f"[{fold}] v4 relabel: {st[['rows', 'pos_old', 'pos_new', 'flip_to_pos', 'flip_to_neg']].sum().to_dict()}")
        v4 = v4_codex(fold)
        # V4 rows are eligible (always observed) cells: an unchanged y would refit the identical frozen models
        v4h = v4 if not (st.flip_to_pos.sum() + st.flip_to_neg.sum()) else \
            v4_codex(fold, V.predict_cv(fold, V.fit(fold, f"hyb_{mode}", y, V4_SEEDS)[0]))
        nl = noloc_codex(fold, noloc_fit(fit_m, "cache", log=log))
        nlh = noloc_codex(fold, noloc_fit(fit_m, f"cache_hyb_{mode}", log=log))
        pickle.dump(dict(v4=v4, v4h=v4h, nl=nl, nlh=nlh), open(OUT / f"cv_{fold}_{mode}.pkl", "wb"))
        mix = lambda a, b: {p: 0.5 * a[p] + 0.5 * b[p] for p in a}
        res.append(score(fold, {"v2_bin": None, "A": mix(v4, nl), "A_hyb": mix(v4h, nlh),
                                "A_v4hyb_only": mix(v4h, nl), "A_nlhyb_only": mix(v4, nlh), "v4": v4, "v4_hyb": v4h,
                                "noloc": nl, "noloc_hyb": nlh}))
        log(f"[{fold}]\n{res[-1].round(4).to_string()}")
    return pd.concat(res, ignore_index=True)


def stage_final(mode="pure", log=print):
    """Final members on hybrid labels -> released T+30 probabilities (pickle) for A_hyb and stage 2."""
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


def out_name(name: str, mode: str) -> str:
    """Hybrid modes deliver to outputs/t2/queue_onset_<name>[_<mode>].csv; controls go to outputs/t2b/hybrid/dryrun/."""
    if mode.split("_")[0] in ("fill", "pure", "mix"):
        return f"../t2/queue_onset_{name}{'' if mode == 'pure' else '_' + mode}.csv"
    return f"hybrid/dryrun/queue_onset_{name}_{mode}.csv"


def write(onset: dict, name: str, ref="queue_onset_A.csv") -> tuple:
    """onset: {(panel, split): P30 or bool D30 [n_onset, L]} -> outputs/t2b/<name> (path from out_name; ongoing = v2 binaries)."""
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


def write_a_hyb(mode="pure", private=None):
    """private: label mode of the private-split members (default `mode`; tfb.t2b.hyb_clean_march); named after it."""
    F = {s: pickle.load(open(OUT / f"final_{m}.pkl", "rb")) for s, m in (("validation", mode), ("private", private or mode))}; on = {}
    for p in io.T2_PANELS:
        for s in ("validation", "private"):
            wi = pd.read_csv(io.t2_dir(p, s) / "window_index.csv", dtype=str); wo = wi[wi.condition == COND]
            on[(p, s)] = 0.5 * long_to_grid(F[s]["v4h"], p, s, wo) + 0.5 * F[s]["nlh"][(p, s)]
    return write(on, out_name("A_hyb", private or mode))


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


if __name__ == "__main__":
    import argparse, json, time
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="all"); ap.add_argument("--mode", default="pure")
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    lg = lambda s: print(f"[hybrid {time.strftime('%H:%M:%S')}] {s}", flush=True)
    if a.stage in ("audit", "all"):
        au = build_cache(a.mode, lg); au.to_csv(OUT / f"label_audit_{a.mode}.csv", index=False)
        lg("\n" + au.drop(columns="mode").set_index("panel").T.to_string())
    if a.stage in ("cv", "all"):
        t = stage_cv(a.mode, lg); t.to_csv(OUT / f"cv_{a.mode}.csv", index=False)
        lg("\n" + t.pivot(index="model", columns="fold", values="onset").round(4).to_string())
    if a.stage in ("final", "all"):
        stage_final(a.mode, lg)
    if a.stage in ("final", "all", "write"):
        ver, ch = write_a_hyb(a.mode); json.dump(ver, open(OUT / f"verification_{a.mode}.json", "w"), indent=1)
        ch.to_csv(OUT / f"changes_{a.mode}.csv", index=False); lg("\n" + ch.groupby("split").sum(numeric_only=True).to_string())
