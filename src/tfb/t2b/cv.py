"""Temporal CV on LB-like windows: fit on earlier months, evaluate held-out months.

Training rows get location priors computed without their own month (no in-sample prior);
evaluation rows get priors from all fit days. Labels for fitting: conservative denoised truth (build.den_labels).
Scored windows: block_day >= 1 (selector scans); block_day == -1 are training-only extras.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .. import io
from . import evaluate as ev, model as M, tm

FOLDS = {"janfeb": (["2030-06", "2030-07", "2030-08", "2030-09", "2030-10", "2030-11", "2030-12"], ["2031-01", "2031-02"]),
         "octnov": (["2030-06", "2030-07", "2030-08", "2030-09"], ["2030-10", "2030-11"])}
CONDS = ("queue_onset", "queue_ongoing")


def _days(B, months):
    d = pd.to_datetime(pd.Series(io.split_dates("train")))
    return np.flatnonzero(d.dt.strftime("%Y-%m").isin(months).to_numpy())


def _sub(B, mask):
    return {k: B[k][mask] for k in ("S", "Q", "RON", "ROFF", "Y", "YD", "E")}


TM = {"mode": None}  # None = train priors only; "days" / "sameday" = test-month priors (tm.py)
LAB = {"raw": False}  # fit labels: denoised YD (recipe) or raw Y (tfb.t2b.v2like)


def rows(panel, B, cond, months, prior_months, loo=False, extra=False):
    """Rows of windows in `months`; priors from prior_months (minus own month if loo),
    optionally blended with the window's own month's masked data before T (TM['mode'])."""
    meta = B["meta"]; out = []
    for mon in months:
        mk = ((meta.month == mon) & (meta.condition == cond) & ((meta.block_day != -1) | extra)).to_numpy()
        if not mk.any():
            continue
        pm = [m for m in prior_months if not (loo and m == mon)]
        pr = M.priors(panel, _days(B, pm))
        if TM["mode"]:
            pr = tm.window_priors(panel, "train", meta.t.to_numpy()[mk], pr, TM["mode"])
        X, names, idx = M.panel_rows(_sub(B, mk), B["net"], meta[mk], pr, cond)
        sb = _sub(B, mk)
        y = sb["Y" if LAB["raw"] else "YD"][idx[:, 0], idx[:, 1] - 1, idx[:, 2]]; e = sb["E"][idx[:, 0], idx[:, 1] - 1, idx[:, 2]]
        out.append((X, names, idx, y, e, np.flatnonzero(mk)))
    return out


def run_fold(fold: str, panels=io.T2_PANELS, variants=None, seeds=(42,), log=print, extra=None, train_extra=False):
    """extra: {key 'PANEL@tag': bundle with meta month/condition} predicted with the same fold models."""
    fit_m, ev_m = FOLDS[fold]
    variants = variants or {"full": (), "noloc": M.LOC}
    Bs = {p: ev.load_train(p) for p in panels}
    for k, B in (extra or {}).items():
        B.setdefault("net", Bs[k.split("@")[0]]["net"]); Bs[k] = B
    res = []; probs = {}
    for cond in CONDS:
        tr = {p: rows(p, Bs[p], cond, fit_m, fit_m, loo=True, extra=train_extra[cond] if isinstance(train_extra, dict) else train_extra) for p in panels}
        te = {p: rows(p.split("@")[0], Bs[p], cond, ev_m, fit_m) for p in Bs}
        names = next(r[1] for p in panels for r in tr[p])
        Xtr = np.concatenate([r[0] for p in panels for r in tr[p]]); ytr = np.concatenate([r[3] for p in panels for r in tr[p]])
        etr = np.concatenate([r[4] for p in panels for r in tr[p]])
        Xtr, ytr = Xtr[etr], ytr[etr]  # train on observed (eligible) cells only
        log(f"[{fold}] {cond}: train rows {len(ytr):,} pos {ytr.mean():.4f}")
        for vname, drop in variants.items():
            ms = [M.fit(Xtr, ytr, names, drop=drop, seed=s) for s in seeds]
            for p in Bs:
                for (X, _, idx, y, e, wpos) in te[p]:
                    pr = np.mean([M.predict(m, X) for m in ms], 0)
                    L = len(Bs[p]["net"]["v_f"])
                    P = M.to_grid(pr, idx, len(wpos), L)
                    probs[(vname, p, cond, wpos[0])] = (P, wpos)
    return probs, Bs


def score(probs, Bs, scale=1.0, decode=M.decode):
    out = []
    for (vname, p, cond, _), (P, wpos) in probs.items():
        B = Bs[p]; D = decode(P, scale)
        m = B["meta"].iloc[wpos][["t", "month", "block_day", "condition", "date"]].copy()
        m["panel"] = p; m["variant"] = vname
        m["iou"] = ev.window_iou(D, B["Y"][wpos], B["E"][wpos]); m["iou_den"] = ev.window_iou(D, B["YD"][wpos], B["E"][wpos])
        out.append(m)
    return pd.concat(out, ignore_index=True)


def summary(df):
    rows = []
    for (v,), g in df.groupby(["variant"]):
        for sub, gg in (("all", g), ("lb_like", g[g.block_day == 1])):
            r = ev.macro(gg); rd = ev.macro(gg, "iou_den")
            rows.append(dict(variant=v, windows=sub, n=len(gg), total=r["total"], onset=r["onset"], ongoing=r["ongoing"],
                             total_den=rd["total"], **{k: v2 for k, v2 in r["panel"].items()}))
    return pd.DataFrame(rows)
