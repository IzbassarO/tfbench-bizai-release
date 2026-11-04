"""Q2 — structure / balance features in the family B noloc members, measured on the Q0 harness (tfb.t2b.harness).

Features (tfb.t2b.features.q2_features, flag FT.Q2): cap_next / cap, lanes_next - lanes, k_crit_next / k_crit, km to the
nearest downstream capacity drop (< 0.9 cap); dN over 15 / 30 / 60 min on the corridor and on the 6 links ending at j
(N = q / v * len from observed cells <= T, storage-normalised); upstream flow / cap; inflow to the cluster tail vs
0.69 * cap of the head; density / k_crit at T and max over 3 links downstream; tail movement over 6 slots.
Onset: noloc on hybrid windows (cache_hyb_pure, extras) -> nl_q2, A_hyb_q2 = decode30(0.5 v4 + 0.5 nl_q2).
Ongoing: noloc raw (cache) -> nl_q2; E25_q2 = decode(0.75 v2 + 0.25 nl_q2) on the windows where v2 exists.
Paired dIoU vs the recipe members on the same windows, both cuts, by condition.

  PYTHONPATH=src python -m tfb.t2b.q2feat   -> outputs/t2b/harness/q2.csv
"""
from __future__ import annotations
import pickle, time
import numpy as np
import pandas as pd
from .. import io
from . import evaluate as ev, features as FT, harness as HS, hybrid as H, model as M
from .ladder import decode30

ON, OG = HS.ON, HS.OG
Q2_COLS = ("dN_", "cap_next", "lanes_next", "kcrit_next", "d_capdrop", "q_up_cap", "qin_tail", "k_down_max3", "k_T_r", "c_vtail6")
# LOMO permutation importance < 0.002 (tfb.t2b.onset_lomo, see report, Sec. III-C): dropped in the trimmed onset member nl_q2_trim
Q2_BALLAST = ("dN_cor_15", "qin_tail_r", "q_up_cap", "dN_seg_60", "dN_cor_30", "k_T_r", "c_vtail6")


def log(s):
    print(f"[q2 {time.strftime('%H:%M:%S')}] {s}", flush=True)


def run() -> pd.DataFrame:
    probs = pickle.load(open(HS.OUT / "probs.pkl", "rb")); rows = []
    Ws = {p: pd.read_parquet(HS.OUT / f"windows_{p}.parquet") for p in io.T2_PANELS}
    Bs = {p: HS.bundle(p, Ws[p]) for p in io.T2_PANELS}
    FT.Q2["on"] = True
    for cut, (fit_m, ev_m) in HS.CUTS.items():
        for cond, cdir, extra in ((ON, "cache_hyb_pure", True), (OG, "cache", False)):
            ms = H.noloc_fit(fit_m, cdir, log=log, cond=cond, extra=extra)
            names = [n for n in M.panel_rows(HS.cv._sub(Bs[io.T2_PANELS[0]], np.array([0])), Bs[io.T2_PANELS[0]]["net"],
                                             Bs[io.T2_PANELS[0]]["meta"].iloc[[0]], M.priors(io.T2_PANELS[0], HS.cv._days(None, fit_m)), cond)[1]]
            keep = [n for n in names if n not in M.LOC]
            imp = pd.Series(np.mean([m[0].feature_importance("gain") for m in ms], 0), index=keep).sort_values(ascending=False); tot = imp.sum()
            q2n = [n for n in keep if n.startswith(Q2_COLS)]
            log(f"{cut} {cond}: q2 gain share {imp[q2n].sum() / tot:.3f}; top q2: " + ", ".join(f"{n} #{list(imp.index).index(n) + 1}" for n in imp[q2n].sort_values(ascending=False).index[:5]))
            for p in io.T2_PANELS:
                B = Bs[p]; d = probs[(cut, p, cond)]; mk = d["mk"]; W = Ws[p]
                G, wpos = HS.noloc_grid(ms, p, B, cond, cut); assert (wpos == mk).all()
                Y, E = B["Y"][mk], B["E"][mk]; et = W.elig_truth.to_numpy()[mk]; ts = W.t.to_numpy()[mk]
                if cond == ON:
                    dec = {"A_hyb": decode30(0.5 * d["v4"] + 0.5 * d["nl"][:, 5]), "A_hyb_q2": decode30(0.5 * d["v4"] + 0.5 * G[:, 5]),
                           "nl": decode30(d["nl"][:, 5]), "nl_q2": decode30(G[:, 5])}
                else:
                    dec = {"nl": HS.dec6(d["nl"]), "nl_q2": HS.dec6(G)}
                    if "v2" in d:
                        has, P2 = d["v2"]; idx = np.flatnonzero(has)
                        for name, D in (("E25", HS.dec6(0.75 * P2 + 0.25 * d["nl"][has])), ("E25_q2", HS.dec6(0.75 * P2 + 0.25 * G[has]))):
                            rows.append(pd.DataFrame(dict(cut=cut, panel=p, condition=cond, model=name, t=ts[idx], elig_truth=et[idx], iou=ev.window_iou(D, Y[idx], E[idx]))))
                for name, D in dec.items():
                    rows.append(pd.DataFrame(dict(cut=cut, panel=p, condition=cond, model=name, t=ts, elig_truth=et, iou=ev.window_iou(D, Y, E))))
                log(f"{cut} {p} {cond}: " + " ".join(f"{k} {ev.window_iou(D, Y, E).mean():.4f}" for k, D in dec.items()))
    FT.Q2["on"] = False
    df = pd.concat(rows, ignore_index=True); df.to_parquet(HS.OUT / "q2_iou.parquet")
    return df


def paired(df: pd.DataFrame, pairs=(("A_hyb_q2", "A_hyb"), ("nl_q2", "nl"), ("E25_q2", "E25"))) -> pd.DataFrame:
    out = []; key = ["cut", "panel", "t"]
    for cond, g in df.groupby("condition"):
        for a, b in pairs:
            if a not in set(g.model) or b not in set(g.model):
                continue
            x = g[g.model == a].set_index(key).iou; y = g[g.model == b].set_index(key).iou; d = (x - y.reindex(x.index)).dropna()
            row = dict(condition=cond, model=a, ref=b, n=len(d), mean=x.mean(), ref_mean=y.reindex(x.index).mean(), delta=d.mean(),
                       se_delta=d.std(ddof=1) / np.sqrt(len(d)), changed=int((d != 0).sum()))
            for c, gc in d.groupby(level=0):
                row[f"{c}_delta"] = gc.mean()
            out.append(row)
    return pd.DataFrame(out)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    df = run(); t = paired(df[df.elig_truth]); t.to_csv(HS.OUT / "q2.csv", index=False)
    print(t.round(4).to_string(index=False))
    log("Q2 DONE")
