"""CV analysis: our windows (LB-like) + head-to-head/ensembles on family A's official-like windows."""
from __future__ import annotations
import pickle
import numpy as np
import pandas as pd
from .. import io
from . import cv, evaluate as ev, features as FT, model as M

OUT = io.OUTPUTS / "t2b"


def _grids(probs):
    """{(panel_key, cond, wpos0): {variant: P}, wpos}"""
    g = {}
    for (v, p, c, w0), (P, wpos) in probs.items():
        g.setdefault((p, c, w0), [{}, wpos])[0][v] = P
    return g


def _rare(B, wpos, cond, fold, panel):
    """Non-recurrent flag: queued cells (at T for ongoing, truth T+30 for onset) mostly at links whose
    fit-period queue frequency at that hour is < .05."""
    pr = M.priors(panel, cv._days(None, cv.FOLDS[fold][0]))
    t = B["meta"].t.to_numpy()[wpos]; hr = ((t % 288) // 12).astype(int)
    out = np.zeros(len(wpos), bool)
    for i, w in enumerate(wpos):
        if cond == "queue_ongoing":
            s = FT.ffill_slots(B["S"][w:w + 1])[0, 12]; q = np.nan_to_num(s, nan=999) <= B["net"]["vcut"]
            f = pr["ph"][q, hr[i]]
        else:
            f = pr["ph"][B["Y"][w, 5], ((t[i] + 6) % 288) // 12]
        out[i] = len(f) > 0 and np.mean(f < 0.05) >= 0.5
    return out


def window_table(fold, probs, codex_b, blends=None):
    blends = blends or {"blend": {"full": 0.5, "noloc": 0.5}}
    Bs = {p: ev.load_train(p) for p in io.T2_PANELS}
    rows = []
    for (pk, cond, _), (Pv, wpos) in _grids(probs).items():
        panel = pk.split("@")[0]
        if "@" in pk:
            cb = codex_b[pk]; Y, YD, E = cb["Y"][wpos], cb["YD"][wpos], cb["E"][wpos]
            meta = cb["meta"].iloc[wpos]
            cands = dict(Pv)
            for b, w in blends.items():
                cands[b] = sum(w[k] * Pv[k] for k in w) / sum(w.values())
            cx = {k: v for k, v in cb["codex"].items() if len(v) == len(cb["meta"])}
            dec = {k: M.decode(P) for k, P in cands.items()}
            for k in ("v1", "v2", "full", "no_location"):
                if f"prediction_{k}" in cx:
                    dec[f"codex_{k}"] = cx[f"prediction_{k}"][wpos].astype(bool)
            mine = cands[next(iter(blends))]
            dec["ens_prob_v2"] = M.decode(0.5 * mine + 0.5 * cx["probability_v2"][wpos])
            dec["ens_bin_v2"] = M.decode(0.5 * mine + 0.5 * cx["prediction_v2"][wpos])
            dec["ens_bin_v1"] = M.decode(0.5 * mine + 0.5 * cx["prediction_v1"][wpos])
            rare = np.zeros(len(wpos), bool); bd = np.zeros(len(wpos), int); src = "codex"
        else:
            B = Bs[panel]; Y, YD, E = B["Y"][wpos], B["YD"][wpos], B["E"][wpos]
            meta = B["meta"].iloc[wpos]; cands = dict(Pv)
            for b, w in blends.items():
                cands[b] = sum(w[k] * Pv[k] for k in w) / sum(w.values())
            dec = {k: M.decode(P) for k, P in cands.items()}
            rare = _rare(B, wpos, cond, fold, panel); bd = meta.block_day.to_numpy(); src = "ours"
        if cond == "queue_onset":
            for k in dec:
                dec[k][:, :5] = False
        for k, D in dec.items():
            rows.append(pd.DataFrame({"fold": fold, "src": src, "panel": panel, "condition": cond, "model": k,
                                      "t": meta.t.to_numpy(), "block_day": bd, "rare": rare,
                                      "iou": ev.window_iou(D, Y, E), "iou_den": ev.window_iou(D, YD, E),
                                      "n_pred": D.sum((1, 2)), "n_true": (Y & E).sum((1, 2))}))
    return pd.concat(rows, ignore_index=True)


def summarize(df):
    out = []
    for (fold, src, model), g in df.groupby(["fold", "src", "model"]):
        for sub, gg in (("all", g), ("lb_like", g[g.block_day == 1] if src == "ours" else g)):
            if src == "codex" and sub == "lb_like":
                continue
            r = ev.macro(gg); rd = ev.macro(gg, "iou_den")
            rare = gg[gg.rare]; nr = gg[~gg.rare]
            out.append(dict(fold=fold, src=src, set=sub, model=model, n=len(gg), S_queue=r["total"], onset=r["onset"],
                            ongoing=r["ongoing"], S_queue_den=rd["total"], onset_den=rd["onset"], ongoing_den=rd["ongoing"],
                            n_rare=len(rare), iou_rare=rare.iou.mean() if len(rare) else np.nan, iou_recurrent=nr.iou.mean(),
                            **{p: r["panel"].get(p, np.nan) for p in io.T2_PANELS}))
    return pd.DataFrame(out)


def load_fold(fold, tag=None):
    z = pickle.load(open(OUT / "cv" / f"probs_{fold}{'_' + tag if tag else ''}.pkl", "rb"))
    return z["probs"], z["codex"]
