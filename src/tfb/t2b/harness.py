"""Q0 — CV harness of ALL admissible Task 2 windows on the train months (official selector, no 5-per-panel quota).

Selector (configs/synthetic_release_v1.json + SCORING_SPEC, as the user fixed it on 2026-11-01): origins every 5 min
inside the day; latent label = hybrid train speed (released truth on eligible cells, imputer elsewhere) <= 0.6 v_f;
history T-60..T-5 (12 slots), horizon T+5..T+30; onset = no queued cell in the history and a queue in the horizon
(the first such T, i.e. the queue is born at T+30); ongoing = an established queue in the history (>= 2 queued slots on
one link, competition contract) and a queue in the horizon, persistence IoU (state at T-5 repeated, all cells) <= 0.9;
history coverage (eligible cells) >= 0.7; spacing >= 360 min within (panel, condition); every window kept.
Scoring: official per-window IoU on eligible cells against the raw truth (tfb.t2b.evaluate.window_iou).

Members (honest cuts, same as the family A folds so that the frozen V4 models apply):
  cut octnov: fit Jun..Sep, evaluate Oct, Nov, Dec;  cut janfeb: fit Jun..Dec, evaluate Jan, Feb.
  onset   v4 = frozen V4 normal (3 seeds, features from the deployment builder on the training overlay),
          nl = family B noloc on hybrid windows (cache_hyb_pure, 3 seeds);  A_hyb = decode30(0.5 v4 + 0.5 nl)
  ongoing nl = family B noloc raw (cache, 3 seeds), decoded;  persistence = last observed history state;
          E25 (0.75 family A v2 + 0.25 nl) only where family A's dense eval windows provide probability_v2 (v2 is not
          reproducible on new train windows without family A's dataset pipeline).

  PYTHONPATH=src python -m tfb.t2b.harness [--panels ...]   -> outputs/t2b/harness/{windows.parquet, iou.parquet}
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
import pandas as pd
from .. import cache, io
from ..t2.decode import expected_iou_decode
from ..t2.v2_experiment import load_model
from . import build, codex, cv, data, evaluate as ev, hybrid as H, model as M, windows as W
from .ladder import decode30

OUT = io.OUTPUTS / "t2b" / "harness"
DAY, SPACING = 288, 72
CUTS = {"octnov": (["2030-06", "2030-07", "2030-08", "2030-09"], ["2030-10", "2030-11", "2030-12"]),
        "janfeb": (["2030-06", "2030-07", "2030-08", "2030-09", "2030-10", "2030-11", "2030-12"], ["2031-01", "2031-02"])}
ON, OG = "queue_onset", "queue_ongoing"
V4_SEEDS = (42, 137, 271)


def log(s):
    print(f"[harness {time.strftime('%H:%M:%S')}] {s}", flush=True)


def select(panel: str, mode: str = "pure") -> pd.DataFrame:
    """Every admissible window of a train panel, per-condition 360-min clock, no quota. mode: hybrid label mode of the
    latent queue state ("pure" = the v4t1 imputer, transductive; "mix" with H.HYB_MIX = the clean imputer's labels)."""
    d = data.load(panel, "train")
    ls = H.label_speed(panel, mode); ls = np.where(np.isfinite(ls), ls, d["speed"])
    lab = np.nan_to_num(ls, nan=999) <= d["vcut"]
    cand = W.candidates(lab, d["elig"])
    cand["pers_iou"] = W.candidates(lab, np.ones_like(d["elig"])).pers_iou.to_numpy()   # all-cell persistence
    cand["birth"] = [int(lab[t + 1:t + 7].any(1).argmax()) for t in cand.t]      # first queued horizon slot (latent)
    cand["elig_truth"] = [bool((np.nan_to_num(d["speed"][t + 1:t + 7], nan=999) <= d["vcut"])[d["elig"][t + 1:t + 7]].any()) for t in cand.t]
    ok = cand[(cand.coverage >= 0.7) & (cand.pers_iou <= 0.9) & ((cand.condition != ON) | (cand.birth == 5))]
    keep = []
    for cond in (ON, OG):
        last = -10 ** 9
        for r in ok[ok.condition == cond].itertuples():
            if r.t - last >= SPACING:
                keep.append(r.Index); last = r.t
    w = cand.loc[sorted(keep)].copy()
    w["panel"] = panel; w["date"] = [d["dates"][t // DAY] for t in w.t]; w["month"] = w.date.astype(str).str[:7]
    w["block_day"] = 0; w["dirty_skipped"] = int((ok.condition == "queue_onset_dirty").sum())
    return w.reset_index(drop=True)


def bundle(panel: str, w: pd.DataFrame) -> dict:
    d = data.load(panel, "train"); ts = w.t.to_numpy()
    S, Q, RON, ROFF = build._pack(d, ts)
    raw = np.nan_to_num(d["speed"], nan=999) <= d["vcut"]; den = build.den_labels(d["speed"], d["vcut"])
    B = dict(meta=w.reset_index(drop=True), S=S, Q=Q, RON=RON, ROFF=ROFF, Y=np.stack([raw[t + 1:t + 7] for t in ts]),
             YD=np.stack([den[t + 1:t + 7] for t in ts]), E=np.stack([d["elig"][t + 1:t + 7] for t in ts]))
    B["net"] = ev.load_train(panel)["net"]
    return B


def persistence(B: dict) -> np.ndarray:
    """Last observed speed in T-60..T (masked row at T included) <= v_cut, repeated over the horizon."""
    S = B["S"]; last = np.full(S.shape[::2], np.nan, np.float32)
    for k in range(S.shape[1]):
        m = np.isfinite(S[:, k]); last[m] = S[:, k][m]
    q = np.nan_to_num(last, nan=999) <= B["net"]["vcut"]
    return np.repeat(q[:, None], 6, 1)


def noloc_grid(ms, panel, B, cond, cut) -> np.ndarray:
    """[n_cond_windows, 6, L] family B noloc probabilities for the windows of `cond` in the cut's eval months (meta order)."""
    fit_m, ev_m = CUTS[cut]; parts = []
    for (X, _, idx, _, _, wpos) in cv.rows(panel, B, cond, ev_m, fit_m):
        parts.append((M.to_grid(np.mean([M.predict(m, X) for m in ms], 0), idx, len(wpos), len(B["net"]["v_f"])), wpos))
    parts.sort(key=lambda t: t[1][0])
    return np.concatenate([g for g, _ in parts]), np.concatenate([w for _, w in parts])


def v4_p30(panel: str, origins: np.ndarray, cut: str, chunk: int = 64) -> np.ndarray:
    """[n, L] frozen V4 (normal) T+30 probabilities in t2b link order."""
    from ..t2 import v4_hyb as V
    ms = [load_model(V.OUT / "v4/models" / cut / "normal" / f"seed{s}.txt", 4) for s in V4_SEEDS]
    o = data.downstream_order(panel, cache.load(panel, "train")["links"]); out = []
    for i in range(0, len(origins), chunk):
        X = V.window_features(panel, [int(t) for t in origins[i:i + chunk]])
        x = np.stack([X[int(t)][cut] for t in origins[i:i + chunk]])
        out.append(np.mean([m.predict(x.reshape(-1, x.shape[-1])).reshape(x.shape[:2]) for m in ms], 0)[:, o])
    return np.concatenate(out)


def v2_probs(panel: str, B: dict) -> dict:
    """{window position: probability_v2 [6, L]} from family A's dense eval windows that coincide with ours (by origin)."""
    out = {}; t_of = {t: i for i, t in enumerate(B["meta"].t.to_numpy())}
    for fold in cv.FOLDS:
        C = codex.windows(fold, panel, official_only=False)
        if "probability_v2" not in C["codex"]:
            continue
        for i, t in enumerate(C["meta"].t.to_numpy()):
            if t in t_of:
                out[t_of[t]] = C["codex"]["probability_v2"][i]
    return out


def dec6(P: np.ndarray) -> np.ndarray:
    return np.stack([expected_iou_decode(x, force_nonempty=True) for x in P]).astype(bool)


def run(panels=io.T2_PANELS) -> tuple[pd.DataFrame, pd.DataFrame]:
    OUT.mkdir(parents=True, exist_ok=True)
    Ws = {}
    for p in panels:
        f = OUT / f"windows_{p}.parquet"
        Ws[p] = pd.read_parquet(f) if f.exists() else select(p); Ws[p].to_parquet(f)
        log(f"windows {p}: " + ", ".join(f"{c}={n}" for c, n in Ws[p].condition.value_counts().items()) + f"; dirty onset skipped {Ws[p].dirty_skipped.iloc[0]}")
    Bs = {p: bundle(p, Ws[p]) for p in panels}
    rows = []; probs = {}   # (cut, panel, cond) -> window positions, member probabilities (for the decoder work, Q3)
    for cut, (fit_m, ev_m) in CUTS.items():
        ms_on = H.noloc_fit(fit_m, "cache_hyb_pure", log=log, cond=ON, extra=True)
        ms_og = H.noloc_fit(fit_m, "cache", log=log, cond=OG, extra=False)
        for p in panels:
            B = Bs[p]; meta = B["meta"]; v2 = v2_probs(p, B)
            for cond, ms in ((ON, ms_on), (OG, ms_og)):
                mk = np.flatnonzero((meta.condition == cond).to_numpy() & meta.month.isin(ev_m).to_numpy())
                if not len(mk):
                    continue
                G, wpos = noloc_grid(ms, p, B, cond, cut); assert (wpos == mk).all()
                Y, E = B["Y"][mk], B["E"][mk]; preds = {"persistence": persistence(B)[mk], "nl": None}
                probs[(cut, p, cond)] = dict(mk=mk, nl=G.astype(np.float32))
                if cond == ON:
                    v4 = v4_p30(p, meta.t.to_numpy()[mk], cut); probs[(cut, p, cond)]["v4"] = v4.astype(np.float32)
                    preds.update(v4=decode30(v4), nl=decode30(G[:, 5]), A_hyb=decode30(0.5 * v4 + 0.5 * G[:, 5]))
                else:
                    preds["nl"] = dec6(G)
                    has = np.array([i in v2 for i in mk])
                    if has.any():
                        P2 = np.stack([v2[i] for i in mk[has]]); probs[(cut, p, cond)]["v2"] = (has, P2.astype(np.float32))
                        for name, D in (("v2", dec6(P2)), ("E25", dec6(0.75 * P2 + 0.25 * G[has])), ("nl@v2", preds["nl"][has])):
                            rows.append(pd.DataFrame(dict(cut=cut, panel=p, condition=cond, model=name, t=meta.t.to_numpy()[mk[has]],
                                                          month=meta.month.to_numpy()[mk[has]], iou=ev.window_iou(D, Y[has], E[has]))))
                for name, D in preds.items():
                    rows.append(pd.DataFrame(dict(cut=cut, panel=p, condition=cond, model=name, t=meta.t.to_numpy()[mk],
                                                  month=meta.month.to_numpy()[mk], iou=ev.window_iou(D, Y, E))))
                log(f"{cut} {p} {cond}: n {len(mk)} " + " ".join(f"{k} {rows[-1 - j].iou.mean():.3f}" for j, k in enumerate(reversed(list(preds)))))
    import pickle
    pickle.dump(probs, open(OUT / "probs.pkl", "wb"))
    iou = pd.concat(rows, ignore_index=True); iou.to_parquet(OUT / "iou.parquet")
    win = pd.concat(Ws.values(), ignore_index=True); win.to_parquet(OUT / "windows.parquet")
    return win, iou


def table(iou: pd.DataFrame) -> pd.DataFrame:
    """condition x model: n windows, pooled mean IoU +- SE, family macro (panel -> family -> mean)."""
    out = []
    for (cond, model), g in iou.groupby(["condition", "model"]):
        out.append(dict(condition=cond, model=model, n=len(g), mean=g.iou.mean(), se=g.iou.std(ddof=1) / np.sqrt(len(g)),
                        macro=g.groupby("panel").iou.mean().groupby(lambda p: ev.FAMILY[p]).mean().mean(),
                        **{m: round(x, 4) for m, x in g.groupby("month").iou.mean().items()}))
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--panels", nargs="*", default=list(io.T2_PANELS)); a = ap.parse_args()
    win, iou = run(a.panels)
    t = table(iou); t.to_csv(OUT / "table.csv", index=False)
    print(t.to_string(index=False))
    print(win.groupby(["panel", "condition"]).size().unstack(fill_value=0).to_string())
    log("HARNESS DONE")
