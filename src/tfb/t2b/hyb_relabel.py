"""Hybrid Task 2 labels rebuilt on the current best Task 1 imputer (all-in producers, RUNTIME=cpu).

Imputer = 0.5 LightGBM 8x (mean of seeds 42/7/123) + 0.5 MLP e20 (mean of seeds 42/7/123), speed only, on the
released-like hidden layer of tfb.t1.hybrid (eligible cells observed, every other train cell hidden). The current labels
(mode "pure") use LightGBM 5x seed 42 at 4000 trees. Labels are train targets only (queued = speed <= 0.6 v_f);
features stay the released data at or before T, so the online rule and the leakage tests are untouched.

Steps (durable in a shared dir, default <store>/hyb_relabel; resumable per panel; heartbeat after each durable save):
  validate  imputer on the holdouts (c_trans8 + mlp8e20 holdout models, hidden eligible cells): RMSE and queue
            recall/precision/F1 for LGBM / MLP / mix at several tree counts. Mix must match or beat LGBM on the
            worst-holdout F1, else the stage stops (no CV files, status rejected).
  build     train_hybrid_speed_mix/<panel>.parquet (tfb.t1.hybrid.build layout) with the finals at the chosen tree count
  cv        tfb.t2b.hybrid audit + stage_cv in mode "mix"; CV npz per panel/fold in the tfb.allin format:
              onset   P = 0.5 v4_hyb(mix) + 0.5 noloc_hyb(mix) at T+30; reference = A_hyb (mode pure) decoded
              ongoing P = 0.75 v2 + 0.25 noloc ongoing refit on mix labels; reference = the same on raw labels, decoded
  final     per condition whose mean CV gain > 0 (the allin rule; the strict min-fold gate is reported too):
              outputs/t2/queue_onset_AhybMix_E25.csv, queue_onset_Ahyb_E25mix.csv (copied to the shared dir)

allin queue.yaml (component onset|ongoing, runtime cpu):  producer: tfb.t2b.hyb_relabel:produce_onset | produce_ongoing
standalone: python -m tfb.t2b.hyb_relabel --results DIR [--params JSON] [--force]
"""
from __future__ import annotations
import torch  # before lightgbm (macOS OpenMP order)
import argparse, json, shutil, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from .. import cache, io
from ..t1 import features as FE, mlp as MLP, hybrid as T1H
from ..t1.base import day_mask
from ..t1.train import SPEED_CAP, holdout_days
from ..t2.decode import expected_iou_decode
from . import codex, cv, evaluate as ev, hybrid as H, run as R
from .postproc import atomic_json, write_cv
from .probavg import _fam, _prob_grid

STAGE = "hyb_relabel"
OG = "queue_ongoing"
T1 = io.OUTPUTS / "t1"
DEFAULTS = dict(w_mlp=0.5, levels=(2000, 4000, 0), lgbm=("_v6t1", "_v6t1_seed7", "_v6t1_seed123"),
                mlp=("mlp_mlp8e20", "mlp_mlp8e20_seed7", "mlp_mlp8e20_seed123"), min_gain=0.004, e25_w=0.25)


def load_mlp(path: Path):
    z = torch.load(path, weights_only=False); prep = MLP.Prep(); prep.__dict__.update(z["prep"])
    net = MLP.Net(len(prep.num), len(prep.flag), prep.n_panel, z["width"], z.get("depth", 1)); net.load_state_dict(z["state"])
    return MLP.heads(net, prep, z["mu"], z["sd"])["speed"], prep.cols


def impute(lgbms, mlps, q, pn_train, exclude_days, sel, levels, w):
    """{name: [T,L] speed at sel, NaN elsewhere}: lgbm@n, mlp, mix@n (= (1 - w) lgbm@n + w mlp), n trees (0 = all)."""
    F = FE.panel_fields(q, pn_train, exclude_days); lcols = lgbms[0].feature_name()
    names = [f"lgbm@{n}" for n in levels] + ["mlp"] + [f"mix@{n}" for n in levels]
    out = {k: np.full(sel.shape, np.nan, np.float32) for k in names}; r, c = np.nonzero(sel)
    for s in range(0, len(r), 1_000_000):
        rr, cc = r[s:s + 1_000_000], c[s:s + 1_000_000]; df = FE.build(q, F, rr, cc); base = df.base_s.to_numpy()
        rm = np.mean([(MLP.Head._cache.clear(), h.predict(df[cols]))[1] for h, cols in mlps], 0); X = df[lcols]  # Head caches by id(df)
        out["mlp"][rr, cc] = np.clip(base + rm, 3.0, SPEED_CAP)
        rl, start = np.zeros(len(rr)), 0
        for n in sorted(levels, key=lambda n: n or 10 ** 9):  # cumulative, as tfb.t1.hybrid.impute: one pass of the largest
            end = min(n or 10 ** 9, min(m.num_trees() for m in lgbms))
            if end > start:
                rl = rl + np.mean([m.predict(X, start_iteration=start, num_iteration=end - start, num_threads=8) for m in lgbms], 0); start = end
            out[f"lgbm@{n}"][rr, cc] = np.clip(base + rl, 3.0, SPEED_CAP)
            out[f"mix@{n}"][rr, cc] = np.clip(base + (1 - w) * rl + w * rm, 3.0, SPEED_CAP)
    return out


def validate(p: dict, d: Path, beat, log=print) -> pd.DataFrame:
    """Per (holdout, panel) rows saved to d/validate/ (resumable)."""
    (d / "validate").mkdir(parents=True, exist_ok=True); run = io.CACHE / "t1_runs"; parts = []
    for hold in ("last30", "autumn"):
        days = holdout_days(hold); models = None
        for panel in T1H.PANELS:
            f = d / "validate" / f"{hold}_{panel}.csv"
            if not f.exists():
                models = models or ([lgb.Booster(model_file=str(run / f"c_trans8_{hold}" / "lgbm_speed.txt"))],
                                    [load_mlp(run / f"mlp8e20_{hold}" / "mlp.pt")])
                pn = cache.load(panel, "train"); evm = pn.target & day_mask(pn, days)[:, None]
                q = T1H.hidden_layer(pn, ~pn.elig | evm); idx = np.flatnonzero(evm); keep = np.zeros(evm.size, bool)
                keep[np.random.default_rng(0).choice(idx, min(T1H.VAL_CELLS, len(idx)), replace=False)] = True
                evm = keep.reshape(evm.shape); vf = np.broadcast_to(pn.v_f, evm.shape)[evm]; y = pn.speed[evm]; yq = y <= 0.6 * vf; rows = []
                for k, pred in impute(*models, q, pn, days, evm, p["levels"], p["w_mlp"]).items():
                    x = pred[evm]; pq_ = x <= 0.6 * vf
                    rows.append(dict(holdout=hold, panel=panel, method=k, n=int(evm.sum()), se=float(((x - y) ** 2).sum()),
                                     tp=int((pq_ & yq).sum()), queued=int(yq.sum()), pred_queued=int(pq_.sum())))
                pd.DataFrame(rows).to_csv(f.with_suffix(".tmp"), index=False); f.with_suffix(".tmp").replace(f)
                log(f"[hyb_relabel] validate {hold} {panel}"); beat(dict(step="validate", holdout=hold, panel=panel)); del pn, q
            parts.append(pd.read_csv(f))
    t = pd.concat(parts).groupby(["holdout", "method"]).sum(numeric_only=True).reset_index()
    t["rmse"] = np.sqrt(t.se / t.n); t["recall"] = t.tp / t.queued; t["precision"] = t.tp / t.pred_queued.clip(lower=1)
    t["f1"] = 2 * t.tp / (t.queued + t.pred_queued).clip(lower=1)
    return t


def choose(t: pd.DataFrame) -> dict:
    """Best mix level and best LGBM level by worst-holdout F1 (then worst RMSE); mix passes if its F1 >= LGBM's."""
    w = t.groupby("method").agg(f1=("f1", "min"), rmse=("rmse", "max")).reset_index().sort_values(["f1", "rmse"], ascending=[False, True])
    best = lambda pre: w[w.method.str.startswith(pre)].iloc[0]
    m, l = best("mix@"), best("lgbm@")
    return dict(mix=m.method, mix_f1=float(m.f1), mix_rmse=float(m.rmse), lgbm=l.method, lgbm_f1=float(l.f1),
                lgbm_rmse=float(l.rmse), n_trees=int(m.method.split("@")[1]), accepted=bool(m.f1 >= l.f1))


def build(p: dict, n_trees: int, d: Path, beat, log=print) -> str:
    """d/train_hybrid_speed_mix/<panel>.parquet (a parquet dataset that tfb.t2b.hybrid reads with a panel filter)."""
    import pyarrow as pa, pyarrow.parquet as pq
    out = d / "train_hybrid_speed_mix"; out.mkdir(parents=True, exist_ok=True); models = None
    for panel in T1H.PANELS:
        f = out / f"{panel}.parquet"
        if f.exists():
            continue
        models = models or ([lgb.Booster(model_file=str(T1 / f"lgbm_speed{s}.txt")) for s in p["lgbm"]], [load_mlp(T1 / m / "mlp.pt") for m in p["mlp"]])
        pn = cache.load(panel, "train"); imp = ~pn.elig; T, L = imp.shape
        pred = impute(*models, T1H.hidden_layer(pn, imp), pn, None, imp, (n_trees,), p["w_mlp"])[f"mix@{n_trees}"]
        ts = (np.repeat(pn.dates, io.STEPS_PER_DAY).astype(object) + "T"
              + np.tile([f"{m // 60:02d}:{m % 60:02d}:00Z" for m in range(0, 1440, 5)], len(pn.dates)).astype(object))
        t = pa.table({"panel": pa.array([panel] * (T * L)).dictionary_encode(),
                      "timestamp": pa.array(np.repeat(ts, L)).dictionary_encode(),
                      "link_id": pa.array(np.tile(np.asarray(pn.links, object), T)).dictionary_encode(),
                      "speed_hybrid": np.where(imp, pred, pn.speed).astype(np.float32).ravel(), "is_imputed": imp.ravel()})
        pq.write_table(t, f.with_suffix(".tmp")); f.with_suffix(".tmp").replace(f)
        vf = np.broadcast_to(pn.v_f, imp.shape)
        log(f"[hyb_relabel] build {panel}: imputed {imp.sum()}, queued share imputed {(pred[imp] <= 0.6 * vf[imp]).mean():.4f}")
        beat(dict(step="build", panel=panel)); del pn
    return str(out)


def _decode(P):
    return np.stack([expected_iou_decode(x, force_nonempty=True) for x in P]).astype(bool)


def cv_files(p: dict, d: Path, beat, log=print, mode="mix", conds=("onset", "ongoing"), ref_mode="pure") -> dict:
    """CV of label mode `mode` (labels from H.HYB_MIX) -> d/cv/{onset,ongoing}_<fold>_<panel>.npz (tfb.allin format)
    + fold gains (family macro); reference = A_hyb in label mode `ref_mode` (onset) / E25 raw (ongoing)."""
    import pickle
    H.build_cache(mode, log).to_csv(d / f"label_audit_{mode}.csv", index=False)
    on = H.stage_cv(mode, log); on.to_csv(d / f"cv_onset_{mode}.csv", index=False); beat(dict(step="cv_onset"), check_budget=False)
    files, gains = {c: [] for c in conds}, {c: {} for c in conds}
    for fold, (fit_m, _) in cv.FOLDS.items():
        Zm = pickle.load(open(H.OUT / f"cv_{fold}_{mode}.pkl", "rb")); Zp = pickle.load(open(H.OUT / f"cv_{fold}_{ref_mode}.pkl", "rb"))
        nl = {tag: H.noloc_codex(fold, H.noloc_fit(fit_m, cdir, log=log, cond=OG, extra=False), cond=OG, h=None)
              for tag, cdir in (("raw", "cache"), ("mix", f"cache_hyb_{mode}"))} if "ongoing" in conds else {}
        rows = []
        for panel in io.T2_PANELS:
            B = codex.windows(fold, panel); wid = B["meta"].window_id.to_numpy()
            mk = (B["meta"].condition == "queue_onset").to_numpy(); P = np.zeros(B["Y"][mk].shape)
            P[:, 5] = 0.5 * Zm["v4h"][panel] + 0.5 * Zm["nlh"][panel]; ref = H.dec30(0.5 * Zp["v4h"][panel] + 0.5 * Zp["nlh"][panel])
            files["onset"].append(write_cv(d, "onset", fold, panel, P, B["Y"][mk], B["E"][mk], ref, wid[mk]))
            rows += [dict(cond="onset", panel=panel, iou=x, model=m) for m, D in (("mix", H.dec30(P[:, 5])), ("ref", ref))
                     for x in ev.window_iou(D, B["Y"][mk], B["E"][mk])]
            if "ongoing" not in conds:
                continue
            mk = (B["meta"].condition == OG).to_numpy(); v2 = B["codex"]["probability_v2"][mk]; w = p["e25_w"]
            assert len(nl["mix"][panel]) == mk.sum(), (fold, panel)
            P = (1 - w) * v2 + w * nl["mix"][panel]; ref = _decode((1 - w) * v2 + w * nl["raw"][panel])
            files["ongoing"].append(write_cv(d, "ongoing", fold, panel, P, B["Y"][mk], B["E"][mk], ref, wid[mk]))
            rows += [dict(cond="ongoing", panel=panel, iou=x, model=m) for m, D in (("mix", _decode(P)), ("ref", ref))
                     for x in ev.window_iou(D, B["Y"][mk], B["E"][mk])]
        r = pd.DataFrame(rows)
        for cond, g in r.groupby("cond"):
            gains[cond][fold] = float(_fam(g[g.model == "mix"]) - _fam(g[g.model == "ref"]))
        log(f"[hyb_relabel] cv {fold}: {({c: round(v[fold], 5) for c, v in gains.items()})}"); beat(dict(step="cv", fold=fold), check_budget=False)
    return dict(files=files, gains=gains,
                passed={c: bool(np.mean(list(g.values())) > 0) for c, g in gains.items()},
                strict={c: bool(min(g.values()) >= p["min_gain"]) for c, g in gains.items()})


def write_ongoing(nl: dict, src: str, dst: str, w: float) -> str:
    """outputs/t2/<dst> = <src> with ongoing rows = decode((1 - w) v2 raw + w nl) (as tfb.t2b.ladder.write_e25)."""
    sub = codex.submission(io.OUTPUTS / "t2" / src)
    pv2 = pd.read_parquet(io.OUTPUTS / "t2" / "probs_v2.parquet").rename(columns={"queue_prob": "queue_pred"}); parts = []
    for panel in io.T2_PANELS:
        for s in ("validation", "private"):
            wi = pd.read_csv(io.t2_dir(panel, s) / "window_index.csv", dtype=str); og = (wi.condition == OG).to_numpy()
            D = codex.grid(sub, panel, s, wi); D[og] = _decode((1 - w) * _prob_grid(pv2, panel, s, wi)[og] + w * nl[(panel, s)])
            parts.append(R.to_rows(D, panel, s, wi))
    pd.concat(parts, ignore_index=True).to_csv(io.OUTPUTS / "t2" / dst, index=False)
    assert R.verify([f"../t2/{dst}"])["key_sets_equal"]
    return dst


def final(cond: str, p: dict, d: Path, beat, log=print, mode="mix") -> str:
    """Released predictions for one condition -> outputs/t2/<file> + a durable copy in d."""
    if not (io.OUTPUTS / "t2b" / f"cache_hyb_{mode}").exists():
        H.build_cache(mode, log)  # VM restarted after cv
    if cond == "onset":
        H.stage_final(mode, log); H.write_a_hyb(mode)
        f = "queue_onset_AhybMix_E25.csv" if mode == "mix" else f"queue_onset_Ahyb_{mode}_E25.csv"
        H.splice(f"queue_onset_A_hyb_{mode}.csv", "queue_onset_E25.csv", f)
    else:
        ms = H.noloc_fit(R.ALL_MONTHS, f"cache_hyb_{mode}", log=log, cond=OG, extra=False)
        f = write_ongoing(H.released_noloc(ms, cond=OG, h=None), "queue_onset_Ahyb_E25.csv", "queue_onset_Ahyb_E25mix.csv", p["e25_w"])
    shutil.copy2(io.OUTPUTS / "t2" / f, d / f); beat(dict(step=f"final_{cond}"))
    return str(d / f)


def pipeline(d: Path, params=None, beat=lambda *a, **k: None, force=False, conds=("onset", "ongoing"), log=print) -> dict:
    p = dict(DEFAULTS, **(params or {})); d = Path(d); d.mkdir(parents=True, exist_ok=True)
    H.HYB_MIX = d / "train_hybrid_speed_mix"  # labels read from the durable parquet dataset

    def step(name, fn):
        f = d / f"{name}.json"
        if f.exists() and not force:
            return json.loads(f.read_text())
        v = fn(); atomic_json(f, v); beat(dict(step=name, done=True)); return v
    r = dict(stage=STAGE, runtime="cpu", params=p)

    def _val():
        t = validate(p, d, beat, log); t.to_csv(d / "validate.csv", index=False); return choose(t)
    r["validate"] = v = step("validate", _val)
    if v["accepted"]:
        r["build"] = step("build", lambda: dict(path=build(p, v["n_trees"], d, beat, log), n_trees=v["n_trees"]))
        r["cv"] = c = step("cv", lambda: cv_files(p, d, beat, log))
        r["final"] = {k: step(f"final_{k}", lambda k=k: final(k, p, d, beat, log)) for k in conds if c["passed"][k]}
    atomic_json(d / "result.json", r)
    return r


def _produce(cond, config, ctx):
    """tfb.allin producer: shared durable dir <store>/hyb_relabel, so the onset and ongoing jobs share every step."""
    r = pipeline(Path(ctx.store.root) / STAGE, config.get("params"), ctx.heartbeat, conds=(cond,))
    if not r["validate"]["accepted"]:
        raise RuntimeError(f"hyb_relabel: mix imputer rejected on the holdouts: {r['validate']}")
    out = dict(cv_files=r["cv"]["files"][cond], hyb_relabel_cv=r["cv"]["gains"][cond], hyb_relabel_strict=r["cv"]["strict"][cond])
    if cond in r["final"]:
        out["test_predictions"] = r["final"][cond]
    return out


def produce_onset(config, ctx):
    return _produce("onset", config, ctx)


def produce_ongoing(config, ctx):
    return _produce("ongoing", config, ctx)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--results", type=Path, default=io.OUTPUTS / "allin")
    ap.add_argument("--params", default="{}"); ap.add_argument("--force", action="store_true")
    a = ap.parse_args(); print(json.dumps(pipeline(a.results / STAGE, json.loads(a.params), force=a.force), indent=1, default=str))
