"""Ongoing candidate (see report, Sec. III-C): family B noloc ongoing member "v2l" (slow-link features + harness training windows)
blended with family A v2, ongoing rows only, on top of candidate_a2692183afdc1226.

Member: recipe noloc ongoing features + FT.V2L slow columns (tfb.t2b.v2like), training windows = recipe cache +
every admissible harness ongoing window (outputs/t2b/cache_hw), all train months, 3 seeds, denoised labels, no
location priors. Ongoing decode = exact expected-IoU top-m of W * v2 + (1 - W) * v2l over the 6 horizons (W = 0.5).
Harness (Q0, 292 windows where v2 exists): E50 vs E25 +0.0061 +- 0.0019, E25-weight +0.0052 +- 0.0014; all 1923
windows: member alone 0.855 vs 0.799.

  PYTHONPATH=src python -m tfb.t2b.ongoing_v2l [--w 0.5]
    -> outputs/t2b/v2l/final_probs_v2l.pkl, outputs/t2/queue_onset_Ahyb_mix_clean_E50v2l.csv,
       outputs/submissions/cand_a2692183_og50v2l.{csv,manifest.json,sha256}
"""
from __future__ import annotations
import argparse, hashlib, json, pickle, time
import numpy as np
import pandas as pd
import polars as pl
from .. import io
from ..t2.decode import expected_iou_decode
from ..submit.checks import check
from ..submit.diff import diff
from . import codex, cv, evaluate as ev, features as FT, harness as HS, hybrid as H, model as M, run as R
from .ladder import T2, _prob_grid
from .v2like import build_cache_hw

OUT = io.OUTPUTS / "t2b" / "v2l"
BASE = io.OUTPUTS / "submissions" / "candidate_a2692183afdc1226.csv"
DEPLOYED = "queue_onset_Ahyb_mix_clean_E25.csv"
KEYS = ["window_id", "timestamp", "link_id"]


def log(s):
    print(f"[v2l {time.strftime('%H:%M:%S')}] {s}", flush=True)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def fit_final(seeds=H.SEEDS, q2=False, alive=False, cache="cache_hw", windows=None):
    """v2l member on all train months; q2 / alive / seeds = the see report, Sec. III-C variants (tfb.t2b.v2like VARIANTS v2l_*);
    cache / windows: the training cache and the admissible windows it adds (default the harness's, pure labels)."""
    from .q2feat import Q2_BALLAST
    build_cache_hw(log, windows, cache)
    FT.V2L["slow"] = True; FT.Q2["on"] = q2; FT.ALIVE["on"] = alive
    try:
        ms = H.noloc_fit(R.ALL_MONTHS, cache, log=log, cond=HS.OG, extra=False, seeds=seeds, drop=tuple(M.LOC) + (Q2_BALLAST if q2 else ()))
        P = H.released_noloc(ms, cond=HS.OG, h=None)
        p0 = io.T2_PANELS[0]; B = ev.load_train(p0, cache)
        names = M.panel_rows(cv._sub(B, np.array([0])), B["net"], B["meta"].iloc[[0]], M.priors(p0, cv._days(None, R.ALL_MONTHS)), HS.OG)[1]
    finally:
        FT.V2L["slow"] = False; FT.Q2["on"] = FT.ALIVE["on"] = False
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


def assemble(qfile: str, cand: str, w: float, extra: dict, base_csv=BASE, base_label="candidate_a2692183afdc1226 (sha f1841210)",
             onset_ok=False, changed=None, evidence=None) -> dict:
    """Full submission = base candidate with queue_pred replaced from <qfile> (ongoing windows; onset too when onset_ok)."""
    q = pd.read_csv(T2 / qfile, dtype={k: str for k in KEYS})
    key = (pl.read_csv(io.RELEASE / "submission_key.csv", infer_schema_length=0).filter(pl.col("task") == "queue")
           .select(["submission_id"] + KEYS).with_columns(pl.col("submission_id").cast(pl.Int64)))
    new = pl.from_pandas(q[KEYS + ["queue_pred"]]).join(key, on=KEYS, how="inner"); assert new.height == key.height == 174000
    base = pl.read_csv(base_csv, schema_overrides={"submission_id": pl.Int64, "task": pl.Utf8, "speed_kmh": pl.Float64, "flow_vph": pl.Float64, "queue_pred": pl.Float64, "path_flow": pl.Float64})
    sub = (base.join(new.select(["submission_id", pl.col("queue_pred").cast(pl.Float64).alias("qn")]), on="submission_id", how="left")
           .with_columns(pl.coalesce(["qn", "queue_pred"]).alias("queue_pred")).select(base.columns))
    out = io.OUTPUTS / "submissions"; csv = out / f"{cand}.csv"; sub.write_csv(csv, float_precision=4)
    errs = check(csv); assert not errs, errs
    d = diff(csv, base_csv); assert all(r["task"] == "queue" for r in d["by"]), d
    # every changed row must be an ongoing-window row (unless the onset rows are meant to change too)
    on = {wid for p in io.T2_PANELS for s in ("validation", "private") for wid in pd.read_csv(io.t2_dir(p, s) / "window_index.csv", dtype=str).query("condition == 'queue_onset'").window_id}
    m = sub.join(base.select(["submission_id", pl.col("queue_pred").alias("qb")]), on="submission_id").filter(pl.col("queue_pred") != pl.col("qb")).join(key, on="submission_id")
    n_on = m.filter(pl.col("window_id").is_in(list(on))).height
    assert onset_ok or not n_on, "onset rows changed"
    h = sha(csv)
    man = dict(name=cand, created=time.strftime("%Y-%m-%dT%H:%M:%S"), base=base_label, sha256=h, rows=sub.height,
               changed=changed or (f"Queue ONGOING rows only: decode({w} v2 + {1 - w:.2f} family B noloc v2l) over 6 horizons; v2l = recipe noloc ongoing + slow-link features (FT.V2L slow), "
                                   "training windows = cache + harness admissible ongoing windows (cache_hw), 3 seeds, all train months. Onset / State / ODME = base"),
               evidence=evidence or "see report, Sec. III-C (E266): Q0 harness 292 v2 windows E50 vs E25 +0.0061 +- 0.0019 (both cuts +), E25-weight +0.0052 +- 0.0014; member alone 0.855 vs 0.799 on 1923",
               queue_file=qfile, checks="OK", diff_vs_base=d, changed_rows_ongoing_only=not n_on, changed_onset_rows=n_on, **extra)
    (out / f"{cand}.manifest.json").write_text(json.dumps(man, indent=1, default=str)); (out / f"{cand}.sha256").write_text(f"{h}  {cand}.csv\n")
    log(f"{cand}: sha256 {h}  changed rows {d['changed_rows']} ({d['by']})")
    return man


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--w", type=float, default=0.5); ap.add_argument("--variant", default="v2l")
    a = ap.parse_args()
    from .v2like import VARIANTS
    cfg = VARIANTS[a.variant]; vtag = a.variant.replace("_", "")
    OUT.mkdir(parents=True, exist_ok=True); tag = f"E{int(round(a.w * 100))}{vtag}"
    ms, P, keep = fit_final(seeds=cfg.get("seeds", H.SEEDS), q2=cfg.get("q2", False), alive=cfg.get("alive", False))
    pickle.dump(dict(v2l=P, keep=keep), open(OUT / f"final_probs_{vtag}.pkl", "wb"))
    imp = pd.Series(np.mean([m[0].feature_importance("gain") for m in ms], 0), index=keep); imp /= imp.sum()
    slow = [n for n in keep if n.startswith(("slow", "cor_", "up_min", "down_min"))]
    log(f"slow gain share {imp[slow].sum():.3f}; top: {imp.sort_values(ascending=False).head(12).round(3).to_dict()}")
    imp.sort_values(ascending=False).to_csv(OUT / f"importance_{vtag}.csv")
    qfile = f"queue_onset_Ahyb_mix_clean_{tag}.csv"; ver, ch = write_queue(P, a.w, qfile); ch.to_csv(OUT / f"changes_{tag}.csv", index=False)
    log("\n" + ch.to_string(index=False)); log("\n" + ch.groupby("split").sum(numeric_only=True).to_string())
    extra = dict(variant=a.variant, config=cfg, slow_gain_share=float(imp[slow].sum()), ongoing_changes=ch.to_dict("records"))
    assemble(qfile, f"cand_a2692183_og{int(round(a.w * 100))}{vtag}", a.w, extra)
    log("DONE")
