"""Hybrid train speed for Task 2 labels: released speed on eligible cells (finite, pct_observed >= 75), Task 1 LGBM
imputation on every other mainline cell (missing, or pct < 75 = generator time-interpolated).

usage:
  python -m tfb.t1.hybrid build <n_trees|0=all> [model_suffix ...]   (default _v4t1 _v4t1_seed7 = outputs/t1/lgbm_speed<suffix>.txt)
        -> outputs/t1/train_hybrid_speed.parquet  (panel, timestamp, link_id, speed_hybrid, is_imputed)
  python -m tfb.t1.hybrid validate <run>             (holdout models cache/t1_runs/<run>_{last30,autumn})
        -> reports/t1_runs/hybrid_<run>.csv: RMSE and queue (v <= 0.6 v_f) recall at hidden eligible cells (a sample of
           VAL_CELLS target cells per panel), imputer at several tree counts vs same-day forward fill vs linear time interpolation
Imputer input = a released-like masked layer built from the truth: eligible cells observed, every imputed cell hidden
(pct stays, as released). Speed model only; seeds averaged. Inference only (no training).
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import lightgbm as lgb
from .. import io, cache
from . import features as FE
from .base import day_mask, interp_time
from .train import SPEED_CAP, holdout_days, run_dir

OUT = io.OUTPUTS / "t1" / "train_hybrid_speed.parquet"
PANELS = io.T2_PANELS        # Task 2 labels only exist here
VAL_CELLS, LEVELS = 30_000, (1000, 2000, 4000, 0)   # 0 = all trees


def hidden_layer(pn, hide):
    q = cache.Panel(pn); obs = pn.elig & ~hide
    for k, v in (("mspeed", pn.speed), ("mflow", pn.flow), ("mocc", pn.occ)):
        q[k] = np.where(obs, v, np.nan).astype(np.float32)
    q["mpct"] = pn.pct; q["melig"] = obs
    return q


def impute(models, q, pn_train, exclude_days, sel, levels=(0,)):
    """{n_trees: [T,L] mean over speed models of base + residual of the first n_trees trees (0 = all)} at the cells sel;
    NaN elsewhere. Levels are cumulative, so all of them cost one pass of the largest."""
    F = FE.panel_fields(q, pn_train, exclude_days); cols = models[0].feature_name()
    out = {n: np.full(sel.shape, np.nan, np.float32) for n in levels}; r, c = np.nonzero(sel)
    for s in range(0, len(r), 1_000_000):
        rr, cc = r[s:s + 1_000_000], c[s:s + 1_000_000]
        df = FE.build(q, F, rr, cc)[cols]; res = np.zeros(len(rr)); start = 0
        for n in sorted(levels, key=lambda n: n or 10 ** 9):
            end = min(n or 10 ** 9, min(m.num_trees() for m in models))
            res += np.mean([m.predict(df, start_iteration=start, num_iteration=end - start, num_threads=8) for m in models], axis=0) if end > start else 0
            out[n][rr, cc] = np.clip(df.base_s.to_numpy() + res, 3.0, SPEED_CAP); start = end
    return out


def ffill_day(sp, valid):
    """Same-day forward fill (the earlier Task 2 label rule); leading gaps stay NaN."""
    a = np.where(valid, sp, np.nan).reshape(-1, io.STEPS_PER_DAY, sp.shape[1])
    return pd.DataFrame(a.transpose(1, 0, 2).reshape(io.STEPS_PER_DAY, -1)).ffill().to_numpy() \
        .reshape(io.STEPS_PER_DAY, a.shape[0], -1).transpose(1, 0, 2).reshape(sp.shape)


def build(n_trees=0, suffixes=("_v4t1", "_v4t1_seed7")):
    models = [lgb.Booster(model_file=str(io.OUTPUTS / "t1" / f"lgbm_speed{s}.txt")) for s in suffixes]
    writer = None
    for panel in PANELS:
        pn = cache.load(panel, "train"); imp = ~pn.elig
        pred = impute(models, hidden_layer(pn, imp), pn, None, imp, (n_trees,))[n_trees]
        T, L = imp.shape
        ts = (np.repeat(pn.dates, io.STEPS_PER_DAY).astype(object) + "T"
              + np.tile([f"{m // 60:02d}:{m % 60:02d}:00Z" for m in range(0, 1440, 5)], len(pn.dates)).astype(object))
        t = pa.table({"panel": pa.array([panel] * (T * L)).dictionary_encode(),
                      "timestamp": pa.array(np.repeat(ts, L)).dictionary_encode(),
                      "link_id": pa.array(np.tile(np.asarray(pn.links, object), T)).dictionary_encode(),
                      "speed_hybrid": np.where(imp, pred, pn.speed).astype(np.float32).ravel(),
                      "is_imputed": imp.ravel()})
        writer = writer or pq.ParquetWriter(OUT, t.schema)
        writer.write_table(t)
        print(f"{panel}: {T * L} cells, imputed {imp.sum()} ({imp.mean():.3f}); queued share imputed "
              f"{(pred[imp] <= 0.6 * np.broadcast_to(pn.v_f, imp.shape)[imp]).mean():.4f} vs observed "
              f"{(pn.speed[~imp] <= 0.6 * np.broadcast_to(pn.v_f, imp.shape)[~imp]).mean():.4f}", flush=True)
        del pn
    writer.close(); print("wrote", OUT)


def validate(run):
    rows_out = []
    for name in ("last30", "autumn"):
        days = holdout_days(name)
        models = [lgb.Booster(model_file=str(run_dir(run, name) / "lgbm_speed.txt"))]
        for panel in PANELS:
            pn = cache.load(panel, "train"); rows = day_mask(pn, days)
            ev = pn.target & rows[:, None]; q = hidden_layer(pn, ~pn.elig | ev)
            idx = np.flatnonzero(ev); keep = np.zeros(ev.size, bool)
            keep[np.random.default_rng(0).choice(idx, min(VAL_CELLS, len(idx)), replace=False)] = True; ev = keep.reshape(ev.shape)
            valid = np.isfinite(q.mspeed)
            cands = {f"imputer_{n or 'all'}": p for n, p in impute(models, q, pn, days, ev, LEVELS).items()}
            cands.update(ffill_day=ffill_day(q.mspeed, valid), interp_time=interp_time(q.mspeed, valid)[0])
            vf = np.broadcast_to(pn.v_f, ev.shape)[ev]; y = pn.speed[ev]; yq = y <= 0.6 * vf
            for k, p in cands.items():
                p = p[ev]; ok = np.isfinite(p); pq_ = p <= 0.6 * vf
                rows_out.append(dict(holdout=name, panel=panel, method=k, n=int(ev.sum()), coverage=float(ok.mean()),
                                     rmse=float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))), queued=int(yq.sum()),
                                     recall=float((pq_ & yq & ok).sum() / max(yq.sum(), 1)),
                                     precision=float((pq_ & yq & ok).sum() / max((pq_ & ok).sum(), 1))))
            print(name, panel, {r["method"]: (round(r["rmse"], 2), round(r["recall"], 3)) for r in rows_out[-len(cands):]}, flush=True)
            del pn, q
    d = pd.DataFrame(rows_out); d.to_csv(io.PROJECT / "reports" / "t1_runs" / f"hybrid_{run}.csv", index=False)
    w = d.assign(se=d.rmse ** 2 * d.n, tp=d.recall * d.queued)
    g = w.groupby(["holdout", "method"]).agg(n=("n", "sum"), se=("se", "sum"), tp=("tp", "sum"), queued=("queued", "sum"))
    g["rmse"] = np.sqrt(g.se / g.n); g["recall"] = g.tp / g.queued
    print(g[["n", "rmse", "queued", "recall"]].to_string())


if __name__ == "__main__":
    build(int(sys.argv[2]), tuple(sys.argv[3:]) or ("_v4t1", "_v4t1_seed7")) if sys.argv[1] == "build" else validate(sys.argv[2])
