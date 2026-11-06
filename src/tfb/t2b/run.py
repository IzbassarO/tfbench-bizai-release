"""Family B physics queue model.

  python -m tfb.t2b.run --stage windows|final

windows: release-like train windows -> outputs/t2b/cache/
final:   fit on all train months, predict released validation+private windows, write
         outputs/t2b/queue_submission_t2b.csv (+ ensemble with family A) and verify coverage.
"""
from __future__ import annotations
import argparse, json, pickle, time
import numpy as np
import pandas as pd
from .. import io
from . import build, codex, cv, evaluate as ev, model as M

OUT = io.OUTPUTS / "t2b"
ALL_MONTHS = ["2030-06", "2030-07", "2030-08", "2030-09", "2030-10", "2030-11", "2030-12", "2031-01", "2031-02"]
CONFIG = dict(variants={"full": (), "noloc": M.LOC}, blend={"full": 0.5, "noloc": 0.5}, seeds=(42, 7, 2031),
              # family A binary weight in the ensemble per condition
              ens_codex_weight={"queue_onset": 0.4, "queue_ongoing": 1.0},
              train_extra={"queue_onset": True, "queue_ongoing": False},
              codex_file=io.OUTPUTS / "t2" / "queue_submission_v2.csv")


def log(s):
    print(f"[t2b {time.strftime('%H:%M:%S')}] {s}", flush=True)


def stage_windows():
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    for p in io.T2_PANELS:
        b = build.train_windows(p)
        b["meta"].to_parquet(OUT / "cache" / f"{p}_train_meta.parquet")
        np.savez_compressed(OUT / "cache" / f"{p}_train.npz", **{k: v for k, v in b.items() if k != "meta"})
        log(f"windows {p}: {len(b['meta'])}")


def blend(P: dict, w: dict) -> np.ndarray:
    return sum(w[k] * P[k] for k in w) / sum(w.values())


def stage_final():
    seeds = CONFIG["seeds"]; Bs = {p: ev.load_train(p) for p in io.T2_PANELS}
    rel = {(p, s): build.released(p, s) for p in io.T2_PANELS for s in ("validation", "private")}
    P = {}
    for cond in cv.CONDS:
        tr = {p: cv.rows(p, Bs[p], cond, ALL_MONTHS, ALL_MONTHS, loo=True, extra=CONFIG["train_extra"][cond]) for p in io.T2_PANELS}
        names = next(r[1] for p in io.T2_PANELS for r in tr[p])
        Xtr = np.concatenate([r[0] for p in io.T2_PANELS for r in tr[p]])
        ytr = np.concatenate([r[3] for p in io.T2_PANELS for r in tr[p]])
        etr = np.concatenate([r[4] for p in io.T2_PANELS for r in tr[p]])
        Xtr, ytr = Xtr[etr], ytr[etr]; del tr
        log(f"final {cond}: {len(ytr):,} rows")
        for vname, drop in CONFIG["variants"].items():
            ms = [M.fit(Xtr, ytr, names, drop=drop, seed=s) for s in seeds]
            for (p, s), R in rel.items():
                mk = (R["meta"].condition == cond).to_numpy()
                pr = M.priors(p, cv._days(None, ALL_MONTHS))
                sub = {k: R[k][mk] for k in ("S", "Q", "RON", "ROFF")}
                X, _, idx = M.panel_rows(sub, Bs[p]["net"], R["meta"][mk], pr, cond)
                prob = np.mean([M.predict(m, X) for m in ms], 0)
                G = P.setdefault((p, s), {}).setdefault(vname, np.zeros((len(mk), 6, len(Bs[p]["net"]["v_f"]))))
                G[mk] = M.to_grid(prob, idx, mk.sum(), G.shape[-1])
    pickle.dump(P, open(OUT / "final_probs.pkl", "wb"))
    write(P, rel)


def write(P, rel):
    csub = codex.submission(CONFIG["codex_file"])
    outs = {"t2b": [], "ens": []}
    for (p, s), R in rel.items():
        Pm = blend(P[(p, s)], CONFIG["blend"])
        wi = pd.read_csv(io.t2_dir(p, s) / "window_index.csv", dtype=str)
        G = codex.grid(csub, p, s, wi)
        w = np.where((R["meta"].condition == "queue_onset").to_numpy(), CONFIG["ens_codex_weight"]["queue_onset"],
                     CONFIG["ens_codex_weight"]["queue_ongoing"])[:, None, None]
        for name, prob in (("t2b", Pm), ("ens", (1 - w) * Pm + w * G)):
            D = M.decode(prob)
            D[(R["meta"].condition == "queue_onset").to_numpy(), :5] = False
            outs[name].append(to_rows(D, p, s, wi))
    for name, parts in outs.items():
        df = pd.concat(parts, ignore_index=True)
        f = OUT / f"queue_submission_{name}.csv"
        df.to_csv(f, index=False)
        log(f"wrote {f.name}: {len(df):,} rows, positives {int(df.queue_pred.sum()):,}")
    json.dump(verify(), open(OUT / "submission_verification.json", "w"), indent=1)


def to_rows(D, panel, split, wi):
    from . import data
    tpl = pd.read_csv(io.t2_dir(panel, split) / "sample_submission_queue.csv", dtype=str)
    d = data.load(panel, split); pos = {l: i for i, l in enumerate(d["links"])}
    wpos = {w: i for i, w in enumerate(wi.window_id)}
    start = pd.to_datetime(wi.forecast_origin, utc=True).to_numpy()
    r = tpl.window_id.map(wpos).to_numpy()
    k = ((pd.to_datetime(tpl.timestamp, utc=True).to_numpy() - start[r]) / np.timedelta64(5, "m")).astype(int) - 1
    assert k.min() == 0 and k.max() == 5
    tpl["queue_pred"] = D[r, k, tpl.link_id.map(pos).to_numpy()].astype(int)
    return tpl


def verify(files=None) -> dict:
    """Each output == union of validation+private templates on keys; binary; no duplicates; matches submission_key."""
    import polars as pl
    keys = pd.concat([pd.read_csv(io.t2_dir(p, s) / "sample_submission_queue.csv", dtype=str)[["window_id", "timestamp", "link_id"]]
                      for p in io.T2_PANELS for s in ("validation", "private")], ignore_index=True)
    sk = pl.read_csv(io.RELEASE / "submission_key.csv", columns=["task", "window_id", "timestamp", "link_id"],
                     schema_overrides={c: pl.Utf8 for c in ("task", "window_id", "timestamp", "link_id")}).filter(pl.col("task") == "queue")
    kset = set(zip(keys.window_id, keys.timestamp, keys.link_id))
    skset = set(zip(sk["window_id"].to_list(), sk["timestamp"].to_list(), sk["link_id"].to_list()))
    res = {"template_rows": len(keys), "submission_key_queue_rows": sk.height, "key_sets_equal": kset == skset}
    for f in files or ("queue_submission_t2b.csv", "queue_submission_ens.csv"):
        df = pd.read_csv(OUT / f, dtype={"window_id": str, "timestamp": str, "link_id": str})
        s = set(zip(df.window_id, df.timestamp, df.link_id))
        res[f] = {"rows": len(df), "unique": len(s) == len(df), "equals_templates": s == kset,
                  "binary": bool(df.queue_pred.isin([0, 1]).all()), "positives": int(df.queue_pred.sum())}
    log(json.dumps(res))
    assert all(v["rows"] == 174000 and v["unique"] and v["equals_templates"] and v["binary"]
               for k, v in res.items() if k.endswith(".csv")) and res["key_sets_equal"]
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["windows", "final"], required=True)
    {"windows": stage_windows, "final": stage_final}[ap.parse_args().stage]()
