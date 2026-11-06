"""Assemble the Kaggle upload file from per-task files.

usage: python -m tfb.submit.assemble <name> --state STATE.csv --queue QUEUE.csv --odme PATH_FLOW.csv [--out-dir DIR] [--note "..."]
Writes <out-dir>/<name>.csv, .sha256 and .manifest.json. A missing file or an unmatched key row raises.
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, time
from pathlib import Path
import polars as pl
from .. import io
from . import checks

KEYS = {"state": ["panel", "timestamp", "station_id", "link_id", "mask_regime"],
        "queue": ["window_id", "timestamp", "link_id"], "odme": ["panel", "departure_time", "path_id"]}
VALS = {"state": ["speed_kmh", "flow_vph"], "queue": ["queue_pred"], "odme": ["path_flow"]}


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=io.PROJECT, text=True).strip()
    except Exception:
        return "nogit"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("--state", required=True); ap.add_argument("--queue")
    ap.add_argument("--odme", required=True); ap.add_argument("--note", default="")
    ap.add_argument("--out-dir", default=str(io.OUTPUTS / "submissions"))
    a = ap.parse_args(argv)
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    files = {"state": Path(a.state), "odme": Path(a.odme)}
    qp = Path(a.queue) if a.queue else io.OUTPUTS / "t2" / "queue_submission.csv"
    if not qp.exists():
        raise SystemExit(f"queue file {qp} missing")
    files["queue"] = qp
    key = pl.read_csv(io.RELEASE / "submission_key.csv", infer_schema_length=0)  # all utf8
    key = key.with_columns(pl.col("submission_id").cast(pl.Int64))
    parts = []
    for task, keys in KEYS.items():
        t = pl.read_csv(files[task], infer_schema_length=0)
        t = t.select(keys + VALS[task]).with_columns([pl.col(v).cast(pl.Float64) for v in VALS[task]])
        if t.select(keys).is_duplicated().any():
            raise SystemExit(f"{task}: duplicate keys in {files[task]}")
        k = key.filter(pl.col("task") == task).select(["submission_id"] + keys)
        j = k.join(t, on=keys, how="left")
        miss = j.filter(pl.col(VALS[task][0]).is_null()).height
        if miss:
            print(j.filter(pl.col(VALS[task][0]).is_null()).head(5))
            raise SystemExit(f"{task}: {miss} of {k.height} key rows unmatched in {files[task]}")
        for c in ["speed_kmh", "flow_vph", "queue_pred", "path_flow"]:
            if c not in j.columns:
                j = j.with_columns(pl.lit(0.0).alias(c))
        parts.append(j.select(["submission_id", pl.lit(task).alias("task"), "speed_kmh", "flow_vph", "queue_pred", "path_flow"]))
    sub = pl.concat(parts).sort("submission_id")
    out = out_dir / f"{a.name}.csv"
    sub.write_csv(out, float_precision=4)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    (out_dir / f"{a.name}.sha256").write_text(f"{sha}  {out.name}\n")
    errs = checks.check(out)
    man = {"name": a.name, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": git_rev(), "sha256": sha,
           "rows": sub.height, "inputs": {k: str(v) for k, v in files.items()},
           "input_sha256": {k: hashlib.sha256(v.read_bytes()).hexdigest()[:16] for k, v in files.items()},
           "checks": errs or "OK", "note": a.note}
    for k, v in files.items():
        m = v.with_suffix(".meta.json")
        if m.exists():
            man[f"{k}_meta"] = json.loads(m.read_text())
    (out_dir / f"{a.name}.manifest.json").write_text(json.dumps(man, indent=2))
    print(json.dumps(man, indent=2))
    if errs:
        raise SystemExit("checks failed")
    return out
