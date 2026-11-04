"""Validate a full Kaggle submission against sample_submission.csv + submission_key.csv.

usage: python -m tfb.submit.checks outputs/submissions/<name>.csv
"""
from __future__ import annotations
import sys
import polars as pl
from .. import io

COLS = ["submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow"]
OWN = {"state": ["speed_kmh", "flow_vph"], "queue": ["queue_pred"], "odme": ["path_flow"]}


def check(path) -> list[str]:
    errs: list[str] = []
    sub = pl.read_csv(path, schema_overrides={"submission_id": pl.Int64, "task": pl.Utf8,
                      "speed_kmh": pl.Float64, "flow_vph": pl.Float64, "queue_pred": pl.Float64, "path_flow": pl.Float64})
    if sub.columns != COLS:
        errs.append(f"columns {sub.columns} != {COLS}")
    sample = pl.read_csv(io.RELEASE / "sample_submission.csv", columns=["submission_id", "task"],
                         schema_overrides={"submission_id": pl.Int64, "task": pl.Utf8})
    if sub.height != sample.height:
        errs.append(f"row count {sub.height} != sample {sample.height}")
    if sub["submission_id"].n_unique() != sub.height:
        errs.append("duplicate submission_id")
    j = sample.join(sub, on="submission_id", how="left", suffix="_s")
    miss = j.filter(pl.col("task_s").is_null()).height
    if miss:
        errs.append(f"{miss} sample ids missing from submission")
    extra = sub.join(sample, on="submission_id", how="anti").height
    if extra:
        errs.append(f"{extra} unknown submission_ids")
    bad_task = j.filter(pl.col("task") != pl.col("task_s")).height
    if bad_task:
        errs.append(f"{bad_task} rows with wrong task label")
    for c in COLS[2:]:
        n = sub.filter(pl.col(c).is_null() | pl.col(c).is_nan() | pl.col(c).is_infinite()).height
        if n:
            errs.append(f"{n} null/NaN/inf in {c}")
    for task, own in OWN.items():
        rows = sub.filter(pl.col("task") == task)
        for c in COLS[2:]:
            if c not in own:
                n = rows.filter(pl.col(c) != 0).height
                if n:
                    errs.append(f"{task}: {n} rows with non-zero unused column {c}")
    q = sub.filter(pl.col("task") == "queue")
    n = q.filter(~pl.col("queue_pred").is_in([0.0, 1.0])).height
    if n:
        errs.append(f"queue: {n} queue_pred not in {{0,1}}")
    n = sub.filter((pl.col("task") == "odme") & (pl.col("path_flow") < 0)).height
    if n:
        errs.append(f"odme: {n} negative path_flow")
    st = sub.filter(pl.col("task") == "state")
    n = st.filter((pl.col("speed_kmh") <= 0) | (pl.col("flow_vph") <= 0)).height
    if n:
        errs.append(f"state: {n} rows with speed or flow <= 0 (suspicious)")
    # per task x split coverage and the 50-vph S_FD trap, via the key
    key = pl.read_csv(io.RELEASE / "submission_key.csv", columns=["submission_id", "task", "split", "panel", "mask_regime"],
                      schema_overrides={"submission_id": pl.Int64, "task": pl.Utf8, "split": pl.Utf8, "panel": pl.Utf8, "mask_regime": pl.Utf8})
    k = key.join(sub.select(["submission_id", "flow_vph", "queue_pred", "path_flow"]), on="submission_id", how="left")
    cov = k.group_by(["task", "split"]).agg(pl.len().alias("n"), pl.col("flow_vph").is_null().sum().alias("missing")).sort(["task", "split"])
    print(cov)
    trap = (k.filter(pl.col("task") == "state").group_by(["split", "panel", "mask_regime"])
            .agg((pl.col("flow_vph") < 50).mean().alias("share_lt50"), pl.len().alias("n")).filter(pl.col("share_lt50") > 0.0))
    if trap.height:
        print("state cells < 50 vph by panel/regime:\n", trap)
    bad = trap.filter(pl.col("share_lt50") > 0.2)
    if bad.height:
        errs.append(f"S_FD trap: {bad.height} panel/regime groups with >20% cells < 50 vph")
    q1 = k.filter(pl.col("task") == "queue").group_by("split").agg(pl.col("queue_pred").mean().alias("pos_rate"))
    print("queue positive rate:", q1)
    return errs


if __name__ == "__main__":
    e = check(sys.argv[1])
    for x in e:
        print("FAIL:", x)
    print("OK" if not e else f"{len(e)} problems")
    sys.exit(1 if e else 0)
