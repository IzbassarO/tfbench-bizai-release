"""Reproduce Task 2 probes and verify queue artifacts without editing v1.

Shared full-submission checks use a temporary carrier under outputs/t2/v3.
Nothing here submits to Kaggle or writes another task's output file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import polars as pl

from .run import PROJECT
from ..io import RELEASE as RELEASE_ROOT   # TFB_RELEASE (run.py --data), else data/kaggle_public


KEYS = ["window_id", "timestamp", "link_id"]
QUEUE_COLUMNS = [*KEYS, "queue_pred"]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _path(args, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return path if path.exists() else Path(args.output) / path


def _contracts(args):
    root = Path(args.root)
    parts = []
    for panel in sorted((root / "task2").iterdir()):
        if not panel.is_dir():
            continue
        for split in ("validation", "private"):
            path = panel / split / "sample_submission_queue.csv"
            if path.exists():
                parts.append(pl.read_csv(path, columns=KEYS, schema_overrides={k: pl.String for k in KEYS}))
    if not parts:
        raise ValueError("No validation/private Task 2 templates")
    expected = pl.concat(parts)
    if expected.height == 0 or expected.is_duplicated().any() or expected.null_count().row(0) != (0, 0, 0):
        raise ValueError("Task 2 template keys are duplicated or null")
    schema = {"submission_id": pl.String, "task": pl.String, **{k: pl.String for k in KEYS}}
    key = (pl.scan_csv(root / "submission_key.csv", schema_overrides=schema)
           .select(["submission_id", "task", *KEYS]).filter(pl.col("task") == "queue")
           .collect(engine="streaming"))
    sample = (pl.scan_csv(root / "sample_submission.csv", schema_overrides={"submission_id": pl.String, "task": pl.String})
              .select(["submission_id", "task"]).filter(pl.col("task") == "queue")
              .collect(engine="streaming"))
    if key.select(KEYS).is_duplicated().any() or key["submission_id"].n_unique() != key.height:
        raise ValueError("Common queue key has duplicate cells or submission IDs")
    if sample["submission_id"].n_unique() != sample.height:
        raise ValueError("Common sample has duplicate queue submission IDs")
    if (sample.height != key.height or sample.join(key, on=["submission_id", "task"], how="anti").height
            or key.join(sample, on=["submission_id", "task"], how="anti").height):
        raise ValueError("Common sample queue rows differ from submission_key")
    if (expected.height != key.height or expected.join(key, on=KEYS, how="anti").height
            or key.join(expected, on=KEYS, how="anti").height):
        raise ValueError("Per-task queue templates differ from common key")
    return expected, key


def _validate_queue(path, expected, key):
    queue = pl.read_csv(path, schema_overrides={**{k: pl.String for k in KEYS}, "queue_pred": pl.Float64})
    if queue.columns != QUEUE_COLUMNS:
        raise ValueError(f"Wrong queue schema: {path}")
    if queue.select(KEYS).is_duplicated().any():
        raise ValueError(f"Duplicate queue cells: {path}")
    values = queue["queue_pred"]
    if values.null_count() or not values.is_in([0., 1.]).all():
        raise ValueError(f"queue_pred must contain finite binary values: {path}")
    if (queue.height != expected.height or queue.join(expected, on=KEYS, how="anti").height
            or expected.join(queue, on=KEYS, how="anti").height):
        raise ValueError(f"Per-task queue coverage mismatch: {path}")
    if queue.join(key, on=KEYS, how="anti").height or key.join(queue, on=KEYS, how="anti").height:
        raise ValueError(f"Common queue coverage mismatch: {path}")
    result = dict(rows=queue.height, positive_rows=int(values.sum()), sha256=_sha(path),
                  complete_per_task=True, complete_common_template=True)
    return queue, result


def verify(args, filenames):
    """Verify complete queue row sets; write only v3's verification report."""
    if isinstance(filenames, (str, Path)):
        filenames = [filenames]
    filenames = list(filenames)
    if not filenames:
        raise ValueError("At least one queue file is required")
    expected, key = _contracts(args)
    results = {}
    for filename in filenames:
        path = _path(args, filename)
        if path.name in results:
            raise ValueError("Queue filenames must have distinct basenames")
        _, results[path.name] = _validate_queue(path, expected, key)
    _dump(Path(args.output) / "v3/submission_verification.json", results)
    return results


def build_probe(args):
    """Zero published onset rows, preserving every ongoing cell and row order.

    Existing probe bytes must match the deterministic reconstruction exactly;
    a differing artifact is rejected rather than silently replaced.
    """
    source = _path(args, getattr(args, "source", "queue_submission_v2.csv"))
    destination = _path(args, getattr(args, "probe", "probe_onset_zero_v2.csv"))
    if not destination.resolve().is_relative_to(Path(args.output).resolve()):
        raise ValueError("Probe writes must stay inside Task 2 output")
    expected, key = _contracts(args)
    _, source_result = _validate_queue(source, expected, key)
    frame = pd.read_csv(source, dtype={k: str for k in KEYS})
    condition_parts = []
    for panel in sorted((Path(args.root) / "task2").iterdir()):
        if not panel.is_dir():
            continue
        for split in ("validation", "private"):
            index = panel / split / "window_index.csv"
            if index.exists():
                condition_parts.append(pd.read_csv(index, usecols=["window_id", "condition"], dtype=str))
    if not condition_parts:
        raise ValueError("No published queue condition indices")
    conditions = pd.concat(condition_parts, ignore_index=True)
    if conditions.window_id.duplicated().any() or not conditions.condition.isin(["queue_onset", "queue_ongoing"]).all():
        raise ValueError("Invalid published window conditions")
    mapped = frame.window_id.map(conditions.set_index("window_id").condition)
    if mapped.isna().any() or set(frame.window_id) != set(conditions.window_id):
        raise ValueError("Submission window set differs from published condition indices")
    onset, ongoing = mapped.eq("queue_onset"), mapped.eq("queue_ongoing")
    old_ongoing = frame.loc[ongoing, "queue_pred"].to_numpy().copy()
    zeroed = int(frame.loc[onset, "queue_pred"].sum())
    frame.loc[onset, "queue_pred"] = 0
    if not (frame.loc[ongoing, "queue_pred"].to_numpy() == old_ongoing).all():
        raise AssertionError("Probe changed ongoing cells")
    rebuilt = frame.to_csv(index=False).encode("utf-8")
    if destination.exists() and destination.read_bytes() != rebuilt:
        raise ValueError("Existing probe differs from deterministic reconstruction")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(rebuilt)
    _, probe_result = _validate_queue(destination, expected, key)
    results = {source.name: source_result, destination.name: probe_result}
    _dump(Path(args.output) / "v3/submission_verification.json", results)
    manifest = dict(source=source.name, source_sha256=_sha(source), probe=destination.name,
        sha256=results[destination.name]["sha256"], rows=len(frame),
        onset_windows=int(conditions.condition.eq("queue_onset").sum()),
        ongoing_windows=int(conditions.condition.eq("queue_ongoing").sum()),
        onset_rows=int(onset.sum()), ongoing_rows=int(ongoing.sum()),
        onset_positive_cells_zeroed=zeroed, ongoing_predictions_unchanged=True, submitted=False)
    _dump(Path(args.output) / "v3/probe_manifest.json", manifest)
    return manifest


def make_check(args, queue):
    """Run the shared checker on a temporary carrier containing this queue."""
    queue_path = _path(args, queue)
    expected, key = _contracts(args)
    candidate, result = _validate_queue(queue_path, expected, key)
    _dump(Path(args.output) / "v3/submission_verification.json", {queue_path.name: result})
    mapped = key.join(candidate, on=KEYS, how="left", validate="1:1").select(
        "submission_id", pl.col("queue_pred").alias("candidate_queue"))
    if mapped["candidate_queue"].null_count() or mapped.height != candidate.height:
        raise ValueError("Queue carrier mapping failed")
    carrier_path = Path(getattr(args, "carrier", PROJECT / "outputs/submissions/baseline_v0.csv"))
    carrier = pl.read_csv(carrier_path, schema_overrides={"submission_id": pl.String})
    columns = carrier.columns
    joined = (carrier.join(mapped, on="submission_id", how="left", validate="1:1")
              .with_columns(pl.when(pl.col("task") == "queue").then(pl.col("candidate_queue"))
                            .otherwise(pl.col("queue_pred")).alias("queue_pred")).select(columns))
    directory = Path(args.output) / "v3/qa"
    directory.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^a-zA-Z0-9_.-]", "_", queue_path.stem)
    with tempfile.TemporaryDirectory(prefix=f"{stem}_", dir=directory) as temporary:
        path = Path(temporary) / "combined.csv"
        joined.write_csv(path)
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "TFB_RELEASE": str(Path(args.root).resolve())}
        checked = subprocess.run(["make", "check", f"SUB={path.resolve()}"], cwd=PROJECT,
                                 env=env, text=True, capture_output=True)
        log = checked.stdout + checked.stderr
        (directory / f"make_check_{stem}.log").write_text(log)
        manifest = dict(queue=str(queue_path), queue_sha256=result["sha256"], carrier=str(carrier_path),
                        carrier_sha256=_sha(carrier_path), returncode=checked.returncode,
                        temporary_carrier_removed=True, submitted=False)
        _dump(directory / f"make_check_{stem}.json", manifest)
        if checked.returncode:
            raise RuntimeError(f"Shared make check failed for {queue_path}; see {directory / ('make_check_' + stem + '.log')}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RELEASE_ROOT)
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/t2")
    parser.add_argument("--stage", choices=["probe", "verify", "check", "all"], default="check")
    parser.add_argument("--queue", type=Path, action="append")
    parser.add_argument("--source", type=Path, default=Path("queue_submission_v2.csv"))
    parser.add_argument("--probe", type=Path, default=Path("probe_onset_zero_v2.csv"))
    parser.add_argument("--carrier", type=Path, default=PROJECT / "outputs/submissions/baseline_v0.csv")
    args = parser.parse_args()
    queues = args.queue or [args.probe if args.stage == "all" else Path("queue_submission_v3_robust.csv")]
    if args.stage in {"probe", "all"}:
        print(json.dumps(build_probe(args), indent=2))
    if args.stage in {"verify", "all"}:
        print(json.dumps(verify(args, queues), indent=2))
    if args.stage in {"check", "all"}:
        for queue in queues:
            print(json.dumps(make_check(args, queue), indent=2))


if __name__ == "__main__":
    main()
