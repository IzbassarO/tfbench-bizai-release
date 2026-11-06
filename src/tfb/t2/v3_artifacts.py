"""Queue file contracts: expected keys from the release and validation of a queue csv.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl



KEYS = ["window_id", "timestamp", "link_id"]
QUEUE_COLUMNS = [*KEYS, "queue_pred"]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
