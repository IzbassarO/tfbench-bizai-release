"""Which rows of a submission differ from its base (by task, split, panel, Task 2 condition) -> stored in its manifest.

usage: python -m tfb.submit.diff <name|path.csv> <base name|path.csv>
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import polars as pl
from .. import io

VALS = ["speed_kmh", "flow_vph", "queue_pred", "path_flow"]


def _path(x) -> Path:
    p = Path(x)
    return p if p.suffix == ".csv" else io.OUTPUTS / "submissions" / f"{x}.csv"


def diff(new, base) -> dict:
    a, b = pl.read_csv(_path(new)), pl.read_csv(_path(base))
    j = a.join(b, on="submission_id", suffix="_b")
    ch = j.filter(pl.any_horizontal([pl.col(c) != pl.col(f"{c}_b") for c in VALS]))
    key = pl.read_csv(io.RELEASE / "submission_key.csv", columns=["submission_id", "split", "panel", "window_id"],
                      infer_schema_length=0).with_columns(pl.col("submission_id").cast(pl.Int64))
    cond = pl.concat([pl.read_csv(f, columns=["window_id", "condition"], infer_schema_length=0)
                      for f in sorted((io.RELEASE / "task2").glob("*/*/window_index.csv"))])
    ch = ch.join(key, on="submission_id").join(cond, on="window_id", how="left").with_columns(pl.col("condition").fill_null("-"))
    by = ch.group_by(["task", "split", "condition"]).agg(pl.len().alias("rows"), pl.col("panel").n_unique().alias("panels"),
                                                         pl.col("window_id").n_unique().alias("windows")).sort(["task", "split", "condition"])
    return {"base": _path(base).stem, "changed_rows": ch.height, "by": by.to_dicts()}


def record(new, base) -> dict:
    d = diff(new, base)
    m = _path(new).with_suffix(".manifest.json")
    man = json.loads(m.read_text()); man["diff_vs_base"] = d; m.write_text(json.dumps(man, indent=2))
    print(f"{_path(new).stem} vs {d['base']}: {d['changed_rows']} rows changed")
    for r in d["by"]:
        print(" ", r)
    return d


if __name__ == "__main__":
    record(sys.argv[1], sys.argv[2])
