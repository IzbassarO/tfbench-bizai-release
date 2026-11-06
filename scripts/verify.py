"""Compare a reproduced submission with the submitted final, row by row, and bound the private-score difference.

  python scripts/verify.py OURS.csv REF.csv [--data DIR]
    REF = the byte-exact rebuild of the submitted final (python run.py --final v3 --from-components ->
          outputs/submission_v3.csv, whose sha256 run.py checks against the submitted file).

The private truth is hidden, but the official scores are means of per-group terms whose change has a bound
computable from the two predictions alone:
  state    per panel x regime, by Minkowski |RMSE(a, y) - RMSE(b, y)| <= RMSE(a, b), so
           |dS_state| <= 0.54 RMSE(dv) / 25 + 0.46 RMSE(dq / lanes) / 600, then regime -> panel -> family -> mean (official)
  queue    a window whose queue_pred is unchanged keeps its IoU; a changed window can move by at most 1 (strict bound);
           also printed: flips / cells queued in both files per window (exact bound when every horizon cell is
           eligible; eligibility of the hidden horizon is not released)
           windows -> panel x condition -> panel -> family -> mean over the 4 Task 2 families (official)
  odme     relative L1 change of path_flow per panel (S_ODME is a ratio of L1 link-count errors: reported, not bounded)
  physics  not bounded (needs the hidden fields); reported through the state differences it is computed from
dS_total <= 0.35 dS_state + 0.30 dS_queue (+ the reported odme / physics changes). Exit 1 if the bound exceeds
1 % of REF's public score (0.88369 v3, 0.88317 v1: 0.0088) or the files differ in rows / keys / format.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
FAMILY = lambda p: p.rsplit("_", 1)[0]   # D7_I10_E -> D7_I10 (5 families; Task 2: D12_I405 has no windows)
COLS = ["submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow"]


def read(p):
    return pl.read_csv(p, schema_overrides={"submission_id": pl.Int64, "task": pl.Utf8, **{c: pl.Float64 for c in COLS[2:]}})


def official_mean(df: pl.DataFrame, val: str, levels: list[str]) -> float:
    """Equal-weight mean up the official hierarchy: mean over levels[0] groups within levels[1:] ... -> one number."""
    for i in range(len(levels)):
        df = df.group_by(levels[i + 1:] or [pl.lit(1).alias("_")]).agg(pl.col(val).mean())
    return float(df[val][0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ours", type=Path); ap.add_argument("ref", type=Path)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "kaggle_public"); ap.add_argument("--public", type=float, default=0.88369)
    a = ap.parse_args()
    A, B = read(a.ours), read(a.ref)
    if A.columns != COLS or B.columns != COLS or A.height != B.height or not (A["submission_id"] == B["submission_id"]).all() \
            or not (A["task"] == B["task"]).all():
        print("FAIL: columns / rows / submission_id / task differ"); return 1
    key = pl.read_csv(a.data / "submission_key.csv", infer_schema_length=0).with_columns(pl.col("submission_id").cast(pl.Int64))
    d = (A.join(B, on="submission_id", suffix="_ref").join(key.drop("task"), on="submission_id"))
    out = {}

    # ---------------------------------------------------------------- state
    lanes = pl.concat([pl.read_csv(f, columns=["link_id", "lanes"], schema_overrides={"link_id": pl.Utf8, "lanes": pl.Float64}).with_columns(pl.lit(f.parts[-3]).alias("panel"))
                       for f in sorted((a.data / "corridors").glob("*/network/fd_parameters.csv"))])
    lanes = lanes.with_columns(pl.when(pl.col("lanes").is_between(1, 16)).then(pl.col("lanes")).otherwise(1.0).alias("lanes"))
    s = d.filter(pl.col("task") == "state").join(lanes.unique(["panel", "link_id"]), on=["panel", "link_id"], how="left") \
         .with_columns(pl.col("lanes").fill_null(1.0), dv=pl.col("speed_kmh") - pl.col("speed_kmh_ref"), dq=pl.col("flow_vph") - pl.col("flow_vph_ref"))
    g = s.group_by(["split", "panel", "mask_regime"]).agg(rv=(pl.col("dv") ** 2).mean().sqrt(), rq=((pl.col("dq") / pl.col("lanes")) ** 2).mean().sqrt(),
                                                          cells=pl.len(), changed=((pl.col("dv").abs() > 0) | (pl.col("dq").abs() > 0)).sum())
    g = g.with_columns(bound=0.54 * pl.col("rv") / 25 + 0.46 * pl.col("rq") / 600, family=pl.col("panel").map_elements(FAMILY, return_dtype=pl.Utf8))
    for split in ("validation", "private"):
        out[f"state_{split}"] = official_mean(g.filter(pl.col("split") == split), "bound", ["mask_regime", "panel", "family"])
    print(f"state: {int(g['changed'].sum())} of {int(g['cells'].sum())} cells differ; max |dv| {s['dv'].abs().max():.4f} km/h, "
          f"max |dq| {s['dq'].abs().max():.3f} vph; worst panel x regime RMSE dv {g['rv'].max():.5f}, dq/lane {g['rq'].max():.5f}")

    # ---------------------------------------------------------------- queue
    q = d.filter(pl.col("task") == "queue").with_columns(flip=(pl.col("queue_pred") != pl.col("queue_pred_ref")))
    wi = pl.concat([pl.read_csv(f, columns=["window_id", "condition"]) for f in sorted((a.data / "task2").glob("*/*/window_index.csv"))])
    w = q.group_by(["split", "panel", "window_id"]).agg(flips=pl.col("flip").sum(), both=((pl.col("queue_pred") == 1) & (pl.col("queue_pred_ref") == 1)).sum()) \
         .join(wi, on="window_id").with_columns(bound=(pl.col("flips") > 0).cast(pl.Float64),
                                                 typical=pl.min_horizontal(1.0, pl.col("flips") / pl.max_horizontal(1, pl.col("both"))),
                                                 family=pl.col("panel").map_elements(FAMILY, return_dtype=pl.Utf8))
    for split in ("validation", "private"):
        for k in ("bound", "typical"):
            out[f"queue_{k}_{split}"] = official_mean(w.filter(pl.col("split") == split), k, ["window_id", "condition", "panel", "family"])
        out[f"queue_{split}"] = out[f"queue_bound_{split}"]
    print(f"queue: {int(q['flip'].sum())} of {q.height} cells flipped in {int((w['flips'] > 0).sum())} of {w.height} windows")

    # ---------------------------------------------------------------- odme
    o = d.filter(pl.col("task") == "odme").group_by(["split", "panel"]).agg(
        rel_l1=(pl.col("path_flow") - pl.col("path_flow_ref")).abs().sum() / (pl.col("path_flow_ref").abs().sum() + 1e-9))
    print(f"odme: max relative L1 change of path_flow per panel {o['rel_l1'].max():.2e}")

    lim = 0.01 * a.public
    for split in ("validation", "private"):
        tot = 0.35 * out[f"state_{split}"] + 0.30 * out[f"queue_{split}"]
        print(f"{split:10s} |dS_state| <= {out[f'state_{split}']:.6f}  |dS_queue| <= {out[f'queue_{split}']:.6f}  "
              f"|dS_total| <= {tot:.6f} (+ odme / physics as reported)  limit 1 % = {lim:.4f} -> {'OK' if tot <= lim else 'OVER'}")
        print(f"{'':10s} queue, typical (exact if every horizon cell is eligible: per window flips / cells queued in both): "
              f"|dS_queue| <= {out[f'queue_typical_{split}']:.6f}, |dS_total| <= {0.35 * out[f'state_{split}'] + 0.30 * out[f'queue_typical_{split}']:.6f}")
    worst = max(0.35 * out[f"state_{s}"] + 0.30 * out[f"queue_{s}"] for s in ("validation", "private"))
    return 0 if worst <= lim and o["rel_l1"].max() < 1e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
