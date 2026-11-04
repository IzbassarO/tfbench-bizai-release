"""Portable ongoing-queue model for the CV/LB-gap investigation.

All candidate blending policies are global, causal and fixed in this module.
The exact released-window gate is evaluated separately before deployment.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .io import dump_json, panel_names
from .run import PROJECT, log
from .train import Classifier, aggregate_scores, fit_classifier, score_window
from .v2_decode import decode_calibrated
from .v2_experiment import FOLDS, load_model
from .v2_predict import member_parts
from .v3_incidents import window_shift_metrics


CONDITION = "queue_ongoing"
POLICIES = ("v2", "portable25", "portable50", "portable100", "portable_rare25", "portable_rare50", "portable_rare100",
            "no_location_rare25", "no_location_rare50", "no_location_rare100",
            "portable_place25", "portable_place50", "no_location_place50")
STATIC = {"free_speed_kmh", "capacity_total_vph", "critical_density_total", "lanes",
          "spacing_up_km", "spacing_down_km"}
CALENDAR = {"origin_time_sin", "origin_time_cos", "origin_day_of_week", "origin_is_weekend",
            "forecast_time_sin", "forecast_time_cos", "forecast_day_of_week"}


def portable_mask(names):
    """Keep relative queue/traffic physics; discard absolute identity/calendar."""
    return np.array([not (n.startswith("location_") or n in STATIC | CALENDAR) for n in names])


def blend_fraction(policy, metrics):
    if policy == "v2": return 0.
    if "_place" in policy:
        if metrics.get("spatial_stratum")=="unknown" or metrics.get("recurrence_stratum")=="unknown":return 0.
        return float(policy.rsplit("place",1)[1])/100 if metrics.get("rare_place_fraction",0) >= .5 else 0.
    if policy.startswith("portable_rare"):
        return float(policy.removeprefix("portable_rare")) / 100 if metrics["recurrence_stratum"] == "nonrecurrent" else 0.
    if policy.startswith("no_location_rare"):
        return float(policy.removeprefix("no_location_rare")) / 100 if metrics["recurrence_stratum"] == "nonrecurrent" else 0.
    if policy.startswith("portable"):
        return float(policy.removeprefix("portable")) / 100
    raise ValueError(f"Unknown fixed policy: {policy}")


def fold_source(args, fold):
    return args.output / "v3/exact_data" if fold == "exact" else args.output / "datasets" / (FOLDS[fold] + "_x")


def fit_portable(args, fold):
    path = args.output / "v3/models" / fold / "portable_ongoing.txt"
    if path.exists(): return load_model(path, args.threads)
    xs, ys, ws = [], [], []
    source = fold_source(args, fold)
    for panel in args.panels:
        with np.load(source / f"{panel}_{CONDITION}.npz") as d:
            x = d["train_x_extended" if fold=="exact" else "train_x"]
            names = d["feature_names_extended" if fold=="exact" else "feature_names"].tolist()
            y, w = d["train_y"], d["train_weight"]
            now = x[:, names.index("queue_now")] > .5
            rare = np.nan_to_num(x[:, names.index("location_queue_prior_target")], nan=0) < .05
            tail = np.abs(x[:, names.index("queue_tail_distance_links")]) <= 3
            head = np.abs(x[:, names.index("queue_head_distance_links")]) <= 3
            # Target-dependent weights are confined to the training population.
            w = w * (1 + 2 * (now & ~y) + 1.5 * (y & rare) + .5 * (tail | head))
            xs.append(x); ys.append(y); ws.append(w)
            if "augment_x" in d and len(d["augment_y"]):
                xs.append(d["augment_x"]); ys.append(d["augment_y"])
                ws.append(d["augment_weight"] * .35)
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    del xs, ys, ws
    gc.collect()
    mask = portable_mask(names)
    log(f"v3 portable fit {fold}: {len(y):,} rows, {mask.sum()} of {len(mask)} features")
    reduced = fit_classifier(x[:, mask], y, w, list(np.asarray(names)[mask]),
                             seed=42, rounds=args.rounds, threads=args.threads)
    model = Classifier(reduced.booster, mask, reduced.constant, args.threads)
    path.parent.mkdir(parents=True, exist_ok=True); model.save(path)
    dump_json(path.with_suffix(".training.json"), dict(fold=fold, seed=42,
        rounds=args.rounds, threads=args.threads, rows=len(y), features=list(np.asarray(names)[mask]),
        dropped=list(np.asarray(names)[~mask]), augmentation_weight=.35,
        x_sha256=hashlib.sha256(memoryview(np.ascontiguousarray(x)).cast("B")).hexdigest(),
        y_sha256=hashlib.sha256(memoryview(np.ascontiguousarray(y)).cast("B")).hexdigest(),
        weights_sha256=hashlib.sha256(memoryview(np.ascontiguousarray(w)).cast("B")).hexdigest()))
    del x, y, w
    gc.collect()
    return model


def predict_portable(args, fold):
    model = fit_portable(args, fold)
    source = fold_source(args, fold)
    dest = args.output / "v3/portable" / fold; dest.mkdir(parents=True, exist_ok=True)
    for panel in args.panels:
        path = dest / f"{panel}_{CONDITION}.npz"
        if path.exists():
            with np.load(path) as old:
                if old["probability"].dtype==np.float64:continue
        with np.load(source / f"{panel}_{CONDITION}.npz") as d:
            x, y = d["eval_x_extended" if fold=="exact" else "eval_x"], d["eval_y"]
            if not len(x): continue
            p = model.predict(x.reshape(-1, x.shape[-1])).reshape(y.shape)
            meta = pd.read_csv(source / f"{panel}_{CONDITION}_eval.csv")
            np.savez(path, probability=p, origin=meta.origin.to_numpy(dtype=str),
                     window_id=meta.window_id.to_numpy(dtype=str))
        log(f"v3 portable predictions {fold}/{panel}")


def v2_config(args, fold, panel, condition):
    selection = json.loads((args.output / "v2/selection.json").read_text())
    member = selection.get(panel, {}).get(condition, "baseline")
    if member == "baseline": return {}
    _, _, calibrated, _, label = member_parts(member)
    key_fold = "janfeb" if fold == "exact" else fold
    return json.loads((args.output / "v2/calibration.json").read_text())[f"{key_fold}/{panel}/{condition}/{label}"] if calibrated else {}


def score_temporal(args, fold):
    """Compare fixed global ongoing policies; retain the frozen V2 onset."""
    source = fold_source(args, fold)
    rows = []
    for panel in args.panels:
        for condition in ("queue_onset", CONDITION):
            with np.load(args.output / "v3/eval" / fold / f"{panel}_{condition}.npz") as d:
                y, e, base = d["truth"], d["eligible"], d["probability_v2"]
                original = d["prediction_v2"]
                no_location = d["probability_no_location"] if condition == CONDITION else None
            meta = pd.read_csv(args.output / "v3/eval" / fold / f"{panel}_{condition}.csv")
            if condition == CONDITION:
                with np.load(args.output / "v3/portable" / fold / f"{panel}_{condition}.npz") as d:
                    assert np.array_equal(d["origin"], meta.origin.to_numpy(dtype=str))
                    assert np.array_equal(d["window_id"], meta.window_id.to_numpy(dtype=str))
                    portable = d["probability"]
                with np.load(source / f"{panel}_{condition}.npz") as d:
                    x, names = d["eval_x"], d["feature_names"].tolist()
                # Each gate's routing prior excludes its evaluation dates.
                # The June-Dec reference is shared only by Jan-Feb and March.
                prior = np.load(args.output / "datasets" / FOLDS[fold] / f"{panel}_prior.npy")
                config = v2_config(args, fold, panel, condition)
            for i, record in enumerate(meta.to_dict("records")):
                metrics = window_shift_metrics(x[i], names, prior, origin=record["origin"]) if condition == CONDITION else {"recurrence_stratum":"no_recent_queue"}
                for policy in POLICIES:
                    fraction = blend_fraction(policy, metrics) if condition == CONDITION else 0.
                    alternative = no_location if policy.startswith("no_location_") else portable if condition == CONDITION else None
                    pred = decode_calibrated((1-fraction)*base[i]+fraction*alternative[i], config) if fraction else original[i]
                    rows.append({**record, **metrics, "fold":fold, "model":policy,
                        "iou":score_window(pred,y[i],e[i]), "blend_fraction":fraction,
                        "eligible_true_cells":int((y[i]&e[i]).sum()),
                        "predicted_cells":int(pred.sum()),
                        "fp":int((pred.astype(bool)&~y[i]&e[i]).sum()),
                        "fn":int((~pred.astype(bool)&y[i]&e[i]).sum())})
    detail = pd.DataFrame(rows)
    dest = args.output / "v3/portable" / fold
    detail.to_csv(dest / "window_scores.csv", index=False)
    primary = detail[detail.official_like.astype(str).str.lower().eq("true")]
    summary = aggregate_scores(primary)
    summary.to_csv(dest / "scores.csv", index=False)
    log(f"v3 portable scores {fold}\n"+summary[summary.panel.eq("OVERALL")].to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=PROJECT/"data/kaggle_public")
    p.add_argument("--output", type=Path, default=PROJECT/"outputs/t2")
    p.add_argument("--fold", choices=["janfeb","octnov","exact","final"], default="janfeb")
    p.add_argument("--stage", choices=["fit","predict","score","all"], default="all")
    p.add_argument("--threads",type=int,default=2);p.add_argument("--rounds",type=int,default=350)
    args=p.parse_args();args.panels=panel_names(args.root)
    if args.stage=="fit":fit_portable(args,args.fold)
    if args.stage in {"predict","all"}:predict_portable(args,args.fold)
    if args.stage in {"score","all"} and args.fold in {"janfeb","octnov"}:score_temporal(args,args.fold)


if __name__ == "__main__": main()
