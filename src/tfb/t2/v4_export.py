"""Export V2 deployment marginals, with exact binary replay checks.

``queue_prob`` is a float64 model/blend probability BEFORE calibration,
cardinality selection, or spatial decoding. The V2 decoder configuration is
recorded separately. Onset T+5 through T+25 probabilities are identically zero.
This module loads existing models only; it never refits or overwrites V2.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .context import build_day_context
from .features import build_features
from .io import (dump_json, history_arrays, panel_names, read_network,
                 read_panel_arrays, read_window_assets, resolve_root)
from .run import PROJECT, enrich_features, log
from .v2_data import extra_features
from .v2_decode import decode_calibrated
from .v2_experiment import SEEDS, load_model
from .v2_predict import member_parts
from .v3_artifacts import KEYS, _contracts, _validate_queue
from ..io import RELEASE as RELEASE_ROOT   # TFB_RELEASE (run.py --data), else data/kaggle_public


@dataclass
class WindowProbability:
    panel: str
    split: str
    window_id: str
    origin: object
    condition: str
    link_ids: np.ndarray
    probability: np.ndarray
    config: dict
    member: str


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _remember(provenance, path):
    path = Path(path)
    provenance[str(path.resolve())] = _sha(path)


def _frozen_hashes(output):
    names = ("queue_submission.csv", "queue_submission_v1.csv",
             "queue_submission_v2.csv", "submission_verification.json")
    return {str(output / name): _sha(output / name) for name in names if (output / name).exists()}


def iter_window_probabilities(args, provenance):
    """Read legal window inputs and reconstruct the selected models.

    The dates-filtered panel reader does not write a raw-data cache. It supplies
    only the masked row at T, ramps through T, and masked same-day context
    clipped at T. The published window history supplies T-60 through T-5.
    Priors and model parameters are the existing full-train V2 artifacts.
    """
    selected_path = args.output / "v2/selection.json"
    calibration_path = args.output / "v2/calibration.json"
    selected = json.loads(selected_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    for path in (selected_path, calibration_path):
        _remember(provenance, path)
    models = {}

    def probability_for(condition, variant, x, extended, names, extra_names, n):
        onset = condition == "queue_onset"
        uses_extra = variant in {"features", "incident", "augmented", "robust"}
        key = condition, variant
        if key not in models:
            members = [str(seed) for seed in SEEDS] if onset and variant != "baseline" else (
                ["median"] if onset or variant == "median_v1" else ["full", "no_location"])
            loaded = []
            for member in members:
                old = args.output / "models/final_enhanced_release" / f"{condition}_{member}.txt"
                path = old if variant in {"baseline", "median_v1"} and old.exists() else (
                    args.output / "v2/models/final" / variant / f"{condition}_{member}.txt")
                model = load_model(path, args.threads)
                _remember(provenance, path)
                _remember(provenance, path.with_suffix(".mask.npy"))
                loaded.append(model)
            models[key] = loaded
        mx = extended if uses_extra else x
        model_names = names + extra_names if uses_extra else names
        if onset:
            mx = mx[-n:]
        for model in models[key]:
            if len(model.mask) != len(model_names) or model.booster.feature_name() != [
                    name for name, take in zip(model_names, model.mask) if take]:
                raise ValueError(f"Model feature schema differs: {condition}/{variant}")
        result = sum(model.predict(mx) for model in models[key]) / len(models[key])
        probability = np.zeros((6, n), np.float64)
        if onset:
            probability[-1] = result
        else:
            probability[:] = result.reshape(6, n)
        return probability

    for panel_id, panel in enumerate(args.panels):
        network = read_network(args.root, panel)
        for path in sorted((args.root / "corridors" / panel / "network").glob("*.csv")):
            _remember(provenance, path)
        n = len(network.link_ids)
        prior_path = args.output / "datasets/final_enhanced_release" / f"{panel}_prior.npy"
        onset_path = args.output / "datasets/final_enhanced_release" / f"{panel}_onset_prior.npy"
        prior, onset_prior = np.load(prior_path), np.load(onset_path)
        for path in (prior_path, onset_path):
            _remember(provenance, path)
        for split in ("validation", "private"):
            index, history, _ = read_window_assets(args.root, panel, split)
            for name in ("window_index.csv", "window_history.parquet"):
                _remember(provenance, args.root / "task2" / panel / split / name)
            days = set(index.date.astype(str))
            days |= {str(np.datetime64(day) - np.timedelta64(1, "D")) for day in days}
            for channel in ("mainline_states_masked", "ramp_states"):
                for path in sorted((args.root / "corridors" / panel / split / channel).rglob("*.parquet")):
                    if path.stem[-10:].replace("_", "-") in days:
                        _remember(provenance, path)
            current = read_panel_arrays(args.root, network, split, args.output / "cache", masked=True, dates=days)
            for window in index.itertuples():
                member = selected.get(panel, {}).get(window.condition, "baseline")
                if member == "baseline":
                    variant, fraction, calibrated, use_extent, label = "baseline", 1., False, False, "baseline"
                else:
                    variant, fraction, calibrated, use_extent, label = member_parts(member)
                if use_extent:
                    raise ValueError("Extent-model selection needs a separate count export; V2 currently uses none")
                arrays = history_arrays(history[history.window_id.eq(window.window_id)], window.forecast_origin, network, current)
                x, names = build_features(arrays.speed, arrays.flow, arrays.occupancy, origin=12,
                    timestamps=arrays.timestamps, ramp_on=arrays.ramp_on, location_prior=prior,
                    panel_id=panel_id, **network.feature_kwargs())
                origin_idx = int((arrays.timestamps[-1] - current.timestamps[0]) / np.timedelta64(5, "m"))
                day_context, day_names = build_day_context(current.speed, current.flow, current.timestamps,
                    origin=origin_idx, vf=network.vf)
                x, names = enrich_features(x, names, arrays, network, onset_prior=onset_prior,
                    day_context=day_context, day_names=day_names)
                extra, extra_names = extra_features(arrays, network)
                extended = np.column_stack([x, extra])
                probability = probability_for(window.condition, variant, x, extended, names, extra_names, n)
                if fraction < 1:
                    probability = fraction * probability + (1 - fraction) * probability_for(
                        window.condition, "baseline", x, extended, names, extra_names, n)
                config = calibration[f"janfeb/{panel}/{window.condition}/{label}"] if calibrated else {}
                yield WindowProbability(panel, split, str(window.window_id), window.forecast_origin,
                    window.condition, network.link_ids, probability, config, member)
            del current
            gc.collect()
            log(f"v4 exported V2 probabilities {panel}/{split}")


def export_probabilities(args):
    """Export all template cells only after exact V2 replay and coverage pass."""
    args.root, args.output = Path(args.root), Path(args.output)
    source = args.output / "queue_submission_v2.csv"
    immutable = _frozen_hashes(args.output)
    expected, key = _contracts(args)
    _validate_queue(source, expected, key)
    frame = pd.read_csv(source, dtype={name: str for name in KEYS})
    probabilities = np.full(len(frame), np.nan, np.float64)
    seen = np.zeros(len(frame), bool)
    provenance, summaries, configs = {}, [], {}
    window_rows = frame.groupby("window_id", sort=False).indices
    for item in iter_window_probabilities(args, provenance):
        p = np.asarray(item.probability, np.float64)
        if item.condition not in {"queue_onset", "queue_ongoing"}:
            raise ValueError("Unknown window condition")
        onset = item.condition == "queue_onset"
        if p.shape != (6, len(item.link_ids)) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Window probabilities must be finite [6,link] marginals in [0,1]")
        if onset and np.count_nonzero(p[:-1]):
            raise ValueError("Onset pre-T+30 marginals must be zero")
        if item.window_id not in window_rows:
            raise ValueError("Model produced a window outside the template")
        rows = window_rows[item.window_id]
        if seen[rows].any():
            raise ValueError("Window probabilities exported twice")
        out = frame.iloc[rows]
        origin = pd.Timestamp(item.origin)
        origin = origin.tz_localize("UTC") if origin.tzinfo is None else origin.tz_convert("UTC")
        horizon = (pd.to_datetime(out.timestamp, utc=True) - origin).dt.total_seconds().to_numpy() / 300
        if not np.isfinite(horizon).all() or not np.equal(horizon, np.floor(horizon)).all() or not ((horizon >= 1) & (horizon <= 6)).all():
            raise ValueError("Template has unexpected forecast horizon")
        hi = horizon.astype(int) - 1
        links = pd.Index(item.link_ids.astype(str))
        if not links.is_unique:
            raise ValueError("Network has duplicate link IDs")
        li = links.get_indexer(out.link_id)
        if (li < 0).any() or len(rows) != p.size or len(set(zip(hi, li))) != p.size:
            raise ValueError("Window does not cover every horizon/link cell exactly once")
        prediction = decode_calibrated(p, item.config, onset=onset)[hi, li]
        mismatch = int(np.count_nonzero(prediction != out.queue_pred.to_numpy()))
        if mismatch:
            raise ValueError(f"V2 replay differs at {item.window_id}: {mismatch} cells")
        probabilities[rows] = p[hi, li]
        seen[rows] = True
        configs[item.window_id] = {k: v for k, v in item.config.items() if k != "fit"}
        summaries.append(dict(panel=item.panel, split=item.split, condition=item.condition,
            window_id=item.window_id, origin=str(origin), member=item.member, rows=len(rows),
            probability_sum=float(p.sum()), probability_min=float(p.min()), probability_max=float(p.max()),
            predicted_positive=int(prediction.sum()), replay_mismatches=mismatch))
    if not seen.all():
        raise ValueError(f"Missing probabilities for {int((~seen).sum())} V2 cells")
    if immutable != _frozen_hashes(args.output):
        raise RuntimeError("V1/V2 source artifacts changed during export")
    destination = args.output / "probs_v2.parquet"
    result = frame[KEYS].copy()
    result["queue_prob"] = probabilities
    temporary = destination.with_suffix(".tmp.parquet")
    result.to_parquet(temporary, index=False)
    temporary.replace(destination)
    target = args.output / "v4"
    target.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(target / "v2_probability_summary.csv", index=False)
    manifest = dict(rows=len(result), windows=len(summaries), source=str(source), source_sha256=_sha(source),
        output=str(destination), sha256=_sha(destination), dtype="float64",
        columns=[*KEYS, "queue_prob"], probability_stage="raw selected model/blend, before calibration and decoding",
        onset_t5_t25_zero=True, all_binary_predictions_replayed=True, replay_mismatches=0,
        frozen_artifact_hashes=immutable, input_sha256=provenance, decoder_configs=configs,
        causal_inputs="published T-60..T-5 history, masked T, ramps <=T, masked same-day context <=T; train-fitted priors")
    dump_json(target / "v2_probability_export_manifest.json", manifest)
    log(f"v4 V2 probability export: {len(result):,} cells, all binary rows exactly replayed")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RELEASE_ROOT)
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/t2")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    args.root = resolve_root(args.root)
    args.panels = panel_names(args.root)
    manifest = export_probabilities(args)
    print(json.dumps({k: manifest[k] for k in ("rows", "windows", "output", "sha256", "replay_mismatches")}, indent=2))


if __name__ == "__main__":
    main()
