"""Deploy the frozen V4 onset ensemble while preserving every V2 ongoing row.

Only existing final models/capacities are read. Raw float64 probabilities are
decoded over all links at T+30; T+5..T+25 remain zero. Shared QA writes its own
temporary full carrier and evidence under outputs/t2/v4, never v1/v2/v3 proof.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd
import polars as pl

from .context import _current_day_times, build_day_context
from .features import build_features
from .io import (dump_json, history_arrays, panel_names, read_network,
                 read_panel_arrays, read_window_assets, resolve_root)
from .run import PROJECT, enrich_features, log
from .train import decode_probabilities
from .v2_data import extra_features
from .v2_experiment import load_model
from .v3_artifacts import KEYS, _contracts, _validate_queue
from .v4_export import WindowProbability, _remember, _sha
from .v4_features import build_onset_features
from ..io import RELEASE as RELEASE_ROOT   # TFB_RELEASE (run.py --data), else data/kaggle_public


SEEDS = (42, 137, 271)


def build_window_features(local, current, network, *, location_prior, onset_prior,
                          capacity, panel_id, origin_idx):
    """Match training's protected legal-history overlay in both day builders."""
    n = len(network.link_ids)
    if len(local.timestamps) != 13:
        raise ValueError("V4 deployment requires the thirteen legal history rows")
    if not 0 <= origin_idx < len(current.timestamps) or current.timestamps[origin_idx] != local.timestamps[-1]:
        raise ValueError("V4 day-array origin must match the last legal history row")
    x, names = build_features(local.speed, local.flow, local.occupancy, origin=12,
        timestamps=local.timestamps, ramp_on=local.ramp_on, location_prior=location_prior,
        panel_id=panel_id, **network.feature_kwargs())
    # Generated training windows protect their twelve raw history slots in
    # masked same-day context. Explicitly reproduce this at deployment, even
    # if an organizer mask were to differ from its published history layer.
    start, times = _current_day_times(np.asarray(current.timestamps), int(origin_idx))
    speed = current.speed[start:origin_idx + 1].copy()
    flow = current.flow[start:origin_idx + 1].copy()
    history_times = np.asarray(local.timestamps[:-1]).astype("datetime64[m]")
    overlap = (history_times >= times[0]) & (history_times < times[-1])
    positions = ((history_times[overlap] - times[0]) / np.timedelta64(5, "m")).astype(int)
    speed[positions] = local.speed[:-1][overlap]
    flow[positions] = local.flow[:-1][overlap]
    context, context_names = build_day_context(speed, flow, times, origin=len(times)-1, vf=network.vf)
    x, names = enrich_features(x, names, local, network, onset_prior=onset_prior,
        day_context=context, day_names=context_names)
    extra, extra_names = extra_features(local, network)
    added, added_names = build_onset_features(local, network, capacity,
        dayarrays=current, origin_idx=origin_idx)
    return np.column_stack([x[-n:], extra[-n:], added]), names + extra_names + added_names


def _immutable_hashes(output):
    paths = [output / name for name in ("queue_submission.csv", "queue_submission_v1.csv",
        "queue_submission_v2.csv", "submission_verification.json", "probs_v2.parquet")]
    paths += [output / "v3/submission_verification.json", output / "v4/selection.json"]
    return {str(path): _sha(path) for path in paths if path.exists()}


def _selection(args):
    """Optional fixed family selection; no inference-time data routing."""
    path = args.output / "v4/selection.json"
    config = json.loads(path.read_text()) if path.exists() else {"family": "normal", "variant": "normal"}
    if not isinstance(config, dict):
        raise ValueError("V4 selection must be a JSON object")
    family = config.get("family")
    if family == "normal":
        variant = config.get("variant", "normal")
        if variant != "normal":
            raise ValueError("Unknown V4 normal model variant")
    elif family == "compact":
        from .v4_compact import VARIANTS
        variant = config.get("variant")
        if variant not in VARIANTS:
            raise ValueError("Unknown V4 compact model variant")
    else:
        raise ValueError("Unknown V4 model family")
    if "seeds" in config and config["seeds"] != list(SEEDS):
        raise ValueError("V4 deployment requires the fixed three-seed ensemble")
    return {"family": family, "variant": variant, "seeds": list(SEEDS)}


def _model_probability(models, x, names, *, selection=None):
    """Validate the centrally defined training mask/schema before prediction."""
    names = list(names)
    if selection is not None and selection["family"] == "compact":
        from .v4_compact import compact_mask
        expected = np.asarray(compact_mask(names, selection["variant"]), bool)
    else:
        from .v4_model import onset_feature_mask
        expected = np.asarray(onset_feature_mask(names), bool)
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != len(names) or expected.shape != (len(names),):
        raise ValueError("V4 full feature width differs from feature names/mask")
    if len(models) != len(SEEDS):
        raise ValueError("V4 requires all three frozen seed models")
    values = []
    for seed, model in zip(SEEDS, models):
        if not np.array_equal(np.asarray(model.mask, bool), expected):
            raise ValueError("V4 stored feature mask differs from central onset mask")
        selected_names = [name for name, keep in zip(names, expected) if keep]
        if model.booster.feature_name() != selected_names:
            raise ValueError("V4 booster feature schema differs from generated names")
        if int(model.booster.params.get("seed", -1)) != seed:
            raise ValueError("V4 frozen model seed differs from required ensemble")
        probability = np.asarray(model.predict(x), np.float64)
        if probability.shape != (len(x),) or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError("V4 model probabilities must be finite marginals in [0,1]")
        values.append(probability)
    return np.mean(values, axis=0, dtype=np.float64)


def iter_onset_probabilities(args, provenance):
    selection = _selection(args)
    selection_path = args.output / "v4/selection.json"
    if selection_path.exists():
        _remember(provenance, selection_path)
    folder = (args.output / "v4/compact/models/final" / selection["variant"]
              if selection["family"] == "compact" else args.output / "v4/models/final/normal")
    models = []
    for seed in SEEDS:
        path = folder / f"seed{seed}.txt"
        model = load_model(path, args.threads)
        for source in (path, path.with_suffix(".mask.npy"), path.with_suffix(".training.json")):
            _remember(provenance, source)
        metadata = json.loads(path.with_suffix(".training.json").read_text())
        if metadata["seed"] != seed or metadata["rounds"] != model.booster.current_iteration():
            raise ValueError("V4 final model differs from its frozen training metadata")
        if selection["family"] == "compact" and (
                metadata.get("variant") != selection["variant"] or metadata.get("fold") != "final"
                or metadata.get("heldout") is not None):
            raise ValueError("V4 compact final model selection differs from its training metadata")
        models.append(model)
    for panel_id, panel in enumerate(args.panels):
        network = read_network(args.root, panel)
        n = len(network.link_ids)
        for path in sorted((args.root / "corridors" / panel / "network").glob("*.csv")):
            _remember(provenance, path)
        base = args.output / "datasets/final_enhanced_release"
        prior_path, event_path = base / f"{panel}_prior.npy", base / f"{panel}_onset_prior.npy"
        prior, event_prior = np.load(prior_path), np.load(event_path)
        capacity_path = args.output / "v4/data/final" / f"{panel}_capacity.npz"
        with np.load(capacity_path) as data:
            capacity = {key: data[key] for key in data.files}
        if "capacity_discharge_only" not in capacity:
            raise ValueError("V4 requires the frozen discharge-only capacity branch")
        schema_path = args.output / "v4/data/final" / f"{panel}.npz"
        with np.load(schema_path) as data:
            frozen_names = data["feature_names"].tolist()
        for path in (prior_path, event_path, capacity_path):
            _remember(provenance, path)
        provenance[str(schema_path.resolve()) + ":feature_names"] = hashlib.sha256(
            json.dumps(frozen_names).encode()).hexdigest()
        for split in ("validation", "private"):
            index, history, _ = read_window_assets(args.root, panel, split)
            for name in ("window_index.csv", "window_history.parquet"):
                _remember(provenance, args.root / "task2" / panel / split / name)
            days = set(index.date.astype(str))
            days |= {str(np.datetime64(day) - np.timedelta64(1, "D")) for day in days}
            current = read_panel_arrays(args.root, network, split, args.output / "cache", masked=True, dates=days)
            for channel in ("mainline_states_masked", "ramp_states"):
                for path in sorted((args.root / "corridors" / panel / split / channel).rglob("*.parquet")):
                    if path.stem[-10:].replace("_", "-") in days:
                        _remember(provenance, path)
            for window in index[index.condition.eq("queue_onset")].itertuples():
                local = history_arrays(history[history.window_id.eq(window.window_id)], window.forecast_origin, network, current)
                origin_idx = int((local.timestamps[-1] - current.timestamps[0]) / np.timedelta64(5, "m"))
                full, full_names = build_window_features(local, current, network, location_prior=prior,
                    onset_prior=event_prior, capacity=capacity, panel_id=panel_id, origin_idx=origin_idx)
                if full_names != frozen_names:
                    raise ValueError(f"V4 deployment feature names differ from final training: {panel}")
                probability = np.zeros((6, n), np.float64)
                probability[-1] = _model_probability(models, full, full_names, selection=selection)
                yield WindowProbability(panel, split, str(window.window_id), window.forecast_origin,
                    "queue_onset", network.link_ids, probability, {},
                    f"v4_{selection['family']}_{selection['variant']}_mean3")
            del current
            gc.collect()
            log(f"V4 onset inference {panel}/{split}")


def _conditions(args):
    parts = []
    for panel in args.panels:
        for split in ("validation", "private"):
            path = args.root / "task2" / panel / split / "window_index.csv"
            parts.append(pd.read_csv(path, usecols=["window_id", "condition"], dtype=str))
    result = pd.concat(parts, ignore_index=True)
    if result.window_id.duplicated().any() or not result.condition.isin(["queue_onset", "queue_ongoing"]).all():
        raise ValueError("Invalid published window condition mapping")
    return result.set_index("window_id").condition


def _source_probabilities(path, frame):
    source = pd.read_parquet(path)
    if list(source.columns) != [*KEYS, "queue_prob"] or source[KEYS].isna().any().any() or source.duplicated(KEYS).any():
        raise ValueError("Frozen V2 probability keys/schema are invalid")
    if len(source) != len(frame):
        raise ValueError("Frozen V2 probability coverage differs")
    index = pd.MultiIndex.from_frame(frame[KEYS])
    aligned = source.set_index(KEYS).reindex(index).queue_prob.to_numpy(np.float64)
    if not np.isfinite(aligned).all() or ((aligned < 0) | (aligned > 1)).any():
        raise ValueError("Frozen V2 probabilities missing or outside [0,1]")
    return aligned


def verify_submission(args, path):
    expected, key = _contracts(args)
    _, result = _validate_queue(path, expected, key)
    dump_json(args.output / "v4/submission_verification.json", {Path(path).name: result})
    return result


def shared_check(args, queue):
    """Use the shared checker without writing any existing verification file."""
    expected, key = _contracts(args)
    candidate, checked_queue = _validate_queue(queue, expected, key)
    mapped = key.join(candidate, on=KEYS, how="left", validate="1:1").select(
        "submission_id", pl.col("queue_pred").alias("new_queue"))
    carrier_path = Path(getattr(args, "carrier", PROJECT / "outputs/submissions/baseline_v0.csv"))
    carrier_hash = _sha(carrier_path)
    carrier = pl.read_csv(carrier_path, schema_overrides={"submission_id": pl.String})
    columns = carrier.columns
    joined = (carrier.join(mapped, on="submission_id", how="left", validate="1:1")
        .with_columns(pl.when(pl.col("task") == "queue").then(pl.col("new_queue"))
                      .otherwise(pl.col("queue_pred")).alias("queue_pred")).select(columns))
    if joined.filter(pl.col("task") == "queue")["queue_pred"].null_count():
        raise ValueError("Missing queue mapping in temporary carrier")
    target = args.output / "v4/qa"
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="onset_", dir=target) as directory:
        path = Path(directory) / "combined.csv"
        joined.write_csv(path)
        result = subprocess.run(["make", "check", f"SUB={path.resolve()}"], cwd=PROJECT,
            env={**os.environ, "TFB_RELEASE": str(args.root.resolve()), "PYTHONDONTWRITEBYTECODE": "1"},
            text=True, capture_output=True)
        (target / "make_check_queue_submission_v4_onset.log").write_text(result.stdout + result.stderr)
        manifest = dict(queue=str(queue), queue_sha256=checked_queue["sha256"], carrier=str(carrier_path),
            carrier_sha256=carrier_hash, returncode=result.returncode, temporary_carrier_removed=True, submitted=False)
        dump_json(target / "make_check_queue_submission_v4_onset.json", manifest)
        if result.returncode:
            raise RuntimeError("V4 shared make check failed; see v4/qa log")
    if _sha(carrier_path) != carrier_hash:
        raise RuntimeError("Read-only shared carrier changed during V4 QA")
    return manifest


def predict(args, *, check=True):
    args.root, args.output = Path(args.root), Path(args.output)
    immutable = _immutable_hashes(args.output)
    selection = _selection(args)
    source = args.output / "queue_submission_v2.csv"
    expected, key = _contracts(args)
    _validate_queue(source, expected, key)
    frame = pd.read_csv(source, dtype={name: str for name in KEYS})
    original = frame.copy(deep=True)
    conditions = frame.window_id.map(_conditions(args))
    if conditions.isna().any():
        raise ValueError("Frozen V2 contains a window absent from the published indices")
    onset = conditions.eq("queue_onset").to_numpy()
    ongoing = ~onset
    probabilities = _source_probabilities(args.output / "probs_v2.parquet", frame)
    old_probability = probabilities.copy()
    rows_by_window = frame.groupby("window_id", sort=False).indices
    seen = np.zeros(len(frame), bool)
    provenance, changes = {}, []
    for item in iter_onset_probabilities(args, provenance):
        if item.condition != "queue_onset" or item.window_id not in rows_by_window:
            raise ValueError("V4 must predict only published onset windows")
        rows = rows_by_window[item.window_id]
        if not onset[rows].all() or seen[rows].any():
            raise ValueError("Duplicate onset window or attempted ongoing replacement")
        p = np.asarray(item.probability, np.float64)
        if p.shape != (6, len(item.link_ids)) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or p[:-1].any():
            raise ValueError("V4 requires finite [6,link] probabilities with zero early onset horizons")
        out = frame.iloc[rows]
        origin = pd.Timestamp(item.origin)
        origin = origin.tz_localize("UTC") if origin.tzinfo is None else origin.tz_convert("UTC")
        horizon = (pd.to_datetime(out.timestamp, utc=True) - origin).dt.total_seconds().to_numpy() / 300
        if not np.equal(horizon, np.floor(horizon)).all() or not ((horizon >= 1) & (horizon <= 6)).all():
            raise ValueError("Unexpected onset forecast horizon")
        hi = horizon.astype(int) - 1
        links = pd.Index(np.asarray(item.link_ids).astype(str))
        li = links.get_indexer(out.link_id)
        if not links.is_unique or (li < 0).any() or len(rows) != p.size or len(set(zip(hi, li))) != p.size:
            raise ValueError("Incomplete onset link/horizon coverage")
        decoded = decode_probabilities(p, onset=True)[hi, li]
        frame.loc[rows, "queue_pred"] = decoded
        probabilities[rows] = p[hi, li]
        seen[rows] = True
        before = original.iloc[rows].queue_pred.to_numpy()
        changes.append(dict(panel=item.panel, split=item.split, window_id=item.window_id,
            changed_cells=int(np.count_nonzero(decoded != before)),
            v2_positive=int(before.sum()), v4_positive=int(decoded.sum()), probability_sum=float(p.sum())))
    if not seen[onset].all() or seen[ongoing].any():
        raise ValueError("V4 onset prediction coverage is incomplete")
    pd.testing.assert_frame_equal(frame.loc[ongoing], original.loc[ongoing], check_exact=True)
    np.testing.assert_array_equal(probabilities[ongoing], old_probability[ongoing])
    if immutable != _immutable_hashes(args.output):
        raise RuntimeError("Frozen source or QA artifacts changed during V4 prediction")
    destination = args.output / "queue_submission_v4_onset.csv"
    temporary = destination.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False)
    _validate_queue(temporary, expected, key)
    temporary.replace(destination)
    probability_path = args.output / "probs_v4_onset.parquet"
    probability_frame = frame[KEYS].copy()
    probability_frame["queue_prob"] = probabilities
    probability_temporary = probability_path.with_suffix(".tmp.parquet")
    probability_frame.to_parquet(probability_temporary, index=False)
    probability_temporary.replace(probability_path)
    verification = verify_submission(args, destination)
    qa = shared_check(args, destination) if check else None
    if immutable != _immutable_hashes(args.output):
        raise RuntimeError("Frozen source or QA artifacts changed during V4 verification")
    target = args.output / "v4"
    pd.DataFrame(changes).to_csv(target / "prediction_changes.csv", index=False)
    manifest = dict(rows=len(frame), onset_rows=int(onset.sum()), ongoing_rows=int(ongoing.sum()),
        onset_windows=len(changes), seeds=list(SEEDS), output=str(destination), sha256=_sha(destination),
        model_selection=selection,
        probability_output=str(probability_path), probability_sha256=_sha(probability_path),
        probability_stage="float64 raw V4 onset mean; frozen raw V2 ongoing probabilities copied",
        onset_t5_t25_zero=True, ongoing_predictions_unchanged=True, ongoing_probabilities_unchanged=True,
        changed_cells=sum(row["changed_cells"] for row in changes), positive_rows=int(frame.queue_pred.sum()),
        frozen_artifact_hashes=immutable, input_sha256=provenance, verification=verification,
        shared_check=qa, submitted=False)
    dump_json(target / "prediction_manifest.json", manifest)
    log(f"V4 onset output: {len(frame):,} rows; {manifest['changed_cells']} onset cells changed; ongoing exact V2")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RELEASE_ROOT)
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/t2")
    parser.add_argument("--carrier", type=Path, default=PROJECT / "outputs/submissions/baseline_v0.csv")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--skip-shared-check", action="store_true")
    args = parser.parse_args()
    args.root = resolve_root(args.root)
    args.panels = panel_names(args.root)
    manifest = predict(args, check=not args.skip_shared_check)
    print(json.dumps({key: manifest[key] for key in ("rows", "sha256", "changed_cells", "ongoing_predictions_unchanged")}, indent=2))


if __name__ == "__main__":
    main()
