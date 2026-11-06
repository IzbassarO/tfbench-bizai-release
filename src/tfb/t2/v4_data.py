"""Onset-only V4 data with window grouping and label-only head rarity.

The existing V2 288-column samples and evaluation cohorts stay unchanged.
New features use causal legal history and masked same-day prefixes. Future
heads and training-head frequencies are metadata only, never feature columns.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from .features import build_features
from .io import PanelArrays, _dense_assign, dump_json, panel_names, parquet_frame, read_network, read_panel_arrays, resolve_root
from .labels import make_labels
from .run import PROJECT, log
from .v2_data import CUTOFF, extra_features
from .v2_experiment import FOLDS
from ..io import RELEASE as RELEASE_ROOT   # TFB_RELEASE (run.py --data), else data/kaggle_public

FOLDS_V4 = ("janfeb", "octnov", "final")
EXPECTED_NEW_FEATURES = 104


def downstream_head(labels):
    """Maximum topology index of ANY raw queue; -1 means no observed head."""
    q = np.asarray(labels, bool)
    if q.ndim == 1:
        return int(np.flatnonzero(q)[-1]) if q.any() else -1
    if q.ndim != 2:
        raise ValueError("Head labels must have shape [link] or [window, link]")
    return np.where(q.any(axis=1), q.shape[1] - 1 - np.argmax(q[:, ::-1], axis=1), -1).astype(np.int32)


def onset_head_rarity(train_heads, train_origins, eval_heads, eval_origins, *, radius_minutes=60):
    """Event frequency of the eval head in a circular origin-TOD neighborhood.

    One training window contributes one event, regardless of corridor length
    or its number of sampled cells. Unknown training heads are counted and
    reported separately; known-head events form the frequency denominator.
    Zero-support bins have frequency 0 plus an explicit unsupported flag.
    Unknown eval heads retain NaN frequency and are not called nonrecurrent.
    """
    train_heads, eval_heads = np.asarray(train_heads, int), np.asarray(eval_heads, int)
    train_times = pd.to_datetime(np.asarray(train_origins)).to_numpy(dtype="datetime64[m]").astype(np.int64) % 1440
    eval_times = pd.to_datetime(np.asarray(eval_origins)).to_numpy(dtype="datetime64[m]").astype(np.int64) % 1440
    if len(train_heads) != len(train_times) or len(eval_heads) != len(eval_times):
        raise ValueError("Each head requires exactly one origin")
    if not 0 <= radius_minutes <= 720:
        raise ValueError("Radius must be within [0,720] minutes")
    rows = []
    for head, minute in zip(eval_heads, eval_times):
        difference = np.abs(train_times - minute)
        nearby = np.minimum(difference, 1440 - difference) <= radius_minutes
        known = nearby & (train_heads >= 0)
        same = known & (train_heads == head) if head >= 0 else np.zeros_like(known)
        n, count = int(known.sum()), int(same.sum())
        frequency = float(count / n) if n and head >= 0 else (0. if head >= 0 else np.nan)
        rows.append(dict(head_prior=frequency, head_event_count=n, head_matching_events=count,
            head_total_events=int(nearby.sum()), head_unknown_events=int((nearby & (train_heads < 0)).sum()),
            head_prior_unsupported=n == 0, nonrecurrent_onset=bool(head >= 0 and frequency < .05)))
    return pd.DataFrame(rows, columns=["head_prior", "head_event_count", "head_matching_events",
        "head_total_events", "head_unknown_events", "head_prior_unsupported", "nonrecurrent_onset"])


def _source(args, fold):
    base = args.output / "datasets" / FOLDS[fold]
    return base, base.parent / (base.name + "_x")


def prepare_metadata(args, fold, panel, network, arrays):
    base, source = _source(args, fold)
    destination = args.output / "v4/data" / fold
    destination.mkdir(parents=True, exist_ok=True)
    windows = pd.read_csv(base / f"{panel}_windows.csv")
    windows = windows[windows.condition.eq("queue_onset")].copy()
    cutoff = np.datetime64(CUTOFF[fold], "m")
    source_path = source / f"{panel}_queue_onset.npz"
    with np.load(source_path) as data:
        train_y, train_median, weights = data["train_y"], data["train_median"], data["train_weight"]
        eval_y, eval_eligible = data["eval_y"], data["eval_eligible"]
        eval_median = data["eval_median"]
    meta = pd.read_csv(source / f"{panel}_queue_onset_eval.csv")
    repeated = {key: [] for key in ["train_window_id", "train_origin", "train_panel", "train_head_index", "train_link_index"]}
    training_rows, eval_windows = [], []
    offset = 0
    for window in windows.itertuples():
        origin = int(window.origin_idx)
        y = make_labels(arrays.speed[origin + 6:origin + 7], network.vf)[0]
        head = downstream_head(y)
        valid = arrays.eligible[origin + 6]
        is_train = arrays.timestamps[origin + 6] < cutoff
        if is_train:
            # Onset sampling keeps every eligible T+30 cell, including every
            # negative. Replay labels to prove sample order before adding IDs.
            n = int(valid.sum())
            np.testing.assert_array_equal(train_y[offset:offset+n], y[valid])
            assert np.all(weights[offset:offset+n] == 1.)
            info = dict(window_id=window.window_id, panel=panel, origin=str(arrays.timestamps[origin]),
                origin_idx=origin, head_index=head, head_link=str(network.link_ids[head]) if head >= 0 else "",
                raw_T30_queue_cells=int(y.sum()), eligible_T30_cells=n,
                train_row_start=offset, train_row_stop=offset+n)
            training_rows.append(info)
            for key, value in [("train_window_id", window.window_id), ("train_origin", str(arrays.timestamps[origin])),
                ("train_panel", panel), ("train_head_index", head)]:
                repeated[key].append(np.repeat(value, n))
            repeated["train_link_index"].append(np.flatnonzero(valid).astype(np.int32))
            offset += n
        else:
            eval_windows.append(dict(window_id=window.window_id, origin=str(arrays.timestamps[origin]), origin_idx=origin))
    assert offset == len(train_y), (fold, panel, "Onset training sample replay failed")
    training = pd.DataFrame(training_rows)
    expected = pd.DataFrame(eval_windows, columns=["window_id", "origin", "origin_idx"])
    assert np.array_equal(expected.window_id.to_numpy(str), meta.window_id.to_numpy(str))
    assert np.array_equal(expected.origin.to_numpy(str), meta.origin.to_numpy(str))
    assert len(meta) == len(eval_y)
    heads = downstream_head(eval_y[:, -1]) if len(eval_y) else np.empty(0, np.int32)
    meta["head_index"] = heads
    meta["head_link"] = [str(network.link_ids[h]) if h >= 0 else "" for h in heads]
    meta["head_known"] = heads >= 0
    meta["origin_idx"] = expected.origin_idx.to_numpy(int)
    rarity = onset_head_rarity(training.head_index, training.origin, heads, meta.origin)
    for column in rarity: meta[column] = rarity[column].to_numpy()
    packed = {key: np.concatenate(value) for key, value in repeated.items()}
    packed.update(train_y=train_y, train_median=train_median, train_weight=weights,
        eval_y=eval_y, eval_median=eval_median, eval_eligible=eval_eligible)
    if fold != "final":
        baseline_path = args.output / "v3/eval" / fold / f"{panel}_queue_onset.npz"
        with np.load(baseline_path) as baseline:
            for key in ("window_id", "origin"):
                assert np.array_equal(baseline[key], meta[key].to_numpy(str)), (fold, panel, key)
            np.testing.assert_array_equal(baseline["truth"], eval_y)
            np.testing.assert_array_equal(baseline["eligible"], eval_eligible)
            packed["baseline_probability"] = baseline["probability_v2"]
            packed["baseline_prediction"] = baseline["prediction_v2"]
    np.savez(destination / f"{panel}_metadata.npz", **packed)
    training.to_csv(destination / f"{panel}_train_windows.csv", index=False)
    meta.to_csv(destination / f"{panel}_eval.csv", index=False)
    return training, meta


def _load_masked_day(root, network, arrays, date):
    paths = sorted((root / "corridors" / network.panel / "train/mainline_states_masked").rglob(
        f"synthetic_mainline_{str(date).replace('-', '_')}.parquet"))
    if len(paths) != 1:
        raise ValueError(f"Expected one masked day file for {network.panel}/{date}, found {len(paths)}")
    times = np.datetime64(date, "m") + np.arange(288) * np.timedelta64(5, "m")
    shape = (288, len(network.link_ids))
    speed, flow, occupancy = [np.full(shape, np.nan, np.float32) for _ in range(3)]
    eligible = np.zeros(shape, bool)
    frame = parquet_frame(paths[0], ["timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy", "is_score_eligible"])
    _dense_assign(frame, (speed, flow, occupancy, eligible), times[0],
        dict(zip(network.link_ids, range(shape[1]))))
    lo = int(np.searchsorted(arrays.timestamps, times[0]))
    return PanelArrays(times, speed, flow, occupancy, eligible, arrays.ramp_on[lo:lo+288])


def causal_history_and_day(arrays, masked_day, origin):
    """Build protected raw history plus masked T and masked same-day prefix."""
    timestamp = arrays.timestamps[origin]
    slot = int((timestamp - masked_day.timestamps[0]) / np.timedelta64(5, "m"))
    if slot < 12 or slot >= len(masked_day.timestamps) or masked_day.timestamps[slot] != timestamp:
        raise ValueError("Origin requires a complete same-day history")
    history_values = [values[origin-12:origin+1].copy() for values in (arrays.speed, arrays.flow, arrays.occupancy)]
    day_values = [values[:slot+1].copy() for values in (masked_day.speed, masked_day.flow, masked_day.occupancy)]
    for local, prefix, masked, raw in zip(history_values, day_values,
        (masked_day.speed, masked_day.flow, masked_day.occupancy),
        (arrays.speed, arrays.flow, arrays.occupancy)):
        local[-1] = masked[slot]
        prefix[slot-12:slot] = raw[origin-12:origin]
    history = PanelArrays(arrays.timestamps[origin-12:origin+1], *history_values,
        arrays.eligible[origin-12:origin+1], arrays.ramp_on[origin-12:origin+1])
    day = PanelArrays(masked_day.timestamps[:slot+1], *day_values,
        masked_day.eligible[:slot+1], masked_day.ramp_on[:slot+1])
    return history, day, slot


def prepare_features(args, fold, panel, network, arrays):
    from .v4_features import fit_empirical_capacity, build_onset_features
    base, source = _source(args, fold)
    destination = args.output / "v4/data" / fold
    with np.load(destination / f"{panel}_metadata.npz") as meta_data:
        packed = {key: meta_data[key] for key in meta_data.files}
    with np.load(source / f"{panel}_queue_onset.npz") as data:
        train_x, eval_x = data["train_x"], data["eval_x"]
        old_names = data["feature_names"].tolist()
    assert train_x.shape[1] == eval_x.shape[-1] == len(old_names) == 288
    prior = np.load(base / f"{panel}_prior.npy")
    capacity = fit_empirical_capacity(arrays, network, CUTOFF[fold])
    numeric = {key: value for key, value in capacity.items() if isinstance(value, np.ndarray)}
    np.savez(destination / f"{panel}_capacity.npz", **numeric)
    dump_json(destination / f"{panel}_capacity.json", {key: value for key, value in capacity.items() if not isinstance(value, np.ndarray)})
    windows = pd.read_csv(base / f"{panel}_windows.csv")
    windows = windows[windows.condition.eq("queue_onset")]
    days = {}; train_extra=[]; eval_extra=[]; offset=0; eval_offset=0
    for window in windows.itertuples():
        origin = int(window.origin_idx)
        date = str(arrays.timestamps[origin].astype("datetime64[D]"))
        if date not in days: days[date] = _load_masked_day(args.root, network, arrays, date)
        history, day, slot = causal_history_and_day(arrays, days[date], origin)
        x, names = build_onset_features(history, network, capacity, dayarrays=day, origin_idx=slot)
        assert x.shape == (len(network.link_ids), len(names))
        if len(names) != EXPECTED_NEW_FEATURES:
            raise ValueError(f"V4 schema requires {EXPECTED_NEW_FEATURES} new columns, found {len(names)}")
        assert not any("head_index" in name or "head_prior" in name or name.startswith("location_") for name in names)
        # Independently replay the original 183 columns and the 58 V2 columns
        # to prove that added metadata/features align with the samples.
        original, original_names = build_features(history.speed, history.flow, history.occupancy,
            origin=12, timestamps=history.timestamps, ramp_on=history.ramp_on,
            location_prior=prior, panel_id=panel_names(args.root).index(panel), **network.feature_kwargs())
        v2_extra, v2_names = extra_features(history, network)
        n=len(network.link_ids)
        assert old_names[:183] == original_names and old_names[230:] == v2_names
        if arrays.timestamps[origin+6] < np.datetime64(CUTOFF[fold], "m"):
            valid=arrays.eligible[origin+6]; count=int(valid.sum()); sl=slice(offset,offset+count)
            np.testing.assert_array_equal(train_x[sl,:183], original[-n:][valid])
            np.testing.assert_array_equal(train_x[sl,230:], v2_extra[-n:][valid])
            train_extra.append(x[valid]); offset+=count
        else:
            np.testing.assert_array_equal(eval_x[eval_offset,:,:183], original[-n:])
            np.testing.assert_array_equal(eval_x[eval_offset,:,230:], v2_extra[-n:])
            eval_extra.append(x); eval_offset+=1
    assert offset == len(train_x) and eval_offset == len(eval_x)
    packed.update(train_x=np.column_stack([train_x,np.concatenate(train_extra)]),
        eval_x=np.concatenate([eval_x,np.stack(eval_extra)],axis=-1) if eval_extra else
            np.empty((0,len(network.link_ids),288+len(names)),np.float32), feature_names=np.asarray(old_names+names))
    temporary=destination/f"{panel}.tmp.npz"
    np.savez(temporary,**packed);temporary.replace(destination/f"{panel}.npz")
    dump_json(destination/f"{panel}_build.json",dict(fold=fold,panel=panel,training_rows=offset,
        eval_windows=eval_offset,source_columns=288,new_columns=len(names),
        training_cutoff_exclusive=CUTOFF[fold],base_feature_alignment_checked=241,
        raw_heads_are_metadata_only=True,head_event_frequency_is_metadata_only=True))
    log(f"V4 saved {fold}/{panel}: {offset:,} rows, {eval_offset} eval, {len(names)} new columns")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=RELEASE_ROOT)
    parser.add_argument("--output",type=Path,default=PROJECT/"outputs/t2")
    parser.add_argument("--fold",action="append",choices=FOLDS_V4)
    parser.add_argument("--panel",action="append")
    parser.add_argument("--stage",choices=["metadata","features","all"],default="all")
    args=parser.parse_args();args.root=resolve_root(args.root)
    for fold in args.fold or FOLDS_V4:
        for panel in args.panel or panel_names(args.root):
            network=read_network(args.root,panel)
            arrays=read_panel_arrays(args.root,network,"train",args.output/"cache")
            if args.stage in {"metadata","all"}:prepare_metadata(args,fold,panel,network,arrays)
            if args.stage in {"features","all"}:prepare_features(args,fold,panel,network,arrays)
            del arrays;gc.collect()


if __name__=="__main__":main()
