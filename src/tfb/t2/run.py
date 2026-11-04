"""Rebuild causal Task 2 models, CV, reports and submissions from the raw release.

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m tfb.t2.run --stage all
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .features import build_features
from .io import (current_observations, dump_json, history_arrays, panel_names,
                 read_network, read_panel_arrays, read_window_assets, resolve_root)
from .labels import make_labels
from .train import aggregate_scores, decode_probabilities, fit_classifier, score_window
from .windows import select_windows

PROJECT = Path(__file__).resolve().parents[3]
CONDITIONS = ("queue_onset", "queue_ongoing")


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def dataset_tag(args, final=False):
    if getattr(args, "dataset_name", None):
        return args.dataset_name
    tag = "final" if final else "cv"
    tag = tag if args.feature_set == "base" else tag + "_enhanced"
    return tag + "_release" if args.window_policy == "release" else tag


def experiment_tag(args):
    return args.feature_set + ("_release" if args.window_policy == "release" else "")


def location_prior(labels, eligible, timestamps, end_date):
    """Smoothed link × time-of-day queue frequency, fitted on model train dates."""
    chosen = timestamps < np.datetime64(end_date, "m")
    n_days = int(chosen.sum() // 288)
    q = (labels[chosen] & eligible[chosen]).reshape(n_days, 288, -1).sum(axis=0, dtype=np.float64)
    n = eligible[chosen].reshape(n_days, 288, -1).sum(axis=0, dtype=np.float64)
    q = sum(np.roll(q, offset, axis=0) for offset in range(-3, 4))
    n = sum(np.roll(n, offset, axis=0) for offset in range(-3, 4))
    return ((q + .1) / (n + 2.)).astype(np.float32)


def enrich_features(x, names, arrays, network, *, onset_prior, day_context, day_names):
    from .extras import build_short_extras
    extra, extra_names = build_short_extras(arrays.speed, arrays.flow, arrays.timestamps,
        origin=len(arrays.timestamps)-1, vf=network.vf, capacity=network.capacity,
        distance_km=network.distance_km)
    n_links = len(network.link_ids)
    common = np.column_stack([extra, day_context])
    origin_epoch = arrays.timestamps[-1].astype("datetime64[m]").astype(int)
    prior_rows = []
    for horizon in (5, 10, 15, 20, 25, 30):
        epoch = origin_epoch + horizon
        slot = int(epoch % 1440 // 5)
        weekend = int((epoch // 1440 + 3) % 7 >= 5)
        p = onset_prior[weekend, slot]
        neighbors = np.pad(p, (2, 2), constant_values=0.)
        nearby = np.maximum.reduce([neighbors[offset:offset+n_links] for offset in range(5)])
        prior_rows.append(np.column_stack([p, nearby]))
    enhanced = np.column_stack([x, np.tile(common, (6, 1)), np.concatenate(prior_rows)])
    return enhanced.astype(np.float32), names + extra_names + day_names + [
        "location_onset_event_prior_target", "location_onset_event_prior_nearby"]


def _features_at(arrays, network, origin, current, current_row, prior, panel_id):
    start = origin - 12
    speed = arrays.speed[start:origin + 1].copy()
    flow = arrays.flow[start:origin + 1].copy()
    occ = arrays.occupancy[start:origin + 1].copy()
    for dest, name in [(speed, "speed_kmh"), (flow, "flow_vph"), (occ, "occupancy")]:
        dest[-1] = current[name][current_row]
    x, names = build_features(speed, flow, occ, origin=12,
                          timestamps=arrays.timestamps[start:origin + 1],
                          ramp_on=arrays.ramp_on[start:origin + 1],
                          location_prior=prior, panel_id=panel_id, **network.feature_kwargs())
    if "day_context" in current:
        from .io import PanelArrays
        local = PanelArrays(arrays.timestamps[start:origin+1], speed, flow, occ,
                            arrays.eligible[start:origin+1], arrays.ramp_on[start:origin+1])
        onset_prior = current["onset_prior"]
        fold = current["onset_prior_folds"][current_row]
        if fold >= 0:
            onset_prior = current["onset_prior_oof"][int(fold)]
        return enrich_features(x, names, local, network, onset_prior=onset_prior,
                               day_context=current["day_context"][current_row],
                               day_names=current["day_context_names"])
    return x, names


def prepare(args, *, final=False):
    tag = dataset_tag(args, final)
    artifact_dir = args.output / "datasets" / tag
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cutoff = getattr(args, "train_cutoff", None) or ("2031-03-01" if final else "2031-01-01")
    eval_end = getattr(args, "eval_end", None)
    window_audit, label_audit, counts = [], [], []
    for panel in args.panels:
        panel_id = panel_names(args.root).index(panel)
        meta_path = artifact_dir / f"{panel}_build.json"
        build_config = dict(feature_set=args.feature_set, window_policy=args.window_policy, schema_version=5, negative_sample=args.negative_sample,
                            seed=args.seed, onset_prior_folds=args.onset_prior_folds, cutoff=cutoff)
        if eval_end is not None:
            build_config["eval_end"] = eval_end
        compatible = (args.feature_set == "base" and not meta_path.exists()) or (
            meta_path.exists() and json.loads(meta_path.read_text()) == build_config)
        if all((artifact_dir / f"{panel}_{c}.npz").exists() for c in CONDITIONS) and compatible and not args.refresh:
            log(f"prepare {tag} {panel}: cached")
            continue
        network = read_network(args.root, panel)
        arrays = read_panel_arrays(args.root, network, "train", args.output / "cache", refresh=args.refresh)
        raw = make_labels(arrays.speed, network.vf)
        smooth = make_labels(arrays.speed, network.vf, method="median3")
        if not final:
            # Centered target denoising must not borrow a held-out Jan 1 slot.
            boundary = int(np.searchsorted(arrays.timestamps, np.datetime64(cutoff, "m")))
            smooth[boundary - 1] = raw[boundary - 1]
        # Selector can only detect queues at observable cells; supervised
        # target/scorer retain the raw eligibility mask separately.
        visible_queue = raw & arrays.eligible
        if args.window_policy == "release":
            from .labels import make_selector_labels
            from .windows import select_rolling_windows
            selector_labels = make_selector_labels(arrays.speed, network.vf, method="ffill", timestamps=arrays.timestamps)
            released_proxy = select_windows(selector_labels, arrays.timestamps, arrays.eligible,
                min_spacing_minutes=360, seed=args.seed, history_includes_origin=False,
                condition_policy="contract", spacing_policy="shared", max_per_condition=5)
            windows = select_rolling_windows(selector_labels, arrays.timestamps, arrays.eligible,
                                              block_days=30, stride_days=7, seed=args.seed)
            del selector_labels
        else:
            windows = select_windows(visible_queue, arrays.timestamps, arrays.eligible,
                                     min_spacing_minutes=360, seed=args.seed)
            released_proxy = windows
        if eval_end is not None:
            # A fold never needs examples whose targets extend past its end.
            end_times = pd.to_datetime(windows.origin_time).to_numpy(dtype="datetime64[m]") + np.timedelta64(30, "m")
            windows = windows.loc[end_times < np.datetime64(eval_end, "m")].reset_index(drop=True)
        windows["panel"] = panel
        windows["window_id"] = [f"LOCAL_{panel}_{i:05d}" for i in range(len(windows))]
        windows.to_csv(artifact_dir / f"{panel}_windows.csv", index=False)
        prior = location_prior(raw, arrays.eligible, arrays.timestamps, cutoff)
        np.save(artifact_dir / f"{panel}_prior.npy", prior)
        official = pd.read_csv(args.root / "task2" / panel / "train" / "window_index.csv")
        official_times = pd.to_datetime(official.forecast_origin, utc=True).dt.tz_localize(None)
        for condition in CONDITIONS:
            generated = windows[windows.condition.eq(condition)]
            shipped = official[official.condition.eq(condition)]
            first_five = released_proxy[released_proxy.condition.eq(condition)].head(5)
            shipped_set = set(official_times[official.condition.eq(condition)].astype(str))
            matches = len(set(pd.to_datetime(first_five.origin_time).astype(str)) & shipped_set)
            window_audit.append(dict(panel=panel, condition=condition, generated=len(generated),
                shipped=len(shipped), first_five_exact_matches=matches,
                fraction_first_queue_T30=float(generated.first_queue_step.eq(6).mean()),
                mean_history_coverage=float(generated.history_coverage.mean()),
                mean_persistence_iou=float(generated.persistence_iou.mean())))
        eligible = arrays.eligible
        label_audit.append(dict(panel=panel, eligible_cells=int(eligible.sum()),
            raw_queued=int((raw & eligible).sum()), median_queued=int((smooth & eligible).sum()),
            changed_labels=int(((raw != smooth) & eligible).sum()),
            changed_fraction=float(((raw != smooth) & eligible).sum() / eligible.sum())))
        origins = windows.origin_idx.to_numpy(dtype=int)
        log(f"prepare {tag} {panel}: {len(windows)} windows; read legal masked T rows")
        current = current_observations(args.root, network, arrays.timestamps[origins],
                                       include_context=args.feature_set == "enhanced",
                                       released_history=arrays if args.feature_set == "enhanced" else None)
        if args.feature_set == "enhanced":
            from .priors import fit_onset_prior
            onset_prior, onset_info = fit_onset_prior(raw, arrays.eligible, arrays.timestamps,
                                                     end_date=cutoff, smoothing_minutes=60)
            current["onset_prior"] = onset_prior
            # Avoid training-target encoding: every training day's conditional
            # event-prior feature is fitted with its date fold excluded.
            day_number = arrays.timestamps.astype("datetime64[D]").astype(int)
            folds = day_number % args.onset_prior_folds
            origin_folds = folds[origins].copy()
            origin_folds[arrays.timestamps[origins] >= np.datetime64(cutoff)] = -1
            current["onset_prior_folds"] = origin_folds
            current["onset_prior_oof"] = []
            for fold in range(args.onset_prior_folds):
                fit_labels, fit_eligible = raw.copy(), arrays.eligible.copy()
                fit_labels[folds == fold] = False
                fit_eligible[folds == fold] = False
                fitted, _ = fit_onset_prior(fit_labels, fit_eligible, arrays.timestamps,
                                            end_date=cutoff, smoothing_minutes=60)
                current["onset_prior_oof"].append(fitted)
                del fit_labels, fit_eligible
            np.save(artifact_dir / f"{panel}_onset_prior.npy", onset_prior)
            dump_json(artifact_dir / f"{panel}_onset_prior_info.json",
                      {k: v for k, v in onset_info.items() if not isinstance(v, np.ndarray)})
        names = None
        for condition in CONDITIONS:
            train_x, train_y, train_median, train_weight = [], [], [], []
            eval_x, eval_y, eval_median, eval_eligible, metadata = [], [], [], [], []
            rng = np.random.default_rng(args.seed + panel_id * 101 + int(condition.endswith("ongoing")))
            selected = windows[windows.condition.eq(condition)]
            for row_no, row in enumerate(selected.itertuples()):
                origin, current_row = int(row.origin_idx), int(row.Index)
                x, names = _features_at(arrays, network, origin, current, current_row, prior, panel_id)
                truth = raw[origin + 1:origin + 7]
                denoised = smooth[origin + 1:origin + 7]
                target_eligible = arrays.eligible[origin + 1:origin + 7]
                is_train = arrays.timestamps[origin + 6] < np.datetime64(cutoff)
                n_links = len(network.link_ids)
                if condition == "queue_onset":
                    model_x, y, ym, valid = x[-n_links:], truth[-1], denoised[-1], target_eligible[-1]
                else:
                    model_x, y, ym, valid = x, truth.reshape(-1), denoised.reshape(-1), target_eligible.reshape(-1)
                if is_train:
                    # Inverse-probability weights preserve calibrated probabilities
                    # while bounding memory on the shared 16 GB machine.
                    sample_p = 1. if condition == "queue_onset" else args.negative_sample
                    keep = valid & (y | ym | (rng.random(len(y)) < sample_p))
                    weight = np.where(y | ym, 1., 1. / sample_p).astype(np.float32)
                    train_x.append(model_x[keep])
                    train_y.append(y[keep])
                    train_median.append(ym[keep])
                    train_weight.append(weight[keep])
                elif not final:
                    eval_x.append(model_x)
                    eval_y.append(truth)
                    eval_median.append(denoised)
                    eval_eligible.append(target_eligible)
                    now = visible_queue[origin]
                    tod = int(arrays.timestamps[origin].astype("datetime64[m]").astype(int) % 1440 // 5)
                    rare_fraction = float((prior[tod, now] < .05).mean()) if now.any() else 0.
                    metadata.append(dict(window_id=row.window_id, panel=panel, condition=condition,
                                         origin=str(arrays.timestamps[origin]),
                                         official_like=bool(getattr(row, "official_like", True)),
                                         rare_queue_fraction=rare_fraction, shift_weight=1. + 4. * rare_fraction))
                if row_no % 250 == 0:
                    log(f"{panel} {condition}: features {row_no}/{len(selected)}")
            if not train_x:
                raise ValueError(f"No training rows for {panel}/{condition}")
            n_model_cells = n_links if condition == "queue_onset" else 6 * n_links
            packed = dict(train_x=np.concatenate(train_x), train_y=np.concatenate(train_y),
                          train_median=np.concatenate(train_median), train_weight=np.concatenate(train_weight),
                          eval_x=np.stack(eval_x) if eval_x else np.empty((0, n_model_cells, len(names)), np.float32),
                          eval_y=np.stack(eval_y) if eval_y else np.empty((0, 6, n_links), bool),
                          eval_median=np.stack(eval_median) if eval_median else np.empty((0, 6, n_links), bool),
                          eval_eligible=np.stack(eval_eligible) if eval_eligible else np.empty((0, 6, n_links), bool),
                          feature_names=np.asarray(names))
            np.savez(artifact_dir / f"{panel}_{condition}.npz", **packed)
            pd.DataFrame(metadata, columns=["window_id", "panel", "condition", "origin", "official_like", "rare_queue_fraction", "shift_weight"]).to_csv(
                artifact_dir / f"{panel}_{condition}_eval.csv", index=False)
            counts.append(dict(panel=panel, condition=condition, train_rows=len(packed["train_y"]),
                               eval_windows=len(eval_x), eval_official_like=sum(m["official_like"] for m in metadata),
                               total_windows=len(selected)))
            log(f"saved {panel} {condition}: {len(packed['train_y']):,} train cells, {len(eval_x)} CV windows")
            del packed, train_x, train_y, train_median, train_weight, eval_x, eval_y, eval_median, eval_eligible
            gc.collect()
        del arrays, raw, smooth, visible_queue, current
        dump_json(meta_path, build_config)
        gc.collect()
    if window_audit:
        for filename, entries, keys in [("window_audit.csv", window_audit, ["panel", "condition"]),
                                         ("label_audit.csv", label_audit, ["panel"]),
                                         ("dataset_counts.csv", counts, ["panel", "condition"])]:
            path = artifact_dir / filename
            frame = pd.DataFrame(entries)
            if path.exists():
                frame = pd.concat([pd.read_csv(path), frame], ignore_index=True).drop_duplicates(keys, keep="last")
            frame.sort_values(keys).to_csv(path, index=False)


def fit_models(args, *, final=False, median=False, conditions=None):
    tag = dataset_tag(args, final)
    source = args.output / "datasets" / tag
    models, names = {}, None
    for condition in (conditions or CONDITIONS):
        xs, ys, weights = [], [], []
        for panel in args.panels:
            with np.load(source / f"{panel}_{condition}.npz", allow_pickle=False) as d:
                xs.append(d["train_x"])
                ys.append(d["train_median" if median else "train_y"])
                weights.append(d["train_weight"])
                names = d["feature_names"].tolist()
        x, y, weight = np.concatenate(xs), np.concatenate(ys), np.concatenate(weights)
        del xs, ys, weights
        gc.collect()
        members = [("median", False, len(names))] if median else [
            ("full", False, len(names)), ("no_location", True, len(names))]
        if args.feature_set == "enhanced" and condition == "queue_onset" and not median:
            members.append(("base183", False, 183))
        for member, no_location, columns in members:
            model_path = args.output / "models" / tag / f"{condition}_{member}.txt"
            model = None
            if args.reuse_models and model_path.exists() and model_path.with_suffix(".mask.npy").exists():
                from .train import Classifier
                import lightgbm as lgb
                booster = lgb.Booster(model_file=str(model_path))
                mask = np.load(model_path.with_suffix(".mask.npy"))
                expected_names = [n for n, take in zip(names[:columns], mask) if take]
                if (booster.current_iteration() == args.rounds and
                    int(booster.params.get("seed", -1)) == args.seed and
                    int(booster.params.get("num_threads", -1)) == args.threads and
                    booster.feature_name() == expected_names and len(mask) == columns):
                    model = Classifier(booster, mask, threads=args.threads)
                    log(f"reuse {tag} {condition} {member}")
            if model is None:
                log(f"fit {tag} {condition} {member}: {len(y):,} rows × {columns} features")
                model = fit_classifier(x[:, :columns], y, weight, names[:columns],
                no_location=no_location, seed=args.seed, rounds=args.rounds, threads=args.threads,
                model_path=model_path)
            models[(condition, member)] = model
        del x, y, weight
        gc.collect()
    return models


def cv(args):
    models = fit_models(args)
    median_models = fit_models(args, median=True) if args.median_experiment else {}
    details, probability_parts = [], {}
    source = args.output / "datasets" / dataset_tag(args)
    for panel in args.panels:
        network = read_network(args.root, panel)
        prior = np.load(source / f"{panel}_prior.npy")
        for condition in CONDITIONS:
            with np.load(source / f"{panel}_{condition}.npz", allow_pickle=False) as d:
                x, truth, eligible = d["eval_x"], d["eval_y"], d["eval_eligible"]
                median_truth, names = d["eval_median"], d["feature_names"].tolist()
            meta = pd.read_csv(source / f"{panel}_{condition}_eval.csv")
            if len(x) == 0:
                continue
            flat = x.reshape(-1, x.shape[-1])
            full = models[(condition, "full")].predict(flat).reshape(x.shape[:2])
            no_location = models[(condition, "no_location")].predict(flat).reshape(x.shape[:2])
            median_p = median_models[(condition, "median")].predict(flat).reshape(x.shape[:2]) if median_models else None
            base_p = models[(condition, "base183")].predict(flat[:, :183]).reshape(x.shape[:2]) if (condition, "base183") in models else None
            np.savez(args.output / f"cv_probabilities_{experiment_tag(args)}_{panel}_{condition}.npz", full=full,
                     no_location=no_location, truth=truth, eligible=eligible)
            for i, record in enumerate(meta.to_dict("records")):
                onset = condition == "queue_onset"
                n_links = len(network.link_ids)
                idx_now = names.index("queue_now")
                now = x[i, :n_links, idx_now] > .5
                persistence = np.broadcast_to(now, (6, n_links)).copy()
                prior_p = np.zeros((6, n_links))
                tod = int(np.datetime64(record["origin"], "m").astype(int) % 1440 // 5)
                prior_p[-1] = prior[(tod + 6) % 288]
                baseline = decode_probabilities(prior_p, onset=True) if onset else persistence
                p_full, p_no = np.zeros((6, n_links)), np.zeros((6, n_links))
                if onset:
                    p_full[-1], p_no[-1] = full[i], no_location[i]
                else:
                    p_full, p_no = full[i].reshape(6, n_links), no_location[i].reshape(6, n_links)
                candidates = {"persistence": persistence, "onset_prior": baseline,
                              "lgb_full": decode_probabilities(p_full, onset=onset),
                              "lgb_no_location": decode_probabilities(p_no, onset=onset),
                              "blend_50": decode_probabilities((p_full + p_no) / 2, onset=onset)}
                if args.feature_set == "enhanced":
                    p_event = np.zeros((6, n_links))
                    if onset:
                        p_event[-1] = x[i, :, names.index("location_onset_event_prior_target")]
                    candidates["onset_event_prior"] = decode_probabilities(p_event, onset=True) if onset else persistence
                    candidates["onset_lgb_prior_50"] = decode_probabilities((p_full+p_event)/2, onset=True) if onset else candidates["blend_50"]
                    p_base = np.zeros((6, n_links))
                    if onset:
                        p_base[-1] = base_p[i]
                    candidates["lgb_base_onset"] = decode_probabilities(p_base, onset=True) if onset else candidates["blend_50"]
                if not onset:
                    physics = x[i, :, names.index("queue_motion_forecast_inside")].reshape(6, n_links)
                    candidates["physics_motion"] = np.where(np.isfinite(physics), physics > .5, persistence)
                else:
                    candidates["physics_motion"] = candidates["lgb_full"]
                if median_p is not None:
                    p = np.zeros((6, n_links))
                    if onset:
                        p[-1] = median_p[i]
                    else:
                        p = median_p[i].reshape(6, n_links)
                    candidates["lgb_median_labels"] = decode_probabilities(p, onset=onset)
                for name, prediction in candidates.items():
                    details.append({**record, "model": name,
                                    "iou": score_window(prediction, truth[i], eligible[i]),
                                    "iou_median_truth": score_window(prediction, median_truth[i], eligible[i]),
                                    "true_positive_cells": int((truth[i] & eligible[i]).sum()),
                                    "predicted_positive_cells": int(prediction.sum())})
            del x, flat
            gc.collect()
    detail = pd.DataFrame(details)
    primary = detail
    if args.window_policy == "release":
        primary = detail[detail.official_like.isin([True, "True", 1])]
    summary = aggregate_scores(primary)
    dense = aggregate_scores(detail)
    dense.to_csv(args.output / f"cv_scores_dense_{experiment_tag(args)}.csv", index=False)
    detail.to_csv(args.output / "cv_window_scores.csv", index=False)
    summary.to_csv(args.output / "cv_scores.csv", index=False)
    detail.to_csv(args.output / f"cv_window_scores_{experiment_tag(args)}.csv", index=False)
    summary.to_csv(args.output / f"cv_scores_{experiment_tag(args)}.csv", index=False)
    monthly = []
    for month, group in primary.groupby(primary.origin.str[:7]):
        frame = aggregate_scores(group)
        frame["month"] = month
        monthly.append(frame)
    month_frame = pd.concat(monthly)
    month_frame.to_csv(args.output / "cv_monthly_scores.csv", index=False)
    month_frame.to_csv(args.output / f"cv_monthly_scores_{experiment_tag(args)}.csv", index=False)
    overall = summary[summary.panel.eq("OVERALL")]
    log("CV overall\n" + overall.to_string(index=False))
    return choose_models(args, summary)


def choose_models(args, summary=None):
    if summary is None:
        summary = pd.read_csv(args.output / f"cv_scores_{experiment_tag(args)}.csv")
    overall = summary[summary.panel.eq("OVERALL")]
    # Condition-specific selection maximizes the shift-weighted held-out score.
    # The predefined 50/50 blend is selected unless it loses materially.
    learned = overall[overall.model.isin(["lgb_full", "lgb_no_location", "blend_50"])]
    onset_candidates = overall[overall.model.isin(["lgb_full", "lgb_no_location", "blend_50", "onset_prior", "onset_event_prior", "onset_lgb_prior_50", "lgb_median_labels", "lgb_base_onset"])]
    onset_member = str(onset_candidates.sort_values(["onset", "model"], ascending=[False, True]).iloc[0].model)
    ongoing_candidates = overall[overall.model.isin(["lgb_full", "lgb_no_location", "blend_50", "persistence", "physics_motion", "lgb_median_labels"])]
    ongoing_member = str(ongoing_candidates.sort_values(["shift_ongoing", "model"], ascending=[False, True]).iloc[0].model)
    blend = learned[learned.model.eq("blend_50")].iloc[0]
    best = ongoing_candidates[ongoing_candidates.model.eq(ongoing_member)].iloc[0]
    if ongoing_member in ("lgb_full", "lgb_no_location", "blend_50", "lgb_median_labels") and best.shift_ongoing - blend.shift_ongoing < .01:
        ongoing_member = "blend_50"
    choice = dict(onset=onset_member, ongoing=ongoing_member, seed=args.seed,
                  rounds=args.rounds, threads=args.threads, feature_set=args.feature_set, window_policy=args.window_policy,
                  selection="Onset by CV IoU, ongoing by shift-weighted CV; predefined 50/50 blend within .01 of best learned member.")
    dump_json(args.output / "chosen_model.json", choice)
    dump_json(args.output / f"chosen_model_{experiment_tag(args)}.json", choice)
    return choice


def _member_probability(models, condition, member, x, names):
    if member == "lgb_base_onset":
        return models[(condition, "base183")].predict(x[:, :183])
    if member == "lgb_median_labels":
        return models[(condition, "median")].predict(x)
    if member == "onset_prior":
        return x[:, names.index("location_queue_prior_target")]
    if member == "onset_event_prior":
        return x[:, names.index("location_onset_event_prior_target")]
    full = models[(condition, "full")].predict(x)
    if member == "onset_lgb_prior_50":
        return (full + x[:, names.index("location_onset_event_prior_target")]) / 2
    if member == "lgb_full":
        return full
    no_location = models[(condition, "no_location")].predict(x)
    return no_location if member == "lgb_no_location" else (full + no_location) / 2


def predict(args):
    choice_path = args.output / f"chosen_model_{experiment_tag(args)}.json"
    if not choice_path.exists():
        raise FileNotFoundError("Run --stage cv before prediction/model selection")
    choice = json.loads(choice_path.read_text())
    if choice.get("feature_set") != args.feature_set or choice.get("window_policy", "strict") != args.window_policy:
        raise ValueError("Selected configuration and requested feature set differ")
    models = fit_models(args, final=True)
    if "lgb_median_labels" in (choice["onset"], choice["ongoing"]):
        selected = [condition for condition, key in zip(CONDITIONS, ("onset", "ongoing"))
                    if choice[key] == "lgb_median_labels"]
        models.update(fit_models(args, final=True, median=True, conditions=selected))
    outputs, fallbacks = [], []
    dataset_dir = args.output / "datasets" / dataset_tag(args, True)
    for panel in args.panels:
        panel_id = panel_names(args.root).index(panel)
        network = read_network(args.root, panel)
        prior = np.load(dataset_dir / f"{panel}_prior.npy")
        onset_prior = np.load(dataset_dir / f"{panel}_onset_prior.npy") if args.feature_set == "enhanced" else None
        link_index = dict(zip(network.link_ids, range(len(network.link_ids))))
        for split in ("validation", "private"):
            index, history, template = read_window_assets(args.root, panel, split)
            days = set(index.date.astype(str))
            days |= {str(np.datetime64(day) - np.timedelta64(1, "D")) for day in days}
            current = read_panel_arrays(args.root, network, split, args.output / "cache", masked=True, dates=days)
            for window in index.itertuples():
                h = history[history.window_id.eq(window.window_id)]
                arrays = history_arrays(h, window.forecast_origin, network, current)
                x, names = build_features(arrays.speed, arrays.flow, arrays.occupancy, origin=12,
                    timestamps=arrays.timestamps, ramp_on=arrays.ramp_on, location_prior=prior,
                    panel_id=panel_id, **network.feature_kwargs())
                if args.feature_set == "enhanced":
                    from .context import build_day_context
                    origin_idx = int((arrays.timestamps[-1] - current.timestamps[0]) / np.timedelta64(5, "m"))
                    day_context, day_names = build_day_context(current.speed, current.flow, current.timestamps,
                                                               origin=origin_idx, vf=network.vf)
                    x, names = enrich_features(x, names, arrays, network, onset_prior=onset_prior,
                                                day_context=day_context, day_names=day_names)
                onset = window.condition == "queue_onset"
                model_x = x[-len(network.link_ids):] if onset else x
                member = choice["onset" if onset else "ongoing"]
                now = x[:len(network.link_ids), names.index("queue_now")] > .5
                fallback = np.broadcast_to(now, (6, len(network.link_ids)))
                if member == "persistence":
                    decoded = fallback.astype(np.uint8)
                elif member == "physics_motion":
                    motion = x[:, names.index("queue_motion_forecast_inside")].reshape(fallback.shape)
                    decoded = np.where(np.isfinite(motion), motion > .5, fallback).astype(np.uint8)
                else:
                    p = _member_probability(models, window.condition, member, model_x, names)
                    probability = np.zeros(fallback.shape)
                    if onset:
                        probability[-1] = p
                    else:
                        probability = p.reshape(probability.shape)
                    decoded = decode_probabilities(probability, onset=onset)
                out = template[template.window_id.eq(window.window_id)].copy()
                times = pd.to_datetime(out.timestamp, utc=True)
                origin = pd.Timestamp(window.forecast_origin)
                hi = ((times - origin).dt.total_seconds() / 300).to_numpy(dtype=int) - 1
                li = out.link_id.map(link_index).to_numpy(dtype=int)
                if not np.all((hi >= 0) & (hi < 6)):
                    raise ValueError("Template timestamp outside forecast horizon")
                out["queue_pred"] = decoded[hi, li]
                safe = out.copy()
                safe["queue_pred"] = fallback[hi, li].astype(np.uint8)
                outputs.append(out)
                fallbacks.append(safe)
            log(f"predicted {panel} {split}: {len(template):,} rows")
            del current
            gc.collect()
    prediction = pd.concat(outputs, ignore_index=True)
    fallback = pd.concat(fallbacks, ignore_index=True)
    prediction.to_csv(args.output / "queue_submission.csv", index=False)
    fallback.to_csv(args.output / "queue_submission_persistence.csv", index=False)
    verify_submissions(args)


def verify_submissions(args, filenames=None):
    templates = []
    for panel in args.panels:
        for split in ("validation", "private"):
            path = args.root / "task2" / panel / split / "sample_submission_queue.csv"
            templates.append(pd.read_csv(path, usecols=["window_id", "timestamp", "link_id"], dtype=str))
    expected = pd.concat(templates, ignore_index=True)
    keys = ["window_id", "timestamp", "link_id"]
    # Read only required columns, streaming the large common key and template.
    key_parts, sample_ids = [], []
    for chunk in pd.read_csv(args.root / "submission_key.csv", dtype=str, chunksize=250000,
                              usecols=["submission_id", "task", *keys]):
        key_parts.append(chunk[chunk.task.eq("queue")].copy())
    common_key = pd.concat(key_parts, ignore_index=True)
    for chunk in pd.read_csv(args.root / "sample_submission.csv", dtype=str,
                              chunksize=250000, usecols=["submission_id", "task"]):
        sample_ids.extend(chunk.loc[chunk.task.eq("queue"), "submission_id"].tolist())
    if set(sample_ids) != set(common_key.submission_id) or len(sample_ids) != len(common_key):
        raise AssertionError("Queue rows in common template differ from submission_key")
    results = {}
    for filename in (filenames or ("queue_submission.csv", "queue_submission_persistence.csv")):
        path = args.output / filename
        actual = pd.read_csv(path, dtype={k: str for k in keys})
        if actual.duplicated(keys).any() or not actual.queue_pred.isin([0, 1]).all():
            raise AssertionError(f"Invalid queue predictions: {filename}")
        check = expected.merge(actual[keys], on=keys, how="outer", indicator=True, validate="one_to_one")
        if not check._merge.eq("both").all() or len(actual) != len(expected):
            raise AssertionError(f"Per-task template coverage mismatch: {filename}")
        check = common_key.merge(actual, on=keys, how="outer", indicator=True, validate="one_to_one")
        if not check._merge.eq("both").all():
            raise AssertionError(f"Common submission_key coverage mismatch: {filename}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        results[filename] = dict(rows=len(actual), positive_rows=int(actual.queue_pred.sum()),
                                 sha256=digest, complete_per_task=True, complete_common_template=True)
    dump_json(args.output / "submission_verification.json", results)
    log("Task 2 submission coverage, binary values and common-key verification passed")


def write_reports(args):
    summary = pd.read_csv(args.output / f"cv_scores_{experiment_tag(args)}.csv")
    overall = summary[summary.panel.eq("OVERALL")]
    choice = json.loads((args.output / f"chosen_model_{experiment_tag(args)}.json").read_text())
    summary.to_csv(args.output / "cv_scores.csv", index=False)
    dump_json(args.output / "chosen_model.json", choice)
    for prefix in ("cv_window_scores", "cv_monthly_scores"):
        source = args.output / f"{prefix}_{experiment_tag(args)}.csv"
        if not source.exists() and prefix == "cv_monthly_scores":
            detail = pd.read_csv(args.output / f"cv_window_scores_{experiment_tag(args)}.csv")
            if args.window_policy == "release":
                detail = detail[detail.official_like.isin([True, "True", 1])]
            monthly = []
            for month, subset in detail.groupby(detail.origin.str[:7]):
                group = aggregate_scores(subset)
                group["month"] = month
                monthly.append(group)
            pd.concat(monthly).to_csv(source, index=False)
        frame = pd.read_csv(source)
        frame.to_csv(args.output / f"{prefix}.csv", index=False)
    audit_dir = args.output / "datasets" / dataset_tag(args)
    counts = pd.read_csv(audit_dir / "dataset_counts.csv")
    audit = pd.read_csv(audit_dir / "window_audit.csv")
    labels = pd.read_csv(audit_dir / "label_audit.csv")
    with np.load(audit_dir / f"{args.panels[0]}_queue_onset.npz", allow_pickle=False) as first:
        feature_names = first["feature_names"].tolist()
    n_features = len(feature_names)
    n_location = sum(n.startswith("location_") for n in feature_names)
    dense_path = args.output / f"cv_scores_dense_{experiment_tag(args)}.csv"
    dense_summary = pd.read_csv(dense_path) if dense_path.exists() else summary
    dense_overall = dense_summary[dense_summary.panel.eq("OVERALL")]
    protocol_text = ("Primary CV uses five onset and five ongoing windows per panel in each held-out calendar month: "
        "80 onset + 80 ongoing windows across January and February. Training augments quota-limited selections from "
        "overlapping 30-day blocks starting every seven days and at month starts. The union may overlap; each source "
        "block separately obeys the five-per-condition quota and shared six-hour spacing. Dense-union held-out scores "
        "are reported separately and do not determine primary model selection. Selector state uses same-day forward "
        "fill of missing raw speeds, the release-contract established-history condition, and shared spacing. These "
        "extra details improve released-train origin matching from 39/80 to 67/80; hidden underlying labels still "
        "prevent exact replication. The shared spacing is an empirical property of all 240 released windows, "
        "stronger than the within-condition published minimum.") if args.window_policy == "release" else (
        "This initial diagnostic uses the supplied strict T-inclusive predicate, eligibility-gated queue proxies "
        "and independent condition spacing. It has 39/40 onset-origin matches but 0/40 ongoing-origin matches, "
        "and is retained as a negative selector experiment.")
    comparisons = []
    for feature_set in ("base", "enhanced", "base_release", "enhanced_release"):
        score_path = args.output / f"cv_scores_{feature_set}.csv"
        if score_path.exists():
            frame = pd.read_csv(score_path)
            frame = frame[frame.panel.eq("OVERALL")].copy()
            frame.insert(0, "features", feature_set)
            comparisons.append(frame)
    verification = json.loads((args.output / "submission_verification.json").read_text()) if (args.output / "submission_verification.json").exists() else {}
    checks = []
    test_log = args.output / "tests.log"
    if test_log.exists():
        passed = [line.strip() for line in test_log.read_text().splitlines() if " passed" in line]
        if passed:
            checks.append("Scoped tests: " + passed[-1] + ".")
    combined_check = args.output / "make_check_learned.log"
    if combined_check.exists() and combined_check.read_text().strip().endswith("OK"):
        checks.append("The shared full-submission checker passed a temporary baseline carrier with its queue rows replaced by these predictions.")
    reproduction_path = args.output / "reproducibility.json"
    if reproduction_path.exists():
        reproduction = json.loads(reproduction_path.read_text())
        if reproduction.get("cached_full_pipeline_identical"):
            checks.append("The full pipeline with cached feature/model artifacts reproduced both CSVs byte for byte.")
        if reproduction.get("fresh_refit_identical"):
            checks.append("A fresh final-model refit from the same raw-derived datasets also reproduced both CSVs byte for byte.")
    validation_text = " ".join(checks)
    onset_score = float(overall[overall.model.eq(choice["onset"])].onset.iloc[0])
    ongoing_row = overall[overall.model.eq(choice["ongoing"])].iloc[0]
    combined = (onset_score + float(ongoing_row.ongoing)) / 2
    shift_combined = (onset_score + float(ongoing_row.shift_ongoing)) / 2
    def table(frame):
        cols = frame.columns.tolist()
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        for row in frame.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |")
        return "\n".join(lines)
    experiments = f"""# Task 2 experiments

Generated automatically by `python -m tfb.t2.run`. All scores use Jan–Feb 2031 held out from June–December training, raw train-observation threshold proxies and official eligible-cell IoU. Feature/decode decisions never use future eligibility; it enters only supervised training selection and the scorer. Final models refit all 273 train days. The held-out months also guide model selection, so these are development CV estimates.

{protocol_text}

## Models and overall scores

{table(overall[['model','onset','ongoing','total','shift_ongoing','shift_total']])}

Persistence uses the latest causal observation through T, carrying T−5 forward if T is masked. The ordinary occupancy-prior rule predicts only T+30 at historical bottlenecks with expected-IoU decoding. Separate LightGBM onset/ongoing classifiers use {n_features} causal features ({args.feature_set}), total-link FD quantities, calendar, spatial trends, ramps, queue components, head/tail velocities and shockwave forecasts. The no-location member removes {n_location} explicit location/prior columns; static geometry/FD quantities can still weakly identify a link. Enhanced features append masked full-day context, filled-neighbor/slow-link geometry and conditional onset-event priors; each training day’s five-way fold is excluded from its conditional prior. Full-day context uses only the released masked prefix through T, with protected raw window-history observations overlaid in training. `lgb_base_onset` trains an independent onset classifier using only the first 183 columns, allowing the simpler onset member alongside enhanced ongoing models. `onset_event_prior` predicts the conditional event prior alone at T+30; `onset_lgb_prior_50` averages it with the full onset classifier before decoding. `physics_motion` extrapolates empirical queue boundaries for ongoing, using the full onset model. The median experiment changes only training labels, retaining the same windows and raw-proxy evaluation; its median-truth sensitivity appears in the feature-specific `cv_window_scores_*.csv`.

Shift weights are 1+4r, where r is the fraction of current queued observable links with train time-of-day queue frequency below .05. These weights emphasize non-recurrent queues but are not a calibrated incident simulator. Overall aggregation averages conditions, directions and the four Task 2 families equally.

## Per-panel results

{table(summary[summary.panel.ne('OVERALL')][['model','panel','n_onset','n_ongoing','onset','ongoing','total','shift_ongoing']])}

## Dense-union CV diagnostic

{table(dense_overall[['model','n_onset','n_ongoing','onset','ongoing','total','shift_ongoing','shift_total']])}

## Feature-set and window-policy experiments

{table(pd.concat(comparisons)[['features','model','n_onset','n_ongoing','onset','ongoing','total','shift_ongoing','shift_total']])}

The original `base`/`enhanced` runs share the initial dense selector and are paired feature ablations. The `_release` runs change the window protocol and primary sample size; their absolute scores should not be treated as a paired feature-only gain.

## Window reconstruction and training sizes

The actual organizer selector and underlying-state labels are not public. Our deterministic selectors apply the published 70% history coverage, horizon queue and <=.9 persistence IoU safeguards. {protocol_text} This is an empirical policy replication, not a claim of exact hidden-label reconstruction. `first_five_exact_matches` uses a separately capped full-train pass, never union.head(5).

{table(audit)}

{table(counts)}

Negative ongoing training cells are sampled at {args.negative_sample:.3f}, with inverse-probability weights. All eligible positive cells under either label variant are kept. All links/times remain in CV and deployment. Onset trains and predicts T+30 only; its other five horizons are zero. No Task 1 reconstructions enter this pipeline.

## Label denoising

{table(labels)}

`median3` is a centered temporal median only for train target sensitivity. Features always remain causal and use no centered smoothing. Hidden underlying-state labels may disagree with both proxies.

## Determinism and artifacts

Seed={args.seed}, LightGBM deterministic=True, force_col_wise=True, fixed threads={args.threads}, rounds={args.rounds}. Chosen configuration: `{json.dumps(choice, sort_keys=True)}`. Per-window scores, Jan/Feb separate results, feature importances, models and raw-state caches are in `outputs/t2/`. No Kaggle submissions were made.

{validation_text} Evidence is in `tests.log`, `submission_verification.json` and `reproducibility.json`; the independent template/model audit is `see report, Sec. III-C`. A separate transfer experiment in `see report, Sec. III-C` found no compelling improvement from the earlier broad-window models, which were excluded from the final ensemble.
"""
    args.reports.mkdir(parents=True, exist_ok=True)
    (args.reports / "codex_02_task2_experiments.md").write_text(experiments)
    selected_panels = []
    for panel in args.panels:
        a = summary[summary.panel.eq(panel) & summary.model.eq(choice["onset"])].iloc[0]
        b = summary[summary.panel.eq(panel) & summary.model.eq(choice["ongoing"])].iloc[0]
        selected_panels.append(dict(panel=panel, onset=a.onset, ongoing=b.ongoing,
                                    total=(a.onset+b.ongoing)/2, shift_ongoing=b.shift_ongoing))
    status = f"""# Task 2 status

Prepared complete validation/private queue files; **nothing submitted to Kaggle**.

Chosen onset `{choice['onset']}`; ongoing `{choice['ongoing']}`. Jan–Feb development CV: onset **{onset_score:.4f}**, ongoing **{float(ongoing_row.ongoing):.4f}**, S_queue **{combined:.4f}**; shift-weighted S_queue **{shift_combined:.4f}**. Implied queue leaderboard contribution **0.30 × S_queue = {0.30*combined:.4f}** (shift proxy: {0.30*shift_combined:.4f}). These are local noisy-label estimates, not measured leaderboard scores.

{protocol_text}

{table(pd.DataFrame(selected_panels))}

All model CV results:

{table(overall[['model','onset','ongoing','total','shift_ongoing','shift_total']])}

Artifacts: `outputs/t2/queue_submission.csv`, `outputs/t2/queue_submission_persistence.csv`, `outputs/t2/submission_verification.json`. Coverage and hashes: `{json.dumps(verification, sort_keys=True)}`.

{validation_text} Independent artifact checks are in `see report, Sec. III-C`. The broad-window model-transfer experiment did not justify changing the selected ensemble; see `see report, Sec. III-C`.

Rebuild from raw data with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m tfb.t2.run --stage all --feature-set {args.feature_set} --window-policy {args.window_policy} --refresh --median-experiment
```

The pipeline writes only Task 2 owned source/output/report paths. Runtime/environment metadata is in `outputs/t2/run_metadata.json`. `see report, Sec. III-C` records the authoritative scorer and schema audit, including conflicting selector descriptions.

Risks: hidden queues use noise-free underlying speeds, while CV targets noisy released measurements; no released Task 2 truth exists. Selection relies on the disclosed late-horizon onset artifact. Primary quota CV has only 160 windows; Jan/Feb development tuning can overestimate transfer. March/April incident frequency differs from train; the rare-queue weight is a proxy. Exact selector internals are unpublished; even the improved proxy matches only 67/80 train origins. Explicit location removal retains static FD/geometry fingerprints. No validation/private target or future observation was used.

Next ideas, ranked by expected gain:

1. Incident-focused ongoing training, especially D7_I10_W, D7_I210_W and D7_I405_S: generate non-recurrent/boundary-shift examples and tune on separate autumn held-out incidents.
2. Hybrid queue-label repair and onset localization for D7_I10_W, D7_I405_S and D7_I210_W, using FD/flow consistency and spatial neighbors; evaluate against unchanged raw eligible proxies.
3. Calibrate queue-component probabilities and expected-IoU cardinality using a disjoint month; cell independence is approximate.
4. Ensemble onset seeds with bottleneck-specific discharge/demand features, gating gains on two temporal holdouts.
5. Learn queue-tail/head motion residuals around recurrent bottlenecks and blend a geometry-only member for incident robustness.
"""
    (args.reports / "codex_STATUS.md").write_text(status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=Path(os.environ.get("TFB_RELEASE", str(PROJECT / "data/kaggle_public"))))
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/t2")
    parser.add_argument("--reports", type=Path, default=PROJECT / "reports")
    parser.add_argument("--stage", choices=["all", "prepare", "cv", "select", "prepare-final", "fit-final", "predict", "verify", "report"], default="all")
    parser.add_argument("--panel", action="append")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=350)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--negative-sample", type=float, default=.25)
    parser.add_argument("--median-experiment", action="store_true", default=True)
    parser.add_argument("--no-median-experiment", action="store_false", dest="median_experiment")
    parser.add_argument("--feature-set", choices=["base", "enhanced"], default="enhanced")
    parser.add_argument("--window-policy", choices=["strict", "release"], default="release")
    parser.add_argument("--onset-prior-folds", type=int, default=5)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--reuse-models", action="store_true", help="Reuse cached classifiers when seed, tree count and feature schema match")
    args = parser.parse_args()
    args.root = resolve_root(args.release_root)
    args.panels = args.panel or panel_names(args.root)
    if not 0 < args.negative_sample <= 1:
        parser.error("--negative-sample must be in (0,1]")
    if args.onset_prior_folds < 2:
        parser.error("--onset-prior-folds must be at least 2")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if args.stage in ("all", "prepare"):
        prepare(args)
    if args.stage in ("all", "cv"):
        cv(args)
    if args.stage == "select":
        choose_models(args)
    if args.stage in ("all", "prepare-final"):
        prepare(args, final=True)
    if args.stage == "fit-final":
        fit_models(args, final=True)
    if args.stage in ("all", "predict"):
        predict(args)
    if args.stage == "verify":
        verify_submissions(args)
    if args.stage in ("all", "report"):
        write_reports(args)
    import lightgbm, pyarrow
    metadata = dict(stage=args.stage, elapsed_seconds=time.monotonic()-started,
        python=platform.python_version(), platform=platform.platform(), numpy=np.__version__,
        pandas=pd.__version__, lightgbm=lightgbm.__version__, pyarrow=pyarrow.__version__,
        seed=args.seed, rounds=args.rounds, threads=args.threads, panels=args.panels,
        feature_set=args.feature_set, window_policy=args.window_policy,
        refresh=args.refresh, reuse_models=args.reuse_models,
        finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    runtime_dir = args.output / "runtime"
    dump_json(runtime_dir / f"{args.stage}_{experiment_tag(args)}.json", metadata)
    metadata["stage_metadata"] = {p.stem: json.loads(p.read_text()) for p in sorted(runtime_dir.glob("*.json"))}
    dump_json(args.output / "run_metadata.json", metadata)
    log(f"completed {args.stage} in {time.monotonic()-started:.1f}s")


if __name__ == "__main__":
    main()
