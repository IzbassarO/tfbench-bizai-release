"""Causal recurrence-shift diagnostics; these are not incident ground truth.

No target, future eligibility, incident schedule or future observation enters
``window_shift_metrics``. Score transport uses local CV labels only after its
causal strata have been frozen. It retains the competition's equal family,
direction and condition weights, and exposes missing stratum support instead
of silently substituting an optimistic score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


CONDITIONS = ("queue_onset", "queue_ongoing")


def window_shift_metrics(
    x, names, prior, *, origin, n_links=None, rare_threshold=0.05,
    rare_fraction_threshold=0.5, max_observation_age=15.0,
) -> dict:
    """Measure causal queue rarity and geometry from one feature matrix.

    ``x`` is link-major within each horizon, either one or six horizons. Only
    its FIRST horizon's current-state columns are used. ``prior`` is an
    earlier-training queue-frequency table [daily_bin,link] or
    [weekday/weekend,daily_bin,link]. Use the same reference table across source
    and target populations when interpreting population change.

    Nonrecurrent means at least half of recently observed queued links have
    historical queue frequency below .05. It is a predefined rarity proxy,
    not an incident detector. Missing/stale evidence is a distinct stratum.
    Onset windows with no visible queue cannot be assigned a future incident
    identity without leaking their future location; they remain queue-free.
    """
    values = np.asarray(x)
    p = np.asarray(prior, dtype=float)
    if p.ndim not in (2, 3) or p.shape[-2] == 0 or (p.ndim == 3 and p.shape[0] != 2):
        raise ValueError("prior must be daily-bin x link, optionally by two day types")
    n = p.shape[-1] if n_links is None else int(n_links)
    if n <= 0 or n != p.shape[-1] or values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("feature schema and prior link count must match")
    if values.shape[0] not in (n, 6 * n) or len(set(names)) != len(names):
        raise ValueError("features must contain one or six ordered horizons with unique names")
    if not 0 <= rare_threshold <= 1 or not 0 <= rare_fraction_threshold <= 1 or max_observation_age < 0:
        raise ValueError("invalid recurrence thresholds")
    lookup = {name: i for i, name in enumerate(names)}

    def column(name, default=np.nan):
        return np.asarray(values[:n, lookup[name]], float) if name in lookup else np.full(n, default)

    # This whitelist deliberately excludes forecast feature columns, labels,
    # future quality and posterior predictions.
    last_speed = column("speed_ratio_last_observed")
    raw_speed = column("speed_ratio_lag0")
    age = column("speed_observation_age_minutes")
    if "speed_ratio_last_observed" not in lookup:
        last_speed = raw_speed.copy()
    if "speed_observation_age_minutes" not in lookup:
        age = np.where(np.isfinite(raw_speed), 0.0, np.nan)
    known = np.isfinite(last_speed) & np.isfinite(age) & (age <= max_observation_age)
    inferred = np.isfinite(last_speed) & (last_speed <= .6)
    queued = known & inferred
    observed_queue = np.isfinite(raw_speed) & (raw_speed <= .6)
    # Normalize aware timestamps explicitly: NumPy's implicit timezone parsing
    # is deprecated, and calendar/prior bins must refer to the UTC instant.
    stamp = pd.Timestamp(origin)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    epoch = stamp.to_datetime64().astype("datetime64[m]")
    if np.isnat(epoch):
        raise ValueError("origin must be a finite timestamp")
    minute = int(epoch.astype(int) % 1440)
    daytype = int((epoch.astype(int) // 1440 + 3) % 7 >= 5)
    daily = p[daytype] if p.ndim == 3 else p
    qprior = daily[minute * len(daily) // 1440]
    all_time = p.reshape(-1, n)
    peak_prior = np.max(np.where(np.isfinite(all_time), all_time, -np.inf), axis=0)
    peak_prior[~np.isfinite(all_time).any(axis=0)] = np.nan
    prior_known = queued & np.isfinite(qprior)
    rare_fraction = float((qprior[prior_known] < rare_threshold).mean()) if prior_known.any() else np.nan
    if queued.any() and prior_known.sum() == queued.sum():
        stratum = "nonrecurrent" if rare_fraction >= rare_fraction_threshold else "recurrent"
    elif queued.any() or known.mean() < .5 or inferred.any():
        stratum = "unknown"
    else:
        stratum = "no_recent_queue"
    padded = np.r_[False, queued, False].astype(np.int8)
    starts, ends = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1) - 1
    locations = column("location_distance_km")
    at = np.flatnonzero(queued)
    speeds = last_speed[queued]
    ages = column("day_queue_run_minutes")[queued]
    finite_ages = ages[np.isfinite(ages)]
    slow = known & (last_speed <= .8)
    place_known = queued & np.isfinite(peak_prior)
    place_fraction = float((peak_prior[place_known] < rare_threshold).mean()) if place_known.any() else np.nan
    timing_known = prior_known & np.isfinite(peak_prior)
    spatial_stratum = ("nonrecurrent" if place_fraction >= rare_fraction_threshold else "recurrent") if queued.any() and place_known.sum() == queued.sum() else ("unknown" if stratum == "unknown" else "no_recent_queue")
    joint_stratum = stratum if stratum in ("unknown", "no_recent_queue") else (
        "novel_place" if spatial_stratum == "nonrecurrent" else "unusual_time" if stratum == "nonrecurrent" else "recurrent")
    result = {
        "recurrence_stratum": stratum,
        "n_links": n,
        "queue_recent_count": int(queued.sum()),
        "queue_observed_T_count": int(observed_queue.sum()),
        "queue_inferred_count": int(inferred.sum()),
        "queue_stale_count": int((inferred & ~known).sum()),
        "known_recent_fraction": float(known.mean()),
        "observed_T_fraction": float(np.isfinite(raw_speed).mean()),
        "rare_queue_fraction": rare_fraction,
        "rare_place_fraction": place_fraction,
        "unusual_time_fraction": float(((qprior[timing_known] < rare_threshold) &
            (peak_prior[timing_known] >= rare_threshold)).mean()) if timing_known.any() else np.nan,
        "spatial_stratum": spatial_stratum,
        "joint_stratum": joint_stratum,
        "mean_queue_prior": float(qprior[prior_known].mean()) if prior_known.any() else np.nan,
        "queue_components": len(starts),
        "largest_component_links": int(np.max(ends - starts + 1)) if len(starts) else 0,
        "queue_span_links": int(at[-1] - at[0] + 1) if len(at) else 0,
        "queue_span_km": float(locations[at[-1]] - locations[at[0]]) if len(at) else 0.0,
        "queue_first_link_index": int(at[0]) if len(at) else np.nan,
        "queue_last_link_index": int(at[-1]) if len(at) else np.nan,
        "queue_centroid_link_index": float(at.mean()) if len(at) else np.nan,
        "queue_centroid_km": float(locations[at].mean()) if len(at) else np.nan,
        "minimum_queue_speed_ratio": float(speeds.min()) if len(speeds) else np.nan,
        "median_queue_age_minutes": float(np.median(finite_ages)) if len(finite_ages) else np.nan,
        "oldest_queue_age_minutes": float(finite_ages.max()) if len(finite_ages) else np.nan,
        "slow80_count": int(slow.sum()),
        "slow80_rare_fraction": float((qprior[slow & np.isfinite(qprior)] < rare_threshold).mean()) if (slow & np.isfinite(qprior)).any() else np.nan,
    }
    for name in ("day_corridor_queue_to_peak_ratio", "day_corridor_queue_change30_count",
                 "day_corridor_minutes_since_peak", "global_speed_ratio_slope15"):
        data = column(name)
        result[name] = float(data[0])
    return result


def causal_nonrecurrent_score(x, names, prior, *, origin, unknown_value=np.nan, **kwargs):
    """Portable [0,1] rarity score for an ongoing model's causal gate.

    This is the fraction of recently observed queued links with rare train
    queue frequency. A well-observed queue-free prefix returns zero; stale or
    insufficient evidence returns ``unknown_value``. It is not a calibrated
    probability of an incident. The caller owns a date-excluded fitted prior.
    """
    metrics = window_shift_metrics(x, names, prior, origin=origin, **kwargs)
    if metrics["recurrence_stratum"] == "unknown":
        return float(unknown_value)
    return float(metrics["rare_queue_fraction"]) if metrics["queue_recent_count"] else 0.0


def stratum_proportions(metrics):
    """Target/source panel-condition proportions, retaining zero-count strata."""
    rows = []
    labels = ("recurrent", "nonrecurrent", "no_recent_queue", "unknown")
    if metrics.duplicated(["panel", "condition", "window_id"]).any():
        raise ValueError("proportions require unique windows")
    for (panel, condition), group in metrics.groupby(["panel", "condition"], sort=True):
        for label in labels:
            count = int(group.recurrence_stratum.eq(label).sum())
            rows.append(dict(panel=panel, family=panel.rsplit("_", 1)[0], condition=condition,
                recurrence_stratum=label, n=count, total=len(group), proportion=count / len(group)))
    return pd.DataFrame(rows)


def _official_weights(target, conditions):
    panels = target[["panel"]].drop_duplicates().copy()
    panels["family"] = panels.panel.str.rsplit("_", n=1).str[0]
    family_sizes = panels.groupby("family").size()
    n_families = len(family_sizes)
    return {row.panel: 1.0 / (n_families * family_sizes[row.family] * len(conditions))
            for row in panels.itertuples()}


def mixture_weighted_scores(
    detail, target_metrics, *, stratum="recurrence_stratum", score="iou",
    conditions=CONDITIONS, fallback_pool=None,
) -> dict[str, pd.DataFrame]:
    """Transport CV stratum errors to a target causal population.

    Required source columns are model,panel,condition,window_id,stratum,score;
    target rows have no scores, just panel,condition,window_id,stratum. Target
    proportions are panel/condition-specific. Unsupported strata give bounds
    [0,1], a NaN point estimate and explicit unsupported mass. They are never
    dropped from the denominator. ``fallback_pool='condition'`` or
    ``'family_condition'`` permits borrowing CV stratum means, under the strong
    stated assumption that those conditional errors transfer across panels.

    Outputs: stratum support/means, panel-condition estimates and hierarchical
    aggregate rows. Conditions/directions/families retain equal official
    weights; window or cell counts cannot change those top-level weights.
    """
    if fallback_pool not in (None, "family_condition", "condition"):
        raise ValueError("fallback_pool must be None, family_condition or condition")
    source, target = detail.copy(), target_metrics.copy()
    required = {"model", "panel", "condition", "window_id", stratum, score}
    if not required <= set(source) or not {"panel", "condition", "window_id", stratum} <= set(target):
        raise ValueError("source/target lack required score or causal stratum metadata")
    source = source[source.condition.isin(conditions)]
    target = target[target.condition.isin(conditions)]
    if source.duplicated(["model", "panel", "condition", "window_id"]).any() or target.duplicated(["panel", "condition", "window_id"]).any():
        raise ValueError("window keys must be unique within each model")
    if target.empty or source.empty or not np.isfinite(source[score]).all() or not source[score].between(0, 1).all():
        raise ValueError("nonempty populations and finite [0,1] source scores are required")
    if source[stratum].isna().any() or target[stratum].isna().any():
        raise ValueError("missing recurrence evidence must be an explicit unknown stratum")
    weights = _official_weights(target, conditions)
    strata_rows, pc_rows = [], []
    for model, by_model in source.groupby("model", sort=True):
        for panel, weight in weights.items():
            family = panel.rsplit("_", 1)[0]
            for condition in conditions:
                src = by_model[by_model.panel.eq(panel) & by_model.condition.eq(condition)]
                tgt = target[target.panel.eq(panel) & target.condition.eq(condition)]
                if src.empty or tgt.empty:
                    raise ValueError(f"missing panel/condition group {model}/{panel}/{condition}")
                lower, support, borrowed, variance, tv = 0., 0., 0., 0., 0.
                for label in sorted(set(src[stratum]) | set(tgt[stratum])):
                    local = src[src[stratum].eq(label)]
                    n_target = int(tgt[stratum].eq(label).sum())
                    probability = n_target / len(tgt)
                    original_n = len(local)
                    pool_name = "local"
                    if local.empty and probability > 0 and fallback_pool is not None:
                        local = by_model[by_model.condition.eq(condition) & by_model[stratum].eq(label)]
                        if fallback_pool == "family_condition":
                            local = local[local.panel.str.rsplit("_", n=1).str[0].eq(family)]
                        pool_name = fallback_pool if len(local) else "unsupported"
                    elif local.empty:
                        pool_name = "unsupported"
                    mean = float(local[score].mean()) if len(local) else np.nan
                    if probability > 0 and len(local):
                        lower += probability * mean
                        support += probability
                        borrowed += probability if original_n == 0 else 0.
                        variance += probability ** 2 * (float(local[score].var(ddof=1)) / len(local) if len(local) > 1 else np.nan)
                    tv += .5 * abs(probability - original_n / len(src))
                    strata_rows.append(dict(model=model, panel=panel, family=family, condition=condition,
                        **{stratum:label}, n_cv=original_n, n_target=n_target,
                        target_probability=probability, mean_iou=mean, pool=pool_name,
                        n_pool=len(local)))
                pc_rows.append(dict(model=model, panel=panel, family=family, condition=condition,
                    weight=weight, n_cv=len(src), n_target=len(tgt), source_score=float(src[score].mean()),
                    transport_score=lower if support >= 1 - 1e-12 else np.nan,
                    supported_transport_score=lower / support if support else np.nan,
                    lower_bound=lower, upper_bound=lower + max(0., 1 - support),
                    supported_mass=support, borrowed_mass=borrowed, composition_tv_bound=tv,
                    source_mean_se_only=float(np.sqrt(variance)) if np.isfinite(variance) else np.nan))
    pc = pd.DataFrame(pc_rows)
    aggregate = []

    def aggregate_row(group, model, level, panel="ALL", family="ALL", condition="ALL"):
        w = group.weight.to_numpy(dtype=float, copy=True)
        w /= w.sum()
        lower = float(np.dot(w, group.lower_bound))
        support = float(np.dot(w, group.supported_mass))
        aggregate.append(dict(model=model, level=level, panel=panel, family=family, condition=condition,
            source_score=float(np.dot(w, group.source_score)),
            transport_score=lower if support >= 1 - 1e-12 else np.nan,
            supported_transport_score=lower / support if support else np.nan,
            lower_bound=lower, upper_bound=float(np.dot(w, group.upper_bound)),
            supported_mass=support, borrowed_mass=float(np.dot(w, group.borrowed_mass)),
            composition_tv_bound=float(np.dot(w, group.composition_tv_bound)),
            n_cv=int(group.n_cv.sum()), n_target=int(group.n_target.sum())))

    for model, group in pc.groupby("model", sort=True):
        aggregate_row(group, model, "overall")
        for condition, subset in group.groupby("condition", sort=True):
            aggregate_row(subset, model, "condition", condition=condition)
        for family, subset in group.groupby("family", sort=True):
            aggregate_row(subset, model, "family", family=family)
        for panel, subset in group.groupby("panel", sort=True):
            aggregate_row(subset, model, "panel", panel=panel, family=panel.rsplit("_", 1)[0])
    return {"strata": pd.DataFrame(strata_rows), "panel_condition": pc,
            "aggregate": pd.DataFrame(aggregate)}


def npz_feature_prefix(path, indices, n_links):
    """Read only first-horizon feature rows from uncompressed cached NPZ.

    Largest existing eval arrays exceed 200 MB. Streaming selected prefixes
    keeps this read-only diagnostic to a few MB while other agents fit models.
    """
    with zipfile.ZipFile(path) as archive, archive.open("eval_x.npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"unsupported cached feature NPY version {version}")
        if len(shape) != 3 or fortran or dtype.hasobject or shape[1] < n_links:
            raise ValueError("expected C-ordered window x cell x feature numeric cache")
        start, row_bytes = stream.tell(), shape[1] * shape[2] * dtype.itemsize
        length = n_links * shape[2] * dtype.itemsize
        output = []
        for index in indices:
            if not 0 <= index < shape[0]:
                raise ValueError("feature row index outside cached array")
            stream.seek(start + int(index) * row_bytes)
            data = stream.read(length)
            if len(data) != length:
                raise ValueError("truncated feature cache")
            output.append(np.frombuffer(data, dtype=dtype).reshape(n_links, shape[2]).copy())
        return output


def score_probability_packs(directory, cv_metrics):
    """Score root's all-panel fixed predictions, using causal metadata strata."""
    from .train import score_window
    keys = ["window_id", "panel", "condition"]
    metrics = cv_metrics.set_index(keys)
    rows = []
    for path in sorted(Path(directory).glob("*.npz")):
        meta = pd.read_csv(path.with_suffix(".csv"))
        with np.load(path, allow_pickle=False) as data:
            if not np.array_equal(data["window_id"], meta.window_id.to_numpy(dtype=str)) or not np.array_equal(data["origin"], meta.origin.to_numpy(dtype=str)):
                raise ValueError(f"prediction metadata alignment mismatch: {path}")
            keep = [i for i, r in enumerate(meta.itertuples()) if (r.window_id, r.panel, r.condition) in metrics.index]
            truth, eligible = data["truth"], data["eligible"]
            for model in data["model_names"].astype(str):
                pred = data[f"prediction_{model}"]
                for i in keep:
                    record = meta.iloc[i].to_dict()
                    metric = metrics.loc[tuple(record[k] for k in keys)]
                    rows.append({**record, "model": model,
                        "recurrence_stratum": metric.recurrence_stratum,
                        "spatial_stratum": metric.spatial_stratum,
                        "joint_stratum": metric.joint_stratum,
                        "iou": score_window(pred[i], truth[i], eligible[i])})
    if not rows:
        raise ValueError("no complete all-panel prediction packs found")
    return pd.DataFrame(rows)


def stratified_scores(detail):
    """Conditional CV scores under the original official sampling weights."""
    rows = []
    for model, group in detail.groupby("model", sort=True):
        weights = _official_weights(group, CONDITIONS)
        counts = group.groupby(["panel", "condition"]).size()
        group = group.copy()
        group["official_window_weight"] = [weights[r.panel] / counts.loc[(r.panel, r.condition)] for r in group.itertuples()]
        for (condition, label), subset in group.groupby(["condition", "recurrence_stratum"], sort=True):
            w = subset.official_window_weight.to_numpy(float)
            rows.append(dict(model=model, condition=condition, recurrence_stratum=label,
                n_windows=len(subset), n_panels=subset.panel.nunique(),
                iou=float(np.average(subset.iou, weights=w)),
                empirical_sd=float(subset.iou.std(ddof=1)) if len(subset) > 1 else np.nan,
                population_weight=float(w.sum())))
    return pd.DataFrame(rows)


def label_mix_transports(label_scores, cv_metrics, target_metrics, *, stratum="recurrence_stratum"):
    """Joint proxy-label and composition sensitivity, computed directly.

    Each label method supplies its OWN conditional CV error means. These means
    are transported using the same frozen causal strata and target proportions;
    a separate label delta is never added to a separate population-shift delta.
    """
    keys = ["window_id", "panel", "condition"]
    source = label_scores.copy()
    if "official_like" in source:
        source = source[source.official_like.astype(str).str.lower().eq("true")]
    if stratum in source:
        source = source.drop(columns=stratum)
    source = source.merge(cv_metrics[[*keys, stratum]], on=keys, how="left", validate="many_to_one")
    results = {name: [] for name in ("strata", "panel_condition", "aggregate")}
    for method, group in source.groupby("label_method", sort=True):
        tables = mixture_weighted_scores(group, target_metrics, stratum=stratum)
        for name, table in tables.items():
            table.insert(0, "label_method", method)
            results[name].append(table)
    return {name: pd.concat(parts, ignore_index=True) for name, parts in results.items()}


def bootstrap_transport_bounds(detail, target, *, model, n_bootstrap=1000, seed=20261024):
    """Window bootstrap within panel/condition; returns a sensitivity range.

    This treats windows as exchangeable within their groups. It cannot account
    for shared incidents, selector bias or proxy-label bias, and therefore is
    not a calibrated confidence interval for leaderboard performance.
    """
    source = detail[detail.model.eq(model)]
    weights = _official_weights(target, CONDITIONS)
    prepared = []
    for panel, weight in weights.items():
        for condition in CONDITIONS:
            s = source[source.panel.eq(panel) & source.condition.eq(condition)]
            t = target[target.panel.eq(panel) & target.condition.eq(condition)]
            if s.empty or t.empty:
                raise ValueError("bootstrap needs all panel/condition groups")
            prepared.append((s.recurrence_stratum.to_numpy(), s.iou.to_numpy(float),
                             t.recurrence_stratum.to_numpy(), weight))
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_bootstrap):
        original, lower, upper, unsupported = 0., 0., 0., 0.
        for s, y, t, weight in prepared:
            si = rng.integers(0, len(s), len(s))
            ti = rng.integers(0, len(t), len(t))
            sb, yb, tb = s[si], y[si], t[ti]
            original += weight * yb.mean()
            for label in np.unique(tb):
                probability = np.mean(tb == label)
                ys = yb[sb == label]
                if len(ys):
                    lower += weight * probability * ys.mean()
                    upper += weight * probability * ys.mean()
                else:
                    upper += weight * probability
                    unsupported += weight * probability
        rows.append(dict(model=model, iteration=b, source_score=original,
                         lower_bound=lower, upper_bound=upper, unsupported_mass=unsupported))
    return pd.DataFrame(rows)


def collect_metrics(root, output, *, target_split="validation"):
    from .context import build_day_context
    from .features import build_features
    from .io import history_arrays, panel_names, read_network, read_panel_arrays, read_window_assets

    project_output = output.parents[1]
    source = project_output / "datasets/cv_enhanced_release"
    final_source = project_output / "datasets/final_enhanced_release"
    cv_rows, target_rows = [], []
    for panel in panel_names(root):
        network = read_network(root, panel)
        prior = np.load(source / f"{panel}_prior.npy")
        deploy_prior = np.load(final_source / f"{panel}_prior.npy")
        for condition in CONDITIONS:
            meta = pd.read_csv(source / f"{panel}_{condition}_eval.csv")
            indices = np.flatnonzero(meta.official_like.astype(str).str.lower().eq("true"))
            path = source / f"{panel}_{condition}.npz"
            with np.load(path) as data:
                names = data["feature_names"].tolist()
            matrices = npz_feature_prefix(path, indices, len(network.link_ids))
            for i, x in zip(indices, matrices):
                record = meta.iloc[i]
                cv_rows.append({"split": "cv", "window_id": record.window_id, "panel": panel,
                    "condition": condition, "origin": record.origin,
                    **window_shift_metrics(x, names, prior, origin=record.origin)})
        index, history, _ = read_window_assets(root, panel, target_split)
        days = set(index.date.astype(str))
        days |= {str(np.datetime64(day) - np.timedelta64(1, "D")) for day in days}
        current = read_panel_arrays(root, network, target_split, output / "cache", masked=True, dates=days)
        for row in index.itertuples():
            local = history_arrays(history[history.window_id.eq(row.window_id)], row.forecast_origin, network, current)
            x, names = build_features(local.speed, local.flow, local.occupancy, origin=12,
                timestamps=local.timestamps, ramp_on=local.ramp_on, location_prior=prior,
                panel_id=0, **network.feature_kwargs())
            at = int((local.timestamps[-1] - current.timestamps[0]) / np.timedelta64(5, "m"))
            day, day_names = build_day_context(current.speed, current.flow, current.timestamps, origin=at, vf=network.vf)
            x, names = np.column_stack([x, np.tile(day, (6, 1))]), names + day_names
            reference = window_shift_metrics(x, names, prior, origin=local.timestamps[-1])
            deployed = window_shift_metrics(x, names, deploy_prior, origin=local.timestamps[-1])
            target_rows.append({"split": target_split, "window_id": row.window_id, "panel": panel,
                "condition": row.condition, "origin": str(local.timestamps[-1]), **reference,
                "deploy_prior_rare_queue_fraction": deployed["rare_queue_fraction"],
                "deploy_prior_recurrence_stratum": deployed["recurrence_stratum"]})
        del current
    return pd.DataFrame(cv_rows), pd.DataFrame(target_rows)


def main():
    project = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project / "data/kaggle_public")
    parser.add_argument("--output", type=Path, default=project / "outputs/t2/v3/incidents")
    parser.add_argument("--target-split", choices=["validation", "private"], default="validation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cv_metrics, target_metrics = collect_metrics(args.root, args.output, target_split=args.target_split)
    cv_metrics.to_csv(args.output / "cv_metrics.csv", index=False)
    target_metrics.to_csv(args.output / f"{args.target_split}_metrics.csv", index=False)
    stratum_proportions(cv_metrics).to_csv(args.output / "cv_stratum_proportions.csv", index=False)
    stratum_proportions(target_metrics).to_csv(args.output / "target_stratum_proportions.csv", index=False)
    project_output = args.output.parents[1]
    detail = pd.read_csv(project_output / "cv_window_scores_enhanced_release.csv")
    detail = detail[detail.official_like.astype(str).str.lower().eq("true")].copy()
    keys = ["window_id", "panel", "condition"]
    extra = cv_metrics[[*keys, "recurrence_stratum", "spatial_stratum", "joint_stratum"]]
    detail = detail.merge(extra, on=keys, validate="many_to_one", how="left")
    choice = json.loads((project_output / "chosen_model_enhanced_release.json").read_text())
    frozen = detail[((detail.condition == "queue_onset") & (detail.model == choice["onset"])) |
                    ((detail.condition == "queue_ongoing") & (detail.model == choice["ongoing"]))].copy()
    frozen["model"] = "frozen_v1"
    detail = pd.concat([detail, frozen], ignore_index=True)
    detail.to_csv(args.output / "cv_stratified_detail.csv", index=False)
    for pool in (None, "family_condition", "condition"):
        tables = mixture_weighted_scores(detail, target_metrics, fallback_pool=pool)
        tag = pool or "strict"
        for name, table in tables.items():
            table.to_csv(args.output / f"transport_{tag}_{name}.csv", index=False)
    pack_dir = project_output / "v3/eval/janfeb"
    if len(list(pack_dir.glob("*.npz"))) == 16:
        packed = score_probability_packs(pack_dir, cv_metrics)
        packed.to_csv(args.output / "pack_stratified_detail.csv", index=False)
        stratified_scores(packed).to_csv(args.output / "pack_stratified_scores.csv", index=False)
        for pool in (None, "family_condition", "condition"):
            tables = mixture_weighted_scores(packed, target_metrics, fallback_pool=pool)
            tag = pool or "strict"
            for name, table in tables.items():
                table.to_csv(args.output / f"pack_transport_{tag}_{name}.csv", index=False)
        joint = mixture_weighted_scores(packed, target_metrics, stratum="joint_stratum")
        for name, table in joint.items():
            table.to_csv(args.output / f"pack_transport_joint_{name}.csv", index=False)
        boot = pd.concat([bootstrap_transport_bounds(packed, target_metrics, model=model) for model in ("v1", "v2")], ignore_index=True)
        boot.to_csv(args.output / "bootstrap_transport_bounds.csv", index=False)
    stratified_scores(detail).to_csv(args.output / "cv_stratified_scores.csv", index=False)
    labels_path = project_output / "v3/labels/window_scores.csv"
    if labels_path.exists():
        labels = pd.read_csv(labels_path)
        for stratum, tag in (("recurrence_stratum", "strict"), ("joint_stratum", "joint")):
            for name, table in label_mix_transports(labels, cv_metrics, target_metrics, stratum=stratum).items():
                table.to_csv(args.output / f"label_mix_transport_{tag}_{name}.csv", index=False)
    manifest = dict(target_split=args.target_split, reference_prior="June–December 2030, fixed across populations",
        causal_stratum="recent <=15min observed queued links; nonrecurrent if >=50% have prior<.05",
        n_cv=len(cv_metrics), n_target=len(target_metrics), incident_ground_truth_available=False,
        future_observations_used=False, target_labels_used=False)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
