"""Causal, deterministic features for Task 2 queue forecasting.

Rows are horizon-major: all links at T+5, then all links at T+10, through
T+30. Links must be ordered upstream to downstream. Every dynamic input is
sliced to [T-60, T] before computing any feature; there is no interpolation
from future rows. Missing values are retained for LightGBM, with observation
indicators and causal last-observation features supplied separately.

``capacity`` and ``kcrit`` are TOTAL link values, as in the released network
CSV files. Flow is total veh/h, speed km/h, density total veh/km, and distance
km. Historical location priors must be fitted by the caller on training data.
"""
from __future__ import annotations

import numpy as np


NO_LOCATION_FEATURE_PREFIXES = ("location_",)


def location_feature_mask(feature_names: list[str]) -> np.ndarray:
    """True for features removed from the model without location priors."""
    return np.asarray(
        [name.startswith(NO_LOCATION_FEATURE_PREFIXES) for name in feature_names],
        dtype=bool,
    )


def _mean(values: np.ndarray, axis: int = 0) -> np.ndarray:
    valid = np.isfinite(values)
    count = np.sum(valid, axis=axis)
    total = np.sum(np.where(valid, values, 0.0), axis=axis)
    return np.divide(total, count, out=np.full(np.shape(total), np.nan), where=count > 0)


def _extreme(values: np.ndarray, which: str) -> np.ndarray:
    valid = np.isfinite(values)
    if which == "min":
        out = np.min(np.where(valid, values, np.inf), axis=0)
    else:
        out = np.max(np.where(valid, values, -np.inf), axis=0)
    return np.where(np.any(valid, axis=0), out, np.nan)


def _std(values: np.ndarray) -> np.ndarray:
    center = _mean(values)
    return np.sqrt(_mean((values - center[None, :]) ** 2))


def _fill_forward(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.where(np.isfinite(values), np.arange(len(values))[:, None], -1)
    indices = np.maximum.accumulate(indices, axis=0)
    filled = values[np.maximum(indices, 0), np.arange(values.shape[1])[None, :]]
    filled[indices < 0] = np.nan
    return filled, indices


def _shift(values: np.ndarray, offset: int) -> np.ndarray:
    """At link i return link i+offset, without wrapping corridor boundaries."""
    out = np.full_like(values, np.nan, dtype=np.float64)
    if abs(offset) >= len(values):
        return out
    if offset > 0:
        out[:-offset] = values[offset:]
    elif offset < 0:
        out[-offset:] = values[:offset]
    else:
        out[:] = values
    return out


def _slope(values: np.ndarray, minutes: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    mean_t = np.divide(
        np.sum(np.where(valid, minutes[:, None], 0), axis=0),
        count,
        out=np.full(values.shape[1], np.nan),
        where=count > 0,
    )
    centered_t = minutes[:, None] - mean_t
    numerator = np.sum(np.where(valid, centered_t * values, 0), axis=0)
    denominator = np.sum(np.where(valid, centered_t ** 2, 0), axis=0)
    return np.divide(
        numerator, denominator, out=np.full(values.shape[1], np.nan),
        where=(count >= 2) & (denominator > 0),
    )


def _segments(queue: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.r_[False, queue, False].astype(np.int8)
    changes = np.diff(padded)
    return np.flatnonzero(changes == 1), np.flatnonzero(changes == -1) - 1


def _boundary_features(
    speed: np.ndarray, flow: np.ndarray, vf: np.ndarray,
    distance: np.ndarray, minutes: np.ndarray,
) -> dict[str, np.ndarray]:
    n = len(distance)
    queued = np.isfinite(speed) & (speed <= 0.6 * vf[None, :])
    now = queued[-1]
    tails, heads = _segments(now)
    result = {
        "queue_now": now.astype(float),
        "queue_count": np.full(n, now.sum(), dtype=float),
        "queue_component_count": np.full(n, len(tails), dtype=float),
        "queue_upstream_count": np.cumsum(now) - now,
        "queue_downstream_count": np.cumsum(now[::-1])[::-1] - now,
    }
    for count, label in ((4, "15"), (7, "30"), (13, "60")):
        result[f"queue_fraction_{label}"] = _mean(queued[-count:].astype(float))
    # Duration of the current observed queue run, capped by available history.
    queue_run = np.zeros(n, dtype=float)
    for row in queued:
        queue_run = np.where(row, queue_run + 5.0, 0.0)
    result["queue_age_minutes"] = queue_run
    boundary_names = (
        "queue_tail_distance_km", "queue_head_distance_km",
        "queue_tail_distance_links", "queue_head_distance_links",
        "queue_nearest_distance_km", "queue_component_length_km",
        "queue_tail_wave_15_kmh", "queue_tail_wave_30_kmh",
        "queue_head_wave_15_kmh", "queue_head_wave_30_kmh",
        "queue_tail_velocity_15_kmh", "queue_tail_velocity_30_kmh",
        "queue_head_velocity_15_kmh", "queue_head_velocity_30_kmh",
    )
    result.update({name: np.full(n, np.nan) for name in boundary_names})
    if len(tails) == 0:
        return result
    # Assign the nearest whole component, keeping each tail/head pair together.
    separation = np.maximum(
        np.maximum(distance[tails][None, :] - distance[:, None], 0),
        distance[:, None] - distance[heads][None, :],
    )
    component = np.argmin(separation, axis=1)
    tail, head = tails[component], heads[component]
    result["queue_tail_distance_km"] = distance - distance[tail]
    result["queue_head_distance_km"] = distance - distance[head]
    result["queue_tail_distance_links"] = np.arange(n) - tail
    result["queue_head_distance_links"] = np.arange(n) - head
    result["queue_nearest_distance_km"] = separation[np.arange(n), component]
    result["queue_component_length_km"] = distance[head] - distance[tail]
    density = flow / np.maximum(speed, 5.0)
    for steps, label in ((3, "15"), (6, "30")):
        q = _mean(flow[-(steps + 1):])
        k = _mean(density[-(steps + 1):])
        for boundary, upstream, downstream in (
            ("tail", tails - 1, tails), ("head", heads, heads + 1),
        ):
            valid = (upstream >= 0) & (downstream < n)
            wave = np.full(len(tails), np.nan)
            u, d = upstream[valid], downstream[valid]
            dq, dk = q[u] - q[d], k[u] - k[d]
            ratio = np.divide(dq, dk, out=np.full(len(u), np.nan), where=np.abs(dk) > 0.5)
            wave[valid] = np.clip(ratio, -40.0, 40.0)
            result[f"queue_{boundary}_wave_{label}_kmh"] = wave[component]
        previous_index = max(0, len(queued) - 1 - steps)
        elapsed = minutes[-1] - minutes[previous_index]
        past_tails, past_heads = _segments(queued[previous_index])
        if elapsed <= 0 or len(past_tails) == 0:
            continue
        for boundary, current, previous in (
            ("tail", tails, past_tails), ("head", heads, past_heads),
        ):
            nearest = np.argmin(
                np.abs(distance[current, None] - distance[previous][None, :]), axis=1,
            )
            velocity = (distance[current] - distance[previous[nearest]]) * 60.0 / elapsed
            result[f"queue_{boundary}_velocity_{label}_kmh"] = np.clip(velocity, -40, 40)[component]
    return result


def build_features(
    speed: np.ndarray,
    flow: np.ndarray,
    occupancy: np.ndarray | None,
    *,
    origin: int,
    vf: np.ndarray,
    capacity: np.ndarray,
    kcrit: np.ndarray,
    lanes: np.ndarray,
    distance_km: np.ndarray,
    timestamps: np.ndarray,
    ramp_on: np.ndarray | None = None,
    location_prior: np.ndarray | None = None,
    panel_id: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """Build one window's causal features in upstream-to-downstream order.

    ``origin`` is the index of released row T. Time arrays have shape
    (n_times, n_links); timestamps is datetime64-compatible. ``ramp_on`` is
    link-attached total on-flow with missing observations represented by NaN.
    ``location_prior`` may have any number of equally spaced daily bins and
    shape (n_daily_bins, n_links). It must use training dates only.
    """
    raw_speed = np.asarray(speed)
    if raw_speed.ndim != 2 or not 0 <= origin < raw_speed.shape[0]:
        raise ValueError("speed must be time x link and origin must be a valid row")
    n_times, n_links = raw_speed.shape
    if n_links == 0:
        raise ValueError("at least one link is required")
    start = max(0, origin - 12)

    def history(values: np.ndarray | None) -> np.ndarray:
        if values is None:
            return np.full((origin + 1 - start, n_links), np.nan)
        arr = np.asarray(values)
        if arr.shape != (n_times, n_links):
            raise ValueError("dynamic arrays must all have speed's time x link shape")
        # Slice BEFORE conversion, filling, statistics, or ramp processing.
        out = np.array(arr[start:origin + 1], dtype=np.float64, copy=True)
        out[~np.isfinite(out)] = np.nan
        return out

    v, q, occ, ramp = (history(a) for a in (speed, flow, occupancy, ramp_on))
    static = []
    for name, values in (("vf", vf), ("capacity", capacity), ("kcrit", kcrit),
                         ("lanes", lanes), ("distance_km", distance_km)):
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.shape != (n_links,) or not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must have one finite value per link")
        if name != "distance_km" and np.any(arr <= 0):
            raise ValueError(f"{name} must be positive")
        static.append(arr)
    vf, capacity, kcrit, lanes, distance = static
    if np.any(np.diff(distance) < 0):
        raise ValueError("distance_km must be ordered upstream to downstream")
    raw_time = np.asarray(timestamps)
    if raw_time.ndim != 1 or len(raw_time) != n_times:
        raise ValueError("timestamps must have one entry per time row")
    times = raw_time[start:origin + 1].astype("datetime64[m]")
    if np.any(np.isnat(times)) or np.any(np.diff(times).astype("timedelta64[m]").astype(int) <= 0):
        raise ValueError("history timestamps must be finite and strictly increasing")
    elapsed = (times - times[-1]).astype("timedelta64[m]").astype(float)
    v_fill, v_indices = _fill_forward(v)
    q_fill, _ = _fill_forward(q)
    speed_ratio = v / vf[None, :]
    flow_ratio = q / capacity[None, :]
    density_ratio = (q / np.maximum(v, 5.0)) / kcrit[None, :]
    series = {"speed_ratio": speed_ratio, "flow_ratio": flow_ratio,
              "density_ratio": density_ratio, "occupancy": occ}
    features: dict[str, np.ndarray] = {}

    def add(name: str, values: np.ndarray | float) -> None:
        arr = np.asarray(values, dtype=np.float64)
        features[name] = np.broadcast_to(arr, (n_links,)).copy()

    add("location_link_index", np.arange(n_links))
    add("location_link_fraction", np.arange(n_links) / max(n_links - 1, 1))
    add("location_distance_km", distance)
    add("location_panel_id", panel_id)
    for name, values in (("free_speed_kmh", vf), ("capacity_total_vph", capacity),
                         ("critical_density_total", kcrit), ("lanes", lanes)):
        add(name, values)
    add("spacing_up_km", distance - _shift(distance, -1))
    add("spacing_down_km", _shift(distance, 1) - distance)
    for name, values in series.items():
        for steps in (0, 1, 3, 6, 12):
            at = len(values) - 1 - steps
            add(f"{name}_lag{steps * 5}", values[at] if at >= 0 else np.nan)
        for length, label in ((4, "15"), (7, "30"), (13, "60")):
            chunk = values[-length:]
            add(f"{name}_mean{label}", _mean(chunk))
            if name != "occupancy":
                add(f"{name}_slope{label}", _slope(chunk, elapsed[-length:]))
        if name in ("speed_ratio", "flow_ratio"):
            add(f"{name}_min60", _extreme(values, "min"))
            add(f"{name}_max60", _extreme(values, "max"))
            add(f"{name}_std60", _std(values))
        add(f"{name}_observed_fraction", np.isfinite(values).mean(axis=0))
        add(f"{name}_missing_now", ~np.isfinite(values[-1]))
    add("speed_ratio_last_observed", v_fill[-1] / vf)
    add("flow_ratio_last_observed", q_fill[-1] / capacity)
    last_indices = v_indices[-1]
    observation_age = np.where(last_indices >= 0, -elapsed[np.maximum(last_indices, 0)], np.nan)
    add("speed_observation_age_minutes", observation_age)
    for offset in tuple(range(-5, 0)) + tuple(range(1, 6)):
        tag = f"{'up' if offset < 0 else 'down'}{abs(offset)}"
        for name in ("speed_ratio", "flow_ratio", "density_ratio"):
            values = series[name]
            add(f"{tag}_{name}_now", _shift(values[-1], offset))
            add(f"{tag}_{name}_slope15", _shift(_slope(values[-4:], elapsed[-4:]), offset))
    boundary = _boundary_features(v_fill, q_fill, vf, distance, elapsed)
    for name, values in boundary.items():
        add(name, values)
    add("global_speed_ratio_mean", _mean(speed_ratio[-1], axis=0))
    add("global_flow_ratio_mean", _mean(flow_ratio[-1], axis=0))
    add("global_speed_ratio_slope15", _mean(_slope(speed_ratio[-4:], elapsed[-4:]), axis=0))
    ramp_ratio = ramp / capacity[None, :]
    add("ramp_on_ratio_now", ramp_ratio[-1])
    add("ramp_on_ratio_mean15", _mean(ramp_ratio[-4:]))
    add("ramp_on_ratio_mean60", _mean(ramp_ratio))
    add("ramp_on_ratio_slope15", _slope(ramp_ratio[-4:], elapsed[-4:]))
    add("ramp_on_ratio_slope30", _slope(ramp_ratio[-7:], elapsed[-7:]))
    add("ramp_on_observed_fraction", np.isfinite(ramp).mean(axis=0))
    add("ramp_on_missing_now", ~np.isfinite(ramp[-1]))
    for offset in (-2, -1, 1, 2):
        add(f"ramp_on_{'up' if offset < 0 else 'down'}{abs(offset)}_ratio_now", _shift(ramp_ratio[-1], offset))

    # Calendar values at future forecast times are known at T and are causal.
    epoch_minutes = int(times[-1].astype(np.int64))
    origin_minute = epoch_minutes % 1440
    day = epoch_minutes // 1440
    dow = (day + 3) % 7  # 1970-01-01 was Thursday, Monday=0.
    add("origin_time_sin", np.sin(2 * np.pi * origin_minute / 1440))
    add("origin_time_cos", np.cos(2 * np.pi * origin_minute / 1440))
    add("origin_day_of_week", dow)
    add("origin_is_weekend", dow >= 5)
    prior = None
    if location_prior is not None:
        prior = np.asarray(location_prior, dtype=np.float64)
        if prior.ndim != 2 or prior.shape[1] != n_links or prior.shape[0] == 0:
            raise ValueError("location_prior must be daily-time-bin x link")
        add("location_queue_prior_origin", prior[(origin_minute * len(prior)) // 1440])
    else:
        # Keep the same feature schema whether or not a prior is supplied.
        add("location_queue_prior_origin", np.nan)
    base_names = list(features)
    base = np.column_stack([features[name] for name in base_names])
    horizon_names = [
        "horizon_minutes", "forecast_time_sin", "forecast_time_cos",
        "forecast_day_of_week", "location_queue_prior_target",
        "queue_tail_wave_forecast_distance_km", "queue_head_wave_forecast_distance_km",
        "queue_tail_motion_forecast_distance_km", "queue_head_motion_forecast_distance_km",
        "queue_wave_forecast_inside", "queue_motion_forecast_inside",
        "speed_ratio_linear_forecast", "flow_ratio_linear_forecast",
    ]
    blocks = []
    for horizon in (5, 10, 15, 20, 25, 30):
        target_minute = (origin_minute + horizon) % 1440
        target_dow = (dow + (origin_minute + horizon) // 1440) % 7
        tail_wave = boundary["queue_tail_distance_km"] - boundary["queue_tail_wave_15_kmh"] * horizon / 60
        head_wave = boundary["queue_head_distance_km"] - boundary["queue_head_wave_15_kmh"] * horizon / 60
        tail_motion = boundary["queue_tail_distance_km"] - boundary["queue_tail_velocity_15_kmh"] * horizon / 60
        head_motion = boundary["queue_head_distance_km"] - boundary["queue_head_velocity_15_kmh"] * horizon / 60
        wave_inside = np.where(np.isfinite(tail_wave) & np.isfinite(head_wave), (tail_wave >= 0) & (head_wave <= 0), np.nan)
        motion_inside = np.where(np.isfinite(tail_motion) & np.isfinite(head_motion), (tail_motion >= 0) & (head_motion <= 0), np.nan)
        target_prior = prior[(target_minute * len(prior)) // 1440] if prior is not None else np.full(n_links, np.nan)
        dynamic = np.column_stack([
            np.full(n_links, horizon),
            np.full(n_links, np.sin(2 * np.pi * target_minute / 1440)),
            np.full(n_links, np.cos(2 * np.pi * target_minute / 1440)),
            np.full(n_links, target_dow), target_prior,
            tail_wave, head_wave, tail_motion, head_motion, wave_inside, motion_inside,
            features["speed_ratio_last_observed"] + features["speed_ratio_slope15"] * horizon,
            features["flow_ratio_last_observed"] + features["flow_ratio_slope15"] * horizon,
        ])
        blocks.append(np.column_stack([base, dynamic]))
    result = np.concatenate(blocks, axis=0).astype(np.float32)
    result[~np.isfinite(result)] = np.nan
    return result, base_names + horizon_names
