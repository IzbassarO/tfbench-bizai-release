"""Small causal extension for early slowdowns and nearby bottlenecks.

This module deliberately leaves the original feature schema untouched.
Output rows are links in upstream-to-downstream order; callers may repeat
them across the six forecast horizons. Distances to a slow link are positive
when that link is downstream of the target link, negative when upstream.
"""
from __future__ import annotations

import numpy as np


def _neighbor(values: np.ndarray, offset: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    if abs(offset) >= len(values):
        return out
    if offset > 0:
        out[:-offset] = values[offset:]
    else:
        out[-offset:] = values[:offset]
    return out


def _last_observation(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(values)
    last = np.max(np.where(valid, np.arange(len(values))[:, None], -1), axis=0)
    result = values[np.maximum(last, 0), np.arange(values.shape[1])]
    return np.where(last >= 0, result, np.nan), last


def _finite_min(values: np.ndarray, axis=0) -> np.ndarray:
    finite = np.isfinite(values)
    minimum = np.min(np.where(finite, values, np.inf), axis=axis)
    return np.where(np.any(finite, axis=axis), minimum, np.nan)


def build_short_extras(
    speed: np.ndarray,
    flow: np.ndarray,
    timestamps: np.ndarray,
    *,
    origin: int,
    vf: np.ndarray,
    capacity: np.ndarray,
    distance_km: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Return 30 per-link features using observations no later than T.

    Dynamic arrays have shape time x link. Only the last thirteen rows ending
    at ``origin`` are used. Neighbor ratios use the last available observation
    in that history, keeping an entirely unobserved link missing. Capacity is
    total link veh/h. A capacity ratio below one denotes reduced capacity in
    the named adjacent link. One-sided slow-link distances are nonnegative.
    """
    raw_v, raw_q = np.asarray(speed), np.asarray(flow)
    if raw_v.ndim != 2 or raw_q.shape != raw_v.shape or raw_v.shape[1] == 0:
        raise ValueError("speed and flow must share a nonempty time x link shape")
    if not 0 <= origin < len(raw_v):
        raise ValueError("origin must identify a time row")
    n_links = raw_v.shape[1]
    start = max(0, origin - 12)
    # Slice before conversion or any statistics, including timestamps.
    v = np.array(raw_v[start:origin + 1], dtype=np.float64, copy=True)
    q = np.array(raw_q[start:origin + 1], dtype=np.float64, copy=True)
    v[~np.isfinite(v)] = np.nan
    q[~np.isfinite(q)] = np.nan
    times_raw = np.asarray(timestamps)
    if times_raw.ndim != 1 or len(times_raw) != len(raw_v):
        raise ValueError("timestamps must have one value per time row")
    times = times_raw[start:origin + 1].astype("datetime64[m]")
    if np.any(np.isnat(times)) or np.any(np.diff(times).astype("timedelta64[m]").astype(int) <= 0):
        raise ValueError("history timestamps must be finite and increasing")
    static = []
    for name, values in (("vf", vf), ("capacity", capacity), ("distance_km", distance_km)):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(values) != n_links or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must have one finite value per link")
        if name != "distance_km" and np.any(values <= 0):
            raise ValueError(f"{name} must be positive")
        static.append(values)
    vf, capacity, distance = static
    if np.any(np.diff(distance) < 0):
        raise ValueError("distance_km must increase upstream to downstream")
    last_v, last_v_index = _last_observation(v)
    last_q, _ = _last_observation(q)
    speed_ratio, flow_ratio = last_v / vf, last_q / capacity
    features: dict[str, np.ndarray] = {}

    def add(name: str, values) -> None:
        features[f"extra_{name}"] = np.broadcast_to(values, (n_links,)).copy()

    for offset in (-3, -2, -1, 1, 2, 3):
        side = "up" if offset < 0 else "down"
        add(f"{side}{abs(offset)}_last_speed_ratio", _neighbor(speed_ratio, offset))
        add(f"{side}{abs(offset)}_last_flow_ratio", _neighbor(flow_ratio, offset))
    for threshold, label in ((0.7, "70"), (0.8, "80")):
        slow = np.isfinite(speed_ratio) & (speed_ratio <= threshold)
        add(f"slow{label}_count", float(slow.sum()))
        signed, absolute, upstream, downstream = [np.full(n_links, np.nan) for _ in range(4)]
        if np.any(slow):
            delta = distance[slow][None, :] - distance[:, None]
            # Stable argmin chooses the first upstream-to-downstream slow link
            # for equal-distance ties.
            closest = np.argmin(np.abs(delta), axis=1)
            signed = delta[np.arange(n_links), closest]
            absolute = np.abs(signed)
            upstream = _finite_min(np.where(delta <= 0, -delta, np.nan), axis=1)
            downstream = _finite_min(np.where(delta >= 0, delta, np.nan), axis=1)
        add(f"slow{label}_nearest_signed_km", signed)
        add(f"slow{label}_nearest_absolute_km", absolute)
        add(f"slow{label}_upstream_distance_km", upstream)
        add(f"slow{label}_downstream_distance_km", downstream)
    add("corridor_min_speed_ratio", _finite_min(speed_ratio))
    finite_speed = speed_ratio[np.isfinite(speed_ratio)]
    add("corridor_speed_ratio_lower_quartile", np.quantile(finite_speed, 0.25) if len(finite_speed) else np.nan)
    add("capacity_ratio_up1", _neighbor(capacity, -1) / capacity)
    add("capacity_ratio_down1", _neighbor(capacity, 1) / capacity)
    add("capacity_ratio_min_down3", _finite_min(np.stack([_neighbor(capacity, offset) for offset in (1, 2, 3)])) / capacity)
    for side, offsets in (("up", (-3, -2, -1)), ("down", (1, 2, 3))):
        add(f"{side}_min_last_speed_ratio3", _finite_min(np.stack([_neighbor(speed_ratio, offset) for offset in offsets])))
    age = (times[-1] - times[np.maximum(last_v_index, 0)]).astype("timedelta64[m]").astype(float)
    add("last_speed_observation_age_minutes", np.where(last_v_index >= 0, age, np.nan))
    result = np.column_stack(list(features.values())).astype(np.float32)
    result[~np.isfinite(result)] = np.nan
    return result, list(features)
