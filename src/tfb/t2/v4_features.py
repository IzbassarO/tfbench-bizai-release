"""Portable onset demand/capacity evidence, with train-only capacity fitting.

The fitted per-link capacities are physical denominators, never classifier
identity columns. Support counts and training frequencies remain metadata.
Runtime features use the window prefix through T and, optionally, the masked
same-UTC-day prefix through T. Earlier-day activation priors are not used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import _extreme, _fill_forward, _mean, _shift, _slope


def _minute(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    out = stamp.to_datetime64().astype("datetime64[m]")
    if np.isnat(out):
        raise ValueError("a finite timestamp is required")
    return out


def _vector(values, n, name):
    out = np.asarray(values, float)
    if out.shape != (n,) or not np.isfinite(out).all() or (out <= 0).any():
        raise ValueError(f"{name} must contain one positive finite value per link")
    return out


def fit_empirical_capacity(arrays, network, enddate):
    """Fit robust observed/discharge capacities strictly before ``enddate``.

    Daily 99.5th-percentile flows provide an observed-flow ceiling. Where at
    least 48 observations across three days show a queued/free boundary,
    daily discharge 75th percentiles provide a more direct capacity proxy.
    A boundary is a slow link followed by a free one, or a free link just
    downstream of a slow one. Discharge estimates cannot fall below 80% of
    the observed-flow ceiling. Sparse links use their FD capacity.

    These are observational capacity proxies: demand-limited detectors may
    never reveal true capacity. The caller must fit on raw TRAIN only and
    retain the same dictionary for every evaluation split of that fold.
    ``capacity_discharge_only`` uses the supported boundary estimate or FD
    capacity; it NEVER falls back to observed demand peaks or their floor.
    Counts, support and raw fitted values are returned for audit, not inputs
    to a classifier. This function never changes the supplied arrays.
    """
    cutoff = _minute(enddate)
    times = np.asarray(arrays.timestamps)
    n = len(network.link_ids)
    if times.ndim != 1 or arrays.flow.shape != (len(times), n) or arrays.speed.shape != arrays.flow.shape:
        raise ValueError("aligned training timestamps, speed and flow required")
    stop = int(np.searchsorted(times, cutoff))
    prefix = times[:stop].astype("datetime64[m]")
    if len(prefix) and (np.isnat(prefix).any() or (np.diff(prefix) <= np.timedelta64(0, "m")).any()):
        raise ValueError("training prefix timestamps must be finite and increasing")
    q = np.array(arrays.flow[:stop], np.float32, copy=True)
    v = np.asarray(arrays.speed[:stop])
    vf = _vector(network.vf, n, "vf")
    fd = _vector(network.capacity, n, "capacity")
    eligible = np.asarray(arrays.eligible[:stop], bool)
    if eligible.shape != q.shape:
        raise ValueError("eligible must match observations")
    valid = eligible & np.isfinite(q) & np.isfinite(v) & (q > 0) & (v > 0)
    # Very large flow outliers should not define a denominator. This is a
    # broad physical guard, not censoring at the nominal FD capacity.
    valid &= q <= 1.5 * fd
    q[~valid] = np.nan
    ratio = np.divide(v, vf.astype(np.float32), dtype=np.float32)
    ratio[~valid] = np.nan
    slow = (ratio >= .15) & (ratio <= .70)
    free = ratio >= .80
    boundary = np.zeros_like(free)
    if n > 1:
        boundary[:, :-1] |= slow[:, :-1] & free[:, 1:]
        boundary[:, 1:] |= free[:, 1:] & slow[:, :-1]
    dates = prefix.astype("datetime64[D]")
    blocks = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1, len(prefix)] if len(prefix) else np.array([0])
    peak = np.full((max(0, len(blocks) - 1), n), np.nan)
    discharge = peak.copy()
    discharge_support = np.zeros(n, np.int64)
    for day, (lo, hi) in enumerate(zip(blocks[:-1], blocks[1:])):
        for link in range(n):
            values = q[lo:hi, link]
            observed = values[np.isfinite(values)]
            if len(observed) >= 12:
                peak[day, link] = np.quantile(observed, .995)
            values = values[boundary[lo:hi, link]]
            discharge_support[link] += len(values)
            if len(values) >= 3:
                discharge[day, link] = np.quantile(values, .75)
    support = valid.sum(axis=0, dtype=np.int64)
    days = np.isfinite(peak).sum(axis=0, dtype=np.int64)
    discharge_days = np.isfinite(discharge).sum(axis=0, dtype=np.int64)
    ceiling, discharge_cap = np.full(n, np.nan), np.full(n, np.nan)
    for link in range(n):
        if days[link]:
            ceiling[link] = np.median(peak[np.isfinite(peak[:, link]), link])
        if discharge_days[link]:
            discharge_cap[link] = np.median(discharge[np.isfinite(discharge[:, link]), link])
    use_peak = (support >= 48) & (days >= 3) & np.isfinite(ceiling)
    use_discharge = use_peak & (discharge_support >= 48) & (discharge_days >= 3) & np.isfinite(discharge_cap)
    fitted = np.where(use_peak, ceiling, fd)
    fitted = np.where(use_discharge, np.maximum(discharge_cap, .8 * ceiling), fitted)
    fitted = np.clip(fitted, .15 * fd, 1.5 * fd)
    discharge_only = np.clip(np.where(use_discharge, discharge_cap, fd), .15 * fd, 1.5 * fd)
    return dict(capacity=fitted.astype(np.float32), support=support,
        capacity_discharge_only=discharge_only.astype(np.float32),
        daily_support=days, discharge_support=discharge_support, discharge_days=discharge_days,
        daily_peak_capacity=ceiling.astype(np.float32), discharge_capacity=discharge_cap.astype(np.float32),
        used_discharge=use_discharge, used_fd_fallback=~use_peak, enddate=str(cutoff),
        method="daily_p995_median; boundary_discharge_daily_p75_median_if_supported; floor_0.8_daily_peak",
        metadata_not_classifier_features=True)


def _pool(values, offsets):
    return _mean(np.stack([np.stack([_shift(row, offset) for row in values]) for offset in offsets]), axis=0)


def _times_at(timestamps, origin, count=None):
    raw = np.asarray(timestamps)
    start = 0 if count is None else max(0, origin + 1 - count)
    selected = raw[start:origin + 1]
    times = (selected.astype("datetime64[m]") if selected.dtype.kind == "M"
             else np.array([_minute(t) for t in selected]))
    if np.isnat(times).any() or (np.diff(times) != np.timedelta64(5, "m")).any():
        raise ValueError("selected observations must be a finite five-minute grid")
    return start, times


def build_onset_features(history, network, empirical_capacity, *, dayarrays=None, origin_idx=None):
    """Return ``[link, feature]`` float32 portable onset features.

    ``history`` normally contains the thirteen legal T-60..T rows. With a
    supplied ``dayarrays`` and ``origin_idx``, that index identifies T in the
    day array, and any rows after T in either input are ignored. Without an
    explicit day index, T is the final history timestamp. Day inputs must be
    masked released observations; legal window rows overlay their same-day
    prefix, preserving the masked row at T. No feature exposes a fitted raw
    capacity, fit support, link identity, calendar value or earlier-day prior.

    Upstream arrival and T+30 extrapolations are causal linear proxies rather
    than a simulator; exits and unreleased demand cannot be reconstructed.
    """
    n = len(network.link_ids)
    if np.asarray(history.speed).ndim != 2 or history.speed.shape[1] != n:
        raise ValueError("history speed must be time x ordered link")
    if not len(history.timestamps):
        raise ValueError("history must contain T")
    if origin_idx is not None and dayarrays is None:
        raise ValueError("origin_idx indexes dayarrays and requires dayarrays")
    if dayarrays is not None and origin_idx is not None:
        if not isinstance(origin_idx, (int, np.integer)) or not 0 <= origin_idx < len(dayarrays.timestamps):
            raise ValueError("origin_idx must identify a day-array row")
        origin = _minute(dayarrays.timestamps[origin_idx])
    else:
        origin = _minute(history.timestamps[-1])
    first = _minute(history.timestamps[0])
    at = int((origin - first) / np.timedelta64(5, "m"))
    if not 0 <= at < len(history.timestamps) or _minute(history.timestamps[at]) != origin:
        raise ValueError("T must occur exactly in history")
    start, times = _times_at(history.timestamps, at, 13)
    minutes = (times - times[-1]).astype("timedelta64[m]").astype(float)
    vf = _vector(network.vf, n, "vf")
    fd = _vector(network.capacity, n, "capacity")
    kcrit = _vector(network.kcrit, n, "kcrit")
    ce = _vector(empirical_capacity["capacity"] if isinstance(empirical_capacity, dict) else empirical_capacity,
                 n, "empirical capacity")
    cd = _vector(empirical_capacity.get("capacity_discharge_only", fd) if isinstance(empirical_capacity, dict) else fd,
                 n, "discharge-only capacity")
    distance = np.asarray(network.distance_km, float)
    if distance.shape != (n,) or not np.isfinite(distance).all() or (np.diff(distance) < 0).any():
        raise ValueError("distance must be ordered upstream to downstream")

    def observed(arrays, key, lo, hi):
        value = getattr(arrays, key, None)
        if value is None:
            return np.full((hi - lo, n), np.nan)
        if np.shape(value) != (len(arrays.timestamps), n):
            raise ValueError(f"{key} must align with timestamps and network")
        result = np.array(value[lo:hi], float, copy=True)
        result[~np.isfinite(result)] = np.nan
        if key in {"speed", "flow", "ramp_on"}:
            result[result < 0] = np.nan
        return result

    vr, qr, rr = [observed(history, key, start, at + 1) for key in ("speed", "flow", "ramp_on")]
    v, vi = _fill_forward(vr)
    q, qi = _fill_forward(qr)
    r, _ = _fill_forward(rr)
    density_raw = np.divide(qr, np.maximum(vr, 5.), out=np.full_like(qr, np.nan),
                            where=np.isfinite(qr) & np.isfinite(vr))
    density, _ = _fill_forward(density_raw)
    speed, flow, density, ramp = v / vf, q / ce, density / kcrit, r / ce
    columns = {}

    def add(name, value):
        columns["v4_" + name] = np.broadcast_to(value, (n,)).copy()

    slopes = {}
    for label, values in (("flow_empirical_ratio", flow), ("density_ratio", density),
                          ("speed_ratio", speed), ("ramp_empirical_ratio", ramp)):
        add(label + "_now", values[-1])
        for lag, rows in ((15, 3), (30, 6), (60, 12)):
            add(f"{label}_lag{lag}", values[max(0, len(values) - 1 - rows)])
            slope = _slope(values[-(rows + 1):], minutes[-(rows + 1):])
            add(f"{label}_slope{lag}", slope)
            slopes[(label, lag)] = slope
        add(label + "_mean15", _mean(values[-4:]))
    add("speed_observation_age_minutes", np.where(vi[-1] >= 0, -minutes[np.maximum(vi[-1], 0)], np.nan))
    add("flow_observation_age_minutes", np.where(qi[-1] >= 0, -minutes[np.maximum(qi[-1], 0)], np.nan))
    add("flow_observed_fraction", np.isfinite(qr).mean(axis=0))
    add("density_observed_fraction", np.isfinite(density_raw).mean(axis=0))
    add("flow_nominal_ratio_now", q[-1] / fd)
    add("speed_drop15", speed[max(0, len(speed) - 4)] - speed[-1])
    for trend in (15, 30):
        projected = flow[-1] + 30 * slopes[("flow_empirical_ratio", trend)]
        add(f"flow_projected30_trend{trend}", np.clip(projected, 0., 4.))
    add("density_projected30", np.clip(density[-1] + 30 * slopes[("density_ratio", 30)], 0., 8.))
    growth = slopes[("flow_empirical_ratio", 30)]
    ttc = np.divide(1 - flow[-1], growth, out=np.full(n, np.nan), where=growth > 1e-5)
    add("minutes_to_empirical_capacity", np.clip(ttc, -60., 240.))
    add("ramp_projected30_ratio", np.clip(ramp[-1] + 30 * slopes[("ramp_empirical_ratio", 30)], 0., 2.))
    raw_flow_slope30 = _slope(q[-7:], minutes[-7:])
    discharge_growth = raw_flow_slope30 / cd
    add("flow_discharge_only_ratio_now", q[-1] / cd)
    add("flow_discharge_only_ratio_slope30", discharge_growth)
    add("flow_discharge_only_projected30", np.clip(q[-1] / cd + 30 * discharge_growth, 0., 4.))
    add("minutes_to_discharge_capacity", np.clip(np.divide(1 - q[-1] / cd, discharge_growth,
        out=np.full(n, np.nan), where=discharge_growth > 1e-5), -60., 240.))
    for side, offsets in (("upstream3", (-1, -2, -3)), ("downstream3", (1, 2, 3))):
        pooled_q = _pool(q, offsets) / ce
        pooled_k = _pool(density, offsets)
        pooled_v = _pool(speed, offsets)
        add(side + "_flow_empirical_ratio_now", pooled_q[-1])
        add(side + "_flow_empirical_ratio_slope30", _slope(pooled_q[-7:], minutes[-7:]))
        add(side + "_density_ratio_now", pooled_k[-1])
        add(side + "_speed_ratio_now", pooled_v[-1])
        add(side + "_density_imbalance", pooled_k[-1] - density[-1])
    upstream = _pool(q, (-1, -2, -3))
    upstream_slope = _slope(upstream[-7:], minutes[-7:])
    travel = np.array([np.mean(distance[i] - distance[max(0, i - 3):i]) if i else np.nan for i in range(n)])
    travel = travel / vf * 60
    add("upstream_arrival30_empirical_ratio", np.clip((upstream[-1] + np.maximum(30 - travel, 0) * upstream_slope) / ce, 0., 4.))
    add("upstream_flow_discharge_only_ratio_now", upstream[-1] / cd)
    add("upstream_arrival30_discharge_only_ratio", np.clip((upstream[-1] + np.maximum(30 - travel, 0) * upstream_slope) / cd, 0., 4.))
    add("upstream_arrival_minus_current_ratio", (upstream[-1] - q[-1]) / ce)
    add("upstream_ramp_growth30_ratio", _slope(_pool(r, (-2, -1, 0))[-7:], minutes[-7:]) / ce)
    add("upstream_demand_plus_ramp_ratio", (upstream[-1] + _pool(r, (-2, -1, 0))[-1]) / ce)
    projected = np.clip(flow[-1] + 30 * growth, 0., 4.)
    for width in (5, 10):
        anchors = np.array([i + np.argmin(ce[i:min(n, i + width + 1)]) for i in range(n)])
        denom = ce[anchors]
        add(f"downstream{width}_capacity_contrast", denom / ce)
        add(f"downstream{width}_bottleneck_distance_km", distance[anchors] - distance)
        add(f"downstream{width}_bottleneck_flow_stress", q[-1, anchors] / denom)
        add(f"downstream{width}_bottleneck_density_ratio", density[-1, anchors])
        add(f"downstream{width}_bottleneck_speed_ratio", speed[-1, anchors])
        add(f"downstream{width}_bottleneck_flow_slope30", _slope(q[-7:, anchors], minutes[-7:]) / denom)
        add(f"downstream{width}_demand_stress", q[-1] / denom)
        add(f"downstream{width}_demand_stress_projected30", (q[-1] + 30 * growth * ce) / denom)
        add(f"downstream{width}_flow_gap_ratio", (q[-1] - q[-1, anchors]) / denom)
        add(f"downstream{width}_maximum_projected_stress", np.array([
            _extreme(projected[i:min(n, i + width + 1), None], "max")[0] for i in range(n)]))
        discharge_anchor = np.array([i + np.argmin(cd[i:min(n, i + width + 1)]) for i in range(n)])
        discharge_denom = cd[discharge_anchor]
        add(f"downstream{width}_discharge_capacity_contrast", discharge_denom / cd)
        add(f"downstream{width}_discharge_demand_stress", q[-1] / discharge_denom)
        add(f"downstream{width}_discharge_bottleneck_stress", q[-1, discharge_anchor] / discharge_denom)
        add(f"downstream{width}_discharge_demand_projected30", (q[-1] + 30 * raw_flow_slope30) / discharge_denom)
    finite = flow[-1, np.isfinite(flow[-1])]
    rank = np.full(n, np.nan)
    if len(finite):
        order = np.sort(finite)
        valid = np.isfinite(flow[-1])
        rank[valid] = (np.searchsorted(order, flow[-1, valid], "left") +
                       np.searchsorted(order, flow[-1, valid], "right")) / (2 * len(order))
    add("current_stress_corridor_percentile", rank)
    add("stress_minus_corridor_peak", flow[-1] - _extreme(flow[-1:, :].T, "max")[0])
    add("projected_stress_corridor_peak", _extreme(projected[:, None], "max")[0])

    # Optional same-day adaptation. The legal window prefix is overlaid to
    # make training/deployment behavior agree when masked context has holes.
    day_values = None
    if dayarrays is not None:
        dfirst = _minute(dayarrays.timestamps[0])
        day_at = int((origin - dfirst) / np.timedelta64(5, "m")) if origin_idx is None else int(origin_idx)
        if not 0 <= day_at < len(dayarrays.timestamps) or _minute(dayarrays.timestamps[day_at]) != origin:
            raise ValueError("T must occur exactly in dayarrays")
        slots_today = int((origin - origin.astype("datetime64[D]")) / np.timedelta64(5, "m")) + 1
        dstart, day_times = _times_at(dayarrays.timestamps, day_at, slots_today)
        dv, dq = [observed(dayarrays, key, dstart, day_at + 1) for key in ("speed", "flow")]
        overlap = times >= origin.astype("datetime64[D]")
        indexes = ((times[overlap] - day_times[0]) / np.timedelta64(5, "m")).astype(int)
        valid = (indexes >= 0) & (indexes < len(day_times))
        dv[indexes[valid]] = vr[overlap][valid]
        dq[indexes[valid]] = qr[overlap][valid]
        day_density = np.divide(dq, np.maximum(dv, 5.), out=np.full_like(dq, np.nan),
                                where=np.isfinite(dq) & np.isfinite(dv)) / kcrit
        day_values = (dv / vf, dq / ce, day_times, dq / cd, day_density)
    for label in ("flow_peak_ratio", "flow_peak120_ratio", "density_peak_ratio", "slow70_observed_fraction",
                  "minutes_since_near_capacity", "minutes_since_slow70", "speed_minimum_ratio",
                  "discharge_flow_peak_ratio", "discharge_flow_peak120_ratio", "minutes_since_discharge_near_capacity"):
        add("day_" + label, np.nan)
    if day_values is not None:
        dv, dq, dt, dq_discharge, kd = day_values
        elapsed = (dt[-1] - dt).astype("timedelta64[m]").astype(float)
        columns["v4_day_flow_peak_ratio"] = _extreme(dq, "max")
        columns["v4_day_flow_peak120_ratio"] = _extreme(dq[elapsed <= 120], "max")
        columns["v4_day_discharge_flow_peak_ratio"] = _extreme(dq_discharge, "max")
        columns["v4_day_discharge_flow_peak120_ratio"] = _extreme(dq_discharge[elapsed <= 120], "max")
        columns["v4_day_density_peak_ratio"] = _extreme(kd, "max")
        columns["v4_day_slow70_observed_fraction"] = _mean(np.where(np.isfinite(dv), (dv <= .7).astype(float), np.nan))
        for name, event in (("near_capacity", dq >= .9), ("slow70", dv <= .7),
                            ("discharge_near_capacity", dq_discharge >= .9)):
            recent = np.max(np.where(event, np.arange(len(dt))[:, None], -1), axis=0)
            columns[f"v4_day_minutes_since_{name}"] = np.where(recent >= 0, elapsed[np.maximum(recent, 0)], np.nan)
        columns["v4_day_speed_minimum_ratio"] = _extreme(dv, "min")
    names = list(columns)
    result = np.column_stack([columns[name] for name in names])
    result[~np.isfinite(result)] = np.nan
    return result.astype(np.float32), names
