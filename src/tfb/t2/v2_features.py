"""Causal demand/discharge features and training-only spatial augmentation.

The feature builder returns *additional* columns in horizon-major order. It
does not read priors, labels, full-day context, or any observation after T.
Flow/capacity and density/critical-density use total-link quantities. Queue
mass and clearance are explanatory proxies, not a forward traffic simulator:
unreleased boundary flows, off-ramps and incident capacity are unknown.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import _fill_forward, _mean, _segments, _slope


def _static(values, name, n_links, *, positive=True):
    out = np.asarray(values, dtype=np.float64).reshape(-1)
    if out.shape != (n_links,) or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain one finite value per link")
    if positive and np.any(out <= 0):
        raise ValueError(f"{name} must be positive")
    return out


def _pool(values, anchors, offsets):
    """Mean of available neighbors at each time, without corridor wrapping."""
    indices = anchors[:, None] + np.asarray(offsets)[None, :]
    valid = (indices >= 0) & (indices < values.shape[1])
    gathered = values[:, np.clip(indices, 0, values.shape[1] - 1)]
    return _mean(np.where(valid[None, :, :], gathered, np.nan), axis=2)


def _region(values, mask, *, total=False):
    """Aggregate time x link values over each target link's component region."""
    valid = np.isfinite(values)
    numerator = np.where(valid, values, 0.0) @ mask.T
    count = valid.astype(float) @ mask.T
    if total:
        return np.where(count > 0, numerator, np.nan)
    return np.divide(numerator, count, out=np.full_like(numerator, np.nan), where=count > 0)


def _wave(q_up, q_down, k_up, k_down):
    dq, dk = q_up - q_down, k_up - k_down
    result = np.divide(dq, dk, out=np.full_like(dq, np.nan), where=np.abs(dk) > 0.5)
    return np.clip(result, -40.0, 40.0)


def build_v2_features(
    speed, flow, occupancy, timestamps, *, origin, vf, capacity, kcrit,
    lanes, distance_km, ramp_on=None,
) -> tuple[np.ndarray, list[str]]:
    """Return compact causal discharge/demand extras, shape ``[6*links,F]``.

    Only T-60..T is inspected. Last observations carry forward inside that
    prefix; observation coverage remains explicit. Current queued components
    use the released .60*v_f threshold. Each target link uses its nearest
    component; with no current queue it uses the slowest observed link among
    itself and the next five downstream links as a prospective bottleneck.

    Demand pools the three links immediately upstream of the queue tail;
    discharge pools three downstream of its head. At the corridor's downstream
    edge the head's own detector is an explicit discharge proxy. Densities are
    derived only from contemporaneously available speed/flow pairs before
    causal filling. Slopes use actual history timestamps and units per minute.
    Occupancy is accepted for the common observation interface but not used.
    """
    raw_speed = np.asarray(speed)
    if raw_speed.ndim != 2 or raw_speed.shape[1] == 0:
        raise ValueError("speed must have nonempty time x link shape")
    if not isinstance(origin, (int, np.integer)) or not 0 <= origin < len(raw_speed):
        raise ValueError("origin must identify a time row")
    n_times, n_links = raw_speed.shape
    start = max(0, int(origin) - 12)

    def history(values):
        if values is None:
            return np.full((origin + 1 - start, n_links), np.nan)
        values = np.asarray(values)
        if values.shape != (n_times, n_links):
            raise ValueError("dynamic inputs must share speed's time x link shape")
        selected = np.array(values[start:origin + 1], dtype=np.float64, copy=True)
        selected[~np.isfinite(selected)] = np.nan
        return selected

    v_raw, q_raw, ramps = [history(values) for values in (speed, flow, ramp_on)]
    if occupancy is not None and np.asarray(occupancy).shape != raw_speed.shape:
        raise ValueError("occupancy must share speed's time x link shape")
    static = [_static(values, name, n_links) for values, name in
              ((vf, "vf"), (capacity, "capacity"), (kcrit, "kcrit"), (lanes, "lanes"))]
    vf, capacity, kcrit, lanes = static
    distance = _static(distance_km, "distance_km", n_links, positive=False)
    if np.any(np.diff(distance) < 0):
        raise ValueError("distance_km must be ordered upstream to downstream")
    raw_times = np.asarray(timestamps)
    if raw_times.ndim != 1 or len(raw_times) != n_times:
        raise ValueError("timestamps must have one value per time row")
    times = raw_times[start:origin + 1].astype("datetime64[m]")
    if np.any(np.isnat(times)) or np.any(np.diff(times).astype("timedelta64[m]").astype(int) <= 0):
        raise ValueError("history timestamps must be finite and increasing")
    minutes = (times - times[-1]).astype("timedelta64[m]").astype(float)
    v, v_indices = _fill_forward(v_raw)
    q, _ = _fill_forward(q_raw)
    k_raw = q_raw / np.maximum(v_raw, 5.0)
    k, _ = _fill_forward(k_raw)
    v_ratio, q_ratio, k_ratio = v / vf, q / capacity, k / kcrit
    queued = np.isfinite(v) & (v_ratio <= 0.6)
    tails, heads = _segments(queued[-1])
    link = np.arange(n_links)
    active = np.full(n_links, len(tails) > 0)
    if len(tails):
        separation = np.maximum(np.maximum(distance[tails][None, :] - distance[:, None], 0),
                                distance[:, None] - distance[heads][None, :])
        nearest = np.argmin(separation, axis=1)
        tail, head = tails[nearest], heads[nearest]
    else:
        ahead = link[:, None] + np.arange(6)[None, :]
        valid = ahead < n_links
        scores = v_ratio[-1, np.minimum(ahead, n_links - 1)]
        scores = np.where(valid & np.isfinite(scores), scores, np.inf)
        head = np.minimum(link + np.argmin(scores, axis=1), n_links - 1)
        tail = head.copy()
    anchor_capacity, anchor_kcrit = capacity[head], kcrit[head]
    exact_region = ((link[None, :] >= tail[:, None]) &
                    (link[None, :] <= head[:, None]) & active[:, None])
    # A fixed three-link halo measures recent queue shrinkage around the
    # currently surviving component, without consulting future boundaries.
    halo = ((link[None, :] >= (tail - 3)[:, None]) &
            (link[None, :] <= (head + 3)[:, None]) & active[:, None])
    if n_links == 1:
        lengths = np.ones(1)
    else:
        steps = np.diff(distance)
        lengths = np.r_[steps[0], (steps[:-1] + steps[1:]) / 2, steps[-1]]
    demand_q = _pool(q, tail, (-3, -2, -1))
    demand_k = _pool(k, tail, (-3, -2, -1))
    discharge_q = _pool(q, head, (1, 2, 3))
    discharge_k = _pool(k, head, (1, 2, 3))
    at_downstream_edge = head == n_links - 1
    discharge_q[:, at_downstream_edge] = q[:, head[at_downstream_edge]]
    discharge_k[:, at_downstream_edge] = k[:, head[at_downstream_edge]]
    demand_mean, discharge_mean = _mean(demand_q[-4:]), _mean(discharge_q[-4:])
    demand_slope = _slope(demand_q[-4:], minutes[-4:])
    discharge_slope = _slope(discharge_q[-4:], minutes[-4:])
    density_up, density_down = _mean(demand_k[-4:]), _mean(discharge_k[-4:])
    q_head, k_head = _mean(q[-4:, head]), _mean(k[-4:, head])
    q_tail, k_tail = _mean(q[-4:, tail]), _mean(k[-4:, tail])
    tail_wave = _wave(demand_mean, q_tail, density_up, k_tail)
    head_wave = _wave(q_head, discharge_mean, k_head, density_down)
    # Delayed vehicles relative to the same flow travelling at free speed.
    delayed_density = np.maximum(k - q / vf, 0.0)
    delayed = np.where(queued, delayed_density * lengths, 0.0)
    delayed[~np.isfinite(k) | ~np.isfinite(q)] = np.nan
    mass = _region(delayed, halo, total=True)
    queue_count = queued.astype(float) @ halo.T
    component_speed = _region(v_ratio, exact_region)
    component_density = _region(k_ratio, exact_region)
    for values in (mass, queue_count):
        values[:, ~active] = 0.0
    current_mass = mass[-1]
    net_inflow = demand_mean - discharge_mean
    draining = discharge_mean - demand_mean
    clearance = np.divide(current_mass * 60, draining, out=np.full(n_links, np.nan), where=draining > 1.0)
    run = np.zeros(n_links)
    for row in queued:
        run = np.where(row, run + 5, 0)
    oldest = np.max(np.where(exact_region, run[None, :], 0), axis=1)
    observed_fraction = _region(np.isfinite(v_raw[-1:]).astype(float), exact_region)[-1]
    component_ramp = _region(ramps, exact_region, total=True)
    component_ramp_coverage = _region(np.isfinite(ramps).astype(float), exact_region)
    features = {}

    def add(name, values):
        features[f"v2_{name}"] = np.broadcast_to(values, (n_links,)).copy()

    add("anchor_distance_downstream_km", distance[head] - distance)
    add("anchor_is_queued", active.astype(float))
    add("anchor_capacity_ratio", anchor_capacity / capacity)
    add("anchor_lanes_ratio", lanes[head] / lanes)
    ahead_capacity = _pool(capacity[None, :], link, (1, 2, 3))  # Mean scale as additional geometry.
    add("downstream3_capacity_mean_ratio", ahead_capacity[-1] / capacity)
    add("upstream3_capacity_mean_ratio", _pool(capacity[None, :], link, (-3, -2, -1))[-1] / capacity)
    add("queue_extent_km", np.where(active, distance[head] - distance[tail], 0.0))
    add("queue_current_link_count", exact_region.sum(axis=1).astype(float))
    add("queue_current_observed_fraction", observed_fraction)
    add("discharge_uses_corridor_edge_proxy", at_downstream_edge.astype(float))
    for side, offsets in (("upstream", (-3, -2, -1)), ("downstream", (1, 2, 3))):
        local_q = _pool(q_ratio, link, offsets)
        add(f"{side}_flow_ratio_mean15", _mean(local_q[-4:]))
        add(f"{side}_flow_ratio_slope15", _slope(local_q[-4:], minutes[-4:]))
        add(f"{side}_speed_ratio_mean15", _mean(_pool(v_ratio, link, offsets)[-4:]))
        add(f"{side}_density_ratio_mean15", _mean(_pool(k_ratio, link, offsets)[-4:]))
    for boundary, indices in (("head", head), ("tail", tail)):
        add(f"{boundary}_speed_ratio_slope15", _slope(v_ratio[-4:, indices], minutes[-4:]))
        add(f"{boundary}_flow_ratio_mean15", _mean(q_ratio[-4:, indices]))
        add(f"{boundary}_flow_ratio_slope15", _slope(q_ratio[-4:, indices], minutes[-4:]))
        add(f"{boundary}_density_ratio_mean15", _mean(k_ratio[-4:, indices]))
    for name, mean_q, slope_q, mean_k in (
        ("demand", demand_mean, demand_slope, density_up),
        ("discharge", discharge_mean, discharge_slope, density_down),
    ):
        add(f"{name}_mean15_vph", mean_q)
        add(f"{name}_capacity_ratio", mean_q / anchor_capacity)
        add(f"{name}_slope15_capacity_ratio", slope_q / anchor_capacity)
        add(f"{name}_density_ratio", mean_k / anchor_kcrit)
    add("net_inflow_capacity_ratio", net_inflow / anchor_capacity)
    add("net_inflow_slope15_capacity_ratio", (demand_slope - discharge_slope) / anchor_capacity)
    add("boundary_density_jump_ratio", (density_up - density_down) / anchor_kcrit)
    add("tail_wave15_kmh", tail_wave)
    add("head_wave15_kmh", head_wave)
    add("demand_change30_capacity_ratio", (demand_mean - _mean(demand_q[-7:-3])) / anchor_capacity if len(q) >= 7 else np.nan)
    add("discharge_change30_capacity_ratio", (discharge_mean - _mean(discharge_q[-7:-3])) / anchor_capacity if len(q) >= 7 else np.nan)
    add("queue_delayed_vehicles", current_mass)
    for back, label in ((3, "15"), (6, "30")):
        at = max(0, len(q) - 1 - back)
        add(f"queue_delayed_vehicles_change{label}", current_mass - mass[at])
        add(f"queue_link_count_change{label}", queue_count[-1] - queue_count[at])
    add("queue_speed_recovery_slope15", _slope(component_speed[-4:], minutes[-4:]))
    add("queue_density_slope15", _slope(component_density[-4:], minutes[-4:]))
    add("queue_oldest_visible_age_minutes", oldest)
    add("ramp_inside_capacity_ratio_mean15", _mean(component_ramp[-4:]) / anchor_capacity)
    add("ramp_inside_capacity_ratio_slope15", _slope(component_ramp[-4:], minutes[-4:]) / anchor_capacity)
    add("ramp_inside_observed_fraction", _mean(component_ramp_coverage[-4:]))
    add("clearance_minutes", clearance)
    add("positive_net_discharge_capacity_ratio", np.maximum(draining, 0) / anchor_capacity)
    names = list(features)
    common = np.column_stack([features[name] for name in names])
    future_names = ["v2_projected_delayed_vehicles", "v2_projected_remaining_fraction",
                    "v2_clearance_before_horizon", "v2_head_wave_distance_at_horizon_km"]
    blocks = []
    for horizon in (5, 10, 15, 20, 25, 30):
        projected = np.maximum(current_mass + net_inflow * horizon / 60, 0)
        remaining = np.divide(projected, current_mass, out=np.full(n_links, np.nan), where=current_mass > 0)
        clear = np.where(np.isfinite(clearance), (clearance <= horizon).astype(float), np.nan)
        head_distance = distance - distance[head] - head_wave * horizon / 60
        blocks.append(np.column_stack([common, projected, remaining, clear, head_distance]))
    result = np.concatenate(blocks).astype(np.float32)
    result[~np.isfinite(result)] = np.nan
    return result, names + future_names


@dataclass
class ShiftedExample:
    """Synthetic training observation/target tensors and geometric coverage.

    ``source_index[destination]`` is -1 at unknown edges. ``target_eligible``
    shifts the supplied eligibility mask; if absent, it only denotes mapped
    cells and callers must combine it with actual source target eligibility.
    """

    speed: np.ndarray
    flow: np.ndarray
    occupancy: np.ndarray
    target_labels: np.ndarray
    target_eligible: np.ndarray
    source_index: np.ndarray
    destination_valid: np.ndarray


def shift_history_and_targets(
    history_speed, history_flow, history_occ, target_labels, *, vf, capacity,
    kcrit, shift, target_eligible=None,
) -> ShiftedExample:
    """Move a training example spatially without wrapping corridor endpoints.

    Positive ``shift`` moves the pattern downstream. Speed and total flow
    preserve source ``v/vf`` and ``q/capacity`` at the destination. This also
    preserves density/critical-density when the source/destination FD satisfy
    the same capacity/(vf*kcrit) ratio, as the released FD does. Incompatible
    FD scales are rejected rather than silently creating inconsistent states.

    Occupancy is shifted unchanged as a dimensionless congestion signal: this
    API lacks detector occupancy/jam-density calibration for exact transport.
    Unknown new edges are NaN inputs and False, ineligible targets. This is
    deterministic training augmentation, not a simulated incident, a scoring
    target correction, or a validation/private feature transformation. Supply
    only the causal training history; target tensors never enter features.
    """
    raw_v, raw_q = np.asarray(history_speed), np.asarray(history_flow)
    if raw_v.ndim != 2 or raw_v.shape[1] == 0 or raw_q.shape != raw_v.shape:
        raise ValueError("history speed/flow must share nonempty time x link shape")
    if not isinstance(shift, (int, np.integer)) or isinstance(shift, (bool, np.bool_)):
        raise ValueError("shift must be an integer number of links")
    n_links = raw_v.shape[1]
    raw_occ = None if history_occ is None else np.asarray(history_occ)
    if raw_occ is not None and raw_occ.shape != raw_v.shape:
        raise ValueError("history occupancy must share history speed shape")
    labels = np.asarray(target_labels)
    if labels.ndim != 2 or labels.shape[1] != n_links or not np.isin(labels, [0, 1]).all():
        raise ValueError("target_labels must be binary horizon x link")
    if target_eligible is None:
        eligible = np.ones(labels.shape, bool)
    else:
        eligible = np.asarray(target_eligible)
        if eligible.shape != labels.shape or not np.isin(eligible, [0, 1]).all():
            raise ValueError("target_eligible must be binary and match target_labels")
        eligible = eligible.astype(bool, copy=False)
    vf = _static(vf, "vf", n_links)
    capacity = _static(capacity, "capacity", n_links)
    kcrit = _static(kcrit, "kcrit", n_links)
    source_index = np.arange(n_links) - int(shift)
    valid = (source_index >= 0) & (source_index < n_links)
    source_index = np.where(valid, source_index, -1)
    source = source_index[valid]
    fd_scale = capacity / (vf * kcrit)
    if not np.allclose(fd_scale[source], fd_scale[valid], rtol=1e-4, atol=1e-6):
        raise ValueError("source/destination FD capacity/(vf*kcrit) ratios are incompatible")

    def moved(values, scale=None):
        dtype = np.result_type(np.asarray(values).dtype, np.float32)
        out = np.full(raw_v.shape, np.nan, dtype=dtype)
        selected = np.asarray(values)[:, source].astype(np.float64, copy=True)
        selected[~np.isfinite(selected)] = np.nan
        if scale is not None:
            selected *= scale[None, :]
        out[:, valid] = selected
        return out

    speed = moved(raw_v, vf[valid] / vf[source])
    flow = moved(raw_q, capacity[valid] / capacity[source])
    occupancy = np.full(raw_v.shape, np.nan, np.float32) if raw_occ is None else moved(raw_occ)
    new_labels = np.zeros(labels.shape, bool)
    new_eligible = np.zeros(labels.shape, bool)
    new_labels[:, valid] = labels[:, source].astype(bool)
    new_eligible[:, valid] = eligible[:, source]
    return ShiftedExample(speed, flow, occupancy, new_labels, new_eligible,
                          source_index.astype(np.int64), valid)
