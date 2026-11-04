"""Full-day causal queue context from released masked observations.

The input is a contiguous five-minute calendar grid. Every calculation uses
only the current UTC day through the row at ``origin``. Missing speeds carry
forward within that day; a link without any observation stays unknown.
"""
from __future__ import annotations

import numpy as np


FEATURE_NAMES = (
    "day_queue_now",
    "day_queue_run_minutes",
    "day_queue_cumulative_minutes",
    "day_minutes_since_first_queue",
    "day_queue_episodes",
    "day_speed_observation_age_minutes",
    "day_speed_observed_fraction",
    "day_corridor_queue_count",
    "day_corridor_queue_peak_count",
    "day_corridor_queue_to_peak_ratio",
    "day_corridor_minutes_since_peak",
    "day_corridor_queue_change_since_peak",
    "day_corridor_queue_change30_count",
    "day_corridor_minutes_since_first_queue",
    "day_corridor_known_fraction",
)


def _utc_minute(value) -> np.datetime64:
    # Released timestamps are UTC, either datetime64 or ISO text ending in Z.
    if isinstance(value, (str, np.str_)) and value.endswith("Z"):
        value = value[:-1]
    return np.datetime64(value, "m")


def _current_day_times(timestamps, origin: int) -> tuple[int, np.ndarray]:
    """Locate the day using T alone, then convert only that day's prefix.

    On the full five-minute grid the time of day determines the index of
    midnight. Future timestamps and other days need not even be parseable.
    """
    time = _utc_minute(timestamps[origin])
    if np.isnat(time):
        raise ValueError("the origin timestamp must be finite")
    minute = int((time - time.astype("datetime64[D]")).astype("timedelta64[m]").astype(int))
    if minute % 5:
        raise ValueError("timestamps must lie on a five-minute calendar grid")
    start = max(0, origin - minute // 5)
    selected = timestamps[start:origin + 1]
    if selected.dtype.kind == "M":
        times = selected.astype("datetime64[m]")
    else:
        times = np.asarray([_utc_minute(value) for value in selected], dtype="datetime64[m]")
    expected = time - np.arange(len(times) - 1, -1, -1) * np.timedelta64(5, "m")
    if np.any(np.isnat(times)) or not np.array_equal(times, expected):
        raise ValueError("current-day timestamps must be a contiguous five-minute grid")
    return start, times


def build_day_context(speed, flow, timestamps, *, origin, vf) -> tuple[np.ndarray, list[str]]:
    """Return ``[link, feature]`` float32 context and its ``day_*`` names.

    ``speed`` and ``flow`` have shape ``[time, link]``. ``vf`` is a positive
    free speed per link, or a scalar. Flow is accepted with the common
    observation interface; these queue context features depend on speed.

    Queue duration and cumulative minutes count five minutes per queued
    slot, including T. Missing readings after an observed state retain that
    state; only an observed free state ends a queue run. An episode begins
    at the first known queued state or a free-to-queued transition. First
    queue age is NaN until a queue has been observed. Queue metrics remain
    NaN for links never observed today; their observation fraction is zero.

    Corridor counts include links with a known carried-forward state. A
    peak tie uses the most recent peak. A known queue-free prefix has a
    current/peak ratio of zero; an entirely unknown prefix has NaN counts.
    The signed change since peak is current minus peak; the 30-minute
    change also describes growth. Global values are repeated across links.

    Inputs may span multiple days or start after the calendar day begins;
    context is limited to the available portion of the origin's UTC day.
    No numeric conversion or statistic touches another day's or a future
    row's observations, and no future timestamp is parsed.
    """
    raw_speed = np.asarray(speed)
    raw_flow = np.asarray(flow)
    raw_times = np.asarray(timestamps)
    if raw_speed.ndim != 2 or raw_speed.shape[1] == 0:
        raise ValueError("speed must have shape [time, link] with at least one link")
    if raw_flow.shape != raw_speed.shape:
        raise ValueError("flow must have the same [time, link] shape as speed")
    if raw_times.ndim != 1 or len(raw_times) != len(raw_speed):
        raise ValueError("timestamps must have one entry per time row")
    if not isinstance(origin, (int, np.integer)) or not 0 <= origin < len(raw_speed):
        raise ValueError("origin must be a valid time-row index")
    start, times = _current_day_times(raw_times, int(origin))
    # Determine the day slice before converting or inspecting measured values.
    values = np.array(raw_speed[start:origin + 1], dtype=np.float64, copy=True)
    values[~np.isfinite(values)] = np.nan
    n_slots, n_links = values.shape
    try:
        free_speed = np.broadcast_to(np.asarray(vf, dtype=np.float64), (n_links,))
    except ValueError as exc:
        raise ValueError("vf must be scalar or have one value per link") from exc
    if not np.all(np.isfinite(free_speed)) or np.any(free_speed <= 0):
        raise ValueError("vf must contain positive finite free speeds")

    observed = np.isfinite(values)
    slot = np.arange(n_slots)[:, None]
    last = np.maximum.accumulate(np.where(observed, slot, -1), axis=0)
    known = last >= 0
    filled = values[np.maximum(last, 0), np.arange(n_links)[None, :]]
    queued = known & (filled <= 0.6 * free_speed[None, :])
    known_now = known[-1]
    elapsed = (times[-1] - times).astype("timedelta64[m]").astype(float)

    last_free = np.maximum.accumulate(np.where(~queued, slot, -1), axis=0)
    run = np.where(queued[-1], (n_slots - 1 - last_free[-1]) * 5.0, 0.0)
    cumulative = queued.sum(axis=0).astype(float) * 5.0
    any_queue = queued.any(axis=0)
    first_queue = queued.argmax(axis=0)
    first_age = np.where(any_queue, elapsed[first_queue], np.nan)
    previous = np.concatenate((np.zeros((1, n_links), dtype=bool), queued[:-1]), axis=0)
    episodes = (queued & ~previous).sum(axis=0).astype(float)
    observation_age = np.where(known_now, elapsed[np.maximum(last[-1], 0)], np.nan)

    local = [
        queued[-1].astype(float), run, cumulative, first_age, episodes,
        observation_age, observed.mean(axis=0),
    ]
    # Unknown is not evidence of a queue-free link.
    for index in (0, 1, 2, 3, 4, 5):
        local[index] = np.where(known_now, local[index], np.nan)

    corridor_known = known.any(axis=1)
    count = np.where(corridor_known, queued.sum(axis=1).astype(float), np.nan)
    if corridor_known[-1]:
        current = count[-1]
        peak = np.nanmax(count)
        peak_slot = np.flatnonzero(count == peak)[-1]
        ratio = current / peak if peak > 0 else 0.0
        since_peak = elapsed[peak_slot]
        change_peak = current - peak
        # Early-day windows use their earliest available state, never yesterday.
        past_slot = max(0, n_slots - 1 - 6)
        change30 = current - count[past_slot]
        first = np.flatnonzero(count > 0)
        since_first = elapsed[first[0]] if first.size else np.nan
    else:
        current = peak = ratio = since_peak = change_peak = change30 = since_first = np.nan
    global_values = (
        current, peak, ratio, since_peak, change_peak, change30, since_first,
        known_now.mean(),
    )
    columns = local + [np.full(n_links, value) for value in global_values]
    return np.column_stack(columns).astype(np.float32), list(FEATURE_NAMES)
