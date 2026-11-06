"""Deterministic Task 2 proxy-window reconstruction from train observations.

The organizer release contains the policy and scorer, but not its selector
implementation or the hidden underlying queue labels.  This module implements
the published constraints and a chronological scan; truncation to the five
earliest windows is explicit.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


_COLUMNS = [
    "origin_idx", "origin_time", "forecast_origin", "window_type", "condition",
    "date", "first_queue_horizon", "first_queue_step", "history_coverage",
    "persistence_iou", "n_queue_at_origin", "n_queue_horizon", "n_eligible_horizon",
]


def select_windows(
    labels,
    timestamps,
    eligible=None,
    *,
    min_spacing_minutes: float = 360,
    seed: int = 42,
    max_per_condition: int | Mapping[str, int] | None = None,
    min_history_coverage: float = 0.7,
    max_persistence_iou: float | None = 0.9,
    require_horizon_queue: bool = True,
    history_includes_origin: bool = True,
    condition_policy: str = "strict",
    spacing_policy: str = "within_condition",
) -> pd.DataFrame:
    """Scan eligible origins chronologically and retain separated windows.

    Inputs are ``[time, link]`` queue labels and a strictly increasing five-minute
    timestamp grid.  Gaps and midnight boundaries are allowed in the input, but
    a window must be contiguous and stay within one calendar day.  Histories
    span T-60..T by default; targets always span T+5..T+30.  At least 70% of
    history cells must be eligible, if an eligibility mask is supplied.

    An onset has no queue in its complete history and at least one target
    queue; an ongoing has at least one queue in its final history slot and a
    target queue.  A chronological onset scan admits the earliest origin
    before a new event, making the first queued target T+30.  Each condition
    has an independent spacing clock by default. Persistence IoU is measured only on
    eligible target cells and candidates above .9 are rejected.

    ``history_includes_origin=False`` reproduces the public baseline's
    T-60..T-5 history, condition at T-5, and persistence at T-5.  The row at T is
    allowed as a feature in either mode.  Eligibility controls coverage and
    persistence scoring, not the existence of a queue in an observed state.
    Pass labels with unknown cells False and an appropriate eligibility mask.

    ``condition_policy='contract'`` instead uses the downloaded competition
    contract: ongoing means at least two queued history observations on one
    link; onset means no such established queue.  It does not require a queue at the
    last history slot, and an isolated noisy history queue can remain onset.

    ``spacing_policy='shared'`` uses one chronological spacing clock for both
    conditions.  No released train/validation/private origins violate this
    stronger constraint, and it reproduces substantially more ongoing origins
    when paired with gap-filled train selector labels.  The public docs specify
    only within-condition spacing, which remains the default.

    ``max_per_condition`` truncates each chronological list (or accepts a
    condition-to-limit dict).  ``seed`` is recorded for reproducibility; the
    scan is deterministic and uses no random sampling.
    """
    raw = np.asarray(labels)
    if raw.ndim != 2:
        raise ValueError("labels must have shape [time, link]")
    if raw.dtype != bool and (not np.isin(raw, [0, 1]).all()):
        raise ValueError("labels must be boolean or contain only 0 and 1")
    queued = raw.astype(bool, copy=False)
    times = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if hasattr(times, "as_unit"):
        times = times.as_unit("ns")
    if len(times) != queued.shape[0]:
        raise ValueError("timestamps length must equal labels.shape[0]")
    if times.hasnans or not times.is_monotonic_increasing or times.has_duplicates:
        raise ValueError("timestamps must be non-null and strictly increasing")
    if eligible is None:
        valid = np.ones(queued.shape, dtype=bool)
    else:
        raw_eligible = np.asarray(eligible)
        if raw_eligible.shape != queued.shape:
            raise ValueError("eligible must have the same [time, link] shape as labels")
        if raw_eligible.dtype != bool and not np.isin(raw_eligible, [0, 1]).all():
            raise ValueError("eligible must be boolean or contain only 0 and 1")
        valid = raw_eligible.astype(bool, copy=False)
    if not 0 <= min_history_coverage <= 1:
        raise ValueError("min_history_coverage must be between 0 and 1")
    if max_persistence_iou is not None and not 0 <= max_persistence_iou <= 1:
        raise ValueError("max_persistence_iou must be None or between 0 and 1")
    if not np.isfinite(min_spacing_minutes) or min_spacing_minutes < 0:
        raise ValueError("min_spacing_minutes must be finite and non-negative")
    if max_per_condition is not None:
        limits = max_per_condition.values() if isinstance(max_per_condition, Mapping) else [max_per_condition]
        if any(not isinstance(n, (int, np.integer)) or n < 0 for n in limits):
            raise ValueError("max_per_condition limits must be non-negative integers")
    if queued.shape[1] == 0:
        raise ValueError("labels must have at least one link")
    if condition_policy not in {"strict", "contract"}:
        raise ValueError("condition_policy must be 'strict' or 'contract'")
    if spacing_policy not in {"within_condition", "shared"}:
        raise ValueError("spacing_policy must be 'within_condition' or 'shared'")

    # Prefix sums keep the chronological scan inexpensive on all 273 days.
    any_queue = queued.any(axis=1)
    queue_prefix = np.r_[0, np.cumsum(any_queue, dtype=np.int64)]
    coverage_prefix = np.r_[0, np.cumsum(valid.sum(axis=1), dtype=np.int64)]
    five_minutes_ns = pd.Timedelta(5.0, unit="m").value
    gaps = np.r_[0, np.cumsum(np.diff(times.asi8) != five_minutes_ns)]
    days = times.normalize()
    spacing = pd.Timedelta(float(min_spacing_minutes), unit="m")
    last_origin: dict[str, pd.Timestamp] = {}
    counts: dict[str, int] = {"queue_onset": 0, "queue_ongoing": 0}
    rows: list[dict] = []
    limits_by_condition = {
        c: max_per_condition.get(c) if isinstance(max_per_condition, Mapping) else max_per_condition
        for c in counts
    }
    for origin in range(12, len(times) - 6):
        if all(limits_by_condition[c] is not None and counts[c] >= limits_by_condition[c] for c in counts):
            break
        start, stop = origin - 12, origin + 7
        if days[start] != days[stop - 1] or gaps[stop - 1] != gaps[start]:
            continue
        history_stop = origin + int(history_includes_origin)
        history_n = history_stop - start
        coverage = (coverage_prefix[history_stop] - coverage_prefix[start]) / (history_n * queued.shape[1])
        if coverage < min_history_coverage:
            continue
        has_target_queue = queue_prefix[stop] > queue_prefix[origin + 1]
        if require_horizon_queue and not has_target_queue:
            continue
        has_history_queue = queue_prefix[history_stop] > queue_prefix[start]
        at_idx = history_stop - 1
        if condition_policy == "contract":
            established = bool((queued[start:history_stop].sum(axis=0) >= 2).any())
            condition = "queue_ongoing" if established else "queue_onset"
        elif not has_history_queue:
            condition = "queue_onset"
        elif any_queue[at_idx]:
            condition = "queue_ongoing"
        else:
            continue
        limit = max_per_condition.get(condition) if isinstance(max_per_condition, Mapping) else max_per_condition
        if limit is not None and counts[condition] >= limit:
            continue
        time = times[origin]
        clock = "shared" if spacing_policy == "shared" else condition
        if clock in last_origin and time - last_origin[clock] < spacing:
            continue
        future = queued[origin + 1:stop]
        future_valid = valid[origin + 1:stop]
        if not future_valid.any():
            continue
        persistence = np.broadcast_to(queued[at_idx], future.shape)
        inter = np.count_nonzero(persistence & future & future_valid)
        union = np.count_nonzero((persistence | future) & future_valid)
        persistence_iou = inter / union if union else 1.0
        if max_persistence_iou is not None and persistence_iou > max_persistence_iou:
            continue
        queued_steps = np.flatnonzero(future.any(axis=1))
        first_step = int(queued_steps[0] + 1) if len(queued_steps) else 0
        rows.append({
            "origin_idx": origin,
            "origin_time": time,
            "forecast_origin": time,
            "window_type": condition,
            "condition": condition,
            "date": str(time.date()),
            "first_queue_horizon": first_step * 5,
            "first_queue_step": first_step,
            "history_coverage": float(coverage),
            "persistence_iou": float(persistence_iou),
            "n_queue_at_origin": int(queued[origin].sum()),
            "n_queue_horizon": int(future.sum()),
            "n_eligible_horizon": int(future_valid.sum()),
        })
        last_origin[clock] = time
        counts[condition] += 1
    result = pd.DataFrame(rows, columns=_COLUMNS)
    result.attrs.update({
        "selector": "chronological_public_policy_proxy",
        "seed": int(seed),
        "history_includes_origin": bool(history_includes_origin),
        "condition_policy": condition_policy,
        "spacing_policy": spacing_policy,
        "minimum_history_coverage": float(min_history_coverage),
        "maximum_persistence_iou": max_persistence_iou,
        "minimum_spacing_minutes": float(min_spacing_minutes),
        "organizer_implementation_available": False,
    })
    return result


def select_rolling_windows(
    labels,
    timestamps,
    eligible,
    *,
    block_days: int = 30,
    stride_days: int = 7,
    seed: int = 42,
) -> pd.DataFrame:
    """Union quota-based selectors over rolling blocks and calendar months.

    Every block uses five windows per condition, a shared six-hour clock, the
    competition-contract established-queue rule, and published T-60..T-5
    history. Rolling blocks begin every ``stride_days`` from the first date,
    plus each calendar-month start; they span ``block_days`` and are clipped
    at the input split boundary. Separate calendar-month selections stop
    exactly at the next month boundary and mark their rows ``official_like``.

    The union retains global ``origin_idx`` values and deduplicates by origin
    and condition. ``n_selection_blocks`` counts contributing scans. Spacing
    holds within each source block; overlapping blocks deliberately augment
    training with origins closer than six hours. Use ``official_like`` rows
    for monthly CV, and a separate full-split ``select_windows(max_per_condition=5)``
    call when auditing the released train origins. Never compare union.head(5)
    to the released index.
    """
    queued = np.asarray(labels)
    valid = np.asarray(eligible)
    if queued.ndim != 2 or valid.shape != queued.shape:
        raise ValueError("labels and eligible must share shape [time, link]")
    times = pd.DatetimeIndex(pd.to_datetime(timestamps)).as_unit("ns")
    if len(times) != len(queued):
        raise ValueError("timestamps length must equal labels.shape[0]")
    if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("timestamps must be non-null and strictly increasing")
    for name, value in (("block_days", block_days), ("stride_days", stride_days)):
        if not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    extra = ["official_like", "n_selection_blocks"]
    if len(times) == 0:
        empty = pd.DataFrame(columns=_COLUMNS + extra)
        empty["official_like"] = empty["official_like"].astype(bool)
        return empty
    first_day, last_day = times[0].normalize(), times[-1].normalize()
    month_starts = pd.date_range(first_day.replace(day=1), last_day, freq="MS")
    rolling_starts = set(pd.date_range(first_day, last_day, freq=f"{stride_days}D"))
    rolling_starts.update(month_starts)
    scans = [(start, start + pd.Timedelta(float(block_days), unit="D"), False)
             for start in sorted(rolling_starts)]
    scans += [(start, start + pd.offsets.MonthBegin(1), True) for start in month_starts]
    parts = []
    n_nonempty_scans = 0
    for start, end, official_like in scans:
        lo = int(times.searchsorted(start, side="left"))
        hi = int(times.searchsorted(end, side="left"))
        if hi - lo < 19:
            continue
        selected = select_windows(
            queued[lo:hi], times[lo:hi], valid[lo:hi], seed=seed,
            max_per_condition=5, history_includes_origin=False,
            condition_policy="contract", spacing_policy="shared",
        )
        if selected.empty:
            continue
        selected["origin_idx"] = selected.origin_idx.astype(np.int64) + lo
        selected["official_like"] = bool(official_like)
        parts.append(selected)
        n_nonempty_scans += 1
    if not parts:
        result = pd.DataFrame(columns=_COLUMNS + extra)
        result["official_like"] = result["official_like"].astype(bool)
    else:
        pooled = pd.concat(parts, ignore_index=True)
        keys = ["origin_idx", "condition"]
        grouped = pooled.groupby(keys, sort=False)
        flags = grouped.official_like.max().rename("official_like")
        multiplicity = grouped.size().rename("n_selection_blocks")
        result = (pooled.drop(columns="official_like").drop_duplicates(keys)
                  .merge(flags, on=keys, how="left", validate="one_to_one")
                  .merge(multiplicity, on=keys, how="left", validate="one_to_one")
                  .sort_values(["origin_idx", "condition"], kind="stable").reset_index(drop=True))
        result["official_like"] = result.official_like.astype(bool)
    result.attrs.update({
        "selector": "rolling_quota_public_policy_proxy",
        "block_days": int(block_days), "stride_days": int(stride_days),
        "seed": int(seed), "n_nonempty_selection_blocks": n_nonempty_scans,
        "union_spacing_enforced": False,
        "monthly_official_like": True,
    })
    return result
