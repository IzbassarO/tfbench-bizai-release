"""Training-only conditional bottleneck priors for selection-matched onset.

Calendar lookup at the target time is causal: its table is fitted on permitted
train dates, then evaluated using only the known target date and time of day.
No observation from the forecast window's future enters this feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fit_onset_prior(
    labels,
    eligible,
    timestamps,
    *,
    end_date,
    smoothing_minutes: float = 60,
) -> tuple[np.ndarray, dict]:
    """Fit ``P(link queued | corridor onset event, daytype, time of day)``.

    Returns ``(prior, info)``. ``prior`` is float32 ``[2, 288, n_links]`` with
    daytype 0=weekday and 1=weekend.  ``info`` includes raw/smoothed event counts
    by daytype and five-minute time bin, as well as total/monthly event counts.

    An event is a queued slot preceded by at least 18 consecutive queue-free
    five-minute slots on the same UTC date.  Queue existence uses all observed
    queue proxies in ``labels`` (unknown observations should already be False),
    consistent with strict window reconstruction.  ``eligible`` controls the
    per-link target numerator and denominator, so unscoreable cells never
    become negative labels.  Events are included only if their timestamp is
    *strictly before* ``end_date``.  Set ``end_date='2031-01-01'`` for Jan-Feb CV.

    Local event targets are pooled within plus/minus ``smoothing_minutes`` of
    each time bin, wrapping at midnight, separately for weekdays/weekends.
    Four Beta pseudoevents shrink each link's local rate toward its global
    frequency across all fitted onset events.  This has a conditional-event
    interpretation; the ordinary all-timestep queue frequency would dilute the
    small but guaranteed onset queue with a large number of non-event zeros.
    With no fitted events, every prior entry is zero.
    """
    raw_labels = np.asarray(labels)
    if raw_labels.ndim != 2 or raw_labels.shape[1] == 0:
        raise ValueError("labels must have shape [time, link] with at least one link")
    if raw_labels.dtype != bool and not np.isin(raw_labels, [0, 1]).all():
        raise ValueError("labels must be boolean or contain only 0 and 1")
    queued = raw_labels.astype(bool, copy=False)
    raw_eligible = np.asarray(eligible)
    if raw_eligible.shape != queued.shape:
        raise ValueError("eligible must have the same shape as labels")
    if raw_eligible.dtype != bool and not np.isin(raw_eligible, [0, 1]).all():
        raise ValueError("eligible must be boolean or contain only 0 and 1")
    valid = raw_eligible.astype(bool, copy=False)
    times = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True)).as_unit("ns")
    if len(times) != len(queued):
        raise ValueError("timestamps length must match labels.shape[0]")
    if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("timestamps must be non-null and strictly increasing")
    if len(times) and ((times.minute % 5 != 0) | (times.second != 0) | (times.microsecond != 0) | (times.nanosecond != 0)).any():
        raise ValueError("timestamps must lie on the five-minute calendar grid")
    if not np.isfinite(smoothing_minutes) or not 0 <= smoothing_minutes <= 720:
        raise ValueError("smoothing_minutes must be finite and between 0 and 720")
    cutoff = pd.Timestamp(end_date)
    if pd.isna(cutoff):
        raise ValueError("end_date must be a valid date or timestamp")
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")

    any_queue = queued.any(axis=1)
    prefix = np.r_[0, np.cumsum(any_queue, dtype=np.int64)]
    gaps = np.r_[0, np.cumsum(np.diff(times.asi8) != 300_000_000_000)]
    dates = times.normalize()
    candidates = np.flatnonzero(any_queue)
    candidates = candidates[candidates >= 18]
    if len(candidates):
        previous = candidates - 18
        keep = (
            (prefix[candidates] == prefix[previous])
            & (gaps[candidates] == gaps[previous])
            & (dates[candidates] == dates[previous])
            & (times[candidates] < cutoff)
        )
        event_indices = candidates[keep]
    else:
        event_indices = np.array([], dtype=np.int64)

    n_links = queued.shape[1]
    positive = np.zeros((2, 288, n_links), dtype=np.float64)
    observed = np.zeros_like(positive)
    counts = np.zeros((2, 288), dtype=np.int64)
    event_times = times[event_indices]
    daytype = np.asarray(event_times.dayofweek >= 5, dtype=np.int64)
    slots = np.asarray((event_times.hour * 60 + event_times.minute) // 5, dtype=np.int64)
    targets = queued[event_indices] & valid[event_indices]
    known = valid[event_indices]
    np.add.at(positive, (daytype, slots), targets)
    np.add.at(observed, (daytype, slots), known)
    np.add.at(counts, (daytype, slots), 1)

    global_numerator = targets.sum(axis=0, dtype=np.float64)
    global_denominator = known.sum(axis=0, dtype=np.float64)
    global_rate = np.divide(global_numerator, global_denominator,
                            out=np.zeros(n_links), where=global_denominator > 0)
    radius = int(np.floor(float(smoothing_minutes) / 5.0))
    # At a full half-day radius, +/-144 denotes the same bin twice; use each
    # of the 288 calendar bins once rather than artificially double-counting.
    offsets = range(-radius, radius + 1) if radius < 144 else range(-144, 144)
    local_positive = np.zeros_like(positive)
    local_observed = np.zeros_like(observed)
    local_counts = np.zeros_like(counts)
    for offset in offsets:
        local_positive += np.roll(positive, offset, axis=1)
        local_observed += np.roll(observed, offset, axis=1)
        local_counts += np.roll(counts, offset, axis=1)
    strength = 4.0
    prior = (local_positive + strength * global_rate) / (local_observed + strength)
    months = event_times.strftime("%Y-%m")
    month_names, month_counts = np.unique(months, return_counts=True)
    info = {
        "n_events": int(len(event_indices)),
        "n_events_by_daytype": {
            "weekday": int(counts[0].sum()), "weekend": int(counts[1].sum())
        },
        "n_events_by_month": {str(m): int(n) for m, n in zip(month_names, month_counts)},
        "event_counts": counts,
        "smoothed_event_counts": local_counts.astype(np.float32),
        "global_link_frequency": global_rate.astype(np.float32),
        "shrinkage_events": strength,
        "end_date": cutoff.isoformat(),
        "smoothing_minutes": float(smoothing_minutes),
    }
    return prior.astype(np.float32), info
