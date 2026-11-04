"""Train-only queue proxies from the released noisy observation layer.

The official target uses the unreleased underlying speed.  These labels are
therefore proxies, including when ``method='raw'``.  Temporal smoothing here is
allowed only for training targets; it must never be applied to forecast inputs.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def make_labels(speed, vf, method: str = "raw") -> np.ndarray:
    """Return a boolean ``[time, link]`` queue proxy, using ``speed <= .60 * vf``.

    ``vf`` may be scalar, a per-link vector, or broadcastable to ``speed``.
    Non-finite speeds, non-finite free speeds, and non-positive free speeds
    produce False; callers must separately retain an observation/eligibility
    mask to keep such cells out of training and evaluation.

    ``median3`` (also ``temporal_median``) thresholds a centered, three-slot
    temporal median wherever all three observations exist; edges and gaps
    retain the raw observation.  It is a noncausal *label* sensitivity
    analysis, never a feature transformation.  Missing original observations
    remain unknown/False, rather than becoming fabricated training labels.
    """
    values = np.asarray(speed, dtype=float)
    if values.ndim != 2:
        raise ValueError("speed must have shape [time, link]")
    try:
        free_speed = np.broadcast_to(np.asarray(vf, dtype=float), values.shape)
    except ValueError as exc:
        raise ValueError("vf must broadcast to speed's [time, link] shape") from exc
    valid = np.isfinite(values) & np.isfinite(free_speed) & (free_speed > 0)
    if method == "raw":
        denoised = values
    elif method in {"median3", "temporal_median", "temporal_median3"}:
        before = np.full_like(values, np.nan)
        after = np.full_like(values, np.nan)
        before[1:] = values[:-1]
        after[:-1] = values[1:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            neighborhood = np.stack((before, values, after))
            median = np.nanmedian(neighborhood, axis=0)
        denoised = np.where(np.isfinite(neighborhood).all(axis=0), median, values)
    else:
        raise ValueError("method must be 'raw' or 'median3' (temporal_median)")
    return valid & (denoised <= 0.60 * free_speed)


def make_selector_labels(speed, vf, method: str = "interp_median3", *, timestamps=None) -> np.ndarray:
    """Return gap-filled, train-only proxy labels for window selection.

    This function is deliberately separate from the raw supervised-label API.
    It approximates the complete underlying state used by the hidden selector,
    whose persistence safeguard otherwise mistakes missing current queue cells
    for queue evolution.  Interpolation and centered smoothing use future
    *training labels* and must never be used to construct forecast features.

    Methods: ``raw``/``median3`` preserve :func:`make_labels`; ``ffill`` forward
    fills observations; ``interp`` linearly interpolates at most three slots
    from each side of a gap; ``interp_median3`` then applies the existing
    three-slot median rule.  An eight-slot interior gap therefore retains its
    middle two missing slots.  Leading/trailing gaps fill at most three slots.

    Supply ``timestamps`` to prevent interpolation/median neighborhoods crossing
    UTC midnight or a missing time slot.  Without timestamps, the input is one
    contiguous segment.  Float32 inputs retain float32 interpolation arithmetic;
    thresholding and median arithmetic use the existing float64 label routine.
    Eligibility must remain separate: do not erase an observed queue merely
    because its quality flag makes it ineligible for scoring.
    """
    if method in {"raw", "median3", "temporal_median", "temporal_median3"}:
        return make_labels(speed, vf, method)
    if method not in {"ffill", "interp", "interp_median3"}:
        raise ValueError("unknown selector-label method")
    original = np.asarray(speed)
    if original.ndim != 2:
        raise ValueError("speed must have shape [time, link]")
    dtype = np.float32 if np.issubdtype(original.dtype, np.floating) and original.dtype.itemsize <= 4 else np.float64
    values = original.astype(dtype, copy=True)
    values[~np.isfinite(values)] = np.nan
    try:
        free_speed = np.broadcast_to(np.asarray(vf), values.shape)
    except ValueError as exc:
        raise ValueError("vf must broadcast to speed's [time, link] shape") from exc
    if timestamps is None:
        boundaries = np.array([0, len(values)])
    else:
        times = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True)).as_unit("ns")
        if len(times) != len(values):
            raise ValueError("timestamps length must equal speed.shape[0]")
        if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
            raise ValueError("timestamps must be non-null and strictly increasing")
        breaks = (np.diff(times.asi8) != 300_000_000_000) | (times.normalize()[1:] != times.normalize()[:-1])
        boundaries = np.r_[0, np.flatnonzero(breaks) + 1, len(values)]
    result = np.zeros(values.shape, dtype=bool)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if stop <= start:
            continue
        frame = pd.DataFrame(values[start:stop])
        if method == "ffill":
            filled = frame.ffill().to_numpy()
        else:
            filled = frame.interpolate(method="linear", axis=0, limit=3,
                                       limit_direction="both").to_numpy()
        label_method = "median3" if method == "interp_median3" else "raw"
        result[start:stop] = make_labels(filled, free_speed[start:stop], label_method)
    return result
