"""Fail-closed validation of v2 prediction packs and comparison cohorts."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _metadata(meta, columns):
    if not isinstance(meta, pd.DataFrame) or any(c not in meta for c in columns):
        raise ValueError("metadata must contain " + ", ".join(columns))
    if meta[list(columns)].isna().any().any():
        raise ValueError("metadata contains null window identifiers")
    if meta.window_id.astype(str).duplicated().any():
        raise ValueError("metadata window IDs must be unique")
    return meta[list(columns)].astype(str).reset_index(drop=True)


def validate_probability_pack(p, y, e, meta, stored_origin, stored_ids) -> None:
    """Validate row alignment and N×6×L predictions without changing arrays.

    The caller must additionally compare truth/eligibility with the source
    dataset arrays: matching IDs alone cannot establish target provenance.
    """
    p, y, e = np.asarray(p), np.asarray(y), np.asarray(e)
    if p.ndim != 3 or p.shape[0] == 0 or p.shape[1] != 6 or p.shape[2] == 0:
        raise ValueError("probability must have nonempty shape [window, 6, link]")
    if y.shape != p.shape or e.shape != p.shape:
        raise ValueError("probability, truth and eligibility shapes must match")
    try:
        good_p = np.isfinite(p).all() and not ((p < 0) | (p > 1)).any()
    except TypeError as exc:
        raise ValueError("probability must be numeric") from exc
    if not good_p:
        raise ValueError("probabilities must be finite and between 0 and 1")
    if not np.isin(y, [0, 1]).all() or not np.isin(e, [0, 1]).all():
        raise ValueError("truth and eligibility must be finite binary arrays")
    keys = _metadata(meta, ("window_id", "origin"))
    if len(keys) != len(p):
        raise ValueError("probability and metadata window counts differ")
    origin, ids = np.asarray(stored_origin), np.asarray(stored_ids)
    if origin.ndim != 1 or ids.ndim != 1:
        raise ValueError("stored origin and window IDs must be one-dimensional")
    if not np.array_equal(origin.astype(str), keys.origin.to_numpy()):
        raise ValueError("Probability/metadata origin mismatch")
    if not np.array_equal(ids.astype(str), keys.window_id.to_numpy()):
        raise ValueError("Probability/metadata window ID mismatch")


def validate_candidate_cohort(candidate_meta, baseline_meta) -> None:
    """Require identical ordered window IDs, origins and conditions.

    No implicit sorting or joins are performed: predictions must already use
    the same row order before probability blending and paired gate evaluation.
    """
    columns = ("window_id", "origin", "condition")
    candidate = _metadata(candidate_meta, columns)
    baseline = _metadata(baseline_meta, columns)
    if not candidate.equals(baseline):
        raise ValueError("candidate and baseline window cohorts/order differ")
