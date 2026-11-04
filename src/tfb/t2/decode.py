"""Deterministic maximum expected-IoU decoding of queue probabilities.

For a fixed prediction size, selecting the highest probabilities is optimal
under independent Bernoulli labels. If the selected set has size ``m``, its
intersection is independent of the positives outside it. Consequently its
expected IoU is exactly ``sum(p_selected) * E[1 / (m + positives_outside)]``.
The outside count has a Poisson-binomial distribution, updated in O(n**2)
time and O(n) memory while visiting all possible prediction sizes.

Only caller-provided eligible cells participate. Eligibility must come from
released metadata or a local scorer, never hidden future observations.
"""
from __future__ import annotations

import numpy as np


EXACT_CELL_LIMIT = 2048


def expected_iou_topm_scores(sorted_probabilities: np.ndarray) -> np.ndarray:
    """Expected IoUs for m=0..n, with probabilities sorted descending.

The empty-set score is the probability that every label is zero, matching
the official empty-union convention. For more than 2048 cells we use the
usual ratio-of-expectations approximation, ``sum(p[:m]) / (m + sum(p[m:]))``.
All panels except D7_I405_N fit the exact calculation; that panel has 2,112 cells.
"""
    p = np.asarray(sorted_probabilities, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    n = p.size
    scores = np.zeros(n + 1, dtype=np.float64)
    scores[0] = np.prod(1.0 - p)
    if n == 0:
        return scores
    prefix = np.cumsum(p)
    if n > EXACT_CELL_LIMIT:
        m = np.arange(1, n + 1, dtype=np.float64)
        scores[1:] = prefix / (m + prefix[-1] - prefix)
        return scores
    outside_distribution = np.ones(1, dtype=np.float64)
    for m in range(n, 0, -1):
        denominator = m + np.arange(outside_distribution.size, dtype=np.float64)
        scores[m] = prefix[m - 1] * np.sum(outside_distribution / denominator)
        probability = p[m - 1]
        expanded = np.zeros(outside_distribution.size + 1, dtype=np.float64)
        expanded[:-1] += outside_distribution * (1.0 - probability)
        expanded[1:] += outside_distribution * probability
        outside_distribution = expanded
    return scores


def expected_iou_decode(
    probabilities: np.ndarray,
    eligible: np.ndarray | None = None,
    force_nonempty: bool = True,
) -> np.ndarray:
    """Choose the top-m set maximizing expected space-time IoU.

    Shape is preserved. Equal probabilities retain original flattened order;
    equal expected scores prefer fewer positive cells. If no cells are
    eligible, the result is empty even when ``force_nonempty`` is true.
    """
    values = np.asarray(probabilities, dtype=np.float64)
    if eligible is None:
        mask = np.ones(values.shape, dtype=bool)
    else:
        mask = np.asarray(eligible, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("eligible must have the same shape as probabilities")
    result = np.zeros(values.shape, dtype=np.uint8)
    indices = np.flatnonzero(mask.reshape(-1))
    if indices.size == 0:
        return result
    p = values.reshape(-1)[indices]
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("eligible probabilities must be finite and lie in [0, 1]")
    order = np.argsort(-p, kind="stable")
    scores = expected_iou_topm_scores(p[order])
    offset = 1 if force_nonempty else 0
    count = offset + int(np.argmax(scores[offset:]))
    result.reshape(-1)[indices[order[:count]]] = 1
    return result
