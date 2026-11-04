"""Small, disjoint-fold calibration of expected-IoU queue decoding.

Inference accepts only predicted probabilities and a JSON-serializable config.
Future eligibility is used exclusively to score calibration targets. Calibration
window IDs and evaluation IDs are mandatory so accidental fold overlap fails.
"""
from __future__ import annotations

import copy
import json

import numpy as np

from .decode import expected_iou_decode


_REGULARIZATION = 0.0025
_IDENTITY = {"temperature": 1.0, "intercept": 0.0, "cardinality_scale": 1.0,
             "spatial_mode": "none", "gap_penalty": 0.0}


def _probability(values) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] != 6 or p.shape[1] == 0:
        raise ValueError("probability must have shape [6, link] with at least one link")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be finite and between 0 and 1")
    return p


def _transform(p, temperature, intercept):
    if temperature == 1.0 and intercept == 0.0:
        return p
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    logits = np.log(clipped) - np.log1p(-clipped)
    logits = logits / temperature + intercept
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))


def decode_calibrated(probability, config, *, onset: bool = False) -> np.ndarray:
    """Decode all cells without future observations, eligibility, or targets.

    Onset returns zeros at T+5..T+25 and localizes at T+30. ``interval`` mode
    selects a single contiguous interval in topology-ordered links, with extent
    derived from exact expected-IoU cardinality and its fitted scale. For each
    possible start, probability mass is penalized for low-probability gaps
    below the ordinary top-m cutoff. Multiple separated queues remain possible
    under the identity ``none`` mode; interval mode is a calibration candidate.
    """
    values = _probability(probability)
    if "onset" in config and bool(config["onset"]) != bool(onset):
        raise ValueError("onset must match the fitted calibration condition")
    parameters = {k: config.get(k, value) for k, value in _IDENTITY.items()}
    temperature, intercept = float(parameters["temperature"]), float(parameters["intercept"])
    scale, penalty = float(parameters["cardinality_scale"]), float(parameters["gap_penalty"])
    mode = parameters["spatial_mode"]
    if not np.isfinite([temperature, intercept, scale, penalty]).all() or temperature <= 0 or scale <= 0 or penalty < 0:
        raise ValueError("invalid calibration parameters")
    if mode not in {"none", "interval"} or (mode == "interval" and not onset):
        raise ValueError("interval localization is available only for onset")
    p = _transform(values[-1] if onset else values, temperature, intercept)
    baseline = expected_iou_decode(p, force_nonempty=True)
    m = int(np.clip(np.floor(int(baseline.sum()) * scale + 0.5), 1, p.size))
    if mode == "none" and scale == 1.0:
        decoded = baseline
    elif mode == "none":
        decoded = np.zeros(p.shape, dtype=np.uint8)
        order = np.argsort(-p.reshape(-1), kind="stable")
        decoded.reshape(-1)[order[:m]] = 1
    else:
        order = np.argsort(-p, kind="stable")
        cutoff = float(p[order[m - 1]])
        utility = p - penalty * np.maximum(cutoff - p, 0.0)
        prefix = np.r_[0.0, np.cumsum(utility)]
        start = int(np.argmax(prefix[m:] - prefix[:-m]))
        decoded = np.zeros(p.shape, dtype=np.uint8)
        decoded[start:start + m] = 1
    if onset:
        result = np.zeros(values.shape, dtype=np.uint8)
        result[-1] = decoded
        return result
    return decoded.astype(np.uint8, copy=False)


def _candidate_grid(onset):
    candidates = [copy.deepcopy(_IDENTITY)]
    for temperature in (0.8, 1.0, 1.25):
        for intercept in (-0.25, 0.0, 0.25):
            if temperature == 1.0 and intercept == 0.0:
                continue
            candidates.append({**_IDENTITY, "temperature": temperature, "intercept": intercept})
    for scale in (0.8, 1.2):
        candidates.append({**_IDENTITY, "cardinality_scale": scale})
    if onset:
        for scale in (0.8, 1.0, 1.2):
            for penalty in (0.0, 0.5):
                candidates.append({**_IDENTITY, "cardinality_scale": scale,
                                   "spatial_mode": "interval", "gap_penalty": penalty})
    return candidates


def _complexity(parameters):
    return (np.log(parameters["temperature"]) / np.log(1.25)) ** 2 + (parameters["intercept"] / .25) ** 2 + ((parameters["cardinality_scale"] - 1) / .2) ** 2 + (1 + parameters["gap_penalty"] if parameters["spatial_mode"] != "none" else 0)


def _ids(values, name):
    if values is None:
        raise ValueError("provide calibration_indices and evaluation_indices to verify disjoint folds")
    result = []
    for value in values:
        if isinstance(value, np.generic):
            value = value.item()
        if not isinstance(value, (str, int, float, bool)) or value is None:
            value = str(value)
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"{name} must contain finite, stable window IDs")
        result.append(value)
    keys = [json.dumps(x, sort_keys=True) for x in result]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{name} must contain unique window IDs")
    return result, set(keys)


def fit_cardinality_calibration(
    probability_list,
    truth_list,
    eligible_list,
    *,
    onset: bool = False,
    calibration_indices=None,
    evaluation_indices=None,
) -> dict:
    """Fit a predefined small grid on a disjoint month, shrinking to identity.

    Both ID lists are required. Supply one unique calibration window ID per
    probability array and IDs for the distinct evaluation month. Overlap is
    rejected. IDs must have a common namespace across folds, e.g. panel+origin;
    separate per-fold row numbers cannot establish disjointness. The caller is
    responsible for supplying out-of-sample probabilities and genuine month
    assignments; the fitter cannot verify their provenance from arrays alone.

    The objective is mean eligible-cell window IoU minus 0.0025 times parameter
    distance from identity. Stable grid order resolves ties toward identity.
    Only the chosen fixed parameters are used at inference. No per-window
    cardinality targets, quality masks, or target arrays are retained.
    """
    probabilities, truths, eligibility = list(probability_list), list(truth_list), list(eligible_list)
    if not probabilities or len(probabilities) != len(truths) or len(probabilities) != len(eligibility):
        raise ValueError("calibration lists must have the same nonzero length")
    calibration_ids, calibration_keys = _ids(calibration_indices, "calibration_indices")
    evaluation_ids, evaluation_keys = _ids(evaluation_indices, "evaluation_indices")
    if len(calibration_ids) != len(probabilities) or not evaluation_ids:
        raise ValueError("provide one ID per calibration window and at least one held-out evaluation ID")
    if calibration_keys & evaluation_keys:
        raise ValueError("calibration and evaluation window IDs must be disjoint")
    prepared = []
    for probability, truth, eligible in zip(probabilities, truths, eligibility):
        p = _probability(probability)
        y, mask = np.asarray(truth), np.asarray(eligible)
        if y.shape != p.shape or mask.shape != p.shape:
            raise ValueError("probability, truth, and eligible shapes must match")
        if not np.isin(y, [0, 1]).all() or not np.isin(mask, [0, 1]).all():
            raise ValueError("truth and eligible must be boolean or binary")
        prepared.append((p, y.astype(bool), mask.astype(bool)))
    trials = []
    best, best_objective = None, -np.inf
    for parameters in _candidate_grid(onset):
        scores = []
        for p, y, mask in prepared:
            pred = decode_calibrated(p, parameters, onset=onset).astype(bool)
            union = np.count_nonzero((pred | y) & mask)
            intersection = np.count_nonzero(pred & y & mask)
            scores.append(0.0 if not mask.any() else (intersection / union if union else 1.0))
        score = float(np.mean(scores))
        regularizer = float(_REGULARIZATION * _complexity(parameters))
        objective = score - regularizer
        trials.append({**parameters, "mean_iou": score, "regularization_penalty": regularizer,
                       "objective": objective})
        if objective > best_objective + 1e-12:
            best, best_objective = copy.deepcopy(parameters), objective
    config = {"version": 1, "onset": bool(onset), **best}
    selected = next(t for t in trials if all(t[k] == best[k] for k in _IDENTITY))
    config["fit"] = {
        "n_windows": len(prepared), "mean_iou": selected["mean_iou"],
        "identity_mean_iou": trials[0]["mean_iou"], "objective": best_objective,
        "regularization": _REGULARIZATION, "n_candidates": len(trials),
        "calibration_indices": calibration_ids, "evaluation_indices": evaluation_ids,
        "disjointness_verified": True, "candidate_scores": trials,
    }
    return config
