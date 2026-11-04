"""Deterministic cell classifiers and condition/family-balanced CV metrics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gc

import lightgbm as lgb
import numpy as np
import pandas as pd

from .decode import expected_iou_decode
from .features import location_feature_mask


@dataclass
class Classifier:
    booster: object
    mask: np.ndarray
    constant: float | None = None
    threads: int = 2

    def predict(self, x):
        if self.constant is not None:
            return np.full(len(x), self.constant)
        selected = x if self.mask.all() else x[:, self.mask]
        return self.booster.predict(selected, num_threads=self.threads)

    def save(self, path: Path):
        if self.booster is not None:
            self.booster.save_model(str(path))
        np.save(path.with_suffix(".mask.npy"), self.mask)


def fit_classifier(x, y, weights, names, *, no_location=False, seed=42, overrides=None,
                   rounds=350, threads=2, model_path: Path | None = None):
    mask = ~location_feature_mask(names) if no_location else np.ones(len(names), bool)
    if len(y) == 0:
        raise ValueError("Cannot fit classifier with no training labels")
    if len(np.unique(y)) < 2:
        return Classifier(None, mask, float(np.mean(y)), threads)
    params = dict(
        objective="binary", metric="binary_logloss", learning_rate=.045,
        num_leaves=31, max_depth=-1, min_data_in_leaf=100,
        max_bin=127, feature_fraction=.9, bagging_fraction=1.0,
        lambda_l2=10., verbosity=-1, seed=seed, feature_fraction_seed=seed,
        bagging_seed=seed, data_random_seed=seed, deterministic=True,
        force_col_wise=True, num_threads=threads,
    )
    params.update(overrides or {})
    selected = x if mask.all() else x[:, mask]
    dataset = lgb.Dataset(selected, label=y, weight=weights,
                          feature_name=[n for n, take in zip(names, mask) if take],
                          free_raw_data=True)
    booster = lgb.train(params, dataset, num_boost_round=rounds)
    model = Classifier(booster, mask, threads=threads)
    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        importance = pd.DataFrame({"feature": [n for n, take in zip(names, mask) if take],
                                   "gain": booster.feature_importance("gain")})
        importance.sort_values("gain", ascending=False).to_csv(model_path.with_suffix(".importance.csv"), index=False)
    del dataset
    gc.collect()
    return model


def score_window(prediction, truth, eligible):
    mask = np.asarray(eligible, bool)
    if not mask.any():
        return 0.0
    prediction, truth = np.asarray(prediction, bool), np.asarray(truth, bool)
    intersection = np.count_nonzero(prediction & truth & mask)
    union = np.count_nonzero((prediction | truth) & mask)
    return float(intersection / union if union else 1.)


def aggregate_scores(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, by_model in detail.groupby("model", sort=True):
        panels = []
        for panel, group in by_model.groupby("panel", sort=True):
            values = {}
            for condition, label in [("queue_onset", "onset"), ("queue_ongoing", "ongoing")]:
                subset = group[group.condition.eq(condition)]
                values[label] = float(subset.iou.mean()) if len(subset) else np.nan
            ongoing = group[group.condition.eq("queue_ongoing")]
            values["shift_ongoing"] = float(np.average(ongoing.iou, weights=ongoing.shift_weight)) if len(ongoing) else np.nan
            values["total"] = (values["onset"] + values["ongoing"]) / 2
            values["shift_total"] = (values["onset"] + values["shift_ongoing"]) / 2
            row = {"model": model, "panel": panel, "family": panel.rsplit("_", 1)[0],
                   "n_onset": int(group.condition.eq("queue_onset").sum()),
                   "n_ongoing": int(group.condition.eq("queue_ongoing").sum()), **values}
            rows.append(row)
            panels.append(row)
        panel_frame = pd.DataFrame(panels)
        columns = ["onset", "ongoing", "total", "shift_ongoing", "shift_total"]
        family = panel_frame.groupby("family")[columns].mean()
        rows.append({"model": model, "panel": "OVERALL", "family": "ALL",
                     "n_onset": int(by_model.condition.eq("queue_onset").sum()),
                     "n_ongoing": int(by_model.condition.eq("queue_ongoing").sum()),
                     **family.mean().to_dict()})
    return pd.DataFrame(rows)


def decode_probabilities(probability, *, onset=False):
    """Deployable decoding, never consulting future eligibility or observations."""
    p = np.asarray(probability).copy()
    if onset:
        result = np.zeros_like(p, dtype=np.uint8)
        result[-1] = expected_iou_decode(p[-1], force_nonempty=True)
        return result
    return expected_iou_decode(p, force_nonempty=True)
