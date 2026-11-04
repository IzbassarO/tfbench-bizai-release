"""Clean Task 1 speed imputer for the Task 2 hybrid labels.

Rule (organizers, 2026-10-29): no data after T, also not indirectly through fitting. The recipe's v4t1 imputer is the
pooled LightGBM of tfb.t1.train with 5 transductive pseudo-masks over the whole validation/private months, so its
labels carry March/April observations into every Task 2 model. This imputer is the same LightGBM (seed 42, rounds cap
20000, early stopping on the same 5% split, the same sampled train cells) with no validation/private rows at all, speed
model only (the hybrid labels use speed only).

exclude_days: also leave these train days out (an honest pseudo-label simulation for tfb.t2b.march_refit).
test_splits/trans: transductive masks over these released splits' observed cells (tfb.t2b.hyb_clean_march: ("validation",)
= March, all <= T for every April window). Default none: the rule-(B) clean imputer.
"""
from __future__ import annotations
import json
from pathlib import Path
from .. import io
from . import train as T


def fit(out: Path, exclude_days=None, rounds=20000, seed=42, panels=None, test_splits=(), trans=0) -> Path:
    """Train (or reuse) out/lgbm_speed.txt. Restores tfb.t1.train's globals afterwards."""
    out = Path(out); model = out / "lgbm_speed.txt"
    if model.exists():
        return model
    out.mkdir(parents=True, exist_ok=True)
    saved = T.SEED, T.ROUNDS, dict(T.PARAMS), T.TRANS
    T.SEED, T.ROUNDS, T.TRANS = seed, rounds, trans if test_splits else 0; T.PARAMS["seed"] = seed
    try:
        X, ys, w, rng = T.build_training(exclude_days, list(panels or io.panels()), test_splits=tuple(test_splits))
        sets = T.make_sets(X, ys["speed"], rng, w); del X
        booster = T.fit_sets(*sets, {"speed": ys["speed"]})["speed"]
    finally:
        T.SEED, T.ROUNDS, T.TRANS = saved[0], saved[1], saved[3]; T.PARAMS.clear(); T.PARAMS.update(saved[2])
    tmp = out / "lgbm_speed.tmp.txt"; booster.save_model(str(tmp)); tmp.replace(model)
    (out / "meta.json").write_text(json.dumps(dict(
        rounds=rounds, best_iteration=booster.best_iteration, seed=seed, panels=list(panels or io.panels()),
        exclude_days=None if exclude_days is None else [int(d) for d in exclude_days],
        rows=f"train target cells + {trans} transductive masks over {list(test_splits)}" if test_splits
        else "train target cells only; no validation/private rows (no transductive masks)"), indent=1))
    return model
