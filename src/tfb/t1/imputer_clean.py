"""Speed imputer for the Task 2 hybrid labels: the pooled LightGBM of tfb.t1.train (seed 42, rounds cap 20000, early
stopping on the same 5% split, the same sampled train cells), speed model only, fitted on train rows only.
"""
from __future__ import annotations
import json
from pathlib import Path
from .. import io
from . import train as T


def fit(out: Path, rounds=20000, seed=42) -> Path:
    """Train (or reuse) out/lgbm_speed.txt. Restores tfb.t1.train's globals afterwards."""
    out = Path(out); model = out / "lgbm_speed.txt"
    if model.exists():
        return model
    out.mkdir(parents=True, exist_ok=True)
    saved = T.SEED, T.ROUNDS, dict(T.PARAMS), T.TRANS
    T.SEED, T.ROUNDS, T.TRANS = seed, rounds, 0; T.PARAMS["seed"] = seed
    try:
        X, ys, w, rng = T.build_training(None, list(io.panels()))
        sets = T.make_sets(X, ys["speed"], rng, w); del X
        booster = T.fit_sets(*sets, {"speed": ys["speed"]})["speed"]
    finally:
        T.SEED, T.ROUNDS, T.TRANS = saved[0], saved[1], saved[3]; T.PARAMS.clear(); T.PARAMS.update(saved[2])
    tmp = out / "lgbm_speed.tmp.txt"; booster.save_model(str(tmp)); tmp.replace(model)
    (out / "meta.json").write_text(json.dumps(dict(rounds=rounds, best_iteration=booster.best_iteration, seed=seed, panels=list(io.panels()),
                                                   rows="train target cells only; no validation/private rows"), indent=1))
    return model
