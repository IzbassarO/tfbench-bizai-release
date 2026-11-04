"""Read-only access to family A's Task 2 CV windows/predictions (outputs/t2/v3/eval) and submission."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .. import io
from . import build, data

EVAL = io.OUTPUTS / "t2" / "v3" / "eval"


def windows(fold: str, panel: str, official_only: bool = True) -> dict:
    """family A eval windows for a panel (both conditions) as a train_at() bundle + family A arrays."""
    metas, arrs = [], {}
    for cond in ("queue_onset", "queue_ongoing"):
        z = np.load(EVAL / fold / f"{panel}_{cond}.npz", allow_pickle=True)
        csv = pd.read_csv(EVAL / fold / f"{panel}_{cond}.csv")
        keep = csv.official_like.to_numpy(bool) if official_only else np.ones(len(csv), bool)
        assert (csv.origin.to_numpy() == z["origin"]).all()
        csv = csv[keep].assign(condition=cond)
        metas.append(csv)
        for k in z.files:
            if k.startswith(("probability_", "prediction_")) or k in ("truth", "eligible"):
                arrs.setdefault(k, []).append(z[k][keep])
    meta = pd.concat(metas, ignore_index=True)
    B = build.train_at(panel, meta.origin.tolist())
    B["meta"] = B["meta"].assign(condition=meta.condition.to_numpy(), month=meta.origin.str[:7].to_numpy(),
                                 block_day=0, window_id=meta.window_id.to_numpy(), shift_weight=meta.shift_weight.to_numpy())
    from .. import cache
    o = data.downstream_order(panel, cache.load(panel, "train")["links"])  # family A arrays are in cache order
    B["codex"] = {k: np.concatenate(v)[..., o] for k, v in arrs.items() if len(v) == 2}
    assert (B["codex"]["truth"] == B["Y"]).all()
    return B


def submission(path=io.OUTPUTS / "t2" / "queue_submission.csv") -> pd.DataFrame:
    return pd.read_csv(path, dtype={"window_id": str, "timestamp": str, "link_id": str})


def grid(sub: pd.DataFrame, panel: str, split: str, wi: pd.DataFrame) -> np.ndarray:
    """family A binary predictions as [W, 6, L] in our downstream link order."""
    d = data.load(panel, split)
    pos = {l: i for i, l in enumerate(d["links"])}; wpos = {w: i for i, w in enumerate(wi.window_id)}
    s = sub[sub.window_id.isin(wpos)]
    G = np.zeros((len(wi), 6, len(pos)), bool)
    start = pd.to_datetime(wi.forecast_origin, utc=True).to_numpy()
    r = s.window_id.map(wpos).to_numpy()
    k = ((pd.to_datetime(s.timestamp, utc=True).to_numpy() - start[r]) / np.timedelta64(5, "m")).astype(int) - 1
    G[r, k, s.link_id.map(pos).to_numpy()] = s.queue_pred.to_numpy() == 1
    return G
