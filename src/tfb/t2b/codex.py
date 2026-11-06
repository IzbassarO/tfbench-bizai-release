"""Read-only access to family A's Task 2 queue files."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .. import io
from . import data


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
