"""Hybrid train labels from the clean imputer: released truth on eligible cells, imputed speed elsewhere.

  labels  tfb.t1.imputer_clean's booster at n_trees -> outputs/t1/train_hybrid_speed_clean/<panel>.parquet
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import lightgbm as lgb
from .. import cache, io
from ..t1 import hybrid as T1H


def write_labels(f: Path, panel: str, pn, imp, pred):
    """One panel of the tfb.t1.hybrid parquet layout (panel, timestamp, link_id, speed_hybrid, is_imputed)."""
    import pyarrow as pa, pyarrow.parquet as pq
    T, L = imp.shape
    ts = (np.repeat(pn.dates, io.STEPS_PER_DAY).astype(object) + "T"
          + np.tile([f"{m // 60:02d}:{m % 60:02d}:00Z" for m in range(0, 1440, 5)], len(pn.dates)).astype(object))
    t = pa.table({"panel": pa.array([panel] * (T * L)).dictionary_encode(),
                  "timestamp": pa.array(np.repeat(ts, L)).dictionary_encode(),
                  "link_id": pa.array(np.tile(np.asarray(pn.links, object), T)).dictionary_encode(),
                  "speed_hybrid": np.where(imp, pred, pn.speed).astype(np.float32).ravel(), "is_imputed": imp.ravel()})
    pq.write_table(t, f.with_suffix(".tmp")); f.with_suffix(".tmp").replace(f)


def labels(model: Path, n_trees: int, out: Path, log=print) -> str:
    """Train hybrid speed: released truth on eligible cells, clean-imputer speed on every other cell (as tfb.t1.hybrid.build)."""
    out.mkdir(parents=True, exist_ok=True); booster = lgb.Booster(model_file=str(model))
    for panel in T1H.PANELS:
        f = out / f"{panel}.parquet"
        if f.exists():
            continue
        pn = cache.load(panel, "train"); imp = ~pn.elig
        pred = T1H.impute([booster], T1H.hidden_layer(pn, imp), pn, None, imp, (n_trees,))[n_trees]
        write_labels(f, panel, pn, imp, pred)
        vf = np.broadcast_to(pn.v_f, imp.shape)
        log(f"[hyb_clean] labels {panel}: imputed {int(imp.sum())}, queued share imputed {(pred[imp] <= 0.6 * vf[imp]).mean():.4f}")
        del pn
    return str(out)
