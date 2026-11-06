"""Scoring helpers: official per-window IoU on eligible cells, macro aggregation."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .. import io
from .data import load as _load


def load_train(panel: str, cache: str = "cache") -> dict:
    z = np.load(io.OUTPUTS / "t2b" / cache / f"{panel}_train.npz")
    b = {k: z[k] for k in z.files}
    b["meta"] = pd.read_parquet(io.OUTPUTS / "t2b" / cache / f"{panel}_train_meta.parquet")
    d = _load(panel, "train")
    b["vcut"] = d["vcut"]; b["net"] = {k: d[k] for k in ("v_f", "cap", "k_crit", "k_jam", "lanes", "length", "vcut")}
    return b


def window_iou(P: np.ndarray, Y: np.ndarray, E: np.ndarray) -> np.ndarray:
    """P, Y, E: [n, 6, L] bool -> official IoU per window (eligible only; empty union 1; no eligible 0)."""
    inter = (P & Y & E).sum((1, 2)); uni = ((P | Y) & E).sum((1, 2))
    out = np.where(uni == 0, 1.0, inter / np.maximum(uni, 1))
    return np.where(E.sum((1, 2)) == 0, 0.0, out)
