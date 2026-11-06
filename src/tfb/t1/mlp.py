"""Feature scaling of the Task 1 MLP member.
"""
from __future__ import annotations
import numpy as np
from . import features as FE

TARGETS = ("speed", "flow_lane")


class Prep:
    """Robust scaling (median / IQR from training rows), NaN -> 0 plus a missing flag for columns that have NaNs."""

    def fit(self, X, rows):
        self.cols = list(X.columns); self.num = [c for c in self.cols if c not in FE.CAT]
        S = X[self.num].to_numpy(np.float32)[rows]
        q25, self.med, q75 = np.nanpercentile(S, [25, 50, 75], axis=0)
        self.iqr = np.where(q75 - q25 > 1e-6, q75 - q25, np.nanstd(S, 0) + 1e-6).astype(np.float32)
        self.med = np.nan_to_num(self.med).astype(np.float32)
        self.flag = np.flatnonzero(np.isnan(X[self.num].to_numpy(np.float32)).any(0))
        self.n_panel = int(X.panel.max()) + 1
        return self

    def transform(self, X):
        A = X[self.num].to_numpy(np.float32)
        F = np.isnan(A[:, self.flag]).astype(np.uint8)
        A -= self.med; A /= self.iqr; np.clip(A, -10, 10, out=A); np.nan_to_num(A, copy=False)
        return A, F, X.panel.to_numpy(np.int64), X.regime.to_numpy(np.int64)
