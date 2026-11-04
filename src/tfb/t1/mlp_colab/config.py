"""Frozen control and reproducible random search (depth follows mlp.Net)."""
from itertools import product
import random

CONTROL = dict(width=512, depth=1, lr=0.002, dropout=0.1, epochs=20,
               batch=16384, loss="mse", scheduler="onecycle")
V3 = dict(CONTROL, epochs=40)
V2 = dict(CONTROL, width=1024, depth=2)
FIXED_NAMES = {0: "e20", 1: "V3_e40_onecycle", 2: "V2_1024_depth2"}
GRID = dict(width=[512, 1024, 2048], depth=[2, 3, 4], lr=[0.001, 0.003],
            dropout=[0.0, 0.1], epochs=[20, 40], batch=[4096, 16384],
            loss=["mse", "huber"])


def configurations(n=30, scheduler="onecycle"):
    space = [dict(zip(GRID, v)) for v in product(*GRID.values())]
    return [CONTROL.copy(), V3.copy(), V2.copy()] + [dict(c, scheduler=scheduler) for c in random.Random(42).sample(space, n)]
