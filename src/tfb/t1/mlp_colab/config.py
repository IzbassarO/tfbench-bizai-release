"""MLP base configuration; a run config is this dict with its overrides (depth follows the network in runner.Net)."""

CONTROL = dict(width=512, depth=1, lr=0.002, dropout=0.1, epochs=20,
               batch=16384, loss="mse", scheduler="onecycle")
