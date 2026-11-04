"""MLP member for Task 1 (see report, Sec. III-A): same features, residual targets, early-stopping hold and prediction
path as the pooled LightGBM (tfb.t1.train); only the learner differs. CPU torch, deterministic for a fixed seed/platform.

usage:
  python -m tfb.t1.mlp holdout [last30|autumn] [panel ...]   -> cache/t1_runs/<TFB_TAG>_<holdout>/{<panel>.npz, mlp.pt}
  python -m tfb.t1.mlp final                                  -> outputs/t1/state_submission_<TFB_TAG>[_seed<S>].csv
env: TFB_SEED, TFB_TAG, TFB_TRANS (as tfb.t1.train); TFB_MLP_EPOCHS=10, TFB_MLP_BATCH=16384, TFB_MLP_LR=2e-3, TFB_MLP_WIDTH=512,
     TFB_MLP_DEPTH=1 (hidden width x width layers), TFB_MLP_LOSS=mse|metric (metric: rows weighted like the S_state macro
     family -> panel -> regime, heads weighted 0.54/25 : 0.46/600 in target-std units)
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd
import torch  # before lightgbm (via .train): two OpenMP runtimes in the wrong order segfault on macOS
from torch import nn
from .. import io, cache
from . import train as T
from . import features as FE

EPOCHS = int(os.environ.get("TFB_MLP_EPOCHS", 10))
BATCH = int(os.environ.get("TFB_MLP_BATCH", 16384))
LR = float(os.environ.get("TFB_MLP_LR", 2e-3))
WIDTH = int(os.environ.get("TFB_MLP_WIDTH", 512))
DEPTH = int(os.environ.get("TFB_MLP_DEPTH", 1))
LOSS = os.environ.get("TFB_MLP_LOSS", "mse")
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


class Net(nn.Module):
    def __init__(self, d_num, d_flag, n_panel, width=WIDTH, depth=DEPTH):
        super().__init__()
        self.ep, self.er = nn.Embedding(n_panel, 4), nn.Embedding(4, 2)
        first = nn.Linear(d_num + d_flag + 6, width)   # created first: init order (RNG) must match the depth-1 network
        hidden = [m for _ in range(depth) for m in (nn.Linear(width, width), nn.SiLU(), nn.Dropout(0.1))]
        self.body = nn.Sequential(first, nn.SiLU(), nn.Dropout(0.1), *hidden,
                                  nn.Linear(width, width // 2), nn.SiLU(), nn.Linear(width // 2, len(TARGETS)))

    def forward(self, a, f, p, r):
        return self.body(torch.cat([a, f, self.ep(p), self.er(r)], 1))


def _t(*arrs):
    return [torch.from_numpy(np.ascontiguousarray(x)) for x in arrs]


def _forward(net, A, F, P, R, chunk=262144):
    net.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(A), chunk):
            a, f, p, r = _t(A[s:s + chunk], F[s:s + chunk].astype(np.float32), P[s:s + chunk], R[s:s + chunk])
            out.append(net(a, f, p, r).numpy())
    return np.concatenate(out)


def fit(X, ys, w, rng):
    """Same hold draw as tfb.t1.train.make_sets (5 % rows) -> early stopping on its MSE (in target-std units)."""
    torch.manual_seed(T.SEED); torch.use_deterministic_algorithms(True); torch.set_num_threads(8)
    hold = rng.random(len(X)) < 0.05; tr = np.flatnonzero(~hold); va = np.flatnonzero(hold)
    prep = Prep().fit(X, tr); A, F, P, R = prep.transform(X); del X
    Y = np.stack([ys[k] for k in TARGETS], 1).astype(np.float32)
    mu, sd = Y[tr].mean(0), Y[tr].std(0); Y = (Y - mu) / sd
    W = np.ones(len(Y), np.float32) if w is None else w.astype(np.float32)
    hw = np.ones(len(TARGETS), np.float32)
    if LOSS == "metric":   # S_state is a macro mean over families -> panels -> regimes of 0.54 (1 - RMSE_s/25) + 0.46 (1 - RMSE_q/600)
        fam = io.family_of(); name = {v: k for k, v in FE.PANEL_ID.items()}; n_fam = pd.Series(fam).value_counts()
        key = P * 4 + R; u, cnt = np.unique(key[tr], return_counts=True); g = dict(zip(u, cnt))
        fw = {p: 1.0 / n_fam[fam[name[p]]] for p in np.unique(P)}
        gw = np.array([fw[k // 4] / g[k] if k in g else 0.0 for k in key], np.float32)
        W = W * gw / gw[tr].mean()
        hw = np.array([0.54 * sd[0] / 25, 0.46 * sd[1] / 600], np.float32); hw = hw / hw.mean()
    hwt = torch.from_numpy(hw)
    net = Net(A.shape[1], F.shape[1], prep.n_panel)
    steps = (len(tr) + BATCH - 1) // BATCH
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * steps)
    order = np.random.default_rng(T.SEED + 3); best, best_state, bad = np.inf, None, 0
    for ep in range(EPOCHS):
        net.train(); t0 = time.time(); perm = order.permutation(tr)
        for s in range(0, len(perm), BATCH):
            b = np.sort(perm[s:s + BATCH])
            a, f, p, r, y, wt = _t(A[b], F[b].astype(np.float32), P[b], R[b], Y[b], W[b])
            loss = ((net(a, f, p, r) - y) ** 2 * wt[:, None] * hwt).sum() / (wt.sum() * len(TARGETS))
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        pv = _forward(net, A[va], F[va], P[va], R[va]); mse = (((pv - Y[va]) ** 2) * W[va, None]).sum(0) / W[va].sum() * hw
        rmse = np.sqrt(mse) * sd
        print(f"  epoch {ep + 1}/{EPOCHS}: hold rmse speed {rmse[0] / np.sqrt(hw[0]):.3f} flow_lane {rmse[1] / np.sqrt(hw[1]):.3f} ({time.time() - t0:.0f}s)", flush=True)
        if mse.mean() < best - 1e-5:
            best, bad = mse.mean(), 0; best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 2:
                break
    net.load_state_dict(best_state)
    return net, prep, mu, sd


class Head:
    """LightGBM-like .predict(df) for tfb.t1.train.predict_panel; both targets come from one forward pass per frame."""
    _cache = {}

    def __init__(self, net, prep, mu, sd, k):
        self.net, self.prep, self.mu, self.sd, self.k = net, prep, mu, sd, k

    def predict(self, df, num_threads=8):
        key = id(df)
        if key not in Head._cache:
            Head._cache.clear(); Head._cache[key] = _forward(self.net, *self.prep.transform(df)) * self.sd + self.mu
        return Head._cache[key][:, self.k]


def heads(net, prep, mu, sd):
    return {k: Head(net, prep, mu, sd, i) for i, k in enumerate(TARGETS)}


def save(out, net, prep, mu, sd):
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "prep": prep.__dict__, "mu": mu, "sd": sd, "width": WIDTH, "depth": DEPTH, "loss": LOSS}, out / "mlp.pt")


def run_holdout(name, panels):
    days = T.holdout_days(name)
    net, prep, mu, sd = fit(*T.build_training(days, panels, test_splits=("train",)))
    out = T.run_dir(T.TAG, name); save(out, net, prep, mu, sd); ms = heads(net, prep, mu, sd); preds = {}
    for panel in panels:
        pn = cache.load(panel, "train"); rows = T.day_mask(pn, days)
        sp, fl, _ = T.predict_panel(ms, prep.cols, pn, pn, days, rows)
        np.savez(out / f"{panel}.npz", rows=np.flatnonzero(rows), sp=sp[rows], fl=fl[rows])
        preds[panel] = (sp, fl); del pn
    s, d, phys = T.score_preds(preds, days, name)
    print(name, f"S_state={s:.5f}")
    T.log_experiment(task="t1", change=f"{T.TAG}: MLP {WIDTH}x{DEPTH} loss={LOSS} epochs<={EPOCHS} lr={LR} trans={T.TRANS} seed={T.SEED} ({'all' if len(panels) == 10 else ','.join(panels)})",
                     local_metric=f"S_state_{name}", local_value=round(s, 5), notes=f"S_phys_proxy={phys.S_phys_proxy.mean():.4f}; run={out.name}")


def run_final(panels):
    net, prep, mu, sd = fit(*T.build_training(None, panels, test_splits=("validation", "private")))
    suffix = f"_{T.TAG}" + (f"_seed{T.SEED}" if T.SEED != 42 else "")
    save(io.OUTPUTS / "t1" / f"mlp{suffix}", net, prep, mu, sd); ms = heads(net, prep, mu, sd); preds = {}
    for panel in panels:
        pn_train = cache.load(panel, "train")
        for split in ("validation", "private"):
            pn = cache.load(panel, split)
            sp, fl, _ = T.predict_panel(ms, prep.cols, pn, pn_train, None, np.ones(len(pn.mspeed), bool))
            preds[(panel, split)] = (sp, fl)
            lt50 = float((fl[pn.target] < 50).mean()); print(panel, split, "done; share<50vph", lt50, flush=True); assert lt50 < 0.01
    out = io.OUTPUTS / "t1" / f"state_submission{suffix}.csv"
    T.write_state(preds, out, meta={"model": "MLP on the tfb.t1 features (residual targets)", "tag": T.TAG, "seed": T.SEED, "trans": T.TRANS,
                                    "epochs_max": EPOCHS, "batch": BATCH, "lr": LR, "width": WIDTH, "depth": DEPTH, "loss": LOSS})
    print("wrote", out)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "holdout":
        run_holdout(sys.argv[2], sys.argv[3:] or io.panels())
    elif mode == "final":
        run_final(sys.argv[2:] or io.panels())
