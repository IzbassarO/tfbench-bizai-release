"""MLP training and prediction on the exported features.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import contextlib
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
import torch.nn.functional as functional

# Optional progress callback, called after each durable epoch.
PROGRESS_CALLBACK = None


class Net(nn.Module):
    """Fully connected network: depth hidden layers of the given width."""
    def __init__(self, d_num, d_flag, n_panel, width=512, depth=1, dropout=0.1):
        super().__init__()
        self.ep, self.er = nn.Embedding(n_panel, 4), nn.Embedding(4, 2)
        first = nn.Linear(d_num + d_flag + 6, width)
        hidden = [m for _ in range(depth) for m in
                  (nn.Linear(width, width), nn.SiLU(), nn.Dropout(dropout))]
        self.body = nn.Sequential(first, nn.SiLU(), nn.Dropout(dropout), *hidden,
                                  nn.Linear(width, width // 2), nn.SiLU(), nn.Linear(width // 2, 2))

    def forward(self, a, f, p, r):
        return self.body(torch.cat([a, f, self.ep(p), self.er(r)], 1))


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)); tmp.replace(path)


def atomic_torch(path, value):
    tmp = path.with_suffix(".tmp"); torch.save(value, tmp); tmp.replace(path)


def load_arrays(paths, meta, labels=True):
    cols = [f"a{i}" for i in range(meta["n_num"])] + [f"f{i}" for i in range(meta["n_flag"])]
    cat = ["panel", "regime"]
    extra = ["y_speed", "y_flow_lane", "valid"] if labels else []
    # Read just one Arrow shard at a time; avoid a second full float32 feature copy.
    n = meta["n_rows"] if labels else sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
    A = np.empty((n, meta["n_num"]), np.float16)
    F = np.empty((n, meta["n_flag"]), np.float16)
    P, R = np.empty(n, np.int64), np.empty(n, np.int64)
    Y, valid = np.empty((n, 2), np.float32), np.empty(n, bool)
    offset = 0
    for path in paths:
        table = pq.read_table(path, columns=cols + cat + extra)
        m = len(table); sl = slice(offset, offset + m)
        for j in range(A.shape[1]): A[sl, j] = table[f"a{j}"].to_numpy()
        for j in range(F.shape[1]): F[sl, j] = table[f"f{j}"].to_numpy()
        P[sl], R[sl] = table["panel"].to_numpy(), table["regime"].to_numpy()
        if labels:
            Y[sl] = np.column_stack([table[k].to_numpy() for k in extra[:2]])
            valid[sl] = table["valid"].to_numpy()
        offset += m
    arrays = (A, F, P, R)
    if labels:
        Y = (Y - np.asarray(meta["mu"], np.float32)) / np.asarray(meta["sd"], np.float32)
        return arrays, Y, valid
    return arrays


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def synchronize(device):
    if device.type == "cuda": torch.cuda.synchronize()


def scheduler_for(opt, cfg, steps):
    name = cfg.get("scheduler", "onecycle")
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * steps)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"] * steps)
    if name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    raise ValueError(f"Unknown scheduler {name}")


def checkpoint_rng(order):
    return dict(order=order.bit_generator.state, cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def best_epoch(history):
    """Epoch of the selected weights, respecting the improvement tolerance."""
    return next(h['epoch'] for h in history if h['valid_mse'] == history[-1]['best'])


def fit(data, meta, cfg, seed, precision, device, directory):
    directory.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    arrays, y, valid = data
    tr, va = np.flatnonzero(~valid), np.flatnonzero(valid)
    A, F, P, R = [torch.as_tensor(a, device=device) for a in arrays]
    Y = torch.as_tensor(y, device=device)
    def forward(b):
        return net(A[b].float(), F[b].float(), P[b], R[b])
    net = Net(meta["n_num"], meta["n_flag"], meta["n_panel"], cfg["width"], cfg["depth"], cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    steps = (len(tr) + cfg["batch"] - 1) // cfg["batch"]
    sch = scheduler_for(opt, cfg, steps)
    amp = precision == "amp"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    order = np.random.default_rng(seed + 3)
    best, best_state, bad, start, history = np.inf, None, 0, 0, []
    resume = directory / "resume.pt"
    if resume.exists():
        z = torch.load(resume, map_location="cpu", weights_only=False)
        net.load_state_dict(z["state"]); opt.load_state_dict(z["optimizer"])
        sch.load_state_dict(z["scheduler"]); scaler.load_state_dict(z["scaler"])
        best, best_state, bad, start, history = (z[k] for k in ("best", "best_state", "bad", "epoch", "history"))
        order.bit_generator.state = z["order"]; torch.set_rng_state(z["cpu_rng"])
        if z["cuda_rng"] and device.type == "cuda": torch.cuda.set_rng_state_all(z["cuda_rng"])
        print(f"resume epoch {start}, {directory}", flush=True)
    autocast = lambda: torch.autocast("cuda", dtype=torch.float16) if amp else contextlib.nullcontext()
    for epoch in range(start, cfg["epochs"]):
        if bad >= 5: break
        synchronize(device); begin = time.perf_counter(); net.train()
        perm = order.permutation(tr)
        for s in range(0, len(perm), cfg["batch"]):
            b = torch.as_tensor(np.sort(perm[s:s+cfg["batch"]]), device=device)
            with autocast():
                pred = forward(b)
                # Loss and targets are always float32, also inside AMP.
                error = pred.float() - Y[b]
                loss = error.square().mean() if cfg["loss"] == "mse" else functional.huber_loss(pred.float(), Y[b], delta=1.0)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
        net.eval(); total = 0.0
        # Validate in fp32; early stopping uses standardized MSE for both losses.
        with torch.no_grad():
            for s in range(0, len(va), cfg["batch"]):
                b = torch.as_tensor(va[s:s+cfg["batch"]], device=device)
                total += float((forward(b) - Y[b]).square().sum().item())
        mse = total / (2 * len(va))
        if not np.isfinite(mse): raise FloatingPointError(f"Nonfinite validation MSE: {directory}")
        synchronize(device); seconds = time.perf_counter() - begin
        if mse < best - 1e-5:
            best, bad = mse, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else: bad += 1
        row = dict(epoch=epoch+1, seconds=seconds, valid_mse=mse, best=best)
        history.append(row)
        print(f"{directory.name} epoch {epoch+1}/{cfg['epochs']}: mse={mse:.6f}, {seconds:.1f}s", flush=True)
        z = dict(state={k: v.detach().cpu() for k, v in net.state_dict().items()},
                 optimizer=opt.state_dict(), scheduler=sch.state_dict(), scaler=scaler.state_dict(),
                 best=best, best_state=best_state, bad=bad, epoch=epoch+1, history=history, **checkpoint_rng(order))
        atomic_torch(resume, z)
        if PROGRESS_CALLBACK is not None: PROGRESS_CALLBACK(row)
    net.load_state_dict(best_state)
    atomic_torch(directory / "model.pt", dict(state=best_state, config=cfg, seed=seed,
                 precision=precision, meta=meta, best_mse=best, history=history,
                 best_epoch=best_epoch(history)))
    # The optimizer is only needed while the run is incomplete.
    del A, F, P, R, Y, opt, sch, scaler
    return net, history


def predict(net, bundle, hold, meta, device, directory, chunk=32768, split=None):
    net.eval()
    for panel in meta["panels"]:
        path = bundle / hold / "predict"
        if split is not None: path = path / split
        path = path / f"{panel}.parquet"
        arrays = load_arrays([path], meta, labels=False)
        pred = []
        with torch.no_grad():
            for s in range(0, len(arrays[0]), chunk):
                a, f, p, r = [torch.as_tensor(x[s:s+chunk], device=device) for x in arrays]
                # Predictions are evaluated in float32.
                pred.append(net(a.float(), f.float(), p, r).cpu().numpy())
        residual = np.concatenate(pred) * np.asarray(meta["sd"], np.float32) + np.asarray(meta["mu"], np.float32)
        z = pq.read_table(path, columns=["base_s", "base_f", "lanes", "row", "col"])
        sp = np.clip(z["base_s"].to_numpy() + residual[:, 0], 3.0, 120.75).astype(np.float32)
        fl = np.maximum(z["base_f"].to_numpy() + residual[:, 1] * z["lanes"].to_numpy(), 100.0).astype(np.float32)
        if not np.isfinite(sp).all() or not np.isfinite(fl).all(): raise FloatingPointError(panel)
        target = directory / f"{panel}.npz"; tmp = directory / f"{panel}.tmp.npz"
        np.savez_compressed(tmp, row=z["row"].to_numpy(), col=z["col"].to_numpy(), sp=sp, fl=fl)
        tmp.replace(target)


def write_final_csv(bundle, prediction_dirs, output):
    """Template order/keys, TOTAL flow, all validation AND private targets."""
    import gzip
    manifest = json.loads((bundle / "manifest.json").read_text()); meta = manifest["holdouts"]["final"]
    tmp = output.with_name(output.name + ".tmp")
    count = 0
    with gzip.open(tmp, "wt") as stream:
        for panel in meta["panels"]:
            for split in meta["test_splits"]:
                tpl = pd.read_parquet(bundle / "final/templates" / split / f"{panel}.parquet")
                arrays = []
                for directory in prediction_dirs:
                    with np.load(directory / split / f"{panel}.npz") as z:
                        arrays.append(np.column_stack((z["sp"], z["fl"])))
                pred = np.mean(arrays, axis=0)
                order = tpl.pop("pred_pos").to_numpy()
                if len(pred) != len(tpl) or not np.isfinite(pred).all(): raise ValueError(f"Invalid final: {panel}/{split}")
                if len(np.unique(order)) != len(order): raise ValueError("Duplicate template targets")
                tpl["speed_kmh"] = np.round(pred[order, 0].astype(np.float64), 3)
                tpl["flow_vph"] = np.round(pred[order, 1].astype(np.float64), 2)
                tpl.to_csv(stream, index=False, header=count == 0)
                count += len(tpl)
    tmp.replace(output)
    return count
