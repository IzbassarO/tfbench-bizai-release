"""Standalone local/Colab runner. All durable output is under --results (Drive).

No Kaggle or submission interfaces. Run `python -m tfb.t1.mlp_colab.runner --help`.
"""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import contextlib
import gc
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
import torch.nn.functional as functional
from .config import configurations, FIXED_NAMES
from .scoring import evaluate, delta_table

# Set by an orchestrator to record progress / pause only after a durable epoch.
PROGRESS_CALLBACK = None


class Net(nn.Module):
    """Initialization/forward order matches mlp.Net, including depth semantics."""
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
        # Validate in fp32, as mlp._forward; early stopping uses standardized MSE for both losses.
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
    # The optimizer is only needed while this holdout is incomplete.
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
                # All exported holdout predictions are evaluated in float32.
                pred.append(net(a.float(), f.float(), p, r).cpu().numpy())
        residual = np.concatenate(pred) * np.asarray(meta["sd"], np.float32) + np.asarray(meta["mu"], np.float32)
        z = pq.read_table(path, columns=["base_s", "base_f", "lanes", "row", "col"])
        sp = np.clip(z["base_s"].to_numpy() + residual[:, 0], 3.0, 120.75).astype(np.float32)
        fl = np.maximum(z["base_f"].to_numpy() + residual[:, 1] * z["lanes"].to_numpy(), 100.0).astype(np.float32)
        if not np.isfinite(sp).all() or not np.isfinite(fl).all(): raise FloatingPointError(panel)
        target = directory / f"{panel}.npz"; tmp = directory / f"{panel}.tmp.npz"
        np.savez_compressed(tmp, row=z["row"].to_numpy(), col=z["col"].to_numpy(), sp=sp, fl=fl)
        tmp.replace(target)


def run_one(bundle, results, hold, cfg, config_id, seed, precision, device, data):
    manifest = json.loads((bundle / "manifest.json").read_text()); meta = manifest["holdouts"][hold]
    directory = results / f"cfg{config_id:03d}" / f"seed{seed}_{precision}" / hold
    done = directory / "done.json"
    identity = dict(bundle_id=manifest["bundle_id"], config=cfg, config_id=config_id,
                    seed=seed, precision=precision, hold=hold)
    if done.exists():
        record = json.loads(done.read_text())
        if any(record[k] != v for k, v in identity.items()): raise ValueError(f"Stale result: {directory}")
        return record
    directory.mkdir(parents=True, exist_ok=True)
    ident = directory / "identity.json"
    if ident.exists() and json.loads(ident.read_text()) != identity: raise ValueError(f"Stale checkpoint: {directory}")
    atomic_json(ident, identity)
    begin = time.perf_counter()
    net, history = fit(data, meta, cfg, seed, precision, device, directory)
    predict(net, bundle, hold, meta, device, directory)
    record = dict(**identity, **evaluate(bundle, hold, [directory]),
                  seconds_per_epoch=float(np.mean([h["seconds"] for h in history])),
                  epochs_run=len(history), best_epoch=best_epoch(history), elapsed_seconds=time.perf_counter()-begin,
                  gpu=torch.cuda.get_device_name() if device.type == "cuda" else platform.processor(),
                  torch=torch.__version__, numpy=np.__version__, path=str(directory))
    atomic_json(done, record)
    (directory / "resume.pt").unlink(missing_ok=True)
    del net; gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    return record


def records_table(results):
    rows = []
    for path in Path(results).glob("cfg*/seed*/*/done.json"):
        r = json.loads(path.read_text()); r["seeds"] = str(r["seed"]); rows.append(r)
    return delta_table(rows)


def select_top(results, n=3):
    tab = records_table(results)
    tab = tab[(tab.seeds == "42") & ((tab.precision == "amp") | (tab.config_id == 0))]
    # Config 0 is fp32 throughout; there is one control per holdout, never a Mac baseline.
    if tab.empty: raise ValueError("Run the sweep first")
    pivot = tab.pivot(index="config_id", columns="hold", values="dJ").dropna()
    if not {"last30", "autumn"}.issubset(pivot.columns): raise ValueError("Both holdouts are required")
    pivot["mean_dJ"] = pivot[["last30", "autumn"]].mean(1)
    pivot["worst_dJ"] = pivot[["last30", "autumn"]].min(1)
    pivot = pivot.drop(index=0, errors="ignore").sort_values(["mean_dJ", "worst_dJ", "config_id"], ascending=[False, False, True])
    return [int(i) for i in pivot.head(n).index]


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


def run_final(args, cfgs, device):
    if args.final_bundle is None: raise ValueError("--final-bundle is required; export with --final-only")
    manifest = json.loads((args.final_bundle / "manifest.json").read_text())
    source = json.loads((args.bundle / "manifest.json").read_text())
    if manifest.get("kind") != "final": raise ValueError("A full-train/test bundle is required")
    if manifest["source_sha256"] != source["source_sha256"]:
        raise ValueError("Holdout and final features were built from different source versions")
    ids = args.ids if args.ids is not None else json.loads((args.results / "top3.json").read_text())
    # Recompute the gate from predictions. No manually-written approval flag can bypass it.
    for i in ids:
        for hold in ("last30", "autumn"):
            base = args.results / "cfg000/seed42_fp32" / hold
            seeds = [args.results / f"cfg{i:03d}" / f"seed{s}_fp32" / hold for s in (42, 7, 123)]
            if not all((d / "done.json").exists() for d in [base, *seeds]):
                raise ValueError(f"cfg{i}: fp32 confirmation on all three seeds and both holdouts is required")
            dJ = evaluate(args.bundle, hold, seeds)["J"] - evaluate(args.bundle, hold, [base])["J"]
            if dJ < args.gate_threshold: raise ValueError(f"cfg{i}/{hold}: gate failed, dJ={dJ:+.6f}")
    meta = manifest["holdouts"]["final"]; data = None
    begin = time.perf_counter(); limit = args.max_hours * 3600
    if args.cu_per_hour is not None: limit = min(limit, args.cu_budget / args.cu_per_hour * 3600)
    for i in ids:
        directories = []
        for seed in (42, 7, 123):
            out = args.results / "final" / f"cfg{i:03d}" / f"seed{seed}_fp32"
            directories.append(out)
            identity = dict(bundle_id=manifest["bundle_id"], config=cfgs[i], seed=seed, precision="fp32")
            marker = out / "done.json"
            if marker.exists():
                record = json.loads(marker.read_text())
                if any(record[k] != v for k, v in identity.items()): raise ValueError(f"Stale final: {out}")
                continue
            if time.perf_counter()-begin >= limit:
                print("Session budget reached. Rerun final to resume."); return
            out.mkdir(parents=True, exist_ok=True)
            ident = out / "identity.json"
            if ident.exists() and json.loads(ident.read_text()) != identity: raise ValueError(f"Stale final checkpoint: {out}")
            atomic_json(ident, identity)
            if data is None:
                data = load_arrays([args.final_bundle / "final/train" / f"{p}.parquet" for p in meta["panels"]], meta)
            net, history = fit(data, meta, cfgs[i], seed, "fp32", device, out)
            for split in meta["test_splits"]:
                (out / split).mkdir(exist_ok=True)
                predict(net, args.final_bundle, "final", meta, device, out / split, split=split)
            csv = out / f"state_submission_colab_cfg{i:03d}_seed{seed}.csv.gz"
            n_rows = write_final_csv(args.final_bundle, [out], csv)
            atomic_json(marker, dict(**identity, n_rows=n_rows, csv=str(csv), history=history))
            (out / "resume.pt").unlink(missing_ok=True)
            del net; gc.collect()
            if device.type == "cuda": torch.cuda.empty_cache()
        output = args.results / "final" / f"cfg{i:03d}" / f"state_submission_colab_cfg{i:03d}_s3.csv.gz"
        count = write_final_csv(args.final_bundle, directories, output)
        atomic_json(output.with_name(output.name + ".meta.json"), dict(config=cfgs[i], seeds=[42, 7, 123],
                    n_rows=count, kind="raw Task 1 MLP component, no blend/pp2", bundle_id=manifest["bundle_id"]))
        print(f"Saved tfb.submit-compatible Task 1 component: {output} ({count:,} rows)", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("mode", choices=["sweep", "confirm", "final", "table", "benchmark"])
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--scheduler", choices=["onecycle", "cosine", "constant"], default="onecycle",
                   help="Scheduler for random configs; fixed e20/V3/V2 keep onecycle")
    p.add_argument("--final-bundle", type=Path, help="Full-train/test bundle exported with --final-only")
    p.add_argument("--gate-threshold", type=float, default=0.0001,
                   help="Require fp32 3-seed mean dJ >= this on each holdout before final")
    p.add_argument("--ids", type=int, nargs="*")
    p.add_argument("--max-hours", type=float, default=5.0,
                   help="Stop between runs at this session budget; rerun to resume")
    p.add_argument("--cu-per-hour", type=float, default=None,
                   help="Current rate shown by Colab; no hardcoded provider prices")
    p.add_argument("--cu-budget", type=float, default=155.0)
    args = p.parse_args(argv)
    if args.mode == "table":
        print(records_table(args.results).to_string(index=False)); return
    if args.mode == "benchmark":
        benchmark(args); return
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("Select L4 GPU in Colab")
    device = torch.device(args.device)
    cfgs = configurations(args.trials, args.scheduler)
    args.results.mkdir(parents=True, exist_ok=True)
    plan = args.results / "plan.json"
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    value = dict(configurations=cfgs, bundle_id=manifest["bundle_id"], fixed_names={str(k): v for k, v in FIXED_NAMES.items()},
                 ranking="mean dJ over last30/autumn; worst dJ tie break")
    if plan.exists() and json.loads(plan.read_text()) != value: raise ValueError("Plan/bundle changed; use a new results folder")
    atomic_json(plan, value)
    if args.mode == "final":
        run_final(args, cfgs, device)
        return
    if args.mode == "confirm":
        ids = args.ids if args.ids is not None else select_top(args.results)
        if not ids or 0 in ids: raise ValueError("Choose candidate IDs >= 1")
        atomic_json(args.results / "top3.json", ids)
        jobs = [(i, s, "fp32") for i in [0, *ids] for s in ([42] if i == 0 else [42, 7, 123])]
    else:
        ids = args.ids if args.ids is not None else list(range(len(cfgs)))
        jobs = [(i, 42, "fp32" if i == 0 or device.type == "cpu" else "amp") for i in ids]
    limit = args.max_hours * 3600
    if args.cu_per_hour is not None:
        limit = min(limit, args.cu_budget / args.cu_per_hour * 3600)
    start = time.perf_counter()
    # Fixed experiments finish both holdouts in the requested order, before the
    # random search. Random trials then use holdout-major IO to avoid 30 reloads.
    if args.mode == "sweep":
        groups = [([job for job in jobs if job[0] == i], ["last30", "autumn"]) for i in (0, 1, 2)]
        groups += [([job for job in jobs if job[0] >= 3], ["last30", "autumn"])]
    else:
        groups = [(jobs, ["last30", "autumn"])]
    for group, holds in groups:
      if not group: continue
      for hold in holds:
        meta = manifest["holdouts"][hold]; data = None
        for i, seed, precision in group:
            path = args.results / f"cfg{i:03d}" / f"seed{seed}_{precision}" / hold / "done.json"
            if path.exists(): continue
            if time.perf_counter() - start >= limit:
                print("Session budget reached. Run this cell again to resume."); return
            if data is None:
                data = load_arrays([args.bundle / hold / "train" / f"{p}.parquet" for p in meta["panels"]], meta)
            try:
                r = run_one(args.bundle, args.results, hold, cfgs[i], i, seed, precision, device, data)
                print(json.dumps(r), flush=True)
                # A completed config/holdout is committed to Drive before proceeding.
                tab = records_table(args.results)
                tmp = args.results / "summary.tmp.csv"; tab.to_csv(tmp, index=False); tmp.replace(args.results / "summary.csv")
            except torch.cuda.OutOfMemoryError:
                atomic_json(path.parent / "oom.json", dict(config=cfgs[i], seed=seed, precision=precision,
                            gpu=torch.cuda.get_device_name(), error="OOM; batch/architecture unchanged"))
                print(f"OOM cfg{i}; recorded on Drive, continuing", flush=True)
                gc.collect(); torch.cuda.empty_cache()
        del data; gc.collect()


def benchmark(args):
    """Bounded real CPU fwd+bwd timing; extrapolation explicitly excludes feature export."""
    seed_all(42); meta = json.loads((args.bundle / "manifest.json").read_text())["holdouts"]["last30"]
    cfgs = configurations(args.trials, args.scheduler); ids = args.ids or [0]
    report = []
    for i in ids:
        cfg = cfgs[i]; batch = cfg["batch"]
        net = Net(meta["n_num"], meta["n_flag"], meta["n_panel"], cfg["width"], cfg["depth"], cfg["dropout"])
        opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=1e-5)
        a = torch.randn(batch, meta["n_num"]); f = torch.zeros(batch, meta["n_flag"])
        panel = torch.zeros(batch, dtype=torch.int64); reg = panel.clone(); y = torch.randn(batch, 2)
        times = []
        for step in range(5):
            t0 = time.perf_counter(); pred = net(a, f, panel, reg)
            loss = (pred-y).square().mean() if cfg["loss"] == "mse" else functional.huber_loss(pred, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if step >= 2: times.append(time.perf_counter()-t0)
        seconds = float(np.median(times)); n = meta["n_rows"] - meta["n_valid"]
        ep = seconds * ((n+batch-1)//batch)
        report.append(dict(config_id=i, config=cfg, seconds_per_batch=seconds, seconds_per_epoch=ep,
                           max_hours_per_seed=ep*cfg["epochs"]/3600, max_hours_3seeds=ep*cfg["epochs"]*3/3600,
                           train_rows=n, threads=8, machine=platform.platform(), timing="synthetic real optimizer steps; excludes IO/validation/features"))
        del net, opt; gc.collect()
    atomic_json(args.results / "cpu_estimate.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
