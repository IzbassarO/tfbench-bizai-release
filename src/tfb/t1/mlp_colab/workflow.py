"""MLP final training over the exported features and the checked component csv.
"""
from __future__ import annotations
import torch  # before LightGBM on macOS
import gc
import hashlib
import json
from pathlib import Path

from .config import CONTROL
from . import runner


def resolve_config(value):
    """A complete config dict (missing keys from CONTROL) -> config, tag."""
    cfg = dict(CONTROL, **value)
    required = set(CONTROL)
    if set(cfg) != required: raise ValueError(f"Expected config keys {sorted(required)}")
    if cfg["loss"] not in ("mse", "huber") or cfg["scheduler"] not in ("onecycle", "cosine", "constant"):
        raise ValueError("Unsupported loss/scheduler")
    if cfg["epochs"] < 1 or cfg["batch"] < 1 or cfg["width"] < 2 or cfg["depth"] < 0:
        raise ValueError("Invalid architecture/training size")
    tag = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    return cfg, tag


def final_train(bundle, results, config, seeds=(42, 7, 123), device="cuda"):
    cfg, tag = resolve_config(config); meta_manifest = json.loads((bundle / "manifest.json").read_text())
    if not seeds or len(set(seeds)) != len(seeds): raise ValueError("SEEDS must be nonempty and unique")
    meta = meta_manifest["holdouts"]["final"]; results = Path(results) / tag / "final"
    directories = []; data = None; dev = torch.device(device)
    for seed in seeds:
        out = results / f"seed{seed}_fp32"; out.mkdir(parents=True, exist_ok=True); directories.append(out)
        identity = dict(bundle_id=meta_manifest["bundle_id"], config=cfg, seed=seed, precision="fp32")
        ident = out / "identity.json"
        if ident.exists() and json.loads(ident.read_text()) != identity: raise ValueError(f"Stale final: {out}")
        runner.atomic_json(ident, identity)
        if (out / "done.json").exists(): continue
        if data is None: data = runner.load_arrays([bundle / "final/train" / f"{p}.parquet" for p in meta["panels"]], meta)
        net, history = runner.fit(data, meta, cfg, seed, "fp32", dev, out)
        for split in meta["test_splits"]:
            (out / split).mkdir(exist_ok=True)
            runner.predict(net, bundle, "final", meta, dev, out / split, split=split)
        csv = out / f"state_submission_{tag}_seed{seed}.csv.gz"
        n = runner.write_final_csv(bundle, [out], csv)
        runner.atomic_json(out / "done.json", dict(**identity, rows=n, history=history, csv=str(csv)))
        (out / "resume.pt").unlink(missing_ok=True)
        del net; gc.collect()
        if dev.type == "cuda": torch.cuda.empty_cache()
    output = results / ("state_submission_" + tag + "_s" + "-".join(map(str, seeds)) + ".csv.gz")
    runner.write_final_csv(bundle, directories, output)
    return output
