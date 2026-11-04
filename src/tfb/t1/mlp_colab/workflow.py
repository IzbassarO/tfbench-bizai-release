"""Colab orchestration: cached features, Mac/Colab gates and checked CSVs.

This module never calls Kaggle submit. The notebook's final opt-in cell owns that.
"""
from __future__ import annotations
import torch  # before LightGBM on macOS
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import zipfile
import numpy as np
import pandas as pd

from tfb import io
from .config import CONTROL, V2, V3, configurations
from . import runner
from .scoring import evaluate

EXPECTED_SHA = "47c5bf41afbe6783cc45783f4ffff18e0db85be1f2630e2489c49c4f48fc0a13"
ROBUST_SHA = "7903cba18f8f4e3c857b5b34dd3a2ac67f173bc9154674475ce5b31dba6b06fb"
COMPETITION = "2026-ieee-big-data-traffic-flow-bench"
QUEUE = {"Ahyb": "queue_onset_Ahyb_E25.csv", "6s": "queue_onset_Ahyb_6s_E25.csv"}
FEATURE_SOURCES = ("src/tfb/t1/mlp.py", "src/tfb/t1/train.py", "src/tfb/t1/features.py")


def code_revision():
    marker = io.PROJECT / ".colab_code_revision"
    if marker.exists(): return marker.read_text().strip()
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=io.PROJECT, text=True).strip()


def component_manifest():
    direct = io.PROJECT / "bundle_manifest.json"
    return direct if direct.exists() else io.OUTPUTS / "tfb_colab_bundle/bundle_manifest.json"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8*1024*1024), b""): h.update(b)
    return h.hexdigest()


def resolve_config(value):
    """e20/V2/V3, 'sweep:3'..'sweep:32', integer ID, or a complete config dict."""
    if isinstance(value, str) and value in ("e20", "V2", "V3"):
        cfg = {"e20": CONTROL, "V2": V2, "V3": V3}[value].copy()
    elif isinstance(value, dict):
        cfg = dict(CONTROL, **value)
    else:
        i = int(str(value).removeprefix("sweep:"))
        cfg = configurations()[i].copy()
    required = set(CONTROL)
    if set(cfg) != required: raise ValueError(f"Expected config keys {sorted(required)}")
    if cfg["loss"] not in ("mse", "huber") or cfg["scheduler"] not in ("onecycle", "cosine", "constant"):
        raise ValueError("Unsupported loss/scheduler")
    if cfg["epochs"] < 1 or cfg["batch"] < 1 or cfg["width"] < 2 or cfg["depth"] < 0:
        raise ValueError("Invalid architecture/training size")
    tag = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    return cfg, tag


def extract_zip(archive, destination):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination.resolve()): raise ValueError("Unsafe archive path")
            if (item.external_attr >> 16) & 0o170000 == 0o120000: raise ValueError("Symlink in archive")
        z.extractall(destination)


def restore_components(archive, repo, verify=True):
    """Copy the archive from Drive once, extract to VM disk, verify original files."""
    repo = Path(repo); marker = repo / ".colab_components_ok.json"
    identity = dict(size=Path(archive).stat().st_size, name=Path(archive).name)
    if marker.exists() and json.loads(marker.read_text()) == identity: return
    local = repo.parent / "tfb_colab_bundle.local.zip"
    shutil.copy2(archive, local)
    extract_zip(local, repo)
    manifest = json.loads((repo / "bundle_manifest.json").read_text())
    if verify:
        for f in manifest["files"]:
            path = repo / f["path"]
            if digest(path) != f["sha256"]: raise ValueError(f"Bundle checksum: {f['path']}")
    runner.atomic_json(marker, identity)
    print(f"Components restored: {len(manifest['files'])} files")


def restore_cpu_components(archive, repo):
    """Add CPU-stage inputs without replacing the canonical bundle manifest."""
    archive, repo = Path(archive), Path(repo)
    marker = repo / ".colab_cpu_components_ok.json"
    identity = dict(name=archive.name, size=archive.stat().st_size)
    if marker.exists() and json.loads(marker.read_text()) == identity: return
    local = repo.parent / "tfb_cpu_bundle.local.zip"; shutil.copy2(archive, local)
    staging = repo.parent / "tfb_cpu_components"; extract_zip(local, staging)
    manifest_path = staging / "bundle_manifest_cpu.json"
    if not manifest_path.exists(): manifest_path = staging / "bundle_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for f in manifest["files"]:
            if digest(staging / f["path"]) != f["sha256"]: raise ValueError(f"CPU bundle checksum: {f['path']}")
    n = 0
    for directory in ("cache", "outputs", "data"):
        for src in (staging / directory).rglob("*"):
            if not src.is_file(): continue
            dst = repo / src.relative_to(staging)
            if dst.exists():
                if digest(src) != digest(dst): raise ValueError(f"CPU bundle conflicts with existing input: {dst}")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst); n += 1
    if n == 0 and not (repo / "outputs/t2b/hybrid/cv_janfeb_pure.pkl").exists():
        raise ValueError("CPU bundle must contain repo-relative cache/, outputs/, data/ paths")
    runner.atomic_json(marker, identity)
    print(f"CPU components ready: {n} added files")


def ensure_raw_release(drive_root, credentials):
    """Kaggle archive on Drive, raw extraction on VM. No organizer-only downloads."""
    drive_root = Path(drive_root); raw = drive_root / "raw"; raw.mkdir(parents=True, exist_ok=True)
    archive = raw / f"{COMPETITION}.zip"; marker = raw / "download_done.json"
    if not marker.exists():
        # Credentials live only in process environment, never on Drive or in notebook output.
        cred = json.loads(credentials)
        if not all(k in cred for k in ("username", "key")): raise ValueError("KAGGLE_JSON needs username/key")
        env = os.environ.copy(); env.update(KAGGLE_USERNAME=cred["username"], KAGGLE_KEY=cred["key"])
        local = io.PROJECT.parent / "tfb_kaggle_download"; local.mkdir(exist_ok=True)
        call = ("from kaggle.api.kaggle_api_extended import KaggleApi; "
                "api=KaggleApi(); api.authenticate(); "
                f"api.competition_download_files({COMPETITION!r}, path={str(local)!r}, force=True, quiet=True)")
        subprocess.run([os.sys.executable, "-c", call], env=env, check=True)
        downloaded = local / archive.name
        if not zipfile.is_zipfile(downloaded): raise ValueError("Kaggle download is not a ZIP archive")
        temp = archive.with_suffix(".tmp"); shutil.copy2(downloaded, temp); temp.replace(archive)
        runner.atomic_json(marker, dict(competition=COMPETITION, size=archive.stat().st_size, sha256=digest(archive)))
    record = json.loads(marker.read_text())
    if not archive.exists() or archive.stat().st_size != record["size"]: raise ValueError("Incomplete raw archive cache")
    done = io.RELEASE / ".colab_raw_ready.json"
    if not done.exists() or json.loads(done.read_text()) != record:
        local = io.PROJECT.parent / "tfb_raw.local.zip"; shutil.copy2(archive, local)
        if digest(local) != record["sha256"]: raise ValueError("Raw archive checksum mismatch")
        staging = io.PROJECT.parent / "tfb_raw_unpacked"; extract_zip(local, staging)
        candidates = [p.parent.parent for p in staging.rglob("config/corridors.json")]
        if not candidates:
            # Some Kaggle downloads wrap the released package in one inner ZIP.
            for inner in sorted(staging.rglob("*.zip")):
                extract_zip(inner, inner.parent / (inner.stem + "_unpacked"))
            candidates = [p.parent.parent for p in staging.rglob("config/corridors.json")]
        if len(candidates) != 1: raise ValueError("Cannot identify unique competition release root")
        release = candidates[0]
        for name in ("task1", "task2", "task3", "task4", "corridors", "config"):
            if not (release / name).is_dir(): raise ValueError(f"Missing release directory {name}")
        for name in ("task1", "task2", "task3", "task4", "corridors", "config", "sample_submission.csv", "submission_key.csv"):
            source = release / name
            if source.is_dir(): shutil.copytree(source, io.RELEASE / name, dirs_exist_ok=True)
            else: shutil.copy2(source, io.RELEASE / name)
        runner.atomic_json(done, record)
    print(f"Public release ready; cached archive: {archive}")


def feature_bundle(drive_root, final=False, build=True):
    """Build once in Colab; float16 standardized inputs and float32 residual labels."""
    from tfb.t1 import train as T, features as FE
    from .export import export_hold, export_final
    T.SEED, T.TRANS, T.TW, T.CW, T.DENSITY, FE.FDF = 42, 8, 1.0, 1.0, False, False
    sources = {r: digest(io.PROJECT / r) for r in FEATURE_SOURCES}
    # Include transformation code and source bundle, so stale features cannot masquerade as a fresh export.
    identity = dict(source_sha256=sources, exporter_sha256=digest(Path(__file__).with_name("export.py")),
                    components_sha256=digest(component_manifest()), trans=8, seed=42, schema=2)
    cache_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    root = Path(drive_root) / "features" / cache_id / ("final" if final else "holdouts")
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["cache_identity"] != identity: raise ValueError("Feature cache identity mismatch")
        print(f"Reusing features: {root}"); return root
    if not build:
        raise FileNotFoundError(f"Waiting for the GPU feature builder: {manifest_path}")
    os.environ["TFB_MLP_STAGING"] = str(io.PROJECT.parent / "tfb_feature_staging" / cache_id)
    panels = io.panels()
    meta = {"final": export_final(root, panels)} if final else {h: export_hold(root, h, panels) for h in ("last30", "autumn")}
    manifest = dict(schema=2, kind="final" if final else "holdout", cache_identity=identity,
                    source_sha256=sources, control=CONTROL, family=io.family_of(), seed=42, trans=8,
                    alphas=json.loads((io.PROJECT / "reports/physics_alphas_pp2.json").read_text()),
                    holdouts=meta, versions={k: importlib.metadata.version(k) for k in ("torch", "numpy", "pandas", "pyarrow")})
    manifest["bundle_id"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    runner.atomic_json(manifest_path, manifest)
    return root


def mac_baseline_seeds(bundle, hold):
    panels = json.loads((bundle / 'manifest.json').read_text())['holdouts'][hold]['panels']
    if all((io.CACHE/'t1_runs'/f'{tag}_{hold}'/f'{p}.npz').is_file()
           for tag in ('mlp8e20','mlp8e20s7','mlp8e20s123') for p in panels):
        return [42,7,123]
    return [42]


def mac_predictions(bundle, hold, destination):
    """Convert original rows/sp/fl caches to compact target-cell arrays; average in the scorer."""
    meta = json.loads((bundle / "manifest.json").read_text())["holdouts"][hold]
    tags = ["mlp8e20", "mlp8e20s7", "mlp8e20s123"] if len(mac_baseline_seeds(bundle, hold)) == 3 else ["mlp8e20"]
    directories = []
    for tag in tags:
        out = destination / f"{tag}_{hold}"; out.mkdir(parents=True, exist_ok=True); directories.append(out)
        for panel in meta["panels"]:
            with np.load(bundle / hold / "score" / f"{panel}.npz") as c, np.load(io.CACHE / "t1_runs" / f"{tag}_{hold}" / f"{panel}.npz") as z:
                if not np.array_equal(z["rows"], c["rows"]): raise ValueError("Mac prediction row mismatch")
                rr, cc = c["target_row"], c["target_col"]
                np.savez_compressed(out / f"{panel}.npz", row=c["rows"][rr], col=cc,
                                    sp=z["sp"][rr, cc].astype(np.float32), fl=z["fl"][rr, cc].astype(np.float32))
    return directories


def gate(bundle, results, config, seeds=(42,), precision="fp32", device="cuda"):
    cfg, tag = resolve_config(config); results = Path(results)
    if not seeds or len(set(seeds)) != len(seeds): raise ValueError("SEEDS must be nonempty and unique")
    manifest = json.loads((bundle / "manifest.json").read_text()); rows = []
    if device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("Choose an L4 runtime")
    dev = torch.device(device)
    for hold in ("last30", "autumn"):
        meta = manifest["holdouts"][hold]; data = None
        jobs = [(CONTROL, 0, 42, "fp32", results / "control")]
        jobs += [(cfg, 1, s, precision, results / tag) for s in seeds if not (cfg == CONTROL and s == 42 and precision == "fp32")]
        for c, i, seed, prec, root in jobs:
            directory = root / f"cfg{i:03d}/seed{seed}_{prec}" / hold
            if (directory / "done.json").exists():
                z = json.loads((directory / "done.json").read_text())
                if z["bundle_id"] != manifest["bundle_id"] or z["config"] != c:
                    raise ValueError(f"Stale result: {directory}")
            if not (directory / "done.json").exists():
                if data is None: data = runner.load_arrays([bundle / hold / "train" / f"{p}.parquet" for p in meta["panels"]], meta)
                runner.run_one(bundle, root, hold, c, i, seed, prec, dev, data)
        control = results / "control/cfg000/seed42_fp32" / hold
        baseline = evaluate(bundle, hold, [control])
        mac = mac_predictions(bundle, hold, results / "mac")
        mac_score = evaluate(bundle, hold, mac)
        for name, dirs in [("Colab e20 control", [control]), ("Mac e20 " + ("42+7+123" if hold == "last30" else "42"), mac)]:
            s = evaluate(bundle, hold, dirs)
            rows.append(dict(hold=hold, model=name, seeds="42" if name.startswith("Colab") else ("42+7+123" if hold == "last30" else "42"),
                             **s, dJ_colab=s["J"]-baseline["J"], dJ_mac=s["J"]-mac_score["J"]))
        pred_dirs = []
        for seed in seeds:
            d = control if cfg == CONTROL and seed == 42 and precision == "fp32" else results / tag / f"cfg001/seed{seed}_{precision}" / hold
            pred_dirs.append(d)
            s = evaluate(bundle, hold, [d]); record = json.loads((d / "done.json").read_text())
            rows.append(dict(hold=hold, model=tag, seeds=str(seed), precision=precision,
                             seconds_per_epoch=record["seconds_per_epoch"], epochs_run=record["epochs_run"],
                             **s, dJ_colab=s["J"]-baseline["J"], dJ_mac=s["J"]-mac_score["J"]))
        if len(seeds) > 1:
            s = evaluate(bundle, hold, pred_dirs)
            rows.append(dict(hold=hold, model=tag, seeds="+".join(map(str, seeds)), precision=precision,
                             **s, dJ_colab=s["J"]-baseline["J"], dJ_mac=s["J"]-mac_score["J"]))
        del data; gc.collect()
    table = pd.DataFrame(rows); table.to_csv(results / f"gate_{tag}_{precision}.csv", index=False)
    runner.atomic_json(results / tag / "gate_manifest.json", dict(config=cfg, seeds=list(seeds), precision=precision,
                       bundle_id=manifest["bundle_id"], gate_file=f"gate_{tag}_{precision}.csv"))
    return table


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


def assemble_component(mlp, destination, config, seeds, onset="Ahyb", check=False):
    """Run the repository's exact blend -> pp2 -> assemble/checks path."""
    from tfb.t1.blend import main as blend
    from tfb.t3.apply import main as pp2
    from tfb.submit.assemble import main as assemble
    from tfb.submit.checks import check as checks
    cfg, tag = resolve_config(config)
    name = "colab_check" if check else f"colab_{tag}_s{'-'.join(map(str, seeds))}_{onset}"
    raw = io.OUTPUTS / "t1" / f"state_{name}.csv"; raw.parent.mkdir(parents=True, exist_ok=True)
    blended = io.OUTPUTS / "t1" / f"state_{name}_pp2.csv"
    blend([str(io.OUTPUTS / "t1/state_submission_ens_v6t1s3.csv") + "@0.5", str(mlp) + "@0.5"], raw)
    pp2(raw, blended, "fd_v_a0.5", "", io.PROJECT / "reports/physics_alphas_pp2.json")
    output = assemble([name, "--state", str(blended), "--queue", str(io.OUTPUTS / "t2" / QUEUE[onset]),
                       "--odme", str(io.OUTPUTS / "t4/path_flow_submission.csv"),
                       "--note", "Colab: cached LGBM/queue/ODME + 0.5 MLP -> pp2; no automatic upload"])
    errs = checks(output)
    if errs: raise ValueError(f"Submission checks failed: {errs}")
    sha = digest(output)
    expected = EXPECTED_SHA if onset == "Ahyb" else ROBUST_SHA
    if check and sha != expected: raise ValueError(f"Assembly regression: {sha} != {expected}")
    metadata = json.loads(output.with_suffix(".manifest.json").read_text())
    metadata.update(mode="check" if check else "submit", commit=code_revision(),
                    config=cfg, seeds=list(seeds), onset=onset, sha256=sha,
                    workflow_sha256=digest(Path(__file__)),
                    code_sha256={p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                    component_bundle_sha256=digest(component_manifest()), checks="OK")
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    saved = destination / output.name
    tmp = saved.with_suffix(".csv.tmp"); shutil.copy2(output, tmp); tmp.replace(saved)
    if digest(saved) != sha: raise IOError("Drive CSV checksum mismatch")
    runner.atomic_json(saved.with_suffix(".manifest.json"), metadata)
    saved.with_suffix(".sha256").write_text(f"{sha}  {saved.name}\n")
    print(f"Saved {saved}\nsha256 {sha}")
    return saved


def check_cached(destination):
    # Canonical check is always Ahyb; ONSET changes only newly-built candidates.
    return assemble_component(io.OUTPUTS / "t1/state_submission_mlp8e20_s3.csv", destination,
                              "e20", [42, 7, 123], onset="Ahyb", check=True)
