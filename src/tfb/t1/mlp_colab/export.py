"""Memory-bounded export; never reads validation/private truth or organizer files.

PYTHONPATH=src .venv/bin/python -m tfb.t1.mlp_colab.export
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import zipfile

import torch  # OpenMP import order on macOS, as in mlp.py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tfb import cache, io
from tfb.t1 import train as T, features as FE
from tfb.t1.mlp import Prep, TARGETS
from tfb.t3.proxy import ramp_ok
from .config import CONTROL


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_features(path, A, F, P, R, **extra):
    """Native Parquet FLOAT16, not a pandas float32 upcast; labels stay FLOAT32."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = {f"a{i}": pa.array(A[:, i].astype(np.float16)) for i in range(A.shape[1])}
    cols.update({f"f{i}": pa.array(F[:, i].astype(np.float16)) for i in range(F.shape[1])})
    cols.update(panel=pa.array(P.astype(np.int16)), regime=pa.array(R.astype(np.int8)))
    cols.update({k: pa.array(v) for k, v in extra.items()})
    pq.write_table(pa.table(cols), path, compression="zstd", row_group_size=131072)


def fit_prep(paths, cols, hold):
    """Same float32 inputs/quantiles as Prep.fit, one feature in RAM at a time."""
    prep = Prep(); prep.cols = cols; prep.num = [c for c in cols if c not in FE.CAT]
    med, iqr, flag = [], [], []
    tr = ~hold
    for j, c in enumerate(prep.num):
        a = np.concatenate([pd.read_parquet(p, columns=[c])[c].to_numpy(np.float32) for p in paths])
        if np.isnan(a).any():
            flag.append(j)
        s = a[tr]; q25, m, q75 = np.nanpercentile(s, [25, 50, 75])
        med.append(np.nan_to_num(m))
        iqr.append(q75 - q25 if q75 - q25 > 1e-6 else np.nanstd(s) + 1e-6)
    prep.med = np.asarray(med, np.float32); prep.iqr = np.asarray(iqr, np.float32)
    prep.flag = np.asarray(flag, np.int64)
    prep.n_panel = max(FE.PANEL_ID.values()) + 1
    return prep


def export_hold(root, name, panels):
    out = root / name; out.mkdir(parents=True, exist_ok=True)
    marker = out / "meta.json"
    if marker.exists():
        print(f"{name}: completed export reused", flush=True)
        return json.loads(marker.read_text())
    days = T.holdout_days(name)
    rng, rng_t = np.random.default_rng(42), np.random.default_rng(43)
    raw = Path(os.environ.get("TFB_MLP_STAGING", root.parent / (root.name + "_staging"))) / name
    raw.mkdir(parents=True, exist_ok=True)
    paths, counts, cols = [], [], None
    # Exactly build_training(days, panels, test_splits=('train',)), streamed by panel.
    for panel in panels:
        pn = cache.load(panel, "train"); rows_ok = ~T.day_mask(pn, days)
        F = FE.panel_fields(pn, pn, exclude_days=days)
        r, c = T.sample_cells(pn, rows_ok, T.ROWS_PER_PANEL, rng)
        df = FE.build(pn, F, r, c); del F
        parts = [(df, *T.labels(df, pn.speed[r, c], pn.flow[r, c], pn.lanes[c]))]
        parts += T.trans_rows(pn, pn, ~rows_ok, days, rng_t, T.TRANS_ROWS)
        frame = pd.concat([p[0] for p in parts], ignore_index=True)
        cols = list(frame.columns)
        for i, k in enumerate(TARGETS):
            frame["y_" + k] = np.concatenate([p[i + 1] for p in parts]).astype(np.float32)
        p = raw / f"{panel}.parquet"; frame.to_parquet(p, compression="zstd", index=False)
        counts.append(len(frame)); paths.append(p)
        print(f"{name} raw {panel}: {len(frame):,}; cumulative {sum(counts):,}", flush=True)
        del pn, frame, parts, df; gc.collect()
    hold = rng.random(sum(counts)) < 0.05
    prep = fit_prep(paths, cols, hold)
    Y = np.ascontiguousarray(np.concatenate([pd.read_parquet(p, columns=["y_" + k for k in TARGETS]).to_numpy(np.float32) for p in paths]))
    mu, sd = Y[~hold].mean(0), Y[~hold].std(0)
    offset = 0
    for panel, path, n in zip(panels, paths, counts):
        frame = pd.read_parquet(path)
        A, F, P, R = prep.transform(frame[cols])
        write_features(out / "train" / path.name, A, F, P, R,
                       y_speed=Y[offset:offset+n, 0], y_flow_lane=Y[offset:offset+n, 1],
                       valid=hold[offset:offset+n])
        offset += n; del frame, A, F, P, R; gc.collect()
        pn = cache.load(panel, "train"); rows = T.day_mask(pn, days); ix = np.flatnonzero(rows)
        r, c = np.nonzero(pn.target & rows[:, None])
        fields = FE.panel_fields(pn, pn, days); df = FE.build(pn, fields, r, c)[cols]; del fields
        A, F, P, R = prep.transform(df)
        # Reconstruction operands must retain float32 precision, unlike model features.
        write_features(out / "predict" / path.name, A, F, P, R,
                       base_s=df.base_s.to_numpy(np.float32), base_f=df.base_f.to_numpy(np.float32),
                       lanes=pn.lanes[c].astype(np.float32), row=r.astype(np.int32), col=c.astype(np.int16))
        context = {k: pn[k][ix] for k in ("speed", "flow", "elig", "target")}
        context.update({k: pn[k] for k in ("lanes", "v_f", "cap", "k_crit", "k_jam", "length", "links")})
        context.update(panel=np.asarray(panel), rows=ix.astype(np.int32),
                       regime=pn.regime[days], rok=ramp_ok(pn)[ix], target_row=(r - ix[0]).astype(np.int32),
                       target_col=c.astype(np.int16))
        lgb_tags = ["c_trans8", "cw3_trans8", "c_trans5"] if name == "last30" else ["c_trans8"]
        ls, lf = [], []
        for tag in lgb_tags:
            with np.load(T.run_dir(tag, name) / f"{panel}.npz") as z:
                if not np.array_equal(z["rows"], ix):
                    raise ValueError(f"{tag}/{panel}: holdout row mismatch")
                ls.append(z["sp"]); lf.append(z["fl"])
        context.update(lgb_sp=np.mean(ls, axis=0).astype(np.float32), lgb_fl=np.mean(lf, axis=0).astype(np.float32))
        (out / "score").mkdir(exist_ok=True)
        np.savez_compressed(out / "score" / f"{panel}.npz", **context)
        print(f"{name} packed {panel}: {len(r):,} prediction cells", flush=True)
        del pn, context, df, A, F, P, R, ls, lf; gc.collect()
    np.savez(out / "prep.npz", **{k: np.asarray(v) for k, v in prep.__dict__.items()}, mu=mu, sd=sd)
    meta = dict(name=name, panels=panels, n_rows=sum(counts), n_valid=int(hold.sum()),
                n_num=len(prep.num), n_flag=len(prep.flag), n_panel=prep.n_panel,
                mu=mu.tolist(), sd=sd.tolist(), counts=counts,
                split_sha256=hashlib.sha256(hold.tobytes()).hexdigest(),
                train_tags=lgb_tags, days=days.tolist())
    marker.write_text(json.dumps(meta, indent=2))
    return meta


def export_final(root, panels):
    """Full train with eight observed-cell masks on validation/private, no test truth.

    This is a separate optional archive: the sweep need not upload final features.
    Training rows and the validation draw are frozen at seed 42, as for holdouts.
    """
    out = root / "final"; out.mkdir(parents=True, exist_ok=True)
    marker = out / "meta.json"
    if marker.exists(): return json.loads(marker.read_text())
    raw = Path(os.environ.get("TFB_MLP_STAGING", root.parent / (root.name + "_staging"))) / "final"; raw.mkdir(parents=True, exist_ok=True)
    rng, rng_t = np.random.default_rng(42), np.random.default_rng(43)
    paths, counts, cols = [], [], None
    for panel in panels:
        pn = cache.load(panel, "train"); fields = FE.panel_fields(pn, pn, None)
        r, c = T.sample_cells(pn, np.ones(len(pn.mspeed), bool), T.ROWS_PER_PANEL, rng)
        df = FE.build(pn, fields, r, c); del fields
        parts = [(df, *T.labels(df, pn.speed[r, c], pn.flow[r, c], pn.lanes[c]))]
        for split in ("validation", "private"):
            pt = cache.load(panel, split)
            parts += T.trans_rows(pt, pn, np.ones(len(pt.mspeed), bool), None, rng_t, T.TRANS_ROWS // 2)
            del pt
        frame = pd.concat([p[0] for p in parts], ignore_index=True); cols = list(frame.columns)
        for i, k in enumerate(TARGETS):
            frame["y_" + k] = np.concatenate([p[i+1] for p in parts]).astype(np.float32)
        path = raw / f"{panel}.parquet"; frame.to_parquet(path, compression="zstd", index=False)
        paths.append(path); counts.append(len(frame))
        print(f"final raw {panel}: {len(frame):,}; cumulative {sum(counts):,}", flush=True)
        del frame, parts, df, pn; gc.collect()
    valid = rng.random(sum(counts)) < 0.05
    prep = fit_prep(paths, cols, valid)
    Y = np.ascontiguousarray(np.concatenate([pd.read_parquet(p, columns=["y_"+k for k in TARGETS]).to_numpy(np.float32) for p in paths]))
    mu, sd = Y[~valid].mean(0), Y[~valid].std(0)
    offset = 0
    for panel, path, n in zip(panels, paths, counts):
        df = pd.read_parquet(path)
        write_features(out / "train" / path.name, *prep.transform(df[cols]),
                       y_speed=Y[offset:offset+n, 0], y_flow_lane=Y[offset:offset+n, 1], valid=valid[offset:offset+n])
        offset += n; del df; gc.collect()
        pn_train = cache.load(panel, "train")
        for split in ("validation", "private"):
            pn = cache.load(panel, split); r, c = np.nonzero(pn.target)
            fields = FE.panel_fields(pn, pn_train, None); df = FE.build(pn, fields, r, c)[cols]; del fields
            write_features(out / "predict" / split / path.name, *prep.transform(df),
                           base_s=df.base_s.to_numpy(np.float32), base_f=df.base_f.to_numpy(np.float32),
                           lanes=pn.lanes[c].astype(np.float32), row=r.astype(np.int32), col=c.astype(np.int16))
            if os.environ.get("TFB_TESTMASK"):   # tfb.t1.testmask: targets include hidden cells, no submission template
                del pn, df; gc.collect(); continue
            tpl = io.t1_template(panel, split)
            rr = io.ts_index(tpl.timestamp, {d: i for i, d in enumerate(pn.dates)})
            cc = tpl.link_id.map({l: i for i, l in enumerate(pn.links)}).to_numpy()
            flat, tpl_flat = r * len(pn.links) + c, rr * len(pn.links) + cc
            pos = np.searchsorted(flat, tpl_flat)
            if len(tpl) != len(r) or not np.array_equal(flat[pos], tpl_flat):
                raise ValueError(f"Template/prediction mismatch: {panel}/{split}")
            tpl = tpl[["panel", "timestamp", "station_id", "link_id", "mask_regime"]].copy()
            tpl["pred_pos"] = pos.astype(np.int32)
            dest = out / "templates" / split / path.name; dest.parent.mkdir(parents=True, exist_ok=True)
            tpl.to_parquet(dest, compression="zstd", index=False)
            print(f"final packed {panel}/{split}: {len(r):,} test targets", flush=True)
            del pn, df, tpl; gc.collect()
        del pn_train; gc.collect()
    np.savez(out / "prep.npz", **{k: np.asarray(v) for k, v in prep.__dict__.items()}, mu=mu, sd=sd)
    meta = dict(name="final", panels=panels, n_rows=sum(counts), n_valid=int(valid.sum()),
                n_num=len(prep.num), n_flag=len(prep.flag), n_panel=prep.n_panel,
                mu=mu.tolist(), sd=sd.tolist(), counts=counts,
                split_sha256=hashlib.sha256(valid.tobytes()).hexdigest(), test_splits=["validation", "private"])
    marker.write_text(json.dumps(meta, indent=2)); return meta


def package(root, archive):
    """Also callable after source updates, without rebuilding any data."""
    for rel in ("__init__.py", "io.py", "eval/__init__.py", "eval/score_t1.py",
                "t1/__init__.py", "t3/__init__.py", "t3/post.py", "t3/proxy.py"):
        target = root / "src/tfb" / rel; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(io.PROJECT / "src/tfb" / rel, target)
    dest = root / "src/tfb/t1/mlp_colab"; dest.mkdir(parents=True, exist_ok=True)
    for p in [Path(__file__).parent / name for name in ("__init__.py", "config.py", "runner.py", "scoring.py", "gate.py")]:
        shutil.copy2(p, dest / p.name)
    config = root / "data/kaggle_public/config"; config.mkdir(parents=True, exist_ok=True)
    shutil.copy2(io.RELEASE / "config/corridors.json", config / "corridors.json")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest.pop("bundle_id", None); manifest.pop("files", None)
    files = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*"))
             if p.is_file() and p.name != "manifest.json" and "__pycache__" not in p.parts}
    manifest["files"] = files
    manifest["bundle_id"] = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    archive.parent.mkdir(parents=True, exist_ok=True)
    tmp = archive.with_suffix(".partial.zip")
    # Parquet/npz already compressed; no extra CPU or temporary uncompressed copy.
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for rel in ["manifest.json", *files]:
            z.write(root / rel, rel)
    tmp.replace(archive)
    print(f"ZIP {archive} ({archive.stat().st_size / 1e9:.2f} GB), sha256={digest(archive)}", flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path)
    p.add_argument("--zip", type=Path)
    p.add_argument("--package-only", action="store_true")
    p.add_argument("--final-only", action="store_true", help="Optional separate full-train + test export")
    args = p.parse_args()
    args.root = args.root or io.OUTPUTS / ("t1_mlp_colab/final_bundle" if args.final_only else "t1_mlp_colab/bundle")
    args.zip = args.zip or io.OUTPUTS / ("t1_mlp_colab/t1_mlp_colab_final.zip" if args.final_only else "t1_mlp_colab/t1_mlp_colab.zip")
    if not args.package_only:
        # Set imported constants explicitly: caller's shell must not change the experiment.
        T.SEED, T.TRANS, T.TW, T.CW, T.DENSITY, FE.FDF = 42, 8, 1.0, 1.0, False, False
        panels = io.panels()
        meta = {"final": export_final(args.root, panels)} if args.final_only else {h: export_hold(args.root, h, panels) for h in ("last30", "autumn")}
        manifest = dict(schema=1, kind="final" if args.final_only else "holdout", control=CONTROL, seed=42, trans=8, holdouts=meta,
                        family=io.family_of(), frozen_sampling=True,
                        alphas=json.loads((io.PROJECT / "reports/physics_alphas_pp2.json").read_text()),
                        versions={k: importlib.metadata.version(k) for k in ("torch", "numpy", "pandas", "pyarrow")},
                        source_sha256={r: digest(io.PROJECT / r) for r in
                                       ("src/tfb/t1/mlp.py", "src/tfb/t1/train.py", "src/tfb/t1/features.py")})
        (args.root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    package(args.root, args.zip)


if __name__ == "__main__":
    main()
