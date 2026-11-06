"""Memory-bounded feature export; never reads validation/private truth or organizer files.
"""
from __future__ import annotations
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tfb import cache, io
from tfb.t1 import train as T, features as FE
from tfb.t1.mlp import Prep, TARGETS


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


def export_final(root, panels):
    """Full train with eight observed-cell masks on validation/private, no test truth.

    Training rows and the validation draw are fixed at seed 42.
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
