"""Strongly regularized additive/low-order onset physics experiments.

Explicit local traffic physics only. No location, historical activation,
static FD/geometry identity, mixed-peak capacity or legacy boundary geometry
is admitted. Future head labels select offline LOBO partitions, never runtime
features or deployment routing.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .io import dump_json, panel_names
from .run import PROJECT, log
from .train import Classifier, decode_probabilities, score_window
from .v2_experiment import load_model
from ..io import RELEASE as RELEASE_ROOT   # TFB_RELEASE (run.py --data), else data/kaggle_public


FEATURES = (
    "v4_flow_discharge_only_ratio_now", "v4_flow_discharge_only_ratio_slope30",
    "v4_flow_discharge_only_projected30", "v4_minutes_to_discharge_capacity",
    "v4_upstream_flow_discharge_only_ratio_now", "v4_upstream_arrival30_discharge_only_ratio",
    "v4_density_ratio_now", "v4_density_ratio_lag15", "v4_density_ratio_lag30",
    "v4_density_ratio_slope15", "v4_density_ratio_slope30",
    "v4_speed_ratio_now", "v4_speed_ratio_lag15", "v4_speed_ratio_lag30",
    "v4_speed_ratio_slope15", "v4_speed_ratio_slope30", "v4_speed_drop15",
    "v4_upstream3_density_ratio_now", "v4_upstream3_density_imbalance", "v4_upstream3_speed_ratio_now",
    "v4_downstream3_density_ratio_now", "v4_downstream3_density_imbalance", "v4_downstream3_speed_ratio_now",
    "v4_day_discharge_flow_peak_ratio", "v4_day_discharge_flow_peak120_ratio",
    "v4_day_minutes_since_discharge_near_capacity", "v4_day_slow70_observed_fraction",
    "v4_day_minutes_since_slow70", "v4_day_speed_minimum_ratio", "v4_day_density_peak_ratio",
    "ramp_on_ratio_now", "ramp_on_ratio_slope15", "ramp_on_ratio_slope30", "ramp_on_observed_fraction",
)
CONTRAST_FEATURES = tuple(f"v4_downstream{width}_discharge_{name}" for width in (5, 10)
    for name in ("capacity_contrast", "demand_stress", "bottleneck_stress", "demand_projected30"))
VARIANTS = {"stump500": (1, 500, 2), "stump1000": (1, 1000, 2),
            "depth2_500": (2, 500, 4), "depth2_1000": (2, 1000, 4),
            "depth4_100": (4, 100, 15), "depth4_200": (4, 200, 15),
            "depth6_100": (6, 100, 31), "contrast_depth4_100": (4, 100, 15),
            "contrast_depth6_100": (6, 100, 31), "fd_depth4_100": (4, 100, 15),
            "fd_depth6_100": (6, 100, 31), "balanced_contrast_depth6_100": (6, 100, 31),
            "balanced_fd_depth6_100": (6, 100, 31), "extra_contrast_depth6_100": (6, 100, 31),
            "extra_balanced_contrast_depth6_100": (6, 100, 31)}


def compact_mask(names, variant=None):
    """Shared full-schema mask: 34 base, 42 contrast, or 33 FD-dropout fields."""
    if len(names) != len(set(names)):
        raise ValueError("Compact onset schema contains duplicate names")
    family = variant.removeprefix("extra_").removeprefix("balanced_") if variant else ""
    features = FEATURES + (CONTRAST_FEATURES if family.startswith("contrast_") else ())
    if family.startswith("fd_"):
        features = tuple(n for n in features if n != "v4_day_minutes_since_discharge_near_capacity")
    missing = set(features) - set(names)
    if missing:
        raise ValueError(f"Compact onset physics fields missing: {sorted(missing)}")
    return np.array([name in features for name in names], bool)


def load_compact_pool(args, fold, *, include_extra=False):
    xs, ys, ws, panels, heads, windows, ratios = [], [], [], [], [], [], []
    names = None
    for pi, panel in enumerate(args.panels):
        paths = [args.output / "v4/data" / fold / f"{panel}.npz"]
        if include_extra:
            paths.append(args.output / "v4/extra" / fold / f"{panel}.npz")
        for path in paths:
            with np.load(path) as d:
                current_names = d["feature_names"].tolist()
                if names is not None and current_names != names:
                    raise ValueError("Panel feature names differ")
                names = current_names
                # Retain only the 42-field union, then select a variant's subset.
                full = d["train_x"]
                xs.append(full[:, compact_mask(names, "contrast_")])
                with np.load(args.output / "v4/data" / fold / f"{panel}_capacity.npz") as cap:
                    cd = cap["capacity_discharge_only"]
                link = d["train_link_index"]
                ratios.append(cd[link] / full[:, names.index("capacity_total_vph")])
                del full
                ys.append(d["train_y"]); ws.append(d["train_weight"])
                heads.append(d["train_head_index"]); windows.append(d["train_window_id"])
                panels.append(np.full(len(ys[-1]), pi, np.int16))
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(ws),
            np.concatenate(panels), np.concatenate(heads), np.concatenate(windows), names, np.concatenate(ratios))


def _hash(value):
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast("B")).hexdigest()


def fd_dropout(x, names, cd_to_fd):
    """Replace ALL retained window-cell CD denominators by FD, without labels.

    The caller applies this uniformly to every retained window, never to a
    target class or head. Event age cannot be reconstructed from a summary
    and must be omitted. Stress and slopes rescale; stored projected ratios
    were clipped before this transform, so their rescaling is an augmentation
    approximation when those caps bind. Capacity time is recomputed.
    """
    values = np.asarray(x)
    ratio = np.asarray(cd_to_fd, float)
    if values.ndim != 2 or values.shape[1] != len(names) or ratio.shape != (len(values),):
        raise ValueError("FD dropout requires aligned cell rows and physical ratios")
    if not np.isfinite(ratio).all() or (ratio <= 0).any():
        raise ValueError("CD/FD ratios must be positive finite")
    if "v4_day_minutes_since_discharge_near_capacity" in names:
        raise ValueError("FD dropout must omit the unreconstructable near-capacity age")
    out = np.array(values, float, copy=True)
    rescale = ("v4_flow_discharge_only_ratio_now", "v4_flow_discharge_only_ratio_slope30",
        "v4_flow_discharge_only_projected30", "v4_upstream_flow_discharge_only_ratio_now",
        "v4_upstream_arrival30_discharge_only_ratio", "v4_day_discharge_flow_peak_ratio",
        "v4_day_discharge_flow_peak120_ratio")
    for name in rescale:
        if name in names:
            out[:, names.index(name)] *= ratio
    if "v4_minutes_to_discharge_capacity" in names:
        now = out[:, names.index("v4_flow_discharge_only_ratio_now")]
        slope = out[:, names.index("v4_flow_discharge_only_ratio_slope30")]
        out[:, names.index("v4_minutes_to_discharge_capacity")] = np.clip(np.divide(
            1 - now, slope, out=np.full(len(out), np.nan), where=slope > 1e-5), -60., 240.)
    return out.astype(values.dtype, copy=False)


def balance_window_weights(panels, heads, windows, keep=None):
    """Train-only head-domain balance, computed AFTER whole-window exclusion.

    Unknown heads retain factor one; known heads use the requested square-root
    inverse window-frequency weighting, clipped [.5,10]. One event, rather
    than its number of link cells, contributes to each domain's counts.
    """
    panels, heads, windows = np.asarray(panels), np.asarray(heads), np.asarray(windows)
    keep = np.ones(len(heads), bool) if keep is None else np.asarray(keep, bool)
    if panels.shape != heads.shape or windows.shape != heads.shape or keep.shape != heads.shape:
        raise ValueError("Window/domain metadata must align")
    factors = np.full(len(heads), np.nan)
    retained = np.flatnonzero(keep)
    if not len(retained):
        return factors
    _, first = np.unique(windows[keep], return_index=True)
    first = retained[first]
    for panel in np.unique(panels[keep]):
        events = first[(panels[first] == panel) & (heads[first] >= 0)]
        domain, counts = np.unique(heads[events], return_counts=True)
        factors[keep & (panels == panel) & (heads < 0)] = 1.
        for head, count in zip(domain, counts):
            factor = np.clip(np.sqrt(len(events) / (len(domain) * count)), .5, 10.)
            factors[keep & (panels == panel) & (heads == head)] = factor
    return factors


def fit_compact(args, pool, fold, variant, *, seed=42, heldout=None, radius=0):
    x, y, w, panels, heads, windows, names, fd_ratio = pool
    depth, leaf, leaves = VARIANTS[variant]
    keep = np.ones(len(y), bool)
    folder = args.output / "v4/compact/models" / fold / variant
    if heldout is None:
        path = folder / f"seed{seed}.txt"
    else:
        panel, head = heldout
        keep = ~((panels == args.panels.index(panel)) & (heads >= 0) & (np.abs(heads - head) <= radius))
        path = folder / "lobo" / f"radius{radius}" / f"{panel}_head{head}.txt"
    mask = compact_mask(names, variant)
    union_mask = compact_mask(names, "contrast_")
    selected = x[keep][:, mask[union_mask]]
    target_y, target_w = y[keep], w[keep]
    balanced = variant.removeprefix("extra_").startswith("balanced_")
    if balanced:
        factors = balance_window_weights(panels, heads, windows, keep)
        target_w = target_w * factors[keep]
    dropout = variant.removeprefix("extra_").removeprefix("balanced_").startswith("fd_")
    if dropout:
        altered = fd_dropout(selected, list(np.asarray(names)[mask]), fd_ratio[keep])
        selected = np.concatenate([selected, altered])
        target_y = np.concatenate([target_y, target_y])
        target_w = np.concatenate([target_w, target_w * .5])
    params = dict(objective="binary", metric="binary_logloss", learning_rate=.05,
        max_depth=depth, num_leaves=leaves, min_data_in_leaf=leaf, max_bin=63,
        feature_fraction=.9, bagging_fraction=1., lambda_l2=50., verbosity=-1,
        deterministic=True, force_col_wise=True, num_threads=args.threads,
        seed=seed, feature_fraction_seed=seed, bagging_seed=seed, data_random_seed=seed)
    signature = dict(fold=fold, variant=variant, seed=seed, rounds=args.rounds, threads=args.threads,
        params=params, names=list(np.asarray(names)[mask]), full_columns=len(names),
        rows=int(keep.sum()), excluded_rows=int((~keep).sum()),
        excluded_windows=int(len(np.unique(windows[~keep]))), heldout=heldout, radius=radius,
        x_sha256=_hash(selected), y_sha256=_hash(target_y), weight_sha256=_hash(target_w),
        keep_sha256=_hash(keep), schema=f"compact_physics{mask.sum()}", future_head_partition_only=True)
    if dropout:
        signature.update(augmented_rows=len(selected), normalization_dropout_weight=.5,
                         augmentation="every retained window duplicated; all local CD denominators become FD")
    if balanced:
        signature.update(window_head_balance="sqrt(n_windows_panel/(n_heads_panel*n_windows_head)), clipped .5..10; after exclusion",
                         balance_min=float(factors[keep].min()), balance_max=float(factors[keep].max()))
    # JSON converts tuples to lists; compare the canonical serialized form.
    signature = json.loads(json.dumps(signature))
    record = path.with_suffix(".training.json")
    if path.exists():
        if not record.exists() or json.loads(record.read_text()) != signature:
            raise ValueError(f"Stale compact model: {path}")
        model = load_model(path, args.threads)
        if not np.array_equal(model.mask, mask) or model.booster.feature_name() != signature["names"]:
            raise ValueError(f"Cached compact mask/schema mismatch: {path}")
        return model
    log(f"compact {fold}/{variant}/{path.stem}: {keep.sum():,} rows, excluded {len(y)-keep.sum():,}")
    data = lgb.Dataset(selected, label=target_y, weight=target_w,
        feature_name=list(np.asarray(names)[mask]), free_raw_data=True)
    booster = lgb.train(params, data, num_boost_round=args.rounds)
    model = Classifier(booster, mask, threads=args.threads)
    path.parent.mkdir(parents=True, exist_ok=True); model.save(path)
    dump_json(record, signature)
    pd.DataFrame(dict(feature=np.asarray(names)[mask], gain=booster.feature_importance("gain"))).sort_values(
        "gain", ascending=False).to_csv(path.with_suffix(".importance.csv"), index=False)
    del data, selected; gc.collect()
    return model


def predict_normal(args, pool, fold, variant, seeds):
    models = [fit_compact(args, pool, fold, variant, seed=seed) for seed in seeds]
    if fold == "final":
        return
    dest = args.output / "v4/compact/probabilities" / fold / variant; dest.mkdir(parents=True, exist_ok=True)
    for panel in args.panels:
        source = args.output / "v4/data" / fold
        with np.load(source / f"{panel}.npz") as d:
            x = d["eval_x"]; flat = x.reshape(-1, x.shape[-1])
            ps = np.stack([model.predict(flat).reshape(x.shape[:2]) for model in models])
        meta = pd.read_csv(source / f"{panel}_eval.csv")
        np.savez(dest / f"{panel}.npz", probability=ps.mean(axis=0), seed_probability=ps,
            seeds=np.asarray(seeds), window_id=meta.window_id.to_numpy(str), origin=meta.origin.to_numpy(str))


def target_groups(args, fold):
    source = args.output / "v4/data" / fold
    training = pd.read_csv(source / "D7_I405_S_train_windows.csv")
    dominant = int(training[training.head_index.ge(0)].head_index.value_counts().index[0])
    return [("D12_I5_N", 105), ("D12_I5_N", 238), ("D12_I5_S", 45),
            ("D7_I10_E", 8), ("D7_I10_W", 25), ("D7_I405_S", dominant)]


def predict_lobo(args, pool, fold, variant, radius):
    dest = args.output / "v4/compact/probabilities" / fold / variant / f"lobo_radius{radius}"
    dest.mkdir(parents=True, exist_ok=True)
    for panel, head in target_groups(args, fold):
        model = fit_compact(args, pool, fold, variant, heldout=(panel, head), radius=radius)
        source = args.output / "v4/data" / fold
        meta = pd.read_csv(source / f"{panel}_eval.csv")
        take = np.flatnonzero(meta.head_index.to_numpy(int) == head)
        with np.load(source / f"{panel}.npz") as d:
            x = d["eval_x"][take]
        if len(x):
            p = model.predict(x.reshape(-1, x.shape[-1])).reshape(x.shape[:2])
        else:
            p = np.empty((0, x.shape[1]))
        np.savez(dest / f"{panel}_head{head}.npz", probability=p,
            window_id=meta.iloc[take].window_id.to_numpy(str), origin=meta.iloc[take].origin.to_numpy(str))


def _score(probability, truth, eligible):
    p = np.zeros(truth.shape, float); p[-1] = probability
    pred = decode_probabilities(p, onset=True).astype(bool)
    return dict(iou=score_window(pred, truth, eligible), predicted_cells=int(pred.sum()),
        true_cells=int((truth & eligible).sum()), fp=int((pred & ~truth & eligible).sum()),
        fn=int((~pred & truth & eligible).sum()))


def evaluate(args, fold, variants, radii):
    rows = []
    for panel in args.panels:
        source = args.output / "v4/data" / fold
        meta = pd.read_csv(source / f"{panel}_eval.csv")
        with np.load(source / f"{panel}_metadata.npz") as d:
            y, e = d["eval_y"], d["eval_eligible"]
            baseline = d["baseline_prediction"]
        rootnormal = args.output / "v4/probabilities" / fold / f"{panel}_normal.npz"
        if rootnormal.exists():
            with np.load(rootnormal) as d:
                assert np.array_equal(d["window_id"], meta.window_id.to_numpy(str))
                regular = d["seed_probability"][0]
        else:
            regular = None
        for variant in variants:
            base = args.output / "v4/compact/probabilities" / fold / variant
            with np.load(base / f"{panel}.npz") as d:
                assert np.array_equal(d["window_id"], meta.window_id.to_numpy(str))
                normal = d["probability"]
            lobo = {}
            for radius in radii:
                for path in (base / f"lobo_radius{radius}").glob(f"{panel}_head*.npz"):
                    with np.load(path) as d:
                        for wid, p in zip(d["window_id"], d["probability"]): lobo[(radius, str(wid))] = p
            root_lobo = {}
            for path in (args.output / "v4/lobo" / fold / "radius0").glob(f"{panel}_head*.npz"):
                with np.load(path) as d:
                    for wid, p in zip(d["window_id"], d["probability"]): root_lobo[str(wid)] = p
            for i, record in enumerate(meta.to_dict("records")):
                common = {**record, "fold": fold, "variant": variant}
                rows.append({**common, "model": "compact_normal", "radius": -1,
                             **_score(normal[i], y[i], e[i])})
                for radius in radii:
                    key = (radius, record["window_id"])
                    if key not in lobo: continue
                    for name, pp in [("compact_lobo", lobo[key]), ("compact_normal_matched", normal[i]),
                                     ("root316_normal_matched", regular[i] if regular is not None else None),
                                     ("root316_lobo_radius0_matched", root_lobo.get(record["window_id"]))]:
                        if pp is not None:
                            rows.append({**common, "model": name, "radius": radius, **_score(pp, y[i], e[i])})
                    pred = baseline[i].astype(bool)
                    rows.append({**common, "model": "v2_frozen_matched", "radius": radius,
                        "iou": score_window(pred, y[i], e[i]), "predicted_cells": int(pred.sum()),
                        "true_cells": int((y[i] & e[i]).sum()), "fp": int((pred & ~y[i] & e[i]).sum()),
                        "fn": int((~pred & y[i] & e[i]).sum())})
    detail = pd.DataFrame(rows)
    dest = args.output / "v4/compact"; dest.mkdir(parents=True, exist_ok=True)
    detail_path = dest / f"{fold}_windows.csv"
    if detail_path.exists():
        previous = pd.read_csv(detail_path)
        detail = pd.concat([previous[~previous.variant.isin(variants)], detail], ignore_index=True)
    detail.to_csv(detail_path, index=False)
    summaries = []
    for keys, g in detail.groupby(["variant", "model", "radius", "panel"], sort=True):
        base = dict(zip(["variant", "model", "radius", "panel"], keys))
        primary = g.official_like.astype(str).str.lower().eq("true").to_numpy()
        known = g.head_index.ge(0) & ~g.head_prior_unsupported.astype(bool)
        rare = (known & g.nonrecurrent_onset.astype(bool)).to_numpy()
        for subset, keep in [("primary", primary), ("dense", np.ones(len(g), bool))]:
            for stratum, part in [("all", np.ones(len(g), bool)), ("rare", rare), ("recurrent", known.to_numpy() & ~rare)]:
                z = g[keep & part]
                if len(z):
                    summaries.append({**base, "fold": fold, "subset": subset, "stratum": stratum, "n": len(z),
                        **{f: float(z[f].mean()) for f in ("iou", "predicted_cells", "true_cells", "fp", "fn")}})
    summary = pd.DataFrame(summaries)
    overall = []
    for keys, g in summary.groupby(["variant", "model", "radius", "subset", "stratum"]):
        overall.append({**dict(zip(["variant", "model", "radius", "subset", "stratum"], keys)),
            "fold": fold, "panel": "SUPPORTED_PANEL_MEAN", "supported_panels": len(g), "n": int(g.n.sum()),
            **{f: float(g[f].mean()) for f in ("iou", "predicted_cells", "true_cells", "fp", "fn")}})
    pd.concat([summary, pd.DataFrame(overall)], ignore_index=True).to_csv(dest / f"{fold}_scores.csv", index=False)
    log("compact summary\n" + pd.DataFrame(overall).query("subset=='primary' and stratum=='all'").to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=RELEASE_ROOT)
    p.add_argument("--output", type=Path, default=PROJECT / "outputs/t2")
    p.add_argument("--fold", choices=["janfeb", "octnov", "final"], default="janfeb")
    p.add_argument("--variant", action="append", choices=list(VARIANTS))
    p.add_argument("--seed", action="append", type=int)
    p.add_argument("--radius", action="append", type=int)
    p.add_argument("--stage", choices=["normal", "lobo", "evaluate", "all"], default="all")
    p.add_argument("--rounds", type=int, default=250); p.add_argument("--threads", type=int, default=2)
    args = p.parse_args(); args.panels = panel_names(args.root)
    variants, seeds, radii = args.variant or list(VARIANTS), args.seed or [42], args.radius or [0]
    pool, loaded_extra = None, None
    for variant in variants:
        include_extra = variant.startswith("extra_")
        if args.stage != "evaluate" and (pool is None or loaded_extra != include_extra):
            pool = load_compact_pool(args, args.fold, include_extra=include_extra)
            loaded_extra = include_extra
        if args.stage in {"normal", "all"}: predict_normal(args, pool, args.fold, variant, seeds)
        if args.stage in {"lobo", "all"} and args.fold != "final":
            for radius in radii: predict_lobo(args, pool, args.fold, variant, radius)
    if args.stage in {"evaluate", "all"} and args.fold != "final": evaluate(args, args.fold, variants, radii)


if __name__ == "__main__": main()
