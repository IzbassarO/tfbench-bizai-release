"""Task 2 v2: two temporal gates, disjoint calibration, frozen v1 output.

All artifacts are isolated under outputs/t2/v2. Nothing is submitted.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import platform
import time
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd

from .io import dump_json, panel_names, resolve_root
from .run import CONDITIONS, PROJECT, log
from .train import Classifier, fit_classifier, score_window
from .v2_decode import decode_calibrated, fit_cardinality_calibration
from .v2_validation import validate_probability_pack, validate_candidate_cohort

WEAK = ("D7_I10_W", "D7_I210_W", "D7_I405_S")
FOLDS = {"sep": "v2_sep_cal", "octnov": "v2_octnov", "dec": "v2_dec_cal",
         "janfeb": "cv_enhanced_release", "final": "final_enhanced_release"}
SEEDS = (42, 137, 271, 419, 655)


def prepare_base(args,fold):
    from .run import prepare
    from .v2_data import CUTOFF
    end={"sep":"2030-10-01","octnov":"2030-12-01","dec":"2031-01-01"}.get(fold)
    a=args_for_fold(args,fold)
    a.train_cutoff=CUTOFF[fold];a.eval_end=end
    prepare(a,final=fold=="final")


def args_for_fold(args, fold, *, extras=False):
    tag = FOLDS[fold]
    return SimpleNamespace(**{**vars(args), "dataset_name": tag + ("_x" if extras else ""),
        "feature_set": "enhanced", "window_policy": "release", "negative_sample": .25,
        "seed": 42, "onset_prior_folds": 5, "reuse_models": True, "refresh": False})


def load_model(path, threads=2):
    return Classifier(lgb.Booster(model_file=str(path)),
                      np.load(path.with_suffix(".mask.npy")), threads=threads)


def load_pool(args, fold, condition, *, extras=False, median=False, incident=False, augment=False):
    source = args.output / "datasets" / (FOLDS[fold] + ("_x" if extras else ""))
    xs, ys, ws = [], [], []
    for panel in args.panels:
        with np.load(source / f"{panel}_{condition}.npz") as data:
            x = data["train_x"]
            y = data["train_median" if median else "train_y"]
            w = data["train_weight"]
            names = data["feature_names"].tolist()
            if incident and panel in WEAK:
                now = x[:, names.index("queue_now")] > .5
                rare = np.nan_to_num(x[:, names.index("location_queue_prior_target")], nan=0) < .05
                tail = np.abs(x[:, names.index("queue_tail_distance_links")]) <= 3
                head = np.abs(x[:, names.index("queue_head_distance_links")]) <= 3
                # Fit on training targets only. Emphasize observed clearing
                # cells, rare queues and moving boundaries, not held-out cases.
                w = w * (1 + 2 * (now & ~y) + 1.5 * (y & rare) + .5 * (tail | head))
            xs.append(x); ys.append(y); ws.append(w)
            if augment and panel in WEAK and "augment_x" in data and len(data["augment_y"]):
                xs.append(data["augment_x"]); ys.append(data["augment_y"])
                ws.append(data["augment_weight"] * .35)
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    del xs, ys, ws
    gc.collect()
    return x, y, w, names


def fit_group(args, fold, condition, variant):
    """Variants are fixed before outer-gate evaluation; model reuse is isolated."""
    extras = variant in {"features", "incident", "augmented", "robust"}
    onset = condition == "queue_onset"
    median = onset or variant == "median_v1"
    x, y, w, names = load_pool(args, fold, condition, extras=extras, median=median,
        incident=variant in {"incident", "augmented"}, augment=variant in {"augmented","robust"})
    members = [(str(seed), variant == "robust", seed) for seed in SEEDS] if onset and variant != "baseline" else (
        [("median", False, 42)] if median else [("full", False, 42), ("no_location", True, 42)])
    result = []
    for member, no_location, seed in members:
        # Frozen v1 Jan-Feb/final models already exist and must stay untouched.
        old = args.output / "models" / FOLDS[fold] / f"{condition}_{member}.txt"
        path = args.output / "v2" / "models" / fold / variant / f"{condition}_{member}.txt"
        signature=dict(fold=fold,condition=condition,variant=variant,seed=seed,rounds=args.rounds,
            threads=args.threads,rows=len(y),feature_names=names,no_location=no_location,
            x_sha256=hashlib.sha256(memoryview(np.ascontiguousarray(x)).cast("B")).hexdigest(),
            y_sha256=hashlib.sha256(memoryview(np.ascontiguousarray(y)).cast("B")).hexdigest(),
            weight_sha256=hashlib.sha256(memoryview(np.ascontiguousarray(w)).cast("B")).hexdigest())
        signature_path=path.with_suffix(".training.json")
        if signature_path.exists() and json.loads(signature_path.read_text())!=signature:
            raise ValueError(f"Cached model training signature mismatch: {path}")
        if variant in {"baseline","median_v1"} and old.exists():
            model = load_model(old, args.threads)
        elif path.exists():
            model = load_model(path, args.threads)
        else:
            log(f"v2 fit {fold}/{variant}/{condition}/{member}: {len(y):,} cells")
            model = fit_classifier(x, y, w, names, no_location=no_location, seed=seed,
                rounds=args.rounds, threads=args.threads, model_path=path)
        expected_names=[n for n,take in zip(names,model.mask) if take]
        if model.booster.feature_name()!=expected_names:
            raise ValueError(f"Cached model feature schema mismatch: {path}")
        if int(model.booster.params.get("seed",-1))!=seed or model.booster.current_iteration()!=args.rounds:
            raise ValueError(f"Cached model parameters mismatch: {path}")
        # V1 model files remain untouched; provenance for their reuse is local.
        dump_json(signature_path,signature)
        result.append(model)
    del x, y, w
    gc.collect()
    return result, extras


def predict_fold(args, fold, condition, variant):
    outdir = args.output / "v2" / "probabilities" / fold
    outdir.mkdir(parents=True, exist_ok=True)
    files = [outdir / f"{panel}_{condition}_{variant}.npz" for panel in WEAK]
    if all(path.exists() and "origin" in np.load(path).files for path in files):
        log(f"v2 probabilities {fold}/{variant}/{condition}: cached")
        return
    models, extras = fit_group(args, fold, condition, variant)
    source = args.output / "datasets" / (FOLDS[fold] + ("_x" if extras else ""))
    for panel, path in zip(WEAK, files):
        with np.load(source / f"{panel}_{condition}.npz") as data:
            x = data["eval_x"]
            shape = x.shape
            p = sum(model.predict(x.reshape(-1, shape[-1])) for model in models) / len(models)
            p = p.reshape(shape[:2])
            n = data["eval_y"].shape[-1]
            probability = np.zeros((len(x), 6, n))
            if condition == "queue_onset": probability[:, -1] = p
            else: probability[:] = p.reshape(-1, 6, n)
            meta = pd.read_csv(source / f"{panel}_{condition}_eval.csv")
            validate_candidate_cohort(meta,_meta(args,fold,panel,condition))
            np.savez(path, probability=probability, truth=data["eval_y"], eligible=data["eval_eligible"],
                     origin=meta.origin.to_numpy(dtype=str), window_id=meta.window_id.to_numpy(dtype=str))
    del models
    gc.collect()


def _read_probability(args, fold, panel, condition, variant):
    path = args.output / "v2" / "probabilities" / fold / f"{panel}_{condition}_{variant}.npz"
    with np.load(path) as d:
        meta = _meta(args, fold, panel, condition)
        # Legacy cached probabilities without identifiers are unsafe to align.
        if "origin" not in d or not np.array_equal(d["origin"], meta.origin.to_numpy(dtype=str)):
            raise ValueError(f"Probability/metadata origin mismatch: {path}")
        if not np.array_equal(d["window_id"], meta.window_id.to_numpy(dtype=str)):
            raise ValueError(f"Probability/metadata window ID mismatch: {path}")
        validate_probability_pack(d["probability"],d["truth"],d["eligible"],meta,d["origin"],d["window_id"])
        with np.load(args.output/"datasets"/FOLDS[fold]/f"{panel}_{condition}.npz") as base:
            if not np.array_equal(d["truth"],base["eval_y"]) or not np.array_equal(d["eligible"],base["eval_eligible"]):
                raise ValueError(f"Probability targets differ from frozen fold cohort: {path}")
        return d["probability"], d["truth"], d["eligible"]


def _meta(args, fold, panel, condition):
    return pd.read_csv(args.output / "datasets" / FOLDS[fold] / f"{panel}_{condition}_eval.csv")


def fit_extent(args,fold):
    source=args.output/"datasets"/(FOLDS[fold]+"_x")
    path=args.output/"v2"/"models"/fold/"extent.txt"
    if path.exists(): return lgb.Booster(model_file=str(path))
    xs,ys=[],[]
    for panel in args.panels:
        with np.load(source/f"{panel}_queue_onset.npz") as d:
            xs.append(d["extent_x"]);ys.append(d["extent_y"])
    x,y=np.concatenate(xs),np.concatenate(ys)
    model=lgb.train(dict(objective="poisson",learning_rate=.03,num_leaves=7,
        min_data_in_leaf=20,lambda_l2=10.,verbosity=-1,seed=42,deterministic=True,
        force_col_wise=True,num_threads=args.threads),lgb.Dataset(x,label=y),num_boost_round=150)
    path.parent.mkdir(parents=True,exist_ok=True);model.save_model(str(path))
    return model


def predict_extent(args,fold):
    model=fit_extent(args,fold)
    out=args.output/"v2"/"probabilities"/fold;out.mkdir(parents=True,exist_ok=True)
    for panel in WEAK:
        with np.load(args.output/"datasets"/(FOLDS[fold]+"_x")/f"{panel}_queue_onset.npz") as d:
            count=model.predict(d["eval_extent_x"],num_threads=args.threads)
        meta=_meta(args,fold,panel,"queue_onset")
        np.savez(out/f"{panel}_extent.npz",count=count,origin=meta.origin.to_numpy(dtype=str))


def extent_decode(probability,count,config):
    pred=decode_calibrated(probability,config,onset=True)
    m0=int(pred.sum()); n=probability.shape[-1]
    alpha=float(config.get("extent_alpha",0.))
    if alpha==0.:return pred
    m=int(np.clip(np.floor((1-alpha)*m0+alpha*count+.5),1,n))
    result=np.zeros_like(pred)
    if config.get("spatial_mode")=="interval":
        from .v2_decode import _transform
        p=_transform(probability[-1],config.get("temperature",1.),config.get("intercept",0.))
        cutoff=np.sort(p)[-m]
        utility=p-float(config.get("gap_penalty",0.))*np.maximum(cutoff-p,0.)
        prefix=np.r_[0,np.cumsum(utility)]
        start=int(np.argmax(prefix[m:]-prefix[:-m]));result[-1,start:start+m]=1
    else:
        order=np.argsort(-probability[-1],kind="stable");result[-1,order[:m]]=1
    return result


def evaluate(args):
    """Calibrate in Sep/Dec; evaluate unchanged configs in Oct-Nov/Jan-Feb."""
    details, configs = [], {}
    for outer, calibration in [("octnov", "sep"), ("janfeb", "dec")]:
        for panel in WEAK:
            for condition in CONDITIONS:
                onset = condition == "queue_onset"
                calmeta, meta = _meta(args, calibration, panel, condition), _meta(args, outer, panel, condition)
                variants = ["baseline", "seed5", "features", "robust"] if onset else ["baseline", "features", "incident", "augmented", "median_v1"]
                available = []
                for variant in variants:
                    path = args.output / "v2" / "probabilities" / outer / f"{panel}_{condition}_{variant}.npz"
                    if path.exists(): available.append(variant)
                for variant in available:
                    pc, yc, ec = _read_probability(args, calibration, panel, condition, variant)
                    p, y, e = _read_probability(args, outer, panel, condition, variant)
                    variants_p = [(variant, pc, p)]
                    if variant != "baseline":
                        bc, _, _ = _read_probability(args, calibration, panel, condition, "baseline")
                        b, _, _ = _read_probability(args, outer, panel, condition, "baseline")
                        for fraction in (.25, .5, .75):
                            variants_p.append((f"{variant}_mix{int(fraction*100)}", bc*(1-fraction)+pc*fraction, b*(1-fraction)+p*fraction))
                    for label, calp, evalp in variants_p:
                        config = fit_cardinality_calibration(calp, yc, ec, onset=onset,
                            calibration_indices=[f"{panel}:{t}" for t in calmeta.origin],
                            evaluation_indices=[f"{panel}:{t}" for t in meta.origin])
                        configs[f"{outer}/{panel}/{condition}/{label}"] = config
                        for calibrated in (False, True):
                            name = label + ("_cal" if calibrated else "")
                            for i, record in enumerate(meta.to_dict("records")):
                                pred = decode_calibrated(evalp[i], config if calibrated else {}, onset=onset)
                                details.append({**record, "fold":outer,"model":name,
                                    "iou":score_window(pred,y[i],e[i]),"predicted_cells":int(pred.sum()),
                                    "fp":int((pred.astype(bool)&~y[i]&e[i]).sum()),
                                    "fn":int((~pred.astype(bool)&y[i]&e[i]).sum())})
                        if onset and label=="features" and (args.output/"v2"/"probabilities"/outer/f"{panel}_extent.npz").exists():
                            with np.load(args.output/"v2"/"probabilities"/calibration/f"{panel}_extent.npz") as d:
                                if not np.array_equal(d["origin"],calmeta.origin.to_numpy(dtype=str)):raise ValueError("Extent calibration alignment")
                                counts_c=d["count"]
                            with np.load(args.output/"v2"/"probabilities"/outer/f"{panel}_extent.npz") as d:
                                if not np.array_equal(d["origin"],meta.origin.to_numpy(dtype=str)):raise ValueError("Extent evaluation alignment")
                                counts=d["count"]
                            best=(-np.inf,0.)
                            for alpha in (0.,.25,.5,1.):
                                cfg={**config,"extent_alpha":alpha}
                                obj=np.mean([score_window(extent_decode(pp,cc,cfg),yy,ee) for pp,cc,yy,ee in zip(calp,counts_c,yc,ec)])-.0025*alpha
                                if obj>best[0]+1e-12:best=(float(obj),alpha)
                            cfg={**config,"extent_alpha":best[1],"extent_calibration_objective":best[0]}
                            configs[f"{outer}/{panel}/{condition}/features_extent"]=cfg
                            for i,record in enumerate(meta.to_dict("records")):
                                pred=extent_decode(evalp[i],counts[i],cfg).astype(bool)
                                details.append({**record,"fold":outer,"model":"features_extent_cal",
                                    "iou":score_window(pred,y[i],e[i]),"predicted_cells":int(pred.sum()),
                                    "fp":int((pred&~y[i]&e[i]).sum()),"fn":int((~pred&y[i]&e[i]).sum())})
    out = args.output / "v2"
    dump_json(out / "calibration.json", configs)
    detail = pd.DataFrame(details)
    detail.to_csv(out / "window_scores.csv", index=False)
    primary = detail[detail.official_like.astype(str).str.lower().eq("true")]
    rows=[]
    for (fold,panel,condition,model),g in primary.groupby(["fold","panel","condition","model"]):
        rows.append(dict(fold=fold,panel=panel,condition=condition,model=model,n=len(g),
            iou=float(g.iou.mean()),shift_iou=float(g.iou.mean() if condition=="queue_onset" else np.average(g.iou,weights=g.shift_weight))))
    scores=pd.DataFrame(rows)
    scores.to_csv(out / "scores.csv",index=False)
    choose_gated(args,scores)
    shutil.copy2(out/"selection.json",out/"selection_formal.json")
    shutil.copy2(out/"gates.csv",out/"gates_formal.csv")
    nonempty=nonempty_scores(args,primary)
    nonempty.to_csv(out/"scores_nonempty.csv",index=False)
    choose_gated(args,scores,nonempty)


def nonempty_scores(args,primary):
    """Diagnostic robustness cohort; primary gate windows remain unchanged."""
    rows=[]
    for (fold,panel,condition),group in primary.groupby(["fold","panel","condition"]):
        with np.load(args.output/"v2/probabilities"/fold/f"{panel}_{condition}_baseline.npz") as pack:
            valid=dict(zip(pack["window_id"],(pack["truth"]&pack["eligible"]).sum(axis=(1,2))>0))
        selected=group[group.window_id.map(valid)]
        for model,g in selected.groupby("model"):
            rows.append(dict(fold=fold,panel=panel,condition=condition,model=model,n=len(g),
                iou=float(g.iou.mean()),shift_iou=float(g.iou.mean() if condition=="queue_onset" else np.average(g.iou,weights=g.shift_weight))))
    return pd.DataFrame(rows)


def assert_complete(args):
    """Do not publish a full comparison after silently skipping a failed fit."""
    missing=[]
    for fold in ("sep","octnov","dec","janfeb"):
        base=args.output/"v2/probabilities"/fold
        for panel in WEAK:
            for condition in CONDITIONS:
                variants=("baseline","seed5","features","robust") if condition=="queue_onset" else (
                    "baseline","features","incident","augmented","median_v1")
                missing.extend(str(base/f"{panel}_{condition}_{v}.npz") for v in variants
                               if not (base/f"{panel}_{condition}_{v}.npz").exists())
            if not (base/f"{panel}_extent.npz").exists():missing.append(str(base/f"{panel}_extent.npz"))
    if missing:raise ValueError("Incomplete v2 experiments; missing: "+", ".join(missing))


def choose_gated(args,scores,nonempty=None):
    """Joint panel improvement in BOTH folds, with no condition regressions.

    An onset already at 1.0 may tie in one fold. It can change only if the
    resulting panel total strictly improves in both folds, and neither changed
    condition reduces ordinary or shift IoU in either fold.
    """
    selections, gates = {}, []
    for panel in WEAK:
        selections[panel] = {}
        candidates={}
        for condition in CONDITIONS:
            s=scores[(scores.panel==panel)&(scores.condition==condition)]
            baseline=s[s.model=="baseline"].set_index("fold")
            if set(baseline.index)!={"janfeb","octnov"} or baseline.index.has_duplicates or not np.isfinite(baseline[["iou","shift_iou"]]).all().all():
                raise ValueError(f"Missing or invalid baseline gates for {panel}/{condition}")
            candidates[condition]=[("baseline",np.zeros(2),np.zeros(2))]
            for model,g in s.groupby("model"):
                if model=="baseline": continue
                g=g.set_index("fold")
                if set(g.index)!={"janfeb","octnov"} or g.index.has_duplicates: continue
                if not np.isfinite(g[["iou","shift_iou"]]).all().all(): continue
                if "n" in g and not np.array_equal(g.loc[baseline.index,"n"],baseline.n):continue
                delta=(g.iou-baseline.iou).reindex(["janfeb","octnov"])
                shift=(g.shift_iou-baseline.shift_iou).reindex(["janfeb","octnov"])
                passes=bool((delta>=-1e-9).all() and (shift>=-1e-9).all()
                            and ((delta>1e-9).any() or (shift>1e-9).any()))
                formal_passes=passes
                nd,ns=pd.Series(np.nan,index=["janfeb","octnov"]),pd.Series(np.nan,index=["janfeb","octnov"])
                guard=True
                if nonempty is not None:
                    sub=nonempty[(nonempty.panel==panel)&(nonempty.condition==condition)]
                    nb=sub[sub.model=="baseline"].set_index("fold")
                    nm=sub[sub.model==model].set_index("fold")
                    guard=bool(set(nb.index)=={"janfeb","octnov"} and set(nm.index)=={"janfeb","octnov"}
                        and not nb.index.has_duplicates and not nm.index.has_duplicates)
                    if guard:
                        nd=(nm.iou-nb.iou).reindex(["janfeb","octnov"])
                        ns=(nm.shift_iou-nb.shift_iou).reindex(["janfeb","octnov"])
                        guard=bool(np.isfinite(nd).all() and np.isfinite(ns).all()
                            and (nd>=-1e-9).all() and (ns>=-1e-9).all()
                            and np.array_equal(nm.loc[nb.index,"n"],nb.n))
                    passes=passes and guard
                gates.append(dict(panel=panel,condition=condition,model=model,
                    janfeb_delta=float(delta.loc["janfeb"]),octnov_delta=float(delta.loc["octnov"]),
                    janfeb_shift_delta=float(shift.loc["janfeb"]),octnov_shift_delta=float(shift.loc["octnov"]),
                    formal_nonregressing=formal_passes,nonempty_guard=guard,
                    janfeb_nonempty_delta=float(nd.loc["janfeb"]),octnov_nonempty_delta=float(nd.loc["octnov"]),
                    janfeb_nonempty_shift_delta=float(ns.loc["janfeb"]),octnov_nonempty_shift_delta=float(ns.loc["octnov"]),
                    nonregressing=passes))
                if passes:candidates[condition].append((model,delta.to_numpy(),shift.to_numpy()))
        eligible=[]
        for on in candidates["queue_onset"]:
            for og in candidates["queue_ongoing"]:
                delta=(on[1]+og[1])/2
                shift=(on[1]+og[2])/2
                if (delta>1e-9).all() and (shift>=-1e-9).all():
                    eligible.append((float(delta.min()),float(delta.mean()),
                        -int(on[0]!="baseline")-int(og[0]!="baseline"),on[0],og[0]))
        if eligible:
            best=max(eligible)
            selections[panel]=dict(queue_onset=best[-2],queue_ongoing=best[-1])
        else:selections[panel]=dict.fromkeys(CONDITIONS,"baseline")
    for row in gates:
        row["accepted"]=row["model"]==selections[row["panel"]][row["condition"]]
    dump_json(args.output / "v2" / "selection.json",selections)
    pd.DataFrame(gates).to_csv(args.output / "v2" / "gates.csv",index=False)
    log(("V2 guarded selection: " if nonempty is not None else "V2 formal selection: ")+json.dumps(selections))


def main():
    started=time.perf_counter()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=PROJECT/"data/kaggle_public")
    p.add_argument("--output",type=Path,default=PROJECT/"outputs/t2")
    p.add_argument("--stage",choices=["prepare-base","baseline","seeds","features","ongoing","median-ongoing","extent","evaluate","prepare-extra","predict","report","all"],default="all")
    p.add_argument("--fold",action="append",choices=list(FOLDS))
    p.add_argument("--feature-variant",choices=["all","features","robust"],default="all")
    p.add_argument("--rounds",type=int,default=350);p.add_argument("--threads",type=int,default=2)
    args=p.parse_args(); args.root=resolve_root(args.root);args.panels=panel_names(args.root)
    frozen=args.output/"queue_submission_v1.csv"
    if not frozen.exists():shutil.copy2(args.output/"queue_submission.csv",frozen)
    if frozen.read_bytes()!=(args.output/"queue_submission.csv").read_bytes():
        raise ValueError("Frozen v1 and current queue_submission.csv differ; refusing to overwrite either")
    folds=args.fold or ["sep","octnov","dec","janfeb"]
    if args.stage in {"prepare-base","all"}:
        for fold in folds:prepare_base(args,fold)
    if args.stage in {"prepare-extra","all"}:
        from .v2_data import prepare_extra
        for fold in folds: prepare_extra(args,fold)
    for fold in folds:
        if fold=="final": continue
        if args.stage in {"baseline","all"}:
            for condition in CONDITIONS: predict_fold(args,fold,condition,"baseline")
        if args.stage in {"seeds","all"}: predict_fold(args,fold,"queue_onset","seed5")
        if args.stage in {"features","all"}:
            if args.feature_variant!="robust":
                for condition in CONDITIONS: predict_fold(args,fold,condition,"features")
            if args.feature_variant!="features":predict_fold(args,fold,"queue_onset","robust")
        if args.stage in {"ongoing","all"}:
            for variant in ("incident","augmented"):predict_fold(args,fold,"queue_ongoing",variant)
        if args.stage in {"median-ongoing","all"}:predict_fold(args,fold,"queue_ongoing","median_v1")
        if args.stage in {"extent","all"}:predict_extent(args,fold)
    if args.stage in {"evaluate","all"}:
        assert_complete(args)
        evaluate(args)
    if args.stage in {"predict","all"}:
        from .v2_predict import predict_v2
        predict_v2(args)
    if args.stage in {"report","all"}:
        from .v2_predict import write_report
        write_report(args)
    source=Path(__file__).parent
    metadata=dict(stage=args.stage,folds=folds,seconds=time.perf_counter()-started,
        seeds=list(SEEDS),threads=args.threads,rounds=args.rounds,python=platform.python_version(),
        versions={name:version(name) for name in ("numpy","pandas","pyarrow","lightgbm","scipy")},
        source_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(source.glob("*.py"))},
        frozen_v1_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),submitted=False)
    dump_json(args.output/"v2"/f"run_{args.stage}.json",metadata)


if __name__=="__main__":main()
