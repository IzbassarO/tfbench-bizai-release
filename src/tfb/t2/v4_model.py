"""Location-free onset models and whole-window leave-bottleneck-out evaluation."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .io import dump_json,panel_names
from .run import PROJECT,log
from .train import Classifier,decode_probabilities,fit_classifier,score_window
from .v2_experiment import load_model
from .v3_robust import portable_mask

SEEDS=(42,137,271)


def onset_feature_mask(names,policy="discharge"):
    mask=portable_mask(names)
    if policy=="all":return mask
    if policy!="discharge":raise ValueError("Unknown onset feature policy")
    for i,name in enumerate(names):
        if name.startswith("v4_"):
            # Daily free-flow peaks are demand-limited; their denominators
            # can encode scenario activation. Use supported discharge/FD
            # capacity and denominator-independent speed/density evidence.
            mask[i]&=(any(k in name for k in ("discharge","density","speed","observed_fraction","observation_age"))
                      or name=="v4_flow_nominal_ratio_now")
            if "_bottleneck_" in name and "discharge" not in name:mask[i]=False
    return mask


def load_pool(args,fold):
    xs,ys,ws,panels,heads=[],[],[],[],[]
    source=args.output/"v4/data"/fold
    for pi,panel in enumerate(args.panels):
        with np.load(source/f"{panel}.npz") as d:
            xs.append(d["train_x"]);ys.append(d["train_y"]);ws.append(d["train_weight"])
            names=d["feature_names"].tolist()
            if len(names)!=392:raise ValueError("V4 requires the frozen 288+104 column schema")
            heads.append(d["train_head_index"]);panels.append(np.full(len(ys[-1]),pi,np.int16))
    return np.concatenate(xs),np.concatenate(ys),np.concatenate(ws),np.concatenate(panels),np.concatenate(heads),names


def fit_model(args,x,y,w,names,path,*,seed=42,keep=None):
    mask=onset_feature_mask(names,getattr(args,"feature_policy","discharge"))
    # Physical-capacity support counts and absolute scalars are metadata only.
    if any(n.startswith("location_") for n in np.asarray(names)[mask]):raise ValueError("Location identity reached V4")
    if keep is None:keep=np.ones(len(y),bool)
    selected=x[keep][:,mask]
    signature=dict(seed=seed,rounds=args.rounds,threads=args.threads,
        rows=int(keep.sum()),excluded_rows=int((~keep).sum()),names=list(np.asarray(names)[mask]),
        dropped=list(np.asarray(names)[~mask]),y_sha256=hashlib.sha256(y[keep].tobytes()).hexdigest(),
        weights_sha256=hashlib.sha256(w[keep].tobytes()).hexdigest(),
        features_sha256=hashlib.sha256(memoryview(np.ascontiguousarray(selected)).cast("B")).hexdigest(),
        deterministic=True,feature_policy=getattr(args,"feature_policy","discharge"),
        source="raw onset labels; no location priors or earlier-day split priors")
    record=path.with_suffix(".training.json")
    if path.exists():
        if not record.exists() or json.loads(record.read_text())!=signature:
            raise ValueError(f"Stale V4 training cache: {path}")
        del selected;gc.collect()
        model=load_model(path,args.threads)
        if model.booster.feature_name()!=signature["names"]:raise ValueError("Cached V4 feature schema mismatch")
        return model
    log(f"v4 fit {path.parent.name}/{path.stem}: {keep.sum():,} rows, {mask.sum()} portable columns")
    m=fit_classifier(selected,y[keep],w[keep],list(np.asarray(names)[mask]),seed=seed,
        rounds=args.rounds,threads=args.threads)
    model=Classifier(m.booster,mask,m.constant,args.threads)
    path.parent.mkdir(parents=True,exist_ok=True);model.save(path)
    dump_json(record,signature)
    del selected;gc.collect()
    return model


def fit_normal(args,fold):
    x,y,w,_,_,names=load_pool(args,fold)
    models=[]
    for seed in SEEDS:
        group="normal" if args.feature_policy=="discharge" else "all_normal"
        path=args.output/"v4/models"/fold/group/f"seed{seed}.txt"
        models.append(fit_model(args,x,y,w,names,path,seed=seed))
    del x,y,w;gc.collect()
    if fold=="final":return
    dest=args.output/"v4/probabilities"/fold;dest.mkdir(parents=True,exist_ok=True)
    for panel in args.panels:
        with np.load(args.output/"v4/data"/fold/f"{panel}.npz") as d:
            ex=d["eval_x"];flat=ex.reshape(-1,ex.shape[-1])
            ps=np.stack([m.predict(flat).reshape(ex.shape[:2]) for m in models])
            meta=pd.read_csv(args.output/"v4/data"/fold/f"{panel}_eval.csv")
            np.savez(dest/f"{panel}_{group}.npz",probability=ps.mean(0),seed_probability=ps,
                seeds=np.array(SEEDS),truth=d["eval_y"],eligible=d["eval_eligible"],
                window_id=meta.window_id.to_numpy(dtype=str),origin=meta.origin.to_numpy(dtype=str))
    log(f"v4 normal predictions complete {fold}")


def fit_lobo(args,fold):
    x,y,w,train_panels,train_heads,names=load_pool(args,fold)
    source=args.output/"v4/data"/fold
    evaluations={};jobs=[]
    for pi,panel in enumerate(args.panels):
        meta=pd.read_csv(source/f"{panel}_eval.csv")
        with np.load(source/f"{panel}.npz") as d:evaluations[panel]=d["eval_x"]
        if args.primary_only:meta=meta[meta.official_like.astype(str).str.lower().eq("true")|meta.nonrecurrent_onset.astype(bool)]
        for head in sorted(set(meta.head_index.astype(int))-{ -1 }):jobs.append((pi,panel,head))
    tree="lobo" if args.feature_policy=="discharge" else "lobo_all"
    dest=args.output/"v4"/tree/fold/f"radius{args.radius}";dest.mkdir(parents=True,exist_ok=True)
    # Nearby evaluated heads can exclude the identical set of training
    # windows. Fit that row set once; retain separate packs for every head.
    grouped={}
    for spec in jobs:
        pi,panel,head=spec
        excluded_heads=tuple(np.unique(train_heads[(train_panels==pi)&(np.abs(train_heads-head)<=args.radius)&(train_heads>=0)]))
        grouped.setdefault((pi,excluded_heads),[]).append(spec)
    def job(group):
        pi,panel,head=group[0]
        path=dest/f"{panel}_head{head}.txt";pack=path.with_suffix(".npz")
        excluded=(train_panels==pi)&(np.abs(train_heads-head)<=args.radius)&(train_heads>=0)
        normal_group="normal" if args.feature_policy=="discharge" else "all_normal"
        normal=args.output/"v4/models"/fold/normal_group/"seed42.txt"
        fit_path=normal if not excluded.any() and normal.with_suffix(".training.json").exists() else path
        model=fit_model(args,x,y,w,names,fit_path,seed=42,keep=~excluded)
        for _,panel,evaluated_head in group:
            meta=pd.read_csv(source/f"{panel}_eval.csv")
            take=np.flatnonzero(meta.head_index.to_numpy(int)==evaluated_head)
            ex=evaluations[panel][take]
            p=model.predict(ex.reshape(-1,ex.shape[-1])).reshape(ex.shape[:2])
            pack=dest/f"{panel}_head{evaluated_head}.npz"
            np.savez(pack,probability=p,window_id=meta.iloc[take].window_id.to_numpy(dtype=str),
                origin=meta.iloc[take].origin.to_numpy(dtype=str),head_index=evaluated_head,
                excluded_training_rows=int(excluded.sum()),training_rows=int((~excluded).sum()),
                model_source=str(fit_path))
            log(f"v4 LOBO {fold}/{panel}/head{evaluated_head}: {len(take)} eval, excluded {excluded.sum():,} cells")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:list(executor.map(job,list(grouped.values())))
    dump_json(dest/"protocol.json",dict(fold=fold,seed=42,radius=args.radius,groups=len(jobs),
        distinct_exclusion_groups=len(grouped),identical_row_sets_share_one_fitted_model=True,
        exclusion="entire training window if same panel and its raw T+30 downstream head is within radius of evaluation head",
        future_head_used_only_for_holdout_partition=True,features_use_no_head_target=True,
        empirical_capacity="fitted on fold training observations; retained as permitted physical calibration, not location frequency"))


def evaluate(args,fold):
    rows=[];cohort_sizes={}
    for panel in args.panels:
        source=args.output/"v4/data"/fold
        meta=pd.read_csv(source/f"{panel}_eval.csv")
        cohort_sizes[(panel,"dense")]=len(meta)
        cohort_sizes[(panel,"primary")]=int(meta.official_like.astype(str).str.lower().eq("true").sum())
        with np.load(args.output/"v4/probabilities"/fold/f"{panel}_normal.npz") as d:
            p,seedp,y,e=d["probability"],d["seed_probability"],d["truth"],d["eligible"]
            assert np.array_equal(d["window_id"],meta.window_id.to_numpy(dtype=str))
        with np.load(args.output/"v3/eval"/fold/f"{panel}_queue_onset.npz") as d:
            assert np.array_equal(d["window_id"],meta.window_id.to_numpy(dtype=str))
            baseline=d["prediction_v2"]
        ablation_path=args.output/"v4/probabilities"/fold/f"{panel}_all_normal.npz"
        all_p=None
        if ablation_path.exists():
            with np.load(ablation_path) as d:
                assert np.array_equal(d["window_id"],meta.window_id.to_numpy(dtype=str))
                all_p=d["probability"]
        lobo={}
        for path in (args.output/"v4/lobo"/fold/f"radius{args.radius}").glob(f"{panel}_head*.npz"):
            with np.load(path) as d:
                for wid,pp in zip(d["window_id"],d["probability"]):lobo[str(wid)]=pp
        for i,r in enumerate(meta.to_dict("records")):
            choices={"v2_normal":baseline[i]}
            candidates=[("v4_mean3_normal",p[i]),("v4_seed42_normal",seedp[0,i]),("v4_seed42_lobo",lobo.get(r["window_id"]))]
            if all_p is not None:candidates.append(("v4_all_mean3_normal",all_p[i]))
            for name,pp in candidates:
                if pp is None:continue
                allp=np.zeros(y[i].shape,float);allp[-1]=pp
                choices[name]=decode_probabilities(allp,onset=True)
            for model,pred in choices.items():
                rows.append({**r,"fold":fold,"lobo_radius":args.radius,"model":model,"iou":score_window(pred,y[i],e[i]),
                    "true_cells":int((y[i]&e[i]).sum()),"predicted_cells":int(pred.sum()),
                    "fp":int((pred.astype(bool)&~y[i]&e[i]).sum()),"fn":int((~pred.astype(bool)&y[i]&e[i]).sum())})
    detail=pd.DataFrame(rows);dest=args.output/"v4/evaluation";dest.mkdir(parents=True,exist_ok=True)
    tag=fold if args.radius==0 else f"{fold}_radius{args.radius}"
    detail.to_csv(dest/f"{tag}_windows.csv",index=False)
    summaries=[]
    for subset,mask in (("primary",detail.official_like.astype(str).str.lower().eq("true")),("dense",np.ones(len(detail),bool))):
        known=detail.head_index.ge(0)
        supported=~detail.head_prior_unsupported.astype(bool)
        rare=detail.nonrecurrent_onset.astype(bool)
        for label,partition in (("all",mask),("rare",mask&known&supported&rare),
                ("recurrent",mask&known&supported&~rare),("unknown_head",mask&~known),
                ("unsupported_tod",mask&known&~supported)):
            for (model,panel),g in detail[partition].groupby(["model","panel"]):
                summaries.append(dict(fold=fold,subset=subset,stratum=label,model=model,panel=panel,n=len(g),onset_iou=g.iou.mean()))
    summary=pd.DataFrame(summaries)
    # Panel means retain equal family/direction weight; also expose sparse
    # rare-stratum support instead of treating an absent panel as score zero.
    overall=[]
    for keys,g in summary.groupby(["fold","subset","stratum","model"]):
        overall.append(dict(zip(["fold","subset","stratum","model"],keys)),)
        overall[-1].update(panel="OVERALL_SUPPORTED",n=int(g.n.sum()),supported_panels=len(g),onset_iou=g.onset_iou.mean())
        fold_name,subset,stratum,model=keys
        masses=np.array([row.n/cohort_sizes[(row.panel,subset)]/len(args.panels) for row in g.itertuples()])
        weighted=dict(zip(["fold","subset","stratum","model"],keys))
        weighted.update(panel="OVERALL_WEIGHTED",n=int(g.n.sum()),supported_panels=len(g),
            support_mass=float(masses.sum()),onset_iou=float(np.average(g.onset_iou,weights=masses)))
        overall.append(weighted)
    pd.concat([summary,pd.DataFrame(overall)],ignore_index=True).to_csv(dest/f"{tag}_scores.csv",index=False)
    log(f"v4 scores {fold}\n"+pd.DataFrame(overall).to_string(index=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=PROJECT/"data/kaggle_public")
    p.add_argument("--output",type=Path,default=PROJECT/"outputs/t2")
    p.add_argument("--fold",choices=["janfeb","octnov","final"],default="janfeb")
    p.add_argument("--stage",choices=["normal","lobo","evaluate","all"],default="all")
    p.add_argument("--threads",type=int,default=2);p.add_argument("--rounds",type=int,default=350)
    p.add_argument("--workers",type=int,default=3);p.add_argument("--radius",type=int,default=0)
    p.add_argument("--primary-only",action="store_true")
    p.add_argument("--feature-policy",choices=["discharge","all"],default="discharge")
    args=p.parse_args();args.panels=panel_names(args.root)
    if args.stage in {"normal","all"}:fit_normal(args,args.fold)
    if args.fold!="final":
        if args.stage in {"lobo","all"}:fit_lobo(args,args.fold)
        if args.stage in {"evaluate","all"}:evaluate(args,args.fold)


if __name__=="__main__":main()
