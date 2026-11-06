"""Location-free V4 onset models on all train months.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np

from .io import dump_json,panel_names
from .run import PROJECT,log
from .train import Classifier, fit_classifier
from .v2_experiment import load_model
from .v3_robust import portable_mask
from ..io import RELEASE as RELEASE_ROOT   # TFB_RELEASE (run.py --data), else data/kaggle_public

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
            if len(names)!=392:raise ValueError("V4 requires the 288+104 column schema")
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
    for seed in SEEDS:
        fit_model(args,x,y,w,names,args.output/"v4/models"/fold/"normal"/f"seed{seed}.txt",seed=seed)
    del x,y,w;gc.collect()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=RELEASE_ROOT)
    p.add_argument("--output",type=Path,default=PROJECT/"outputs/t2")
    p.add_argument("--fold",choices=["final"],default="final")
    p.add_argument("--threads",type=int,default=2);p.add_argument("--rounds",type=int,default=350)
    args=p.parse_args();args.panels=panel_names(args.root)
    fit_normal(args,args.fold)


if __name__=="__main__":main()
