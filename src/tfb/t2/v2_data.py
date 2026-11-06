"""Append causal features and training-only counterfactual queue translations.

Base feature values/row sampling are those of the v1 dataset. Every
augmentation is built from a training window, never a held-out label, and its
target-eligibility mask is translated without wrapping corridor boundaries.
"""
from __future__ import annotations

import gc
import json
import shutil

import numpy as np
import pandas as pd

from .features import build_features
from .io import PanelArrays, current_observations, dump_json, read_network, read_panel_arrays
from .labels import make_labels
from .run import CONDITIONS, enrich_features, log
from .v2_experiment import FOLDS, WEAK
from .v2_features import build_v2_features, shift_history_and_targets

CUTOFF = {"sep":"2030-09-01", "octnov":"2030-10-01", "dec":"2030-12-01",
          "janfeb":"2031-01-01", "final":"2031-03-01"}


def legal_history(arrays, network, origin, current, row):
    start=origin-12
    v,q,o=(a[start:origin+1].copy() for a in (arrays.speed,arrays.flow,arrays.occupancy))
    for values,name in [(v,"speed_kmh"),(q,"flow_vph"),(o,"occupancy")]: values[-1]=current[name][row]
    return PanelArrays(arrays.timestamps[start:origin+1],v,q,o,
                       arrays.eligible[start:origin+1],arrays.ramp_on[start:origin+1])


def extra_features(history,network):
    return build_v2_features(history.speed,history.flow,history.occupancy,history.timestamps,
        origin=len(history.timestamps)-1,ramp_on=history.ramp_on,**network.feature_kwargs())


def synthetic_features(history,network,truth,eligible,day_context,day_names,prior,panel_id,shift):
    shifted=shift_history_and_targets(history.speed,history.flow,history.occupancy,truth,
        vf=network.vf,capacity=network.capacity,kcrit=network.kcrit,shift=shift,
        target_eligible=eligible)
    n=len(network.link_ids)
    context=np.full_like(day_context,np.nan)
    valid=shifted.destination_valid
    context[valid]=day_context[shifted.source_index[valid]]
    # No synthetic ramp observation is invented. The normalized dynamic state
    # moves; destination static FD/geometry and historical priors stay fixed.
    local=PanelArrays(history.timestamps,shifted.speed,shifted.flow,shifted.occupancy,
        np.broadcast_to(valid,(13,n)),np.full_like(history.ramp_on,np.nan))
    x,names=build_features(local.speed,local.flow,local.occupancy,origin=12,
        timestamps=local.timestamps,ramp_on=local.ramp_on,location_prior=prior,
        panel_id=panel_id,**network.feature_kwargs())
    # Synthetic queues are intentionally non-recurrent. Their conditional
    # onset-event prior is unknown rather than copied from the source location.
    x,names=enrich_features(x,names,local,network,onset_prior=np.zeros((2,288,n)),
        day_context=context,day_names=day_names)
    extra,extra_names=extra_features(local,network)
    return np.column_stack([x,extra]),names+extra_names,shifted.target_labels,shifted.target_eligible


def extent_summary(x,names,origin):
    """Causal window features for a small onset-cardinality regression model."""
    # Aggregate ALL links, before any target-quality mask or cell sampling.
    # Otherwise future eligibility would silently enter extent features.
    selected=names
    epoch=np.datetime64(origin,"m").astype(int)
    values=[np.sin(2*np.pi*(epoch%1440)/1440),np.cos(2*np.pi*(epoch%1440)/1440),
            (epoch//1440+3)%7]
    for name in selected:
        a=x[:,names.index(name)].astype(float)
        a=a[np.isfinite(a)]
        values.extend([float(np.mean(a)),float(np.min(a)),float(np.max(a))] if len(a) else [np.nan]*3)
    # Different corridor sizes should not masquerade as demand differences.
    return np.array([len(x),*values],np.float32)


def prepare_extra(args,fold):
    source=args.output/"datasets"/FOLDS[fold]
    dest=args.output/"datasets"/(FOLDS[fold]+"_x")
    dest.mkdir(parents=True,exist_ok=True)
    cutoff=np.datetime64(CUTOFF[fold],"m")
    for panel_id,panel in enumerate(args.panels):
        base_config=json.loads((source/f"{panel}_build.json").read_text())
        if np.datetime64(base_config["cutoff"],"m")!=cutoff:
            raise ValueError(f"Base dataset cutoff mismatch for {fold}/{panel}")
        marker=dest/f"{panel}_v2_build.json"
        if marker.exists() and all((dest/f"{panel}_{c}.npz").exists() for c in CONDITIONS):
            log(f"v2 extra {fold}/{panel}: cached"); continue
        network=read_network(args.root,panel)
        arrays=read_panel_arrays(args.root,network,"train",args.output/"cache")
        raw=make_labels(arrays.speed,network.vf)
        smooth=make_labels(arrays.speed,network.vf,method="median3")
        boundary=int(np.searchsorted(arrays.timestamps,cutoff))
        if boundary<len(raw): smooth[boundary-1]=raw[boundary-1]
        windows=pd.read_csv(source/f"{panel}_windows.csv")
        origins=windows.origin_idx.to_numpy(int)
        current=current_observations(args.root,network,arrays.timestamps[origins],
            include_context=True,released_history=arrays)
        prior=np.load(source/f"{panel}_prior.npy")
        n=len(network.link_ids)
        for condition in CONDITIONS:
            with np.load(source/f"{panel}_{condition}.npz") as d:
                packed={k:d[k] for k in d.files}
            old_names=packed["feature_names"].tolist()
            additions,eval_additions,ay,ax,aw=[],[],[],[],[]
            extent_x,extent_y,eval_extent_x=[],[],[]
            rng=np.random.default_rng(42+panel_id*101+int(condition.endswith("ongoing")))
            aug_rng=np.random.default_rng(9841+panel_id)
            offset=0
            for row in windows[windows.condition.eq(condition)].itertuples():
                origin=int(row.origin_idx); row_idx=int(row.Index)
                history=legal_history(arrays,network,origin,current,row_idx)
                extra,extra_names=extra_features(history,network)
                truth=raw[origin+1:origin+7]; median=smooth[origin+1:origin+7]
                valid=arrays.eligible[origin+1:origin+7]
                onset=condition=="queue_onset"
                y=truth[-1] if onset else truth.reshape(-1)
                ym=median[-1] if onset else median.reshape(-1)
                mask=valid[-1] if onset else valid.reshape(-1)
                model_extra=extra[-n:] if onset else extra
                is_train=arrays.timestamps[origin+6]<cutoff
                if is_train:
                    sample_p=1. if onset else .25
                    keep=mask&(y|ym|(rng.random(len(y))<sample_p))
                    count=int(keep.sum())
                    if not np.array_equal(packed["train_y"][offset:offset+count],y[keep]):
                        raise AssertionError(f"v1 training row alignment failed {fold}/{panel}/{condition}")
                    if not np.array_equal(packed["train_median"][offset:offset+count],ym[keep]):
                        raise AssertionError("Median target alignment failed")
                    additions.append(model_extra[keep]); offset+=count
                    if onset:
                        extent_x.append(extent_summary(model_extra,extra_names,arrays.timestamps[origin]))
                        extent_y.append(float((truth[-1]&valid[-1]).sum())/max(float(valid[-1].mean()),.1))
                    if panel in WEAK:
                        # One modest translated history per source window;
                        # both upstream and downstream boundary shifts occur.
                        shift=int(aug_rng.choice([-5,-3,-2,2,3,5]))
                        sx,snames,sy,sv=synthetic_features(history,network,median if onset else truth,valid,
                            current["day_context"][row_idx],current["day_context_names"],prior,panel_id,shift)
                        if onset: sx,sy,sv=sx[-n:],sy[-1],sv[-1]
                        sy=sy.reshape(-1);sv=sv.reshape(-1)
                        take=sv&(sy|(aug_rng.random(len(sy))<.25))
                        ax.append(sx[take]);ay.append(sy[take]);aw.append(np.where(sy[take],1.,4.).astype(np.float32))
                else:
                    eval_additions.append(model_extra)
                    if onset:
                        eval_extent_x.append(extent_summary(model_extra,extra_names,arrays.timestamps[origin]))
            if offset!=len(packed["train_x"]): raise AssertionError("Training sampler failed exact coverage")
            packed["train_x"]=np.column_stack([packed["train_x"],np.concatenate(additions)])
            if len(eval_additions): packed["eval_x"]=np.concatenate([packed["eval_x"],np.stack(eval_additions)],axis=-1)
            else: packed["eval_x"]=np.empty((*packed["eval_x"].shape[:2],len(old_names)+len(extra_names)),np.float32)
            packed["feature_names"]=np.array(old_names+extra_names)
            packed.update(augment_x=np.concatenate(ax) if ax else np.empty((0,len(old_names)+len(extra_names)),np.float32),
                    augment_y=np.concatenate(ay) if ay else np.empty(0,bool),augment_weight=np.concatenate(aw) if aw else np.empty(0,np.float32))
            if onset:
                packed.update(extent_x=np.stack(extent_x),extent_y=np.array(extent_y,np.float32),
                    eval_extent_x=np.stack(eval_extent_x) if eval_extent_x else np.empty((0,len(extent_x[0])),np.float32))
            np.savez(dest/f"{panel}_{condition}.npz",**packed)
            shutil.copy2(source/f"{panel}_{condition}_eval.csv",dest/f"{panel}_{condition}_eval.csv")
            log(f"v2 extra {fold}/{panel}/{condition}: {len(packed['train_x']):,} original + {sum(len(a) for a in ay):,} synthetic")
            del packed,additions,eval_additions,ax,ay,aw
            gc.collect()
        dump_json(marker,dict(schema=1,cutoff=str(cutoff),base_dataset=FOLDS[fold],
            augmentation="FD-normalized nonwrapping shifts -5,-3,-2,2,3,5, training-only, weight .35",
            onset_early_horizons=False))
        del arrays,current,raw,smooth
        gc.collect()
