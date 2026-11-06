"""Apply the gated Task 2 v2 choice: fit the selected members on all train months and write queue_submission_v2.csv.
"""
from __future__ import annotations

import gc
import hashlib
import json

import numpy as np
import pandas as pd

from .context import build_day_context
from .features import build_features
from .io import dump_json, history_arrays, read_network, read_panel_arrays, read_window_assets
from .run import enrich_features, log, verify_submissions
from .v2_data import extra_features, extent_summary, prepare_extra
from .v2_decode import decode_calibrated
from .v2_experiment import WEAK, extent_decode, fit_extent, fit_group, prepare_base


def member_parts(name):
    calibrated=name.endswith("_cal")
    label=name[:-4] if calibrated else name
    if label=="features_extent":return "features",1.,calibrated,True,label
    if "_mix" in label:
        variant,amount=label.rsplit("_mix",1)
        return variant,int(amount)/100.,calibrated,False,label
    return label,1.,calibrated,False,label


def predict_v2(args):
    selected=json.loads((args.output/"v2/selection.json").read_text())
    calibration=json.loads((args.output/"v2/calibration.json").read_text())
    frozen=args.output/"queue_submission_v1.csv"
    original_hash=hashlib.sha256((args.output/"queue_submission.csv").read_bytes()).hexdigest()
    if frozen.read_bytes()!=(args.output/"queue_submission.csv").read_bytes():raise ValueError("v1 changed")
    frame=pd.read_csv(frozen)
    needs_extra=any(member_parts(member)[0] in {"features","incident","augmented","robust"}
        for members in selected.values() for member in members.values())
    prepare_base(args,"final")
    if needs_extra:
        prepare_extra(args,"final")
    model_cache={}
    extent_model=None
    changes=[]
    for panel in WEAK:
        panel_id=args.panels.index(panel)
        network=read_network(args.root,panel);n=len(network.link_ids)
        prior=np.load(args.output/"datasets/final_enhanced_release"/f"{panel}_prior.npy")
        onset_prior=np.load(args.output/"datasets/final_enhanced_release"/f"{panel}_onset_prior.npy")
        link_index=dict(zip(network.link_ids,range(n)))
        for split in ("validation","private"):
            index,history,template=read_window_assets(args.root,panel,split)
            days=set(index.date.astype(str))
            days|={str(np.datetime64(day)-np.timedelta64(1,"D")) for day in days}
            current=read_panel_arrays(args.root,network,split,args.output/"cache",masked=True,dates=days)
            for window in index.itertuples():
                member=selected[panel][window.condition]
                if member=="baseline":continue
                variant,fraction,calibrated,use_extent,label=member_parts(member)
                onset=window.condition=="queue_onset"
                arrays=history_arrays(history[history.window_id.eq(window.window_id)],window.forecast_origin,network,current)
                x,names=build_features(arrays.speed,arrays.flow,arrays.occupancy,origin=12,
                    timestamps=arrays.timestamps,ramp_on=arrays.ramp_on,location_prior=prior,
                    panel_id=panel_id,**network.feature_kwargs())
                origin_idx=int((arrays.timestamps[-1]-current.timestamps[0])/np.timedelta64(5,"m"))
                day_context,day_names=build_day_context(current.speed,current.flow,current.timestamps,
                    origin=origin_idx,vf=network.vf)
                x,names=enrich_features(x,names,arrays,network,onset_prior=onset_prior,
                    day_context=day_context,day_names=day_names)
                extra,extra_names=extra_features(arrays,network)
                model_x=x[-n:] if onset else x
                extended=np.column_stack([x,extra]); extended=extended[-n:] if onset else extended
                def probability_for(v):
                    key=(window.condition,v)
                    if key not in model_cache:model_cache[key]=fit_group(args,"final",window.condition,v)
                    models,uses_extra=model_cache[key]
                    mx=extended if uses_extra else model_x
                    result=sum(m.predict(mx) for m in models)/len(models)
                    p=np.zeros((6,n))
                    if onset:p[-1]=result
                    else:p[:]=result.reshape(6,n)
                    return p
                p=probability_for(variant)
                if fraction<1:p=fraction*p+(1-fraction)*probability_for("baseline")
                config=calibration[f"janfeb/{panel}/{window.condition}/{label}"] if calibrated else {}
                if use_extent:
                    if extent_model is None:extent_model=fit_extent(args,"final")
                    summary=extent_summary(extra[-n:],extra_names,arrays.timestamps[-1])
                    count=float(extent_model.predict(summary[None,:],num_threads=args.threads)[0])
                    decoded=extent_decode(p,count,config)
                else:decoded=decode_calibrated(p,config,onset=onset)
                rows=frame.window_id.eq(window.window_id)
                out=frame.loc[rows]
                times=pd.to_datetime(out.timestamp,utc=True)
                hi=((times-pd.Timestamp(window.forecast_origin)).dt.total_seconds()/300).to_numpy(int)-1
                li=out.link_id.map(link_index).to_numpy(int)
                if not np.all((hi>=0)&(hi<6)):raise ValueError("Unexpected horizon")
                old=out.queue_pred.to_numpy().copy();new=decoded[hi,li]
                frame.loc[rows,"queue_pred"]=new
                changes.append(dict(panel=panel,split=split,window_id=window.window_id,condition=window.condition,
                    model=member,changed_cells=int((old!=new).sum()),v1_positive=int(old.sum()),v2_positive=int(new.sum())))
            del current
            gc.collect()
            log(f"v2 predicted {panel}/{split}")
    destination=args.output/"queue_submission_v2.csv"
    temporary=destination.with_suffix(".tmp.csv")
    frame.to_csv(temporary,index=False)
    temporary.replace(destination)
    assert hashlib.sha256((args.output/"queue_submission.csv").read_bytes()).hexdigest()==original_hash
    assert frozen.read_bytes()==(args.output/"queue_submission.csv").read_bytes()
    pd.DataFrame(changes,columns=["panel","split","window_id","condition","model","changed_cells","v1_positive","v2_positive"]).to_csv(args.output/"v2/prediction_changes.csv",index=False)
    # Keep the v1 verification report as it is.
    verification=args.output/"submission_verification.json"
    old_verification=verification.read_bytes() if verification.exists() else None
    try:
        verify_submissions(args,filenames=["queue_submission_v1.csv","queue_submission_v2.csv"])
        (args.output/"v2/submission_verification.json").write_bytes(verification.read_bytes())
    finally:
        if old_verification is not None:verification.write_bytes(old_verification)
    dump_json(args.output/"v2/prediction_manifest.json",dict(v1_sha256=original_hash,
        v2_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),rows=len(frame),
        changed_cells=sum(row["changed_cells"] for row in changes),selected=selected,
        calibration_policy="Dec 2030 out-of-sample predictions; fixed for deployment"))


