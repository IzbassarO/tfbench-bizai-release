"""Apply gated Task 2 v2 changes without touching the frozen v1 submission."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .context import build_day_context
from .features import build_features
from .io import dump_json, history_arrays, read_network, read_panel_arrays, read_window_assets
from .run import PROJECT, enrich_features, log, verify_submissions
from .v2_data import extra_features, extent_summary, prepare_extra
from .v2_decode import decode_calibrated
from .v2_experiment import (CONDITIONS, WEAK, extent_decode, fit_extent, fit_group,
                            prepare_base)


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
    # Keep the original v1 verification report frozen too.
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
        calibration_policy="Dec 2030 out-of-sample predictions; fixed for deployment",
        submitted=False))


def markdown(frame,columns=None):
    if columns:frame=frame[columns]
    def cell(value):
        if isinstance(value,(float,np.floating)):return f"{value:.4f}"
        return str(value)
    return "| "+" | ".join(frame.columns)+" |\n|"+"---|"*len(frame.columns)+"\n"+"\n".join("| "+" | ".join(cell(v) for v in row)+" |" for row in frame.itertuples(index=False,name=None))


def write_report(args):
    out=args.output/"v2"
    selection=json.loads((out/"selection.json").read_text())
    scores=pd.read_csv(out/"scores.csv")
    gates=pd.read_csv(out/"gates.csv")
    manifest=json.loads((out/"prediction_manifest.json").read_text())
    rows=[]
    for fold in ("janfeb","octnov"):
        for panel in WEAK:
            values={}
            for version in ("v1","v2"):
                for condition,label in zip(CONDITIONS,("onset","ongoing")):
                    model="baseline" if version=="v1" else selection[panel][condition]
                    row=scores[(scores.fold==fold)&(scores.panel==panel)&(scores.condition==condition)&(scores.model==model)].iloc[0]
                    values[f"{version}_{label}"]=row.iou
                    if label=="ongoing":values[f"{version}_shift_ongoing"]=row.shift_iou
                values[f"{version}_total"]=(values[f"{version}_onset"]+values[f"{version}_ongoing"])/2
                values[f"{version}_shift_total"]=(values[f"{version}_onset"]+values[f"{version}_shift_ongoing"])/2
            rows.append(dict(fold=fold,panel=panel,**values,delta=values["v2_total"]-values["v1_total"],
                shift_delta=values["v2_shift_total"]-values["v1_shift_total"]))
    comparison=pd.DataFrame(rows);comparison.to_csv(out/"v1_v2_comparison.csv",index=False)
    window_scores=pd.read_csv(out/"window_scores.csv")
    cases=[]; sensitivity=[]
    for panel in WEAK:
        for condition in CONDITIONS:
            chosen=selection[panel][condition]
            if chosen=="baseline":continue
            for fold in ("janfeb","octnov"):
                subset=window_scores[(window_scores.panel==panel)&(window_scores.condition==condition)&
                    (window_scores.fold==fold)&window_scores.official_like.astype(str).str.lower().eq("true")]
                before=subset[subset.model=="baseline"].set_index("window_id")
                after=subset[subset.model==chosen].set_index("window_id")
                with np.load(out/"probabilities"/fold/f"{panel}_{condition}_baseline.npz") as pack:
                    counts=dict(zip(pack["window_id"],(pack["truth"]&pack["eligible"]).sum(axis=(1,2))))
                    raw_counts=dict(zip(pack["window_id"],pack["truth"].sum(axis=(1,2))))
                nonempty=np.array([counts[wid]>0 for wid in after.index])
                raw_empty=np.array([raw_counts[wid]==0 for wid in after.index])
                all_delta=float((after.iou-before.loc[after.index].iou).mean())
                delta_nonempty=float((after.loc[nonempty].iou-before.loc[after.index[nonempty]].iou).mean()) if nonempty.any() else np.nan
                empty_gain=float((after.loc[~nonempty].iou-before.loc[after.index[~nonempty]].iou).sum()/len(after))
                raw_empty_gain=float((after.loc[raw_empty].iou-before.loc[after.index[raw_empty]].iou).sum()/len(after))
                sensitivity.append(dict(fold=fold,panel=panel,condition=condition,model=chosen,
                    nonempty_windows=int(nonempty.sum()),empty_windows=int((~nonempty).sum()),
                    delta_all=all_delta,delta_nonempty=delta_nonempty,empty_gain_contribution=empty_gain,
                    raw_empty_gain_contribution=raw_empty_gain))
                for wid,r in after.iterrows():
                    old=before.loc[wid]
                    cases.append(dict(fold=fold,panel=panel,condition=condition,origin=r.origin,
                        v1=old.iou,v2=r.iou,delta=r.iou-old.iou,fp_v1=old.fp,fp_v2=r.fp,
                        fn_v1=old.fn,fn_v2=r.fn,eligible_true_cells=int(counts[wid]),raw_true_cells=int(raw_counts[wid])))
    case_frame=pd.DataFrame(cases)
    if len(case_frame):
        case_frame=case_frame.loc[case_frame.delta.abs().sort_values(ascending=False).index]
        case_frame.to_csv(out/"selected_window_deltas.csv",index=False)
    sensitivity=pd.DataFrame(sensitivity)
    if len(sensitivity):sensitivity.to_csv(out/"nonempty_sensitivity.csv",index=False)
    # Frozen V1 scores on all eight panels; untouched panels retain predictions.
    original=pd.read_csv(args.output/"cv_scores_enhanced_release.csv")
    on=original[(original.model=="lgb_median_labels")&(original.panel!="OVERALL")].set_index("panel").onset
    og=original[(original.model=="blend_50")&(original.panel!="OVERALL")].set_index("panel")
    final=[]
    for panel in args.panels:
        baseline=(float(on[panel])+float(og.loc[panel,"ongoing"]))/2
        baseline_shift=(float(on[panel])+float(og.loc[panel,"shift_ongoing"]))/2
        match=comparison[(comparison.fold=="janfeb")&(comparison.panel==panel)]
        if len(match):
            r=match.iloc[0];on2,og2,shift2=r.v2_onset,r.v2_ongoing,r.v2_shift_ongoing
        else:on2,og2,shift2=float(on[panel]),float(og.loc[panel,"ongoing"]),float(og.loc[panel,"shift_ongoing"])
        final.append(dict(panel=panel,v1=baseline,v2=(on2+og2)/2,delta=(on2+og2)/2-baseline,
            onset_v2=on2,ongoing_v2=og2,shift_v1=baseline_shift,shift_v2=(on2+shift2)/2))
    allpanels=pd.DataFrame(final)
    # Each of four families has exactly two scored directional panels.
    v1=float(allpanels.v1.mean());v2=float(allpanels.v2.mean())
    sh1=float(allpanels.shift_v1.mean());sh2=float(allpanels.shift_v2.mean())
    accepted=[(p,c,m) for p,cs in selection.items() for c,m in cs.items() if m!="baseline"]
    recommended="Recommend v2 for review: every changed panel passes both temporal and shift gates; each changed condition avoids regression in either fold." if accepted else "Recommend keeping v1: no candidate passed all required gates; v2 is the frozen fallback."
    proxy_driven=bool(len(sensitivity) and ((sensitivity.delta_all>0)&(sensitivity.delta_nonempty<0)&
        (sensitivity.raw_empty_gain_contribution>sensitivity.delta_all)).any())
    if proxy_driven:
        recommended="Recommend keeping v1 for the main assembler. V2 passes the formal temporal/shift gates, but at least one selected gain is entirely explained by empty raw-label horizons while nonempty horizons get worse. Keep v2 as a reviewed experiment until its gain is confirmed with better queue labels."
    protocol="""Primary tests independently select five onset and five ongoing windows per panel per held-out month. Jan–Feb models train June–December 2030; Oct–Nov models train June–September 2030. The baseline in each fold uses the frozen v1 architecture and selected members, refitted strictly before that holdout. September calibration predictions come from June–August fits; December calibration predictions come from June–November fits. All rolling-window targets must end before each training cutoff. Conditional priors exclude training date folds; held-out priors use earlier training dates only. Median labels are purged at the cutoff.\n\nCalibration uses the dense rolling union within September/December, separately per weak panel and condition. Its fixed small grid chooses probability temperature/intercept, expected-IoU cardinality scale, and optional contiguous onset interval. A Poisson extent model predicts T+30 queue count from causal all-link summaries; its blend strength is fitted on the disjoint calibration month. Inference never receives future eligibility. Onset keeps T+5..T+25 at zero.\n\nAcceptance requires each changed panel’s combined S_queue to strictly improve in both Jan–Feb and Oct–Nov, with no reduction in either shift-weighted S_queue. Each changed condition must additionally avoid ordinary and shift-weighted regression in either fold and improve at least one metric. A condition at a score ceiling may tie in one fold only when the combined panel still improves in both. Unchanged conditions and all five other panels retain v1. After the median-label sensitivity exposed an empty-horizon artifact, an additional guard requires each changed condition to avoid ordinary and shift regression in both nonempty-horizon subsets. Nonempty means at least one eligible raw queued target. This guard removes candidates; it never removes windows from the primary gate. The unfiltered formal choices remain in selection_formal.json and gates_formal.csv. Selection ranks guarded passing onset/ongoing pairs by the smaller panel improvement, then mean improvement and fewer changes. The extra guard was added after inspecting that sensitivity, so this remains a development gate, not an untouched final test; candidate comparisons can still induce selection optimism."""
    methods="""Fixed five onset seeds: 42, 137, 271, 419, 655. Each LightGBM fit uses deterministic=True and two threads by default, 350 rounds. The 58 new columns describe current bottleneck inflow/discharge, density imbalance, mass/clearance, recovery trends and horizon projections. Incident weights emphasize training-only rare queues, dissolution negatives and boundaries. An additional ongoing median-label sensitivity reuses the frozen v1 feature architecture and is scored against unchanged raw truth. One counterfactual per weak-panel training window translates normalized dynamic states by −5, −3, −2, +2, +3 or +5 links without wrapping, while destination static geometry/FD and priors remain fixed. Synthetic ramp values are missing; event priors are zero; target eligibility translates only for training. Synthetic weight is 0.35. Onset robust members additionally remove explicit location columns. These counterfactuals are regularization examples, not validated incident simulations."""
    experiment=("# Task 2 v2 experiments\n\n"+protocol+"\n\n"+methods+"\n\n## Selected changes\n\n"+
        markdown(pd.DataFrame([dict(panel=p,condition=c,model=m) for p,cs in selection.items() for c,m in cs.items()]))+
        "\n\n## Primary per-panel comparison\n\n"+markdown(comparison,["fold","panel","v1_onset","v2_onset","v1_ongoing","v2_ongoing","v1_total","v2_total","delta","shift_delta"])+
        ("\n\n## Largest selected window changes\n\n"+markdown(case_frame.head(15))+"\n\nAll primary windows remain in the averages. Improvements on the empty/raw-label cases in report09 are proxy-sensitive.\n" if len(case_frame) else "")+
        ("\n\n## Nonempty-horizon sensitivity\n\nDiagnostic only; no primary windows or gate scores are removed. Nonempty means at least one eligible queued target. Empty gain contribution is the empty-window summed delta divided by all primary windows; raw-empty contribution additionally requires no raw queue anywhere, including ineligible cells.\n\n"+markdown(sensitivity)+"\n" if len(sensitivity) else "")+
        "\n\n## Every candidate and gate\n\nThe formal winner before the nonempty guard is retained separately. Full median-label I405_S ongoing gains +0.093617 in Jan–Feb because one all-empty raw horizon changes from 0 to 1; its nonempty-window mean falls by 0.007979. The production guard rejects that candidate. The 50% median blend keeps both nonempty holdouts positive without an empty-window gain.\n\nPositive deltas are absolute IoU changes, not percentages. Nonregressing marks eligible condition candidates; accepted marks the deployed condition choice. The pair must also pass both strict panel-total gates; selection.json records it.\n\n"+markdown(gates)+
        "\n\nPer-window scores, full calibration trials, fixed seed models, raw-derived caches and prediction changes are under outputs/t2/v2. Error diagnosis precedes these experiments in see report, Sec. III-C. The gate retains zero-raw-queue and early-onset proxy mismatches rather than excluding difficult examples.\n")
    (PROJECT/"reports/codex_14_v2_experiments.md").write_text(experiment)
    v1_experiments=PROJECT/"reports/codex_02_task2_experiments.md"
    marker="<!-- TASK2_V2 -->"
    previous=v1_experiments.read_text().split(marker)[0].rstrip() if v1_experiments.exists() else "# Task 2 experiments"
    v1_experiments.write_text(previous+"\n\n"+marker+"\n## Frozen-v1 / gated-v2 iteration\n\nAll new models, five-seed ensembles, counterfactual/incident training, onset extent and disjoint calibration results are logged in [codex_14_v2_experiments.md](codex_14_v2_experiments.md), with every candidate’s two temporal deltas and both shift deltas. The original v1 experiments above remain unchanged.\n")
    tests=(out/"tests.log").read_text().strip().splitlines()[-1] if (out/"tests.log").exists() else "pending final scoped test run"
    report=(f"# Task 2 status: frozen v1 and gated v2\n\n{recommended}\n\n"
        f"Jan–Feb all-panel S_queue: **v1 {v1:.4f} → v2 {v2:.4f}** (Δ {v2-v1:+.4f}). Shift-weighted S_queue: **{sh1:.4f} → {sh2:.4f}**. Expected queue contribution: **0.30 × {v2:.4f} = {0.30*v2:.4f}**, a local proxy estimate, not a measured leaderboard score. Nothing submitted to Kaggle.\n\n"
        +markdown(allpanels,["panel","v1","v2","delta","onset_v2","ongoing_v2","shift_v1","shift_v2"])+
        "\n\nFrozen v1 uses the median-label onset classifier and 50/50 full/no-location ongoing blend. V2 condition choices (`baseline` retains that frozen member):\n\n"+
        markdown(pd.DataFrame([dict(panel=p,condition=c,model=m) for p,cs in selection.items() for c,m in cs.items()]))+
        "\n\nBoth temporal gates for the three weak panels:\n\n"+markdown(comparison,["fold","panel","v1_total","v2_total","delta","shift_delta"])+
        ("\n\nNonempty-horizon sensitivity (diagnostic, primary scores unchanged):\n\n"+markdown(sensitivity)+"\n" if len(sensitivity) else "")+
        "\n\n"+protocol+"\n\n"+methods+"\n\n"
        f"Artifacts: outputs/t2/queue_submission_v1.csv (frozen), outputs/t2/queue_submission_v2.csv (candidate), and unchanged outputs/t2/queue_submission.csv. Each has {manifest['rows']:,} required rows. V2 changes {manifest['changed_cells']:,} cell predictions; complete per-task and common-template coverage is recorded in outputs/t2/v2/submission_verification.json. V1 SHA256 `{manifest['v1_sha256']}`; v2 `{manifest['v2_sha256']}`. Scoped tests: {tests}.\n\n"
        "Detailed failure patterns: [codex_09_weak_panels.md](codex_09_weak_panels.md). All candidates, negative results and per-panel deltas: [codex_14_v2_experiments.md](codex_14_v2_experiments.md). Independent audit: [codex_13_v2_final_audit.md](codex_13_v2_final_audit.md). Frozen v1 status is preserved in outputs/t2/codex_STATUS_v1.md.\n\n"
        "Rebuild v2 from raw data with one command (v1 queue file is deliberately retained as the unchanged-panel carrier):\n\n```bash\nPYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m tfb.t2.v2_experiment --stage all\n```\n\n"
        "Risks: only ten primary windows per condition/panel in each two-month holdout; hidden truth is noise-free while local truth is noisy. Three weak-panel ongoing horizons have no raw queue despite filled selector events, and one onset has early positives inconsistent with the released T+30 artifact. Calibration is based on a single earlier month per gate; final refit changes the model population. Spatial shifts cannot reproduce a distant new bottleneck, incident capacity loss or unknown ramp/off-flow. Testing multiple variants on both gates can still overfit their joint selection. No future target quality, future ramp data or Task 1 reconstruction enters features or decoding.\n\n"
        "Next ideas ranked by expected gain:\n\n1. Train truly non-recurrent incident episodes with plausible capacity reductions and explicit event durations; use a fresh temporal gate.\n2. Correct onset bottleneck ranking with causal corridor-wide demand routing and geometry-only members; monitor wrong-site distances.\n3. Balance ongoing loss by window and episode size, especially tiny short-lived queues, instead of pooled cells.\n4. Use component-aware joint extent/head probabilities and episode-level dependence in expected-IoU decoding.\n5. Validate calibration transfer across several strictly earlier months and quantify bootstrap uncertainty by day.\n")
    (PROJECT/"reports/codex_STATUS.md").write_text(report)
    dump_json(out/"score_summary.json",dict(v1=v1,v2=v2,shift_v1=sh1,shift_v2=sh2,
        expected_queue_contribution=.30*v2,proxy_driven=proxy_driven,recommendation=recommended))
