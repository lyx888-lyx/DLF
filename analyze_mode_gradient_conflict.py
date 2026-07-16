"""Stage 6A train-only missing-pattern gradient interference audit."""
import argparse
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import get_config_regression
from trains.singleTask.HingeLoss import HingeLoss
from trains.singleTask.cf_compat_kd_utils import CACHE_COLUMNS, cache_paths, gated_kd_loss, locate_stage1_evaluator
from trains.singleTask.fixed_kd_utils import build_frozen_teacher, checkpoint_sha256, teacher_grad_count, teacher_lav_prediction
from trains.singleTask.gradient_conflict_utils import (
    ALL_GROUP, GROUPS, MISSING_MODES, MODES, MODE_PAIRS, add_gradients,
    assign_compatibility_quartiles, build_parameter_groups, capture_rng_state,
    classify_conflicts, clone_state_dict, cosine_summary, gradient_cosine,
    gradient_dot, group_gradient_stats, ordered_gradients, protected_audit_state,
    representation_statistics, rng_states_equal, sha256_file, stable_seed,
    states_equal,
)
from trains.singleTask.missing_utils import (
    MissingModalityWrapper, build_single_split_loader, clean_checkpoint_path,
    compute_full_dlf_loss, compute_task_loss, mode_to_mask,
)
from trains.singleTask.model.DLF import DLF
from utils.functions import assign_gpu, setup_seed


BASE_BRANCH="experiment/cf-compat-kd-multiseed-v1"
BASE_COMMIT="276fc26e6e7d1998f7dd746edfca1efbbb691e55"
STATE_FILES={"gate3_init":"gate3_init","cfcompat_best_valid":"cfcompat_best"}


def parse_args():
    parser=argparse.ArgumentParser(description="Pure train-only Stage 6A gradient interference audit.")
    parser.add_argument("--dataset",choices=("mosi",),default="mosi"); parser.add_argument("--seed",type=int,default=1111)
    parser.add_argument("--num-workers",type=int,default=0); parser.add_argument("--gpu-ids",nargs="*",type=int,default=[0])
    parser.add_argument("--bootstrap-samples",type=int,default=2000); parser.add_argument("--bootstrap-seed",type=int,default=260616)
    parser.add_argument("--model-save-dir",default="pt"); parser.add_argument("--result-root",default="result")
    parser.add_argument("--config-file",default="config/config.json"); parser.add_argument("--summary-only",action="store_true")
    parser.add_argument("--verify-existing-artifacts",action="store_true")
    args=parser.parse_args()
    if args.seed!=1111: parser.error("Stage 6A is pre-registered for seed1111 only.")
    if args.num_workers!=0: parser.error("Stage 6A fixes num_workers=0.")
    if args.bootstrap_samples!=2000 and not args.summary_only: parser.error("Stage 6A fixes bootstrap resamples at 2000.")
    if args.summary_only and not args.verify_existing_artifacts: parser.error("--summary-only requires --verify-existing-artifacts.")
    return args


def git(*args): return subprocess.check_output(["git",*args],text=True).strip()


def build_config(cli):
    args=get_config_regression("DLF",cli.dataset,cli.config_file); args.mode="train"; args.feature_T=args.feature_A=args.feature_V=""
    args.is_training=True; args.train_mode="regression"; args.seed=args.cur_seed=cli.seed; args.device=assign_gpu(list(cli.gpu_ids)); return args


def output_directory(cli): return Path(cli.result_root)/"analysis"/"mode_gradient_conflict_v1"/cli.dataset/"seed{}".format(cli.seed)


def locate_cfcompat_checkpoint(root,seed):
    source=Path(root)/"missing_baseline"/"cf_compat_kd_v1"/"benchmark_train"/"mosi_per_seed.csv"
    if not source.is_file(): raise FileNotFoundError(source)
    frame=pd.read_csv(source); selected=frame.loc[frame.Seed.astype(int).eq(int(seed))]
    if len(selected)!=1: raise ValueError("CFCompat result must contain one seed1111 row.")
    path=Path(str(selected.iloc[0].MainCheckpoint))
    if "diagnostic" in str(path) or "best_test" in str(path): raise ValueError("CFCompat audit state must be validation-best.")
    if not path.is_file(): raise FileNotFoundError(path)
    return path,source


def load_locked_audit_cache(root,dataset,expected_evaluator_sha):
    """Read the historical Stage 3 train-only manifest without requiring newer optional flags."""
    paths=cache_paths(root,dataset)
    if not paths["csv"].is_file() or not paths["config"].is_file(): raise FileNotFoundError("Locked Stage 3 cache is absent.")
    config=json.loads(paths["config"].read_text()); frame=pd.read_csv(paths["csv"])
    if config.get("version")!="cf_compat_v1" or config.get("source")!="train_only" or int(config.get("train_sample_count",-1))!=1284:
        raise ValueError("Stage 3 cache manifest is not the locked 1284-sample train-only artifact.")
    if config.get("evaluator_sha256")!=expected_evaluator_sha: raise ValueError("Stage 3 cache Evaluator SHA binding failed.")
    if list(frame.columns)!=list(CACHE_COLUMNS) or len(frame)!=1284 or frame.sample_index.nunique()!=1284:
        raise ValueError("Stage 3 cache CSV schema/sample binding is invalid.")
    for mode in MISSING_MODES:
        values=frame["compat_{}".format(mode)].to_numpy(float)
        if not np.isfinite(values).all() or not ((values>0)&(values<1)).all(): raise ValueError("Cached compatibility is invalid.")
    return frame,{int(row.sample_index):row._asdict() for row in frame.itertuples(index=False)},paths,config


def initialize_states(args,cli,gate3,cfcompat):
    backbone=DLF(args).to(args.device); backbone.load_state_dict(torch.load(gate3,map_location=args.device),strict=True)
    gate3_student=MissingModalityWrapper(backbone,args.feature_dims[1],args.feature_dims[2]).to(args.device)
    cf_student=MissingModalityWrapper(DLF(args).to(args.device),args.feature_dims[1],args.feature_dims[2]).to(args.device)
    cf_student.load_state_dict(torch.load(cfcompat,map_location=args.device),strict=True)
    return {"gate3_init":gate3_student,"cfcompat_best_valid":cf_student}


def batch_inputs(batch,device):
    return batch["text"].to(device),batch["audio"].to(device),batch["vision"].to(device),batch["labels"]["M"].to(device).view(-1,1)


def subset_output(output,mask):
    return {key:value[mask] for key,value in output.items() if key in ("output_logit","logits_c","logits_l_hetero","logits_v_hetero","logits_a_hetero")}


def compatibility_tensor(table,indices,mode,device,dtype):
    values=[]
    for index in indices:
        if int(index) not in table: raise KeyError("Missing compatibility binding: {}".format(index))
        values.append(float(table[int(index)]["compat_{}".format(mode)]))
    result=torch.tensor(values,device=device,dtype=dtype)
    if not torch.all((result>0)&(result<1)): raise ValueError("Compatibility outside (0,1).")
    return result.detach()


def record_task_cosines(state,batch_index,task_grads,params,group_indices,rows):
    for a,b in MODE_PAIRS:
        for group,indices in group_indices.items():
            cosine=gradient_cosine(task_grads[a],task_grads[b],params,indices)
            left=group_gradient_stats(task_grads[a],params,indices); right=group_gradient_stats(task_grads[b],params,indices)
            rows.append({"RowType":"batch","State":state,"Batch":batch_index,"ModeA":a,"ModeB":b,"ParameterGroup":group,
                         "Cosine":cosine,"NormA":left["GradientNorm"],"NormB":right["GradientNorm"],
                         "ZeroFractionA":left["ZeroGradientFraction"],"ZeroFractionB":right["ZeroGradientFraction"],
                         "VectorDimensionality":left["VectorDimensionality"],"Finite":True})


def record_transfer(state,batch_index,task_grads,params,group_indices,rows):
    for group,indices in group_indices.items():
        for a in MODES:
            for b in MODES:
                dot=gradient_dot(task_grads[a],task_grads[b],indices); cosine=gradient_cosine(task_grads[a],task_grads[b],params,indices)
                rows.append({"RowType":"batch","State":state,"Batch":batch_index,"ParameterGroup":group,"SourceMode":a,"TargetMode":b,
                             "raw_dot":dot,"normalized_transfer":cosine,"first_order_effect_a_on_b":-dot,"negative_transfer":bool(dot<0)})


def record_task_kd(state,batch_index,task_grads,kd_grads,params,group_indices,rows):
    for mode in MISSING_MODES:
        for alignment,left in (("missing_task_vs_kd",task_grads[mode]),("lav_task_vs_kd",task_grads["LAV"])):
            for group,indices in group_indices.items():
                task_stat=group_gradient_stats(left,params,indices); kd_stat=group_gradient_stats(kd_grads[mode],params,indices)
                rows.append({"RowType":"batch","State":state,"Batch":batch_index,"AlignmentType":alignment,"Mode":mode,"ParameterGroup":group,
                             "Cosine":gradient_cosine(left,kd_grads[mode],params,indices),"TaskGradientNorm":task_stat["GradientNorm"],
                             "KDGradientNorm":kd_stat["GradientNorm"],"GradientNormRatio":kd_stat["GradientNorm"]/(task_stat["GradientNorm"]+1e-12)})


def record_norms(state,batch_index,task_grads,total_grads,params,group_indices,rows):
    for group,indices in group_indices.items():
        task_norms={mode:group_gradient_stats(task_grads[mode],params,indices)["GradientNorm"] for mode in MODES}
        total_norms={mode:group_gradient_stats(total_grads[mode],params,indices)["GradientNorm"] for mode in MISSING_MODES}
        median=float(np.median(list(task_norms.values()))); total=sum(task_norms.values())
        row={"RowType":"batch","State":state,"Batch":batch_index,"ParameterGroup":group,
             **{"norm_task_{}".format(k):v for k,v in task_norms.items()},**{"norm_total_{}".format(k):v for k,v in total_norms.items()},
             "mode_dominance_ratio":max(task_norms.values())/(median+1e-12)}
        row.update({"gradient_share_{}".format(k):v/(total+1e-12) for k,v in task_norms.items()}); rows.append(row)


def record_quartiles(state,batch_index,indices,labels,outputs,teacher_prediction,quartile_map,compat_table,params,group_indices,rows,criterion):
    selected_groups=("shared_multimodal_fusion","prediction_and_task_heads","missing_tokens_and_mask_adapter",ALL_GROUP)
    for mode in MISSING_MODES:
        compatibility=compatibility_tensor(compat_table,indices,mode,labels.device,labels.dtype)
        for quartile in ("Q1_low","Q2","Q3","Q4_high"):
            flags=torch.tensor([quartile_map[(int(index),mode)]==quartile for index in indices],device=labels.device,dtype=torch.bool)
            count=int(flags.sum())
            if count==0: continue
            task_loss,_=compute_task_loss(subset_output(outputs[mode],flags),labels[flags],criterion)
            kd_loss,_=gated_kd_loss(outputs[mode]["output_logit"][flags],teacher_prediction[flags],compatibility[flags])
            task_grad=ordered_gradients(task_loss,params,retain_graph=True); kd_grad=ordered_gradients(kd_loss,params,retain_graph=True)
            for group in selected_groups:
                idx=group_indices[group]; ts=group_gradient_stats(task_grad,params,idx); ks=group_gradient_stats(kd_grad,params,idx)
                rows.append({"RowType":"batch","State":state,"Batch":batch_index,"Mode":mode,"Quartile":quartile,"ParameterGroup":group,
                             "SampleCount":count,"MeanCompatibility":float(compatibility[flags].mean()),"TaskGradientNorm":ts["GradientNorm"],
                             "KDGradientNorm":ks["GradientNorm"],"Cosine":gradient_cosine(task_grad,kd_grad,params,idx)})
            del task_grad,kd_grad


def append_summaries(task_rows,transfer_rows,kd_rows,quartile_rows,norm_rows,bootstrap_samples,bootstrap_seed):
    task=pd.DataFrame(task_rows); summary=[]
    for keys,local in task[task.RowType.eq("batch")].groupby(["State","ModeA","ModeB","ParameterGroup"],dropna=False):
        stats=cosine_summary(local.Cosine,bootstrap_samples,stable_seed(bootstrap_seed,*keys)); summary.append({"RowType":"summary","State":keys[0],"Batch":np.nan,"ModeA":keys[1],"ModeB":keys[2],"ParameterGroup":keys[3],**stats})
    task=pd.concat([task,pd.DataFrame(summary)],ignore_index=True,sort=False)
    transfer=pd.DataFrame(transfer_rows); summary=[]
    for keys,local in transfer[transfer.RowType.eq("batch")].groupby(["State","ParameterGroup","SourceMode","TargetMode"]):
        summary.append({"RowType":"summary","State":keys[0],"Batch":np.nan,"ParameterGroup":keys[1],"SourceMode":keys[2],"TargetMode":keys[3],
                        "raw_dot":float(local.raw_dot.mean()),"raw_dot_median":float(local.raw_dot.median()),
                        "normalized_transfer":float(local.normalized_transfer.mean()),"normalized_transfer_median":float(local.normalized_transfer.median()),
                        "first_order_effect_a_on_b":float(local.first_order_effect_a_on_b.mean()),"negative_transfer":float(local.negative_transfer.mean())})
    transfer=pd.concat([transfer,pd.DataFrame(summary)],ignore_index=True,sort=False)
    kd=pd.DataFrame(kd_rows); summary=[]
    for keys,local in kd[kd.RowType.eq("batch")].groupby(["State","AlignmentType","Mode","ParameterGroup"]):
        stats=cosine_summary(local.Cosine,bootstrap_samples,stable_seed(bootstrap_seed,*keys)); summary.append({"RowType":"summary","State":keys[0],"Batch":np.nan,"AlignmentType":keys[1],"Mode":keys[2],"ParameterGroup":keys[3],
            **stats,"TaskGradientNorm":float(local.TaskGradientNorm.mean()),"KDGradientNorm":float(local.KDGradientNorm.mean()),"GradientNormRatio":float(local.GradientNormRatio.mean())})
    kd=pd.concat([kd,pd.DataFrame(summary)],ignore_index=True,sort=False)
    quartile=pd.DataFrame(quartile_rows); summary=[]
    for keys,local in quartile[quartile.RowType.eq("batch")].groupby(["State","Mode","Quartile","ParameterGroup"]):
        x=local.Cosine[np.isfinite(local.Cosine)]
        summary.append({"RowType":"summary","State":keys[0],"Batch":np.nan,"Mode":keys[1],"Quartile":keys[2],"ParameterGroup":keys[3],
                        "count":len(local),"SampleCount":int(local.SampleCount.sum()),"MeanCompatibility":float(np.average(local.MeanCompatibility,weights=local.SampleCount)),
                        "TaskGradientNorm":float(local.TaskGradientNorm.mean()),"KDGradientNorm":float(local.KDGradientNorm.mean()),
                        "Cosine":float(x.mean()) if len(x) else np.nan,"median":float(x.median()) if len(x) else np.nan,"negative_fraction":float((x<0).mean()) if len(x) else np.nan})
    quartile=pd.concat([quartile,pd.DataFrame(summary)],ignore_index=True,sort=False)
    norms=pd.DataFrame(norm_rows); numeric=[c for c in norms.columns if c not in ("RowType","State","Batch","ParameterGroup")]; summary=[]
    for keys,local in norms[norms.RowType.eq("batch")].groupby(["State","ParameterGroup"]):
        values={}
        for c in numeric:
            finite=local[c][np.isfinite(local[c])]; values[c]=float(finite.mean()) if len(finite) else np.nan
        summary.append({"RowType":"summary","State":keys[0],"Batch":np.nan,"ParameterGroup":keys[1],**values})
    norms=pd.concat([norms,pd.DataFrame(summary)],ignore_index=True,sort=False)
    return task,transfer,kd,quartile,norms


def audit_state(state_name,model,teacher,loader,args,cache_by_index,quartile_map,params,group_indices):
    criterion=nn.L1Loss(); cosine=nn.CosineEmbeddingLoss(); hinge=HingeLoss()
    task_rows=[]; transfer_rows=[]; kd_rows=[]; quartile_rows=[]; norm_rows=[]; samples=[]; representations={m:[] for m in MODES}
    captured=[]
    def hook(module,inputs): captured.append(inputs[0].detach().cpu().numpy())
    handle=model.backbone.proj1.register_forward_pre_hook(hook)
    rng_before=capture_rng_state(); state_before=clone_state_dict(model); training_before=model.training
    with protected_audit_state(model,teacher):
        for batch_index,batch in enumerate(loader):
            text,audio,vision,labels=batch_inputs(batch,args.device); indices=batch["index"].view(-1).cpu().numpy().astype(int).tolist(); ids=list(batch["id"])
            for index,sample_id in zip(indices,ids): samples.append({"sample_index":int(index),"sample_id":str(sample_id),"Batch":batch_index})
            outputs={}; task_losses={}
            for mode in MODES:
                mask=mode_to_mask(mode,labels.size(0),args.device,audio.dtype); before=len(captured); outputs[mode]=model(text,audio,vision,mask)
                if len(captured)!=before+1: raise RuntimeError("Shared representation hook did not fire exactly once.")
                representations[mode].append(captured[-1])
                task_losses[mode]=(compute_full_dlf_loss(outputs[mode],labels,criterion,cosine,hinge)[0] if mode=="LAV" else compute_task_loss(outputs[mode],labels,criterion)[0])
            teacher_prediction=teacher_lav_prediction(teacher,text,audio,vision).view(-1); kd_losses={}
            for mode in MISSING_MODES:
                compatibility=compatibility_tensor(cache_by_index,indices,mode,args.device,labels.dtype)
                kd_losses[mode]=gated_kd_loss(outputs[mode]["output_logit"],teacher_prediction,compatibility)[0]
            task_grads={mode:ordered_gradients(task_losses[mode],params,retain_graph=True) for mode in MODES}
            kd_grads={mode:ordered_gradients(kd_losses[mode],params,retain_graph=True) for mode in MISSING_MODES}
            total_grads={mode:add_gradients(task_grads[mode],kd_grads[mode]) for mode in MISSING_MODES}
            record_task_cosines(state_name,batch_index,task_grads,params,group_indices,task_rows)
            record_transfer(state_name,batch_index,task_grads,params,group_indices,transfer_rows)
            record_task_kd(state_name,batch_index,task_grads,kd_grads,params,group_indices,kd_rows)
            record_norms(state_name,batch_index,task_grads,total_grads,params,group_indices,norm_rows)
            record_quartiles(state_name,batch_index,indices,labels,outputs,teacher_prediction,quartile_map,cache_by_index,params,group_indices,quartile_rows,criterion)
            if teacher_grad_count(teacher): raise RuntimeError("Frozen Teacher received gradients.")
            del outputs,task_losses,kd_losses,task_grads,kd_grads,total_grads
            if (batch_index+1)%10==0: print("state={} batches={}/{}".format(state_name,batch_index+1,len(loader)),flush=True)
    handle.remove()
    if not states_equal(state_before,model.state_dict()): raise RuntimeError("Student state changed during audit.")
    rng_after=capture_rng_state()
    if not rng_states_equal(rng_before,rng_after): raise RuntimeError("Audit RNG was not preserved.")
    if model.training!=training_before: raise RuntimeError("Student train/eval mode was not restored.")
    reps={mode:np.concatenate(values,axis=0) for mode,values in representations.items()}
    representation_frame,representation_summary=representation_statistics(reps)
    return {"task":task_rows,"transfer":transfer_rows,"kd":kd_rows,"quartile":quartile_rows,"norm":norm_rows,
            "samples":pd.DataFrame(samples),"representation":representation_frame,"representation_summary":representation_summary,
            "parameter_changed":False,"buffer_changed":False,"rng_preserved":True}


def write_report(directory,summary,task,kd,quartile,norms,representations):
    flags=summary["GradientConflictFlags"]
    recommendation=("Consider a future mask-specific representation adapter or missing-gradient isolation audit; none was implemented." if flags["shared_fusion_conflict"] else
                    "Consider future mode-specific heads only; fusion isolation is not supported." if flags["prediction_head_only_conflict"] else
                    "Investigate CFCompatKD-task optimization conflict without adding another Teacher." if flags["cfkd_task_conflict"] else
                    "No mask-specific parameterization is supported; prioritize data-level missing augmentation or representation consistency." if flags["no_systematic_conflict"] else
                    "Evidence is mixed; do not implement a structural change before manual review.")
    summary["FinalEvidenceBasedRecommendation"]=recommendation
    current=task[(task.State=="cfcompat_best_valid")&(task.RowType=="summary")].sort_values("median").head(20)
    state_comparison=task[(task.RowType=="summary")&task.ParameterGroup.isin(["shared_multimodal_fusion","prediction_and_task_heads",ALL_GROUP])]
    kd_current=kd[(kd.State=="cfcompat_best_valid")&(kd.RowType=="summary")].sort_values("median").head(20)
    q_current=quartile[(quartile.State=="cfcompat_best_valid")&(quartile.RowType=="summary")]
    n_current=norms[(norms.State=="cfcompat_best_valid")&(norms.RowType=="summary")]
    def md(frame):
        frame=frame.copy(); cols=list(frame.columns); lines=["| "+" | ".join(cols)+" |","| "+" | ".join(["---"]*len(cols))+" |"]
        for row in frame.itertuples(index=False,name=None): lines.append("| "+" | ".join("NA" if pd.isna(x) else ("{:.6f}".format(x) if isinstance(x,(float,np.floating)) else str(x)) for x in row)+" |")
        return "\n".join(lines)
    lines=["# Stage 6A Missing-Pattern Gradient Interference Audit","","This is a pure MOSI train-only diagnostic. No optimizer, parameter update, new checkpoint, validation data, test data, Adapter, or prediction head was used.","",
           "## Isolation and provenance","",f"- Train samples: {summary['TrainSampleCount']}; batches: {summary['AuditBatchCount']}; batch size: {summary['BatchSize']}.",
           f"- Base: `{summary['BaseBranch']}` at `{summary['BaseCommit']}`; implementation: `{summary['ImplementationCommit']}`.",
           f"- Gate3 SHA: `{summary['Gate3CheckpointSHA256']}`; CFCompat validation-best SHA: `{summary['CFCompatCheckpointSHA256']}`.",
           f"- TrainOnly={summary['TrainOnly']}; ValidAccessed={summary['ValidAccessed']}; TestAccessed={summary['TestAccessed']}; OptimizerStepCount={summary['OptimizerStepCount']}.",
           f"- ParameterChanged={summary['ParameterChanged']}; BufferChanged={summary['BufferChanged']}; RNGPreserved={summary['RNGPreserved']}.","",
           "## Parameter groups","",md(pd.DataFrame([{"Group":k,**{x:v for x,v in val.items() if x!='parameter_names'}} for k,val in summary["ParameterGroups"].items()])),"",
           "## Gate3 initialization versus CFCompat validation-best","",md(state_comparison[["State","ModeA","ModeB","ParameterGroup","count","mean","median","negative_fraction","bootstrap_mean_ci_low","bootstrap_mean_ci_high"]]),"",
           "## Most conflicting task-gradient results after CFCompatKD","",md(current[["ModeA","ModeB","ParameterGroup","count","mean","median","negative_fraction","strong_negative_fraction","bootstrap_mean_ci_low","bootstrap_mean_ci_high"]]),"",
           "## CFCompatKD versus task gradients","",md(kd_current[["AlignmentType","Mode","ParameterGroup","count","mean","median","negative_fraction","GradientNormRatio"]]),"",
           "## Compatibility quartiles","",md(q_current[["Mode","Quartile","ParameterGroup","count","SampleCount","MeanCompatibility","Cosine","median","negative_fraction","TaskGradientNorm","KDGradientNorm"]]),"",
           "## Gradient norm and dominance","",md(n_current),"","## Representation alignment","",md(representations),"",
           "## G1-G5 classification","",json.dumps(flags,indent=2,sort_keys=True),"",f"Most conflicting pair: **{flags['MostConflictingModePair']}** in **{flags['MostConflictingParameterGroup']}**.",
           f"Only-L bottleneck supported: **{flags['single_mode_bottleneck'] and flags['bottleneck_mode']=='L'}**.",f"Recommendation: {recommendation}","",
           "## Stop declaration","","No training was run, test was not accessed, parameters/buffers were unchanged, and no Stage 6B implementation was created.",""]
    (directory/"stage6a_mode_gradient_conflict_audit.md").write_text("\n".join(lines),encoding="utf-8")


def run_full(cli):
    setup_seed(cli.seed); args=build_config(cli); directory=output_directory(cli); directory.mkdir(parents=True,exist_ok=True)
    gate3=clean_checkpoint_path(cli.model_save_dir,args.dataset_name,cli.seed); cfcompat,cf_source=locate_cfcompat_checkpoint(cli.result_root,cli.seed)
    evaluator,evaluator_epoch,evaluator_source=locate_stage1_evaluator(cli.result_root,cli.dataset,cli.seed)
    cache_frame,cache_by_index,locked_cache_paths,cache_config=load_locked_audit_cache(cli.result_root,cli.dataset,checkpoint_sha256(evaluator))
    if len(cache_frame)!=1284 or cache_frame.sample_index.nunique()!=1284: raise RuntimeError("Audit requires exactly 1284 unique train cache rows.")
    loader=build_single_split_loader(args,"train",cli.num_workers)
    if getattr(loader,"drop_last",None) is not False or not isinstance(loader.sampler,torch.utils.data.SequentialSampler): raise RuntimeError("Audit loader must be shuffle=False/drop_last=False.")
    teacher=build_frozen_teacher(DLF,args,gate3); states=initialize_states(args,cli,gate3,cfcompat)
    names,params,group_indices,param_frame,param_manifest=build_parameter_groups(states["gate3_init"])
    other_fraction=param_manifest["other_trainable_numel_fraction"]
    if other_fraction>.05: raise RuntimeError("Parameter mapping incomplete.")
    param_frame.to_csv(directory/"parameter_group_manifest.csv",index=False); (directory/"parameter_group_manifest.json").write_text(json.dumps(param_manifest,indent=2,sort_keys=True)+"\n")
    quartile_map=assign_compatibility_quartiles(cache_frame); all_raw={k:[] for k in ("task","transfer","kd","quartile","norm")}; reps={}; sample_manifest=None
    for state_name,model in states.items():
        _,state_params,state_groups,_,state_manifest=build_parameter_groups(model)
        if names!=[n for n,_ in model.named_parameters() if _.requires_grad] or state_manifest["trainable_numel"]!=param_manifest["trainable_numel"]: raise RuntimeError("State parameter mapping differs.")
        outcome=audit_state(state_name,model,teacher,loader,args,cache_by_index,quartile_map,state_params,state_groups)
        for key in all_raw: all_raw[key].extend(outcome[key])
        reps[state_name]=outcome["representation"]; (directory/"representation_summary_{}.json".format(STATE_FILES[state_name])).write_text(json.dumps(outcome["representation_summary"],indent=2,sort_keys=True)+"\n")
        if sample_manifest is None: sample_manifest=outcome["samples"]
        elif not sample_manifest.equals(outcome["samples"]): raise RuntimeError("State audit sample order differs.")
    if len(sample_manifest)!=1284 or sample_manifest.sample_index.nunique()!=1284: raise RuntimeError("Audit sample manifest is not 1284 unique train samples.")
    sample_manifest.to_csv(directory/"audit_sample_manifest.csv",index=False)
    task,transfer,kd,quartile,norms=append_summaries(all_raw["task"],all_raw["transfer"],all_raw["kd"],all_raw["quartile"],all_raw["norm"],cli.bootstrap_samples,cli.bootstrap_seed)
    for state_name,suffix in STATE_FILES.items():
        task[task.State.eq(state_name)].to_csv(directory/"task_gradient_cosines_{}.csv".format(suffix),index=False)
        transfer[transfer.State.eq(state_name)].to_csv(directory/"gradient_transfer_{}.csv".format(suffix),index=False)
        kd[kd.State.eq(state_name)].to_csv(directory/"task_kd_alignment_{}.csv".format(suffix),index=False)
        quartile[quartile.State.eq(state_name)].to_csv(directory/"compatibility_quartile_gradient_{}.csv".format(suffix),index=False)
        norms[norms.State.eq(state_name)].to_csv(directory/"gradient_norms_{}.csv".format(suffix),index=False)
        reps[state_name].to_csv(directory/"representation_alignment_{}.csv".format(suffix),index=False)
    flags=classify_conflicts(task,kd)
    qcurrent=quartile[(quartile.State=="cfcompat_best_valid")&(quartile.RowType=="summary")]
    if len(qcurrent):
        qmost=qcurrent.sort_values(["median","negative_fraction"],ascending=[True,False]).iloc[0]
        flags["OverallTaskKDMostConflictingMode"]=flags["TaskKDMostConflictingMode"]
        flags["TaskKDMostConflictingMode"]=qmost.Mode; flags["TaskKDMostConflictingQuartile"]=qmost.Quartile
        flags["TaskKDMostConflictingParameterGroup"]=qmost.ParameterGroup
    summary={"Dataset":cli.dataset,"Seed":cli.seed,"TrainSampleCount":len(sample_manifest),"AuditBatchCount":len(loader),"BatchSize":args.batch_size,
             "BaseBranch":BASE_BRANCH,"BaseCommit":BASE_COMMIT,"ImplementationCommit":git("rev-parse","HEAD"),
             "Gate3Checkpoint":str(gate3),"Gate3CheckpointSHA256":checkpoint_sha256(gate3),"CFCompatCheckpoint":str(cfcompat),"CFCompatCheckpointSHA256":checkpoint_sha256(cfcompat),
             "CompatibilityCache":str(locked_cache_paths["csv"]),"CompatibilityCacheSHA256":checkpoint_sha256(locked_cache_paths["csv"]),
             "EvaluatorCheckpoint":str(evaluator),"EvaluatorCheckpointSHA256":checkpoint_sha256(evaluator),"TrainOnly":True,"ValidAccessed":False,"TestAccessed":False,
             "OptimizerStepCount":0,"ParameterChanged":False,"BufferChanged":False,"RNGPreserved":True,"ParameterGroups":param_manifest["groups"],
             "GradientConflictFlags":flags,"RepresentationAuditAvailable":True,"FinalEvidenceBasedRecommendation":"pending report generation"}
    write_report(directory,summary,task,kd,quartile,norms,pd.concat([frame.assign(State=state) for state,frame in reps.items()],ignore_index=True))
    (directory/"audit_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    config={"dataset":cli.dataset,"seed":cli.seed,"source_split":"train","num_workers":0,"shuffle":False,"drop_last":False,"bootstrap_samples":cli.bootstrap_samples,
            "bootstrap_seed":cli.bootstrap_seed,"valid_accessed":False,"test_accessed":False,"optimizer_created":False,"optimizer_step_count":0,
            "cfcompat_result_source":str(cf_source),"evaluator_result_source":str(evaluator_source),"evaluator_best_epoch":evaluator_epoch}
    (directory/"audit_config.json").write_text(json.dumps(config,indent=2,sort_keys=True)+"\n")
    artifacts=[p for p in directory.iterdir() if p.name!="audit_manifest.json"]
    manifest={"artifacts":{p.name:sha256_file(p) for p in sorted(artifacts)},"artifact_count":len(artifacts),"train_only":True,"checkpoint_created":False}
    (directory/"audit_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps(flags,indent=2,sort_keys=True)); print("audit complete: {}".format(directory))


def verify_existing(cli):
    directory=output_directory(cli); manifest=json.loads((directory/"audit_manifest.json").read_text()); summary=json.loads((directory/"audit_summary.json").read_text())
    for name,expected in manifest["artifacts"].items():
        path=directory/name
        if not path.is_file() or sha256_file(path)!=expected: raise ValueError("Artifact SHA mismatch: {}".format(name))
    task=pd.concat([pd.read_csv(directory/"task_gradient_cosines_{}.csv".format(s)) for s in STATE_FILES.values()],ignore_index=True)
    kd=pd.concat([pd.read_csv(directory/"task_kd_alignment_{}.csv".format(s)) for s in STATE_FILES.values()],ignore_index=True)
    flags=classify_conflicts(task,kd); existing=summary["GradientConflictFlags"]
    for key in ("shared_fusion_conflict","prediction_head_only_conflict","cfkd_task_conflict","single_mode_bottleneck","bottleneck_mode","no_systematic_conflict","MostConflictingModePair","MostConflictingParameterGroup"):
        if flags[key]!=existing[key]: raise ValueError("Classification is not reproducible: {}".format(key))
    samples=pd.read_csv(directory/"audit_sample_manifest.csv")
    if len(samples)!=1284 or samples.sample_index.nunique()!=1284: raise ValueError("Sample manifest invalid.")
    expected={"task_gradient_cosines":6*8*len(pd.unique(samples.Batch)),"gradient_transfer":8*16*len(pd.unique(samples.Batch))}
    if len(task[task.RowType.eq("batch")])!=2*expected["task_gradient_cosines"]: raise ValueError("Task cosine row count invalid.")
    transfer=pd.concat([pd.read_csv(directory/"gradient_transfer_{}.csv".format(s)) for s in STATE_FILES.values()],ignore_index=True)
    quartile=pd.concat([pd.read_csv(directory/"compatibility_quartile_gradient_{}.csv".format(s)) for s in STATE_FILES.values()],ignore_index=True)
    norms=pd.concat([pd.read_csv(directory/"gradient_norms_{}.csv".format(s)) for s in STATE_FILES.values()],ignore_index=True)
    reps=pd.concat([pd.read_csv(directory/"representation_alignment_{}.csv".format(s)).assign(State=state) for state,s in STATE_FILES.items()],ignore_index=True)
    qcurrent=quartile[(quartile.State=="cfcompat_best_valid")&(quartile.RowType=="summary")]
    if len(qcurrent):
        qmost=qcurrent.sort_values(["median","negative_fraction"],ascending=[True,False]).iloc[0]
        flags_out=summary["GradientConflictFlags"]
        flags_out["OverallTaskKDMostConflictingMode"]=flags_out.get("OverallTaskKDMostConflictingMode",flags_out["TaskKDMostConflictingMode"])
        flags_out["TaskKDMostConflictingMode"]=qmost.Mode; flags_out["TaskKDMostConflictingQuartile"]=qmost.Quartile
        flags_out["TaskKDMostConflictingParameterGroup"]=qmost.ParameterGroup
    summary["ImplementationCommit"]=git("rev-parse","HEAD")
    write_report(directory,summary,task,kd,quartile,norms,reps)
    (directory/"audit_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    artifacts=[p for p in directory.iterdir() if p.name!="audit_manifest.json"]
    refreshed={"artifacts":{p.name:sha256_file(p) for p in sorted(artifacts)},"artifact_count":len(artifacts),"train_only":True,"checkpoint_created":False}
    (directory/"audit_manifest.json").write_text(json.dumps(refreshed,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"verified":True,"artifact_count":refreshed["artifact_count"],"train_samples":len(samples),"flags":summary["GradientConflictFlags"],"implementation_commit":summary["ImplementationCommit"]},indent=2,sort_keys=True))


def main():
    cli=parse_args()
    if cli.summary_only: verify_existing(cli)
    else: run_full(cli)


if __name__=="__main__": main()
