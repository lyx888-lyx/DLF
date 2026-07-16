"""Pure diagnostic utilities for Stage 6A missing-pattern gradient audits."""
import hashlib
import json
import math
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch


MODES = ("LAV", "LA", "LV", "L")
MISSING_MODES = ("LA", "LV", "L")
MODE_PAIRS = (("LAV", "LA"), ("LAV", "LV"), ("LAV", "L"),
              ("LA", "LV"), ("LA", "L"), ("LV", "L"))
GROUPS = ("text_backbone_or_projection", "audio_backbone_or_projection",
          "vision_backbone_or_projection", "shared_multimodal_fusion",
          "prediction_and_task_heads", "missing_tokens_and_mask_adapter",
          "other_trainable")
ALL_GROUP = "all_shared_parameters"


def capture_rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().clone(),
            "cuda": [x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None}


def restore_rng_state(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state["cuda"])


def rng_states_equal(left, right):
    same = left["python"] == right["python"] and np.array_equal(left["numpy"][1], right["numpy"][1])
    same = same and left["numpy"][2:] == right["numpy"][2:] and torch.equal(left["torch"], right["torch"])
    if left["cuda"] is None or right["cuda"] is None: return same and left["cuda"] is right["cuda"]
    return same and len(left["cuda"]) == len(right["cuda"]) and all(torch.equal(a, b) for a, b in zip(left["cuda"], right["cuda"]))


def clone_state_dict(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def states_equal(left, right):
    return set(left) == set(right) and all(torch.equal(left[name], right[name].detach().cpu()) for name in left)


@contextmanager
def protected_audit_state(model, teacher=None):
    rng = capture_rng_state(); state = clone_state_dict(model); training = model.training
    teacher_training = teacher.training if teacher is not None else None
    try:
        model.eval()
        if teacher is not None: teacher.eval()
        yield {"rng_before": rng, "state_before": state}
    finally:
        if not states_equal(state, model.state_dict()): model.load_state_dict(state, strict=True)
        model.train(training)
        if teacher is not None: teacher.train(teacher_training)
        restore_rng_state(rng)


def parameter_group_for_name(name):
    if name in ("missing_audio_token", "missing_vision_token") or name.startswith("mask_adapter."):
        return "missing_tokens_and_mask_adapter"
    head_roots = ("proj1_c", "proj2_c", "out_layer_c", "proj1_l_low", "proj2_l_low", "out_layer_l_low",
                  "proj1_v_low", "proj2_v_low", "out_layer_v_low", "proj1_a_low", "proj2_a_low", "out_layer_a_low",
                  "proj1_l_high", "proj2_l_high", "out_layer_l_high", "proj1_v_high", "proj2_v_high", "out_layer_v_high",
                  "proj1_a_high", "proj2_a_high", "out_layer_a_high", "proj1", "proj2", "out_layer")
    root = name[len("backbone."):].split(".", 1)[0] if name.startswith("backbone.") else ""
    if root in head_roots: return "prediction_and_task_heads"
    text = ("text_model", "proj_l", "encoder_s_l", "decoder_l", "proj_cosine_l", "align_c_l")
    audio = ("proj_a", "encoder_s_a", "decoder_a", "proj_cosine_a", "align_c_a")
    vision = ("proj_v", "encoder_s_v", "decoder_v", "proj_cosine_v", "align_c_v")
    if root in text: return "text_backbone_or_projection"
    if root in audio: return "audio_backbone_or_projection"
    if root in vision: return "vision_backbone_or_projection"
    if root == "encoder_c" or root.startswith("self_attentions_c_") or root.startswith("trans_") or root.startswith("projector_"):
        return "shared_multimodal_fusion"
    return "other_trainable"


def build_parameter_groups(model):
    names=[]; params=[]; group_indices={group: [] for group in GROUPS}; rows=[]; seen=set()
    for index, (name, parameter) in enumerate(model.named_parameters()):
        if not parameter.requires_grad: continue
        if id(parameter) in seen: raise ValueError("Trainable parameter appears twice: {}".format(name))
        seen.add(id(parameter)); group=parameter_group_for_name(name)
        names.append(name); params.append(parameter); group_indices[group].append(index)
        rows.append({"ParameterName": name, "Group": group, "Shape": "x".join(map(str, parameter.shape)),
                     "Numel": int(parameter.numel()), "Trainable": True})
    if len(params) != sum(len(v) for v in group_indices.values()): raise RuntimeError("Parameter groups are not exhaustive.")
    all_indices=list(range(len(params))); group_indices[ALL_GROUP]=all_indices
    total=sum(p.numel() for p in params); other=sum(params[i].numel() for i in group_indices["other_trainable"])
    manifest={"groups": {}, "trainable_parameter_count": len(params), "trainable_numel": total,
              "other_trainable_numel_fraction": other/total if total else 0.0, "mapping_complete": other/total <= .05}
    for group in GROUPS:
        idx=group_indices[group]; manifest["groups"][group]={"parameter_count":len(idx),"trainable_numel":sum(params[i].numel() for i in idx),
                                                              "parameter_names":[names[i] for i in idx]}
    if not manifest["mapping_complete"]: raise RuntimeError("other_trainable exceeds 5% of trainable parameters.")
    return names, params, group_indices, pd.DataFrame(rows), manifest


def ordered_gradients(loss, parameters, retain_graph=True):
    grads=torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    for grad in grads:
        if grad is not None and not torch.isfinite(grad).all(): raise FloatingPointError("Non-finite gradient detected.")
    return grads


def add_gradients(left, right):
    result=[]
    for a,b in zip(left,right):
        if a is None and b is None: result.append(None)
        elif a is None: result.append(b)
        elif b is None: result.append(a)
        else: result.append(a+b)
    return tuple(result)


def flatten_group_gradient(grads, parameters, indices):
    chunks=[]
    for index in indices:
        grad=grads[index]
        chunks.append(torch.zeros_like(parameters[index]).reshape(-1) if grad is None else grad.reshape(-1))
    return torch.cat(chunks) if chunks else torch.empty(0, device=parameters[0].device if parameters else "cpu")


def group_gradient_stats(grads, parameters, indices):
    dimension=sum(parameters[i].numel() for i in indices); norm_sq=0.0; zero=0
    for i in indices:
        grad=grads[i]
        if grad is None: zero += parameters[i].numel()
        else:
            norm_sq += float(torch.sum(grad.detach().double()*grad.detach().double()).cpu())
            zero += int(torch.count_nonzero(grad.detach()==0).cpu())
    return {"GradientNorm": math.sqrt(norm_sq), "ZeroGradientFraction": zero/dimension if dimension else float("nan"),
            "Finite": True, "VectorDimensionality": int(dimension)}


def gradient_dot(left, right, indices):
    total=0.0
    for i in indices:
        if left[i] is not None and right[i] is not None:
            total += float(torch.sum(left[i].detach().double()*right[i].detach().double()).cpu())
    return total


def gradient_cosine(left, right, parameters, indices):
    a=group_gradient_stats(left,parameters,indices); b=group_gradient_stats(right,parameters,indices)
    if a["GradientNorm"] < 1e-12 or b["GradientNorm"] < 1e-12: return float("nan")
    return gradient_dot(left,right,indices)/(a["GradientNorm"]*b["GradientNorm"])


def bootstrap_intervals(values, samples=2000, seed=260616):
    x=np.asarray(values,dtype=float); x=x[np.isfinite(x)]
    if x.size==0: return {"bootstrap_mean_ci_low":float("nan"),"bootstrap_mean_ci_high":float("nan"),
                          "bootstrap_median_ci_low":float("nan"),"bootstrap_median_ci_high":float("nan")}
    rng=np.random.RandomState(seed); means=np.empty(samples); medians=np.empty(samples)
    for i in range(samples):
        draw=x[rng.randint(0,len(x),len(x))]; means[i]=draw.mean(); medians[i]=np.median(draw)
    return {"bootstrap_mean_ci_low":float(np.quantile(means,.025)),"bootstrap_mean_ci_high":float(np.quantile(means,.975)),
            "bootstrap_median_ci_low":float(np.quantile(medians,.025)),"bootstrap_median_ci_high":float(np.quantile(medians,.975))}


def stable_seed(*parts):
    return int(hashlib.sha256("|".join(map(str,parts)).encode()).hexdigest()[:8],16)


def cosine_summary(values, bootstrap_samples=2000, seed=260616):
    x=np.asarray(values,dtype=float); x=x[np.isfinite(x)]
    if x.size==0:
        result={key:float("nan") for key in ("mean","std","median","min","p10","p25","p75","p90","max","negative_fraction","strong_negative_fraction","positive_fraction")}; result["count"]=0
    else:
        result={"count":int(x.size),"mean":float(x.mean()),"std":float(x.std()),"median":float(np.median(x)),"min":float(x.min()),
                "p10":float(np.quantile(x,.1)),"p25":float(np.quantile(x,.25)),"p75":float(np.quantile(x,.75)),
                "p90":float(np.quantile(x,.9)),"max":float(x.max()),"negative_fraction":float((x<0).mean()),
                "strong_negative_fraction":float((x<-.2).mean()),"positive_fraction":float((x>0).mean())}
    result.update(bootstrap_intervals(x,bootstrap_samples,seed)); return result


def assign_compatibility_quartiles(frame):
    result={}
    for mode in MISSING_MODES:
        ordered=frame.sort_values(["compat_{}".format(mode),"sample_index"],kind="mergesort")
        labels=pd.qcut(np.arange(len(ordered)),4,labels=["Q1_low","Q2","Q3","Q4_high"])
        for index,label in zip(ordered.sample_index.astype(int),labels): result[(int(index),mode)]=str(label)
    if len(result)!=len(frame)*3: raise RuntimeError("Quartile assignment is incomplete.")
    return result


def linear_cka(x, y):
    x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
    x=x-x.mean(0,keepdims=True); y=y-y.mean(0,keepdims=True)
    cross=x.T@y; numerator=np.sum(cross*cross); denominator=math.sqrt(np.sum((x.T@x)**2)*np.sum((y.T@y)**2))
    return float(numerator/denominator) if denominator>0 else float("nan")


def effective_rank(matrix):
    singular=np.linalg.svd(np.asarray(matrix,dtype=np.float64)-np.mean(matrix,axis=0,keepdims=True),compute_uv=False)
    total=singular.sum()
    if total<=0: return 0.0
    p=singular[singular>0]/total
    return float(np.exp(-np.sum(p*np.log(p))))


def representation_statistics(representations):
    pairs=[]; summaries={}
    for mode,matrix in representations.items():
        x=np.asarray(matrix,dtype=np.float64); norms=np.linalg.norm(x,axis=1); gram=x@x.T; denom=np.outer(norms,norms)
        upper=np.triu_indices(len(x),1); pairwise=np.divide(gram[upper],denom[upper],out=np.full(len(upper[0]),np.nan),where=denom[upper]>0)
        summaries[mode]={"sample_count":len(x),"feature_dim":x.shape[1],"norm_mean":float(norms.mean()),"norm_std":float(norms.std()),
                         "norm_median":float(np.median(norms)),"norm_p10":float(np.quantile(norms,.1)),"norm_p90":float(np.quantile(norms,.9)),
                         "feature_variance_mean":float(np.var(x,axis=0).mean()),"effective_rank":effective_rank(x),
                         "mean_pairwise_sample_cosine":float(np.nanmean(pairwise))}
    for a,b in MODE_PAIRS:
        x=np.asarray(representations[a],dtype=np.float64); y=np.asarray(representations[b],dtype=np.float64)
        denom=np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1)
        cos=np.divide(np.sum(x*y,axis=1),denom,out=np.full(len(x),np.nan),where=denom>0)
        pairs.append({"ModeA":a,"ModeB":b,"SampleCount":len(x),"SameSampleCosineMean":float(np.nanmean(cos)),
                      "SameSampleCosineStd":float(np.nanstd(cos)),"SameSampleCosineMedian":float(np.nanmedian(cos)),
                      "LinearCKA":linear_cka(x,y),"MeanRepresentationDisplacement":float(np.linalg.norm(x-y,axis=1).mean())})
    return pd.DataFrame(pairs), summaries


def classify_conflicts(task_summary, kd_summary):
    current=task_summary[(task_summary.State=="cfcompat_best_valid")&(task_summary.RowType=="summary")]
    def g1(group):
        local=current[current.ParameterGroup==group]
        condition_a=((local["median"]<0)&(local.negative_fraction>=.60)).sum()>=2
        lav=local[local.ModeA.eq("LAV")]
        condition_b=(lav.bootstrap_mean_ci_high<0).any()
        return bool(condition_a or condition_b)
    fusion=g1("shared_multimodal_fusion"); head=g1("prediction_and_task_heads")
    kd_current=kd_summary[(kd_summary.State=="cfcompat_best_valid")&(kd_summary.RowType=="summary")&(kd_summary.AlignmentType=="missing_task_vs_kd")]
    kd_conflict=bool(((kd_current["median"]<0)&(kd_current.negative_fraction>=.60)).any())
    missing_pairs=current[current.ModeA.isin(MISSING_MODES)&current.ModeB.isin(MISSING_MODES)]
    bottleneck=None
    for mode in MISSING_MODES:
        involving=current[((current.ModeA==mode)|(current.ModeB==mode))&~((current.ModeA=="LAV")&(current.ModeB==mode))]
        other=missing_pairs[(missing_pairs.ModeA!=mode)&(missing_pairs.ModeB!=mode)]
        if (involving.negative_fraction>=.60).sum()>=2 and not (other.negative_fraction>=.60).any(): bottleneck=mode; break
    major=current[current.ParameterGroup.isin(["shared_multimodal_fusion","prediction_and_task_heads",ALL_GROUP])]
    no_systematic=bool(not fusion and not head and ((major["median"]>=0)|(major.negative_fraction<.40)).all())
    most=current.sort_values(["median","negative_fraction"],ascending=[True,False]).iloc[0]
    kd_most=kd_current.sort_values(["median","negative_fraction"],ascending=[True,False]).iloc[0] if len(kd_current) else None
    return {"shared_fusion_conflict":fusion,"prediction_head_only_conflict":bool(head and not fusion),"cfkd_task_conflict":kd_conflict,
            "single_mode_bottleneck":bottleneck is not None,"bottleneck_mode":bottleneck,"no_systematic_conflict":no_systematic,
            "MostConflictingModePair":"{}-{}".format(most.ModeA,most.ModeB),"MostConflictingParameterGroup":most.ParameterGroup,
            "LowestMedianCosine":float(most["median"]),"HighestNegativeFraction":float(current.negative_fraction.max()),
            "TaskKDMostConflictingMode":None if kd_most is None else kd_most.Mode,
            "TaskKDMostConflictingQuartile":None}


def sha256_file(path):
    digest=hashlib.sha256()
    with open(path,"rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()
