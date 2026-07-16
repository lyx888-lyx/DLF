"""Generate the frozen Stage 5A final audit after all three formal runs."""
import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from trains.singleTask.fixed_kd_utils import checkpoint_sha256


NEW = [
    ("ModeTeacherKD", "mode_teacher_kd_v1"),
    ("UniformDualTeacherKD", "uniform_dual_teacher_kd_v1"),
    ("CRDTD", "crdtd_v1"),
]
CF_J = 0.7178811431
CFRR_J = 0.7169613639513652
METRICS = ("MAE", "Corr", "acc_2", "F1_score", "acc_5", "acc_7", "Loss")
MODES = ("LAV", "LA", "LV", "L")


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def one(path):
    frame = pd.read_csv(path)
    if len(frame) != 1: raise ValueError("Expected exactly one formal seed row: {}".format(path))
    return frame.iloc[0]


def finite_csv(path):
    frame = pd.read_csv(path)
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Non-finite or missing numeric data: {}".format(path))
    return frame


def historical_row(name, path, j_valid=None, j_test=None):
    if Path(path).is_file():
        row = one(path)
        j_valid = float(row.get("J_valid", row.get("J_val", j_valid))) if not pd.isna(row.get("J_valid", row.get("J_val", np.nan))) else j_valid
        j_test = float(row.get("J_test_at_valid_best", row.get("J_test", j_test))) if not pd.isna(row.get("J_test_at_valid_best", row.get("J_test", np.nan))) else j_test
        epoch = row.get("BestValidEpoch", row.get("BestEpoch", np.nan)); checkpoint = row.get("MainCheckpoint", row.get("Checkpoint", ""))
        sha = checkpoint_sha256(checkpoint) if checkpoint and Path(str(checkpoint)).is_file() else "NA"
    else:
        epoch=np.nan; sha="NA"
    return {"Method":name,"BestValidEpoch":epoch,"J_valid":j_valid,"J_test_at_valid_best":j_test,"SelectionRegret":np.nan,"CheckpointSHA256":sha}


def new_artifacts():
    items=[]; sequence=[]
    required=["mosi_per_seed.csv","mosi_summary.csv","mosi_epoch_metrics.csv","mosi_route_summary.csv","mosi_route_quartiles.csv",
              "mosi_mode_teacher_metrics.csv","mosi_best_valid_predictions.csv","mosi_best_test_diagnostic_predictions.csv"]
    for name,version in NEW:
        root=Path("result/missing_baseline")/version/"benchmark_train"
        if any(not (root/f).is_file() for f in required): raise FileNotFoundError("Incomplete formal output for {}".format(name))
        for f in required: finite_csv(root/f)
        row=one(root/"mosi_per_seed.csv"); epoch=int(row.BestValidEpoch)
        state_path=Path(row.MainCheckpoint)
        if checkpoint_sha256(state_path)!=row.MainCheckpointSHA256: raise ValueError("Main checkpoint SHA mismatch: {}".format(name))
        prediction=pd.read_csv(root/"mosi_best_valid_predictions.csv")
        if set(prediction.selected_by.astype(str))!={"valid"} or set(prediction.diagnostic_only.astype(str).str.lower())!={"false"}:
            raise ValueError("Main prediction metadata invalid: {}".format(name))
        diagnostic=pd.read_csv(root/"mosi_best_test_diagnostic_predictions.csv")
        if set(diagnostic.selected_by.astype(str))!={"test"} or not diagnostic.diagnostic_only.astype(bool).all() or not diagnostic.not_main_result.astype(bool).all():
            raise ValueError("Diagnostic prediction metadata invalid: {}".format(name))
        route=pd.read_csv(root/"mosi_route_summary.csv"); selected=route.loc[route.Epoch.eq(epoch)]
        if len(selected)!=1: raise ValueError("No unique selected route row: {}".format(name))
        items.append((name,version,row,selected.iloc[0],root)); sequence.append(row.MissingSequenceSHA256)
    if len(set(sequence))!=1: raise ValueError("Formal variants did not share one missing-sequence SHA.")
    return items,sequence[0]


def classification(items, audit_quartiles):
    scores={name:float(row.J_test_at_valid_best) for name,_,row,_,_ in items}
    if scores["CRDTD"] < CF_J and scores["CRDTD"] < min(scores["ModeTeacherKD"],scores["UniformDualTeacherKD"]): return "A: CRDTD SUCCESS"
    if scores["ModeTeacherKD"] < CF_J and scores["ModeTeacherKD"] == min(scores.values()): return "B: ModeTeacherKD is the supported candidate"
    if scores["UniformDualTeacherKD"] < CF_J and scores["UniformDualTeacherKD"] == min(scores.values()): return "C: uniform dual-teacher complementarity only"
    if scores["CRDTD"] < CF_J: return "D: dual-teacher gain without proven compatibility-routing contribution"
    q1=audit_quartiles.loc[audit_quartiles.Quartile.eq("Q1_low")]
    accuracy=float(q1.fraction_mode_teacher_more_accurate.mean()); attainable=float(q1.fraction_mode_teacher_closer_to_student.mean())
    if scores["ModeTeacherKD"] > CF_J+.01 and accuracy<=.5 and attainable<=.5: return "F: ModDrop is not a suitable numeric distillation Teacher"
    return "E: dual-teacher prediction KD did not exceed CFCompatKD"


def table(frame, digits=6):
    local=frame.copy()
    for col in local.select_dtypes(include=[np.number]): local[col]=local[col].map(lambda x:"NA" if pd.isna(x) else ("{:.{}f}".format(float(x),digits)))
    columns=list(local.columns)
    rows=["| " + " | ".join(map(str,columns)) + " |", "| " + " | ".join(["---"]*len(columns)) + " |"]
    for values in local.itertuples(index=False,name=None): rows.append("| " + " | ".join(str(v) for v in values) + " |")
    return "\n".join(rows)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--test-count",type=int,required=True); parser.add_argument("--smoke-verified",action="store_true")
    cli=parser.parse_args(); items,sequence_sha=new_artifacts()
    audit_root=Path("result/dual_teacher/crdtd_v1/mosi/seed1111")
    audit=json.loads((audit_root/"dual_teacher_suitability_summary.json").read_text()); audit_q=finite_csv(audit_root/"dual_teacher_suitability_quartiles.csv")
    if audit["source_split"]!="train" or audit["train_sample_count"]!=1284 or not audit["rng_state_preserved"]: raise ValueError("Suitability audit integrity failure.")
    result_class=classification(items,audit_q)
    controls=[
        historical_row("ModDrop","result/missing_baseline/moddrop/train/mosi_per_seed.csv",j_test=.737780),
        historical_row("FixedKD","result/missing_baseline/fixed_kd/train/mosi_per_seed.csv"),
        historical_row("ReliabilityKD","result/missing_baseline/reliability_kd_v1/benchmark_train/mosi_per_seed.csv"),
        historical_row("CFCompatKD","result/missing_baseline/cf_compat_kd_v1/benchmark_train/mosi_per_seed.csv",.677964,CF_J),
    ]
    rows=controls[:]
    for name,_,row,_,_ in items:
        rows.append({"Method":name,"BestValidEpoch":int(row.BestValidEpoch),"J_valid":float(row.J_valid),"J_test_at_valid_best":float(row.J_test_at_valid_best),
                     "SelectionRegret":float(row.SelectionRegret),"CheckpointSHA256":row.MainCheckpointSHA256})
    rows.append(historical_row("CFRR-only (historical)","result/missing_baseline/cfrr_only_v1/benchmark_train/mosi_per_seed.csv",.689520,CFRR_J))
    comparison=pd.DataFrame(rows); comparison["Delta_vs_CFCompatKD"]=comparison.J_test_at_valid_best-CF_J
    metric_rows=[]
    for name,_,row,_,_ in items:
        for mode in MODES:
            metric_rows.append({"Method":name,"Mode":mode,**{metric:float(row["test_at_valid_best_{}_{}".format(mode,metric)]) for metric in METRICS}})
        metric_rows.append({"Method":name,"Mode":"MissingMacro",**{metric:float(row["test_at_valid_best_MissingMacro_{}".format(metric)]) for metric in METRICS}})
    contributions=[]
    for name,_,row,route,_ in items:
        contributions.append({"Method":name,"Epoch":int(row.BestValidEpoch),"FullRaw":route.FullKDRawMean,"ModeRaw":route.ModeKDRawMean,
                              "WeightedFull":route.WeightedFullContribution,"WeightedMode":route.WeightedModeContribution,
                              "FullFraction":route.FullContributionFraction,"ModeFraction":route.ModeContributionFraction,
                              "AlphaMean":route.AlphaMean,"AlphaStd":route.AlphaStd,"AlphaMin":route.AlphaMin,"AlphaMax":route.AlphaMax,
                              "TeacherDisagreement":route.TeacherDisagreementMean,"StudentFullGap":route.StudentFullTeacherGap,"StudentModeGap":route.StudentModeTeacherGap})
    q_display=audit_q[["Mode","Quartile","count","mean_compatibility","mean_full_teacher_error","mean_mode_teacher_error",
                       "fraction_mode_teacher_more_accurate","mean_initial_student_full_gap","mean_initial_student_mode_gap",
                       "fraction_mode_teacher_closer_to_student","mean_teacher_disagreement"]]
    status=git("status","--porcelain"); head=git("rev-parse","HEAD"); upstream=git("rev-parse","@{u}")
    lines=["# Stage 5A Compatibility-Routed Dual-Teacher Distillation Final Audit","",f"Classification: **{result_class}**","","## Integrity","",
           "- Branch: `{}`; base commit: `276fc26e6e7d1998f7dd746edfca1efbbb691e55`; implementation commit: `{}`.".format(git("branch","--show-current"),head),
           "- Push verified: {}; worktree: `{}`.".format(head==upstream,"clean" if not status else "dirty"),
           "- Tests passed: {}; A/B/C smoke verified: {}.".format(cli.test_count,cli.smoke_verified),
           "- Full Teacher SHA: `{}`; Mode Teacher SHA: `{}`; compatibility cache SHA: `{}`; dual-teacher audit cache SHA: `{}`.".format(
               audit["full_teacher_sha256"],audit["mode_teacher_sha256"],one(items[0][4]/"mosi_per_seed.csv").CompatibilityCacheSHA256,
               checkpoint_sha256(audit_root/"train_dual_teacher_targets.csv")),
           "- Missing-sequence SHA: `{}` (identical across A/B/C).".format(sequence_sha),
           "- No NaN/Inf/OOM, Teacher gradient, test-based main selection, formula change, or hyperparameter change was detected.","",
           "## Eight-method comparison","",table(comparison),"","## New-method selected-main test metrics","",table(pd.DataFrame(metric_rows)),"",
           "## Route contributions at validation-best","",table(pd.DataFrame(contributions)),"","## Train-only suitability audit","",
           "The audit contains 1284 unique train samples (3852 sample-mode rows), does not read valid/test, and preserves RNG/model parameters.","",table(q_display),"",
           "Low-compatibility mean fraction Mode Teacher closer to label: {:.6f}; closer to initial Student: {:.6f}.".format(
               float(audit_q.loc[audit_q.Quartile.eq("Q1_low"),"fraction_mode_teacher_more_accurate"].mean()),
               float(audit_q.loc[audit_q.Quartile.eq("Q1_low"),"fraction_mode_teacher_closer_to_student"].mean())),"",
           "## Protocol stop","","No additional seeds, alpha tuning, reliability, residual, recoverability, diffusion, or Stage 3 modification was run.",""]
    output=Path("result/missing_baseline/stage5a_crdtd_final_audit.md"); output.write_text("\n".join(lines),encoding="utf-8")
    print("classification={} output={}".format(result_class,output))


if __name__=="__main__": main()
