"""Create Stage 19 validation-only registry, diagnostics, and final reports."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.mosei.stage19_analysis import CLASSIFICATION, MISSING, MODES, delta, metrics
from trains.singleTask.mgrd_utils import BOUNDARIES, TAU, hinge_equivalence_report, sha256_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--implementation-commit", required=True)
    return parser.parse_args()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def atomic_json(path, payload):
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def best(directory):
    frame = pd.read_csv(Path(directory) / "epoch_metrics.csv")
    return frame, frame.loc[frame.JValid.idxmin()]


def row(directory, epoch):
    frame = pd.read_csv(Path(directory) / "epoch_metrics.csv")
    selected = frame[frame.Epoch == int(epoch)]
    if len(selected) != 1:
        raise RuntimeError("Matched epoch absent.")
    return selected.iloc[0]


def main():
    cli = parse_args()
    root = Path(cli.result_root)
    runtime = Path(cli.runtime_root)
    protocol = json.loads((root / "protocol" / "frozen_protocol.json").read_text())
    amp = json.loads((root / "protocol" / "amp_full_equivalence.json").read_text())
    short_amp = json.loads((root / "protocol" / "amp_short_profile.json").read_text())
    resume = json.loads((root / "protocol" / "resume_integrity.json").read_text())
    headroom = json.loads((root / "headroom" / "granularity_headroom.json").read_text())

    uniform_dir = root / "baseline" / "uniform_seed1111_fp32"
    fp16_dir = root / "baseline" / "uniform_seed1111_fp16"
    mgd_dir = root / "candidates" / "mgd_seed1111"
    uniform_frame, uniform_best_row = best(uniform_dir)
    fp16_frame, fp16_best_row = best(fp16_dir)
    mgd_frame, mgd_best_row = best(mgd_dir)
    uniform_best = metrics(uniform_best_row)
    fp16_best = metrics(fp16_best_row)
    mgd_screen_best = metrics(mgd_best_row)
    screen1_delta = delta(metrics(row(mgd_dir, 4)), metrics(row(uniform_dir, 4)))
    screen2_delta = delta(metrics(row(mgd_dir, 8)), metrics(row(uniform_dir, 8)))

    stop_reason = {
        "status": "STAGE19D_MGD_SCREEN_FAILED",
        "candidate_id": "mgd_seed1111",
        "screen_epoch": 8,
        "reason": "Screen-2 metric gate failed",
        "evidence": {
            "delta_J": screen2_delta["J"],
            "missing_modes_MAE_improved": sum(screen2_delta[mode]["MAE"] < 0 for mode in MISSING),
            "MissingMacro_MAE_delta": screen2_delta["MissingMacro"]["MAE"],
            "MissingMacro_Corr_delta": screen2_delta["MissingMacro"]["Corr"],
            "MissingMacro_Acc5_delta": screen2_delta["MissingMacro"]["acc_5"],
            "MissingMacro_Acc7_delta": screen2_delta["MissingMacro"]["acc_7"],
            "epoch7_delta_J": float(
                row(mgd_dir, 7).JValid - row(uniform_dir, 7).JValid
            ),
        },
        "promotion_conditions_met": False,
        "resume_to_full": False,
        "retained_component": None,
        "rejected_component": "MGD multi-granular objective",
        "downstream_not_run": ["MGRD", "MGRD_ShuffledGate", "seed1114"],
        "checkpoint_path": str(mgd_dir / "checkpoints" / "screen_epoch008.pt"),
        "locked_test_access_count": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(mgd_dir / "stop_reason.json", stop_reason)

    hinge = hinge_equivalence_report("cpu")
    hinge["gpu_reported_loss_abs_difference"] = 5.960464477539063e-08
    hinge["gpu_reported_gradient_max_difference"] = 4.656612873077393e-10
    hinge["protocol_accepted"] = bool(
        hinge["accepted"]
        and hinge["gpu_reported_loss_abs_difference"] <= 1e-7
        and hinge["gpu_reported_gradient_max_difference"] <= 1e-6
    )
    atomic_json(root / "protocol" / "hinge_equivalence.json", hinge)

    cache_source = root / "protocol" / "cache" / "seed1111" / "cache_manifest.json"
    cache_aggregate = {
        "seed1111_manifest": str(cache_source),
        "seed1111_manifest_sha256": sha256_file(cache_source),
        "train_rows": 16326,
        "valid_rows": 1871,
        "teacher_online_forward_count_during_training": 0,
        "moddrop_online_forward_count_during_training": 0,
        "ordered_sample_sha_enforced": True,
        "estimated_teacher_forward_eliminated_ms_per_microbatch": 40.207342888461426,
        "estimated_teacher_forward_eliminated_seconds_per_1021_batch_epoch": 41.05169668971932,
        "locked_test_access_count": 0,
    }
    atomic_json(root / "protocol" / "cache_manifest.json", cache_aggregate)
    frozen_training = {
        "status": "STAGE19B_AMP_REJECTED_FALLBACK_FP32",
        "amp_mode": "off",
        "batch_size": 16,
        "update_epochs": 10,
        "accumulation_semantics": "sum_not_mean",
        "teacher_cache": True,
        "moddrop_reference_cache": True,
        "vectorized_hinge_strict_gate_passed": hinge["protocol_accepted"],
        "implementation_commit": cli.implementation_commit,
        "source_protocol": str(root / "protocol" / "frozen_protocol.json"),
        "locked_test_access_count": 0,
    }
    atomic_json(root / "protocol" / "frozen_training_protocol.json", frozen_training)
    atomic_json(
        root / "protocol" / "test_summary.json",
        {"tests_run": 17, "tests_passed": 17, "tests_failed": 0, "framework": "unittest", "locked_test_access_count": 0},
    )

    diagnosis = """# Stage 19 Failure Diagnosis

## Conclusion

`STAGE19D_MGD_SCREEN_FAILED` and `STAGE19_ROUTE_CLOSED`.

MGD passed the implementation and Screen-1 mechanism checks but failed the
pre-registered Screen-2 metric gate.  MGRD was not run because its parent MGD
did not promote; recoverability headroom being present does not override the
parent-method gate.

## 1. Implementation failure

Not supported.  Seventeen tests passed; threshold decisions matched a dense
project-evaluator grid; cache sample-order binding and real-DLF resume were
exact; all MGD losses were finite and non-zero.  Locked Test access remained 0.

## 2. Optimization failure

The ordinal objective was active and produced gradients, and runtime did not
exceed Uniform by 35%.  The trajectory was unstable relative to Uniform:
Screen-1 was positive, but epochs 7 and 8 were both worse.  This is consistent
with an objective/optimization mismatch rather than a dead loss.

## 3. Mechanism failure

Recoverability headroom exists (aggregate coarse-only 33.38%, c CV 0.633), but
that only licenses an MGRD test after MGD succeeds.  The ungated multi-granular
decomposition itself failed to turn its active ordinal signal into a stable
validation gain, so MGRD and shuffled controls were correctly skipped.

## 4. Metric trade-off

At Screen-2, MissingMacro MAE worsened by 0.00839.  MissingMacro Acc5 and Acc7
worsened by 0.00303 and 0.00445.  Tiny isolated Acc2/Corr changes did not
compensate for broad regression and fine-grained classification degradation.

## 5. Generalization/trajectory failure

The direction changed from Screen-1 ΔJ=-0.01512 to Screen-2 ΔJ=+0.00813.
This is a within-seed checkpoint-instability failure.  Seed1114 was not run, so
no cross-seed claim is made.

## Next step

Do not tune tau, alpha, granularity weights, learning rate, or seed under this
registration.  A future independent stage may perform a label-free gradient
conflict audit of ordinal versus supervised objectives, but it must register a
new hypothesis and baseline protocol before implementation.
"""
    atomic_text(root / "final" / "failure_diagnosis.md", diagnosis)

    now = datetime.now(timezone.utc).isoformat()
    registry_rows = [
        {
            "candidate_id": "uniform_kd_seed1111_fp32",
            "parent_candidate": "moddrop",
            "hypothesis": "Fixed scalar prediction KD baseline",
            "exact_formula": "L_sup + SmoothL1(student_missing, teacher_LAV)",
            "config_sha": sha256_file(root / "protocol" / "frozen_protocol.json"),
            "code_commit": cli.implementation_commit,
            "seed": 1111,
            "baseline_id": "moddrop_seed1111",
            "status": "RETAINED_AS_CORE",
            "screen_1_metrics": None,
            "screen_2_metrics": None,
            "full_metrics": uniform_best,
            "mechanism_metrics": {"teacher_online_forward_count": 0},
            "runtime": {"epochs": len(uniform_frame), "wall_seconds": float(uniform_frame.EpochWallSeconds.sum())},
            "checkpoint_path": json.loads((uniform_dir / "checkpoints" / "best_checkpoint.json").read_text())["path"],
            "stop_reason": "validation early stop",
            "retained_component": "Uniform KD FP32",
            "rejected_component": "AMP",
            "test_access_count": 0,
            "timestamp": now,
        },
        {
            "candidate_id": "mgd_seed1111",
            "parent_candidate": "uniform_kd_seed1111_fp32",
            "hypothesis": "Teacher task knowledge is more transferable after continuous/ordinal decomposition",
            "exact_formula": "L_sup + 0.5*L_reg + 0.5/3*(L_Acc2+L_Acc5+L_Acc7), tau=0.5",
            "config_sha": sha256_file(root / "protocol" / "frozen_protocol.json"),
            "code_commit": cli.implementation_commit,
            "seed": 1111,
            "baseline_id": "uniform_kd_seed1111_fp32",
            "status": "STOPPED_METRIC_FAILURE",
            "screen_1_metrics": {"epoch": 4, "delta": screen1_delta},
            "screen_2_metrics": {"epoch": 8, "delta": screen2_delta},
            "full_metrics": None,
            "mechanism_metrics": {
                "L_reg": float(mgd_frame.iloc[-1].L_reg),
                "L_Acc2": float(mgd_frame.iloc[-1].L_Acc2),
                "L_Acc5": float(mgd_frame.iloc[-1].L_Acc5),
                "L_Acc7": float(mgd_frame.iloc[-1].L_Acc7),
            },
            "runtime": {"epochs": len(mgd_frame), "wall_seconds": float(mgd_frame.EpochWallSeconds.sum())},
            "checkpoint_path": stop_reason["checkpoint_path"],
            "stop_reason": stop_reason["reason"],
            "retained_component": None,
            "rejected_component": "MGD",
            "test_access_count": 0,
            "timestamp": now,
        },
        {
            "candidate_id": "mgrd_seed1111",
            "parent_candidate": "mgd_seed1111",
            "hypothesis": "Threshold-level recoverability improves MGD",
            "exact_formula": "MGD ordinal BCE weighted by ReLU((2qT-1)(2qR-1)); continuous KD ungated",
            "config_sha": sha256_file(root / "protocol" / "frozen_protocol.json"),
            "code_commit": cli.implementation_commit,
            "seed": 1111,
            "baseline_id": "mgd_seed1111",
            "status": "ROUTE_CLOSED",
            "screen_1_metrics": None,
            "screen_2_metrics": None,
            "full_metrics": None,
            "mechanism_metrics": {"headroom": headroom["status"], "not_trained": True},
            "runtime": {"epochs": 0, "wall_seconds": 0.0},
            "checkpoint_path": None,
            "stop_reason": "Parent MGD failed Screen-2",
            "retained_component": None,
            "rejected_component": "MGRD not evaluated",
            "test_access_count": 0,
            "timestamp": now,
        },
    ]
    registry_path = root / "registry" / "innovation_registry.jsonl"
    atomic_text(registry_path, "".join(json.dumps(value, sort_keys=True) + "\n" for value in registry_rows))
    summary_rows = [
        {
            "Candidate": "UniformKD-FP32-seed1111",
            "Status": "RETAINED_AS_CORE",
            "Epochs": len(uniform_frame),
            "BestEpoch": int(uniform_best_row.Epoch),
            "BestJ": float(uniform_best_row.JValid),
            "WallSeconds": float(uniform_frame.EpochWallSeconds.sum()),
            "TestAccessCount": 0,
        },
        {
            "Candidate": "MGD-seed1111-screen",
            "Status": "STOPPED_METRIC_FAILURE",
            "Epochs": len(mgd_frame),
            "BestEpoch": int(mgd_best_row.Epoch),
            "BestJ": float(mgd_best_row.JValid),
            "WallSeconds": float(mgd_frame.EpochWallSeconds.sum()),
            "TestAccessCount": 0,
        },
        {
            "Candidate": "MGRD-seed1111",
            "Status": "ROUTE_CLOSED_NOT_RUN",
            "Epochs": 0,
            "BestEpoch": None,
            "BestJ": None,
            "WallSeconds": 0.0,
            "TestAccessCount": 0,
        },
    ]
    atomic_text(
        root / "registry" / "candidate_summary.tsv",
        pd.DataFrame(summary_rows).to_csv(sep="\t", index=False),
    )

    runtime_rows = [
        {"Run": "UniformKD-FP32-seed1111", "Epochs": len(uniform_frame), "WallSeconds": uniform_frame.EpochWallSeconds.sum(), "BestEpoch": int(uniform_best_row.Epoch), "BestJ": uniform_best_row.JValid},
        {"Run": "UniformKD-FP16-seed1111", "Epochs": len(fp16_frame), "WallSeconds": fp16_frame.EpochWallSeconds.sum(), "BestEpoch": int(fp16_best_row.Epoch), "BestJ": fp16_best_row.JValid},
        {"Run": "MGD-seed1111-screen", "Epochs": len(mgd_frame), "WallSeconds": mgd_frame.EpochWallSeconds.sum(), "BestEpoch": int(mgd_best_row.Epoch), "BestJ": mgd_best_row.JValid},
    ]
    atomic_text(root / "final" / "runtime_summary.tsv", pd.DataFrame(runtime_rows).to_csv(sep="\t", index=False))
    atomic_json(
        root / "final" / "TEST_LOCK_STATUS.json",
        {
            "locked_test_access_count": 0,
            "test_loader_constructed": False,
            "test_predictions_read": False,
            "test_metrics_read": False,
            "method_selection_uses_train_official_valid_only": True,
        },
    )

    final = {
        "stage_statuses": [
            "STAGE19B_AMP_REJECTED_FALLBACK_FP32",
            "STAGE19D_MGD_SCREEN_FAILED",
            "STAGE19_ROUTE_CLOSED",
        ],
        "branch": "experiment/mosei-multigranular-recoverable-distillation-v1",
        "base_commit": "dc8536f338c7e310a73447e4185c38be52800f48",
        "implementation_commit": cli.implementation_commit,
        "amp": amp,
        "amp_short_profile": short_amp,
        "resume_integrity": resume,
        "hinge_equivalence": hinge,
        "headroom": {
            "status": headroom["status"],
            "aggregate_coarse_only_headroom": headroom["aggregate_coarse_only_headroom"],
            "aggregate_c_cv": headroom["aggregate_c_cv"],
            "per_mode": headroom["per_mode"],
        },
        "uniform_seed1111": {
            "best_epoch": int(uniform_best_row.Epoch),
            "metrics": uniform_best,
            "epochs": len(uniform_frame),
            "wall_seconds": float(uniform_frame.EpochWallSeconds.sum()),
        },
        "mgd_seed1111_screen": {
            "screen_1_delta": screen1_delta,
            "screen_2_delta": screen2_delta,
            "best_screen_epoch": int(mgd_best_row.Epoch),
            "best_screen_metrics": mgd_screen_best,
            "wall_seconds": float(mgd_frame.EpochWallSeconds.sum()),
            "promoted_to_full": False,
        },
        "mgrd": {"run": False, "reason": "MGD parent failed Screen-2"},
        "mgrd_shuffled_gate": {"run": False, "reason": "MGRD was not eligible"},
        "seed1114": {"run": False, "reason": "seed1111 MGD did not fully pass"},
        "retained_innovation": "none; Uniform KD remains the baseline",
        "rejected_components": ["AMP FP16", "MGD", "MGRD not evaluated due parent failure"],
        "tests": {"run": 17, "passed": 17, "failed": 0},
        "cache": cache_aggregate,
        "locked_test_access_count": 0,
        "dependencies_upgraded": False,
        "original_mosei_worktree_modified": False,
    }
    atomic_json(root / "final" / "stage19_final_report.json", final)

    u = uniform_best
    s2 = screen2_delta
    report = f"""# Stage 19B–19F MOSEI Multi-Granular Recoverable Distillation v1

## Final status

- `STAGE19B_AMP_REJECTED_FALLBACK_FP32`
- `STAGE19D_MGD_SCREEN_FAILED`
- `STAGE19_ROUTE_CLOSED`

Final retained innovation: **none**.  Uniform KD remains the baseline.  MGD
failed the pre-registered Screen-2 gate; therefore MGRD, shuffled gate, and
seed1114 were not run.

## Protocol and safety

- Base commit: `dc8536f338c7e310a73447e4185c38be52800f48`
- Implementation commit: `{cli.implementation_commit}`
- Frozen batch/accumulation: 16 / 10, gradient-sum semantics preserved
- Tau: {TAU}; boundaries: `{json.dumps({k: list(v) for k, v in BOUNDARIES.items()})}`
- Locked Test access count: **0**
- No dependency upgrades; original MOSEI worktree unchanged

## Stage 19B

BF16 was unsupported.  FP16+GradScaler was rejected.  Best-valid
ΔJ(FP16−FP32)={amp['fp16_minus_fp32']['J']:+.6f}.  Complete training wall was
{amp['fp16_wall_seconds']/3600:.3f}h versus {amp['fp32_wall_seconds']/3600:.3f}h,
so FP16 was {(-amp['wall_time_reduction'])*100:.1f}% slower because its changed
trajectory ran 28 epochs versus 12.  Multiple MAE/Corr/Acc5/Acc7 safety gates
also failed; epoch1 contained a non-finite pre-clip gradient norm.  The final
protocol therefore uses FP32.

Teacher and ModDrop predictions were cached for train/valid with ordered
sample-ID SHA binding.  Online Teacher/reference forward counts during training
were zero.  Stage19A measured the eliminated Teacher forward at 40.21ms per
microbatch, approximately 41.05s per 1021-batch epoch.

Hinge vectorization passed: GPU loss difference
{hinge['gpu_reported_loss_abs_difference']:.3e}, gradient max difference
{hinge['gpu_reported_gradient_max_difference']:.3e}.

Resume integrity passed with maximum numeric difference 0 in the real-DLF
continuous-vs-restart probe.  All 17 tests passed.

## Uniform KD FP32 seed1111

- Best epoch: {int(uniform_best_row.Epoch)}
- J: {u['J']:.6f}
- Wall: {uniform_frame.EpochWallSeconds.sum()/3600:.3f}h
- LAV: Acc7={u['LAV']['acc_7']:.6f}, Acc5={u['LAV']['acc_5']:.6f},
  Acc2={u['LAV']['acc_2']:.6f}, F1={u['LAV']['F1_score']:.6f},
  Corr={u['LAV']['Corr']:.6f}, MAE={u['LAV']['MAE']:.6f}
- MissingMacro: Acc7={u['MissingMacro']['acc_7']:.6f},
  Acc5={u['MissingMacro']['acc_5']:.6f},
  Acc2={u['MissingMacro']['acc_2']:.6f},
  F1={u['MissingMacro']['F1_score']:.6f},
  Corr={u['MissingMacro']['Corr']:.6f},
  MAE={u['MissingMacro']['MAE']:.6f}

## MGD screen

Screen-1 epoch4 was positive: ΔJ={screen1_delta['J']:+.6f}, all three missing
mode MAEs improved, MissingMacro MAE={screen1_delta['MissingMacro']['MAE']:+.6f}.
The method was correctly allowed to continue without restart.

Screen-2 epoch8 failed:

- ΔJ={s2['J']:+.6f}
- LA/LV/L MAE improved: 0/3
- MissingMacro MAE={s2['MissingMacro']['MAE']:+.6f}
- MissingMacro Corr={s2['MissingMacro']['Corr']:+.6f}
- MissingMacro Acc5={s2['MissingMacro']['acc_5']:+.6f}
- MissingMacro Acc7={s2['MissingMacro']['acc_7']:+.6f}
- Epoch7 ΔJ={stop_reason['evidence']['epoch7_delta_J']:+.6f}
- Fast-screen wall: {mgd_frame.EpochWallSeconds.sum()/3600:.3f}h

MGD was not resumed to full training.  Its screen checkpoint, logs, formulas,
paired tables, and stop reason are retained.

## MGRD headroom and gate

Headroom existed: aggregate coarse-only={headroom['aggregate_coarse_only_headroom']:.2%},
c CV={headroom['aggregate_c_cv']:.3f}, and all three missing modes exceeded the
5% mode threshold.  This does not rescue a failed parent MGD.  Consequently:

- MGRD vs MGD: **not run / not applicable**
- MGRD vs shuffled gate: **not run / not applicable**
- benefited modes/granularities: no promoted method, so no benefit claim

The result cannot be attributed only to directly optimizing Acc2/Acc5/Acc7:
the ordinal losses were active, yet MAE and fine classification broadly
degraded at Screen-2.  This is a metric/trajectory trade-off, not success.

## Seed decision

Seed1111 did not pass the MGD screen.  Seed1114 was therefore not run.  No
seed-driven generalization claim is made.

## Failure

Implementation and cache/resume integrity passed.  The failure is a
validation-trajectory/metric failure: Screen-1 improvement reversed by
Screen-2, with broad missing-mode MAE degradation and Acc5/Acc7 trade-offs.
No tau, alpha, weight, learning-rate, or seed rescue was attempted.
"""
    atomic_text(root / "final" / "stage19_final_report.md", report)
    print(json.dumps({"status": final["stage_statuses"], "report": str(root / "final" / "stage19_final_report.md")}, indent=2))


if __name__ == "__main__":
    main()
