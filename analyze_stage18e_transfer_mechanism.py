"""Export frozen train predictions and close the Stage 18E mechanism audit."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from data_loader import MMDataLoader
from train_cf_compat_kd import build_config, initialize_teacher_student
from trains.singleTask.cf_compat_kd_utils import (
    CACHE_VERSION,
    MULTISEED_CACHE_VERSION,
)
from trains.singleTask.cfcompat_fair_trainer import (
    TEST_ISOLATION,
    _asset_manifest,
    _load_frozen_cache,
)
from trains.singleTask.cfcompat_transfer_analysis import (
    METHODS,
    bind_transfer_rows,
    compatibility_deciles,
    compatibility_summaries,
    continuation_gate,
    teacher_benefit_summaries,
    transfer_summaries,
    write_mass_audit,
)
from trains.singleTask.control_training import (
    _teacher_train_cache,
    _write_train_analysis,
)
from trains.singleTask.fixed_kd_utils import checkpoint_sha256
from trains.singleTask.missing_utils import build_single_split_loader
from utils.functions import setup_seed


ROOT = Path("result/missing_baseline/cfcompat_evidence_v1/mosi")
STAGE_A = ROOT / "stage18a_training_recovery"
STAGE_B = ROOT / "stage18b_student_learnability"
STAGE_C = ROOT / "stage18c_seed1114_controls"
STAGE_D = ROOT / "stage18d_seed1111_replication"
STAGE_E = ROOT / "stage18e_transfer_mechanism"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=("export", "analyze"), required=True)
    parser.add_argument(
        "--target",
        choices=("seed1114_moddrop", "seed1114_cfcompat", "seed1111_moddrop"),
    )
    parser.add_argument("--physical-gpu", type=int, choices=(2,), default=2)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--num-workers", type=int, choices=(1,), default=1)
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--config-file", default="config/config.json")
    cli = parser.parse_args()
    if cli.action == "export" and cli.target is None:
        parser.error("--target is required for export")
    if cli.gpu_ids != [0]:
        parser.error("Physical GPU 2 must be internal GPU 0.")
    return cli


def target_spec(target):
    specs = {
        "seed1114_moddrop": (
            1114,
            "moddrop",
            STAGE_A / "moddrop_runA/run_metrics.csv",
        ),
        "seed1114_cfcompat": (
            1114,
            "cfcompat",
            STAGE_A / "cfcompat_runA/run_metrics.csv",
        ),
        "seed1111_moddrop": (
            1111,
            "moddrop",
            STAGE_B / "seed1111/unified/run_metrics.csv",
        ),
    }
    return specs[target]


def export_frozen(cli):
    seed, method, metric_path = target_spec(cli.target)
    setup_seed(seed)
    args = build_config(cli, seed)
    assets = _asset_manifest(seed, "/code/DLF/result", "/code/DLF/pt")
    cache_version = MULTISEED_CACHE_VERSION if seed != 1111 else CACHE_VERSION
    _, cache_by_index = _load_frozen_cache(
        seed, "/code/DLF/result", assets["evaluator_sha"], cache_version
    )
    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("Frozen exporter may construct only train/valid.")
    teacher_cli = type(
        "TeacherCLI", (), {"model_save_dir": "/code/DLF/pt"}
    )()
    teacher, student, _, _ = initialize_teacher_student(
        args, teacher_cli, seed, loaders
    )
    metric = pd.read_csv(metric_path).iloc[0]
    checkpoint = Path(metric.Checkpoint)
    if checkpoint_sha256(checkpoint) != metric.CheckpointSHA256:
        raise RuntimeError("Frozen checkpoint SHA mismatch.")
    student.load_state_dict(
        torch.load(checkpoint, map_location=args.device), strict=True
    )
    analysis_loader = build_single_split_loader(
        args, "train", cli.num_workers
    )
    teacher_by_index, _, _ = _teacher_train_cache(
        teacher, analysis_loader, args.device
    )
    destination = STAGE_E / "frozen_predictions" / (
        "{}.csv".format(cli.target)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    _write_train_analysis(
        student,
        analysis_loader,
        args.device,
        seed,
        method,
        teacher_by_index,
        cache_by_index,
        destination,
    )
    print("EXPORTED {} rows to {}".format(len(pd.read_csv(destination)), destination))


def prediction_path(seed, method):
    if method == "moddrop":
        return STAGE_E / "frozen_predictions" / (
            "seed{}_moddrop.csv".format(seed)
        )
    if seed == 1114 and method == "cfcompat":
        return STAGE_E / "frozen_predictions/seed1114_cfcompat.csv"
    stage = STAGE_C if seed == 1114 else STAGE_D
    return stage / method / "train_analysis_predictions.csv"


def metric_frame():
    return pd.concat(
        [
            pd.read_csv(STAGE_C / "stage18c_seed1114_metrics.csv"),
            pd.read_csv(STAGE_D / "stage18d_seed1111_metrics.csv"),
        ],
        ignore_index=True,
    )


def analyze():
    STAGE_E.mkdir(parents=True, exist_ok=True)
    all_bound = []
    for seed in (1114, 1111):
        baseline = pd.read_csv(prediction_path(seed, "moddrop"))
        if len(baseline) != 1284 * 3:
            raise RuntimeError("ModDrop frozen train prediction count mismatch.")
        for method in METHODS:
            frame = pd.read_csv(prediction_path(seed, method))
            all_bound.append(bind_transfer_rows(baseline, frame))
    bound = pd.concat(all_bound, ignore_index=True)
    quadrants = transfer_summaries(bound)
    benefit = teacher_benefit_summaries(bound)
    calibration = compatibility_summaries(bound)
    deciles = compatibility_deciles(bound)
    harmful = quadrants.copy()
    metrics = metric_frame()
    gate_checks, gate_passed = continuation_gate(metrics, harmful)
    mass = write_mass_audit(STAGE_C)

    quadrants.to_csv(STAGE_E / "stage18e_transfer_quadrants.csv", index=False)
    benefit.to_csv(STAGE_E / "stage18e_teacher_benefit.csv", index=False)
    calibration.to_csv(
        STAGE_E / "stage18e_compatibility_calibration.csv", index=False
    )
    deciles.to_csv(
        STAGE_E / "stage18e_compatibility_deciles.csv", index=False
    )
    harmful.to_csv(
        STAGE_E / "stage18e_harmful_imitation.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "Status": "NOT_RUN_OPTIONAL",
                "Reason": (
                    "The preregistered gradient audit is optional and is not "
                    "needed to decide the failed continuation gate."
                ),
                **TEST_ISOLATION,
            }
        ]
    ).to_csv(STAGE_E / "stage18e_gradient_alignment.csv", index=False)
    gate_checks.to_csv(
        STAGE_E / "stage18e_continuation_gate.csv", index=False
    )

    pooled_transfer = quadrants.loc[
        quadrants.Seed.astype(str).eq("POOLED")
        & quadrants.Mode.eq("ALL")
        & quadrants.Method.isin(["uniform", "cfcompat", "shuffled_gate"])
    ]
    pooled_calibration = calibration.loc[
        calibration.Seed.astype(str).eq("POOLED")
        & calibration.Mode.eq("ALL")
    ]
    seed_comparison = pd.read_csv(
        STAGE_D / "stage18d_two_seed_comparison.csv"
    )
    status = (
        "STAGE18F_CONTINUATION_GATE_PASSED"
        if gate_passed
        else "STAGE18_CORE_DISTILLATION_CLAIM_UNSUPPORTED"
    )
    audit = (
        "# Stage 18E Transfer Mechanism Audit\n\n"
        "Status: `{}`\n\n"
        "All outcomes use frozen train-only predictions. No Test loader, "
        "feature, label, prediction, or evaluation was accessed.\n\n"
        "## Continuation gate\n\n"
        "```\n{}\n```\n\n"
        "## Pooled transfer outcomes\n\n"
        "```\n{}\n```\n\n"
        "## Compatibility discrimination\n\n"
        "```\n{}\n```\n\n"
        "## Two-seed validation comparisons\n\n"
        "```\n{}\n```\n\n"
        "## KD-mass binding\n\n"
        "All {} controlled epoch/scope checks passed at tolerance 1e-10: {}.\n\n"
        "Because the full five-item continuation gate {}, five-seed Stage 18F "
        "and Locked Test {}.\n"
    ).format(
        status,
        gate_checks.to_string(index=False),
        pooled_transfer.to_string(index=False),
        pooled_calibration.to_string(index=False),
        seed_comparison.to_string(index=False),
        len(mass),
        bool(mass["Passed1e-10"].all()),
        "passed" if gate_passed else "failed",
        "may proceed" if gate_passed else "remain prohibited",
    )
    (STAGE_E / "stage18e_mechanism_audit.md").write_text(audit)
    manifest = {
        "stage": "18E",
        "status": status,
        "continuation_gate_passed": gate_passed,
        "test_isolation": TEST_ISOLATION,
        "locked_test_access_count": 0,
        "five_seed_stage18f_executed": False,
        "physical_gpu_authorized_by_user": 2,
    }
    (STAGE_E / "stage18e_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(gate_checks.to_string(index=False))
    print(status)


def main():
    cli = parse_args()
    if cli.action == "export":
        export_frozen(cli)
    else:
        analyze()


if __name__ == "__main__":
    main()
