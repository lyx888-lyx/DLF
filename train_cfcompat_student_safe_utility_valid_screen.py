"""Student-safe dynamic-utility plus residual-CFCompat Valid-only screen.

This exploratory v3 keeps the current-Student interval and explicit unsafe-KD
abstention from v2.  It compares uniform active-sample KD against dynamic
utility/difficulty weighting and a residual form of the original frozen
CFCompat prior.  Only Train and official Valid are constructed.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import train_cfcompat_safe_projection_valid_screen as base
import train_cfcompat_student_safe_abstain_valid_screen as v2
from train_cf_compat_kd import batch_to_device
from trains.singleTask.cf_compat_kd_utils import (
    compatibility_for_modes,
    gated_kd_loss,
)
from trains.singleTask.cfcompat_safe_projection_utils import (
    derive_valid_events,
    group_summary,
    overall_from_events,
    replay_gate,
)
from trains.singleTask.cfcompat_stability_utils import (
    load_stage3_reference,
    preserve_rng_state,
)
from trains.singleTask.cfcompat_student_safe_utility_utils import (
    CANDIDATE_RUNS,
    FORMAL_SEEDS,
    METHOD,
    OUTPUT_TAG,
    PRIMARY_RUN,
    RESIDUAL_CFCOMPAT_ALPHA,
    RUNS,
    UNIFORM_RUN,
    UTILITY_RUN,
    VERSION,
    candidate_gate,
    dynamic_difficulty,
    dynamic_utility,
    frozen_thresholds,
    jsonable,
    residual_compatibility,
    student_safe_project_teacher,
    utility_projection_summary,
)
from trains.singleTask.fixed_kd_utils import (
    _restore_normal_position_cache,
    checkpoint_sha256,
    teacher_lav_prediction,
)
from trains.singleTask.missing_utils import (
    MISSING_MODES,
    build_single_split_loader,
    compute_full_dlf_loss,
    compute_task_loss,
    mode_to_mask,
)


_ORIGINAL_TRAIN_TRAJECTORY = base.train_trajectory
_CURRENT_RUN = None
_ACTIVE_TAU_BY_MODE = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Student-safe utility/residual-CFCompat v3 Valid-only screen."
    )
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--log-dir", default="log/windows_valid_only_prereq_v1")
    parser.add_argument("--config-file", default="config/config.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if int(args.num_workers) != 1:
        parser.error("The v3 protocol fixes num_workers=1.")
    args.seeds = [1112] if args.smoke_test else list(FORMAL_SEEDS)
    args.max_epochs = 2 if args.smoke_test else None
    return args


def result_paths(cli):
    output = (
        Path(cli.result_root)
        / "missing_baseline"
        / OUTPUT_TAG
        / cli.dataset
        / "valid_screen"
    )
    model = (
        Path(cli.model_save_dir)
        / "missing_baseline"
        / OUTPUT_TAG
        / cli.dataset
        / "valid_screen"
    )
    if cli.smoke_test:
        output, model = output / "smoke", model / "smoke"
    if output.exists() or model.exists():
        if not cli.overwrite:
            raise FileExistsError(
                "v3 output already exists; inspect it or use --overwrite: {} / {}".format(
                    output, model
                )
            )
        if output.exists():
            shutil.rmtree(output)
        if model.exists():
            shutil.rmtree(model)
    output.mkdir(parents=True, exist_ok=True)
    model.mkdir(parents=True, exist_ok=True)
    return output, model


def create_logger(cli):
    directory = Path(cli.log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    kind = "smoke" if cli.smoke_test else "formal"
    path = directory / "DLF-mosi-student-safe-utility-v3-{}-{}.log".format(
        kind, datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    logger = logging.getLogger("student_safe_utility_valid_screen")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def initial_student_difficulty_scales(student, args, num_workers):
    """Freeze per-mode Train-only median initial-Student absolute errors."""
    errors = {mode: [] for mode in MISSING_MODES}
    was_training = bool(student.training)
    with preserve_rng_state():
        loader = build_single_split_loader(args, "train", num_workers)
        try:
            student.eval()
            with torch.inference_mode():
                for batch in loader:
                    text, audio, vision, labels = batch_to_device(
                        batch, args.device
                    )
                    for mode in MISSING_MODES:
                        mask = mode_to_mask(
                            mode,
                            labels.size(0),
                            args.device,
                            audio.dtype,
                        )
                        prediction = student(
                            text, audio, vision, mask
                        )["output_logit"].view(-1)
                        normal = prediction.detach().clone()
                        _restore_normal_position_cache()
                        errors[mode].append(
                            torch.abs(normal - labels.view(-1)).cpu()
                        )
        finally:
            student.train(was_training)
            _restore_normal_position_cache()

    scales = {}
    for mode in MISSING_MODES:
        if not errors[mode]:
            raise RuntimeError("Train-only difficulty pass was empty for {}.".format(mode))
        values = torch.cat(errors[mode]).numpy().astype(np.float64)
        if len(values) != 1284 or not np.isfinite(values).all():
            raise RuntimeError(
                "Difficulty scale requires 1284 finite Train errors for {}.".format(mode)
            )
        tau = float(np.median(values))
        if not math.isfinite(tau) or tau <= 0.0:
            raise RuntimeError("Difficulty scale is not positive for {}.".format(mode))
        scales[mode] = tau
    return scales


def load_assets(cli, args, loaders, seed):
    """Reuse the v2 memory-safe assets and add frozen Train-only scales."""
    global _ACTIVE_TAU_BY_MODE
    teacher, student, valid_bundle, assets = v2.load_assets(
        cli, args, loaders, seed
    )
    if _CURRENT_RUN is None:
        raise RuntimeError("v3 trajectory run was not bound before asset loading.")

    if _CURRENT_RUN == "cfcompat_replay":
        tau_by_mode = {}
    elif _CURRENT_RUN in CANDIDATE_RUNS:
        tau_by_mode = initial_student_difficulty_scales(
            student, args, cli.num_workers
        )
    else:
        raise RuntimeError("Unknown bound v3 run: {}".format(_CURRENT_RUN))

    valid_bundle.difficulty_tau_by_mode = dict(tau_by_mode)
    assets["difficulty_tau_by_mode"] = dict(tau_by_mode)
    assets["difficulty_scale_source"] = (
        "initial_student_train_only_per_mode_median_absolute_error"
    )
    assets["residual_cfcompat_alpha"] = RESIDUAL_CFCOMPAT_ALPHA
    _ACTIVE_TAU_BY_MODE = dict(tau_by_mode)
    return teacher, student, valid_bundle, assets


def tau_tensor_for_modes(bundle, modes, device, dtype):
    values = []
    for mode in modes:
        if str(mode) not in bundle.difficulty_tau_by_mode:
            raise KeyError("Difficulty scale is absent for mode {}.".format(mode))
        values.append(float(bundle.difficulty_tau_by_mode[str(mode)]))
    tensor = torch.as_tensor(values, device=device, dtype=dtype).view(-1)
    if not torch.isfinite(tensor).all() or torch.any(tensor <= 0.0):
        raise FloatingPointError("Bound difficulty scales are invalid.")
    return tensor


def forward_objective(
    run,
    batch,
    missing_mask,
    modes,
    args,
    teacher,
    evaluator_bundle,
    student,
    cache_by_index,
    criterion,
    cosine,
    hinge,
):
    text, audio, vision, labels = batch_to_device(batch, args.device)
    missing_mask = missing_mask.to(device=args.device, dtype=audio.dtype)
    full_mask = mode_to_mask("LAV", labels.size(0), args.device, audio.dtype)

    full_output = student(text, audio, vision, full_mask)
    full_loss, _ = compute_full_dlf_loss(
        full_output, labels, criterion, cosine, hinge
    )
    missing_output = student(text, audio, vision, missing_mask)
    missing_loss, _ = compute_task_loss(missing_output, labels, criterion)
    teacher_prediction = teacher_lav_prediction(teacher, text, audio, vision)

    indices = batch["index"].view(-1).cpu().numpy().astype(int).tolist()
    compatibility = compatibility_for_modes(
        cache_by_index,
        indices,
        list(modes),
        args.device,
        labels.dtype,
    ).view(-1)

    projection_records = []
    current_student = None
    if run == "cfcompat_replay":
        kd_target = teacher_prediction.detach().view(-1, 1)
        gate = compatibility
    elif run in CANDIDATE_RUNS:
        current_student = missing_output["output_logit"].detach().view(-1, 1)
        kd_target, projection = student_safe_project_teacher(
            current_student, teacher_prediction, labels
        )
        active = projection["active"].to(
            device=args.device, dtype=compatibility.dtype
        )
        utility = dynamic_utility(current_student, kd_target, labels)
        tau = tau_tensor_for_modes(
            evaluator_bundle,
            modes,
            args.device,
            labels.dtype,
        )
        difficulty = dynamic_difficulty(current_student, labels, tau)
        residual = residual_compatibility(compatibility)

        if run == UNIFORM_RUN:
            gate = active
        elif run == UTILITY_RUN:
            gate = active * utility * difficulty
        elif run == PRIMARY_RUN:
            gate = active * utility * difficulty * residual
        else:
            raise ValueError("Unknown v3 candidate: {}".format(run))

        for offset in range(labels.size(0)):
            record = {
                key: (
                    bool(value[offset].detach().cpu())
                    if value.dtype == torch.bool
                    else float(value[offset].detach().cpu())
                )
                for key, value in projection.items()
            }
            record.update(
                {
                    "mode": str(modes[offset]),
                    "utility": float(utility[offset].detach().cpu()),
                    "difficulty": float(difficulty[offset].detach().cpu()),
                    "compatibility": float(
                        compatibility[offset].detach().cpu()
                    ),
                    "residual_compatibility": float(
                        residual[offset].detach().cpu()
                    ),
                    "final_gate": float(gate[offset].detach().cpu()),
                    "difficulty_tau": float(tau[offset].detach().cpu()),
                }
            )
            projection_records.append(record)
    else:
        raise ValueError("Unknown v3 run: {}".format(run))

    kd_loss, each_kd = gated_kd_loss(
        missing_output["output_logit"], kd_target, gate
    )
    total_loss = full_loss + missing_loss + kd_loss
    if not torch.isfinite(total_loss):
        raise FloatingPointError("NaN/Inf in student-safe utility objective.")

    denominator = float(gate.sum().detach().cpu()) + 1e-8
    diagnostics = {
        "full_loss": float(full_loss.detach().cpu()),
        "missing_loss": float(missing_loss.detach().cpu()),
        "kd_loss": float(kd_loss.detach().cpu()),
        "mean_gate": float(gate.detach().mean().cpu()),
        "weighted_kd": float(
            (gate.detach() * each_kd.detach().view(-1)).sum().cpu()
            / denominator
        ),
        "baseline_missing_MAE": (
            float(
                torch.abs(
                    current_student.view(-1) - labels.view(-1)
                ).mean().cpu()
            )
            if current_student is not None
            else float("nan")
        ),
        "safe_target_MAE": (
            float(
                torch.abs(kd_target.view(-1) - labels.view(-1)).mean().cpu()
            )
            if current_student is not None
            else float("nan")
        ),
    }
    return total_loss, diagnostics, projection_records


def reference_prediction_rows(evaluator_bundle, teacher, loader, device):
    return v2.reference_prediction_rows(
        evaluator_bundle, teacher, loader, device
    )


def method_name(run):
    mapping = {
        "cfcompat_replay": "DLF-CFCompatKD-v1",
        UNIFORM_RUN: "DLF-Student-Safe-Abstain-Uniform-v2",
        UTILITY_RUN: "DLF-Student-Safe-Dynamic-Utility-v3",
        PRIMARY_RUN: METHOD,
    }
    if run not in mapping:
        raise ValueError("Unknown run: {}".format(run))
    return mapping[run]


def train_trajectory(cli, logger, output_root, model_root, seed, run):
    """Run the frozen Stage-3 trajectory with v3 objective hooks."""
    global _CURRENT_RUN, _ACTIVE_TAU_BY_MODE
    if _CURRENT_RUN is not None:
        raise RuntimeError("Nested v3 trajectory binding is forbidden.")
    _CURRENT_RUN = str(run)
    _ACTIVE_TAU_BY_MODE = None
    try:
        result, epoch_rows, raw_events = _ORIGINAL_TRAIN_TRAJECTORY(
            cli, logger, output_root, model_root, seed, run
        )
        tau_by_mode = dict(_ACTIVE_TAU_BY_MODE or {})
    finally:
        _CURRENT_RUN = None
        _ACTIVE_TAU_BY_MODE = None

    result["Method"] = method_name(run)
    result["StudentSafeAnchor"] = "current_missing_student_prediction_detached"
    result["UnsafeKDAbstention"] = bool(run in CANDIDATE_RUNS)
    result["DynamicUtilityUsed"] = bool(run in (UTILITY_RUN, PRIMARY_RUN))
    result["DynamicDifficultyUsed"] = bool(run in (UTILITY_RUN, PRIMARY_RUN))
    result["CompatibilityUsed"] = bool(
        run in ("cfcompat_replay", PRIMARY_RUN)
    )
    result["ResidualCFCompatAlpha"] = (
        RESIDUAL_CFCOMPAT_ALPHA if run == PRIMARY_RUN else float("nan")
    )
    for mode in MISSING_MODES:
        result["DifficultyTau{}".format(mode)] = float(
            tau_by_mode.get(mode, float("nan"))
        )

    utility_projection_keys = (
        "projection_mean_utility",
        "projection_mean_active_utility",
        "projection_mean_difficulty",
        "projection_mean_compatibility",
        "projection_mean_residual_compatibility",
        "projection_mean_final_gate",
        "projection_gate_effective_sample_size",
        "projection_active_compatibility_utility_pearson",
        "projection_active_compatibility_utility_spearman",
    )
    for key in utility_projection_keys:
        if key not in result:
            result[key] = float("nan")

    for row in epoch_rows:
        row["current_student_missing_MAE"] = row.pop(
            "baseline_missing_MAE", float("nan")
        )
        row["unsafe_KD_abstention"] = bool(run in CANDIDATE_RUNS)
        row["dynamic_utility_used"] = bool(run in (UTILITY_RUN, PRIMARY_RUN))
        row["dynamic_difficulty_used"] = bool(
            run in (UTILITY_RUN, PRIMARY_RUN)
        )
        for mode in MISSING_MODES:
            row["difficulty_tau_{}".format(mode)] = float(
                tau_by_mode.get(mode, float("nan"))
            )

    run_dir = output_root / "seed{}".format(seed) / run
    pd.DataFrame(epoch_rows).to_csv(run_dir / "epoch_metrics.csv", index=False)
    return result, epoch_rows, raw_events


def render_report(summary):
    lines = [
        "# Student-Safe Dynamic-Utility Residual-CFCompat v3",
        "",
        "## Frozen exploratory decision",
        "",
        "- Verdict: `{}`".format(summary["verdict"]),
        "- Formal seeds: `{}`".format(
            ", ".join(str(seed) for seed in summary["protocol"]["formal_seeds"])
        ),
        "- Official Test constructed: `False`",
        "- Safe anchor: detached current missing-modality Student prediction",
        "- Unsafe/equal target: explicit KD abstention (`gate=0`)",
        "- Dynamic utility: fraction of current absolute error removed",
        "- Difficulty scale: initial-Student Train-only per-mode median error",
        "- Residual CFCompat: `0.5 + 0.5 * compatibility`",
        "",
    ]
    if not summary["candidate_gates"]:
        lines.extend(["Smoke-only run; no formal decision gate was evaluated.", ""])
        return "\n".join(lines) + "\n"

    lines.extend(["## Candidate gates", ""])
    for run in CANDIDATE_RUNS:
        gate = summary["candidate_gates"][run]
        lines.extend(
            [
                "### {}".format(run),
                "",
                "- Passed: `{}`".format(gate["passed"]),
                "- Mean Valid-J degradation: `{:+.6f}`".format(
                    gate["mean_J_degradation_vs_CFCompatKD"]
                ),
                "- Mean harmful-imitation reduction: `{:+.6f}`".format(
                    gate["mean_harmful_imitation_reduction"]
                ),
                "",
            ]
        )
        for seed in FORMAL_SEEDS:
            item = gate["per_seed"][str(seed)]
            lines.append(
                "- Seed {}: J degradation `{:+.6f}`, harmful reduction `{:+.6f}`, abstain `{:.4f}`".format(
                    seed,
                    item["J_degradation_vs_CFCompatKD"],
                    item["harmful_reduction"],
                    item["projection_abstain_fraction"],
                )
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def bind_v3_hooks():
    base.RUNS = RUNS
    base.METHOD = METHOD
    base.load_assets = load_assets
    base.forward_objective = forward_objective
    base.reference_prediction_rows = reference_prediction_rows
    base.projection_summary = utility_projection_summary


def verdict_from_gates(gates):
    if gates[PRIMARY_RUN]["passed"]:
        return "PROMOTE_UTILITY_RESIDUAL_CFCOMPAT_TO_1111_1114_EXTENSION"
    if gates[UTILITY_RUN]["passed"]:
        return "DYNAMIC_UTILITY_PASSED_RESIDUAL_CFCOMPAT_FAILED"
    if gates[UNIFORM_RUN]["passed"]:
        return "STUDENT_SAFE_UNIFORM_PASSED_DYNAMIC_UTILITY_FAILED"
    return "STOP_STUDENT_SAFE_UTILITY_V3_FAILED"


def main():
    cli = parse_args()
    bind_v3_hooks()
    output_root, model_root = result_paths(cli)
    logger, log_path = create_logger(cli)

    grid_rows = []
    epoch_rows = []
    raw_events = []
    stage3_references = []
    replay_checks = {}

    for seed in cli.seeds:
        reference_series, reference_path = load_stage3_reference(
            cli.result_root, seed
        )
        reference = {
            str(key): jsonable(value) for key, value in reference_series.items()
        }
        reference["ReferencePath"] = str(reference_path.resolve())
        reference["ReferenceSHA256"] = checkpoint_sha256(reference_path)
        stage3_references.append(reference)

        for run in RUNS:
            result, local_epochs, local_events = train_trajectory(
                cli, logger, output_root, model_root, seed, run
            )
            grid_rows.append(result)
            epoch_rows.extend(local_epochs)
            raw_events.extend(local_events)
            if run == "cfcompat_replay" and not cli.smoke_test:
                check = replay_gate(result, reference)
                replay_checks[str(seed)] = check
                if not check["passed"]:
                    raise RuntimeError(
                        "Seed {} Stage-3 replay failed: {}".format(seed, check)
                    )

    raw_frame = pd.DataFrame(raw_events)
    events = derive_valid_events(raw_frame)
    overall = overall_from_events(events)
    groups = group_summary(events)

    if cli.smoke_test:
        gates = {}
        verdict = "SMOKE_ONLY_NO_DECISION"
    else:
        gates = {
            run: candidate_gate(run, grid_rows, epoch_rows, groups)
            for run in CANDIDATE_RUNS
        }
        verdict = verdict_from_gates(gates)

    prefix = "student_safe_utility_v3"
    artifacts = {
        f"{prefix}_valid_grid_summary.csv": pd.DataFrame(grid_rows),
        f"{prefix}_all_epoch_metrics.csv": pd.DataFrame(epoch_rows),
        f"{prefix}_raw_valid_events.csv": raw_frame,
        f"{prefix}_valid_events.csv": events,
        f"{prefix}_overall_metrics.csv": overall,
        f"{prefix}_group_metrics.csv": groups,
        f"{prefix}_stage3_references.csv": pd.DataFrame(stage3_references),
    }
    for name, frame in artifacts.items():
        frame.to_csv(output_root / name, index=False)

    source_records = []
    for row in grid_rows:
        source_records.append(
            {
                "seed": int(row["Seed"]),
                "run": str(row["Run"]),
                "checkpoint": row["MainCheckpoint"],
                "checkpoint_sha256": row["MainCheckpointSHA256"],
                "teacher_checkpoint": row["TeacherCheckpoint"],
                "teacher_sha256": row["TeacherSHA256"],
                "evaluator_checkpoint": row["EvaluatorCheckpoint"],
                "evaluator_sha256": row["EvaluatorSHA256"],
                "evaluator_source": row["EvaluatorSource"],
                "evaluator_source_sha256": row["EvaluatorSourceSHA256"],
                "compatibility_cache": row["CompatibilityCache"],
                "compatibility_cache_sha256": row["CompatibilityCacheSHA256"],
                "compatibility_config": row["CompatibilityConfig"],
                "compatibility_config_sha256": row["CompatibilityConfigSHA256"],
                "difficulty_tau_LA": jsonable(row["DifficultyTauLA"]),
                "difficulty_tau_LV": jsonable(row["DifficultyTauLV"]),
                "difficulty_tau_L": jsonable(row["DifficultyTauL"]),
            }
        )

    manifest = {
        "version": VERSION,
        "method": METHOD,
        "base_branch": "feature/cfcompat-student-safe-abstain-valid-screen-v2",
        "implementation_branch": (
            "feature/cfcompat-student-safe-utility-residual-valid-screen-v3"
        ),
        "exploratory_after_v1_v2": True,
        "formal_seeds": list(cli.seeds),
        "runs": list(RUNS),
        "decision_split": "official_valid_only",
        "official_test_constructed": False,
        "test_loader_construction_count": 0,
        "test_loader_traversal_count": 0,
        "additional_inference_parameters": 0,
        "difficulty_scale_source": (
            "initial_student_train_only_per_mode_median_absolute_error"
        ),
        "residual_cfcompat_alpha": RESIDUAL_CFCOMPAT_ALPHA,
        "source_records": source_records,
        "stage3_references": [
            {
                "seed": int(row["Seed"]),
                "path": row["ReferencePath"],
                "sha256": row["ReferenceSHA256"],
            }
            for row in stage3_references
        ],
        "artifacts": {},
    }
    for name in artifacts:
        path = output_root / name
        manifest["artifacts"][name] = {
            "path": str(path.resolve()),
            "sha256": checkpoint_sha256(path),
        }
    manifest_path = output_root / f"{prefix}_source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "version": VERSION,
        "method": METHOD,
        "verdict": verdict,
        "baseline_replay_gates": replay_checks,
        "candidate_gates": gates,
        "protocol": {
            "exploratory_after_v1_v2": True,
            "formal_seeds": list(cli.seeds),
            "runs": list(RUNS),
            "safe_target": (
                "teacher_clipped_to_closed_detached_current_student_to_train_label_interval"
            ),
            "abstention": (
                "gate_zero_when_projected_target_equals_current_student"
            ),
            "dynamic_utility": (
                "fraction_of_current_absolute_label_error_removed_by_safe_target"
            ),
            "dynamic_difficulty": (
                "current_absolute_error_over_error_plus_train_only_initial_student_mode_median"
            ),
            "difficulty_scale_source": (
                "initial_student_train_only_per_mode_median_absolute_error"
            ),
            "residual_cfcompat": "alpha_plus_one_minus_alpha_times_compatibility",
            "residual_cfcompat_alpha": RESIDUAL_CFCOMPAT_ALPHA,
            "primary_gate": (
                "active_times_utility_times_difficulty_times_residual_cfcompat"
            ),
            "lambda_kd": 1.0,
            "optimizer": "Adam",
            "update_epochs": 10,
            "checkpoint_selection": "minimum_official_valid_J",
            "official_test_constructed": False,
            "official_test_authorized": False,
            "next_stage_on_main_pass": "MOSI_seed1111_seed1114_extension",
            "additional_inference_parameters": 0,
            "frozen_candidate_thresholds": frozen_thresholds(),
        },
    }
    summary_path = output_root / f"{prefix}_valid_screen_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = output_root / f"{prefix}_valid_screen_report.md"
    report_path.write_text(render_report(summary), encoding="utf-8")

    logger.info(
        "complete verdict=%s output=%s log=%s", verdict, output_root, log_path
    )
    print("Student-Safe Dynamic-Utility v3 screen complete")
    if not cli.smoke_test:
        for run in CANDIDATE_RUNS:
            gate = gates[run]
            print(
                run,
                "mean J degradation:",
                "{:+.6f}".format(gate["mean_J_degradation_vs_CFCompatKD"]),
                "mean harmful reduction:",
                "{:+.6f}".format(gate["mean_harmful_imitation_reduction"]),
            )
    print("verdict:", verdict)
    print("official Test was not constructed")
    print("report:", report_path)


if __name__ == "__main__":
    main()
