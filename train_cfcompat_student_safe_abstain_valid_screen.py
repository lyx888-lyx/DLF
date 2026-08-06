"""Current-Student interval plus unsafe-KD abstention Valid-only screen.

This exploratory v2 reuses the frozen Windows prerequisite assets and the
original CFCompatKD replay.  Candidate Teacher targets are clipped to the
closed interval between the current missing-modality Student prediction
(detached) and the training label.  A sample explicitly receives KD gate zero
when the clipped target equals the current Student prediction.
"""
from __future__ import annotations

import argparse
import gc
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
import train_cfcompat_safe_projection_valid_screen_v2 as hardened
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
from trains.singleTask.cfcompat_student_safe_abstain_utils import (
    CANDIDATE_RUNS,
    FORMAL_SEEDS,
    METHOD,
    OUTPUT_TAG,
    RUNS,
    VERSION,
    candidate_gate,
    jsonable,
    student_projection_summary,
    student_safe_project_teacher,
)
from trains.singleTask.fixed_kd_utils import (
    checkpoint_sha256,
    teacher_lav_prediction,
)
from trains.singleTask.missing_utils import (
    MISSING_MODES,
    compute_full_dlf_loss,
    compute_task_loss,
    mode_to_mask,
)


_ORIGINAL_LOAD_ASSETS = base.load_assets
_ORIGINAL_TRAIN_TRAJECTORY = base.train_trajectory


class FrozenValidReference:
    """CPU-only Valid reference bundle with no trainable parameter interface."""

    def __init__(self, valid_reference):
        self.valid_reference = valid_reference

    def parameters(self):
        return iter(())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Current-Student safe-abstention CFCompatKD Valid-only screen."
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
        parser.error("The v2 protocol fixes num_workers=1.")
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
                "v2 output already exists; inspect it or use --overwrite: {} / {}".format(
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
    path = directory / "DLF-mosi-student-safe-abstain-v2-{}-{}.log".format(
        kind, datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    logger = logging.getLogger("student_safe_abstain_valid_screen")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def load_assets(cli, args, loaders, seed):
    """Cache frozen Valid references, then release the ModDrop evaluator."""
    teacher, student, evaluator, assets = _ORIGINAL_LOAD_ASSETS(
        cli, args, loaders, seed
    )
    with preserve_rng_state():
        valid_reference = hardened.reference_prediction_rows(
            evaluator, teacher, loaders["valid"], args.device
        )
    if len(valid_reference) != 229:
        raise RuntimeError("MOSI Valid reference cache must contain 229 samples.")

    del evaluator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    assets["baseline_valid_cache_sample_count"] = int(len(valid_reference))
    assets["student_safe_anchor"] = "current_missing_student_prediction_detached"
    assets["unsafe_kd_abstention"] = True
    return teacher, student, FrozenValidReference(valid_reference), assets


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
    del evaluator_bundle
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
        gate = active if run == "student_safe_uniform" else active * compatibility
        for offset in range(labels.size(0)):
            projection_records.append(
                {
                    key: (
                        bool(value[offset].detach().cpu())
                        if value.dtype == torch.bool
                        else float(value[offset].detach().cpu())
                    )
                    for key, value in projection.items()
                }
            )
    else:
        raise ValueError("Unknown student-safe run: {}".format(run))

    kd_loss, each_kd = gated_kd_loss(
        missing_output["output_logit"], kd_target, gate
    )
    total_loss = full_loss + missing_loss + kd_loss
    if not torch.isfinite(total_loss):
        raise FloatingPointError("NaN/Inf in student-safe abstention objective.")

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
        # The frozen v1 trajectory expects these two keys.  The wrapper renames
        # the first one to current_student_missing_MAE in written v2 outputs.
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
    del teacher, loader, device
    return evaluator_bundle.valid_reference.copy()


def method_name(run):
    if run == "cfcompat_replay":
        return "DLF-CFCompatKD-v1"
    if run == "student_safe_uniform":
        return "DLF-Student-Safe-Abstain-Uniform-v2"
    if run == "student_safe_cfcompat":
        return METHOD
    raise ValueError("Unknown run: {}".format(run))


def train_trajectory(cli, logger, output_root, model_root, seed, run):
    """Run the frozen Stage-3 trajectory with v2 objective hooks."""
    result, epoch_rows, raw_events = _ORIGINAL_TRAIN_TRAJECTORY(
        cli, logger, output_root, model_root, seed, run
    )
    result["Method"] = method_name(run)
    result["StudentSafeAnchor"] = "current_missing_student_prediction_detached"
    result["UnsafeKDAbstention"] = bool(run in CANDIDATE_RUNS)
    result["CompatibilityUsed"] = bool(
        run in ("cfcompat_replay", "student_safe_cfcompat")
    )
    for key in (
        "projection_active_fraction",
        "projection_abstain_fraction",
        "projection_raw_teacher_better_fraction",
    ):
        if key not in result:
            result[key] = float("nan")

    for row in epoch_rows:
        row["current_student_missing_MAE"] = row.pop(
            "baseline_missing_MAE", float("nan")
        )
        row["unsafe_KD_abstention"] = bool(run in CANDIDATE_RUNS)

    run_dir = output_root / "seed{}".format(seed) / run
    pd.DataFrame(epoch_rows).to_csv(run_dir / "epoch_metrics.csv", index=False)
    return result, epoch_rows, raw_events


def render_report(summary):
    lines = [
        "# Current-Student Safe-Abstention CFCompatKD v2",
        "",
        "## Frozen exploratory decision",
        "",
        "- Verdict: `{}`".format(summary["verdict"]),
        "- Formal seeds: `1112, 1113, 1115`",
        "- Official Test constructed: `False`",
        "- Anchor: detached current missing-modality Student prediction",
        "- Unsafe/equal target: explicit KD abstention (`gate=0`)",
        "",
        "## Candidate gates",
        "",
    ]
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


def bind_v2_hooks():
    # The frozen trajectory resolves these names dynamically from its module.
    base.RUNS = RUNS
    base.METHOD = METHOD
    base.load_assets = load_assets
    base.forward_objective = forward_objective
    base.reference_prediction_rows = reference_prediction_rows
    base.projection_summary = student_projection_summary


def main():
    cli = parse_args()
    bind_v2_hooks()
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
        if gates["student_safe_cfcompat"]["passed"]:
            verdict = "PROMOTE_STUDENT_SAFE_CFCOMPAT_TO_1111_1114_EXTENSION"
        elif gates["student_safe_uniform"]["passed"]:
            verdict = "STUDENT_SAFE_UNIFORM_PASSED_CFCOMPAT_WEIGHTING_FAILED"
        else:
            verdict = "STOP_STUDENT_SAFE_ABSTENTION_V2_FAILED"

    prefix = "student_safe_abstain"
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
            }
        )

    manifest = {
        "version": VERSION,
        "method": METHOD,
        "base_branch": "feature/cfcompat-safe-projection-valid-screen-v1",
        "implementation_branch": "feature/cfcompat-student-safe-abstain-valid-screen-v2",
        "exploratory_after_v1": True,
        "formal_seeds": list(cli.seeds),
        "runs": list(RUNS),
        "decision_split": "official_valid_only",
        "official_test_constructed": False,
        "test_loader_construction_count": 0,
        "test_loader_traversal_count": 0,
        "additional_inference_parameters": 0,
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
            "exploratory_after_v1": True,
            "formal_seeds": list(cli.seeds),
            "runs": list(RUNS),
            "safe_target": "teacher_clipped_to_closed_detached_current_student_to_train_label_interval",
            "abstention": "gate_zero_when_projected_target_equals_current_student",
            "student_safe_uniform_uses_compatibility": False,
            "student_safe_cfcompat_uses_compatibility": True,
            "lambda_kd": 1.0,
            "optimizer": "Adam",
            "update_epochs": 10,
            "checkpoint_selection": "minimum_official_valid_J",
            "official_test_constructed": False,
            "official_test_authorized": False,
            "next_stage_on_main_pass": "MOSI_seed1111_seed1114_extension",
            "additional_inference_parameters": 0,
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
    print("Current-Student Safe-Abstention v2 screen complete")
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
