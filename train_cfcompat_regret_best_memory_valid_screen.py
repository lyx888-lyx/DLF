"""Regret-Aware Best-So-Far Memory CFCompatKD v4.2, Seed1113 Valid-only dev.

Exactly one new Seed1113 trajectory is trained. Audited v2, v4 and v4.1 Valid
references are reused without retraining. Official Test is never constructed.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import train_cfcompat_safe_projection_valid_screen as base
import train_cfcompat_safe_projection_valid_screen_v2 as hardened
import train_cfcompat_regret_preserve_valid_screen as v4
import train_cfcompat_regret_preserve_guard_valid_screen as v4p1
from train_cf_compat_kd import batch_to_device
from trains.singleTask.cf_compat_kd_utils import compatibility_for_modes
from trains.singleTask.cfcompat_regret_best_memory_utils import (
    BestSoFarPredictionMemory,
    DEV_SEED,
    LAMBDA_PRESERVE_V4P2,
    METHOD,
    OUTPUT_TAG,
    RUN,
    RUNS,
    VERSION,
    best_memory_projection_summary,
    dev_candidate_gate,
    frozen_thresholds,
    negative_transfer_summary,
    regret_best_memory_decision,
    tiered_kd_loss,
)
from trains.singleTask.cfcompat_safe_projection_utils import (
    derive_valid_events,
    group_summary,
    overall_from_events,
)
from trains.singleTask.cfcompat_stability_utils import preserve_rng_state
from trains.singleTask.fixed_kd_utils import checkpoint_sha256, teacher_lav_prediction
from trains.singleTask.missing_utils import (
    MISSING_MODES,
    compute_full_dlf_loss,
    compute_task_loss,
    mode_to_mask,
)


_ORIGINAL_LOAD_ASSETS = base.load_assets
_ORIGINAL_TRAIN_TRAJECTORY = base.train_trajectory
_ACTIVE_TRAIN_BASELINE_FRAME = None
_ACTIVE_BEST_MEMORY = None
_ACTIVE_TRAIN_DECISIONS = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Best-So-Far Regret Memory v4.2 Valid-only Seed1113 development screen."
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
        parser.error("v4.2 fixes num_workers=1.")
    args.seeds = [DEV_SEED]
    args.max_epochs = 2 if args.smoke_test else None
    return args


def result_paths(cli):
    output = Path(cli.result_root) / "missing_baseline" / OUTPUT_TAG / cli.dataset / "valid_screen" / "seed1113_dev"
    model = Path(cli.model_save_dir) / "missing_baseline" / OUTPUT_TAG / cli.dataset / "valid_screen" / "seed1113_dev"
    if cli.smoke_test:
        output, model = output / "smoke", model / "smoke"
    if output.exists() or model.exists():
        if not cli.overwrite:
            raise FileExistsError("v4.2 output already exists; inspect it or use --overwrite: {} / {}".format(output, model))
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
    path = directory / "DLF-mosi-regret-best-memory-v4p2-seed1113-{}-{}.log".format(
        kind, datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    logger = logging.getLogger("regret_best_memory_valid_screen")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def load_assets(cli, args, loaders, seed):
    """Build frozen references and initialize best memory without RNG/shuffle drift."""
    global _ACTIVE_TRAIN_BASELINE_FRAME, _ACTIVE_BEST_MEMORY
    if int(seed) != DEV_SEED:
        raise RuntimeError("v4.2 development is locked to Seed1113.")
    teacher, student, evaluator, assets = _ORIGINAL_LOAD_ASSETS(cli, args, loaders, seed)
    with preserve_rng_state():
        valid_reference = hardened.reference_prediction_rows(
            evaluator, teacher, loaders["valid"], args.device
        )
        train_baseline = v4.train_baseline_prediction_rows(
            evaluator, loaders["train"], args.device
        )
    if len(valid_reference) != 229:
        raise RuntimeError("MOSI Valid reference cache must contain 229 samples.")
    _ACTIVE_TRAIN_BASELINE_FRAME = train_baseline.copy()
    _ACTIVE_BEST_MEMORY = BestSoFarPredictionMemory(train_baseline)
    bundle = v4.FrozenRegretReference(valid_reference, train_baseline)
    del evaluator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    assets["baseline_valid_cache_sample_count"] = int(len(valid_reference))
    assets["baseline_train_cache_sample_count"] = int(len(train_baseline))
    assets["train_loader_generator_preserved"] = True
    assets["regret_anchor"] = "per_sample_per_missing_mode_best_so_far_prediction_memory"
    assets["memory_initialization"] = "frozen_validation_best_moddrop_missing_prediction"
    assets["routing_policy"] = "strong_weak_distill_memory_preserve_abstain"
    return teacher, student, bundle, assets


def forward_objective(
    run, batch, missing_mask, modes, args, teacher, evaluator_bundle, student,
    cache_by_index, criterion, cosine, hinge,
):
    global _ACTIVE_BEST_MEMORY, _ACTIVE_TRAIN_DECISIONS
    if run != RUN:
        raise ValueError("Unknown v4.2 run: {}".format(run))
    if _ACTIVE_BEST_MEMORY is None or _ACTIVE_TRAIN_DECISIONS is None:
        raise RuntimeError("v4.2 memory/decision recorder is not active.")

    text, audio, vision, labels = batch_to_device(batch, args.device)
    missing_mask = missing_mask.to(device=args.device, dtype=audio.dtype)
    full_mask = mode_to_mask("LAV", labels.size(0), args.device, audio.dtype)
    full_output = student(text, audio, vision, full_mask)
    full_loss, _ = compute_full_dlf_loss(full_output, labels, criterion, cosine, hinge)
    missing_output = student(text, audio, vision, missing_mask)
    missing_loss, _ = compute_task_loss(missing_output, labels, criterion)
    current_student = missing_output["output_logit"].detach().view(-1, 1)
    teacher_prediction = teacher_lav_prediction(teacher, text, audio, vision).view(-1, 1)

    indices = batch["index"].view(-1).cpu().numpy().astype(int).tolist()
    mode_list = list(modes)
    compatibility = compatibility_for_modes(
        cache_by_index, indices, mode_list, args.device, labels.dtype
    ).view(-1)
    baseline_prediction = v4.baseline_for_modes(
        evaluator_bundle, indices, mode_list, labels, args.device, labels.dtype
    )

    memory = _ACTIVE_BEST_MEMORY.bind_and_update(
        indices=indices,
        modes=mode_list,
        current_prediction=current_student,
        labels=labels,
        event_start_ordinal=len(_ACTIVE_TRAIN_DECISIONS) + 1,
        device=args.device,
        dtype=labels.dtype,
    )
    memory_prediction = memory["memory_after_prediction"]
    decision = regret_best_memory_decision(
        current_student, teacher_prediction, memory_prediction, labels, compatibility
    )
    kd_loss, each_kd, eligible_mass, effective_gate = tiered_kd_loss(
        missing_output["output_logit"],
        decision["teacher_safe_target"],
        decision["strong_gate"],
        decision["weak_gate"],
    )
    each_preserve = F.smooth_l1_loss(
        missing_output["output_logit"].view(-1),
        decision["preserve_safe_target"].view(-1),
        reduction="none",
    )
    preserve_weight = decision["preserve_gate"].to(each_preserve)
    preserve_loss = torch.sum(preserve_weight * each_preserve) / (
        torch.sum(preserve_weight) + 1e-8
    )
    total_loss = full_loss + missing_loss + kd_loss + LAMBDA_PRESERVE_V4P2 * preserve_loss
    if not torch.isfinite(total_loss):
        raise FloatingPointError("NaN/Inf in v4.2 best-memory objective.")

    identifiers = list(batch["id"])
    projection_records = []
    projection = decision["teacher_projection"]
    for offset in range(labels.size(0)):
        record = {
            key: (
                bool(value[offset].detach().cpu())
                if value.dtype == torch.bool
                else float(value[offset].detach().cpu())
            )
            for key, value in projection.items()
        }
        record.update({
            "sample_index": int(indices[offset]),
            "sample_id": str(identifiers[offset]),
            "mode": str(mode_list[offset]),
            "label": float(labels[offset].detach().cpu()),
            "student_prediction": float(current_student[offset].detach().cpu()),
            "baseline_prediction": float(baseline_prediction[offset].detach().cpu()),
            "teacher_prediction": float(teacher_prediction[offset].detach().cpu()),
            "memory_before_prediction": float(memory["memory_before_prediction"][offset].detach().cpu()),
            "memory_before_error": float(memory["memory_before_error"][offset].detach().cpu()),
            "memory_after_prediction": float(memory_prediction[offset].detach().cpu()),
            "memory_after_error": float(memory["memory_after_error"][offset].detach().cpu()),
            "memory_updated": bool(memory["memory_updated"][offset].detach().cpu()),
            "memory_improvement": float(memory["memory_improvement"][offset].detach().cpu()),
            "memory_update_count_after": int(memory["memory_update_count_after"][offset].detach().cpu()),
            "teacher_safe_target": float(decision["teacher_safe_target"][offset].detach().cpu()),
            "preserve_safe_target": float(decision["preserve_safe_target"][offset].detach().cpu()),
            "strong_distill": bool(decision["strong_distill"][offset].detach().cpu()),
            "weak_distill": bool(decision["weak_distill"][offset].detach().cpu()),
            "preserve": bool(decision["preserve"][offset].detach().cpu()),
            "memory_abstain": bool(decision["abstain"][offset].detach().cpu()),
            "teacher_better_any": bool(decision["teacher_better_any"][offset].detach().cpu()),
            "strong_candidate": bool(decision["strong_candidate"][offset].detach().cpu()),
            "weak_candidate": bool(decision["weak_candidate"][offset].detach().cpu()),
            "memory_direction_correct": bool(decision["memory_direction_correct"][offset].detach().cpu()),
            "current_regressed": bool(decision["current_regressed"][offset].detach().cpu()),
            "memory_error": float(decision["memory_error"][offset].detach().cpu()),
            "teacher_error": float(decision["teacher_error"][offset].detach().cpu()),
            "current_error": float(decision["current_error"][offset].detach().cpu()),
            "teacher_advantage_vs_memory": float(decision["teacher_advantage_vs_memory"][offset].detach().cpu()),
            "current_regret_vs_memory": float(decision["current_regret_vs_memory"][offset].detach().cpu()),
            "compatibility": float(compatibility[offset].detach().cpu()),
            "mild_compatibility": float(decision["mild_compatibility"][offset].detach().cpu()),
            "strong_gate": float(decision["strong_gate"][offset].detach().cpu()),
            "weak_gate": float(decision["weak_gate"][offset].detach().cpu()),
            "eligible_distill_mass": float(eligible_mass[offset].detach().cpu()),
            "effective_distill_gate": float(effective_gate[offset].detach().cpu()),
            "preserve_gate": float(decision["preserve_gate"][offset].detach().cpu()),
            "distill_loss_each": float(each_kd[offset].detach().cpu()),
            "preserve_loss_each": float(each_preserve[offset].detach().cpu()),
        })
        record["event_ordinal"] = len(_ACTIVE_TRAIN_DECISIONS) + 1
        projection_records.append(record)
        _ACTIVE_TRAIN_DECISIONS.append(dict(record))

    diagnostics = {
        "full_loss": float(full_loss.detach().cpu()),
        "missing_loss": float(missing_loss.detach().cpu()),
        "kd_loss": float(kd_loss.detach().cpu()),
        "mean_gate": float(effective_gate.mean().detach().cpu()),
        "weighted_kd": float(kd_loss.detach().cpu()),
        "baseline_missing_MAE": float(
            torch.abs(memory_prediction.view(-1) - labels.view(-1)).mean().cpu()
        ),
        "safe_target_MAE": float(
            torch.abs(decision["teacher_safe_target"].view(-1) - labels.view(-1)).mean().cpu()
        ),
    }
    return total_loss, diagnostics, projection_records


def reference_prediction_rows(evaluator_bundle, teacher, loader, device):
    del teacher, loader, device
    return evaluator_bundle.valid_reference.copy()


def train_trajectory(cli, logger, output_root, model_root, seed, run):
    global _ACTIVE_TRAIN_BASELINE_FRAME, _ACTIVE_BEST_MEMORY, _ACTIVE_TRAIN_DECISIONS
    if int(seed) != DEV_SEED or run != RUN:
        raise RuntimeError("v4.2 may train only Seed1113 / regret_best_memory_cfcompat.")
    _ACTIVE_TRAIN_BASELINE_FRAME = None
    _ACTIVE_BEST_MEMORY = None
    _ACTIVE_TRAIN_DECISIONS = []
    try:
        result, epoch_rows, raw_events = _ORIGINAL_TRAIN_TRAJECTORY(
            cli, logger, output_root, model_root, seed, run
        )
        train_baseline = _ACTIVE_TRAIN_BASELINE_FRAME.copy()
        final_memory = _ACTIVE_BEST_MEMORY.to_frame()
        decisions = pd.DataFrame(_ACTIVE_TRAIN_DECISIONS)
    finally:
        _ACTIVE_TRAIN_BASELINE_FRAME = None
        _ACTIVE_BEST_MEMORY = None
        _ACTIVE_TRAIN_DECISIONS = None

    expected = 1284 * int(result["TrainEpochCount"])
    if len(train_baseline) != 1284 or len(final_memory) != 1284 * len(MISSING_MODES) or len(decisions) != expected:
        raise RuntimeError("v4.2 Train baseline/memory/decision count mismatch.")
    decisions["Epoch"] = ((decisions.event_ordinal.astype(int) - 1) // 1284) + 1

    run_dir = output_root / "seed{}".format(seed) / run
    baseline_path = run_dir / "train_frozen_moddrop_baseline_cache.csv"
    memory_path = run_dir / "train_best_so_far_memory_final.csv"
    decisions_path = run_dir / "train_best_memory_decisions.csv"
    train_baseline.to_csv(baseline_path, index=False)
    final_memory.to_csv(memory_path, index=False)
    decisions.to_csv(decisions_path, index=False)

    result.update({
        "Method": METHOD,
        "RegretAnchor": "per_sample_per_missing_mode_best_so_far_prediction_memory",
        "MemoryInitialization": "frozen_validation_best_moddrop_missing_prediction",
        "RoutingPolicy": "strong_weak_distill_memory_preserve_abstain",
        "TrainBaselineCache": str(baseline_path.resolve()),
        "TrainBaselineCacheSHA256": checkpoint_sha256(baseline_path),
        "FinalBestMemory": str(memory_path.resolve()),
        "FinalBestMemorySHA256": checkpoint_sha256(memory_path),
        "TrainDecisionRecords": str(decisions_path.resolve()),
        "TrainDecisionRecordsSHA256": checkpoint_sha256(decisions_path),
        "MemoryFinalImprovedFraction": float((final_memory.best_error < final_memory.initial_error - 1e-12).mean()),
        "MemoryFinalMeanErrorImprovement": float((final_memory.initial_error - final_memory.best_error).mean()),
        "MemoryTotalUpdateCount": int(final_memory.update_count.sum()),
    })
    for row in epoch_rows:
        local = decisions.loc[decisions.Epoch.astype(int).eq(int(row["Epoch"]))]
        if len(local) != 1284:
            raise RuntimeError("v4.2 epoch decision accounting failed.")
        row["strong_distill_fraction"] = float(local.strong_distill.astype(bool).mean())
        row["weak_distill_fraction"] = float(local.weak_distill.astype(bool).mean())
        row["preserve_fraction"] = float(local.preserve.astype(bool).mean())
        row["memory_abstain_fraction"] = float(local.memory_abstain.astype(bool).mean())
        row["memory_update_fraction"] = float(local.memory_updated.astype(bool).mean())
    pd.DataFrame(epoch_rows).to_csv(run_dir / "epoch_metrics.csv", index=False)
    return result, epoch_rows, raw_events, train_baseline, final_memory, decisions


def v4p1_reference_paths(cli):
    root = Path(cli.result_root) / "missing_baseline" / "cfcompat_regret_preserve_guard_v4p1" / cli.dataset / "valid_screen" / "seed1113_dev"
    return {
        "root": root,
        "grid": root / "regret_guard_v4p1_candidate_grid.csv",
        "raw": root / "regret_guard_v4p1_candidate_raw_valid_events.csv",
        "summary": root / "regret_guard_v4p1_valid_screen_summary.json",
        "manifest": root / "regret_guard_v4p1_source_manifest.json",
        "audit": root / "regret_guard_v4p1_audit_check.json",
    }


def load_v4p1_reference(cli):
    paths = v4p1_reference_paths(cli)
    missing = [str(path) for key, path in paths.items() if key != "root" and not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing audited v4.1 reference artifacts:\n" + "\n".join(missing))
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    if not audit.get("passed", False):
        raise RuntimeError("Frozen v4.1 audit did not pass.")
    grid = pd.read_csv(paths["grid"])
    raw = pd.read_csv(paths["raw"])
    if len(grid) != 1 or int(grid.iloc[0].Seed) != DEV_SEED or str(grid.iloc[0].Run) != "regret_preserve_guard_cfcompat":
        raise RuntimeError("Frozen v4.1 grid is not the expected Seed1113 candidate.")
    if len(raw) != 4 * 229 or set(raw.Split.astype(str)) != {"valid"}:
        raise RuntimeError("Frozen v4.1 Valid predictions are incomplete.")
    sources = {
        key: {"path": str(path.resolve()), "sha256": checkpoint_sha256(path)}
        for key, path in paths.items() if key != "root"
    }
    return grid, raw, sources


def render_report(summary):
    gate = summary.get("candidate_gate")
    lines = [
        "# Regret-Aware Best-So-Far Memory CFCompatKD v4.2", "",
        "- Development seed: `1113`",
        "- New trajectories trained: `1`",
        "- Frozen references: audited v2 + v4 + v4.1",
        "- Official Test constructed: `False`",
        "- Memory key: `Train sample x missing mode`",
        "- Memory initialization: frozen validation-best ModDrop prediction",
        "- Memory update: detached current Student only when Train-label error strictly improves",
        "- Preserve target: current best memory, safely clipped to current-Student-to-label interval",
        "- Strong/weak Teacher KD: retained from v4.1 (`>=0.02` / `(0,0.02)` at true `0.25x` weak strength)",
        "",
    ]
    if not gate:
        lines += ["Smoke-only run; no scientific promotion decision.", ""]
        return "\n".join(lines) + "\n"
    lines += [
        "## Development decision", "",
        "- Verdict: `{}`".format(summary["verdict"]),
        "- Candidate Valid-J: `{:.6f}`".format(gate["candidate_J"]),
        "- v4 Valid-J: `{:.6f}`".format(gate["v4_J"]),
        "- Better-and-correct gain change vs v4.1: `{:+.6f}`".format(gate["better_correct_change_vs_v4p1"]),
        "- Candidate negative-transfer rate: `{:.4f}`".format(gate["candidate_negative_transfer_rate"]),
        "- Candidate severe-negative-transfer rate: `{:.4f}`".format(gate["candidate_severe_negative_transfer_rate"]),
        "", "### Checks", "",
    ]
    for name, value in gate["checks"].items():
        lines.append("- {}: `{}`".format(name, value))
    lines.append("")
    return "\n".join(lines) + "\n"


def bind_hooks():
    base.RUNS = RUNS
    base.METHOD = METHOD
    base.load_assets = load_assets
    base.forward_objective = forward_objective
    base.reference_prediction_rows = reference_prediction_rows
    base.projection_summary = best_memory_projection_summary


def main():
    cli = parse_args()
    bind_hooks()
    output_root, model_root = result_paths(cli)
    logger, log_path = create_logger(cli)

    v2_grid, v2_raw, v2_sources = v4.load_v2_references(cli)
    v4_grid, v4_raw, v4_sources = v4p1.load_v4_reference(cli)
    v4p1_grid, v4p1_raw, v4p1_sources = load_v4p1_reference(cli)
    result, epoch_rows, candidate_raw, train_baseline, final_memory, train_decisions = train_trajectory(
        cli, logger, output_root, model_root, DEV_SEED, RUN
    )
    candidate_grid = pd.DataFrame([result])
    candidate_raw_frame = pd.DataFrame(candidate_raw)
    combined_raw = pd.concat(
        [v2_raw, v4_raw, v4p1_raw, candidate_raw_frame], ignore_index=True, sort=False
    )
    events = derive_valid_events(combined_raw)
    overall = overall_from_events(events)
    groups = group_summary(events)
    transfer = negative_transfer_summary(events)

    if cli.smoke_test:
        gate = None
        verdict = "SMOKE_ONLY_NO_DECISION"
    else:
        gate = dev_candidate_gate(result, v4_grid, v4p1_grid, groups, transfer, epoch_rows)
        verdict = (
            "PROMOTE_REGRET_BEST_MEMORY_V4P2_TO_3SEED_VALID_SCREEN"
            if gate["passed"]
            else "STOP_REGRET_BEST_MEMORY_V4P2_SINGLE_SEED_DEV_FAILED"
        )

    prefix = "regret_best_memory_v4p2"
    artifacts = {
        f"{prefix}_candidate_grid.csv": candidate_grid,
        f"{prefix}_all_epoch_metrics.csv": pd.DataFrame(epoch_rows),
        f"{prefix}_candidate_raw_valid_events.csv": candidate_raw_frame,
        f"{prefix}_v2_reference_grid.csv": v2_grid,
        f"{prefix}_v2_reference_raw_valid_events.csv": v2_raw,
        f"{prefix}_v4_reference_grid.csv": v4_grid,
        f"{prefix}_v4_reference_raw_valid_events.csv": v4_raw,
        f"{prefix}_v4p1_reference_grid.csv": v4p1_grid,
        f"{prefix}_v4p1_reference_raw_valid_events.csv": v4p1_raw,
        f"{prefix}_combined_valid_events.csv": events,
        f"{prefix}_overall_metrics.csv": overall,
        f"{prefix}_group_metrics.csv": groups,
        f"{prefix}_transfer_metrics.csv": transfer,
        f"{prefix}_train_baseline_cache.csv": train_baseline,
        f"{prefix}_final_memory.csv": final_memory,
        f"{prefix}_train_decisions.csv": train_decisions,
    }
    for name, frame in artifacts.items():
        frame.to_csv(output_root / name, index=False)

    source_record = {
        "seed": DEV_SEED,
        "run": RUN,
        "checkpoint": result["MainCheckpoint"], "checkpoint_sha256": result["MainCheckpointSHA256"],
        "teacher_checkpoint": result["TeacherCheckpoint"], "teacher_sha256": result["TeacherSHA256"],
        "evaluator_checkpoint": result["EvaluatorCheckpoint"], "evaluator_sha256": result["EvaluatorSHA256"],
        "evaluator_source": result["EvaluatorSource"], "evaluator_source_sha256": result["EvaluatorSourceSHA256"],
        "compatibility_cache": result["CompatibilityCache"], "compatibility_cache_sha256": result["CompatibilityCacheSHA256"],
        "compatibility_config": result["CompatibilityConfig"], "compatibility_config_sha256": result["CompatibilityConfigSHA256"],
        "train_baseline_cache": result["TrainBaselineCache"], "train_baseline_cache_sha256": result["TrainBaselineCacheSHA256"],
        "final_best_memory": result["FinalBestMemory"], "final_best_memory_sha256": result["FinalBestMemorySHA256"],
        "train_decision_records": result["TrainDecisionRecords"], "train_decision_records_sha256": result["TrainDecisionRecordsSHA256"],
    }
    manifest = {
        "version": VERSION,
        "method": METHOD,
        "base_commit": "6d4b82b4d214e63adc5185d01d9dfd4e47d3d4b6",
        "implementation_branch": "feature/cfcompat-regret-best-memory-v4p2",
        "development_seed": DEV_SEED,
        "runs": list(RUNS),
        "decision_split": "official_valid_only",
        "official_test_constructed": False,
        "test_loader_construction_count": 0,
        "test_loader_traversal_count": 0,
        "additional_inference_parameters": 0,
        "v2_reference_sources": v2_sources,
        "v4_reference_sources": v4_sources,
        "v4p1_reference_sources": v4p1_sources,
        "source_record": source_record,
        "artifacts": {},
    }
    for name in artifacts:
        path = output_root / name
        manifest["artifacts"][name] = {
            "path": str(path.resolve()), "sha256": checkpoint_sha256(path)
        }
    manifest_path = output_root / f"{prefix}_source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary = {
        "version": VERSION,
        "method": METHOD,
        "verdict": verdict,
        "candidate_gate": gate,
        "protocol": {
            "development_seed": DEV_SEED,
            "new_trajectories_trained": 1,
            "runs": list(RUNS),
            "reference_runs": [
                "cfcompat_replay", "student_safe_uniform",
                "regret_preserve_cfcompat", "regret_preserve_guard_cfcompat",
            ],
            "regret_anchor": "per_sample_per_missing_mode_best_so_far_prediction_memory",
            "memory_initialization": "frozen_validation_best_moddrop_missing_prediction",
            "memory_update_split": "train_only",
            "memory_update_rule": "replace_with_detached_current_student_if_absolute_train_label_error_strictly_improves",
            "routing_policy": "strong_weak_distill_memory_preserve_abstain",
            "weak_kd_normalization": "eligible_unscaled_cfcompat_mass_denominator_effective_scaled_numerator",
            "lambda_kd": 1.0,
            "lambda_preserve": LAMBDA_PRESERVE_V4P2,
            "optimizer": "Adam",
            "update_epochs": 10,
            "checkpoint_selection": "minimum_official_valid_J",
            "official_test_constructed": False,
            "official_test_authorized": False,
            "next_stage_on_pass": "frozen_three_seed_1112_1113_1115_valid_screen",
            "frozen_thresholds": frozen_thresholds(),
        },
    }
    summary_path = output_root / f"{prefix}_valid_screen_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path = output_root / f"{prefix}_valid_screen_report.md"
    report_path.write_text(render_report(summary), encoding="utf-8")

    logger.info("complete verdict=%s output=%s log=%s", verdict, output_root, log_path)
    print("Regret-Aware Best-So-Far Memory CFCompatKD v4.2 screen complete")
    if gate is not None:
        print("candidate J:", "{:.6f}".format(gate["candidate_J"]))
        print("v4 J:", "{:.6f}".format(gate["v4_J"]))
        print("better-correct change vs v4.1:", "{:+.6f}".format(gate["better_correct_change_vs_v4p1"]))
        print("candidate negative-transfer rate:", "{:.6f}".format(gate["candidate_negative_transfer_rate"]))
        print("candidate severe-negative-transfer rate:", "{:.6f}".format(gate["candidate_severe_negative_transfer_rate"]))
        print("decision fractions:", gate["decision_fractions"])
    print("memory final improved fraction:", "{:.6f}".format(result["MemoryFinalImprovedFraction"]))
    print("memory total update count:", result["MemoryTotalUpdateCount"])
    print("verdict:", verdict)
    print("official Test was not constructed")
    print("report:", report_path)


if __name__ == "__main__":
    main()
