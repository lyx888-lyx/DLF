"""Held-out-seed, validation-only Safe-CFCompatKD screen."""
from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_loader import MMDataLoader
from train_cf_compat_kd import (
    _flatten,
    batch_to_device,
    build_config,
    initialize_teacher_student,
    prediction_rows,
)
from trains.singleTask.HingeLoss import HingeLoss
from trains.singleTask.cf_compat_kd_utils import (
    MULTISEED_CACHE_VERSION,
    build_frozen_evaluator,
    compatibility_for_modes,
    gated_kd_loss,
    modes_from_masks,
)
from trains.singleTask.cfcompat_safe_projection_utils import (
    FORMAL_SEEDS,
    METHOD,
    OUTPUT_TAG,
    RUNS,
    VERSION,
    aggregate_candidate_gate,
    derive_valid_events,
    group_summary,
    jsonable,
    overall_from_events,
    projection_summary,
    replay_gate,
    safe_project_teacher,
)
from trains.singleTask.cfcompat_sam_utils import (
    capture_rng_state,
    load_locked_counterfactual_cache,
    portable_locate_stage1_evaluator,
    restore_rng_state,
    rng_states_equal,
)
from trains.singleTask.cfcompat_stability_utils import (
    MissingSequenceHasher,
    expected_missing_sequence_sha,
    load_stage3_reference,
    preserve_rng_state,
)
from trains.singleTask.fixed_kd_utils import (
    assert_teacher_not_in_optimizer,
    checkpoint_sha256,
    teacher_grad_count,
    teacher_lav_prediction,
)
from trains.singleTask.missing_utils import (
    MISSING_MODES,
    compute_full_dlf_loss,
    compute_task_loss,
    count_missing_modes,
    evaluate_all_modes,
    mode_to_mask,
    sample_missing_masks,
    validation_objective,
)
from trains.singleTask.model.DLF import DLF
from utils.functions import setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Safe-CFCompatKD held-out-seed Valid-only screen."
    )
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(FORMAL_SEEDS))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--log-dir", default="log/missing_baseline")
    parser.add_argument("--config-file", default="config/config.json")
    args = parser.parse_args()
    if tuple(args.seeds) != FORMAL_SEEDS:
        parser.error("Formal screen fixes held-out seeds to 1112 1113 1115.")
    if int(args.num_workers) != 1:
        parser.error("Formal screen fixes num_workers=1.")
    if args.smoke_test:
        args.max_epochs = 2 if args.max_epochs is None else min(2, args.max_epochs)
    elif args.max_epochs is not None:
        parser.error("Formal runs use the frozen DLF early-stop rule.")
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
    output.mkdir(parents=True, exist_ok=True)
    model.mkdir(parents=True, exist_ok=True)
    return output, model


def create_logger(cli):
    directory = Path(cli.log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    kind = "smoke" if cli.smoke_test else "formal"
    path = directory / "DLF-mosi-safe-cfcompat-valid-screen-{}-{}.log".format(
        kind, datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    logger = logging.getLogger("safe_cfcompat_valid_screen")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def run_tag(run):
    if run not in RUNS:
        raise ValueError("Unknown Safe-CFCompat run: {}".format(run))
    return run


def load_assets(cli, args, loaders, seed):
    evaluator_checkpoint, evaluator_best_epoch, evaluator_source = (
        portable_locate_stage1_evaluator(
            cli.result_root,
            cli.dataset,
            seed,
            multiseed=True,
            smoke=False,
        )
    )
    evaluator_sha = checkpoint_sha256(evaluator_checkpoint)
    cache_frame, cache_by_index, cache_paths = load_locked_counterfactual_cache(
        cli.result_root,
        cli.dataset,
        version=MULTISEED_CACHE_VERSION,
        seed=seed,
        expected_evaluator_sha=evaluator_sha,
    )
    if len(cache_frame) != 1284:
        raise RuntimeError("Safe-CFCompat requires the locked 1284-sample cache.")

    teacher, student, teacher_checkpoint, teacher_sha = initialize_teacher_student(
        args, cli, seed, loaders
    )

    # Constructing a second frozen model may consume initialization RNG.  Make
    # it observationally invisible so the replay trajectory remains exact.
    state_before = capture_rng_state()
    evaluator = build_frozen_evaluator(DLF, args, evaluator_checkpoint)
    state_after = capture_rng_state()
    restore_rng_state(state_before)
    if not rng_states_equal(state_before, capture_rng_state()):
        raise RuntimeError("Restoring RNG after evaluator construction failed.")
    if rng_states_equal(state_before, state_after):
        evaluator_construction_consumed_rng = False
    else:
        evaluator_construction_consumed_rng = True
    evaluator.eval()
    for parameter in evaluator.parameters():
        parameter.requires_grad_(False)

    return teacher, student, evaluator, {
        "teacher_checkpoint": teacher_checkpoint,
        "teacher_sha": teacher_sha,
        "evaluator_checkpoint": evaluator_checkpoint,
        "evaluator_sha": evaluator_sha,
        "evaluator_best_epoch": evaluator_best_epoch,
        "evaluator_source": evaluator_source,
        "cache_by_index": cache_by_index,
        "cache_csv": cache_paths["csv"],
        "cache_config": cache_paths["config"],
        "evaluator_construction_consumed_rng": evaluator_construction_consumed_rng,
    }


def _frozen_baseline_prediction(evaluator, text, audio, vision, missing_mask):
    with preserve_rng_state():
        with torch.inference_mode():
            prediction = evaluator(text, audio, vision, missing_mask)["output_logit"]
    return prediction.detach().view(-1, 1)


def forward_objective(
    run,
    batch,
    missing_mask,
    modes,
    args,
    teacher,
    evaluator,
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
        cache_by_index, indices, list(modes), args.device, labels.dtype
    ).view(-1)

    projection_records = []
    if run == "cfcompat_replay":
        kd_target = teacher_prediction.detach().view(-1, 1)
        gate = compatibility
        baseline_prediction = None
    else:
        baseline_prediction = _frozen_baseline_prediction(
            evaluator, text, audio, vision, missing_mask
        )
        kd_target, diagnostics = safe_project_teacher(
            baseline_prediction, teacher_prediction, labels
        )
        gate = (
            torch.ones_like(compatibility)
            if run == "safe_uniform"
            else compatibility
        )
        for offset in range(labels.size(0)):
            projection_records.append(
                {
                    key: (
                        float(value[offset].detach().cpu())
                        if value.dtype != torch.bool
                        else bool(value[offset].detach().cpu())
                    )
                    for key, value in diagnostics.items()
                }
            )

    kd_loss, each_kd = gated_kd_loss(
        missing_output["output_logit"], kd_target, gate
    )
    total_loss = full_loss + missing_loss + kd_loss
    if not torch.isfinite(total_loss):
        raise FloatingPointError("NaN/Inf in Safe-CFCompat objective.")

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
    }
    if baseline_prediction is not None:
        diagnostics["baseline_missing_MAE"] = float(
            torch.abs(baseline_prediction.view(-1) - labels.view(-1)).mean().cpu()
        )
        diagnostics["safe_target_MAE"] = float(
            torch.abs(kd_target.view(-1) - labels.view(-1)).mean().cpu()
        )
    else:
        diagnostics["baseline_missing_MAE"] = float("nan")
        diagnostics["safe_target_MAE"] = float("nan")
    return total_loss, diagnostics, projection_records


def reference_prediction_rows(evaluator, teacher, loader, device):
    evaluator.eval()
    teacher.eval()
    rows = []
    with torch.inference_mode():
        for batch in loader:
            text, audio, vision, labels = batch_to_device(batch, device)
            teacher_prediction = teacher_lav_prediction(
                teacher, text, audio, vision
            ).view(-1)
            baseline = {}
            for mode in ("LAV",) + MISSING_MODES:
                mask = mode_to_mask(mode, labels.size(0), device, audio.dtype)
                baseline[mode] = evaluator(
                    text, audio, vision, mask
                )["output_logit"].view(-1)
            indices = batch["index"].view(-1).cpu().numpy().astype(int)
            ids = list(batch["id"])
            for offset, index in enumerate(indices):
                row = {
                    "sample_index": int(index),
                    "sample_id": str(ids[offset]),
                    "label": float(labels[offset].item()),
                    "teacher_prediction": float(
                        teacher_prediction[offset].detach().cpu()
                    ),
                }
                for mode in ("LAV",) + MISSING_MODES:
                    row[f"baseline_{mode}_pred"] = float(
                        baseline[mode][offset].detach().cpu()
                    )
                rows.append(row)
    return pd.DataFrame(rows).sort_values("sample_index", kind="mergesort")


def raw_events_for_run(seed, run, candidate_predictions, reference_predictions):
    merged = candidate_predictions.merge(
        reference_predictions,
        on=["sample_index", "label"],
        how="inner",
        suffixes=("_candidate", "_reference"),
        validate="one_to_one",
    )
    if len(merged) != len(candidate_predictions):
        raise RuntimeError("Valid candidate/reference prediction binding failed.")
    rows = []
    for row in merged.itertuples(index=False):
        sample_id = getattr(row, "sample_id_candidate", getattr(row, "sample_id", ""))
        for mode in ("LAV",) + MISSING_MODES:
            rows.append(
                {
                    "Seed": int(seed),
                    "Run": str(run),
                    "Mode": mode,
                    "sample_index": int(row.sample_index),
                    "sample_id": str(sample_id),
                    "label": float(row.label),
                    "baseline_prediction": float(
                        getattr(row, f"baseline_{mode}_pred")
                    ),
                    "candidate_prediction": float(getattr(row, f"{mode}_pred")),
                    "teacher_prediction": float(row.teacher_prediction),
                    "Split": "valid",
                    "SelectedBy": "validation_J",
                }
            )
    return rows


def train_trajectory(cli, logger, output_root, model_root, seed, run):
    setup_seed(seed)
    args = build_config(cli, seed)
    if int(args.update_epochs) != 10:
        raise RuntimeError("Safe-CFCompat fixes original update_epochs=10.")
    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("Safe-CFCompat may construct only train/valid loaders.")
    teacher, student, evaluator, assets = load_assets(cli, args, loaders, seed)
    optimizer = optim.Adam(student.parameters(), lr=args.learning_rate)
    assert_teacher_not_in_optimizer(teacher, optimizer)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if any(id(parameter) in optimizer_ids for parameter in evaluator.parameters()):
        raise RuntimeError("Frozen baseline evaluator entered the optimizer.")
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience
    )
    criterion = nn.L1Loss()
    cosine = nn.CosineEmbeddingLoss()
    hinge = HingeLoss()
    missing_generator = torch.Generator().manual_seed(int(seed) + 104729)
    missing_hasher = MissingSequenceHasher()

    run_dir = output_root / f"seed{seed}" / run_tag(run)
    checkpoint = model_root / f"seed{seed}" / run_tag(run) / "best_valid.pth"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    best_j = float("inf")
    best_epoch = 0
    best_metrics = None
    epoch_rows = []
    batch_sizes = None
    last_epoch = 0
    all_projection_records = []
    logger.info(
        "seed=%s run=%s objective=%s test=forbidden",
        seed,
        run,
        "original_cfcompat" if run == "cfcompat_replay" else "safe_projection",
    )

    for epoch in range(1, (cli.max_epochs or 1000) + 1):
        last_epoch = epoch
        student.train()
        optimizer.zero_grad()
        counts = Counter({"LA": 0, "LV": 0, "L": 0})
        epoch_batch_sizes = []
        objective_rows = []
        epoch_projection_records = []

        for step, batch in enumerate(loaders["train"], 1):
            labels_cpu = batch["labels"]["M"].view(-1, 1)
            batch_size = int(labels_cpu.size(0))
            epoch_batch_sizes.append(batch_size)
            missing_mask = sample_missing_masks(
                batch_size,
                missing_generator,
                torch.device("cpu"),
                torch.float32,
            )
            modes = tuple(modes_from_masks(missing_mask))
            missing_hasher.update(modes)
            counts.update(count_missing_modes(missing_mask))
            loss, diagnostics, projection_records = forward_objective(
                run,
                batch,
                missing_mask,
                modes,
                args,
                teacher,
                evaluator,
                student,
                assets["cache_by_index"],
                criterion,
                cosine,
                hinge,
            )
            loss.backward()
            if teacher_grad_count(teacher) or teacher_grad_count(evaluator):
                raise RuntimeError("A frozen model received gradients.")
            if step % int(args.update_epochs) == 0 or step == len(loaders["train"]):
                optimizer.step()
                optimizer.zero_grad()
            objective_rows.append(diagnostics)
            epoch_projection_records.extend(projection_records)

        if batch_sizes is None:
            batch_sizes = epoch_batch_sizes
        elif batch_sizes != epoch_batch_sizes:
            raise RuntimeError("Batch-size sequence changed across epochs.")

        valid = evaluate_all_modes(
            student, loaders["valid"], args.device, "moddrop", criterion
        )
        j_valid = validation_objective(valid)
        if not math.isfinite(j_valid):
            raise FloatingPointError("Non-finite Valid J.")
        scheduler.step(j_valid)
        is_best = j_valid <= best_j - 1e-6
        if is_best:
            best_j = j_valid
            best_epoch = epoch
            best_metrics = valid
            torch.save(student.state_dict(), checkpoint)

        local_projection = (
            projection_summary(epoch_projection_records)
            if epoch_projection_records
            else {
                "sample_count": 0,
                "wrong_direction_fraction": 0.0,
                "overshoot_fraction": 0.0,
                "zero_width_fraction": 0.0,
                "unchanged_fraction": 1.0,
                "projected_to_baseline_fraction": 0.0,
                "projected_to_label_fraction": 0.0,
                "mean_teacher_target_abs_shift": 0.0,
                "mean_safe_interval_width": 0.0,
            }
        )
        all_projection_records.extend(epoch_projection_records)
        row = {
            "Seed": int(seed),
            "Run": run,
            "Epoch": int(epoch),
            "J_valid": float(j_valid),
            "IsBestValid": bool(is_best),
            "full_loss": float(np.mean([item["full_loss"] for item in objective_rows])),
            "missing_loss": float(np.mean([item["missing_loss"] for item in objective_rows])),
            "KD_loss": float(np.mean([item["kd_loss"] for item in objective_rows])),
            "mean_gate": float(np.mean([item["mean_gate"] for item in objective_rows])),
            "baseline_missing_MAE": float(
                np.nanmean([item["baseline_missing_MAE"] for item in objective_rows])
            ) if run != "cfcompat_replay" else float("nan"),
            "safe_target_MAE": float(
                np.nanmean([item["safe_target_MAE"] for item in objective_rows])
            ) if run != "cfcompat_replay" else float("nan"),
            **local_projection,
            **_flatten(valid, "valid"),
        }
        epoch_rows.append(row)
        logger.info(
            "seed=%s run=%s epoch=%s J_valid=%.6f LAV=%.6f MissingMacro=%.6f KD=%.6f wrong=%.4f overshoot=%.4f unchanged=%.4f",
            seed,
            run,
            epoch,
            j_valid,
            valid["LAV"]["MAE"],
            np.mean([valid[mode]["MAE"] for mode in MISSING_MODES]),
            row["KD_loss"],
            row["wrong_direction_fraction"],
            row["overshoot_fraction"],
            row["unchanged_fraction"],
        )
        if epoch - best_epoch >= args.early_stop:
            break

    if not checkpoint.is_file() or best_metrics is None:
        raise RuntimeError("Validation-best checkpoint is absent.")
    expected_sha, expected_count = expected_missing_sequence_sha(
        seed, last_epoch, batch_sizes
    )
    if (
        missing_hasher.hexdigest() != expected_sha
        or missing_hasher.count != expected_count
    ):
        raise RuntimeError("Missing-mode sequence hash differs from Stage 3.")

    student.load_state_dict(torch.load(checkpoint, map_location=args.device), strict=True)
    final_valid = evaluate_all_modes(
        student, loaders["valid"], args.device, "moddrop", criterion
    )
    candidate_predictions = prediction_rows(student, loaders["valid"], args.device)
    reference_predictions = reference_prediction_rows(
        evaluator, teacher, loaders["valid"], args.device
    )
    raw_events = raw_events_for_run(
        seed, run, candidate_predictions, reference_predictions
    )
    pd.DataFrame(epoch_rows).to_csv(run_dir / "epoch_metrics.csv", index=False)
    pd.DataFrame(raw_events).to_csv(run_dir / "valid_raw_events.csv", index=False)

    overall_projection = (
        projection_summary(all_projection_records)
        if all_projection_records
        else {
            "sample_count": 0,
            "wrong_direction_fraction": 0.0,
            "overshoot_fraction": 0.0,
            "zero_width_fraction": 0.0,
            "unchanged_fraction": 1.0,
            "projected_to_baseline_fraction": 0.0,
            "projected_to_label_fraction": 0.0,
            "mean_teacher_target_abs_shift": 0.0,
            "mean_safe_interval_width": 0.0,
        }
    )
    result = {
        "Seed": int(seed),
        "Run": run,
        "Method": (
            "DLF-CFCompatKD-v1"
            if run == "cfcompat_replay"
            else ("DLF-Safe-UniformKD-v1" if run == "safe_uniform" else METHOD)
        ),
        "BestValidEpoch": int(best_epoch),
        "TrainEpochCount": int(last_epoch),
        "J_valid": float(validation_objective(final_valid)),
        "MainCheckpoint": str(checkpoint.resolve()),
        "MainCheckpointSHA256": checkpoint_sha256(checkpoint),
        "MissingSequenceSHA256": missing_hasher.hexdigest(),
        "MissingSequenceCount": int(missing_hasher.count),
        "TeacherCheckpoint": str(Path(assets["teacher_checkpoint"]).resolve()),
        "TeacherSHA256": assets["teacher_sha"],
        "EvaluatorCheckpoint": str(Path(assets["evaluator_checkpoint"]).resolve()),
        "EvaluatorSHA256": assets["evaluator_sha"],
        "EvaluatorBestEpoch": int(assets["evaluator_best_epoch"]),
        "EvaluatorSource": str(Path(assets["evaluator_source"]).resolve()),
        "EvaluatorSourceSHA256": checkpoint_sha256(assets["evaluator_source"]),
        "CompatibilityCache": str(Path(assets["cache_csv"]).resolve()),
        "CompatibilityCacheSHA256": checkpoint_sha256(assets["cache_csv"]),
        "CompatibilityConfig": str(Path(assets["cache_config"]).resolve()),
        "CompatibilityConfigSHA256": checkpoint_sha256(assets["cache_config"]),
        "EvaluatorConstructionConsumedRNG": bool(
            assets["evaluator_construction_consumed_rng"]
        ),
        "TestConstructed": False,
        **{f"projection_{key}": value for key, value in overall_projection.items()},
        **_flatten(final_valid, "valid"),
    }
    return result, epoch_rows, raw_events


def render_report(summary):
    lines = [
        "# Safe-CFCompatKD v1: held-out-seed Valid screen",
        "",
        "## Frozen decision",
        "",
        "- Verdict: `{}`".format(summary["verdict"]),
        "- Formal seeds: `1112, 1113, 1115`",
        "- Official Test constructed: `False`",
        "- Main candidate: `safe_cfcompat`",
        "- Ablation: `safe_uniform`",
        "",
        "## Candidate gates",
        "",
    ]
    for run in ("safe_uniform", "safe_cfcompat"):
        gate = summary["candidate_gates"][run]
        lines.extend(
            [
                "### {}".format(run),
                "",
                "- Passed: `{}`".format(gate["passed"]),
                "- Mean Valid-J gain versus original CFCompatKD: `{:+.6f}`".format(
                    gate["mean_gain_valid_J_vs_CFCompatKD"]
                ),
                "",
            ]
        )
        for seed in FORMAL_SEEDS:
            item = gate["per_seed"][str(seed)]
            lines.append(
                "- Seed {} gain: `{:+.6f}`".format(
                    seed, item["gain_valid_J_vs_CFCompatKD"]
                )
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    cli = parse_args()
    output_root, model_root = result_paths(cli)
    logger, log_path = create_logger(cli)
    grid_rows = []
    epoch_rows = []
    raw_events = []
    stage3_references = []
    replay_checks = {}

    for seed in FORMAL_SEEDS:
        reference_series, reference_path = load_stage3_reference(cli.result_root, seed)
        reference = {str(key): jsonable(value) for key, value in reference_series.items()}
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
            if run == "cfcompat_replay":
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
    gates = {
        run: aggregate_candidate_gate(run, grid_rows, epoch_rows, groups)
        for run in ("safe_uniform", "safe_cfcompat")
    }
    if gates["safe_cfcompat"]["passed"]:
        verdict = "PROMOTE_SAFE_CFCompatKD_TO_MOSEI_SINGLE_SEED_VALID_SCREEN"
    elif gates["safe_uniform"]["passed"]:
        verdict = "SAFE_PROJECTION_ABLATION_PASSED_CFCompat_EXTENSION_FAILED"
    else:
        verdict = "STOP_SAFE_CFCompatKD_HELDOUT_VALID_FAILED"

    artifacts = {
        "safe_projection_valid_grid_summary.csv": pd.DataFrame(grid_rows),
        "safe_projection_all_epoch_metrics.csv": pd.DataFrame(epoch_rows),
        "safe_projection_raw_valid_events.csv": raw_frame,
        "safe_projection_valid_events.csv": events,
        "safe_projection_overall_metrics.csv": overall,
        "safe_projection_group_metrics.csv": groups,
        "safe_projection_stage3_references.csv": pd.DataFrame(stage3_references),
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
        "base_branch": "feature/cfcompat-sam-valid-screen-v1",
        "formal_seeds": list(FORMAL_SEEDS),
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
    (output_root / "safe_projection_source_manifest.json").write_text(
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
            "formal_seeds": list(FORMAL_SEEDS),
            "runs": list(RUNS),
            "safe_target": "teacher_clipped_to_closed_DLF_prediction_to_train_label_interval",
            "safe_uniform_uses_compatibility": False,
            "safe_cfcompat_uses_compatibility": True,
            "lambda_kd": 1.0,
            "optimizer": "Adam",
            "update_epochs": 10,
            "checkpoint_selection": "minimum_official_valid_J",
            "official_test_constructed": False,
            "official_test_authorized": False,
            "next_stage_on_main_pass": "MOSEI_single_seed_valid_screen",
            "additional_inference_parameters": 0,
        },
    }
    (output_root / "safe_projection_valid_screen_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "safe_projection_valid_screen_report.md").write_text(
        render_report(summary), encoding="utf-8"
    )
    logger.info(
        "complete verdict=%s safe_cfcompat_mean_gain=%+.6f output=%s log=%s",
        verdict,
        gates["safe_cfcompat"]["mean_gain_valid_J_vs_CFCompatKD"],
        output_root,
        log_path,
    )
    print("Safe-CFCompatKD held-out-seed Valid screen complete")
    print(
        "safe-uniform mean Valid J gain:",
        "{:+.6f}".format(
            gates["safe_uniform"]["mean_gain_valid_J_vs_CFCompatKD"]
        ),
    )
    print(
        "safe-cfcompat mean Valid J gain:",
        "{:+.6f}".format(
            gates["safe_cfcompat"]["mean_gain_valid_J_vs_CFCompatKD"]
        ),
    )
    print("verdict:", verdict)
    print("official Test was not constructed")
    print("report:", output_root / "safe_projection_valid_screen_report.md")


if __name__ == "__main__":
    main()
