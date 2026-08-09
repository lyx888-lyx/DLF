"""CFCompatKD v13: Train-only sentinel safety on the actual Adam parameter step.

v13 is a metric-bearing candidate motivated by the v12.3 window audit.  It
keeps the v12 residual architecture, objective, missing RNG, raw asymmetric
gradient surgery, Adam optimizer, five video-grouped folds, Train-holdout
checkpoint selector, and 4-of-5 inference consensus unchanged.

The added mechanism is an actual-step functional-safety layer:

1. Before each fold trains, build two fixed sentinel groups from only that
   fold's Train 4/5 videos: historical Teacher-beneficial events and historical
   Frozen-S0-beneficial events (both use the already-established 0.02 margins).
2. Cache frozen S0 residual features for every sentinel LA/LV/L event so the
   safety gradient is cheap and persistent at every optimizer window.
3. Run the exact v12 raw gradient surgery and let Adam update its moments and
   propose its real parameter displacement.
4. Before accepting that displacement, project it per missing mode onto the
   intersection of the Teacher-beneficial and S0-beneficial MAE safety
   halfspaces.  Adam's internal state is not rewritten; every future proposed
   step is checked again.

No epoch switch, safety strength, Valid calibration, inference gate, or new
selection threshold is introduced.  Official Valid is first used after all five
fold models are frozen.  Official Test is never constructed or accessed.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from data_loader import MMDataLoader
from train_cf_compat_kd import _flatten, batch_to_device, build_config
import train_cfcompat_gradient_surgery_valid_screen_v12 as v12
import train_cfcompat_regret_preserve_valid_screen as v4
import train_cfcompat_safe_projection_valid_screen as base
import train_cfcompat_frozen_backbone_adapter_isolation_valid_screen_v7 as v7
import train_cfcompat_crossfit_residual_consensus_valid_screen_v9 as v9
import train_cfcompat_conservative_crossfit_residual_valid_screen_v10 as v10
from trains.singleTask.HingeLoss import HingeLoss
from trains.singleTask.cfcompat_adapter_isolation_utils import mechanism_transfer_summary
from trains.singleTask.cfcompat_crossfit_residual_consensus_utils import (
    FrozenS0CrossfitConsensus,
    bank_state_cpu,
    consensus_summary,
    deterministic_video_group_folds,
    frozen_features_from_base,
    module_state_sha256,
)
from trains.singleTask.cfcompat_gradient_surgery_numerical_hotfix import (
    asymmetric_project_supervised_stable,
)
from trains.singleTask.cfcompat_gradient_surgery_utils import (
    add_gradient_tuples,
    assign_gradient_tuple,
    detached_gradient_tuple,
    development_signal_gate,
    summarize_surgery_windows,
    zero_gradient_tuple,
)
from trains.singleTask.cfcompat_adam_step_safety_utils import (
    BENEFICIAL_NTR_MAX_DEGRADATION_VS_V8,
    CONSENSUS_MIN_AGREE,
    DEV_SEED,
    DISTILL_MARGIN,
    J_MAX_DEGRADATION_VS_V8,
    METHOD,
    NEAR_OPTIMAL_REL_TOL,
    NONBENEFICIAL_NTR_MAX_DEGRADATION_VS_V12,
    NONBENEFICIAL_NTR_REDUCTION_REQUIRED_VS_V8,
    N_FOLDS,
    OUTPUT_TAG,
    OVERALL_NTR_MAX_DEGRADATION_VS_V4,
    RUN,
    S0_PROTECT_MARGIN,
    VERSION,
    clone_tensor_tuple,
    direct_v12_metric_improvement_gate,
    flatten_tensor_tuple,
    jsonable,
    project_two_halfspaces,
    unflatten_like,
    vector_cosine,
)
from trains.singleTask.cfcompat_regret_preserve_utils import (
    LAMBDA_PRESERVE,
    MILD_CFCOMPAT_BASE,
    MILD_CFCOMPAT_SCALE,
    PRESERVE_MARGIN,
    regret_projection_summary,
)
from trains.singleTask.cfcompat_sample_residual_utils import (
    MAX_ABS_RESIDUAL,
    RESIDUAL_HIDDEN_DIM,
)
from trains.singleTask.cfcompat_stability_utils import MissingSequenceHasher
from trains.singleTask.fixed_kd_utils import (
    assert_teacher_not_in_optimizer,
    checkpoint_sha256,
    teacher_grad_count,
    teacher_lav_prediction,
)
from trains.singleTask.missing_utils import (
    MISSING_MODES,
    count_missing_modes,
    evaluate_all_modes,
    mode_to_mask,
    sample_missing_masks,
    validation_objective,
)
from utils.functions import setup_seed


SAFETY_GROUP_ORDER = ("TEACHER_BENEFICIAL", "S0_BENEFICIAL")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Adam-step functional-safety CFCompatKD v13"
    )
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--log-dir", default="log/missing_baseline")
    parser.add_argument("--config-file", default="config/config.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if int(args.num_workers) != 1:
        parser.error("v13 keeps formal v12 num_workers=1 for Train/holdout loaders.")
    args.seeds = [DEV_SEED]
    args.max_epochs = 2 if args.smoke_test else None
    return args


def result_paths(cli):
    output = (
        Path(cli.result_root)
        / "missing_baseline"
        / OUTPUT_TAG
        / cli.dataset
        / "valid_screen"
        / "seed1113_dev"
    )
    model = (
        Path(cli.model_save_dir)
        / "missing_baseline"
        / OUTPUT_TAG
        / cli.dataset
        / "valid_screen"
        / "seed1113_dev"
    )
    if cli.smoke_test:
        output, model = output / "smoke", model / "smoke"
    if output.exists() or model.exists():
        if not cli.overwrite:
            raise FileExistsError(
                "v13 output exists; inspect it or use --overwrite: {} / {}".format(
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
    path = directory / "DLF-mosi-adam-step-safety-v13-{}-{}.log".format(
        kind, datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    logger = logging.getLogger("adam_step_safety_v13")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def load_v12_reference(cli):
    root = (
        Path(cli.result_root)
        / "missing_baseline"
        / "cfcompat_gradient_surgery_v12"
        / cli.dataset
        / "valid_screen"
        / "seed1113_dev"
    )
    grid_path = root / "gradient_surgery_v12_candidate_grid.csv"
    raw_path = root / "gradient_surgery_v12_candidate_raw_valid_events.csv"
    summary_path = root / "gradient_surgery_v12_valid_screen_summary.json"
    if not grid_path.is_file() or not raw_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Missing frozen formal v12 artifacts under {}".format(root))
    grid = pd.read_csv(grid_path)
    raw = pd.read_csv(raw_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if len(grid) != 1 or int(grid.iloc[0].Seed) != DEV_SEED:
        raise RuntimeError("Frozen v12 grid is not unique Seed1113.")
    if set(raw.Split.astype(str)) != {"valid"}:
        raise RuntimeError("Frozen v12 raw events contain non-Valid split.")
    if bool(summary.get("protocol", {}).get("official_test_accessed", True)):
        raise RuntimeError("Frozen v12 summary does not certify Test isolation.")
    return grid.iloc[0].to_dict(), raw, summary


def make_safety_loader(train_dataset, train_indices, batch_size):
    # This loader is separate from the shuffled training loader and uses no RNG;
    # building the sentinel cache cannot consume the v12 Train sampler sequence.
    return DataLoader(
        Subset(train_dataset, list(train_indices)),
        batch_size=int(batch_size),
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )


def build_fold_sentinel_cache(
    fold,
    student,
    safety_loader,
    teacher,
    evaluator_bundle,
    args,
):
    """Freeze per-mode Teacher/S0-beneficial Train-only residual features."""
    student.s0.eval()
    teacher.eval()
    pieces = {
        (group, mode): {"features": [], "mask": [], "s0_prediction": [], "label": [], "sample_index": []}
        for group in SAFETY_GROUP_ORDER
        for mode in MISSING_MODES
    }
    overlap_counts = Counter()
    total_mode_events = Counter()

    with torch.no_grad():
        for batch in safety_loader:
            text, audio, vision, labels = batch_to_device(batch, args.device)
            indices = batch["index"].view(-1).cpu().numpy().astype(int).tolist()
            label_flat = labels.view(-1)
            teacher_prediction = teacher_lav_prediction(
                teacher, text, audio, vision
            ).view(-1)
            teacher_error = torch.abs(teacher_prediction - label_flat)
            batch_size = int(labels.size(0))

            for mode in MISSING_MODES:
                mode_list = [mode] * batch_size
                mask = mode_to_mask(mode, batch_size, args.device, audio.dtype)
                base_output = student.s0(text, audio, vision, mask)
                features = frozen_features_from_base(base_output, student.feature_dim)
                s0_prediction = base_output["output_logit"].view(-1)
                baseline = v4.baseline_for_modes(
                    evaluator_bundle,
                    indices,
                    mode_list,
                    labels,
                    args.device,
                    labels.dtype,
                ).view(-1)
                baseline_error = torch.abs(baseline - label_flat)
                s0_error = torch.abs(s0_prediction - label_flat)
                teacher_beneficial = (
                    baseline_error - teacher_error >= DISTILL_MARGIN
                )
                s0_beneficial = (
                    baseline_error - s0_error >= S0_PROTECT_MARGIN
                )
                selections = {
                    "TEACHER_BENEFICIAL": teacher_beneficial,
                    "S0_BENEFICIAL": s0_beneficial,
                }
                total_mode_events[mode] += batch_size
                overlap_counts[mode] += int((teacher_beneficial & s0_beneficial).sum().cpu())

                for group, select in selections.items():
                    if not bool(select.any()):
                        continue
                    target = pieces[(group, mode)]
                    target["features"].append(features[select].detach().cpu())
                    target["mask"].append(mask[select].detach().cpu())
                    target["s0_prediction"].append(s0_prediction[select].detach().cpu())
                    target["label"].append(label_flat[select].detach().cpu())
                    selected_indices = torch.as_tensor(indices, dtype=torch.long)[select.detach().cpu()]
                    target["sample_index"].append(selected_indices.clone())

    cache = {}
    manifest_rows = []
    for group in SAFETY_GROUP_ORDER:
        for mode in MISSING_MODES:
            source = pieces[(group, mode)]
            if not source["features"]:
                raise RuntimeError(
                    "Empty Train-only sentinel group fold={} group={} mode={}".format(
                        fold, group, mode
                    )
                )
            entry = {
                "features": torch.cat(source["features"], dim=0).to(args.device),
                "mask": torch.cat(source["mask"], dim=0).to(args.device),
                "s0_prediction": torch.cat(source["s0_prediction"], dim=0).to(args.device),
                "label": torch.cat(source["label"], dim=0).to(args.device),
                "sample_index": torch.cat(source["sample_index"], dim=0),
            }
            count = int(entry["label"].numel())
            if entry["features"].size(0) != count or entry["mask"].size(0) != count:
                raise RuntimeError("Sentinel tensor length mismatch.")
            cache[(group, mode)] = entry
            manifest_rows.append(
                {
                    "Fold": int(fold),
                    "Group": str(group),
                    "Mode": str(mode),
                    "N": count,
                    "TrainModeEventN": int(total_mode_events[mode]),
                    "Prevalence": float(count / max(int(total_mode_events[mode]), 1)),
                    "SelectionMargin": float(
                        DISTILL_MARGIN if group == "TEACHER_BENEFICIAL" else S0_PROTECT_MARGIN
                    ),
                }
            )
    for mode in MISSING_MODES:
        manifest_rows.append(
            {
                "Fold": int(fold),
                "Group": "TEACHER_AND_S0_BENEFICIAL_OVERLAP",
                "Mode": str(mode),
                "N": int(overlap_counts[mode]),
                "TrainModeEventN": int(total_mode_events[mode]),
                "Prevalence": float(overlap_counts[mode] / max(int(total_mode_events[mode]), 1)),
                "SelectionMargin": float(DISTILL_MARGIN),
            }
        )
    return cache, pd.DataFrame(manifest_rows)


def sentinel_loss_and_gradient(student, entry, parameters):
    delta = student.bank.delta(entry["features"], entry["mask"]).view(-1)
    prediction = entry["s0_prediction"].view(-1) + delta
    label = entry["label"].view(-1)
    loss = torch.mean(torch.abs(prediction - label))
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite sentinel safety loss.")
    gradient = detached_gradient_tuple(loss, parameters, retain_graph=False)
    return float(loss.detach().cpu()), gradient


def adam_step_with_actual_safety(
    student,
    optimizer,
    parameters,
    raw_update_gradient,
    sentinel_cache,
):
    """Let Adam propose a step, then project the real displacement for safety."""
    safety = {}
    for mode in MISSING_MODES:
        safety[mode] = {}
        for group in SAFETY_GROUP_ORDER:
            loss, gradient = sentinel_loss_and_gradient(
                student, sentinel_cache[(group, mode)], parameters
            )
            safety[mode][group] = {
                "loss": loss,
                "gradient_tuple": gradient,
                "gradient_vector": flatten_tensor_tuple(gradient),
            }

    before = clone_tensor_tuple(parameters)
    optimizer.step()
    proposed_after = clone_tensor_tuple(parameters)
    proposed_delta_tuple = tuple(
        after - prior for prior, after in zip(before, proposed_after)
    )
    proposed_delta = flatten_tensor_tuple(proposed_delta_tuple)
    raw_update_vector = flatten_tensor_tuple(raw_update_gradient)
    proposed_effective = -proposed_delta
    raw_to_adam_cosine = vector_cosine(raw_update_vector, proposed_effective)

    final_delta = proposed_delta.clone()
    mode_rows = []
    for mode in MISSING_MODES:
        teacher_gradient = safety[mode]["TEACHER_BENEFICIAL"]["gradient_vector"]
        s0_gradient = safety[mode]["S0_BENEFICIAL"]["gradient_vector"]
        raw_teacher_dot = float(torch.dot(raw_update_vector, teacher_gradient).detach().cpu())
        raw_s0_dot = float(torch.dot(raw_update_vector, s0_gradient).detach().cpu())
        projected, diagnostic = project_two_halfspaces(
            final_delta,
            teacher_gradient,
            s0_gradient,
        )
        final_delta = projected
        mode_rows.append(
            {
                "Mode": str(mode),
                "teacher_sentinel_loss_before": float(safety[mode]["TEACHER_BENEFICIAL"]["loss"]),
                "s0_sentinel_loss_before": float(safety[mode]["S0_BENEFICIAL"]["loss"]),
                "raw_surgery_teacher_gradient_dot": raw_teacher_dot,
                "raw_surgery_s0_gradient_dot": raw_s0_dot,
                "raw_surgery_to_adam_effective_cosine": float(raw_to_adam_cosine),
                **diagnostic,
            }
        )

    final_delta_tuple = unflatten_like(final_delta, before)
    with torch.no_grad():
        for parameter, prior, displacement in zip(parameters, before, final_delta_tuple):
            parameter.copy_(prior + displacement)

    # Final functional-safety sign audit at the exact pre-step gradients.
    for row, mode in zip(mode_rows, MISSING_MODES):
        teacher_gradient = safety[mode]["TEACHER_BENEFICIAL"]["gradient_vector"]
        s0_gradient = safety[mode]["S0_BENEFICIAL"]["gradient_vector"]
        teacher_dot = float(torch.dot(final_delta, teacher_gradient).detach().cpu())
        s0_dot = float(torch.dot(final_delta, s0_gradient).detach().cpu())
        scale_teacher = max(
            float(torch.linalg.vector_norm(final_delta) * torch.linalg.vector_norm(teacher_gradient)),
            1.0,
        )
        scale_s0 = max(
            float(torch.linalg.vector_norm(final_delta) * torch.linalg.vector_norm(s0_gradient)),
            1.0,
        )
        tolerance = 128.0 * torch.finfo(torch.float64).eps
        if teacher_dot > tolerance * scale_teacher or s0_dot > tolerance * scale_s0:
            raise RuntimeError(
                "Final v13 Adam displacement violates sentinel safety in mode {}: {} / {}".format(
                    mode, teacher_dot, s0_dot
                )
            )
        row["final_teacher_gradient_dot_global"] = teacher_dot
        row["final_s0_gradient_dot_global"] = s0_dot

    proposed_norm = float(torch.linalg.vector_norm(proposed_delta).detach().cpu())
    final_norm = float(torch.linalg.vector_norm(final_delta).detach().cpu())
    removed_norm = float(torch.linalg.vector_norm(proposed_delta - final_delta).detach().cpu())
    projected_mode_count = int(sum(bool(row["projected"]) for row in mode_rows))
    global_record = {
        "projected_any": bool(projected_mode_count > 0),
        "projected_mode_count": projected_mode_count,
        "proposed_delta_l2_global": proposed_norm,
        "final_delta_l2_global": final_norm,
        "removed_delta_l2_fraction_global": float(removed_norm / proposed_norm) if proposed_norm > 0.0 else 0.0,
        "raw_surgery_to_adam_effective_cosine": float(raw_to_adam_cosine),
    }
    for row in mode_rows:
        row.update(global_record)
    return mode_rows


def summarize_step_safety(frame: pd.DataFrame):
    if frame.empty:
        raise ValueError("No v13 actual-step safety rows.")
    unique_windows = frame.drop_duplicates(["Fold", "Epoch", "UpdateWindow"])
    return {
        "window_count": int(len(unique_windows)),
        "projected_window_count": int(unique_windows.projected_any.astype(bool).sum()),
        "projected_window_fraction": float(unique_windows.projected_any.astype(bool).mean()),
        "mean_projected_mode_count": float(unique_windows.projected_mode_count.astype(float).mean()),
        "mean_removed_delta_l2_fraction": float(unique_windows.removed_delta_l2_fraction_global.astype(float).mean()),
        "mean_raw_surgery_to_adam_effective_cosine": float(unique_windows.raw_surgery_to_adam_effective_cosine.astype(float).mean()),
        "mode_projection_rate": {
            mode: float(
                frame.loc[frame.Mode.astype(str).eq(mode), "projected"].astype(bool).mean()
            )
            for mode in MISSING_MODES
        },
    }


def train_one_fold_v13(
    cli,
    logger,
    args,
    fold,
    train_loader,
    holdout_loader,
    safety_loader,
    teacher,
    s0_student,
    evaluator_bundle,
    assets,
    model_root,
    s0_sha,
):
    student = v9.fresh_fold_student(s0_student, args.device)
    parameters = list(student.bank.parameters())
    if not parameters:
        raise RuntimeError("Fold residual bank has no trainable parameters.")

    sentinel_cache, sentinel_manifest = build_fold_sentinel_cache(
        fold,
        student,
        safety_loader,
        teacher,
        evaluator_bundle,
        args,
    )
    optimizer = optim.Adam(parameters, lr=args.learning_rate)
    assert_teacher_not_in_optimizer(teacher, optimizer)
    if any(float(group.get("weight_decay", 0.0)) != 0.0 for group in optimizer.param_groups):
        raise RuntimeError("v13 actual-step projection assumes formal v12 Adam weight_decay=0.")
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if any(id(parameter) in optimizer_ids for parameter in student.s0.parameters()):
        raise RuntimeError("Frozen S0 entered fold {} optimizer.".format(fold))
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience
    )
    criterion = nn.L1Loss()
    cosine = nn.CosineEmbeddingLoss()
    hinge = HingeLoss()
    missing_generator = torch.Generator().manual_seed(
        DEV_SEED + 104729 + 7919 * int(fold)
    )
    missing_hasher = MissingSequenceHasher()

    fold_dir = model_root / "seed1113" / "fold{}".format(fold)
    fold_dir.mkdir(parents=True, exist_ok=True)
    conservative_checkpoint = fold_dir / "conservative_train_holdout_bank.pth"
    absolute_best_checkpoint = fold_dir / "absolute_best_train_holdout_bank.pth"

    absolute_best_j = float("inf")
    absolute_best_epoch = 0
    last_epoch = 0
    epoch_rows = []
    epoch_snapshots = []
    decision_records = []
    surgery_rows = []
    safety_rows = []
    batch_sizes = None
    all_projection_records = []

    logger.info(
        "fold=%s train=%s holdout=%s sentinel_events=%s optimizer=v12_surgery_plus_adam_step_safety valid=unseen",
        fold,
        len(train_loader.dataset),
        len(holdout_loader.dataset),
        int(sum(row.N for row in sentinel_manifest.itertuples() if row.Group in SAFETY_GROUP_ORDER)),
    )

    for epoch in range(1, (cli.max_epochs or 1000) + 1):
        last_epoch = epoch
        student.train()
        optimizer.zero_grad()
        epoch_batch_sizes = []
        objective_rows = []
        epoch_projection_records = []
        epoch_surgery_rows = []
        epoch_safety_rows = []
        counts = Counter({"LA": 0, "LV": 0, "L": 0})
        accumulated_supervised = zero_gradient_tuple(parameters)
        accumulated_selective = zero_gradient_tuple(parameters)
        window_steps = 0
        update_window_index = 0

        for step, batch in enumerate(train_loader, 1):
            labels_cpu = batch["labels"]["M"].view(-1, 1)
            batch_size = int(labels_cpu.size(0))
            epoch_batch_sizes.append(batch_size)
            missing_mask = sample_missing_masks(
                batch_size,
                missing_generator,
                torch.device("cpu"),
                torch.float32,
            )
            modes = tuple(v12.modes_from_masks(missing_mask))
            missing_hasher.update(modes)
            counts.update(count_missing_modes(missing_mask))

            before_decisions = len(decision_records)
            supervised_loss, selective_loss, diagnostics, projection_records = (
                v12.forward_objective_components(
                    batch,
                    missing_mask,
                    modes,
                    args,
                    teacher,
                    evaluator_bundle,
                    student,
                    assets["cache_by_index"],
                    criterion,
                    cosine,
                    hinge,
                    decision_records,
                )
            )
            for record in decision_records[before_decisions:]:
                record["Fold"] = int(fold)
                record["Epoch"] = int(epoch)

            supervised_gradient = detached_gradient_tuple(
                supervised_loss, parameters, retain_graph=True
            )
            selective_gradient = detached_gradient_tuple(
                selective_loss, parameters, retain_graph=False
            )
            accumulated_supervised = add_gradient_tuples(
                accumulated_supervised, supervised_gradient
            )
            accumulated_selective = add_gradient_tuples(
                accumulated_selective, selective_gradient
            )
            window_steps += 1

            if teacher_grad_count(teacher):
                raise RuntimeError("Frozen Teacher received gradients.")
            v9.assert_s0_no_gradients(student)

            if step % int(args.update_epochs) == 0 or step == len(train_loader):
                update_window_index += 1
                projected_supervised, update_gradient, surgery = (
                    asymmetric_project_supervised_stable(
                        accumulated_supervised, accumulated_selective
                    )
                )
                optimizer.zero_grad()
                assign_gradient_tuple(parameters, update_gradient)
                v9.assert_s0_no_gradients(student)

                local_safety_rows = adam_step_with_actual_safety(
                    student,
                    optimizer,
                    parameters,
                    update_gradient,
                    sentinel_cache,
                )
                optimizer.zero_grad()
                for local in local_safety_rows:
                    local.update(
                        {
                            "Fold": int(fold),
                            "Epoch": int(epoch),
                            "UpdateWindow": int(update_window_index),
                            "MicrobatchCount": int(window_steps),
                            "LastTrainStep": int(step),
                            "LearningRate": float(optimizer.param_groups[0]["lr"]),
                        }
                    )
                    safety_rows.append(dict(local))
                    epoch_safety_rows.append(dict(local))

                surgery.update(
                    {
                        "Fold": int(fold),
                        "Epoch": int(epoch),
                        "UpdateWindow": int(update_window_index),
                        "MicrobatchCount": int(window_steps),
                        "LastTrainStep": int(step),
                        "LearningRate": float(optimizer.param_groups[0]["lr"]),
                    }
                )
                surgery_rows.append(dict(surgery))
                epoch_surgery_rows.append(dict(surgery))
                accumulated_supervised = zero_gradient_tuple(parameters)
                accumulated_selective = zero_gradient_tuple(parameters)
                window_steps = 0

            objective_rows.append(diagnostics)
            epoch_projection_records.extend(projection_records)

        if window_steps != 0:
            raise RuntimeError("Gradient accumulation window was not flushed.")
        if batch_sizes is None:
            batch_sizes = epoch_batch_sizes
        elif batch_sizes != epoch_batch_sizes:
            raise RuntimeError("Fold {} batch-size sequence changed.".format(fold))

        holdout = evaluate_all_modes(
            student, holdout_loader, args.device, "moddrop", criterion
        )
        holdout_j = validation_objective(holdout)
        if not math.isfinite(holdout_j):
            raise FloatingPointError("Non-finite Train-holdout J.")
        scheduler.step(holdout_j)
        is_absolute_best = holdout_j <= absolute_best_j - 1e-6
        if is_absolute_best:
            absolute_best_j = float(holdout_j)
            absolute_best_epoch = int(epoch)

        epoch_snapshots.append(
            {
                "Epoch": int(epoch),
                "HoldoutJ": float(holdout_j),
                "state": bank_state_cpu(student),
                "metrics": holdout,
            }
        )
        local_projection = regret_projection_summary(epoch_projection_records)
        local_surgery = summarize_surgery_windows(epoch_surgery_rows)
        local_step_safety = summarize_step_safety(pd.DataFrame(epoch_safety_rows))
        all_projection_records.extend(epoch_projection_records)
        row = {
            "Fold": int(fold),
            "Epoch": int(epoch),
            "TrainN": int(len(train_loader.dataset)),
            "HoldoutN": int(len(holdout_loader.dataset)),
            "HoldoutJ": float(holdout_j),
            "IsAbsoluteBestTrainHoldout": bool(is_absolute_best),
            "full_loss": float(np.mean([x["full_loss"] for x in objective_rows])),
            "missing_loss": float(np.mean([x["missing_loss"] for x in objective_rows])),
            "KD_loss": float(np.mean([x["kd_loss"] for x in objective_rows])),
            "preserve_loss": float(np.mean([x["preserve_loss"] for x in objective_rows])),
            "selective_loss": float(np.mean([x["selective_loss"] for x in objective_rows])),
            "mean_gate": float(np.mean([x["mean_gate"] for x in objective_rows])),
            **{f"surgery_{key}": value for key, value in local_surgery.items()},
            **{f"step_safety_{key}": value for key, value in local_step_safety.items() if key != "mode_projection_rate"},
            **{
                "step_safety_projection_rate_{}".format(mode): local_step_safety["mode_projection_rate"][mode]
                for mode in MISSING_MODES
            },
            **local_projection,
            **_flatten(holdout, "train_holdout"),
        }
        epoch_rows.append(row)
        logger.info(
            "fold=%s epoch=%s holdout_J=%.6f projected_windows=%.3f removed_delta=%.3f raw_adam_cos=%.3f absolute_best=%s",
            fold,
            epoch,
            holdout_j,
            local_step_safety["projected_window_fraction"],
            local_step_safety["mean_removed_delta_l2_fraction"],
            local_step_safety["mean_raw_surgery_to_adam_effective_cosine"],
            is_absolute_best,
        )
        if epoch - absolute_best_epoch >= args.early_stop:
            break

    if not epoch_snapshots:
        raise RuntimeError("Fold {} produced no epoch snapshots.".format(fold))
    selector = v10.select_earliest_near_optimal(
        epoch_snapshots, NEAR_OPTIMAL_REL_TOL
    )
    by_epoch = {int(item["Epoch"]): item for item in epoch_snapshots}
    selected_snapshot = by_epoch[int(selector["selected_epoch"])]
    best_snapshot = by_epoch[int(selector["absolute_best_epoch"])]
    torch.save(selected_snapshot["state"], conservative_checkpoint)
    torch.save(best_snapshot["state"], absolute_best_checkpoint)

    student.bank.load_state_dict(best_snapshot["state"], strict=True)
    best_residual = v10.residual_magnitude_summary(student, holdout_loader, args.device)
    student.bank.load_state_dict(selected_snapshot["state"], strict=True)
    selected_residual = v10.residual_magnitude_summary(student, holdout_loader, args.device)
    if module_state_sha256(student.s0) != s0_sha:
        raise RuntimeError("Frozen S0 changed after fold {} selection.".format(fold))

    for row in epoch_rows:
        row["WithinOnePercentOfAbsoluteBest"] = bool(
            float(row["HoldoutJ"]) <= float(selector["near_optimal_cutoff_J"]) + 1e-12
        )
        row["SelectedConservative"] = bool(
            int(row["Epoch"]) == int(selector["selected_epoch"])
        )

    projection = regret_projection_summary(all_projection_records)
    surgery_summary = summarize_surgery_windows(surgery_rows)
    step_safety_summary = summarize_step_safety(pd.DataFrame(safety_rows))
    residual_ratio = (
        float(selected_residual["mean_abs_residual"] / best_residual["mean_abs_residual"])
        if best_residual["mean_abs_residual"] > 0
        else 0.0
    )
    fold_result = {
        "Fold": int(fold),
        "TrainEpochCount": int(last_epoch),
        "AbsoluteBestTrainHoldoutEpoch": int(selector["absolute_best_epoch"]),
        "AbsoluteBestTrainHoldoutJ": float(selector["absolute_best_holdout_J"]),
        "ConservativeSelectedEpoch": int(selector["selected_epoch"]),
        "ConservativeSelectedTrainHoldoutJ": float(selector["selected_holdout_J"]),
        "NearOptimalCutoffJ": float(selector["near_optimal_cutoff_J"]),
        "NearOptimalRelativeTolerance": float(selector["relative_tolerance"]),
        "SelectedRelativeJDegradation": float(selector["selected_relative_J_degradation"]),
        "EpochReductionVsAbsoluteBest": int(selector["epoch_reduction_vs_absolute_best"]),
        "AbsoluteBestHoldoutMeanAbsResidual": float(best_residual["mean_abs_residual"]),
        "ConservativeHoldoutMeanAbsResidual": float(selected_residual["mean_abs_residual"]),
        "ConservativeToBestResidualMagnitudeRatio": residual_ratio,
        "MissingSequenceSHA256": missing_hasher.hexdigest(),
        "MissingSequenceCount": int(missing_hasher.count),
        "ConservativeCheckpoint": str(conservative_checkpoint.resolve()),
        "ConservativeCheckpointSHA256": checkpoint_sha256(conservative_checkpoint),
        "AbsoluteBestCheckpoint": str(absolute_best_checkpoint.resolve()),
        "AbsoluteBestCheckpointSHA256": checkpoint_sha256(absolute_best_checkpoint),
        **{f"surgery_{key}": value for key, value in surgery_summary.items()},
        **{f"step_safety_{key}": value for key, value in step_safety_summary.items() if key != "mode_projection_rate"},
        **{
            "step_safety_projection_rate_{}".format(mode): step_safety_summary["mode_projection_rate"][mode]
            for mode in MISSING_MODES
        },
        **{f"projection_{key}": value for key, value in projection.items()},
        **_flatten(selected_snapshot["metrics"], "selected_train_holdout"),
        **_flatten(best_snapshot["metrics"], "absolute_best_train_holdout"),
    }
    selected_state_cpu = {
        key: value.detach().cpu().clone()
        for key, value in selected_snapshot["state"].items()
    }
    del optimizer, student, epoch_snapshots, sentinel_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (
        selected_state_cpu,
        pd.DataFrame(epoch_rows),
        pd.DataFrame(decision_records),
        pd.DataFrame(surgery_rows),
        pd.DataFrame(safety_rows),
        sentinel_manifest,
        fold_result,
    )


def main():
    cli = parse_args()
    output_root, model_root = result_paths(cli)
    logger, log_path = create_logger(cli)
    setup_seed(DEV_SEED)
    args = build_config(cli, DEV_SEED)
    if int(args.update_epochs) != 10:
        raise RuntimeError("v13 keeps formal v12 update_epochs=10.")
    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("v13 may construct only train/valid loaders; Test is forbidden.")

    v12_grid, v12_raw, v12_summary = load_v12_reference(cli)
    v10_grid, v10_raw, v10_summary = v12.load_v10_reference(cli)
    v8_grid, v8_raw = v9.load_v8_reference(cli)
    v4_grid, v4_raw = v7.load_v4_reference(cli)

    v4._ACTIVE_TRAIN_BASELINE_FRAME = None
    try:
        teacher, s0_student, evaluator_bundle, assets = v4.load_assets(
            cli, args, loaders, DEV_SEED
        )
        train_baseline = v4._ACTIVE_TRAIN_BASELINE_FRAME.copy()
    finally:
        v4._ACTIVE_TRAIN_BASELINE_FRAME = None

    s0_sha_before = module_state_sha256(s0_student)
    assignment = deterministic_video_group_folds(
        list(loaders["train"].dataset.ids), N_FOLDS
    )
    assignment.to_csv(output_root / "adam_step_safety_v13_fold_assignment.csv", index=False)

    bank_states = []
    fold_epoch_frames = []
    fold_decision_frames = []
    fold_surgery_frames = []
    fold_safety_frames = []
    sentinel_frames = []
    fold_results = []
    for fold in range(N_FOLDS):
        train_loader, holdout_loader, train_indices, holdout_indices = v9.make_fold_loaders(
            loaders["train"].dataset, assignment, fold, args, cli.num_workers
        )
        safety_loader = make_safety_loader(
            loaders["train"].dataset,
            train_indices,
            args.batch_size,
        )
        (
            state,
            epoch_frame,
            decisions,
            surgery_frame,
            safety_frame,
            sentinel_manifest,
            fold_result,
        ) = train_one_fold_v13(
            cli,
            logger,
            args,
            fold,
            train_loader,
            holdout_loader,
            safety_loader,
            teacher,
            s0_student,
            evaluator_bundle,
            assets,
            model_root,
            s0_sha_before,
        )
        bank_states.append(state)
        fold_epoch_frames.append(epoch_frame)
        fold_decision_frames.append(decisions)
        fold_surgery_frames.append(surgery_frame)
        fold_safety_frames.append(safety_frame)
        sentinel_frames.append(sentinel_manifest)
        fold_result["TrainN"] = int(len(train_indices))
        fold_result["HoldoutN"] = int(len(holdout_indices))
        fold_result["TrainVideoCount"] = int(
            assignment.loc[assignment.sample_index.isin(train_indices), "video_id"].nunique()
        )
        fold_result["HoldoutVideoCount"] = int(
            assignment.loc[assignment.sample_index.isin(holdout_indices), "video_id"].nunique()
        )
        fold_results.append(fold_result)

    if module_state_sha256(s0_student) != s0_sha_before:
        raise RuntimeError("S0 changed across v13 fold training.")

    # Official Valid is first used here, after all Train-only folds are frozen.
    candidate = FrozenS0CrossfitConsensus(
        s0_student,
        bank_states,
        hidden_dim=RESIDUAL_HIDDEN_DIM,
        max_abs_residual=MAX_ABS_RESIDUAL,
        min_agree=CONSENSUS_MIN_AGREE,
    ).to(args.device)
    if any(parameter.requires_grad for parameter in candidate.parameters()):
        raise RuntimeError("Final v13 consensus model must be inference-only.")

    criterion = nn.L1Loss()
    final_valid = evaluate_all_modes(
        candidate, loaders["valid"], args.device, "moddrop", criterion
    )
    candidate_j = validation_objective(final_valid)
    final_predictions = v7.snapshot_predictions(candidate, loaders["valid"], args.device)
    reference_predictions = evaluator_bundle.valid_reference.copy()
    raw_events = base.raw_events_for_run(
        DEV_SEED, RUN, final_predictions, reference_predictions
    )
    events = base.derive_valid_events(pd.DataFrame(raw_events))
    consensus_events = v9.consensus_diagnostic_rows(
        candidate, loaders["valid"], args.device
    )
    consensus_stats = consensus_summary(consensus_events)

    check = (
        events.loc[
            events.Mode.astype(str).isin(MISSING_MODES),
            ["sample_index", "Mode", "candidate_prediction"],
        ]
        .merge(
            consensus_events[["sample_index", "Mode", "candidate_prediction"]],
            on=["sample_index", "Mode"],
            suffixes=("_official", "_diagnostic"),
            validate="one_to_one",
        )
    )
    max_diff = float(
        np.max(
            np.abs(
                check.candidate_prediction_official.to_numpy(float)
                - check.candidate_prediction_diagnostic.to_numpy(float)
            )
        )
    )
    if max_diff > 1e-6:
        raise RuntimeError("v13 consensus diagnostic path drifted: {}".format(max_diff))

    v12_events = base.derive_valid_events(v12_raw)
    v10_events = base.derive_valid_events(v10_raw)
    v8_events = base.derive_valid_events(v8_raw)
    v4_events = base.derive_valid_events(v4_raw)
    candidate_transfer = mechanism_transfer_summary(events, RUN)
    v12_transfer = mechanism_transfer_summary(v12_events, str(v12_raw.Run.iloc[0]))
    v10_transfer = mechanism_transfer_summary(v10_events, str(v10_raw.Run.iloc[0]))
    v8_transfer = mechanism_transfer_summary(v8_events, str(v8_raw.Run.iloc[0]))
    v4_transfer = mechanism_transfer_summary(v4_events, str(v4_raw.Run.iloc[0]))

    legacy_gate = None
    direct_gate = None
    if not cli.smoke_test:
        legacy_gate = development_signal_gate(
            float(candidate_j),
            float(v8_grid["J_valid"]),
            candidate_transfer,
            v8_transfer,
            v4_transfer,
        )
        direct_gate = direct_v12_metric_improvement_gate(
            float(candidate_j),
            float(v12_grid["J_valid"]),
            candidate_transfer,
            v12_transfer,
        )

    if cli.smoke_test:
        verdict = "SMOKE_ONLY_NO_METRIC_DECISION"
    elif direct_gate["passed"] and legacy_gate["passed"]:
        verdict = "PROMOTE_V13_METRIC_IMPROVEMENT_AND_LEGACY_TARGET_PASS"
    elif direct_gate["passed"]:
        verdict = "V13_IMPROVES_V12_BUT_LEGACY_TARGET_GATE_NOT_MET"
    else:
        verdict = "V13_NO_PROMOTION_DIRECT_METRIC_IMPROVEMENT_GATE_FAILED"

    fold_metrics = pd.concat(fold_epoch_frames, ignore_index=True)
    fold_decisions = pd.concat(fold_decision_frames, ignore_index=True)
    surgery_windows = pd.concat(fold_surgery_frames, ignore_index=True)
    step_safety_windows = pd.concat(fold_safety_frames, ignore_index=True)
    sentinel_manifest = pd.concat(sentinel_frames, ignore_index=True)
    fold_manifest = pd.DataFrame(fold_results)
    surgery_summary = summarize_surgery_windows(surgery_windows.to_dict("records"))
    step_safety_summary = summarize_step_safety(step_safety_windows)

    transfer_rows = []
    for source, summary in (
        ("v13_candidate", candidate_transfer),
        ("v12_frozen", v12_transfer),
        ("v10_frozen", v10_transfer),
        ("v8_frozen", v8_transfer),
        ("v4_frozen", v4_transfer),
    ):
        for group in ("all_missing", "teacher_beneficial", "teacher_nonbeneficial"):
            transfer_rows.append({"source": source, "group": group, **summary[group]})

    pd.DataFrame(
        [
            {
                "Seed": DEV_SEED,
                "Run": RUN,
                "Method": METHOD,
                "J_valid": float(candidate_j),
                "S0StateSHA256": s0_sha_before,
                "S0StateUnchanged": bool(module_state_sha256(s0_student) == s0_sha_before),
                "CrossfitFolds": N_FOLDS,
                "ConsensusMinAgree": CONSENSUS_MIN_AGREE,
                "NearOptimalRelativeTolerance": NEAR_OPTIMAL_REL_TOL,
                "ResidualHiddenDim": RESIDUAL_HIDDEN_DIM,
                "MaxAbsResidual": MAX_ABS_RESIDUAL,
                "RawGradientSurgery": "v12_exact_asymmetric_projection",
                "ActualStepSafety": "per_mode_two_halfspace_projection_teacher_and_s0_beneficial",
                "OfficialValidUsedForFoldSelection": False,
                "TestConstructed": False,
                **_flatten(final_valid, "valid"),
            }
        ]
    ).to_csv(output_root / "adam_step_safety_v13_candidate_grid.csv", index=False)
    pd.DataFrame(raw_events).to_csv(
        output_root / "adam_step_safety_v13_candidate_raw_valid_events.csv", index=False
    )
    events.to_csv(output_root / "adam_step_safety_v13_candidate_valid_events.csv", index=False)
    train_baseline.to_csv(output_root / "adam_step_safety_v13_train_baseline_cache.csv", index=False)
    fold_manifest.to_csv(output_root / "adam_step_safety_v13_fold_manifest.csv", index=False)
    fold_metrics.to_csv(output_root / "adam_step_safety_v13_fold_epoch_metrics.csv", index=False)
    fold_decisions.to_csv(output_root / "adam_step_safety_v13_fold_train_decisions.csv", index=False)
    surgery_windows.to_csv(output_root / "adam_step_safety_v13_raw_surgery_windows.csv", index=False)
    step_safety_windows.to_csv(output_root / "adam_step_safety_v13_actual_step_safety_windows.csv", index=False)
    sentinel_manifest.to_csv(output_root / "adam_step_safety_v13_sentinel_manifest.csv", index=False)
    consensus_events.to_csv(output_root / "adam_step_safety_v13_valid_consensus_events.csv", index=False)
    pd.DataFrame(transfer_rows).to_csv(
        output_root / "adam_step_safety_v13_transfer_summary.csv", index=False
    )

    summary = {
        "version": VERSION,
        "method": METHOD,
        "verdict": verdict,
        "direct_v12_metric_improvement_gate": jsonable(direct_gate) if direct_gate is not None else None,
        "legacy_v8_v4_target_gate": jsonable(legacy_gate) if legacy_gate is not None else None,
        "candidate_transfer": jsonable(candidate_transfer),
        "frozen_v12_transfer": jsonable(v12_transfer),
        "frozen_v10_transfer": jsonable(v10_transfer),
        "frozen_v8_transfer": jsonable(v8_transfer),
        "frozen_v4_transfer": jsonable(v4_transfer),
        "candidate_J": float(candidate_j),
        "frozen_v12_J": float(v12_grid["J_valid"]),
        "frozen_v10_J": float(v10_grid["J_valid"]),
        "frozen_v8_J": float(v8_grid["J_valid"]),
        "frozen_v4_J": float(v4_grid["J_valid"]),
        "consensus_summary": jsonable(consensus_stats),
        "raw_gradient_surgery_summary": jsonable(surgery_summary),
        "actual_step_safety_summary": jsonable(step_safety_summary),
        "sentinel_manifest": jsonable(sentinel_manifest.to_dict("records")),
        "fold_manifest": jsonable(fold_manifest.to_dict("records")),
        "parameter_isolation": {
            "entire_s0_path_trainable": False,
            "s0_state_sha256_before": s0_sha_before,
            "s0_state_sha256_after": module_state_sha256(s0_student),
            "s0_state_unchanged": bool(module_state_sha256(s0_student) == s0_sha_before),
            "per_fold_architecture": "v12_exact_v8_residual_head_bank",
            "final_consensus_model_trainable": False,
        },
        "protocol": {
            "development_seed": DEV_SEED,
            "base_objective": "v12_exact_v4_supervised_missing_plus_distill_plus_preserve",
            "raw_gradient_surgery": "v12_exact_numerical_hotfix_projection",
            "single_added_v13_mechanism": "Train_only_functional_sentinel_projection_of_actual_Adam_parameter_displacement",
            "sentinel_source": "only_current_fold_Train_4of5_videos",
            "teacher_beneficial_definition": "baseline_error_minus_teacher_error_ge_{:.2f}".format(DISTILL_MARGIN),
            "s0_beneficial_definition": "baseline_error_minus_s0_error_ge_{:.2f}".format(S0_PROTECT_MARGIN),
            "sentinel_safety_loss": "mean_absolute_error_to_label",
            "safety_constraints": "per_missing_mode_Teacher_beneficial_and_S0_beneficial_first_order_MAE_nonincrease",
            "actual_step_rule": "Adam_updates_moments_and_proposes_delta_then_closest_feasible_delta_is_applied",
            "adam_internal_state_projected": False,
            "adam_weight_decay": 0.0,
            "new_safety_strength_hyperparameter": False,
            "epoch_specific_safety_switch": False,
            "optimizer": "Adam_same_as_v12",
            "update_epochs": int(args.update_epochs),
            "fold_assignment": "same deterministic video-grouped 5-fold as v12",
            "fold_checkpoint_selector": "same earliest epoch within 1% of absolute best Train-video-holdout J as v12",
            "selector_relative_tolerance": NEAR_OPTIMAL_REL_TOL,
            "consensus_rule": "{}-of-{} same-sign then median correction else zero".format(
                CONSENSUS_MIN_AGREE, N_FOLDS
            ),
            "official_valid_first_used_after_all_fold_models_frozen": True,
            "official_valid_used_for_fold_training": False,
            "official_valid_used_for_fold_checkpoint_selection": False,
            "official_valid_used_for_sentinel_selection": False,
            "official_test_constructed": False,
            "official_test_accessed": False,
            "lambda_preserve": LAMBDA_PRESERVE,
            "preserve_margin": PRESERVE_MARGIN,
            "mild_cfcompat": "{:.2f}+{:.2f}*compatibility".format(
                MILD_CFCOMPAT_BASE, MILD_CFCOMPAT_SCALE
            ),
        },
        "frozen_metric_criteria": {
            "direct_v12_gate": {
                "J_strict_improvement": True,
                "beneficial_NTR_strict_improvement": True,
                "overall_NTR_strict_improvement": True,
                "nonbeneficial_NTR_max_degradation": NONBENEFICIAL_NTR_MAX_DEGRADATION_VS_V12,
            },
            "legacy_target_gate": {
                "J_max_degradation_vs_v8": J_MAX_DEGRADATION_VS_V8,
                "beneficial_NTR_max_degradation_vs_v8": BENEFICIAL_NTR_MAX_DEGRADATION_VS_V8,
                "nonbeneficial_NTR_reduction_required_vs_v8": NONBENEFICIAL_NTR_REDUCTION_REQUIRED_VS_V8,
                "overall_NTR_max_degradation_vs_v4": OVERALL_NTR_MAX_DEGRADATION_VS_V4,
            },
        },
        "valid_prediction_path_max_abs_difference": max_diff,
        "v12p3_mechanism_basis": {
            "teacher_beneficial_raw_harm_windows": "75/315",
            "teacher_beneficial_additional_adam_transform_harm_windows": "35/315",
            "s0_beneficial_raw_harm_windows": "91/315",
            "s0_beneficial_additional_adam_transform_harm_windows": "33/315",
            "teacher_nonbeneficial_additional_adam_transform_harm_windows": "0/315",
            "nonlinear_finite_step_harm_is_rare": True,
        },
        "frozen_v12_verdict": v12_summary.get("verdict"),
        "frozen_v10_verdict": v10_summary.get("verdict"),
    }
    summary_path = output_root / "adam_step_safety_v13_valid_screen_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    logger.info(
        "complete verdict=%s candidate_J=%.6f projected_windows=%.4f beneficial_NTR=%.4f nonbeneficial_NTR=%.4f output=%s log=%s",
        verdict,
        candidate_j,
        step_safety_summary["projected_window_fraction"],
        candidate_transfer["teacher_beneficial"]["negative_transfer_rate"],
        candidate_transfer["teacher_nonbeneficial"]["negative_transfer_rate"],
        output_root,
        log_path,
    )
    print("Adam-Step Functional Safety CFCompatKD v13 complete")
    print("candidate J:", candidate_j)
    print("frozen v12 J:", v12_grid["J_valid"])
    print("frozen v8 J:", v8_grid["J_valid"])
    print("S0 state unchanged:", module_state_sha256(s0_student) == s0_sha_before)
    print("actual-step projected window fraction:", step_safety_summary["projected_window_fraction"])
    print("mean removed Adam displacement fraction:", step_safety_summary["mean_removed_delta_l2_fraction"])
    print("mean raw-surgery vs Adam-effective cosine:", step_safety_summary["mean_raw_surgery_to_adam_effective_cosine"])
    print("consensus applied rate:", consensus_stats["consensus_applied_rate"])
    if direct_gate is not None:
        print("direct metric improvement vs v12 passed:", direct_gate["passed"])
        print("J delta vs v12:", direct_gate["delta_J_candidate_minus_v12"])
        print("beneficial NTR reduction vs v12:", direct_gate["beneficial_NTR_reduction_vs_v12"])
        print("overall NTR reduction vs v12:", direct_gate["overall_NTR_reduction_vs_v12"])
        print("nonbeneficial NTR degradation vs v12:", direct_gate["nonbeneficial_NTR_degradation_vs_v12"])
    if legacy_gate is not None:
        print("legacy v8/v4 target gate passed:", legacy_gate["passed"])
    print("verdict:", verdict)
    print("official Test was not constructed or accessed")
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
