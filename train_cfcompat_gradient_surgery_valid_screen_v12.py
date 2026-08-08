"""CFCompatKD v12: asymmetric gradient surgery on the frozen-S0 residual bank.

v12 is the direct mechanism test motivated by the v11 Train-OOF objective audit.
Relative to v10, the model, five deterministic video-grouped folds, v4 objective,
missing-mode RNG, Adam optimizer, update window (10 mini-batches), Train-only
1%-near-optimal conservative checkpoint selector, and 4-of-5 median consensus
are unchanged.  The single mechanism change is optimizer geometry:

* accumulate the residual gradient of supervised missing-label loss;
* accumulate the residual gradient of selective KD + 0.25*PRESERVE;
* once per original optimizer update window, if their dot product is negative,
  project only the supervised gradient off the selective gradient;
* add the untouched selective gradient and perform the Adam step.

Official Valid is first used after all five fold banks are frozen.  Official Test
is never constructed or accessed.
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
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_loader import MMDataLoader
from train_cf_compat_kd import _flatten, batch_to_device, build_config
import train_cfcompat_regret_preserve_valid_screen as v4
import train_cfcompat_safe_projection_valid_screen as base
import train_cfcompat_frozen_backbone_adapter_isolation_valid_screen_v7 as v7
import train_cfcompat_crossfit_residual_consensus_valid_screen_v9 as v9
import train_cfcompat_conservative_crossfit_residual_valid_screen_v10 as v10
from trains.singleTask.HingeLoss import HingeLoss
from trains.singleTask.cf_compat_kd_utils import (
    compatibility_for_modes,
    gated_kd_loss,
    modes_from_masks,
)
from trains.singleTask.cfcompat_adapter_isolation_utils import mechanism_transfer_summary
from trains.singleTask.cfcompat_crossfit_residual_consensus_utils import (
    FrozenS0CrossfitConsensus,
    bank_state_cpu,
    consensus_summary,
    deterministic_video_group_folds,
    module_state_sha256,
)
from trains.singleTask.cfcompat_gradient_surgery_utils import (
    BENEFICIAL_NTR_MAX_DEGRADATION_VS_V8,
    CONSENSUS_MIN_AGREE,
    DEV_SEED,
    J_MAX_DEGRADATION_VS_V8,
    METHOD,
    NEAR_OPTIMAL_REL_TOL,
    NONBENEFICIAL_NTR_REDUCTION_REQUIRED_VS_V8,
    N_FOLDS,
    OUTPUT_TAG,
    OVERALL_NTR_MAX_DEGRADATION_VS_V4,
    RUN,
    VERSION,
    add_gradient_tuples,
    asymmetric_project_supervised,
    assign_gradient_tuple,
    detached_gradient_tuple,
    development_signal_gate,
    jsonable,
    summarize_surgery_windows,
    zero_gradient_tuple,
)
from trains.singleTask.cfcompat_regret_preserve_utils import (
    DISTILL_MARGIN,
    LAMBDA_PRESERVE,
    MILD_CFCOMPAT_BASE,
    MILD_CFCOMPAT_SCALE,
    PRESERVE_MARGIN,
    regret_preserve_decision,
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
    compute_full_dlf_loss,
    compute_task_loss,
    count_missing_modes,
    evaluate_all_modes,
    mode_to_mask,
    sample_missing_masks,
    validation_objective,
)
from utils.functions import setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Asymmetric gradient-surgery CFCompatKD v12"
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
        parser.error("v12 fixes num_workers=1.")
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
                "v12 output exists; inspect it or use --overwrite: {} / {}".format(
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
    path = directory / "DLF-mosi-gradient-surgery-v12-{}-{}.log".format(
        kind, datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    logger = logging.getLogger("gradient_surgery_v12")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def load_v10_reference(cli):
    root = (
        Path(cli.result_root)
        / "missing_baseline"
        / "cfcompat_conservative_crossfit_residual_v10"
        / cli.dataset
        / "valid_screen"
        / "seed1113_dev"
    )
    grid_path = root / "conservative_crossfit_v10_candidate_grid.csv"
    raw_path = root / "conservative_crossfit_v10_candidate_raw_valid_events.csv"
    summary_path = root / "conservative_crossfit_v10_valid_screen_summary.json"
    if not grid_path.is_file() or not raw_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Missing frozen v10 reference artifacts under {}".format(root))
    grid = pd.read_csv(grid_path)
    raw = pd.read_csv(raw_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if len(grid) != 1 or int(grid.iloc[0].Seed) != DEV_SEED:
        raise RuntimeError("Frozen v10 grid is not unique Seed1113.")
    if set(raw.Split.astype(str)) != {"valid"}:
        raise RuntimeError("Frozen v10 raw events contain non-Valid split.")
    return grid.iloc[0].to_dict(), raw, summary


def forward_objective_components(
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
    decision_records,
):
    """Exact v4 residual objective, separated into supervised/selective parts."""
    text, audio, vision, labels = batch_to_device(batch, args.device)
    missing_mask = missing_mask.to(device=args.device, dtype=audio.dtype)
    full_mask = mode_to_mask("LAV", labels.size(0), args.device, audio.dtype)

    # The v8/v9/v10 residual bank is inactive for LAV; keep this computation only
    # for exact diagnostics.  It contributes zero residual-head gradient.
    with torch.no_grad():
        full_output = student(text, audio, vision, full_mask)
        full_loss, _ = compute_full_dlf_loss(full_output, labels, criterion, cosine, hinge)

    missing_output = student(text, audio, vision, missing_mask)
    missing_loss, _ = compute_task_loss(missing_output, labels, criterion)
    current_student = missing_output["output_logit"].detach().view(-1, 1)
    teacher_prediction = teacher_lav_prediction(
        teacher, text, audio, vision
    ).view(-1, 1)

    indices = batch["index"].view(-1).cpu().numpy().astype(int).tolist()
    mode_list = list(modes)
    compatibility = compatibility_for_modes(
        cache_by_index, indices, mode_list, args.device, labels.dtype
    ).view(-1)
    baseline_prediction = v4.baseline_for_modes(
        evaluator_bundle, indices, mode_list, labels, args.device, labels.dtype
    )
    decision = regret_preserve_decision(
        current_student,
        teacher_prediction,
        baseline_prediction,
        labels,
        compatibility,
    )

    kd_loss, each_kd = gated_kd_loss(
        missing_output["output_logit"],
        decision["teacher_safe_target"],
        decision["distill_gate"],
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

    supervised_loss = missing_loss
    selective_loss = kd_loss + LAMBDA_PRESERVE * preserve_loss
    if not torch.isfinite(supervised_loss) or not torch.isfinite(selective_loss):
        raise FloatingPointError("Non-finite v12 objective component.")

    projection_records = []
    teacher_projection = decision["teacher_projection"]
    identifiers = list(batch["id"])
    for offset in range(labels.size(0)):
        record = {
            key: (
                bool(value[offset].detach().cpu())
                if value.dtype == torch.bool
                else float(value[offset].detach().cpu())
            )
            for key, value in teacher_projection.items()
        }
        record.update(
            {
                "sample_index": int(indices[offset]),
                "sample_id": str(identifiers[offset]),
                "mode": str(mode_list[offset]),
                "label": float(labels[offset].detach().cpu()),
                "student_prediction": float(current_student[offset].detach().cpu()),
                "baseline_prediction": float(baseline_prediction[offset].detach().cpu()),
                "teacher_prediction": float(teacher_prediction[offset].detach().cpu()),
                "teacher_safe_target": float(decision["teacher_safe_target"][offset].detach().cpu()),
                "preserve_safe_target": float(decision["preserve_safe_target"][offset].detach().cpu()),
                "distill": bool(decision["distill"][offset].detach().cpu()),
                "preserve": bool(decision["preserve"][offset].detach().cpu()),
                "decision_abstain": bool(decision["abstain"][offset].detach().cpu()),
                "teacher_beneficial": bool(decision["teacher_beneficial"][offset].detach().cpu()),
                "current_regressed": bool(decision["current_regressed"][offset].detach().cpu()),
                "baseline_error": float(decision["baseline_error"][offset].detach().cpu()),
                "teacher_error": float(decision["teacher_error"][offset].detach().cpu()),
                "current_error": float(decision["current_error"][offset].detach().cpu()),
                "teacher_advantage_vs_baseline": float(
                    decision["teacher_advantage_vs_baseline"][offset].detach().cpu()
                ),
                "current_regret_vs_baseline": float(
                    decision["current_regret_vs_baseline"][offset].detach().cpu()
                ),
                "compatibility": float(compatibility[offset].detach().cpu()),
                "mild_compatibility": float(decision["mild_compatibility"][offset].detach().cpu()),
                "distill_gate": float(decision["distill_gate"][offset].detach().cpu()),
                "preserve_gate": float(decision["preserve_gate"][offset].detach().cpu()),
                "distill_loss_each": float(each_kd[offset].detach().cpu()),
                "preserve_loss_each": float(each_preserve[offset].detach().cpu()),
            }
        )
        record["event_ordinal"] = len(decision_records) + 1
        projection_records.append(record)
        decision_records.append(dict(record))

    diagnostics = {
        "full_loss": float(full_loss.detach().cpu()),
        "missing_loss": float(missing_loss.detach().cpu()),
        "kd_loss": float(kd_loss.detach().cpu()),
        "preserve_loss": float(preserve_loss.detach().cpu()),
        "selective_loss": float(selective_loss.detach().cpu()),
        "mean_gate": float(decision["distill_gate"].detach().mean().cpu()),
    }
    return supervised_loss, selective_loss, diagnostics, projection_records


def train_one_fold_surgery(
    cli,
    logger,
    args,
    fold,
    train_loader,
    holdout_loader,
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
    optimizer = optim.Adam(parameters, lr=args.learning_rate)
    assert_teacher_not_in_optimizer(teacher, optimizer)
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
    batch_sizes = None
    all_projection_records = []

    logger.info(
        "fold=%s train=%s holdout=%s optimizer=asymmetric_gradient_surgery update_window=%s valid=unseen",
        fold,
        len(train_loader.dataset),
        len(holdout_loader.dataset),
        int(args.update_epochs),
    )

    for epoch in range(1, (cli.max_epochs or 1000) + 1):
        last_epoch = epoch
        student.train()
        optimizer.zero_grad()
        epoch_batch_sizes = []
        objective_rows = []
        epoch_projection_records = []
        epoch_surgery_rows = []
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
            modes = tuple(modes_from_masks(missing_mask))
            missing_hasher.update(modes)
            counts.update(count_missing_modes(missing_mask))

            before = len(decision_records)
            supervised_loss, selective_loss, diagnostics, projection_records = (
                forward_objective_components(
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
            for record in decision_records[before:]:
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
                    asymmetric_project_supervised(
                        accumulated_supervised, accumulated_selective
                    )
                )
                optimizer.zero_grad()
                assign_gradient_tuple(parameters, update_gradient)
                v9.assert_s0_no_gradients(student)
                optimizer.step()
                optimizer.zero_grad()

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
            **local_projection,
            **_flatten(holdout, "train_holdout"),
        }
        epoch_rows.append(row)
        logger.info(
            "fold=%s epoch=%s holdout_J=%.6f MissingMacro=%.6f conflict_windows=%.3f mean_cos=%.3f absolute_best=%s",
            fold,
            epoch,
            holdout_j,
            np.mean([holdout[mode]["MAE"] for mode in MISSING_MODES]),
            local_surgery["conflict_window_fraction"],
            local_surgery["mean_pre_surgery_cosine"],
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
        **{f"projection_{key}": value for key, value in projection.items()},
        **_flatten(selected_snapshot["metrics"], "selected_train_holdout"),
        **_flatten(best_snapshot["metrics"], "absolute_best_train_holdout"),
    }
    selected_state_cpu = {
        key: value.detach().cpu().clone()
        for key, value in selected_snapshot["state"].items()
    }
    del optimizer, student, epoch_snapshots
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (
        selected_state_cpu,
        pd.DataFrame(epoch_rows),
        pd.DataFrame(decision_records),
        pd.DataFrame(surgery_rows),
        fold_result,
    )


def main():
    cli = parse_args()
    output_root, model_root = result_paths(cli)
    logger, log_path = create_logger(cli)
    setup_seed(DEV_SEED)
    args = build_config(cli, DEV_SEED)
    if int(args.update_epochs) != 10:
        raise RuntimeError("v12 fixes original update_epochs=10.")
    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("v12 may construct only train/valid loaders; Test is forbidden.")

    v10_grid, v10_raw, v10_summary = load_v10_reference(cli)
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
    assignment.to_csv(output_root / "gradient_surgery_v12_fold_assignment.csv", index=False)

    bank_states = []
    fold_epoch_frames = []
    fold_decision_frames = []
    fold_surgery_frames = []
    fold_results = []
    for fold in range(N_FOLDS):
        train_loader, holdout_loader, train_indices, holdout_indices = v9.make_fold_loaders(
            loaders["train"].dataset, assignment, fold, args, cli.num_workers
        )
        state, epoch_frame, decisions, surgery_frame, fold_result = train_one_fold_surgery(
            cli,
            logger,
            args,
            fold,
            train_loader,
            holdout_loader,
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
        raise RuntimeError("S0 changed across v12 fold training.")

    # Official Valid is first used here, after every Train-only fold checkpoint is frozen.
    candidate = FrozenS0CrossfitConsensus(
        s0_student,
        bank_states,
        hidden_dim=RESIDUAL_HIDDEN_DIM,
        max_abs_residual=MAX_ABS_RESIDUAL,
        min_agree=CONSENSUS_MIN_AGREE,
    ).to(args.device)
    if any(parameter.requires_grad for parameter in candidate.parameters()):
        raise RuntimeError("Final v12 consensus model must be inference-only.")

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
        raise RuntimeError("v12 consensus diagnostic path drifted: {}".format(max_diff))

    v10_events = base.derive_valid_events(v10_raw)
    v8_events = base.derive_valid_events(v8_raw)
    v4_events = base.derive_valid_events(v4_raw)
    candidate_transfer = mechanism_transfer_summary(events, RUN)
    v10_transfer = mechanism_transfer_summary(v10_events, str(v10_raw.Run.iloc[0]))
    v8_transfer = mechanism_transfer_summary(v8_events, str(v8_raw.Run.iloc[0]))
    v4_transfer = mechanism_transfer_summary(v4_events, str(v4_raw.Run.iloc[0]))

    gate = None if cli.smoke_test else development_signal_gate(
        float(candidate_j),
        float(v8_grid["J_valid"]),
        candidate_transfer,
        v8_transfer,
        v4_transfer,
    )
    verdict = (
        "SMOKE_ONLY_NO_MECHANISM_DECISION"
        if gate is None
        else (
            "MECHANISM_SIGNAL_POSITIVE_ASYMMETRIC_GRADIENT_SURGERY"
            if gate["passed"]
            else "MECHANISM_SIGNAL_NEGATIVE_OR_MIXED_ASYMMETRIC_GRADIENT_SURGERY"
        )
    )

    fold_metrics = pd.concat(fold_epoch_frames, ignore_index=True)
    fold_decisions = pd.concat(fold_decision_frames, ignore_index=True)
    surgery_windows = pd.concat(fold_surgery_frames, ignore_index=True)
    fold_manifest = pd.DataFrame(fold_results)
    surgery_summary = summarize_surgery_windows(surgery_windows.to_dict("records"))

    transfer_rows = []
    for source, summary in (
        ("v12_candidate", candidate_transfer),
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
                "GradientSurgery": "project_supervised_off_selective_on_negative_dot",
                "OfficialValidUsedForFoldSelection": False,
                "TestConstructed": False,
                **_flatten(final_valid, "valid"),
            }
        ]
    ).to_csv(output_root / "gradient_surgery_v12_candidate_grid.csv", index=False)
    pd.DataFrame(raw_events).to_csv(
        output_root / "gradient_surgery_v12_candidate_raw_valid_events.csv", index=False
    )
    events.to_csv(output_root / "gradient_surgery_v12_candidate_valid_events.csv", index=False)
    train_baseline.to_csv(output_root / "gradient_surgery_v12_train_baseline_cache.csv", index=False)
    fold_manifest.to_csv(output_root / "gradient_surgery_v12_fold_manifest.csv", index=False)
    fold_metrics.to_csv(output_root / "gradient_surgery_v12_fold_epoch_metrics.csv", index=False)
    fold_decisions.to_csv(output_root / "gradient_surgery_v12_fold_train_decisions.csv", index=False)
    surgery_windows.to_csv(output_root / "gradient_surgery_v12_update_windows.csv", index=False)
    consensus_events.to_csv(output_root / "gradient_surgery_v12_valid_consensus_events.csv", index=False)
    pd.DataFrame(transfer_rows).to_csv(
        output_root / "gradient_surgery_v12_transfer_summary.csv", index=False
    )

    summary = {
        "version": VERSION,
        "method": METHOD,
        "verdict": verdict,
        "mechanism_signal_gate": jsonable(gate) if gate is not None else None,
        "candidate_transfer": jsonable(candidate_transfer),
        "frozen_v10_transfer": jsonable(v10_transfer),
        "frozen_v8_transfer": jsonable(v8_transfer),
        "frozen_v4_transfer": jsonable(v4_transfer),
        "candidate_J": float(candidate_j),
        "frozen_v10_J": float(v10_grid["J_valid"]),
        "frozen_v8_J": float(v8_grid["J_valid"]),
        "frozen_v4_J": float(v4_grid["J_valid"]),
        "consensus_summary": jsonable(consensus_stats),
        "gradient_surgery_summary": jsonable(surgery_summary),
        "fold_manifest": jsonable(fold_manifest.to_dict("records")),
        "parameter_isolation": {
            "entire_s0_path_trainable": False,
            "s0_state_sha256_before": s0_sha_before,
            "s0_state_sha256_after": module_state_sha256(s0_student),
            "s0_state_unchanged": bool(module_state_sha256(s0_student) == s0_sha_before),
            "per_fold_architecture": "v10_exact_v8_residual_head_bank",
            "per_fold_hidden_dim": RESIDUAL_HIDDEN_DIM,
            "per_fold_max_abs_residual": MAX_ABS_RESIDUAL,
            "final_consensus_model_trainable": False,
        },
        "protocol": {
            "development_seed": DEV_SEED,
            "base_objective": "v4_supervised_missing_plus_distill_plus_preserve",
            "single_mechanism_change_vs_v10": "asymmetric_gradient_surgery_at_original_optimizer_update_window",
            "surgery_supervised_component": "missing_task_loss_residual_gradient",
            "surgery_selective_component": "KD_loss_plus_lambda_preserve_times_preserve_loss",
            "surgery_rule": "if dot(supervised,selective)<0 project only supervised to selective-orthogonal halfspace",
            "selective_gradient_modified": False,
            "optimizer": "Adam_unchanged",
            "update_epochs": int(args.update_epochs),
            "fold_assignment": "same deterministic video-grouped 5-fold as v10",
            "fold_checkpoint_selector": "same earliest epoch within 1% of absolute best Train-video-holdout J as v10",
            "selector_relative_tolerance": NEAR_OPTIMAL_REL_TOL,
            "consensus_rule": "{}-of-{} same-sign then median correction else zero".format(
                CONSENSUS_MIN_AGREE, N_FOLDS
            ),
            "official_valid_first_used_after_all_fold_models_frozen": True,
            "official_valid_used_for_fold_training": False,
            "official_valid_used_for_fold_checkpoint_selection": False,
            "official_test_constructed": False,
            "official_test_accessed": False,
            "distill_margin": DISTILL_MARGIN,
            "preserve_margin": PRESERVE_MARGIN,
            "lambda_preserve": LAMBDA_PRESERVE,
            "mild_cfcompat": "{:.2f}+{:.2f}*compatibility".format(
                MILD_CFCOMPAT_BASE, MILD_CFCOMPAT_SCALE
            ),
        },
        "frozen_signal_thresholds": {
            "J_max_degradation_vs_v8": J_MAX_DEGRADATION_VS_V8,
            "beneficial_NTR_max_degradation_vs_v8": BENEFICIAL_NTR_MAX_DEGRADATION_VS_V8,
            "nonbeneficial_NTR_reduction_required_vs_v8": NONBENEFICIAL_NTR_REDUCTION_REQUIRED_VS_V8,
            "overall_NTR_max_degradation_vs_v4": OVERALL_NTR_MAX_DEGRADATION_VS_V4,
        },
        "valid_prediction_path_max_abs_difference": max_diff,
        "v11_mechanism_basis": {
            "supervised_missing_harmed_oof_teacher_beneficial_in_folds": "5/5",
            "selective_only_improved_oof_teacher_beneficial_in_folds": "5/5",
            "this_experiment_tests": "whether removing only direct supervised-vs-selective gradient conflict preserves repair while reducing beneficial harm",
        },
        "frozen_v10_verdict": v10_summary.get("verdict"),
    }
    summary_path = output_root / "gradient_surgery_v12_valid_screen_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    logger.info(
        "complete verdict=%s candidate_J=%.6f conflict_windows=%.4f beneficial_NTR=%.4f nonbeneficial_NTR=%.4f output=%s log=%s",
        verdict,
        candidate_j,
        surgery_summary["conflict_window_fraction"],
        candidate_transfer["teacher_beneficial"]["negative_transfer_rate"],
        candidate_transfer["teacher_nonbeneficial"]["negative_transfer_rate"],
        output_root,
        log_path,
    )
    print("Asymmetric Gradient Surgery CFCompatKD v12 complete")
    print("candidate J:", candidate_j)
    print("frozen v10 J:", v10_grid["J_valid"])
    print("frozen v8 J:", v8_grid["J_valid"])
    print("frozen v4 J:", v4_grid["J_valid"])
    print("S0 state unchanged:", module_state_sha256(s0_student) == s0_sha_before)
    print("gradient-surgery conflict window fraction:", surgery_summary["conflict_window_fraction"])
    print("mean conflict cosine:", surgery_summary["mean_conflict_cosine"])
    print("mean supervised L2 removed on conflict:", surgery_summary["mean_removed_fraction_on_conflict"])
    print("consensus applied rate:", consensus_stats["consensus_applied_rate"])
    if gate is not None:
        print(
            "nonbeneficial-Teacher NTR reduction vs v8:",
            gate["nonbeneficial_teacher_NTR_reduction_vs_v8"],
        )
        print(
            "beneficial-Teacher NTR degradation vs v8:",
            gate["beneficial_teacher_NTR_degradation_vs_v8"],
        )
    print("verdict:", verdict)
    print("official Test was not constructed or accessed")
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
