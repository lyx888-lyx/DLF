"""CFCompatKD v7: hard frozen-backbone / missing-adapter isolation.

Seed1113 Valid-only mechanism test.  The v4 DISTILL/PRESERVE/ABSTAIN objective
is preserved exactly.  The shared DLF backbone is frozen in both parameters and
module state; only MissingModalityWrapper's two missing tokens and mask_adapter
are optimized.  Per-epoch Valid sample predictions are saved for failure-onset
analysis.  Official Test is never constructed or accessed.
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
from train_cf_compat_kd import _flatten, batch_to_device, build_config, prediction_rows
import train_cfcompat_regret_preserve_valid_screen as v4
import train_cfcompat_safe_projection_valid_screen as base
from trains.singleTask.HingeLoss import HingeLoss
from trains.singleTask.cf_compat_kd_utils import compatibility_for_modes, gated_kd_loss, modes_from_masks
from trains.singleTask.cfcompat_adapter_isolation_utils import (
    BENEFICIAL_TEACHER_NTR_REDUCTION_REQUIRED,
    DEV_SEED,
    J_MAX_DEGRADATION_VS_V4,
    METHOD,
    NONBENEFICIAL_TEACHER_NTR_MAX_DEGRADATION,
    OUTPUT_TAG,
    OVERALL_NTR_MAX_DEGRADATION,
    RUN,
    RUNS,
    VERSION,
    assert_no_backbone_gradients,
    build_epoch_event_trajectory,
    clip_failure_epoch_summary,
    development_signal_gate,
    enforce_isolation_train_mode,
    epoch_transfer_summary,
    failure_onset_table,
    freeze_shared_backbone,
    jsonable,
    mechanism_transfer_summary,
    module_state_sha256,
    prediction_frame_to_long,
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
from trains.singleTask.cfcompat_stability_utils import (
    MissingSequenceHasher,
    expected_missing_sequence_sha,
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
from utils.functions import setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Frozen-backbone / missing-adapter isolation CFCompatKD v7"
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
        parser.error("v7 fixes num_workers=1.")
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
                "v7 output already exists; inspect it or use --overwrite: {} / {}".format(
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
    path = directory / "DLF-mosi-frozen-backbone-adapter-v7-{}-{}.log".format(
        kind, datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    logger = logging.getLogger("frozen_backbone_adapter_v7")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def snapshot_predictions(model, loader, device):
    """Prediction snapshot with RNG and loader generator restored exactly."""
    generator = getattr(loader, "generator", None)
    state = generator.get_state().clone() if generator is not None else None
    try:
        with preserve_rng_state():
            frame = prediction_rows(model, loader, device)
    finally:
        if state is not None:
            generator.set_state(state)
    if state is not None and not torch.equal(generator.get_state(), state):
        raise RuntimeError("Valid loader generator changed during trajectory snapshot.")
    if len(frame) != 229 or frame.sample_index.nunique() != 229:
        raise RuntimeError("Valid prediction snapshot must contain 229 unique samples.")
    return frame.reset_index(drop=True)


def load_v4_reference(cli):
    root = (
        Path(cli.result_root)
        / "missing_baseline"
        / "cfcompat_regret_preserve_v4"
        / cli.dataset
        / "valid_screen"
        / "seed1113_dev"
    )
    grid_path = root / "regret_preserve_v4_candidate_grid.csv"
    raw_path = root / "regret_preserve_v4_candidate_raw_valid_events.csv"
    if not grid_path.is_file() or not raw_path.is_file():
        raise FileNotFoundError("Missing frozen v4 reference artifacts under {}".format(root))
    grid = pd.read_csv(grid_path)
    raw = pd.read_csv(raw_path)
    if len(grid) != 1 or int(grid.iloc[0].Seed) != DEV_SEED:
        raise RuntimeError("Frozen v4 candidate grid is not unique Seed1113.")
    if set(raw.Split.astype(str)) != {"valid"}:
        raise RuntimeError("Frozen v4 raw reference contains a non-Valid split.")
    return grid.iloc[0].to_dict(), raw


def capture_trainable_state(student):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in student.named_parameters()
        if parameter.requires_grad
    }


def trainable_delta_summary(before, student):
    rows = []
    current = dict(student.named_parameters())
    for name, initial in before.items():
        final = current[name].detach().cpu()
        delta = final - initial
        rows.append(
            {
                "parameter": name,
                "numel": int(final.numel()),
                "initial_l2": float(initial.pow(2).sum().sqrt()),
                "final_l2": float(final.pow(2).sum().sqrt()),
                "delta_l2": float(delta.pow(2).sum().sqrt()),
                "max_abs_delta": float(delta.abs().max()) if delta.numel() else 0.0,
            }
        )
    return pd.DataFrame(rows)


def forward_objective(
    batch, missing_mask, modes, args, teacher, evaluator_bundle, student,
    cache_by_index, criterion, cosine, hinge, decision_records,
):
    """Exact v4 objective under the v7 trainable-parameter restriction."""
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
    modes = list(modes)
    compatibility = compatibility_for_modes(
        cache_by_index, indices, modes, args.device, labels.dtype
    ).view(-1)
    baseline_prediction = v4.baseline_for_modes(
        evaluator_bundle, indices, modes, labels, args.device, labels.dtype
    )
    decision = regret_preserve_decision(
        current_student, teacher_prediction, baseline_prediction, labels, compatibility
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
    total_loss = full_loss + missing_loss + kd_loss + LAMBDA_PRESERVE * preserve_loss
    if not torch.isfinite(total_loss):
        raise FloatingPointError("NaN/Inf in v7 isolated v4 objective.")

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
                "mode": str(modes[offset]),
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
                "teacher_advantage_vs_baseline": float(decision["teacher_advantage_vs_baseline"][offset].detach().cpu()),
                "current_regret_vs_baseline": float(decision["current_regret_vs_baseline"][offset].detach().cpu()),
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
        "mean_gate": float(decision["distill_gate"].detach().mean().cpu()),
        "baseline_missing_MAE": float(
            torch.abs(baseline_prediction.view(-1) - labels.view(-1)).mean().cpu()
        ),
        "safe_target_MAE": float(
            torch.abs(decision["teacher_safe_target"].view(-1) - labels.view(-1)).mean().cpu()
        ),
    }
    return total_loss, diagnostics, projection_records


def train_trajectory(cli, logger, output_root, model_root):
    setup_seed(DEV_SEED)
    args = build_config(cli, DEV_SEED)
    if int(args.update_epochs) != 10:
        raise RuntimeError("v7 fixes original update_epochs=10.")
    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("v7 may construct only train/valid loaders.")

    # Reuse v4 frozen assets and objective, but own the training loop so the
    # parameter isolation and per-epoch Valid trajectory are directly audited.
    v4._ACTIVE_TRAIN_BASELINE_FRAME = None
    decision_records = []
    try:
        teacher, student, evaluator_bundle, assets = v4.load_assets(
            cli, args, loaders, DEV_SEED
        )
        train_baseline = v4._ACTIVE_TRAIN_BASELINE_FRAME.copy()
        isolation = freeze_shared_backbone(student)
        backbone_sha_before = module_state_sha256(student.backbone)
        trainable_before = capture_trainable_state(student)

        trainable_parameters = [
            parameter for parameter in student.parameters() if parameter.requires_grad
        ]
        optimizer = optim.Adam(trainable_parameters, lr=args.learning_rate)
        assert_teacher_not_in_optimizer(teacher, optimizer)
        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if any(id(parameter) in optimizer_ids for parameter in student.backbone.parameters()):
            raise RuntimeError("Frozen backbone entered the optimizer.")
        if any(id(parameter) in optimizer_ids for parameter in evaluator_bundle.parameters()):
            raise RuntimeError("Frozen baseline bundle entered the optimizer.")

        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=args.patience
        )
        criterion = nn.L1Loss()
        cosine = nn.CosineEmbeddingLoss()
        hinge = HingeLoss()
        missing_generator = torch.Generator().manual_seed(DEV_SEED + 104729)
        missing_hasher = MissingSequenceHasher()

        run_dir = output_root / "seed1113" / RUN
        checkpoint = model_root / "seed1113" / RUN / "best_valid.pth"
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)

        # Epoch 0 is the exact frozen initial Student function.
        student.eval()
        s0_valid_metrics = evaluate_all_modes(
            student, loaders["valid"], args.device, "moddrop", criterion
        )
        s0_j = validation_objective(s0_valid_metrics)
        s0_prediction_frame = snapshot_predictions(student, loaders["valid"], args.device)
        epoch_prediction_frames = [
            prediction_frame_to_long(s0_prediction_frame, 0, False)
        ]

        best_j = float("inf")
        best_epoch = 0
        best_metrics = None
        epoch_rows = []
        batch_sizes = None
        last_epoch = 0
        all_projection_records = []
        logger.info(
            "seed=%s run=%s isolation=frozen_backbone trainables=%s test=forbidden",
            DEV_SEED,
            RUN,
            isolation["trainable_names"],
        )

        for epoch in range(1, (cli.max_epochs or 1000) + 1):
            last_epoch = epoch
            enforce_isolation_train_mode(student)
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
                loss.backward()
                if teacher_grad_count(teacher):
                    raise RuntimeError("Frozen Teacher received gradients.")
                assert_no_backbone_gradients(student)
                if step % int(args.update_epochs) == 0 or step == len(loaders["train"]):
                    optimizer.step()
                    optimizer.zero_grad()
                objective_rows.append(diagnostics)
                epoch_projection_records.extend(projection_records)

            if batch_sizes is None:
                batch_sizes = epoch_batch_sizes
            elif batch_sizes != epoch_batch_sizes:
                raise RuntimeError("Batch-size sequence changed across epochs.")

            backbone_sha_epoch = module_state_sha256(student.backbone)
            if backbone_sha_epoch != backbone_sha_before:
                raise RuntimeError("Frozen backbone state changed during epoch {}.".format(epoch))

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

            epoch_prediction_frames.append(
                prediction_frame_to_long(
                    snapshot_predictions(student, loaders["valid"], args.device),
                    epoch,
                    is_best,
                )
            )

            local_projection = regret_projection_summary(epoch_projection_records)
            all_projection_records.extend(epoch_projection_records)
            row = {
                "Seed": DEV_SEED,
                "Run": RUN,
                "Epoch": int(epoch),
                "J_valid": float(j_valid),
                "IsBestValid": bool(is_best),
                "backbone_state_sha256": backbone_sha_epoch,
                "full_loss": float(np.mean([x["full_loss"] for x in objective_rows])),
                "missing_loss": float(np.mean([x["missing_loss"] for x in objective_rows])),
                "KD_loss": float(np.mean([x["kd_loss"] for x in objective_rows])),
                "mean_gate": float(np.mean([x["mean_gate"] for x in objective_rows])),
                "baseline_missing_MAE": float(np.mean([x["baseline_missing_MAE"] for x in objective_rows])),
                "safe_target_MAE": float(np.mean([x["safe_target_MAE"] for x in objective_rows])),
                **local_projection,
                **_flatten(valid, "valid"),
            }
            epoch_rows.append(row)
            logger.info(
                "epoch=%s J=%.6f MissingMacro=%.6f trainables=%s backbone_unchanged=True",
                epoch,
                j_valid,
                np.mean([valid[mode]["MAE"] for mode in MISSING_MODES]),
                isolation["trainable_parameter_count"],
            )
            if epoch - best_epoch >= args.early_stop:
                break

        if not checkpoint.is_file() or best_metrics is None:
            raise RuntimeError("Validation-best checkpoint is absent.")
        expected_sha, expected_count = expected_missing_sequence_sha(
            DEV_SEED, last_epoch, batch_sizes
        )
        if missing_hasher.hexdigest() != expected_sha or missing_hasher.count != expected_count:
            raise RuntimeError("Missing-mode sequence hash differs from frozen Stage 3 sequence.")

        backbone_sha_before_reload = module_state_sha256(student.backbone)
        if backbone_sha_before_reload != backbone_sha_before:
            raise RuntimeError("Backbone changed before checkpoint reload.")
        student.load_state_dict(torch.load(checkpoint, map_location=args.device), strict=True)
        backbone_sha_after = module_state_sha256(student.backbone)
        if backbone_sha_after != backbone_sha_before:
            raise RuntimeError("Backbone changed after loading validation-best checkpoint.")

        final_valid = evaluate_all_modes(
            student, loaders["valid"], args.device, "moddrop", criterion
        )
        final_predictions = snapshot_predictions(student, loaders["valid"], args.device)
        reference_predictions = evaluator_bundle.valid_reference.copy()
        raw_events = base.raw_events_for_run(
            DEV_SEED, RUN, final_predictions, reference_predictions
        )
        decisions = pd.DataFrame(decision_records)
        expected_decisions = 1284 * int(last_epoch)
        if len(decisions) != expected_decisions:
            raise RuntimeError(
                "v7 Train decision count mismatch: {} != {}".format(
                    len(decisions), expected_decisions
                )
            )
        decisions["Epoch"] = ((decisions.event_ordinal.astype(int) - 1) // 1284) + 1

        epoch_predictions = pd.concat(epoch_prediction_frames, ignore_index=True)
        epoch_predictions["SelectedBestValid"] = epoch_predictions.Epoch.astype(int).eq(best_epoch)
        trajectory = build_epoch_event_trajectory(
            epoch_predictions, reference_predictions, best_epoch
        )
        transfer_by_epoch = epoch_transfer_summary(trajectory)
        failure_onset = failure_onset_table(trajectory)
        clip_epoch_summary = clip_failure_epoch_summary(trajectory)
        trainable_delta = trainable_delta_summary(trainable_before, student)

        overall_projection = regret_projection_summary(all_projection_records)
        result = {
            "Seed": DEV_SEED,
            "Run": RUN,
            "Method": METHOD,
            "BestValidEpoch": int(best_epoch),
            "TrainEpochCount": int(last_epoch),
            "J_valid": float(validation_objective(final_valid)),
            "S0_J_valid": float(s0_j),
            "MainCheckpoint": str(checkpoint.resolve()),
            "MainCheckpointSHA256": checkpoint_sha256(checkpoint),
            "MissingSequenceSHA256": missing_hasher.hexdigest(),
            "MissingSequenceCount": int(missing_hasher.count),
            "TeacherCheckpoint": str(Path(assets["teacher_checkpoint"]).resolve()),
            "TeacherSHA256": assets["teacher_sha"],
            "EvaluatorCheckpoint": str(Path(assets["evaluator_checkpoint"]).resolve()),
            "EvaluatorSHA256": assets["evaluator_sha"],
            "BackboneStateSHA256Before": backbone_sha_before,
            "BackboneStateSHA256After": backbone_sha_after,
            "BackboneStateUnchanged": True,
            "TrainableNames": ";".join(isolation["trainable_names"]),
            "TrainableParameterCount": isolation["trainable_parameter_count"],
            "TotalParameterCount": isolation["total_parameter_count"],
            "TrainableParameterFraction": isolation["trainable_parameter_fraction"],
            "TestConstructed": False,
            **{f"projection_{key}": value for key, value in overall_projection.items()},
            **_flatten(final_valid, "valid"),
        }
        return {
            "result": result,
            "epoch_rows": pd.DataFrame(epoch_rows),
            "raw_events": pd.DataFrame(raw_events),
            "train_baseline": train_baseline,
            "train_decisions": decisions,
            "s0_valid_predictions": s0_prediction_frame,
            "valid_reference": reference_predictions,
            "epoch_predictions": epoch_predictions,
            "trajectory": trajectory,
            "transfer_by_epoch": transfer_by_epoch,
            "failure_onset": failure_onset,
            "clip_epoch_summary": clip_epoch_summary,
            "trainable_delta": trainable_delta,
            "isolation": isolation,
            "backbone_sha_before": backbone_sha_before,
            "backbone_sha_after": backbone_sha_after,
        }
    finally:
        v4._ACTIVE_TRAIN_BASELINE_FRAME = None


def main():
    cli = parse_args()
    output_root, model_root = result_paths(cli)
    logger, log_path = create_logger(cli)
    v4_grid, v4_raw = load_v4_reference(cli)
    bundle = train_trajectory(cli, logger, output_root, model_root)

    result = bundle["result"]
    raw_events = bundle["raw_events"]
    events = base.derive_valid_events(raw_events)
    v4_events = base.derive_valid_events(v4_raw)
    candidate_transfer = mechanism_transfer_summary(events, RUN)
    v4_run = str(v4_raw.Run.iloc[0])
    v4_transfer = mechanism_transfer_summary(v4_events, v4_run)
    gate = None if cli.smoke_test else development_signal_gate(
        float(result["J_valid"]),
        float(v4_grid["J_valid"]),
        candidate_transfer,
        v4_transfer,
    )
    verdict = (
        "SMOKE_ONLY_NO_MECHANISM_DECISION"
        if gate is None
        else (
            "MECHANISM_SIGNAL_POSITIVE_HARD_ADAPTER_ISOLATION"
            if gate["passed"]
            else "MECHANISM_SIGNAL_NEGATIVE_OR_MIXED_HARD_ADAPTER_ISOLATION"
        )
    )

    transfer_rows = []
    for source, summary in (("v7_candidate", candidate_transfer), ("v4_frozen", v4_transfer)):
        for group in ("all_missing", "teacher_beneficial", "teacher_nonbeneficial"):
            transfer_rows.append({"source": source, "group": group, **summary[group]})

    artifacts = {
        "frozen_backbone_v7_candidate_grid.csv": pd.DataFrame([result]),
        "frozen_backbone_v7_epoch_metrics.csv": bundle["epoch_rows"],
        "frozen_backbone_v7_candidate_raw_valid_events.csv": raw_events,
        "frozen_backbone_v7_candidate_valid_events.csv": events,
        "frozen_backbone_v7_transfer_summary.csv": pd.DataFrame(transfer_rows),
        "frozen_backbone_v7_train_baseline_cache.csv": bundle["train_baseline"],
        "frozen_backbone_v7_train_decisions.csv": bundle["train_decisions"],
        "frozen_backbone_v7_s0_valid_predictions.csv": bundle["s0_valid_predictions"],
        "frozen_backbone_v7_valid_reference_cache.csv": bundle["valid_reference"],
        "frozen_backbone_v7_valid_epoch_predictions.csv": bundle["epoch_predictions"],
        "frozen_backbone_v7_valid_epoch_event_trajectory.csv": bundle["trajectory"],
        "frozen_backbone_v7_valid_epoch_transfer_summary.csv": bundle["transfer_by_epoch"],
        "frozen_backbone_v7_valid_failure_onset.csv": bundle["failure_onset"],
        "frozen_backbone_v7_valid_clip_failure_epoch_summary.csv": bundle["clip_epoch_summary"],
        "frozen_backbone_v7_trainable_parameter_delta.csv": bundle["trainable_delta"],
        "frozen_backbone_v7_v4_reference_raw_valid_events.csv": v4_raw,
    }
    for name, frame in artifacts.items():
        frame.to_csv(output_root / name, index=False)

    selected_epoch_summary = bundle["transfer_by_epoch"].loc[
        bundle["transfer_by_epoch"].Epoch.astype(int).eq(int(result["BestValidEpoch"]))
    ].to_dict("records")
    summary = {
        "version": VERSION,
        "method": METHOD,
        "verdict": verdict,
        "mechanism_signal_gate": jsonable(gate) if gate is not None else None,
        "candidate_transfer": jsonable(candidate_transfer),
        "frozen_v4_transfer": jsonable(v4_transfer),
        "selected_best_epoch_transfer_and_drift": jsonable(selected_epoch_summary),
        "parameter_isolation": {
            **jsonable(bundle["isolation"]),
            "backbone_state_sha256_before": bundle["backbone_sha_before"],
            "backbone_state_sha256_after": bundle["backbone_sha_after"],
            "backbone_state_unchanged": bool(
                bundle["backbone_sha_before"] == bundle["backbone_sha_after"]
            ),
            "backbone_forced_eval_during_train": True,
        },
        "protocol": {
            "development_seed": DEV_SEED,
            "new_trajectories_trained": 1,
            "base_objective": "v4_distill_preserve_abstain_unchanged",
            "shared_backbone_parameters_trainable": False,
            "shared_backbone_module_state_trainable": False,
            "trainable_scope": "missing_audio_token_missing_vision_token_mask_adapter_only",
            "per_epoch_valid_sample_predictions_saved": True,
            "checkpoint_selection": "minimum_official_valid_J",
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
            "J_max_degradation_vs_v4": J_MAX_DEGRADATION_VS_V4,
            "beneficial_teacher_NTR_reduction_required": BENEFICIAL_TEACHER_NTR_REDUCTION_REQUIRED,
            "nonbeneficial_teacher_NTR_max_degradation": NONBENEFICIAL_TEACHER_NTR_MAX_DEGRADATION,
            "overall_NTR_max_degradation": OVERALL_NTR_MAX_DEGRADATION,
        },
    }
    summary_path = output_root / "frozen_backbone_v7_valid_screen_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    logger.info("complete verdict=%s output=%s log=%s", verdict, output_root, log_path)
    print("Frozen-Backbone Adapter-Isolation CFCompatKD v7 complete")
    print("candidate J:", result["J_valid"])
    print("S0 J:", result["S0_J_valid"])
    print("frozen v4 J:", v4_grid["J_valid"])
    print("backbone state unchanged:", result["BackboneStateUnchanged"])
    print("trainable names:", result["TrainableNames"])
    if gate is not None:
        print(
            "beneficial-Teacher NTR reduction vs v4:",
            gate["beneficial_teacher_NTR_reduction_vs_v4"],
        )
    print("verdict:", verdict)
    print("official Test was not constructed or accessed")
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
