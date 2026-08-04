"""Locked two-seed Valid-only screen for multi-teacher CFCompatKD."""
from __future__ import annotations

import argparse
import gc
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
    CACHE_VERSION,
    MULTISEED_CACHE_VERSION,
    compatibility_for_modes,
    gated_kd_loss,
    modes_from_masks,
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
)
from trains.singleTask.cfcompat_teacher_consensus_utils import (
    BASELINE_RUN,
    CANDIDATE_RUNS,
    CONSENSUS_RUN,
    FORMAL_SEEDS,
    MEAN_RUN,
    METHOD,
    OUTPUT_TAG,
    RUNS,
    TEACHER_SEEDS,
    VERSION,
    aggregate_candidate_gate,
    consensus_cache_paths,
    consensus_for_indices,
    load_consensus_cache,
    make_consensus_cache,
    method_gate,
    replay_gate,
    verdict_from_gates,
    write_consensus_cache,
)
from trains.singleTask.fixed_kd_utils import (
    assert_teacher_not_in_optimizer,
    build_frozen_teacher,
    checkpoint_sha256,
    teacher_grad_count,
    teacher_lav_prediction,
)
from trains.singleTask.missing_utils import (
    MISSING_MODES,
    build_single_split_loader,
    clean_checkpoint_path,
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
        description="CFCompatKD multi-teacher consensus two-seed Valid screen."
    )
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(FORMAL_SEEDS))
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--log-dir", default="log/missing_baseline")
    parser.add_argument("--config-file", default="config/config.json")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-epochs", type=int)
    args = parser.parse_args()
    if tuple(args.seeds) != FORMAL_SEEDS:
        parser.error("Formal teacher-consensus screen fixes seeds to 1111 1114.")
    if int(args.num_workers) != 1:
        parser.error("Formal teacher-consensus screen fixes num_workers=1.")
    if args.smoke_test:
        args.max_epochs = 2 if args.max_epochs is None else min(2, args.max_epochs)
    elif args.max_epochs is not None:
        parser.error("Formal runs use the original frozen early-stop rule.")
    return args


def result_paths(cli):
    root = (
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
        root, model = root / "smoke", model / "smoke"
    root.mkdir(parents=True, exist_ok=True)
    model.mkdir(parents=True, exist_ok=True)
    return root, model


def create_logger(cli):
    directory = Path(cli.log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    kind = "smoke" if cli.smoke_test else "formal"
    path = directory / "DLF-mosi-cfcompat-teacher-consensus-{}-{}.log".format(
        kind, datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    logger = logging.getLogger("cfcompat_teacher_consensus")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def json_value(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def series_to_dict(series):
    return {str(key): json_value(value) for key, value in series.items()}


def aligned_batch_ids(value, batch_size):
    if isinstance(value, (list, tuple)):
        if len(value) == batch_size and not any(
            torch.is_tensor(item) and item.numel() == batch_size for item in value
        ):
            return [str(item) for item in value]
        components = []
        for item in value:
            if torch.is_tensor(item) and item.numel() == batch_size:
                components.append(item.detach().cpu().view(-1).tolist())
            elif isinstance(item, np.ndarray) and len(item) == batch_size:
                components.append(item.tolist())
            elif isinstance(item, (list, tuple)) and len(item) == batch_size:
                components.append(list(item))
        if components:
            return [
                "|".join(str(component[index]) for component in components)
                for index in range(batch_size)
            ]
    if torch.is_tensor(value) and value.numel() == batch_size:
        return [str(item) for item in value.detach().cpu().view(-1).tolist()]
    raise RuntimeError("Unable to align sample IDs for batch size {}.".format(batch_size))


def collect_teacher_predictions(teacher, loader, args):
    rows = []
    teacher.eval()
    with torch.no_grad():
        for batch in loader:
            text = batch["text"].to(args.device)
            audio = batch["audio"].to(args.device)
            vision = batch["vision"].to(args.device)
            labels = batch["labels"]["M"].view(-1).cpu().numpy().astype(np.float64)
            indices = batch["index"].view(-1).cpu().numpy().astype(np.int64)
            identifiers = aligned_batch_ids(batch["id"], len(indices))
            prediction = (
                teacher_lav_prediction(teacher, text, audio, vision)
                .view(-1)
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            for position, index in enumerate(indices):
                rows.append(
                    {
                        "sample_index": int(index),
                        "sample_id": str(identifiers[position]),
                        "label": float(labels[position]),
                        "prediction": float(prediction[position]),
                    }
                )
    frame = pd.DataFrame(rows).sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    if len(frame) != 1284 or frame.sample_index.duplicated().any():
        raise RuntimeError("Each clean Teacher must predict 1284 unique train samples.")
    if not np.array_equal(frame.sample_index.to_numpy(dtype=np.int64), np.arange(1284)):
        raise RuntimeError("Teacher train predictions are not bound to contiguous indices.")
    if not np.isfinite(frame[["label", "prediction"]].to_numpy(dtype=np.float64)).all():
        raise FloatingPointError("Teacher train predictions contain NaN/Inf.")
    return frame


def build_train_consensus_cache(cli, output_root, logger):
    setup_seed(1111)
    args = build_config(cli, 1111)
    paths = consensus_cache_paths(output_root)
    before = capture_rng_state()
    predictions_by_seed = {}
    metadata = None
    teacher_records = []
    try:
        loader = build_single_split_loader(args, "train", cli.num_workers)
        for teacher_seed in TEACHER_SEEDS:
            checkpoint = clean_checkpoint_path(
                cli.model_save_dir, args.dataset_name, teacher_seed
            )
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    "Required clean Teacher checkpoint absent: {}".format(checkpoint)
                )
            teacher = build_frozen_teacher(DLF, args, checkpoint)
            local = collect_teacher_predictions(teacher, loader, args)
            local_metadata = local[["sample_index", "sample_id", "label"]].copy()
            if metadata is None:
                metadata = local_metadata
            else:
                if not np.array_equal(
                    metadata.sample_index.to_numpy(dtype=np.int64),
                    local_metadata.sample_index.to_numpy(dtype=np.int64),
                ):
                    raise RuntimeError("Teacher sample indices differ across seeds.")
                if not metadata.sample_id.astype(str).equals(local_metadata.sample_id.astype(str)):
                    raise RuntimeError("Teacher sample IDs differ across seeds.")
                if not np.allclose(
                    metadata.label.to_numpy(dtype=np.float64),
                    local_metadata.label.to_numpy(dtype=np.float64),
                    atol=0.0,
                    rtol=0.0,
                ):
                    raise RuntimeError("Teacher labels differ across seeds.")
            predictions_by_seed[int(teacher_seed)] = local.prediction.to_numpy(dtype=np.float64)
            teacher_records.append(
                {
                    "seed": int(teacher_seed),
                    "path": str(checkpoint.resolve()),
                    "sha256": checkpoint_sha256(checkpoint),
                }
            )
            del teacher, local
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        frame, variance_median = make_consensus_cache(metadata, predictions_by_seed)
    finally:
        restore_rng_state(before)
    after = capture_rng_state()
    preserved = rng_states_equal(before, after)
    if not preserved:
        raise RuntimeError("Teacher-consensus cache construction changed training RNG state.")
    config = write_consensus_cache(
        frame,
        paths,
        variance_median,
        teacher_records,
        rng_state_preserved=preserved,
    )
    logger.info(
        "built train-only teacher cache teachers=%s samples=%s variance_median=%.9g",
        TEACHER_SEEDS,
        len(frame),
        variance_median,
    )
    return load_consensus_cache(paths) + (paths, config)


def load_assets(cli, args, loaders, seed):
    multiseed = int(seed) != 1111
    evaluator_checkpoint, evaluator_best_epoch, evaluator_source = (
        portable_locate_stage1_evaluator(
            cli.result_root,
            cli.dataset,
            seed,
            multiseed=multiseed,
            smoke=False,
        )
    )
    evaluator_sha = checkpoint_sha256(evaluator_checkpoint)
    cache_version = MULTISEED_CACHE_VERSION if multiseed else CACHE_VERSION
    cache_frame, cache_by_index, cache_paths = load_locked_counterfactual_cache(
        cli.result_root,
        cli.dataset,
        version=cache_version,
        seed=seed if multiseed else None,
        expected_evaluator_sha=evaluator_sha,
    )
    if len(cache_frame) != 1284:
        raise RuntimeError("Consensus screen requires the locked 1284-sample compatibility cache.")
    teacher, student, teacher_checkpoint, teacher_sha = initialize_teacher_student(
        args, cli, seed, loaders
    )
    return teacher, student, {
        "teacher_checkpoint": teacher_checkpoint,
        "teacher_sha": teacher_sha,
        "evaluator_checkpoint": evaluator_checkpoint,
        "evaluator_sha": evaluator_sha,
        "evaluator_best_epoch": evaluator_best_epoch,
        "evaluator_source": evaluator_source,
        "cache_by_index": cache_by_index,
        "cache_csv": cache_paths["csv"],
        "cache_config": cache_paths["config"],
    }


def prepare_batch(batch, missing_generator, missing_hasher, counts):
    labels = batch["labels"]["M"].view(-1, 1)
    missing_mask = sample_missing_masks(
        labels.size(0),
        missing_generator,
        torch.device("cpu"),
        torch.float32,
    )
    modes = modes_from_masks(missing_mask)
    missing_hasher.update(modes)
    counts.update(count_missing_modes(missing_mask))
    return {
        "batch": batch,
        "missing_mask_cpu": missing_mask.cpu(),
        "modes": tuple(modes),
    }


def forward_objective(
    prepared,
    args,
    run,
    teacher,
    student,
    compatibility_cache,
    consensus_cache,
    criterion,
    cosine,
    hinge,
):
    batch = prepared["batch"]
    text, audio, vision, labels = batch_to_device(batch, args.device)
    missing_mask = prepared["missing_mask_cpu"].to(
        device=args.device, dtype=audio.dtype
    )
    modes = list(prepared["modes"])
    full_mask = mode_to_mask("LAV", labels.size(0), args.device, audio.dtype)
    full_output = student(text, audio, vision, full_mask)
    full_loss, _ = compute_full_dlf_loss(full_output, labels, criterion, cosine, hinge)
    missing_output = student(text, audio, vision, missing_mask)
    missing_loss, _ = compute_task_loss(missing_output, labels, criterion)
    indices = batch["index"].view(-1).cpu().numpy().astype(int).tolist()
    compatibility = compatibility_for_modes(
        compatibility_cache,
        indices,
        modes,
        args.device,
        labels.dtype,
    )
    if run == BASELINE_RUN:
        if teacher is None:
            raise RuntimeError("Single-Teacher replay requires the live frozen Teacher.")
        teacher_target = teacher_lav_prediction(teacher, text, audio, vision)
        gate = compatibility.detach()
        kd_loss, _ = gated_kd_loss(
            missing_output["output_logit"], teacher_target, gate
        )
        variance = torch.zeros_like(gate)
        consensus_weight = torch.ones_like(gate)
    else:
        teacher_target, variance, consensus_weight = consensus_for_indices(
            consensus_cache,
            indices,
            args.device,
            labels.dtype,
        )
        gate = method_gate(run, compatibility, consensus_weight)
        kd_loss, _ = gated_kd_loss(
            missing_output["output_logit"], teacher_target, gate
        )
    total_loss = full_loss + missing_loss + kd_loss
    if not torch.isfinite(total_loss):
        raise FloatingPointError("NaN/Inf in teacher-consensus CFCompatKD objective.")
    return total_loss, {
        "full_loss": float(full_loss.detach().cpu()),
        "missing_loss": float(missing_loss.detach().cpu()),
        "kd_loss": float(kd_loss.detach().cpu()),
        "compatibility_mean": float(compatibility.detach().mean().cpu()),
        "teacher_variance_mean": float(variance.detach().mean().cpu()),
        "consensus_weight_mean": float(consensus_weight.detach().mean().cpu()),
        "final_gate_mean": float(gate.detach().mean().cpu()),
        "teacher_target_mean": float(teacher_target.detach().mean().cpu()),
    }


def train_trajectory(
    cli,
    logger,
    output_root,
    model_root,
    seed,
    run,
    consensus_cache,
    consensus_paths,
    consensus_config,
):
    if run not in RUNS:
        raise ValueError("Unknown locked run {}.".format(run))
    setup_seed(seed)
    args = build_config(cli, seed)
    if int(args.update_epochs) != 10:
        raise RuntimeError("Teacher-consensus screen fixes update_epochs=10.")
    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("Teacher-consensus screen may construct only train/valid loaders.")
    teacher, student, assets = load_assets(cli, args, loaders, seed)
    optimizer = optim.Adam(student.parameters(), lr=args.learning_rate)
    assert_teacher_not_in_optimizer(teacher, optimizer)
    live_teacher_used = run == BASELINE_RUN
    if not live_teacher_used:
        del teacher
        teacher = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.patience,
    )
    criterion = nn.L1Loss()
    cosine = nn.CosineEmbeddingLoss()
    hinge = HingeLoss()
    missing_generator = torch.Generator().manual_seed(int(seed) + 104729)
    missing_hasher = MissingSequenceHasher()

    run_dir = output_root / "seed{}".format(seed) / run
    checkpoint = model_root / "seed{}".format(seed) / run / "best_valid.pth"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_j = float("inf")
    best_epoch = 0
    best_valid_metrics = None
    epoch_rows = []
    batch_sizes = None
    last_epoch = 0
    logger.info(
        "seed=%s run=%s optimizer=Adam target=%s gate=%s test=forbidden",
        seed,
        run,
        "single_teacher" if run == BASELINE_RUN else "five_teacher_mean",
        "compatibility_times_consensus" if run == CONSENSUS_RUN else "compatibility",
    )

    for epoch in range(1, (cli.max_epochs or 1000) + 1):
        last_epoch = epoch
        student.train()
        optimizer.zero_grad()
        counts = Counter({"LA": 0, "LV": 0, "L": 0})
        epoch_batch_sizes = []
        objective_rows = []
        window = []
        for step, batch in enumerate(loaders["train"], 1):
            batch_size = int(batch["labels"]["M"].shape[0])
            epoch_batch_sizes.append(batch_size)
            window.append(
                prepare_batch(batch, missing_generator, missing_hasher, counts)
            )
            flush = len(window) == int(args.update_epochs) or step == len(loaders["train"])
            if not flush:
                continue
            for prepared in window:
                loss, diagnostics = forward_objective(
                    prepared,
                    args,
                    run,
                    teacher,
                    student,
                    assets["cache_by_index"],
                    consensus_cache,
                    criterion,
                    cosine,
                    hinge,
                )
                loss.backward()
                objective_rows.append(diagnostics)
            if live_teacher_used and teacher_grad_count(teacher):
                raise RuntimeError("Frozen single Teacher received gradients.")
            optimizer.step()
            optimizer.zero_grad()
            window = []

        if batch_sizes is None:
            batch_sizes = epoch_batch_sizes
        elif batch_sizes != epoch_batch_sizes:
            raise RuntimeError("Batch-size sequence changed across epochs.")
        if epoch == 1 and int(seed) == 1111:
            expected_counts = Counter({"LA": 435, "LV": 430, "L": 419})
            if counts != expected_counts:
                raise RuntimeError(
                    "Missing-mask sequence changed: {} vs {}".format(counts, expected_counts)
                )

        valid = evaluate_all_modes(
            student,
            loaders["valid"],
            args.device,
            "moddrop",
            criterion,
        )
        j_valid = validation_objective(valid)
        if not math.isfinite(j_valid):
            raise FloatingPointError("Non-finite Valid J.")
        scheduler.step(j_valid)
        is_best = j_valid <= best_j - 1e-6
        if is_best:
            best_j = j_valid
            best_epoch = epoch
            best_valid_metrics = valid
            torch.save(student.state_dict(), checkpoint)
        row = {
            "Seed": int(seed),
            "Run": run,
            "Epoch": int(epoch),
            "J_valid": float(j_valid),
            "IsBestValid": bool(is_best),
            "full_loss": float(np.mean([item["full_loss"] for item in objective_rows])),
            "missing_loss": float(np.mean([item["missing_loss"] for item in objective_rows])),
            "KD_loss": float(np.mean([item["kd_loss"] for item in objective_rows])),
            "compatibility_mean": float(
                np.mean([item["compatibility_mean"] for item in objective_rows])
            ),
            "teacher_variance_mean": float(
                np.mean([item["teacher_variance_mean"] for item in objective_rows])
            ),
            "consensus_weight_mean": float(
                np.mean([item["consensus_weight_mean"] for item in objective_rows])
            ),
            "final_gate_mean": float(
                np.mean([item["final_gate_mean"] for item in objective_rows])
            ),
            "teacher_target_mean": float(
                np.mean([item["teacher_target_mean"] for item in objective_rows])
            ),
            **_flatten(valid, "valid"),
        }
        epoch_rows.append(row)
        logger.info(
            "seed=%s run=%s epoch=%s J=%.6f LAV=%.6f MissingMacro=%.6f "
            "KD=%.6f compat=%.6f consensus=%.6f gate=%.6f",
            seed,
            run,
            epoch,
            j_valid,
            valid["LAV"]["MAE"],
            np.mean([valid[mode]["MAE"] for mode in MISSING_MODES]),
            row["KD_loss"],
            row["compatibility_mean"],
            row["consensus_weight_mean"],
            row["final_gate_mean"],
        )
        if epoch - best_epoch >= args.early_stop:
            break

    if not checkpoint.is_file() or best_valid_metrics is None:
        raise RuntimeError("Validation-best checkpoint absent for seed={} run={}.".format(seed, run))
    expected_sha, expected_count = expected_missing_sequence_sha(seed, last_epoch, batch_sizes)
    if missing_hasher.hexdigest() != expected_sha or missing_hasher.count != expected_count:
        raise RuntimeError("Missing-mode sequence hash differs from original Stage 3.")

    student.load_state_dict(torch.load(checkpoint, map_location=args.device), strict=True)
    final_valid = evaluate_all_modes(
        student,
        loaders["valid"],
        args.device,
        "moddrop",
        criterion,
    )
    predictions = prediction_rows(student, loaders["valid"], args.device)
    predictions["Seed"] = int(seed)
    predictions["Run"] = run
    predictions["Split"] = "valid"
    predictions["SelectedBy"] = "validation_J"
    pd.DataFrame(epoch_rows).to_csv(run_dir / "epoch_metrics.csv", index=False)
    predictions.to_csv(run_dir / "valid_predictions.csv", index=False)

    result = {
        "Seed": int(seed),
        "Run": run,
        "Method": "DLF-CFCompatKD-v1" if run == BASELINE_RUN else METHOD,
        "TeacherTarget": "single_seed_teacher" if run == BASELINE_RUN else "five_teacher_mean",
        "DistillationGate": (
            "counterfactual_compatibility_times_teacher_consensus"
            if run == CONSENSUS_RUN
            else "counterfactual_compatibility"
        ),
        "BestValidEpoch": int(best_epoch),
        "TrainEpochCount": int(last_epoch),
        "J_valid": float(validation_objective(final_valid)),
        "MainCheckpoint": str(checkpoint),
        "MainCheckpointSHA256": checkpoint_sha256(checkpoint),
        "MissingSequenceSHA256": missing_hasher.hexdigest(),
        "MissingSequenceCount": int(missing_hasher.count),
        "BatchSizeSequenceJSON": json.dumps([int(value) for value in batch_sizes]),
        "StudentInitializationCheckpoint": str(assets["teacher_checkpoint"]),
        "StudentInitializationSHA256": assets["teacher_sha"],
        "LiveTeacherUsedDuringTraining": bool(live_teacher_used),
        "FrozenTeacherGradientCount": int(
            teacher_grad_count(teacher) if live_teacher_used else 0
        ),
        "TeacherEnsembleSeedCount": int(len(TEACHER_SEEDS) if run != BASELINE_RUN else 1),
        "TeacherConsensusCache": str(consensus_paths["csv"]),
        "TeacherConsensusCacheSHA256": checkpoint_sha256(consensus_paths["csv"]),
        "TeacherVarianceMedian": float(consensus_config["variance_median"]),
        "EvaluatorCheckpoint": str(assets["evaluator_checkpoint"]),
        "EvaluatorSHA256": assets["evaluator_sha"],
        "EvaluatorBestEpoch": int(assets["evaluator_best_epoch"]),
        "CompatibilityCache": str(assets["cache_csv"]),
        "CompatibilityCacheSHA256": checkpoint_sha256(assets["cache_csv"]),
        "TestConstructed": False,
        **_flatten(final_valid, "valid"),
    }
    return result, epoch_rows, predictions


def render_report(summary):
    lines = [
        "# CFCompatKD teacher-consensus v1: dual-seed Valid screen",
        "",
        "## Decision",
        "",
        "- Verdict: `{}`".format(summary["verdict"]),
        "- Ensemble-mean gate passed: `{}`".format(
            summary["candidate_gates"][MEAN_RUN]["passed"]
        ),
        "- Consensus-calibrated gate passed: `{}`".format(
            summary["candidate_gates"][CONSENSUS_RUN]["passed"]
        ),
        "- Official Test constructed: `False`",
        "- Additional inference parameters: `0`",
        "",
        "## Valid results",
        "",
        "| Seed | Baseline J | Ensemble mean J | Mean gain | Consensus J | Consensus gain |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed in FORMAL_SEEDS:
        baseline = summary["baselines"][str(seed)]
        mean_row = next(
            row for row in summary["candidates"]
            if int(row["Seed"]) == seed and row["Run"] == MEAN_RUN
        )
        consensus_row = next(
            row for row in summary["candidates"]
            if int(row["Seed"]) == seed and row["Run"] == CONSENSUS_RUN
        )
        lines.append(
            "| {} | {:.6f} | {:.6f} | {:+.6f} | {:.6f} | {:+.6f} |".format(
                seed,
                baseline["J_valid"],
                mean_row["J_valid"],
                baseline["J_valid"] - mean_row["J_valid"],
                consensus_row["J_valid"],
                baseline["J_valid"] - consensus_row["J_valid"],
            )
        )
    lines.extend(
        [
            "",
            "## Frozen teacher consensus",
            "",
            "- Teacher seeds: `{}`".format(list(TEACHER_SEEDS)),
            "- Target: arithmetic mean of five clean LAV predictions",
            "- Disagreement: population variance across five predictions",
            "- Scale: train-only median variance `{:.9g}`".format(
                summary["teacher_consensus"]["variance_median"]
            ),
            "- Weight: `1/(1+variance/median_train_variance)`",
            "",
            "The ensemble-mean run is an ablation. The consensus-calibrated run is "
            "the primary extension. Neither result is selected by a hyperparameter grid.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    cli = parse_args()
    output_root, model_root = result_paths(cli)
    logger, log_path = create_logger(cli)
    (
        consensus_frame,
        consensus_by_index,
        consensus_config_loaded,
        consensus_paths,
        consensus_config,
    ) = build_train_consensus_cache(cli, output_root, logger)
    if consensus_config_loaded != consensus_config:
        raise RuntimeError("Written and reloaded consensus cache configurations differ.")

    baselines = {}
    baseline_results = []
    candidate_results = []
    all_epochs = []
    all_predictions = []
    replay_checks = {}
    baseline_sources = []

    for seed in FORMAL_SEEDS:
        reference_series, reference_path = load_stage3_reference(cli.result_root, seed)
        reference = series_to_dict(reference_series)
        baselines[str(seed)] = reference
        baseline_sources.append(
            {
                "seed": int(seed),
                "path": str(reference_path.resolve()),
                "sha256": checkpoint_sha256(reference_path),
            }
        )
        replay, epochs, predictions = train_trajectory(
            cli,
            logger,
            output_root,
            model_root,
            seed,
            BASELINE_RUN,
            consensus_by_index,
            consensus_paths,
            consensus_config,
        )
        check = replay_gate(replay, reference)
        replay_checks[str(seed)] = check
        if not cli.smoke_test and not check["passed"]:
            raise RuntimeError("Seed {} Stage-3 replay failed: {}".format(seed, check))
        replay["BaselineResult"] = str(reference_path)
        replay["BaselineResultSHA256"] = checkpoint_sha256(reference_path)
        baseline_results.append(replay)
        all_epochs.extend(epochs)
        all_predictions.append(predictions)

        for run in CANDIDATE_RUNS:
            result, local_epochs, local_predictions = train_trajectory(
                cli,
                logger,
                output_root,
                model_root,
                seed,
                run,
                consensus_by_index,
                consensus_paths,
                consensus_config,
            )
            candidate_results.append(result)
            all_epochs.extend(local_epochs)
            all_predictions.append(local_predictions)

    candidate_gates = {
        run: aggregate_candidate_gate(
            run,
            candidate_results,
            {int(seed): baselines[str(seed)] for seed in FORMAL_SEEDS},
            all_epochs,
        )
        for run in CANDIDATE_RUNS
    }
    verdict = verdict_from_gates(candidate_gates)

    grid_path = output_root / "teacher_consensus_valid_grid_summary.csv"
    epochs_path = output_root / "teacher_consensus_all_epoch_metrics.csv"
    predictions_path = output_root / "teacher_consensus_all_valid_predictions.csv"
    pd.DataFrame(baseline_results + candidate_results).to_csv(grid_path, index=False)
    pd.DataFrame(all_epochs).to_csv(epochs_path, index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(predictions_path, index=False)

    source_manifest = {
        "version": VERSION,
        "method": METHOD,
        "base_branch": "feature/cfcompat-sam-valid-screen-v1",
        "formal_seeds": list(FORMAL_SEEDS),
        "teacher_seeds": list(TEACHER_SEEDS),
        "runs": list(RUNS),
        "candidate_selection": "none_each_candidate_has_an_independent_gate",
        "teacher_consensus_cache": {
            "csv": str(consensus_paths["csv"].resolve()),
            "csv_sha256": checkpoint_sha256(consensus_paths["csv"]),
            "config": str(consensus_paths["config"].resolve()),
            "config_sha256": checkpoint_sha256(consensus_paths["config"]),
        },
        "baseline_sources": baseline_sources,
        "artifacts": {
            grid_path.name: {
                "path": str(grid_path.resolve()),
                "sha256": checkpoint_sha256(grid_path),
            },
            epochs_path.name: {
                "path": str(epochs_path.resolve()),
                "sha256": checkpoint_sha256(epochs_path),
            },
            predictions_path.name: {
                "path": str(predictions_path.resolve()),
                "sha256": checkpoint_sha256(predictions_path),
            },
        },
        "checkpoints": [
            {
                "seed": int(row["Seed"]),
                "run": row["Run"],
                "path": str(Path(row["MainCheckpoint"]).resolve()),
                "sha256": row["MainCheckpointSHA256"],
            }
            for row in baseline_results + candidate_results
        ],
        "test_constructed": False,
        "test_loader_construction_count": 0,
        "test_loader_traversal_count": 0,
        "additional_inference_parameters": 0,
    }
    manifest_path = output_root / "teacher_consensus_source_manifest.json"
    manifest_path.write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "version": VERSION,
        "method": METHOD,
        "verdict": verdict,
        "baselines": baselines,
        "baseline_replays": baseline_results,
        "baseline_replay_gates": replay_checks,
        "candidates": candidate_results,
        "candidate_gates": candidate_gates,
        "teacher_consensus": {
            "teacher_seeds": list(TEACHER_SEEDS),
            "sample_count": int(len(consensus_frame)),
            "variance_median": float(consensus_config["variance_median"]),
            "weight_formula": consensus_config["consensus_weight_formula"],
        },
        "protocol": {
            "decision_split": "official_valid_only_two_seeds",
            "formal_seeds": list(FORMAL_SEEDS),
            "teacher_seeds": list(TEACHER_SEEDS),
            "candidate_runs": list(CANDIDATE_RUNS),
            "candidate_selection": "none",
            "base_optimizer": "Adam",
            "update_epochs": 10,
            "original_full_and_missing_losses_unchanged": True,
            "single_teacher_baseline_exact_replay_required": not cli.smoke_test,
            "teacher_cache_source": "official_train_only",
            "teacher_models_used_only_during_cache_construction": True,
            "official_test_constructed": False,
            "official_test_authorized": False,
            "next_stage_on_pass": "mosei_single_seed_valid_screen",
            "additional_inference_parameters": 0,
        },
    }
    summary_path = output_root / "teacher_consensus_valid_screen_summary.json"
    report_path = output_root / "teacher_consensus_valid_screen_report.md"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_report(summary), encoding="utf-8")
    logger.info(
        "complete verdict=%s mean_gain_mean=%+.6f mean_gain_consensus=%+.6f output=%s log=%s",
        verdict,
        candidate_gates[MEAN_RUN]["mean_gain_valid_J"],
        candidate_gates[CONSENSUS_RUN]["mean_gain_valid_J"],
        output_root,
        log_path,
    )
    print("CFCompatKD teacher-consensus dual-seed Valid screen complete")
    print(
        "ensemble-mean Valid J gain:",
        "{:+.6f}".format(candidate_gates[MEAN_RUN]["mean_gain_valid_J"]),
    )
    print(
        "consensus-calibrated Valid J gain:",
        "{:+.6f}".format(candidate_gates[CONSENSUS_RUN]["mean_gain_valid_J"]),
    )
    print("verdict:", verdict)
    print("official Test was not constructed")
    print("report:", report_path)


if __name__ == "__main__":
    main()
