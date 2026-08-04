"""Locked two-seed, validation-only SAM screen for CFCompatKD."""
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
    CACHE_VERSION,
    MULTISEED_CACHE_VERSION,
    compatibility_for_modes,
    gate_weights,
    gated_kd_loss,
    modes_from_masks,
)
from trains.singleTask.cfcompat_sam_utils import (
    FORMAL_SEEDS,
    METHOD,
    OUTPUT_TAG,
    SAM_RHOS,
    VERSION,
    SAMController,
    aggregate_valid_gate,
    capture_rng_state,
    load_locked_counterfactual_cache,
    portable_locate_stage1_evaluator,
    replay_gate,
    restore_rng_state,
    rng_states_equal,
    select_rho,
)
from trains.singleTask.cfcompat_stability_utils import (
    MissingSequenceHasher,
    expected_missing_sequence_sha,
    load_stage3_reference,
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
        description="CFCompatKD + SAM two-seed validation-only screen."
    )
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(FORMAL_SEEDS)
    )
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
        parser.error(
            "Formal SAM screen fixes seeds to 1111 1114 in that order."
        )
    if args.num_workers != 1:
        parser.error("Formal SAM screen fixes num_workers=1.")
    if args.smoke_test:
        args.max_epochs = (
            2 if args.max_epochs is None else min(2, args.max_epochs)
        )
    elif args.max_epochs is not None:
        parser.error("Formal runs use the frozen DLF early-stop rule.")
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
    path = directory / "DLF-mosi-cfcompat-sam-valid-screen-{}-{}.log".format(
        kind, datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    logger = logging.getLogger("cfcompat_sam_valid_screen")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
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
    return {
        str(key): json_value(value) for key, value in series.items()
    }


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
    cache_frame, cache_by_index, cache_paths = (
        load_locked_counterfactual_cache(
            cli.result_root,
            cli.dataset,
            version=cache_version,
            seed=seed if multiseed else None,
            expected_evaluator_sha=evaluator_sha,
        )
    )
    if len(cache_frame) != 1284:
        raise RuntimeError(
            "MOSI SAM screen requires the locked 1284-sample cache."
        )
    teacher, student, teacher_checkpoint, teacher_sha = (
        initialize_teacher_student(args, cli, seed, loaders)
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


def rho_tag(rho):
    return (
        "adam_replay"
        if rho is None
        else "sam_rho_{}".format(str(rho).replace(".", "p"))
    )


def prepare_batch(
    batch,
    missing_generator,
    missing_hasher,
    counts,
):
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
    teacher,
    student,
    cache_by_index,
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

    full_mask = mode_to_mask(
        "LAV", labels.size(0), args.device, audio.dtype
    )
    full_output = student(text, audio, vision, full_mask)
    full_loss, _ = compute_full_dlf_loss(
        full_output, labels, criterion, cosine, hinge
    )
    missing_output = student(text, audio, vision, missing_mask)
    missing_loss, _ = compute_task_loss(
        missing_output, labels, criterion
    )
    teacher_prediction = teacher_lav_prediction(
        teacher, text, audio, vision
    )
    indices = (
        batch["index"].view(-1).cpu().numpy().astype(int).tolist()
    )
    compatibility = compatibility_for_modes(
        cache_by_index,
        indices,
        modes,
        args.device,
        labels.dtype,
    )
    gate, _ = gate_weights(
        compatibility, teacher_prediction, labels, "compat"
    )
    kd_loss, _ = gated_kd_loss(
        missing_output["output_logit"], teacher_prediction, gate
    )
    total_loss = full_loss + missing_loss + kd_loss
    if not torch.isfinite(total_loss):
        raise FloatingPointError("NaN/Inf in CFCompatKD objective.")
    return total_loss, {
        "full_loss": float(full_loss.detach().cpu()),
        "missing_loss": float(missing_loss.detach().cpu()),
        "kd_loss": float(kd_loss.detach().cpu()),
    }


def run_adam_window(
    window,
    args,
    teacher,
    student,
    optimizer,
    assets,
    criterion,
    cosine,
    hinge,
):
    diagnostics = []
    for prepared in window:
        loss, row = forward_objective(
            prepared,
            args,
            teacher,
            student,
            assets["cache_by_index"],
            criterion,
            cosine,
            hinge,
        )
        loss.backward()
        diagnostics.append(row)
    if teacher_grad_count(teacher):
        raise RuntimeError(
            "Frozen Teacher received gradients in Adam replay."
        )
    optimizer.step()
    optimizer.zero_grad()
    return diagnostics, None


def run_sam_window(
    window,
    args,
    teacher,
    student,
    optimizer,
    sam,
    assets,
    criterion,
    cosine,
    hinge,
):
    optimizer.zero_grad()
    state_before = capture_rng_state()
    first_rows = []
    for prepared in window:
        loss, row = forward_objective(
            prepared,
            args,
            teacher,
            student,
            assets["cache_by_index"],
            criterion,
            cosine,
            hinge,
        )
        loss.backward()
        first_rows.append(row)
    if teacher_grad_count(teacher):
        raise RuntimeError(
            "Frozen Teacher received first-pass SAM gradients."
        )
    state_after_first = capture_rng_state()
    first_grad_norm = sam.ascent_step()

    optimizer.zero_grad()
    restore_rng_state(state_before)
    second_rows = []
    for prepared in window:
        loss, row = forward_objective(
            prepared,
            args,
            teacher,
            student,
            assets["cache_by_index"],
            criterion,
            cosine,
            hinge,
        )
        loss.backward()
        second_rows.append(row)
    if teacher_grad_count(teacher):
        raise RuntimeError(
            "Frozen Teacher received second-pass SAM gradients."
        )
    state_after_second = capture_rng_state()
    if not rng_states_equal(state_after_first, state_after_second):
        sam.restore()
        raise RuntimeError(
            "SAM second pass consumed a different RNG trajectory."
        )
    second_grad_norm = float(sam.grad_norm().detach().cpu())
    if not math.isfinite(second_grad_norm) or second_grad_norm <= 0:
        sam.restore()
        raise RuntimeError(
            "SAM second-pass gradient norm is invalid."
        )
    sam.descent_step(optimizer)
    optimizer.zero_grad()
    return second_rows, {
        "first_grad_norm": first_grad_norm,
        "second_grad_norm": second_grad_norm,
        "first_kd_loss": float(
            np.mean([row["kd_loss"] for row in first_rows])
        ),
        "second_kd_loss": float(
            np.mean([row["kd_loss"] for row in second_rows])
        ),
    }


def train_trajectory(
    cli,
    logger,
    output_root,
    model_root,
    seed,
    rho,
):
    setup_seed(seed)
    args = build_config(cli, seed)
    if int(args.update_epochs) != 10:
        raise RuntimeError(
            "SAM screen is frozen for the original update_epochs=10."
        )
    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError(
            "SAM validation screen may construct only train/valid loaders."
        )
    teacher, student, assets = load_assets(
        cli, args, loaders, seed
    )
    optimizer = optim.Adam(
        student.parameters(), lr=args.learning_rate
    )
    assert_teacher_not_in_optimizer(teacher, optimizer)
    sam = (
        None
        if rho is None
        else SAMController(student.parameters(), rho)
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.patience,
    )
    criterion = nn.L1Loss()
    cosine = nn.CosineEmbeddingLoss()
    hinge = HingeLoss()
    missing_generator = torch.Generator().manual_seed(
        int(seed) + 104729
    )
    missing_hasher = MissingSequenceHasher()

    tag = rho_tag(rho)
    run_dir = output_root / f"seed{seed}" / tag
    checkpoint = (
        model_root / f"seed{seed}" / tag / "best_valid.pth"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    best_j = float("inf")
    best_epoch = 0
    best_valid_metrics = None
    epoch_rows = []
    batch_sizes = None
    last_epoch = 0
    logger.info(
        "seed=%s run=%s rho=%s optimizer=%s update_epochs=%s "
        "test=forbidden",
        seed,
        tag,
        rho,
        "Adam" if rho is None else "SAM(Adam)",
        args.update_epochs,
    )

    for epoch in range(1, (cli.max_epochs or 1000) + 1):
        last_epoch = epoch
        student.train()
        optimizer.zero_grad()
        counts = Counter({"LA": 0, "LV": 0, "L": 0})
        epoch_batch_sizes = []
        objective_rows = []
        sam_rows = []
        window = []

        for step, batch in enumerate(loaders["train"], 1):
            batch_size = int(batch["labels"]["M"].shape[0])
            epoch_batch_sizes.append(batch_size)
            window.append(
                prepare_batch(
                    batch,
                    missing_generator,
                    missing_hasher,
                    counts,
                )
            )
            flush = (
                len(window) == int(args.update_epochs)
                or step == len(loaders["train"])
            )
            if not flush:
                continue
            if rho is None:
                local_rows, local_sam = run_adam_window(
                    window,
                    args,
                    teacher,
                    student,
                    optimizer,
                    assets,
                    criterion,
                    cosine,
                    hinge,
                )
            else:
                local_rows, local_sam = run_sam_window(
                    window,
                    args,
                    teacher,
                    student,
                    optimizer,
                    sam,
                    assets,
                    criterion,
                    cosine,
                    hinge,
                )
            objective_rows.extend(local_rows)
            if local_sam is not None:
                sam_rows.append(local_sam)
            window = []

        if batch_sizes is None:
            batch_sizes = epoch_batch_sizes
        elif batch_sizes != epoch_batch_sizes:
            raise RuntimeError(
                "Batch-size sequence changed across epochs."
            )
        if epoch == 1 and int(seed) == 1111:
            expected_counts = Counter(
                {"LA": 435, "LV": 430, "L": 419}
            )
            if counts != expected_counts:
                raise RuntimeError(
                    f"Missing-mask sequence changed: "
                    f"{counts} vs {expected_counts}"
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
            "Run": tag,
            "Rho": 0.0 if rho is None else float(rho),
            "Optimizer": "Adam" if rho is None else "SAM(Adam)",
            "Epoch": int(epoch),
            "J_valid": float(j_valid),
            "IsBestValid": bool(is_best),
            "full_loss": float(
                np.mean(
                    [item["full_loss"] for item in objective_rows]
                )
            ),
            "missing_loss": float(
                np.mean(
                    [item["missing_loss"] for item in objective_rows]
                )
            ),
            "KD_loss": float(
                np.mean(
                    [item["kd_loss"] for item in objective_rows]
                )
            ),
            "sam_first_grad_norm": (
                float(
                    np.mean(
                        [
                            item["first_grad_norm"]
                            for item in sam_rows
                        ]
                    )
                )
                if sam_rows
                else 0.0
            ),
            "sam_second_grad_norm": (
                float(
                    np.mean(
                        [
                            item["second_grad_norm"]
                            for item in sam_rows
                        ]
                    )
                )
                if sam_rows
                else 0.0
            ),
            **_flatten(valid, "valid"),
        }
        epoch_rows.append(row)
        logger.info(
            "seed=%s run=%s epoch=%s J_valid=%.6f LAV=%.6f "
            "MissingMacro=%.6f KD=%.6f grad1=%.6f grad2=%.6f",
            seed,
            tag,
            epoch,
            j_valid,
            valid["LAV"]["MAE"],
            np.mean(
                [valid[mode]["MAE"] for mode in MISSING_MODES]
            ),
            row["KD_loss"],
            row["sam_first_grad_norm"],
            row["sam_second_grad_norm"],
        )
        if epoch - best_epoch >= args.early_stop:
            break

    if not checkpoint.is_file() or best_valid_metrics is None:
        raise RuntimeError(
            f"Validation-best checkpoint is absent for "
            f"seed={seed} run={tag}."
        )
    expected_sha, expected_count = expected_missing_sequence_sha(
        seed, last_epoch, batch_sizes
    )
    if (
        missing_hasher.hexdigest() != expected_sha
        or missing_hasher.count != expected_count
    ):
        raise RuntimeError(
            "Missing-mode sequence hash differs from Stage 3."
        )

    student.load_state_dict(
        torch.load(checkpoint, map_location=args.device), strict=True
    )
    final_valid = evaluate_all_modes(
        student,
        loaders["valid"],
        args.device,
        "moddrop",
        criterion,
    )
    predictions = prediction_rows(
        student, loaders["valid"], args.device
    )
    predictions["Seed"] = int(seed)
    predictions["Run"] = tag
    predictions["Rho"] = 0.0 if rho is None else float(rho)
    predictions["Split"] = "valid"
    predictions["SelectedBy"] = "validation_J"
    pd.DataFrame(epoch_rows).to_csv(
        run_dir / "epoch_metrics.csv", index=False
    )
    predictions.to_csv(
        run_dir / "valid_predictions.csv", index=False
    )

    result = {
        "Seed": int(seed),
        "Run": tag,
        "Rho": 0.0 if rho is None else float(rho),
        "Method": (
            "DLF-CFCompatKD-v1" if rho is None else METHOD
        ),
        "Optimizer": "Adam" if rho is None else "SAM(Adam)",
        "BestValidEpoch": int(best_epoch),
        "J_valid": float(validation_objective(final_valid)),
        "MainCheckpoint": str(checkpoint),
        "MainCheckpointSHA256": checkpoint_sha256(checkpoint),
        "MissingSequenceSHA256": missing_hasher.hexdigest(),
        "MissingSequenceCount": int(missing_hasher.count),
        "TeacherCheckpoint": str(assets["teacher_checkpoint"]),
        "TeacherSHA256": assets["teacher_sha"],
        "EvaluatorCheckpoint": str(
            assets["evaluator_checkpoint"]
        ),
        "EvaluatorSHA256": assets["evaluator_sha"],
        "EvaluatorBestEpoch": int(
            assets["evaluator_best_epoch"]
        ),
        "CompatibilityCache": str(assets["cache_csv"]),
        "CompatibilityCacheSHA256": checkpoint_sha256(
            assets["cache_csv"]
        ),
        "TestConstructed": False,
        **_flatten(final_valid, "valid"),
    }
    return result, epoch_rows, predictions


def render_report(summary):
    lines = [
        "# CFCompatKD + SAM v1: two-seed Valid screen",
        "",
        "## Decision",
        "",
        f"- Verdict: `{summary['verdict']}`",
        f"- Selected rho: `{summary['selected_rho']}`",
        f"- Dual-seed Valid gate passed: "
        f"`{summary['valid_gate']['passed']}`",
        "- Official Test constructed: `False`",
        "- Next stage on pass: `train-only grouped stability screen`",
        "",
        "## Valid summary",
        "",
        "| Seed | Baseline J | SAM J | Gain |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for seed in FORMAL_SEEDS:
        base = summary["baselines"][str(seed)]
        candidate = next(
            row
            for row in summary["selected_candidates"]
            if int(row["Seed"]) == seed
        )
        lines.append(
            "| {} | {:.6f} | {:.6f} | {:+.6f} |".format(
                seed,
                base["J_valid"],
                candidate["J_valid"],
                base["J_valid"] - candidate["J_valid"],
            )
        )
    return "\n".join(lines) + "\n"


def main():
    cli = parse_args()
    output_root, model_root = result_paths(cli)
    logger, log_path = create_logger(cli)
    baselines = {}
    baseline_results = []
    candidate_results = []
    all_epochs = []
    all_predictions = []
    replay_checks = {}

    for seed in FORMAL_SEEDS:
        reference_series, reference_path = load_stage3_reference(
            cli.result_root, seed
        )
        reference = series_to_dict(reference_series)
        baselines[str(seed)] = reference
        replay, epochs, predictions = train_trajectory(
            cli,
            logger,
            output_root,
            model_root,
            seed,
            rho=None,
        )
        check = replay_gate(replay, reference)
        replay_checks[str(seed)] = check
        if not check["passed"]:
            raise RuntimeError(
                f"Seed {seed} Stage-3 replay failed: {check}"
            )
        replay["BaselineResult"] = str(reference_path)
        replay["BaselineResultSHA256"] = checkpoint_sha256(
            reference_path
        )
        baseline_results.append(replay)
        all_epochs.extend(epochs)
        all_predictions.append(predictions)

        for rho in SAM_RHOS:
            result, local_epochs, local_predictions = train_trajectory(
                cli,
                logger,
                output_root,
                model_root,
                seed,
                rho=float(rho),
            )
            candidate_results.append(result)
            all_epochs.extend(local_epochs)
            all_predictions.append(local_predictions)

    selected_rho = select_rho(candidate_results)
    valid_gate = aggregate_valid_gate(
        selected_rho,
        candidate_results,
        {
            int(seed): baselines[str(seed)]
            for seed in FORMAL_SEEDS
        },
        all_epochs,
    )
    selected_candidates = [
        row
        for row in candidate_results
        if math.isclose(
            float(row["Rho"]), float(selected_rho), abs_tol=0.0
        )
    ]
    verdict = (
        "PROMOTE_SAM_TO_TRAIN_ONLY_GROUP_SCREEN"
        if valid_gate["passed"]
        else "STOP_SAM_DUAL_SEED_VALID_FAILED"
    )

    pd.DataFrame(
        baseline_results + candidate_results
    ).to_csv(
        output_root / "sam_valid_grid_summary.csv", index=False
    )
    pd.DataFrame(all_epochs).to_csv(
        output_root / "sam_all_epoch_metrics.csv", index=False
    )
    pd.concat(all_predictions, ignore_index=True).to_csv(
        output_root / "sam_all_valid_predictions.csv", index=False
    )

    source_manifest = {
        "version": VERSION,
        "method": METHOD,
        "base_branch": (
            "experiment/cfcompat-distillation-evidence-v1"
        ),
        "formal_seeds": list(FORMAL_SEEDS),
        "rho_grid": list(SAM_RHOS),
        "selected_rho": float(selected_rho),
        "test_constructed": False,
        "test_loader_construction_count": 0,
        "test_loader_traversal_count": 0,
        "additional_inference_parameters": 0,
        "checkpoints": [
            {
                "seed": int(row["Seed"]),
                "run": row["Run"],
                "path": row["MainCheckpoint"],
                "sha256": row["MainCheckpointSHA256"],
            }
            for row in baseline_results + candidate_results
        ],
    }
    (
        output_root / "sam_source_manifest.json"
    ).write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True)
        + "\n",
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
        "selected_rho": float(selected_rho),
        "selected_candidates": selected_candidates,
        "valid_gate": valid_gate,
        "protocol": {
            "decision_split": "official_valid_only_two_seeds",
            "formal_seeds": list(FORMAL_SEEDS),
            "rho_grid": list(SAM_RHOS),
            "base_optimizer": "Adam",
            "update_epochs": 10,
            "same_microbatch_window_two_pass": True,
            "same_missing_masks_two_pass": True,
            "same_dropout_rng_two_pass": True,
            "original_cfcompat_objective_unchanged": True,
            "official_test_constructed": False,
            "official_test_authorized": False,
            "next_stage_on_pass": (
                "train_only_group_stability_screen"
            ),
            "additional_inference_parameters": 0,
        },
    }
    (
        output_root / "sam_valid_screen_summary.json"
    ).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (
        output_root / "sam_valid_screen_report.md"
    ).write_text(
        render_report(summary), encoding="utf-8"
    )
    logger.info(
        "complete verdict=%s selected_rho=%.6g mean_gain=%+.6f "
        "output=%s log=%s",
        verdict,
        selected_rho,
        valid_gate["mean_gain_valid_J"],
        output_root,
        log_path,
    )
    print("CFCompatKD + SAM dual-seed Valid screen complete")
    print("selected rho:", selected_rho)
    print(
        "mean Valid J gain:",
        "{:+.6f}".format(valid_gate["mean_gain_valid_J"]),
    )
    print("verdict:", verdict)
    print("official Test was not constructed")
    print("report:", output_root / "sam_valid_screen_report.md")


if __name__ == "__main__":
    main()
