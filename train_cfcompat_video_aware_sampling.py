"""Train CFCompatKD with deterministic source-video-aware batches only."""

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
    build_single_split_loader,
    compute_full_dlf_loss,
    compute_task_loss,
    count_missing_modes,
    evaluate_all_modes,
    mode_to_mask,
    regression_metrics,
    sample_missing_masks,
    validation_objective,
)
from trains.singleTask.video_aware_sampling_utils import (
    DISCOVERY_REPLAY_TOLERANCE,
    METHOD,
    OUTPUT_TAG,
    REPLAY_TOLERANCE,
    SAMPLES_PER_VIDEO,
    VERSION,
    aggregate_batch_statistics,
    batch_video_statistics,
    build_video_aware_train_loader,
    candidate_test_gate,
    candidate_valid_gate,
    load_locked_counterfactual_cache,
    parse_video_id,
    portable_locate_stage1_evaluator,
)
from utils.functions import setup_seed


FORMAL_SEEDS = (1111, 1114)


def parse_args():
    parser = argparse.ArgumentParser(
        description="CFCompatKD source-video-aware batch sampling screen."
    )
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seed", type=int, choices=FORMAL_SEEDS, default=1114)
    parser.add_argument(
        "--samples-per-video", type=int, default=SAMPLES_PER_VIDEO
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

    if args.num_workers != 1:
        parser.error("Formal video-aware sampling fixes num_workers=1.")
    if args.samples_per_video != SAMPLES_PER_VIDEO:
        parser.error("Formal video-aware sampling fixes samples_per_video=4.")
    if args.smoke_test:
        args.max_epochs = 2 if args.max_epochs is None else min(2, args.max_epochs)
    elif args.max_epochs is not None:
        parser.error("Formal runs use the frozen DLF early-stop rule.")
    return args


def result_paths(cli):
    root = (
        Path(cli.result_root)
        / "missing_baseline"
        / OUTPUT_TAG
        / cli.dataset
        / "seed{}".format(cli.seed)
    )
    model = (
        Path(cli.model_save_dir)
        / "missing_baseline"
        / OUTPUT_TAG
        / cli.dataset
        / "seed{}".format(cli.seed)
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
    path = directory / "DLF-{}-video-aware-sampling-seed{}-{}-{}.log".format(
        cli.dataset,
        cli.seed,
        kind,
        datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    logger = logging.getLogger("cfcompat_video_aware_sampling")
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


def load_assets(cli, args, loaders):
    multiseed = int(cli.seed) != 1111
    evaluator_checkpoint, evaluator_best_epoch, evaluator_source = (
        portable_locate_stage1_evaluator(
            cli.result_root,
            cli.dataset,
            cli.seed,
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
        seed=cli.seed if multiseed else None,
        expected_evaluator_sha=evaluator_sha,
    )
    teacher, student, teacher_checkpoint, teacher_sha = initialize_teacher_student(
        args, cli, cli.seed, loaders
    )
    return teacher, student, {
        "teacher_checkpoint": teacher_checkpoint,
        "teacher_sha": teacher_sha,
        "evaluator_checkpoint": evaluator_checkpoint,
        "evaluator_sha": evaluator_sha,
        "evaluator_best_epoch": evaluator_best_epoch,
        "evaluator_source": evaluator_source,
        "cache_frame": cache_frame,
        "cache_by_index": cache_by_index,
        "cache_version": cache_version,
        "cache_csv": cache_paths["csv"],
        "cache_config": cache_paths["config"],
    }


def trajectory_tag(video_aware: bool) -> str:
    return "video_aware_sampling" if video_aware else "stage3_replay"


def train_trajectory(cli, logger, output_root, model_root, video_aware):
    setup_seed(cli.seed)
    args = build_config(cli, cli.seed)
    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("Training may construct only train and valid loaders.")

    sampler = None
    if video_aware:
        loaders["train"], sampler = build_video_aware_train_loader(
            args,
            cli.num_workers,
            cli.seed,
            samples_per_video=cli.samples_per_video,
        )

    teacher, student, assets = load_assets(cli, args, loaders)
    optimizer = optim.Adam(student.parameters(), lr=args.learning_rate)
    assert_teacher_not_in_optimizer(teacher, optimizer)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience
    )
    criterion = nn.L1Loss()
    cosine = nn.CosineEmbeddingLoss()
    hinge = HingeLoss()
    missing_generator = torch.Generator().manual_seed(cli.seed + 104729)
    missing_hasher = MissingSequenceHasher()

    tag = trajectory_tag(video_aware)
    run_dir = output_root / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_root / tag / "best_valid.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    best_j = float("inf")
    best_epoch = 0
    best_valid_metrics = None
    epoch_rows = []
    batch_sizes = None
    last_epoch = 0

    logger.info(
        "run=%s seed=%s video_aware=%s objective=original_cfcompat test=locked",
        tag,
        cli.seed,
        video_aware,
    )

    for epoch in range(1, (cli.max_epochs or 1000) + 1):
        last_epoch = epoch
        student.train()
        optimizer.zero_grad()
        counts = Counter({"LA": 0, "LV": 0, "L": 0})
        kd_losses = []
        batch_rows = []
        epoch_batch_sizes = []

        for step, batch in enumerate(loaders["train"], 1):
            text, audio, vision, labels = batch_to_device(batch, args.device)
            epoch_batch_sizes.append(int(labels.size(0)))

            full_mask = mode_to_mask(
                "LAV", labels.size(0), args.device, audio.dtype
            )
            full_output = student(text, audio, vision, full_mask)
            full_loss, _ = compute_full_dlf_loss(
                full_output, labels, criterion, cosine, hinge
            )

            missing_mask = sample_missing_masks(
                labels.size(0),
                missing_generator,
                args.device,
                audio.dtype,
            )
            modes = modes_from_masks(missing_mask)
            missing_hasher.update(modes)
            counts.update(count_missing_modes(missing_mask))
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
                assets["cache_by_index"],
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
            kd_losses.append(float(kd_loss.detach().cpu()))

            total_loss = full_loss + missing_loss + kd_loss
            if not torch.isfinite(total_loss):
                raise FloatingPointError("NaN/Inf in sampler-only trajectory.")
            total_loss.backward()
            if teacher_grad_count(teacher):
                raise RuntimeError("Frozen Teacher received gradients.")
            if step % args.update_epochs == 0 or step == len(loaders["train"]):
                optimizer.step()
                optimizer.zero_grad()

            videos = [parse_video_id(value) for value in list(batch["id"])]
            batch_rows.append(batch_video_statistics(videos))

        if batch_sizes is None:
            batch_sizes = epoch_batch_sizes
        elif batch_sizes != epoch_batch_sizes:
            raise RuntimeError("Batch-size sequence changed across epochs.")

        if epoch == 1 and cli.seed == 1111:
            expected_counts = Counter({"LA": 435, "LV": 430, "L": 419})
            if counts != expected_counts:
                raise RuntimeError(
                    "Missing-mask sequence changed: {} vs {}".format(
                        counts, expected_counts
                    )
                )

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
            best_valid_metrics = valid
            torch.save(student.state_dict(), checkpoint)

        diagnostics = aggregate_batch_statistics(batch_rows)
        row = {
            "Seed": int(cli.seed),
            "Run": tag,
            "Epoch": int(epoch),
            "VideoAwareSampler": bool(video_aware),
            "SamplesPerVideo": int(cli.samples_per_video) if video_aware else 0,
            "J_valid": float(j_valid),
            "IsBestValid": bool(is_best),
            "KD_loss": float(np.mean(kd_losses)),
            **diagnostics,
            **_flatten(valid, "valid"),
        }
        epoch_rows.append(row)
        logger.info(
            "run=%s epoch=%s J_valid=%.6f LAV=%.6f MissingMacro=%.6f "
            "videos_per_batch=%.3f repeated_fraction=%.3f",
            tag,
            epoch,
            j_valid,
            valid["LAV"]["MAE"],
            np.mean([valid[mode]["MAE"] for mode in MISSING_MODES]),
            diagnostics["video_count"],
            diagnostics["repeated_sample_fraction"],
        )
        if epoch - best_epoch >= args.early_stop:
            break

    if not checkpoint.is_file() or best_valid_metrics is None:
        raise RuntimeError("Validation-best checkpoint is absent for {}.".format(tag))

    expected_sha, expected_count = expected_missing_sequence_sha(
        cli.seed, last_epoch, batch_sizes
    )
    if (
        missing_hasher.hexdigest() != expected_sha
        or missing_hasher.count != expected_count
    ):
        raise RuntimeError("Missing-mode sequence hash differs from Stage 3.")

    student.load_state_dict(
        torch.load(checkpoint, map_location=args.device), strict=True
    )
    final_valid = evaluate_all_modes(
        student, loaders["valid"], args.device, "moddrop", criterion
    )
    valid_predictions = prediction_rows(
        student, loaders["valid"], args.device
    )
    valid_predictions["Seed"] = cli.seed
    valid_predictions["Run"] = tag
    valid_predictions["Split"] = "valid"
    valid_predictions["SelectedBy"] = "validation_J"

    pd.DataFrame(epoch_rows).to_csv(
        run_dir / "epoch_metrics.csv", index=False
    )
    valid_predictions.to_csv(
        run_dir / "valid_predictions.csv", index=False
    )

    result = {
        "Seed": int(cli.seed),
        "Run": tag,
        "Method": METHOD if video_aware else "DLF-CFCompatKD-v1",
        "VideoAwareSampler": bool(video_aware),
        "SamplesPerVideo": int(cli.samples_per_video) if video_aware else 0,
        "BestValidEpoch": int(best_epoch),
        "J_valid": float(validation_objective(final_valid)),
        "MainCheckpoint": str(checkpoint),
        "MainCheckpointSHA256": checkpoint_sha256(checkpoint),
        "MissingSequenceSHA256": missing_hasher.hexdigest(),
        "MissingSequenceCount": int(missing_hasher.count),
        "SamplerEpochCount": int(sampler.epoch) if sampler is not None else None,
        "TeacherCheckpoint": str(assets["teacher_checkpoint"]),
        "TeacherSHA256": assets["teacher_sha"],
        "EvaluatorCheckpoint": str(assets["evaluator_checkpoint"]),
        "EvaluatorSHA256": assets["evaluator_sha"],
        "EvaluatorBestEpoch": int(assets["evaluator_best_epoch"]),
        "CompatibilityCache": str(assets["cache_csv"]),
        "CompatibilityCacheSHA256": checkpoint_sha256(assets["cache_csv"]),
        "TestConstructed": False,
        **_flatten(final_valid, "valid"),
    }
    return result, epoch_rows, valid_predictions, args


def replay_gate(replay: dict, reference: dict) -> dict:
    differences = {
        "J_valid": abs(float(replay["J_valid"]) - float(reference["J_valid"])),
    }
    for mode in ("LAV",) + MISSING_MODES:
        key = "valid_{}_MAE".format(mode)
        differences[key] = abs(float(replay[key]) - float(reference[key]))
    epoch_match = int(replay["BestValidEpoch"]) == int(reference["BestValidEpoch"])
    passed = epoch_match and all(
        value <= REPLAY_TOLERANCE for value in differences.values()
    )
    return {
        "passed": bool(passed),
        "epoch_match": bool(epoch_match),
        "tolerance": REPLAY_TOLERANCE,
        "differences": differences,
    }


def discovery_replay_gate(cli, candidate):
    if int(cli.seed) != 1111:
        return {"available": False, "passed": None, "reason": "seed_not_1111"}
    path = (
        Path(cli.result_root)
        / "missing_baseline"
        / "cfcompat_video_vrex_v1"
        / "mosi"
        / "seed1111"
        / "video_vrex_summary.json"
    )
    if not path.is_file():
        return {"available": False, "passed": None, "reason": "source_absent"}
    source = json.loads(path.read_text(encoding="utf-8"))["sampler_control"]
    keys = [
        "J_valid",
        "valid_LAV_MAE",
        "valid_LA_MAE",
        "valid_LV_MAE",
        "valid_L_MAE",
    ]
    differences = {
        key: abs(float(candidate[key]) - float(source[key])) for key in keys
    }
    epoch_match = int(candidate["BestValidEpoch"]) == int(source["BestValidEpoch"])
    return {
        "available": True,
        "passed": bool(
            epoch_match
            and all(
                value <= DISCOVERY_REPLAY_TOLERANCE
                for value in differences.values()
            )
        ),
        "epoch_match": bool(epoch_match),
        "tolerance": DISCOVERY_REPLAY_TOLERANCE,
        "differences": differences,
        "source": str(path),
        "source_sha256": checkpoint_sha256(path),
    }


def evaluate_and_predictions(model, loader, device, criterion):
    model.eval()
    modes = ("LAV",) + MISSING_MODES
    collected = {
        mode: {"prediction": [], "label": [], "loss": []}
        for mode in modes
    }
    rows = []
    with torch.no_grad():
        for batch in loader:
            text, audio, vision, labels = batch_to_device(batch, device)
            predictions = {}
            for mode in modes:
                mask = mode_to_mask(
                    mode, labels.size(0), device, audio.dtype
                )
                output = model(text, audio, vision, mask)["output_logit"]
                predictions[mode] = output
                collected[mode]["prediction"].append(output.detach().cpu())
                collected[mode]["label"].append(labels.detach().cpu())
                collected[mode]["loss"].append(
                    float(criterion(output, labels).detach().cpu())
                )
            indices = batch["index"].view(-1).cpu().numpy().astype(int)
            sample_ids = list(batch["id"])
            for offset, index in enumerate(indices):
                rows.append({
                    "sample_id": str(sample_ids[offset]),
                    "sample_index": int(index),
                    "label": float(labels[offset].item()),
                    **{
                        "{}_pred".format(mode): float(
                            predictions[mode][offset].item()
                        )
                        for mode in modes
                    },
                })

    metrics = {}
    for mode, values in collected.items():
        prediction = torch.cat(values["prediction"], dim=0)
        labels = torch.cat(values["label"], dim=0)
        row = regression_metrics(prediction, labels)
        row["Loss"] = float(np.mean(values["loss"]))
        metrics[mode] = row
    frame = pd.DataFrame(rows).sort_values(
        "sample_index", kind="mergesort"
    )
    return metrics, frame


def evaluate_frozen_test(cli, args, candidate, baseline, logger):
    test_loader = build_single_split_loader(args, "test", cli.num_workers)
    valid_loaders = MMDataLoader(args, cli.num_workers)
    teacher, student, _ = load_assets(cli, args, valid_loaders)
    del teacher
    student.load_state_dict(
        torch.load(candidate["MainCheckpoint"], map_location=args.device),
        strict=True,
    )
    criterion = nn.L1Loss()
    test_metrics, predictions = evaluate_and_predictions(
        student, test_loader, args.device, criterion
    )
    result = dict(candidate)
    result["J_test_at_valid_best"] = float(validation_objective(test_metrics))
    result.update(_flatten(test_metrics, "test_at_valid_best"))
    result["TestConstructed"] = True
    result["TestLoaderConstructionCount"] = 1
    result["TestLoaderTraversalCount"] = 1
    gate = candidate_test_gate(result, baseline)
    logger.info(
        "frozen_test J=%.6f gain=%+.6f passed=%s",
        result["J_test_at_valid_best"],
        gate["gain_test_J"],
        gate["passed"],
    )
    return result, gate, predictions


def previous_seed_passed(cli, seed):
    path = (
        Path(cli.result_root)
        / "missing_baseline"
        / OUTPUT_TAG
        / cli.dataset
        / "seed{}".format(seed)
        / "video_aware_sampling_summary.json"
    )
    if not path.is_file():
        return False, path
    payload = json.loads(path.read_text(encoding="utf-8"))
    return bool(
        payload.get("valid_gate", {}).get("passed")
        and payload.get("test_gate", {}).get("passed")
    ), path


def render_report(summary):
    baseline = summary["baseline"]
    candidate = summary["candidate"]
    valid_gate = summary["valid_gate"]
    lines = [
        "# CFCompatKD + Video-Aware Batch Sampling v1",
        "",
        "## Decision",
        "",
        "- Verdict: `{}`".format(summary["verdict"]),
        "- Seed: `{}`".format(summary["seed"]),
        "- Samples per video: `4`",
        "- New loss: `none`",
        "- Additional inference parameters: `0`",
        "- Baseline replay passed: `{}`".format(
            summary["baseline_replay_gate"]["passed"]
        ),
        "- Validation gate passed: `{}`".format(valid_gate["passed"]),
        "- Test constructed: `{}`".format(
            summary["protocol"]["test_constructed"]
        ),
        "",
        "## Validation",
        "",
        "| Metric | CFCompatKD | Video-aware sampling | Gain |",
        "| --- | ---: | ---: | ---: |",
        "| Valid J | {:.6f} | {:.6f} | {:+.6f} |".format(
            baseline["J_valid"],
            candidate["J_valid"],
            valid_gate["gain_valid_J"],
        ),
        "| Valid LAV MAE | {:.6f} | {:.6f} | {:+.6f} |".format(
            baseline["valid_LAV_MAE"],
            candidate["valid_LAV_MAE"],
            valid_gate["gain_valid_LAV_MAE"],
        ),
        "| Valid MissingMacro gain |  |  | {:+.6f} |".format(
            valid_gate["gain_valid_MissingMacro_MAE"]
        ),
        "",
    ]
    if summary.get("test_gate") is not None:
        lines.extend([
            "## Frozen test",
            "",
            "- Test J gain: `{:+.6f}`".format(
                summary["test_gate"]["gain_test_J"]
            ),
            "- Test gate passed: `{}`".format(
                summary["test_gate"]["passed"]
            ),
            "",
        ])
    return "\n".join(lines)


def main():
    cli = parse_args()
    logger, log_path = create_logger(cli)
    output_root, model_root = result_paths(cli)

    domain_audit_path = output_root / "domain_audit" / "video_domain_audit.json"
    if not domain_audit_path.is_file():
        raise FileNotFoundError(
            "Run audit_video_aware_sampling_domains.py first: {}".format(
                domain_audit_path
            )
        )
    domain_audit = json.loads(
        domain_audit_path.read_text(encoding="utf-8")
    )
    if not domain_audit.get("sampler_viable", False):
        raise RuntimeError("Video-domain audit did not approve the sampler.")

    baseline_series, baseline_path = load_stage3_reference(
        cli.result_root, cli.seed
    )
    baseline = series_to_dict(baseline_series)

    replay, replay_epochs, replay_predictions, _ = train_trajectory(
        cli, logger, output_root, model_root, video_aware=False
    )
    replay_check = replay_gate(replay, baseline)
    if not replay_check["passed"]:
        raise RuntimeError(
            "Stage3 replay failed; sampler screen is invalid: {}".format(
                replay_check
            )
        )

    candidate, candidate_epochs, candidate_valid_predictions, candidate_args = (
        train_trajectory(
            cli, logger, output_root, model_root, video_aware=True
        )
    )
    discovery_gate = discovery_replay_gate(cli, candidate)
    if discovery_gate.get("available") and not discovery_gate.get("passed"):
        raise RuntimeError(
            "Seed1111 sampler did not reproduce the discovery run: {}".format(
                discovery_gate
            )
        )

    valid_gate = candidate_valid_gate(candidate, baseline, candidate_epochs)
    test_gate = None
    test_predictions = None
    final_candidate = dict(candidate)

    if valid_gate["passed"]:
        final_candidate, test_gate, test_predictions = evaluate_frozen_test(
            cli, candidate_args, candidate, baseline, logger
        )
        if test_gate["passed"]:
            if cli.seed == 1114:
                verdict = "PROMOTE_SEED1114_RUN_SEED1111_FORMAL"
            else:
                seed1114_passed, _ = previous_seed_passed(cli, 1114)
                verdict = (
                    "PROMOTE_MOSI_TWO_SEEDS_RUN_MOSEI"
                    if seed1114_passed
                    else "PROMOTE_SEED1111_WAIT_FOR_SEED1114"
                )
        else:
            verdict = "STOP_VIDEO_AWARE_SAMPLING_TEST_GATE_FAILED"
    else:
        verdict = "STOP_VIDEO_AWARE_SAMPLING_VALID_GATE_FAILED"

    pd.DataFrame(replay_epochs + candidate_epochs).to_csv(
        output_root / "video_aware_sampling_all_epoch_metrics.csv",
        index=False,
    )
    pd.concat(
        [replay_predictions, candidate_valid_predictions], ignore_index=True
    ).to_csv(
        output_root / "video_aware_sampling_all_valid_predictions.csv",
        index=False,
    )
    if test_predictions is not None:
        test_predictions["Seed"] = cli.seed
        test_predictions["Run"] = candidate["Run"]
        test_predictions["Split"] = "test"
        test_predictions["SelectedBy"] = "validation_gate_then_frozen_test"
        test_predictions.to_csv(
            output_root / "video_aware_sampling_test_predictions.csv",
            index=False,
        )

    comparison = []
    for metric in [
        "J_valid",
        "valid_LAV_MAE",
        "valid_LA_MAE",
        "valid_LV_MAE",
        "valid_L_MAE",
    ]:
        comparison.append({
            "metric": metric,
            "baseline": float(baseline[metric]),
            "candidate": float(final_candidate[metric]),
            "gain_positive_is_better": (
                float(baseline[metric]) - float(final_candidate[metric])
            ),
            "decision_metric": True,
        })
    if test_gate is not None:
        for metric in [
            "J_test_at_valid_best",
            "test_at_valid_best_LAV_MAE",
            "test_at_valid_best_LA_MAE",
            "test_at_valid_best_LV_MAE",
            "test_at_valid_best_L_MAE",
        ]:
            comparison.append({
                "metric": metric,
                "baseline": float(baseline[metric]),
                "candidate": float(final_candidate[metric]),
                "gain_positive_is_better": (
                    float(baseline[metric]) - float(final_candidate[metric])
                ),
                "decision_metric": False,
            })
    pd.DataFrame(comparison).to_csv(
        output_root / "video_aware_sampling_comparison.csv", index=False
    )

    source_manifest = {
        "version": VERSION,
        "method": METHOD,
        "seed": int(cli.seed),
        "baseline_result": str(baseline_path),
        "baseline_result_sha256": checkpoint_sha256(baseline_path),
        "domain_audit": str(domain_audit_path),
        "domain_audit_sha256": checkpoint_sha256(domain_audit_path),
        "candidate_checkpoint": final_candidate["MainCheckpoint"],
        "candidate_checkpoint_sha256": final_candidate["MainCheckpointSHA256"],
        "teacher_checkpoint": final_candidate["TeacherCheckpoint"],
        "teacher_sha256": final_candidate["TeacherSHA256"],
        "evaluator_checkpoint": final_candidate["EvaluatorCheckpoint"],
        "evaluator_sha256": final_candidate["EvaluatorSHA256"],
        "compatibility_cache": final_candidate["CompatibilityCache"],
        "compatibility_cache_sha256": final_candidate[
            "CompatibilityCacheSHA256"
        ],
        "test_constructed_after_valid_gate": bool(valid_gate["passed"]),
        "test_loader_traversal_count": 1 if test_gate is not None else 0,
        "additional_inference_parameters": 0,
    }
    (output_root / "video_aware_sampling_source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "version": VERSION,
        "method": METHOD,
        "seed": int(cli.seed),
        "verdict": verdict,
        "baseline": baseline,
        "baseline_replay": replay,
        "baseline_replay_gate": replay_check,
        "candidate": final_candidate,
        "discovery_replay_gate": discovery_gate,
        "valid_gate": valid_gate,
        "test_gate": test_gate,
        "protocol": {
            "decision_split": "valid_then_one_frozen_test",
            "video_domain": "source_video_id",
            "samples_per_video": SAMPLES_PER_VIDEO,
            "exact_epoch_coverage": True,
            "oversampling": False,
            "replacement": False,
            "new_loss": False,
            "original_cfcompat_objective_unchanged": True,
            "test_constructed": bool(valid_gate["passed"]),
            "test_loader_traversal_count": 1 if test_gate is not None else 0,
            "additional_inference_parameters": 0,
        },
    }
    (output_root / "video_aware_sampling_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "video_aware_sampling_report.md").write_text(
        render_report(summary) + "\n", encoding="utf-8"
    )

    logger.info(
        "complete verdict=%s valid_gain=%+.6f result=%s log=%s",
        verdict,
        valid_gate["gain_valid_J"],
        output_root,
        log_path,
    )
    print("CFCompat video-aware sampling complete")
    print("seed:", cli.seed)
    print("verdict:", verdict)
    print("valid J gain:", "{:+.6f}".format(valid_gate["gain_valid_J"]))
    if test_gate is not None:
        print("test J gain:", "{:+.6f}".format(test_gate["gain_test_J"]))
    else:
        print("test was not constructed because the Valid gate failed")
    print("report:", output_root / "video_aware_sampling_report.md")


if __name__ == "__main__":
    main()
