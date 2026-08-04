"""Train CFCompatKD with source-video V-REx under a locked Valid-first screen."""

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
    sample_missing_masks,
    validation_objective,
)
from trains.singleTask.video_vrex_utils import (
    METHOD,
    OUTPUT_TAG,
    REPLAY_TOLERANCE,
    SAMPLER_CONTROL_LAMBDA,
    SAMPLES_PER_VIDEO,
    VERSION,
    VREX_LAMBDAS,
    aggregate_vrex_diagnostics,
    build_video_aware_train_loader,
    candidate_test_gate,
    candidate_valid_gate,
    load_locked_counterfactual_cache,
    portable_locate_stage1_evaluator,
    video_ids_from_batch,
    video_risk_variance,
)
from utils.functions import setup_seed


FORMAL_SEEDS = (1111, 1114)


def parse_args():
    parser = argparse.ArgumentParser(
        description="CFCompatKD source-video V-REx validation screen."
    )
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seed", type=int, choices=FORMAL_SEEDS, default=1111)
    parser.add_argument("--fixed-lambda", type=float)
    parser.add_argument("--samples-per-video", type=int, default=SAMPLES_PER_VIDEO)
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
        parser.error("Formal Video-VREx fixes num_workers=1.")
    if args.samples_per_video != SAMPLES_PER_VIDEO:
        parser.error("Video-VREx fixes --samples-per-video at 4.")
    if args.seed == 1111 and args.fixed_lambda is not None:
        parser.error("Seed1111 performs the frozen three-point lambda screen.")
    if args.seed == 1114:
        if args.fixed_lambda is None:
            parser.error("Seed1114 requires the lambda selected by seed1111.")
        if not any(math.isclose(args.fixed_lambda, value, abs_tol=0.0) for value in VREX_LAMBDAS):
            parser.error("Seed1114 lambda must be one of 0.01, 0.1, 1.0.")
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
    path = directory / "DLF-{}-video-vrex-seed{}-{}-{}.log".format(
        cli.dataset,
        cli.seed,
        kind,
        datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    logger = logging.getLogger("cfcompat_video_vrex")
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


def lambda_tag(value: float, original_replay: bool = False) -> str:
    if original_replay:
        return "stage3_replay"
    if math.isclose(value, 0.0, abs_tol=0.0):
        return "video_sampler_control_lambda0"
    return "lambda_{}".format(str(value).replace(".", "p"))


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
    cache_frame, cache_by_index = load_locked_counterfactual_cache(
        cli.result_root,
        cli.dataset,
        version=cache_version,
        seed=cli.seed if multiseed else None,
        expected_evaluator_sha=evaluator_sha,
    )
    if cli.dataset == "mosi" and len(cache_frame) != 1284:
        raise RuntimeError("MOSI Video-VREx requires the locked 1284-sample cache.")
    teacher, student, teacher_checkpoint, teacher_sha = initialize_teacher_student(
        args, cli, cli.seed, loaders
    )
    assets = {
        "teacher_checkpoint": teacher_checkpoint,
        "teacher_sha": teacher_sha,
        "evaluator_checkpoint": evaluator_checkpoint,
        "evaluator_sha": evaluator_sha,
        "evaluator_best_epoch": evaluator_best_epoch,
        "evaluator_source": evaluator_source,
        "cache_frame": cache_frame,
        "cache_by_index": cache_by_index,
        "cache_version": cache_version,
    }
    return teacher, student, assets


def train_trajectory(
    cli,
    logger,
    output_root: Path,
    model_root: Path,
    lambda_vrex: float,
    video_aware: bool,
    original_replay: bool = False,
):
    setup_seed(cli.seed)
    args = build_config(cli, cli.seed)
    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("Training may construct only train and valid loaders.")

    sampler = None
    if video_aware:
        train_loader, sampler = build_video_aware_train_loader(
            args,
            cli.num_workers,
            cli.seed,
            samples_per_video=cli.samples_per_video,
        )
        loaders["train"] = train_loader

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

    tag = lambda_tag(lambda_vrex, original_replay=original_replay)
    run_dir = output_root / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_root / tag / "best_valid.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    best_j = float("inf")
    best_epoch = 0
    epoch_rows = []
    best_valid_metrics = None
    batch_sizes = None
    last_epoch = 0

    logger.info(
        "run=%s seed=%s video_aware=%s lambda_vrex=%.6g test=locked",
        tag,
        cli.seed,
        video_aware,
        lambda_vrex,
    )

    for epoch in range(1, (cli.max_epochs or 1000) + 1):
        last_epoch = epoch
        student.train()
        optimizer.zero_grad()
        counts = Counter({"LA": 0, "LV": 0, "L": 0})
        kd_losses = []
        vrex_rows = []
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

            videos = video_ids_from_batch(batch["id"])
            vrex = video_risk_variance(
                full_output["output_logit"],
                missing_output["output_logit"],
                labels,
                videos,
            )
            vrex_rows.append(vrex.diagnostics)

            if video_aware and step == 1:
                if vrex.diagnostics["eligible_video_count"] < 2:
                    raise RuntimeError("Video-aware first batch cannot estimate V-REx.")
                if lambda_vrex > 0:
                    gradients = torch.autograd.grad(
                        vrex.penalty,
                        (
                            student.backbone.proj1.weight,
                            student.backbone.out_layer.weight,
                        ),
                        retain_graph=True,
                        allow_unused=False,
                    )
                    if any(
                        not torch.isfinite(gradient).all()
                        or float(gradient.detach().pow(2).sum()) <= 0
                        for gradient in gradients
                    ):
                        raise RuntimeError("Video-VREx gradient did not reach DLF.")

            total_loss = (
                full_loss
                + missing_loss
                + kd_loss
                + float(lambda_vrex) * vrex.penalty
            )
            if not torch.isfinite(total_loss):
                raise FloatingPointError("NaN/Inf in Video-VREx trajectory.")
            total_loss.backward()
            if teacher_grad_count(teacher):
                raise RuntimeError("Frozen Teacher received gradients.")
            if step % args.update_epochs == 0 or step == len(loaders["train"]):
                optimizer.step()
                optimizer.zero_grad()

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

        diagnostics = aggregate_vrex_diagnostics(vrex_rows)
        row = {
            "Seed": cli.seed,
            "Run": tag,
            "Epoch": epoch,
            "VideoAwareSampler": bool(video_aware),
            "LambdaVREx": float(lambda_vrex),
            "J_valid": float(j_valid),
            "IsBestValid": bool(is_best),
            "KD_loss": float(np.mean(kd_losses)),
            **diagnostics,
            **_flatten(valid, "valid"),
        }
        epoch_rows.append(row)
        logger.info(
            "run=%s epoch=%s J_valid=%.6f LAV=%.6f MissingMacro=%.6f "
            "vrex=%.6f eligible_videos=%.3f eligible_fraction=%.3f",
            tag,
            epoch,
            j_valid,
            valid["LAV"]["MAE"],
            np.mean([valid[mode]["MAE"] for mode in MISSING_MODES]),
            diagnostics["vrex_penalty"],
            diagnostics["eligible_video_count"],
            diagnostics["eligible_sample_fraction"],
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
    valid_predictions["LambdaVREx"] = float(lambda_vrex)
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
        "Method": "DLF-CFCompatKD-v1" if original_replay else METHOD,
        "VideoAwareSampler": bool(video_aware),
        "LambdaVREx": float(lambda_vrex),
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


def select_lambda(candidate_results):
    positive = [
        result
        for result in candidate_results
        if any(
            math.isclose(result["LambdaVREx"], value, abs_tol=0.0)
            for value in VREX_LAMBDAS
        )
    ]
    if len(positive) != len(VREX_LAMBDAS):
        raise RuntimeError("Positive lambda grid is incomplete.")
    return min(
        positive,
        key=lambda result: (float(result["J_valid"]), float(result["LambdaVREx"])),
    )


def evaluate_selected_test(cli, args, selected, baseline, logger):
    test_loader = build_single_split_loader(args, "test", cli.num_workers)
    valid_loaders = MMDataLoader(args, cli.num_workers)
    teacher, student, _ = load_assets(cli, args, valid_loaders)
    del teacher
    student.load_state_dict(
        torch.load(selected["MainCheckpoint"], map_location=args.device),
        strict=True,
    )
    criterion = nn.L1Loss()
    test_metrics = evaluate_all_modes(
        student, test_loader, args.device, "moddrop", criterion
    )
    test_predictions = prediction_rows(student, test_loader, args.device)
    candidate = dict(selected)
    candidate["J_test_at_valid_best"] = float(validation_objective(test_metrics))
    candidate.update(_flatten(test_metrics, "test_at_valid_best"))
    candidate["TestConstructed"] = True
    candidate["TestLoaderConstructionCount"] = 1
    candidate["TestLoaderTraversalCount"] = 1
    gate = candidate_test_gate(candidate, baseline)
    logger.info(
        "selected lambda=%.6g frozen_test_J=%.6f gain=%+.6f passed=%s",
        candidate["LambdaVREx"],
        candidate["J_test_at_valid_best"],
        gate["gain_test_J"],
        gate["passed"],
    )
    return candidate, gate, test_predictions


def render_report(summary):
    baseline = summary["baseline"]
    selected = summary["selected_candidate"]
    valid_gate = summary["valid_gate"]
    lines = [
        "# CFCompatKD + Video-VREx v1",
        "",
        "## Decision",
        "",
        "- Verdict: `{}`".format(summary["verdict"]),
        "- Selected lambda: `{}`".format(selected["LambdaVREx"]),
        "- Baseline replay passed: `{}`".format(summary["baseline_replay_gate"]["passed"]),
        "- Validation gate passed: `{}`".format(valid_gate["passed"]),
        "- Test constructed: `{}`".format(summary["protocol"]["test_constructed"]),
        "",
        "## Validation",
        "",
        "| Metric | CFCompatKD | Video-VREx | Gain |",
        "| --- | ---: | ---: | ---: |",
        "| Valid J | {:.6f} | {:.6f} | {:+.6f} |".format(
            baseline["J_valid"], selected["J_valid"], valid_gate["gain_valid_J"]
        ),
        "| Valid LAV MAE | {:.6f} | {:.6f} | {:+.6f} |".format(
            baseline["valid_LAV_MAE"],
            selected["valid_LAV_MAE"],
            valid_gate["gain_valid_LAV_MAE"],
        ),
        "| Valid MissingMacro gain |  |  | {:+.6f} |".format(
            valid_gate["gain_valid_MissingMacro_MAE"]
        ),
        "",
    ]
    if summary.get("test_gate") is not None:
        lines.extend(
            [
                "## Frozen test",
                "",
                "- Test J gain: `{:+.6f}`".format(
                    summary["test_gate"]["gain_test_J"]
                ),
                "- Test gate passed: `{}`".format(summary["test_gate"]["passed"]),
                "",
            ]
        )
    return "\n".join(lines)


def main():
    cli = parse_args()
    logger, log_path = create_logger(cli)
    output_root, model_root = result_paths(cli)

    audit_path = (
        Path(cli.result_root)
        / "missing_baseline"
        / OUTPUT_TAG
        / cli.dataset
        / "domain_audit"
        / "video_domain_audit.json"
    )
    if not audit_path.is_file():
        raise FileNotFoundError(
            "Run audit_video_domains.py before training: {}".format(audit_path)
        )
    domain_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not domain_audit.get("vrex_batch_viable", False):
        raise RuntimeError("Video-domain audit did not approve V-REx batches.")

    baseline_series, baseline_path = load_stage3_reference(
        cli.result_root, cli.seed
    )
    baseline = series_to_dict(baseline_series)

    replay, replay_epochs, _, _ = train_trajectory(
        cli,
        logger,
        output_root,
        model_root,
        lambda_vrex=0.0,
        video_aware=False,
        original_replay=True,
    )
    replay_check = replay_gate(replay, baseline)
    if not replay_check["passed"]:
        raise RuntimeError(
            "Stage3 replay failed; Video-VREx screen is invalid: {}".format(
                replay_check
            )
        )

    results = []
    epoch_rows = list(replay_epochs)
    valid_predictions = []

    control, control_epochs, control_predictions, _ = train_trajectory(
        cli,
        logger,
        output_root,
        model_root,
        lambda_vrex=SAMPLER_CONTROL_LAMBDA,
        video_aware=True,
    )
    results.append(control)
    epoch_rows.extend(control_epochs)
    valid_predictions.append(control_predictions)

    lambda_values = (
        (float(cli.fixed_lambda),)
        if cli.seed == 1114
        else VREX_LAMBDAS
    )
    last_args = None
    for value in lambda_values:
        result, local_epochs, local_predictions, last_args = train_trajectory(
            cli,
            logger,
            output_root,
            model_root,
            lambda_vrex=float(value),
            video_aware=True,
        )
        results.append(result)
        epoch_rows.extend(local_epochs)
        valid_predictions.append(local_predictions)

    selected = (
        results[-1]
        if cli.seed == 1114
        else select_lambda(results)
    )
    selected_epochs = [
        row for row in epoch_rows if row["Run"] == selected["Run"]
    ]
    valid_gate = candidate_valid_gate(selected, baseline, selected_epochs)

    test_gate = None
    test_predictions = None
    selected_candidate = dict(selected)
    if valid_gate["passed"]:
        selected_candidate, test_gate, test_predictions = evaluate_selected_test(
            cli, last_args, selected, baseline, logger
        )
        verdict = (
            "PROMOTE_SEED1111_RUN_SEED1114"
            if test_gate["passed"] and cli.seed == 1111
            else (
                "PROMOTE_TWO_SEEDS_RUN_MOSEI"
                if test_gate["passed"] and cli.seed == 1114
                else "STOP_VIDEO_VREX_TEST_GATE_FAILED"
            )
        )
    else:
        verdict = "STOP_VIDEO_VREX_VALID_GATE_FAILED"

    grid = pd.DataFrame(results).sort_values(
        ["LambdaVREx"], kind="mergesort"
    )
    grid["EligibleForSelection"] = grid.LambdaVREx.isin(VREX_LAMBDAS)
    grid["SelectedByValidJ"] = grid.Run.eq(selected["Run"])
    grid.to_csv(output_root / "video_vrex_grid_summary.csv", index=False)
    pd.DataFrame(epoch_rows).to_csv(
        output_root / "video_vrex_all_epoch_metrics.csv", index=False
    )
    pd.concat(valid_predictions, ignore_index=True).to_csv(
        output_root / "video_vrex_all_valid_predictions.csv", index=False
    )
    if test_predictions is not None:
        test_predictions["Seed"] = cli.seed
        test_predictions["Run"] = selected["Run"]
        test_predictions["LambdaVREx"] = selected["LambdaVREx"]
        test_predictions["Split"] = "test"
        test_predictions["SelectedBy"] = "validation_J_and_valid_gate"
        test_predictions.to_csv(
            output_root / "video_vrex_selected_test_predictions.csv", index=False
        )

    source_manifest = {
        "version": VERSION,
        "method": METHOD,
        "seed": int(cli.seed),
        "baseline_result": str(baseline_path),
        "baseline_result_sha256": checkpoint_sha256(baseline_path),
        "domain_audit": str(audit_path),
        "domain_audit_sha256": checkpoint_sha256(audit_path),
        "selected_checkpoint": selected_candidate["MainCheckpoint"],
        "selected_checkpoint_sha256": selected_candidate["MainCheckpointSHA256"],
        "teacher_checkpoint": selected_candidate["TeacherCheckpoint"],
        "teacher_sha256": selected_candidate["TeacherSHA256"],
        "evaluator_checkpoint": selected_candidate["EvaluatorCheckpoint"],
        "evaluator_sha256": selected_candidate["EvaluatorSHA256"],
        "test_constructed_after_valid_gate": bool(valid_gate["passed"]),
        "additional_inference_parameters": 0,
    }
    (output_root / "video_vrex_source_manifest.json").write_text(
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
        "sampler_control": control,
        "selected_candidate": selected_candidate,
        "valid_gate": valid_gate,
        "test_gate": test_gate,
        "lambda_grid": list(lambda_values),
        "protocol": {
            "decision_split": "valid_then_one_frozen_test",
            "video_domain": "source_video_id",
            "samples_per_video": SAMPLES_PER_VIDEO,
            "sampler_control_is_not_selectable": True,
            "positive_lambda_grid": list(VREX_LAMBDAS),
            "test_constructed": bool(valid_gate["passed"]),
            "test_loader_traversal_count": 1 if test_gate is not None else 0,
            "original_cfcompat_objective_unchanged": True,
            "additional_inference_parameters": 0,
        },
    }
    (output_root / "video_vrex_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "video_vrex_report.md").write_text(
        render_report(summary) + "\n", encoding="utf-8"
    )

    logger.info(
        "complete verdict=%s selected_lambda=%.6g valid_gain=%+.6f result=%s log=%s",
        verdict,
        selected["LambdaVREx"],
        valid_gate["gain_valid_J"],
        output_root,
        log_path,
    )
    print("CFCompat Video-VREx complete")
    print("seed:", cli.seed)
    print("selected lambda:", selected["LambdaVREx"])
    print("verdict:", verdict)
    print("valid J gain:", "{:+.6f}".format(valid_gate["gain_valid_J"]))
    if test_gate is not None:
        print("test J gain:", "{:+.6f}".format(test_gate["gain_test_J"]))
    else:
        print("test was not constructed because the Valid gate failed")
    print("report:", output_root / "video_vrex_report.md")


if __name__ == "__main__":
    main()
