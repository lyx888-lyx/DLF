"Train CFCompatKD with mode-consistent C-Mixup at the DLF fusion feature."

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
    _diagnostic_rows,
    _flatten,
    batch_to_device,
    build_config,
    initialize_teacher_student,
)
from trains.singleTask.HingeLoss import HingeLoss
from trains.singleTask.cf_compat_kd_utils import (
    CACHE_VERSION,
    MULTISEED_CACHE_VERSION,
    cache_paths,
    compatibility_for_modes,
    gate_weights,
    gated_kd_loss,
    load_counterfactual_cache,
    locate_stage1_evaluator,
    modes_from_masks,
)
from trains.singleTask.cfcompat_stability_utils import (
    MissingSequenceHasher,
    expected_missing_sequence_sha,
    load_stage3_reference,
)
from trains.singleTask.cmixup_regression_utils import (
    DEFAULT_ALPHA,
    DEFAULT_BANDWIDTH,
    DEFAULT_MIX_WEIGHT,
    VERSION,
    FusionInputCapture,
    ModeConsistentCMixupSampler,
    aggregate_mix_diagnostics,
    compute_mode_consistent_cmixup_loss,
    promotion_gate,
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
    mode_to_mask,
    regression_metrics,
    sample_missing_masks,
    validation_objective,
)
from utils.functions import setup_seed


METHOD = "DLF-CFCompatKD-ModeConsistentCMixup-v1"
FORMAL_SEEDS = (1111, 1114)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mode-consistent fusion-level C-Mixup for CFCompatKD."
    )
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seed", type=int, choices=FORMAL_SEEDS, default=1111)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--bandwidth", type=float, default=DEFAULT_BANDWIDTH)
    parser.add_argument("--mix-weight", type=float, default=DEFAULT_MIX_WEIGHT)
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
        parser.error("Formal C-Mixup fixes num_workers=1 to match Stage 3.")
    if not math.isclose(args.alpha, DEFAULT_ALPHA, abs_tol=0.0):
        parser.error("Formal C-Mixup fixes --alpha at 2.0.")
    if not math.isclose(args.bandwidth, DEFAULT_BANDWIDTH, abs_tol=0.0):
        parser.error("Formal C-Mixup fixes --bandwidth at 0.5.")
    if not math.isclose(args.mix_weight, DEFAULT_MIX_WEIGHT, abs_tol=0.0):
        parser.error("Formal C-Mixup fixes --mix-weight at 1.0.")
    if args.smoke_test:
        args.max_epochs = 2 if args.max_epochs is None else min(2, args.max_epochs)
    elif args.max_epochs is not None:
        parser.error("Formal runs use the frozen DLF early-stop epoch rule.")
    return args


def paths(cli):
    result = (
        Path(cli.result_root)
        / "missing_baseline"
        / "cfcompat_cmixup_v1"
        / cli.dataset
        / "seed{}".format(cli.seed)
    )
    model = (
        Path(cli.model_save_dir)
        / "missing_baseline"
        / "cfcompat_cmixup_v1"
        / cli.dataset
        / "seed{}".format(cli.seed)
    )
    if cli.smoke_test:
        result, model = result / "smoke", model / "smoke"
    result.mkdir(parents=True, exist_ok=True)
    model.mkdir(parents=True, exist_ok=True)
    return result, model / "best_valid.pth"


def create_logger(cli):
    directory = Path(cli.log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    kind = "smoke" if cli.smoke_test else "formal"
    path = directory / "DLF-mosi-cfcompat-cmixup-seed{}-{}-{}.log".format(
        cli.seed, kind, datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    logger = logging.getLogger("cfcompat_cmixup")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def _json_value(value):
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
    return {str(key): _json_value(value) for key, value in series.items()}


def evaluate_and_predictions(model, loader, device, criterion):
    """Evaluate all modes and retain predictions in exactly one loader pass."""
    model.eval()
    collected = {
        mode: {"prediction": [], "label": [], "loss": []}
        for mode in ("LAV",) + MISSING_MODES
    }
    rows = []
    with torch.no_grad():
        for batch in loader:
            text, audio, vision, labels = batch_to_device(batch, device)
            predictions = {}
            for mode in ("LAV",) + MISSING_MODES:
                mask = mode_to_mask(mode, labels.size(0), device, audio.dtype)
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
                rows.append(
                    {
                        "sample_id": str(sample_ids[offset]),
                        "sample_index": int(index),
                        "label": float(labels[offset].item()),
                        **{
                            "{}_pred".format(mode): float(
                                predictions[mode][offset].item()
                            )
                            for mode in predictions
                        },
                    }
                )
    metrics = {}
    for mode, values in collected.items():
        prediction = torch.cat(values["prediction"], dim=0)
        labels = torch.cat(values["label"], dim=0)
        row = regression_metrics(prediction, labels)
        row["Loss"] = float(np.mean(values["loss"]))
        metrics[mode] = row
    predictions = pd.DataFrame(rows).sort_values(
        "sample_index", kind="mergesort"
    )
    return metrics, predictions


def flatten(metrics, prefix):
    result = {}
    for mode, values in metrics.items():
        for key, value in values.items():
            result["{}_{}_{}".format(prefix, mode, key)] = float(value)
    return result


def comparison_rows(candidate, baseline):
    metrics = ["J_valid", "valid_LAV_MAE"]
    metrics.extend("valid_{}_MAE".format(mode) for mode in MISSING_MODES)
    metrics.extend(
        "test_at_valid_best_{}_MAE".format(mode)
        for mode in ("LAV",) + MISSING_MODES
    )
    rows = []
    for metric in metrics:
        if metric not in candidate or metric not in baseline:
            continue
        base = float(baseline[metric])
        value = float(candidate[metric])
        rows.append(
            {
                "metric": metric,
                "baseline": base,
                "candidate": value,
                "gain_positive_is_better": base - value,
                "decision_metric": metric.startswith("valid_")
                or metric == "J_valid",
            }
        )
    return rows


def render_report(summary):
    baseline = summary["baseline"]
    candidate = summary["candidate"]
    gate = summary["promotion_gate"]
    lines = [
        "# CFCompatKD + Mode-Consistent C-Mixup v1",
        "",
        "## Decision",
        "",
        "- Verdict: `{}`".format(summary["verdict"]),
        "- Decision split: Valid only",
        "- Test loader constructed after validation checkpoint selection: yes",
        "",
        "## Core metrics",
        "",
        "| Metric | CFCompatKD | C-Mixup | Gain |",
        "| --- | ---: | ---: | ---: |",
        "| Valid J | {:.6f} | {:.6f} | {:+.6f} |".format(
            float(baseline["J_valid"]),
            float(candidate["J_valid"]),
            float(gate["gain_valid_J"]),
        ),
        "| Valid LAV MAE | {:.6f} | {:.6f} | {:+.6f} |".format(
            float(baseline["valid_LAV_MAE"]),
            float(candidate["valid_LAV_MAE"]),
            float(gate["gain_valid_LAV_MAE"]),
        ),
        "| Test LAV MAE at Valid-best | {:.6f} | {:.6f} | {:+.6f} |".format(
            float(baseline["test_at_valid_best_LAV_MAE"]),
            float(candidate["test_at_valid_best_LAV_MAE"]),
            float(baseline["test_at_valid_best_LAV_MAE"])
            - float(candidate["test_at_valid_best_LAV_MAE"]),
        ),
        "",
        "## Frozen gate",
        "",
    ]
    for key, value in gate["checks"].items():
        lines.append("- {}: **{}**".format(key, "PASS" if value else "FAIL"))
    lines.extend(
        [
            "",
            "## Method boundary",
            "",
            "- C-Mixup is applied only to the differentiable input of the original DLF `proj1`.",
            "- Full and sampled-missing views reuse the same partner and interpolation coefficient.",
            "- Partners are selected only within the same LA/LV/L missing mode.",
            "- Original samples retain the unchanged full task, missing task, and CFCompatKD losses.",
            "- Mixed samples receive regression loss only; no synthetic compatibility or Teacher target is created.",
            "- No inference parameter, module, Teacher, router, expert, or second-stage prediction is added.",
            "",
        ]
    )
    return "\n".join(lines)


def train_one_seed(cli, logger):
    setup_seed(cli.seed)
    args = build_config(cli, cli.seed)
    multiseed = int(cli.seed) != 1111
    evaluator_checkpoint, evaluator_best_epoch, evaluator_source = (
        locate_stage1_evaluator(
            cli.result_root,
            cli.dataset,
            cli.seed,
            multiseed=multiseed,
            smoke=False,
        )
    )
    evaluator_sha = checkpoint_sha256(evaluator_checkpoint)
    cache_version = MULTISEED_CACHE_VERSION if multiseed else CACHE_VERSION
    cache_frame, cache_by_index = load_counterfactual_cache(
        cli.result_root,
        cli.dataset,
        version=cache_version,
        seed=cli.seed if multiseed else None,
        expected_evaluator_sha=evaluator_sha,
    )
    if len(cache_frame) != 1284:
        raise RuntimeError("C-Mixup requires the audited 1284-sample cache.")
    cache_file = cache_paths(
        cli.result_root,
        cli.dataset,
        version=cache_version,
        seed=cli.seed if multiseed else None,
    )["csv"]
    cache_sha = checkpoint_sha256(cache_file)

    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("Training may construct only train and valid loaders.")
    teacher, student, teacher_checkpoint, teacher_sha = initialize_teacher_student(
        args, cli, cli.seed, loaders
    )
    optimizer = optim.Adam(student.parameters(), lr=args.learning_rate)
    assert_teacher_not_in_optimizer(teacher, optimizer)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience
    )
    criterion = nn.L1Loss()
    cosine = nn.CosineEmbeddingLoss()
    hinge = HingeLoss()
    missing_generator = torch.Generator().manual_seed(cli.seed + 104729)
    mix_sampler = ModeConsistentCMixupSampler(
        seed=cli.seed, alpha=cli.alpha, bandwidth=cli.bandwidth
    )
    missing_hasher = MissingSequenceHasher()
    batch_sizes = None

    result_dir, checkpoint = paths(cli)
    best_j = float("inf")
    best_epoch = 0
    epoch_rows = []
    best_valid_metrics = None

    baseline_series, baseline_path = load_stage3_reference(
        cli.result_root, cli.seed
    )
    baseline = series_to_dict(baseline_series)
    logger.info(
        "seed=%s method=%s baseline=%s baseline_J_valid=%.6f",
        cli.seed,
        METHOD,
        baseline_path,
        float(baseline["J_valid"]),
    )
    logger.info(
        "alpha=%.3f bandwidth=%.3f mix_weight=%.3f; test locked until selection",
        cli.alpha,
        cli.bandwidth,
        cli.mix_weight,
    )

    last_epoch = 0
    for epoch in range(1, (cli.max_epochs or 1000) + 1):
        last_epoch = epoch
        student.train()
        optimizer.zero_grad()
        counts = Counter({"LA": 0, "LV": 0, "L": 0})
        gate_records = []
        mix_rows = []
        kd_losses = []
        epoch_batch_sizes = []

        for step, batch in enumerate(loaders["train"], 1):
            text, audio, vision, labels = batch_to_device(batch, args.device)
            epoch_batch_sizes.append(int(labels.size(0)))

            full_mask = mode_to_mask(
                "LAV", labels.size(0), args.device, audio.dtype
            )
            with FusionInputCapture(student.backbone.proj1) as capture:
                full_output = student(text, audio, vision, full_mask)
                full_fusion = capture.pop()
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
                missing_fusion = capture.pop()
                missing_loss, _ = compute_task_loss(
                    missing_output, labels, criterion
                )
                capture.assert_empty()

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
            gate, reliability = gate_weights(
                compatibility, teacher_prediction, labels, "compat"
            )
            kd_loss, each_kd = gated_kd_loss(
                missing_output["output_logit"], teacher_prediction, gate
            )
            kd_losses.append(float(kd_loss.detach()))

            mix_loss, mix_diagnostics, _ = compute_mode_consistent_cmixup_loss(
                student=student,
                full_fusion=full_fusion,
                missing_fusion=missing_fusion,
                labels=labels,
                modes=modes,
                sampler=mix_sampler,
                criterion=criterion,
            )
            mix_rows.append(mix_diagnostics)

            if step == 1:
                mix_gradients = torch.autograd.grad(
                    mix_loss,
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
                    for gradient in mix_gradients
                ):
                    raise RuntimeError(
                        "C-Mixup did not reach the original DLF tail."
                    )

            total_loss = (
                full_loss
                + missing_loss
                + kd_loss
                + cli.mix_weight * mix_loss
            )
            if not torch.isfinite(total_loss):
                raise FloatingPointError("NaN/Inf in C-Mixup loss.")
            total_loss.backward()
            if teacher_grad_count(teacher):
                raise RuntimeError("Frozen Teacher received gradients.")
            if step % args.update_epochs == 0 or step == len(loaders["train"]):
                optimizer.step()
                optimizer.zero_grad()

            denominator = float(gate.sum().detach().cpu()) + 1e-8
            student_error = torch.abs(
                missing_output["output_logit"].detach().view(-1)
                - labels.view(-1)
            )
            teacher_error = torch.abs(
                teacher_prediction.detach().view(-1) - labels.view(-1)
            )
            teacher_student_gap = torch.abs(
                missing_output["output_logit"].detach().view(-1)
                - teacher_prediction.detach().view(-1)
            )
            for offset, index in enumerate(indices):
                mode = modes[offset]
                gate_records.append(
                    {
                        "sample_index": index,
                        "mode": mode,
                        "delta": cache_by_index[index]["delta_{}".format(mode)],
                        "compat": float(compatibility[offset]),
                        "reliability": float(reliability[offset]),
                        "gate": float(gate[offset]),
                        "kd": float(each_kd[offset].detach()),
                        "weighted_kd_contribution": float(
                            gate[offset].detach()
                            * each_kd[offset].detach()
                            / denominator
                        ),
                        "student_missing_abs_label_error": float(
                            student_error[offset]
                        ),
                        "teacher_error": float(teacher_error[offset]),
                        "teacher_student_abs_gap": float(
                            teacher_student_gap[offset]
                        ),
                    }
                )

        if batch_sizes is None:
            batch_sizes = epoch_batch_sizes
        elif batch_sizes != epoch_batch_sizes:
            raise RuntimeError("Train batch-size sequence changed across epochs.")
        if (
            epoch == 1
            and cli.seed == 1111
            and counts != Counter({"LA": 435, "LV": 430, "L": 419})
        ):
            raise RuntimeError(
                "Epoch1 missing counts must be LA=435 LV=430 L=419."
            )

        valid_metrics, _ = evaluate_and_predictions(
            student, loaders["valid"], args.device, criterion
        )
        j_valid = validation_objective(valid_metrics)
        if not math.isfinite(j_valid):
            raise FloatingPointError("Non-finite Valid J.")
        scheduler.step(j_valid)
        is_best = j_valid <= best_j - 1e-6
        if is_best:
            best_j = j_valid
            best_epoch = epoch
            best_valid_metrics = valid_metrics
            torch.save(student.state_dict(), checkpoint)

        gate_summary, _ = _diagnostic_rows(
            gate_records, cli.seed, epoch, "compat"
        )
        mix_summary = aggregate_mix_diagnostics(mix_rows)
        row = {
            "Seed": cli.seed,
            "Epoch": epoch,
            "Method": METHOD,
            "J_valid": j_valid,
            "IsBestValid": is_best,
            "KD_loss": float(np.mean(kd_losses)),
            "gate_mean": gate_summary["gate_mean"],
            "ESS_fraction": gate_summary["ESS_fraction"],
            **mix_summary,
            **flatten(valid_metrics, "valid"),
        }
        epoch_rows.append(row)
        logger.info(
            "epoch=%s J_valid=%.6f LAV=%.6f MissingMacro=%.6f "
            "mix=%.6f+%.6f active=%.4f label_dist=%.4f",
            epoch,
            j_valid,
            valid_metrics["LAV"]["MAE"],
            np.mean([valid_metrics[mode]["MAE"] for mode in MISSING_MODES]),
            row["mix_full_loss"],
            row["mix_missing_loss"],
            row["mix_active_fraction"],
            row["mix_mean_partner_label_distance"],
        )
        if epoch - best_epoch >= args.early_stop:
            break

    if not checkpoint.is_file() or best_valid_metrics is None:
        raise RuntimeError("Validation-best C-Mixup checkpoint is absent.")
    expected_sha, expected_count = expected_missing_sequence_sha(
        cli.seed, last_epoch, batch_sizes
    )
    actual_sha = missing_hasher.hexdigest()
    if actual_sha != expected_sha or missing_hasher.count != expected_count:
        raise RuntimeError(
            "Missing-mode sequence differs from frozen Stage 3 sampling."
        )

    student.load_state_dict(
        torch.load(checkpoint, map_location=args.device), strict=True
    )
    final_valid, valid_predictions = evaluate_and_predictions(
        student, loaders["valid"], args.device, criterion
    )

    # Test is created and traversed only after the validation-selected
    # checkpoint has been frozen.
    test_loader = build_single_split_loader(args, "test", cli.num_workers)
    final_test, test_predictions = evaluate_and_predictions(
        student, test_loader, args.device, criterion
    )

    candidate = {
        "Seed": cli.seed,
        "Method": METHOD,
        "BestValidEpoch": best_epoch,
        "J_valid": validation_objective(final_valid),
        "J_test_at_valid_best": validation_objective(final_test),
        "MainCheckpoint": str(checkpoint),
        "MainCheckpointSHA256": checkpoint_sha256(checkpoint),
        "TeacherCheckpoint": str(teacher_checkpoint),
        "TeacherSHA256": teacher_sha,
        "EvaluatorCheckpoint": str(evaluator_checkpoint),
        "EvaluatorSHA256": evaluator_sha,
        "EvaluatorBestEpoch": int(evaluator_best_epoch),
        "CompatibilityCache": str(cache_file),
        "CompatibilityCacheSHA256": cache_sha,
        "MissingSequenceSHA256": actual_sha,
        "CMixupPartnerSequenceSHA256": mix_sampler.hexdigest(),
        "CMixupDrawCount": int(mix_sampler.draw_count),
        "CMixupAlpha": cli.alpha,
        "CMixupBandwidth": cli.bandwidth,
        "CMixupWeight": cli.mix_weight,
        "TestLoaderConstructionCount": 1,
        "TestLoaderTraversalCount": 1,
        "SelectedBy": "validation_J",
        "StudentOnlyInference": True,
        "AdditionalInferenceParameters": 0,
        **flatten(final_valid, "valid"),
        **flatten(final_test, "test_at_valid_best"),
    }
    gate = promotion_gate(candidate, baseline, epoch_rows)
    verdict = (
        "PROMOTE_SEED1111_RUN_SEED1114"
        if gate["passed"] and cli.seed == 1111
        else (
            "PROMOTE_TWO_SEEDS_RUN_FIVE_SEEDS"
            if gate["passed"] and cli.seed == 1114
            else "STOP_CMIXUP_SEED_GATE_FAILED"
        )
    )

    valid_predictions["Seed"] = cli.seed
    valid_predictions["Method"] = METHOD
    valid_predictions["Split"] = "valid"
    valid_predictions["SelectedBy"] = "validation_J"
    test_predictions["Seed"] = cli.seed
    test_predictions["Method"] = METHOD
    test_predictions["Split"] = "test"
    test_predictions["SelectedBy"] = "validation_J"

    pd.DataFrame(epoch_rows).to_csv(
        result_dir / "cmixup_epoch_metrics.csv", index=False
    )
    valid_predictions.to_csv(
        result_dir / "cmixup_valid_predictions.csv", index=False
    )
    test_predictions.to_csv(
        result_dir / "cmixup_test_predictions.csv", index=False
    )
    pd.DataFrame(comparison_rows(candidate, baseline)).to_csv(
        result_dir / "cmixup_comparison.csv", index=False
    )

    source_manifest = {
        "version": VERSION,
        "seed": cli.seed,
        "base_method": "DLF-CFCompatKD-v1",
        "baseline_result": str(baseline_path),
        "baseline_result_sha256": checkpoint_sha256(baseline_path),
        "teacher_checkpoint": str(teacher_checkpoint),
        "teacher_sha256": teacher_sha,
        "evaluator_checkpoint": str(evaluator_checkpoint),
        "evaluator_sha256": evaluator_sha,
        "evaluator_source": str(evaluator_source),
        "compatibility_cache": str(cache_file),
        "compatibility_cache_sha256": cache_sha,
        "candidate_checkpoint": str(checkpoint),
        "candidate_checkpoint_sha256": candidate["MainCheckpointSHA256"],
        "selection_split": "valid",
        "test_created_after_selection": True,
        "test_loader_traversal_count": 1,
        "no_inference_architecture_change": True,
    }
    (result_dir / "cmixup_source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n"
    )
    summary = {
        "version": VERSION,
        "method": METHOD,
        "seed": cli.seed,
        "verdict": verdict,
        "baseline": baseline,
        "candidate": candidate,
        "promotion_gate": gate,
        "protocol": {
            "decision_split": "valid",
            "alpha": cli.alpha,
            "bandwidth": cli.bandwidth,
            "mix_weight": cli.mix_weight,
            "fusion_location": "DLF.proj1 forward-pre-hook input",
            "same_partner_full_and_missing": True,
            "same_missing_mode_only": True,
            "mixed_samples_use_teacher_or_compatibility": False,
            "original_cfcompat_objective_unchanged": True,
            "test_used_for_selection": False,
            "test_loader_traversal_count": 1,
            "additional_inference_parameters": 0,
        },
    }
    (result_dir / "cmixup_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (result_dir / "cmixup_report.md").write_text(render_report(summary) + "\n")
    logger.info(
        "complete seed=%s verdict=%s baseline_J=%.6f candidate_J=%.6f "
        "gain=%+.6f test_LAV=%.6f",
        cli.seed,
        verdict,
        float(baseline["J_valid"]),
        float(candidate["J_valid"]),
        gate["gain_valid_J"],
        float(candidate["test_at_valid_best_LAV_MAE"]),
    )
    return result_dir, summary


def main():
    cli = parse_args()
    logger, log_path = create_logger(cli)
    result_dir, summary = train_one_seed(cli, logger)
    logger.info(
        "result=%s report=%s log=%s",
        result_dir,
        result_dir / "cmixup_report.md",
        log_path,
    )
    print("CFCompat C-Mixup complete")
    print("seed:", cli.seed)
    print("verdict:", summary["verdict"])
    print(
        "valid J gain:",
        "{:+.6f}".format(summary["promotion_gate"]["gain_valid_J"]),
    )
    print(
        "valid LAV MAE gain:",
        "{:+.6f}".format(
            summary["promotion_gate"]["gain_valid_LAV_MAE"]
        ),
    )
    print("report:", result_dir / "cmixup_report.md")


if __name__ == "__main__":
    main()
