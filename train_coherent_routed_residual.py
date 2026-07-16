"""Stage 4A.1 coherent compatibility-routed residual distillation."""
import argparse
import json
import logging
import math
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import get_config_regression
from data_loader import MMDataLoader
from trains.singleTask.HingeLoss import HingeLoss
from trains.singleTask.cf_compat_kd_utils import modes_from_masks
from trains.singleTask.cf_residual_utils import (
    MODE_ORDER,
    build_residual_student,
    correction_gaps,
    evaluate_residual_modes,
    evaluation_objectives,
    flatten_evaluation,
    load_residual_cache,
    prediction_rows,
    residual_cache_paths,
    residual_diagnostic_rows,
    residual_targets_for_modes,
)
from trains.singleTask.coherent_routed_residual_utils import (
    ROUTE_VARIANTS,
    alignment_paths,
    build_teacher_evaluator_alignment,
    distribution,
    gradient_alignment_audit,
    missing_sequence_sha256,
    route_diagnostic_rows,
    shared_route_loss,
)
from trains.singleTask.fixed_kd_utils import (
    assert_teacher_not_in_optimizer,
    build_frozen_teacher,
    checkpoint_sha256,
    teacher_grad_count,
    teacher_lav_prediction,
)
from trains.singleTask.missing_utils import (
    build_single_split_loader,
    clean_checkpoint_path,
    compute_full_dlf_loss,
    compute_task_loss,
    count_missing_modes,
    mode_to_mask,
    sample_missing_masks,
    write_result_csvs,
)
from trains.singleTask.model.DLF import DLF
from utils.functions import assign_gpu, setup_seed


RESULT_FILES = (
    "mosi_per_seed.csv", "mosi_summary.csv", "mosi_epoch_metrics.csv",
    "mosi_route_summary.csv", "mosi_route_quartiles.csv",
    "mosi_residual_mode_metrics.csv", "mosi_best_valid_predictions.csv",
    "mosi_best_test_diagnostic_predictions.csv", "gradient_alignment_init.csv",
    "gradient_alignment_best_valid.csv",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Stage 4A.1 coherent shared-budget routing.")
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1111])
    parser.add_argument("--route-variant", choices=tuple(ROUTE_VARIANTS))
    parser.add_argument("--lambda-route", type=float, default=1.0)
    parser.add_argument("--build-alignment-audit-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--log-dir", default="log/missing_baseline")
    parser.add_argument("--config-file", default="config/config.json")
    cli = parser.parse_args(argv)
    if cli.seeds != [1111]:
        parser.error("Stage 4A.1 is locked to exactly seed1111.")
    if cli.lambda_route != 1.0:
        parser.error("Stage 4A.1 fixes lambda_route=1.0.")
    if not cli.build_alignment_audit_only and cli.route_variant is None:
        parser.error("--route-variant is required for training.")
    if cli.max_epochs is not None and cli.max_epochs < 1:
        parser.error("--max-epochs must be positive.")
    if cli.smoke_test:
        cli.max_epochs = 2 if cli.max_epochs is None else min(cli.max_epochs, 2)
    return cli


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def git_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def build_config(cli, seed=1111):
    args = get_config_regression("DLF", cli.dataset, cli.config_file)
    args.mode = "train"; args.feature_T = args.feature_A = args.feature_V = ""
    args.is_training = True; args.train_mode = "regression"
    args.seed = args.cur_seed = int(seed); args.device = assign_gpu(list(cli.gpu_ids))
    return args


def batch_to_device(batch, device):
    return (batch["text"].to(device), batch["audio"].to(device), batch["vision"].to(device),
            batch["labels"]["M"].to(device).view(-1, 1))


def method_paths(cli):
    version = ROUTE_VARIANTS[cli.route_variant]["version"]
    result = Path(cli.result_root) / "missing_baseline" / version / "benchmark_train"
    main = Path(cli.model_save_dir) / "missing_baseline" / version / "DLF_mosi_seed1111_best_valid.pth"
    if cli.smoke_test:
        result = result / "smoke"; main = main.parent / "smoke" / main.name
    diagnostic = main.parent / "diagnostic" / main.name.replace("_best_valid.pth", "_best_test_diagnostic.pth")
    return result, main, diagnostic


def create_logger(cli):
    tag = "alignment-audit" if cli.build_alignment_audit_only else ROUTE_VARIANTS[cli.route_variant]["log_tag"]
    kind = "smoke" if cli.smoke_test else "train"
    directory = Path(cli.log_dir); directory.mkdir(parents=True, exist_ok=True)
    path = directory / "DLF-mosi-{}-seed1111-{}-{}.log".format(tag, kind, datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = logging.getLogger("coherent_routed_residual"); logger.handlers.clear(); logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(formatter); logger.addHandler(handler)
    return logger, path


def _autograd_norm(loss, parameters):
    gradients = torch.autograd.grad(loss, list(parameters), retain_graph=True, allow_unused=True)
    return math.sqrt(sum(float(gradient.detach().pow(2).sum()) for gradient in gradients if gradient is not None))


def _grad_norm(parameters):
    return math.sqrt(sum(float(parameter.grad.detach().pow(2).sum()) for parameter in parameters if parameter.grad is not None))


def initialize_models(args, cli, valid_loader, scales):
    gate3 = clean_checkpoint_path(cli.model_save_dir, cli.dataset, 1111)
    if not gate3.is_file():
        raise FileNotFoundError("Gate 3 seed1111 validation-best checkpoint missing: {}".format(gate3))
    gate3_sha = checkpoint_sha256(gate3)
    teacher = build_frozen_teacher(DLF, args, gate3)
    student = build_residual_student(args, gate3, scales); student.eval()
    first = next(iter(valid_loader)); text, audio, vision, _ = batch_to_device(first, args.device)
    lav_mask = mode_to_mask("LAV", text.size(0), args.device, audio.dtype)
    with torch.no_grad():
        teacher_prediction = teacher_lav_prediction(teacher, text, audio, vision)
        output = student(text, audio, vision, lav_mask)
    torch.testing.assert_close(output["base_output_logit"], teacher_prediction, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(output["corrected_output_logit"], output["base_output_logit"], rtol=0, atol=0)
    if torch.count_nonzero(output["predicted_residual"]):
        raise RuntimeError("LAV residual must be exactly zero.")
    return teacher, student, gate3, gate3_sha


def _formal_output_guard(cli, result_dir, main_checkpoint, diagnostic_checkpoint):
    if cli.smoke_test:
        return
    existing = [result_dir / name for name in RESULT_FILES if (result_dir / name).exists()]
    existing.extend(path for path in (main_checkpoint, diagnostic_checkpoint) if path.exists())
    if existing:
        raise FileExistsError("Formal Stage 4A.1 outputs already exist; selective reruns are forbidden: {}".format(existing))


def train_one_seed(cli, logger, log_path):
    seed = 1111; setup_seed(seed); start_time = utc_now(); args = build_config(cli, seed)
    cache_frame, cache_by_index, scales, cache_config = load_residual_cache(cli.result_root, cli.dataset, seed)
    if len(cache_frame) != 1284 or cache_config["ResidualCacheSHA256"] != "8702d2ff15b80594094ea74469ad393a681992ecce1891c7095208f8c1004790":
        raise RuntimeError("Stage 4A.1 must reuse the locked Stage 4A residual cache.")
    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("Benchmark training must construct train and valid loaders only.")
    test_loader = build_single_split_loader(args, "test", cli.num_workers)
    audit_loader = build_single_split_loader(args, "train", cli.num_workers)
    teacher, student, gate3, gate3_sha = initialize_models(args, cli, loaders["valid"], scales)
    result_dir, main_checkpoint, diagnostic_checkpoint = method_paths(cli)
    _formal_output_guard(cli, result_dir, main_checkpoint, diagnostic_checkpoint)
    main_checkpoint.parent.mkdir(parents=True, exist_ok=True); diagnostic_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    criterion, cosine, hinge = nn.L1Loss(), nn.CosineEmbeddingLoss(), HingeLoss()
    variant = cli.route_variant; method = ROUTE_VARIANTS[variant]["method"]
    gradient_init = gradient_alignment_audit(
        student, teacher, audit_loader, cache_by_index, scales, args.device, variant, criterion, "initialization")
    optimizer = optim.Adam(student.parameters(), lr=args.learning_rate)
    assert_teacher_not_in_optimizer(teacher, optimizer)
    head_ids = {id(parameter) for parameter in student.residual_heads.parameters()}
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if not head_ids.issubset(optimizer_ids):
        raise RuntimeError("Residual heads are absent from the optimizer.")
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=args.patience)
    missing_generator = torch.Generator().manual_seed(seed + 104729)
    best_valid_j = best_test_j = float("inf"); best_valid_epoch = best_test_epoch = 0
    epoch_rows, route_summaries, route_quartiles, mode_metrics = [], [], [], []
    epoch1_modes = []
    logger.info("pid=%s commit=%s device_request=%s method=%s variant=%s", os.getpid(), git_head(), cli.gpu_ids, method, variant)
    logger.info("teacher=%s sha=%s evaluator=%s evaluator_sha=%s residual_cache=%s residual_sha=%s",
                gate3, gate3_sha, cache_config["EvaluatorCheckpoint"], cache_config["EvaluatorSHA256"],
                residual_cache_paths(cli.result_root, cli.dataset)["csv"], cache_config["ResidualCacheSHA256"])
    logger.info("shared route is exactly mean(C*d+(1-C)*u); valid/test are Student-only")
    for epoch in range(1, (cli.max_epochs or 1000) + 1):
        student.train(); optimizer.zero_grad(); counts = Counter({mode: 0 for mode in MODE_ORDER}); records = []
        full_total = missing_total = 0.0
        for step, batch in enumerate(loaders["train"], 1):
            text, audio, vision, labels = batch_to_device(batch, args.device)
            full_output = student(text, audio, vision, mode_to_mask("LAV", labels.size(0), args.device, audio.dtype))
            full_loss, _ = compute_full_dlf_loss(full_output, labels, criterion, cosine, hinge)
            missing_mask = sample_missing_masks(labels.size(0), missing_generator, args.device, audio.dtype)
            modes = modes_from_masks(missing_mask); counts.update(count_missing_modes(missing_mask))
            if epoch == 1:
                epoch1_modes.extend(modes)
            missing_output = student(text, audio, vision, missing_mask)
            missing_loss, _ = compute_task_loss(missing_output, labels, criterion)
            indices = batch["index"].view(-1).cpu().numpy().astype(int).tolist()
            compatibility, target_residual, target_z = residual_targets_for_modes(
                cache_by_index, indices, modes, scales, args.device, labels.dtype)
            teacher_prediction = teacher_lav_prediction(teacher, text, audio, vision)
            routed = shared_route_loss(missing_output, teacher_prediction, target_z, compatibility, variant)
            if epoch == 1 and step == 1:
                torch.testing.assert_close(missing_output["corrected_output_logit"], missing_output["base_output_logit"], rtol=0, atol=0)
                if torch.count_nonzero(missing_output["predicted_residual"]):
                    raise RuntimeError("Zero-init first-batch residual is not exactly zero.")
                direct_head_norm = _autograd_norm(routed["weighted_direct"].mean(), student.residual_heads.parameters())
                if variant == "corrected_joint_shared" and direct_head_norm <= 0:
                    raise RuntimeError("Variant B direct KD did not reach residual heads.")
                if variant == "corrected_stopres_shared" and direct_head_norm != 0:
                    raise RuntimeError("Variant C direct KD reached residual heads.")
                if variant == "corrected_stopres_shared" and _autograd_norm(
                        routed["weighted_direct"].mean(), student.base_student.parameters()) <= 0:
                    raise RuntimeError("Variant C direct KD did not reach the base Student.")
            total_loss = full_loss + missing_loss + routed["loss"]
            if not torch.isfinite(total_loss):
                raise FloatingPointError("NaN/Inf in Stage 4A.1 loss.")
            total_loss.backward()
            if teacher_grad_count(teacher):
                raise RuntimeError("Frozen Teacher received gradients.")
            if epoch == 1 and step == 1:
                if _grad_norm(student.residual_heads.parameters()) <= 0 or _grad_norm(student.base_student.parameters()) <= 0:
                    raise RuntimeError("Student/residual total gradients must be non-zero.")
            if step % args.update_epochs == 0 or step == len(loaders["train"]):
                optimizer.step(); optimizer.zero_grad()
            full_total += float(full_loss.detach()); missing_total += float(missing_loss.detach())
            base = missing_output["base_output_logit"].detach().view(-1)
            corrected = missing_output["corrected_output_logit"].detach().view(-1)
            predicted = missing_output["predicted_residual"].detach().view(-1)
            teacher_flat = teacher_prediction.detach().view(-1)
            for offset, index in enumerate(indices):
                records.append({
                    "sample_index": index, "mode": modes[offset], "compatibility": float(compatibility[offset]),
                    "complement": float(1.0 - compatibility[offset]), "target_residual": float(target_residual[offset]),
                    "predicted_residual": float(predicted[offset]), "base_prediction": float(base[offset]),
                    "corrected_prediction": float(corrected[offset]), "label": float(labels.view(-1)[offset]),
                    "direct_raw": float(routed["direct_each"][offset]), "residual_raw": float(routed["residual_each"][offset]),
                    "weighted_direct": float(routed["weighted_direct"][offset]),
                    "weighted_residual": float(routed["weighted_residual"][offset]),
                    "teacher_base_gap": float(torch.abs(base[offset] - teacher_flat[offset])),
                    "teacher_corrected_gap": float(torch.abs(corrected[offset] - teacher_flat[offset])),
                    "base_corrected_gap": float(torch.abs(corrected[offset] - base[offset])),
                })
        if epoch == 1 and counts != Counter({"LA": 435, "LV": 430, "L": 419}):
            raise RuntimeError("First epoch missing counts must be LA=435 LV=430 L=419.")
        route_summary, epoch_quartiles = route_diagnostic_rows(records, seed, epoch, method, variant)
        # Per-sample tensors are asserted bit-exact in ``shared_route_loss``.
        # This persisted diagnostic is reconstructed through Python floats.
        if not math.isclose(route_summary["MeanRouteWeightSum"], 1.0, rel_tol=0.0, abs_tol=1e-7):
            raise RuntimeError("Shared route weights must sum to one within serialization precision.")
        _, per_mode, _ = residual_diagnostic_rows(records, seed, epoch, method)
        for row in per_mode:
            row["RouteVariant"] = variant
        route_summaries.append(route_summary); route_quartiles.extend(epoch_quartiles); mode_metrics.extend(per_mode)
        valid = evaluate_residual_modes(student, loaders["valid"], args.device, criterion)
        test = evaluate_residual_modes(student, test_loader, args.device, criterion)
        j_valid, j_valid_base = evaluation_objectives(valid); j_test, j_test_base = evaluation_objectives(test)
        if not all(math.isfinite(value) for value in (j_valid, j_valid_base, j_test, j_test_base)):
            raise FloatingPointError("Non-finite Stage 4A.1 benchmark metric.")
        scheduler.step(j_valid)
        is_best_valid = j_valid <= best_valid_j - 1e-6; is_best_test = j_test <= best_test_j - 1e-6
        if is_best_valid:
            best_valid_j, best_valid_epoch = j_valid, epoch; torch.save(student.state_dict(), main_checkpoint)
        if is_best_test:
            best_test_j, best_test_epoch = j_test, epoch; torch.save(student.state_dict(), diagnostic_checkpoint)
        record_frame = pd.DataFrame(records); batches = len(loaders["train"])
        epoch_rows.append({
            "Seed": seed, "Epoch": epoch, "Method": method, "RouteVariant": variant,
            "J_valid_corrected": j_valid, "J_test_corrected": j_test,
            "J_valid_base": j_valid_base, "J_test_base": j_test_base,
            "IsBestValid": is_best_valid, "IsBestTestDiagnostic": is_best_test,
            "FullLoss": full_total / batches, "MissingLoss": missing_total / batches,
            **{key: route_summary[key] for key in ("RouteLoss", "DirectRawLossMean", "ResidualRawLossMean",
                "WeightedDirectContribution", "WeightedResidualContribution", "DirectContributionFraction",
                "ResidualContributionFraction", "MeanRouteWeightSum")},
            "TeacherStudentBaseGap": float(record_frame.teacher_base_gap.mean()),
            "TeacherStudentCorrectedGap": float(record_frame.teacher_corrected_gap.mean()),
            "BaseCorrectedAbsGap": float(record_frame.base_corrected_gap.mean()),
            "LA_count": counts["LA"], "LV_count": counts["LV"], "L_count": counts["L"],
            **{"compatibility_{}".format(k): v for k, v in distribution(record_frame.compatibility).items()},
            **{"predicted_residual_{}".format(k): v for k, v in distribution(record_frame.predicted_residual, True).items()},
            **flatten_evaluation(valid, "valid"), **flatten_evaluation(test, "test"),
        })
        logger.info("epoch=%s LA=%s LV=%s L=%s J_valid=%.6f J_test=%.6f route=%.6f direct_frac=%.4f residual_frac=%.4f",
                    epoch, counts["LA"], counts["LV"], counts["L"], j_valid, j_test, route_summary["RouteLoss"],
                    route_summary["DirectContributionFraction"], route_summary["ResidualContributionFraction"])
        if epoch - best_valid_epoch >= args.early_stop:
            break
    if not main_checkpoint.is_file() or not diagnostic_checkpoint.is_file():
        raise RuntimeError("Main and diagnostic checkpoints must both exist.")
    sequence_sha = missing_sequence_sha256(epoch1_modes)
    student.load_state_dict(torch.load(main_checkpoint, map_location=args.device), strict=True)
    final_valid = evaluate_residual_modes(student, loaders["valid"], args.device, criterion)
    final_test = evaluate_residual_modes(student, test_loader, args.device, criterion)
    gradient_best = gradient_alignment_audit(
        student, teacher, audit_loader, cache_by_index, scales, args.device, variant, criterion, "best_valid")
    valid_predictions = prediction_rows(student, loaders["valid"], args.device)
    valid_predictions["selected_by"] = "validation"; valid_predictions["diagnostic_only"] = False
    valid_gaps = correction_gaps(student, loaders["valid"], args.device); test_gaps = correction_gaps(student, test_loader, args.device)
    student.load_state_dict(torch.load(diagnostic_checkpoint, map_location=args.device), strict=True)
    diagnostic_test = evaluate_residual_modes(student, test_loader, args.device, criterion)
    diagnostic_predictions = prediction_rows(student, test_loader, args.device)
    diagnostic_predictions["selected_by"] = "test"; diagnostic_predictions["diagnostic_only"] = True
    diagnostic_predictions["not_main_result"] = True
    diagnostic_gaps = correction_gaps(student, test_loader, args.device)
    student.load_state_dict(torch.load(main_checkpoint, map_location=args.device), strict=True)
    final_j_valid, final_base_j_valid = evaluation_objectives(final_valid)
    final_j_test, final_base_j_test = evaluation_objectives(final_test)
    cache_path = residual_cache_paths(cli.result_root, cli.dataset)["csv"]
    result = {
        "Seed": seed, "Method": method, "RouteVariant": variant, "BestValidEpoch": best_valid_epoch,
        "J_valid_corrected": final_j_valid, "J_test_corrected_at_valid_best": final_j_test,
        "J_valid_base": final_base_j_valid, "J_test_base_at_valid_best": final_base_j_test,
        "J_valid": final_j_valid, "J_test_at_valid_best": final_j_test,
        "BestObservedTestEpoch": best_test_epoch, "BestObservedTestJ": best_test_j,
        "SelectionRegret": final_j_test - best_test_j, "TotalEpochs": len(epoch_rows),
        "MainCheckpoint": str(main_checkpoint), "MainCheckpointSHA256": checkpoint_sha256(main_checkpoint),
        "DiagnosticCheckpoint": str(diagnostic_checkpoint), "DiagnosticCheckpointSHA256": checkpoint_sha256(diagnostic_checkpoint),
        "TeacherCheckpoint": str(gate3), "TeacherSHA256": gate3_sha,
        "StudentInitCheckpoint": str(gate3), "StudentInitSHA256": gate3_sha,
        "EvaluatorCheckpoint": cache_config["EvaluatorCheckpoint"], "EvaluatorSHA256": cache_config["EvaluatorSHA256"],
        "CompatibilityCache": cache_config["SourceCompatibilityCache"],
        "CompatibilityCacheSHA256": cache_config["SourceCompatibilityCacheSHA256"],
        "ResidualCache": str(cache_path), "ResidualCacheSHA256": cache_config["ResidualCacheSHA256"],
        "MissingSequenceSHA256": sequence_sha, "CodeCommit": git_head(), "PID": os.getpid(),
        "StartTime": start_time, "EndTime": utc_now(), "ExitCode": 0, "CUDADevice": "cuda:{}".format(cli.gpu_ids[0]),
        "LogPath": str(log_path), "EvaluatorForwardDuringTraining": False, "ValidTestUsedTeacher": False,
        "ValidTestUsedEvaluator": False, "ValidTestUsedTrainCache": False, "TeacherGradientDetected": False,
        "EvaluatorGradientDetected": False, "TestBasedMainSelection": False,
        **flatten_evaluation(final_valid, "valid"), **flatten_evaluation(final_test, "test_at_valid_best"),
        **flatten_evaluation(diagnostic_test, "test_diagnostic"),
        **{"valid_{}".format(key): value for key, value in valid_gaps.items()},
        **{"test_at_valid_best_{}".format(key): value for key, value in test_gaps.items()},
        **{"test_diagnostic_{}".format(key): value for key, value in diagnostic_gaps.items()},
    }
    return (result, epoch_rows, route_summaries, route_quartiles, mode_metrics,
            valid_predictions, diagnostic_predictions, gradient_init, gradient_best)


def write_outputs(cli, outputs):
    (result, epochs, route_summaries, quartiles, mode_metrics, valid, diagnostic,
     gradient_init, gradient_best) = outputs
    result_dir, _, _ = method_paths(cli); result_dir.mkdir(parents=True, exist_ok=True)
    write_result_csvs([result], result_dir, cli.dataset)
    pd.DataFrame(epochs).to_csv(result_dir / "mosi_epoch_metrics.csv", index=False)
    pd.DataFrame(route_summaries).to_csv(result_dir / "mosi_route_summary.csv", index=False)
    pd.DataFrame(quartiles).to_csv(result_dir / "mosi_route_quartiles.csv", index=False)
    pd.DataFrame(mode_metrics).to_csv(result_dir / "mosi_residual_mode_metrics.csv", index=False)
    valid.to_csv(result_dir / "mosi_best_valid_predictions.csv", index=False)
    diagnostic.to_csv(result_dir / "mosi_best_test_diagnostic_predictions.csv", index=False)
    pd.DataFrame(gradient_init).to_csv(result_dir / "gradient_alignment_init.csv", index=False)
    pd.DataFrame(gradient_best).to_csv(result_dir / "gradient_alignment_best_valid.csv", index=False)
    return result_dir


def main(argv=None):
    cli = parse_args(argv); logger, log_path = create_logger(cli); args = build_config(cli, 1111)
    if cli.build_alignment_audit_only:
        paths, metadata, frame = build_teacher_evaluator_alignment(
            args, cli.result_root, cli.model_save_dir, cli.num_workers)
        logger.info("alignment audit complete samples=%s csv_sha=%s train-only=true no valid/test",
                    len(frame), metadata["AlignmentCSVSHA256"])
        logger.info("artifacts=%s log=%s", paths["directory"], log_path); return
    outputs = train_one_seed(cli, logger, log_path); result_dir = write_outputs(cli, outputs)
    logger.info("complete result=%s main_checkpoint=%s sequence_sha=%s log=%s",
                result_dir, outputs[0]["MainCheckpoint"], outputs[0]["MissingSequenceSHA256"], log_path)


if __name__ == "__main__":
    main()
