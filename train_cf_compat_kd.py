"""Stage 3B counterfactual compatibility-gated prediction KD benchmark entrypoint."""
import argparse
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

from config import get_config_regression
from data_loader import MMDataLoader
from trains.singleTask.HingeLoss import HingeLoss
from trains.singleTask.cf_compat_kd_utils import (
    CACHE_VERSION,
    MULTISEED_CACHE_VERSION,
    MULTISEED_SMOKE_CACHE_VERSION,
    build_counterfactual_cache,
    build_frozen_evaluator,
    cache_paths,
    compatibility_for_modes,
    distribution_stats,
    effective_sample_size,
    gate_weights,
    gated_kd_loss,
    load_counterfactual_cache,
    locate_stage1_evaluator,
    modes_from_masks,
    write_counterfactual_cache,
)
from trains.singleTask.fixed_kd_utils import (
    assert_initial_lav_equivalence,
    assert_teacher_not_in_optimizer,
    build_frozen_teacher,
    checkpoint_sha256,
    compute_validation_gaps,
    teacher_grad_count,
    teacher_lav_prediction,
)
from trains.singleTask.missing_utils import (
    MISSING_MODES,
    MissingModalityWrapper,
    build_single_split_loader,
    clean_checkpoint_path,
    compute_full_dlf_loss,
    compute_task_loss,
    count_missing_modes,
    evaluate_all_modes,
    flatten_mode_metrics,
    mode_to_mask,
    sample_missing_masks,
    validation_objective,
    write_result_csvs,
)
from trains.singleTask.model.DLF import DLF
from utils.functions import assign_gpu, setup_seed


METHODS = {
    "compat": ("DLF-CFCompatKD-v1", "cf_compat_kd_v1", "cfcompatkd"),
    "reliability_compat": ("DLF-ReliabilityCFCompatKD-v1", "reliability_cf_compat_kd_v1", "reliabilitycfcompatkd"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3B counterfactual compatibility-gated KD.")
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1111])
    parser.add_argument("--gate-mode", choices=tuple(METHODS), default="compat")
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--lambda-kd", type=float, default=1.0)
    parser.add_argument("--build-gate-cache-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--log-dir", default="log/missing_baseline")
    parser.add_argument("--config-file", default="config/config.json")
    parser.add_argument("--multiseed-replication", action="store_true")
    args = parser.parse_args()
    if args.eta != 1.0 or args.lambda_kd != 1.0:
        parser.error("Stage 3B fixes --eta and --lambda-kd at 1.0.")
    if args.max_epochs is not None and args.max_epochs < 1:
        parser.error("--max-epochs must be positive.")
    if args.smoke_test:
        args.max_epochs = 2 if args.max_epochs is None else min(2, args.max_epochs)
    if args.multiseed_replication:
        if args.gate_mode != "compat":
            parser.error("Stage 3B-M permits only --gate-mode compat.")
        if len(args.seeds) != 1 or args.seeds[0] not in (1112, 1113, 1114, 1115):
            parser.error("Stage 3B-M runs exactly one new seed (1112-1115) per process.")
    return args


def build_config(cli, seed):
    args = get_config_regression("DLF", cli.dataset, cli.config_file)
    args.mode = "train"
    args.feature_T = args.feature_A = args.feature_V = ""
    args.is_training = True
    args.train_mode = "regression"
    args.seed = args.cur_seed = int(seed)
    args.device = assign_gpu(list(cli.gpu_ids))
    return args


def batch_to_device(batch, device):
    return (
        batch["text"].to(device),
        batch["audio"].to(device),
        batch["vision"].to(device),
        batch["labels"]["M"].to(device).view(-1, 1),
    )


def method_paths(cli, dataset, seed=None):
    _, version, _ = METHODS[cli.gate_mode]
    if getattr(cli, "multiseed_replication", False):
        if seed is None:
            if len(cli.seeds) != 1:
                raise ValueError("Stage 3B-M path resolution requires one seed.")
            seed = cli.seeds[0]
        result = Path(cli.result_root) / "missing_baseline" / version / "benchmark_multiseed"
        main_root = Path(cli.model_save_dir) / "missing_baseline" / version / "benchmark_multiseed"
        if cli.smoke_test:
            result, main_root = result / "smoke", main_root / "smoke"
        result = result / "seed{}".format(int(seed))
        main = main_root / "seed{}".format(int(seed)) / "DLF_{}_seed{{}}_best_valid.pth".format(dataset)
    else:
        result = Path(cli.result_root) / "missing_baseline" / version / "benchmark_train"
        if cli.smoke_test:
            result = result / "smoke"
        main = Path(cli.model_save_dir) / "missing_baseline" / version / "DLF_{}_seed{{}}_best_valid.pth".format(dataset)
        if cli.smoke_test:
            main = main.parent / "smoke" / main.name
    diagnostic = main.parent / "diagnostic" / main.name.replace("_best_valid.pth", "_best_test_diagnostic.pth")
    return result, main, diagnostic


def create_logger(cli):
    _, _, tag = METHODS[cli.gate_mode]
    directory = Path(cli.log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    kind = "smoke" if cli.smoke_test else "train"
    if getattr(cli, "multiseed_replication", False):
        path = directory / "DLF-{}-{}-multiseed-seed{}-{}-{}.log".format(
            cli.dataset, tag, cli.seeds[0], kind, datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        path = directory / "DLF-{}-{}-{}-{}.log".format(cli.dataset, tag, kind, datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = logging.getLogger("cf_compat_kd")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def initialize_teacher_student(args, cli, seed, loaders):
    checkpoint = clean_checkpoint_path(cli.model_save_dir, args.dataset_name, seed)
    if not checkpoint.is_file():
        raise FileNotFoundError("Gate 3 teacher checkpoint missing: {}".format(checkpoint))
    sha = checkpoint_sha256(checkpoint)
    teacher = build_frozen_teacher(DLF, args, checkpoint)
    backbone = DLF(args).to(args.device)
    backbone.load_state_dict(torch.load(checkpoint, map_location=args.device), strict=True)
    student = MissingModalityWrapper(backbone, args.feature_dims[1], args.feature_dims[2]).to(args.device)
    student.eval()
    first = next(iter(loaders["valid"]))
    text, audio, vision, _ = batch_to_device(first, args.device)
    assert_initial_lav_equivalence(teacher, student, text, audio, vision)
    return teacher, student, checkpoint, sha


def prediction_rows(model, loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            text, audio, vision, labels = batch_to_device(batch, device)
            predictions = {}
            for mode in ("LAV",) + MISSING_MODES:
                mask = mode_to_mask(mode, labels.size(0), device, audio.dtype)
                predictions[mode] = model(text, audio, vision, mask)["output_logit"].view(-1).cpu().numpy()
            for offset, index in enumerate(batch["index"].view(-1).cpu().numpy().astype(int)):
                rows.append({
                    "sample_id": str(list(batch["id"])[offset]), "sample_index": int(index),
                    "label": float(labels[offset].item()),
                    **{"{}_pred".format(mode): float(predictions[mode][offset]) for mode in predictions},
                })
    return pd.DataFrame(rows).sort_values("sample_index", kind="mergesort")


def _grad_norm(gradients):
    return math.sqrt(sum(float(g.detach().pow(2).sum()) for g in gradients if g is not None))


def _diagnostic_rows(records, seed, epoch, gate_mode):
    """Return true gate aggregate, compatibility quartiles, and R/C 2x2 rows."""
    data = pd.DataFrame(records)
    if data.empty:
        raise RuntimeError("No gate diagnostics were recorded.")
    gate = distribution_stats(data.gate)
    gate_summary = {
        "Seed": seed, "Epoch": epoch, "GateMode": gate_mode, "SampleCount": len(data),
        **{"gate_{}".format(key): value for key, value in gate.items()},
        "ESS": effective_sample_size(data.gate), "ESS_fraction": effective_sample_size(data.gate) / len(data),
        "mean_reliability": float(data.reliability.mean()), "mean_compat": float(data.compat.mean()),
        "mean_delta": float(data.delta.mean()), "mean_unweighted_kd": float(data.kd.mean()),
        "mean_weighted_kd_contribution": float(data.weighted_kd_contribution.mean()),
        "mean_student_missing_abs_label_error": float(data.student_missing_abs_label_error.mean()),
    }
    ordered = data.sort_values(["compat", "sample_index"], kind="mergesort").copy()
    ordered["compat_quartile"] = pd.qcut(np.arange(len(ordered)), 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    quartiles = []
    for group, local in ordered.groupby("compat_quartile", observed=False):
        row = {
            "Seed": seed, "Epoch": epoch, "GateMode": gate_mode, "GroupType": "compat_quartile",
            "Group": str(group), "count": len(local), "mean_delta": float(local.delta.mean()),
            "mean_compat": float(local.compat.mean()), "mean_reliability": float(local.reliability.mean()),
            "mean_final_gate": float(local.gate.mean()), "mean_unweighted_kd": float(local.kd.mean()),
            "mean_weighted_kd_contribution": float(local.weighted_kd_contribution.mean()),
            "mean_student_missing_abs_label_error": float(local.student_missing_abs_label_error.mean()),
        }
        row.update({"{}_count".format(mode): int((local["mode"] == mode).sum()) for mode in MISSING_MODES})
        quartiles.append(row)
    if gate_mode == "reliability_compat":
        r_med, c_med = data.reliability.median(), data.compat.median()
        labels = np.char.add(np.where(data.reliability >= r_med, "R_high_", "R_low_"), np.where(data.compat >= c_med, "C_high", "C_low"))
        for name in ("R_high_C_high", "R_high_C_low", "R_low_C_high", "R_low_C_low"):
            local = data.loc[labels == name]
            if len(local):
                quartiles.append({
                    "Seed": seed, "Epoch": epoch, "GateMode": gate_mode, "GroupType": "reliability_compat_2x2",
                    "Group": name, "count": len(local), "mean_teacher_error": float(local.teacher_error.mean()),
                    "mean_delta": float(local.delta.mean()), "mean_compat": float(local.compat.mean()),
                    "mean_reliability": float(local.reliability.mean()), "mean_final_gate": float(local.gate.mean()),
                    "mean_unweighted_kd": float(local.kd.mean()),
                    "mean_student_missing_abs_label_error": float(local.student_missing_abs_label_error.mean()),
                })
    return gate_summary, quartiles


def _flatten(metrics, prefix):
    return {"{}_{}".format(prefix, key): value for key, value in flatten_mode_metrics(metrics).items()}


def build_gate_cache_only(cli, seed, logger):
    """This path constructs one non-shuffled train loader and no valid/test loader."""
    setup_seed(seed)
    args = build_config(cli, seed)
    multiseed = getattr(cli, "multiseed_replication", False)
    evaluator_checkpoint, best_epoch, source_csv = locate_stage1_evaluator(
        cli.result_root, cli.dataset, seed, multiseed=multiseed, smoke=cli.smoke_test)
    missing_generator = torch.Generator().manual_seed(int(seed) + 104729)
    train_loader = build_single_split_loader(args, "train", cli.num_workers)
    evaluator = build_frozen_evaluator(DLF, args, evaluator_checkpoint)
    frame = build_counterfactual_cache(evaluator, train_loader, args.device, missing_generator)
    cache_version = (MULTISEED_SMOKE_CACHE_VERSION if cli.smoke_test else MULTISEED_CACHE_VERSION) if multiseed else CACHE_VERSION
    paths, config = write_counterfactual_cache(
        frame, cli.result_root, cli.dataset, evaluator_checkpoint, best_epoch,
        version=cache_version, seed=seed if multiseed else None, rng_state_preserved=True)
    if len(frame) != 1284 or frame.sample_index.nunique() != 1284:
        raise RuntimeError("MOSI cache audit expected exactly 1284 unique train samples.")
    logger.info("train-only cache built samples=%s evaluator=%s source_csv=%s sha=%s",
                len(frame), evaluator_checkpoint, source_csv, config["evaluator_sha256"])
    for mode, values in __import__("json").load(open(paths["summary"])) .items():
        logger.info("cache mode=%s compat=[%.6f,%.6f] spearman=%.9f", mode, values["compat"]["min"],
                    values["compat"]["max"], values["corr_delta_compat_spearman"])
    return paths


def train_one_seed(cli, seed, logger):
    setup_seed(seed)
    args = build_config(cli, seed)
    multiseed = getattr(cli, "multiseed_replication", False)
    evaluator_checkpoint, evaluator_best_epoch, evaluator_source = locate_stage1_evaluator(
        cli.result_root, cli.dataset, seed, multiseed=multiseed, smoke=cli.smoke_test)
    evaluator_sha = checkpoint_sha256(evaluator_checkpoint)
    cache_version = (MULTISEED_SMOKE_CACHE_VERSION if cli.smoke_test else MULTISEED_CACHE_VERSION) if multiseed else CACHE_VERSION
    cache_frame, cache_by_index = load_counterfactual_cache(
        cli.result_root, cli.dataset, version=cache_version,
        seed=seed if multiseed else None, expected_evaluator_sha=evaluator_sha)
    if len(cache_frame) != 1284:
        raise RuntimeError("Stage 3B requires the audited 1284-sample MOSI train cache.")
    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("Benchmark training must construct train/valid loaders.")
    test_loader = build_single_split_loader(args, "test", cli.num_workers)
    teacher, student, teacher_checkpoint, teacher_sha = initialize_teacher_student(args, cli, seed, loaders)
    optimizer = optim.Adam(student.parameters(), lr=args.learning_rate)
    assert_teacher_not_in_optimizer(teacher, optimizer)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=args.patience)
    criterion, cosine, hinge = nn.L1Loss(), nn.CosineEmbeddingLoss(), HingeLoss()
    missing_generator = torch.Generator().manual_seed(int(seed) + 104729)
    result_dir, main_template, diagnostic_template = method_paths(cli, cli.dataset, seed)
    main_checkpoint = Path(str(main_template).format(seed))
    diagnostic_checkpoint = Path(str(diagnostic_template).format(seed))
    main_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    method, _, _ = METHODS[cli.gate_mode]
    best_valid_j, best_valid_epoch = float("inf"), 0
    best_valid_metrics = best_valid_test = None
    best_test_j, best_test_epoch = float("inf"), 0
    epoch_rows, gate_summaries, quartiles = [], [], []
    logger.info("method=%s seed=%s benchmark/original protocol: valid and test every epoch; main checkpoint selected by J_valid only", method, seed)
    logger.info("teacher=%s teacher_sha=%s evaluator=%s evaluator_sha=%s evaluator_best_epoch=%s evaluator_source=%s",
                teacher_checkpoint, teacher_sha, evaluator_checkpoint, evaluator_sha, evaluator_best_epoch, evaluator_source)
    for epoch in range(1, (cli.max_epochs or 1000) + 1):
        student.train()
        optimizer.zero_grad()
        counts = Counter({"LA": 0, "LV": 0, "L": 0})
        gate_records = []; batch_kd_losses = []
        for step, batch in enumerate(loaders["train"], 1):
            text, audio, vision, labels = batch_to_device(batch, args.device)
            full_mask = mode_to_mask("LAV", labels.size(0), args.device, audio.dtype)
            full_loss, _ = compute_full_dlf_loss(student(text, audio, vision, full_mask), labels, criterion, cosine, hinge)
            missing_mask = sample_missing_masks(labels.size(0), missing_generator, args.device, audio.dtype)
            modes = modes_from_masks(missing_mask)
            counts.update(count_missing_modes(missing_mask))
            missing_output = student(text, audio, vision, missing_mask)
            missing_loss, _ = compute_task_loss(missing_output, labels, criterion)
            teacher_prediction = teacher_lav_prediction(teacher, text, audio, vision)
            indices = batch["index"].view(-1).cpu().numpy().astype(int).tolist()
            compatibility = compatibility_for_modes(cache_by_index, indices, modes, args.device, labels.dtype)
            gate, reliability = gate_weights(compatibility, teacher_prediction, labels, cli.gate_mode)
            kd_loss, each_kd = gated_kd_loss(missing_output["output_logit"], teacher_prediction, gate)
            batch_kd_losses.append(float(kd_loss.detach()))
            if step == 1:
                gradients = torch.autograd.grad(kd_loss, [p for p in student.parameters() if p.requires_grad],
                                                retain_graph=True, allow_unused=True)
                if _grad_norm(gradients) <= 0:
                    raise RuntimeError("Gated KD did not reach student parameters.")
            total_loss = full_loss + missing_loss + kd_loss
            if not torch.isfinite(total_loss):
                raise FloatingPointError("NaN/Inf in Stage 3B loss.")
            total_loss.backward()
            if teacher_grad_count(teacher):
                raise RuntimeError("Frozen full teacher received gradients.")
            if step % args.update_epochs == 0 or step == len(loaders["train"]):
                optimizer.step()
                optimizer.zero_grad()
            denominator = float(gate.sum().detach().cpu()) + 1e-8
            student_error = torch.abs(missing_output["output_logit"].detach().view(-1) - labels.view(-1))
            teacher_error = torch.abs(teacher_prediction.detach().view(-1) - labels.view(-1))
            teacher_student_gap = torch.abs(missing_output["output_logit"].detach().view(-1) - teacher_prediction.detach().view(-1))
            for offset, index in enumerate(indices):
                mode = modes[offset]
                gate_records.append({
                    "sample_index": index, "mode": mode, "delta": cache_by_index[index]["delta_{}".format(mode)],
                    "compat": float(compatibility[offset]), "reliability": float(reliability[offset]),
                    "gate": float(gate[offset]), "kd": float(each_kd[offset].detach()),
                    "weighted_kd_contribution": float(gate[offset].detach() * each_kd[offset].detach() / denominator),
                    "student_missing_abs_label_error": float(student_error[offset]),
                    "teacher_error": float(teacher_error[offset]),
                    "teacher_student_abs_gap": float(teacher_student_gap[offset]),
                })
        if epoch == 1 and int(seed) == 1111 and counts != Counter({"LA": 435, "LV": 430, "L": 419}):
            raise RuntimeError("First epoch missing-mask counts must be LA=435 LV=430 L=419.")
        valid = evaluate_all_modes(student, loaders["valid"], args.device, "moddrop", criterion)
        test = evaluate_all_modes(student, test_loader, args.device, "moddrop", criterion)
        j_valid, j_test = validation_objective(valid), validation_objective(test)
        if not (math.isfinite(j_valid) and math.isfinite(j_test)):
            raise FloatingPointError("Non-finite benchmark metric.")
        scheduler.step(j_valid)
        is_best_valid = j_valid <= best_valid_j - 1e-6
        is_best_test = j_test <= best_test_j - 1e-6
        if is_best_valid:
            best_valid_j, best_valid_epoch, best_valid_metrics, best_valid_test = j_valid, epoch, valid, test
            torch.save(student.state_dict(), main_checkpoint)
        if is_best_test:
            best_test_j, best_test_epoch = j_test, epoch
            torch.save(student.state_dict(), diagnostic_checkpoint)
        gate_summary, gate_quartiles = _diagnostic_rows(gate_records, seed, epoch, cli.gate_mode)
        gate_summaries.append(gate_summary)
        quartiles.extend(gate_quartiles)
        row = {
            "Seed": seed, "Epoch": epoch, "Method": method, "J_valid": j_valid, "J_test": j_test,
            "IsBestValid": is_best_valid, "IsBestTestDiagnostic": is_best_test,
            "KD_loss": float(np.mean(batch_kd_losses)),
            "train_teacher_student_abs_gap": float(np.mean([entry["teacher_student_abs_gap"] for entry in gate_records])),
            "ESS": gate_summary["ESS"], "ESS_fraction": gate_summary["ESS_fraction"],
            **{"gate_{}".format(key): value for key, value in distribution_stats([entry["gate"] for entry in gate_records]).items()},
            **_flatten(valid, "valid"), **_flatten(test, "test"),
        }
        epoch_rows.append(row)
        logger.info("epoch=%s LA=%s LV=%s L=%s J_valid=%.6f J_test=%.6f KD=%.6f gate=%.6f ESS_fraction=%.6f",
                    epoch, counts["LA"], counts["LV"], counts["L"], j_valid, j_test, row["KD_loss"],
                    row["gate_mean"], row["ESS_fraction"])
        if epoch - best_valid_epoch >= args.early_stop:
            break
    if not main_checkpoint.is_file() or not diagnostic_checkpoint.is_file():
        raise RuntimeError("Both main and diagnostic checkpoints must exist.")
    student.load_state_dict(torch.load(main_checkpoint, map_location=args.device), strict=True)
    final_valid = evaluate_all_modes(student, loaders["valid"], args.device, "moddrop", criterion)
    final_test = evaluate_all_modes(student, test_loader, args.device, "moddrop", criterion)
    valid_predictions = prediction_rows(student, loaders["valid"], args.device)
    student.load_state_dict(torch.load(diagnostic_checkpoint, map_location=args.device), strict=True)
    diagnostic_test = evaluate_all_modes(student, test_loader, args.device, "moddrop", criterion)
    diagnostic_predictions = prediction_rows(student, test_loader, args.device)
    diagnostic_predictions["selected_by"] = "test"
    diagnostic_predictions["diagnostic_only"] = True
    diagnostic_predictions["not_main_result"] = True
    diagnostic_gaps = compute_validation_gaps(teacher, student, test_loader, args.device)
    student.load_state_dict(torch.load(main_checkpoint, map_location=args.device), strict=True)
    valid_gaps = compute_validation_gaps(teacher, student, loaders["valid"], args.device)
    test_gaps = compute_validation_gaps(teacher, student, test_loader, args.device)
    result = {
        "Seed": seed, "Method": method, "BestValidEpoch": best_valid_epoch,
        "J_valid": validation_objective(final_valid), "J_test_at_valid_best": validation_objective(final_test),
        "BestObservedTestEpoch": best_test_epoch, "BestObservedTestJ": best_test_j,
        "MainCheckpoint": str(main_checkpoint), "DiagnosticCheckpoint": str(diagnostic_checkpoint),
        "TeacherCheckpoint": str(teacher_checkpoint), "TeacherSHA256": teacher_sha,
        "StudentInitCheckpoint": str(teacher_checkpoint), "StudentInitSHA256": teacher_sha,
        "EvaluatorCheckpoint": str(evaluator_checkpoint), "EvaluatorSHA256": evaluator_sha,
        "EvaluatorBestEpoch": evaluator_best_epoch,
        **_flatten(final_valid, "valid"), **_flatten(final_test, "test_at_valid_best"),
        **_flatten(diagnostic_test, "test_diagnostic"), **{"valid_{}".format(k): v for k, v in valid_gaps.items()},
        **{"test_at_valid_best_{}".format(k): v for k, v in test_gaps.items()},
        **{"test_diagnostic_{}".format(k): v for k, v in diagnostic_gaps.items()},
    }
    return result, epoch_rows, gate_summaries, quartiles, valid_predictions, diagnostic_predictions


def main():
    cli = parse_args()
    logger, log_path = create_logger(cli)
    if cli.build_gate_cache_only:
        for seed in cli.seeds:
            build_gate_cache_only(cli, seed, logger)
        logger.info("cache audit complete; no valid/test loader was constructed; log=%s", log_path)
        return
    rows, epochs, summaries, quartiles = [], [], [], []
    valid_predictions, diagnostic_predictions = [], []
    for seed in cli.seeds:
        row, per_epoch, gate_summary, gate_quartiles, valid, diagnostic = train_one_seed(cli, seed, logger)
        rows.append(row); epochs.extend(per_epoch); summaries.extend(gate_summary); quartiles.extend(gate_quartiles)
        valid_predictions.append(valid); diagnostic_predictions.append(diagnostic)
    result_dir, _, _ = method_paths(cli, cli.dataset, cli.seeds[0] if cli.seeds else None)
    write_result_csvs(rows, result_dir, cli.dataset)
    pd.DataFrame(epochs).to_csv(result_dir / "{}_epoch_metrics.csv".format(cli.dataset), index=False)
    pd.DataFrame(summaries).to_csv(result_dir / "{}_gate_summary.csv".format(cli.dataset), index=False)
    pd.DataFrame(quartiles).to_csv(result_dir / "{}_gate_quartiles.csv".format(cli.dataset), index=False)
    pd.concat(valid_predictions, ignore_index=True).to_csv(result_dir / "{}_best_valid_predictions.csv".format(cli.dataset), index=False)
    pd.concat(diagnostic_predictions, ignore_index=True).to_csv(result_dir / "{}_best_test_diagnostic_predictions.csv".format(cli.dataset), index=False)
    logger.info("complete results=%s log=%s", result_dir, log_path)


if __name__ == "__main__":
    main()
