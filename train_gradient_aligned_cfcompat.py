"""Stage 6B gradient-aligned counterfactual compatibility distillation."""
import argparse
import copy
import hashlib
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

from analyze_mode_gradient_conflict import load_locked_audit_cache
from config import get_config_regression
from data_loader import MMDataLoader
from trains.singleTask.HingeLoss import HingeLoss
from trains.singleTask.cf_compat_kd_utils import (
    CACHE_VERSION, compatibility_for_modes, gated_kd_loss,
    locate_stage1_evaluator, modes_from_masks,
)
from trains.singleTask.fixed_kd_utils import (
    assert_initial_lav_equivalence, assert_teacher_not_in_optimizer,
    build_frozen_teacher, checkpoint_sha256, teacher_grad_count,
    teacher_lav_prediction,
)
from trains.singleTask.gradient_aligned_cfcompat_utils import (
    GRADIENT_POLICIES, MissingSequenceDigest, add_gradient_tuples,
    clone_gradients, combine_task_kd_gradients, compare_tensor_tuples,
    finite_quantiles, gradient_dot, group_gradient_metrics,
    gradient_norm, ordered_autograd, replay_corrected_total,
    trainable_named_parameters, write_parameter_gradients,
)
from trains.singleTask.gradient_conflict_utils import (
    ALL_GROUP, assign_compatibility_quartiles, build_parameter_groups,
    capture_rng_state, clone_state_dict, protected_audit_state,
    representation_statistics, restore_rng_state, states_equal,
)
from trains.singleTask.missing_utils import (
    MISSING_MODES, MissingModalityWrapper, build_single_split_loader,
    clean_checkpoint_path, compute_full_dlf_loss, compute_task_loss,
    count_missing_modes, evaluate_all_modes, flatten_mode_metrics,
    mode_to_mask, sample_missing_masks, validation_objective,
    write_result_csvs,
)
from trains.singleTask.model.DLF import DLF
from utils.functions import assign_gpu, setup_seed


METHODS = {
    "manual_replay": ("DLF-CFCompatKD-ManualReplay-v1", "cfcompat_manual_replay_v1", "cfcompat-manual-replay"),
    "conflict_drop": ("DLF-CFCompatKD-ConflictDrop-v1", "cfcompat_conflict_drop_v1", "cfcompat-conflict-drop"),
    "task_anchored_projection": ("DLF-GA-CFCompatKD-v1", "ga_cfcompat_v1", "ga-cfcompat"),
}
EXPECTED_MANUAL = {
    "BestValidEpoch": 9,
    "J_valid": 0.677964,
    "J_test_at_valid_best": 0.717881,
    "test_at_valid_best_LAV_MAE": 0.715761,
    "test_at_valid_best_MissingMacro_MAE": 0.720002,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 6B GA-CFCompatKD benchmark.")
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1111])
    parser.add_argument("--gradient-policy", choices=GRADIENT_POLICIES, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--log-dir", default="log/missing_baseline")
    parser.add_argument("--config-file", default="config/config.json")
    args = parser.parse_args()
    if args.seeds != [1111]:
        parser.error("Stage 6B is fixed to seed1111.")
    if args.num_workers != 0:
        parser.error("Stage 6B fixes --num-workers at 0.")
    if args.max_epochs is not None and args.max_epochs < 1:
        parser.error("--max-epochs must be positive.")
    if args.smoke_test:
        args.max_epochs = min(args.max_epochs or 2, 2)
    return args


def build_config(cli, seed):
    args = get_config_regression("DLF", cli.dataset, cli.config_file)
    args.mode = "train"; args.feature_T = args.feature_A = args.feature_V = ""
    args.is_training = True; args.train_mode = "regression"
    args.seed = args.cur_seed = int(seed); args.device = assign_gpu(list(cli.gpu_ids))
    return args


def method_paths(cli, dataset):
    _, version, _ = METHODS[cli.gradient_policy]
    result = Path(cli.result_root) / "missing_baseline" / version / "benchmark_train"
    main = Path(cli.model_save_dir) / "missing_baseline" / version / "DLF_{}_seed{{}}_best_valid.pth".format(dataset)
    if cli.smoke_test:
        result = result / "smoke"; main = main.parent / "smoke" / main.name
    diagnostic = main.parent / "diagnostic" / main.name.replace("_best_valid.pth", "_best_test_diagnostic.pth")
    return result, main, diagnostic


def create_logger(cli):
    _, _, tag = METHODS[cli.gradient_policy]
    directory = Path(cli.log_dir); directory.mkdir(parents=True, exist_ok=True)
    kind = "smoke" if cli.smoke_test else "train"
    path = directory / "DLF-{}-{}-{}-{}.log".format(cli.dataset, tag, kind, datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = logging.getLogger("gradient_aligned_cfcompat"); logger.handlers.clear(); logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(formatter); logger.addHandler(handler)
    return logger, path


def batch_to_device(batch, device):
    return (batch["text"].to(device), batch["audio"].to(device), batch["vision"].to(device),
            batch["labels"]["M"].to(device).view(-1, 1))


def initialize_teacher_student(args, cli, seed, valid_loader):
    checkpoint = clean_checkpoint_path(cli.model_save_dir, args.dataset_name, seed)
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    sha = checkpoint_sha256(checkpoint)
    teacher = build_frozen_teacher(DLF, args, checkpoint)
    backbone = DLF(args).to(args.device); backbone.load_state_dict(torch.load(checkpoint, map_location=args.device), strict=True)
    student = MissingModalityWrapper(backbone, args.feature_dims[1], args.feature_dims[2]).to(args.device)
    student.eval(); batch = next(iter(valid_loader)); text, audio, vision, _ = batch_to_device(batch, args.device)
    assert_initial_lav_equivalence(teacher, student, text, audio, vision)
    return teacher, student, checkpoint, sha


def _subset_output(output, flags):
    keys = ("output_logit", "logits_c", "logits_l_hetero", "logits_v_hetero", "logits_a_hetero")
    return {key: output[key][flags] for key in keys}


def _batch_losses(student, teacher, batch, mask, cache_by_index, args, criterion, cosine, hinge):
    text, audio, vision, labels = batch_to_device(batch, args.device)
    full_mask = mode_to_mask("LAV", labels.size(0), args.device, audio.dtype)
    full_output = student(text, audio, vision, full_mask)
    full_loss, _ = compute_full_dlf_loss(full_output, labels, criterion, cosine, hinge)
    missing_output = student(text, audio, vision, mask)
    missing_loss, _ = compute_task_loss(missing_output, labels, criterion)
    teacher_prediction = teacher_lav_prediction(teacher, text, audio, vision)
    modes = modes_from_masks(mask)
    indices = batch["index"].view(-1).cpu().numpy().astype(int).tolist()
    compatibility = compatibility_for_modes(cache_by_index, indices, modes, args.device, labels.dtype)
    kd_loss, each_kd = gated_kd_loss(missing_output["output_logit"], teacher_prediction, compatibility)
    return full_loss, missing_loss, kd_loss, each_kd, compatibility, modes, labels


def _optimizer_state_difference(left, right):
    max_abs = 0.0; mismatch = 0
    def compare(a, b):
        nonlocal max_abs, mismatch
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b): mismatch += 1; return
            for key in a: compare(a[key], b[key])
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b): mismatch += 1; return
            for first, second in zip(a, b): compare(first, second)
        elif torch.is_tensor(a) and torch.is_tensor(b):
            if a.shape != b.shape: mismatch += 1; return
            difference = float(torch.max(torch.abs(a.detach().float() - b.detach().float())).cpu()) if a.numel() else 0.0
            max_abs = max(max_abs, difference)
            if difference > 1e-6: mismatch += 1
        elif a != b: mismatch += 1
    compare(left, right)
    return {"max_abs_optimizer_state_difference": max_abs, "optimizer_state_mismatch_count": mismatch}


def run_manual_equivalence_gate(student, teacher, batch, mask, cache_by_index, args, optimizer, scheduler):
    """Real DLF single-batch gradient and single-step replay gate."""
    names, parameters = trainable_named_parameters(student)
    state = clone_state_dict(student); optimizer_state = copy.deepcopy(optimizer.state_dict())
    scheduler_state = copy.deepcopy(scheduler.state_dict()); rng = capture_rng_state(); training = student.training
    criterion, cosine, hinge = nn.L1Loss(), nn.CosineEmbeddingLoss(), HingeLoss()
    try:
        student.train(); forward_rng = capture_rng_state(); optimizer.zero_grad(set_to_none=True)
        full, missing, kd, _, _, _, _ = _batch_losses(student, teacher, batch, mask, cache_by_index, args, criterion, cosine, hinge)
        (full + missing + kd).backward()
        backward_gradients = clone_gradients(parameters)
        optimizer.step(); scheduler.step(1.0)
        backward_parameters = tuple(parameter.detach().clone() for parameter in parameters)
        backward_optimizer = copy.deepcopy(optimizer.state_dict()); backward_scheduler = copy.deepcopy(scheduler.state_dict())

        student.load_state_dict(state, strict=True); optimizer.load_state_dict(optimizer_state); scheduler.load_state_dict(scheduler_state)
        restore_rng_state(forward_rng); optimizer.zero_grad(set_to_none=True)
        full, missing, kd, _, _, _, _ = _batch_losses(student, teacher, batch, mask, cache_by_index, args, criterion, cosine, hinge)
        task_gradients = ordered_autograd(full + missing, parameters, retain_graph=True)
        kd_gradients = ordered_autograd(kd, parameters, retain_graph=True)
        reference_total = ordered_autograd(full + missing + kd, parameters, retain_graph=False)
        used, _, _ = combine_task_kd_gradients(parameters, task_gradients, kd_gradients, "manual_replay")
        total = replay_corrected_total(reference_total, task_gradients, kd_gradients, used)
        write_parameter_gradients(parameters, total, accumulate=False)
        manual_gradients = clone_gradients(parameters)
        gradient_comparison = compare_tensor_tuples(backward_gradients, manual_gradients)
        optimizer.step(); scheduler.step(1.0)
        manual_parameters = tuple(parameter.detach().clone() for parameter in parameters)
        parameter_comparison = compare_tensor_tuples(backward_parameters, manual_parameters)
        optimizer_comparison = _optimizer_state_difference(backward_optimizer, optimizer.state_dict())
        scheduler_match = backward_scheduler == scheduler.state_dict()
        passed = (gradient_comparison["cosine"] >= .999999 and
                  (gradient_comparison["max_abs_difference"] <= 1e-6 or gradient_comparison["max_relative_difference"] <= 1e-5) and
                  parameter_comparison["cosine"] >= .999999 and
                  (parameter_comparison["max_abs_difference"] <= 1e-6 or parameter_comparison["max_relative_difference"] <= 1e-5) and
                  optimizer_comparison["optimizer_state_mismatch_count"] == 0 and scheduler_match)
        return {"Passed": bool(passed), "ParameterNameCount": len(names),
                "GradientComparison": gradient_comparison, "ParameterComparison": parameter_comparison,
                "OptimizerComparison": optimizer_comparison, "SchedulerStateEqual": scheduler_match}
    finally:
        student.load_state_dict(state, strict=True); optimizer.load_state_dict(optimizer_state); scheduler.load_state_dict(scheduler_state)
        optimizer.zero_grad(set_to_none=True); student.train(training); restore_rng_state(rng)


def prediction_rows(model, loader, device):
    model.eval(); rows = []
    with torch.no_grad():
        for batch in loader:
            text, audio, vision, labels = batch_to_device(batch, device); predictions = {}
            for mode in ("LAV",) + MISSING_MODES:
                mask = mode_to_mask(mode, labels.size(0), device, audio.dtype)
                predictions[mode] = model(text, audio, vision, mask)["output_logit"].view(-1).cpu().numpy()
            for offset, index in enumerate(batch["index"].view(-1).cpu().numpy().astype(int)):
                rows.append({"sample_id": str(list(batch["id"])[offset]), "sample_index": int(index),
                             "label": float(labels[offset]), **{"{}_pred".format(mode): float(value[offset]) for mode, value in predictions.items()}})
    return pd.DataFrame(rows).sort_values("sample_index", kind="mergesort")


def _flatten(metrics, prefix):
    flat = flatten_mode_metrics(metrics)
    for key in next(iter(metrics.values())):
        flat["MissingMacro_{}".format(key)] = float(np.mean([metrics[mode][key] for mode in MISSING_MODES]))
    return {"{}_{}".format(prefix, key): value for key, value in flat.items()}


def representation_audit(model, loader, args, max_batches=None):
    captured = []; representations = {mode: [] for mode in ("LAV",) + MISSING_MODES}
    def hook(module, inputs): captured.append(inputs[0].detach().cpu().numpy())
    handle = model.backbone.proj1.register_forward_pre_hook(hook)
    with protected_audit_state(model):
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches: break
                text, audio, vision, labels = batch_to_device(batch, args.device)
                for mode in ("LAV",) + MISSING_MODES:
                    before = len(captured); mask = mode_to_mask(mode, labels.size(0), args.device, audio.dtype)
                    model(text, audio, vision, mask)
                    if len(captured) != before + 1: raise RuntimeError("Representation hook count changed.")
                    representations[mode].append(captured[-1])
    handle.remove(); arrays = {mode: np.concatenate(values) for mode, values in representations.items()}
    pair_frame, summary = representation_statistics(arrays)
    return {"mode_summary": summary, "cross_mode": pair_frame.to_dict(orient="records")}


def gradient_probe(model, teacher, loader, args, cache_frame, cache_by_index, policy, parameters, max_batches=4):
    criterion, cosine, hinge = nn.L1Loss(), nn.CosineEmbeddingLoss(), HingeLoss()
    quartiles = assign_compatibility_quartiles(cache_frame); records = []
    with protected_audit_state(model, teacher):
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches: break
            text, audio, vision, labels = batch_to_device(batch, args.device)
            indices = batch["index"].view(-1).cpu().numpy().astype(int).tolist()
            full_mask = mode_to_mask("LAV", labels.size(0), args.device, audio.dtype)
            full_output = model(text, audio, vision, full_mask)
            teacher_prediction = teacher_lav_prediction(teacher, text, audio, vision)
            for mode in MISSING_MODES:
                mask = mode_to_mask(mode, labels.size(0), args.device, audio.dtype)
                missing_output = model(text, audio, vision, mask)
                scopes = [("all", None, torch.ones(labels.size(0), device=args.device, dtype=torch.bool))]
                for quartile in ("Q1_low", "Q2", "Q3", "Q4_high"):
                    flags = torch.tensor([quartiles[(index, mode)] == quartile for index in indices], device=args.device)
                    if int(flags.sum()): scopes.append(("quartile", quartile, flags))
                for scope, quartile, flags in scopes:
                    missing_loss, _ = compute_task_loss(_subset_output(missing_output, flags), labels[flags], criterion)
                    if scope == "all":
                        full_loss, _ = compute_full_dlf_loss(full_output, labels, criterion, cosine, hinge)
                        task_loss = full_loss + missing_loss
                    else:
                        # Stage 6A's verified quartile definition is the
                        # subset missing-task gradient; full DLF intermediates
                        # do not share a single batch-axis layout.
                        task_loss = missing_loss
                    compatibility = compatibility_for_modes(cache_by_index, [index for index, flag in zip(indices, flags.tolist()) if flag],
                                                            [mode] * int(flags.sum()), args.device, labels.dtype)
                    kd_loss, _ = gated_kd_loss(missing_output["output_logit"][flags], teacher_prediction[flags], compatibility)
                    task = ordered_autograd(task_loss, parameters, retain_graph=True)
                    raw = ordered_autograd(kd_loss, parameters, retain_graph=True)
                    _, _, metrics = combine_task_kd_gradients(parameters, task, raw, policy)
                    records.append({"Batch": batch_index, "Scope": scope, "Mode": mode, "Quartile": quartile,
                                    "SampleCount": int(flags.sum()), **metrics})
    frame = pd.DataFrame(records); rows = []
    for keys, local in frame.groupby(["Scope", "Mode", "Quartile"], dropna=False):
        rows.append({"Scope": keys[0], "Mode": keys[1], "Quartile": keys[2], "BatchCount": len(local),
                     "SampleCount": int(local.SampleCount.sum()), "ConflictFraction": float(local.ConflictFlag.mean()),
                     "RawCosineMean": float(local.RawCosine.mean()), "RawCosineMedian": float(local.RawCosine.median()),
                     "UsedCosineMean": float(local.ProjectedCosine.mean()), "UsedCosineMedian": float(local.ProjectedCosine.median()),
                     "RemovedFractionMean": float(local.RemovedFraction.mean()),
                     "TaskGradNormMean": float(local.GlobalTaskGradNorm.mean()),
                     "KDRawGradNormMean": float(local.GlobalKDRawGradNorm.mean()),
                     "KDUsedGradNormMean": float(local.GlobalKDUsedGradNorm.mean())})
    return pd.DataFrame(rows)


def _epoch_gradient_summary(step_frame, epoch, losses, j_valid, j_test, best_valid, best_test):
    local = step_frame[step_frame.Epoch.eq(epoch)]; raw = finite_quantiles(local.RawCosine); used = finite_quantiles(local.ProjectedCosine)
    removed = finite_quantiles(local.RemovedFraction); retained = finite_quantiles(local.KDRetainedNormRatio)
    coefficients = local.ProjectionCoefficient[np.isfinite(local.ProjectionCoefficient)]
    descent = local.ActualStepTaskDescent[np.isfinite(local.ActualStepTaskDescent)]
    return {"Seed": 1111, "Epoch": epoch, "Method": local.Method.iloc[0], "GradientPolicy": local.GradientPolicy.iloc[0],
            "StepCount": len(local), "ConflictCount": int(local.ConflictFlag.sum()), "ConflictFraction": float(local.ConflictFlag.mean()),
            "TaskGradNormMean": float(local.GlobalTaskGradNorm.mean()), "KDRawGradNormMean": float(local.GlobalKDRawGradNorm.mean()),
            "KDUsedGradNormMean": float(local.GlobalKDUsedGradNorm.mean()), "TotalGradNormMean": float(local.GlobalTotalGradNorm.mean()),
            **{"RawCosine{}".format(key): value for key, value in raw.items()},
            **{"UsedCosine{}".format(key): value for key, value in used.items()},
            "RemovedFractionMean": removed["Mean"], "RemovedFractionMedian": removed["Median"], "RemovedFractionP90": removed["P90"],
            "KDRetainedNormRatioMean": retained["Mean"], "KDRetainedNormRatioMedian": retained["Median"],
            "ProjectionCoefficientMean": float(coefficients.mean()) if len(coefficients) else 0.0,
            "ProjectionCoefficientMin": float(coefficients.min()) if len(coefficients) else 0.0,
            "ActualStepTaskDescentMean": float(descent.mean()) if len(descent) else float("nan"),
            "ActualStepTaskDescentPositiveFraction": float((descent > 0).mean()) if len(descent) else float("nan"),
            "FullLoss": losses[0], "MissingLoss": losses[1], "KDLoss": losses[2], "J_valid": j_valid, "J_test": j_test,
            "IsBestValid": best_valid, "IsBestTestDiagnostic": best_test}


def _manual_gate_source(cli):
    return Path(cli.result_root) / "missing_baseline" / "cfcompat_manual_replay_v1" / "benchmark_train" / "mosi_per_seed.csv"


def assert_manual_formal_gate(cli):
    source = _manual_gate_source(cli)
    if not source.is_file(): raise RuntimeError("Manual Replay formal gate artifact is absent: {}".format(source))
    row = pd.read_csv(source).iloc[0]
    numeric = ("J_valid", "J_test_at_valid_best", "test_at_valid_best_LAV_MAE", "test_at_valid_best_MissingMacro_MAE")
    if int(row.BestValidEpoch) != 9 or any(abs(float(row[key]) - EXPECTED_MANUAL[key]) > 1e-4 for key in numeric):
        raise RuntimeError("IMPLEMENTATION REPLAY FAILURE: formal Manual Replay result did not reproduce Stage 3.")


def train_one_seed(cli, seed, logger):
    setup_seed(seed); args = build_config(cli, seed)
    evaluator_checkpoint, _, _ = locate_stage1_evaluator(cli.result_root, cli.dataset, seed)
    evaluator_sha = checkpoint_sha256(evaluator_checkpoint)
    cache_frame, cache_by_index, _, _ = load_locked_audit_cache(
        cli.result_root, cli.dataset, evaluator_sha)
    if len(cache_frame) != 1284: raise RuntimeError("Stage 6B requires the locked 1284-sample train cache.")
    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"}: raise RuntimeError("Training must construct only train/valid through MMDataLoader.")
    test_loader = build_single_split_loader(args, "test", cli.num_workers)
    audit_loader = build_single_split_loader(args, "train", cli.num_workers)
    teacher, student, init_checkpoint, init_sha = initialize_teacher_student(args, cli, seed, loaders["valid"])
    names, parameters = trainable_named_parameters(student)
    group_names, group_parameters, group_indices, _, group_manifest = build_parameter_groups(student)
    if names != group_names or [id(value) for value in parameters] != [id(value) for value in group_parameters]:
        raise RuntimeError("Stage 6A parameter manifest order changed.")
    optimizer = optim.Adam(student.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=args.patience)
    assert_teacher_not_in_optimizer(teacher, optimizer)
    criterion, cosine, hinge = nn.L1Loss(), nn.CosineEmbeddingLoss(), HingeLoss()
    missing_generator = torch.Generator().manual_seed(seed + 104729)
    result_dir, main_template, diagnostic_template = method_paths(cli, cli.dataset)
    result_dir.mkdir(parents=True, exist_ok=True)
    main_checkpoint = Path(str(main_template).format(seed)); diagnostic_checkpoint = Path(str(diagnostic_template).format(seed))
    main_checkpoint.parent.mkdir(parents=True, exist_ok=True); diagnostic_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    method = METHODS[cli.gradient_policy][0]

    first_batch = next(iter(audit_loader)); first_labels = first_batch["labels"]["M"].view(-1, 1)
    gate_generator = torch.Generator().manual_seed(seed + 104729)
    gate_mask = sample_missing_masks(first_labels.size(0), gate_generator, args.device, first_batch["audio"].dtype)
    equivalence = run_manual_equivalence_gate(student, teacher, first_batch, gate_mask, cache_by_index, args, optimizer, scheduler)
    (result_dir / "mosi_manual_replay_equivalence.json").write_text(json.dumps(equivalence, indent=2, sort_keys=True) + "\n")
    if not equivalence["Passed"]: raise RuntimeError("IMPLEMENTATION REPLAY FAILURE: single-batch/step gate failed.")

    probe_init = gradient_probe(student, teacher, audit_loader, args, cache_frame, cache_by_index, cli.gradient_policy, parameters)
    probe_init.to_csv(result_dir / "mosi_gradient_probe_init.csv", index=False)
    rep_limit = 4 if cli.smoke_test else None
    (result_dir / "mosi_representation_init.json").write_text(json.dumps(representation_audit(student, audit_loader, args, rep_limit), indent=2, sort_keys=True) + "\n")

    best_valid_j = best_test_j = float("inf"); best_valid_epoch = best_test_epoch = 0
    step_rows = []; group_rows = []; epoch_rows = []; epoch_gradient_rows = []
    digest = MissingSequenceDigest(); final_epoch = 0
    for epoch in range(1, (cli.max_epochs or 1000) + 1):
        final_epoch = epoch; student.train(); optimizer.zero_grad(set_to_none=True)
        counts = Counter({"LA": 0, "LV": 0, "L": 0}); losses = [[], [], []]; first_update_done = False
        accumulated_task = tuple(None for _ in parameters)
        for step, batch in enumerate(loaders["train"], 1):
            mask = sample_missing_masks(batch["labels"]["M"].size(0), missing_generator, args.device, batch["audio"].dtype)
            if epoch == 1: digest.update(mask)
            counts.update(count_missing_modes(mask))
            full_loss, missing_loss, kd_loss, _, _, modes, _ = _batch_losses(student, teacher, batch, mask, cache_by_index, args, criterion, cosine, hinge)
            task_gradients = ordered_autograd(full_loss + missing_loss, parameters, retain_graph=True)
            kd_gradients = ordered_autograd(kd_loss, parameters, retain_graph=True)
            reference_total = ordered_autograd(full_loss + missing_loss + kd_loss, parameters, retain_graph=False)
            used_gradients, _, metrics = combine_task_kd_gradients(parameters, task_gradients, kd_gradients, cli.gradient_policy)
            total_gradients = replay_corrected_total(reference_total, task_gradients, kd_gradients, used_gradients)
            metrics["GlobalTotalGradNorm"] = gradient_norm(total_gradients)
            write_parameter_gradients(parameters, total_gradients, accumulate=True)
            if not first_update_done: accumulated_task = add_gradient_tuples(accumulated_task, task_gradients)
            actual_descent = float("nan"); update_boundary = step % args.update_epochs == 0 or step == len(loaders["train"])
            if update_boundary:
                before = tuple(parameter.detach().clone() for parameter in parameters) if not first_update_done else None
                optimizer.step()
                if not first_update_done:
                    delta = tuple(parameter.detach() - previous for parameter, previous in zip(parameters, before))
                    actual_descent = -gradient_dot(accumulated_task, delta); first_update_done = True
                    accumulated_task = tuple(None for _ in parameters)
                optimizer.zero_grad(set_to_none=True)
            if teacher_grad_count(teacher): raise RuntimeError("Frozen Full Teacher received gradients.")
            losses[0].append(float(full_loss.detach())); losses[1].append(float(missing_loss.detach())); losses[2].append(float(kd_loss.detach()))
            step_row = {"Seed": seed, "Epoch": epoch, "Step": step, "Method": method, "GradientPolicy": cli.gradient_policy,
                        "MissingLA": modes.count("LA"), "MissingLV": modes.count("LV"), "MissingL": modes.count("L"),
                        "OptimizerStep": update_boundary, "ActualStepTaskDescent": actual_descent, **metrics}
            step_rows.append(step_row)
            for local in group_gradient_metrics(task_gradients, kd_gradients, used_gradients, parameters, group_indices):
                group_rows.append({"Seed": seed, "Epoch": epoch, "Step": step, "Method": method,
                                   "GradientPolicy": cli.gradient_policy, **local})
            del task_gradients, kd_gradients, reference_total, used_gradients, total_gradients
        if epoch == 1 and counts != Counter({"LA": 435, "LV": 430, "L": 419}):
            raise RuntimeError("Epoch1 missing counts must be LA=435 LV=430 L=419.")
        valid = evaluate_all_modes(student, loaders["valid"], args.device, "moddrop", criterion)
        test = evaluate_all_modes(student, test_loader, args.device, "moddrop", criterion)
        j_valid, j_test = validation_objective(valid), validation_objective(test); scheduler.step(j_valid)
        is_best_valid = j_valid <= best_valid_j - 1e-6; is_best_test = j_test <= best_test_j - 1e-6
        if is_best_valid: best_valid_j, best_valid_epoch = j_valid, epoch; torch.save(student.state_dict(), main_checkpoint)
        if is_best_test: best_test_j, best_test_epoch = j_test, epoch; torch.save(student.state_dict(), diagnostic_checkpoint)
        mean_losses = tuple(float(np.mean(values)) for values in losses)
        local_steps = pd.DataFrame(step_rows)
        epoch_gradient_rows.append(_epoch_gradient_summary(local_steps, epoch, mean_losses, j_valid, j_test, is_best_valid, is_best_test))
        epoch_rows.append({"Seed": seed, "Epoch": epoch, "Method": method, "GradientPolicy": cli.gradient_policy,
                           "J_valid": j_valid, "J_test": j_test, "IsBestValid": is_best_valid,
                           "IsBestTestDiagnostic": is_best_test, "FullLoss": mean_losses[0],
                           "MissingLoss": mean_losses[1], "KDLoss": mean_losses[2],
                           **_flatten(valid, "valid"), **_flatten(test, "test")})
        logger.info("epoch=%s LA=%s LV=%s L=%s J_valid=%.9f J_test=%.9f conflict=%.6f", epoch, counts["LA"], counts["LV"], counts["L"], j_valid, j_test, epoch_gradient_rows[-1]["ConflictFraction"])
        if epoch - best_valid_epoch >= args.early_stop: break

    final_state = clone_state_dict(student)
    probe_final = gradient_probe(student, teacher, audit_loader, args, cache_frame, cache_by_index, cli.gradient_policy, parameters)
    probe_final.to_csv(result_dir / "mosi_gradient_probe_final.csv", index=False)
    (result_dir / "mosi_representation_final.json").write_text(json.dumps(representation_audit(student, audit_loader, args, rep_limit), indent=2, sort_keys=True) + "\n")
    student.load_state_dict(torch.load(main_checkpoint, map_location=args.device), strict=True)
    probe_best = gradient_probe(student, teacher, audit_loader, args, cache_frame, cache_by_index, cli.gradient_policy, parameters)
    probe_best.to_csv(result_dir / "mosi_gradient_probe_best_valid.csv", index=False)
    (result_dir / "mosi_representation_best_valid.json").write_text(json.dumps(representation_audit(student, audit_loader, args, rep_limit), indent=2, sort_keys=True) + "\n")
    final_valid = evaluate_all_modes(student, loaders["valid"], args.device, "moddrop", criterion)
    final_test = evaluate_all_modes(student, test_loader, args.device, "moddrop", criterion)
    valid_predictions = prediction_rows(student, loaders["valid"], args.device)
    student.load_state_dict(torch.load(diagnostic_checkpoint, map_location=args.device), strict=True)
    diagnostic_test = evaluate_all_modes(student, test_loader, args.device, "moddrop", criterion)
    diagnostic_predictions = prediction_rows(student, test_loader, args.device)
    diagnostic_predictions["selected_by"] = "test"; diagnostic_predictions["diagnostic_only"] = True; diagnostic_predictions["not_main_result"] = True
    student.load_state_dict(final_state, strict=True)

    step_frame = pd.DataFrame(step_rows); group_frame = pd.DataFrame(group_rows)
    group_summary = group_frame.groupby(["Seed", "Epoch", "Method", "GradientPolicy", "ParameterGroup"], as_index=False).agg(
        StepCount=("Step", "count"), TaskGradNormMean=("TaskGradNorm", "mean"), RawKDGradNormMean=("RawKDGradNorm", "mean"),
        UsedKDGradNormMean=("UsedKDGradNorm", "mean"), RawTaskKDCosineMean=("RawTaskKDCosine", "mean"),
        UsedTaskKDCosineMean=("UsedTaskKDCosine", "mean"), RemovedNormFractionMean=("RemovedNormFraction", "mean"))
    result = {"Seed": seed, "Method": method, "GradientPolicy": cli.gradient_policy, "BestValidEpoch": best_valid_epoch,
              "J_valid": validation_objective(final_valid), "J_test_at_valid_best": validation_objective(final_test),
              "BestObservedTestEpoch": best_test_epoch, "BestObservedTestJ": best_test_j,
              "SelectionRegret": validation_objective(final_test) - best_test_j,
              "MainCheckpoint": str(main_checkpoint), "MainCheckpointSHA256": checkpoint_sha256(main_checkpoint),
              "DiagnosticCheckpoint": str(diagnostic_checkpoint), "TeacherCheckpoint": str(init_checkpoint), "TeacherSHA256": init_sha,
              "StudentInitCheckpoint": str(init_checkpoint), "StudentInitSHA256": init_sha,
              "CompatibilityCache": str(Path(cli.result_root) / "counterfactual_compatibility" / CACHE_VERSION / cli.dataset / "train_counterfactual_compatibility.csv"),
              "CompatibilityCacheSHA256": checkpoint_sha256(Path(cli.result_root) / "counterfactual_compatibility" / CACHE_VERSION / cli.dataset / "train_counterfactual_compatibility.csv"),
              "MissingSequenceSHA256": digest.hexdigest(), "TotalSteps": len(step_frame),
              "ConflictStepCount": int(step_frame.ConflictFlag.sum()), "ConflictStepFraction": float(step_frame.ConflictFlag.mean()),
              "MeanRawCosine": float(step_frame.RawCosine.mean()), "MedianRawCosine": float(step_frame.RawCosine.median()),
              "MeanUsedCosine": float(step_frame.ProjectedCosine.mean()), "MedianUsedCosine": float(step_frame.ProjectedCosine.median()),
              "MeanRemovedFraction": float(step_frame.RemovedFraction.mean()), "MedianRemovedFraction": float(step_frame.RemovedFraction.median()),
              "MeanKDRetainedNormRatio": float(step_frame.KDRetainedNormRatio.mean()),
              "ActualTaskDescentFraction": float((step_frame.ActualStepTaskDescent.dropna() > 0).mean()),
              "TeacherGradientCount": teacher_grad_count(teacher), "TrainableParameterCount": len(parameters),
              "TrainableNumel": group_manifest["trainable_numel"], "FinalEpoch": final_epoch,
              **_flatten(final_valid, "valid"), **_flatten(final_test, "test_at_valid_best"), **_flatten(diagnostic_test, "test_diagnostic")}
    return result, pd.DataFrame(epoch_rows), step_frame, pd.DataFrame(epoch_gradient_rows), group_summary, valid_predictions, diagnostic_predictions


def main():
    cli = parse_args(); logger, log_path = create_logger(cli)
    if not cli.smoke_test and cli.gradient_policy != "manual_replay": assert_manual_formal_gate(cli)
    rows = []
    for seed in cli.seeds:
        result, epochs, steps, gradient_epochs, groups, valid_predictions, diagnostic_predictions = train_one_seed(cli, seed, logger)
        rows.append(result); result_dir, _, _ = method_paths(cli, cli.dataset)
        epochs.to_csv(result_dir / "mosi_epoch_metrics.csv", index=False)
        steps.to_csv(result_dir / "mosi_gradient_step_metrics.csv", index=False)
        gradient_epochs.to_csv(result_dir / "mosi_gradient_epoch_summary.csv", index=False)
        groups.to_csv(result_dir / "mosi_gradient_group_summary.csv", index=False)
        valid_predictions.to_csv(result_dir / "mosi_best_valid_predictions.csv", index=False)
        diagnostic_predictions.to_csv(result_dir / "mosi_best_test_diagnostic_predictions.csv", index=False)
    result_dir, _, _ = method_paths(cli, cli.dataset); write_result_csvs(rows, result_dir, cli.dataset)
    if not cli.smoke_test and cli.gradient_policy == "manual_replay":
        row = rows[0]
        numeric = ("J_valid", "J_test_at_valid_best", "test_at_valid_best_LAV_MAE", "test_at_valid_best_MissingMacro_MAE")
        passed = int(row["BestValidEpoch"]) == 9 and all(abs(row[key] - EXPECTED_MANUAL[key]) <= 1e-4 for key in numeric)
        gate = {"Passed": passed, "Expected": EXPECTED_MANUAL, "Observed": {key: row[key] for key in EXPECTED_MANUAL}, "Classification": "PASS" if passed else "IMPLEMENTATION REPLAY FAILURE"}
        (result_dir / "mosi_formal_replay_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
        if not passed: raise RuntimeError("IMPLEMENTATION REPLAY FAILURE")
    logger.info("complete results=%s log=%s", result_dir, log_path)


if __name__ == "__main__":
    main()
