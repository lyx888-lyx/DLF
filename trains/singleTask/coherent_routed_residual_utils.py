"""Shared-budget routing and offline audits for Stage 4A.1 CCRRD."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .cf_compat_kd_utils import modes_from_masks, stable_average_ranks
from .cf_residual_utils import (
    MODE_ORDER,
    load_residual_cache,
    residual_targets_for_modes,
    rng_states_equal,
)
from .fixed_kd_utils import (
    build_frozen_teacher,
    capture_rng_state,
    checkpoint_sha256,
    restore_rng_state,
    teacher_lav_prediction,
)
from .missing_utils import (
    build_single_split_loader,
    clean_checkpoint_path,
    compute_task_loss,
    sample_missing_masks,
)
from .model.DLF import DLF


ROUTE_VARIANTS = {
    "base_shared": {
        "method": "DLF-CFCompatCFRR-BaseShared-v1",
        "version": "cfcompat_cfrr_base_shared_v1",
        "log_tag": "cfrr-base-shared",
    },
    "corrected_joint_shared": {
        "method": "DLF-CFCompatCFRR-CorrectedJointShared-v1",
        "version": "cfcompat_cfrr_corrected_joint_v1",
        "log_tag": "cfrr-corrected-joint",
    },
    "corrected_stopres_shared": {
        "method": "DLF-CCRRD-v1",
        "version": "ccrrd_v1",
        "log_tag": "ccrrd",
    },
}
ALIGNMENT_VERSION = "coherent_routing_v1"


def direct_prediction(output, route_variant):
    """Select the fixed direct-KD value/gradient path for one CCRRD variant."""
    if route_variant == "base_shared":
        return output["base_output_logit"]
    if route_variant == "corrected_joint_shared":
        return output["corrected_output_logit"]
    if route_variant == "corrected_stopres_shared":
        return output["base_output_logit"] + output["predicted_residual"].detach()
    raise ValueError("Unknown coherent route variant: {}".format(route_variant))


def shared_route_loss(output, teacher_target, target_z, compatibility, route_variant):
    """Exactly mean(C*d + (1-C)*u), with no per-channel renormalization."""
    prediction = direct_prediction(output, route_variant).view(-1)
    teacher = teacher_target.detach().view(-1).to(prediction)
    predicted_z = output["predicted_residual_z"].view(-1)
    target = target_z.detach().view(-1).to(predicted_z)
    compat = compatibility.detach().view(-1).to(prediction)
    if not (prediction.shape == teacher.shape == predicted_z.shape == target.shape == compat.shape):
        raise ValueError("CCRRD routed tensors must all have shape [batch].")
    direct_each = F.smooth_l1_loss(prediction, teacher, reduction="none")
    residual_each = F.smooth_l1_loss(predicted_z, target, reduction="none")
    weighted_direct = compat * direct_each
    weighted_residual = (1.0 - compat) * residual_each
    route_each = weighted_direct + weighted_residual
    route_loss = route_each.mean()
    if not torch.isfinite(route_each).all():
        raise FloatingPointError("Non-finite CCRRD shared route loss.")
    torch.testing.assert_close(compat + (1.0 - compat), torch.ones_like(compat), rtol=0, atol=0)
    return {
        "loss": route_loss,
        "route_each": route_each,
        "direct_each": direct_each,
        "residual_each": residual_each,
        "weighted_direct": weighted_direct,
        "weighted_residual": weighted_residual,
        "compatibility": compat,
    }


def missing_sequence_sha256(modes):
    if not modes or any(mode not in MODE_ORDER for mode in modes):
        raise ValueError("Missing sequence must contain only LA/LV/L.")
    return hashlib.sha256("|".join(modes).encode("utf-8")).hexdigest()


def _pearson(first, second):
    first = np.asarray(first, dtype=np.float64); second = np.asarray(second, dtype=np.float64)
    if len(first) < 2 or first.std() == 0 or second.std() == 0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _spearman(first, second):
    return _pearson(stable_average_ranks(first), stable_average_ranks(second))


def distribution(values, include_abs=False):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Finite non-empty values required.")
    q = np.quantile(values, [.1, .25, .5, .75, .9, .95])
    result = {
        "mean": float(values.mean()), "std": float(values.std(ddof=0)),
        "min": float(values.min()), "p10": float(q[0]), "p25": float(q[1]),
        "median": float(q[2]), "p75": float(q[3]), "p90": float(q[4]),
        "p95": float(q[5]), "max": float(values.max()),
    }
    if include_abs:
        result["mean_abs"] = float(np.abs(values).mean())
    return result


def route_diagnostic_rows(records, seed, epoch, method, route_variant):
    frame = pd.DataFrame(records)
    numeric = frame.select_dtypes(include=[np.number]).to_numpy()
    if len(frame) == 0 or not np.isfinite(numeric).all():
        raise ValueError("Finite routed records required.")
    direct_sum = float(frame.weighted_direct.sum())
    residual_sum = float(frame.weighted_residual.sum())
    denominator = direct_sum + residual_sum + 1e-8
    summary = {
        "Seed": seed, "Epoch": epoch, "Method": method, "RouteVariant": route_variant,
        "SampleCount": int(len(frame)), "DirectRawLossMean": float(frame.direct_raw.mean()),
        "ResidualRawLossMean": float(frame.residual_raw.mean()),
        "WeightedDirectContribution": float(frame.weighted_direct.mean()),
        "WeightedResidualContribution": float(frame.weighted_residual.mean()),
        "RouteLoss": float((frame.weighted_direct + frame.weighted_residual).mean()),
        "DirectContributionFraction": direct_sum / denominator,
        "ResidualContributionFraction": residual_sum / denominator,
        "MeanRouteWeightSum": float((frame.compatibility + frame.complement).mean()),
        **{"Compatibility_{}".format(k): v for k, v in distribution(frame.compatibility).items()},
        **{"PredictedResidual_{}".format(k): v for k, v in distribution(frame.predicted_residual, True).items()},
    }
    ordered = frame.sort_values(["compatibility", "sample_index"], kind="mergesort").copy()
    ordered["quartile"] = pd.qcut(np.arange(len(ordered)), 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    quartiles = []
    for quartile, local in ordered.groupby("quartile", observed=False):
        base_error = np.abs(local.base_prediction - local.label)
        corrected_error = np.abs(local.corrected_prediction - local.label)
        row = {
            "Seed": seed, "Epoch": epoch, "Method": method, "RouteVariant": route_variant,
            "Quartile": str(quartile), "Count": int(len(local)),
            "MeanCompatibility": float(local.compatibility.mean()),
            "DirectRawLossMean": float(local.direct_raw.mean()),
            "ResidualRawLossMean": float(local.residual_raw.mean()),
            "WeightedDirectContribution": float(local.weighted_direct.mean()),
            "WeightedResidualContribution": float(local.weighted_residual.mean()),
            "RouteLoss": float((local.weighted_direct + local.weighted_residual).mean()),
            "BaseLabelMAE": float(base_error.mean()), "CorrectedLabelMAE": float(corrected_error.mean()),
            "MeanErrorReduction": float((base_error - corrected_error).mean()),
            "fraction_corrected_better": float((base_error > corrected_error).mean()),
            "fraction_corrected_worse": float((base_error < corrected_error).mean()),
        }
        row.update({"{}_count".format(mode): int(local["mode"].eq(mode).sum()) for mode in MODE_ORDER})
        quartiles.append(row)
    return summary, quartiles


def alignment_paths(result_root, dataset="mosi", seed=1111):
    directory = Path(result_root) / "counterfactual_residual" / ALIGNMENT_VERSION / dataset / "seed{}".format(seed)
    return {"directory": directory, "csv": directory / "teacher_evaluator_alignment.csv",
            "json": directory / "teacher_evaluator_alignment.json"}


def _alignment_pair(first, second):
    first = np.asarray(first, np.float64); second = np.asarray(second, np.float64)
    return {
        "MAE": float(np.mean(np.abs(first - second))),
        "RMSE": float(np.sqrt(np.mean(np.square(first - second)))),
        "Pearson": _pearson(first, second), "Spearman": _spearman(first, second),
        "SignAgreement": float(np.mean(np.sign(first) == np.sign(second))),
    }


def build_teacher_evaluator_alignment(args, result_root, model_save_dir="pt", num_workers=0):
    """Create the diagnostic-only train alignment without valid/test or evaluator forward."""
    before = capture_rng_state()
    paths = alignment_paths(result_root, "mosi", 1111)
    try:
        cache_frame, cache_by_index, _, cache_config = load_residual_cache(result_root, "mosi", 1111)
        gate3 = clean_checkpoint_path(model_save_dir, "mosi", 1111)
        teacher = build_frozen_teacher(DLF, args, gate3)
        loader = build_single_split_loader(args, "train", num_workers)
        rows = []
        for batch in loader:
            text = batch["text"].to(args.device); audio = batch["audio"].to(args.device); vision = batch["vision"].to(args.device)
            teacher_prediction = teacher_lav_prediction(teacher, text, audio, vision).view(-1).cpu().numpy()
            indices = batch["index"].view(-1).cpu().numpy().astype(int)
            identifiers = list(batch["id"]); labels = batch["labels"]["M"].view(-1).cpu().numpy()
            for offset, index in enumerate(indices):
                cached = cache_by_index[int(index)]; teacher_value = float(teacher_prediction[offset])
                row = {"sample_index": int(index), "sample_id": str(identifiers[offset]), "label": float(labels[offset]),
                       "teacher_LAV_pred": teacher_value, "evaluator_LAV_pred": float(cached["evaluator_LAV_pred"]),
                       "teacher_minus_evaluator_LAV": teacher_value - float(cached["evaluator_LAV_pred"])}
                for mode in MODE_ORDER:
                    evaluator = float(cached["evaluator_{}_pred".format(mode)])
                    row["evaluator_{}_pred".format(mode)] = evaluator
                    row["evaluator_residual_{}".format(mode)] = float(cached["residual_{}".format(mode)])
                    row["teacher_needed_residual_{}".format(mode)] = teacher_value - evaluator
                rows.append(row)
        frame = pd.DataFrame(rows).sort_values("sample_index", kind="mergesort").reset_index(drop=True)
        if len(frame) != 1284 or frame.sample_index.nunique() != 1284 or not np.array_equal(frame.sample_index, np.arange(1284)):
            raise RuntimeError("Alignment audit must visit all 1284 train samples exactly once.")
        pair = _alignment_pair(frame.teacher_LAV_pred, frame.evaluator_LAV_pred)
        difference = frame.teacher_minus_evaluator_LAV.to_numpy(float); abs_difference = np.abs(difference)
        q = np.quantile(abs_difference, [.1, .25, .5, .75, .9, .95])
        pair.update({"MeanDifference": float(difference.mean()), "StdDifference": float(difference.std(ddof=0)),
                     "P10AbsDifference": float(q[0]), "P25AbsDifference": float(q[1]), "MedianAbsDifference": float(q[2]),
                     "P75AbsDifference": float(q[3]), "P90AbsDifference": float(q[4]), "P95AbsDifference": float(q[5]),
                     "MaxAbsDifference": float(abs_difference.max())})
        mode_stats = {}
        for mode in MODE_ORDER:
            evaluator_residual = frame["evaluator_residual_{}".format(mode)].to_numpy(float)
            teacher_needed = frame["teacher_needed_residual_{}".format(mode)].to_numpy(float)
            mode_stats[mode] = _alignment_pair(evaluator_residual, teacher_needed)
            mode_stats[mode].update({"MeanAbsEvaluatorResidual": float(np.abs(evaluator_residual).mean()),
                                     "MeanAbsTeacherNeededResidual": float(np.abs(teacher_needed).mean())})
        metadata = {
            "TeacherCheckpoint": str(gate3), "TeacherSHA256": checkpoint_sha256(gate3),
            "EvaluatorCheckpoint": cache_config["EvaluatorCheckpoint"], "EvaluatorSHA256": cache_config["EvaluatorSHA256"],
            "CompatibilityCache": cache_config["SourceCompatibilityCache"],
            "CompatibilityCacheSHA256": cache_config["SourceCompatibilityCacheSHA256"],
            "ResidualCacheSHA256": cache_config["ResidualCacheSHA256"], "TrainSampleCount": 1284,
            "CreatedFromTrainOnly": True, "DiagnosticOnly": True,
            "TeacherEvaluatorLAV": pair, "Modes": mode_stats,
        }
        paths["directory"].mkdir(parents=True, exist_ok=True)
        frame.to_csv(paths["csv"], index=False)
        metadata["AlignmentCSVSHA256"] = checkpoint_sha256(paths["csv"])
        paths["json"].write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    finally:
        restore_rng_state(before)
    after = capture_rng_state()
    if not rng_states_equal(before, after):
        raise RuntimeError("Teacher–Evaluator alignment audit changed RNG state.")
    return paths, metadata, frame


def _parameter_groups(student):
    residual_ids = {id(parameter) for parameter in student.residual_heads.parameters()}
    base_head_ids = {id(parameter) for parameter in student.base_student.backbone.out_layer.parameters()}
    groups = {"shared_backbone": [], "base_prediction_head": [], "residual_heads": []}
    for parameter in student.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in residual_ids:
            groups["residual_heads"].append(parameter)
        elif id(parameter) in base_head_ids:
            groups["base_prediction_head"].append(parameter)
        else:
            groups["shared_backbone"].append(parameter)
    return groups


def _flat_grad(loss, parameters, retain_graph):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    pieces = [torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.detach().reshape(-1)
              for parameter, gradient in zip(parameters, gradients)]
    return torch.cat(pieces) if pieces else torch.zeros(1, device=loss.device)


def _cosine(first, second):
    denominator = float(first.norm() * second.norm())
    return float(torch.dot(first, second) / denominator) if denominator > 0 else 0.0


def gradient_alignment_audit(student, teacher, loader, cache_by_index, scales, device, route_variant, criterion, stage):
    """Read-only four-batch train audit; restores RNG and model mode."""
    before_rng = capture_rng_state(); was_training = student.training
    parameter_snapshot = {name: parameter.detach().clone() for name, parameter in student.named_parameters()}
    groups = _parameter_groups(student)
    accumulators = {component: {group: None for group in groups} for component in ("direct", "residual", "task")}
    generator = torch.Generator().manual_seed(1111 + 104729)
    student.train()
    try:
        for batch_number, batch in enumerate(loader, 1):
            if batch_number > 4:
                break
            text = batch["text"].to(device); audio = batch["audio"].to(device); vision = batch["vision"].to(device)
            labels = batch["labels"]["M"].to(device).view(-1, 1)
            mask = sample_missing_masks(labels.size(0), generator, device, audio.dtype)
            modes = modes_from_masks(mask); indices = batch["index"].view(-1).cpu().numpy().astype(int).tolist()
            output = student(text, audio, vision, mask)
            compatibility, _, target_z = residual_targets_for_modes(cache_by_index, indices, modes, scales, device, labels.dtype)
            teacher_prediction = teacher_lav_prediction(teacher, text, audio, vision)
            routed = shared_route_loss(output, teacher_prediction, target_z, compatibility, route_variant)
            losses = {"direct": routed["weighted_direct"].mean(),
                      "residual": routed["weighted_residual"].mean(),
                      "task": compute_task_loss(output, labels, criterion)[0]}
            for component, loss in losses.items():
                for group_name, parameters in groups.items():
                    flat = _flat_grad(loss, parameters, retain_graph=True)
                    accumulators[component][group_name] = flat if accumulators[component][group_name] is None else accumulators[component][group_name] + flat
        if batch_number < 4:
            raise RuntimeError("Gradient audit requires four train-only batches.")
        rows = []
        for group_name in groups:
            direct = accumulators["direct"][group_name] / 4.0
            residual = accumulators["residual"][group_name] / 4.0
            task = accumulators["task"][group_name] / 4.0
            rows.append({
                "RouteVariant": route_variant, "AuditStage": stage, "ParameterGroup": group_name, "BatchCount": 4,
                "DirectGradNorm": float(direct.norm()), "ResidualGradNorm": float(residual.norm()),
                "TaskGradNorm": float(task.norm()), "DirectResidualCosine": _cosine(direct, residual),
                "DirectTaskCosine": _cosine(direct, task), "ResidualTaskCosine": _cosine(residual, task),
            })
        direct_residual_norm = next(row["DirectGradNorm"] for row in rows if row["ParameterGroup"] == "residual_heads")
        direct_backbone_norm = next(row["DirectGradNorm"] for row in rows if row["ParameterGroup"] == "shared_backbone")
        if route_variant == "corrected_joint_shared" and direct_residual_norm <= 0:
            raise RuntimeError("Joint corrected direct KD must reach residual heads.")
        if route_variant == "corrected_stopres_shared" and direct_residual_norm != 0:
            raise RuntimeError("Stop-residual direct KD gradient must be exactly zero on residual heads.")
        if route_variant == "corrected_stopres_shared" and direct_backbone_norm <= 0:
            raise RuntimeError("Stop-residual direct KD must reach the Student backbone.")
        for name, parameter in student.named_parameters():
            if not torch.equal(parameter.detach(), parameter_snapshot[name]):
                raise RuntimeError("Gradient audit changed model parameters: {}".format(name))
    finally:
        student.train(was_training)
        restore_rng_state(before_rng)
    after_rng = capture_rng_state()
    if not rng_states_equal(before_rng, after_rng):
        raise RuntimeError("Gradient audit changed RNG state.")
    return rows
