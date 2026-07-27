"""Losses and diagnostics for function-space consensus distillation V7."""

from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn.functional as F


REGION_NAMES = (
    "strong_negative",
    "negative",
    "boundary",
    "positive",
    "strong_positive",
)
ORDINAL_THRESHOLDS = (-1.5, -0.5, 0.5, 1.5)


def region_index(labels: torch.Tensor) -> torch.Tensor:
    values = labels.view(-1)
    result = torch.full_like(values, 2, dtype=torch.long)
    result[values < -1.5] = 0
    result[(values >= -1.5) & (values < -0.5)] = 1
    result[(values > 0.5) & (values <= 1.5)] = 3
    result[values > 1.5] = 4
    return result


def safe_corr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    x = prediction.view(-1).double()
    y = target.view(-1).double()
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.sqrt(x.square().sum() * y.square().sum()).clamp_min(1e-12)
    return float((x * y).sum().div(denominator).item())


def balanced_sign_accuracy(prediction: torch.Tensor, target: torch.Tensor) -> float:
    predicted = prediction.view(-1) >= 0
    truth = target.view(-1) >= 0
    recalls = []
    for value in (False, True):
        mask = truth == value
        if mask.any():
            recalls.append((predicted[mask] == truth[mask]).float().mean())
    return float(torch.stack(recalls).mean().item()) if recalls else 0.0


def _region_balanced_mae(prediction, labels):
    regions = region_index(labels)
    terms = []
    for index in range(len(REGION_NAMES)):
        mask = regions == index
        if mask.any():
            terms.append(torch.abs(prediction[mask] - labels[mask]).mean())
    return torch.stack(terms).mean() if terms else prediction.new_zeros(())


def _region_bias(prediction, labels):
    regions = region_index(labels)
    terms = []
    for index in range(len(REGION_NAMES)):
        mask = regions == index
        if mask.any():
            terms.append((labels[mask] - prediction[mask]).mean().square())
    return torch.stack(terms).mean() if terms else prediction.new_zeros(())


def _ordinal_loss(logits, labels):
    thresholds = logits.new_tensor(ORDINAL_THRESHOLDS).view(1, -1)
    target = (labels > thresholds).float()
    return F.binary_cross_entropy_with_logits(logits, target)


def _pairwise_function_loss(student_prediction, teacher_prediction, agreement):
    student_delta = student_prediction - student_prediction.transpose(0, 1)
    teacher_delta = teacher_prediction - teacher_prediction.transpose(0, 1)
    pair_weight = agreement @ agreement.transpose(0, 1)
    eye = torch.eye(len(student_prediction), device=student_prediction.device)
    pair_weight = pair_weight * (1.0 - eye)
    error = F.smooth_l1_loss(student_delta, teacher_delta, reduction="none")
    return (pair_weight * error).sum() / pair_weight.sum().clamp_min(1.0)


def _kernel_loss(student_feature, teacher_kernel, kernel_variance):
    student_feature = F.normalize(student_feature, dim=1)
    student_kernel = student_feature @ student_feature.transpose(0, 1)
    confidence = torch.exp(-kernel_variance).clamp_min(0.05)
    pair_weight = confidence @ confidence.transpose(0, 1)
    eye = torch.eye(len(student_feature), device=student_feature.device)
    pair_weight = pair_weight * (1.0 - eye)
    error = (student_kernel - teacher_kernel).square()
    return (pair_weight * error).sum() / pair_weight.sum().clamp_min(1.0)


def consensus_distillation_loss(
    student: Mapping[str, torch.Tensor],
    committee: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    weights: Mapping[str, float],
) -> Dict[str, torch.Tensor]:
    prediction = student["prediction"]
    base = student["base_prediction"]
    consensus = committee["consensus"].detach()
    agreement = committee["agreement"].detach()

    base_error = torch.abs(base.detach() - labels)
    teacher_error = torch.abs(consensus - labels)
    advantage = (base_error - teacher_error).detach()
    soft_advantage = 0.50 + torch.sigmoid(advantage / 0.08)
    transfer_weight = (agreement * soft_advantage).clamp(0.10, 3.0)

    supervised = F.l1_loss(prediction, labels)
    base_supervised = F.l1_loss(base, labels)
    distill_error = F.smooth_l1_loss(prediction, consensus, reduction="none")
    distill = (transfer_weight * distill_error).mean()
    pairwise = _pairwise_function_loss(prediction, consensus, agreement)
    kernel = _kernel_loss(
        student["feature"], committee["kernel"], committee["kernel_variance"]
    )
    ordinal = _ordinal_loss(student["ordinal_logits"], labels)
    target_scale = (
        torch.abs(prediction.detach() - labels)
        + committee["dispersion"].detach()
    ).clamp_min(1e-4)
    predicted_scale = F.softplus(student["log_scale"])
    uncertainty = F.smooth_l1_loss(predicted_scale, target_scale)
    region = _region_balanced_mae(prediction, labels)
    bias = _region_bias(prediction, labels)
    disagreement_shrink = (
        student["correction"].abs() * committee["dispersion"].detach()
    ).mean()
    correction_shrink = student["correction"].abs().mean()

    losses = {
        "supervised": supervised,
        "base_supervised": base_supervised,
        "distill": distill,
        "pairwise": pairwise,
        "kernel": kernel,
        "ordinal": ordinal,
        "uncertainty": uncertainty,
        "region": region,
        "bias": bias,
        "disagreement_shrink": disagreement_shrink,
        "correction_shrink": correction_shrink,
    }
    total = prediction.new_zeros(())
    for name, value in losses.items():
        total = total + float(weights.get(name, 0.0)) * value
    losses["total"] = total
    return losses


def prediction_diagnostics(student, committee, labels):
    residual_target = labels - student["base_prediction"]
    residual_prediction = student["prediction"] - student["base_prediction"]
    teacher_gain = (
        torch.abs(student["base_prediction"] - labels)
        - torch.abs(committee["consensus"] - labels)
    )
    return {
        "mae": float(torch.abs(student["prediction"] - labels).mean().item()),
        "base_mae": float(torch.abs(student["base_prediction"] - labels).mean().item()),
        "teacher_consensus_mae": float(
            torch.abs(committee["consensus"] - labels).mean().item()
        ),
        "residual_corr": safe_corr(residual_prediction, residual_target),
        "sign_balanced_accuracy": balanced_sign_accuracy(
            residual_prediction, residual_target
        ),
        "teacher_gain": float(teacher_gain.mean().item()),
        "teacher_dispersion": float(committee["dispersion"].mean().item()),
        "predicted_scale_error_corr": safe_corr(
            F.softplus(student["log_scale"]),
            torch.abs(student["prediction"] - labels),
        ),
    }
