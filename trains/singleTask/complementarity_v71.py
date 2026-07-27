"""Committee fitting, losses and diagnostics for V7.1."""

from __future__ import annotations

import hashlib
from typing import Dict, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

from .model.CPFD_DLF import soft_region_membership


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
    denominator = torch.sqrt(
        x.square().sum() * y.square().sum()
    ).clamp_min(1e-12)
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


def group_ids(sample_ids: Sequence[object], groups: int = 3) -> torch.Tensor:
    return torch.tensor([
        int(hashlib.sha1(str(value).encode("utf-8")).hexdigest(), 16) % groups
        for value in sample_ids
    ], dtype=torch.long)


def committee_dispersion(predictions: torch.Tensor) -> torch.Tensor:
    median = predictions.median(dim=1).values
    mad = torch.abs(predictions - median.unsqueeze(1)).median(dim=1).values
    std = predictions.std(dim=1, unbiased=False)
    return mad + 0.5 * std


def apply_global_committee(
    predictions: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    return (predictions * weights.view(1, -1, 1)).sum(dim=1)


def apply_region_committee(
    predictions: torch.Tensor,
    anchor: torch.Tensor,
    region_weights: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    membership = soft_region_membership(anchor, temperature)
    sample_weights = membership @ region_weights
    return (predictions * sample_weights.unsqueeze(-1)).sum(dim=1)


def _fit_global_weights(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    groups: torch.Tensor,
    steps: int = 600,
    lr: float = 0.05,
) -> torch.Tensor:
    teacher_count = predictions.size(1)
    logits = torch.zeros(teacher_count, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=lr)
    uniform = torch.full((teacher_count,), 1.0 / teacher_count)
    best = None

    for _ in range(int(steps)):
        weights = torch.softmax(logits, dim=0)
        prediction = apply_global_committee(predictions, weights)
        overall = torch.abs(prediction - labels).mean()
        group_losses = []
        for group in torch.unique(groups):
            mask = groups == group
            if mask.any():
                group_losses.append(
                    torch.abs(prediction[mask] - labels[mask]).mean()
                )
        stability = (
            torch.stack(group_losses).std(unbiased=False)
            if len(group_losses) > 1
            else overall.new_zeros(())
        )
        kl = (
            weights
            * (weights.clamp_min(1e-8) / uniform).log()
        ).sum()
        objective = overall + 0.30 * stability + 0.01 * kl
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
        row = (float(objective.item()), weights.detach().clone())
        if best is None or row[0] < best[0]:
            best = row
    return best[1]


def _fit_region_weights(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    anchor: torch.Tensor,
    groups: torch.Tensor,
    global_weights: torch.Tensor,
    regularization: float,
    temperature: float,
    steps: int = 700,
    lr: float = 0.04,
) -> torch.Tensor:
    init = global_weights.clamp_min(1e-6).log().repeat(5, 1)
    logits = init.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=lr)
    best = None

    for _ in range(int(steps)):
        weights = torch.softmax(logits, dim=1)
        prediction = apply_region_committee(
            predictions, anchor, weights, temperature
        )
        overall = torch.abs(prediction - labels).mean()
        group_losses = []
        for group in torch.unique(groups):
            mask = groups == group
            if mask.any():
                group_losses.append(
                    torch.abs(prediction[mask] - labels[mask]).mean()
                )
        stability = (
            torch.stack(group_losses).std(unbiased=False)
            if len(group_losses) > 1
            else overall.new_zeros(())
        )
        global_target = global_weights.view(1, -1).expand_as(weights)
        kl = (
            weights
            * (
                weights.clamp_min(1e-8)
                / global_target.clamp_min(1e-8)
            ).log()
        ).sum(dim=1).mean()
        smoothness = (weights[1:] - weights[:-1]).square().mean()
        objective = (
            overall
            + 0.30 * stability
            + float(regularization) * kl
            + 0.05 * smoothness
        )
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
        row = (float(objective.item()), weights.detach().clone())
        if best is None or row[0] < best[0]:
            best = row
    return best[1]


def fit_committee_cv(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    anchor: torch.Tensor,
    sample_ids: Sequence[object],
    temperature: float = 0.55,
    regularizations: Tuple[float, ...] = (0.03, 0.06, 0.12, 0.25, 0.50),
    steps: int = 600,
) -> Dict[str, object]:
    """Fit global and softly region-conditioned simplex weights with 3-fold CV."""
    predictions = predictions.detach().cpu().float()
    labels = labels.detach().cpu().float()
    anchor = anchor.detach().cpu().float()
    groups = group_ids(sample_ids, 3)

    rows = []
    global_fold_maes = []
    region_fold_maes = {float(value): [] for value in regularizations}

    for held_out in range(3):
        train_mask = groups != held_out
        valid_mask = groups == held_out
        global_weights = _fit_global_weights(
            predictions[train_mask],
            labels[train_mask],
            groups[train_mask],
            steps=steps,
        )
        global_prediction = apply_global_committee(
            predictions[valid_mask], global_weights
        )
        global_mae = float(
            torch.abs(global_prediction - labels[valid_mask]).mean().item()
        )
        global_fold_maes.append(global_mae)
        rows.append({
            "kind": "global",
            "regularization": None,
            "fold": held_out,
            "mae": global_mae,
        })

        for regularization in regularizations:
            region_weights = _fit_region_weights(
                predictions[train_mask],
                labels[train_mask],
                anchor[train_mask],
                groups[train_mask],
                global_weights,
                regularization=float(regularization),
                temperature=temperature,
                steps=steps,
            )
            region_prediction = apply_region_committee(
                predictions[valid_mask],
                anchor[valid_mask],
                region_weights,
                temperature,
            )
            region_mae = float(
                torch.abs(region_prediction - labels[valid_mask]).mean().item()
            )
            region_fold_maes[float(regularization)].append(region_mae)
            rows.append({
                "kind": "region",
                "regularization": float(regularization),
                "fold": held_out,
                "mae": region_mae,
            })

    global_tensor = torch.tensor(global_fold_maes)
    global_cv_mean = float(global_tensor.mean().item())
    global_cv_std = float(global_tensor.std(unbiased=False).item())
    global_score = global_cv_mean + 0.25 * global_cv_std

    best_reg = None
    best_region_score = float("inf")
    for regularization, values in region_fold_maes.items():
        tensor = torch.tensor(values)
        score = float(
            tensor.mean().item() + 0.25 * tensor.std(unbiased=False).item()
        )
        if score < best_region_score:
            best_region_score = score
            best_reg = float(regularization)

    global_weights = _fit_global_weights(
        predictions, labels, groups, steps=steps
    )
    region_weights = _fit_region_weights(
        predictions,
        labels,
        anchor,
        groups,
        global_weights,
        regularization=best_reg,
        temperature=temperature,
        steps=steps,
    )
    selected = (
        "region_simplex"
        if best_region_score < global_score - 1e-4
        else "global_simplex"
    )
    return {
        "global_weights": global_weights,
        "region_weights": region_weights,
        "selected": selected,
        "selected_regularization": best_reg,
        "global_cv_score": global_score,
        "region_cv_score": best_region_score,
        "cv_rows": rows,
    }


def _region_balanced_mae(prediction, labels):
    regions = region_index(labels)
    values = []
    for index in range(len(REGION_NAMES)):
        mask = regions == index
        if mask.any():
            values.append(torch.abs(prediction[mask] - labels[mask]).mean())
    return torch.stack(values).mean() if values else prediction.new_zeros(())


def _region_bias(prediction, labels):
    regions = region_index(labels)
    values = []
    for index in range(len(REGION_NAMES)):
        mask = regions == index
        if mask.any():
            values.append((labels[mask] - prediction[mask]).mean().square())
    return torch.stack(values).mean() if values else prediction.new_zeros(())


def _ordinal_loss(logits, labels):
    thresholds = logits.new_tensor(ORDINAL_THRESHOLDS).view(1, -1)
    target = (labels > thresholds).float()
    return F.binary_cross_entropy_with_logits(logits, target)


def _pairwise_residual_loss(student_residual, teacher_residual, confidence):
    student_delta = student_residual - student_residual.transpose(0, 1)
    teacher_delta = teacher_residual - teacher_residual.transpose(0, 1)
    pair_weight = confidence @ confidence.transpose(0, 1)
    eye = torch.eye(len(student_residual), device=student_residual.device)
    pair_weight = pair_weight * (1.0 - eye)
    error = F.smooth_l1_loss(
        student_delta, teacher_delta, reduction="none"
    )
    return (pair_weight * error).sum() / pair_weight.sum().clamp_min(1.0)


def complementarity_distillation_loss(
    student: Mapping[str, torch.Tensor],
    teacher_prediction: torch.Tensor,
    dispersion: torch.Tensor,
    labels: torch.Tensor,
    weights: Mapping[str, float],
    residual_clip: float = 0.35,
    confidence_temperature: float = 0.12,
) -> Dict[str, torch.Tensor]:
    prediction = student["prediction"]
    base = student["base_prediction"]
    correction = student["correction"]
    teacher_residual = (
        teacher_prediction.detach() - base.detach()
    ).clamp(-float(residual_clip), float(residual_clip))

    confidence = torch.exp(
        -dispersion.detach() / max(float(confidence_temperature), 1e-6)
    ).clamp_min(1e-4)
    confidence = confidence / confidence.mean().clamp_min(1e-6)

    supervised = F.l1_loss(prediction, labels)
    base_supervised = F.l1_loss(base, labels)
    residual_error = F.smooth_l1_loss(
        correction, teacher_residual, reduction="none"
    )
    residual_distill = (confidence * residual_error).mean()
    pairwise = _pairwise_residual_loss(
        correction, teacher_residual, confidence
    )
    ordinal = _ordinal_loss(student["ordinal_logits"], labels)
    region = _region_balanced_mae(prediction, labels)
    bias = _region_bias(prediction, labels)
    target_scale = (
        torch.abs(prediction.detach() - labels) + dispersion.detach()
    ).clamp_min(1e-4)
    uncertainty = F.smooth_l1_loss(
        F.softplus(student["log_scale"]), target_scale
    )
    disagreement_shrink = (
        correction.abs() * dispersion.detach()
    ).mean()
    correction_shrink = correction.abs().mean()

    losses = {
        "supervised": supervised,
        "base_supervised": base_supervised,
        "residual_distill": residual_distill,
        "pairwise": pairwise,
        "ordinal": ordinal,
        "region": region,
        "bias": bias,
        "uncertainty": uncertainty,
        "disagreement_shrink": disagreement_shrink,
        "correction_shrink": correction_shrink,
    }
    total = prediction.new_zeros(())
    for name, value in losses.items():
        total = total + float(weights.get(name, 0.0)) * value
    losses["total"] = total
    return losses


def selection_stats(
    anchor: torch.Tensor,
    prediction: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, object]:
    gain = (
        torch.abs(anchor - labels).view(-1)
        - torch.abs(prediction - labels).view(-1)
    )
    selected = torch.abs(prediction - anchor).view(-1) > 1e-8
    count = int(selected.sum().item())
    return {
        "mae": float(torch.abs(prediction - labels).mean().item()),
        "mean_realized_gain": float(gain.mean().item()),
        "correction_count": count,
        "correction_rate": float(selected.float().mean().item()),
        "correction_precision": (
            float((gain[selected] > 0).float().mean().item())
            if count else None
        ),
        "harm_over_005_rate": float((gain < -0.05).float().mean().item()),
        "harm_over_010_rate": float((gain < -0.10).float().mean().item()),
        "mean_abs_change": float(torch.abs(prediction - anchor).mean().item()),
    }
