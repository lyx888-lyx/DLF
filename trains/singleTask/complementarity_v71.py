"""V7.1 committee fitting and diagnostics for historical Hybrid replay.

This file restores the historical V7.1 committee math on the analysis branch.
The only adaptation is that ``soft_region_membership`` is inlined from the
historical ``CPFD_DLF.py`` instead of importing the full V7.1 student model
stack, which is not needed for committee-only reconstruction.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Sequence, Tuple

import torch


REGION_CENTERS = (-2.25, -1.0, 0.0, 1.0, 2.25)
REGION_NAMES = (
    "strong_negative",
    "negative",
    "boundary",
    "positive",
    "strong_positive",
)


def soft_region_membership(
    prediction: torch.Tensor,
    temperature: float = 0.55,
) -> torch.Tensor:
    """Exact historical V7.1 deployable soft sentiment-region membership."""
    centers = prediction.new_tensor(REGION_CENTERS).view(1, -1)
    distance = torch.abs(prediction.view(-1, 1) - centers)
    return torch.softmax(-distance / max(float(temperature), 1e-6), dim=1)


def group_ids(sample_ids: Sequence[object], groups: int = 3) -> torch.Tensor:
    return torch.tensor([
        int(hashlib.sha1(str(value).encode("utf-8")).hexdigest(), 16) % groups
        for value in sample_ids
    ], dtype=torch.long)


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
        objective = overall + 0.35 * stability + 0.01 * kl
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
            + 0.35 * stability
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
    """Exact historical V7.1 3-fold committee fit."""
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
