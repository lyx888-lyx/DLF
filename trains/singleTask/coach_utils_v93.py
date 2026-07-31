"""Shared metrics, fold construction, and serialization helpers for V9.3."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from .model.OrdinalAdvantageCoachV93 import REGION_NAMES, region_index
from .oof_group_splits_v92 import build_nested_group_folds, conversation_group_id


def cpu_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def safe_metrics(metrics_fn, prediction, labels) -> Dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics_fn(
            prediction.detach().cpu(), labels.detach().cpu()
        ).items()
    }


def region_metrics(probabilities: torch.Tensor, labels: torch.Tensor):
    target = region_index(labels).cpu()
    predicted = probabilities.argmax(dim=1).cpu()
    precision_values = []
    recall_values = []
    f1_values = []
    for category in range(len(REGION_NAMES)):
        true_positive = int(
            ((predicted == category) & (target == category)).sum().item()
        )
        false_positive = int(
            ((predicted == category) & (target != category)).sum().item()
        )
        false_negative = int(
            ((predicted != category) & (target == category)).sum().item()
        )
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
    expected = (
        probabilities.cpu() * torch.arange(len(REGION_NAMES)).view(1, -1)
    ).sum(dim=1)
    return {
        "accuracy": float((predicted == target).float().mean().item()),
        "macro_precision": float(np.mean(precision_values)),
        "macro_recall": float(np.mean(recall_values)),
        "macro_f1": float(np.mean(f1_values)),
        "ordinal_index_mae": float(
            torch.abs(expected - target.float()).mean().item()
        ),
    }


def gain_diagnostics(
    predicted_gain,
    win_probability,
    realized_gain,
    margin,
):
    prediction = predicted_gain.view(-1).cpu().numpy()
    probability = win_probability.view(-1).cpu().numpy()
    actual = realized_gain.view(-1).cpu().numpy()
    correlation = 0.0
    if (
        len(actual) >= 2
        and np.std(prediction) > 0
        and np.std(actual) > 0
    ):
        correlation = float(np.corrcoef(prediction, actual)[0, 1])
    target = actual > float(margin)
    guessed = probability >= 0.5
    true_positive = int(np.sum(target & guessed))
    false_positive = int(np.sum(~target & guessed))
    false_negative = int(np.sum(target & ~guessed))
    return {
        "gain_mae": float(np.mean(np.abs(prediction - actual))),
        "gain_corr": correlation,
        "win_accuracy": float(np.mean(guessed == target)),
        "win_precision": float(
            true_positive / max(1, true_positive + false_positive)
        ),
        "win_recall": float(
            true_positive / max(1, true_positive + false_negative)
        ),
        "positive_rate": float(np.mean(target)),
    }


def safe_group_folds(sample_ids, labels, requested_folds, seed):
    group_count = len(
        {conversation_group_id(value) for value in sample_ids}
    )
    last_error = None
    for fold_count in range(
        min(int(requested_folds), group_count), 1, -1
    ):
        for fraction in (0.20, 0.25, 0.33):
            try:
                return build_nested_group_folds(
                    sample_ids,
                    labels,
                    outer_folds=fold_count,
                    inner_valid_fraction=fraction,
                    seed=int(seed),
                )
            except (RuntimeError, ValueError) as error:
                last_error = error
    raise RuntimeError(
        f"unable to construct leakage-safe grouped folds: {last_error}"
    )


def json_scalar(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value
