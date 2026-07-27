import itertools
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .need_direction_v31 import predict_classifier, stratified_indices
from .residual_direction_core import (
    SPECIALIST_NAMES,
    _build_folds,
    _fusion,
    _predict_specialist,
    _regions,
    _train_specialist,
)

logger = logging.getLogger('MMSA')
REGION_NAMES = ('strong_negative', 'negative', 'boundary', 'positive', 'strong_positive')
SCENARIOS = {
    'original': (1.0, 1.0, 1.0, 1.0, 1.0),
    'negative_heavy': (2.5, 1.8, 1.0, 0.8, 0.8),
    'positive_heavy': (0.8, 0.8, 1.0, 1.8, 2.5),
    'boundary_heavy': (1.0, 1.0, 2.5, 1.0, 1.0),
    'strong_heavy': (2.0, 1.0, 1.0, 1.0, 2.0),
}


def cross_fit_specialists_for_views(
    train_cache, valid_views, test_views, device, seed, fold_count, save_dir, **kwargs
):
    folds = _build_folds(
        train_cache['labels'], train_cache['expert_matrix'], fold_count,
        seed, kwargs['residual_margin']
    )
    sample_count = len(train_cache['labels'])
    oof = torch.zeros(sample_count, len(SPECIALIST_NAMES), dtype=train_cache['context'].dtype)
    valid_predictions = [[] for _ in valid_views]
    test_predictions = [[] for _ in test_views]
    history, all_indices = [], torch.arange(sample_count)
    for specialist_fold, holdout in enumerate(folds, start=1):
        mask = torch.ones(sample_count, dtype=torch.bool)
        mask[holdout] = False
        train_indices = all_indices[mask]
        logger.info(
            'V3.1 specialist fold %d/%d train=%d holdout=%d',
            specialist_fold, len(folds), len(train_indices), len(holdout)
        )
        model, fold_history = _train_specialist(
            train_cache['context'][train_indices],
            train_cache['expert_matrix'][train_indices],
            train_cache['labels'][train_indices],
            device=device, seed=seed + 1009 * specialist_fold, **kwargs
        )
        oof[holdout] = _predict_specialist(model, train_cache['context'][holdout], device)
        for view_index, view in enumerate(valid_views):
            valid_predictions[view_index].append(_predict_specialist(model, view['context'], device))
        for view_index, view in enumerate(test_views):
            test_predictions[view_index].append(_predict_specialist(model, view['context'], device))
        history.extend({'specialist_fold': specialist_fold, **row} for row in fold_history)
        torch.save(
            {'state_dict': model.state_dict(), 'specialist_fold': specialist_fold,
             'specialist_names': list(SPECIALIST_NAMES)},
            Path(save_dir) / f'v31_specialist_fold_{specialist_fold}.pth'
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return (
        oof,
        [torch.stack(values).mean(0) for values in valid_predictions],
        [torch.stack(values).mean(0) for values in test_predictions],
        history,
    )


def apply_view_policy(
    classifier_output, residuals, fusion, need_threshold, direction_threshold,
    direction_margin, alpha_up, alpha_down
):
    need_probability = torch.sigmoid(classifier_output['need_logit'])
    direction_probability = F.softmax(classifier_output['direction_logits'], dim=1)
    down_probability, up_probability = direction_probability[:, 0], direction_probability[:, 1]
    choose_up = (
        (need_probability >= need_threshold)
        & (up_probability >= direction_threshold)
        & ((up_probability - down_probability) >= direction_margin)
        & (float(alpha_up) > 0)
    )
    choose_down = (
        (need_probability >= need_threshold)
        & (down_probability >= direction_threshold)
        & ((down_probability - up_probability) >= direction_margin)
        & (float(alpha_down) > 0)
    )
    correction = torch.zeros_like(fusion)
    correction[choose_up] = float(alpha_up) * residuals[choose_up, 0:1]
    correction[choose_down] = float(alpha_down) * residuals[choose_down, 1:2]
    action = torch.zeros(len(fusion), dtype=torch.long)
    action[choose_down], action[choose_up] = 1, 2
    return {
        'prediction': fusion + correction,
        'correction': correction,
        'action': action,
        'need_probability': need_probability,
        'down_probability': down_probability,
        'up_probability': up_probability,
    }


def apply_ensemble_policy(outputs, residual_views, views, policy):
    view_results = [
        apply_view_policy(output, residuals, _fusion(view['expert_matrix']), **policy)
        for output, residuals, view in zip(outputs, residual_views, views)
    ]
    fusion = torch.stack([_fusion(view['expert_matrix']) for view in views]).mean(0)
    prediction = torch.stack([result['prediction'] for result in view_results]).mean(0)
    return {
        'fusion': fusion,
        'prediction': prediction,
        'correction': prediction - fusion,
        'view_results': view_results,
    }


def selection_stats(fusion, prediction, labels, correction):
    gain = torch.abs(fusion - labels).view(-1) - torch.abs(prediction - labels).view(-1)
    selected = correction.abs().view(-1) > 1e-8
    return {
        'mae': float(torch.abs(prediction - labels).mean().item()),
        'mean_realized_gain': float(gain.mean().item()),
        'correction_rate': float(selected.float().mean().item()),
        'mean_abs_correction': float(correction.abs().mean().item()),
        'harm_over_005_rate': float((gain < -0.05).float().mean().item()),
        'harm_over_010_rate': float((gain < -0.10).float().mean().item()),
        'correction_precision': float((gain[selected] > 0).float().mean().item()) if selected.any() else None,
    }


def _scenario_rows(fusion, prediction, labels, correction):
    error = torch.abs(prediction - labels).view(-1)
    gain = torch.abs(fusion - labels).view(-1) - error
    regions = _regions(labels)
    counts = torch.bincount(regions, minlength=len(REGION_NAMES)).to(error.dtype)
    scenarios = dict(SCENARIOS)
    scenarios['balanced'] = tuple((counts.sum() / counts.clamp_min(1.0)).tolist())
    rows = {}
    for name, factors in scenarios.items():
        weight = torch.tensor(factors, dtype=error.dtype)[regions]
        weight = weight / weight.sum().clamp_min(1e-12)
        rows[name] = {
            'mae': float((weight * error).sum().item()),
            'harm_005': float((weight * (gain < -0.05).float()).sum().item()),
            'mean_abs_correction': float((weight * correction.abs().view(-1)).sum().item()),
        }
    return rows


def calibrate_ensemble_policy(
    classifier, valid_views, valid_residual_views, device, residual_margin,
    fold_count=3, need_thresholds=None, direction_thresholds=None,
    direction_margins=None, alpha_values=None, robustness_weight=0.50,
    harm_penalty=0.25, precision_penalty=0.08, target_precision=0.65,
    correction_penalty=0.02
):
    need_thresholds = need_thresholds or [0.50, 0.60, 0.70, 0.80, 0.90]
    direction_thresholds = direction_thresholds or [0.55, 0.60, 0.65, 0.70, 0.80]
    direction_margins = direction_margins or [0.0, 0.10, 0.20]
    alpha_values = alpha_values or [0.0, 0.25, 0.50, 0.75, 1.0]
    outputs = [
        predict_classifier(classifier, view['context'], residuals, device)
        for view, residuals in zip(valid_views, valid_residual_views)
    ]
    labels = valid_views[0]['labels']
    fusion = torch.stack([_fusion(view['expert_matrix']) for view in valid_views]).mean(0)
    folds = stratified_indices(labels, fusion, residual_margin, fold_count, 19001)
    rows = []
    for need_threshold, direction_threshold, direction_margin, alpha_up, alpha_down in itertools.product(
        need_thresholds, direction_thresholds, direction_margins, alpha_values, alpha_values
    ):
        policy = {
            'need_threshold': float(need_threshold),
            'direction_threshold': float(direction_threshold),
            'direction_margin': float(direction_margin),
            'alpha_up': float(alpha_up),
            'alpha_down': float(alpha_down),
        }
        result = apply_ensemble_policy(outputs, valid_residual_views, valid_views, policy)
        records = []
        for fold_index, indices in enumerate(folds, start=1):
            records.extend(
                {'fold': fold_index, 'scenario': name, **values}
                for name, values in _scenario_rows(
                    result['fusion'][indices], result['prediction'][indices],
                    labels[indices], result['correction'][indices]
                ).items()
            )
        mean_mae = float(np.mean([row['mae'] for row in records]))
        worst_mae = float(np.max([row['mae'] for row in records]))
        worst_harm = float(np.max([row['harm_005'] for row in records]))
        mean_correction = float(np.mean([row['mean_abs_correction'] for row in records]))
        stats = selection_stats(result['fusion'], result['prediction'], labels, result['correction'])
        precision = target_precision if stats['correction_precision'] is None else stats['correction_precision']
        objective = (
            mean_mae + robustness_weight * (worst_mae - mean_mae)
            + harm_penalty * worst_harm
            + precision_penalty * max(0.0, target_precision - precision)
            + correction_penalty * mean_correction
        )
        rows.append({
            **policy, 'objective': objective, 'cv_mean_mae': mean_mae,
            'cv_worst_mae': worst_mae, 'cv_worst_harm_005': worst_harm, **stats
        })
    best = min(
        rows,
        key=lambda row: (
            row['objective'], row['cv_worst_mae'], row['harm_over_005_rate'],
            -(row['correction_precision'] if row['correction_precision'] is not None else 1.0),
            row['correction_rate'],
        )
    )
    return (
        {key: float(best[key]) for key in (
            'need_threshold', 'direction_threshold', 'direction_margin',
            'alpha_up', 'alpha_down'
        )},
        rows,
        outputs,
    )
