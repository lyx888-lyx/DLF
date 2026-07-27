import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .residual_direction_core import SPECIALIST_NAMES, _fusion, _regions
from .residual_direction_specialist import _candidate_diagnostics, _region_diagnostics
from .residual_quantile_v32 import (
    fit_ridge_baseline,
    predict_quantiles,
    predict_ridge,
    quantile_metrics,
    train_quantile_model,
)
from .selective_policy_v31 import cross_fit_specialists_for_views
from .specialist_residual import _safe_metrics

logger = logging.getLogger('MMSA')
REGION_NAMES = ('strong_negative', 'negative', 'boundary', 'positive', 'strong_positive')
SCENARIOS = {
    'original': (1.0, 1.0, 1.0, 1.0, 1.0),
    'negative_heavy': (2.5, 1.8, 1.0, 0.8, 0.8),
    'positive_heavy': (0.8, 0.8, 1.0, 1.8, 2.5),
    'boundary_heavy': (1.0, 1.0, 2.5, 1.0, 1.0),
    'strong_heavy': (2.0, 1.0, 1.0, 1.0, 2.0),
}


def _stratified_folds(labels, fusion, fold_count, seed):
    residual = (labels - fusion).view(-1)
    direction = torch.ones_like(residual, dtype=torch.long)
    direction[residual < -0.10] = 0
    direction[residual > 0.10] = 2
    strata = _regions(labels) * 3 + direction
    generator = torch.Generator().manual_seed(int(seed))
    folds = [[] for _ in range(fold_count)]
    for value in torch.unique(strata).tolist():
        indices = torch.nonzero(strata == value, as_tuple=False).view(-1)
        indices = indices[torch.randperm(len(indices), generator=generator)]
        for position, sample_index in enumerate(indices.tolist()):
            folds[position % fold_count].append(sample_index)
    result = [torch.tensor(sorted(values), dtype=torch.long) for values in folds]
    if any(len(values) == 0 for values in result):
        raise RuntimeError('Policy calibration produced an empty fold.')
    return result


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
            'harm_010': float((weight * (gain < -0.10).float()).sum().item()),
            'mean_abs_correction': float((weight * correction.abs().view(-1)).sum().item()),
        }
    return rows


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


def apply_quantile_view_policy(
    quantiles,
    specialist_residuals,
    fusion,
    interval_margin,
    max_interval_width,
    alpha_up,
    alpha_down,
):
    q10, _, q90 = quantiles[:, 0], quantiles[:, 1], quantiles[:, 2]
    width = q90 - q10
    choose_up = (
        (q10 >= float(interval_margin))
        & (width <= float(max_interval_width))
        & (float(alpha_up) > 0)
    )
    choose_down = (
        (q90 <= -float(interval_margin))
        & (width <= float(max_interval_width))
        & (float(alpha_down) > 0)
    )
    correction = torch.zeros_like(fusion)
    correction[choose_up] = float(alpha_up) * specialist_residuals[choose_up, 0:1]
    correction[choose_down] = float(alpha_down) * specialist_residuals[choose_down, 1:2]
    action = torch.zeros(len(fusion), dtype=torch.long)
    action[choose_down] = 1
    action[choose_up] = 2
    return {
        'prediction': fusion + correction,
        'correction': correction,
        'action': action,
        'interval_width': width,
    }


def apply_quantile_ensemble_policy(quantile_views, residual_views, views, policy):
    results = [
        apply_quantile_view_policy(
            quantiles,
            residuals,
            _fusion(view['expert_matrix']),
            **policy,
        )
        for quantiles, residuals, view in zip(quantile_views, residual_views, views)
    ]
    fusion = torch.stack([_fusion(view['expert_matrix']) for view in views]).mean(0)
    prediction = torch.stack([result['prediction'] for result in results]).mean(0)
    return {
        'fusion': fusion,
        'prediction': prediction,
        'correction': prediction - fusion,
        'view_results': results,
    }


def _objective(records, stats, robustness_weight, harm_penalty, correction_penalty):
    mean_mae = float(np.mean([row['mae'] for row in records]))
    worst_mae = float(np.max([row['mae'] for row in records]))
    worst_harm = float(np.max([row['harm_005'] for row in records]))
    mean_correction = float(np.mean([row['mean_abs_correction'] for row in records]))
    return (
        mean_mae
        + robustness_weight * (worst_mae - mean_mae)
        + harm_penalty * worst_harm
        + correction_penalty * mean_correction
    ), mean_mae, worst_mae, worst_harm


def calibrate_quantile_policy(
    quantile_views,
    residual_views,
    valid_views,
    fold_count=3,
    interval_margins=None,
    max_interval_widths=None,
    alpha_values=None,
    robustness_weight=0.50,
    harm_penalty=0.30,
    correction_penalty=0.02,
):
    interval_margins = interval_margins or [0.0, 0.05, 0.10, 0.20]
    max_interval_widths = max_interval_widths or [0.50, 0.75, 1.00, 1.50, 3.00]
    alpha_values = alpha_values or [0.0, 0.25, 0.50, 0.75, 1.0]
    labels = valid_views[0]['labels']
    fusion = torch.stack([_fusion(view['expert_matrix']) for view in valid_views]).mean(0)
    folds = _stratified_folds(labels, fusion, fold_count, 19001)
    rows = []
    for margin, width, alpha_up, alpha_down in itertools.product(
        interval_margins, max_interval_widths, alpha_values, alpha_values
    ):
        policy = {
            'interval_margin': float(margin),
            'max_interval_width': float(width),
            'alpha_up': float(alpha_up),
            'alpha_down': float(alpha_down),
        }
        result = apply_quantile_ensemble_policy(
            quantile_views, residual_views, valid_views, policy
        )
        records = []
        for fold_index, indices in enumerate(folds, start=1):
            records.extend(
                {'fold': fold_index, 'scenario': name, **values}
                for name, values in _scenario_rows(
                    result['fusion'][indices], result['prediction'][indices],
                    labels[indices], result['correction'][indices],
                ).items()
            )
        stats = selection_stats(
            result['fusion'], result['prediction'], labels, result['correction']
        )
        objective, mean_mae, worst_mae, worst_harm = _objective(
            records, stats, robustness_weight, harm_penalty, correction_penalty
        )
        rows.append({
            **policy,
            'objective': objective,
            'cv_mean_mae': mean_mae,
            'cv_worst_mae': worst_mae,
            'cv_worst_harm_005': worst_harm,
            **stats,
        })
    best = min(
        rows,
        key=lambda row: (
            row['objective'], row['cv_worst_mae'], row['harm_over_005_rate'],
            -(row['correction_precision'] if row['correction_precision'] is not None else 0.0),
            row['correction_rate'],
        ),
    )
    return {
        key: float(best[key])
        for key in ('interval_margin', 'max_interval_width', 'alpha_up', 'alpha_down')
    }, rows


def _apply_direct_view(predicted_residual, fusion, alpha, clip_value):
    correction = float(alpha) * predicted_residual.clamp(-float(clip_value), float(clip_value))
    return fusion + correction


def calibrate_direct_baseline(
    valid_predictions,
    valid_views,
    name,
    fold_count=3,
    alpha_values=None,
    clip_values=None,
):
    alpha_values = alpha_values or [0.0, 0.25, 0.50, 0.75, 1.0]
    clip_values = clip_values or [0.25, 0.50, 0.75, 1.00]
    labels = valid_views[0]['labels']
    fusion = torch.stack([_fusion(view['expert_matrix']) for view in valid_views]).mean(0)
    folds = _stratified_folds(labels, fusion, fold_count, 29001)
    rows = []
    for alpha, clip_value in itertools.product(alpha_values, clip_values):
        per_view = [
            _apply_direct_view(prediction, _fusion(view['expert_matrix']), alpha, clip_value)
            for prediction, view in zip(valid_predictions, valid_views)
        ]
        final = torch.stack(per_view).mean(0)
        correction = final - fusion
        records = []
        for indices in folds:
            records.extend(
                values
                for values in _scenario_rows(
                    fusion[indices], final[indices], labels[indices], correction[indices]
                ).values()
            )
        stats = selection_stats(fusion, final, labels, correction)
        objective, mean_mae, worst_mae, worst_harm = _objective(
            records, stats, 0.50, 0.30, 0.02
        )
        rows.append({
            'baseline': name,
            'alpha': float(alpha),
            'clip_value': float(clip_value),
            'objective': objective,
            'cv_mean_mae': mean_mae,
            'cv_worst_mae': worst_mae,
            'cv_worst_harm_005': worst_harm,
            **stats,
        })
    best = min(rows, key=lambda row: (row['objective'], row['harm_over_005_rate'], row['mae']))
    return {'alpha': float(best['alpha']), 'clip_value': float(best['clip_value'])}, rows


def apply_direct_baseline(predictions, views, policy):
    fusion = torch.stack([_fusion(view['expert_matrix']) for view in views]).mean(0)
    final = torch.stack([
        _apply_direct_view(prediction, _fusion(view['expert_matrix']), **policy)
        for prediction, view in zip(predictions, views)
    ]).mean(0)
    return final, selection_stats(fusion, final, views[0]['labels'], final - fusion)


def _quantile_region_rows(quantiles, labels, fusion, residual_margin):
    target = labels - fusion
    regions = _regions(labels)
    rows = []
    for index, name in enumerate(REGION_NAMES):
        mask = regions == index
        if mask.any():
            rows.append({
                'region': name,
                'count': int(mask.sum().item()),
                **quantile_metrics(quantiles[mask], target[mask], residual_margin),
            })
    return rows


def run_aligned_semantic_quantile_v32(
    caches,
    metrics_fn,
    device,
    save_dir,
    seed,
    specialist_folds=5,
    specialist_epochs=25,
    specialist_learning_rate=8e-4,
    specialist_shared_dim=96,
    specialist_adapter_dim=48,
    specialist_dropout=0.15,
    max_residual=0.75,
    boundary_max_residual=0.35,
    batch_size=128,
    residual_margin=0.10,
    boundary_label_threshold=0.50,
    off_region_anchor_weight=0.01,
    quantile_epochs=80,
    quantile_learning_rate=4e-4,
    quantile_hidden_dim=128,
    semantic_hidden_dim=96,
    quantile_dropout=0.20,
    quantile_patience=10,
    max_quantile_center=1.50,
    max_quantile_width=1.50,
    median_loss_weight=0.25,
    calibration_folds=3,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    specialist_kwargs = dict(
        epochs=specialist_epochs,
        learning_rate=specialist_learning_rate,
        shared_dim=specialist_shared_dim,
        adapter_dim=specialist_adapter_dim,
        dropout=specialist_dropout,
        max_residual=max_residual,
        boundary_max_residual=boundary_max_residual,
        batch_size=batch_size,
        residual_margin=residual_margin,
        boundary_threshold=boundary_label_threshold,
        off_anchor_weight=off_region_anchor_weight,
    )
    oof_residuals, valid_residual_views, test_residual_views, specialist_history = (
        cross_fit_specialists_for_views(
            caches['train'], caches['valid_views'], caches['test_views'],
            device, seed, specialist_folds, save_dir, **specialist_kwargs
        )
    )
    pd.DataFrame(specialist_history).to_csv(
        save_dir / 'v32_specialist_fold_history.csv', index=False
    )

    quantile_model, selection_history, final_history, best_epoch = train_quantile_model(
        caches['train'], oof_residuals, device, seed + 12001,
        residual_margin=residual_margin,
        epochs=quantile_epochs,
        learning_rate=quantile_learning_rate,
        hidden_dim=quantile_hidden_dim,
        semantic_hidden_dim=semantic_hidden_dim,
        dropout=quantile_dropout,
        batch_size=batch_size,
        max_center=max_quantile_center,
        max_width=max_quantile_width,
        patience=quantile_patience,
        median_loss_weight=median_loss_weight,
    )
    pd.DataFrame(selection_history).to_csv(
        save_dir / 'v32_quantile_selection_history.csv', index=False
    )
    pd.DataFrame(final_history).to_csv(
        save_dir / 'v32_quantile_final_history.csv', index=False
    )

    valid_quantile_views = [
        predict_quantiles(quantile_model, view, residuals, device)
        for view, residuals in zip(caches['valid_views'], valid_residual_views)
    ]
    test_quantile_views = [
        predict_quantiles(quantile_model, view, residuals, device)
        for view, residuals in zip(caches['test_views'], test_residual_views)
    ]
    policy, calibration_rows = calibrate_quantile_policy(
        valid_quantile_views, valid_residual_views, caches['valid_views'],
        fold_count=calibration_folds,
    )
    pd.DataFrame(calibration_rows).to_csv(
        save_dir / 'v32_quantile_policy_calibration.csv', index=False
    )
    test_result = apply_quantile_ensemble_policy(
        test_quantile_views, test_residual_views, caches['test_views'], policy
    )

    ridge_model, ridge_cv = fit_ridge_baseline(
        caches['train'], oof_residuals, residual_margin, seed + 17001
    )
    pd.DataFrame(ridge_cv).to_csv(save_dir / 'v32_ridge_cv.csv', index=False)
    valid_ridge_views = [
        predict_ridge(ridge_model, view, residuals)
        for view, residuals in zip(caches['valid_views'], valid_residual_views)
    ]
    test_ridge_views = [
        predict_ridge(ridge_model, view, residuals)
        for view, residuals in zip(caches['test_views'], test_residual_views)
    ]
    ridge_policy, ridge_rows = calibrate_direct_baseline(
        valid_ridge_views, caches['valid_views'], 'ridge', calibration_folds
    )

    valid_median_views = [values[:, 1:2] for values in valid_quantile_views]
    test_median_views = [values[:, 1:2] for values in test_quantile_views]
    median_policy, median_rows = calibrate_direct_baseline(
        valid_median_views, caches['valid_views'], 'quantile_median', calibration_folds
    )
    pd.DataFrame(ridge_rows + median_rows).to_csv(
        save_dir / 'v32_baseline_calibration.csv', index=False
    )

    test_labels = caches['test_views'][0]['labels']
    fusion = test_result['fusion']
    fusion_metrics = _safe_metrics(metrics_fn, fusion, test_labels)
    routed_metrics = _safe_metrics(metrics_fn, test_result['prediction'], test_labels)
    route_stats = selection_stats(
        fusion, test_result['prediction'], test_labels, test_result['correction']
    )
    ridge_prediction, ridge_stats = apply_direct_baseline(
        test_ridge_views, caches['test_views'], ridge_policy
    )
    median_prediction, median_stats = apply_direct_baseline(
        test_median_views, caches['test_views'], median_policy
    )

    ensemble_experts = torch.stack([
        view['expert_matrix'] for view in caches['test_views']
    ]).mean(0)
    ensemble_residuals = torch.stack(test_residual_views).mean(0)
    ensemble_cache = {
        'expert_matrix': ensemble_experts,
        'labels': test_labels,
        'sample_ids': caches['test_views'][0]['sample_ids'],
    }
    specialist_rows, oracle, errors = _candidate_diagnostics(
        ensemble_cache, ensemble_residuals, metrics_fn
    )
    oracle_metrics = _safe_metrics(metrics_fn, oracle, test_labels)

    ensemble_quantiles = torch.stack(test_quantile_views).mean(0)
    quantile_test_metrics = quantile_metrics(
        ensemble_quantiles, test_labels - fusion, residual_margin
    )
    pd.DataFrame(_quantile_region_rows(
        ensemble_quantiles, test_labels, fusion, residual_margin
    )).to_csv(save_dir / 'v32_test_quantile_region_diagnostics.csv', index=False)
    pd.DataFrame(specialist_rows).to_csv(
        save_dir / 'v32_test_specialist_diagnostics.csv', index=False
    )
    pd.DataFrame(_region_diagnostics(ensemble_cache, ensemble_residuals)).to_csv(
        save_dir / 'v32_test_specialist_region_diagnostics.csv', index=False
    )
    correlation = np.nan_to_num(np.corrcoef(errors.numpy(), rowvar=False), nan=0.0)
    action_names = ('fusion',) + SPECIALIST_NAMES
    pd.DataFrame(correlation, index=action_names, columns=action_names).to_csv(
        save_dir / 'v32_test_specialist_error_correlation.csv'
    )

    baseline_rows = [
        {'model': 'fusion', **fusion_metrics},
        {'model': 'quantile_specialist_router', **routed_metrics, **route_stats},
        {'model': 'ridge_residual', **_safe_metrics(metrics_fn, ridge_prediction, test_labels), **ridge_stats},
        {'model': 'quantile_median_residual', **_safe_metrics(metrics_fn, median_prediction, test_labels), **median_stats},
        {'model': 'specialist_oracle', **oracle_metrics},
    ]
    pd.DataFrame(baseline_rows).to_csv(
        save_dir / 'v32_test_baseline_comparison.csv', index=False
    )

    sample_ids = ensemble_cache['sample_ids']
    if len(sample_ids) != len(test_labels):
        sample_ids = [str(index) for index in range(len(test_labels))]
    prediction_data = {
        'sample_id': sample_ids,
        'target': test_labels.view(-1).numpy(),
        'fusion_prediction': fusion.view(-1).numpy(),
        'router_prediction': test_result['prediction'].view(-1).numpy(),
        'router_correction': test_result['correction'].view(-1).numpy(),
        'q10_residual': ensemble_quantiles[:, 0].numpy(),
        'q50_residual': ensemble_quantiles[:, 1].numpy(),
        'q90_residual': ensemble_quantiles[:, 2].numpy(),
        'ridge_prediction': ridge_prediction.view(-1).numpy(),
        'quantile_median_prediction': median_prediction.view(-1).numpy(),
    }
    for index, name in enumerate(SPECIALIST_NAMES):
        prediction_data[f'{name}_residual'] = ensemble_residuals[:, index].numpy()
    pd.DataFrame(prediction_data).to_csv(
        save_dir / 'aligned_semantic_quantile_v32_predictions.csv', index=False
    )

    per_view_rows = []
    for index, (view, quantiles, residuals, result) in enumerate(zip(
        caches['test_views'], test_quantile_views, test_residual_views,
        test_result['view_results'],
    ), start=1):
        view_fusion = _fusion(view['expert_matrix'])
        stats = selection_stats(
            view_fusion, result['prediction'], test_labels, result['correction']
        )
        per_view_rows.append({
            'view': index,
            'fusion_mae': float(torch.abs(view_fusion - test_labels).mean().item()),
            **stats,
            **{f'quantile_{key}': value for key, value in quantile_metrics(
                quantiles, test_labels - view_fusion, residual_margin
            ).items()},
        })
    pd.DataFrame(per_view_rows).to_csv(
        save_dir / 'v32_test_per_view_routing.csv', index=False
    )

    oracle_space = fusion_metrics['MAE'] - oracle_metrics['MAE']
    recovered = fusion_metrics['MAE'] - routed_metrics['MAE']
    summary = {
        'seed': int(seed),
        'method': 'aligned_semantic_quantile_v32',
        'quantile_best_epoch': int(best_epoch),
        'selective_policy': policy,
        'ridge_policy': ridge_policy,
        'quantile_median_policy': median_policy,
        'fusion_metrics': fusion_metrics,
        'residual_router_metrics': routed_metrics,
        'ridge_residual_metrics': _safe_metrics(metrics_fn, ridge_prediction, test_labels),
        'quantile_median_metrics': _safe_metrics(metrics_fn, median_prediction, test_labels),
        'specialist_oracle_metrics': oracle_metrics,
        **route_stats,
        **{f'quantile_{key}': value for key, value in quantile_test_metrics.items()},
        'oracle_gap_recovery_ratio': recovered / oracle_space if oracle_space > 1e-12 else 0.0,
    }
    with open(save_dir / 'aligned_semantic_quantile_v32_summary.json', 'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
    torch.save({
        'state_dict': quantile_model.state_dict(),
        'policy': policy,
        'best_epoch': int(best_epoch),
        'ridge_model': ridge_model,
        'ridge_policy': ridge_policy,
        'quantile_median_policy': median_policy,
    }, save_dir / 'aligned_semantic_quantile_v32_models.pth')
    logger.info(
        'V3.2 TEST fusion_MAE=%.4f routed_MAE=%.4f ridge_MAE=%.4f median_MAE=%.4f oracle_MAE=%.4f corr=%.4f sign_bal=%.4f policy=%s',
        fusion_metrics['MAE'], routed_metrics['MAE'],
        summary['ridge_residual_metrics']['MAE'], summary['quantile_median_metrics']['MAE'],
        oracle_metrics['MAE'], quantile_test_metrics['residual_correlation'],
        quantile_test_metrics['sign_balanced_accuracy'], policy,
    )
    return summary
