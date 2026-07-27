import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .conformal_policy_v33 import (
    apply_ensemble_policy,
    apply_target_stacking,
    calibrate_policy,
    calibrate_target_stacking,
    conformalize,
    coverage_rows,
    fit_split_conformal,
    selection_stats,
)
from .ordinal_quantile_v33 import cross_fit_ordinal_quantile, output_metrics
from .quantile_policy_v32 import apply_direct_baseline, calibrate_direct_baseline
from .residual_direction_core import SPECIALIST_NAMES, _fusion, _regions
from .residual_direction_specialist import _candidate_diagnostics, _region_diagnostics
from .residual_quantile_v32 import fit_ridge_baseline, predict_ridge
from .selective_policy_v31 import cross_fit_specialists_for_views
from .specialist_residual import _safe_metrics

logger = logging.getLogger('MMSA')
REGION_NAMES = ('strong_negative', 'negative', 'boundary', 'positive', 'strong_positive')


def _route_region_rows(fusion, prediction, labels, correction, action):
    regions = _regions(labels)
    rows = []
    for index, name in enumerate(REGION_NAMES):
        mask = regions == index
        if not mask.any():
            continue
        stats = selection_stats(
            fusion[mask], prediction[mask], labels[mask], correction[mask], action[mask]
        )
        rows.append({'region': name, 'count': int(mask.sum().item()), **stats})
    return rows


def _ensemble_output(outputs):
    return {
        'quantiles': torch.stack([output['quantiles'] for output in outputs]).mean(0),
        'ordinal_logits': torch.stack([output['ordinal_logits'] for output in outputs]).mean(0),
        'region_probs': torch.stack([output['region_probs'] for output in outputs]).mean(0),
        'direct_target': torch.stack([output['direct_target'] for output in outputs]).mean(0),
    }


def run_ordinal_conformal_v33(
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
    model_folds=5,
    model_epochs=80,
    model_learning_rate=4e-4,
    model_hidden_dim=128,
    semantic_hidden_dim=96,
    model_dropout=0.20,
    model_patience=10,
    max_quantile_center=1.75,
    max_quantile_width=1.75,
    ordinal_loss_weight=0.30,
    target_loss_weight=0.25,
    median_loss_weight=0.20,
    conformal_coverage=0.80,
    min_policy_selected=15,
    min_policy_coverage=0.05,
    min_policy_wilson_lcb=0.60,
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
        save_dir / 'v33_specialist_fold_history.csv', index=False
    )

    (
        oof_output,
        valid_outputs,
        test_outputs,
        selection_history,
        fold_history,
        best_epoch,
    ) = cross_fit_ordinal_quantile(
        caches['train'],
        oof_residuals,
        caches['valid_views'],
        valid_residual_views,
        caches['test_views'],
        test_residual_views,
        device,
        seed + 12001,
        save_dir,
        fold_count=model_folds,
        residual_margin=residual_margin,
        epochs=model_epochs,
        learning_rate=model_learning_rate,
        hidden_dim=model_hidden_dim,
        semantic_hidden_dim=semantic_hidden_dim,
        dropout=model_dropout,
        batch_size=batch_size,
        max_center=max_quantile_center,
        max_width=max_quantile_width,
        ordinal_loss_weight=ordinal_loss_weight,
        target_loss_weight=target_loss_weight,
        median_loss_weight=median_loss_weight,
        patience=model_patience,
    )
    pd.DataFrame(selection_history).to_csv(
        save_dir / 'v33_model_selection_history.csv', index=False
    )
    pd.DataFrame(fold_history).to_csv(
        save_dir / 'v33_model_fold_history.csv', index=False
    )

    train_fusion = _fusion(caches['train']['expert_matrix'])
    train_residual_target = caches['train']['labels'] - train_fusion
    conformal = fit_split_conformal(
        oof_output['quantiles'], train_residual_target, conformal_coverage
    )
    coverage_diagnostics = []
    for calibrated, name in ((False, 'raw_oof'), (True, 'conformal_oof')):
        rows = coverage_rows(
            oof_output['quantiles'], caches['train']['labels'], train_fusion,
            conformal if calibrated else None,
        )
        coverage_diagnostics.extend({'split': name, **row} for row in rows)
    pd.DataFrame(coverage_diagnostics).to_csv(
        save_dir / 'v33_oof_conformal_coverage.csv', index=False
    )
    (save_dir / 'v33_conformal.json').write_text(
        json.dumps(conformal, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    policy, policy_rows = calibrate_policy(
        valid_outputs,
        valid_residual_views,
        caches['valid_views'],
        conformal,
        min_selected=min_policy_selected,
        min_coverage=min_policy_coverage,
        min_wilson_lcb=min_policy_wilson_lcb,
    )
    pd.DataFrame(policy_rows).to_csv(
        save_dir / 'v33_policy_calibration.csv', index=False
    )
    test_result = apply_ensemble_policy(
        test_outputs, test_residual_views, caches['test_views'], conformal, policy
    )

    stacking_policy, stacking_rows = calibrate_target_stacking(
        valid_outputs, caches['valid_views']
    )
    pd.DataFrame(stacking_rows).to_csv(
        save_dir / 'v33_target_stacking_calibration.csv', index=False
    )
    stacking_prediction, stacking_stats = apply_target_stacking(
        test_outputs, caches['test_views'], stacking_policy
    )

    ridge_model, ridge_cv = fit_ridge_baseline(
        caches['train'], oof_residuals, residual_margin, seed + 17001
    )
    pd.DataFrame(ridge_cv).to_csv(save_dir / 'v33_ridge_cv.csv', index=False)
    valid_ridge = [
        predict_ridge(ridge_model, view, residuals)
        for view, residuals in zip(caches['valid_views'], valid_residual_views)
    ]
    test_ridge = [
        predict_ridge(ridge_model, view, residuals)
        for view, residuals in zip(caches['test_views'], test_residual_views)
    ]
    ridge_policy, ridge_rows = calibrate_direct_baseline(
        valid_ridge, caches['valid_views'], 'ridge_v33', 3
    )
    pd.DataFrame(ridge_rows).to_csv(
        save_dir / 'v33_ridge_calibration.csv', index=False
    )
    ridge_prediction, ridge_stats = apply_direct_baseline(
        test_ridge, caches['test_views'], ridge_policy
    )

    labels = caches['test_views'][0]['labels']
    fusion = test_result['fusion']
    fusion_metrics = _safe_metrics(metrics_fn, fusion, labels)
    routed_metrics = _safe_metrics(metrics_fn, test_result['prediction'], labels)
    stacking_metrics = _safe_metrics(metrics_fn, stacking_prediction, labels)
    ridge_metrics = _safe_metrics(metrics_fn, ridge_prediction, labels)
    route_stats = selection_stats(
        fusion, test_result['prediction'], labels,
        test_result['correction'], test_result['action'],
    )

    ensemble_experts = torch.stack([
        view['expert_matrix'] for view in caches['test_views']
    ]).mean(0)
    ensemble_residuals = torch.stack(test_residual_views).mean(0)
    ensemble_cache = {
        'expert_matrix': ensemble_experts,
        'labels': labels,
        'sample_ids': caches['test_views'][0]['sample_ids'],
    }
    specialist_rows, oracle, errors = _candidate_diagnostics(
        ensemble_cache, ensemble_residuals, metrics_fn
    )
    oracle_metrics = _safe_metrics(metrics_fn, oracle, labels)
    pd.DataFrame(specialist_rows).to_csv(
        save_dir / 'v33_test_specialist_diagnostics.csv', index=False
    )
    pd.DataFrame(_region_diagnostics(ensemble_cache, ensemble_residuals)).to_csv(
        save_dir / 'v33_test_specialist_region_diagnostics.csv', index=False
    )
    correlation = np.nan_to_num(np.corrcoef(errors.numpy(), rowvar=False), nan=0.0)
    action_names = ('fusion',) + SPECIALIST_NAMES
    pd.DataFrame(correlation, index=action_names, columns=action_names).to_csv(
        save_dir / 'v33_test_specialist_error_correlation.csv'
    )

    ensemble_output = _ensemble_output(test_outputs)
    model_test_metrics = output_metrics(
        ensemble_output, labels, fusion, residual_margin
    )
    coverage_rows_test = []
    for calibrated, name in ((False, 'raw_test'), (True, 'conformal_test')):
        rows = coverage_rows(
            ensemble_output['quantiles'], labels, fusion,
            conformal if calibrated else None,
        )
        coverage_rows_test.extend({'split': name, **row} for row in rows)
    pd.DataFrame(coverage_rows_test).to_csv(
        save_dir / 'v33_test_conformal_coverage.csv', index=False
    )
    pd.DataFrame(_route_region_rows(
        fusion, test_result['prediction'], labels,
        test_result['correction'], test_result['action'],
    )).to_csv(save_dir / 'v33_test_route_region_diagnostics.csv', index=False)

    per_view_rows = []
    for view_index, (view, output, residuals, result) in enumerate(zip(
        caches['test_views'], test_outputs, test_residual_views,
        test_result['view_results'],
    ), start=1):
        view_fusion = _fusion(view['expert_matrix'])
        stats = selection_stats(
            view_fusion, result['prediction'], labels,
            result['correction'], result['action'],
        )
        metrics = output_metrics(output, labels, view_fusion, residual_margin)
        per_view_rows.append({
            'view': view_index,
            'fusion_mae': float(torch.abs(view_fusion - labels).mean().item()),
            **stats,
            **metrics,
        })
    pd.DataFrame(per_view_rows).to_csv(
        save_dir / 'v33_test_per_view_routing.csv', index=False
    )

    baseline_rows = [
        {'model': 'fusion', **fusion_metrics},
        {'model': 'ordinal_conformal_router', **routed_metrics, **route_stats},
        {'model': 'direct_target_stacking', **stacking_metrics, **stacking_stats},
        {'model': 'ridge_residual', **ridge_metrics, **ridge_stats},
        {'model': 'specialist_oracle', **oracle_metrics},
    ]
    pd.DataFrame(baseline_rows).to_csv(
        save_dir / 'v33_test_baseline_comparison.csv', index=False
    )

    calibrated_quantiles = conformalize(ensemble_output['quantiles'], conformal)
    region_probs = ensemble_output['region_probs']
    sample_ids = ensemble_cache['sample_ids']
    if len(sample_ids) != len(labels):
        sample_ids = [str(index) for index in range(len(labels))]
    prediction_data = {
        'sample_id': sample_ids,
        'target': labels.view(-1).numpy(),
        'fusion_prediction': fusion.view(-1).numpy(),
        'router_prediction': test_result['prediction'].view(-1).numpy(),
        'router_correction': test_result['correction'].view(-1).numpy(),
        'router_action': test_result['action'].numpy(),
        'raw_q10': ensemble_output['quantiles'][:, 0].numpy(),
        'q50': ensemble_output['quantiles'][:, 1].numpy(),
        'raw_q90': ensemble_output['quantiles'][:, 2].numpy(),
        'conformal_q10': calibrated_quantiles[:, 0].numpy(),
        'conformal_q90': calibrated_quantiles[:, 2].numpy(),
        'prob_strong_negative': region_probs[:, 0].numpy(),
        'prob_negative': region_probs[:, 1].numpy(),
        'prob_boundary': region_probs[:, 2].numpy(),
        'prob_positive': region_probs[:, 3].numpy(),
        'prob_strong_positive': region_probs[:, 4].numpy(),
        'direct_target_prediction': ensemble_output['direct_target'].view(-1).numpy(),
        'stacking_prediction': stacking_prediction.view(-1).numpy(),
        'ridge_prediction': ridge_prediction.view(-1).numpy(),
    }
    for index, name in enumerate(SPECIALIST_NAMES):
        prediction_data[f'{name}_residual'] = ensemble_residuals[:, index].numpy()
    pd.DataFrame(prediction_data).to_csv(
        save_dir / 'ordinal_conformal_v33_predictions.csv', index=False
    )

    oracle_space = fusion_metrics['MAE'] - oracle_metrics['MAE']
    recovered = fusion_metrics['MAE'] - routed_metrics['MAE']
    summary = {
        'seed': int(seed),
        'method': 'ordinal_conformal_stacking_v33',
        'model_best_epoch': int(best_epoch),
        'conformal': conformal,
        'selective_policy': policy,
        'target_stacking_policy': stacking_policy,
        'ridge_policy': ridge_policy,
        'fusion_metrics': fusion_metrics,
        'residual_router_metrics': routed_metrics,
        'target_stacking_metrics': stacking_metrics,
        'ridge_residual_metrics': ridge_metrics,
        'specialist_oracle_metrics': oracle_metrics,
        **route_stats,
        **{f'model_{key}': value for key, value in model_test_metrics.items()},
        'oracle_gap_recovery_ratio': recovered / oracle_space if oracle_space > 1e-12 else 0.0,
    }
    with open(save_dir / 'ordinal_conformal_v33_summary.json', 'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
    torch.save(
        {
            'conformal': conformal,
            'policy': policy,
            'target_stacking_policy': stacking_policy,
            'ridge_model': ridge_model,
            'ridge_policy': ridge_policy,
            'best_epoch': int(best_epoch),
        },
        save_dir / 'ordinal_conformal_v33_models.pth',
    )
    logger.info(
        'V3.3 TEST fusion_MAE=%.4f routed_MAE=%.4f stacking_MAE=%.4f ridge_MAE=%.4f oracle_MAE=%.4f residual_corr=%.4f sign_bal=%.4f ordinal_bal=%.4f policy=%s',
        fusion_metrics['MAE'], routed_metrics['MAE'], stacking_metrics['MAE'],
        ridge_metrics['MAE'], oracle_metrics['MAE'],
        model_test_metrics['residual_correlation'],
        model_test_metrics['sign_balanced_accuracy'],
        model_test_metrics['ordinal_balanced_accuracy'], policy,
    )
    return summary
