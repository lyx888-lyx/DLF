import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .candidate_gain_system_v34 import _load_or_train_ordinal_outputs
from .down_alpha_gain_v341 import (
    ALPHAS,
    apply_logit_calibrators,
    cross_fit_down_alpha_gain,
    down_alpha_targets,
    fit_logit_calibrators,
    gain_model_metrics,
)
from .down_alpha_policy_v341 import (
    alpha_precision_coverage_rows,
    apply_ensemble_policy,
    calibrate_down_alpha_policy,
    selection_stats,
    test_alpha_diagnostics,
)
from .residual_direction_core import SPECIALIST_NAMES, _fusion, _regions
from .residual_direction_specialist import _candidate_diagnostics, _region_diagnostics
from .selective_policy_v31 import cross_fit_specialists_for_views
from .specialist_residual import _safe_metrics

logger = logging.getLogger('MMSA')
REGION_NAMES = ('strong_negative', 'negative', 'boundary', 'positive', 'strong_positive')


def _ensemble_output(outputs):
    return {
        key: torch.stack([output[key] for output in outputs]).mean(0)
        for key in ('predicted_gain', 'benefit_logits', 'harm_logits')
    }


def _route_region_rows(policy_name, fusion, prediction, labels, correction):
    regions = _regions(labels)
    rows = []
    for index, name in enumerate(REGION_NAMES):
        mask = regions == index
        if mask.any():
            rows.append({
                'policy_name': policy_name,
                'region': name,
                'count': int(mask.sum().item()),
                **selection_stats(
                    fusion[mask], prediction[mask], labels[mask], correction[mask]
                ),
            })
    return rows


def _calibration_rows(output, targets, calibrators):
    calibrated = apply_logit_calibrators(output, calibrators)
    rows = []
    for alpha_index, alpha in enumerate(ALPHAS):
        for name, logits_key, target_key in (
            ('benefit', 'benefit_logits', 'benefit'),
            ('harm', 'harm_logits', 'severe_harm'),
        ):
            raw = torch.sigmoid(output[logits_key][:, alpha_index])
            probability = calibrated[name][:, alpha_index]
            target = targets[target_key][:, alpha_index]
            params = calibrators[name][alpha_index]
            rows.append({
                'alpha_down': float(alpha),
                'target': name,
                'scale': float(params['scale']),
                'bias': float(params['bias']),
                'positive_rate': float(target.mean().item()),
                'raw_brier': float((raw - target).pow(2).mean().item()),
                'calibrated_brier': float((probability - target).pow(2).mean().item()),
                'raw_mean_probability': float(raw.mean().item()),
                'calibrated_mean_probability': float(probability.mean().item()),
            })
    return rows


def _gain_region_rows(output, cache, residuals, calibrators):
    targets = down_alpha_targets(cache, residuals, ALPHAS)
    probabilities = apply_logit_calibrators(output, calibrators)
    regions = _regions(cache['labels'])
    rows = []
    for region_index, region_name in enumerate(REGION_NAMES):
        mask = regions == region_index
        if not mask.any():
            continue
        for alpha_index, alpha in enumerate(ALPHAS):
            actual = targets['gains'][mask, alpha_index]
            rows.append({
                'region': region_name,
                'alpha_down': float(alpha),
                'count': int(mask.sum().item()),
                'actual_mean_gain': float(actual.mean().item()),
                'predicted_mean_gain': float(
                    output['predicted_gain'][mask, alpha_index].mean().item()
                ),
                'actual_positive_rate': float((actual > 0).float().mean().item()),
                'actual_harm_010_rate': float((actual < -0.10).float().mean().item()),
                'mean_benefit_probability': float(
                    probabilities['benefit'][mask, alpha_index].mean().item()
                ),
                'mean_harm_probability': float(
                    probabilities['harm'][mask, alpha_index].mean().item()
                ),
            })
    return rows


def run_down_alpha_v341(
    caches,
    metrics_fn,
    device,
    save_dir,
    seed,
    v33_model_dir,
    reuse_v33_models=True,
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
    ordinal_folds=5,
    ordinal_epochs=80,
    ordinal_learning_rate=4e-4,
    ordinal_hidden_dim=128,
    semantic_hidden_dim=96,
    ordinal_dropout=0.20,
    ordinal_patience=10,
    max_quantile_center=1.75,
    max_quantile_width=1.75,
    ordinal_loss_weight=0.30,
    target_loss_weight=0.25,
    median_loss_weight=0.20,
    gain_folds=5,
    gain_selection_folds=3,
    gain_epochs=60,
    gain_learning_rate=4e-4,
    gain_hidden_dim=128,
    gain_alpha_hidden_dim=64,
    gain_dropout=0.20,
    gain_patience=8,
    max_predicted_gain=0.75,
    gain_loss_weight=1.0,
    benefit_loss_weight=0.75,
    harm_loss_weight=0.50,
    min_policy_selected=25,
    min_policy_coverage=0.05,
    min_policy_precision=0.65,
    min_policy_wilson_lcb=0.50,
    max_policy_harm_010=0.02,
    min_oof_fold_gain=-0.002,
    min_positive_oof_folds=2,
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
        save_dir / 'v341_specialist_fold_history.csv', index=False
    )

    (
        oof_ordinal,
        valid_ordinal_outputs,
        test_ordinal_outputs,
        ordinal_selection_history,
        ordinal_fold_history,
        ordinal_best_epoch,
        reused_ordinal,
    ) = _load_or_train_ordinal_outputs(
        caches['train'], oof_residuals,
        caches['valid_views'], valid_residual_views,
        caches['test_views'], test_residual_views,
        device, seed + 12001, save_dir, v33_model_dir, reuse_v33_models,
        ordinal_folds, residual_margin, ordinal_epochs, ordinal_learning_rate,
        ordinal_hidden_dim, semantic_hidden_dim, ordinal_dropout, batch_size,
        max_quantile_center, max_quantile_width, ordinal_loss_weight,
        target_loss_weight, median_loss_weight, ordinal_patience,
    )
    if ordinal_selection_history:
        pd.DataFrame(ordinal_selection_history).to_csv(
            save_dir / 'v341_ordinal_selection_history.csv', index=False
        )
    if ordinal_fold_history:
        pd.DataFrame(ordinal_fold_history).to_csv(
            save_dir / 'v341_ordinal_fold_history.csv', index=False
        )

    (
        oof_gain_output,
        valid_gain_outputs,
        test_gain_outputs,
        gain_selection_history,
        gain_fold_history,
        gain_best_epoch,
    ) = cross_fit_down_alpha_gain(
        caches['train'], oof_residuals, oof_ordinal,
        caches['valid_views'], valid_residual_views, valid_ordinal_outputs,
        caches['test_views'], test_residual_views, test_ordinal_outputs,
        device, seed + 24001, save_dir,
        alphas=ALPHAS,
        fold_count=gain_folds,
        selection_fold_count=gain_selection_folds,
        epochs=gain_epochs,
        learning_rate=gain_learning_rate,
        hidden_dim=gain_hidden_dim,
        alpha_hidden_dim=gain_alpha_hidden_dim,
        dropout=gain_dropout,
        batch_size=batch_size,
        max_gain=max_predicted_gain,
        gain_loss_weight=gain_loss_weight,
        benefit_loss_weight=benefit_loss_weight,
        harm_loss_weight=harm_loss_weight,
        patience=gain_patience,
    )
    pd.DataFrame(gain_selection_history).to_csv(
        save_dir / 'v341_gain_selection_history.csv', index=False
    )
    pd.DataFrame(gain_fold_history).to_csv(
        save_dir / 'v341_gain_fold_history.csv', index=False
    )

    oof_targets = down_alpha_targets(caches['train'], oof_residuals, ALPHAS)
    calibrators = fit_logit_calibrators(oof_gain_output, oof_targets, ALPHAS)
    (save_dir / 'v341_probability_calibrators.json').write_text(
        json.dumps(calibrators, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    pd.DataFrame(_calibration_rows(
        oof_gain_output, oof_targets, calibrators
    )).to_csv(save_dir / 'v341_probability_calibration.csv', index=False)

    policy_set, policy_rows, fold_diagnostics = calibrate_down_alpha_policy(
        oof_gain_output, oof_residuals, caches['train'],
        valid_gain_outputs, valid_residual_views, caches['valid_views'],
        calibrators,
        alphas=ALPHAS,
        min_selected=min_policy_selected,
        min_coverage=min_policy_coverage,
        min_precision=min_policy_precision,
        min_wilson_lcb=min_policy_wilson_lcb,
        max_harm_010=max_policy_harm_010,
        min_fold_gain=min_oof_fold_gain,
        min_positive_folds=min_positive_oof_folds,
    )
    pd.DataFrame(policy_rows).to_csv(
        save_dir / 'v341_policy_calibration.csv', index=False
    )
    pd.DataFrame(fold_diagnostics).to_csv(
        save_dir / 'v341_oof_fold_policy_diagnostics.csv', index=False
    )
    pd.DataFrame(alpha_precision_coverage_rows(
        oof_gain_output, oof_residuals, caches['train'], calibrators, ALPHAS
    )).to_csv(save_dir / 'v341_oof_precision_coverage.csv', index=False)

    test_results = {
        name: apply_ensemble_policy(
            test_gain_outputs, test_residual_views, caches['test_views'],
            policy, calibrators, ALPHAS,
        )
        for name, policy in policy_set.items()
    }
    labels = caches['test_views'][0]['labels']
    formal_result = test_results['formal_safe_policy']
    fusion = formal_result['fusion']
    fusion_metrics = _safe_metrics(metrics_fn, fusion, labels)

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
        save_dir / 'v341_test_specialist_diagnostics.csv', index=False
    )
    pd.DataFrame(_region_diagnostics(ensemble_cache, ensemble_residuals)).to_csv(
        save_dir / 'v341_test_specialist_region_diagnostics.csv', index=False
    )
    correlation = np.nan_to_num(np.corrcoef(errors.numpy(), rowvar=False), nan=0.0)
    pd.DataFrame(
        correlation,
        index=('fusion',) + SPECIALIST_NAMES,
        columns=('fusion',) + SPECIALIST_NAMES,
    ).to_csv(save_dir / 'v341_test_specialist_error_correlation.csv')

    ensemble_gain_output = _ensemble_output(test_gain_outputs)
    test_targets = down_alpha_targets(ensemble_cache, ensemble_residuals, ALPHAS)
    model_metrics = gain_model_metrics(
        ensemble_gain_output, test_targets, calibrators, ALPHAS
    )
    pd.DataFrame(test_alpha_diagnostics(
        ensemble_gain_output, ensemble_cache, ensemble_residuals,
        calibrators, ALPHAS,
    )).to_csv(save_dir / 'v341_test_alpha_gain_diagnostics.csv', index=False)
    pd.DataFrame(_gain_region_rows(
        ensemble_gain_output, ensemble_cache, ensemble_residuals, calibrators
    )).to_csv(save_dir / 'v341_test_alpha_region_diagnostics.csv', index=False)

    baseline_rows = [{'model': 'fusion', **fusion_metrics}]
    route_region_rows = []
    result_summaries = {}
    for policy_name, result in test_results.items():
        metrics = _safe_metrics(metrics_fn, result['prediction'], labels)
        stats = selection_stats(
            fusion, result['prediction'], labels, result['correction']
        )
        baseline_rows.append({'model': policy_name, **metrics, **stats})
        route_region_rows.extend(_route_region_rows(
            policy_name, fusion, result['prediction'], labels, result['correction']
        ))
        result_summaries[policy_name] = {
            'policy': policy_set[policy_name],
            'metrics': metrics,
            **stats,
        }
    baseline_rows.append({'model': 'specialist_oracle', **oracle_metrics})
    pd.DataFrame(baseline_rows).to_csv(
        save_dir / 'v341_test_baseline_comparison.csv', index=False
    )
    pd.DataFrame(route_region_rows).to_csv(
        save_dir / 'v341_test_route_region_diagnostics.csv', index=False
    )

    per_view_rows = []
    for view_index, (view, result) in enumerate(zip(
        caches['test_views'], formal_result['view_results']
    ), start=1):
        view_fusion = _fusion(view['expert_matrix'])
        per_view_rows.append({
            'view': view_index,
            'fusion_mae': float(torch.abs(view_fusion - labels).mean().item()),
            **selection_stats(
                view_fusion, result['prediction'], labels, result['correction']
            ),
        })
    pd.DataFrame(per_view_rows).to_csv(
        save_dir / 'v341_test_per_view_routing.csv', index=False
    )

    probabilities = apply_logit_calibrators(ensemble_gain_output, calibrators)
    sample_ids = ensemble_cache['sample_ids']
    if len(sample_ids) != len(labels):
        sample_ids = [str(index) for index in range(len(labels))]
    prediction_data = {
        'sample_id': sample_ids,
        'target': labels.view(-1).numpy(),
        'fusion_prediction': fusion.view(-1).numpy(),
        'down_residual': ensemble_residuals[:, 1].numpy(),
    }
    for policy_name, result in test_results.items():
        prediction_data[f'{policy_name}_prediction'] = result['prediction'].view(-1).numpy()
        prediction_data[f'{policy_name}_correction'] = result['correction'].view(-1).numpy()
    for alpha_index, alpha in enumerate(ALPHAS):
        suffix = str(alpha).replace('.', 'p')
        prediction_data[f'alpha_{suffix}_predicted_gain'] = (
            ensemble_gain_output['predicted_gain'][:, alpha_index].numpy()
        )
        prediction_data[f'alpha_{suffix}_benefit_probability'] = (
            probabilities['benefit'][:, alpha_index].numpy()
        )
        prediction_data[f'alpha_{suffix}_harm_probability'] = (
            probabilities['harm'][:, alpha_index].numpy()
        )
        prediction_data[f'alpha_{suffix}_actual_gain'] = (
            test_targets['gains'][:, alpha_index].numpy()
        )
    pd.DataFrame(prediction_data).to_csv(
        save_dir / 'down_alpha_v341_predictions.csv', index=False
    )

    formal_metrics = result_summaries['formal_safe_policy']['metrics']
    oracle_space = fusion_metrics['MAE'] - oracle_metrics['MAE']
    recovered = fusion_metrics['MAE'] - formal_metrics['MAE']
    summary = {
        'seed': int(seed),
        'method': 'down_alpha_conditioned_v341',
        'alphas': [float(value) for value in ALPHAS],
        'reused_v33_ordinal_models': bool(reused_ordinal),
        'ordinal_best_epoch': int(ordinal_best_epoch),
        'gain_best_epoch': int(gain_best_epoch),
        'probability_calibrators': calibrators,
        'policies': policy_set,
        'fusion_metrics': fusion_metrics,
        'formal_safe_result': result_summaries['formal_safe_policy'],
        'best_valid_gain_shadow_result': result_summaries['best_valid_gain_policy'],
        'best_oof_stable_shadow_result': result_summaries['best_oof_stable_policy'],
        'specialist_oracle_metrics': oracle_metrics,
        **{f'gain_model_{key}': value for key, value in model_metrics.items()},
        'oracle_gap_recovery_ratio': recovered / oracle_space if oracle_space > 1e-12 else 0.0,
    }
    with open(save_dir / 'down_alpha_v341_summary.json', 'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
    torch.save({
        'calibrators': calibrators,
        'policies': policy_set,
        'best_epoch': int(gain_best_epoch),
        'alphas': [float(value) for value in ALPHAS],
    }, save_dir / 'down_alpha_v341_policy.pth')

    formal_stats = result_summaries['formal_safe_policy']
    logger.info(
        'V3.4.1 TEST fusion_MAE=%.4f formal_MAE=%.4f valid_shadow_MAE=%.4f oof_shadow_MAE=%.4f oracle_MAE=%.4f precision=%s lcb=%s coverage=%.4f policy=%s',
        fusion_metrics['MAE'],
        formal_stats['metrics']['MAE'],
        result_summaries['best_valid_gain_policy']['metrics']['MAE'],
        result_summaries['best_oof_stable_policy']['metrics']['MAE'],
        oracle_metrics['MAE'],
        'NA' if formal_stats['correction_precision'] is None else f"{formal_stats['correction_precision']:.4f}",
        'NA' if formal_stats['correction_precision_wilson_lcb'] is None else f"{formal_stats['correction_precision_wilson_lcb']:.4f}",
        formal_stats['correction_rate'],
        policy_set['formal_safe_policy'],
    )
    return summary
