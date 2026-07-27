import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .candidate_gain_policy_v34 import (
    apply_ensemble_policy,
    apply_view_policy,
    calibrate_candidate_gain_policy,
    selection_stats,
    test_gain_diagnostics,
)
from .candidate_gain_v34 import (
    CandidateGainNet,
    candidate_gain_targets,
    cross_fit_candidate_gain,
    gain_model_metrics,
)
from .ordinal_quantile_v33 import (
    OrdinalConformalNet,
    _predict as predict_ordinal_model,
    cross_fit_ordinal_quantile,
    ordinal_probabilities,
    stratified_folds as ordinal_stratified_folds,
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


def _ensemble_ordinal_output(outputs):
    logits = torch.stack([output['ordinal_logits'] for output in outputs]).mean(0)
    return {
        'quantiles': torch.stack([output['quantiles'] for output in outputs]).mean(0),
        'ordinal_logits': logits,
        'region_probs': ordinal_probabilities(logits),
        'direct_target': torch.stack([output['direct_target'] for output in outputs]).mean(0),
    }


def _load_or_train_ordinal_outputs(
    train_cache,
    oof_residuals,
    valid_views,
    valid_residual_views,
    test_views,
    test_residual_views,
    device,
    seed,
    save_dir,
    v33_model_dir,
    reuse_v33_models,
    fold_count,
    residual_margin,
    epochs,
    learning_rate,
    hidden_dim,
    semantic_hidden_dim,
    dropout,
    batch_size,
    max_center,
    max_width,
    ordinal_loss_weight,
    target_loss_weight,
    median_loss_weight,
    patience,
):
    v33_model_dir = Path(v33_model_dir)
    checkpoints = [
        v33_model_dir / f'v33_multitask_fold_{index}.pth'
        for index in range(1, int(fold_count) + 1)
    ]
    if reuse_v33_models and all(path.is_file() for path in checkpoints):
        logger.info('Reusing %d V3.3 ordinal/quantile fold models.', len(checkpoints))
        fusion = _fusion(train_cache['expert_matrix'])
        folds = ordinal_stratified_folds(
            train_cache['labels'], fusion, residual_margin,
            fold_count, seed + 2718,
        )
        n = len(train_cache['labels'])
        oof_output = {
            'quantiles': torch.zeros(n, 3),
            'ordinal_logits': torch.zeros(n, 4),
            'direct_target': torch.zeros(n, 1),
        }
        valid_accumulators = [list() for _ in valid_views]
        test_accumulators = [list() for _ in test_views]
        best_epochs = []
        for fold_index, (holdout, checkpoint) in enumerate(zip(folds, checkpoints), start=1):
            model = OrdinalConformalNet(
                train_cache['context'].size(1),
                train_cache['semantic'].size(1),
                oof_residuals.size(1),
                hidden_dim,
                semantic_hidden_dim,
                dropout,
                max_center,
                max_width,
            ).to(device)
            payload = torch.load(checkpoint, map_location=device)
            model.load_state_dict(payload['state_dict'])
            best_epochs.append(int(payload.get('best_epoch', -1)))
            holdout_cache = {
                key: value[holdout]
                if torch.is_tensor(value) and value.ndim > 0 and len(value) == n
                else value
                for key, value in train_cache.items()
            }
            prediction = predict_ordinal_model(
                model, holdout_cache, oof_residuals[holdout], device
            )
            for key in ('quantiles', 'ordinal_logits', 'direct_target'):
                oof_output[key][holdout] = prediction[key]
            for view_index, (view, residuals) in enumerate(zip(valid_views, valid_residual_views)):
                valid_accumulators[view_index].append(
                    predict_ordinal_model(model, view, residuals, device)
                )
            for view_index, (view, residuals) in enumerate(zip(test_views, test_residual_views)):
                test_accumulators[view_index].append(
                    predict_ordinal_model(model, view, residuals, device)
                )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        oof_output['region_probs'] = ordinal_probabilities(oof_output['ordinal_logits'])

        def average(values_by_view):
            rows = []
            for values in values_by_view:
                logits = torch.stack([value['ordinal_logits'] for value in values]).mean(0)
                rows.append({
                    'quantiles': torch.stack([value['quantiles'] for value in values]).mean(0),
                    'ordinal_logits': logits,
                    'region_probs': ordinal_probabilities(logits),
                    'direct_target': torch.stack([value['direct_target'] for value in values]).mean(0),
                })
            return rows

        valid_outputs = average(valid_accumulators)
        test_outputs = average(test_accumulators)
        best_epoch = max(best_epochs) if best_epochs else -1
        return oof_output, valid_outputs, test_outputs, [], [], best_epoch, True

    logger.info('V3.3 fold models unavailable or reuse disabled; cross-fitting ordinal features.')
    values = cross_fit_ordinal_quantile(
        train_cache,
        oof_residuals,
        valid_views,
        valid_residual_views,
        test_views,
        test_residual_views,
        device,
        seed,
        save_dir,
        fold_count=fold_count,
        residual_margin=residual_margin,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        semantic_hidden_dim=semantic_hidden_dim,
        dropout=dropout,
        batch_size=batch_size,
        max_center=max_center,
        max_width=max_width,
        ordinal_loss_weight=ordinal_loss_weight,
        target_loss_weight=target_loss_weight,
        median_loss_weight=median_loss_weight,
        patience=patience,
    )
    return (*values, False)


def _route_region_rows(fusion, prediction, labels, correction, action):
    regions = _regions(labels)
    rows = []
    for index, name in enumerate(REGION_NAMES):
        mask = regions == index
        if mask.any():
            rows.append({
                'region': name,
                'count': int(mask.sum().item()),
                **selection_stats(
                    fusion[mask], prediction[mask], labels[mask],
                    correction[mask], action[mask],
                ),
            })
    return rows


def _q50_apply(outputs, residual_views, views, threshold, alpha_up, alpha_down):
    results = []
    for output, residuals, view in zip(outputs, residual_views, views):
        fusion = _fusion(view['expert_matrix'])
        q50 = output['quantiles'][:, 1]
        choose_up = (q50 >= float(threshold)) & (float(alpha_up) > 0)
        choose_down = (q50 <= -float(threshold)) & (float(alpha_down) > 0)
        correction = torch.zeros_like(fusion)
        correction[choose_up] = float(alpha_up) * residuals[choose_up, 0:1]
        correction[choose_down] = float(alpha_down) * residuals[choose_down, 1:2]
        action = torch.zeros(len(fusion), dtype=torch.long)
        action[choose_down] = 1
        action[choose_up] = 2
        results.append({
            'prediction': fusion + correction,
            'correction': correction,
            'action': action,
        })
    fusion = torch.stack([_fusion(view['expert_matrix']) for view in views]).mean(0)
    prediction = torch.stack([result['prediction'] for result in results]).mean(0)
    correction = prediction - fusion
    votes = torch.stack([result['action'] for result in results])
    action = torch.zeros(len(fusion), dtype=torch.long)
    for sample_index in range(len(fusion)):
        nonzero = votes[:, sample_index]
        nonzero = nonzero[nonzero > 0]
        if len(nonzero):
            action[sample_index] = int(torch.bincount(nonzero, minlength=3).argmax().item())
    action[correction.abs().view(-1) <= 1e-8] = 0
    return {
        'fusion': fusion,
        'prediction': prediction,
        'correction': correction,
        'action': action,
        'view_results': results,
    }


def _calibrate_q50_policy(valid_outputs, valid_residual_views, valid_views):
    labels = valid_views[0]['labels']
    rows, feasible = [], []
    for threshold, alpha_up, alpha_down in itertools.product(
        [0.20, 0.30, 0.40, 0.50, 0.60], [0.0, 0.25, 0.50], [0.0, 0.25, 0.50]
    ):
        result = _q50_apply(
            valid_outputs, valid_residual_views, valid_views,
            threshold, alpha_up, alpha_down,
        )
        stats = selection_stats(
            result['fusion'], result['prediction'], labels,
            result['correction'], result['action'],
        )
        fallback = alpha_up == 0 and alpha_down == 0
        lcb = stats['correction_precision_wilson_lcb']
        allowed = fallback or (
            stats['correction_count'] >= 15
            and stats['correction_rate'] >= 0.05
            and stats['mean_realized_gain'] > 0
            and stats['harm_over_010_rate'] <= 0.02
            and lcb is not None and lcb >= 0.60
        )
        row = {
            'threshold': float(threshold),
            'alpha_up': float(alpha_up),
            'alpha_down': float(alpha_down),
            'feasible': bool(allowed),
            **stats,
        }
        rows.append(row)
        if allowed:
            feasible.append(row)
    best = min(feasible, key=lambda row: (
        row['mae'], row['harm_over_010_rate'],
        -(row['correction_precision_wilson_lcb'] or 0.0), row['correction_rate'],
    ))
    return {
        'threshold': float(best['threshold']),
        'alpha_up': float(best['alpha_up']),
        'alpha_down': float(best['alpha_down']),
    }, rows


def _gain_region_rows(output, cache, residuals):
    targets = candidate_gain_targets(cache, residuals)
    benefit_probability = torch.sigmoid(output['benefit_logits'])
    harm_probability = torch.sigmoid(output['harm_logits'])
    regions = _regions(cache['labels'])
    rows = []
    for region_index, region_name in enumerate(REGION_NAMES):
        mask = regions == region_index
        if not mask.any():
            continue
        for candidate_index, candidate_name in enumerate(('up', 'down')):
            actual = targets['gains'][mask, candidate_index]
            predicted = output['predicted_gain'][mask, candidate_index]
            rows.append({
                'region': region_name,
                'candidate': candidate_name,
                'count': int(mask.sum().item()),
                'actual_mean_gain': float(actual.mean().item()),
                'predicted_mean_gain': float(predicted.mean().item()),
                'actual_positive_rate': float((actual > 0).float().mean().item()),
                'actual_harm_010_rate': float((actual < -0.10).float().mean().item()),
                'mean_benefit_probability': float(
                    benefit_probability[mask, candidate_index].mean().item()
                ),
                'mean_harm_probability': float(
                    harm_probability[mask, candidate_index].mean().item()
                ),
            })
    return rows


def run_candidate_gain_v34(
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
    gain_candidate_hidden_dim=64,
    gain_dropout=0.20,
    gain_patience=8,
    max_predicted_gain=1.50,
    gain_loss_weight=1.0,
    benefit_loss_weight=0.75,
    harm_loss_weight=0.50,
    min_policy_selected=15,
    min_policy_coverage=0.05,
    min_policy_wilson_lcb=0.60,
    max_policy_harm_010=0.02,
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
        save_dir / 'v34_specialist_fold_history.csv', index=False
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
            save_dir / 'v34_ordinal_selection_history.csv', index=False
        )
    if ordinal_fold_history:
        pd.DataFrame(ordinal_fold_history).to_csv(
            save_dir / 'v34_ordinal_fold_history.csv', index=False
        )

    (
        oof_gain_output,
        valid_gain_outputs,
        test_gain_outputs,
        gain_selection_history,
        gain_fold_history,
        gain_best_epoch,
    ) = cross_fit_candidate_gain(
        caches['train'], oof_residuals, oof_ordinal,
        caches['valid_views'], valid_residual_views, valid_ordinal_outputs,
        caches['test_views'], test_residual_views, test_ordinal_outputs,
        device, seed + 24001, save_dir,
        fold_count=gain_folds,
        selection_fold_count=gain_selection_folds,
        epochs=gain_epochs,
        learning_rate=gain_learning_rate,
        hidden_dim=gain_hidden_dim,
        candidate_hidden_dim=gain_candidate_hidden_dim,
        dropout=gain_dropout,
        batch_size=batch_size,
        max_gain=max_predicted_gain,
        gain_loss_weight=gain_loss_weight,
        benefit_loss_weight=benefit_loss_weight,
        harm_loss_weight=harm_loss_weight,
        patience=gain_patience,
    )
    pd.DataFrame(gain_selection_history).to_csv(
        save_dir / 'v34_gain_selection_history.csv', index=False
    )
    pd.DataFrame(gain_fold_history).to_csv(
        save_dir / 'v34_gain_fold_history.csv', index=False
    )

    policy, curve_rows, policy_rows, fold_diagnostics = calibrate_candidate_gain_policy(
        oof_gain_output, oof_residuals, caches['train'],
        valid_gain_outputs, valid_residual_views, caches['valid_views'],
        min_selected=min_policy_selected,
        min_coverage=min_policy_coverage,
        min_wilson_lcb=min_policy_wilson_lcb,
        max_harm_010=max_policy_harm_010,
    )
    pd.DataFrame(curve_rows).to_csv(
        save_dir / 'v34_candidate_precision_coverage.csv', index=False
    )
    pd.DataFrame(policy_rows).to_csv(
        save_dir / 'v34_joint_policy_calibration.csv', index=False
    )
    pd.DataFrame(fold_diagnostics).to_csv(
        save_dir / 'v34_oof_fold_policy_diagnostics.csv', index=False
    )
    test_result = apply_ensemble_policy(
        test_gain_outputs, test_residual_views, caches['test_views'], policy
    )

    q50_policy, q50_rows = _calibrate_q50_policy(
        valid_ordinal_outputs, valid_residual_views, caches['valid_views']
    )
    pd.DataFrame(q50_rows).to_csv(
        save_dir / 'v34_q50_policy_calibration.csv', index=False
    )
    q50_test_result = _q50_apply(
        test_ordinal_outputs, test_residual_views, caches['test_views'],
        **q50_policy,
    )

    labels = caches['test_views'][0]['labels']
    fusion = test_result['fusion']
    fusion_metrics = _safe_metrics(metrics_fn, fusion, labels)
    routed_metrics = _safe_metrics(metrics_fn, test_result['prediction'], labels)
    q50_metrics = _safe_metrics(metrics_fn, q50_test_result['prediction'], labels)
    route_stats = selection_stats(
        fusion, test_result['prediction'], labels,
        test_result['correction'], test_result['action'],
    )
    q50_stats = selection_stats(
        fusion, q50_test_result['prediction'], labels,
        q50_test_result['correction'], q50_test_result['action'],
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
        save_dir / 'v34_test_specialist_diagnostics.csv', index=False
    )
    pd.DataFrame(_region_diagnostics(ensemble_cache, ensemble_residuals)).to_csv(
        save_dir / 'v34_test_specialist_region_diagnostics.csv', index=False
    )
    correlation = np.nan_to_num(np.corrcoef(errors.numpy(), rowvar=False), nan=0.0)
    pd.DataFrame(
        correlation,
        index=('fusion',) + SPECIALIST_NAMES,
        columns=('fusion',) + SPECIALIST_NAMES,
    ).to_csv(save_dir / 'v34_test_specialist_error_correlation.csv')

    ensemble_gain_output = _ensemble_output(test_gain_outputs)
    ensemble_ordinal_output = _ensemble_ordinal_output(test_ordinal_outputs)
    test_targets = candidate_gain_targets(ensemble_cache, ensemble_residuals)
    gain_metrics = gain_model_metrics(ensemble_gain_output, test_targets)
    pd.DataFrame(test_gain_diagnostics(
        ensemble_gain_output, ensemble_cache, ensemble_residuals
    )).to_csv(save_dir / 'v34_test_gain_diagnostics.csv', index=False)
    pd.DataFrame(_gain_region_rows(
        ensemble_gain_output, ensemble_cache, ensemble_residuals
    )).to_csv(save_dir / 'v34_test_gain_region_diagnostics.csv', index=False)
    pd.DataFrame(_route_region_rows(
        fusion, test_result['prediction'], labels,
        test_result['correction'], test_result['action'],
    )).to_csv(save_dir / 'v34_test_route_region_diagnostics.csv', index=False)

    per_view_rows = []
    for view_index, (view, output, residuals, result) in enumerate(zip(
        caches['test_views'], test_gain_outputs, test_residual_views,
        test_result['view_results'],
    ), start=1):
        view_fusion = _fusion(view['expert_matrix'])
        stats = selection_stats(
            view_fusion, result['prediction'], labels,
            result['correction'], result['action'],
        )
        view_targets = candidate_gain_targets(view, residuals)
        per_view_rows.append({
            'view': view_index,
            'fusion_mae': float(torch.abs(view_fusion - labels).mean().item()),
            **stats,
            **gain_model_metrics(output, view_targets),
        })
    pd.DataFrame(per_view_rows).to_csv(
        save_dir / 'v34_test_per_view_routing.csv', index=False
    )

    pd.DataFrame([
        {'model': 'fusion', **fusion_metrics},
        {'model': 'candidate_gain_router', **routed_metrics, **route_stats},
        {'model': 'q50_direction_baseline', **q50_metrics, **q50_stats},
        {'model': 'specialist_oracle', **oracle_metrics},
    ]).to_csv(save_dir / 'v34_test_baseline_comparison.csv', index=False)

    benefit_probability = torch.sigmoid(ensemble_gain_output['benefit_logits'])
    harm_probability = torch.sigmoid(ensemble_gain_output['harm_logits'])
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
        'q50_baseline_prediction': q50_test_result['prediction'].view(-1).numpy(),
        'q50_residual': ensemble_ordinal_output['quantiles'][:, 1].numpy(),
        'up_predicted_gain': ensemble_gain_output['predicted_gain'][:, 0].numpy(),
        'down_predicted_gain': ensemble_gain_output['predicted_gain'][:, 1].numpy(),
        'up_benefit_probability': benefit_probability[:, 0].numpy(),
        'down_benefit_probability': benefit_probability[:, 1].numpy(),
        'up_harm_probability': harm_probability[:, 0].numpy(),
        'down_harm_probability': harm_probability[:, 1].numpy(),
        'up_actual_gain': test_targets['gains'][:, 0].numpy(),
        'down_actual_gain': test_targets['gains'][:, 1].numpy(),
    }
    for index, name in enumerate(SPECIALIST_NAMES):
        prediction_data[f'{name}_residual'] = ensemble_residuals[:, index].numpy()
    pd.DataFrame(prediction_data).to_csv(
        save_dir / 'candidate_gain_v34_predictions.csv', index=False
    )

    oracle_space = fusion_metrics['MAE'] - oracle_metrics['MAE']
    recovered = fusion_metrics['MAE'] - routed_metrics['MAE']
    summary = {
        'seed': int(seed),
        'method': 'candidate_gain_selector_v34',
        'reused_v33_ordinal_models': bool(reused_ordinal),
        'ordinal_best_epoch': int(ordinal_best_epoch),
        'gain_best_epoch': int(gain_best_epoch),
        'selective_policy': policy,
        'q50_policy': q50_policy,
        'fusion_metrics': fusion_metrics,
        'residual_router_metrics': routed_metrics,
        'q50_direction_metrics': q50_metrics,
        'specialist_oracle_metrics': oracle_metrics,
        **route_stats,
        **{f'gain_model_{key}': value for key, value in gain_metrics.items()},
        'oracle_gap_recovery_ratio': recovered / oracle_space if oracle_space > 1e-12 else 0.0,
    }
    with open(save_dir / 'candidate_gain_v34_summary.json', 'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
    torch.save(
        {
            'policy': policy,
            'q50_policy': q50_policy,
            'gain_best_epoch': int(gain_best_epoch),
        },
        save_dir / 'candidate_gain_v34_policy.pth',
    )
    logger.info(
        'V3.4 TEST fusion_MAE=%.4f routed_MAE=%.4f q50_MAE=%.4f oracle_MAE=%.4f '
        'precision=%s lcb=%s coverage=%.4f up_corr=%.4f down_corr=%.4f policy=%s',
        fusion_metrics['MAE'], routed_metrics['MAE'], q50_metrics['MAE'],
        oracle_metrics['MAE'],
        'NA' if route_stats['correction_precision'] is None else f"{route_stats['correction_precision']:.4f}",
        'NA' if route_stats['correction_precision_wilson_lcb'] is None else f"{route_stats['correction_precision_wilson_lcb']:.4f}",
        route_stats['correction_rate'],
        gain_metrics['up_gain_correlation'], gain_metrics['down_gain_correlation'],
        policy,
    )
    return summary
