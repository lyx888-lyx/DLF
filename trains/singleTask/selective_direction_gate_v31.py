import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .need_direction_v31 import (
    ensemble_classifier_metrics,
    predict_classifier,
    train_need_direction_classifier,
)
from .residual_direction_core import SPECIALIST_NAMES, _fusion, _regions
from .selective_policy_v31 import (
    apply_ensemble_policy,
    apply_view_policy,
    calibrate_ensemble_policy,
    cross_fit_specialists_for_views,
    selection_stats,
)
from .specialist_residual import _safe_metrics

logger = logging.getLogger('MMSA')
ACTION_NAMES = ('fusion', 'up', 'down', 'boundary')
REGION_NAMES = ('strong_negative', 'negative', 'boundary', 'positive', 'strong_positive')


def candidate_diagnostics(views, residual_views, metrics_fn):
    labels = views[0]['labels']
    fusion = torch.stack([_fusion(view['expert_matrix']) for view in views]).mean(0)
    candidates = [fusion]
    for specialist_index in range(len(SPECIALIST_NAMES)):
        candidates.append(
            torch.stack([
                _fusion(view['expert_matrix']) + residuals[:, specialist_index:specialist_index + 1]
                for view, residuals in zip(views, residual_views)
            ]).mean(0)
        )
    matrix = torch.cat(candidates, dim=1)
    errors = torch.abs(matrix - labels)
    winner = errors.argmin(1)
    oracle = matrix.gather(1, winner.unsqueeze(1))
    rows = []
    for index, name in enumerate(ACTION_NAMES):
        prediction = matrix[:, index:index + 1]
        gain = torch.abs(fusion - labels) - torch.abs(prediction - labels)
        rows.append({
            'candidate': name,
            **_safe_metrics(metrics_fn, prediction, labels),
            'oracle_win_rate': float((winner == index).float().mean().item()),
            'mean_gain_vs_fusion': float(gain.mean().item()),
            'gain_over_005_rate': float((gain > 0.05).float().mean().item()),
            'harm_over_005_rate': float((gain < -0.05).float().mean().item()),
            'mean_residual': float((prediction - fusion).mean().item()),
            'mean_abs_residual': float((prediction - fusion).abs().mean().item()),
        })
    return rows, oracle, errors, matrix


def region_diagnostics(views, residual_views):
    labels = views[0]['labels']
    fusion = torch.stack([_fusion(view['expert_matrix']) for view in views]).mean(0)
    candidates = [fusion]
    for specialist_index in range(len(SPECIALIST_NAMES)):
        candidates.append(
            torch.stack([
                _fusion(view['expert_matrix']) + residuals[:, specialist_index:specialist_index + 1]
                for view, residuals in zip(views, residual_views)
            ]).mean(0)
        )
    matrix = torch.cat(candidates, dim=1)
    groups, rows = _regions(labels), []
    for group_index, group_name in enumerate(REGION_NAMES):
        mask = groups == group_index
        if not mask.any():
            continue
        for candidate_index, candidate_name in enumerate(ACTION_NAMES):
            prediction = matrix[mask, candidate_index:candidate_index + 1]
            target, base = labels[mask], fusion[mask]
            gain = torch.abs(base - target) - torch.abs(prediction - target)
            rows.append({
                'region': group_name, 'candidate': candidate_name,
                'count': int(mask.sum().item()),
                'mae': float(torch.abs(prediction - target).mean().item()),
                'mean_gain_vs_fusion': float(gain.mean().item()),
                'gain_over_005_rate': float((gain > 0.05).float().mean().item()),
                'harm_over_005_rate': float((gain < -0.05).float().mean().item()),
            })
    return rows


def per_view_rows(outputs, residual_views, views, policy):
    rows = []
    for view_index, (output, residuals, view) in enumerate(
        zip(outputs, residual_views, views), start=1
    ):
        result = apply_view_policy(
            output, residuals, _fusion(view['expert_matrix']), **policy
        )
        rows.append({
            'view': view_index,
            **selection_stats(
                _fusion(view['expert_matrix']), result['prediction'],
                view['labels'], result['correction']
            ),
        })
    return rows


def run_backbone_crossfit_direction_v31(
    caches, metrics_fn, device, save_dir, seed,
    specialist_folds=5, specialist_epochs=25,
    specialist_learning_rate=8e-4, specialist_shared_dim=96,
    specialist_adapter_dim=48, specialist_dropout=0.15,
    max_residual=0.75, boundary_max_residual=0.35,
    batch_size=128, residual_margin=0.10,
    boundary_label_threshold=0.50, off_region_anchor_weight=0.01,
    classifier_epochs=60, classifier_learning_rate=5e-4,
    classifier_hidden_dim=64, classifier_dropout=0.15,
    classifier_patience=8, label_smoothing=0.05,
    direction_loss_weight=1.0, calibration_folds=3
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    train_cache = caches['train']
    valid_views, test_views = caches['valid_views'], caches['test_views']
    specialist_kwargs = {
        'epochs': specialist_epochs,
        'learning_rate': specialist_learning_rate,
        'shared_dim': specialist_shared_dim,
        'adapter_dim': specialist_adapter_dim,
        'dropout': specialist_dropout,
        'max_residual': max_residual,
        'boundary_max_residual': boundary_max_residual,
        'batch_size': batch_size,
        'residual_margin': residual_margin,
        'boundary_threshold': boundary_label_threshold,
        'off_anchor_weight': off_region_anchor_weight,
    }
    oof, valid_residuals, test_residuals, specialist_history = (
        cross_fit_specialists_for_views(
            train_cache, valid_views, test_views, device, seed,
            specialist_folds, save_dir, **specialist_kwargs
        )
    )
    pd.DataFrame(specialist_history).to_csv(
        save_dir / 'v31_specialist_fold_history.csv', index=False
    )
    classifier, selection_history, final_history, best_epoch = (
        train_need_direction_classifier(
            train_cache, oof, device, seed + 12001,
            residual_margin=residual_margin,
            epochs=classifier_epochs,
            learning_rate=classifier_learning_rate,
            hidden_dim=classifier_hidden_dim,
            dropout=classifier_dropout,
            batch_size=batch_size,
            patience=classifier_patience,
            label_smoothing=label_smoothing,
            direction_loss_weight=direction_loss_weight,
        )
    )
    pd.DataFrame(selection_history).to_csv(
        save_dir / 'v31_classifier_selection_history.csv', index=False
    )
    pd.DataFrame(final_history).to_csv(
        save_dir / 'v31_classifier_final_history.csv', index=False
    )
    policy, calibration_rows, _ = calibrate_ensemble_policy(
        classifier, valid_views, valid_residuals, device,
        residual_margin, fold_count=calibration_folds
    )
    pd.DataFrame(calibration_rows).to_csv(
        save_dir / 'v31_selective_policy_calibration.csv', index=False
    )
    test_outputs = [
        predict_classifier(classifier, view['context'], residuals, device)
        for view, residuals in zip(test_views, test_residuals)
    ]
    result = apply_ensemble_policy(test_outputs, test_residuals, test_views, policy)
    labels = test_views[0]['labels']
    fusion_metrics = _safe_metrics(metrics_fn, result['fusion'], labels)
    router_metrics = _safe_metrics(metrics_fn, result['prediction'], labels)
    specialist_rows, oracle, errors, candidate_matrix = candidate_diagnostics(
        test_views, test_residuals, metrics_fn
    )
    oracle_metrics = _safe_metrics(metrics_fn, oracle, labels)
    stats = selection_stats(
        result['fusion'], result['prediction'], labels, result['correction']
    )
    classifier_stats = ensemble_classifier_metrics(
        test_outputs, test_views, residual_margin
    )
    oracle_space = fusion_metrics['MAE'] - oracle_metrics['MAE']
    recovered = fusion_metrics['MAE'] - router_metrics['MAE']
    summary = {
        'seed': int(seed),
        'backbone_mode': 'exact_group_crossfit_per_view_routing',
        'classifier_best_epoch': int(best_epoch),
        'selective_policy': policy,
        'fusion_metrics': fusion_metrics,
        'residual_router_metrics': router_metrics,
        'specialist_oracle_metrics': oracle_metrics,
        **stats, **classifier_stats,
        'oracle_gap_recovery_ratio': (
            recovered / oracle_space if oracle_space > 1e-12 else 0.0
        ),
    }
    with open(
        save_dir / 'backbone_crossfit_direction_v31_summary.json',
        'w', encoding='utf-8'
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
    pd.DataFrame(specialist_rows).to_csv(
        save_dir / 'v31_test_specialist_diagnostics.csv', index=False
    )
    pd.DataFrame(region_diagnostics(test_views, test_residuals)).to_csv(
        save_dir / 'v31_test_specialist_region_diagnostics.csv', index=False
    )
    correlation = np.nan_to_num(np.corrcoef(errors.numpy(), rowvar=False), nan=0.0)
    pd.DataFrame(
        correlation, index=ACTION_NAMES, columns=ACTION_NAMES
    ).to_csv(save_dir / 'v31_test_specialist_error_correlation.csv')
    pd.DataFrame(per_view_rows(
        test_outputs, test_residuals, test_views, policy
    )).to_csv(save_dir / 'v31_test_per_view_routing.csv', index=False)

    data = {
        'sample_id': test_views[0]['sample_ids'],
        'target': labels.view(-1).numpy(),
        'ensemble_fusion_prediction': result['fusion'].view(-1).numpy(),
        'ensemble_router_prediction': result['prediction'].view(-1).numpy(),
        'ensemble_correction': result['correction'].view(-1).numpy(),
        'oracle_prediction': oracle.view(-1).numpy(),
    }
    for index, name in enumerate(ACTION_NAMES):
        data[f'ensemble_{name}_candidate'] = candidate_matrix[:, index].numpy()
    for view_index, (view, residuals, view_result) in enumerate(
        zip(test_views, test_residuals, result['view_results']), start=1
    ):
        data[f'view_{view_index}_fusion'] = _fusion(
            view['expert_matrix']
        ).view(-1).numpy()
        data[f'view_{view_index}_router'] = view_result[
            'prediction'
        ].view(-1).numpy()
        data[f'view_{view_index}_correction'] = view_result[
            'correction'
        ].view(-1).numpy()
        data[f'view_{view_index}_need_probability'] = view_result[
            'need_probability'
        ].numpy()
        data[f'view_{view_index}_down_probability'] = view_result[
            'down_probability'
        ].numpy()
        data[f'view_{view_index}_up_probability'] = view_result[
            'up_probability'
        ].numpy()
        data[f'view_{view_index}_action'] = view_result['action'].numpy()
        for specialist_index, specialist_name in enumerate(SPECIALIST_NAMES):
            data[f'view_{view_index}_{specialist_name}_residual'] = (
                residuals[:, specialist_index].numpy()
            )
    pd.DataFrame(data).to_csv(
        save_dir / 'backbone_crossfit_direction_v31_predictions.csv', index=False
    )
    torch.save(
        {'state_dict': classifier.state_dict(), 'policy': policy,
         'best_epoch': int(best_epoch), 'specialist_names': list(SPECIALIST_NAMES)},
        save_dir / 'v31_need_direction_classifier.pth'
    )
    logger.info(
        'V3.1 TEST fusion_MAE=%.4f routed_MAE=%.4f oracle_MAE=%.4f '
        'policy=%s need_bal_acc=%.4f direction_bal_acc=%.4f '
        'precision=%s coverage=%.4f',
        fusion_metrics['MAE'], router_metrics['MAE'], oracle_metrics['MAE'],
        policy, classifier_stats['need_balanced_accuracy'],
        classifier_stats['direction_balanced_accuracy_on_need'],
        'NA' if stats['correction_precision'] is None else f"{stats['correction_precision']:.4f}",
        stats['correction_rate']
    )
    return summary
