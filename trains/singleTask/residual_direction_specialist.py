import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .specialist_residual import _cache_split, _normalize_cached, _safe_metrics
from .residual_direction_core import (
    SPECIALIST_NAMES, ACTION_NAMES, _fusion, _regions, _cross_fit,
)
from .residual_direction_gate import _train_gate, _predict_gate, _calibrate, _routing_stats

logger = logging.getLogger('MMSA')
REGION_NAMES = ('strong_negative', 'negative', 'boundary', 'positive', 'strong_positive')


def _candidate_diagnostics(cache, residuals, metrics_fn):
    labels, fusion = cache['labels'], _fusion(cache['expert_matrix'])
    predictions = torch.cat([fusion, fusion + residuals], dim=1)
    errors = torch.abs(predictions - labels)
    winner = errors.argmin(1)
    rows = []
    for index, name in enumerate(ACTION_NAMES):
        prediction = predictions[:, index:index + 1]
        gain = torch.abs(fusion - labels) - torch.abs(prediction - labels)
        delta = prediction - fusion
        rows.append({'candidate': name, **_safe_metrics(metrics_fn, prediction, labels),
                     'oracle_win_rate': float((winner == index).float().mean().item()),
                     'mean_gain_vs_fusion': float(gain.mean().item()),
                     'gain_over_005_rate': float((gain > 0.05).float().mean().item()),
                     'harm_over_005_rate': float((gain < -0.05).float().mean().item()),
                     'mean_residual': float(delta.mean().item()),
                     'mean_abs_residual': float(delta.abs().mean().item())})
    return rows, predictions.gather(1, winner.unsqueeze(1)), errors


def _region_diagnostics(cache, residuals):
    labels, fusion = cache['labels'], _fusion(cache['expert_matrix'])
    predictions = torch.cat([fusion, fusion + residuals], dim=1)
    groups, rows = _regions(labels), []
    for group_index, group_name in enumerate(REGION_NAMES):
        mask = groups == group_index
        if not mask.any():
            continue
        for candidate_index, candidate_name in enumerate(ACTION_NAMES):
            prediction = predictions[mask, candidate_index:candidate_index + 1]
            target = labels[mask]
            base = fusion[mask]
            gain = torch.abs(base - target) - torch.abs(prediction - target)
            rows.append({'region': group_name, 'candidate': candidate_name,
                         'count': int(mask.sum().item()),
                         'mae': float(torch.abs(prediction - target).mean().item()),
                         'mean_gain_vs_fusion': float(gain.mean().item()),
                         'gain_over_005_rate': float((gain > 0.05).float().mean().item()),
                         'harm_over_005_rate': float((gain < -0.05).float().mean().item())})
    return rows


def _save_report(save_dir, seed, folds, alphas, metrics_fn, test, test_residuals,
                 gate_result, train, oof, gate_best_epoch):
    save_dir = Path(save_dir)
    labels, fusion = test['labels'], _fusion(test['expert_matrix'])
    prediction = gate_result['prediction']
    correction = gate_result['correction']
    weights = gate_result['weights']
    specialist_rows, oracle, errors = _candidate_diagnostics(test, test_residuals, metrics_fn)
    fusion_metrics = _safe_metrics(metrics_fn, fusion, labels)
    routed_metrics = _safe_metrics(metrics_fn, prediction, labels)
    oracle_metrics = _safe_metrics(metrics_fn, oracle, labels)
    oracle_space = fusion_metrics['MAE'] - oracle_metrics['MAE']
    recovered = fusion_metrics['MAE'] - routed_metrics['MAE']
    summary = {
        'seed': int(seed),
        'oof_folds': int(folds),
        'gate_best_epoch': int(gate_best_epoch),
        'component_alphas': alphas,
        'fusion_metrics': fusion_metrics,
        'residual_router_metrics': routed_metrics,
        'specialist_oracle_metrics': oracle_metrics,
        **_routing_stats(fusion, prediction, labels, correction, weights),
        'oracle_gap_recovery_ratio': recovered / oracle_space if oracle_space > 1e-12 else 0.0,
    }
    with open(save_dir / 'residual_direction_summary.json', 'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)

    pd.DataFrame(specialist_rows).to_csv(save_dir / 'test_specialist_diagnostics.csv', index=False)
    pd.DataFrame(_region_diagnostics(test, test_residuals)).to_csv(
        save_dir / 'test_specialist_region_diagnostics.csv', index=False
    )
    corr = np.nan_to_num(np.corrcoef(errors.numpy(), rowvar=False), nan=0.0)
    pd.DataFrame(corr, index=ACTION_NAMES, columns=ACTION_NAMES).to_csv(
        save_dir / 'test_specialist_error_correlation.csv'
    )

    oof_rows, _, oof_errors = _candidate_diagnostics(train, oof, metrics_fn)
    pd.DataFrame(oof_rows).to_csv(save_dir / 'oof_specialist_diagnostics.csv', index=False)
    pd.DataFrame(_region_diagnostics(train, oof)).to_csv(
        save_dir / 'oof_specialist_region_diagnostics.csv', index=False
    )
    oof_corr = np.nan_to_num(np.corrcoef(oof_errors.numpy(), rowvar=False), nan=0.0)
    pd.DataFrame(oof_corr, index=ACTION_NAMES, columns=ACTION_NAMES).to_csv(
        save_dir / 'oof_specialist_error_correlation.csv'
    )

    sample_ids = test['sample_ids']
    if len(sample_ids) != len(labels):
        sample_ids = [str(i) for i in range(len(labels))]
    data = {
        'sample_id': sample_ids,
        'target': labels.view(-1).numpy(),
        'fusion_prediction': fusion.view(-1).numpy(),
        'residual_router_prediction': prediction.view(-1).numpy(),
        'correction': correction.view(-1).numpy(),
        'fusion_weight': weights[:, 0].numpy(),
    }
    for index, name in enumerate(SPECIALIST_NAMES):
        data[f'{name}_residual'] = test_residuals[:, index].numpy()
        data[f'{name}_weight'] = weights[:, index + 1].numpy()
        data[f'{name}_scaled_component'] = gate_result['components'][:, index].numpy()
    pd.DataFrame(data).to_csv(save_dir / 'residual_direction_predictions.csv', index=False)

    logger.info(
        'RESIDUAL-DIRECTION TEST fusion_MAE=%.4f routed_MAE=%.4f oracle_MAE=%.4f alphas=%s recovery=%.4f',
        fusion_metrics['MAE'], routed_metrics['MAE'], oracle_metrics['MAE'], alphas,
        summary['oracle_gap_recovery_ratio'],
    )
    return summary


def train_residual_direction_system(
    model, dataloader, metrics_fn, device, save_dir, seed, oof_folds=5,
    specialist_epochs=25, gate_epochs=60, gate_patience=8,
    specialist_learning_rate=8e-4, gate_learning_rate=5e-4,
    shared_dim=96, adapter_dim=48, gate_hidden_dim=64, dropout=0.15,
    max_residual=0.75, boundary_max_residual=0.35, batch_size=128,
    residual_margin=0.10, boundary_label_threshold=0.50,
    off_region_anchor_weight=0.01, sentiment_dro_weight=0.25,
    direction_dro_weight=0.25, harm_weight=1.5, direction_loss_weight=0.20,
    oracle_kl_weight=0.0, anchor_weight=0.05, correction_weight=0.01,
    sign_weight=0.05, calibration_alphas=None, calibration_folds=3,
    calibration_robustness_weight=0.50, calibration_harm_penalty=0.20,
    calibration_correction_penalty=0.02,
):
    if calibration_alphas is None:
        calibration_alphas = [0.0, 0.25, 0.50, 0.75, 1.0]
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    cached = {
        split: _cache_split(model, dataloader[split], device, split)
        for split in ('train', 'valid', 'test')
    }
    normalization = _normalize_cached(cached)

    specialist_kwargs = dict(
        epochs=specialist_epochs,
        learning_rate=specialist_learning_rate,
        shared_dim=shared_dim,
        adapter_dim=adapter_dim,
        dropout=dropout,
        max_residual=max_residual,
        boundary_max_residual=boundary_max_residual,
        batch_size=batch_size,
        residual_margin=residual_margin,
        boundary_threshold=boundary_label_threshold,
        off_anchor_weight=off_region_anchor_weight,
    )
    oof, valid_residuals, test_residuals, specialist_history = _cross_fit(
        cached['train'], cached['valid'], cached['test'], device, seed, oof_folds,
        save_dir, **specialist_kwargs
    )
    pd.DataFrame(specialist_history).to_csv(
        save_dir / 'specialist_fold_history.csv', index=False
    )
    torch.save(
        {
            'normalization': normalization,
            'specialist_names': list(SPECIALIST_NAMES),
            'fold_count': int(oof_folds),
        },
        save_dir / 'residual_direction_ensemble_metadata.pth',
    )

    gate_kwargs = dict(
        epochs=gate_epochs,
        learning_rate=gate_learning_rate,
        hidden_dim=gate_hidden_dim,
        dropout=dropout,
        batch_size=batch_size,
        patience=gate_patience,
        boundary_threshold=boundary_label_threshold,
        sentiment_dro_weight=sentiment_dro_weight,
        direction_dro_weight=direction_dro_weight,
        harm_weight=harm_weight,
        direction_loss_weight=direction_loss_weight,
        oracle_kl_weight=oracle_kl_weight,
        anchor_weight=anchor_weight,
        correction_weight=correction_weight,
        sign_weight=sign_weight,
        calibration_robustness_weight=calibration_robustness_weight,
        calibration_harm_penalty=calibration_harm_penalty,
        calibration_correction_penalty=calibration_correction_penalty,
    )
    gate, selection_history, final_history, best_epoch = _train_gate(
        cached['train'], oof, device, seed + 12001, residual_margin, **gate_kwargs
    )
    pd.DataFrame(selection_history).to_csv(
        save_dir / 'gate_selection_history.csv', index=False
    )
    pd.DataFrame(final_history).to_csv(
        save_dir / 'gate_final_history.csv', index=False
    )

    alphas, calibration_rows = _calibrate(
        gate, cached['valid'], valid_residuals, device, calibration_alphas,
        calibration_folds, seed + 15001, residual_margin,
        calibration_robustness_weight, calibration_harm_penalty,
        calibration_correction_penalty,
    )
    pd.DataFrame(calibration_rows).to_csv(
        save_dir / 'multi_scenario_alpha_calibration.csv', index=False
    )
    torch.save(
        {
            'state_dict': gate.state_dict(),
            'component_alphas': alphas,
            'best_epoch': int(best_epoch),
            'specialist_names': list(SPECIALIST_NAMES),
        },
        save_dir / 'residual_direction_gate.pth',
    )

    result = _predict_gate(gate, cached['test'], test_residuals, device, alphas)
    return gate, _save_report(
        save_dir, seed, oof_folds, alphas, metrics_fn, cached['test'],
        test_residuals, result, cached['train'], oof, best_epoch
    )
