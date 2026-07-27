import copy
import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim

from .specialist_residual import FUSION_INDEX, _cache_split, _normalize_cached, _safe_metrics
from .residual_direction_core import (
    SPECIALIST_NAMES, ACTION_NAMES, REGION_NAMES, ResidualDirectionGate,
    _fusion, _regions, _directions, _make_loader, _build_folds,
    _cross_fit, _gate_loss,
)

logger = logging.getLogger('MMSA')
SCENARIOS = {
    'original': (1.0, 1.0, 1.0, 1.0, 1.0),
    'negative_heavy': (2.5, 1.8, 1.0, 0.8, 0.8),
    'positive_heavy': (0.8, 0.8, 1.0, 1.8, 2.5),
    'boundary_heavy': (1.0, 1.0, 2.5, 1.0, 1.0),
    'strong_heavy': (2.0, 1.0, 1.0, 1.0, 2.0),
}


def _scenario_rows(fusion, prediction, labels, correction):
    error = torch.abs(prediction - labels).view(-1)
    gain = torch.abs(fusion - labels).view(-1) - error
    regions = _regions(labels)
    counts = torch.bincount(regions, minlength=len(REGION_NAMES)).to(error.dtype)
    scenarios = dict(SCENARIOS)
    scenarios['balanced'] = tuple((counts.sum() / counts.clamp_min(1.0)).tolist())
    rows = {}
    for name, values in scenarios.items():
        factors = torch.tensor(values, dtype=error.dtype, device=error.device)
        weights = factors[regions]
        weights = weights / weights.sum().clamp_min(1e-12)
        rows[name] = {
            'mae': float((weights * error).sum().item()),
            'harm_005': float((weights * (gain < -0.05).float()).sum().item()),
            'mean_abs_correction': float((weights * correction.abs().view(-1)).sum().item()),
        }
    return rows


def _scenario_objective(rows, robustness_weight, harm_penalty, correction_penalty):
    maes = [row['mae'] for row in rows.values()]
    return (float(np.mean(maes)) + robustness_weight * (float(np.max(maes)) - float(np.mean(maes)))
            + harm_penalty * max(row['harm_005'] for row in rows.values())
            + correction_penalty * float(np.mean([row['mean_abs_correction'] for row in rows.values()])))


def _routing_stats(fusion, prediction, labels, correction, weights=None):
    error, fusion_error = torch.abs(prediction - labels), torch.abs(fusion - labels)
    gain = fusion_error - error
    result = {
        'mae': float(error.mean().item()), 'mean_realized_gain': float(gain.mean().item()),
        'positive_gain_rate': float((gain > 0).float().mean().item()),
        'harm_over_005_rate': float((gain < -0.05).float().mean().item()),
        'harm_over_010_rate': float((gain < -0.10).float().mean().item()),
        'correction_rate': float((correction.abs() > 0.01).float().mean().item()),
        'mean_abs_correction': float(correction.abs().mean().item()),
    }
    region_values = []
    regions = _regions(labels)
    for index, name in enumerate(REGION_NAMES):
        mask = regions == index
        if mask.any():
            value = float(error.view(-1)[mask].mean().item())
            result[f'{name}_mae'] = value
            region_values.append(value)
    result['balanced_group_mae'] = float(np.mean(region_values))
    result['worst_group_mae'] = float(np.max(region_values))
    if weights is not None:
        top = weights.argmax(1)
        for index, name in enumerate(ACTION_NAMES):
            result[f'average_{name}_weight'] = float(weights[:, index].mean().item())
            result[f'top_{name}_rate'] = float((top == index).float().mean().item())
    return result


def _train_gate_once(context, residuals, experts, labels, device, seed, epochs,
                     learning_rate, hidden_dim, dropout, batch_size, residual_margin,
                     validation_indices=None, patience=8, calibration_robustness_weight=0.5,
                     calibration_harm_penalty=0.2, calibration_correction_penalty=0.02,
                     **loss_kwargs):
    gate = ResidualDirectionGate(context.size(1), hidden_dim, dropout).to(device)
    optimizer = optim.AdamW(gate.parameters(), lr=learning_rate, weight_decay=1e-4)
    all_indices = torch.arange(len(labels))
    train_indices = all_indices
    if validation_indices is not None:
        mask = torch.ones(len(labels), dtype=torch.bool)
        mask[validation_indices] = False
        train_indices = all_indices[mask]
    loader = _make_loader(context[train_indices], residuals[train_indices], experts[train_indices],
                          labels[train_indices], batch_size=batch_size, shuffle=True, seed=seed)
    best_state, best_epoch, best_objective, history = None, 0, float('inf'), []
    for epoch in range(1, epochs + 1):
        gate.train()
        totals = {key: 0.0 for key in ('loss', 'mean_l1', 'sentiment_gap', 'direction_gap',
                                        'harm_loss', 'direction_loss', 'oracle_kl',
                                        'anchor_loss', 'correction_loss', 'sign_loss')}
        for batch_context, batch_residuals, batch_experts, batch_labels in loader:
            batch_context, batch_residuals, batch_experts, batch_labels = (
                batch_context.to(device), batch_residuals.to(device),
                batch_experts.to(device), batch_labels.to(device)
            )
            optimizer.zero_grad()
            output = gate(batch_context, batch_residuals)
            loss, parts = _gate_loss(output, batch_residuals, _fusion(batch_experts),
                                     batch_labels, residual_margin=residual_margin, **loss_kwargs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 5.0)
            optimizer.step()
            totals['loss'] += loss.item()
            for key, value in parts.items():
                totals[key] += value.item()
        record = {'epoch': epoch, **{k: v / max(1, len(loader)) for k, v in totals.items()}}
        if validation_indices is not None:
            gate.eval()
            with torch.no_grad():
                c = context[validation_indices].to(device)
                r = residuals[validation_indices].to(device)
                f = _fusion(experts[validation_indices]).to(device)
                y = labels[validation_indices].to(device)
                output = gate(c, r)
                pred = f + output['correction']
                scenario = _scenario_rows(f, pred, y, output['correction'])
                objective = _scenario_objective(scenario, calibration_robustness_weight,
                                                calibration_harm_penalty,
                                                calibration_correction_penalty)
                stats = _routing_stats(f, pred, y, output['correction'], output['weights'])
            record['valid_objective'] = objective
            record.update({f'valid_{k}': v for k, v in stats.items()})
            if objective < best_objective - 1e-6:
                best_state, best_epoch, best_objective = copy.deepcopy(gate.state_dict()), epoch, objective
            if epoch - best_epoch >= patience:
                history.append(record)
                break
        history.append(record)
    if validation_indices is None:
        return gate, history, epochs
    if best_state is None:
        raise RuntimeError('Gate training produced no checkpoint.')
    gate.load_state_dict(best_state)
    return gate, history, best_epoch


def _train_gate(train, oof, device, seed, residual_margin, **kwargs):
    folds = _build_folds(train['labels'], train['expert_matrix'], 5, seed + 2718, residual_margin)
    validation_indices = folds[0]
    _, selection_history, best_epoch = _train_gate_once(
        train['context'], oof, train['expert_matrix'], train['labels'], device, seed,
        residual_margin=residual_margin, validation_indices=validation_indices, **kwargs
    )
    final = dict(kwargs)
    final['epochs'] = max(1, best_epoch)
    gate, final_history, _ = _train_gate_once(
        train['context'], oof, train['expert_matrix'], train['labels'], device, seed + 1,
        residual_margin=residual_margin, validation_indices=None,
        patience=final.pop('patience', 8), **final
    )
    return gate, selection_history, final_history, best_epoch


def _predict_gate(gate, cache, residuals, device, alphas):
    gate.eval()
    with torch.no_grad():
        output = gate(cache['context'].to(device), residuals.to(device))
        alpha = torch.tensor([alphas[name] for name in SPECIALIST_NAMES],
                             dtype=output['components'].dtype, device=device).view(1, -1)
        components = output['components'] * alpha
        correction = components.sum(1, keepdim=True)
        prediction = _fusion(cache['expert_matrix']).to(device) + correction
    return {'prediction': prediction.cpu(), 'correction': correction.cpu(),
            'components': components.cpu(), 'weights': output['weights'].cpu()}


def _calibrate(gate, valid, residuals, device, alpha_values, fold_count, seed,
               residual_margin, robustness_weight, harm_penalty, correction_penalty):
    folds = _build_folds(valid['labels'], valid['expert_matrix'], fold_count, seed, residual_margin)
    rows = []
    for values in itertools.product(alpha_values, repeat=len(SPECIALIST_NAMES)):
        alphas = dict(zip(SPECIALIST_NAMES, values))
        result = _predict_gate(gate, valid, residuals, device, alphas)
        fold_rows = []
        for fold_index, indices in enumerate(folds):
            scenarios = _scenario_rows(_fusion(valid['expert_matrix'][indices]),
                                       result['prediction'][indices], valid['labels'][indices],
                                       result['correction'][indices])
            fold_rows.extend({'fold': fold_index + 1, 'scenario': name, **row}
                             for name, row in scenarios.items())
        maes = [row['mae'] for row in fold_rows]
        harms = [row['harm_005'] for row in fold_rows]
        corrections = [row['mean_abs_correction'] for row in fold_rows]
        mean_mae, worst_mae = float(np.mean(maes)), float(np.max(maes))
        objective = (mean_mae + robustness_weight * (worst_mae - mean_mae)
                     + harm_penalty * float(np.max(harms))
                     + correction_penalty * float(np.mean(corrections)))
        stats = _routing_stats(_fusion(valid['expert_matrix']), result['prediction'],
                               valid['labels'], result['correction'], result['weights'])
        rows.append({**{f'alpha_{k}': float(v) for k, v in alphas.items()},
                     'objective': objective, 'cv_mean_mae': mean_mae,
                     'cv_worst_mae': worst_mae, 'cv_worst_harm_005': float(np.max(harms)),
                     **stats})
    best = min(rows, key=lambda row: (row['objective'], row['cv_worst_mae'],
                                      row['harm_over_005_rate'], row['mean_abs_correction']))
    return {name: float(best[f'alpha_{name}']) for name in SPECIALIST_NAMES}, rows
