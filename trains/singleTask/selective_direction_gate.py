import copy
import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from .residual_direction_core import (
    SPECIALIST_NAMES,
    _cross_fit,
    _fusion,
    _make_loader,
    _regions,
)
from .residual_direction_specialist import _candidate_diagnostics, _region_diagnostics
from .specialist_residual import _safe_metrics

logger = logging.getLogger('MMSA')
DIRECTION_NAMES = ('down', 'keep', 'up')
REGION_NAMES = ('strong_negative', 'negative', 'boundary', 'positive', 'strong_positive')
SCENARIOS = {
    'original': (1.0, 1.0, 1.0, 1.0, 1.0),
    'negative_heavy': (2.5, 1.8, 1.0, 0.8, 0.8),
    'positive_heavy': (0.8, 0.8, 1.0, 1.8, 2.5),
    'boundary_heavy': (1.0, 1.0, 2.5, 1.0, 1.0),
    'strong_heavy': (2.0, 1.0, 1.0, 1.0, 2.0),
}


class SelectiveDirectionClassifier(nn.Module):
    def __init__(self, context_dim, hidden_dim=64, dropout=0.15):
        super().__init__()
        input_dim = context_dim + len(SPECIALIST_NAMES)
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(DIRECTION_NAMES)),
        )

    def forward(self, context, residuals):
        return self.network(torch.cat([context, residuals], dim=1))


def _direction_target(labels, fusion, margin):
    residual = (labels - fusion).view(-1)
    target = torch.full_like(residual, 1, dtype=torch.long)
    target[residual < -margin] = 0
    target[residual > margin] = 2
    return target


def _strata(labels, fusion, margin):
    return _regions(labels) * len(DIRECTION_NAMES) + _direction_target(labels, fusion, margin)


def _stratified_folds(labels, fusion, fold_count, seed, margin):
    groups = _strata(labels, fusion, margin)
    generator = torch.Generator().manual_seed(int(seed))
    folds = [[] for _ in range(fold_count)]
    for group in torch.unique(groups).tolist():
        indices = torch.nonzero(groups == group, as_tuple=False).view(-1)
        indices = indices[torch.randperm(len(indices), generator=generator)]
        for position, sample_index in enumerate(indices.tolist()):
            folds[position % fold_count].append(sample_index)
    return [torch.tensor(sorted(fold), dtype=torch.long) for fold in folds if fold]


def _class_weights(target):
    counts = torch.bincount(target, minlength=len(DIRECTION_NAMES)).float().clamp_min(1.0)
    weights = counts.sum() / torch.sqrt(counts)
    return weights / weights.mean()


def _balanced_accuracy(target, prediction):
    recalls = []
    for index in range(len(DIRECTION_NAMES)):
        mask = target == index
        if mask.any():
            recalls.append((prediction[mask] == index).float().mean())
    return float(torch.stack(recalls).mean().item()) if recalls else 0.0


def _macro_direction_metrics(logits, target):
    probabilities = F.softmax(logits, dim=1)
    prediction = probabilities.argmax(dim=1)
    result = {
        'accuracy': float((prediction == target).float().mean().item()),
        'balanced_accuracy': _balanced_accuracy(target, prediction),
        'mean_confidence': float(probabilities.max(dim=1).values.mean().item()),
    }
    for index, name in enumerate(DIRECTION_NAMES):
        mask = target == index
        result[f'{name}_recall'] = float((prediction[mask] == index).float().mean().item()) if mask.any() else 0.0
    up_down = target != 1
    if up_down.any():
        binary_prediction = torch.where(
            probabilities[:, 2] >= probabilities[:, 0],
            torch.full_like(target, 2),
            torch.zeros_like(target),
        )
        result['up_down_accuracy'] = float((binary_prediction[up_down] == target[up_down]).float().mean().item())
    else:
        result['up_down_accuracy'] = 0.0
    return result


def _predict_logits(model, context, residuals, device, batch_size=512):
    model.eval()
    outputs = []
    loader = DataLoader(TensorDataset(context, residuals), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_context, batch_residuals in loader:
            outputs.append(model(batch_context.to(device), batch_residuals.to(device)).cpu())
    return torch.cat(outputs, dim=0)


def _train_once(
    context,
    residuals,
    expert_matrix,
    labels,
    device,
    seed,
    residual_margin,
    epochs,
    learning_rate,
    hidden_dim,
    dropout,
    batch_size,
    validation_indices=None,
    patience=8,
    label_smoothing=0.05,
):
    model = SelectiveDirectionClassifier(context.size(1), hidden_dim, dropout).to(device)
    fusion = _fusion(expert_matrix)
    target = _direction_target(labels, fusion, residual_margin)
    weights = _class_weights(target).to(device)
    all_indices = torch.arange(len(labels))
    train_indices = all_indices
    if validation_indices is not None:
        mask = torch.ones(len(labels), dtype=torch.bool)
        mask[validation_indices] = False
        train_indices = all_indices[mask]
    loader = _make_loader(
        context[train_indices], residuals[train_indices], target[train_indices],
        batch_size=batch_size, shuffle=True, seed=seed,
    )
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history, best_state, best_epoch, best_score = [], None, 0, -float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_context, batch_residuals, batch_target in loader:
            optimizer.zero_grad()
            logits = model(batch_context.to(device), batch_residuals.to(device))
            loss = F.cross_entropy(
                logits,
                batch_target.to(device),
                weight=weights,
                label_smoothing=label_smoothing,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
        row = {'epoch': epoch, 'train_loss': total_loss / max(1, len(loader))}
        if validation_indices is not None:
            logits = _predict_logits(
                model, context[validation_indices], residuals[validation_indices], device
            )
            metrics = _macro_direction_metrics(logits, target[validation_indices])
            row.update({f'valid_{key}': value for key, value in metrics.items()})
            score = metrics['balanced_accuracy']
            if score > best_score + 1e-6:
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                best_score = score
            if epoch - best_epoch >= patience:
                history.append(row)
                break
        history.append(row)
    if validation_indices is None:
        return model, history, epochs
    if best_state is None:
        raise RuntimeError('Direction classifier did not produce a checkpoint.')
    model.load_state_dict(best_state)
    return model, history, best_epoch


def train_direction_classifier(
    train_cache,
    oof_residuals,
    device,
    seed,
    residual_margin=0.10,
    epochs=60,
    learning_rate=5e-4,
    hidden_dim=64,
    dropout=0.15,
    batch_size=128,
    patience=8,
    label_smoothing=0.05,
):
    fusion = _fusion(train_cache['expert_matrix'])
    folds = _stratified_folds(
        train_cache['labels'], fusion, 5, seed + 2718, residual_margin
    )
    validation_indices = folds[0]
    _, selection_history, best_epoch = _train_once(
        train_cache['context'], oof_residuals, train_cache['expert_matrix'],
        train_cache['labels'], device, seed, residual_margin, epochs,
        learning_rate, hidden_dim, dropout, batch_size,
        validation_indices=validation_indices, patience=patience,
        label_smoothing=label_smoothing,
    )
    model, final_history, _ = _train_once(
        train_cache['context'], oof_residuals, train_cache['expert_matrix'],
        train_cache['labels'], device, seed + 1, residual_margin, max(1, best_epoch),
        learning_rate, hidden_dim, dropout, batch_size,
        validation_indices=None, patience=patience,
        label_smoothing=label_smoothing,
    )
    return model, selection_history, final_history, best_epoch


def _apply_selective_policy(
    logits,
    residuals,
    fusion,
    threshold_up,
    threshold_down,
    score_margin,
    alpha_up,
    alpha_down,
):
    probabilities = F.softmax(logits, dim=1)
    down_probability = probabilities[:, 0]
    up_probability = probabilities[:, 2]
    choose_up = (
        (up_probability >= threshold_up)
        & ((up_probability - down_probability) >= score_margin)
    )
    choose_down = (
        (down_probability >= threshold_down)
        & ((down_probability - up_probability) >= score_margin)
    )
    correction = torch.zeros_like(fusion)
    correction[choose_up] = float(alpha_up) * residuals[choose_up, 0:1]
    correction[choose_down] = float(alpha_down) * residuals[choose_down, 1:2]
    action = torch.full((len(fusion),), 1, dtype=torch.long)
    action[choose_down] = 0
    action[choose_up] = 2
    return {
        'prediction': fusion + correction,
        'correction': correction,
        'probabilities': probabilities,
        'action': action,
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
        factor = torch.tensor(factors, dtype=error.dtype)
        weight = factor[regions]
        weight = weight / weight.sum().clamp_min(1e-12)
        rows[name] = {
            'mae': float((weight * error).sum().item()),
            'harm_005': float((weight * (gain < -0.05).float()).sum().item()),
            'harm_010': float((weight * (gain < -0.10).float()).sum().item()),
            'mean_abs_correction': float((weight * correction.abs().view(-1)).sum().item()),
        }
    return rows


def _selection_stats(fusion, prediction, labels, correction, action):
    gain = torch.abs(fusion - labels).view(-1) - torch.abs(prediction - labels).view(-1)
    selected = action != 1
    result = {
        'mae': float(torch.abs(prediction - labels).mean().item()),
        'mean_realized_gain': float(gain.mean().item()),
        'correction_rate': float(selected.float().mean().item()),
        'mean_abs_correction': float(correction.abs().mean().item()),
        'harm_over_005_rate': float((gain < -0.05).float().mean().item()),
        'harm_over_010_rate': float((gain < -0.10).float().mean().item()),
        'correction_precision': float((gain[selected] > 0).float().mean().item()) if selected.any() else 1.0,
    }
    for index, name in ((0, 'down'), (2, 'up')):
        mask = action == index
        result[f'{name}_coverage'] = float(mask.float().mean().item())
        result[f'{name}_precision'] = float((gain[mask] > 0).float().mean().item()) if mask.any() else 1.0
    return result


def calibrate_selective_policy(
    classifier,
    valid_cache,
    valid_residuals,
    device,
    residual_margin,
    threshold_values=None,
    score_margins=None,
    alpha_values=None,
    fold_count=3,
    robustness_weight=0.50,
    harm_penalty=0.20,
    precision_penalty=0.05,
    target_precision=0.65,
    correction_penalty=0.02,
):
    threshold_values = threshold_values or [0.50, 0.60, 0.70, 0.80, 0.90]
    score_margins = score_margins or [0.0, 0.10, 0.20]
    alpha_values = alpha_values or [0.0, 0.25, 0.50, 0.75, 1.0]
    logits = _predict_logits(classifier, valid_cache['context'], valid_residuals, device)
    fusion = _fusion(valid_cache['expert_matrix'])
    folds = _stratified_folds(
        valid_cache['labels'], fusion, fold_count, 19001, residual_margin
    )
    rows = []
    for threshold_up, threshold_down, score_margin, alpha_up, alpha_down in itertools.product(
        threshold_values, threshold_values, score_margins, alpha_values, alpha_values
    ):
        result = _apply_selective_policy(
            logits, valid_residuals, fusion, threshold_up, threshold_down,
            score_margin, alpha_up, alpha_down,
        )
        scenario_records = []
        for fold_index, indices in enumerate(folds, start=1):
            scenarios = _scenario_rows(
                fusion[indices], result['prediction'][indices], valid_cache['labels'][indices],
                result['correction'][indices],
            )
            scenario_records.extend(
                {'fold': fold_index, 'scenario': name, **values}
                for name, values in scenarios.items()
            )
        mean_mae = float(np.mean([row['mae'] for row in scenario_records]))
        worst_mae = float(np.max([row['mae'] for row in scenario_records]))
        worst_harm = float(np.max([row['harm_005'] for row in scenario_records]))
        mean_correction = float(np.mean([row['mean_abs_correction'] for row in scenario_records]))
        stats = _selection_stats(
            fusion, result['prediction'], valid_cache['labels'], result['correction'], result['action']
        )
        objective = (
            mean_mae
            + robustness_weight * (worst_mae - mean_mae)
            + harm_penalty * worst_harm
            + precision_penalty * max(0.0, target_precision - stats['correction_precision'])
            + correction_penalty * mean_correction
        )
        rows.append({
            'threshold_up': float(threshold_up),
            'threshold_down': float(threshold_down),
            'score_margin': float(score_margin),
            'alpha_up': float(alpha_up),
            'alpha_down': float(alpha_down),
            'objective': float(objective),
            'cv_mean_mae': mean_mae,
            'cv_worst_mae': worst_mae,
            'cv_worst_harm_005': worst_harm,
            **stats,
        })
    best = min(
        rows,
        key=lambda row: (
            row['objective'], row['cv_worst_mae'], row['harm_over_005_rate'],
            -row['correction_precision'], row['correction_rate'],
        ),
    )
    policy = {
        key: float(best[key])
        for key in ('threshold_up', 'threshold_down', 'score_margin', 'alpha_up', 'alpha_down')
    }
    return policy, rows, logits


def run_selective_direction_system(
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
    classifier_epochs=60,
    classifier_learning_rate=5e-4,
    classifier_hidden_dim=64,
    classifier_dropout=0.15,
    classifier_patience=8,
    label_smoothing=0.05,
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
    oof_residuals, valid_residuals, test_residuals, specialist_history = _cross_fit(
        caches['train'], caches['valid'], caches['test'], device, seed,
        specialist_folds, save_dir, **specialist_kwargs
    )
    pd.DataFrame(specialist_history).to_csv(
        save_dir / 'specialist_fold_history.csv', index=False
    )

    classifier, selection_history, final_history, best_epoch = train_direction_classifier(
        caches['train'], oof_residuals, device, seed + 12001,
        residual_margin=residual_margin,
        epochs=classifier_epochs,
        learning_rate=classifier_learning_rate,
        hidden_dim=classifier_hidden_dim,
        dropout=classifier_dropout,
        batch_size=batch_size,
        patience=classifier_patience,
        label_smoothing=label_smoothing,
    )
    pd.DataFrame(selection_history).to_csv(
        save_dir / 'direction_classifier_selection_history.csv', index=False
    )
    pd.DataFrame(final_history).to_csv(
        save_dir / 'direction_classifier_final_history.csv', index=False
    )

    policy, calibration_rows, _ = calibrate_selective_policy(
        classifier, caches['valid'], valid_residuals, device, residual_margin,
        fold_count=calibration_folds,
    )
    pd.DataFrame(calibration_rows).to_csv(
        save_dir / 'selective_policy_calibration.csv', index=False
    )
    torch.save(
        {
            'state_dict': classifier.state_dict(),
            'policy': policy,
            'best_epoch': int(best_epoch),
            'direction_names': list(DIRECTION_NAMES),
        },
        save_dir / 'selective_direction_classifier.pth',
    )

    test_logits = _predict_logits(
        classifier, caches['test']['context'], test_residuals, device
    )
    fusion = _fusion(caches['test']['expert_matrix'])
    result = _apply_selective_policy(
        test_logits, test_residuals, fusion, **policy
    )
    labels = caches['test']['labels']
    fusion_metrics = _safe_metrics(metrics_fn, fusion, labels)
    routed_metrics = _safe_metrics(metrics_fn, result['prediction'], labels)
    specialist_rows, oracle, errors = _candidate_diagnostics(
        caches['test'], test_residuals, metrics_fn
    )
    oracle_metrics = _safe_metrics(metrics_fn, oracle, labels)
    stats = _selection_stats(
        fusion, result['prediction'], labels, result['correction'], result['action']
    )
    target = _direction_target(labels, fusion, residual_margin)
    direction_metrics = _macro_direction_metrics(test_logits, target)
    oracle_space = fusion_metrics['MAE'] - oracle_metrics['MAE']
    recovered = fusion_metrics['MAE'] - routed_metrics['MAE']
    summary = {
        'seed': int(seed),
        'backbone_mode': 'true_outer_crossfit',
        'classifier_best_epoch': int(best_epoch),
        'selective_policy': policy,
        'fusion_metrics': fusion_metrics,
        'residual_router_metrics': routed_metrics,
        'specialist_oracle_metrics': oracle_metrics,
        **stats,
        **{f'direction_{key}': value for key, value in direction_metrics.items()},
        'oracle_gap_recovery_ratio': recovered / oracle_space if oracle_space > 1e-12 else 0.0,
    }
    with open(save_dir / 'backbone_crossfit_direction_summary.json', 'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)

    pd.DataFrame(specialist_rows).to_csv(
        save_dir / 'test_specialist_diagnostics.csv', index=False
    )
    pd.DataFrame(_region_diagnostics(caches['test'], test_residuals)).to_csv(
        save_dir / 'test_specialist_region_diagnostics.csv', index=False
    )
    correlation = np.nan_to_num(np.corrcoef(errors.numpy(), rowvar=False), nan=0.0)
    action_names = ('fusion',) + SPECIALIST_NAMES
    pd.DataFrame(correlation, index=action_names, columns=action_names).to_csv(
        save_dir / 'test_specialist_error_correlation.csv'
    )

    prediction_action = result['action']
    confusion = torch.zeros(len(DIRECTION_NAMES), len(DIRECTION_NAMES), dtype=torch.long)
    for true_value, predicted_value in zip(target.tolist(), prediction_action.tolist()):
        confusion[true_value, predicted_value] += 1
    pd.DataFrame(confusion.numpy(), index=DIRECTION_NAMES, columns=DIRECTION_NAMES).to_csv(
        save_dir / 'test_direction_confusion_matrix.csv'
    )
    probabilities = result['probabilities']
    sample_ids = caches['test']['sample_ids']
    if len(sample_ids) != len(labels):
        sample_ids = [str(index) for index in range(len(labels))]
    prediction_data = {
        'sample_id': sample_ids,
        'target': labels.view(-1).numpy(),
        'fusion_prediction': fusion.view(-1).numpy(),
        'router_prediction': result['prediction'].view(-1).numpy(),
        'correction': result['correction'].view(-1).numpy(),
        'true_direction': [DIRECTION_NAMES[index] for index in target.tolist()],
        'selected_action': [DIRECTION_NAMES[index] for index in prediction_action.tolist()],
        'prob_down': probabilities[:, 0].numpy(),
        'prob_keep': probabilities[:, 1].numpy(),
        'prob_up': probabilities[:, 2].numpy(),
        'up_residual': test_residuals[:, 0].numpy(),
        'down_residual': test_residuals[:, 1].numpy(),
        'boundary_residual': test_residuals[:, 2].numpy(),
    }
    pd.DataFrame(prediction_data).to_csv(
        save_dir / 'backbone_crossfit_direction_predictions.csv', index=False
    )
    logger.info(
        'BACKBONE-CROSSFIT TEST fusion_MAE=%.4f routed_MAE=%.4f oracle_MAE=%.4f policy=%s direction_bal_acc=%.4f precision=%.4f coverage=%.4f',
        fusion_metrics['MAE'], routed_metrics['MAE'], oracle_metrics['MAE'], policy,
        direction_metrics['balanced_accuracy'], stats['correction_precision'], stats['correction_rate'],
    )
    return summary
