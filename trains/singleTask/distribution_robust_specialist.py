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

from .specialist_residual import (
    FUSION_INDEX,
    _cache_split,
    _normalize_cached,
    _safe_metrics,
    _sign_loss,
)

logger = logging.getLogger('MMSA')

SPECIALIST_NAMES = (
    'positive',
    'negative',
    'boundary',
    'strong_positive',
    'strong_negative',
)
TRUE_REGION_NAMES = (
    'strong_negative',
    'negative',
    'boundary',
    'positive',
    'strong_positive',
)
PREDICTED_ALPHA_REGIONS = ('negative', 'boundary', 'positive')


class DistributionRobustSpecialists(nn.Module):
    """Hard-region residual specialists with independent adapters and sign constraints."""

    def __init__(
        self,
        context_dim,
        shared_dim=96,
        adapter_dim=48,
        dropout=0.15,
        max_residual=0.75,
        boundary_max_residual=0.30,
    ):
        super().__init__()
        self.max_residual = float(max_residual)
        self.boundary_max_residual = float(boundary_max_residual)
        self.shared = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.adapters = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(shared_dim, adapter_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(adapter_dim, adapter_dim),
                nn.GELU(),
                nn.LayerNorm(adapter_dim),
                nn.Dropout(dropout),
            )
            for name in SPECIALIST_NAMES
        })
        self.heads = nn.ModuleDict({name: nn.Linear(adapter_dim, 1) for name in SPECIALIST_NAMES})
        with torch.no_grad():
            for name in ('positive', 'negative', 'strong_positive', 'strong_negative'):
                self.heads[name].bias.fill_(-3.0)

    def forward(self, context):
        shared = self.shared(context)
        raw = {
            name: self.heads[name](self.adapters[name](shared))
            for name in SPECIALIST_NAMES
        }
        outputs = [
            self.max_residual * torch.sigmoid(raw['positive']),
            -self.max_residual * torch.sigmoid(raw['negative']),
            self.boundary_max_residual * torch.tanh(raw['boundary']),
            self.max_residual * torch.sigmoid(raw['strong_positive']),
            -self.max_residual * torch.sigmoid(raw['strong_negative']),
        ]
        return torch.cat(outputs, dim=1)


class DistributionRobustGate(nn.Module):
    """Fusion-anchored soft gate trained with group-robust objectives."""

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
            nn.Linear(hidden_dim, len(SPECIALIST_NAMES) + 1),
        )
        with torch.no_grad():
            self.network[-1].bias.zero_()
            self.network[-1].bias[0] = 2.0

    def forward(self, context, residuals):
        logits = self.network(torch.cat([context, residuals], dim=1))
        weights = F.softmax(logits, dim=1)
        correction = torch.sum(weights[:, 1:] * residuals, dim=1, keepdim=True)
        return {'logits': logits, 'weights': weights, 'correction': correction}


def _true_region_index(labels):
    values = labels.view(-1)
    result = torch.full_like(values, 2, dtype=torch.long)
    result[values < -1.5] = 0
    result[(values >= -1.5) & (values < -0.5)] = 1
    result[(values > 0.5) & (values <= 1.5)] = 3
    result[values > 1.5] = 4
    return result


def _predicted_alpha_region_index(fusion_prediction):
    values = fusion_prediction.view(-1)
    result = torch.ones_like(values, dtype=torch.long)
    result[values < -0.5] = 0
    result[values > 0.5] = 2
    return result


def _hard_specialist_masks(labels):
    values = labels.view(-1)
    return torch.stack([
        (values > 0.5) & (values <= 1.5),
        (values >= -1.5) & (values < -0.5),
        torch.abs(values) <= 0.5,
        values > 1.5,
        values < -1.5,
    ], dim=1)


def _make_loader(*tensors, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def _specialist_loss(
    residuals,
    fusion,
    labels,
    off_region_anchor_weight=0.10,
    sign_loss_weight=0.05,
):
    masks = _hard_specialist_masks(labels)
    candidate = fusion + residuals
    target = labels.expand_as(candidate)
    per_sample = F.smooth_l1_loss(candidate, target, reduction='none')
    head_losses = []
    for index in range(len(SPECIALIST_NAMES)):
        active = masks[:, index]
        if active.any():
            head_losses.append(per_sample[active, index].mean())
    task_loss = torch.stack(head_losses).mean() if head_losses else per_sample.mean()

    off_region = ~masks
    off_anchor = residuals[off_region].pow(2).mean() if off_region.any() else residuals.new_tensor(0.0)

    sign_losses = []
    for index in range(len(SPECIALIST_NAMES)):
        active = masks[:, index]
        if active.any():
            sign_losses.append(_sign_loss(candidate[active, index:index + 1], labels[active]))
    sign_loss = torch.stack(sign_losses).mean() if sign_losses else residuals.new_tensor(0.0)

    total = task_loss + off_region_anchor_weight * off_anchor + sign_loss_weight * sign_loss
    return total, {
        'task_loss': task_loss.detach(),
        'off_region_anchor': off_anchor.detach(),
        'sign_loss': sign_loss.detach(),
    }


def _train_specialist_model(
    context,
    expert_matrix,
    labels,
    device,
    seed,
    epochs,
    learning_rate,
    shared_dim,
    adapter_dim,
    dropout,
    max_residual,
    boundary_max_residual,
    batch_size,
    off_region_anchor_weight,
):
    model = DistributionRobustSpecialists(
        context_dim=context.size(1),
        shared_dim=shared_dim,
        adapter_dim=adapter_dim,
        dropout=dropout,
        max_residual=max_residual,
        boundary_max_residual=boundary_max_residual,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loader = _make_loader(
        context,
        expert_matrix,
        labels,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {'loss': 0.0, 'task_loss': 0.0, 'off_region_anchor': 0.0, 'sign_loss': 0.0}
        for batch_context, batch_experts, batch_labels in loader:
            batch_context = batch_context.to(device)
            batch_experts = batch_experts.to(device)
            batch_labels = batch_labels.to(device)
            fusion = batch_experts[:, FUSION_INDEX:FUSION_INDEX + 1]
            optimizer.zero_grad()
            residuals = model(batch_context)
            loss, parts = _specialist_loss(
                residuals,
                fusion,
                batch_labels,
                off_region_anchor_weight=off_region_anchor_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals['loss'] += loss.item()
            for key, value in parts.items():
                totals[key] += value.item()
        history.append({
            'epoch': epoch,
            **{key: value / max(1, len(loader)) for key, value in totals.items()},
        })
    return model, history


def _predict_specialists(model, context, device, batch_size=512):
    model.eval()
    outputs = []
    loader = DataLoader(TensorDataset(context), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (batch_context,) in loader:
            outputs.append(model(batch_context.to(device)).cpu())
    return torch.cat(outputs, dim=0)


def _build_stratified_folds(labels, fold_count, seed):
    if fold_count < 2:
        raise ValueError('oof_folds must be at least 2.')
    region_index = _true_region_index(labels)
    generator = torch.Generator().manual_seed(int(seed))
    folds = [[] for _ in range(fold_count)]
    for region in range(len(TRUE_REGION_NAMES)):
        indices = torch.nonzero(region_index == region, as_tuple=False).view(-1)
        if not len(indices):
            continue
        indices = indices[torch.randperm(len(indices), generator=generator)]
        for position, sample_index in enumerate(indices.tolist()):
            folds[position % fold_count].append(sample_index)
    result = [torch.tensor(sorted(fold), dtype=torch.long) for fold in folds if fold]
    if len(result) < 2:
        raise RuntimeError('Stratified fold construction produced fewer than two folds.')
    return result


def _cross_fit_specialists(
    train_cache,
    valid_cache,
    test_cache,
    device,
    seed,
    fold_count,
    save_dir,
    **train_kwargs,
):
    folds = _build_stratified_folds(train_cache['labels'], fold_count, seed)
    sample_count = len(train_cache['labels'])
    oof_residuals = torch.zeros(sample_count, len(SPECIALIST_NAMES), dtype=train_cache['context'].dtype)
    valid_predictions = []
    test_predictions = []
    fold_history = []
    all_indices = torch.arange(sample_count)

    for fold_index, holdout_indices in enumerate(folds):
        mask = torch.ones(sample_count, dtype=torch.bool)
        mask[holdout_indices] = False
        train_indices = all_indices[mask]
        logger.info(
            'Training distribution-robust specialist fold %d/%d on %d samples; holding out %d.',
            fold_index + 1,
            len(folds),
            len(train_indices),
            len(holdout_indices),
        )
        fold_model, history = _train_specialist_model(
            context=train_cache['context'][train_indices],
            expert_matrix=train_cache['expert_matrix'][train_indices],
            labels=train_cache['labels'][train_indices],
            device=device,
            seed=seed + 1009 * (fold_index + 1),
            **train_kwargs,
        )
        oof_residuals[holdout_indices] = _predict_specialists(
            fold_model, train_cache['context'][holdout_indices], device
        )
        valid_predictions.append(_predict_specialists(fold_model, valid_cache['context'], device))
        test_predictions.append(_predict_specialists(fold_model, test_cache['context'], device))
        for row in history:
            fold_history.append({'fold': fold_index + 1, **row})
        torch.save(
            {
                'state_dict': fold_model.state_dict(),
                'context_dim': int(train_cache['context'].size(1)),
                'specialist_names': list(SPECIALIST_NAMES),
                'fold': fold_index + 1,
            },
            Path(save_dir) / f'specialist_fold_{fold_index + 1}.pth',
        )
        del fold_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    valid_residuals = torch.stack(valid_predictions, dim=0).mean(dim=0)
    test_residuals = torch.stack(test_predictions, dim=0).mean(dim=0)
    return oof_residuals, valid_residuals, test_residuals, fold_history


def _group_means(values, labels):
    regions = _true_region_index(labels)
    means = []
    for region in range(len(TRUE_REGION_NAMES)):
        mask = regions == region
        if mask.any():
            means.append(values.view(-1)[mask].mean())
    return means


def _gate_loss(
    output,
    residuals,
    fusion,
    labels,
    group_dro_weight=0.40,
    harm_weight=2.0,
    oracle_kl_weight=0.05,
    anchor_weight=0.10,
    correction_weight=0.02,
    sign_weight=0.05,
    oracle_temperature=0.10,
):
    prediction = fusion + output['correction']
    prediction_error = torch.abs(prediction - labels)
    fusion_error = torch.abs(fusion - labels)
    mean_l1 = prediction_error.mean()
    group_losses = _group_means(prediction_error, labels)
    worst_group = torch.stack(group_losses).max() if group_losses else mean_l1
    group_dro = F.relu(worst_group - mean_l1)

    harm = F.relu(prediction_error - fusion_error)
    harm_loss = harm.mean()
    anchor_loss = (1.0 - output['weights'][:, :1]).mean()
    correction_loss = torch.abs(output['correction']).mean()
    sign_loss = _sign_loss(prediction, labels)

    candidates = torch.cat([fusion, fusion + residuals], dim=1)
    candidate_errors = torch.abs(candidates - labels)
    target_weights = F.softmax(-candidate_errors / oracle_temperature, dim=1)
    oracle_kl = F.kl_div(
        F.log_softmax(output['logits'], dim=1),
        target_weights,
        reduction='batchmean',
    )

    total = (
        mean_l1
        + group_dro_weight * group_dro
        + harm_weight * harm_loss
        + oracle_kl_weight * oracle_kl
        + anchor_weight * anchor_loss
        + correction_weight * correction_loss
        + sign_weight * sign_loss
    )
    return total, {
        'mean_l1': mean_l1.detach(),
        'worst_group_gap': group_dro.detach(),
        'harm_loss': harm_loss.detach(),
        'oracle_kl': oracle_kl.detach(),
        'anchor_loss': anchor_loss.detach(),
        'correction_loss': correction_loss.detach(),
        'sign_loss': sign_loss.detach(),
    }


def _routing_stats(fusion, prediction, labels, correction):
    fusion_error = torch.abs(fusion - labels)
    prediction_error = torch.abs(prediction - labels)
    realized_gain = fusion_error - prediction_error
    corrected = torch.abs(correction) > 0.01
    group_errors = _group_means(prediction_error, labels)
    balanced_mae = float(torch.stack(group_errors).mean().item()) if group_errors else float(prediction_error.mean().item())
    worst_group_mae = float(torch.stack(group_errors).max().item()) if group_errors else balanced_mae
    result = {
        'mae': float(prediction_error.mean().item()),
        'balanced_group_mae': balanced_mae,
        'worst_group_mae': worst_group_mae,
        'mean_realized_gain': float(realized_gain.mean().item()),
        'positive_gain_rate': float((realized_gain > 0).float().mean().item()),
        'harm_over_005_rate': float((realized_gain < -0.05).float().mean().item()),
        'harm_over_010_rate': float((realized_gain < -0.10).float().mean().item()),
        'correction_rate': float(corrected.float().mean().item()),
        'mean_abs_correction': float(torch.abs(correction).mean().item()),
    }
    regions = _true_region_index(labels)
    for index, name in enumerate(TRUE_REGION_NAMES):
        mask = regions == index
        if mask.any():
            result[f'{name}_mae'] = float(prediction_error.view(-1)[mask].mean().item())
    return result


def _robust_objective(stats, mean_alpha=0.0):
    balanced_gap = max(0.0, stats['balanced_group_mae'] - stats['mae'])
    worst_gap = max(0.0, stats['worst_group_mae'] - stats['balanced_group_mae'])
    return (
        stats['mae']
        + 0.50 * balanced_gap
        + 0.20 * worst_gap
        + 0.20 * stats['harm_over_005_rate']
        + 0.005 * stats['correction_rate']
        + 0.002 * float(mean_alpha)
    )


def _stratified_validation_indices(labels, fraction, seed):
    folds = _build_stratified_folds(labels, max(2, int(round(1.0 / fraction))), seed)
    return folds[0]


def _train_gate_once(
    context,
    residuals,
    expert_matrix,
    labels,
    device,
    seed,
    epochs,
    learning_rate,
    hidden_dim,
    dropout,
    batch_size,
    validation_indices=None,
    patience=8,
    **loss_kwargs,
):
    gate = DistributionRobustGate(context.size(1), hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = optim.AdamW(gate.parameters(), lr=learning_rate, weight_decay=1e-4)
    all_indices = torch.arange(len(labels))
    if validation_indices is None:
        train_indices = all_indices
    else:
        mask = torch.ones(len(labels), dtype=torch.bool)
        mask[validation_indices] = False
        train_indices = all_indices[mask]
    loader = _make_loader(
        context[train_indices], residuals[train_indices], expert_matrix[train_indices], labels[train_indices],
        batch_size=batch_size, shuffle=True, seed=seed,
    )
    best_state = None
    best_epoch = 0
    best_objective = float('inf')
    history = []
    for epoch in range(1, epochs + 1):
        gate.train()
        totals = {
            'loss': 0.0,
            'mean_l1': 0.0,
            'worst_group_gap': 0.0,
            'harm_loss': 0.0,
            'oracle_kl': 0.0,
            'anchor_loss': 0.0,
            'correction_loss': 0.0,
            'sign_loss': 0.0,
        }
        for batch_context, batch_residuals, batch_experts, batch_labels in loader:
            batch_context = batch_context.to(device)
            batch_residuals = batch_residuals.to(device)
            batch_experts = batch_experts.to(device)
            batch_labels = batch_labels.to(device)
            fusion = batch_experts[:, FUSION_INDEX:FUSION_INDEX + 1]
            optimizer.zero_grad()
            output = gate(batch_context, batch_residuals)
            loss, parts = _gate_loss(output, batch_residuals, fusion, batch_labels, **loss_kwargs)
            loss.backward()
            nn.utils.clip_grad_norm_(gate.parameters(), 5.0)
            optimizer.step()
            totals['loss'] += loss.item()
            for key, value in parts.items():
                totals[key] += value.item()
        record = {'epoch': epoch, **{key: value / max(1, len(loader)) for key, value in totals.items()}}
        if validation_indices is not None:
            gate.eval()
            with torch.no_grad():
                valid_context = context[validation_indices].to(device)
                valid_residuals = residuals[validation_indices].to(device)
                valid_fusion = expert_matrix[validation_indices, FUSION_INDEX:FUSION_INDEX + 1].to(device)
                valid_labels = labels[validation_indices].to(device)
                output = gate(valid_context, valid_residuals)
                prediction = valid_fusion + output['correction']
                stats = _routing_stats(valid_fusion, prediction, valid_labels, output['correction'])
                objective = _robust_objective(stats, mean_alpha=1.0)
            record['valid_objective'] = objective
            record.update({f'valid_{key}': value for key, value in stats.items()})
            if objective < best_objective - 1e-6:
                best_objective = objective
                best_epoch = epoch
                best_state = copy.deepcopy(gate.state_dict())
            if epoch - best_epoch >= patience:
                history.append(record)
                break
        history.append(record)
    if validation_indices is None:
        return gate, history, epochs
    if best_state is None:
        raise RuntimeError('Gate training did not produce a valid checkpoint.')
    gate.load_state_dict(best_state)
    return gate, history, best_epoch


def _train_gate_with_oof(train_cache, oof_residuals, device, seed, **train_kwargs):
    validation_indices = _stratified_validation_indices(train_cache['labels'], 0.20, seed + 2718)
    _, selection_history, best_epoch = _train_gate_once(
        context=train_cache['context'],
        residuals=oof_residuals,
        expert_matrix=train_cache['expert_matrix'],
        labels=train_cache['labels'],
        device=device,
        seed=seed,
        validation_indices=validation_indices,
        **train_kwargs,
    )
    final_kwargs = dict(train_kwargs)
    final_kwargs['epochs'] = max(1, best_epoch)
    final_gate, final_history, _ = _train_gate_once(
        context=train_cache['context'],
        residuals=oof_residuals,
        expert_matrix=train_cache['expert_matrix'],
        labels=train_cache['labels'],
        device=device,
        seed=seed + 1,
        validation_indices=None,
        patience=final_kwargs.pop('patience', 8),
        **final_kwargs,
    )
    return final_gate, selection_history, final_history, best_epoch


def _predict_gate(gate, cache, residuals, device, region_alphas=None):
    gate.eval()
    context = cache['context'].to(device)
    residuals_device = residuals.to(device)
    fusion = cache['expert_matrix'][:, FUSION_INDEX:FUSION_INDEX + 1].to(device)
    with torch.no_grad():
        output = gate(context, residuals_device)
        if region_alphas is None:
            alpha = torch.ones_like(fusion)
        else:
            region_index = _predicted_alpha_region_index(fusion)
            alpha_values = torch.tensor(
                [region_alphas[name] for name in PREDICTED_ALPHA_REGIONS],
                dtype=fusion.dtype,
                device=device,
            )
            alpha = alpha_values[region_index].view(-1, 1)
        correction = alpha * output['correction']
        prediction = fusion + correction
    return {
        'prediction': prediction.cpu(),
        'weights': output['weights'].cpu(),
        'correction': correction.cpu(),
        'raw_correction': output['correction'].cpu(),
        'alpha': alpha.cpu(),
    }


def _calibrate_region_alphas(gate, valid_cache, valid_residuals, device, alpha_values):
    fusion = valid_cache['expert_matrix'][:, FUSION_INDEX:FUSION_INDEX + 1]
    labels = valid_cache['labels']
    rows = []
    for combination in itertools.product(alpha_values, repeat=len(PREDICTED_ALPHA_REGIONS)):
        region_alphas = dict(zip(PREDICTED_ALPHA_REGIONS, combination))
        result = _predict_gate(gate, valid_cache, valid_residuals, device, region_alphas)
        stats = _routing_stats(fusion, result['prediction'], labels, result['correction'])
        mean_alpha = float(np.mean(combination))
        objective = _robust_objective(stats, mean_alpha=mean_alpha)
        rows.append({
            **{f'alpha_{name}': float(region_alphas[name]) for name in PREDICTED_ALPHA_REGIONS},
            'mean_alpha': mean_alpha,
            'objective': float(objective),
            **stats,
        })
    best = min(
        rows,
        key=lambda row: (
            row['objective'],
            row['mae'],
            row['harm_over_005_rate'],
            row['mean_alpha'],
        ),
    )
    selected = {name: float(best[f'alpha_{name}']) for name in PREDICTED_ALPHA_REGIONS}
    return selected, rows


def _candidate_diagnostics(cache, residuals, metrics_fn):
    labels = cache['labels']
    fusion = cache['expert_matrix'][:, FUSION_INDEX:FUSION_INDEX + 1]
    candidates = fusion + residuals
    predictions = torch.cat([fusion, candidates], dim=1)
    errors = torch.abs(predictions - labels)
    winner = errors.argmin(dim=1)
    rows = []
    names = ('fusion',) + SPECIALIST_NAMES
    for index, name in enumerate(names):
        prediction = predictions[:, index:index + 1]
        realized_gain = torch.abs(fusion - labels) - torch.abs(prediction - labels)
        rows.append({
            'candidate': name,
            **_safe_metrics(metrics_fn, prediction, labels),
            'oracle_win_rate': float((winner == index).float().mean().item()),
            'mean_gain_vs_fusion': float(realized_gain.mean().item()),
            'gain_over_005_rate': float((realized_gain > 0.05).float().mean().item()),
            'harm_over_005_rate': float((realized_gain < -0.05).float().mean().item()),
        })
    oracle_prediction = predictions.gather(1, winner.unsqueeze(1))
    return rows, oracle_prediction, errors


def _js_divergence(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first = first / max(first.sum(), 1e-12)
    second = second / max(second.sum(), 1e-12)
    middle = 0.5 * (first + second)

    def kl(left, right):
        mask = left > 0
        return float(np.sum(left[mask] * np.log(left[mask] / np.clip(right[mask], 1e-12, None))))

    return 0.5 * kl(first, middle) + 0.5 * kl(second, middle)


def _wasserstein_1d(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    quantiles = np.linspace(0.001, 0.999, 999)
    return float(np.mean(np.abs(np.quantile(first, quantiles) - np.quantile(second, quantiles))))


def _distribution_diagnostics(cached, metrics_fn, save_dir):
    distribution_rows = []
    region_rows = []
    proportions = {}
    for split_name, cache in cached.items():
        labels = cache['labels'].view(-1)
        fusion = cache['expert_matrix'][:, FUSION_INDEX:FUSION_INDEX + 1]
        regions = _true_region_index(labels)
        counts = []
        for region_index, region_name in enumerate(TRUE_REGION_NAMES):
            mask = regions == region_index
            count = int(mask.sum().item())
            counts.append(count)
            distribution_rows.append({
                'split': split_name,
                'region': region_name,
                'count': count,
                'proportion': count / max(1, len(labels)),
                'label_mean': float(labels[mask].mean().item()) if count else None,
            })
            if count:
                metrics = _safe_metrics(metrics_fn, fusion[mask], labels[mask].view(-1, 1))
                region_rows.append({'split': split_name, 'region': region_name, **metrics})
        proportions[split_name] = np.asarray(counts, dtype=float)
        distribution_rows.append({
            'split': split_name,
            'region': '__overall__',
            'count': len(labels),
            'proportion': 1.0,
            'label_mean': float(labels.mean().item()),
            'label_std': float(labels.std(unbiased=False).item()),
            'label_median': float(labels.median().item()),
            'label_min': float(labels.min().item()),
            'label_max': float(labels.max().item()),
        })
    pd.DataFrame(distribution_rows).to_csv(Path(save_dir) / 'split_label_distribution.csv', index=False)
    pd.DataFrame(region_rows).to_csv(Path(save_dir) / 'split_region_fusion_metrics.csv', index=False)
    shift = {}
    train_labels = cached['train']['labels'].view(-1).numpy()
    for split_name in ('valid', 'test'):
        split_labels = cached[split_name]['labels'].view(-1).numpy()
        shift[f'train_to_{split_name}'] = {
            'js_divergence_5bin': _js_divergence(proportions['train'], proportions[split_name]),
            'wasserstein_label': _wasserstein_1d(train_labels, split_labels),
        }
    with open(Path(save_dir) / 'distribution_shift.json', 'w', encoding='utf-8') as file:
        json.dump(shift, file, ensure_ascii=False, indent=2, allow_nan=False)
    return shift


def _save_report(
    save_dir,
    seed,
    fold_count,
    selected_alphas,
    metrics_fn,
    test_cache,
    test_residuals,
    gate_result,
    train_cache,
    oof_residuals,
    specialist_rows,
    specialist_oracle,
    specialist_errors,
    gate_best_epoch,
    distribution_shift,
):
    save_dir = Path(save_dir)
    labels = test_cache['labels']
    fusion = test_cache['expert_matrix'][:, FUSION_INDEX:FUSION_INDEX + 1]
    prediction = gate_result['prediction']
    correction = gate_result['correction']
    weights = gate_result['weights']
    stats = _routing_stats(fusion, prediction, labels, correction)
    fusion_metrics = _safe_metrics(metrics_fn, fusion, labels)
    routed_metrics = _safe_metrics(metrics_fn, prediction, labels)
    oracle_metrics = _safe_metrics(metrics_fn, specialist_oracle, labels)
    oracle_space = fusion_metrics['MAE'] - oracle_metrics['MAE']
    recovered = fusion_metrics['MAE'] - routed_metrics['MAE']
    recovery_ratio = recovered / oracle_space if oracle_space > 1e-12 else 0.0
    names = ('fusion',) + SPECIALIST_NAMES
    average_weights = {name: float(weights[:, index].mean().item()) for index, name in enumerate(names)}
    top_indices = weights.argmax(dim=1)
    top_rates = {name: float((top_indices == index).float().mean().item()) for index, name in enumerate(names)}
    summary = {
        'seed': int(seed),
        'oof_folds': int(fold_count),
        'gate_best_epoch': int(gate_best_epoch),
        'region_alphas': selected_alphas,
        'fusion_metrics': fusion_metrics,
        'residual_router_metrics': routed_metrics,
        'specialist_oracle_metrics': oracle_metrics,
        **stats,
        'oracle_gap_recovery_ratio': float(recovery_ratio),
        'average_gate_weights': average_weights,
        'top_gate_rates': top_rates,
        'distribution_shift': distribution_shift,
    }
    with open(save_dir / 'distribution_robust_summary.json', 'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)

    pd.DataFrame(specialist_rows).to_csv(save_dir / 'test_specialist_diagnostics.csv', index=False)
    error_names = ('fusion',) + SPECIALIST_NAMES
    with np.errstate(invalid='ignore', divide='ignore'):
        correlation = np.corrcoef(specialist_errors.numpy(), rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0)
    pd.DataFrame(correlation, index=error_names, columns=error_names).to_csv(
        save_dir / 'test_specialist_error_correlation.csv'
    )

    oof_rows, _, oof_errors = _candidate_diagnostics(train_cache, oof_residuals, metrics_fn)
    pd.DataFrame(oof_rows).to_csv(save_dir / 'oof_specialist_diagnostics.csv', index=False)
    with np.errstate(invalid='ignore', divide='ignore'):
        oof_correlation = np.corrcoef(oof_errors.numpy(), rowvar=False)
    oof_correlation = np.nan_to_num(oof_correlation, nan=0.0)
    pd.DataFrame(oof_correlation, index=error_names, columns=error_names).to_csv(
        save_dir / 'oof_specialist_error_correlation.csv'
    )

    sample_ids = test_cache['sample_ids']
    if len(sample_ids) != len(labels):
        sample_ids = [str(index) for index in range(len(labels))]
    prediction_data = {
        'sample_id': sample_ids,
        'target': labels.view(-1).numpy(),
        'fusion_prediction': fusion.view(-1).numpy(),
        'residual_router_prediction': prediction.view(-1).numpy(),
        'correction': correction.view(-1).numpy(),
        'alpha': gate_result['alpha'].view(-1).numpy(),
        'fusion_abs_error': torch.abs(fusion - labels).view(-1).numpy(),
        'router_abs_error': torch.abs(prediction - labels).view(-1).numpy(),
    }
    for index, name in enumerate(SPECIALIST_NAMES):
        candidate = fusion + test_residuals[:, index:index + 1]
        prediction_data[f'{name}_residual'] = test_residuals[:, index].numpy()
        prediction_data[f'{name}_prediction'] = candidate.view(-1).numpy()
        prediction_data[f'{name}_weight'] = weights[:, index + 1].numpy()
    prediction_data['fusion_weight'] = weights[:, 0].numpy()
    pd.DataFrame(prediction_data).to_csv(save_dir / 'distribution_robust_predictions.csv', index=False)

    logger.info(
        'DISTRIBUTION-ROBUST TEST fusion_MAE=%.4f routed_MAE=%.4f oracle_MAE=%.4f alphas=%s correction_rate=%.4f harm_010=%.4f recovery=%.4f',
        fusion_metrics['MAE'],
        routed_metrics['MAE'],
        oracle_metrics['MAE'],
        selected_alphas,
        stats['correction_rate'],
        stats['harm_over_010_rate'],
        recovery_ratio,
    )
    return summary


def train_distribution_robust_specialist_system(
    model,
    dataloader,
    metrics_fn,
    device,
    save_dir,
    seed,
    oof_folds=5,
    specialist_epochs=25,
    gate_epochs=60,
    gate_patience=8,
    specialist_learning_rate=8e-4,
    gate_learning_rate=5e-4,
    shared_dim=96,
    adapter_dim=48,
    gate_hidden_dim=64,
    dropout=0.15,
    max_residual=0.75,
    boundary_max_residual=0.30,
    batch_size=128,
    off_region_anchor_weight=0.10,
    group_dro_weight=0.40,
    harm_weight=2.0,
    oracle_kl_weight=0.05,
    anchor_weight=0.10,
    correction_weight=0.02,
    sign_weight=0.05,
    calibration_alphas=None,
):
    """Train stratified fold specialists, an OOF gate, and region-conditional calibration."""
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
    distribution_shift = _distribution_diagnostics(cached, metrics_fn, save_dir)

    specialist_kwargs = {
        'epochs': specialist_epochs,
        'learning_rate': specialist_learning_rate,
        'shared_dim': shared_dim,
        'adapter_dim': adapter_dim,
        'dropout': dropout,
        'max_residual': max_residual,
        'boundary_max_residual': boundary_max_residual,
        'batch_size': batch_size,
        'off_region_anchor_weight': off_region_anchor_weight,
    }
    oof_residuals, valid_residuals, test_residuals, fold_history = _cross_fit_specialists(
        train_cache=cached['train'],
        valid_cache=cached['valid'],
        test_cache=cached['test'],
        device=device,
        seed=seed,
        fold_count=oof_folds,
        save_dir=save_dir,
        **specialist_kwargs,
    )
    pd.DataFrame(fold_history).to_csv(save_dir / 'specialist_fold_history.csv', index=False)
    torch.save(
        {
            'normalization': normalization,
            'specialist_names': list(SPECIALIST_NAMES),
            'context_dim': int(cached['train']['context'].size(1)),
            'fold_count': int(oof_folds),
        },
        save_dir / 'specialist_ensemble_metadata.pth',
    )

    gate_kwargs = {
        'epochs': gate_epochs,
        'learning_rate': gate_learning_rate,
        'hidden_dim': gate_hidden_dim,
        'dropout': dropout,
        'batch_size': batch_size,
        'patience': gate_patience,
        'group_dro_weight': group_dro_weight,
        'harm_weight': harm_weight,
        'oracle_kl_weight': oracle_kl_weight,
        'anchor_weight': anchor_weight,
        'correction_weight': correction_weight,
        'sign_weight': sign_weight,
    }
    gate, gate_selection_history, gate_final_history, gate_best_epoch = _train_gate_with_oof(
        cached['train'],
        oof_residuals,
        device=device,
        seed=seed + 12001,
        **gate_kwargs,
    )
    pd.DataFrame(gate_selection_history).to_csv(save_dir / 'gate_selection_history.csv', index=False)
    pd.DataFrame(gate_final_history).to_csv(save_dir / 'gate_final_history.csv', index=False)

    selected_alphas, calibration_rows = _calibrate_region_alphas(
        gate,
        cached['valid'],
        valid_residuals,
        device,
        calibration_alphas,
    )
    pd.DataFrame(calibration_rows).to_csv(save_dir / 'region_alpha_calibration.csv', index=False)
    torch.save(
        {
            'state_dict': gate.state_dict(),
            'context_dim': int(cached['train']['context'].size(1)),
            'specialist_names': list(SPECIALIST_NAMES),
            'best_epoch': int(gate_best_epoch),
            'region_alphas': selected_alphas,
        },
        save_dir / 'distribution_robust_gate.pth',
    )

    gate_result = _predict_gate(gate, cached['test'], test_residuals, device, selected_alphas)
    specialist_rows, specialist_oracle, specialist_errors = _candidate_diagnostics(
        cached['test'], test_residuals, metrics_fn
    )
    summary = _save_report(
        save_dir=save_dir,
        seed=seed,
        fold_count=oof_folds,
        selected_alphas=selected_alphas,
        metrics_fn=metrics_fn,
        test_cache=cached['test'],
        test_residuals=test_residuals,
        gate_result=gate_result,
        train_cache=cached['train'],
        oof_residuals=oof_residuals,
        specialist_rows=specialist_rows,
        specialist_oracle=specialist_oracle,
        specialist_errors=specialist_errors,
        gate_best_epoch=gate_best_epoch,
        distribution_shift=distribution_shift,
    )
    return gate, summary
