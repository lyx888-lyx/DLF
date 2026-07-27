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

from .specialist_residual import FUSION_INDEX, _cache_split, _normalize_cached, _safe_metrics, _sign_loss

logger = logging.getLogger('MMSA')
SPECIALIST_NAMES = ('up', 'down', 'boundary')
ACTION_NAMES = ('fusion',) + SPECIALIST_NAMES
REGION_NAMES = ('strong_negative', 'negative', 'boundary', 'positive', 'strong_positive')
SCENARIOS = {
    'original': (1.0, 1.0, 1.0, 1.0, 1.0),
    'negative_heavy': (2.5, 1.8, 1.0, 0.8, 0.8),
    'positive_heavy': (0.8, 0.8, 1.0, 1.8, 2.5),
    'boundary_heavy': (1.0, 1.0, 2.5, 1.0, 1.0),
    'strong_heavy': (2.0, 1.0, 1.0, 1.0, 2.0),
}


class ResidualDirectionSpecialists(nn.Module):
    def __init__(self, context_dim, shared_dim=96, adapter_dim=48, dropout=0.15,
                 max_residual=0.75, boundary_max_residual=0.35):
        super().__init__()
        self.max_residual = float(max_residual)
        self.boundary_max_residual = float(boundary_max_residual)
        self.shared = nn.Sequential(
            nn.LayerNorm(context_dim), nn.Linear(context_dim, shared_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.adapters = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(shared_dim, adapter_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(adapter_dim, adapter_dim), nn.GELU(),
                nn.LayerNorm(adapter_dim), nn.Dropout(dropout),
            ) for name in SPECIALIST_NAMES
        })
        self.heads = nn.ModuleDict({name: nn.Linear(adapter_dim, 1) for name in SPECIALIST_NAMES})
        with torch.no_grad():
            self.heads['up'].bias.fill_(-1.5)
            self.heads['down'].bias.fill_(-1.5)
            self.heads['boundary'].bias.zero_()

    def forward(self, context):
        shared = self.shared(context)
        raw = {name: self.heads[name](self.adapters[name](shared)) for name in SPECIALIST_NAMES}
        return torch.cat([
            self.max_residual * torch.sigmoid(raw['up']),
            -self.max_residual * torch.sigmoid(raw['down']),
            self.boundary_max_residual * torch.tanh(raw['boundary']),
        ], dim=1)


class ResidualDirectionGate(nn.Module):
    def __init__(self, context_dim, hidden_dim=64, dropout=0.15):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(context_dim + len(SPECIALIST_NAMES)),
            nn.Linear(context_dim + len(SPECIALIST_NAMES), hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(ACTION_NAMES)),
        )
        with torch.no_grad():
            self.network[-1].bias.zero_()
            self.network[-1].bias[0] = 1.5

    def forward(self, context, residuals):
        logits = self.network(torch.cat([context, residuals], dim=1))
        weights = F.softmax(logits, dim=1)
        components = weights[:, 1:] * residuals
        return {'logits': logits, 'weights': weights, 'components': components,
                'correction': components.sum(dim=1, keepdim=True)}


def _fusion(expert_matrix):
    return expert_matrix[:, FUSION_INDEX:FUSION_INDEX + 1]


def _regions(labels):
    y = labels.view(-1)
    out = torch.full_like(y, 2, dtype=torch.long)
    out[y < -1.5] = 0
    out[(y >= -1.5) & (y < -0.5)] = 1
    out[(y > 0.5) & (y <= 1.5)] = 3
    out[y > 1.5] = 4
    return out


def _directions(labels, fusion, margin):
    residual = (labels - fusion).view(-1)
    out = torch.ones_like(residual, dtype=torch.long)
    out[residual < -margin] = 0
    out[residual > margin] = 2
    return out


def _action_target(labels, fusion, margin, boundary_threshold):
    residual = (labels - fusion).view(-1)
    target = torch.zeros_like(residual, dtype=torch.long)
    target[residual > margin] = 1
    target[residual < -margin] = 2
    boundary = (torch.abs(labels.view(-1)) <= boundary_threshold) & (torch.abs(residual) > margin)
    target[boundary] = 3
    return target


def _make_loader(*tensors, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle,
                      generator=generator if shuffle else None)


def _specialist_loss(residuals, fusion, labels, margin, boundary_threshold,
                     max_residual, boundary_max_residual, off_anchor_weight):
    target = labels - fusion
    masks = torch.stack([
        target.view(-1) > margin,
        target.view(-1) < -margin,
        (torch.abs(labels.view(-1)) <= boundary_threshold) & (torch.abs(target.view(-1)) > margin * 0.5),
    ], dim=1)
    targets = torch.cat([
        target.clamp(0.0, max_residual),
        target.clamp(-max_residual, 0.0),
        target.clamp(-boundary_max_residual, boundary_max_residual),
    ], dim=1)
    losses = []
    for index in range(len(SPECIALIST_NAMES)):
        active = masks[:, index]
        if active.any():
            losses.append(F.smooth_l1_loss(residuals[active, index], targets[active, index]))
    task = torch.stack(losses).mean() if losses else residuals.new_tensor(0.0)
    inactive = ~masks
    anchor = residuals[inactive].pow(2).mean() if inactive.any() else residuals.new_tensor(0.0)
    return task + off_anchor_weight * anchor, {'task_loss': task.detach(), 'off_region_anchor': anchor.detach()}


def _train_specialist(context, experts, labels, device, seed, epochs, learning_rate,
                      shared_dim, adapter_dim, dropout, max_residual,
                      boundary_max_residual, batch_size, residual_margin,
                      boundary_threshold, off_anchor_weight):
    model = ResidualDirectionSpecialists(
        context.size(1), shared_dim, adapter_dim, dropout, max_residual, boundary_max_residual
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loader = _make_loader(context, experts, labels, batch_size=batch_size, shuffle=True, seed=seed)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {'loss': 0.0, 'task_loss': 0.0, 'off_region_anchor': 0.0}
        for batch_context, batch_experts, batch_labels in loader:
            batch_context, batch_experts, batch_labels = (
                batch_context.to(device), batch_experts.to(device), batch_labels.to(device)
            )
            optimizer.zero_grad()
            residuals = model(batch_context)
            loss, parts = _specialist_loss(
                residuals, _fusion(batch_experts), batch_labels, residual_margin,
                boundary_threshold, max_residual, boundary_max_residual, off_anchor_weight
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals['loss'] += loss.item()
            for key, value in parts.items():
                totals[key] += value.item()
        history.append({'epoch': epoch, **{k: v / max(1, len(loader)) for k, v in totals.items()}})
    return model, history


def _predict_specialist(model, context, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(context), batch_size=512, shuffle=False):
            outputs.append(model(batch.to(device)).cpu())
    return torch.cat(outputs, dim=0)


def _build_folds(labels, experts, fold_count, seed, margin):
    if fold_count < 2:
        raise ValueError('fold_count must be at least 2.')
    strata = _regions(labels) * 3 + _directions(labels, _fusion(experts), margin)
    generator = torch.Generator().manual_seed(int(seed))
    folds = [[] for _ in range(fold_count)]
    for group in torch.unique(strata).tolist():
        indices = torch.nonzero(strata == group, as_tuple=False).view(-1)
        indices = indices[torch.randperm(len(indices), generator=generator)]
        for position, sample_index in enumerate(indices.tolist()):
            folds[position % fold_count].append(sample_index)
    return [torch.tensor(sorted(fold), dtype=torch.long) for fold in folds if fold]


def _cross_fit(train, valid, test, device, seed, fold_count, save_dir, **kwargs):
    folds = _build_folds(train['labels'], train['expert_matrix'], fold_count, seed, kwargs['residual_margin'])
    count = len(train['labels'])
    oof = torch.zeros(count, len(SPECIALIST_NAMES), dtype=train['context'].dtype)
    valid_predictions, test_predictions, history = [], [], []
    all_indices = torch.arange(count)
    for fold_index, holdout in enumerate(folds):
        mask = torch.ones(count, dtype=torch.bool)
        mask[holdout] = False
        train_indices = all_indices[mask]
        logger.info('Training residual-direction fold %d/%d on %d samples; holding out %d.',
                    fold_index + 1, len(folds), len(train_indices), len(holdout))
        model, fold_history = _train_specialist(
            train['context'][train_indices], train['expert_matrix'][train_indices],
            train['labels'][train_indices], device=device,
            seed=seed + 1009 * (fold_index + 1), **kwargs
        )
        oof[holdout] = _predict_specialist(model, train['context'][holdout], device)
        valid_predictions.append(_predict_specialist(model, valid['context'], device))
        test_predictions.append(_predict_specialist(model, test['context'], device))
        history.extend({'fold': fold_index + 1, **row} for row in fold_history)
        torch.save({'state_dict': model.state_dict(), 'fold': fold_index + 1,
                    'specialist_names': list(SPECIALIST_NAMES)},
                   Path(save_dir) / f'residual_direction_fold_{fold_index + 1}.pth')
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return oof, torch.stack(valid_predictions).mean(0), torch.stack(test_predictions).mean(0), history


def _group_gap(errors, groups):
    means = [errors.view(-1)[groups == group].mean() for group in torch.unique(groups).tolist()]
    return F.relu(torch.stack(means).max() - errors.mean()) if means else errors.new_tensor(0.0)


def _gate_loss(output, residuals, fusion, labels, residual_margin, boundary_threshold,
               sentiment_dro_weight, direction_dro_weight, harm_weight,
               direction_loss_weight, oracle_kl_weight, anchor_weight,
               correction_weight, sign_weight):
    prediction = fusion + output['correction']
    errors = torch.abs(prediction - labels)
    fusion_errors = torch.abs(fusion - labels)
    mean_l1 = errors.mean()
    sentiment_gap = _group_gap(errors, _regions(labels))
    direction_gap = _group_gap(errors, _directions(labels, fusion, residual_margin))
    harm = F.relu(errors - fusion_errors).mean()
    direction_loss = F.cross_entropy(
        output['logits'], _action_target(labels, fusion, residual_margin, boundary_threshold)
    )
    anchor = (1.0 - output['weights'][:, :1]).mean()
    correction = torch.abs(output['correction']).mean()
    sign = _sign_loss(prediction, labels)
    oracle_kl = prediction.new_tensor(0.0)
    if oracle_kl_weight > 0:
        candidates = torch.cat([fusion, fusion + residuals], dim=1)
        target_weights = F.softmax(-torch.abs(candidates - labels) / 0.10, dim=1)
        oracle_kl = F.kl_div(F.log_softmax(output['logits'], dim=1), target_weights,
                             reduction='batchmean')
    total = (mean_l1 + sentiment_dro_weight * sentiment_gap + direction_dro_weight * direction_gap
             + harm_weight * harm + direction_loss_weight * direction_loss
             + oracle_kl_weight * oracle_kl + anchor_weight * anchor
             + correction_weight * correction + sign_weight * sign)
    return total, {
        'mean_l1': mean_l1.detach(), 'sentiment_gap': sentiment_gap.detach(),
        'direction_gap': direction_gap.detach(), 'harm_loss': harm.detach(),
        'direction_loss': direction_loss.detach(), 'oracle_kl': oracle_kl.detach(),
        'anchor_loss': anchor.detach(), 'correction_loss': correction.detach(),
        'sign_loss': sign.detach(),
    }
