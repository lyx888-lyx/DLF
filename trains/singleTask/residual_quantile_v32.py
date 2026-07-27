import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from .residual_direction_core import _fusion, _regions

QUANTILES = (0.10, 0.50, 0.90)


class SemanticResidualQuantileNet(nn.Module):
    def __init__(
        self,
        context_dim,
        semantic_dim,
        residual_dim=3,
        hidden_dim=128,
        semantic_hidden_dim=96,
        dropout=0.20,
        max_center=1.50,
        max_width=1.50,
    ):
        super().__init__()
        self.max_center = float(max_center)
        self.max_width = float(max_width)
        self.model_encoder = nn.Sequential(
            nn.LayerNorm(context_dim + residual_dim),
            nn.Linear(context_dim + residual_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.semantic_encoder = nn.Sequential(
            nn.LayerNorm(semantic_dim),
            nn.Linear(semantic_dim, semantic_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + semantic_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, context, semantic, specialist_residuals):
        model_hidden = self.model_encoder(
            torch.cat([context, specialist_residuals], dim=1)
        )
        semantic_hidden = self.semantic_encoder(semantic)
        raw = self.fusion(torch.cat([model_hidden, semantic_hidden], dim=1))
        median = self.max_center * torch.tanh(raw[:, 0])
        lower_width = self.max_width * torch.sigmoid(raw[:, 1])
        upper_width = self.max_width * torch.sigmoid(raw[:, 2])
        return torch.stack(
            [median - lower_width, median, median + upper_width], dim=1
        )


def pinball_loss(prediction, target, quantile):
    error = target - prediction
    return torch.maximum(quantile * error, (quantile - 1.0) * error)


def _strata(labels, fusion, margin):
    residual = (labels - fusion).view(-1)
    direction = torch.ones_like(residual, dtype=torch.long)
    direction[residual < -margin] = 0
    direction[residual > margin] = 2
    return _regions(labels) * 3 + direction


def _sample_weights(labels, fusion, margin):
    strata = _strata(labels, fusion, margin)
    counts = torch.bincount(strata, minlength=15).float().clamp_min(1.0)
    values = counts.sum() / torch.sqrt(counts)
    weights = values[strata]
    return weights / weights.mean().clamp_min(1e-8)


def _stratified_folds(labels, fusion, margin, fold_count, seed):
    strata = _strata(labels, fusion, margin)
    generator = torch.Generator().manual_seed(int(seed))
    folds = [[] for _ in range(fold_count)]
    for value in torch.unique(strata).tolist():
        indices = torch.nonzero(strata == value, as_tuple=False).view(-1)
        indices = indices[torch.randperm(len(indices), generator=generator)]
        for position, sample_index in enumerate(indices.tolist()):
            folds[position % fold_count].append(sample_index)
    result = [torch.tensor(sorted(values), dtype=torch.long) for values in folds]
    if any(len(values) == 0 for values in result):
        raise RuntimeError('Quantile stratification produced an empty fold.')
    return result


def quantile_metrics(quantiles, residual_target, margin=0.10):
    target = residual_target.view(-1)
    q10, q50, q90 = quantiles[:, 0], quantiles[:, 1], quantiles[:, 2]
    active = target.abs() > margin
    predicted_direction = q50 >= 0
    true_direction = target >= 0
    recalls = []
    for direction in (False, True):
        mask = active & (true_direction == direction)
        if mask.any():
            recalls.append((predicted_direction[mask] == direction).float().mean())
    sign_balanced = float(torch.stack(recalls).mean().item()) if recalls else 0.0
    confident_up = q10 > margin
    confident_down = q90 < -margin
    confident = confident_up | confident_down
    correct = (confident_up & (target > 0)) | (confident_down & (target < 0))
    precision = float(correct[confident].float().mean().item()) if confident.any() else None
    coverage = float(confident.float().mean().item())
    centered_target = target - target.mean()
    centered_pred = q50 - q50.mean()
    denominator = torch.sqrt(
        centered_target.pow(2).sum() * centered_pred.pow(2).sum()
    ).clamp_min(1e-12)
    correlation = float((centered_target * centered_pred).sum().item() / denominator.item())
    inside = (target >= q10) & (target <= q90)
    return {
        'residual_mae': float(torch.abs(q50 - target).mean().item()),
        'residual_correlation': correlation,
        'sign_balanced_accuracy': sign_balanced,
        'interval_coverage_10_90': float(inside.float().mean().item()),
        'confident_direction_coverage': coverage,
        'confident_direction_precision': precision,
        'mean_interval_width': float((q90 - q10).mean().item()),
    }


def _predict(model, context, semantic, residuals, device, batch_size=512):
    model.eval()
    rows = []
    loader = DataLoader(
        TensorDataset(context, semantic, residuals),
        batch_size=batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for batch_context, batch_semantic, batch_residuals in loader:
            rows.append(
                model(
                    batch_context.to(device),
                    batch_semantic.to(device),
                    batch_residuals.to(device),
                ).cpu()
            )
    return torch.cat(rows, dim=0)


def _train_once(
    context,
    semantic,
    specialist_residuals,
    expert_matrix,
    labels,
    device,
    seed,
    residual_margin,
    epochs,
    learning_rate,
    hidden_dim,
    semantic_hidden_dim,
    dropout,
    batch_size,
    max_center,
    max_width,
    validation_indices=None,
    patience=10,
    median_loss_weight=0.25,
):
    fusion = _fusion(expert_matrix)
    target = (labels - fusion).view(-1)
    weights = _sample_weights(labels, fusion, residual_margin)
    all_indices = torch.arange(len(labels))
    train_indices = all_indices
    if validation_indices is not None:
        mask = torch.ones(len(labels), dtype=torch.bool)
        mask[validation_indices] = False
        train_indices = all_indices[mask]
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        TensorDataset(
            context[train_indices],
            semantic[train_indices],
            specialist_residuals[train_indices],
            target[train_indices],
            weights[train_indices],
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    model = SemanticResidualQuantileNet(
        context.size(1), semantic.size(1), specialist_residuals.size(1),
        hidden_dim, semantic_hidden_dim, dropout, max_center, max_width,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_state, best_epoch, best_score = None, 0, -float('inf')
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {'loss': 0.0, 'pinball': 0.0, 'median_loss': 0.0}
        for batch_context, batch_semantic, batch_residuals, batch_target, batch_weight in loader:
            optimizer.zero_grad()
            prediction = model(
                batch_context.to(device),
                batch_semantic.to(device),
                batch_residuals.to(device),
            )
            batch_target = batch_target.to(device)
            batch_weight = batch_weight.to(device)
            losses = []
            for column, quantile in enumerate(QUANTILES):
                losses.append(
                    pinball_loss(prediction[:, column], batch_target, quantile)
                )
            pinball = torch.stack(losses, dim=1).mean(dim=1)
            pinball = (pinball * batch_weight).mean()
            median_loss = F.smooth_l1_loss(prediction[:, 1], batch_target)
            loss = pinball + median_loss_weight * median_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals['loss'] += loss.item()
            totals['pinball'] += pinball.item()
            totals['median_loss'] += median_loss.item()
        row = {'epoch': epoch, **{k: v / max(1, len(loader)) for k, v in totals.items()}}
        if validation_indices is not None:
            prediction = _predict(
                model,
                context[validation_indices],
                semantic[validation_indices],
                specialist_residuals[validation_indices],
                device,
            )
            metrics = quantile_metrics(
                prediction, target[validation_indices], residual_margin
            )
            row.update({f'valid_{key}': value for key, value in metrics.items()})
            precision = metrics['confident_direction_precision']
            precision_value = 0.0 if precision is None else precision
            coverage_value = min(metrics['confident_direction_coverage'] / 0.20, 1.0)
            score = (
                0.45 * metrics['sign_balanced_accuracy']
                + 0.35 * precision_value
                + 0.20 * coverage_value
            )
            row['valid_selection_score'] = score
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
        raise RuntimeError('Quantile model did not produce a checkpoint.')
    model.load_state_dict(best_state)
    return model, history, best_epoch


def train_quantile_model(
    train_cache,
    oof_specialist_residuals,
    device,
    seed,
    residual_margin=0.10,
    epochs=80,
    learning_rate=4e-4,
    hidden_dim=128,
    semantic_hidden_dim=96,
    dropout=0.20,
    batch_size=128,
    max_center=1.50,
    max_width=1.50,
    patience=10,
    median_loss_weight=0.25,
):
    fusion = _fusion(train_cache['expert_matrix'])
    validation_indices = _stratified_folds(
        train_cache['labels'], fusion, residual_margin, 5, seed + 2718
    )[0]
    _, selection_history, best_epoch = _train_once(
        train_cache['context'], train_cache['semantic'], oof_specialist_residuals,
        train_cache['expert_matrix'], train_cache['labels'], device, seed,
        residual_margin, epochs, learning_rate, hidden_dim, semantic_hidden_dim,
        dropout, batch_size, max_center, max_width,
        validation_indices=validation_indices, patience=patience,
        median_loss_weight=median_loss_weight,
    )
    model, final_history, _ = _train_once(
        train_cache['context'], train_cache['semantic'], oof_specialist_residuals,
        train_cache['expert_matrix'], train_cache['labels'], device, seed + 1,
        residual_margin, max(1, best_epoch), learning_rate, hidden_dim,
        semantic_hidden_dim, dropout, batch_size, max_center, max_width,
        validation_indices=None, patience=patience,
        median_loss_weight=median_loss_weight,
    )
    return model, selection_history, final_history, best_epoch


def predict_quantiles(model, cache, specialist_residuals, device):
    return _predict(
        model, cache['context'], cache['semantic'], specialist_residuals, device
    )


def combined_features(cache, specialist_residuals):
    return torch.cat(
        [cache['context'], cache['semantic'], specialist_residuals], dim=1
    ).float()


def fit_ridge_baseline(
    train_cache,
    oof_specialist_residuals,
    residual_margin,
    seed,
    lambdas=(0.1, 1.0, 10.0, 100.0),
    fold_count=5,
):
    features = combined_features(train_cache, oof_specialist_residuals)
    target = (
        train_cache['labels'] - _fusion(train_cache['expert_matrix'])
    ).view(-1, 1)
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-5)
    x = (features - mean) / std
    x = torch.cat([x, torch.ones(len(x), 1)], dim=1).double()
    y = target.double()
    folds = _stratified_folds(
        train_cache['labels'], _fusion(train_cache['expert_matrix']),
        residual_margin, fold_count, seed,
    )
    all_indices = torch.arange(len(x))
    rows = []
    for value in lambdas:
        fold_maes = []
        for holdout in folds:
            mask = torch.ones(len(x), dtype=torch.bool)
            mask[holdout] = False
            train_index = all_indices[mask]
            xtx = x[train_index].t() @ x[train_index]
            regularizer = torch.eye(xtx.size(0), dtype=xtx.dtype) * float(value)
            regularizer[-1, -1] = 0.0
            weight = torch.linalg.solve(
                xtx + regularizer,
                x[train_index].t() @ y[train_index],
            )
            prediction = x[holdout] @ weight
            fold_maes.append(torch.abs(prediction - y[holdout]).mean().item())
        rows.append({'lambda': float(value), 'cv_residual_mae': float(sum(fold_maes) / len(fold_maes))})
    best_lambda = min(rows, key=lambda row: row['cv_residual_mae'])['lambda']
    xtx = x.t() @ x
    regularizer = torch.eye(xtx.size(0), dtype=xtx.dtype) * float(best_lambda)
    regularizer[-1, -1] = 0.0
    weight = torch.linalg.solve(xtx + regularizer, x.t() @ y).float()
    return {'mean': mean, 'std': std, 'weight': weight, 'lambda': best_lambda}, rows


def predict_ridge(model, cache, specialist_residuals):
    features = combined_features(cache, specialist_residuals)
    x = (features - model['mean']) / model['std']
    x = torch.cat([x, torch.ones(len(x), 1)], dim=1)
    return x @ model['weight']
