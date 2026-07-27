import copy
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from .residual_direction_core import _fusion, _regions

QUANTILES = (0.10, 0.50, 0.90)
REGION_NAMES = ('strong_negative', 'negative', 'boundary', 'positive', 'strong_positive')


class OrdinalConformalNet(nn.Module):
    """Predict residual quantiles, an ordinal sentiment region, and a direct label."""

    def __init__(
        self,
        context_dim,
        semantic_dim,
        residual_dim=3,
        hidden_dim=128,
        semantic_hidden_dim=96,
        dropout=0.20,
        max_center=1.75,
        max_width=1.75,
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
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim + semantic_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.quantile_head = nn.Linear(hidden_dim, 3)
        self.ordinal_score = nn.Linear(hidden_dim, 1)
        self.ordinal_cut_raw = nn.Parameter(torch.tensor([-1.5, 0.3, 0.3, 0.3]))
        self.target_head = nn.Linear(hidden_dim, 1)

    def _cutpoints(self):
        first = self.ordinal_cut_raw[0:1]
        gaps = F.softplus(self.ordinal_cut_raw[1:]) + 1e-3
        return torch.cat([first, first + torch.cumsum(gaps, dim=0)], dim=0)

    def forward(self, context, semantic, specialist_residuals):
        model_hidden = self.model_encoder(torch.cat([context, specialist_residuals], dim=1))
        semantic_hidden = self.semantic_encoder(semantic)
        hidden = self.shared(torch.cat([model_hidden, semantic_hidden], dim=1))

        raw_quantiles = self.quantile_head(hidden)
        median = self.max_center * torch.tanh(raw_quantiles[:, 0])
        lower_width = self.max_width * torch.sigmoid(raw_quantiles[:, 1])
        upper_width = self.max_width * torch.sigmoid(raw_quantiles[:, 2])
        quantiles = torch.stack(
            [median - lower_width, median, median + upper_width], dim=1
        )

        score = self.ordinal_score(hidden)
        ordinal_logits = score - self._cutpoints().view(1, -1)
        direct_target = 3.0 * torch.tanh(self.target_head(hidden))
        return {
            'quantiles': quantiles,
            'ordinal_logits': ordinal_logits,
            'direct_target': direct_target,
        }


def ordinal_probabilities(ordinal_logits):
    cumulative = torch.sigmoid(ordinal_logits)
    probabilities = torch.stack(
        [
            1.0 - cumulative[:, 0],
            cumulative[:, 0] - cumulative[:, 1],
            cumulative[:, 1] - cumulative[:, 2],
            cumulative[:, 2] - cumulative[:, 3],
            cumulative[:, 3],
        ],
        dim=1,
    ).clamp_min(0.0)
    return probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)


def pinball_loss(prediction, target, quantile):
    error = target - prediction
    return torch.maximum(quantile * error, (quantile - 1.0) * error)


def _strata(labels, fusion, margin):
    residual = (labels - fusion).view(-1)
    direction = torch.ones_like(residual, dtype=torch.long)
    direction[residual < -margin] = 0
    direction[residual > margin] = 2
    return _regions(labels) * 3 + direction


def stratified_folds(labels, fusion, margin, fold_count, seed):
    strata = _strata(labels, fusion, margin)
    generator = torch.Generator().manual_seed(int(seed))
    folds = [[] for _ in range(int(fold_count))]
    for value in torch.unique(strata).tolist():
        indices = torch.nonzero(strata == value, as_tuple=False).view(-1)
        indices = indices[torch.randperm(len(indices), generator=generator)]
        for position, sample_index in enumerate(indices.tolist()):
            folds[position % fold_count].append(sample_index)
    result = [torch.tensor(sorted(values), dtype=torch.long) for values in folds]
    if any(len(values) == 0 for values in result):
        raise RuntimeError('V3.3 stratification produced an empty fold.')
    return result


def _auxiliary_weights(labels):
    regions = _regions(labels)
    counts = torch.bincount(regions, minlength=len(REGION_NAMES)).float().clamp_min(1.0)
    values = counts.sum() / torch.sqrt(counts)
    weights = values[regions]
    return weights / weights.mean().clamp_min(1e-8)


def balanced_accuracy(target, prediction, class_count):
    recalls = []
    for value in range(class_count):
        mask = target == value
        if mask.any():
            recalls.append((prediction[mask] == value).float().mean())
    return float(torch.stack(recalls).mean().item()) if recalls else 0.0


def residual_metrics(quantiles, residual_target, margin=0.10):
    target = residual_target.view(-1)
    q10, q50, q90 = quantiles[:, 0], quantiles[:, 1], quantiles[:, 2]
    active = target.abs() > margin
    true_direction = target >= 0
    predicted_direction = q50 >= 0
    recalls = []
    for direction in (False, True):
        mask = active & (true_direction == direction)
        if mask.any():
            recalls.append((predicted_direction[mask] == direction).float().mean())
    sign_balanced = float(torch.stack(recalls).mean().item()) if recalls else 0.0
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
        'mean_interval_width': float((q90 - q10).mean().item()),
    }


def output_metrics(output, labels, fusion, residual_margin):
    regions = _regions(labels)
    region_probs = ordinal_probabilities(output['ordinal_logits'])
    region_prediction = region_probs.argmax(dim=1)
    result = residual_metrics(output['quantiles'], labels - fusion, residual_margin)
    result.update({
        'ordinal_accuracy': float((region_prediction == regions).float().mean().item()),
        'ordinal_balanced_accuracy': balanced_accuracy(regions, region_prediction, 5),
        'direct_target_mae': float(torch.abs(output['direct_target'] - labels).mean().item()),
    })
    return result


def _predict(model, cache, specialist_residuals, device, batch_size=512):
    model.eval()
    quantiles, ordinal_logits, direct_targets = [], [], []
    loader = DataLoader(
        TensorDataset(cache['context'], cache['semantic'], specialist_residuals),
        batch_size=batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for context, semantic, residuals in loader:
            output = model(context.to(device), semantic.to(device), residuals.to(device))
            quantiles.append(output['quantiles'].cpu())
            ordinal_logits.append(output['ordinal_logits'].cpu())
            direct_targets.append(output['direct_target'].cpu())
    logits = torch.cat(ordinal_logits, dim=0)
    return {
        'quantiles': torch.cat(quantiles, dim=0),
        'ordinal_logits': logits,
        'region_probs': ordinal_probabilities(logits),
        'direct_target': torch.cat(direct_targets, dim=0),
    }


def _loss_components(output, labels, fusion, auxiliary_weight, ordinal_weight, target_weight, median_weight):
    residual_target = (labels - fusion).view(-1)
    quantile_losses = []
    for column, quantile in enumerate(QUANTILES):
        quantile_losses.append(
            pinball_loss(output['quantiles'][:, column], residual_target, quantile)
        )
    # Deliberately unweighted: these retain ordinary conditional-quantile meaning.
    pinball = torch.stack(quantile_losses, dim=1).mean()
    median = F.smooth_l1_loss(output['quantiles'][:, 1], residual_target)

    regions = _regions(labels)
    thresholds = torch.arange(4, device=labels.device).view(1, -1)
    ordinal_target = (regions.view(-1, 1) > thresholds).float()
    ordinal_per_sample = F.binary_cross_entropy_with_logits(
        output['ordinal_logits'], ordinal_target, reduction='none'
    ).mean(dim=1)
    ordinal = (ordinal_per_sample * auxiliary_weight).mean()

    # The direct target head remains unweighted so its prediction keeps the
    # ordinary source-distribution regression meaning. Region balancing is
    # supplied by the separate ordinal auxiliary task.
    direct_target = F.smooth_l1_loss(
        output['direct_target'].view(-1), labels.view(-1)
    )
    total = pinball + median_weight * median + ordinal_weight * ordinal + target_weight * direct_target
    return total, pinball, median, ordinal, direct_target


def train_model(
    train_cache,
    specialist_residuals,
    train_indices,
    device,
    seed,
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
    validation_indices=None,
    patience=10,
    residual_margin=0.10,
):
    generator = torch.Generator().manual_seed(int(seed))
    model = OrdinalConformalNet(
        train_cache['context'].size(1),
        train_cache['semantic'].size(1),
        specialist_residuals.size(1),
        hidden_dim,
        semantic_hidden_dim,
        dropout,
        max_center,
        max_width,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    auxiliary_weights = _auxiliary_weights(train_cache['labels'])
    dataset = TensorDataset(
        train_cache['context'][train_indices],
        train_cache['semantic'][train_indices],
        specialist_residuals[train_indices],
        train_cache['expert_matrix'][train_indices],
        train_cache['labels'][train_indices],
        auxiliary_weights[train_indices],
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    best_state, best_epoch, best_score = None, 0, -float('inf')
    history = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        totals = {'loss': 0.0, 'pinball': 0.0, 'median': 0.0, 'ordinal': 0.0, 'target': 0.0}
        for context, semantic, residuals, experts, labels, weights in loader:
            optimizer.zero_grad()
            context = context.to(device)
            semantic = semantic.to(device)
            residuals = residuals.to(device)
            experts = experts.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            output = model(context, semantic, residuals)
            loss, pinball, median, ordinal, direct_target = _loss_components(
                output, labels, _fusion(experts), weights,
                ordinal_loss_weight, target_loss_weight, median_loss_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for name, value in (
                ('loss', loss), ('pinball', pinball), ('median', median),
                ('ordinal', ordinal), ('target', direct_target),
            ):
                totals[name] += float(value.item())
        row = {'epoch': epoch, **{key: value / max(1, len(loader)) for key, value in totals.items()}}
        if validation_indices is not None:
            output = _predict(
                model,
                {key: train_cache[key][validation_indices] if torch.is_tensor(train_cache[key]) and train_cache[key].ndim > 0 and len(train_cache[key]) == len(train_cache['labels']) else train_cache[key]
                 for key in train_cache},
                specialist_residuals[validation_indices],
                device,
            )
            metrics = output_metrics(
                output,
                train_cache['labels'][validation_indices],
                _fusion(train_cache['expert_matrix'][validation_indices]),
                residual_margin,
            )
            row.update({f'valid_{key}': value for key, value in metrics.items()})
            score = (
                0.35 * metrics['sign_balanced_accuracy']
                + 0.25 * max(0.0, metrics['residual_correlation'])
                + 0.25 * metrics['ordinal_balanced_accuracy']
                + 0.15 * max(0.0, 1.0 - metrics['direct_target_mae'] / 1.5)
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
        return model, history, int(epochs)
    if best_state is None:
        raise RuntimeError('V3.3 model selection produced no checkpoint.')
    model.load_state_dict(best_state)
    return model, history, best_epoch


def _subset_cache(cache, indices):
    result = {}
    sample_count = len(cache['labels'])
    for key, value in cache.items():
        if torch.is_tensor(value) and value.ndim > 0 and len(value) == sample_count:
            result[key] = value[indices]
        else:
            result[key] = value
    return result


def cross_fit_ordinal_quantile(
    train_cache,
    oof_specialist_residuals,
    valid_views,
    valid_residual_views,
    test_views,
    test_residual_views,
    device,
    seed,
    save_dir,
    fold_count=5,
    residual_margin=0.10,
    epochs=80,
    learning_rate=4e-4,
    hidden_dim=128,
    semantic_hidden_dim=96,
    dropout=0.20,
    batch_size=128,
    max_center=1.75,
    max_width=1.75,
    ordinal_loss_weight=0.30,
    target_loss_weight=0.25,
    median_loss_weight=0.20,
    patience=10,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fusion = _fusion(train_cache['expert_matrix'])
    folds = stratified_folds(
        train_cache['labels'], fusion, residual_margin, fold_count, seed + 2718
    )
    selection_holdout = folds[0]
    selection_mask = torch.ones(len(train_cache['labels']), dtype=torch.bool)
    selection_mask[selection_holdout] = False
    selection_indices = torch.arange(len(train_cache['labels']))[selection_mask]
    selection_model, selection_history, best_epoch = train_model(
        train_cache, oof_specialist_residuals, selection_indices, device, seed,
        epochs, learning_rate, hidden_dim, semantic_hidden_dim, dropout,
        batch_size, max_center, max_width, ordinal_loss_weight,
        target_loss_weight, median_loss_weight,
        validation_indices=selection_holdout, patience=patience,
        residual_margin=residual_margin,
    )
    del selection_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    n = len(train_cache['labels'])
    oof_quantiles = torch.zeros(n, 3)
    oof_ordinal_logits = torch.zeros(n, 4)
    oof_direct_target = torch.zeros(n, 1)
    valid_accumulators = [list() for _ in valid_views]
    test_accumulators = [list() for _ in test_views]
    fold_history = []
    all_indices = torch.arange(n)
    for fold_index, holdout in enumerate(folds, start=1):
        mask = torch.ones(n, dtype=torch.bool)
        mask[holdout] = False
        train_indices = all_indices[mask]
        model, history, _ = train_model(
            train_cache, oof_specialist_residuals, train_indices, device,
            seed + 1009 * fold_index, max(1, best_epoch), learning_rate,
            hidden_dim, semantic_hidden_dim, dropout, batch_size, max_center,
            max_width, ordinal_loss_weight, target_loss_weight,
            median_loss_weight, validation_indices=None, patience=patience,
            residual_margin=residual_margin,
        )
        holdout_output = _predict(
            model, _subset_cache(train_cache, holdout),
            oof_specialist_residuals[holdout], device,
        )
        oof_quantiles[holdout] = holdout_output['quantiles']
        oof_ordinal_logits[holdout] = holdout_output['ordinal_logits']
        oof_direct_target[holdout] = holdout_output['direct_target']
        for view_index, (view, residuals) in enumerate(zip(valid_views, valid_residual_views)):
            valid_accumulators[view_index].append(_predict(model, view, residuals, device))
        for view_index, (view, residuals) in enumerate(zip(test_views, test_residual_views)):
            test_accumulators[view_index].append(_predict(model, view, residuals, device))
        fold_history.extend({'fold': fold_index, **row} for row in history)
        torch.save(
            {'state_dict': model.state_dict(), 'fold': fold_index, 'best_epoch': int(best_epoch)},
            save_dir / f'v33_multitask_fold_{fold_index}.pth',
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def average_outputs(accumulators):
        result = []
        for values in accumulators:
            logits = torch.stack([value['ordinal_logits'] for value in values]).mean(0)
            result.append({
                'quantiles': torch.stack([value['quantiles'] for value in values]).mean(0),
                'ordinal_logits': logits,
                'region_probs': ordinal_probabilities(logits),
                'direct_target': torch.stack([value['direct_target'] for value in values]).mean(0),
            })
        return result

    oof_output = {
        'quantiles': oof_quantiles,
        'ordinal_logits': oof_ordinal_logits,
        'region_probs': ordinal_probabilities(oof_ordinal_logits),
        'direct_target': oof_direct_target,
    }
    return (
        oof_output,
        average_outputs(valid_accumulators),
        average_outputs(test_accumulators),
        selection_history,
        fold_history,
        best_epoch,
    )
