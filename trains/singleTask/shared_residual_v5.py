import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


REGION_NAMES = ('strong_negative', 'negative', 'boundary', 'positive', 'strong_positive')


def region_index(labels):
    values = torch.as_tensor(labels, dtype=torch.float32).view(-1)
    result = torch.full_like(values, 2, dtype=torch.long)
    result[values < -1.5] = 0
    result[(values >= -1.5) & (values < -0.5)] = 1
    result[(values > 0.5) & (values <= 1.5)] = 3
    result[values > 1.5] = 4
    return result


def pinball_loss(prediction, target, quantile):
    error = target - prediction
    return torch.maximum(quantile * error, (quantile - 1.0) * error).mean()


def safe_corr(prediction, target):
    x = prediction.view(-1).double()
    y = target.view(-1).double()
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.sqrt((x.square().sum()) * (y.square().sum())).clamp_min(1e-12)
    return float((x * y).sum().div(denominator).item())


def balanced_sign_accuracy(prediction, target):
    pred = prediction.view(-1) >= 0
    truth = target.view(-1) >= 0
    recalls = []
    for value in (False, True):
        mask = truth == value
        if mask.any():
            recalls.append((pred[mask] == truth[mask]).float().mean())
    if not recalls:
        return 0.0
    return float(torch.stack(recalls).mean().item())


class FoldFeatureNormalizer:
    def __init__(self, fold_count):
        self.fold_count = int(fold_count)
        self.stats = {}

    def fit(self, fusion_feature, context_feature, fold_ids):
        for fold in range(1, self.fold_count + 1):
            mask = fold_ids.view(-1) == fold
            if not mask.any():
                raise RuntimeError(f'No samples found for fold {fold}.')
            self.stats[fold] = {
                'fusion_mean': fusion_feature[mask].mean(dim=0, keepdim=True),
                'fusion_std': fusion_feature[mask].std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-4),
                'context_mean': context_feature[mask].mean(dim=0, keepdim=True),
                'context_std': context_feature[mask].std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-4),
            }
        return self

    def transform(self, fusion_feature, context_feature, fold_ids):
        fusion = torch.zeros_like(fusion_feature)
        context = torch.zeros_like(context_feature)
        for fold in range(1, self.fold_count + 1):
            mask = fold_ids.view(-1) == fold
            if not mask.any():
                continue
            stats = self.stats[fold]
            fusion[mask] = (fusion_feature[mask] - stats['fusion_mean']) / stats['fusion_std']
            context[mask] = (context_feature[mask] - stats['context_mean']) / stats['context_std']
        return fusion, context

    def state_dict(self):
        return {'fold_count': self.fold_count, 'stats': self.stats}

    @classmethod
    def from_state_dict(cls, state):
        instance = cls(state['fold_count'])
        instance.stats = state['stats']
        return instance


class FoldAdapter(nn.Module):
    def __init__(self, input_dim, output_dim=64, hidden_dim=96, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, x):
        return self.net(x)


class SharedResidualQuantileNet(nn.Module):
    def __init__(
        self,
        fusion_dim,
        context_dim,
        fold_count=3,
        adapter_dim=64,
        adapter_hidden_dim=96,
        context_hidden_dim=48,
        trunk_dim=128,
        dropout=0.20,
        residual_max=1.0,
    ):
        super().__init__()
        self.fold_count = int(fold_count)
        self.residual_max = float(residual_max)
        self.adapters = nn.ModuleList([
            FoldAdapter(
                fusion_dim,
                output_dim=adapter_dim,
                hidden_dim=adapter_hidden_dim,
                dropout=dropout * 0.75,
            )
            for _ in range(self.fold_count)
        ])
        self.context_branch = nn.Sequential(
            nn.LayerNorm(int(context_dim)),
            nn.Linear(int(context_dim), int(context_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout * 0.5)),
            nn.Linear(int(context_hidden_dim), 32),
            nn.GELU(),
        )
        input_dim = int(adapter_dim) + 32
        self.input_project = nn.Linear(input_dim, int(trunk_dim))
        self.trunk = nn.Sequential(
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(trunk_dim), int(trunk_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(trunk_dim), int(trunk_dim)),
        )
        self.trunk_norm = nn.LayerNorm(int(trunk_dim))
        self.quantile_head = nn.Linear(int(trunk_dim), 3)
        self.sign_head = nn.Linear(int(trunk_dim), 1)
        self.magnitude_head = nn.Linear(int(trunk_dim), 1)

    def encode(self, fusion_feature, context_feature, fold_ids):
        adapted = torch.zeros(
            fusion_feature.size(0),
            self.adapters[0].net[-1].out_features,
            device=fusion_feature.device,
            dtype=fusion_feature.dtype,
        )
        for fold in range(1, self.fold_count + 1):
            mask = fold_ids.view(-1) == fold
            if mask.any():
                adapted[mask] = self.adapters[fold - 1](fusion_feature[mask])
        context = self.context_branch(context_feature)
        hidden = self.input_project(torch.cat([adapted, context], dim=1))
        hidden = self.trunk_norm(hidden + self.trunk(hidden))
        return hidden, adapted

    def forward(self, fusion_feature, context_feature, fold_ids):
        hidden, adapted = self.encode(fusion_feature, context_feature, fold_ids)
        raw = self.quantile_head(hidden)
        q50 = self.residual_max * torch.tanh(raw[:, 0:1])
        lower_spread = self.residual_max * torch.sigmoid(raw[:, 1:2])
        upper_spread = self.residual_max * torch.sigmoid(raw[:, 2:3])
        q10 = q50 - lower_spread
        q90 = q50 + upper_spread
        return {
            'q10': q10,
            'q50': q50,
            'q90': q90,
            'sign_logit': self.sign_head(hidden),
            'magnitude': self.residual_max * torch.sigmoid(self.magnitude_head(hidden)),
            'adapted': adapted,
            'hidden': hidden,
        }


def fold_alignment_loss(adapted, fold_ids):
    groups = []
    for fold in sorted(torch.unique(fold_ids.view(-1)).tolist()):
        mask = fold_ids.view(-1) == int(fold)
        if mask.sum() >= 2:
            values = adapted[mask]
            groups.append((values.mean(dim=0), values.std(dim=0, unbiased=False)))
    if len(groups) < 2:
        return adapted.new_zeros(())
    mean_target = torch.stack([item[0] for item in groups]).mean(dim=0)
    std_target = torch.stack([item[1] for item in groups]).mean(dim=0)
    losses = []
    for mean, std in groups:
        losses.append((mean - mean_target).square().mean())
        losses.append((std - std_target).square().mean())
    return torch.stack(losses).mean()


def residual_losses(
    output,
    residual,
    labels,
    fold_ids,
    sign_pos_weight=None,
    huber_delta=0.25,
):
    q10, q50, q90 = output['q10'], output['q50'], output['q90']
    huber = F.huber_loss(q50, residual, delta=float(huber_delta))
    quantile = (
        pinball_loss(q10, residual, 0.10)
        + pinball_loss(q50, residual, 0.50)
        + pinball_loss(q90, residual, 0.90)
    ) / 3.0
    sign_target = (residual >= 0).float()
    sign = F.binary_cross_entropy_with_logits(
        output['sign_logit'], sign_target, pos_weight=sign_pos_weight
    )
    magnitude = F.smooth_l1_loss(output['magnitude'], residual.abs())
    regions = region_index(labels).to(labels.device)
    region_terms, bias_terms = [], []
    for region in range(len(REGION_NAMES)):
        mask = regions == region
        if mask.any():
            region_terms.append(F.huber_loss(q50[mask], residual[mask], delta=float(huber_delta)))
            bias_terms.append((residual[mask] - q50[mask]).mean().square())
    region_loss = torch.stack(region_terms).mean() if region_terms else huber.new_zeros(())
    bias_loss = torch.stack(bias_terms).mean() if bias_terms else huber.new_zeros(())
    alignment = fold_alignment_loss(output['adapted'], fold_ids)
    shrinkage = q50.abs().mean()
    return {
        'huber': huber,
        'quantile': quantile,
        'sign': sign,
        'magnitude': magnitude,
        'region': region_loss,
        'bias': bias_loss,
        'alignment': alignment,
        'shrinkage': shrinkage,
    }


def group_region_sampler_weights(fold_ids, labels):
    regions = region_index(labels)
    keys = fold_ids.view(-1).long() * 10 + regions
    unique, counts = torch.unique(keys, return_counts=True)
    count_map = {int(key): int(count) for key, count in zip(unique, counts)}
    return torch.tensor(
        [1.0 / max(1, count_map[int(key)]) for key in keys], dtype=torch.double
    )


def make_loader(
    fusion_feature,
    context_feature,
    anchor,
    labels,
    fold_ids,
    indices,
    batch_size,
    balanced=True,
):
    indices = torch.as_tensor(indices, dtype=torch.long)
    residual = labels - anchor
    dataset = TensorDataset(
        fusion_feature[indices],
        context_feature[indices],
        anchor[indices],
        labels[indices],
        residual[indices],
        fold_ids[indices],
    )
    if balanced:
        weights = group_region_sampler_weights(fold_ids[indices], labels[indices])
        sampler = WeightedRandomSampler(weights, num_samples=len(indices), replacement=True)
        return DataLoader(dataset, batch_size=int(batch_size), sampler=sampler)
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=False)


@dataclass
class TrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 80
    patience: int = 12
    batch_size: int = 128
    huber_delta: float = 0.25
    huber_weight: float = 1.0
    quantile_weight: float = 0.5
    sign_weight: float = 0.1
    magnitude_weight: float = 0.1
    region_weight: float = 0.3
    bias_weight: float = 0.1
    alignment_weight: float = 0.05
    shrinkage_weight: float = 0.005


def prediction_metrics(output, anchor, labels, fold_ids):
    residual = labels - anchor
    q50 = output['q50']
    prediction = anchor + q50
    absolute = torch.abs(prediction - labels)
    fold_maes = []
    for fold in sorted(torch.unique(fold_ids.view(-1)).tolist()):
        mask = fold_ids.view(-1) == int(fold)
        if mask.any():
            fold_maes.append(float(absolute[mask].mean().item()))
    regions = region_index(labels)
    region_maes = []
    for region in range(len(REGION_NAMES)):
        mask = regions == region
        if mask.any():
            region_maes.append(float(absolute[mask].mean().item()))
    coverage = ((residual >= output['q10']) & (residual <= output['q90'])).float().mean()
    return {
        'final_mae': float(absolute.mean().item()),
        'residual_mae': float(torch.abs(q50 - residual).mean().item()),
        'residual_corr': safe_corr(q50, residual),
        'sign_balanced_accuracy': balanced_sign_accuracy(q50, residual),
        'interval_coverage': float(coverage.item()),
        'interval_width': float((output['q90'] - output['q10']).mean().item()),
        'worst_fold_mae': max(fold_maes) if fold_maes else float('nan'),
        'worst_region_mae': max(region_maes) if region_maes else float('nan'),
    }


@torch.no_grad()
def predict_model(model, fusion_feature, context_feature, anchor, labels, fold_ids, device, batch_size=256):
    model.eval()
    dataset = TensorDataset(fusion_feature, context_feature, anchor, labels, fold_ids)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
    buffers = {key: [] for key in ('q10', 'q50', 'q90', 'sign_logit', 'magnitude')}
    for fusion, context, batch_anchor, batch_labels, batch_fold in loader:
        del batch_anchor, batch_labels
        output = model(fusion.to(device), context.to(device), batch_fold.to(device))
        for key in buffers:
            buffers[key].append(output[key].detach().cpu())
    return {key: torch.cat(value, dim=0) for key, value in buffers.items()}


def train_one_model(
    model,
    fusion_feature,
    context_feature,
    anchor,
    labels,
    fold_ids,
    train_indices,
    valid_indices,
    device,
    config,
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    residual_all = labels - anchor
    train_residual = residual_all[train_indices]
    positives = float((train_residual >= 0).sum().item())
    negatives = float((train_residual < 0).sum().item())
    sign_pos_weight = torch.tensor(
        [negatives / max(positives, 1.0)], dtype=torch.float32, device=device
    )
    loader = make_loader(
        fusion_feature,
        context_feature,
        anchor,
        labels,
        fold_ids,
        train_indices,
        config.batch_size,
        balanced=True,
    )
    best_state = None
    best_epoch = 0
    best_objective = float('inf')
    history = []
    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        totals = {}
        for fusion, context, batch_anchor, batch_labels, batch_residual, batch_fold in loader:
            del batch_anchor
            optimizer.zero_grad()
            output = model(fusion.to(device), context.to(device), batch_fold.to(device))
            losses = residual_losses(
                output,
                batch_residual.to(device),
                batch_labels.to(device),
                batch_fold.to(device),
                sign_pos_weight=sign_pos_weight,
                huber_delta=config.huber_delta,
            )
            total = (
                config.huber_weight * losses['huber']
                + config.quantile_weight * losses['quantile']
                + config.sign_weight * losses['sign']
                + config.magnitude_weight * losses['magnitude']
                + config.region_weight * losses['region']
                + config.bias_weight * losses['bias']
                + config.alignment_weight * losses['alignment']
                + config.shrinkage_weight * losses['shrinkage']
            )
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals['loss'] = totals.get('loss', 0.0) + float(total.detach().item())
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().item())
        row = {'epoch': int(epoch)}
        for key, value in totals.items():
            row[f'train_{key}'] = value / max(1, len(loader))
        if valid_indices is not None:
            valid_indices = torch.as_tensor(valid_indices, dtype=torch.long)
            output = predict_model(
                model,
                fusion_feature[valid_indices],
                context_feature[valid_indices],
                anchor[valid_indices],
                labels[valid_indices],
                fold_ids[valid_indices],
                device,
                batch_size=config.batch_size,
            )
            metrics = prediction_metrics(
                output,
                anchor[valid_indices],
                labels[valid_indices],
                fold_ids[valid_indices],
            )
            row.update({f'valid_{key}': value for key, value in metrics.items()})
            objective = (
                metrics['final_mae']
                + 0.15 * metrics['worst_fold_mae']
                + 0.10 * metrics['worst_region_mae']
            )
            row['valid_objective'] = float(objective)
            if objective < best_objective - 1e-6:
                best_objective = float(objective)
                best_epoch = int(epoch)
                best_state = copy.deepcopy(model.state_dict())
            if epoch - best_epoch >= int(config.patience):
                history.append(row)
                break
        history.append(row)
    if valid_indices is None:
        return model, history, int(config.epochs), None
    if best_state is None:
        raise RuntimeError('Residual model selection failed to produce a checkpoint.')
    model.load_state_dict(best_state)
    return model, history, best_epoch, best_objective
