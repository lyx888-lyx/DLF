import copy
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from .residual_direction_core import _fusion, _regions

CANDIDATE_NAMES = ('up', 'down')


class CandidateGainNet(nn.Module):
    """Predict candidate gain, probability of improvement, and severe-harm risk."""

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        candidate_hidden_dim=64,
        dropout=0.20,
        max_gain=1.50,
    ):
        super().__init__()
        self.max_gain = float(max_gain)
        self.shared = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, candidate_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(candidate_hidden_dim, 3),
            )
            for _ in CANDIDATE_NAMES
        ])

    def forward(self, features):
        hidden = self.shared(features)
        raw = torch.stack([tower(hidden) for tower in self.towers], dim=1)
        return {
            'predicted_gain': self.max_gain * torch.tanh(raw[:, :, 0]),
            'benefit_logits': raw[:, :, 1],
            'harm_logits': raw[:, :, 2],
        }


def candidate_gain_targets(cache, specialist_residuals, severe_harm=0.10):
    fusion = _fusion(cache['expert_matrix'])
    labels = cache['labels']
    fusion_error = torch.abs(fusion - labels)
    candidate_predictions = torch.cat([
        fusion + specialist_residuals[:, 0:1],
        fusion + specialist_residuals[:, 1:2],
    ], dim=1)
    gains = fusion_error - torch.abs(candidate_predictions - labels)
    return {
        'gains': gains,
        'benefit': (gains > 0).float(),
        'severe_harm': (gains < -float(severe_harm)).float(),
    }


def build_candidate_features(cache, specialist_residuals, ordinal_output):
    experts = cache['expert_matrix'].float()
    fusion = _fusion(experts)
    quantiles = ordinal_output['quantiles'].float()
    width = quantiles[:, 2:3] - quantiles[:, 0:1]
    median = quantiles[:, 1:2]
    direct_residual = ordinal_output['direct_target'].float() - fusion
    conflict = cache.get(
        'conflict_score', experts.std(dim=1, keepdim=True, unbiased=False)
    ).float()
    return torch.cat([
        cache['context'].float(),
        cache['semantic'].float(),
        specialist_residuals.float(),
        experts,
        fusion,
        conflict,
        quantiles,
        width,
        median.abs(),
        direct_residual,
        ordinal_output['ordinal_logits'].float(),
        ordinal_output['region_probs'].float(),
    ], dim=1)


def _balanced_accuracy(target, prediction):
    recalls = []
    for value in (0, 1):
        mask = target == value
        if mask.any():
            recalls.append((prediction[mask] == value).float().mean())
    return float(torch.stack(recalls).mean().item()) if recalls else 0.0


def _correlation(prediction, target):
    prediction = prediction.view(-1)
    target = target.view(-1)
    prediction = prediction - prediction.mean()
    target = target - target.mean()
    denominator = torch.sqrt(
        prediction.pow(2).sum() * target.pow(2).sum()
    ).clamp_min(1e-12)
    return float((prediction * target).sum().item() / denominator.item())


def gain_model_metrics(output, targets, top_fraction=0.10):
    predicted_gain = output['predicted_gain']
    benefit_probability = torch.sigmoid(output['benefit_logits'])
    harm_probability = torch.sigmoid(output['harm_logits'])
    rows = {}
    scores = predicted_gain + 0.10 * (benefit_probability - 0.5) - 0.10 * harm_probability
    for index, name in enumerate(CANDIDATE_NAMES):
        actual_gain = targets['gains'][:, index]
        benefit_target = targets['benefit'][:, index].long()
        harm_target = targets['severe_harm'][:, index].long()
        benefit_prediction = (benefit_probability[:, index] >= 0.5).long()
        harm_prediction = (harm_probability[:, index] >= 0.5).long()
        top_count = max(1, int(math.ceil(len(actual_gain) * float(top_fraction))))
        top_indices = torch.topk(scores[:, index], top_count).indices
        rows.update({
            f'{name}_gain_mae': float(torch.abs(predicted_gain[:, index] - actual_gain).mean().item()),
            f'{name}_gain_correlation': _correlation(predicted_gain[:, index], actual_gain),
            f'{name}_benefit_balanced_accuracy': _balanced_accuracy(
                benefit_target, benefit_prediction
            ),
            f'{name}_harm_balanced_accuracy': _balanced_accuracy(
                harm_target, harm_prediction
            ),
            f'{name}_top_precision': float(
                (actual_gain[top_indices] > 0).float().mean().item()
            ),
            f'{name}_top_mean_gain': float(actual_gain[top_indices].mean().item()),
            f'{name}_mean_benefit_probability': float(
                benefit_probability[:, index].mean().item()
            ),
            f'{name}_mean_harm_probability': float(
                harm_probability[:, index].mean().item()
            ),
        })
    return rows


def _gain_stratified_folds(cache, specialist_residuals, fold_count, seed):
    targets = candidate_gain_targets(cache, specialist_residuals)
    benefit = targets['benefit'].long()
    strata = _regions(cache['labels']) * 4 + benefit[:, 0] * 2 + benefit[:, 1]
    generator = torch.Generator().manual_seed(int(seed))
    folds = [[] for _ in range(int(fold_count))]
    for value in torch.unique(strata).tolist():
        indices = torch.nonzero(strata == value, as_tuple=False).view(-1)
        indices = indices[torch.randperm(len(indices), generator=generator)]
        for position, sample_index in enumerate(indices.tolist()):
            folds[position % fold_count].append(sample_index)
    result = [torch.tensor(sorted(values), dtype=torch.long) for values in folds]
    if any(len(values) == 0 for values in result):
        raise RuntimeError('Candidate-gain stratification produced an empty fold.')
    return result


def _positive_weights(target):
    positive = target.sum(dim=0)
    negative = float(len(target)) - positive
    return (negative / positive.clamp_min(1.0)).clamp(0.25, 8.0)


def _predict(model, features, device, batch_size=512):
    model.eval()
    predicted_gain, benefit_logits, harm_logits = [], [], []
    loader = DataLoader(TensorDataset(features), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (batch_features,) in loader:
            output = model(batch_features.to(device))
            predicted_gain.append(output['predicted_gain'].cpu())
            benefit_logits.append(output['benefit_logits'].cpu())
            harm_logits.append(output['harm_logits'].cpu())
    return {
        'predicted_gain': torch.cat(predicted_gain, dim=0),
        'benefit_logits': torch.cat(benefit_logits, dim=0),
        'harm_logits': torch.cat(harm_logits, dim=0),
    }


def _selection_score(metrics):
    values = []
    for name in CANDIDATE_NAMES:
        values.append(
            0.30 * max(0.0, metrics[f'{name}_gain_correlation'])
            + 0.30 * metrics[f'{name}_benefit_balanced_accuracy']
            + 0.25 * metrics[f'{name}_top_precision']
            + 0.15 * metrics[f'{name}_harm_balanced_accuracy']
        )
    return float(sum(values) / len(values))


def train_gain_model(
    train_cache,
    specialist_residuals,
    ordinal_output,
    train_indices,
    device,
    seed,
    epochs=60,
    learning_rate=4e-4,
    hidden_dim=128,
    candidate_hidden_dim=64,
    dropout=0.20,
    batch_size=128,
    max_gain=1.50,
    gain_loss_weight=1.0,
    benefit_loss_weight=0.75,
    harm_loss_weight=0.50,
    validation_indices=None,
    patience=8,
):
    features = build_candidate_features(train_cache, specialist_residuals, ordinal_output)
    targets = candidate_gain_targets(train_cache, specialist_residuals)
    benefit_pos_weight = _positive_weights(targets['benefit']).to(device)
    harm_pos_weight = _positive_weights(targets['severe_harm']).to(device)
    model = CandidateGainNet(
        features.size(1), hidden_dim, candidate_hidden_dim, dropout, max_gain
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        TensorDataset(
            features[train_indices],
            targets['gains'][train_indices],
            targets['benefit'][train_indices],
            targets['severe_harm'][train_indices],
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    best_state, best_epoch, best_score = None, 0, -float('inf')
    history = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        totals = {'loss': 0.0, 'gain_loss': 0.0, 'benefit_loss': 0.0, 'harm_loss': 0.0}
        for batch_features, batch_gain, batch_benefit, batch_harm in loader:
            optimizer.zero_grad()
            output = model(batch_features.to(device))
            batch_gain = batch_gain.to(device)
            batch_benefit = batch_benefit.to(device)
            batch_harm = batch_harm.to(device)
            gain_loss = F.smooth_l1_loss(output['predicted_gain'], batch_gain)
            benefit_loss = F.binary_cross_entropy_with_logits(
                output['benefit_logits'], batch_benefit,
                pos_weight=benefit_pos_weight,
            )
            harm_loss = F.binary_cross_entropy_with_logits(
                output['harm_logits'], batch_harm,
                pos_weight=harm_pos_weight,
            )
            loss = (
                float(gain_loss_weight) * gain_loss
                + float(benefit_loss_weight) * benefit_loss
                + float(harm_loss_weight) * harm_loss
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for key, value in (
                ('loss', loss), ('gain_loss', gain_loss),
                ('benefit_loss', benefit_loss), ('harm_loss', harm_loss),
            ):
                totals[key] += float(value.item())
        row = {'epoch': epoch, **{
            key: value / max(1, len(loader)) for key, value in totals.items()
        }}
        if validation_indices is not None:
            prediction = _predict(model, features[validation_indices], device)
            validation_targets = {
                key: value[validation_indices] for key, value in targets.items()
            }
            metrics = gain_model_metrics(prediction, validation_targets)
            row.update({f'valid_{key}': value for key, value in metrics.items()})
            score = _selection_score(metrics)
            row['valid_selection_score'] = score
            if score > best_score + 1e-6:
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                best_score = score
            if epoch - best_epoch >= int(patience):
                history.append(row)
                break
        history.append(row)
    if validation_indices is None:
        return model, history, int(epochs)
    if best_state is None:
        raise RuntimeError('Candidate-gain model selection produced no checkpoint.')
    model.load_state_dict(best_state)
    return model, history, best_epoch


def cross_fit_candidate_gain(
    train_cache,
    oof_specialist_residuals,
    oof_ordinal_output,
    valid_views,
    valid_residual_views,
    valid_ordinal_outputs,
    test_views,
    test_residual_views,
    test_ordinal_outputs,
    device,
    seed,
    save_dir,
    fold_count=5,
    selection_fold_count=3,
    epochs=60,
    learning_rate=4e-4,
    hidden_dim=128,
    candidate_hidden_dim=64,
    dropout=0.20,
    batch_size=128,
    max_gain=1.50,
    gain_loss_weight=1.0,
    benefit_loss_weight=0.75,
    harm_loss_weight=0.50,
    patience=8,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    folds = _gain_stratified_folds(
        train_cache, oof_specialist_residuals, fold_count, seed + 2718
    )
    all_indices = torch.arange(len(train_cache['labels']))

    selection_histories, selected_epochs = [], []
    for selection_fold, holdout in enumerate(folds[:int(selection_fold_count)], start=1):
        mask = torch.ones(len(all_indices), dtype=torch.bool)
        mask[holdout] = False
        model, history, best_epoch = train_gain_model(
            train_cache, oof_specialist_residuals, oof_ordinal_output,
            all_indices[mask], device, seed + 313 * selection_fold,
            epochs, learning_rate, hidden_dim, candidate_hidden_dim, dropout,
            batch_size, max_gain, gain_loss_weight, benefit_loss_weight,
            harm_loss_weight, validation_indices=holdout, patience=patience,
        )
        selected_epochs.append(int(best_epoch))
        selection_histories.extend(
            {'selection_fold': selection_fold, **row} for row in history
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    selected_epochs.sort()
    best_epoch = selected_epochs[len(selected_epochs) // 2]

    train_features = build_candidate_features(
        train_cache, oof_specialist_residuals, oof_ordinal_output
    )
    n = len(train_features)
    oof_output = {
        'predicted_gain': torch.zeros(n, 2),
        'benefit_logits': torch.zeros(n, 2),
        'harm_logits': torch.zeros(n, 2),
    }
    valid_accumulators = [list() for _ in valid_views]
    test_accumulators = [list() for _ in test_views]
    fold_history = []
    for fold_index, holdout in enumerate(folds, start=1):
        mask = torch.ones(n, dtype=torch.bool)
        mask[holdout] = False
        model, history, _ = train_gain_model(
            train_cache, oof_specialist_residuals, oof_ordinal_output,
            all_indices[mask], device, seed + 1009 * fold_index,
            max(1, best_epoch), learning_rate, hidden_dim,
            candidate_hidden_dim, dropout, batch_size, max_gain,
            gain_loss_weight, benefit_loss_weight, harm_loss_weight,
            validation_indices=None, patience=patience,
        )
        prediction = _predict(model, train_features[holdout], device)
        for key in oof_output:
            oof_output[key][holdout] = prediction[key]
        for view_index, (view, residuals, ordinal) in enumerate(zip(
            valid_views, valid_residual_views, valid_ordinal_outputs
        )):
            features = build_candidate_features(view, residuals, ordinal)
            valid_accumulators[view_index].append(_predict(model, features, device))
        for view_index, (view, residuals, ordinal) in enumerate(zip(
            test_views, test_residual_views, test_ordinal_outputs
        )):
            features = build_candidate_features(view, residuals, ordinal)
            test_accumulators[view_index].append(_predict(model, features, device))
        fold_history.extend({'fold': fold_index, **row} for row in history)
        torch.save(
            {'state_dict': model.state_dict(), 'fold': fold_index, 'best_epoch': best_epoch},
            save_dir / f'v34_gain_fold_{fold_index}.pth',
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def average_outputs(accumulators):
        rows = []
        for values in accumulators:
            rows.append({
                key: torch.stack([value[key] for value in values]).mean(0)
                for key in ('predicted_gain', 'benefit_logits', 'harm_logits')
            })
        return rows

    return (
        oof_output,
        average_outputs(valid_accumulators),
        average_outputs(test_accumulators),
        selection_histories,
        fold_history,
        best_epoch,
    )
