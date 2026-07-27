import copy
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from .candidate_gain_v34 import build_candidate_features
from .residual_direction_core import _fusion, _regions

ALPHAS = (0.10, 0.125, 0.15, 0.20, 0.25)


class DownAlphaGainNet(nn.Module):
    """Predict gain, benefit probability, and severe-harm risk for each Down alpha."""

    def __init__(
        self,
        input_dim,
        alpha_count=len(ALPHAS),
        hidden_dim=128,
        alpha_hidden_dim=64,
        dropout=0.20,
        max_gain=0.75,
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
        self.alpha_towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, alpha_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(alpha_hidden_dim, 3),
            )
            for _ in range(int(alpha_count))
        ])

    def forward(self, features):
        hidden = self.shared(features)
        raw = torch.stack([tower(hidden) for tower in self.alpha_towers], dim=1)
        return {
            'predicted_gain': self.max_gain * torch.tanh(raw[:, :, 0]),
            'benefit_logits': raw[:, :, 1],
            'harm_logits': raw[:, :, 2],
        }


def down_alpha_targets(cache, specialist_residuals, alphas=ALPHAS, severe_harm=0.10):
    fusion = _fusion(cache['expert_matrix'])
    labels = cache['labels']
    fusion_error = torch.abs(fusion - labels)
    down_residual = specialist_residuals[:, 1:2]
    alpha_tensor = torch.tensor(alphas, dtype=fusion.dtype).view(1, -1)
    candidate_predictions = fusion + down_residual * alpha_tensor
    gains = fusion_error - torch.abs(candidate_predictions - labels)
    return {
        'gains': gains,
        'benefit': (gains > 0).float(),
        'severe_harm': (gains < -float(severe_harm)).float(),
    }


def _balanced_accuracy(target, prediction):
    recalls = []
    for value in (0, 1):
        mask = target == value
        if mask.any():
            recalls.append((prediction[mask] == value).float().mean())
    return float(torch.stack(recalls).mean().item()) if recalls else 0.0


def _correlation(prediction, target):
    prediction = prediction.view(-1) - prediction.mean()
    target = target.view(-1) - target.mean()
    denominator = torch.sqrt(
        prediction.pow(2).sum() * target.pow(2).sum()
    ).clamp_min(1e-12)
    return float((prediction * target).sum().item() / denominator.item())


def apply_logit_calibrators(output, calibrators):
    probabilities = {}
    for key, logit_key in (('benefit', 'benefit_logits'), ('harm', 'harm_logits')):
        logits = output[logit_key]
        values = []
        for alpha_index, params in enumerate(calibrators[key]):
            values.append(torch.sigmoid(
                float(params['scale']) * logits[:, alpha_index] + float(params['bias'])
            ))
        probabilities[key] = torch.stack(values, dim=1)
    return probabilities


def _fit_platt(logits, target):
    logits = logits.detach().float().view(-1)
    target = target.detach().float().view(-1)
    mean = float(target.mean().item())
    if mean <= 1e-6 or mean >= 1.0 - 1e-6:
        clipped = min(max(mean, 1e-5), 1.0 - 1e-5)
        return {'scale': 0.0, 'bias': math.log(clipped / (1.0 - clipped))}
    log_scale = torch.tensor(0.0, requires_grad=True)
    bias = torch.tensor(math.log(mean / (1.0 - mean)), requires_grad=True)
    optimizer = optim.LBFGS(
        [log_scale, bias], lr=0.25, max_iter=80, line_search_fn='strong_wolfe'
    )

    def closure():
        optimizer.zero_grad()
        scale = torch.exp(log_scale).clamp(1e-3, 100.0)
        loss = F.binary_cross_entropy_with_logits(scale * logits + bias, target)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except RuntimeError:
        pass
    return {
        'scale': float(torch.exp(log_scale.detach()).clamp(1e-3, 100.0).item()),
        'bias': float(bias.detach().item()),
    }


def fit_logit_calibrators(oof_output, targets, alphas=ALPHAS):
    result = {'alphas': [float(value) for value in alphas], 'benefit': [], 'harm': []}
    for alpha_index in range(len(alphas)):
        result['benefit'].append(_fit_platt(
            oof_output['benefit_logits'][:, alpha_index],
            targets['benefit'][:, alpha_index],
        ))
        result['harm'].append(_fit_platt(
            oof_output['harm_logits'][:, alpha_index],
            targets['severe_harm'][:, alpha_index],
        ))
    return result


def gain_model_metrics(output, targets, calibrators=None, alphas=ALPHAS, top_fraction=0.10):
    if calibrators is None:
        benefit_probability = torch.sigmoid(output['benefit_logits'])
        harm_probability = torch.sigmoid(output['harm_logits'])
    else:
        calibrated = apply_logit_calibrators(output, calibrators)
        benefit_probability = calibrated['benefit']
        harm_probability = calibrated['harm']
    rows = {}
    for alpha_index, alpha in enumerate(alphas):
        actual_gain = targets['gains'][:, alpha_index]
        predicted_gain = output['predicted_gain'][:, alpha_index]
        score = predicted_gain + 0.10 * (benefit_probability[:, alpha_index] - 0.5) \
            - 0.10 * harm_probability[:, alpha_index]
        top_count = max(1, int(math.ceil(len(actual_gain) * float(top_fraction))))
        top_indices = torch.topk(score, top_count).indices
        prefix = f'alpha_{str(alpha).replace(".", "p")}'
        rows.update({
            f'{prefix}_gain_mae': float(torch.abs(predicted_gain - actual_gain).mean().item()),
            f'{prefix}_gain_correlation': _correlation(predicted_gain, actual_gain),
            f'{prefix}_benefit_balanced_accuracy': _balanced_accuracy(
                targets['benefit'][:, alpha_index].long(),
                (benefit_probability[:, alpha_index] >= 0.5).long(),
            ),
            f'{prefix}_harm_balanced_accuracy': _balanced_accuracy(
                targets['severe_harm'][:, alpha_index].long(),
                (harm_probability[:, alpha_index] >= 0.5).long(),
            ),
            f'{prefix}_top_precision': float((actual_gain[top_indices] > 0).float().mean().item()),
            f'{prefix}_top_mean_gain': float(actual_gain[top_indices].mean().item()),
        })
    return rows


def _gain_stratified_folds(cache, specialist_residuals, fold_count, seed, alphas=ALPHAS):
    targets = down_alpha_targets(cache, specialist_residuals, alphas)
    middle = min(range(len(alphas)), key=lambda index: abs(float(alphas[index]) - 0.15))
    benefit = targets['benefit'][:, middle].long()
    severe = targets['severe_harm'][:, middle].long()
    strata = _regions(cache['labels']) * 4 + benefit * 2 + severe
    generator = torch.Generator().manual_seed(int(seed))
    folds = [[] for _ in range(int(fold_count))]
    for value in torch.unique(strata).tolist():
        indices = torch.nonzero(strata == value, as_tuple=False).view(-1)
        indices = indices[torch.randperm(len(indices), generator=generator)]
        for position, sample_index in enumerate(indices.tolist()):
            folds[position % int(fold_count)].append(sample_index)
    result = [torch.tensor(sorted(values), dtype=torch.long) for values in folds]
    if any(len(values) == 0 for values in result):
        raise RuntimeError('V3.4.1 gain stratification produced an empty fold.')
    return result


def _positive_weights(target):
    positive = target.sum(dim=0)
    negative = float(len(target)) - positive
    return (negative / positive.clamp_min(1.0)).clamp(0.25, 8.0)


def _predict(model, features, device, batch_size=512):
    model.eval()
    accumulators = {'predicted_gain': [], 'benefit_logits': [], 'harm_logits': []}
    loader = DataLoader(TensorDataset(features), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (batch_features,) in loader:
            output = model(batch_features.to(device))
            for key in accumulators:
                accumulators[key].append(output[key].cpu())
    return {key: torch.cat(values, dim=0) for key, values in accumulators.items()}


def _selection_score(metrics, alphas=ALPHAS):
    values = []
    for alpha in alphas:
        prefix = f'alpha_{str(alpha).replace(".", "p")}'
        values.append(
            0.30 * max(0.0, metrics[f'{prefix}_gain_correlation'])
            + 0.30 * metrics[f'{prefix}_benefit_balanced_accuracy']
            + 0.25 * metrics[f'{prefix}_top_precision']
            + 0.15 * metrics[f'{prefix}_harm_balanced_accuracy']
        )
    return float(sum(values) / len(values))


def train_down_alpha_model(
    train_cache,
    specialist_residuals,
    ordinal_output,
    train_indices,
    device,
    seed,
    alphas=ALPHAS,
    epochs=60,
    learning_rate=4e-4,
    hidden_dim=128,
    alpha_hidden_dim=64,
    dropout=0.20,
    batch_size=128,
    max_gain=0.75,
    gain_loss_weight=1.0,
    benefit_loss_weight=0.75,
    harm_loss_weight=0.50,
    validation_indices=None,
    patience=8,
):
    features = build_candidate_features(train_cache, specialist_residuals, ordinal_output)
    targets = down_alpha_targets(train_cache, specialist_residuals, alphas)
    benefit_pos_weight = _positive_weights(targets['benefit'][train_indices]).to(device)
    harm_pos_weight = _positive_weights(targets['severe_harm'][train_indices]).to(device)
    model = DownAlphaGainNet(
        features.size(1), len(alphas), hidden_dim, alpha_hidden_dim, dropout, max_gain
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
                output['benefit_logits'], batch_benefit, pos_weight=benefit_pos_weight
            )
            harm_loss = F.binary_cross_entropy_with_logits(
                output['harm_logits'], batch_harm, pos_weight=harm_pos_weight
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
            metrics = gain_model_metrics(prediction, validation_targets, alphas=alphas)
            row.update({f'valid_{key}': value for key, value in metrics.items()})
            score = _selection_score(metrics, alphas)
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
        raise RuntimeError('V3.4.1 gain model selection produced no checkpoint.')
    model.load_state_dict(best_state)
    return model, history, best_epoch


def cross_fit_down_alpha_gain(
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
    alphas=ALPHAS,
    fold_count=5,
    selection_fold_count=3,
    epochs=60,
    learning_rate=4e-4,
    hidden_dim=128,
    alpha_hidden_dim=64,
    dropout=0.20,
    batch_size=128,
    max_gain=0.75,
    gain_loss_weight=1.0,
    benefit_loss_weight=0.75,
    harm_loss_weight=0.50,
    patience=8,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    folds = _gain_stratified_folds(
        train_cache, oof_specialist_residuals, fold_count, seed + 2718, alphas
    )
    n = len(train_cache['labels'])
    all_indices = torch.arange(n)
    selection_histories, selected_epochs = [], []
    for selection_fold, holdout in enumerate(folds[:int(selection_fold_count)], start=1):
        mask = torch.ones(n, dtype=torch.bool)
        mask[holdout] = False
        model, history, best_epoch = train_down_alpha_model(
            train_cache, oof_specialist_residuals, oof_ordinal_output,
            all_indices[mask], device, seed + 313 * selection_fold,
            alphas, epochs, learning_rate, hidden_dim, alpha_hidden_dim,
            dropout, batch_size, max_gain, gain_loss_weight,
            benefit_loss_weight, harm_loss_weight,
            validation_indices=holdout, patience=patience,
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
    oof_output = {
        'predicted_gain': torch.zeros(n, len(alphas)),
        'benefit_logits': torch.zeros(n, len(alphas)),
        'harm_logits': torch.zeros(n, len(alphas)),
    }
    valid_accumulators = [list() for _ in valid_views]
    test_accumulators = [list() for _ in test_views]
    fold_history = []
    for fold_index, holdout in enumerate(folds, start=1):
        mask = torch.ones(n, dtype=torch.bool)
        mask[holdout] = False
        model, history, _ = train_down_alpha_model(
            train_cache, oof_specialist_residuals, oof_ordinal_output,
            all_indices[mask], device, seed + 1009 * fold_index,
            alphas, max(1, best_epoch), learning_rate, hidden_dim,
            alpha_hidden_dim, dropout, batch_size, max_gain,
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
        torch.save({
            'state_dict': model.state_dict(),
            'fold': fold_index,
            'best_epoch': int(best_epoch),
            'alphas': [float(value) for value in alphas],
        }, save_dir / f'v341_down_gain_fold_{fold_index}.pth')
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def average_outputs(accumulators):
        return [{
            key: torch.stack([value[key] for value in values]).mean(0)
            for key in ('predicted_gain', 'benefit_logits', 'harm_logits')
        } for values in accumulators]

    return (
        oof_output,
        average_outputs(valid_accumulators),
        average_outputs(test_accumulators),
        selection_histories,
        fold_history,
        best_epoch,
    )
