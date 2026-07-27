import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from .residual_direction_core import SPECIALIST_NAMES, _fusion, _make_loader, _regions


class NeedDirectionClassifier(nn.Module):
    def __init__(self, context_dim, hidden_dim=64, dropout=0.15):
        super().__init__()
        input_dim = context_dim + len(SPECIALIST_NAMES)
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.need_head = nn.Linear(hidden_dim, 1)
        self.direction_head = nn.Linear(hidden_dim, 2)

    def forward(self, context, residuals):
        hidden = self.encoder(torch.cat([context, residuals], dim=1))
        return {
            'need_logit': self.need_head(hidden).view(-1),
            'direction_logits': self.direction_head(hidden),
        }


def targets(labels, fusion, margin):
    residual = (labels - fusion).view(-1)
    return (torch.abs(residual) > margin).long(), (residual > 0).long()


def balanced_binary_accuracy(target, prediction):
    recalls = []
    for value in (0, 1):
        mask = target == value
        if mask.any():
            recalls.append((prediction[mask] == value).float().mean())
    return float(torch.stack(recalls).mean().item()) if recalls else 0.0


def classifier_metrics(output, need_target, direction_target):
    need_probability = torch.sigmoid(output['need_logit'])
    need_prediction = (need_probability >= 0.5).long()
    direction_probability = F.softmax(output['direction_logits'], dim=1)
    direction_prediction = direction_probability.argmax(dim=1)
    active = need_target.bool()
    return {
        'need_accuracy': float((need_prediction == need_target).float().mean().item()),
        'need_balanced_accuracy': balanced_binary_accuracy(need_target, need_prediction),
        'need_mean_confidence': float(torch.maximum(need_probability, 1 - need_probability).mean().item()),
        'need_recall': float((need_prediction[active] == 1).float().mean().item()) if active.any() else 0.0,
        'keep_recall': float((need_prediction[~active] == 0).float().mean().item()) if (~active).any() else 0.0,
        'direction_accuracy_on_need': float((direction_prediction[active] == direction_target[active]).float().mean().item()) if active.any() else 0.0,
        'direction_balanced_accuracy_on_need': balanced_binary_accuracy(direction_target[active], direction_prediction[active]) if active.any() else 0.0,
        'direction_mean_confidence': float(direction_probability.max(dim=1).values.mean().item()),
    }


def stratified_indices(labels, fusion, margin, fold_count, seed):
    need, direction = targets(labels, fusion, margin)
    strata = _regions(labels) * 4 + need * 2 + direction
    generator = torch.Generator().manual_seed(int(seed))
    folds = [[] for _ in range(fold_count)]
    for value in torch.unique(strata).tolist():
        indices = torch.nonzero(strata == value, as_tuple=False).view(-1)
        indices = indices[torch.randperm(len(indices), generator=generator)]
        for position, sample_index in enumerate(indices.tolist()):
            folds[position % fold_count].append(sample_index)
    result = [torch.tensor(sorted(fold), dtype=torch.long) for fold in folds]
    if any(len(fold) == 0 for fold in result):
        raise RuntimeError('Classifier stratification produced an empty fold.')
    return result


def predict_classifier(model, context, residuals, device, batch_size=512):
    model.eval()
    need_logits, direction_logits = [], []
    loader = DataLoader(TensorDataset(context, residuals), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_context, batch_residuals in loader:
            output = model(batch_context.to(device), batch_residuals.to(device))
            need_logits.append(output['need_logit'].cpu())
            direction_logits.append(output['direction_logits'].cpu())
    return {
        'need_logit': torch.cat(need_logits),
        'direction_logits': torch.cat(direction_logits),
    }


def _class_weights(target):
    counts = torch.bincount(target, minlength=2).float().clamp_min(1)
    weights = counts.sum() / torch.sqrt(counts)
    return weights / weights.mean()


def train_classifier_once(
    context, residuals, expert_matrix, labels, device, seed, residual_margin,
    epochs, learning_rate, hidden_dim, dropout, batch_size,
    validation_indices=None, patience=8, label_smoothing=0.05,
    direction_loss_weight=1.0,
):
    fusion = _fusion(expert_matrix)
    need_target, direction_target = targets(labels, fusion, residual_margin)
    positive = need_target.float().sum()
    negative = float(len(need_target)) - positive
    pos_weight = torch.tensor(float(negative / max(1.0, float(positive))), device=device)
    direction_weights = _class_weights(direction_target[need_target.bool()]).to(device)
    model = NeedDirectionClassifier(context.size(1), hidden_dim, dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    all_indices = torch.arange(len(labels))
    train_indices = all_indices
    if validation_indices is not None:
        mask = torch.ones(len(labels), dtype=torch.bool)
        mask[validation_indices] = False
        train_indices = all_indices[mask]
    loader = _make_loader(
        context[train_indices], residuals[train_indices],
        need_target[train_indices], direction_target[train_indices],
        batch_size=batch_size, shuffle=True, seed=seed,
    )
    history, best_state, best_epoch, best_score = [], None, 0, -float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {'loss': 0.0, 'need_loss': 0.0, 'direction_loss': 0.0}
        for batch_context, batch_residuals, batch_need, batch_direction in loader:
            optimizer.zero_grad()
            output = model(batch_context.to(device), batch_residuals.to(device))
            batch_need = batch_need.to(device)
            batch_direction = batch_direction.to(device)
            need_loss = F.binary_cross_entropy_with_logits(
                output['need_logit'], batch_need.float(), pos_weight=pos_weight
            )
            active = batch_need.bool()
            direction_loss = (
                F.cross_entropy(
                    output['direction_logits'][active], batch_direction[active],
                    weight=direction_weights, label_smoothing=label_smoothing,
                ) if active.any() else need_loss.new_tensor(0.0)
            )
            loss = need_loss + direction_loss_weight * direction_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals['loss'] += loss.item()
            totals['need_loss'] += need_loss.item()
            totals['direction_loss'] += direction_loss.item()
        row = {'epoch': epoch, **{k: v / max(1, len(loader)) for k, v in totals.items()}}
        if validation_indices is not None:
            output = predict_classifier(model, context[validation_indices], residuals[validation_indices], device)
            metrics = classifier_metrics(output, need_target[validation_indices], direction_target[validation_indices])
            row.update({f'valid_{k}': v for k, v in metrics.items()})
            score = 0.4 * metrics['need_balanced_accuracy'] + 0.6 * metrics['direction_balanced_accuracy_on_need']
            row['valid_selection_score'] = score
            if score > best_score + 1e-6:
                best_state, best_epoch, best_score = copy.deepcopy(model.state_dict()), epoch, score
            if epoch - best_epoch >= patience:
                history.append(row)
                break
        history.append(row)
    if validation_indices is None:
        return model, history, epochs
    if best_state is None:
        raise RuntimeError('Need/direction classifier did not produce a checkpoint.')
    model.load_state_dict(best_state)
    return model, history, best_epoch


def train_need_direction_classifier(
    train_cache, oof_residuals, device, seed, residual_margin=0.10,
    epochs=60, learning_rate=5e-4, hidden_dim=64, dropout=0.15,
    batch_size=128, patience=8, label_smoothing=0.05,
    direction_loss_weight=1.0,
):
    fusion = _fusion(train_cache['expert_matrix'])
    validation_indices = stratified_indices(
        train_cache['labels'], fusion, residual_margin, 5, seed + 2718
    )[0]
    _, selection_history, best_epoch = train_classifier_once(
        train_cache['context'], oof_residuals, train_cache['expert_matrix'],
        train_cache['labels'], device, seed, residual_margin, epochs,
        learning_rate, hidden_dim, dropout, batch_size,
        validation_indices=validation_indices, patience=patience,
        label_smoothing=label_smoothing,
        direction_loss_weight=direction_loss_weight,
    )
    model, final_history, _ = train_classifier_once(
        train_cache['context'], oof_residuals, train_cache['expert_matrix'],
        train_cache['labels'], device, seed + 1, residual_margin,
        max(1, best_epoch), learning_rate, hidden_dim, dropout, batch_size,
        validation_indices=None, patience=patience,
        label_smoothing=label_smoothing,
        direction_loss_weight=direction_loss_weight,
    )
    return model, selection_history, final_history, best_epoch


def ensemble_classifier_metrics(outputs, views, residual_margin):
    need_probability = torch.stack(
        [torch.sigmoid(output['need_logit']) for output in outputs]
    ).mean(0)
    direction_probability = torch.stack(
        [F.softmax(output['direction_logits'], dim=1) for output in outputs]
    ).mean(0)
    ensemble_output = {
        'need_logit': torch.logit(need_probability.clamp(1e-6, 1 - 1e-6)),
        'direction_logits': torch.log(direction_probability.clamp_min(1e-8)),
    }
    labels = views[0]['labels']
    fusion = torch.stack([_fusion(view['expert_matrix']) for view in views]).mean(0)
    need, direction = targets(labels, fusion, residual_margin)
    return classifier_metrics(ensemble_output, need, direction)
