import copy
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
from tqdm import tqdm

from .expert_analysis import extract_expert_logits, normalize_batch_ids

logger = logging.getLogger('MMSA')

BASE_EXPERT_NAMES = ('text', 'audio', 'video', 'common', 'fusion')
SPECIALIST_NAMES = ('positive', 'negative', 'boundary', 'strong', 'conflict')
FUSION_INDEX = BASE_EXPERT_NAMES.index('fusion')


class _HighLevelFeatureCapture:
    """Capture the exact tensors fed to DLF's five prediction heads."""

    def __init__(self, model):
        self.values = {}
        self.handles = []
        modules = {
            'text_high': model.out_layer_l_high,
            'audio_high': model.out_layer_a_high,
            'video_high': model.out_layer_v_high,
            'common_high': model.out_layer_c,
            'fusion_high': model.out_layer,
        }
        for name, module in modules.items():
            self.handles.append(
                module.register_forward_pre_hook(self._make_hook(name))
            )

    def _make_hook(self, name):
        def hook(_module, inputs):
            self.values[name] = inputs[0].detach()
        return hook

    def clear(self):
        self.values.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class SpecialistResidualExperts(nn.Module):
    """Five bounded residual experts trained for different sentiment regions."""

    def __init__(
        self,
        context_dim,
        hidden_dim=96,
        head_hidden_dim=32,
        dropout=0.15,
        max_residual=0.75,
    ):
        super().__init__()
        self.max_residual = float(max_residual)
        self.encoder = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden_dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, 1),
            )
            for name in SPECIALIST_NAMES
        })

    def forward(self, context):
        hidden = self.encoder(context)
        raw = torch.cat([self.heads[name](hidden) for name in SPECIALIST_NAMES], dim=1)
        return self.max_residual * torch.tanh(raw)


class ResidualMixtureGate(nn.Module):
    """A fusion-anchored soft gate over bounded specialist corrections."""

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
            self.network[-1].bias[0] = 1.5

    def forward(self, context, residuals):
        logits = self.network(torch.cat([context, residuals], dim=1))
        weights = F.softmax(logits, dim=1)
        correction = torch.sum(weights[:, 1:] * residuals, dim=1, keepdim=True)
        return {
            'logits': logits,
            'weights': weights,
            'correction': correction,
        }


def _safe_metrics(metrics_fn, prediction, target):
    return {key: float(value) for key, value in metrics_fn(prediction, target).items()}


def _stack_base_experts(model_output):
    experts = extract_expert_logits(model_output)
    return torch.cat(
        [experts[name].view(-1, 1) for name in BASE_EXPERT_NAMES],
        dim=1,
    )


def _cosine_distance(left, right):
    return 1.0 - F.cosine_similarity(left, right, dim=1).unsqueeze(1)


def _build_context(captured, expert_matrix):
    required = {'text_high', 'audio_high', 'video_high', 'common_high', 'fusion_high'}
    missing = required.difference(captured)
    if missing:
        raise RuntimeError(f'Missing captured DLF features: {sorted(missing)}')

    text = captured['text_high']
    audio = captured['audio_high']
    video = captured['video_high']
    common = captured['common_high']
    fusion = captured['fusion_high']

    if common.size(1) % 3 != 0:
        raise ValueError('DLF common representation cannot be split into three modalities.')
    common_l, common_v, common_a = torch.chunk(common, 3, dim=1)

    conflicts = torch.cat([
        _cosine_distance(text, audio),
        _cosine_distance(text, video),
        _cosine_distance(audio, video),
        _cosine_distance(common_l, common_a),
        _cosine_distance(common_l, common_v),
        _cosine_distance(common_a, common_v),
    ], dim=1)
    norms = torch.cat([
        value.norm(p=2, dim=1, keepdim=True)
        for value in (text, audio, video, common, fusion)
    ], dim=1)

    fusion_prediction = expert_matrix[:, FUSION_INDEX:FUSION_INDEX + 1]
    prediction_differences = expert_matrix[:, :FUSION_INDEX] - fusion_prediction
    prediction_mean = expert_matrix.mean(dim=1, keepdim=True)
    prediction_std = expert_matrix.std(dim=1, keepdim=True, unbiased=False)
    prediction_min = expert_matrix.min(dim=1, keepdim=True).values
    prediction_max = expert_matrix.max(dim=1, keepdim=True).values
    prediction_span = prediction_max - prediction_min
    prediction_features = torch.cat([
        expert_matrix,
        prediction_differences,
        prediction_mean,
        prediction_std,
        prediction_min,
        prediction_max,
        prediction_span,
    ], dim=1)

    context = torch.cat([
        text,
        audio,
        video,
        common,
        fusion,
        norms,
        conflicts,
        prediction_features,
    ], dim=1)
    conflict_score = conflicts.mean(dim=1, keepdim=True) + prediction_std
    return context, conflict_score


def _cache_split(model, dataloader, device, split_name):
    model.eval()
    expert_matrices = []
    contexts = []
    conflicts = []
    labels_all = []
    ids_all = []
    capture = _HighLevelFeatureCapture(model)
    logger.info('Caching frozen DLF predictions and high-level features for %s.', split_name)
    try:
        with torch.no_grad():
            for batch_data in tqdm(dataloader):
                capture.clear()
                vision = batch_data['vision'].to(device)
                audio = batch_data['audio'].to(device)
                text = batch_data['text'].to(device)
                labels = batch_data['labels']['M'].to(device).view(-1, 1)
                model_output = model(text, audio, vision)
                expert_matrix = _stack_base_experts(model_output)
                context, conflict_score = _build_context(capture.values, expert_matrix)
                expert_matrices.append(expert_matrix.cpu())
                contexts.append(context.cpu())
                conflicts.append(conflict_score.cpu())
                labels_all.append(labels.cpu())
                ids_all.extend(normalize_batch_ids(batch_data.get('id')))
    finally:
        capture.close()

    return {
        'expert_matrix': torch.cat(expert_matrices, dim=0),
        'context': torch.cat(contexts, dim=0),
        'conflict_score': torch.cat(conflicts, dim=0),
        'labels': torch.cat(labels_all, dim=0),
        'sample_ids': ids_all,
    }


def _normalize_cached(cached):
    context_mean = cached['train']['context'].mean(dim=0, keepdim=True)
    context_std = cached['train']['context'].std(dim=0, keepdim=True, unbiased=False)
    context_std = context_std.clamp_min(1e-4)
    conflict_mean = cached['train']['conflict_score'].mean()
    conflict_std = cached['train']['conflict_score'].std(unbiased=False).clamp_min(1e-4)
    for split in cached.values():
        split['context'] = (split['context'] - context_mean) / context_std
        split['conflict_z'] = (split['conflict_score'] - conflict_mean) / conflict_std
    return {
        'context_mean': context_mean,
        'context_std': context_std,
        'conflict_mean': conflict_mean,
        'conflict_std': conflict_std,
    }


def _specialist_priors(labels, conflict_z):
    absolute = torch.abs(labels)
    positive = torch.sigmoid((labels - 0.15) / 0.35)
    negative = torch.sigmoid((-labels - 0.15) / 0.35)
    boundary = torch.exp(-absolute / 0.45)
    strong = torch.sigmoid((absolute - 1.0) / 0.35)
    conflict = torch.sigmoid(conflict_z)
    priors = torch.cat([positive, negative, boundary, strong, conflict], dim=1)
    return 0.10 + 0.90 * priors


def _sign_loss(prediction, labels):
    mask = (torch.abs(labels) > 0.10).float()
    signed_margin = torch.sign(labels) * prediction
    return (F.softplus(-4.0 * signed_margin) * mask).sum() / mask.sum().clamp_min(1.0)


def _specialist_loss(
    residuals,
    fusion_prediction,
    labels,
    priors,
    sign_loss_weight=0.10,
    off_region_anchor_weight=0.03,
):
    candidate_predictions = fusion_prediction + residuals
    task_loss = F.smooth_l1_loss(
        candidate_predictions,
        labels.expand_as(candidate_predictions),
        reduction='none',
    )
    weighted_task = (task_loss * priors).sum() / priors.sum().clamp_min(1.0)

    sign_losses = []
    for index in range(len(SPECIALIST_NAMES)):
        sign_losses.append(_sign_loss(candidate_predictions[:, index:index + 1], labels))
    sign_loss = torch.stack(sign_losses).mean()

    off_region = 1.0 - (priors - 0.10) / 0.90
    off_region_anchor = (off_region * residuals.pow(2)).mean()
    total = (
        weighted_task
        + sign_loss_weight * sign_loss
        + off_region_anchor_weight * off_region_anchor
    )
    return total, {
        'task_loss': weighted_task.detach(),
        'sign_loss': sign_loss.detach(),
        'off_region_anchor': off_region_anchor.detach(),
    }


def _make_loader(*tensors, batch_size, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def _train_specialists(
    context,
    expert_matrix,
    labels,
    conflict_z,
    device,
    seed,
    epochs,
    learning_rate,
    hidden_dim,
    head_hidden_dim,
    dropout,
    max_residual,
    batch_size,
):
    model = SpecialistResidualExperts(
        context_dim=context.size(1),
        hidden_dim=hidden_dim,
        head_hidden_dim=head_hidden_dim,
        dropout=dropout,
        max_residual=max_residual,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loader = _make_loader(
        context,
        expert_matrix,
        labels,
        conflict_z,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {'loss': 0.0, 'task_loss': 0.0, 'sign_loss': 0.0, 'off_region_anchor': 0.0}
        for batch_context, batch_experts, batch_labels, batch_conflict in loader:
            batch_context = batch_context.to(device)
            batch_experts = batch_experts.to(device)
            batch_labels = batch_labels.to(device)
            batch_conflict = batch_conflict.to(device)
            fusion = batch_experts[:, FUSION_INDEX:FUSION_INDEX + 1]
            priors = _specialist_priors(batch_labels, batch_conflict)
            optimizer.zero_grad()
            residuals = model(batch_context)
            loss, parts = _specialist_loss(residuals, fusion, batch_labels, priors)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals['loss'] += loss.item()
            for key, value in parts.items():
                totals[key] += value.item()
        record = {'epoch': epoch}
        for key, value in totals.items():
            record[key] = value / max(1, len(loader))
        history.append(record)
    return model, history


def _predict_specialists(model, context, device, batch_size=512):
    model.eval()
    outputs = []
    loader = DataLoader(TensorDataset(context), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (batch_context,) in loader:
            outputs.append(model(batch_context.to(device)).cpu())
    return torch.cat(outputs, dim=0)


def _build_folds(sample_count, fold_count, seed):
    if fold_count < 2:
        raise ValueError('oof_folds must be at least 2.')
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(sample_count, generator=generator)
    return [chunk for chunk in torch.tensor_split(permutation, fold_count) if chunk.numel()]


def _cross_fit_specialists(
    train_cache,
    device,
    seed,
    fold_count,
    **train_kwargs,
):
    folds = _build_folds(len(train_cache['labels']), fold_count, seed)
    oof_residuals = torch.zeros(
        len(train_cache['labels']),
        len(SPECIALIST_NAMES),
        dtype=train_cache['context'].dtype,
    )
    all_history = []
    all_indices = torch.arange(len(train_cache['labels']))
    for fold_index, holdout_indices in enumerate(folds):
        mask = torch.ones(len(all_indices), dtype=torch.bool)
        mask[holdout_indices] = False
        train_indices = all_indices[mask]
        logger.info(
            'Training specialist fold %d/%d on %d samples; predicting %d held-out samples.',
            fold_index + 1,
            len(folds),
            len(train_indices),
            len(holdout_indices),
        )
        fold_model, fold_history = _train_specialists(
            context=train_cache['context'][train_indices],
            expert_matrix=train_cache['expert_matrix'][train_indices],
            labels=train_cache['labels'][train_indices],
            conflict_z=train_cache['conflict_z'][train_indices],
            device=device,
            seed=seed + 1009 * (fold_index + 1),
            **train_kwargs,
        )
        oof_residuals[holdout_indices] = _predict_specialists(
            fold_model,
            train_cache['context'][holdout_indices],
            device,
        )
        for row in fold_history:
            all_history.append({'fold': fold_index + 1, **row})
        del fold_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return oof_residuals, all_history


def _gate_forward_prediction(gate, context, residuals, fusion, alpha=1.0):
    output = gate(context, residuals)
    prediction = fusion + float(alpha) * output['correction']
    return prediction, output


def _gate_loss(
    output,
    residuals,
    fusion,
    labels,
    harm_weight=2.0,
    oracle_kl_weight=0.25,
    anchor_weight=0.01,
    sign_weight=0.10,
    oracle_temperature=0.10,
):
    prediction = fusion + output['correction']
    prediction_error = torch.abs(prediction - labels)
    fusion_error = torch.abs(fusion - labels)
    l1_loss = prediction_error.mean()
    harm_loss = F.relu(prediction_error - fusion_error).mean()
    anchor_loss = (1.0 - output['weights'][:, :1]).mean()
    sign = _sign_loss(prediction, labels)

    candidate_predictions = torch.cat([fusion, fusion + residuals], dim=1)
    candidate_errors = torch.abs(candidate_predictions - labels)
    target_weights = F.softmax(-candidate_errors / oracle_temperature, dim=1)
    oracle_kl = F.kl_div(
        F.log_softmax(output['logits'], dim=1),
        target_weights,
        reduction='batchmean',
    )
    total = (
        l1_loss
        + harm_weight * harm_loss
        + oracle_kl_weight * oracle_kl
        + anchor_weight * anchor_loss
        + sign_weight * sign
    )
    return total, {
        'l1_loss': l1_loss.detach(),
        'harm_loss': harm_loss.detach(),
        'oracle_kl': oracle_kl.detach(),
        'anchor_loss': anchor_loss.detach(),
        'sign_loss': sign.detach(),
    }


def _routing_stats(fusion, prediction, labels, correction):
    fusion_error = torch.abs(fusion - labels)
    prediction_error = torch.abs(prediction - labels)
    realized_gain = fusion_error - prediction_error
    corrected = torch.abs(correction) > 0.01
    return {
        'mean_realized_gain': float(realized_gain.mean().item()),
        'positive_gain_rate': float((realized_gain > 0).float().mean().item()),
        'harm_over_005_rate': float((realized_gain < -0.05).float().mean().item()),
        'harm_over_010_rate': float((realized_gain < -0.10).float().mean().item()),
        'correction_rate': float(corrected.float().mean().item()),
        'mean_abs_correction': float(torch.abs(correction).mean().item()),
    }


def _gate_validation_objective(stats, mae):
    return (
        mae
        + 0.20 * stats['harm_over_005_rate']
        + 0.01 * stats['correction_rate']
    )


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
    gate = ResidualMixtureGate(context.size(1), hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = optim.AdamW(gate.parameters(), lr=learning_rate, weight_decay=1e-4)

    all_indices = torch.arange(len(labels))
    if validation_indices is None:
        train_indices = all_indices
    else:
        mask = torch.ones(len(labels), dtype=torch.bool)
        mask[validation_indices] = False
        train_indices = all_indices[mask]

    loader = _make_loader(
        context[train_indices],
        residuals[train_indices],
        expert_matrix[train_indices],
        labels[train_indices],
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    best_state = None
    best_epoch = 0
    best_objective = float('inf')
    history = []

    for epoch in range(1, epochs + 1):
        gate.train()
        totals = {
            'loss': 0.0,
            'l1_loss': 0.0,
            'harm_loss': 0.0,
            'oracle_kl': 0.0,
            'anchor_loss': 0.0,
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
            loss, parts = _gate_loss(
                output,
                batch_residuals,
                fusion,
                batch_labels,
                **loss_kwargs,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(gate.parameters(), 5.0)
            optimizer.step()
            totals['loss'] += loss.item()
            for key, value in parts.items():
                totals[key] += value.item()

        record = {'epoch': epoch}
        for key, value in totals.items():
            record[key] = value / max(1, len(loader))

        if validation_indices is not None:
            gate.eval()
            with torch.no_grad():
                valid_context = context[validation_indices].to(device)
                valid_residuals = residuals[validation_indices].to(device)
                valid_fusion = expert_matrix[validation_indices, FUSION_INDEX:FUSION_INDEX + 1].to(device)
                valid_labels = labels[validation_indices].to(device)
                valid_prediction, valid_output = _gate_forward_prediction(
                    gate,
                    valid_context,
                    valid_residuals,
                    valid_fusion,
                )
                valid_stats = _routing_stats(
                    valid_fusion,
                    valid_prediction,
                    valid_labels,
                    valid_output['correction'],
                )
                valid_mae = float(torch.abs(valid_prediction - valid_labels).mean().item())
                objective = _gate_validation_objective(valid_stats, valid_mae)
            record['valid_mae'] = valid_mae
            record['valid_objective'] = objective
            record.update({f'valid_{key}': value for key, value in valid_stats.items()})
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


def _train_gate_with_oof(
    train_cache,
    oof_residuals,
    device,
    seed,
    **train_kwargs,
):
    sample_count = len(train_cache['labels'])
    generator = torch.Generator().manual_seed(int(seed) + 2718)
    permutation = torch.randperm(sample_count, generator=generator)
    valid_count = max(1, int(round(sample_count * 0.20)))
    validation_indices = permutation[:valid_count]

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


def _predict_gate(gate, cache, residuals, device, alpha=1.0):
    gate.eval()
    context = cache['context'].to(device)
    residuals_device = residuals.to(device)
    fusion = cache['expert_matrix'][:, FUSION_INDEX:FUSION_INDEX + 1].to(device)
    with torch.no_grad():
        prediction, output = _gate_forward_prediction(
            gate,
            context,
            residuals_device,
            fusion,
            alpha=alpha,
        )
    return {
        'prediction': prediction.cpu(),
        'weights': output['weights'].cpu(),
        'correction': (float(alpha) * output['correction']).cpu(),
    }


def _calibrate_alpha(gate, valid_cache, valid_residuals, metrics_fn, device, alphas):
    fusion = valid_cache['expert_matrix'][:, FUSION_INDEX:FUSION_INDEX + 1]
    labels = valid_cache['labels']
    rows = []
    for alpha in alphas:
        result = _predict_gate(gate, valid_cache, valid_residuals, device, alpha=alpha)
        metrics = _safe_metrics(metrics_fn, result['prediction'], labels)
        stats = _routing_stats(fusion, result['prediction'], labels, result['correction'])
        objective = _gate_validation_objective(stats, metrics['MAE'])
        rows.append({
            'alpha': float(alpha),
            'objective': float(objective),
            **metrics,
            **stats,
        })
    best = min(rows, key=lambda row: (row['objective'], row['MAE'], row['harm_over_005_rate']))
    return best, rows


def _candidate_diagnostics(cache, residuals, metrics_fn):
    labels = cache['labels']
    fusion = cache['expert_matrix'][:, FUSION_INDEX:FUSION_INDEX + 1]
    candidates = fusion + residuals
    all_predictions = torch.cat([fusion, candidates], dim=1)
    all_errors = torch.abs(all_predictions - labels)
    winner = all_errors.argmin(dim=1)
    rows = []
    names = ('fusion',) + SPECIALIST_NAMES
    for index, name in enumerate(names):
        prediction = all_predictions[:, index:index + 1]
        realized_gain = torch.abs(fusion - labels) - torch.abs(prediction - labels)
        row = {
            'candidate': name,
            **_safe_metrics(metrics_fn, prediction, labels),
            'oracle_win_rate': float((winner == index).float().mean().item()),
            'mean_gain_vs_fusion': float(realized_gain.mean().item()),
            'gain_over_005_rate': float((realized_gain > 0.05).float().mean().item()),
            'harm_over_005_rate': float((realized_gain < -0.05).float().mean().item()),
        }
        rows.append(row)
    oracle_prediction = all_predictions.gather(1, winner.unsqueeze(1))
    return rows, oracle_prediction, all_errors


def _save_report(
    save_dir,
    seed,
    fold_count,
    alpha,
    metrics_fn,
    test_cache,
    test_residuals,
    gate_result,
    oof_cache,
    oof_residuals,
    specialist_rows,
    specialist_oracle,
    specialist_errors,
    gate_best_epoch,
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

    average_weights = {
        name: float(weights[:, index].mean().item())
        for index, name in enumerate(('fusion',) + SPECIALIST_NAMES)
    }
    top_indices = weights.argmax(dim=1)
    top_rates = {
        name: float((top_indices == index).float().mean().item())
        for index, name in enumerate(('fusion',) + SPECIALIST_NAMES)
    }

    summary = {
        'seed': int(seed),
        'oof_folds': int(fold_count),
        'gate_best_epoch': int(gate_best_epoch),
        'calibration_alpha': float(alpha),
        'fusion_metrics': fusion_metrics,
        'residual_router_metrics': routed_metrics,
        'specialist_oracle_metrics': oracle_metrics,
        **stats,
        'oracle_gap_recovery_ratio': float(recovery_ratio),
        'average_gate_weights': average_weights,
        'top_gate_rates': top_rates,
    }
    with open(save_dir / 'specialist_summary.json', 'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)

    pd.DataFrame(specialist_rows).to_csv(
        save_dir / 'test_specialist_diagnostics.csv', index=False
    )
    error_names = ('fusion',) + SPECIALIST_NAMES
    with np.errstate(invalid='ignore', divide='ignore'):
        correlation = np.corrcoef(specialist_errors.numpy(), rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0)
    pd.DataFrame(correlation, index=error_names, columns=error_names).to_csv(
        save_dir / 'test_specialist_error_correlation.csv'
    )

    oof_rows, _, oof_errors = _candidate_diagnostics(
        oof_cache, oof_residuals, metrics_fn
    )
    pd.DataFrame(oof_rows).to_csv(save_dir / 'oof_specialist_diagnostics.csv', index=False)
    with np.errstate(invalid='ignore', divide='ignore'):
        oof_corr = np.corrcoef(oof_errors.numpy(), rowvar=False)
    oof_corr = np.nan_to_num(oof_corr, nan=0.0)
    pd.DataFrame(oof_corr, index=error_names, columns=error_names).to_csv(
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
        'fusion_abs_error': torch.abs(fusion - labels).view(-1).numpy(),
        'router_abs_error': torch.abs(prediction - labels).view(-1).numpy(),
    }
    for index, name in enumerate(SPECIALIST_NAMES):
        candidate = fusion + test_residuals[:, index:index + 1]
        prediction_data[f'{name}_residual'] = test_residuals[:, index].numpy()
        prediction_data[f'{name}_prediction'] = candidate.view(-1).numpy()
        prediction_data[f'{name}_weight'] = weights[:, index + 1].numpy()
    prediction_data['fusion_weight'] = weights[:, 0].numpy()
    pd.DataFrame(prediction_data).to_csv(
        save_dir / 'specialist_predictions.csv', index=False
    )
    logger.info(
        'SPECIALIST TEST fusion_MAE=%.4f routed_MAE=%.4f oracle_MAE=%.4f alpha=%.2f correction_rate=%.4f harm_010=%.4f recovery=%.4f',
        fusion_metrics['MAE'],
        routed_metrics['MAE'],
        oracle_metrics['MAE'],
        alpha,
        stats['correction_rate'],
        stats['harm_over_010_rate'],
        recovery_ratio,
    )
    return summary


def train_specialist_residual_system(
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
    specialist_hidden_dim=96,
    head_hidden_dim=32,
    gate_hidden_dim=64,
    dropout=0.15,
    max_residual=0.75,
    batch_size=128,
    harm_weight=2.0,
    oracle_kl_weight=0.25,
    anchor_weight=0.01,
    sign_weight=0.10,
    calibration_alphas=None,
):
    """Train cross-fitted specialist residuals and a conservative soft gate."""
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

    specialist_kwargs = {
        'epochs': specialist_epochs,
        'learning_rate': specialist_learning_rate,
        'hidden_dim': specialist_hidden_dim,
        'head_hidden_dim': head_hidden_dim,
        'dropout': dropout,
        'max_residual': max_residual,
        'batch_size': batch_size,
    }
    oof_residuals, fold_history = _cross_fit_specialists(
        cached['train'],
        device=device,
        seed=seed,
        fold_count=oof_folds,
        **specialist_kwargs,
    )
    pd.DataFrame(fold_history).to_csv(
        save_dir / 'specialist_fold_history.csv', index=False
    )

    final_specialists, final_specialist_history = _train_specialists(
        context=cached['train']['context'],
        expert_matrix=cached['train']['expert_matrix'],
        labels=cached['train']['labels'],
        conflict_z=cached['train']['conflict_z'],
        device=device,
        seed=seed + 9001,
        **specialist_kwargs,
    )
    pd.DataFrame(final_specialist_history).to_csv(
        save_dir / 'specialist_final_history.csv', index=False
    )
    torch.save(
        {
            'state_dict': final_specialists.state_dict(),
            'context_dim': int(cached['train']['context'].size(1)),
            'specialist_names': list(SPECIALIST_NAMES),
            'max_residual': float(max_residual),
            'normalization': normalization,
        },
        save_dir / 'specialists_final.pth',
    )
    valid_residuals = _predict_specialists(
        final_specialists, cached['valid']['context'], device
    )
    test_residuals = _predict_specialists(
        final_specialists, cached['test']['context'], device
    )

    gate_kwargs = {
        'epochs': gate_epochs,
        'learning_rate': gate_learning_rate,
        'hidden_dim': gate_hidden_dim,
        'dropout': dropout,
        'batch_size': batch_size,
        'patience': gate_patience,
        'harm_weight': harm_weight,
        'oracle_kl_weight': oracle_kl_weight,
        'anchor_weight': anchor_weight,
        'sign_weight': sign_weight,
    }
    gate, gate_selection_history, gate_final_history, gate_best_epoch = _train_gate_with_oof(
        cached['train'],
        oof_residuals,
        device=device,
        seed=seed + 12001,
        **gate_kwargs,
    )
    pd.DataFrame(gate_selection_history).to_csv(
        save_dir / 'gate_selection_history.csv', index=False
    )
    pd.DataFrame(gate_final_history).to_csv(
        save_dir / 'gate_final_history.csv', index=False
    )

    best_alpha, alpha_rows = _calibrate_alpha(
        gate,
        cached['valid'],
        valid_residuals,
        metrics_fn,
        device,
        calibration_alphas,
    )
    pd.DataFrame(alpha_rows).to_csv(save_dir / 'alpha_calibration.csv', index=False)
    torch.save(
        {
            'state_dict': gate.state_dict(),
            'context_dim': int(cached['train']['context'].size(1)),
            'specialist_names': list(SPECIALIST_NAMES),
            'best_epoch': int(gate_best_epoch),
            'calibration_alpha': float(best_alpha['alpha']),
        },
        save_dir / 'residual_gate.pth',
    )

    gate_result = _predict_gate(
        gate,
        cached['test'],
        test_residuals,
        device,
        alpha=best_alpha['alpha'],
    )
    specialist_rows, specialist_oracle, specialist_errors = _candidate_diagnostics(
        cached['test'], test_residuals, metrics_fn
    )
    summary = _save_report(
        save_dir=save_dir,
        seed=seed,
        fold_count=oof_folds,
        alpha=best_alpha['alpha'],
        metrics_fn=metrics_fn,
        test_cache=cached['test'],
        test_residuals=test_residuals,
        gate_result=gate_result,
        oof_cache=cached['train'],
        oof_residuals=oof_residuals,
        specialist_rows=specialist_rows,
        specialist_oracle=specialist_oracle,
        specialist_errors=specialist_errors,
        gate_best_epoch=gate_best_epoch,
    )
    return final_specialists, gate, summary
