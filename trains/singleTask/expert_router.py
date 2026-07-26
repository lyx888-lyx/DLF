import copy
import json
import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from .expert_analysis import extract_expert_logits, normalize_batch_ids

logger = logging.getLogger('MMSA')

EXPERT_NAMES = ('text', 'audio', 'video', 'common', 'fusion')
FUSION_INDEX = EXPERT_NAMES.index('fusion')
ALT_EXPERT_NAMES = EXPERT_NAMES[:-1]


class ConservativeExpertRouter(nn.Module):
    """Predict whether an alternative DLF branch can safely beat fusion."""

    def __init__(self, hidden_dim=32, dropout=0.1):
        super().__init__()
        expert_count = len(EXPERT_NAMES)
        feature_dim = expert_count + (expert_count - 1) + 5
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gain_head = nn.Linear(hidden_dim, len(ALT_EXPERT_NAMES))
        self.benefit_head = nn.Linear(hidden_dim, len(ALT_EXPERT_NAMES))

    @staticmethod
    def build_features(expert_matrix):
        fusion = expert_matrix[:, FUSION_INDEX:FUSION_INDEX + 1]
        differences = expert_matrix[:, :FUSION_INDEX] - fusion
        mean = expert_matrix.mean(dim=1, keepdim=True)
        std = expert_matrix.std(dim=1, keepdim=True, unbiased=False)
        minimum = expert_matrix.min(dim=1, keepdim=True).values
        maximum = expert_matrix.max(dim=1, keepdim=True).values
        span = maximum - minimum
        return torch.cat(
            [expert_matrix, differences, mean, std, minimum, maximum, span],
            dim=1,
        )

    def forward(self, expert_matrix):
        hidden = self.encoder(self.build_features(expert_matrix))
        gain = self.gain_head(hidden)
        benefit_logit = self.benefit_head(hidden)
        score = gain * torch.sigmoid(benefit_logit)
        return {
            'gain': gain,
            'benefit_logit': benefit_logit,
            'score': score,
        }


def _stack_experts(model_output):
    experts = extract_expert_logits(model_output)
    return torch.cat(
        [experts[name].view(-1, 1) for name in EXPERT_NAMES],
        dim=1,
    )


def _true_gains(expert_matrix, labels):
    errors = torch.abs(expert_matrix - labels)
    fusion_error = errors[:, FUSION_INDEX:FUSION_INDEX + 1]
    return fusion_error - errors[:, :FUSION_INDEX]


def _safe_metrics(metrics_fn, prediction, target):
    return {
        key: float(value)
        for key, value in metrics_fn(prediction, target).items()
    }


def _collect_predictions(model, router, dataloader, device):
    model.eval()
    router.eval()
    matrices = []
    labels_all = []
    ids_all = []
    gains_all = []
    benefit_all = []
    scores_all = []

    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            vision = batch_data['vision'].to(device)
            audio = batch_data['audio'].to(device)
            text = batch_data['text'].to(device)
            labels = batch_data['labels']['M'].to(device).view(-1, 1)
            model_output = model(text, audio, vision)
            expert_matrix = _stack_experts(model_output)
            router_output = router(expert_matrix)

            matrices.append(expert_matrix.cpu())
            labels_all.append(labels.cpu())
            gains_all.append(router_output['gain'].cpu())
            benefit_all.append(torch.sigmoid(router_output['benefit_logit']).cpu())
            scores_all.append(router_output['score'].cpu())
            ids_all.extend(normalize_batch_ids(batch_data.get('id')))

    return {
        'expert_matrix': torch.cat(matrices, dim=0),
        'labels': torch.cat(labels_all, dim=0),
        'predicted_gain': torch.cat(gains_all, dim=0),
        'benefit_probability': torch.cat(benefit_all, dim=0),
        'score': torch.cat(scores_all, dim=0),
        'sample_ids': ids_all,
    }


def _route_from_collected(collected, threshold):
    expert_matrix = collected['expert_matrix']
    score = collected['score']
    best_score, best_alt_index = score.max(dim=1)
    fusion_indices = torch.full_like(best_alt_index, FUSION_INDEX)
    selected_indices = torch.where(
        best_score > threshold,
        best_alt_index,
        fusion_indices,
    )
    routed_prediction = expert_matrix.gather(
        1, selected_indices.unsqueeze(1)
    )
    return routed_prediction, selected_indices, best_score


def choose_validation_threshold(collected, metrics_fn, thresholds):
    best = None
    for threshold in thresholds:
        routed_prediction, selected_indices, _ = _route_from_collected(
            collected, threshold
        )
        metrics = _safe_metrics(
            metrics_fn,
            routed_prediction,
            collected['labels'],
        )
        switch_rate = float(
            (selected_indices != FUSION_INDEX).float().mean().item()
        )
        candidate = {
            'threshold': float(threshold),
            'metrics': metrics,
            'switch_rate': switch_rate,
        }
        if best is None:
            best = candidate
            continue
        if metrics['MAE'] < best['metrics']['MAE'] - 1e-8:
            best = candidate
        elif abs(metrics['MAE'] - best['metrics']['MAE']) <= 1e-8:
            if metrics.get('Corr', -1.0) > best['metrics'].get('Corr', -1.0):
                best = candidate
    return best


def _pairwise_ranking_loss(predicted_gain, true_gain, margin=0.02):
    true_difference = true_gain.unsqueeze(2) - true_gain.unsqueeze(1)
    predicted_difference = (
        predicted_gain.unsqueeze(2) - predicted_gain.unsqueeze(1)
    )
    mask = torch.abs(true_difference) > margin
    if not mask.any():
        return predicted_gain.new_tensor(0.0)
    direction = torch.sign(true_difference[mask])
    return F.relu(margin - direction * predicted_difference[mask]).mean()


def train_router(
    model,
    dataloader,
    metrics_fn,
    device,
    save_dir,
    seed,
    epochs=40,
    patience=7,
    learning_rate=1e-3,
    hidden_dim=32,
    dropout=0.1,
    minimum_gain=0.03,
    gain_loss_weight=1.0,
    benefit_loss_weight=0.5,
    ranking_loss_weight=0.2,
    soft_route_loss_weight=0.2,
    temperature=0.1,
    thresholds=None,
):
    """Train a frozen-backbone, fusion-anchored conservative router."""
    if thresholds is None:
        thresholds = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / 'router_best.pth'

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    router = ConservativeExpertRouter(
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    optimizer = optim.Adam(
        router.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    best_state = None
    best_threshold = None
    best_valid_mae = float('inf')
    best_epoch = 0
    history = []

    for epoch in range(1, epochs + 1):
        router.train()
        total_loss = 0.0
        total_gain_loss = 0.0
        total_benefit_loss = 0.0
        total_rank_loss = 0.0
        total_soft_route_loss = 0.0

        for batch_data in tqdm(dataloader['train']):
            vision = batch_data['vision'].to(device)
            audio = batch_data['audio'].to(device)
            text = batch_data['text'].to(device)
            labels = batch_data['labels']['M'].to(device).view(-1, 1)

            with torch.no_grad():
                model_output = model(text, audio, vision)
                expert_matrix = _stack_experts(model_output).detach()
                true_gain = _true_gains(expert_matrix, labels)

            optimizer.zero_grad()
            output = router(expert_matrix)
            predicted_gain = output['gain']
            benefit_logit = output['benefit_logit']

            gain_loss = F.smooth_l1_loss(predicted_gain, true_gain)
            benefit_target = (true_gain > minimum_gain).float()
            benefit_loss = F.binary_cross_entropy_with_logits(
                benefit_logit,
                benefit_target,
            )
            rank_loss = _pairwise_ranking_loss(
                predicted_gain,
                true_gain,
            )

            zero_score = torch.zeros(
                expert_matrix.size(0), 1, device=device
            )
            route_scores = torch.cat([output['score'], zero_score], dim=1)
            route_weights = F.softmax(route_scores / temperature, dim=1)
            soft_prediction = torch.sum(
                route_weights * expert_matrix,
                dim=1,
                keepdim=True,
            )
            soft_route_loss = F.l1_loss(soft_prediction, labels)

            loss = (
                gain_loss_weight * gain_loss
                + benefit_loss_weight * benefit_loss
                + ranking_loss_weight * rank_loss
                + soft_route_loss_weight * soft_route_loss
            )
            loss.backward()
            nn.utils.clip_grad_norm_(router.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            total_gain_loss += gain_loss.item()
            total_benefit_loss += benefit_loss.item()
            total_rank_loss += rank_loss.item()
            total_soft_route_loss += soft_route_loss.item()

        valid_collected = _collect_predictions(
            model,
            router,
            dataloader['valid'],
            device,
        )
        valid_choice = choose_validation_threshold(
            valid_collected,
            metrics_fn,
            thresholds,
        )
        valid_mae = valid_choice['metrics']['MAE']
        batch_count = max(1, len(dataloader['train']))
        epoch_record = {
            'epoch': epoch,
            'loss': total_loss / batch_count,
            'gain_loss': total_gain_loss / batch_count,
            'benefit_loss': total_benefit_loss / batch_count,
            'ranking_loss': total_rank_loss / batch_count,
            'soft_route_loss': total_soft_route_loss / batch_count,
            'valid_threshold': valid_choice['threshold'],
            'valid_switch_rate': valid_choice['switch_rate'],
            **{
                f"valid_{key}": value
                for key, value in valid_choice['metrics'].items()
            },
        }
        history.append(epoch_record)
        logger.info(
            'ROUTER epoch=%d loss=%.4f valid_MAE=%.4f threshold=%.3f switch_rate=%.4f',
            epoch,
            epoch_record['loss'],
            valid_mae,
            valid_choice['threshold'],
            valid_choice['switch_rate'],
        )

        if valid_mae < best_valid_mae - 1e-6:
            best_valid_mae = valid_mae
            best_epoch = epoch
            best_threshold = valid_choice['threshold']
            best_state = copy.deepcopy(router.state_dict())
            torch.save(
                {
                    'router_state_dict': best_state,
                    'threshold': best_threshold,
                    'best_epoch': best_epoch,
                    'seed': int(seed),
                    'expert_names': list(EXPERT_NAMES),
                },
                checkpoint_path,
            )

        if epoch - best_epoch >= patience:
            break

    if best_state is None:
        raise RuntimeError('Router training did not produce a valid checkpoint.')

    router.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(save_dir / 'router_history.csv', index=False)

    test_collected = _collect_predictions(
        model,
        router,
        dataloader['test'],
        device,
    )
    summary = save_router_report(
        collected=test_collected,
        metrics_fn=metrics_fn,
        threshold=best_threshold,
        save_dir=save_dir,
        best_epoch=best_epoch,
        seed=seed,
    )
    return router, summary


def save_router_report(
    collected,
    metrics_fn,
    threshold,
    save_dir,
    best_epoch,
    seed,
):
    save_dir = Path(save_dir)
    expert_matrix = collected['expert_matrix']
    labels = collected['labels']
    routed_prediction, selected_indices, best_score = _route_from_collected(
        collected,
        threshold,
    )
    fusion_prediction = expert_matrix[:, FUSION_INDEX:FUSION_INDEX + 1]
    absolute_errors = torch.abs(expert_matrix - labels)
    oracle_indices = absolute_errors.argmin(dim=1)
    oracle_prediction = expert_matrix.gather(1, oracle_indices.unsqueeze(1))

    routed_error = torch.abs(routed_prediction - labels)
    fusion_error = torch.abs(fusion_prediction - labels)
    oracle_error = torch.abs(oracle_prediction - labels)
    realized_gain = fusion_error - routed_error
    switched = selected_indices != FUSION_INDEX

    selection_counts = torch.bincount(
        selected_indices,
        minlength=len(EXPERT_NAMES),
    ).tolist()
    selection_rates = {
        name: count / len(selected_indices)
        for name, count in zip(EXPERT_NAMES, selection_counts)
    }
    if switched.any():
        switch_precision = float(
            (realized_gain[switched] > 0).float().mean().item()
        )
    else:
        switch_precision = 0.0

    summary = {
        'seed': int(seed),
        'best_epoch': int(best_epoch),
        'threshold': float(threshold),
        'fusion_metrics': _safe_metrics(metrics_fn, fusion_prediction, labels),
        'router_metrics': _safe_metrics(metrics_fn, routed_prediction, labels),
        'oracle_metrics': _safe_metrics(metrics_fn, oracle_prediction, labels),
        'switch_rate': float(switched.float().mean().item()),
        'switch_precision': switch_precision,
        'mean_realized_gain': float(realized_gain.mean().item()),
        'positive_gain_rate': float((realized_gain > 0).float().mean().item()),
        'harm_over_005_rate': float((realized_gain < -0.05).float().mean().item()),
        'harm_over_010_rate': float((realized_gain < -0.10).float().mean().item()),
        'mean_routing_regret': float((routed_error - oracle_error).mean().item()),
        'selection_rates': selection_rates,
    }
    with open(save_dir / 'router_summary.json', 'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)

    sample_ids = collected['sample_ids']
    if len(sample_ids) != len(labels):
        sample_ids = [str(index) for index in range(len(labels))]

    prediction_data = {
        'sample_id': sample_ids,
        'target': labels.view(-1).numpy(),
        'selected_expert': [EXPERT_NAMES[index] for index in selected_indices.tolist()],
        'best_score': best_score.numpy(),
        'fusion_prediction': fusion_prediction.view(-1).numpy(),
        'router_prediction': routed_prediction.view(-1).numpy(),
        'oracle_prediction': oracle_prediction.view(-1).numpy(),
        'fusion_abs_error': fusion_error.view(-1).numpy(),
        'router_abs_error': routed_error.view(-1).numpy(),
        'realized_gain': realized_gain.view(-1).numpy(),
    }
    for index, name in enumerate(EXPERT_NAMES):
        prediction_data[f'{name}_prediction'] = expert_matrix[:, index].numpy()
        prediction_data[f'{name}_abs_error'] = absolute_errors[:, index].numpy()
    for index, name in enumerate(ALT_EXPERT_NAMES):
        prediction_data[f'{name}_predicted_gain'] = (
            collected['predicted_gain'][:, index].numpy()
        )
        prediction_data[f'{name}_benefit_probability'] = (
            collected['benefit_probability'][:, index].numpy()
        )
        prediction_data[f'{name}_router_score'] = (
            collected['score'][:, index].numpy()
        )

    pd.DataFrame(prediction_data).to_csv(
        save_dir / 'router_predictions.csv',
        index=False,
    )

    logger.info('Router results saved to %s', save_dir)
    logger.info(
        'ROUTER TEST fusion_MAE=%.4f router_MAE=%.4f oracle_MAE=%.4f switch_rate=%.4f switch_precision=%.4f',
        summary['fusion_metrics']['MAE'],
        summary['router_metrics']['MAE'],
        summary['oracle_metrics']['MAE'],
        summary['switch_rate'],
        summary['switch_precision'],
    )
    return summary
