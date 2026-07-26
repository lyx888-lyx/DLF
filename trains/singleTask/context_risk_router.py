import copy
import itertools
import json
import logging
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from .expert_analysis import extract_expert_logits, normalize_batch_ids
logger = logging.getLogger('MMSA')
EXPERT_NAMES = ('text', 'audio', 'video', 'common', 'fusion')
FUSION_INDEX = EXPERT_NAMES.index('fusion')
ALT_EXPERT_NAMES = EXPERT_NAMES[:-1]

class ContextRiskExpertRouter(nn.Module):
    """Context-aware router with explicit benefit and harm prediction."""

    def __init__(self, context_dim, hidden_dim=64, context_hidden_dim=64, dropout=0.15):
        super().__init__()
        expert_count = len(EXPERT_NAMES)
        prediction_feature_dim = expert_count + (expert_count - 1) + 5
        self.prediction_encoder = nn.Sequential(nn.Linear(prediction_feature_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.context_encoder = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, context_hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(context_hidden_dim, context_hidden_dim), nn.ReLU(), nn.LayerNorm(context_hidden_dim), nn.Dropout(dropout))
        self.fusion_encoder = nn.Sequential(nn.Linear(hidden_dim + context_hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.gain_head = nn.Linear(hidden_dim, len(ALT_EXPERT_NAMES))
        self.benefit_head = nn.Linear(hidden_dim, len(ALT_EXPERT_NAMES))
        self.harm_head = nn.Linear(hidden_dim, len(ALT_EXPERT_NAMES))

    @staticmethod
    def build_prediction_features(expert_matrix):
        fusion = expert_matrix[:, FUSION_INDEX:FUSION_INDEX + 1]
        differences = expert_matrix[:, :FUSION_INDEX] - fusion
        mean = expert_matrix.mean(dim=1, keepdim=True)
        std = expert_matrix.std(dim=1, keepdim=True, unbiased=False)
        minimum = expert_matrix.min(dim=1, keepdim=True).values
        maximum = expert_matrix.max(dim=1, keepdim=True).values
        span = maximum - minimum
        return torch.cat([expert_matrix, differences, mean, std, minimum, maximum, span], dim=1)

    def forward(self, expert_matrix, context):
        prediction_hidden = self.prediction_encoder(self.build_prediction_features(expert_matrix))
        context_hidden = self.context_encoder(context)
        hidden = self.fusion_encoder(torch.cat([prediction_hidden, context_hidden], dim=1))
        gain = self.gain_head(hidden)
        benefit_logit = self.benefit_head(hidden)
        harm_logit = self.harm_head(hidden)
        benefit_probability = torch.sigmoid(benefit_logit)
        harm_probability = torch.sigmoid(harm_logit)
        risk_adjusted_score = gain * benefit_probability * (1.0 - harm_probability)
        return {'gain': gain, 'benefit_logit': benefit_logit, 'harm_logit': harm_logit, 'benefit_probability': benefit_probability, 'harm_probability': harm_probability, 'score': risk_adjusted_score}

def _stack_experts(model_output):
    experts = extract_expert_logits(model_output)
    return torch.cat([experts[name].view(-1, 1) for name in EXPERT_NAMES], dim=1)

def _sequence_summary(sequence_tensor):
    """Summarize a [sequence, batch, hidden] DLF representation."""
    if sequence_tensor.dim() != 3:
        raise ValueError(f'Expected a three-dimensional sequence tensor, got {sequence_tensor.shape}.')
    last = sequence_tensor[-1]
    mean = sequence_tensor.mean(dim=0)
    return torch.cat([last, mean], dim=1)

def _cosine_distance(left, right):
    return 1.0 - F.cosine_similarity(left, right, dim=1).unsqueeze(1)

def _extract_router_context(model_output):
    """
    Build a compact sample context from representations already returned by DLF.

    No DLF parameter or checkpoint format is changed. The context combines the
    last/mean specific and common states, aligned common features, their norms,
    and pairwise modality-conflict signals.
    """
    s_l = _sequence_summary(model_output['s_l'])
    s_a = _sequence_summary(model_output['s_a'])
    s_v = _sequence_summary(model_output['s_v'])
    c_l = _sequence_summary(model_output['c_l'])
    c_a = _sequence_summary(model_output['c_a'])
    c_v = _sequence_summary(model_output['c_v'])
    c_l_sim = model_output['c_l_sim'].view(model_output['c_l_sim'].size(0), -1)
    c_a_sim = model_output['c_a_sim'].view(model_output['c_a_sim'].size(0), -1)
    c_v_sim = model_output['c_v_sim'].view(model_output['c_v_sim'].size(0), -1)
    raw_context = torch.cat([s_l, s_a, s_v, c_l, c_a, c_v, c_l_sim, c_a_sim, c_v_sim], dim=1)
    norm_features = torch.cat([vector.norm(p=2, dim=1, keepdim=True) for vector in (s_l, s_a, s_v, c_l_sim, c_a_sim, c_v_sim)], dim=1)
    conflict_features = torch.cat([_cosine_distance(s_l, s_a), _cosine_distance(s_l, s_v), _cosine_distance(s_a, s_v), _cosine_distance(c_l_sim, c_a_sim), _cosine_distance(c_l_sim, c_v_sim), _cosine_distance(c_a_sim, c_v_sim)], dim=1)
    return torch.cat([raw_context, norm_features, conflict_features], dim=1)

def _true_gains(expert_matrix, labels):
    errors = torch.abs(expert_matrix - labels)
    fusion_error = errors[:, FUSION_INDEX:FUSION_INDEX + 1]
    return fusion_error - errors[:, :FUSION_INDEX]

def _safe_metrics(metrics_fn, prediction, target):
    return {key: float(value) for key, value in metrics_fn(prediction, target).items()}

def _cache_expert_outputs(model, dataloader, device, split_name):
    """Run frozen DLF once and cache predictions plus contextual representations."""
    model.eval()
    matrices = []
    contexts = []
    labels_all = []
    ids_all = []
    logger.info('Caching frozen DLF expert outputs and context for %s split.', split_name)
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            vision = batch_data['vision'].to(device)
            audio = batch_data['audio'].to(device)
            text = batch_data['text'].to(device)
            labels = batch_data['labels']['M'].to(device).view(-1, 1)
            model_output = model(text, audio, vision)
            matrices.append(_stack_experts(model_output).cpu())
            contexts.append(_extract_router_context(model_output).cpu())
            labels_all.append(labels.cpu())
            ids_all.extend(normalize_batch_ids(batch_data.get('id')))
    return {'expert_matrix': torch.cat(matrices, dim=0), 'context': torch.cat(contexts, dim=0), 'labels': torch.cat(labels_all, dim=0), 'sample_ids': ids_all}

def _score_cached(router, cached, device):
    router.eval()
    with torch.no_grad():
        expert_matrix = cached['expert_matrix'].to(device)
        context = cached['context'].to(device)
        output = router(expert_matrix, context)
    return {'expert_matrix': cached['expert_matrix'], 'context': cached['context'], 'labels': cached['labels'], 'predicted_gain': output['gain'].cpu(), 'benefit_probability': output['benefit_probability'].cpu(), 'harm_probability': output['harm_probability'].cpu(), 'score': output['score'].cpu(), 'sample_ids': cached['sample_ids']}

def _route_from_collected(collected, gain_threshold, benefit_threshold, harm_threshold):
    expert_matrix = collected['expert_matrix']
    predicted_gain = collected['predicted_gain']
    benefit_probability = collected['benefit_probability']
    harm_probability = collected['harm_probability']
    score = collected['score']
    eligible = (predicted_gain > gain_threshold) & (benefit_probability > benefit_threshold) & (harm_probability < harm_threshold)
    masked_score = score.masked_fill(~eligible, float('-inf'))
    best_score, best_alt_index = masked_score.max(dim=1)
    has_eligible = eligible.any(dim=1)
    fusion_indices = torch.full_like(best_alt_index, FUSION_INDEX)
    selected_indices = torch.where(has_eligible, best_alt_index, fusion_indices)
    routed_prediction = expert_matrix.gather(1, selected_indices.unsqueeze(1))
    best_score = torch.where(has_eligible, best_score, torch.zeros_like(best_score))
    return (routed_prediction, selected_indices, best_score)

def _routing_statistics(collected, metrics_fn, gain_threshold, benefit_threshold, harm_threshold):
    routed_prediction, selected_indices, _ = _route_from_collected(collected, gain_threshold, benefit_threshold, harm_threshold)
    labels = collected['labels']
    expert_matrix = collected['expert_matrix']
    fusion_prediction = expert_matrix[:, FUSION_INDEX:FUSION_INDEX + 1]
    routed_error = torch.abs(routed_prediction - labels)
    fusion_error = torch.abs(fusion_prediction - labels)
    realized_gain = fusion_error - routed_error
    switched = selected_indices != FUSION_INDEX
    if switched.any():
        switch_precision = float((realized_gain[switched] > 0).float().mean().item())
        switched_mean_gain = float(realized_gain[switched].mean().item())
    else:
        switch_precision = 1.0
        switched_mean_gain = 0.0
    return {'gain_threshold': float(gain_threshold), 'benefit_threshold': float(benefit_threshold), 'harm_threshold': float(harm_threshold), 'metrics': _safe_metrics(metrics_fn, routed_prediction, labels), 'switch_rate': float(switched.float().mean().item()), 'switch_precision': switch_precision, 'switched_mean_gain': switched_mean_gain, 'harm_over_005_rate': float((realized_gain < -0.05).float().mean().item()), 'harm_over_010_rate': float((realized_gain < -0.1).float().mean().item()), 'mean_realized_gain': float(realized_gain.mean().item())}

def choose_validation_gate(collected, metrics_fn, gain_thresholds, benefit_thresholds, harm_thresholds, minimum_switch_precision=0.6, harm_risk_weight=0.1, switch_rate_weight=0.005, precision_penalty_weight=0.05):
    candidates = []
    for gain_threshold, benefit_threshold, harm_threshold in itertools.product(gain_thresholds, benefit_thresholds, harm_thresholds):
        candidate = _routing_statistics(collected, metrics_fn, gain_threshold, benefit_threshold, harm_threshold)
        precision_shortfall = max(0.0, minimum_switch_precision - candidate['switch_precision'])
        candidate['objective'] = candidate['metrics']['MAE'] + harm_risk_weight * candidate['harm_over_005_rate'] + switch_rate_weight * candidate['switch_rate'] + precision_penalty_weight * precision_shortfall
        candidates.append(candidate)
    no_switch = _routing_statistics(collected, metrics_fn, gain_threshold=1000000.0, benefit_threshold=1.0, harm_threshold=0.0)
    no_switch['objective'] = no_switch['metrics']['MAE']
    candidates.append(no_switch)
    best = min(candidates, key=lambda item: (item['objective'], item['metrics']['MAE'], item['harm_over_005_rate'], item['switch_rate']))
    return (best, candidates)

def _pairwise_ranking_loss(predicted_gain, true_gain, margin=0.02):
    true_difference = true_gain.unsqueeze(2) - true_gain.unsqueeze(1)
    predicted_difference = predicted_gain.unsqueeze(2) - predicted_gain.unsqueeze(1)
    mask = torch.abs(true_difference) > margin
    if not mask.any():
        return predicted_gain.new_tensor(0.0)
    direction = torch.sign(true_difference[mask])
    return F.relu(margin - direction * predicted_difference[mask]).mean()

def _balanced_binary_cross_entropy(logits, target):
    positive_count = target.sum()
    negative_count = target.numel() - positive_count
    if positive_count.item() == 0 or negative_count.item() == 0:
        return F.binary_cross_entropy_with_logits(logits, target)
    pos_weight = (negative_count / positive_count).clamp(1.0, 10.0)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)

def _harm_averse_loss(predicted_gain, benefit_logit, harm_logit, true_gain):
    benefit_probability = torch.sigmoid(benefit_logit)
    harm_probability = torch.sigmoid(harm_logit)
    false_switch_strength = F.relu(predicted_gain) * benefit_probability * (1.0 - harm_probability)
    actual_harm = F.relu(-true_gain)
    harmful_mask = actual_harm > 0
    if not harmful_mask.any():
        return predicted_gain.new_tensor(0.0)
    return (actual_harm[harmful_mask] * false_switch_strength[harmful_mask].pow(2)).mean()

def _candidate_rows(candidates, prefix='valid'):
    rows = []
    for candidate in candidates:
        row = {'gain_threshold': candidate['gain_threshold'], 'benefit_threshold': candidate['benefit_threshold'], 'harm_threshold': candidate['harm_threshold'], 'objective': candidate.get('objective'), 'switch_rate': candidate['switch_rate'], 'switch_precision': candidate['switch_precision'], 'switched_mean_gain': candidate['switched_mean_gain'], 'mean_realized_gain': candidate['mean_realized_gain'], 'harm_over_005_rate': candidate['harm_over_005_rate'], 'harm_over_010_rate': candidate['harm_over_010_rate']}
        row.update({f'{prefix}_{key}': value for key, value in candidate['metrics'].items()})
        rows.append(row)
    return rows

def train_router(model, dataloader, metrics_fn, device, save_dir, seed, epochs=60, patience=10, learning_rate=0.0005, hidden_dim=64, context_hidden_dim=64, dropout=0.15, minimum_gain=0.03, harm_margin=0.03, gain_loss_weight=1.0, benefit_loss_weight=0.6, harm_classification_weight=0.8, ranking_loss_weight=0.2, harm_averse_weight=2.0, minimum_switch_precision=0.6, harm_risk_weight=0.1, switch_rate_weight=0.005, precision_penalty_weight=0.05, gain_thresholds=None, benefit_thresholds=None, harm_thresholds=None, router_batch_size=128):
    """Train a frozen-backbone, context-aware, harm-averse conservative router."""
    if gain_thresholds is None:
        gain_thresholds = [0.02, 0.03, 0.05, 0.08, 0.1, 0.15]
    if benefit_thresholds is None:
        benefit_thresholds = [0.55, 0.65, 0.75, 0.85]
    if harm_thresholds is None:
        harm_thresholds = [0.5, 0.35, 0.2, 0.1]
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / 'router_best.pth'
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    cached = {split: _cache_expert_outputs(model, dataloader[split], device, split) for split in ('train', 'valid', 'test')}
    context_dim = cached['train']['context'].size(1)
    train_dataset = TensorDataset(cached['train']['expert_matrix'], cached['train']['context'], cached['train']['labels'])
    train_loader = DataLoader(train_dataset, batch_size=router_batch_size, shuffle=True)
    router = ContextRiskExpertRouter(context_dim=context_dim, hidden_dim=hidden_dim, context_hidden_dim=context_hidden_dim, dropout=dropout).to(device)
    optimizer = optim.AdamW(router.parameters(), lr=learning_rate, weight_decay=0.0001)
    best_state = None
    best_gate = None
    best_objective = float('inf')
    best_epoch = 0
    history = []
    best_validation_candidates = None
    for epoch in range(1, epochs + 1):
        router.train()
        totals = {'loss': 0.0, 'gain_loss': 0.0, 'benefit_loss': 0.0, 'harm_classification_loss': 0.0, 'ranking_loss': 0.0, 'harm_averse_loss': 0.0}
        for expert_matrix, context, labels in train_loader:
            expert_matrix = expert_matrix.to(device)
            context = context.to(device)
            labels = labels.to(device)
            true_gain = _true_gains(expert_matrix, labels)
            optimizer.zero_grad()
            output = router(expert_matrix, context)
            predicted_gain = output['gain']
            benefit_logit = output['benefit_logit']
            harm_logit = output['harm_logit']
            gain_loss = F.smooth_l1_loss(predicted_gain, true_gain)
            benefit_target = (true_gain > minimum_gain).float()
            harm_target = (true_gain < -harm_margin).float()
            benefit_loss = _balanced_binary_cross_entropy(benefit_logit, benefit_target)
            harm_classification_loss = _balanced_binary_cross_entropy(harm_logit, harm_target)
            ranking_loss = _pairwise_ranking_loss(predicted_gain, true_gain)
            harm_averse_loss = _harm_averse_loss(predicted_gain, benefit_logit, harm_logit, true_gain)
            loss = gain_loss_weight * gain_loss + benefit_loss_weight * benefit_loss + harm_classification_weight * harm_classification_loss + ranking_loss_weight * ranking_loss + harm_averse_weight * harm_averse_loss
            loss.backward()
            nn.utils.clip_grad_norm_(router.parameters(), 5.0)
            optimizer.step()
            totals['loss'] += loss.item()
            totals['gain_loss'] += gain_loss.item()
            totals['benefit_loss'] += benefit_loss.item()
            totals['harm_classification_loss'] += harm_classification_loss.item()
            totals['ranking_loss'] += ranking_loss.item()
            totals['harm_averse_loss'] += harm_averse_loss.item()
        valid_collected = _score_cached(router, cached['valid'], device)
        valid_choice, validation_candidates = choose_validation_gate(valid_collected, metrics_fn, gain_thresholds, benefit_thresholds, harm_thresholds, minimum_switch_precision=minimum_switch_precision, harm_risk_weight=harm_risk_weight, switch_rate_weight=switch_rate_weight, precision_penalty_weight=precision_penalty_weight)
        batch_count = max(1, len(train_loader))
        epoch_record = {'epoch': epoch, **{key: value / batch_count for key, value in totals.items()}, 'valid_objective': valid_choice['objective'], 'valid_gain_threshold': valid_choice['gain_threshold'], 'valid_benefit_threshold': valid_choice['benefit_threshold'], 'valid_harm_threshold': valid_choice['harm_threshold'], 'valid_switch_rate': valid_choice['switch_rate'], 'valid_switch_precision': valid_choice['switch_precision'], 'valid_harm_over_005_rate': valid_choice['harm_over_005_rate'], **{f'valid_{key}': value for key, value in valid_choice['metrics'].items()}}
        history.append(epoch_record)
        logger.info('ROUTER epoch=%d loss=%.4f valid_MAE=%.4f objective=%.4f gain_thr=%.3f benefit_thr=%.2f harm_thr=%.2f switch_rate=%.4f switch_precision=%.4f', epoch, epoch_record['loss'], valid_choice['metrics']['MAE'], valid_choice['objective'], valid_choice['gain_threshold'], valid_choice['benefit_threshold'], valid_choice['harm_threshold'], valid_choice['switch_rate'], valid_choice['switch_precision'])
        if valid_choice['objective'] < best_objective - 1e-06:
            best_objective = valid_choice['objective']
            best_epoch = epoch
            best_gate = {'gain_threshold': valid_choice['gain_threshold'], 'benefit_threshold': valid_choice['benefit_threshold'], 'harm_threshold': valid_choice['harm_threshold']}
            best_state = copy.deepcopy(router.state_dict())
            best_validation_candidates = validation_candidates
            torch.save({'router_state_dict': best_state, 'gate': best_gate, 'best_epoch': best_epoch, 'best_valid_objective': best_objective, 'seed': int(seed), 'expert_names': list(EXPERT_NAMES), 'context_dim': int(context_dim), 'hidden_dim': int(hidden_dim), 'context_hidden_dim': int(context_hidden_dim), 'dropout': float(dropout)}, checkpoint_path)
        if epoch - best_epoch >= patience:
            break
    if best_state is None:
        raise RuntimeError('Router training did not produce a valid checkpoint.')
    router.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(save_dir / 'router_history.csv', index=False)
    pd.DataFrame(_candidate_rows(best_validation_candidates, prefix='valid')).to_csv(save_dir / 'validation_gate_search.csv', index=False)
    test_collected = _score_cached(router, cached['test'], device)
    test_candidates = []
    for gain_threshold, benefit_threshold, harm_threshold in itertools.product(gain_thresholds, benefit_thresholds, harm_thresholds):
        test_candidates.append(_routing_statistics(test_collected, metrics_fn, gain_threshold, benefit_threshold, harm_threshold))
    test_candidates.append(_routing_statistics(test_collected, metrics_fn, gain_threshold=1000000.0, benefit_threshold=1.0, harm_threshold=0.0))
    pd.DataFrame(_candidate_rows(test_candidates, prefix='test')).to_csv(save_dir / 'test_gate_curve_diagnostic.csv', index=False)
    summary = save_router_report(collected=test_collected, metrics_fn=metrics_fn, gate=best_gate, save_dir=save_dir, best_epoch=best_epoch, best_valid_objective=best_objective, seed=seed)
    return (router, summary)

def save_router_report(collected, metrics_fn, gate, save_dir, best_epoch, best_valid_objective, seed):
    save_dir = Path(save_dir)
    expert_matrix = collected['expert_matrix']
    labels = collected['labels']
    routed_prediction, selected_indices, best_score = _route_from_collected(collected, gate['gain_threshold'], gate['benefit_threshold'], gate['harm_threshold'])
    fusion_prediction = expert_matrix[:, FUSION_INDEX:FUSION_INDEX + 1]
    absolute_errors = torch.abs(expert_matrix - labels)
    oracle_indices = absolute_errors.argmin(dim=1)
    oracle_prediction = expert_matrix.gather(1, oracle_indices.unsqueeze(1))
    routed_error = torch.abs(routed_prediction - labels)
    fusion_error = torch.abs(fusion_prediction - labels)
    oracle_error = torch.abs(oracle_prediction - labels)
    realized_gain = fusion_error - routed_error
    switched = selected_indices != FUSION_INDEX
    selection_counts = torch.bincount(selected_indices, minlength=len(EXPERT_NAMES)).tolist()
    selection_rates = {name: count / len(selected_indices) for name, count in zip(EXPERT_NAMES, selection_counts)}
    if switched.any():
        switch_precision = float((realized_gain[switched] > 0).float().mean().item())
        switched_mean_gain = float(realized_gain[switched].mean().item())
    else:
        switch_precision = 1.0
        switched_mean_gain = 0.0
    summary = {'seed': int(seed), 'best_epoch': int(best_epoch), 'best_valid_objective': float(best_valid_objective), 'gate': {key: float(value) for key, value in gate.items()}, 'fusion_metrics': _safe_metrics(metrics_fn, fusion_prediction, labels), 'router_metrics': _safe_metrics(metrics_fn, routed_prediction, labels), 'oracle_metrics': _safe_metrics(metrics_fn, oracle_prediction, labels), 'switch_rate': float(switched.float().mean().item()), 'switch_precision': switch_precision, 'switched_mean_gain': switched_mean_gain, 'mean_realized_gain': float(realized_gain.mean().item()), 'positive_gain_rate': float((realized_gain > 0).float().mean().item()), 'harm_over_005_rate': float((realized_gain < -0.05).float().mean().item()), 'harm_over_010_rate': float((realized_gain < -0.1).float().mean().item()), 'mean_routing_regret': float((routed_error - oracle_error).mean().item()), 'selection_rates': selection_rates}
    with open(save_dir / 'router_summary.json', 'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
    sample_ids = collected['sample_ids']
    if len(sample_ids) != len(labels):
        sample_ids = [str(index) for index in range(len(labels))]
    prediction_data = {'sample_id': sample_ids, 'target': labels.view(-1).numpy(), 'selected_expert': [EXPERT_NAMES[index] for index in selected_indices.tolist()], 'best_score': best_score.numpy(), 'fusion_prediction': fusion_prediction.view(-1).numpy(), 'router_prediction': routed_prediction.view(-1).numpy(), 'oracle_prediction': oracle_prediction.view(-1).numpy(), 'fusion_abs_error': fusion_error.view(-1).numpy(), 'router_abs_error': routed_error.view(-1).numpy(), 'realized_gain': realized_gain.view(-1).numpy()}
    for index, name in enumerate(EXPERT_NAMES):
        prediction_data[f'{name}_prediction'] = expert_matrix[:, index].numpy()
        prediction_data[f'{name}_abs_error'] = absolute_errors[:, index].numpy()
    for index, name in enumerate(ALT_EXPERT_NAMES):
        prediction_data[f'{name}_predicted_gain'] = collected['predicted_gain'][:, index].numpy()
        prediction_data[f'{name}_benefit_probability'] = collected['benefit_probability'][:, index].numpy()
        prediction_data[f'{name}_harm_probability'] = collected['harm_probability'][:, index].numpy()
        prediction_data[f'{name}_router_score'] = collected['score'][:, index].numpy()
    pd.DataFrame(prediction_data).to_csv(save_dir / 'router_predictions.csv', index=False)
    logger.info('Router results saved to %s', save_dir)
    logger.info('ROUTER TEST fusion_MAE=%.4f router_MAE=%.4f oracle_MAE=%.4f switch_rate=%.4f switch_precision=%.4f harm_010=%.4f', summary['fusion_metrics']['MAE'], summary['router_metrics']['MAE'], summary['oracle_metrics']['MAE'], summary['switch_rate'], summary['switch_precision'], summary['harm_over_010_rate'])
    return summary
