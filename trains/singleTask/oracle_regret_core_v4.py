import itertools
import math

import torch
import torch.nn.functional as F


REGION_NAMES = ('strong_negative', 'negative', 'boundary', 'positive', 'strong_positive')


def region_index(labels):
    values = labels.view(-1)
    result = torch.full_like(values, 2, dtype=torch.long)
    result[values < -1.5] = 0
    result[(values >= -1.5) & (values < -0.5)] = 1
    result[(values > 0.5) & (values <= 1.5)] = 3
    result[values > 1.5] = 4
    return result


def oracle_teacher(candidate_values, labels, temperature=0.10):
    errors = torch.abs(candidate_values.detach() - labels)
    probabilities = F.softmax(-errors / max(float(temperature), 1e-4), dim=1)
    return probabilities, errors


def regret_ranking_loss(logits, detached_errors):
    error_i = detached_errors.unsqueeze(2)
    error_j = detached_errors.unsqueeze(1)
    better = error_i < error_j
    regret = (error_j - error_i).clamp_min(0.0)
    logit_i = logits.unsqueeze(2)
    logit_j = logits.unsqueeze(1)
    pair_loss = F.softplus(logit_j - logit_i) * regret
    denominator = regret[better].sum().clamp_min(1e-8)
    return pair_loss[better].sum() / denominator


def oracle_regret_losses(
    output,
    labels,
    teacher_temperature=0.10,
    student_temperature=1.0,
):
    teacher, detached_errors = oracle_teacher(
        output['candidate_values'], labels, teacher_temperature
    )
    log_probabilities = F.log_softmax(
        output['candidate_logits'] / max(float(student_temperature), 1e-4), dim=1
    )
    probabilities = log_probabilities.exp()
    distill = -(teacher * log_probabilities).sum(dim=1).mean()

    candidate_errors = torch.abs(output['candidate_values'] - labels)
    expected_mae = (probabilities * candidate_errors).sum(dim=1).mean()
    routed_prediction = (
        probabilities * output['candidate_values']
    ).sum(dim=1, keepdim=True)
    routed_mae = F.l1_loss(routed_prediction, labels)
    ranking = regret_ranking_loss(output['candidate_logits'], detached_errors)

    oracle_index = detached_errors.argmin(dim=1)
    selected_index = output['candidate_logits'].argmax(dim=1)
    top1_accuracy = (selected_index == oracle_index).float().mean()
    oracle_error = detached_errors.min(dim=1).values
    selected_error = detached_errors.gather(1, selected_index.view(-1, 1)).view(-1)
    mean_regret = (selected_error - oracle_error).mean()
    fusion_error = torch.abs(output['output_logit'].detach() - labels).view(-1)
    oracle_gap = (fusion_error - oracle_error).clamp_min(0.0)
    normalized_regret = (
        (selected_error - oracle_error).clamp_min(0.0).sum()
        / oracle_gap.sum().clamp_min(1e-8)
    )
    return {
        'distill': distill,
        'expected_mae': expected_mae,
        'routed_mae': routed_mae,
        'ranking': ranking,
        'teacher_probabilities': teacher,
        'student_probabilities': probabilities,
        'oracle_index': oracle_index,
        'top1_accuracy': top1_accuracy,
        'mean_regret': mean_regret,
        'normalized_regret': normalized_regret,
    }


def entropy(probabilities):
    return -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1)


def apply_inference_policy(raw, policy):
    fusion = raw['fusion']
    candidates = raw['candidate_values']
    logits = raw['candidate_logits']
    temperature = float(policy.get('temperature', 1.0))
    blend = float(policy.get('blend', 1.0))
    mode = policy.get('mode', 'soft')
    threshold = policy.get('entropy_threshold')

    probabilities = F.softmax(logits / max(temperature, 1e-4), dim=1)
    if mode == 'hard':
        index = logits.argmax(dim=1)
        raw_prediction = candidates.gather(1, index.view(-1, 1))
    else:
        raw_prediction = (probabilities * candidates).sum(dim=1, keepdim=True)
    prediction = fusion + blend * (raw_prediction - fusion)
    uncertainty = entropy(probabilities)
    if threshold is not None:
        fallback = uncertainty > float(threshold)
        prediction = prediction.clone()
        prediction[fallback] = fusion[fallback]
    correction = prediction - fusion
    return {
        'prediction': prediction,
        'correction': correction,
        'probabilities': probabilities,
        'entropy': uncertainty,
    }


def selection_stats(fusion, prediction, labels):
    gain = torch.abs(fusion - labels).view(-1) - torch.abs(prediction - labels).view(-1)
    selected = (prediction - fusion).abs().view(-1) > 1e-8
    count = int(selected.sum().item())
    return {
        'mae': float(torch.abs(prediction - labels).mean().item()),
        'mean_realized_gain': float(gain.mean().item()),
        'correction_rate': float(selected.float().mean().item()),
        'correction_count': count,
        'correction_precision': float((gain[selected] > 0).float().mean().item()) if count else None,
        'harm_over_005_rate': float((gain < -0.05).float().mean().item()),
        'harm_over_010_rate': float((gain < -0.10).float().mean().item()),
        'mean_abs_correction': float((prediction - fusion).abs().mean().item()),
    }


def calibrate_inference_policy(raw, labels):
    fallback = {
        'mode': 'soft', 'temperature': 1.0, 'blend': 0.0,
        'entropy_threshold': None,
    }
    policies = [fallback]
    for mode, temperature, blend, threshold in itertools.product(
        ('soft', 'hard'),
        (0.50, 0.75, 1.00, 1.50, 2.00),
        (0.25, 0.50, 0.75, 1.00),
        (None, 1.40, 1.60, 1.80, 2.00),
    ):
        policies.append({
            'mode': mode,
            'temperature': float(temperature),
            'blend': float(blend),
            'entropy_threshold': threshold,
        })
    rows = []
    for policy in policies:
        result = apply_inference_policy(raw, policy)
        stats = selection_stats(raw['fusion'], result['prediction'], labels)
        objective = (
            stats['mae']
            + 0.15 * stats['harm_over_010_rate']
            + 0.01 * stats['mean_abs_correction']
        )
        rows.append({**policy, 'objective': float(objective), **stats})
    best_mae = min(rows, key=lambda row: (
        row['mae'], row['harm_over_010_rate'], row['mean_abs_correction']
    ))
    robust = min(rows, key=lambda row: (
        row['objective'], row['mae'], row['harm_over_010_rate']
    ))
    fields = ('mode', 'temperature', 'blend', 'entropy_threshold')
    return (
        {key: best_mae[key] for key in fields},
        {key: robust[key] for key in fields},
        rows,
    )


def candidate_diagnostics(raw, labels):
    candidates = raw['candidate_values']
    logits = raw['candidate_logits']
    errors = torch.abs(candidates - labels)
    oracle_index = errors.argmin(dim=1)
    hard_index = logits.argmax(dim=1)
    oracle_prediction = candidates.gather(1, oracle_index.view(-1, 1))
    hard_prediction = candidates.gather(1, hard_index.view(-1, 1))
    fusion_error = torch.abs(raw['fusion'] - labels).view(-1)
    oracle_error = torch.abs(oracle_prediction - labels).view(-1)
    hard_error = torch.abs(hard_prediction - labels).view(-1)
    return {
        'oracle_prediction': oracle_prediction,
        'hard_prediction': hard_prediction,
        'oracle_index': oracle_index,
        'hard_index': hard_index,
        'candidate_top1_accuracy': float((oracle_index == hard_index).float().mean().item()),
        'mean_candidate_regret': float((hard_error - oracle_error).mean().item()),
        'normalized_candidate_regret': float(
            (hard_error - oracle_error).clamp_min(0.0).sum().item()
            / (fusion_error - oracle_error).clamp_min(0.0).sum().clamp_min(1e-8).item()
        ),
    }


def region_rows(fusion, prediction, oracle_prediction, labels):
    regions = region_index(labels)
    rows = []
    for index, name in enumerate(REGION_NAMES):
        mask = regions == index
        if not mask.any():
            continue
        rows.append({
            'region': name,
            'count': int(mask.sum().item()),
            'fusion_mae': float(torch.abs(fusion[mask] - labels[mask]).mean().item()),
            'routed_mae': float(torch.abs(prediction[mask] - labels[mask]).mean().item()),
            'oracle_mae': float(torch.abs(oracle_prediction[mask] - labels[mask]).mean().item()),
            'mean_bias_fusion': float((labels[mask] - fusion[mask]).mean().item()),
            'mean_bias_routed': float((labels[mask] - prediction[mask]).mean().item()),
        })
    return rows
