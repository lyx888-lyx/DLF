import itertools
import math

import numpy as np
import torch

from .candidate_gain_v34 import CANDIDATE_NAMES, candidate_gain_targets
from .residual_direction_core import _fusion, _regions

REGION_NAMES = ('strong_negative', 'negative', 'boundary', 'positive', 'strong_positive')
SCENARIOS = {
    'original': (1.0, 1.0, 1.0, 1.0, 1.0),
    'negative_heavy': (2.5, 1.8, 1.0, 0.8, 0.8),
    'positive_heavy': (0.8, 0.8, 1.0, 1.8, 2.5),
    'boundary_heavy': (1.0, 1.0, 2.5, 1.0, 1.0),
    'strong_heavy': (2.0, 1.0, 1.0, 1.0, 2.0),
}


def wilson_lower_bound(successes, total, confidence_z=1.96):
    if total <= 0:
        return None
    p = successes / total
    z2 = confidence_z ** 2
    denominator = 1.0 + z2 / total
    center = p + z2 / (2.0 * total)
    radius = confidence_z * math.sqrt(
        (p * (1.0 - p) + z2 / (4.0 * total)) / total
    )
    return max(0.0, (center - radius) / denominator)


def selection_stats(fusion, prediction, labels, correction, action=None):
    gain = torch.abs(fusion - labels).view(-1) - torch.abs(prediction - labels).view(-1)
    selected = correction.abs().view(-1) > 1e-8
    count = int(selected.sum().item())
    successes = int((gain[selected] > 0).sum().item()) if count else 0
    result = {
        'mae': float(torch.abs(prediction - labels).mean().item()),
        'mean_realized_gain': float(gain.mean().item()),
        'selected_mean_gain': float(gain[selected].mean().item()) if count else None,
        'correction_rate': float(selected.float().mean().item()),
        'correction_count': count,
        'mean_abs_correction': float(correction.abs().mean().item()),
        'harm_over_005_rate': float((gain < -0.05).float().mean().item()),
        'harm_over_010_rate': float((gain < -0.10).float().mean().item()),
        'correction_precision': float(successes / count) if count else None,
        'correction_precision_wilson_lcb': wilson_lower_bound(successes, count),
    }
    if action is not None:
        for index, name in ((1, 'down'), (2, 'up')):
            mask = action == index
            action_count = int(mask.sum().item())
            action_good = int((gain[mask] > 0).sum().item()) if action_count else 0
            result[f'{name}_count'] = action_count
            result[f'{name}_precision'] = (
                float(action_good / action_count) if action_count else None
            )
    return result


def _candidate_probabilities(output):
    return (
        torch.sigmoid(output['benefit_logits']),
        torch.sigmoid(output['harm_logits']),
    )


def _candidate_score(output):
    benefit_probability, harm_probability = _candidate_probabilities(output)
    return (
        output['predicted_gain']
        + 0.10 * (benefit_probability - 0.5)
        - 0.10 * harm_probability
    )


def apply_view_policy(output, specialist_residuals, fusion, policy):
    benefit_probability, harm_probability = _candidate_probabilities(output)
    score = _candidate_score(output)
    up_eligible = (
        (benefit_probability[:, 0] >= float(policy['up_benefit_threshold']))
        & (output['predicted_gain'][:, 0] >= float(policy['up_gain_threshold']))
        & (harm_probability[:, 0] <= float(policy['up_harm_threshold']))
        & (float(policy['alpha_up']) > 0)
    )
    down_eligible = (
        (benefit_probability[:, 1] >= float(policy['down_benefit_threshold']))
        & (output['predicted_gain'][:, 1] >= float(policy['down_gain_threshold']))
        & (harm_probability[:, 1] <= float(policy['down_harm_threshold']))
        & (float(policy['alpha_down']) > 0)
    )
    margin = float(policy['decision_margin'])
    choose_up = up_eligible & (~down_eligible | ((score[:, 0] - score[:, 1]) >= margin))
    choose_down = down_eligible & (~up_eligible | ((score[:, 1] - score[:, 0]) >= margin))
    conflict = choose_up & choose_down
    if conflict.any():
        up_wins = score[:, 0] >= score[:, 1]
        choose_up = choose_up & (~conflict | up_wins)
        choose_down = choose_down & (~conflict | ~up_wins)

    correction = torch.zeros_like(fusion)
    correction[choose_up] = float(policy['alpha_up']) * specialist_residuals[choose_up, 0:1]
    correction[choose_down] = float(policy['alpha_down']) * specialist_residuals[choose_down, 1:2]
    action = torch.zeros(len(fusion), dtype=torch.long)
    action[choose_down] = 1
    action[choose_up] = 2
    return {
        'prediction': fusion + correction,
        'correction': correction,
        'action': action,
        'score': score,
        'benefit_probability': benefit_probability,
        'harm_probability': harm_probability,
    }


def apply_ensemble_policy(outputs, residual_views, views, policy):
    results = [
        apply_view_policy(output, residuals, _fusion(view['expert_matrix']), policy)
        for output, residuals, view in zip(outputs, residual_views, views)
    ]
    fusion = torch.stack([_fusion(view['expert_matrix']) for view in views]).mean(0)
    prediction = torch.stack([result['prediction'] for result in results]).mean(0)
    correction = prediction - fusion
    votes = torch.stack([result['action'] for result in results])
    action = torch.zeros(len(fusion), dtype=torch.long)
    for sample_index in range(len(fusion)):
        nonzero = votes[:, sample_index]
        nonzero = nonzero[nonzero > 0]
        if len(nonzero):
            action[sample_index] = int(torch.bincount(nonzero, minlength=3).argmax().item())
    action[correction.abs().view(-1) <= 1e-8] = 0
    return {
        'fusion': fusion,
        'prediction': prediction,
        'correction': correction,
        'action': action,
        'view_results': results,
    }


def _single_cache_policy(output, residuals, cache, policy):
    result = apply_view_policy(output, residuals, _fusion(cache['expert_matrix']), policy)
    return {
        'fusion': _fusion(cache['expert_matrix']),
        'prediction': result['prediction'],
        'correction': result['correction'],
        'action': result['action'],
    }


def _scenario_rows(fusion, prediction, labels, correction):
    error = torch.abs(prediction - labels).view(-1)
    gain = torch.abs(fusion - labels).view(-1) - error
    regions = _regions(labels)
    counts = torch.bincount(regions, minlength=len(REGION_NAMES)).to(error.dtype)
    scenarios = dict(SCENARIOS)
    scenarios['balanced'] = tuple((counts.sum() / counts.clamp_min(1.0)).tolist())
    rows = {}
    for name, factors in scenarios.items():
        weight = torch.tensor(factors, dtype=error.dtype)[regions]
        weight = weight / weight.sum().clamp_min(1e-12)
        rows[name] = {
            'mae': float((weight * error).sum().item()),
            'harm_005': float((weight * (gain < -0.05).float()).sum().item()),
            'harm_010': float((weight * (gain < -0.10).float()).sum().item()),
            'mean_abs_correction': float(
                (weight * correction.abs().view(-1)).sum().item()
            ),
        }
    return rows


def _fold_gain_rows(result, labels, source_fold_ids):
    gain = (
        torch.abs(result['fusion'] - labels).view(-1)
        - torch.abs(result['prediction'] - labels).view(-1)
    )
    selected = result['correction'].abs().view(-1) > 1e-8
    rows = []
    for fold_id in torch.unique(source_fold_ids).tolist():
        mask = source_fold_ids == int(fold_id)
        rows.append({
            'fold': int(fold_id),
            'count': int(mask.sum().item()),
            'mean_gain': float(gain[mask].mean().item()),
            'selected_count': int((selected & mask).sum().item()),
            'selected_mean_gain': float(gain[selected & mask].mean().item())
            if (selected & mask).any() else None,
        })
    return rows


def _fallback_policy():
    return {
        'up_benefit_threshold': 1.0,
        'up_gain_threshold': 1.0,
        'up_harm_threshold': 0.0,
        'down_benefit_threshold': 1.0,
        'down_gain_threshold': 1.0,
        'down_harm_threshold': 0.0,
        'decision_margin': 0.0,
        'alpha_up': 0.0,
        'alpha_down': 0.0,
    }


def _candidate_policy(candidate, benefit_threshold, gain_threshold, harm_threshold, alpha):
    policy = _fallback_policy()
    policy[f'{candidate}_benefit_threshold'] = float(benefit_threshold)
    policy[f'{candidate}_gain_threshold'] = float(gain_threshold)
    policy[f'{candidate}_harm_threshold'] = float(harm_threshold)
    policy[f'alpha_{candidate}'] = float(alpha)
    return policy


def build_candidate_curves(
    oof_output,
    oof_residuals,
    train_cache,
    valid_outputs,
    valid_residual_views,
    valid_views,
    benefit_thresholds=None,
    gain_thresholds=None,
    harm_thresholds=None,
    alpha_values=None,
):
    benefit_thresholds = benefit_thresholds or [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    gain_thresholds = gain_thresholds or [0.00, 0.02, 0.05, 0.10, 0.15]
    harm_thresholds = harm_thresholds or [0.10, 0.20, 0.30, 0.40]
    alpha_values = alpha_values or [0.25, 0.50]
    rows = []
    for candidate in CANDIDATE_NAMES:
        for benefit, gain, harm, alpha in itertools.product(
            benefit_thresholds, gain_thresholds, harm_thresholds, alpha_values
        ):
            policy = _candidate_policy(candidate, benefit, gain, harm, alpha)
            oof_result = _single_cache_policy(
                oof_output, oof_residuals, train_cache, policy
            )
            valid_result = apply_ensemble_policy(
                valid_outputs, valid_residual_views, valid_views, policy
            )
            oof_stats = selection_stats(
                oof_result['fusion'], oof_result['prediction'], train_cache['labels'],
                oof_result['correction'], oof_result['action'],
            )
            valid_stats = selection_stats(
                valid_result['fusion'], valid_result['prediction'], valid_views[0]['labels'],
                valid_result['correction'], valid_result['action'],
            )
            fold_rows = _fold_gain_rows(
                oof_result, train_cache['labels'], train_cache['source_fold_ids']
            )
            min_fold_gain = min(row['mean_gain'] for row in fold_rows)
            objective = (
                valid_stats['mae']
                + 0.35 * valid_stats['harm_over_010_rate']
                + 0.08 * max(
                    0.0,
                    0.65 - (valid_stats['correction_precision'] or 0.0),
                )
                - 0.10 * min(0.05, max(-0.05, oof_stats['mean_realized_gain']))
            )
            rows.append({
                'candidate': candidate,
                'benefit_threshold': float(benefit),
                'gain_threshold': float(gain),
                'harm_threshold': float(harm),
                'alpha': float(alpha),
                'ranking_objective': float(objective),
                'oof_min_fold_gain': float(min_fold_gain),
                **{f'oof_{key}': value for key, value in oof_stats.items()},
                **{f'valid_{key}': value for key, value in valid_stats.items()},
            })
    return rows


def _top_candidate_configs(rows, candidate, limit):
    selected = [row for row in rows if row['candidate'] == candidate]
    selected.sort(key=lambda row: (
        row['ranking_objective'],
        -row['oof_min_fold_gain'],
        row['valid_harm_over_010_rate'],
        -float(row['valid_correction_precision'] or 0.0),
    ))
    result = [{
        'benefit_threshold': 1.0,
        'gain_threshold': 1.0,
        'harm_threshold': 0.0,
        'alpha': 0.0,
    }]
    for row in selected:
        config = {
            'benefit_threshold': float(row['benefit_threshold']),
            'gain_threshold': float(row['gain_threshold']),
            'harm_threshold': float(row['harm_threshold']),
            'alpha': float(row['alpha']),
        }
        if config not in result:
            result.append(config)
        if len(result) >= int(limit) + 1:
            break
    return result


def calibrate_candidate_gain_policy(
    oof_output,
    oof_residuals,
    train_cache,
    valid_outputs,
    valid_residual_views,
    valid_views,
    min_selected=15,
    min_coverage=0.05,
    min_wilson_lcb=0.60,
    max_harm_010=0.02,
    candidate_config_limit=12,
    decision_margins=None,
):
    curve_rows = build_candidate_curves(
        oof_output, oof_residuals, train_cache,
        valid_outputs, valid_residual_views, valid_views,
    )
    up_configs = _top_candidate_configs(curve_rows, 'up', candidate_config_limit)
    down_configs = _top_candidate_configs(curve_rows, 'down', candidate_config_limit)
    decision_margins = decision_margins or [0.0, 0.02, 0.05]
    labels = valid_views[0]['labels']
    rows, feasible_rows, fold_diagnostics = [], [], []

    for up, down, margin in itertools.product(
        up_configs, down_configs, decision_margins
    ):
        policy = {
            'up_benefit_threshold': float(up['benefit_threshold']),
            'up_gain_threshold': float(up['gain_threshold']),
            'up_harm_threshold': float(up['harm_threshold']),
            'down_benefit_threshold': float(down['benefit_threshold']),
            'down_gain_threshold': float(down['gain_threshold']),
            'down_harm_threshold': float(down['harm_threshold']),
            'decision_margin': float(margin),
            'alpha_up': float(up['alpha']),
            'alpha_down': float(down['alpha']),
        }
        valid_result = apply_ensemble_policy(
            valid_outputs, valid_residual_views, valid_views, policy
        )
        oof_result = _single_cache_policy(
            oof_output, oof_residuals, train_cache, policy
        )
        valid_stats = selection_stats(
            valid_result['fusion'], valid_result['prediction'], labels,
            valid_result['correction'], valid_result['action'],
        )
        oof_stats = selection_stats(
            oof_result['fusion'], oof_result['prediction'], train_cache['labels'],
            oof_result['correction'], oof_result['action'],
        )
        scenario_values = _scenario_rows(
            valid_result['fusion'], valid_result['prediction'], labels,
            valid_result['correction'],
        )
        mean_mae = float(np.mean([value['mae'] for value in scenario_values.values()]))
        worst_mae = float(np.max([value['mae'] for value in scenario_values.values()]))
        worst_harm = float(np.max([value['harm_010'] for value in scenario_values.values()]))
        fold_rows = _fold_gain_rows(
            oof_result, train_cache['labels'], train_cache['source_fold_ids']
        )
        min_fold_gain = min(row['mean_gain'] for row in fold_rows)
        fallback = policy['alpha_up'] == 0 and policy['alpha_down'] == 0
        lcb = valid_stats['correction_precision_wilson_lcb']
        feasible = fallback or (
            valid_stats['correction_count'] >= int(min_selected)
            and valid_stats['correction_rate'] >= float(min_coverage)
            and valid_stats['mean_realized_gain'] > 0
            and valid_stats['harm_over_010_rate'] <= float(max_harm_010)
            and lcb is not None
            and lcb >= float(min_wilson_lcb)
            and oof_stats['mean_realized_gain'] > 0
            and min_fold_gain > 0
        )
        objective = (
            mean_mae
            + 0.50 * (worst_mae - mean_mae)
            + 0.40 * worst_harm
            + 0.02 * valid_stats['mean_abs_correction']
        )
        row = {
            **policy,
            'objective': float(objective),
            'scenario_mean_mae': mean_mae,
            'scenario_worst_mae': worst_mae,
            'scenario_worst_harm_010': worst_harm,
            'oof_min_fold_gain': float(min_fold_gain),
            'feasible': bool(feasible),
            **{f'valid_{key}': value for key, value in valid_stats.items()},
            **{f'oof_{key}': value for key, value in oof_stats.items()},
        }
        policy_index = len(rows)
        rows.append(row)
        for fold_row in fold_rows:
            fold_diagnostics.append({'policy_index': policy_index, **fold_row})
        if feasible:
            feasible_rows.append(row)

    if not feasible_rows:
        raise RuntimeError('Candidate-gain calibration produced no feasible fallback policy.')
    best = min(feasible_rows, key=lambda row: (
        row['objective'],
        row['scenario_worst_mae'],
        row['valid_harm_over_010_rate'],
        -(row['valid_correction_precision_wilson_lcb'] or 0.0),
        row['valid_correction_rate'],
    ))
    policy_keys = (
        'up_benefit_threshold', 'up_gain_threshold', 'up_harm_threshold',
        'down_benefit_threshold', 'down_gain_threshold', 'down_harm_threshold',
        'decision_margin', 'alpha_up', 'alpha_down',
    )
    return (
        {key: float(best[key]) for key in policy_keys},
        curve_rows,
        rows,
        fold_diagnostics,
    )


def test_gain_diagnostics(output, cache, specialist_residuals):
    targets = candidate_gain_targets(cache, specialist_residuals)
    benefit_probability, harm_probability = _candidate_probabilities(output)
    score = _candidate_score(output)
    rows = []
    for index, name in enumerate(CANDIDATE_NAMES):
        actual_gain = targets['gains'][:, index]
        prediction = output['predicted_gain'][:, index]
        for fraction in (0.05, 0.10, 0.20, 0.30):
            count = max(1, int(math.ceil(len(actual_gain) * fraction)))
            selected = torch.topk(score[:, index], count).indices
            rows.append({
                'candidate': name,
                'top_fraction': float(fraction),
                'count': int(count),
                'precision': float((actual_gain[selected] > 0).float().mean().item()),
                'mean_actual_gain': float(actual_gain[selected].mean().item()),
                'mean_predicted_gain': float(prediction[selected].mean().item()),
                'mean_benefit_probability': float(
                    benefit_probability[selected, index].mean().item()
                ),
                'mean_harm_probability': float(
                    harm_probability[selected, index].mean().item()
                ),
                'harm_over_010_rate': float(
                    (actual_gain[selected] < -0.10).float().mean().item()
                ),
            })
    return rows
