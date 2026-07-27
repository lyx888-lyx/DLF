import itertools
import math

import numpy as np
import torch

from .down_alpha_gain_v341 import ALPHAS, apply_logit_calibrators
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


def selection_stats(fusion, prediction, labels, correction):
    gain = torch.abs(fusion - labels).view(-1) - torch.abs(prediction - labels).view(-1)
    selected = correction.abs().view(-1) > 1e-8
    count = int(selected.sum().item())
    successes = int((gain[selected] > 0).sum().item()) if count else 0
    return {
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


def _alpha_index(alpha, alphas=ALPHAS):
    return min(range(len(alphas)), key=lambda index: abs(float(alphas[index]) - float(alpha)))


def apply_view_policy(output, specialist_residuals, fusion, policy, calibrators, alphas=ALPHAS):
    alpha = float(policy['alpha_down'])
    correction = torch.zeros_like(fusion)
    selected = torch.zeros(len(fusion), dtype=torch.bool)
    probabilities = apply_logit_calibrators(output, calibrators)
    if alpha > 0:
        index = _alpha_index(alpha, alphas)
        selected = (
            (probabilities['benefit'][:, index] >= float(policy['benefit_threshold']))
            & (output['predicted_gain'][:, index] >= float(policy['gain_threshold']))
            & (probabilities['harm'][:, index] <= float(policy['harm_threshold']))
        )
        correction[selected] = alpha * specialist_residuals[selected, 1:2]
    return {
        'prediction': fusion + correction,
        'correction': correction,
        'selected': selected,
        'benefit_probability': probabilities['benefit'],
        'harm_probability': probabilities['harm'],
    }


def apply_ensemble_policy(outputs, residual_views, views, policy, calibrators, alphas=ALPHAS):
    results = [
        apply_view_policy(
            output, residuals, _fusion(view['expert_matrix']),
            policy, calibrators, alphas,
        )
        for output, residuals, view in zip(outputs, residual_views, views)
    ]
    fusion = torch.stack([_fusion(view['expert_matrix']) for view in views]).mean(0)
    prediction = torch.stack([result['prediction'] for result in results]).mean(0)
    return {
        'fusion': fusion,
        'prediction': prediction,
        'correction': prediction - fusion,
        'view_results': results,
    }


def _single_cache_policy(output, residuals, cache, policy, calibrators, alphas=ALPHAS):
    result = apply_view_policy(
        output, residuals, _fusion(cache['expert_matrix']), policy, calibrators, alphas
    )
    return {
        'fusion': _fusion(cache['expert_matrix']),
        'prediction': result['prediction'],
        'correction': result['correction'],
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
            'harm_010': float((weight * (gain < -0.10).float()).sum().item()),
            'mean_abs_correction': float((weight * correction.abs().view(-1)).sum().item()),
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
        'alpha_down': 0.0,
        'benefit_threshold': 1.0,
        'gain_threshold': 1.0,
        'harm_threshold': 0.0,
    }


def _policy_key(policy):
    return tuple((key, float(policy[key])) for key in sorted(policy))


def calibrate_down_alpha_policy(
    oof_output,
    oof_residuals,
    train_cache,
    valid_outputs,
    valid_residual_views,
    valid_views,
    calibrators,
    alphas=ALPHAS,
    min_selected=25,
    min_coverage=0.05,
    min_precision=0.65,
    min_wilson_lcb=0.50,
    max_harm_010=0.02,
    min_fold_gain=-0.002,
    min_positive_folds=2,
    benefit_thresholds=None,
    gain_thresholds=None,
    harm_thresholds=None,
):
    benefit_thresholds = benefit_thresholds or [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    gain_thresholds = gain_thresholds or [-0.02, 0.00, 0.01, 0.02, 0.05, 0.10]
    harm_thresholds = harm_thresholds or [0.10, 0.20, 0.30, 0.40, 0.50]
    labels = valid_views[0]['labels']
    source_fold_ids = train_cache.get('source_fold_ids')
    if source_fold_ids is None:
        raise RuntimeError('V3.4.1 requires source_fold_ids in the OOF train cache.')

    rows = []
    policies = [_fallback_policy()]
    for alpha, benefit, gain, harm in itertools.product(
        alphas, benefit_thresholds, gain_thresholds, harm_thresholds
    ):
        policies.append({
            'alpha_down': float(alpha),
            'benefit_threshold': float(benefit),
            'gain_threshold': float(gain),
            'harm_threshold': float(harm),
        })

    for policy_index, policy in enumerate(policies):
        valid_result = apply_ensemble_policy(
            valid_outputs, valid_residual_views, valid_views,
            policy, calibrators, alphas,
        )
        oof_result = _single_cache_policy(
            oof_output, oof_residuals, train_cache,
            policy, calibrators, alphas,
        )
        valid_stats = selection_stats(
            valid_result['fusion'], valid_result['prediction'], labels,
            valid_result['correction'],
        )
        oof_stats = selection_stats(
            oof_result['fusion'], oof_result['prediction'], train_cache['labels'],
            oof_result['correction'],
        )
        fold_rows = _fold_gain_rows(
            oof_result, train_cache['labels'], source_fold_ids
        )
        positive_folds = sum(row['mean_gain'] > 0 for row in fold_rows)
        worst_fold_gain = min(row['mean_gain'] for row in fold_rows)
        scenarios = _scenario_rows(
            valid_result['fusion'], valid_result['prediction'], labels,
            valid_result['correction'],
        )
        mean_mae = float(np.mean([value['mae'] for value in scenarios.values()]))
        worst_mae = float(np.max([value['mae'] for value in scenarios.values()]))
        worst_harm = float(np.max([value['harm_010'] for value in scenarios.values()]))
        mean_correction = float(np.mean([
            value['mean_abs_correction'] for value in scenarios.values()
        ]))
        is_fallback = float(policy['alpha_down']) == 0.0
        precision = valid_stats['correction_precision']
        lcb = valid_stats['correction_precision_wilson_lcb']
        formal_feasible = is_fallback or (
            valid_stats['correction_count'] >= int(min_selected)
            and valid_stats['correction_rate'] >= float(min_coverage)
            and precision is not None and precision >= float(min_precision)
            and lcb is not None and lcb >= float(min_wilson_lcb)
            and valid_stats['mean_realized_gain'] > 0
            and valid_stats['harm_over_010_rate'] <= float(max_harm_010)
            and oof_stats['mean_realized_gain'] > 0
            and positive_folds >= int(min_positive_folds)
            and worst_fold_gain >= float(min_fold_gain)
        )
        shadow_valid_eligible = (
            not is_fallback
            and valid_stats['correction_count'] >= 15
            and valid_stats['mean_realized_gain'] > 0
            and valid_stats['harm_over_010_rate'] <= 0.05
        )
        shadow_oof_eligible = (
            not is_fallback
            and valid_stats['mean_realized_gain'] > 0
            and oof_stats['mean_realized_gain'] > 0
            and positive_folds >= int(min_positive_folds)
            and worst_fold_gain >= float(min_fold_gain)
        )
        objective = (
            mean_mae
            + 0.50 * (worst_mae - mean_mae)
            + 0.40 * worst_harm
            + 0.02 * mean_correction
        )
        rows.append({
            'policy_index': policy_index,
            **policy,
            'objective': float(objective),
            'scenario_mean_mae': mean_mae,
            'scenario_worst_mae': worst_mae,
            'scenario_worst_harm_010': worst_harm,
            'oof_positive_fold_count': int(positive_folds),
            'oof_worst_fold_gain': float(worst_fold_gain),
            'formal_feasible': bool(formal_feasible),
            'shadow_valid_eligible': bool(shadow_valid_eligible),
            'shadow_oof_eligible': bool(shadow_oof_eligible),
            **{f'valid_{key}': value for key, value in valid_stats.items()},
            **{f'oof_{key}': value for key, value in oof_stats.items()},
        })

    formal_rows = [row for row in rows if row['formal_feasible']]
    formal = min(formal_rows, key=lambda row: (
        row['objective'], row['scenario_worst_mae'],
        row['valid_harm_over_010_rate'],
        -(row['valid_correction_precision_wilson_lcb'] or 0.0),
    ))
    valid_shadow_rows = [row for row in rows if row['shadow_valid_eligible']]
    oof_shadow_rows = [row for row in rows if row['shadow_oof_eligible']]
    best_valid = min(valid_shadow_rows, key=lambda row: (
        row['valid_mae'], row['valid_harm_over_010_rate'],
        -(row['valid_correction_precision'] or 0.0),
    )) if valid_shadow_rows else formal
    best_oof = min(oof_shadow_rows, key=lambda row: (
        -row['oof_mean_realized_gain'], row['valid_mae'],
        row['valid_harm_over_010_rate'],
    )) if oof_shadow_rows else formal

    policy_fields = ('alpha_down', 'benefit_threshold', 'gain_threshold', 'harm_threshold')
    result = {
        'formal_safe_policy': {key: float(formal[key]) for key in policy_fields},
        'best_valid_gain_policy': {key: float(best_valid[key]) for key in policy_fields},
        'best_oof_stable_policy': {key: float(best_oof[key]) for key in policy_fields},
    }
    fold_diagnostics = []
    selected_keys = {_policy_key(policy) for policy in result.values()}
    for row in rows:
        policy = {key: float(row[key]) for key in policy_fields}
        if _policy_key(policy) not in selected_keys:
            continue
        oof_result = _single_cache_policy(
            oof_output, oof_residuals, train_cache, policy, calibrators, alphas
        )
        for fold_row in _fold_gain_rows(
            oof_result, train_cache['labels'], source_fold_ids
        ):
            fold_diagnostics.append({
                'policy_name': next(
                    name for name, value in result.items()
                    if _policy_key(value) == _policy_key(policy)
                ),
                **fold_row,
            })
    return result, rows, fold_diagnostics


def alpha_precision_coverage_rows(
    output, residuals, cache, calibrators, alphas=ALPHAS,
    benefit_thresholds=None, gain_thresholds=None, harm_thresholds=None,
):
    benefit_thresholds = benefit_thresholds or [0.50, 0.60, 0.65, 0.70, 0.75, 0.80]
    gain_thresholds = gain_thresholds or [0.00, 0.01, 0.02, 0.05, 0.10]
    harm_thresholds = harm_thresholds or [0.10, 0.20, 0.30, 0.40, 0.50]
    rows = []
    for alpha, benefit, gain, harm in itertools.product(
        alphas, benefit_thresholds, gain_thresholds, harm_thresholds
    ):
        policy = {
            'alpha_down': float(alpha),
            'benefit_threshold': float(benefit),
            'gain_threshold': float(gain),
            'harm_threshold': float(harm),
        }
        result = _single_cache_policy(output, residuals, cache, policy, calibrators, alphas)
        rows.append({
            **policy,
            **selection_stats(
                result['fusion'], result['prediction'], cache['labels'],
                result['correction'],
            ),
        })
    return rows


def test_alpha_diagnostics(output, cache, residuals, calibrators, alphas=ALPHAS):
    from .down_alpha_gain_v341 import down_alpha_targets

    targets = down_alpha_targets(cache, residuals, alphas)
    probabilities = apply_logit_calibrators(output, calibrators)
    rows = []
    for alpha_index, alpha in enumerate(alphas):
        actual = targets['gains'][:, alpha_index]
        score = (
            output['predicted_gain'][:, alpha_index]
            + 0.10 * (probabilities['benefit'][:, alpha_index] - 0.5)
            - 0.10 * probabilities['harm'][:, alpha_index]
        )
        for fraction in (0.05, 0.10, 0.20, 0.30):
            count = max(1, int(math.ceil(len(actual) * fraction)))
            selected = torch.topk(score, count).indices
            rows.append({
                'alpha_down': float(alpha),
                'top_fraction': float(fraction),
                'count': int(count),
                'precision': float((actual[selected] > 0).float().mean().item()),
                'mean_actual_gain': float(actual[selected].mean().item()),
                'mean_predicted_gain': float(
                    output['predicted_gain'][selected, alpha_index].mean().item()
                ),
                'mean_benefit_probability': float(
                    probabilities['benefit'][selected, alpha_index].mean().item()
                ),
                'mean_harm_probability': float(
                    probabilities['harm'][selected, alpha_index].mean().item()
                ),
                'harm_over_010_rate': float(
                    (actual[selected] < -0.10).float().mean().item()
                ),
            })
    return rows
