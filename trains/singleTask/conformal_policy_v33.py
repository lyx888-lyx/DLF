import itertools
import math

import numpy as np
import torch

from .residual_direction_core import _fusion, _regions

REGION_NAMES = ('strong_negative', 'negative', 'boundary', 'positive', 'strong_positive')
SCENARIOS = {
    'original': (1.0, 1.0, 1.0, 1.0, 1.0),
    'negative_heavy': (2.5, 1.8, 1.0, 0.8, 0.8),
    'positive_heavy': (0.8, 0.8, 1.0, 1.8, 2.5),
    'boundary_heavy': (1.0, 1.0, 2.5, 1.0, 1.0),
    'strong_heavy': (2.0, 1.0, 1.0, 1.0, 2.0),
}


def fit_split_conformal(oof_quantiles, residual_target, target_coverage=0.80):
    target = residual_target.view(-1)
    lower, upper = oof_quantiles[:, 0], oof_quantiles[:, 2]
    scores = torch.maximum(lower - target, target - upper).clamp_min(0.0)
    n = len(scores)
    rank = min(n, max(1, int(math.ceil((n + 1) * float(target_coverage)))))
    offset = float(torch.kthvalue(scores, rank).values.item())
    return {
        'target_coverage': float(target_coverage),
        'offset': offset,
        'sample_count': int(n),
        'score_mean': float(scores.mean().item()),
        'score_max': float(scores.max().item()),
    }


def conformalize(quantiles, conformal):
    result = quantiles.clone()
    result[:, 0] -= float(conformal['offset'])
    result[:, 2] += float(conformal['offset'])
    return result


def coverage_rows(quantiles, labels, fusion, conformal=None):
    values = conformalize(quantiles, conformal) if conformal is not None else quantiles
    target = (labels - fusion).view(-1)
    regions = _regions(labels)
    rows = []
    for region_index, region_name in enumerate(REGION_NAMES):
        mask = regions == region_index
        if not mask.any():
            continue
        lower, upper = values[mask, 0], values[mask, 2]
        rows.append({
            'region': region_name,
            'count': int(mask.sum().item()),
            'coverage': float(((target[mask] >= lower) & (target[mask] <= upper)).float().mean().item()),
            'mean_width': float((upper - lower).mean().item()),
            'median_residual_mae': float(torch.abs(values[mask, 1] - target[mask]).mean().item()),
        })
    lower, upper = values[:, 0], values[:, 2]
    rows.append({
        'region': 'overall',
        'count': int(len(target)),
        'coverage': float(((target >= lower) & (target <= upper)).float().mean().item()),
        'mean_width': float((upper - lower).mean().item()),
        'median_residual_mae': float(torch.abs(values[:, 1] - target).mean().item()),
    })
    return rows


def wilson_lower_bound(successes, total, confidence_z=1.96):
    if total <= 0:
        return None
    p = successes / total
    z2 = confidence_z ** 2
    denominator = 1.0 + z2 / total
    center = p + z2 / (2.0 * total)
    radius = confidence_z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)
    return max(0.0, (center - radius) / denominator)


def selection_stats(fusion, prediction, labels, correction, action=None):
    gain = torch.abs(fusion - labels).view(-1) - torch.abs(prediction - labels).view(-1)
    selected = correction.abs().view(-1) > 1e-8
    selected_count = int(selected.sum().item())
    successes = int((gain[selected] > 0).sum().item()) if selected_count else 0
    result = {
        'mae': float(torch.abs(prediction - labels).mean().item()),
        'mean_realized_gain': float(gain.mean().item()),
        'correction_rate': float(selected.float().mean().item()),
        'correction_count': selected_count,
        'mean_abs_correction': float(correction.abs().mean().item()),
        'harm_over_005_rate': float((gain < -0.05).float().mean().item()),
        'harm_over_010_rate': float((gain < -0.10).float().mean().item()),
        'correction_precision': float(successes / selected_count) if selected_count else None,
        'correction_precision_wilson_lcb': wilson_lower_bound(successes, selected_count),
    }
    if action is not None:
        for index, name in ((1, 'down'), (2, 'up'), (3, 'boundary')):
            mask = action == index
            count = int(mask.sum().item())
            good = int((gain[mask] > 0).sum().item()) if count else 0
            result[f'{name}_count'] = count
            result[f'{name}_precision'] = float(good / count) if count else None
    return result


def apply_view_policy(
    output,
    specialist_residuals,
    fusion,
    conformal,
    interval_margin,
    max_interval_width,
    region_threshold,
    boundary_threshold,
    alpha_up,
    alpha_down,
    alpha_boundary,
):
    quantiles = conformalize(output['quantiles'], conformal)
    lower, upper = quantiles[:, 0], quantiles[:, 2]
    width = upper - lower
    probabilities = output['region_probs']
    negative_probability = probabilities[:, 0] + probabilities[:, 1]
    boundary_probability = probabilities[:, 2]
    positive_probability = probabilities[:, 3] + probabilities[:, 4]

    choose_down = (
        (upper <= -float(interval_margin))
        & (width <= float(max_interval_width))
        & (negative_probability >= float(region_threshold))
        & (float(alpha_down) > 0)
    )
    choose_up = (
        (lower >= float(interval_margin))
        & (width <= float(max_interval_width))
        & (positive_probability >= float(region_threshold))
        & (float(alpha_up) > 0)
    )
    remaining = ~(choose_down | choose_up)
    choose_boundary = (
        remaining
        & (lower <= 0)
        & (upper >= 0)
        & (width <= float(max_interval_width))
        & (boundary_probability >= float(boundary_threshold))
        & (float(alpha_boundary) > 0)
    )

    correction = torch.zeros_like(fusion)
    correction[choose_up] = float(alpha_up) * specialist_residuals[choose_up, 0:1]
    correction[choose_down] = float(alpha_down) * specialist_residuals[choose_down, 1:2]
    correction[choose_boundary] = float(alpha_boundary) * specialist_residuals[choose_boundary, 2:3]
    action = torch.zeros(len(fusion), dtype=torch.long)
    action[choose_down] = 1
    action[choose_up] = 2
    action[choose_boundary] = 3
    return {
        'prediction': fusion + correction,
        'correction': correction,
        'action': action,
        'quantiles': quantiles,
        'negative_probability': negative_probability,
        'boundary_probability': boundary_probability,
        'positive_probability': positive_probability,
    }


def apply_ensemble_policy(outputs, residual_views, views, conformal, policy):
    results = [
        apply_view_policy(
            output, residuals, _fusion(view['expert_matrix']), conformal, **policy
        )
        for output, residuals, view in zip(outputs, residual_views, views)
    ]
    fusion = torch.stack([_fusion(view['expert_matrix']) for view in views]).mean(0)
    prediction = torch.stack([result['prediction'] for result in results]).mean(0)
    correction = prediction - fusion
    action_votes = torch.stack([result['action'] for result in results])
    action = torch.zeros(len(fusion), dtype=torch.long)
    for sample_index in range(len(fusion)):
        values = action_votes[:, sample_index]
        nonzero = values[values > 0]
        if len(nonzero) > 0:
            counts = torch.bincount(nonzero, minlength=4)
            action[sample_index] = int(counts.argmax().item())
    action[correction.abs().view(-1) <= 1e-8] = 0
    return {
        'fusion': fusion,
        'prediction': prediction,
        'correction': correction,
        'action': action,
        'view_results': results,
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
            'mean_abs_correction': float((weight * correction.abs().view(-1)).sum().item()),
        }
    return rows


def calibrate_policy(
    outputs,
    residual_views,
    valid_views,
    conformal,
    min_selected=15,
    min_coverage=0.05,
    min_wilson_lcb=0.60,
    interval_margins=None,
    max_interval_widths=None,
    region_thresholds=None,
    boundary_thresholds=None,
    alpha_values=None,
    robustness_weight=0.50,
    harm_penalty=0.35,
    correction_penalty=0.02,
):
    interval_margins = interval_margins or [0.0, 0.05, 0.10, 0.20]
    max_interval_widths = max_interval_widths or [0.75, 1.00, 1.50, 2.00]
    region_thresholds = region_thresholds or [0.40, 0.50, 0.60, 0.70]
    boundary_thresholds = boundary_thresholds or [0.50, 0.60, 0.70]
    alpha_values = alpha_values or [0.0, 0.5, 1.0]
    labels = valid_views[0]['labels']
    rows, feasible_rows = [], []
    for margin, width, region_threshold, boundary_threshold, alpha_up, alpha_down, alpha_boundary in itertools.product(
        interval_margins,
        max_interval_widths,
        region_thresholds,
        boundary_thresholds,
        alpha_values,
        alpha_values,
        alpha_values,
    ):
        policy = {
            'interval_margin': float(margin),
            'max_interval_width': float(width),
            'region_threshold': float(region_threshold),
            'boundary_threshold': float(boundary_threshold),
            'alpha_up': float(alpha_up),
            'alpha_down': float(alpha_down),
            'alpha_boundary': float(alpha_boundary),
        }
        result = apply_ensemble_policy(outputs, residual_views, valid_views, conformal, policy)
        stats = selection_stats(
            result['fusion'], result['prediction'], labels, result['correction'], result['action']
        )
        scenario_values = _scenario_rows(
            result['fusion'], result['prediction'], labels, result['correction']
        )
        mean_mae = float(np.mean([value['mae'] for value in scenario_values.values()]))
        worst_mae = float(np.max([value['mae'] for value in scenario_values.values()]))
        worst_harm = float(np.max([value['harm_005'] for value in scenario_values.values()]))
        mean_correction = float(np.mean([value['mean_abs_correction'] for value in scenario_values.values()]))
        objective = (
            mean_mae
            + robustness_weight * (worst_mae - mean_mae)
            + harm_penalty * worst_harm
            + correction_penalty * mean_correction
        )
        fallback = alpha_up == 0 and alpha_down == 0 and alpha_boundary == 0
        lcb = stats['correction_precision_wilson_lcb']
        feasible = fallback or (
            stats['correction_count'] >= int(min_selected)
            and stats['correction_rate'] >= float(min_coverage)
            and lcb is not None
            and lcb >= float(min_wilson_lcb)
        )
        row = {
            **policy,
            'objective': objective,
            'scenario_mean_mae': mean_mae,
            'scenario_worst_mae': worst_mae,
            'scenario_worst_harm_005': worst_harm,
            'feasible': bool(feasible),
            **stats,
        }
        rows.append(row)
        if feasible:
            feasible_rows.append(row)
    if not feasible_rows:
        raise RuntimeError('V3.3 policy calibration produced no feasible fallback policy.')
    best = min(
        feasible_rows,
        key=lambda row: (
            row['objective'], row['scenario_worst_mae'], row['harm_over_005_rate'],
            -(row['correction_precision_wilson_lcb'] or 0.0), row['correction_rate'],
        ),
    )
    policy_keys = (
        'interval_margin', 'max_interval_width', 'region_threshold',
        'boundary_threshold', 'alpha_up', 'alpha_down', 'alpha_boundary',
    )
    return {key: float(best[key]) for key in policy_keys}, rows


def calibrate_target_stacking(valid_outputs, valid_views, alpha_values=None):
    alpha_values = alpha_values or [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
    labels = valid_views[0]['labels']
    fusion = torch.stack([_fusion(view['expert_matrix']) for view in valid_views]).mean(0)
    rows = []
    for alpha in alpha_values:
        per_view = [
            (1.0 - float(alpha)) * _fusion(view['expert_matrix'])
            + float(alpha) * output['direct_target']
            for output, view in zip(valid_outputs, valid_views)
        ]
        prediction = torch.stack(per_view).mean(0)
        correction = prediction - fusion
        stats = selection_stats(fusion, prediction, labels, correction)
        scenario_values = _scenario_rows(fusion, prediction, labels, correction)
        mean_mae = float(np.mean([value['mae'] for value in scenario_values.values()]))
        worst_mae = float(np.max([value['mae'] for value in scenario_values.values()]))
        worst_harm = float(np.max([value['harm_005'] for value in scenario_values.values()]))
        objective = mean_mae + 0.50 * (worst_mae - mean_mae) + 0.35 * worst_harm
        rows.append({'alpha': float(alpha), 'objective': objective, **stats})
    best = min(rows, key=lambda row: (row['objective'], row['harm_over_005_rate'], row['mae']))
    return {'alpha': float(best['alpha'])}, rows


def apply_target_stacking(outputs, views, policy):
    alpha = float(policy['alpha'])
    fusion = torch.stack([_fusion(view['expert_matrix']) for view in views]).mean(0)
    prediction = torch.stack([
        (1.0 - alpha) * _fusion(view['expert_matrix']) + alpha * output['direct_target']
        for output, view in zip(outputs, views)
    ]).mean(0)
    return prediction, selection_stats(
        fusion, prediction, views[0]['labels'], prediction - fusion
    )
