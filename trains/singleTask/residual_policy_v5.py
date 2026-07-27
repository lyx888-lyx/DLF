import itertools

import torch

from .backbone_crossfit import _video_group_id
from .shared_residual_v5 import REGION_NAMES, balanced_sign_accuracy, region_index, safe_corr


def ensemble_view_outputs(view_outputs):
    anchors = torch.stack([view['anchor'] for view in view_outputs], dim=0)
    q10 = torch.stack([view['q10'] for view in view_outputs], dim=0)
    q50 = torch.stack([view['q50'] for view in view_outputs], dim=0)
    q90 = torch.stack([view['q90'] for view in view_outputs], dim=0)
    widths = (q90 - q10).clamp_min(1e-4)
    weights = 1.0 / widths
    weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-8)
    residual = (weights * q50).sum(dim=0)
    lower = (weights * q10).sum(dim=0)
    upper = (weights * q90).sum(dim=0)
    anchor = anchors.mean(dim=0)
    corrected_views = anchors + q50
    disagreement = corrected_views.std(dim=0, unbiased=False)
    uncertainty = widths.mean(dim=0) + disagreement
    sign_probability = torch.stack([
        torch.sigmoid(view['sign_logit']) for view in view_outputs
    ], dim=0).mean(dim=0)
    return {
        'anchor': anchor,
        'labels': view_outputs[0]['labels'],
        'sample_ids': view_outputs[0]['sample_ids'],
        'residual': residual,
        'q10': lower,
        'q90': upper,
        'width': upper - lower,
        'disagreement': disagreement,
        'uncertainty': uncertainty,
        'sign_probability': sign_probability,
        'anchors': anchors,
        'view_q50': q50,
        'view_widths': widths,
        'corrected_views': corrected_views,
        'weights': weights,
    }


def apply_residual_policy(raw, policy):
    alpha = float(policy.get('alpha', 1.0))
    coverage = float(policy.get('coverage', 1.0))
    count = len(raw['anchor'])
    selected = torch.zeros(count, dtype=torch.bool)
    if alpha > 0 and coverage > 0:
        selected_count = max(1, min(count, int(round(count * coverage))))
        ranking = torch.argsort(raw['uncertainty'].view(-1))
        selected[ranking[:selected_count]] = True
    correction = torch.zeros_like(raw['residual'])
    correction[selected] = alpha * raw['residual'][selected]
    prediction = raw['anchor'] + correction
    return {'prediction': prediction, 'correction': correction, 'selected': selected}


def selection_stats(anchor, prediction, labels):
    gain = torch.abs(anchor - labels).view(-1) - torch.abs(prediction - labels).view(-1)
    selected = (prediction - anchor).abs().view(-1) > 1e-8
    count = int(selected.sum().item())
    return {
        'mae': float(torch.abs(prediction - labels).mean().item()),
        'mean_realized_gain': float(gain.mean().item()),
        'selected_mean_gain': float(gain[selected].mean().item()) if count else None,
        'correction_rate': float(selected.float().mean().item()),
        'correction_count': count,
        'correction_precision': float((gain[selected] > 0).float().mean().item()) if count else None,
        'harm_over_005_rate': float((gain < -0.05).float().mean().item()),
        'harm_over_010_rate': float((gain < -0.10).float().mean().item()),
        'mean_abs_correction': float((prediction - anchor).abs().mean().item()),
    }


def _group_half_masks(sample_ids):
    groups = sorted({_video_group_id(value) for value in sample_ids})
    mapping = {name: index % 2 for index, name in enumerate(groups)}
    halves = torch.tensor([mapping[_video_group_id(value)] for value in sample_ids])
    return [halves == 0, halves == 1]


def calibrate_residual_policy(raw):
    policies = [{'alpha': 0.0, 'coverage': 0.0}]
    policies.extend(
        {'alpha': alpha, 'coverage': coverage}
        for alpha, coverage in itertools.product(
            (0.25, 0.50, 0.75, 1.00), (0.10, 0.25, 0.50, 0.75, 1.00)
        )
    )
    halves = _group_half_masks(raw['sample_ids'])
    rows = []
    for index, policy in enumerate(policies):
        result = apply_residual_policy(raw, policy)
        stats = selection_stats(raw['anchor'], result['prediction'], raw['labels'])
        half_gains = []
        for mask in halves:
            anchor_mae = torch.abs(raw['anchor'][mask] - raw['labels'][mask]).mean()
            final_mae = torch.abs(result['prediction'][mask] - raw['labels'][mask]).mean()
            half_gains.append(float((anchor_mae - final_mae).item()))
        objective = (
            stats['mae']
            + 0.25 * stats['harm_over_010_rate']
            + 0.02 * stats['mean_abs_correction']
            + 0.25 * max(0.0, -min(half_gains))
        )
        robust = (
            float(policy['alpha']) == 0.0
            or (
                stats['mean_realized_gain'] > 0
                and min(half_gains) >= -0.002
                and stats['harm_over_010_rate'] <= 0.05
            )
        )
        rows.append({
            'policy_index': index,
            **policy,
            'objective': float(objective),
            'half_1_gain': half_gains[0],
            'half_2_gain': half_gains[1],
            'half_worst_gain': min(half_gains),
            'robust_feasible': bool(robust),
            **stats,
        })
    best = min(rows, key=lambda row: (row['mae'], row['harm_over_010_rate']))
    robust_rows = [row for row in rows if row['robust_feasible']]
    robust = min(robust_rows, key=lambda row: (row['objective'], row['mae']))
    return (
        {'alpha': best['alpha'], 'coverage': best['coverage']},
        {'alpha': robust['alpha'], 'coverage': robust['coverage']},
        rows,
    )


def fit_ridge_baseline(anchor_cache, normalizer, valid_raw, test_raw, context_builder):
    train = anchor_cache['train']
    fold_ids = train['source_fold_ids'].long()
    fusion, context = normalizer.transform(
        train['fusion_feature'].float(), context_builder(train), fold_ids
    )
    x = torch.cat([fusion, context], dim=1).double()
    x = torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], dim=1)
    target = (train['labels'] - train['anchor']).double()
    rows, best = [], None
    for ridge in (0.1, 1.0, 10.0, 100.0):
        identity = torch.eye(x.size(1), dtype=x.dtype)
        identity[-1, -1] = 0.0
        weight = torch.linalg.solve(x.T @ x + ridge * identity, x.T @ target)
        split_residuals = {}
        for split_name, views in (
            ('valid', anchor_cache['valid_views']), ('test', anchor_cache['test_views'])
        ):
            predictions = []
            for fold_index, view in enumerate(views, start=1):
                ids = torch.full((len(view['labels']),), fold_index, dtype=torch.long)
                feature, context = normalizer.transform(
                    view['fusion_feature'].float(), context_builder(view), ids
                )
                design = torch.cat([feature, context], dim=1).double()
                design = torch.cat([
                    design, torch.ones(len(design), 1, dtype=design.dtype)
                ], dim=1)
                predictions.append((design @ weight).float())
            split_residuals[split_name] = torch.stack(predictions).mean(dim=0)
        for alpha in (0.0, 0.25, 0.50, 1.0):
            prediction = valid_raw['anchor'] + alpha * split_residuals['valid']
            mae = float(torch.abs(prediction - valid_raw['labels']).mean().item())
            row = {'ridge': ridge, 'alpha': alpha, 'valid_mae': mae}
            rows.append(row)
            if best is None or mae < best['valid_mae']:
                best = {**row, 'test_residual': split_residuals['test']}
    test_prediction = test_raw['anchor'] + best['alpha'] * best['test_residual']
    return best, test_prediction, rows


def region_diagnostics(name, anchor, prediction, labels):
    regions = region_index(labels)
    rows = []
    for index, region_name in enumerate(REGION_NAMES):
        mask = regions == index
        if mask.any():
            rows.append({
                'model': name,
                'region': region_name,
                'count': int(mask.sum().item()),
                'anchor_mae': float(torch.abs(anchor[mask] - labels[mask]).mean().item()),
                'final_mae': float(torch.abs(prediction[mask] - labels[mask]).mean().item()),
                'anchor_bias': float((labels[mask] - anchor[mask]).mean().item()),
                'final_bias': float((labels[mask] - prediction[mask]).mean().item()),
            })
    return rows


def quantile_diagnostics(raw):
    residual = raw['labels'] - raw['anchor']
    covered = ((residual >= raw['q10']) & (residual <= raw['q90'])).view(-1)
    rows = [{
        'level': 'overall', 'group': 'all', 'count': len(residual),
        'coverage': float(covered.float().mean().item()),
        'mean_width': float(raw['width'].mean().item()),
    }]
    regions = region_index(raw['labels'])
    for index, name in enumerate(REGION_NAMES):
        mask = regions == index
        if mask.any():
            rows.append({
                'level': 'region', 'group': name, 'count': int(mask.sum().item()),
                'coverage': float(covered[mask].float().mean().item()),
                'mean_width': float(raw['width'][mask].mean().item()),
            })
    return rows


def per_view_diagnostics(view_outputs):
    rows = []
    for view in view_outputs:
        prediction = view['anchor'] + view['q50']
        residual = view['labels'] - view['anchor']
        rows.append({
            'fold': view['fold'],
            'anchor_mae': float(torch.abs(view['anchor'] - view['labels']).mean().item()),
            'corrected_mae': float(torch.abs(prediction - view['labels']).mean().item()),
            'residual_mae': float(torch.abs(view['q50'] - residual).mean().item()),
            'residual_corr': safe_corr(view['q50'], residual),
            'sign_balanced_accuracy': balanced_sign_accuracy(view['q50'], residual),
            'interval_coverage': float(((residual >= view['q10']) & (residual <= view['q90'])).float().mean().item()),
            'interval_width': float((view['q90'] - view['q10']).mean().item()),
        })
    return rows
