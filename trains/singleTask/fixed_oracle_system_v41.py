import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .backbone_crossfit import _video_group_id
from .fixed_oof_anchor_v41 import build_fixed_oof_anchor_cache_v41
from .fixed_oracle_student_v41 import (
    fixed_oracle_teacher,
    train_fixed_oracle_students_v41,
)


logger = logging.getLogger('MMSA')
REGION_NAMES = (
    'strong_negative',
    'negative',
    'boundary',
    'positive',
    'strong_positive',
)


def _safe_metrics(metrics_fn, prediction, labels):
    result = metrics_fn(prediction.detach().cpu(), labels.detach().cpu())
    return {key: float(value) for key, value in result.items()}


def _region_index(labels):
    values = labels.view(-1)
    result = torch.full_like(values, 2, dtype=torch.long)
    result[values < -1.5] = 0
    result[(values >= -1.5) & (values < -0.5)] = 1
    result[(values > 0.5) & (values <= 1.5)] = 3
    result[values > 1.5] = 4
    return result


def aggregate_fixed_views(outputs, views, offsets):
    if len(outputs) != len(views):
        raise ValueError('Student outputs and anchor views must have equal lengths.')
    labels = views[0]['labels']
    sample_ids = views[0]['sample_ids']
    anchors = torch.stack([view['anchor'] for view in views], dim=0)
    logits = torch.stack([output['logits'] for output in outputs], dim=0)
    ensemble_anchor = anchors.mean(dim=0)
    offset_tensor = torch.as_tensor(offsets, dtype=ensemble_anchor.dtype).view(1, -1)
    return {
        'view_logits': logits,
        'view_anchors': anchors,
        'anchor': ensemble_anchor,
        'candidate_values': ensemble_anchor + offset_tensor,
        'labels': labels,
        'sample_ids': sample_ids,
        'offsets': tuple(float(value) for value in offsets),
    }


def _ensemble_probabilities(raw, temperature):
    view_probabilities = F.softmax(
        raw['view_logits'] / max(float(temperature), 1e-4),
        dim=2,
    )
    probabilities = view_probabilities.mean(dim=0)
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return probabilities, view_probabilities


def _entropy(probabilities):
    return -(
        probabilities * probabilities.clamp_min(1e-8).log()
    ).sum(dim=1)


def _weighted_median_index(probabilities):
    return (probabilities.cumsum(dim=1) < 0.5).sum(dim=1).clamp(
        max=probabilities.size(1) - 1
    )


def apply_fixed_inference_policy(raw, policy):
    probabilities, view_probabilities = _ensemble_probabilities(
        raw, policy.get('temperature', 1.0)
    )
    candidates = raw['candidate_values']
    mode = policy.get('mode', 'soft')
    if mode == 'hard':
        selected_index = probabilities.argmax(dim=1)
        raw_prediction = candidates.gather(1, selected_index.view(-1, 1))
    elif mode == 'median':
        selected_index = _weighted_median_index(probabilities)
        raw_prediction = candidates.gather(1, selected_index.view(-1, 1))
    elif mode == 'soft':
        selected_index = probabilities.argmax(dim=1)
        raw_prediction = (probabilities * candidates).sum(dim=1, keepdim=True)
    else:
        raise ValueError(f'Unknown V4.1 inference mode: {mode}')

    anchor = raw['anchor']
    blend = float(policy.get('blend', 1.0))
    prediction = anchor + blend * (raw_prediction - anchor)
    entropy = _entropy(probabilities)
    top2 = torch.topk(probabilities, k=min(2, probabilities.size(1)), dim=1).values
    margin = (
        top2[:, 0] - top2[:, 1]
        if top2.size(1) > 1
        else top2[:, 0]
    )
    fallback = torch.zeros(len(anchor), dtype=torch.bool)
    entropy_threshold = policy.get('entropy_threshold')
    margin_threshold = policy.get('margin_threshold')
    if entropy_threshold is not None:
        fallback |= entropy > float(entropy_threshold)
    if margin_threshold is not None:
        fallback |= margin < float(margin_threshold)
    if fallback.any():
        prediction = prediction.clone()
        prediction[fallback] = anchor[fallback]

    return {
        'prediction': prediction,
        'correction': prediction - anchor,
        'probabilities': probabilities,
        'view_probabilities': view_probabilities,
        'entropy': entropy,
        'margin': margin,
        'fallback': fallback,
        'selected_index': selected_index,
    }


def selection_stats(anchor, prediction, labels):
    gain = (
        torch.abs(anchor - labels).view(-1)
        - torch.abs(prediction - labels).view(-1)
    )
    selected = (prediction - anchor).abs().view(-1) > 1e-8
    count = int(selected.sum().item())
    return {
        'mae': float(torch.abs(prediction - labels).mean().item()),
        'mean_realized_gain': float(gain.mean().item()),
        'selected_mean_gain': float(gain[selected].mean().item())
        if count else None,
        'correction_rate': float(selected.float().mean().item()),
        'correction_count': count,
        'correction_precision': float(
            (gain[selected] > 0).float().mean().item()
        ) if count else None,
        'harm_over_005_rate': float((gain < -0.05).float().mean().item()),
        'harm_over_010_rate': float((gain < -0.10).float().mean().item()),
        'mean_abs_correction': float(
            (prediction - anchor).abs().mean().item()
        ),
    }


def _group_half_masks(sample_ids):
    group_names = sorted({_video_group_id(value) for value in sample_ids})
    assignment = {
        name: (index % 2) for index, name in enumerate(group_names)
    }
    halves = torch.tensor(
        [assignment[_video_group_id(value)] for value in sample_ids],
        dtype=torch.long,
    )
    return [halves == 0, halves == 1]


def calibrate_fixed_inference_policy(raw):
    labels = raw['labels']
    fallback = {
        'mode': 'soft',
        'temperature': 1.0,
        'blend': 0.0,
        'entropy_threshold': None,
        'margin_threshold': None,
    }
    policies = [fallback]
    for mode, temperature, blend, entropy_threshold, margin_threshold in itertools.product(
        ('soft', 'median', 'hard'),
        (0.35, 0.50, 0.75, 1.00, 1.50, 2.00),
        (0.25, 0.50, 0.75, 1.00),
        (None, 1.20, 1.50, 1.80, 2.00),
        (None, 0.05, 0.10, 0.20, 0.30),
    ):
        policies.append({
            'mode': mode,
            'temperature': float(temperature),
            'blend': float(blend),
            'entropy_threshold': entropy_threshold,
            'margin_threshold': margin_threshold,
        })

    anchor_mae = float(torch.abs(raw['anchor'] - labels).mean().item())
    half_masks = _group_half_masks(raw['sample_ids'])
    rows = []
    for policy_index, policy in enumerate(policies):
        result = apply_fixed_inference_policy(raw, policy)
        stats = selection_stats(raw['anchor'], result['prediction'], labels)
        half_gains = []
        half_maes = []
        for mask in half_masks:
            half_anchor_mae = torch.abs(
                raw['anchor'][mask] - labels[mask]
            ).mean()
            half_mae = torch.abs(
                result['prediction'][mask] - labels[mask]
            ).mean()
            half_gains.append(float((half_anchor_mae - half_mae).item()))
            half_maes.append(float(half_mae.item()))
        objective = (
            stats['mae']
            + 0.25 * stats['harm_over_010_rate']
            + 0.02 * stats['mean_abs_correction']
            + 0.25 * max(0.0, -min(half_gains))
        )
        robust_feasible = (
            float(policy['blend']) == 0.0
            or (
                stats['mean_realized_gain'] > 0
                and min(half_gains) >= -0.002
                and stats['harm_over_010_rate'] <= 0.05
            )
        )
        rows.append({
            'policy_index': int(policy_index),
            **policy,
            'anchor_mae': anchor_mae,
            'objective': float(objective),
            'half_1_gain': half_gains[0],
            'half_2_gain': half_gains[1],
            'half_worst_gain': min(half_gains),
            'half_1_mae': half_maes[0],
            'half_2_mae': half_maes[1],
            'robust_feasible': bool(robust_feasible),
            **stats,
        })

    best_valid = min(rows, key=lambda row: (
        row['mae'],
        row['harm_over_010_rate'],
        row['mean_abs_correction'],
    ))
    robust_rows = [row for row in rows if row['robust_feasible']]
    robust = min(robust_rows, key=lambda row: (
        row['objective'],
        row['mae'],
        row['harm_over_010_rate'],
    ))
    fields = (
        'mode',
        'temperature',
        'blend',
        'entropy_threshold',
        'margin_threshold',
    )
    return (
        {key: best_valid[key] for key in fields},
        {key: robust[key] for key in fields},
        rows,
    )


def candidate_diagnostics(raw, probabilities=None):
    labels = raw['labels']
    candidates = raw['candidate_values']
    if probabilities is None:
        probabilities, _ = _ensemble_probabilities(raw, 1.0)
    teacher = fixed_oracle_teacher(
        raw['anchor'], labels, raw['offsets']
    )
    oracle_index = teacher['oracle_index']
    hard_index = probabilities.argmax(dim=1)
    top2 = torch.topk(
        probabilities, k=min(2, probabilities.size(1)), dim=1
    ).indices
    oracle_prediction = candidates.gather(
        1, oracle_index.view(-1, 1)
    )
    hard_prediction = candidates.gather(
        1, hard_index.view(-1, 1)
    )
    anchor_error = torch.abs(raw['anchor'] - labels).view(-1)
    oracle_error = torch.abs(oracle_prediction - labels).view(-1)
    hard_error = torch.abs(hard_prediction - labels).view(-1)
    predicted_offset = (
        probabilities
        * torch.as_tensor(raw['offsets'], dtype=raw['anchor'].dtype).view(1, -1)
    ).sum(dim=1)
    target_offset = (labels - raw['anchor']).view(-1).clamp(
        min=min(raw['offsets']), max=max(raw['offsets'])
    )
    return {
        'oracle_prediction': oracle_prediction,
        'hard_prediction': hard_prediction,
        'oracle_index': oracle_index,
        'hard_index': hard_index,
        'candidate_top1_accuracy': float(
            (oracle_index == hard_index).float().mean().item()
        ),
        'candidate_top2_accuracy': float(
            (top2 == oracle_index.view(-1, 1)).any(dim=1).float().mean().item()
        ),
        'candidate_within_one_accuracy': float(
            ((oracle_index - hard_index).abs() <= 1).float().mean().item()
        ),
        'candidate_offset_mae': float(
            torch.abs(predicted_offset - target_offset).mean().item()
        ),
        'mean_candidate_regret': float(
            (hard_error - oracle_error).mean().item()
        ),
        'normalized_candidate_regret': float(
            (hard_error - oracle_error).clamp_min(0.0).sum().item()
            / (anchor_error - oracle_error).clamp_min(0.0).sum().clamp_min(1e-8).item()
        ),
    }


def _region_rows(policy_name, anchor, prediction, oracle, labels):
    regions = _region_index(labels)
    rows = []
    for index, name in enumerate(REGION_NAMES):
        mask = regions == index
        if not mask.any():
            continue
        rows.append({
            'policy': policy_name,
            'region': name,
            'count': int(mask.sum().item()),
            'anchor_mae': float(
                torch.abs(anchor[mask] - labels[mask]).mean().item()
            ),
            'routed_mae': float(
                torch.abs(prediction[mask] - labels[mask]).mean().item()
            ),
            'oracle_mae': float(
                torch.abs(oracle[mask] - labels[mask]).mean().item()
            ),
            'anchor_bias': float(
                (labels[mask] - anchor[mask]).mean().item()
            ),
            'routed_bias': float(
                (labels[mask] - prediction[mask]).mean().item()
            ),
        })
    return rows


def _per_view_rows(outputs, views, offsets, metrics_fn):
    rows = []
    for fold_index, (output, view) in enumerate(zip(outputs, views), start=1):
        raw = {
            'view_logits': output['logits'].unsqueeze(0),
            'view_anchors': view['anchor'].unsqueeze(0),
            'anchor': view['anchor'],
            'candidate_values': view['anchor'] + torch.as_tensor(
                offsets, dtype=view['anchor'].dtype
            ).view(1, -1),
            'labels': view['labels'],
            'sample_ids': view['sample_ids'],
            'offsets': tuple(float(value) for value in offsets),
        }
        diagnostics = candidate_diagnostics(raw)
        rows.append({
            'fold': int(fold_index),
            **_safe_metrics(metrics_fn, view['anchor'], view['labels']),
            **{
                key: value for key, value in diagnostics.items()
                if not torch.is_tensor(value)
            },
        })
    return rows


def run_fixed_oof_oracle_v41(
    args,
    metrics_fn,
    device,
    save_dir,
    backbone_dir,
    seed,
    num_workers=1,
    outer_folds=3,
    rebuild_anchor_cache=False,
    offsets=(-0.75, -0.50, -0.25, -0.10, 0.0, 0.10, 0.25, 0.50, 0.75),
    student_epochs=80,
    student_learning_rate=3e-4,
    student_hidden_dim=192,
    student_adapter_dim=128,
    candidate_dim=64,
    student_dropout=0.20,
    student_batch_size=128,
    teacher_temperature=0.10,
    student_temperature=1.0,
    distill_weight=1.0,
    expected_mae_weight=0.50,
    ranking_weight=0.20,
    ordinal_emd_weight=0.50,
    offset_weight=0.50,
    validation_fraction=0.20,
    student_patience=10,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    anchor_cache = build_fixed_oof_anchor_cache_v41(
        args=args,
        num_workers=num_workers,
        device=device,
        backbone_dir=backbone_dir,
        output_dir=save_dir,
        seed=seed,
        outer_folds=outer_folds,
        rebuild=rebuild_anchor_cache,
    )
    student_result = train_fixed_oracle_students_v41(
        anchor_cache=anchor_cache,
        device=device,
        save_dir=save_dir,
        seed=seed,
        offsets=offsets,
        epochs=student_epochs,
        learning_rate=student_learning_rate,
        hidden_dim=student_hidden_dim,
        adapter_dim=student_adapter_dim,
        candidate_dim=candidate_dim,
        dropout=student_dropout,
        batch_size=student_batch_size,
        teacher_temperature=teacher_temperature,
        student_temperature=student_temperature,
        distill_weight=distill_weight,
        expected_mae_weight=expected_mae_weight,
        ranking_weight=ranking_weight,
        ordinal_emd_weight=ordinal_emd_weight,
        offset_weight=offset_weight,
        validation_fraction=validation_fraction,
        patience=student_patience,
    )

    valid_raw = aggregate_fixed_views(
        student_result['valid_outputs'],
        anchor_cache['valid_views'],
        offsets,
    )
    test_raw = aggregate_fixed_views(
        student_result['test_outputs'],
        anchor_cache['test_views'],
        offsets,
    )
    best_policy, robust_policy, calibration_rows = calibrate_fixed_inference_policy(
        valid_raw
    )
    pd.DataFrame(calibration_rows).to_csv(
        save_dir / 'v41_inference_policy_calibration.csv', index=False
    )

    policies = {
        'raw_soft': {
            'mode': 'soft',
            'temperature': 1.0,
            'blend': 1.0,
            'entropy_threshold': None,
            'margin_threshold': None,
        },
        'best_valid': best_policy,
        'robust_valid': robust_policy,
    }
    anchor_metrics = _safe_metrics(
        metrics_fn, test_raw['anchor'], test_raw['labels']
    )
    base_diagnostics = candidate_diagnostics(test_raw)
    oracle_metrics = _safe_metrics(
        metrics_fn,
        base_diagnostics['oracle_prediction'],
        test_raw['labels'],
    )

    baseline_rows = [{'model': 'fixed_anchor', **anchor_metrics}]
    result_summaries = {}
    region_rows = []
    main_result = None
    for name, policy in policies.items():
        result = apply_fixed_inference_policy(test_raw, policy)
        metrics = _safe_metrics(
            metrics_fn, result['prediction'], test_raw['labels']
        )
        stats = selection_stats(
            test_raw['anchor'], result['prediction'], test_raw['labels']
        )
        result_summaries[name] = {
            'policy': policy,
            'metrics': metrics,
            **stats,
        }
        baseline_rows.append({'model': name, **metrics, **stats})
        region_rows.extend(_region_rows(
            name,
            test_raw['anchor'],
            result['prediction'],
            base_diagnostics['oracle_prediction'],
            test_raw['labels'],
        ))
        if name == 'robust_valid':
            main_result = result

    hard_metrics = _safe_metrics(
        metrics_fn,
        base_diagnostics['hard_prediction'],
        test_raw['labels'],
    )
    baseline_rows.extend([
        {'model': 'raw_hard_top1', **hard_metrics},
        {'model': 'fixed_codebook_oracle', **oracle_metrics},
    ])
    pd.DataFrame(baseline_rows).to_csv(
        save_dir / 'v41_test_baseline_comparison.csv', index=False
    )
    pd.DataFrame(region_rows).to_csv(
        save_dir / 'v41_test_region_diagnostics.csv', index=False
    )
    pd.DataFrame(_per_view_rows(
        student_result['test_outputs'],
        anchor_cache['test_views'],
        offsets,
        metrics_fn,
    )).to_csv(save_dir / 'v41_test_per_view_diagnostics.csv', index=False)

    if main_result is None:
        raise RuntimeError('V4.1 robust policy result was not produced.')
    main_diagnostics = candidate_diagnostics(
        test_raw, main_result['probabilities']
    )
    labels = test_raw['labels']
    prediction_data = {
        'sample_id': test_raw['sample_ids'],
        'target': labels.view(-1).numpy(),
        'anchor_prediction': test_raw['anchor'].view(-1).numpy(),
        'robust_prediction': main_result['prediction'].view(-1).numpy(),
        'robust_correction': main_result['correction'].view(-1).numpy(),
        'candidate_entropy': main_result['entropy'].numpy(),
        'candidate_margin': main_result['margin'].numpy(),
        'candidate_fallback': main_result['fallback'].numpy(),
        'oracle_candidate_index': base_diagnostics['oracle_index'].numpy(),
        'student_candidate_index': main_diagnostics['hard_index'].numpy(),
        'oracle_prediction': base_diagnostics['oracle_prediction'].view(-1).numpy(),
    }
    for index, offset in enumerate(offsets):
        token = str(float(offset)).replace('-', 'm').replace('.', 'p')
        prediction_data[f'candidate_{token}'] = (
            test_raw['candidate_values'][:, index].numpy()
        )
        prediction_data[f'probability_{token}'] = (
            main_result['probabilities'][:, index].numpy()
        )
    pd.DataFrame(prediction_data).to_csv(
        save_dir / 'fixed_oof_oracle_v41_predictions.csv', index=False
    )

    confusion = torch.zeros(
        len(offsets), len(offsets), dtype=torch.long
    )
    for target_index, predicted_index in zip(
        base_diagnostics['oracle_index'].tolist(),
        main_diagnostics['hard_index'].tolist(),
    ):
        confusion[int(target_index), int(predicted_index)] += 1
    pd.DataFrame(
        confusion.numpy(),
        index=[f'oracle_{value:g}' for value in offsets],
        columns=[f'pred_{value:g}' for value in offsets],
    ).to_csv(save_dir / 'v41_candidate_confusion.csv')

    robust_metrics = result_summaries['robust_valid']['metrics']
    oracle_space = anchor_metrics['MAE'] - oracle_metrics['MAE']
    recovered = anchor_metrics['MAE'] - robust_metrics['MAE']
    internal_rows = student_result['internal_rows']
    summary = {
        'method': 'fixed_oof_oracle_distillation_v41',
        'seed': int(seed),
        'offsets': [float(value) for value in offsets],
        'anchor_source': 'frozen_v31_outer_fold_oof',
        'backbone_trainable': False,
        'teacher_fixed': True,
        'outer_folds': int(outer_folds),
        'best_valid_policy': best_policy,
        'robust_valid_policy': robust_policy,
        'fixed_anchor_metrics': anchor_metrics,
        'inference_results': result_summaries,
        'raw_hard_top1_metrics': hard_metrics,
        'fixed_codebook_oracle_metrics': oracle_metrics,
        'candidate_top1_accuracy': main_diagnostics['candidate_top1_accuracy'],
        'candidate_top2_accuracy': main_diagnostics['candidate_top2_accuracy'],
        'candidate_within_one_accuracy': main_diagnostics[
            'candidate_within_one_accuracy'
        ],
        'candidate_offset_mae': main_diagnostics['candidate_offset_mae'],
        'mean_candidate_regret': main_diagnostics['mean_candidate_regret'],
        'normalized_candidate_regret': main_diagnostics[
            'normalized_candidate_regret'
        ],
        'internal_generalization_mean': {
            key: float(np.mean([row[key] for row in internal_rows]))
            for key in (
                'soft_mae',
                'hard_mae',
                'anchor_mae',
                'oracle_mae',
                'top1_accuracy',
                'top2_accuracy',
                'within_one_accuracy',
                'offset_mae',
                'mean_regret',
                'normalized_regret',
            )
        },
        'oracle_gap_recovery_ratio': (
            recovered / oracle_space if oracle_space > 1e-12 else 0.0
        ),
    }
    with open(
        save_dir / 'fixed_oof_oracle_v41_summary.json',
        'w',
        encoding='utf-8',
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)

    logger.info(
        'V4.1 TEST anchor_MAE=%.4f robust_MAE=%.4f best_valid_MAE=%.4f '
        'oracle_MAE=%.4f top1=%.4f top2=%.4f within1=%.4f '
        'regret=%.4f recovery=%.4f robust_policy=%s',
        anchor_metrics['MAE'],
        robust_metrics['MAE'],
        result_summaries['best_valid']['metrics']['MAE'],
        oracle_metrics['MAE'],
        summary['candidate_top1_accuracy'],
        summary['candidate_top2_accuracy'],
        summary['candidate_within_one_accuracy'],
        summary['normalized_candidate_regret'],
        summary['oracle_gap_recovery_ratio'],
        robust_policy,
    )
    return summary
