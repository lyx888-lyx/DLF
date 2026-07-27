import copy
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .backbone_crossfit import _video_group_id
from .fixed_oof_anchor_v41 import build_fixed_oof_anchor_cache_v41
from .residual_policy_v5 import (
    apply_residual_policy,
    calibrate_residual_policy,
    ensemble_view_outputs,
    fit_ridge_baseline,
    per_view_diagnostics,
    quantile_diagnostics,
    region_diagnostics,
    selection_stats,
)
from .shared_residual_v5 import (
    FoldFeatureNormalizer,
    SharedResidualQuantileNet,
    TrainingConfig,
    balanced_sign_accuracy,
    predict_model,
    prediction_metrics,
    safe_corr,
    train_one_model,
)

logger = logging.getLogger('MMSA')


def _safe_metrics(metrics_fn, prediction, labels):
    return {
        key: float(value)
        for key, value in metrics_fn(
            prediction.detach().cpu(), labels.detach().cpu()
        ).items()
    }


def build_context_feature(view):
    anchor = view['anchor'].float()
    return torch.cat([
        view['context'].float(),
        anchor,
        anchor.abs(),
        torch.sign(anchor),
        view['conflict_score'].float(),
    ], dim=1)


def group_validation_split(sample_ids, fold_ids, fraction, seed):
    rng = np.random.default_rng(int(seed))
    validation = []
    for fold in sorted(torch.unique(fold_ids.view(-1)).tolist()):
        fold_indices = torch.nonzero(
            fold_ids.view(-1) == int(fold), as_tuple=False
        ).view(-1)
        groups = {}
        for index in fold_indices.tolist():
            groups.setdefault(_video_group_id(sample_ids[index]), []).append(index)
        names = list(groups)
        rng.shuffle(names)
        target = max(1, int(round(len(fold_indices) * float(fraction))))
        count = 0
        for name in names:
            validation.extend(groups[name])
            count += len(groups[name])
            if count >= target:
                break
    validation = torch.tensor(sorted(set(validation)), dtype=torch.long)
    mask = torch.ones(len(sample_ids), dtype=torch.bool)
    mask[validation] = False
    training = torch.nonzero(mask, as_tuple=False).view(-1)
    if len(training) == 0 or len(validation) == 0:
        raise RuntimeError('Grouped residual validation split is empty.')
    return training, validation


def make_model(train_cache, fold_count, config):
    return SharedResidualQuantileNet(
        fusion_dim=train_cache['fusion_feature'].size(1),
        context_dim=build_context_feature(train_cache).size(1),
        fold_count=fold_count,
        adapter_dim=config['adapter_dim'],
        adapter_hidden_dim=config['adapter_hidden_dim'],
        context_hidden_dim=config['context_hidden_dim'],
        trunk_dim=config['trunk_dim'],
        dropout=config['dropout'],
        residual_max=config['residual_max'],
    )


def train_shared_residual_model(
    anchor_cache,
    device,
    save_dir,
    seed,
    selection_repeats,
    validation_fraction,
    model_config,
    training_config,
):
    train = anchor_cache['train']
    fold_count = int(anchor_cache['outer_folds'])
    raw_fusion = train['fusion_feature'].float()
    raw_context = build_context_feature(train)
    anchor = train['anchor'].float()
    labels = train['labels'].float()
    fold_ids = train['source_fold_ids'].long()
    sample_ids = train['sample_ids']
    selection_rows, split_rows, selected_epochs = [], [], []

    for repeat in range(int(selection_repeats)):
        train_indices, valid_indices = group_validation_split(
            sample_ids, fold_ids, validation_fraction, seed + 1709 * repeat
        )
        normalizer = FoldFeatureNormalizer(fold_count).fit(
            raw_fusion[train_indices], raw_context[train_indices], fold_ids[train_indices]
        )
        fusion, context = normalizer.transform(raw_fusion, raw_context, fold_ids)
        torch.manual_seed(int(seed + 5003 * repeat))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed + 5003 * repeat))
        model = make_model(train, fold_count, model_config)
        model, history, best_epoch, best_objective = train_one_model(
            model,
            fusion,
            context,
            anchor,
            labels,
            fold_ids,
            train_indices,
            valid_indices,
            device,
            training_config,
        )
        selected_epochs.append(int(best_epoch))
        selection_rows.extend({'repeat': repeat + 1, **row} for row in history)
        output = predict_model(
            model,
            fusion[valid_indices],
            context[valid_indices],
            anchor[valid_indices],
            labels[valid_indices],
            fold_ids[valid_indices],
            device,
            batch_size=training_config.batch_size,
        )
        metrics = prediction_metrics(
            output, anchor[valid_indices], labels[valid_indices], fold_ids[valid_indices]
        )
        split_rows.append({
            'repeat': repeat + 1,
            'train_count': int(len(train_indices)),
            'valid_count': int(len(valid_indices)),
            'selected_epoch': int(best_epoch),
            'best_objective': float(best_objective),
            **metrics,
        })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    selected_epoch = max(1, int(np.median(selected_epochs)))
    normalizer = FoldFeatureNormalizer(fold_count).fit(raw_fusion, raw_context, fold_ids)
    fusion, context = normalizer.transform(raw_fusion, raw_context, fold_ids)
    final_config = copy.deepcopy(training_config)
    final_config.epochs = selected_epoch
    torch.manual_seed(int(seed + 99173))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed + 99173))
    model = make_model(train, fold_count, model_config)
    model, final_history, _, _ = train_one_model(
        model,
        fusion,
        context,
        anchor,
        labels,
        fold_ids,
        torch.arange(len(labels)),
        None,
        device,
        final_config,
    )
    model = model.to('cpu')
    pd.DataFrame(selection_rows).to_csv(
        Path(save_dir) / 'v5_residual_selection_history.csv', index=False
    )
    pd.DataFrame(split_rows).to_csv(
        Path(save_dir) / 'v5_internal_generalization.csv', index=False
    )
    pd.DataFrame(final_history).to_csv(
        Path(save_dir) / 'v5_residual_final_history.csv', index=False
    )
    torch.save({
        'state_dict': model.state_dict(),
        'normalizer': normalizer.state_dict(),
        'model_config': model_config,
        'training_config': vars(final_config),
        'selected_epoch': selected_epoch,
    }, Path(save_dir) / 'shared_residual_v5_model.pth')
    return model, normalizer, selected_epoch, split_rows


def predict_views(model, normalizer, views, device, batch_size):
    model = model.to(device)
    outputs = []
    for fold_index, view in enumerate(views, start=1):
        fold_ids = torch.full((len(view['labels']),), fold_index, dtype=torch.long)
        fusion, context = normalizer.transform(
            view['fusion_feature'].float(), build_context_feature(view), fold_ids
        )
        output = predict_model(
            model,
            fusion,
            context,
            view['anchor'].float(),
            view['labels'].float(),
            fold_ids,
            device,
            batch_size=batch_size,
        )
        outputs.append({
            **output,
            'anchor': view['anchor'].float(),
            'labels': view['labels'].float(),
            'sample_ids': view['sample_ids'],
            'fold': fold_index,
        })
    return outputs


def run_fixed_anchor_residual_v5(
    args,
    metrics_fn,
    device,
    save_dir,
    backbone_dir,
    seed,
    num_workers=1,
    outer_folds=3,
    rebuild_anchor_cache=False,
    anchor_cache_dir=None,
    selection_repeats=3,
    validation_fraction=0.20,
    model_config=None,
    training_config=None,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_config = model_config or {
        'adapter_dim': 64,
        'adapter_hidden_dim': 96,
        'context_hidden_dim': 48,
        'trunk_dim': 128,
        'dropout': 0.20,
        'residual_max': 1.0,
    }
    training_config = training_config or TrainingConfig()
    anchor_cache_dir = Path(anchor_cache_dir) if anchor_cache_dir is not None else save_dir
    anchor_cache_dir.mkdir(parents=True, exist_ok=True)
    anchor_cache = build_fixed_oof_anchor_cache_v41(
        args=args,
        num_workers=num_workers,
        device=device,
        backbone_dir=backbone_dir,
        output_dir=anchor_cache_dir,
        seed=seed,
        outer_folds=outer_folds,
        rebuild=rebuild_anchor_cache,
    )
    model, normalizer, selected_epoch, internal_rows = train_shared_residual_model(
        anchor_cache,
        device,
        save_dir,
        seed,
        selection_repeats,
        validation_fraction,
        model_config,
        training_config,
    )
    valid_views = predict_views(
        model, normalizer, anchor_cache['valid_views'], device, training_config.batch_size
    )
    test_views = predict_views(
        model, normalizer, anchor_cache['test_views'], device, training_config.batch_size
    )
    valid_raw = ensemble_view_outputs(valid_views)
    test_raw = ensemble_view_outputs(test_views)
    best_policy, robust_policy, policy_rows = calibrate_residual_policy(valid_raw)
    pd.DataFrame(policy_rows).to_csv(
        save_dir / 'v5_residual_policy_calibration.csv', index=False
    )

    policies = {
        'full_residual': {'alpha': 1.0, 'coverage': 1.0},
        'best_valid': best_policy,
        'robust_valid': robust_policy,
    }
    anchor_metrics = _safe_metrics(metrics_fn, test_raw['anchor'], test_raw['labels'])
    rows = [{'model': 'fixed_anchor', **anchor_metrics}]
    results, regions = {}, []
    for name, policy in policies.items():
        result = apply_residual_policy(test_raw, policy)
        metrics = _safe_metrics(metrics_fn, result['prediction'], test_raw['labels'])
        stats = selection_stats(test_raw['anchor'], result['prediction'], test_raw['labels'])
        results[name] = {'policy': policy, 'metrics': metrics, **stats}
        rows.append({'model': name, **metrics, **stats})
        regions.extend(region_diagnostics(
            name, test_raw['anchor'], result['prediction'], test_raw['labels']
        ))

    ridge_best, ridge_prediction, ridge_rows = fit_ridge_baseline(
        anchor_cache, normalizer, valid_raw, test_raw, build_context_feature
    )
    pd.DataFrame(ridge_rows).to_csv(save_dir / 'v5_ridge_calibration.csv', index=False)
    ridge_metrics = _safe_metrics(metrics_fn, ridge_prediction, test_raw['labels'])
    ridge_stats = selection_stats(test_raw['anchor'], ridge_prediction, test_raw['labels'])
    results['ridge'] = {
        'policy': {'ridge': ridge_best['ridge'], 'alpha': ridge_best['alpha']},
        'metrics': ridge_metrics,
        **ridge_stats,
    }
    rows.append({'model': 'ridge', **ridge_metrics, **ridge_stats})
    regions.extend(region_diagnostics(
        'ridge', test_raw['anchor'], ridge_prediction, test_raw['labels']
    ))
    pd.DataFrame(rows).to_csv(save_dir / 'v5_test_baseline_comparison.csv', index=False)
    pd.DataFrame(regions).to_csv(save_dir / 'v5_test_region_diagnostics.csv', index=False)
    pd.DataFrame(quantile_diagnostics(test_raw)).to_csv(
        save_dir / 'v5_test_quantile_diagnostics.csv', index=False
    )
    pd.DataFrame(per_view_diagnostics(test_views)).to_csv(
        save_dir / 'v5_test_per_view_diagnostics.csv', index=False
    )

    main_result = apply_residual_policy(test_raw, robust_policy)
    prediction_data = {
        'sample_id': test_raw['sample_ids'],
        'target': test_raw['labels'].view(-1).numpy(),
        'anchor': test_raw['anchor'].view(-1).numpy(),
        'predicted_residual': test_raw['residual'].view(-1).numpy(),
        'q10': test_raw['q10'].view(-1).numpy(),
        'q90': test_raw['q90'].view(-1).numpy(),
        'uncertainty': test_raw['uncertainty'].view(-1).numpy(),
        'view_disagreement': test_raw['disagreement'].view(-1).numpy(),
        'sign_probability': test_raw['sign_probability'].view(-1).numpy(),
        'v5_prediction': main_result['prediction'].view(-1).numpy(),
        'v5_correction': main_result['correction'].view(-1).numpy(),
        'v5_selected': main_result['selected'].numpy(),
        'ridge_prediction': ridge_prediction.view(-1).numpy(),
    }
    for index in range(len(test_views)):
        prediction_data[f'anchor_view_{index + 1}'] = test_raw['anchors'][index].view(-1).numpy()
        prediction_data[f'residual_view_{index + 1}'] = test_raw['view_q50'][index].view(-1).numpy()
        prediction_data[f'width_view_{index + 1}'] = test_raw['view_widths'][index].view(-1).numpy()
    pd.DataFrame(prediction_data).to_csv(
        save_dir / 'fixed_anchor_residual_v5_predictions.csv', index=False
    )

    residual_target = test_raw['labels'] - test_raw['anchor']
    oracle_prediction = test_raw['anchor'] + residual_target.clamp(
        min=-model_config['residual_max'], max=model_config['residual_max']
    )
    oracle_metrics = _safe_metrics(metrics_fn, oracle_prediction, test_raw['labels'])
    main_metrics = results['robust_valid']['metrics']
    oracle_space = anchor_metrics['MAE'] - oracle_metrics['MAE']
    recovered = anchor_metrics['MAE'] - main_metrics['MAE']
    summary = {
        'method': 'fixed_anchor_shared_continuous_residual_v5',
        'seed': int(seed),
        'anchor_source': 'frozen_v31_outer_fold_oof',
        'backbone_trainable': False,
        'shared_trunk': True,
        'fold_specific_adapters': int(outer_folds),
        'selected_epoch': int(selected_epoch),
        'model_config': model_config,
        'training_config': vars(training_config),
        'best_valid_policy': best_policy,
        'robust_valid_policy': robust_policy,
        'fixed_anchor_metrics': anchor_metrics,
        'inference_results': results,
        'bounded_residual_oracle_metrics': oracle_metrics,
        'residual_diagnostics': {
            'residual_mae': float(torch.abs(test_raw['residual'] - residual_target).mean().item()),
            'residual_corr': safe_corr(test_raw['residual'], residual_target),
            'sign_balanced_accuracy': balanced_sign_accuracy(test_raw['residual'], residual_target),
            'interval_coverage': float(((residual_target >= test_raw['q10']) & (residual_target <= test_raw['q90'])).float().mean().item()),
            'mean_interval_width': float(test_raw['width'].mean().item()),
            'mean_view_disagreement': float(test_raw['disagreement'].mean().item()),
        },
        'internal_generalization_mean': {
            key: float(np.mean([row[key] for row in internal_rows]))
            for key in internal_rows[0] if key != 'repeat'
        },
        'oracle_gap_recovery_ratio': recovered / oracle_space if oracle_space > 1e-12 else 0.0,
    }
    with open(save_dir / 'fixed_anchor_residual_v5_summary.json', 'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
    logger.info(
        'V5 TEST anchor_MAE=%.4f full_MAE=%.4f robust_MAE=%.4f ridge_MAE=%.4f '
        'res_corr=%.4f sign_bal=%.4f coverage=%.4f recovery=%.4f policy=%s',
        anchor_metrics['MAE'],
        results['full_residual']['metrics']['MAE'],
        results['robust_valid']['metrics']['MAE'],
        ridge_metrics['MAE'],
        summary['residual_diagnostics']['residual_corr'],
        summary['residual_diagnostics']['sign_balanced_accuracy'],
        summary['residual_diagnostics']['interval_coverage'],
        summary['oracle_gap_recovery_ratio'],
        robust_policy,
    )
    return summary
