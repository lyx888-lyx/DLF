import copy
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_loader import MMDataLoader
from utils import setup_seed
from .backbone_crossfit import (
    BASE_EXPERT_NAMES,
    REGION_NAMES,
    _predict_cache,
    _subset_loader,
    _video_group_id,
    build_prediction_context,
)
from .backbone_crossfit_full import FullFoldDLFTrainer
from .model import DLF as DLFModel

logger = logging.getLogger('MMSA')


def _region_index(labels):
    values = torch.as_tensor(labels, dtype=torch.float32).view(-1)
    result = torch.full_like(values, 2, dtype=torch.long)
    result[values < -1.5] = 0
    result[(values >= -1.5) & (values < -0.5)] = 1
    result[(values > 0.5) & (values <= 1.5)] = 3
    result[values > 1.5] = 4
    return result


def _global_fold_objective(fold_counts, fold_sizes, target_counts, target_size):
    count_scale = np.sqrt(target_counts + 1.0)
    region_term = np.square((fold_counts - target_counts[None, :]) / count_scale[None, :]).mean()
    size_term = np.square((fold_sizes - target_size) / math.sqrt(target_size + 1.0)).mean()
    nonzero = fold_sizes[fold_sizes > 0]
    ratio_term = 0.0
    if len(nonzero) > 1:
        ratio_term = max(0.0, float(nonzero.max() / nonzero.min()) - 1.25) ** 2
    return float(region_term + 0.35 * size_term + 2.0 * ratio_term)


def build_exact_group_stratified_folds(dataset, indices, fold_count, seed):
    """Build exactly ``fold_count`` non-empty, group-exclusive, region-balanced folds."""
    if fold_count < 2:
        raise ValueError('fold_count must be at least 2.')
    indices = [int(index) for index in indices]
    if len(indices) < fold_count:
        raise ValueError(f'Cannot split {len(indices)} samples into {fold_count} folds.')

    labels = torch.as_tensor(dataset.labels['M'][indices], dtype=torch.float32).view(-1)
    regions = _region_index(labels).tolist()
    groups = {}
    for local_index, sample_index in enumerate(indices):
        group_name = _video_group_id(dataset.ids[sample_index])
        record = groups.setdefault(
            group_name,
            {'indices': [], 'counts': np.zeros(len(REGION_NAMES), dtype=np.float64)},
        )
        record['indices'].append(sample_index)
        record['counts'][regions[local_index]] += 1.0

    if len(groups) < fold_count:
        raise RuntimeError(
            f'Only {len(groups)} video groups are available for {fold_count} requested folds.'
        )

    rng = np.random.default_rng(int(seed))
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(
        key=lambda item: (
            -len(item[1]['indices']),
            -float(np.max(item[1]['counts'])),
            item[0],
        )
    )

    total_counts = sum((record['counts'] for _, record in items), np.zeros(len(REGION_NAMES)))
    target_counts = total_counts / float(fold_count)
    target_size = len(indices) / float(fold_count)
    fold_counts = np.zeros((fold_count, len(REGION_NAMES)), dtype=np.float64)
    fold_sizes = np.zeros(fold_count, dtype=np.float64)
    fold_groups = [[] for _ in range(fold_count)]

    # Seed every fold with one large group. This prevents empty-fold collapse.
    for fold_index, (group_name, record) in enumerate(items[:fold_count]):
        fold_groups[fold_index].append((group_name, record['indices']))
        fold_counts[fold_index] += record['counts']
        fold_sizes[fold_index] += len(record['indices'])

    for group_name, record in items[fold_count:]:
        candidates = []
        for fold_index in range(fold_count):
            candidate_counts = fold_counts.copy()
            candidate_sizes = fold_sizes.copy()
            candidate_counts[fold_index] += record['counts']
            candidate_sizes[fold_index] += len(record['indices'])
            objective = _global_fold_objective(
                candidate_counts, candidate_sizes, target_counts, target_size
            )
            candidates.append((objective, candidate_sizes[fold_index], fold_index))
        _, _, best_fold = min(candidates)
        fold_groups[best_fold].append((group_name, record['indices']))
        fold_counts[best_fold] += record['counts']
        fold_sizes[best_fold] += len(record['indices'])

    folds = [
        torch.tensor(
            sorted(sample_index for _, values in entries for sample_index in values),
            dtype=torch.long,
        )
        for entries in fold_groups
    ]
    if len(folds) != fold_count or any(len(fold) == 0 for fold in folds):
        raise RuntimeError('Exact group split failed to produce all requested non-empty folds.')

    flattened = [int(index) for fold in folds for index in fold.tolist()]
    if len(flattened) != len(indices) or sorted(flattened) != sorted(indices):
        raise RuntimeError('Fold coverage check failed: samples are missing or duplicated.')

    group_to_fold = {}
    for fold_index, entries in enumerate(fold_groups):
        for group_name, _ in entries:
            if group_name in group_to_fold:
                raise RuntimeError(f'Video group {group_name} was assigned more than once.')
            group_to_fold[group_name] = fold_index

    positive_sizes = [len(fold) for fold in folds]
    size_ratio = max(positive_sizes) / max(1, min(positive_sizes))
    if size_ratio > 1.50:
        logger.warning(
            'Fold size ratio is %.3f (>1.50). Group constraints may limit balance.',
            size_ratio,
        )
    return folds


def _raw_cache(matrix, labels, sample_ids, fold_id):
    return {
        'expert_matrix': matrix.float(),
        'context_raw': build_prediction_context(matrix.float()),
        'conflict_score': matrix.float().std(dim=1, keepdim=True, unbiased=False),
        'labels': labels.float().view(-1, 1),
        'sample_ids': [str(value) for value in sample_ids],
        'fold_id': int(fold_id),
    }


def _fit_fold_normalization(train_raw, fold_ids, fold_count):
    stats = {}
    normalized = torch.zeros_like(train_raw['context_raw'])
    for fold_index in range(1, fold_count + 1):
        mask = fold_ids == fold_index
        if not mask.any():
            raise RuntimeError(f'No OOF samples were assigned to fold {fold_index}.')
        context = train_raw['context_raw'][mask]
        mean = context.mean(dim=0, keepdim=True)
        std = context.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-4)
        normalized[mask] = (context - mean) / std
        stats[fold_index] = {'mean': mean, 'std': std}
    train_cache = dict(train_raw)
    train_cache['context'] = normalized
    train_cache['source_fold_ids'] = fold_ids.clone()
    return train_cache, stats


def _normalize_view(raw_cache, stats, fold_index):
    result = dict(raw_cache)
    result['context'] = (
        raw_cache['context_raw'] - stats[fold_index]['mean']
    ) / stats[fold_index]['std']
    result['source_fold_ids'] = torch.full(
        (len(raw_cache['labels']),), int(fold_index), dtype=torch.long
    )
    return result


def build_backbone_crossfit_views_v31(
    args,
    num_workers,
    device,
    save_dir,
    seed,
    outer_folds=3,
    inner_folds=5,
    max_backbone_epochs=100,
    reuse_backbones=True,
):
    """Train exact outer-fold DLFs and return one valid/test view per backbone."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    full_loaders = MMDataLoader(args, num_workers)
    datasets = {name: loader.dataset for name, loader in full_loaders.items()}
    train_dataset = datasets['train']
    all_train_indices = list(range(len(train_dataset)))
    outer = build_exact_group_stratified_folds(
        train_dataset, all_train_indices, outer_folds, seed
    )
    if len(outer) != outer_folds:
        raise RuntimeError(
            f'Requested {outer_folds} outer folds but generated {len(outer)}.'
        )

    oof_matrix = torch.zeros(
        len(train_dataset), len(BASE_EXPERT_NAMES), dtype=torch.float32
    )
    train_fold_ids = torch.zeros(len(train_dataset), dtype=torch.long)
    valid_raw_views, test_raw_views = [], []
    fold_rows, assignment_rows, balance_rows = [], [], []
    all_index_tensor = torch.arange(len(train_dataset))

    for fold_index, holdout in enumerate(outer, start=1):
        train_fold_ids[holdout] = fold_index
        outer_mask = torch.ones(len(train_dataset), dtype=torch.bool)
        outer_mask[holdout] = False
        outer_train = all_index_tensor[outer_mask]
        inner = build_exact_group_stratified_folds(
            train_dataset,
            outer_train.tolist(),
            inner_folds,
            seed + 4099 * fold_index,
        )
        if len(inner) != inner_folds:
            raise RuntimeError(
                f'Requested {inner_folds} inner folds but generated {len(inner)}.'
            )
        inner_valid = inner[(fold_index - 1) % inner_folds]
        inner_valid_set = set(inner_valid.tolist())
        inner_train = torch.tensor(
            [index for index in outer_train.tolist() if index not in inner_valid_set],
            dtype=torch.long,
        )

        for sample_index in holdout.tolist():
            assignment_rows.append({
                'sample_index': int(sample_index),
                'sample_id': str(train_dataset.ids[sample_index]),
                'video_group': _video_group_id(train_dataset.ids[sample_index]),
                'label': float(train_dataset.labels['M'][sample_index]),
                'outer_fold': fold_index,
            })

        fold_args = copy.deepcopy(args)
        fold_seed = int(seed + 100003 * fold_index)
        fold_args['seed'] = fold_seed
        fold_args['cur_seed'] = fold_index
        fold_args['device'] = device
        checkpoint = save_dir / f'backbone_v31_fold_{fold_index}.pth'
        selection_checkpoint = save_dir / f'backbone_v31_fold_{fold_index}_selection.pth'
        metadata_path = save_dir / f'backbone_v31_fold_{fold_index}_metadata.json'

        setup_seed(fold_seed)
        model = getattr(DLFModel, 'DLF')(fold_args).to(device)
        if reuse_backbones and checkpoint.is_file():
            logger.info('Reusing V3.1 outer-fold checkpoint %s', checkpoint)
            model.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
                best_epoch = int(metadata.get('selected_epoch', -1))
                best_value = float(metadata.get('selection_value', float('nan')))
            else:
                best_epoch, best_value = -1, float('nan')
        else:
            selection_train_loader = _subset_loader(
                train_dataset, inner_train, args['batch_size'], num_workers, True
            )
            inner_valid_loader = _subset_loader(
                train_dataset, inner_valid, args['batch_size'], num_workers, False
            )
            logger.info(
                'V3.1 selecting fold %d/%d epoch: inner_train=%d inner_valid=%d '
                'outer_train=%d holdout=%d',
                fold_index, outer_folds, len(inner_train), len(inner_valid),
                len(outer_train), len(holdout),
            )
            _, best_epoch, best_value = FullFoldDLFTrainer(fold_args).train(
                model,
                selection_train_loader,
                inner_valid_loader,
                selection_checkpoint,
                max_epochs=max_backbone_epochs,
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            final_seed = fold_seed + 17
            setup_seed(final_seed)
            fold_args['seed'] = final_seed
            model = getattr(DLFModel, 'DLF')(fold_args).to(device)
            full_outer_loader = _subset_loader(
                train_dataset, outer_train, args['batch_size'], num_workers, True
            )
            model = FullFoldDLFTrainer(fold_args).train_fixed_epochs(
                model, full_outer_loader, max(1, best_epoch), checkpoint
            )
            metadata_path.write_text(
                json.dumps(
                    {
                        'selected_epoch': int(best_epoch),
                        'selection_value': float(best_value),
                        'fold_seed': fold_seed,
                    },
                    indent=2,
                ),
                encoding='utf-8',
            )

        holdout_loader = _subset_loader(
            train_dataset, holdout, args['batch_size'], num_workers, False
        )
        valid_loader = DataLoader(
            datasets['valid'],
            batch_size=args['batch_size'],
            num_workers=num_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            datasets['test'],
            batch_size=args['batch_size'],
            num_workers=num_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
        holdout_cache = _predict_cache(model, holdout_loader, device)
        valid_cache = _predict_cache(model, valid_loader, device)
        test_cache = _predict_cache(model, test_loader, device)
        oof_matrix[holdout] = holdout_cache['expert_matrix']

        valid_raw_views.append(
            _raw_cache(
                valid_cache['expert_matrix'],
                valid_cache['labels'],
                valid_cache['sample_ids'],
                fold_index,
            )
        )
        test_raw_views.append(
            _raw_cache(
                test_cache['expert_matrix'],
                test_cache['labels'],
                test_cache['sample_ids'],
                fold_index,
            )
        )

        fold_rows.append({
            'fold': fold_index,
            'fold_seed': fold_seed,
            'inner_train_count': len(inner_train),
            'inner_valid_count': len(inner_valid),
            'full_outer_train_count': len(outer_train),
            'holdout_count': len(holdout),
            'selected_epoch': int(best_epoch),
            'selection_value': float(best_value),
            'holdout_fusion_mae': float(
                torch.abs(
                    holdout_cache['expert_matrix'][:, -1:] - holdout_cache['labels']
                ).mean().item()
            ),
            'valid_fusion_mae': float(
                torch.abs(
                    valid_cache['expert_matrix'][:, -1:] - valid_cache['labels']
                ).mean().item()
            ),
            'test_fusion_mae': float(
                torch.abs(
                    test_cache['expert_matrix'][:, -1:] - test_cache['labels']
                ).mean().item()
            ),
        })

        holdout_regions = _region_index(
            train_dataset.labels['M'][holdout.numpy()]
        )
        for region_index, region_name in enumerate(REGION_NAMES):
            balance_rows.append({
                'level': 'outer',
                'fold': fold_index,
                'region': region_name,
                'count': int((holdout_regions == region_index).sum().item()),
                'fold_size': len(holdout),
            })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if (train_fold_ids == 0).any():
        raise RuntimeError('Some train samples did not receive an outer fold ID.')

    train_labels = torch.as_tensor(
        train_dataset.labels['M'], dtype=torch.float32
    ).view(-1, 1)
    train_raw = _raw_cache(
        oof_matrix,
        train_labels,
        [str(value) for value in train_dataset.ids],
        0,
    )
    train_cache, fold_norm = _fit_fold_normalization(
        train_raw, train_fold_ids, outer_folds
    )
    valid_views = [
        _normalize_view(view, fold_norm, fold_index)
        for fold_index, view in enumerate(valid_raw_views, start=1)
    ]
    test_views = [
        _normalize_view(view, fold_norm, fold_index)
        for fold_index, view in enumerate(test_raw_views, start=1)
    ]

    pd.DataFrame(fold_rows).to_csv(
        save_dir / 'backbone_v31_fold_history.csv', index=False
    )
    pd.DataFrame(assignment_rows).sort_values('sample_index').to_csv(
        save_dir / 'backbone_v31_oof_assignments.csv', index=False
    )
    pd.DataFrame(balance_rows).to_csv(
        save_dir / 'backbone_v31_fold_balance.csv', index=False
    )

    diagnostics = []
    train_fusion = train_cache['expert_matrix'][:, -1:]
    diagnostics.append({
        'split': 'train_oof',
        'view': 'oof',
        'count': len(train_labels),
        'fusion_mae': float(torch.abs(train_fusion - train_labels).mean().item()),
        'label_mean': float(train_labels.mean().item()),
        'prediction_mean': float(train_fusion.mean().item()),
        'target_minus_prediction_bias': float((train_labels - train_fusion).mean().item()),
    })
    for split_name, views in (('valid', valid_views), ('test', test_views)):
        ensemble_fusion = torch.stack(
            [view['expert_matrix'][:, -1:] for view in views], dim=0
        ).mean(dim=0)
        labels = views[0]['labels']
        for fold_index, view in enumerate(views, start=1):
            fusion = view['expert_matrix'][:, -1:]
            diagnostics.append({
                'split': split_name,
                'view': f'fold_{fold_index}',
                'count': len(labels),
                'fusion_mae': float(torch.abs(fusion - labels).mean().item()),
                'label_mean': float(labels.mean().item()),
                'prediction_mean': float(fusion.mean().item()),
                'target_minus_prediction_bias': float((labels - fusion).mean().item()),
            })
        diagnostics.append({
            'split': split_name,
            'view': 'ensemble',
            'count': len(labels),
            'fusion_mae': float(torch.abs(ensemble_fusion - labels).mean().item()),
            'label_mean': float(labels.mean().item()),
            'prediction_mean': float(ensemble_fusion.mean().item()),
            'target_minus_prediction_bias': float((labels - ensemble_fusion).mean().item()),
        })
    pd.DataFrame(diagnostics).to_csv(
        save_dir / 'backbone_v31_diagnostics.csv', index=False
    )
    torch.save(
        {
            'fold_normalization': fold_norm,
            'outer_folds': outer_folds,
            'context_dim': int(train_cache['context'].size(1)),
            'base_expert_names': list(BASE_EXPERT_NAMES),
        },
        save_dir / 'backbone_v31_metadata.pth',
    )
    return {
        'train': train_cache,
        'valid_views': valid_views,
        'test_views': test_views,
        'outer_folds': outer_folds,
    }
