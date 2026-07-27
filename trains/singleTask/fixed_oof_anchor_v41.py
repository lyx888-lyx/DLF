import copy
import json
import logging
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import MMDataLoader
from utils import setup_seed
from .backbone_crossfit import (
    _stack_experts,
    _subset_loader,
    build_prediction_context,
)
from .backbone_crossfit_v31 import build_exact_group_stratified_folds
from .expert_analysis import normalize_batch_ids
from .model import DLF as DLFModel


logger = logging.getLogger('MMSA')


class _FusionFeatureCapture:
    def __init__(self, model):
        self.value = None
        self.handle = model.out_layer.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        del module
        self.value = inputs[0]

    def close(self):
        self.handle.remove()


@torch.no_grad()
def _collect_anchor_view(model, dataloader, device):
    model.eval()
    capture = _FusionFeatureCapture(model)
    features, matrices, labels, sample_ids = [], [], [], []
    try:
        for batch in tqdm(dataloader, leave=False):
            text = batch['text'].to(device)
            audio = batch['audio'].to(device)
            vision = batch['vision'].to(device)
            target = batch['labels']['M'].to(device).view(-1, 1)
            output = model(text, audio, vision)
            if capture.value is None:
                raise RuntimeError('DLF fusion feature hook did not receive a tensor.')
            features.append(capture.value.detach().cpu())
            matrices.append(_stack_experts(output).detach().cpu())
            labels.append(target.detach().cpu())
            sample_ids.extend(normalize_batch_ids(batch.get('id')))
    finally:
        capture.close()

    matrix = torch.cat(matrices, dim=0).float()
    fusion_feature = torch.cat(features, dim=0).float()
    context = build_prediction_context(matrix)
    anchor = matrix[:, -1:]
    conflict = matrix.std(dim=1, keepdim=True, unbiased=False)
    model_feature = torch.cat(
        [fusion_feature, context, anchor, conflict],
        dim=1,
    )
    return {
        'model_feature': model_feature,
        'fusion_feature': fusion_feature,
        'expert_matrix': matrix,
        'context': context,
        'anchor': anchor,
        'conflict_score': conflict,
        'labels': torch.cat(labels, dim=0).float(),
        'sample_ids': [str(value) for value in sample_ids],
    }


def _full_loader(dataset, batch_size, num_workers):
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )


def _cache_is_compatible(payload, seed, outer_folds, backbone_dir):
    return (
        int(payload.get('seed', -1)) == int(seed)
        and int(payload.get('outer_folds', -1)) == int(outer_folds)
        and str(payload.get('backbone_dir', '')) == str(Path(backbone_dir).resolve())
    )


def build_fixed_oof_anchor_cache_v41(
    args,
    num_workers,
    device,
    backbone_dir,
    output_dir,
    seed,
    outer_folds=3,
    rebuild=False,
):
    """Build fixed OOF anchors and hidden features from frozen V3.1 fold backbones.

    Train samples are evaluated only by the outer-fold backbone that did not train on
    them. Valid and test receive one frozen view per outer-fold backbone.
    """
    backbone_dir = Path(backbone_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / 'fixed_oof_anchor_v41_cache.pth'

    if cache_path.is_file() and not rebuild:
        payload = torch.load(cache_path, map_location='cpu')
        if _cache_is_compatible(payload, seed, outer_folds, backbone_dir):
            logger.info('Reusing fixed V4.1 OOF anchor cache %s', cache_path)
            return payload
        logger.warning('Ignoring incompatible V4.1 anchor cache %s', cache_path)

    loaders = MMDataLoader(args, num_workers)
    datasets = {name: loader.dataset for name, loader in loaders.items()}
    train_dataset = datasets['train']
    all_indices = list(range(len(train_dataset)))
    outer = build_exact_group_stratified_folds(
        train_dataset, all_indices, int(outer_folds), int(seed)
    )
    if len(outer) != int(outer_folds):
        raise RuntimeError(
            f'Expected {outer_folds} exact outer folds, received {len(outer)}.'
        )

    train_cache = None
    source_fold_ids = torch.zeros(len(train_dataset), dtype=torch.long)
    valid_views, test_views, fold_rows = [], [], []

    for fold_index, holdout in enumerate(outer, start=1):
        checkpoint = backbone_dir / f'backbone_v31_fold_{fold_index}.pth'
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f'Missing frozen V3.1 backbone: {checkpoint}. '
                'Run V3.1 first or pass --backbone-dir to the correct seed directory.'
            )

        fold_args = copy.deepcopy(args)
        fold_args['device'] = device
        fold_args['seed'] = int(seed + 100003 * fold_index + 17)
        fold_args['cur_seed'] = fold_index
        setup_seed(fold_args['seed'])
        model = getattr(DLFModel, 'DLF')(fold_args).to(device)
        state = torch.load(checkpoint, map_location=device)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            logger.warning(
                'V4.1 fold %d checkpoint mismatch missing=%d unexpected=%d',
                fold_index,
                len(incompatible.missing_keys),
                len(incompatible.unexpected_keys),
            )
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        holdout_loader = _subset_loader(
            train_dataset, holdout, args['batch_size'], num_workers, False
        )
        valid_loader = _full_loader(
            datasets['valid'], args['batch_size'], num_workers
        )
        test_loader = _full_loader(
            datasets['test'], args['batch_size'], num_workers
        )
        logger.info(
            'V4.1 collecting fixed anchor fold %d/%d: holdout=%d',
            fold_index, outer_folds, len(holdout),
        )
        holdout_view = _collect_anchor_view(model, holdout_loader, device)
        valid_view = _collect_anchor_view(model, valid_loader, device)
        test_view = _collect_anchor_view(model, test_loader, device)

        if train_cache is None:
            train_cache = {
                key: torch.zeros(
                    (len(train_dataset), value.size(1)), dtype=value.dtype
                )
                for key, value in holdout_view.items()
                if torch.is_tensor(value) and value.ndim == 2
            }
            train_cache['sample_ids'] = [''] * len(train_dataset)

        for key, value in holdout_view.items():
            if torch.is_tensor(value) and value.ndim == 2:
                train_cache[key][holdout] = value
        for position, sample_index in enumerate(holdout.tolist()):
            train_cache['sample_ids'][sample_index] = holdout_view['sample_ids'][position]
        source_fold_ids[holdout] = fold_index
        valid_view['source_fold_id'] = int(fold_index)
        test_view['source_fold_id'] = int(fold_index)
        valid_views.append(valid_view)
        test_views.append(test_view)

        fold_rows.append({
            'fold': int(fold_index),
            'holdout_count': int(len(holdout)),
            'checkpoint': str(checkpoint),
            'holdout_anchor_mae': float(
                torch.abs(holdout_view['anchor'] - holdout_view['labels']).mean().item()
            ),
            'valid_anchor_mae': float(
                torch.abs(valid_view['anchor'] - valid_view['labels']).mean().item()
            ),
            'test_anchor_mae': float(
                torch.abs(test_view['anchor'] - test_view['labels']).mean().item()
            ),
            'feature_dim': int(holdout_view['model_feature'].size(1)),
        })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if train_cache is None or (source_fold_ids == 0).any():
        raise RuntimeError('V4.1 fixed OOF cache is incomplete.')
    if any(not value for value in train_cache['sample_ids']):
        train_cache['sample_ids'] = [str(value) for value in train_dataset.ids]
    train_cache['source_fold_ids'] = source_fold_ids

    payload = {
        'seed': int(seed),
        'outer_folds': int(outer_folds),
        'backbone_dir': str(backbone_dir.resolve()),
        'feature_dim': int(train_cache['model_feature'].size(1)),
        'candidate_anchor': 'frozen_v31_oof',
        'train': train_cache,
        'valid_views': valid_views,
        'test_views': test_views,
    }
    torch.save(payload, cache_path)
    pd.DataFrame(fold_rows).to_csv(
        output_dir / 'v41_fixed_anchor_fold_diagnostics.csv', index=False
    )
    (output_dir / 'v41_fixed_anchor_metadata.json').write_text(
        json.dumps({
            'seed': int(seed),
            'outer_folds': int(outer_folds),
            'backbone_dir': str(backbone_dir.resolve()),
            'feature_dim': int(train_cache['model_feature'].size(1)),
            'cache_path': str(cache_path),
        }, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    logger.info(
        'Built fixed V4.1 anchor cache: train=%d valid=%d test=%d dim=%d',
        len(train_cache['labels']),
        len(valid_views[0]['labels']),
        len(test_views[0]['labels']),
        train_cache['model_feature'].size(1),
    )
    return payload
