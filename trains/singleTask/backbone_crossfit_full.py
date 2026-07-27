import copy
import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import MMDataLoader
from utils import setup_seed
from .backbone_crossfit import (
    BASE_EXPERT_NAMES,
    FoldDLFTrainer,
    _group_stratified_folds,
    _normalize_caches,
    _predict_cache,
    _subset_loader,
    _video_group_id,
    build_prediction_context,
)
from .model import DLF as DLFModel

logger = logging.getLogger('MMSA')


class FullFoldDLFTrainer(FoldDLFTrainer):
    """Select an epoch on inner validation, then retrain on the full outer fold."""

    def train_fixed_epochs(self, model, train_loader, epochs, checkpoint_path):
        optimizer = optim.Adam(model.parameters(), lr=self.args.learning_rate)
        update_epochs = max(1, int(self.args.update_epochs))
        total_epochs = max(1, int(epochs))
        for epoch in range(1, total_epochs + 1):
            model.train()
            optimizer.zero_grad()
            losses = []
            for step, batch in enumerate(tqdm(train_loader, leave=False), start=1):
                text = batch['text'].to(self.args.device)
                audio = batch['audio'].to(self.args.device)
                vision = batch['vision'].to(self.args.device)
                labels = batch['labels']['M'].to(self.args.device).view(-1, 1)
                output = model(text, audio, vision)
                loss = self._loss(output, labels)
                loss.backward()
                if self.args.grad_clip != -1.0:
                    nn.utils.clip_grad_value_(list(model.parameters()), self.args.grad_clip)
                if step % update_epochs == 0 or step == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()
                losses.append(loss.item())
            logger.info(
                'Backbone full-data epoch %d/%d train_loss=%.4f',
                epoch,
                total_epochs,
                sum(losses) / max(1, len(losses)),
            )
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        return model


def build_full_backbone_crossfit_caches(
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
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    full_loaders = MMDataLoader(args, num_workers)
    datasets = {name: loader.dataset for name, loader in full_loaders.items()}
    train_dataset = datasets['train']
    all_train_indices = list(range(len(train_dataset)))
    folds = _group_stratified_folds(train_dataset, all_train_indices, outer_folds, seed)

    oof_matrix = torch.zeros(len(train_dataset), len(BASE_EXPERT_NAMES), dtype=torch.float32)
    valid_matrices, test_matrices = [], []
    fold_rows, assignment_rows = [], []
    all_index_tensor = torch.arange(len(train_dataset))

    for fold_index, holdout in enumerate(folds, start=1):
        outer_mask = torch.ones(len(train_dataset), dtype=torch.bool)
        outer_mask[holdout] = False
        outer_train = all_index_tensor[outer_mask]
        inner = _group_stratified_folds(
            train_dataset,
            outer_train.tolist(),
            inner_folds,
            seed + 4099 * fold_index,
        )
        inner_valid = inner[(fold_index - 1) % len(inner)]
        inner_valid_set = set(inner_valid.tolist())
        inner_train = torch.tensor(
            [index for index in outer_train.tolist() if index not in inner_valid_set],
            dtype=torch.long,
        )
        for sample_index in holdout.tolist():
            assignment_rows.append({
                'sample_index': sample_index,
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
        checkpoint = save_dir / f'backbone_fold_{fold_index}.pth'
        selection_checkpoint = save_dir / f'backbone_fold_{fold_index}_selection.pth'

        setup_seed(fold_seed)
        model = getattr(DLFModel, 'DLF')(fold_args).to(device)
        if reuse_backbones and checkpoint.is_file():
            logger.info('Reusing full outer-fold checkpoint %s', checkpoint)
            model.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)
            best_epoch, best_value = -1, float('nan')
        else:
            selection_train_loader = _subset_loader(
                train_dataset, inner_train, args['batch_size'], num_workers, True
            )
            inner_valid_loader = _subset_loader(
                train_dataset, inner_valid, args['batch_size'], num_workers, False
            )
            logger.info(
                'Selecting fold %d/%d epoch: inner_train=%d inner_valid=%d holdout=%d',
                fold_index,
                len(folds),
                len(inner_train),
                len(inner_valid),
                len(holdout),
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
            logger.info(
                'Retraining fold %d on all %d outer-train samples for %d epochs.',
                fold_index,
                len(outer_train),
                best_epoch,
            )
            model = FullFoldDLFTrainer(fold_args).train_fixed_epochs(
                model,
                full_outer_loader,
                best_epoch,
                checkpoint,
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
        valid_matrices.append(valid_cache['expert_matrix'])
        test_matrices.append(test_cache['expert_matrix'])
        fold_rows.append({
            'fold': fold_index,
            'fold_seed': fold_seed,
            'inner_train_count': len(inner_train),
            'inner_valid_count': len(inner_valid),
            'full_outer_train_count': len(outer_train),
            'holdout_count': len(holdout),
            'selected_epoch': best_epoch,
            'selection_value': best_value,
            'holdout_fusion_mae': float(
                torch.abs(
                    holdout_cache['expert_matrix'][:, -1:] - holdout_cache['labels']
                ).mean().item()
            ),
        })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    valid_matrix = torch.stack(valid_matrices, dim=0).mean(dim=0)
    test_matrix = torch.stack(test_matrices, dim=0).mean(dim=0)
    caches = {
        'train': {
            'expert_matrix': oof_matrix,
            'context': build_prediction_context(oof_matrix),
            'conflict_score': oof_matrix.std(dim=1, keepdim=True, unbiased=False),
            'labels': torch.as_tensor(train_dataset.labels['M'], dtype=torch.float32).view(-1, 1),
            'sample_ids': [str(item) for item in train_dataset.ids],
        },
        'valid': {
            'expert_matrix': valid_matrix,
            'context': build_prediction_context(valid_matrix),
            'conflict_score': valid_matrix.std(dim=1, keepdim=True, unbiased=False),
            'labels': torch.as_tensor(datasets['valid'].labels['M'], dtype=torch.float32).view(-1, 1),
            'sample_ids': [str(item) for item in datasets['valid'].ids],
        },
        'test': {
            'expert_matrix': test_matrix,
            'context': build_prediction_context(test_matrix),
            'conflict_score': test_matrix.std(dim=1, keepdim=True, unbiased=False),
            'labels': torch.as_tensor(datasets['test'].labels['M'], dtype=torch.float32).view(-1, 1),
            'sample_ids': [str(item) for item in datasets['test'].ids],
        },
    }
    normalization = _normalize_caches(caches)
    pd.DataFrame(fold_rows).to_csv(save_dir / 'backbone_fold_history.csv', index=False)
    pd.DataFrame(assignment_rows).sort_values('sample_index').to_csv(
        save_dir / 'backbone_oof_assignments.csv', index=False
    )
    diagnostics = []
    for split, cache in caches.items():
        fusion = cache['expert_matrix'][:, -1:]
        diagnostics.append({
            'split': split,
            'count': len(cache['labels']),
            'fusion_mae': float(torch.abs(fusion - cache['labels']).mean().item()),
            'label_mean': float(cache['labels'].mean().item()),
            'prediction_mean': float(fusion.mean().item()),
            'mean_bias_target_minus_prediction': float(
                (cache['labels'] - fusion).mean().item()
            ),
        })
    pd.DataFrame(diagnostics).to_csv(
        save_dir / 'backbone_crossfit_diagnostics.csv', index=False
    )
    torch.save({
        'normalization': normalization,
        'outer_folds': len(folds),
        'context_dim': int(caches['train']['context'].size(1)),
        'base_expert_names': list(BASE_EXPERT_NAMES),
        'full_outer_retrain': True,
    }, save_dir / 'backbone_crossfit_metadata.pth')
    return caches
