import copy
import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data_loader import MMDataLoader
from utils import setup_seed
from .HingeLoss import HingeLoss
from .expert_analysis import extract_expert_logits, normalize_batch_ids
from .model import DLF as DLFModel
from ..utils import MetricsTop

logger = logging.getLogger('MMSA')
BASE_EXPERT_NAMES = ('text', 'audio', 'video', 'common', 'fusion')
REGION_NAMES = ('strong_negative', 'negative', 'boundary', 'positive', 'strong_positive')


class _MSE(nn.Module):
    def forward(self, pred, real):
        diff = real - pred
        return diff.pow(2).sum() / max(1, diff.numel())


def _region_index(labels):
    values = labels.view(-1)
    result = torch.full_like(values, 2, dtype=torch.long)
    result[values < -1.5] = 0
    result[(values >= -1.5) & (values < -0.5)] = 1
    result[(values > 0.5) & (values <= 1.5)] = 3
    result[values > 1.5] = 4
    return result


def _video_group_id(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='replace')
    text = str(value)
    if '[' in text:
        return text.split('[', 1)[0]
    if '$_$' in text:
        return text.split('$_$', 1)[0]
    match = re.match(r'^(.*?)(?:[_-](?:seg)?\d+)$', text, flags=re.IGNORECASE)
    return match.group(1) if match else text


def _group_stratified_folds(dataset, indices, fold_count, seed):
    indices = [int(index) for index in indices]
    labels = torch.as_tensor(dataset.labels['M'][indices], dtype=torch.float32).view(-1)
    regions = _region_index(labels).tolist()
    groups = {}
    for local_index, sample_index in enumerate(indices):
        group = _video_group_id(dataset.ids[sample_index])
        record = groups.setdefault(group, {'indices': [], 'counts': np.zeros(len(REGION_NAMES), dtype=np.float64)})
        record['indices'].append(sample_index)
        record['counts'][regions[local_index]] += 1.0

    rng = np.random.default_rng(int(seed))
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda item: len(item[1]['indices']), reverse=True)
    total_counts = sum((item[1]['counts'] for item in items), np.zeros(len(REGION_NAMES)))
    target_counts = total_counts / float(fold_count)
    target_size = len(indices) / float(fold_count)
    fold_counts = np.zeros((fold_count, len(REGION_NAMES)), dtype=np.float64)
    fold_sizes = np.zeros(fold_count, dtype=np.float64)
    fold_groups = [[] for _ in range(fold_count)]

    for group_name, record in items:
        scores = []
        group_size = len(record['indices'])
        for fold_index in range(fold_count):
            new_counts = fold_counts[fold_index] + record['counts']
            region_score = np.square((new_counts - target_counts) / np.sqrt(target_counts + 1.0)).sum()
            size_score = ((fold_sizes[fold_index] + group_size - target_size) / math.sqrt(target_size + 1.0)) ** 2
            scores.append(float(region_score + 0.20 * size_score))
        best_fold = min(range(fold_count), key=lambda index: (scores[index], fold_sizes[index]))
        fold_groups[best_fold].append((group_name, record['indices']))
        fold_counts[best_fold] += record['counts']
        fold_sizes[best_fold] += group_size

    folds = []
    for entries in fold_groups:
        fold_indices = sorted(index for _, values in entries for index in values)
        if fold_indices:
            folds.append(torch.tensor(fold_indices, dtype=torch.long))
    if len(folds) < 2:
        raise RuntimeError('Group-stratified splitting produced fewer than two folds.')
    return folds


def _subset_loader(dataset, indices, batch_size, num_workers, shuffle):
    return DataLoader(
        Subset(dataset, [int(index) for index in indices]),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )


def _stack_experts(output):
    experts = extract_expert_logits(output)
    return torch.cat([experts[name].view(-1, 1) for name in BASE_EXPERT_NAMES], dim=1)


def build_prediction_context(expert_matrix):
    fusion = expert_matrix[:, -1:]
    differences = expert_matrix[:, :-1] - fusion
    mean = expert_matrix.mean(dim=1, keepdim=True)
    std = expert_matrix.std(dim=1, keepdim=True, unbiased=False)
    minimum = expert_matrix.min(dim=1, keepdim=True).values
    maximum = expert_matrix.max(dim=1, keepdim=True).values
    span = maximum - minimum
    absolute = expert_matrix.abs()
    signed = torch.sign(expert_matrix)
    pairwise = []
    for left in range(expert_matrix.size(1)):
        for right in range(left + 1, expert_matrix.size(1)):
            pairwise.append(torch.abs(expert_matrix[:, left:left + 1] - expert_matrix[:, right:right + 1]))
    return torch.cat([
        expert_matrix, differences, mean, std, minimum, maximum, span,
        absolute, signed, torch.cat(pairwise, dim=1),
    ], dim=1)


def _predict_cache(model, dataloader, device):
    model.eval()
    matrices, labels, sample_ids = [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, leave=False):
            text = batch['text'].to(device)
            audio = batch['audio'].to(device)
            vision = batch['vision'].to(device)
            target = batch['labels']['M'].to(device).view(-1, 1)
            output = model(text, audio, vision)
            matrices.append(_stack_experts(output).cpu())
            labels.append(target.cpu())
            sample_ids.extend(normalize_batch_ids(batch.get('id')))
    matrix = torch.cat(matrices, dim=0)
    context = build_prediction_context(matrix)
    return {
        'expert_matrix': matrix,
        'context': context,
        'conflict_score': matrix.std(dim=1, keepdim=True, unbiased=False),
        'labels': torch.cat(labels, dim=0),
        'sample_ids': sample_ids,
    }


class FoldDLFTrainer:
    """Train one DLF fold without evaluating the official test split."""

    def __init__(self, args):
        self.args = args
        self.l1 = nn.L1Loss()
        self.cosine = nn.CosineEmbeddingLoss()
        self.mse = _MSE()
        self.sim_loss = HingeLoss()
        self.metrics = MetricsTop('regression').getMetics(args.dataset_name)

    def _loss(self, output, labels):
        task = (
            self.l1(output['output_logit'], labels)
            + self.l1(output['logits_c'], labels)
            + 3.0 * self.l1(output['logits_l_hetero'], labels)
            + self.l1(output['logits_v_hetero'], labels)
            + self.l1(output['logits_a_hetero'], labels)
        )
        reconstruction = (
            self.mse(output['recon_l'], output['origin_l'])
            + self.mse(output['recon_v'], output['origin_v'])
            + self.mse(output['recon_a'], output['origin_a'])
        )
        specific = (
            self.mse(output['s_l'].permute(1, 2, 0), output['s_l_r'])
            + self.mse(output['s_v'].permute(1, 2, 0), output['s_v_r'])
            + self.mse(output['s_a'].permute(1, 2, 0), output['s_a_r'])
        )
        feature_width = 50 if self.args.dataset_name == 'mosi' else 10
        negative_target = torch.full((1,), -1.0, device=labels.device)
        orthogonal = (
            self.cosine(output['s_l'].reshape(-1, feature_width), output['c_l'].reshape(-1, feature_width), negative_target)
            + self.cosine(output['s_v'].reshape(-1, feature_width), output['c_v'].reshape(-1, feature_width), negative_target)
            + self.cosine(output['s_a'].reshape(-1, feature_width), output['c_a'].reshape(-1, feature_width), negative_target)
        )
        features, ids = [], []
        for index in range(labels.size(0)):
            for value in (output['c_l_sim'][index], output['c_v_sim'][index], output['c_a_sim'][index]):
                features.append(value.view(1, -1))
                ids.append(labels[index].view(1, -1))
        similarity = self.sim_loss(torch.cat(ids, dim=0), torch.cat(features, dim=0))
        return task + 0.1 * (specific + reconstruction + 0.1 * (similarity + orthogonal))

    def evaluate(self, model, dataloader):
        model.eval()
        predictions, labels, losses = [], [], []
        with torch.no_grad():
            for batch in dataloader:
                text = batch['text'].to(self.args.device)
                audio = batch['audio'].to(self.args.device)
                vision = batch['vision'].to(self.args.device)
                target = batch['labels']['M'].to(self.args.device).view(-1, 1)
                output = model(text, audio, vision)
                losses.append(self.l1(output['output_logit'], target).item())
                predictions.append(output['output_logit'].cpu())
                labels.append(target.cpu())
        prediction, target = torch.cat(predictions), torch.cat(labels)
        result = self.metrics(prediction, target)
        result['Loss'] = float(np.mean(losses))
        return result

    def train(self, model, train_loader, valid_loader, checkpoint_path, max_epochs=100):
        optimizer = optim.Adam(model.parameters(), lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=self.args.patience
        )
        best_value, best_epoch = float('inf'), 0
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        update_epochs = max(1, int(self.args.update_epochs))

        for epoch in range(1, int(max_epochs) + 1):
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

            valid = self.evaluate(model, valid_loader)
            scheduler.step(valid['Loss'])
            logger.info(
                'Backbone epoch %d train_loss=%.4f valid_loss=%.4f valid_MAE=%.4f',
                epoch, float(np.mean(losses)), valid['Loss'], valid['MAE'],
            )
            value = valid['Loss'] if self.args.KeyEval == 'Loss' else -float(valid[self.args.KeyEval])
            if value < best_value - 1e-6:
                best_value, best_epoch = value, epoch
                torch.save(model.state_dict(), checkpoint_path)
            if epoch - best_epoch >= int(self.args.early_stop):
                break
        if not checkpoint_path.is_file():
            raise RuntimeError(f'Backbone fold did not save a checkpoint: {checkpoint_path}')
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.args.device), strict=False)
        return model, best_epoch, best_value


def _normalize_caches(caches):
    mean = caches['train']['context'].mean(dim=0, keepdim=True)
    std = caches['train']['context'].std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-4)
    for cache in caches.values():
        cache['context'] = (cache['context'] - mean) / std
    return {'context_mean': mean, 'context_std': std}


def build_backbone_crossfit_caches(
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
            train_dataset, outer_train.tolist(), inner_folds, seed + 4099 * fold_index
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
        setup_seed(fold_seed)
        checkpoint = save_dir / f'backbone_fold_{fold_index}.pth'
        train_loader = _subset_loader(
            train_dataset, inner_train, args['batch_size'], num_workers, True
        )
        inner_valid_loader = _subset_loader(
            train_dataset, inner_valid, args['batch_size'], num_workers, False
        )
        model = getattr(DLFModel, 'DLF')(fold_args).to(device)
        trainer = FoldDLFTrainer(fold_args)
        if reuse_backbones and checkpoint.is_file():
            logger.info('Reusing backbone fold checkpoint %s', checkpoint)
            model.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)
            best_epoch, best_value = -1, float('nan')
        else:
            logger.info(
                'Training backbone fold %d/%d: inner_train=%d inner_valid=%d holdout=%d',
                fold_index, len(folds), len(inner_train), len(inner_valid), len(holdout),
            )
            model, best_epoch, best_value = trainer.train(
                model, train_loader, inner_valid_loader, checkpoint, max_epochs=max_backbone_epochs
            )

        holdout_loader = _subset_loader(
            train_dataset, holdout, args['batch_size'], num_workers, False
        )
        valid_loader = DataLoader(
            datasets['valid'], batch_size=args['batch_size'], num_workers=num_workers,
            shuffle=False, pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            datasets['test'], batch_size=args['batch_size'], num_workers=num_workers,
            shuffle=False, pin_memory=torch.cuda.is_available(),
        )
        holdout_cache = _predict_cache(model, holdout_loader, device)
        valid_cache = _predict_cache(model, valid_loader, device)
        test_cache = _predict_cache(model, test_loader, device)
        oof_matrix[holdout] = holdout_cache['expert_matrix']
        valid_matrices.append(valid_cache['expert_matrix'])
        test_matrices.append(test_cache['expert_matrix'])
        holdout_mae = float(torch.abs(holdout_cache['expert_matrix'][:, -1:] - holdout_cache['labels']).mean())
        fold_rows.append({
            'fold': fold_index,
            'fold_seed': fold_seed,
            'inner_train_count': len(inner_train),
            'inner_valid_count': len(inner_valid),
            'holdout_count': len(holdout),
            'best_epoch': best_epoch,
            'best_value': best_value,
            'holdout_fusion_mae': holdout_mae,
        })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    valid_matrix = torch.stack(valid_matrices, dim=0).mean(dim=0)
    test_matrix = torch.stack(test_matrices, dim=0).mean(dim=0)
    train_labels = torch.as_tensor(train_dataset.labels['M'], dtype=torch.float32).view(-1, 1)
    valid_labels = torch.as_tensor(datasets['valid'].labels['M'], dtype=torch.float32).view(-1, 1)
    test_labels = torch.as_tensor(datasets['test'].labels['M'], dtype=torch.float32).view(-1, 1)
    caches = {
        'train': {
            'expert_matrix': oof_matrix,
            'context': build_prediction_context(oof_matrix),
            'conflict_score': oof_matrix.std(dim=1, keepdim=True, unbiased=False),
            'labels': train_labels,
            'sample_ids': normalize_batch_ids(train_dataset.ids),
        },
        'valid': {
            'expert_matrix': valid_matrix,
            'context': build_prediction_context(valid_matrix),
            'conflict_score': valid_matrix.std(dim=1, keepdim=True, unbiased=False),
            'labels': valid_labels,
            'sample_ids': normalize_batch_ids(datasets['valid'].ids),
        },
        'test': {
            'expert_matrix': test_matrix,
            'context': build_prediction_context(test_matrix),
            'conflict_score': test_matrix.std(dim=1, keepdim=True, unbiased=False),
            'labels': test_labels,
            'sample_ids': normalize_batch_ids(datasets['test'].ids),
        },
    }
    normalization = _normalize_caches(caches)
    pd.DataFrame(fold_rows).to_csv(save_dir / 'backbone_fold_history.csv', index=False)
    pd.DataFrame(assignment_rows).sort_values('sample_index').to_csv(
        save_dir / 'backbone_oof_assignments.csv', index=False
    )
    diagnostic_rows = []
    for split, cache in caches.items():
        fusion = cache['expert_matrix'][:, -1:]
        diagnostic_rows.append({
            'split': split,
            'count': len(cache['labels']),
            'fusion_mae': float(torch.abs(fusion - cache['labels']).mean().item()),
            'label_mean': float(cache['labels'].mean().item()),
            'prediction_mean': float(fusion.mean().item()),
            'mean_bias_target_minus_prediction': float((cache['labels'] - fusion).mean().item()),
        })
    pd.DataFrame(diagnostic_rows).to_csv(save_dir / 'backbone_crossfit_diagnostics.csv', index=False)
    torch.save(
        {
            'normalization': normalization,
            'outer_folds': len(folds),
            'context_dim': int(caches['train']['context'].size(1)),
            'base_expert_names': list(BASE_EXPERT_NAMES),
        },
        save_dir / 'backbone_crossfit_metadata.pth',
    )
    return caches
