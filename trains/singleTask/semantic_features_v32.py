import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_loader import MMDataLoader
from .BertTextEncoder_import_shim import load_frozen_text_encoder

logger = logging.getLogger('MMSA')


def _masked_stats(sequence, mask=None):
    sequence = sequence.float()
    if mask is None:
        mask = torch.isfinite(sequence).all(dim=-1)
        mask = mask & (sequence.abs().sum(dim=-1) > 0)
    mask = mask.bool()
    weights = mask.float().unsqueeze(-1)
    count = weights.sum(dim=1).clamp_min(1.0)
    mean = (sequence * weights).sum(dim=1) / count
    centered = (sequence - mean.unsqueeze(1)) * weights
    std = torch.sqrt(centered.pow(2).sum(dim=1) / count + 1e-8)
    positive_inf = torch.full_like(sequence, float('inf'))
    negative_inf = torch.full_like(sequence, -float('inf'))
    minimum = torch.where(weights.bool(), sequence, positive_inf).min(dim=1).values
    maximum = torch.where(weights.bool(), sequence, negative_inf).max(dim=1).values
    minimum = torch.where(torch.isfinite(minimum), minimum, torch.zeros_like(minimum))
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    return torch.cat([mean, std, minimum, maximum], dim=1)


def _extract_split_features(dataset, text_encoder, device, batch_size, num_workers):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    rows = []
    text_encoder.eval()
    with torch.no_grad():
        for batch in loader:
            text = batch['text'].to(device)
            audio = batch['audio'].float()
            vision = batch['vision'].float()
            token_mask = text[:, 1, :].float().to(device)
            hidden = text_encoder(text)
            weights = token_mask.unsqueeze(-1)
            text_mean = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            text_cls = hidden[:, 0, :]
            text_stats = torch.cat([text_cls, text_mean], dim=1).cpu()
            audio_stats = _masked_stats(audio)
            vision_stats = _masked_stats(vision)
            rows.append(torch.cat([text_stats, audio_stats, vision_stats], dim=1))
    return torch.cat(rows, dim=0)


def _fit_projection(train_raw, output_dim):
    mean = train_raw.mean(dim=0, keepdim=True)
    std = train_raw.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-5)
    standardized = (train_raw - mean) / std
    max_rank = min(standardized.size(0) - 1, standardized.size(1))
    rank = min(int(output_dim), int(max_rank))
    if rank < 1:
        raise RuntimeError('Semantic feature matrix has insufficient rank.')
    _, _, vh = torch.linalg.svd(standardized, full_matrices=False)
    components = vh[:rank].t().contiguous()
    return {'mean': mean, 'std': std, 'components': components, 'output_dim': rank}


def _project(raw, projection):
    return ((raw - projection['mean']) / projection['std']) @ projection['components']


def attach_aligned_semantic_features(
    args,
    caches,
    num_workers,
    device,
    save_dir,
    semantic_dim=96,
    extraction_batch_size=16,
):
    """Attach frozen-pretrained, fold-aligned semantic features to all caches.

    Text features come from a single frozen pretrained BERT shared by every
    backbone fold. Audio and visual features are deterministic temporal
    statistics. A train-only PCA projection controls dimensionality.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    loaders = MMDataLoader(args, num_workers)
    datasets = {name: loader.dataset for name, loader in loaders.items()}
    logger.info('Extracting shared frozen semantic features.')
    text_encoder = load_frozen_text_encoder(args).to(device)
    for parameter in text_encoder.parameters():
        parameter.requires_grad = False
    raw = {
        split: _extract_split_features(
            datasets[split], text_encoder, device,
            extraction_batch_size, num_workers,
        )
        for split in ('train', 'valid', 'test')
    }
    del text_encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    projection = _fit_projection(raw['train'], semantic_dim)
    projected = {split: _project(values, projection).float() for split, values in raw.items()}
    if len(projected['train']) != len(caches['train']['labels']):
        raise RuntimeError('Train semantic feature order does not match OOF cache.')
    caches['train']['semantic'] = projected['train']
    for split, key in (('valid', 'valid_views'), ('test', 'test_views')):
        for view in caches[key]:
            if len(view['labels']) != len(projected[split]):
                raise RuntimeError(f'{split} semantic feature count mismatch.')
            view['semantic'] = projected[split]

    torch.save(
        {
            'projection': projection,
            'raw_dim': int(raw['train'].size(1)),
            'semantic_dim': int(projected['train'].size(1)),
            'text_feature': 'frozen_bert_cls_plus_masked_mean',
            'audio_visual_feature': 'masked_mean_std_min_max',
        },
        save_dir / 'semantic_v32_metadata.pth',
    )
    return caches
