import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_loader import MMDataLoader
from ..subNets import BertTextEncoder

logger = logging.getLogger('MMSA')


def _masked_stats(sequence):
    sequence = sequence.float()
    mask = torch.isfinite(sequence).all(dim=-1)
    mask = mask & (sequence.abs().sum(dim=-1) > 0)
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
            token_mask = text[:, 1, :].float()
            hidden = text_encoder(text)
            weights = token_mask.unsqueeze(-1)
            masked_mean = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            cls = hidden[:, 0, :]
            text_features = torch.cat([cls, masked_mean], dim=1).cpu()
            audio_features = _masked_stats(batch['audio'])
            vision_features = _masked_stats(batch['vision'])
            rows.append(torch.cat([text_features, audio_features, vision_features], dim=1))
    return torch.cat(rows, dim=0)


def _fit_projection(train_raw, output_dim, seed):
    mean = train_raw.mean(dim=0, keepdim=True)
    std = train_raw.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-5)
    standardized = (train_raw - mean) / std
    max_rank = min(standardized.size(0) - 1, standardized.size(1))
    rank = min(int(output_dim), int(max_rank))
    if rank < 1:
        raise RuntimeError('Semantic feature matrix has insufficient rank.')
    state = torch.random.get_rng_state()
    torch.manual_seed(int(seed))
    try:
        _, _, components = torch.pca_lowrank(
            standardized,
            q=rank,
            center=False,
            niter=4,
        )
    finally:
        torch.random.set_rng_state(state)
    return {
        'mean': mean,
        'std': std,
        'components': components[:, :rank].contiguous(),
        'output_dim': rank,
    }


def _project(raw, projection):
    return ((raw - projection['mean']) / projection['std']) @ projection['components']


def attach_aligned_semantic_features(
    args,
    caches,
    num_workers,
    device,
    save_dir,
    seed,
    semantic_dim=96,
    extraction_batch_size=16,
    reuse_features=True,
):
    """Attach one shared, frozen semantic representation to every backbone view.

    Text is encoded by a single pretrained frozen BERT, independent of all
    outer-fold DLF models. Audio and vision use deterministic temporal moments.
    The concatenated representation is standardized and projected using train
    data only, so fold-specific hidden coordinate systems are never mixed.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    feature_path = save_dir / 'semantic_v32_features.pth'
    if reuse_features and feature_path.is_file():
        payload = torch.load(feature_path, map_location='cpu')
        if int(payload.get('semantic_dim', -1)) == int(semantic_dim):
            logger.info('Reusing cached V3.2 semantic features from %s', feature_path)
            projected = payload['features']
            return _attach(caches, projected)

    loaders = MMDataLoader(args, num_workers)
    datasets = {name: loader.dataset for name, loader in loaders.items()}
    logger.info('Extracting shared frozen BERT/audio/vision semantic features.')
    text_encoder = BertTextEncoder(
        use_finetune=False,
        transformers=args.transformers,
        pretrained=args.pretrained,
    ).to(device)
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

    projection = _fit_projection(raw['train'], semantic_dim, seed)
    projected = {
        split: _project(values, projection).float()
        for split, values in raw.items()
    }
    torch.save(
        {
            'features': projected,
            'projection': projection,
            'raw_dim': int(raw['train'].size(1)),
            'semantic_dim': int(projected['train'].size(1)),
            'text_feature': 'shared_frozen_bert_cls_plus_masked_mean',
            'audio_visual_feature': 'masked_mean_std_min_max',
        },
        feature_path,
    )
    return _attach(caches, projected)


def _attach(caches, projected):
    if len(projected['train']) != len(caches['train']['labels']):
        raise RuntimeError('Train semantic feature order does not match OOF cache.')
    caches['train']['semantic'] = projected['train']
    for split, key in (('valid', 'valid_views'), ('test', 'test_views')):
        for view in caches[key]:
            if len(view['labels']) != len(projected[split]):
                raise RuntimeError(f'{split} semantic feature count mismatch.')
            view['semantic'] = projected[split]
    return caches
