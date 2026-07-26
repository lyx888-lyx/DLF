"""Train the Conservative Expert Router on a frozen DLF checkpoint."""

import argparse
import logging
from pathlib import Path

import torch

from config import get_config_regression
from data_loader import MMDataLoader
from trains.singleTask.expert_router import train_router
from trains.singleTask.model import DLF
from trains.utils import MetricsTop
from utils import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a fusion-anchored conservative router for DLF experts.'
    )
    parser.add_argument('--dataset', choices=['mosi', 'mosei'], default='mosi')
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--config', type=str, default='./config/config.json')
    parser.add_argument('--save-root', type=str, default='./result/router')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--minimum-gain', type=float, default=0.03)
    return parser.parse_args()


def main():
    cli_args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger('MMSA')

    setup_seed(cli_args.seed)
    device = assign_gpu([cli_args.gpu])

    config_path = Path(cli_args.config)
    args = get_config_regression('DLF', cli_args.dataset, config_path)
    args['device'] = device
    args['train_mode'] = 'regression'
    args['feature_T'] = ''
    args['feature_A'] = ''
    args['feature_V'] = ''
    args['seed'] = cli_args.seed

    checkpoint = (
        Path(cli_args.checkpoint)
        if cli_args.checkpoint
        else Path('./pt') / f'DLF{cli_args.dataset}.pth'
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f'DLF checkpoint not found: {checkpoint}. '
            'Train DLF first or pass --checkpoint.'
        )

    dataloader = MMDataLoader(args, cli_args.num_workers)
    model = getattr(DLF, 'DLF')(args).to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    metrics_fn = MetricsTop('regression').getMetics(cli_args.dataset)
    save_dir = (
        Path(cli_args.save_root)
        / cli_args.dataset
        / f'seed_{cli_args.seed}'
    )

    logger.info('Loaded frozen DLF checkpoint from %s', checkpoint)
    logger.info('Router outputs will be saved to %s', save_dir)

    _, summary = train_router(
        model=model,
        dataloader=dataloader,
        metrics_fn=metrics_fn,
        device=device,
        save_dir=save_dir,
        seed=cli_args.seed,
        epochs=cli_args.epochs,
        patience=cli_args.patience,
        learning_rate=cli_args.learning_rate,
        hidden_dim=cli_args.hidden_dim,
        dropout=cli_args.dropout,
        minimum_gain=cli_args.minimum_gain,
    )

    logger.info('Fusion metrics: %s', summary['fusion_metrics'])
    logger.info('Router metrics: %s', summary['router_metrics'])
    logger.info('Oracle metrics: %s', summary['oracle_metrics'])


if __name__ == '__main__':
    main()
