"""Train ORD-DLF: Oracle-Regret Distilled Dynamic Label Fusion."""

import argparse
import logging
from pathlib import Path

import torch

from config import get_config_regression
from data_loader import MMDataLoader
from trains.singleTask.model.ORD_DLF import OracleRegretDLF
from trains.singleTask.oracle_regret_system_v4 import OracleRegretTrainerV4
from trains.utils import MetricsTop
from utils import assign_gpu, setup_seed


def parse_offsets(value):
    offsets = tuple(float(token.strip()) for token in value.split(',') if token.strip())
    if not offsets:
        raise argparse.ArgumentTypeError('At least one candidate offset is required.')
    if min(abs(item) for item in offsets) > 1e-8:
        offsets = tuple(sorted(offsets + (0.0,)))
    return offsets


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Jointly train DLF with a candidate-conditioned energy model using '
            'label-derived soft oracle distributions, expected MAE, and regret ranking.'
        )
    )
    parser.add_argument('--dataset', choices=['mosi', 'mosei'], default='mosi')
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--config', type=str, default='./config/config.json')
    parser.add_argument('--save-root', type=str, default='./result/oracle_regret_distilled_v4')
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-epochs', type=int, default=80)
    parser.add_argument('--early-stop', type=int, default=12)
    parser.add_argument(
        '--candidate-offsets', type=parse_offsets,
        default=(-0.75, -0.50, -0.25, -0.10, 0.0, 0.10, 0.25, 0.50, 0.75),
    )
    parser.add_argument('--router-hidden-dim', type=int, default=192)
    parser.add_argument('--candidate-dim', type=int, default=64)
    parser.add_argument('--router-dropout', type=float, default=0.20)
    parser.add_argument('--router-lr-multiplier', type=float, default=3.0)
    parser.add_argument('--teacher-temperature', type=float, default=0.10)
    parser.add_argument('--student-temperature', type=float, default=1.0)
    parser.add_argument('--distill-weight', type=float, default=0.60)
    parser.add_argument('--expected-mae-weight', type=float, default=0.40)
    parser.add_argument('--routed-mae-weight', type=float, default=1.00)
    parser.add_argument('--ranking-weight', type=float, default=0.20)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument(
        '--init-checkpoint', type=str, default='auto',
        help=(
            'Backbone checkpoint. "auto" uses ./pt/DLF<dataset>.pth when present; '
            '"none" trains from scratch.'
        ),
    )
    return parser.parse_args()


def resolve_checkpoint(value, dataset):
    if value.lower() == 'none':
        return None
    if value.lower() == 'auto':
        path = Path('./pt') / f'DLF{dataset}.pth'
        return path if path.is_file() else None
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f'Initial checkpoint not found: {path}')
    return path


def main():
    cli = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger('MMSA')
    setup_seed(cli.seed)
    device = assign_gpu([cli.gpu])
    args = get_config_regression('DLF', cli.dataset, Path(cli.config))
    args['device'] = device
    args['train_mode'] = 'regression'
    args['feature_T'] = ''
    args['feature_A'] = ''
    args['feature_V'] = ''
    args['seed'] = cli.seed
    args['cur_seed'] = 1
    args['batch_size'] = cli.batch_size

    save_dir = Path(cli.save_root) / cli.dataset / f'seed_{cli.seed}'
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info('V4 outputs: %s', save_dir)
    logger.info('V4 candidate offsets: %s', cli.candidate_offsets)

    dataloaders = MMDataLoader(args, cli.num_workers)
    model = OracleRegretDLF(
        args,
        offsets=cli.candidate_offsets,
        router_hidden_dim=cli.router_hidden_dim,
        candidate_dim=cli.candidate_dim,
        router_dropout=cli.router_dropout,
    ).to(device)
    checkpoint = resolve_checkpoint(cli.init_checkpoint, cli.dataset)
    if checkpoint is not None:
        incompatible = model.load_backbone_checkpoint(checkpoint, map_location=device)
        logger.info(
            'Initialized DLF backbone from %s (missing=%d unexpected=%d).',
            checkpoint, len(incompatible.missing_keys), len(incompatible.unexpected_keys),
        )
    else:
        logger.info('Training V4 backbone from scratch.')

    trainer = OracleRegretTrainerV4(
        args=args,
        metrics_fn=MetricsTop('regression').getMetics(cli.dataset),
        save_dir=save_dir,
        max_epochs=cli.max_epochs,
        early_stop=cli.early_stop,
        router_lr_multiplier=cli.router_lr_multiplier,
        teacher_temperature=cli.teacher_temperature,
        student_temperature=cli.student_temperature,
        distill_weight=cli.distill_weight,
        expected_mae_weight=cli.expected_mae_weight,
        routed_mae_weight=cli.routed_mae_weight,
        ranking_weight=cli.ranking_weight,
        amp=cli.amp,
    )
    model, best_epoch, best_valid_mae = trainer.train(model, dataloaders)
    logger.info('V4 selected epoch=%d raw-valid-routed-MAE=%.6f', best_epoch, best_valid_mae)
    summary = trainer.evaluate_and_save(model, dataloaders, best_epoch)
    logger.info('V4 fusion: %s', summary['fusion_metrics'])
    logger.info('V4 routed: %s', summary['inference_results']['best_valid_mae']['metrics'])
    logger.info('V4 codebook oracle: %s', summary['codebook_oracle_metrics'])


if __name__ == '__main__':
    main()
