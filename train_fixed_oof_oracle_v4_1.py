"""Train V4.1 with frozen V3.1 OOF anchors and a fixed oracle teacher."""

import argparse
import logging
from pathlib import Path

from config import get_config_regression
from trains.singleTask.fixed_oracle_system_v41 import run_fixed_oof_oracle_v41
from trains.utils import MetricsTop
from utils import assign_gpu, setup_seed


def parse_offsets(value):
    offsets = tuple(
        float(token.strip()) for token in value.split(',') if token.strip()
    )
    if not offsets:
        raise argparse.ArgumentTypeError('At least one candidate offset is required.')
    if min(abs(value) for value in offsets) > 1e-8:
        offsets = offsets + (0.0,)
    return tuple(sorted(set(offsets)))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Train fold-specific candidate-energy students on fixed V3.1 OOF '
            'anchors and fixed label-derived oracle distributions.'
        )
    )
    parser.add_argument('--dataset', choices=['mosi', 'mosei'], default='mosi')
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--config', type=str, default='./config/config.json')
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--student-batch-size', type=int, default=128)
    parser.add_argument('--outer-folds', type=int, default=3)
    parser.add_argument(
        '--backbone-dir',
        type=str,
        default='auto',
        help=(
            'Directory containing backbone_v31_fold_*.pth. "auto" resolves '
            'result/backbone_crossfit_direction_v31/<dataset>/seed_<seed>.'
        ),
    )
    parser.add_argument(
        '--save-root',
        type=str,
        default='./result/fixed_oof_oracle_v41',
    )
    parser.add_argument('--rebuild-anchor-cache', action='store_true')
    parser.add_argument(
        '--candidate-offsets',
        type=parse_offsets,
        default=(-0.75, -0.50, -0.25, -0.10, 0.0, 0.10, 0.25, 0.50, 0.75),
    )
    parser.add_argument('--student-epochs', type=int, default=80)
    parser.add_argument('--student-learning-rate', type=float, default=3e-4)
    parser.add_argument('--student-hidden-dim', type=int, default=192)
    parser.add_argument('--student-adapter-dim', type=int, default=128)
    parser.add_argument('--candidate-dim', type=int, default=64)
    parser.add_argument('--student-dropout', type=float, default=0.20)
    parser.add_argument('--teacher-temperature', type=float, default=0.10)
    parser.add_argument('--student-temperature', type=float, default=1.0)
    parser.add_argument('--distill-weight', type=float, default=1.0)
    parser.add_argument('--expected-mae-weight', type=float, default=0.50)
    parser.add_argument('--ranking-weight', type=float, default=0.20)
    parser.add_argument('--ordinal-emd-weight', type=float, default=0.50)
    parser.add_argument('--offset-weight', type=float, default=0.50)
    parser.add_argument('--internal-valid-fraction', type=float, default=0.20)
    parser.add_argument('--student-patience', type=int, default=10)
    return parser.parse_args()


def resolve_backbone_dir(value, dataset, seed):
    if value.lower() == 'auto':
        return (
            Path('./result/backbone_crossfit_direction_v31')
            / dataset
            / f'seed_{seed}'
        )
    return Path(value)


def main():
    cli = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger('MMSA')
    setup_seed(cli.seed)
    device = assign_gpu([cli.gpu])
    args = get_config_regression(
        'DLF', cli.dataset, Path(cli.config)
    )
    args['device'] = device
    args['train_mode'] = 'regression'
    args['feature_T'] = ''
    args['feature_A'] = ''
    args['feature_V'] = ''
    args['seed'] = cli.seed
    args['cur_seed'] = 1
    args['batch_size'] = cli.batch_size

    backbone_dir = resolve_backbone_dir(
        cli.backbone_dir, cli.dataset, cli.seed
    )
    save_dir = (
        Path(cli.save_root)
        / cli.dataset
        / f'seed_{cli.seed}'
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info('V4.1 frozen backbone directory: %s', backbone_dir)
    logger.info('V4.1 outputs: %s', save_dir)
    logger.info('V4.1 candidate offsets: %s', cli.candidate_offsets)

    summary = run_fixed_oof_oracle_v41(
        args=args,
        metrics_fn=MetricsTop('regression').getMetics(cli.dataset),
        device=device,
        save_dir=save_dir,
        backbone_dir=backbone_dir,
        seed=cli.seed,
        num_workers=cli.num_workers,
        outer_folds=cli.outer_folds,
        rebuild_anchor_cache=cli.rebuild_anchor_cache,
        offsets=cli.candidate_offsets,
        student_epochs=cli.student_epochs,
        student_learning_rate=cli.student_learning_rate,
        student_hidden_dim=cli.student_hidden_dim,
        student_adapter_dim=cli.student_adapter_dim,
        candidate_dim=cli.candidate_dim,
        student_dropout=cli.student_dropout,
        student_batch_size=cli.student_batch_size,
        teacher_temperature=cli.teacher_temperature,
        student_temperature=cli.student_temperature,
        distill_weight=cli.distill_weight,
        expected_mae_weight=cli.expected_mae_weight,
        ranking_weight=cli.ranking_weight,
        ordinal_emd_weight=cli.ordinal_emd_weight,
        offset_weight=cli.offset_weight,
        validation_fraction=cli.internal_valid_fraction,
        student_patience=cli.student_patience,
    )
    logger.info('V4.1 fixed anchor: %s', summary['fixed_anchor_metrics'])
    logger.info(
        'V4.1 robust routed: %s',
        summary['inference_results']['robust_valid']['metrics'],
    )
    logger.info(
        'V4.1 best-valid shadow: %s',
        summary['inference_results']['best_valid']['metrics'],
    )
    logger.info(
        'V4.1 fixed codebook oracle: %s',
        summary['fixed_codebook_oracle_metrics'],
    )


if __name__ == '__main__':
    main()
