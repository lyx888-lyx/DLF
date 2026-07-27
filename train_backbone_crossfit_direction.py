"""Train true backbone-cross-fitted DLF direction routing on MOSI/MOSEI."""

import argparse
import logging
from pathlib import Path

from config import get_config_regression
from trains.singleTask.backbone_crossfit import build_backbone_crossfit_caches
from trains.singleTask.selective_direction_gate import run_selective_direction_system
from trains.utils import MetricsTop
from utils import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Train outer-fold DLF backbones, generate true unseen OOF predictions, '
            'then train selective Up/Down/Keep residual routing.'
        )
    )
    parser.add_argument('--dataset', choices=['mosi', 'mosei'], default='mosi')
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--config', type=str, default='./config/config.json')
    parser.add_argument('--save-root', type=str, default='./result/backbone_crossfit_direction')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=1)

    parser.add_argument('--outer-folds', type=int, default=3)
    parser.add_argument('--inner-folds', type=int, default=5)
    parser.add_argument('--max-backbone-epochs', type=int, default=100)
    parser.add_argument('--retrain-backbones', action='store_true')

    parser.add_argument('--specialist-folds', type=int, default=5)
    parser.add_argument('--specialist-epochs', type=int, default=25)
    parser.add_argument('--specialist-learning-rate', type=float, default=8e-4)
    parser.add_argument('--specialist-shared-dim', type=int, default=96)
    parser.add_argument('--specialist-adapter-dim', type=int, default=48)
    parser.add_argument('--specialist-dropout', type=float, default=0.15)
    parser.add_argument('--max-residual', type=float, default=0.75)
    parser.add_argument('--boundary-max-residual', type=float, default=0.35)
    parser.add_argument('--off-region-anchor-weight', type=float, default=0.01)

    parser.add_argument('--classifier-epochs', type=int, default=60)
    parser.add_argument('--classifier-learning-rate', type=float, default=5e-4)
    parser.add_argument('--classifier-hidden-dim', type=int, default=64)
    parser.add_argument('--classifier-dropout', type=float, default=0.15)
    parser.add_argument('--classifier-patience', type=int, default=8)
    parser.add_argument('--label-smoothing', type=float, default=0.05)

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--residual-margin', type=float, default=0.10)
    parser.add_argument('--boundary-label-threshold', type=float, default=0.50)
    parser.add_argument('--calibration-folds', type=int, default=3)
    return parser.parse_args()


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
    logger.info('Outputs will be saved to %s', save_dir)
    logger.info(
        'Training %d outer DLF backbones. Existing fold checkpoints are reused unless --retrain-backbones is set.',
        cli.outer_folds,
    )

    caches = build_backbone_crossfit_caches(
        args=args,
        num_workers=cli.num_workers,
        device=device,
        save_dir=save_dir,
        seed=cli.seed,
        outer_folds=cli.outer_folds,
        inner_folds=cli.inner_folds,
        max_backbone_epochs=cli.max_backbone_epochs,
        reuse_backbones=not cli.retrain_backbones,
    )
    metrics_fn = MetricsTop('regression').getMetics(cli.dataset)
    summary = run_selective_direction_system(
        caches=caches,
        metrics_fn=metrics_fn,
        device=device,
        save_dir=save_dir,
        seed=cli.seed,
        specialist_folds=cli.specialist_folds,
        specialist_epochs=cli.specialist_epochs,
        specialist_learning_rate=cli.specialist_learning_rate,
        specialist_shared_dim=cli.specialist_shared_dim,
        specialist_adapter_dim=cli.specialist_adapter_dim,
        specialist_dropout=cli.specialist_dropout,
        max_residual=cli.max_residual,
        boundary_max_residual=cli.boundary_max_residual,
        batch_size=cli.batch_size,
        residual_margin=cli.residual_margin,
        boundary_label_threshold=cli.boundary_label_threshold,
        off_region_anchor_weight=cli.off_region_anchor_weight,
        classifier_epochs=cli.classifier_epochs,
        classifier_learning_rate=cli.classifier_learning_rate,
        classifier_hidden_dim=cli.classifier_hidden_dim,
        classifier_dropout=cli.classifier_dropout,
        classifier_patience=cli.classifier_patience,
        label_smoothing=cli.label_smoothing,
        calibration_folds=cli.calibration_folds,
    )
    logger.info('Selective policy: %s', summary['selective_policy'])
    logger.info('Fusion metrics: %s', summary['fusion_metrics'])
    logger.info('Router metrics: %s', summary['residual_router_metrics'])
    logger.info('Oracle metrics: %s', summary['specialist_oracle_metrics'])
    logger.info('Direction balanced accuracy: %.4f', summary['direction_balanced_accuracy'])
    logger.info('Correction precision/coverage: %.4f / %.4f', summary['correction_precision'], summary['correction_rate'])


if __name__ == '__main__':
    main()
