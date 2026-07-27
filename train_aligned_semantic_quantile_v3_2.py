"""Run V3.2 aligned semantic residual quantile routing."""

import argparse
import logging
from pathlib import Path

from config import get_config_regression
from trains.singleTask.backbone_crossfit_v31 import build_backbone_crossfit_views_v31
from trains.singleTask.quantile_policy_v32 import run_aligned_semantic_quantile_v32
from trains.singleTask.semantic_features_v32 import attach_aligned_semantic_features
from trains.utils import MetricsTop
from utils import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Reuse or train exact cross-fitted DLF backbones, add one shared frozen '
            'semantic representation, predict residual quantiles, and apply only '
            'interval-certified specialist corrections.'
        )
    )
    parser.add_argument('--dataset', choices=['mosi', 'mosei'], default='mosi')
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--config', type=str, default='./config/config.json')
    parser.add_argument(
        '--save-root', type=str,
        default='./result/aligned_semantic_quantile_v32',
    )
    parser.add_argument(
        '--backbone-root', type=str,
        default='./result/backbone_crossfit_direction_v31',
        help='V3.1 checkpoint root. Existing exact-fold backbones are reused.',
    )
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=1)

    parser.add_argument('--outer-folds', type=int, default=3)
    parser.add_argument('--inner-folds', type=int, default=5)
    parser.add_argument('--max-backbone-epochs', type=int, default=100)
    parser.add_argument('--retrain-backbones', action='store_true')
    parser.add_argument('--backbone-batch-size', type=int, default=32)

    parser.add_argument('--semantic-dim', type=int, default=96)
    parser.add_argument('--semantic-extraction-batch-size', type=int, default=16)
    parser.add_argument('--recompute-semantic-features', action='store_true')

    parser.add_argument('--specialist-folds', type=int, default=5)
    parser.add_argument('--specialist-epochs', type=int, default=25)
    parser.add_argument('--specialist-learning-rate', type=float, default=8e-4)
    parser.add_argument('--specialist-shared-dim', type=int, default=96)
    parser.add_argument('--specialist-adapter-dim', type=int, default=48)
    parser.add_argument('--specialist-dropout', type=float, default=0.15)
    parser.add_argument('--max-residual', type=float, default=0.75)
    parser.add_argument('--boundary-max-residual', type=float, default=0.35)
    parser.add_argument('--off-region-anchor-weight', type=float, default=0.01)

    parser.add_argument('--quantile-epochs', type=int, default=80)
    parser.add_argument('--quantile-learning-rate', type=float, default=4e-4)
    parser.add_argument('--quantile-hidden-dim', type=int, default=128)
    parser.add_argument('--semantic-hidden-dim', type=int, default=96)
    parser.add_argument('--quantile-dropout', type=float, default=0.20)
    parser.add_argument('--quantile-patience', type=int, default=10)
    parser.add_argument('--max-quantile-center', type=float, default=1.50)
    parser.add_argument('--max-quantile-width', type=float, default=1.50)
    parser.add_argument('--median-loss-weight', type=float, default=0.25)

    parser.add_argument('--batch-size', type=int, default=128)
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
    args['batch_size'] = cli.backbone_batch_size

    save_dir = Path(cli.save_root) / cli.dataset / f'seed_{cli.seed}'
    backbone_dir = Path(cli.backbone_root) / cli.dataset / f'seed_{cli.seed}'
    save_dir.mkdir(parents=True, exist_ok=True)
    backbone_dir.mkdir(parents=True, exist_ok=True)
    logger.info('V3.2 outputs: %s', save_dir)
    logger.info('V3.1 backbone checkpoints/cache: %s', backbone_dir)

    caches = build_backbone_crossfit_views_v31(
        args=args,
        num_workers=cli.num_workers,
        device=device,
        save_dir=backbone_dir,
        seed=cli.seed,
        outer_folds=cli.outer_folds,
        inner_folds=cli.inner_folds,
        max_backbone_epochs=cli.max_backbone_epochs,
        reuse_backbones=not cli.retrain_backbones,
    )
    caches = attach_aligned_semantic_features(
        args=args,
        caches=caches,
        num_workers=cli.num_workers,
        device=device,
        save_dir=save_dir,
        seed=cli.seed,
        semantic_dim=cli.semantic_dim,
        extraction_batch_size=cli.semantic_extraction_batch_size,
        reuse_features=not cli.recompute_semantic_features,
    )
    summary = run_aligned_semantic_quantile_v32(
        caches=caches,
        metrics_fn=MetricsTop('regression').getMetics(cli.dataset),
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
        quantile_epochs=cli.quantile_epochs,
        quantile_learning_rate=cli.quantile_learning_rate,
        quantile_hidden_dim=cli.quantile_hidden_dim,
        semantic_hidden_dim=cli.semantic_hidden_dim,
        quantile_dropout=cli.quantile_dropout,
        quantile_patience=cli.quantile_patience,
        max_quantile_center=cli.max_quantile_center,
        max_quantile_width=cli.max_quantile_width,
        median_loss_weight=cli.median_loss_weight,
        calibration_folds=cli.calibration_folds,
    )
    logger.info('V3.2 policy: %s', summary['selective_policy'])
    logger.info('V3.2 fusion: %s', summary['fusion_metrics'])
    logger.info('V3.2 router: %s', summary['residual_router_metrics'])
    logger.info('V3.2 ridge: %s', summary['ridge_residual_metrics'])
    logger.info('V3.2 quantile median: %s', summary['quantile_median_metrics'])
    logger.info('V3.2 oracle: %s', summary['specialist_oracle_metrics'])
    logger.info(
        'Residual correlation %.4f; sign balanced accuracy %.4f; interval precision=%s coverage=%.4f',
        summary['quantile_residual_correlation'],
        summary['quantile_sign_balanced_accuracy'],
        'NA' if summary['quantile_confident_direction_precision'] is None else f"{summary['quantile_confident_direction_precision']:.4f}",
        summary['quantile_confident_direction_coverage'],
    )


if __name__ == '__main__':
    main()
