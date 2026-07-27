"""Run V3.3 ordinal-conformal residual routing with direct target stacking."""

import argparse
import logging
from pathlib import Path

from config import get_config_regression
from trains.singleTask.backbone_crossfit_v31 import build_backbone_crossfit_views_v31
from trains.singleTask.ordinal_conformal_system_v33 import run_ordinal_conformal_v33
from trains.singleTask.semantic_features_v32 import attach_aligned_semantic_features
from trains.utils import MetricsTop
from utils import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Reuse exact cross-fitted DLF backbones, attach one shared frozen semantic '
            'representation, cross-fit an unweighted residual quantile model with '
            'ordinal-region and direct-target heads, conformalize its intervals, and '
            'apply only statistically supported specialist corrections.'
        )
    )
    parser.add_argument('--dataset', choices=['mosi', 'mosei'], default='mosi')
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--config', type=str, default='./config/config.json')
    parser.add_argument(
        '--save-root', type=str,
        default='./result/ordinal_conformal_stacking_v33',
    )
    parser.add_argument(
        '--backbone-root', type=str,
        default='./result/backbone_crossfit_direction_v31',
    )
    parser.add_argument(
        '--semantic-root', type=str,
        default='./result/aligned_semantic_quantile_v32',
        help='Directory containing reusable V3.2 shared semantic features.',
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

    parser.add_argument('--model-folds', type=int, default=5)
    parser.add_argument('--model-epochs', type=int, default=80)
    parser.add_argument('--model-learning-rate', type=float, default=4e-4)
    parser.add_argument('--model-hidden-dim', type=int, default=128)
    parser.add_argument('--semantic-hidden-dim', type=int, default=96)
    parser.add_argument('--model-dropout', type=float, default=0.20)
    parser.add_argument('--model-patience', type=int, default=10)
    parser.add_argument('--max-quantile-center', type=float, default=1.75)
    parser.add_argument('--max-quantile-width', type=float, default=1.75)
    parser.add_argument('--ordinal-loss-weight', type=float, default=0.30)
    parser.add_argument('--target-loss-weight', type=float, default=0.25)
    parser.add_argument('--median-loss-weight', type=float, default=0.20)

    parser.add_argument('--conformal-coverage', type=float, default=0.80)
    parser.add_argument('--min-policy-selected', type=int, default=15)
    parser.add_argument('--min-policy-coverage', type=float, default=0.05)
    parser.add_argument('--min-policy-wilson-lcb', type=float, default=0.60)

    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--residual-margin', type=float, default=0.10)
    parser.add_argument('--boundary-label-threshold', type=float, default=0.50)
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
    semantic_dir = Path(cli.semantic_root) / cli.dataset / f'seed_{cli.seed}'
    save_dir.mkdir(parents=True, exist_ok=True)
    backbone_dir.mkdir(parents=True, exist_ok=True)
    semantic_dir.mkdir(parents=True, exist_ok=True)
    logger.info('V3.3 outputs: %s', save_dir)
    logger.info('Backbone cache: %s', backbone_dir)
    logger.info('Semantic cache: %s', semantic_dir)

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
        save_dir=semantic_dir,
        seed=cli.seed,
        semantic_dim=cli.semantic_dim,
        extraction_batch_size=cli.semantic_extraction_batch_size,
        reuse_features=not cli.recompute_semantic_features,
    )
    summary = run_ordinal_conformal_v33(
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
        model_folds=cli.model_folds,
        model_epochs=cli.model_epochs,
        model_learning_rate=cli.model_learning_rate,
        model_hidden_dim=cli.model_hidden_dim,
        semantic_hidden_dim=cli.semantic_hidden_dim,
        model_dropout=cli.model_dropout,
        model_patience=cli.model_patience,
        max_quantile_center=cli.max_quantile_center,
        max_quantile_width=cli.max_quantile_width,
        ordinal_loss_weight=cli.ordinal_loss_weight,
        target_loss_weight=cli.target_loss_weight,
        median_loss_weight=cli.median_loss_weight,
        conformal_coverage=cli.conformal_coverage,
        min_policy_selected=cli.min_policy_selected,
        min_policy_coverage=cli.min_policy_coverage,
        min_policy_wilson_lcb=cli.min_policy_wilson_lcb,
    )
    logger.info('V3.3 policy: %s', summary['selective_policy'])
    logger.info('V3.3 fusion: %s', summary['fusion_metrics'])
    logger.info('V3.3 router: %s', summary['residual_router_metrics'])
    logger.info('V3.3 target stacking: %s', summary['target_stacking_metrics'])
    logger.info('V3.3 ridge: %s', summary['ridge_residual_metrics'])
    logger.info('V3.3 oracle: %s', summary['specialist_oracle_metrics'])
    logger.info(
        'Residual corr %.4f sign_bal %.4f ordinal_bal %.4f precision=%s lcb=%s coverage=%.4f',
        summary['model_residual_correlation'],
        summary['model_sign_balanced_accuracy'],
        summary['model_ordinal_balanced_accuracy'],
        'NA' if summary['correction_precision'] is None else f"{summary['correction_precision']:.4f}",
        'NA' if summary['correction_precision_wilson_lcb'] is None else f"{summary['correction_precision_wilson_lcb']:.4f}",
        summary['correction_rate'],
    )


if __name__ == '__main__':
    main()
