"""Run V3.4 cross-fitted candidate-gain selection on MOSI/MOSEI."""

import argparse
import logging
from pathlib import Path

from config import get_config_regression
from trains.singleTask.backbone_crossfit_v31 import build_backbone_crossfit_views_v31
from trains.singleTask.candidate_gain_system_v34 import run_candidate_gain_v34
from trains.singleTask.semantic_features_v32 import attach_aligned_semantic_features
from trains.utils import MetricsTop
from utils import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Reuse exact cross-fitted DLF backbones and aligned semantic features, '
            'predict the realized gain and severe-harm risk of Up/Down candidates, '
            'and apply only risk-constrained candidate corrections.'
        )
    )
    parser.add_argument('--dataset', choices=['mosi', 'mosei'], default='mosi')
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--config', type=str, default='./config/config.json')
    parser.add_argument(
        '--save-root', type=str,
        default='./result/candidate_gain_selector_v34',
    )
    parser.add_argument(
        '--backbone-root', type=str,
        default='./result/backbone_crossfit_direction_v31',
    )
    parser.add_argument(
        '--semantic-root', type=str,
        default='./result/aligned_semantic_quantile_v32',
    )
    parser.add_argument(
        '--v33-root', type=str,
        default='./result/ordinal_conformal_stacking_v33',
        help='Directory containing reusable V3.3 ordinal/quantile fold checkpoints.',
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

    parser.add_argument('--ordinal-folds', type=int, default=5)
    parser.add_argument('--ordinal-epochs', type=int, default=80)
    parser.add_argument('--ordinal-learning-rate', type=float, default=4e-4)
    parser.add_argument('--ordinal-hidden-dim', type=int, default=128)
    parser.add_argument('--semantic-hidden-dim', type=int, default=96)
    parser.add_argument('--ordinal-dropout', type=float, default=0.20)
    parser.add_argument('--ordinal-patience', type=int, default=10)
    parser.add_argument('--max-quantile-center', type=float, default=1.75)
    parser.add_argument('--max-quantile-width', type=float, default=1.75)
    parser.add_argument('--ordinal-loss-weight', type=float, default=0.30)
    parser.add_argument('--target-loss-weight', type=float, default=0.25)
    parser.add_argument('--median-loss-weight', type=float, default=0.20)
    parser.add_argument(
        '--retrain-ordinal-models', action='store_true',
        help='Ignore reusable V3.3 fold models and rebuild ordinal features.',
    )

    parser.add_argument('--gain-folds', type=int, default=5)
    parser.add_argument('--gain-selection-folds', type=int, default=3)
    parser.add_argument('--gain-epochs', type=int, default=60)
    parser.add_argument('--gain-learning-rate', type=float, default=4e-4)
    parser.add_argument('--gain-hidden-dim', type=int, default=128)
    parser.add_argument('--gain-candidate-hidden-dim', type=int, default=64)
    parser.add_argument('--gain-dropout', type=float, default=0.20)
    parser.add_argument('--gain-patience', type=int, default=8)
    parser.add_argument('--max-predicted-gain', type=float, default=1.50)
    parser.add_argument('--gain-loss-weight', type=float, default=1.0)
    parser.add_argument('--benefit-loss-weight', type=float, default=0.75)
    parser.add_argument('--harm-loss-weight', type=float, default=0.50)

    parser.add_argument('--min-policy-selected', type=int, default=15)
    parser.add_argument('--min-policy-coverage', type=float, default=0.05)
    parser.add_argument('--min-policy-wilson-lcb', type=float, default=0.60)
    parser.add_argument('--max-policy-harm-010', type=float, default=0.02)

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
    v33_dir = Path(cli.v33_root) / cli.dataset / f'seed_{cli.seed}'
    for directory in (save_dir, backbone_dir, semantic_dir, v33_dir):
        directory.mkdir(parents=True, exist_ok=True)
    logger.info('V3.4 outputs: %s', save_dir)
    logger.info('Backbone cache: %s', backbone_dir)
    logger.info('Semantic cache: %s', semantic_dir)
    logger.info('V3.3 ordinal model cache: %s', v33_dir)

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
    summary = run_candidate_gain_v34(
        caches=caches,
        metrics_fn=MetricsTop('regression').getMetics(cli.dataset),
        device=device,
        save_dir=save_dir,
        seed=cli.seed,
        v33_model_dir=v33_dir,
        reuse_v33_models=not cli.retrain_ordinal_models,
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
        ordinal_folds=cli.ordinal_folds,
        ordinal_epochs=cli.ordinal_epochs,
        ordinal_learning_rate=cli.ordinal_learning_rate,
        ordinal_hidden_dim=cli.ordinal_hidden_dim,
        semantic_hidden_dim=cli.semantic_hidden_dim,
        ordinal_dropout=cli.ordinal_dropout,
        ordinal_patience=cli.ordinal_patience,
        max_quantile_center=cli.max_quantile_center,
        max_quantile_width=cli.max_quantile_width,
        ordinal_loss_weight=cli.ordinal_loss_weight,
        target_loss_weight=cli.target_loss_weight,
        median_loss_weight=cli.median_loss_weight,
        gain_folds=cli.gain_folds,
        gain_selection_folds=cli.gain_selection_folds,
        gain_epochs=cli.gain_epochs,
        gain_learning_rate=cli.gain_learning_rate,
        gain_hidden_dim=cli.gain_hidden_dim,
        gain_candidate_hidden_dim=cli.gain_candidate_hidden_dim,
        gain_dropout=cli.gain_dropout,
        gain_patience=cli.gain_patience,
        max_predicted_gain=cli.max_predicted_gain,
        gain_loss_weight=cli.gain_loss_weight,
        benefit_loss_weight=cli.benefit_loss_weight,
        harm_loss_weight=cli.harm_loss_weight,
        min_policy_selected=cli.min_policy_selected,
        min_policy_coverage=cli.min_policy_coverage,
        min_policy_wilson_lcb=cli.min_policy_wilson_lcb,
        max_policy_harm_010=cli.max_policy_harm_010,
    )
    logger.info('V3.4 policy: %s', summary['selective_policy'])
    logger.info('V3.4 q50 policy: %s', summary['q50_policy'])
    logger.info('V3.4 fusion: %s', summary['fusion_metrics'])
    logger.info('V3.4 router: %s', summary['residual_router_metrics'])
    logger.info('V3.4 q50 baseline: %s', summary['q50_direction_metrics'])
    logger.info('V3.4 oracle: %s', summary['specialist_oracle_metrics'])


if __name__ == '__main__':
    main()
