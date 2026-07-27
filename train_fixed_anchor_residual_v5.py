"""Train FAR-DLF V5: fixed-anchor shared continuous residual learning."""

import argparse
import logging
from pathlib import Path

from config import get_config_regression
from trains.singleTask.fixed_anchor_residual_system_v5 import run_fixed_anchor_residual_v5
from trains.singleTask.shared_residual_v5 import TrainingConfig
from trains.utils import MetricsTop
from utils import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Reuse frozen V3.1 OOF anchors, train fold-specific lightweight adapters '
            'and one shared continuous residual quantile network.'
        )
    )
    parser.add_argument('--dataset', choices=['mosi', 'mosei'], default='mosi')
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--config', type=str, default='./config/config.json')
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--residual-batch-size', type=int, default=128)
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
        default='./result/fixed_anchor_shared_residual_v5',
    )
    parser.add_argument('--rebuild-anchor-cache', action='store_true')
    parser.add_argument(
        '--anchor-cache-dir',
        type=str,
        default='auto',
        help=(
            'Directory containing fixed_oof_anchor_v41_cache.pth. "auto" reuses '
            'result/fixed_oof_oracle_v41/<dataset>/seed_<seed>.'
        ),
    )
    parser.add_argument('--selection-repeats', type=int, default=3)
    parser.add_argument('--internal-valid-fraction', type=float, default=0.20)

    parser.add_argument('--residual-epochs', type=int, default=80)
    parser.add_argument('--residual-patience', type=int, default=12)
    parser.add_argument('--residual-learning-rate', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--adapter-dim', type=int, default=64)
    parser.add_argument('--adapter-hidden-dim', type=int, default=96)
    parser.add_argument('--context-hidden-dim', type=int, default=48)
    parser.add_argument('--trunk-dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.20)
    parser.add_argument('--residual-max', type=float, default=1.0)
    parser.add_argument('--huber-delta', type=float, default=0.25)

    parser.add_argument('--huber-weight', type=float, default=1.0)
    parser.add_argument('--quantile-weight', type=float, default=0.50)
    parser.add_argument('--sign-weight', type=float, default=0.10)
    parser.add_argument('--magnitude-weight', type=float, default=0.10)
    parser.add_argument('--region-weight', type=float, default=0.30)
    parser.add_argument('--bias-weight', type=float, default=0.10)
    parser.add_argument('--alignment-weight', type=float, default=0.05)
    parser.add_argument('--shrinkage-weight', type=float, default=0.005)
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
    args = get_config_regression('DLF', cli.dataset, Path(cli.config))
    args['device'] = device
    args['train_mode'] = 'regression'
    args['feature_T'] = ''
    args['feature_A'] = ''
    args['feature_V'] = ''
    args['seed'] = cli.seed
    args['cur_seed'] = 1
    args['batch_size'] = cli.batch_size

    backbone_dir = resolve_backbone_dir(cli.backbone_dir, cli.dataset, cli.seed)
    save_dir = Path(cli.save_root) / cli.dataset / f'seed_{cli.seed}'
    anchor_cache_dir = (
        Path('./result/fixed_oof_oracle_v41') / cli.dataset / f'seed_{cli.seed}'
        if cli.anchor_cache_dir.lower() == 'auto'
        else Path(cli.anchor_cache_dir)
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info('V5 frozen backbone directory: %s', backbone_dir)
    logger.info('V5 outputs: %s', save_dir)
    logger.info('V5 anchor cache directory: %s', anchor_cache_dir)

    model_config = {
        'adapter_dim': cli.adapter_dim,
        'adapter_hidden_dim': cli.adapter_hidden_dim,
        'context_hidden_dim': cli.context_hidden_dim,
        'trunk_dim': cli.trunk_dim,
        'dropout': cli.dropout,
        'residual_max': cli.residual_max,
    }
    training_config = TrainingConfig(
        learning_rate=cli.residual_learning_rate,
        weight_decay=cli.weight_decay,
        epochs=cli.residual_epochs,
        patience=cli.residual_patience,
        batch_size=cli.residual_batch_size,
        huber_delta=cli.huber_delta,
        huber_weight=cli.huber_weight,
        quantile_weight=cli.quantile_weight,
        sign_weight=cli.sign_weight,
        magnitude_weight=cli.magnitude_weight,
        region_weight=cli.region_weight,
        bias_weight=cli.bias_weight,
        alignment_weight=cli.alignment_weight,
        shrinkage_weight=cli.shrinkage_weight,
    )
    summary = run_fixed_anchor_residual_v5(
        args=args,
        metrics_fn=MetricsTop('regression').getMetics(cli.dataset),
        device=device,
        save_dir=save_dir,
        backbone_dir=backbone_dir,
        seed=cli.seed,
        num_workers=cli.num_workers,
        outer_folds=cli.outer_folds,
        rebuild_anchor_cache=cli.rebuild_anchor_cache,
        anchor_cache_dir=anchor_cache_dir,
        selection_repeats=cli.selection_repeats,
        validation_fraction=cli.internal_valid_fraction,
        model_config=model_config,
        training_config=training_config,
    )
    logger.info('V5 fixed anchor: %s', summary['fixed_anchor_metrics'])
    logger.info('V5 full residual: %s', summary['inference_results']['full_residual']['metrics'])
    logger.info('V5 robust residual: %s', summary['inference_results']['robust_valid']['metrics'])
    logger.info('V5 ridge: %s', summary['inference_results']['ridge']['metrics'])


if __name__ == '__main__':
    main()
