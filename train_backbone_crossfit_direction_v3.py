"""Run full outer-fold DLF retraining and selective direction routing."""

from train_backbone_crossfit_direction import parse_args
import logging
from pathlib import Path

from config import get_config_regression
from trains.singleTask.backbone_crossfit_full import build_full_backbone_crossfit_caches
from trains.singleTask.selective_direction_gate import run_selective_direction_system
from trains.utils import MetricsTop
from utils import assign_gpu, setup_seed


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

    caches = build_full_backbone_crossfit_caches(
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
    summary = run_selective_direction_system(
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
    logger.info(
        'Direction balanced accuracy %.4f; correction precision/coverage %.4f/%.4f',
        summary['direction_balanced_accuracy'],
        summary['correction_precision'],
        summary['correction_rate'],
    )


if __name__ == '__main__':
    main()
