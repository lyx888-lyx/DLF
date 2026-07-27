"""Train residual-direction specialists with robust multi-scenario calibration."""

import argparse
import logging
from pathlib import Path

import torch

from config import get_config_regression
from data_loader import MMDataLoader
from trains.singleTask.model import DLF
from trains.singleTask.residual_direction_specialist import train_residual_direction_system
from trains.utils import MetricsTop
from utils import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Train upward/downward/boundary residual specialists with joint-stratified '
            'OOF predictions and multi-scenario robust component calibration.'
        )
    )
    parser.add_argument('--dataset', choices=['mosi', 'mosei'], default='mosi')
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--config', type=str, default='./config/config.json')
    parser.add_argument('--save-root', type=str, default='./result/residual_direction')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=1)

    parser.add_argument('--oof-folds', type=int, default=5)
    parser.add_argument('--specialist-epochs', type=int, default=25)
    parser.add_argument('--gate-epochs', type=int, default=60)
    parser.add_argument('--gate-patience', type=int, default=8)
    parser.add_argument('--specialist-learning-rate', type=float, default=8e-4)
    parser.add_argument('--gate-learning-rate', type=float, default=5e-4)
    parser.add_argument('--shared-dim', type=int, default=96)
    parser.add_argument('--adapter-dim', type=int, default=48)
    parser.add_argument('--gate-hidden-dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--max-residual', type=float, default=0.75)
    parser.add_argument('--boundary-max-residual', type=float, default=0.35)
    parser.add_argument('--batch-size', type=int, default=128)

    parser.add_argument('--residual-margin', type=float, default=0.10)
    parser.add_argument('--boundary-label-threshold', type=float, default=0.50)
    parser.add_argument('--off-region-anchor-weight', type=float, default=0.01)
    parser.add_argument('--sentiment-dro-weight', type=float, default=0.25)
    parser.add_argument('--direction-dro-weight', type=float, default=0.25)
    parser.add_argument('--harm-weight', type=float, default=1.5)
    parser.add_argument('--direction-loss-weight', type=float, default=0.20)
    parser.add_argument('--oracle-kl-weight', type=float, default=0.0)
    parser.add_argument('--anchor-weight', type=float, default=0.05)
    parser.add_argument('--correction-weight', type=float, default=0.01)
    parser.add_argument('--sign-weight', type=float, default=0.05)

    parser.add_argument('--calibration-folds', type=int, default=3)
    parser.add_argument('--calibration-robustness-weight', type=float, default=0.50)
    parser.add_argument('--calibration-harm-penalty', type=float, default=0.20)
    parser.add_argument('--calibration-correction-penalty', type=float, default=0.02)
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
    save_dir = Path(cli_args.save_root) / cli_args.dataset / f'seed_{cli_args.seed}'
    logger.info('Loaded frozen DLF checkpoint from %s', checkpoint)
    logger.info('Residual-direction outputs will be saved to %s', save_dir)

    _, summary = train_residual_direction_system(
        model=model,
        dataloader=dataloader,
        metrics_fn=metrics_fn,
        device=device,
        save_dir=save_dir,
        seed=cli_args.seed,
        oof_folds=cli_args.oof_folds,
        specialist_epochs=cli_args.specialist_epochs,
        gate_epochs=cli_args.gate_epochs,
        gate_patience=cli_args.gate_patience,
        specialist_learning_rate=cli_args.specialist_learning_rate,
        gate_learning_rate=cli_args.gate_learning_rate,
        shared_dim=cli_args.shared_dim,
        adapter_dim=cli_args.adapter_dim,
        gate_hidden_dim=cli_args.gate_hidden_dim,
        dropout=cli_args.dropout,
        max_residual=cli_args.max_residual,
        boundary_max_residual=cli_args.boundary_max_residual,
        batch_size=cli_args.batch_size,
        residual_margin=cli_args.residual_margin,
        boundary_label_threshold=cli_args.boundary_label_threshold,
        off_region_anchor_weight=cli_args.off_region_anchor_weight,
        sentiment_dro_weight=cli_args.sentiment_dro_weight,
        direction_dro_weight=cli_args.direction_dro_weight,
        harm_weight=cli_args.harm_weight,
        direction_loss_weight=cli_args.direction_loss_weight,
        oracle_kl_weight=cli_args.oracle_kl_weight,
        anchor_weight=cli_args.anchor_weight,
        correction_weight=cli_args.correction_weight,
        sign_weight=cli_args.sign_weight,
        calibration_folds=cli_args.calibration_folds,
        calibration_robustness_weight=cli_args.calibration_robustness_weight,
        calibration_harm_penalty=cli_args.calibration_harm_penalty,
        calibration_correction_penalty=cli_args.calibration_correction_penalty,
    )

    logger.info('Component alphas: %s', summary['component_alphas'])
    logger.info('Fusion metrics: %s', summary['fusion_metrics'])
    logger.info('Residual router metrics: %s', summary['residual_router_metrics'])
    logger.info('Specialist oracle metrics: %s', summary['specialist_oracle_metrics'])
    logger.info('Oracle gap recovery: %.4f', summary['oracle_gap_recovery_ratio'])


if __name__ == '__main__':
    main()
