import copy
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .HingeLoss import HingeLoss
from .expert_analysis import normalize_batch_ids
from .oracle_regret_core_v4 import (
    apply_inference_policy,
    calibrate_inference_policy,
    candidate_diagnostics,
    oracle_regret_losses,
    region_rows,
    selection_stats,
)


logger = logging.getLogger('MMSA')


class MSE(nn.Module):
    def forward(self, prediction, target):
        return (prediction - target).pow(2).mean()


def _safe_metrics(metrics_fn, prediction, labels):
    result = metrics_fn(prediction.detach().cpu(), labels.detach().cpu())
    return {key: float(value) for key, value in result.items()}


class OracleRegretTrainerV4:
    def __init__(
        self,
        args,
        metrics_fn,
        save_dir,
        max_epochs=80,
        early_stop=12,
        router_lr_multiplier=3.0,
        teacher_temperature=0.10,
        student_temperature=1.0,
        distill_weight=0.60,
        expected_mae_weight=0.40,
        routed_mae_weight=1.00,
        ranking_weight=0.20,
        amp=False,
    ):
        self.args = args
        self.metrics_fn = metrics_fn
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_epochs = int(max_epochs)
        self.early_stop = int(early_stop)
        self.router_lr_multiplier = float(router_lr_multiplier)
        self.teacher_temperature = float(teacher_temperature)
        self.student_temperature = float(student_temperature)
        self.distill_weight = float(distill_weight)
        self.expected_mae_weight = float(expected_mae_weight)
        self.routed_mae_weight = float(routed_mae_weight)
        self.ranking_weight = float(ranking_weight)
        self.amp = bool(amp and torch.cuda.is_available())
        self.l1 = nn.L1Loss()
        self.mse = MSE()
        self.cosine = nn.CosineEmbeddingLoss()
        self.sim_loss = HingeLoss()

    def _base_dlf_loss(self, output, labels):
        task = (
            self.l1(output['output_logit'], labels)
            + self.l1(output['logits_c'], labels)
            + 3.0 * self.l1(output['logits_l_hetero'], labels)
            + self.l1(output['logits_v_hetero'], labels)
            + self.l1(output['logits_a_hetero'], labels)
        )
        reconstruction = (
            self.mse(output['recon_l'], output['origin_l'])
            + self.mse(output['recon_v'], output['origin_v'])
            + self.mse(output['recon_a'], output['origin_a'])
        )
        specific = (
            self.mse(output['s_l'].permute(1, 2, 0), output['s_l_r'])
            + self.mse(output['s_v'].permute(1, 2, 0), output['s_v_r'])
            + self.mse(output['s_a'].permute(1, 2, 0), output['s_a_r'])
        )
        reshape_size = 50 if self.args.dataset_name == 'mosi' else 10
        negative = labels.new_full((1,), -1.0)
        orthogonal = (
            self.cosine(
                output['s_l'].reshape(-1, reshape_size),
                output['c_l'].reshape(-1, reshape_size),
                negative,
            )
            + self.cosine(
                output['s_v'].reshape(-1, reshape_size),
                output['c_v'].reshape(-1, reshape_size),
                negative,
            )
            + self.cosine(
                output['s_a'].reshape(-1, reshape_size),
                output['c_a'].reshape(-1, reshape_size),
                negative,
            )
        )
        c_l, c_v, c_a = output['c_l_sim'], output['c_v_sim'], output['c_a_sim']
        features, ids = [], []
        for index in range(labels.size(0)):
            for value in (c_l[index], c_v[index], c_a[index]):
                features.append(value.view(1, -1))
                ids.append(labels[index].view(1, -1))
        similarity = self.sim_loss(torch.cat(ids, dim=0), torch.cat(features, dim=0))
        base = task + 0.1 * (specific + reconstruction + 0.1 * (similarity + orthogonal))
        return base, {
            'task': task,
            'reconstruction': reconstruction,
            'specific': specific,
            'similarity': similarity,
            'orthogonal': orthogonal,
        }

    def _batch_loss(self, output, labels):
        base, components = self._base_dlf_loss(output, labels)
        oracle = oracle_regret_losses(
            output,
            labels,
            teacher_temperature=self.teacher_temperature,
            student_temperature=self.student_temperature,
        )
        total = (
            base
            + self.distill_weight * oracle['distill']
            + self.expected_mae_weight * oracle['expected_mae']
            + self.routed_mae_weight * oracle['routed_mae']
            + self.ranking_weight * oracle['ranking']
        )
        values = {
            'loss': total,
            'base_loss': base,
            'distill_loss': oracle['distill'],
            'expected_mae_loss': oracle['expected_mae'],
            'routed_mae_loss': oracle['routed_mae'],
            'ranking_loss': oracle['ranking'],
            'candidate_top1_accuracy': oracle['top1_accuracy'],
            'mean_candidate_regret': oracle['mean_regret'],
            'normalized_candidate_regret': oracle['normalized_regret'],
            **{f'base_{key}': value for key, value in components.items()},
        }
        return total, values

    def _optimizer(self, model):
        router_parameters = list(model.energy_head.parameters())
        router_ids = {id(parameter) for parameter in router_parameters}
        backbone_parameters = [
            parameter for parameter in model.parameters() if id(parameter) not in router_ids
        ]
        return optim.AdamW(
            [
                {'params': backbone_parameters, 'lr': float(self.args.learning_rate)},
                {
                    'params': router_parameters,
                    'lr': float(self.args.learning_rate) * self.router_lr_multiplier,
                },
            ],
            weight_decay=1e-4,
        )

    def train(self, model, dataloaders):
        device = self.args.device
        model.to(device)
        optimizer = self._optimizer(model)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=int(self.args.patience)
        )
        scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        checkpoint = self.save_dir / 'oracle_regret_v4_best.pth'
        best_state = None
        best_epoch = 0
        best_valid_mae = float('inf')
        history = []

        for epoch in range(1, self.max_epochs + 1):
            model.train()
            totals = {}
            predictions, labels_all = [], []
            optimizer.zero_grad()
            accumulation = max(1, int(getattr(self.args, 'update_epochs', 1)))
            for batch_index, batch in enumerate(dataloaders['train'], start=1):
                text = batch['text'].to(device)
                audio = batch['audio'].to(device)
                vision = batch['vision'].to(device)
                labels = batch['labels']['M'].to(device).view(-1, 1)
                with torch.cuda.amp.autocast(enabled=self.amp):
                    output = model(text, audio, vision)
                    loss, values = self._batch_loss(output, labels)
                    scaled_loss = loss / accumulation
                scaler.scale(scaled_loss).backward()
                if batch_index % accumulation == 0 or batch_index == len(dataloaders['train']):
                    scaler.unscale_(optimizer)
                    if float(self.args.grad_clip) != -1.0:
                        nn.utils.clip_grad_norm_(model.parameters(), float(self.args.grad_clip))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                for key, value in values.items():
                    totals[key] = totals.get(key, 0.0) + float(value.detach().item())
                predictions.append(output['routed_prediction'].detach().cpu())
                labels_all.append(labels.detach().cpu())

            count = max(1, len(dataloaders['train']))
            train_row = {key: value / count for key, value in totals.items()}
            train_metrics = _safe_metrics(
                self.metrics_fn, torch.cat(predictions), torch.cat(labels_all)
            )
            valid_raw = self.collect(model, dataloaders['valid'])
            valid_fusion_metrics = _safe_metrics(
                self.metrics_fn, valid_raw['fusion'], valid_raw['labels']
            )
            valid_raw_result = apply_inference_policy(
                valid_raw,
                {'mode': 'soft', 'temperature': 1.0, 'blend': 1.0, 'entropy_threshold': None},
            )
            valid_routed_metrics = _safe_metrics(
                self.metrics_fn, valid_raw_result['prediction'], valid_raw['labels']
            )
            scheduler.step(valid_routed_metrics['MAE'])
            row = {
                'epoch': epoch,
                **{f'train_{key}': value for key, value in train_row.items()},
                **{f'train_metric_{key}': value for key, value in train_metrics.items()},
                **{f'valid_fusion_{key}': value for key, value in valid_fusion_metrics.items()},
                **{f'valid_raw_routed_{key}': value for key, value in valid_routed_metrics.items()},
            }
            history.append(row)
            logger.info(
                'V4 epoch=%d train_loss=%.4f valid_fusion_MAE=%.4f '
                'valid_routed_MAE=%.4f top1=%.4f regret=%.4f',
                epoch,
                train_row['loss'],
                valid_fusion_metrics['MAE'],
                valid_routed_metrics['MAE'],
                train_row['candidate_top1_accuracy'],
                train_row['mean_candidate_regret'],
            )
            if valid_routed_metrics['MAE'] < best_valid_mae - 1e-6:
                best_valid_mae = valid_routed_metrics['MAE']
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                torch.save(
                    {
                        'state_dict': best_state,
                        'epoch': best_epoch,
                        'valid_routed_mae': best_valid_mae,
                    },
                    checkpoint,
                )
            if epoch - best_epoch >= self.early_stop:
                break

        pd.DataFrame(history).to_csv(
            self.save_dir / 'v4_training_history.csv', index=False
        )
        if best_state is None:
            raise RuntimeError('V4 training did not produce a valid checkpoint.')
        model.load_state_dict(best_state)
        return model, best_epoch, best_valid_mae

    @torch.no_grad()
    def collect(self, model, dataloader):
        model.eval()
        buffers = {
            'fusion': [], 'candidate_values': [], 'candidate_logits': [],
            'labels': [], 'sample_ids': [],
        }
        for batch in dataloader:
            text = batch['text'].to(self.args.device)
            audio = batch['audio'].to(self.args.device)
            vision = batch['vision'].to(self.args.device)
            labels = batch['labels']['M'].to(self.args.device).view(-1, 1)
            output = model(text, audio, vision)
            buffers['fusion'].append(output['output_logit'].detach().cpu())
            buffers['candidate_values'].append(output['candidate_values'].detach().cpu())
            buffers['candidate_logits'].append(output['candidate_logits'].detach().cpu())
            buffers['labels'].append(labels.detach().cpu())
            buffers['sample_ids'].extend(normalize_batch_ids(batch.get('id')))
        return {
            key: torch.cat(value, dim=0) if key != 'sample_ids' else value
            for key, value in buffers.items()
        }

    def evaluate_and_save(self, model, dataloaders, best_epoch):
        valid_raw = self.collect(model, dataloaders['valid'])
        test_raw = self.collect(model, dataloaders['test'])
        best_policy, robust_policy, calibration_rows = calibrate_inference_policy(
            valid_raw, valid_raw['labels']
        )
        pd.DataFrame(calibration_rows).to_csv(
            self.save_dir / 'v4_inference_policy_calibration.csv', index=False
        )

        raw_policy = {
            'mode': 'soft', 'temperature': 1.0, 'blend': 1.0,
            'entropy_threshold': None,
        }
        policies = {
            'raw_soft': raw_policy,
            'best_valid_mae': best_policy,
            'robust_valid': robust_policy,
        }
        fusion_metrics = _safe_metrics(
            self.metrics_fn, test_raw['fusion'], test_raw['labels']
        )
        diagnostics = candidate_diagnostics(test_raw, test_raw['labels'])
        oracle_metrics = _safe_metrics(
            self.metrics_fn, diagnostics['oracle_prediction'], test_raw['labels']
        )
        hard_metrics = _safe_metrics(
            self.metrics_fn, diagnostics['hard_prediction'], test_raw['labels']
        )

        rows = [{'model': 'fusion', **fusion_metrics}]
        results = {}
        for name, policy in policies.items():
            result = apply_inference_policy(test_raw, policy)
            metrics = _safe_metrics(
                self.metrics_fn, result['prediction'], test_raw['labels']
            )
            stats = selection_stats(
                test_raw['fusion'], result['prediction'], test_raw['labels']
            )
            results[name] = {'policy': policy, 'metrics': metrics, **stats}
            rows.append({'model': name, **metrics, **stats})
        rows.extend([
            {'model': 'hard_top1', **hard_metrics},
            {'model': 'codebook_oracle', **oracle_metrics},
        ])
        pd.DataFrame(rows).to_csv(
            self.save_dir / 'v4_test_baseline_comparison.csv', index=False
        )

        main_result = apply_inference_policy(test_raw, best_policy)
        pd.DataFrame(region_rows(
            test_raw['fusion'], main_result['prediction'],
            diagnostics['oracle_prediction'], test_raw['labels'],
        )).to_csv(self.save_dir / 'v4_test_region_diagnostics.csv', index=False)

        probabilities = main_result['probabilities']
        sample_ids = test_raw['sample_ids']
        if len(sample_ids) != len(test_raw['labels']):
            sample_ids = [str(index) for index in range(len(test_raw['labels']))]
        prediction_data = {
            'sample_id': sample_ids,
            'target': test_raw['labels'].view(-1).numpy(),
            'fusion_prediction': test_raw['fusion'].view(-1).numpy(),
            'v4_prediction': main_result['prediction'].view(-1).numpy(),
            'v4_correction': main_result['correction'].view(-1).numpy(),
            'candidate_entropy': main_result['entropy'].numpy(),
            'oracle_prediction': diagnostics['oracle_prediction'].view(-1).numpy(),
            'hard_prediction': diagnostics['hard_prediction'].view(-1).numpy(),
            'oracle_candidate_index': diagnostics['oracle_index'].numpy(),
            'hard_candidate_index': diagnostics['hard_index'].numpy(),
        }
        for index, offset in enumerate(model.candidate_offsets.detach().cpu().tolist()):
            token = str(offset).replace('-', 'm').replace('.', 'p')
            prediction_data[f'candidate_{token}'] = test_raw['candidate_values'][:, index].numpy()
            prediction_data[f'probability_{token}'] = probabilities[:, index].numpy()
        pd.DataFrame(prediction_data).to_csv(
            self.save_dir / 'oracle_regret_v4_predictions.csv', index=False
        )

        main_metrics = results['best_valid_mae']['metrics']
        oracle_space = fusion_metrics['MAE'] - oracle_metrics['MAE']
        recovered = fusion_metrics['MAE'] - main_metrics['MAE']
        summary = {
            'method': 'oracle_regret_distillation_v4',
            'best_epoch': int(best_epoch),
            'candidate_offsets': [float(value) for value in model.candidate_offsets.detach().cpu()],
            'best_valid_mae_policy': best_policy,
            'robust_valid_policy': robust_policy,
            'fusion_metrics': fusion_metrics,
            'inference_results': results,
            'hard_top1_metrics': hard_metrics,
            'codebook_oracle_metrics': oracle_metrics,
            'candidate_top1_accuracy': diagnostics['candidate_top1_accuracy'],
            'mean_candidate_regret': diagnostics['mean_candidate_regret'],
            'normalized_candidate_regret': diagnostics['normalized_candidate_regret'],
            'oracle_gap_recovery_ratio': recovered / oracle_space if oracle_space > 1e-12 else 0.0,
        }
        with open(
            self.save_dir / 'oracle_regret_v4_summary.json', 'w', encoding='utf-8'
        ) as file:
            json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
        logger.info(
            'V4 TEST fusion_MAE=%.4f routed_MAE=%.4f raw_MAE=%.4f '
            'hard_MAE=%.4f oracle_MAE=%.4f top1=%.4f regret=%.4f recovery=%.4f policy=%s',
            fusion_metrics['MAE'],
            main_metrics['MAE'],
            results['raw_soft']['metrics']['MAE'],
            hard_metrics['MAE'],
            oracle_metrics['MAE'],
            diagnostics['candidate_top1_accuracy'],
            diagnostics['mean_candidate_regret'],
            summary['oracle_gap_recovery_ratio'],
            best_policy,
        )
        return summary
