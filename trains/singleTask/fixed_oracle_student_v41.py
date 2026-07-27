import copy
import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from utils import setup_seed
from .backbone_crossfit import _video_group_id


logger = logging.getLogger('MMSA')


class FixedOracleEnergyStudent(nn.Module):
    """Candidate-conditioned student trained on fixed OOF anchors and teachers."""

    def __init__(
        self,
        input_dim,
        offsets,
        hidden_dim=192,
        adapter_dim=128,
        candidate_dim=64,
        dropout=0.20,
    ):
        super().__init__()
        self.register_buffer(
            'offsets', torch.tensor(tuple(offsets), dtype=torch.float32)
        )
        self.context = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
        )
        self.adapter = nn.Sequential(
            nn.Linear(int(hidden_dim), int(adapter_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(adapter_dim), int(hidden_dim)),
        )
        self.candidate = nn.Sequential(
            nn.Linear(6, int(candidate_dim)),
            nn.GELU(),
            nn.Linear(int(candidate_dim), int(candidate_dim)),
            nn.GELU(),
        )
        self.context_to_candidate = nn.Linear(int(hidden_dim), int(candidate_dim))
        self.energy = nn.Sequential(
            nn.Linear(int(hidden_dim) + 2 * int(candidate_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, features, anchor):
        context = self.context(features)
        context = context + self.adapter(context)
        candidate_context = self.context_to_candidate(context)

        batch_size = anchor.size(0)
        offsets = self.offsets.view(1, -1).expand(batch_size, -1)
        anchor_matrix = anchor.view(-1, 1).expand_as(offsets)
        candidate_values = anchor_matrix + offsets
        descriptor = torch.stack([
            candidate_values,
            offsets,
            offsets.abs(),
            offsets.sign(),
            anchor_matrix,
            anchor_matrix.abs(),
        ], dim=-1)
        candidate = self.candidate(descriptor)
        interaction = candidate * candidate_context.unsqueeze(1)
        logits = self.energy(torch.cat([
            context.unsqueeze(1).expand(-1, offsets.size(1), -1),
            candidate,
            interaction,
        ], dim=-1)).squeeze(-1)
        return {
            'logits': logits,
            'candidate_values': candidate_values,
        }


def fixed_oracle_teacher(anchor, labels, offsets, temperature=0.10):
    offset_tensor = torch.as_tensor(
        offsets, dtype=anchor.dtype, device=anchor.device
    ).view(1, -1)
    candidates = anchor + offset_tensor
    errors = torch.abs(candidates - labels)
    probabilities = F.softmax(
        -errors / max(float(temperature), 1e-4), dim=1
    )
    oracle_index = errors.argmin(dim=1)
    return {
        'candidate_values': candidates,
        'errors': errors,
        'probabilities': probabilities,
        'oracle_index': oracle_index,
    }


def regret_ranking_loss(logits, errors):
    error_i = errors.unsqueeze(2)
    error_j = errors.unsqueeze(1)
    better = error_i < error_j
    regret = (error_j - error_i).clamp_min(0.0)
    logit_i = logits.unsqueeze(2)
    logit_j = logits.unsqueeze(1)
    losses = F.softplus(logit_j - logit_i) * regret
    denominator = regret[better].sum().clamp_min(1e-8)
    return losses[better].sum() / denominator


def fixed_oracle_losses(
    output,
    anchor,
    labels,
    offsets,
    teacher_temperature=0.10,
    student_temperature=1.0,
):
    teacher = fixed_oracle_teacher(
        anchor, labels, offsets, teacher_temperature
    )
    log_probabilities = F.log_softmax(
        output['logits'] / max(float(student_temperature), 1e-4),
        dim=1,
    )
    probabilities = log_probabilities.exp()
    distill = -(teacher['probabilities'] * log_probabilities).sum(dim=1).mean()
    expected_mae = (
        probabilities * teacher['errors'].detach()
    ).sum(dim=1).mean()
    ranking = regret_ranking_loss(
        output['logits'], teacher['errors'].detach()
    )

    predicted_cdf = probabilities.cumsum(dim=1)
    teacher_cdf = teacher['probabilities'].detach().cumsum(dim=1)
    ordinal_emd = torch.abs(predicted_cdf - teacher_cdf).mean()

    offsets_tensor = torch.as_tensor(
        offsets, dtype=anchor.dtype, device=anchor.device
    ).view(1, -1)
    predicted_offset = (probabilities * offsets_tensor).sum(dim=1, keepdim=True)
    target_offset = (labels - anchor).clamp(
        min=float(min(offsets)), max=float(max(offsets))
    )
    offset_regression = F.smooth_l1_loss(predicted_offset, target_offset)

    hard_index = output['logits'].argmax(dim=1)
    oracle_index = teacher['oracle_index']
    hard_error = teacher['errors'].gather(
        1, hard_index.view(-1, 1)
    ).view(-1)
    oracle_error = teacher['errors'].min(dim=1).values
    anchor_error = torch.abs(anchor - labels).view(-1)
    regret = hard_error - oracle_error
    oracle_gap = (anchor_error - oracle_error).clamp_min(0.0)
    return {
        'distill': distill,
        'expected_mae': expected_mae,
        'ranking': ranking,
        'ordinal_emd': ordinal_emd,
        'offset_regression': offset_regression,
        'top1_accuracy': (hard_index == oracle_index).float().mean(),
        'within_one_accuracy': (
            (hard_index - oracle_index).abs() <= 1
        ).float().mean(),
        'mean_regret': regret.mean(),
        'normalized_regret': (
            regret.clamp_min(0.0).sum() / oracle_gap.sum().clamp_min(1e-8)
        ),
    }


@torch.no_grad()
def student_metrics(output, anchor, labels, offsets, temperature=1.0):
    teacher = fixed_oracle_teacher(anchor, labels, offsets)
    probabilities = F.softmax(
        output['logits'] / max(float(temperature), 1e-4), dim=1
    )
    hard_index = probabilities.argmax(dim=1)
    oracle_index = teacher['oracle_index']
    hard_prediction = teacher['candidate_values'].gather(
        1, hard_index.view(-1, 1)
    )
    soft_prediction = (
        probabilities * teacher['candidate_values']
    ).sum(dim=1, keepdim=True)
    hard_error = torch.abs(hard_prediction - labels).view(-1)
    oracle_error = teacher['errors'].min(dim=1).values
    anchor_error = torch.abs(anchor - labels).view(-1)
    top2 = torch.topk(probabilities, k=min(2, probabilities.size(1)), dim=1).indices
    top2_hit = (top2 == oracle_index.view(-1, 1)).any(dim=1)
    predicted_offset = (
        probabilities
        * torch.as_tensor(offsets, dtype=anchor.dtype).view(1, -1)
    ).sum(dim=1)
    target_offset = (labels - anchor).view(-1).clamp(
        min=float(min(offsets)), max=float(max(offsets))
    )
    regret = hard_error - oracle_error
    return {
        'soft_mae': float(torch.abs(soft_prediction - labels).mean().item()),
        'hard_mae': float(hard_error.mean().item()),
        'anchor_mae': float(anchor_error.mean().item()),
        'oracle_mae': float(oracle_error.mean().item()),
        'top1_accuracy': float((hard_index == oracle_index).float().mean().item()),
        'top2_accuracy': float(top2_hit.float().mean().item()),
        'within_one_accuracy': float(
            ((hard_index - oracle_index).abs() <= 1).float().mean().item()
        ),
        'offset_mae': float(torch.abs(predicted_offset - target_offset).mean().item()),
        'mean_regret': float(regret.mean().item()),
        'normalized_regret': float(
            regret.clamp_min(0.0).sum().item()
            / (anchor_error - oracle_error).clamp_min(0.0).sum().clamp_min(1e-8).item()
        ),
    }


def _group_validation_split(sample_ids, fraction, seed):
    groups = {}
    for index, sample_id in enumerate(sample_ids):
        groups.setdefault(_video_group_id(sample_id), []).append(index)
    generator = torch.Generator().manual_seed(int(seed))
    names = list(groups)
    order = torch.randperm(len(names), generator=generator).tolist()
    names = [names[index] for index in order]
    target = max(1, int(round(len(sample_ids) * float(fraction))))
    valid, count = [], 0
    for name in names:
        if count >= target and valid:
            break
        valid.extend(groups[name])
        count += len(groups[name])
    valid_set = set(valid)
    train = [index for index in range(len(sample_ids)) if index not in valid_set]
    if not train or not valid:
        raise RuntimeError('V4.1 group validation split produced an empty partition.')
    return torch.tensor(train, dtype=torch.long), torch.tensor(valid, dtype=torch.long)


def _feature_stats(features):
    return {
        'mean': features.mean(dim=0, keepdim=True),
        'std': features.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-4),
    }


def normalize_features(features, stats):
    return (features - stats['mean']) / stats['std']


@torch.no_grad()
def predict_student(model, features, anchor, device, batch_size=512):
    model.eval()
    logits = []
    loader = DataLoader(
        TensorDataset(features, anchor),
        batch_size=int(batch_size),
        shuffle=False,
    )
    for batch_features, batch_anchor in loader:
        output = model(
            batch_features.to(device), batch_anchor.to(device)
        )
        logits.append(output['logits'].detach().cpu())
    return {'logits': torch.cat(logits, dim=0), 'anchor': anchor.clone()}


def _train_model(
    features,
    anchor,
    labels,
    offsets,
    train_indices,
    device,
    seed,
    epochs,
    learning_rate,
    hidden_dim,
    adapter_dim,
    candidate_dim,
    dropout,
    batch_size,
    teacher_temperature,
    student_temperature,
    distill_weight,
    expected_mae_weight,
    ranking_weight,
    ordinal_emd_weight,
    offset_weight,
    validation_indices=None,
    patience=8,
):
    setup_seed(seed)
    stats = _feature_stats(features[train_indices])
    normalized = normalize_features(features, stats)
    model = FixedOracleEnergyStudent(
        normalized.size(1),
        offsets,
        hidden_dim=hidden_dim,
        adapter_dim=adapter_dim,
        candidate_dim=candidate_dim,
        dropout=dropout,
    ).to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=1e-4
    )
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        TensorDataset(
            normalized[train_indices],
            anchor[train_indices],
            labels[train_indices],
        ),
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
    )
    best_state, best_epoch, best_objective = None, 0, float('inf')
    history = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        totals = {}
        for batch_features, batch_anchor, batch_labels in loader:
            optimizer.zero_grad()
            output = model(
                batch_features.to(device), batch_anchor.to(device)
            )
            losses = fixed_oracle_losses(
                output,
                batch_anchor.to(device),
                batch_labels.to(device),
                offsets,
                teacher_temperature=teacher_temperature,
                student_temperature=student_temperature,
            )
            total = (
                float(distill_weight) * losses['distill']
                + float(expected_mae_weight) * losses['expected_mae']
                + float(ranking_weight) * losses['ranking']
                + float(ordinal_emd_weight) * losses['ordinal_emd']
                + float(offset_weight) * losses['offset_regression']
            )
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            values = {'loss': total, **losses}
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().item())

        row = {
            'epoch': int(epoch),
            **{
                f'train_{key}': value / max(1, len(loader))
                for key, value in totals.items()
            },
        }
        if validation_indices is not None:
            prediction = predict_student(
                model,
                normalized[validation_indices],
                anchor[validation_indices],
                device,
                batch_size=batch_size,
            )
            metrics = student_metrics(
                prediction,
                anchor[validation_indices],
                labels[validation_indices],
                offsets,
                temperature=student_temperature,
            )
            row.update({f'valid_{key}': value for key, value in metrics.items()})
            objective = (
                metrics['soft_mae']
                + 0.15 * metrics['normalized_regret']
                + 0.05 * metrics['offset_mae']
            )
            row['valid_objective'] = float(objective)
            if objective < best_objective - 1e-6:
                best_objective = float(objective)
                best_epoch = int(epoch)
                best_state = copy.deepcopy(model.state_dict())
            if epoch - best_epoch >= int(patience):
                history.append(row)
                break
        history.append(row)

    if validation_indices is None:
        return model, stats, history, int(epochs)
    if best_state is None:
        raise RuntimeError('V4.1 student selection produced no checkpoint.')
    model.load_state_dict(best_state)
    return model, stats, history, int(best_epoch)


def train_fixed_oracle_students_v41(
    anchor_cache,
    device,
    save_dir,
    seed,
    offsets,
    epochs=80,
    learning_rate=3e-4,
    hidden_dim=192,
    adapter_dim=128,
    candidate_dim=64,
    dropout=0.20,
    batch_size=128,
    teacher_temperature=0.10,
    student_temperature=1.0,
    distill_weight=1.0,
    expected_mae_weight=0.50,
    ranking_weight=0.20,
    ordinal_emd_weight=0.50,
    offset_weight=0.50,
    validation_fraction=0.20,
    patience=10,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    train_cache = anchor_cache['train']
    fold_count = int(anchor_cache['outer_folds'])
    valid_outputs, test_outputs = [], []
    selection_history, final_history, internal_rows = [], [], []

    for fold_index in range(1, fold_count + 1):
        global_indices = torch.nonzero(
            train_cache['source_fold_ids'] == fold_index,
            as_tuple=False,
        ).view(-1)
        fold_features = train_cache['model_feature'][global_indices]
        fold_anchor = train_cache['anchor'][global_indices]
        fold_labels = train_cache['labels'][global_indices]
        fold_ids = [train_cache['sample_ids'][index] for index in global_indices.tolist()]
        train_local, valid_local = _group_validation_split(
            fold_ids,
            validation_fraction,
            seed + 701 * fold_index,
        )
        logger.info(
            'V4.1 selecting student fold %d/%d train=%d internal_valid=%d',
            fold_index, fold_count, len(train_local), len(valid_local),
        )
        selection_model, selection_stats, history, best_epoch = _train_model(
            fold_features,
            fold_anchor,
            fold_labels,
            offsets,
            train_local,
            device,
            seed + 1009 * fold_index,
            epochs,
            learning_rate,
            hidden_dim,
            adapter_dim,
            candidate_dim,
            dropout,
            batch_size,
            teacher_temperature,
            student_temperature,
            distill_weight,
            expected_mae_weight,
            ranking_weight,
            ordinal_emd_weight,
            offset_weight,
            validation_indices=valid_local,
            patience=patience,
        )
        selection_history.extend(
            {'fold': fold_index, **row} for row in history
        )
        internal_prediction = predict_student(
            selection_model,
            normalize_features(fold_features[valid_local], selection_stats),
            fold_anchor[valid_local],
            device,
            batch_size=batch_size,
        )
        internal_rows.append({
            'fold': int(fold_index),
            'selected_epoch': int(best_epoch),
            'train_count': int(len(train_local)),
            'internal_valid_count': int(len(valid_local)),
            **student_metrics(
                internal_prediction,
                fold_anchor[valid_local],
                fold_labels[valid_local],
                offsets,
                temperature=student_temperature,
            ),
        })
        del selection_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        all_local = torch.arange(len(global_indices))
        final_model, final_stats, history, _ = _train_model(
            fold_features,
            fold_anchor,
            fold_labels,
            offsets,
            all_local,
            device,
            seed + 5003 * fold_index,
            max(1, best_epoch),
            learning_rate,
            hidden_dim,
            adapter_dim,
            candidate_dim,
            dropout,
            batch_size,
            teacher_temperature,
            student_temperature,
            distill_weight,
            expected_mae_weight,
            ranking_weight,
            ordinal_emd_weight,
            offset_weight,
            validation_indices=None,
            patience=patience,
        )
        final_history.extend({'fold': fold_index, **row} for row in history)
        valid_view = anchor_cache['valid_views'][fold_index - 1]
        test_view = anchor_cache['test_views'][fold_index - 1]
        valid_outputs.append(predict_student(
            final_model,
            normalize_features(valid_view['model_feature'], final_stats),
            valid_view['anchor'],
            device,
            batch_size=batch_size,
        ))
        test_outputs.append(predict_student(
            final_model,
            normalize_features(test_view['model_feature'], final_stats),
            test_view['anchor'],
            device,
            batch_size=batch_size,
        ))
        torch.save({
            'state_dict': final_model.state_dict(),
            'feature_stats': final_stats,
            'offsets': [float(value) for value in offsets],
            'fold': int(fold_index),
            'selected_epoch': int(best_epoch),
            'input_dim': int(fold_features.size(1)),
        }, save_dir / f'v41_fixed_student_fold_{fold_index}.pth')
        del final_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pd.DataFrame(selection_history).to_csv(
        save_dir / 'v41_student_selection_history.csv', index=False
    )
    pd.DataFrame(final_history).to_csv(
        save_dir / 'v41_student_final_history.csv', index=False
    )
    pd.DataFrame(internal_rows).to_csv(
        save_dir / 'v41_internal_generalization.csv', index=False
    )
    return {
        'valid_outputs': valid_outputs,
        'test_outputs': test_outputs,
        'internal_rows': internal_rows,
    }
