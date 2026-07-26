import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger('MMSA')


EXPERT_OUTPUT_KEYS = {
    'text': 'logits_l_hetero',
    'audio': 'logits_a_hetero',
    'video': 'logits_v_hetero',
    'common': 'logits_c',
    'fusion': 'output_logit',
}


def extract_expert_logits(model_output):
    """Return the five already-trained DLF prediction branches as experts."""
    if 'expert_logits' in model_output:
        return model_output['expert_logits']

    return {
        expert_name: model_output[output_key]
        for expert_name, output_key in EXPERT_OUTPUT_KEYS.items()
    }


def normalize_batch_ids(batch_ids):
    """Convert the default DataLoader-collated IDs into strings."""
    if batch_ids is None:
        return []
    if torch.is_tensor(batch_ids):
        return [str(item) for item in batch_ids.detach().cpu().tolist()]
    if isinstance(batch_ids, (list, tuple)):
        return [str(item) for item in batch_ids]
    return [str(batch_ids)]


def _to_serializable_metrics(metrics):
    return {key: float(value) for key, value in metrics.items()}


def analyze_expert_pool(expert_predictions, targets, metrics_fn, save_dir, sample_ids=None):
    """
    Evaluate DLF's existing prediction branches without changing model training.

    Outputs:
      - expert_summary.csv/json: metrics and per-expert win rate
      - expert_predictions.csv: per-sample predictions/errors/winner
      - expert_error_correlation.csv: pairwise absolute-error correlation

    Oracle is diagnostic only: it uses the ground-truth label to choose the
    lowest-error expert for each sample and must never be used at inference.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    targets = targets.detach().cpu().view(-1, 1)
    expert_names = list(expert_predictions.keys())
    predictions = {
        name: expert_predictions[name].detach().cpu().view(-1, 1)
        for name in expert_names
    }

    sample_count = targets.size(0)
    for name, pred in predictions.items():
        if pred.size(0) != sample_count:
            raise ValueError(
                f'Expert {name} has {pred.size(0)} predictions, expected {sample_count}.'
            )

    prediction_matrix = torch.cat([predictions[name] for name in expert_names], dim=1)
    absolute_errors = torch.abs(prediction_matrix - targets)
    winner_indices = torch.argmin(absolute_errors, dim=1)
    oracle_predictions = prediction_matrix.gather(1, winner_indices.unsqueeze(1))
    mean_predictions = prediction_matrix.mean(dim=1, keepdim=True)

    win_counts = torch.bincount(winner_indices, minlength=len(expert_names)).cpu().numpy()

    summary_rows = []
    summary_json = {}
    for index, name in enumerate(expert_names):
        metrics = _to_serializable_metrics(metrics_fn(predictions[name], targets))
        win_count = int(win_counts[index])
        row = {
            'expert': name,
            **metrics,
            'win_count': win_count,
            'win_rate': win_count / sample_count,
        }
        summary_rows.append(row)
        summary_json[name] = row

    mean_metrics = _to_serializable_metrics(metrics_fn(mean_predictions, targets))
    mean_row = {
        'expert': 'equal_mean',
        **mean_metrics,
        'win_count': np.nan,
        'win_rate': np.nan,
    }
    summary_rows.append(mean_row)
    summary_json['equal_mean'] = mean_row

    oracle_metrics = _to_serializable_metrics(metrics_fn(oracle_predictions, targets))
    oracle_row = {
        'expert': 'oracle',
        **oracle_metrics,
        'win_count': sample_count,
        'win_rate': 1.0,
    }
    summary_rows.append(oracle_row)
    summary_json['oracle'] = oracle_row

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(save_dir / 'expert_summary.csv', index=False)
    with open(save_dir / 'expert_summary.json', 'w', encoding='utf-8') as file:
        json.dump(summary_json, file, ensure_ascii=False, indent=2)

    if sample_ids is None or len(sample_ids) != sample_count:
        sample_ids = [str(index) for index in range(sample_count)]

    prediction_data = {
        'sample_id': sample_ids,
        'target': targets.view(-1).numpy(),
        'winner': [expert_names[index] for index in winner_indices.tolist()],
        'oracle_prediction': oracle_predictions.view(-1).numpy(),
        'equal_mean_prediction': mean_predictions.view(-1).numpy(),
    }
    for index, name in enumerate(expert_names):
        prediction_data[f'{name}_prediction'] = prediction_matrix[:, index].numpy()
        prediction_data[f'{name}_abs_error'] = absolute_errors[:, index].numpy()

    pd.DataFrame(prediction_data).to_csv(
        save_dir / 'expert_predictions.csv', index=False
    )

    error_array = absolute_errors.numpy()
    with np.errstate(invalid='ignore', divide='ignore'):
        error_correlation = np.corrcoef(error_array, rowvar=False)
    error_correlation = np.nan_to_num(error_correlation, nan=0.0)
    pd.DataFrame(
        error_correlation,
        index=expert_names,
        columns=expert_names,
    ).to_csv(save_dir / 'expert_error_correlation.csv')

    logger.info('Expert Pool V0 results saved to %s', save_dir)
    for row in summary_rows:
        metric_text = ' '.join(
            f'{key}={value:.4f}'
            for key, value in row.items()
            if key not in {'expert', 'win_count', 'win_rate'} and not pd.isna(value)
        )
        logger.info(
            'EXPERT %-10s %s win_rate=%s',
            row['expert'],
            metric_text,
            'N/A' if pd.isna(row['win_rate']) else f"{row['win_rate']:.4f}",
        )

    return summary_json
