"""Strict router-free modality residual team utilities for V9.28.

V9.28 treats the fold-local V9.19 text branch as the immutable anchor and asks
whether audio and vision contain deployment-stable residual information. The
residual heads never receive text tokens or text embeddings. They only receive
pooled features from their assigned non-text modality plus the scalar text
anchor prediction. Per-sample routing and sample-dependent mixture weights are
forbidden; only one set of non-negative global shrinkage coefficients is fitted
on strict inner OOF corrections and then frozen for the outer holdout.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset

AUDIT_VERSION = "modality_residual_team_v928_v1"
TEXT_FEATURE_KEY = "logits_l_hetero"
HEAD_NAMES = ("audio", "vision", "audio_visual")
PRIMARY_STRATEGY = "text_plus_all_residuals"


@dataclass(frozen=True)
class ResidualHeadConfigV928:
    temporal_bins: int = 4
    hidden_dim: int = 64
    dropout: float = 0.15
    correction_max: float = 0.75
    epochs: int = 120
    batch_size: int = 64
    learning_rate: float = 2e-3
    weight_decay: float = 1e-3
    correction_l1: float = 0.01

    def validate(self) -> None:
        if self.temporal_bins < 1:
            raise ValueError("temporal_bins must be positive")
        if self.hidden_dim < 8:
            raise ValueError("hidden_dim is too small")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if self.correction_max <= 0.0:
            raise ValueError("correction_max must be positive")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if min(self.learning_rate, self.weight_decay, self.correction_l1) < 0:
            raise ValueError("optimization values must be non-negative")


@dataclass(frozen=True)
class ResidualTeamConfigV928:
    head: ResidualHeadConfigV928 = ResidualHeadConfigV928()
    coefficient_l2: float = 0.01
    coefficient_upper_bound: float = 1.0
    bootstrap_repetitions: int = 2000
    bootstrap_seed: int = 1111
    required_gain_vs_text: float = 0.003
    required_positive_outer_folds: int = 4
    max_worst_fold_degradation: float = 0.005
    required_gain_ci_low: float = 0.0
    competitive_margin_vs_v921: float = 0.0

    def validate(self) -> None:
        self.head.validate()
        if self.coefficient_l2 < 0:
            raise ValueError("coefficient_l2 must be non-negative")
        if self.coefficient_upper_bound <= 0:
            raise ValueError("coefficient_upper_bound must be positive")
        if self.bootstrap_repetitions < 100:
            raise ValueError("bootstrap_repetitions is too small")
        if self.required_positive_outer_folds < 1:
            raise ValueError("required_positive_outer_folds must be positive")


class BoundedResidualMLP(nn.Module):
    """Low-capacity correction head with a hard output range."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        correction_max: float,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.correction_max = float(correction_max)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout) * 0.5),
            nn.Linear(int(hidden_dim), 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.size(1) != self.input_dim:
            raise ValueError(
                f"expected [N,{self.input_dim}] features, got {tuple(features.shape)}"
            )
        return self.correction_max * torch.tanh(self.network(features))


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _valid_prefix(sequence: np.ndarray, length: int | None) -> np.ndarray:
    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"sequence must be [time,dim], got {values.shape}")
    if length is not None:
        count = max(1, min(int(length), len(values)))
        return values[:count]
    nonzero = np.any(np.abs(values) > 1e-12, axis=1)
    if bool(nonzero.any()):
        return values[: int(np.flatnonzero(nonzero)[-1]) + 1]
    return values[:1]


def summarize_sequence(
    sequence: np.ndarray,
    length: int | None,
    temporal_bins: int,
) -> np.ndarray:
    """Create deterministic compact features without using text information."""
    values = _valid_prefix(sequence, length)
    parts = [values.mean(axis=0), values.std(axis=0)]
    boundaries = np.linspace(0, len(values), int(temporal_bins) + 1)
    for index in range(int(temporal_bins)):
        start = int(np.floor(boundaries[index]))
        end = int(np.floor(boundaries[index + 1]))
        if end <= start:
            end = min(len(values), start + 1)
        chunk = values[start:end]
        if len(chunk) == 0:
            chunk = values[-1:]
        parts.append(chunk.mean(axis=0))
    duration = np.asarray(
        [float(len(values)) / max(1, len(sequence))], dtype=np.float32
    )
    summary = np.concatenate([*parts, duration], axis=0).astype(np.float32)
    if not np.isfinite(summary).all():
        raise FloatingPointError("non-finite modality summary")
    return summary


def build_modality_summary_matrix(
    dataset,
    dataset_indices: Sequence[int],
    modality: str,
    temporal_bins: int,
) -> np.ndarray:
    if modality not in {"audio", "vision", "audio_visual"}:
        raise ValueError(f"unsupported modality: {modality}")
    indices = [int(value) for value in dataset_indices]
    audio_lengths = getattr(dataset, "audio_lengths", None)
    vision_lengths = getattr(dataset, "vision_lengths", None)
    rows = []
    for index in indices:
        local = []
        if modality in {"audio", "audio_visual"}:
            length = None if audio_lengths is None else int(audio_lengths[index])
            local.append(
                summarize_sequence(dataset.audio[index], length, temporal_bins)
            )
        if modality in {"vision", "audio_visual"}:
            length = None if vision_lengths is None else int(vision_lengths[index])
            local.append(
                summarize_sequence(dataset.vision[index], length, temporal_bins)
            )
        rows.append(np.concatenate(local).astype(np.float32))
    matrix = np.stack(rows, axis=0)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise FloatingPointError("invalid modality summary matrix")
    return matrix


def append_anchor_features(
    modality_features: np.ndarray,
    anchor_prediction: Sequence[float],
) -> np.ndarray:
    features = np.asarray(modality_features, dtype=np.float64)
    anchor = np.asarray(anchor_prediction, dtype=np.float64).reshape(-1)
    if features.ndim != 2 or len(features) != len(anchor):
        raise ValueError("modality features and anchor do not align")
    anchor_columns = np.column_stack([anchor, np.abs(anchor), anchor**2])
    result = np.column_stack([features, anchor_columns])
    if not np.isfinite(result).all():
        raise FloatingPointError("non-finite residual-head input")
    return result


def fit_standardizer(features: np.ndarray) -> Dict[str, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return {"mean": mean, "scale": scale}


def apply_standardizer(
    features: np.ndarray,
    standardizer: Mapping[str, np.ndarray],
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    result = (values - np.asarray(standardizer["mean"])) / np.asarray(
        standardizer["scale"]
    )
    if not np.isfinite(result).all():
        raise FloatingPointError("standardized residual features are non-finite")
    return result.astype(np.float32)


def fit_residual_head(
    modality_features: np.ndarray,
    anchor_prediction: Sequence[float],
    labels: Sequence[float],
    config: ResidualHeadConfigV928,
    device,
    seed: int,
) -> Dict[str, object]:
    config.validate()
    seed_everything(seed)
    anchor = np.asarray(anchor_prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(labels, dtype=np.float64).reshape(-1) - anchor
    raw = append_anchor_features(modality_features, anchor)
    standardizer = fit_standardizer(raw)
    x = torch.tensor(
        apply_standardizer(raw, standardizer), dtype=torch.float32
    )
    y = torch.tensor(target, dtype=torch.float32).view(-1, 1)
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=min(int(config.batch_size), len(x)),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
        drop_last=False,
    )
    model = BoundedResidualMLP(
        x.size(1),
        config.hidden_dim,
        config.dropout,
        config.correction_max,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    history = []
    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        total_loss = 0.0
        total_task = 0.0
        total_count = 0
        for batch_x, batch_y in loader:
            prediction = model(batch_x.to(device))
            task = F.smooth_l1_loss(prediction, batch_y.to(device), beta=0.20)
            magnitude = prediction.abs().mean()
            loss = task + float(config.correction_l1) * magnitude
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            count = len(batch_x)
            total_loss += float(loss.detach().item()) * count
            total_task += float(task.detach().item()) * count
            total_count += count
        history.append(
            {
                "epoch": epoch,
                "loss": total_loss / max(1, total_count),
                "task_loss": total_task / max(1, total_count),
            }
        )
    model.eval()
    return {
        "model": model,
        "standardizer": standardizer,
        "config": asdict(config),
        "history": history,
        "training_count": len(x),
    }


@torch.no_grad()
def predict_residual_head(
    bundle: Mapping[str, object],
    modality_features: np.ndarray,
    anchor_prediction: Sequence[float],
    device,
    batch_size: int = 256,
) -> np.ndarray:
    raw = append_anchor_features(modality_features, anchor_prediction)
    values = torch.tensor(
        apply_standardizer(raw, bundle["standardizer"]), dtype=torch.float32
    )
    model = bundle["model"]
    model.eval()
    outputs = []
    for start in range(0, len(values), int(batch_size)):
        outputs.append(model(values[start : start + batch_size].to(device)).cpu())
    result = torch.cat(outputs, dim=0).view(-1).numpy().astype(np.float64)
    if not np.isfinite(result).all():
        raise FloatingPointError("residual prediction is non-finite")
    return result


def crossfit_residual_predictions(
    modality_features: np.ndarray,
    anchor_prediction: Sequence[float],
    labels: Sequence[float],
    fold_index: Sequence[int],
    config: ResidualHeadConfigV928,
    device,
    seed: int,
) -> Dict[str, object]:
    features = np.asarray(modality_features)
    anchor = np.asarray(anchor_prediction, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    folds = np.asarray(fold_index, dtype=np.int64).reshape(-1)
    if not (len(features) == len(anchor) == len(y) == len(folds)):
        raise ValueError("crossfit residual inputs do not align")
    unique_folds = sorted(int(value) for value in np.unique(folds))
    prediction = np.full(len(y), np.nan, dtype=np.float64)
    rows = []
    for fold in unique_folds:
        holdout = np.flatnonzero(folds == fold)
        training = np.flatnonzero(folds != fold)
        if len(holdout) == 0 or len(training) == 0:
            raise RuntimeError("empty residual crossfit partition")
        bundle = fit_residual_head(
            features[training],
            anchor[training],
            y[training],
            config,
            device,
            int(seed) + 1009 * (fold + 1),
        )
        correction = predict_residual_head(
            bundle, features[holdout], anchor[holdout], device
        )
        prediction[holdout] = correction
        residual = y[holdout] - anchor[holdout]
        correlation = spearmanr(correction, residual).statistic
        rows.append(
            {
                "inner_fold": fold,
                "training_count": len(training),
                "holdout_count": len(holdout),
                "correction_mean": float(correction.mean()),
                "correction_abs_mean": float(np.abs(correction).mean()),
                "residual_spearman": (
                    float(correlation) if np.isfinite(correlation) else 0.0
                ),
            }
        )
    if not np.isfinite(prediction).all():
        raise FloatingPointError("crossfit residual prediction is incomplete")
    return {"prediction": prediction, "fold_rows": rows}


def fit_nonnegative_shrinkage(
    corrections: np.ndarray,
    anchor_prediction: Sequence[float],
    labels: Sequence[float],
    coefficient_l2: float,
    upper_bound: float,
) -> np.ndarray:
    matrix = np.asarray(corrections, dtype=np.float64)
    anchor = np.asarray(anchor_prediction, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or len(matrix) != len(anchor) or len(anchor) != len(y):
        raise ValueError("coefficient fit inputs do not align")
    if matrix.shape[1] == 0:
        return np.zeros(0, dtype=np.float64)
    initial = np.zeros(matrix.shape[1], dtype=np.float64)

    def objective(weights: np.ndarray) -> float:
        prediction = anchor + matrix @ weights
        return float(
            np.abs(prediction - y).mean()
            + float(coefficient_l2) * np.square(weights).sum()
        )

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, float(upper_bound))] * matrix.shape[1],
        options={"maxiter": 2000, "ftol": 1e-12, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"residual shrinkage fit failed: {result.message}")
    weights = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, upper_bound)
    if objective(weights) > objective(initial) + 1e-8:
        raise RuntimeError("residual shrinkage worsened the zero-correction anchor")
    return weights


def strategy_definitions() -> Dict[str, tuple[int, ...]]:
    return {
        "text_anchor": (),
        "text_plus_audio": (0,),
        "text_plus_vision": (1,),
        "text_plus_audio_visual_head": (2,),
        "text_plus_audio_and_vision": (0, 1),
        PRIMARY_STRATEGY: (0, 1, 2),
    }


def fit_strategy_weights(
    corrections: np.ndarray,
    anchor_prediction: Sequence[float],
    labels: Sequence[float],
    config: ResidualTeamConfigV928,
) -> Dict[str, np.ndarray]:
    config.validate()
    matrix = np.asarray(corrections, dtype=np.float64)
    result = {}
    for name, columns in strategy_definitions().items():
        weights = np.zeros(matrix.shape[1], dtype=np.float64)
        if columns:
            local = fit_nonnegative_shrinkage(
                matrix[:, columns],
                anchor_prediction,
                labels,
                config.coefficient_l2,
                config.coefficient_upper_bound,
            )
            weights[list(columns)] = local
        result[name] = weights
    return result


def apply_strategy(
    anchor_prediction: Sequence[float],
    corrections: np.ndarray,
    weights: Sequence[float],
) -> np.ndarray:
    anchor = np.asarray(anchor_prediction, dtype=np.float64).reshape(-1)
    matrix = np.asarray(corrections, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[1] != len(weight):
        raise ValueError("correction matrix and weights do not align")
    prediction = anchor + matrix @ weight
    if not np.isfinite(prediction).all():
        raise FloatingPointError("strategy prediction is non-finite")
    return prediction


def prediction_metrics(
    prediction: Sequence[float],
    labels: Sequence[float],
    text_anchor: Sequence[float],
    v921_prediction: Sequence[float] | None = None,
) -> Dict[str, float]:
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    text = np.asarray(text_anchor, dtype=np.float64).reshape(-1)
    if not (len(pred) == len(y) == len(text)):
        raise ValueError("metric values do not align")
    error = np.abs(pred - y)
    text_error = np.abs(text - y)
    gain_text = text_error - error
    result = {
        "sample_count": len(y),
        "mae": float(error.mean()),
        "text_anchor_mae": float(text_error.mean()),
        "gain_vs_text": float(gain_text.mean()),
        "win_rate_vs_text": float((gain_text > 0.0).mean()),
        "harm_over_010_rate_vs_text": float((gain_text < -0.10).mean()),
        "large_gain_010_rate_vs_text": float((gain_text > 0.10).mean()),
    }
    if v921_prediction is not None:
        baseline = np.asarray(v921_prediction, dtype=np.float64).reshape(-1)
        if len(baseline) != len(y):
            raise ValueError("V9.21 prediction does not align")
        baseline_error = np.abs(baseline - y)
        gain = baseline_error - error
        result.update(
            {
                "v921_mae": float(baseline_error.mean()),
                "gain_vs_v921": float(gain.mean()),
                "win_rate_vs_v921": float((gain > 0.0).mean()),
                "harm_over_010_rate_vs_v921": float((gain < -0.10).mean()),
            }
        )
    return result


def serializable_bundle(bundle: Mapping[str, object]) -> Dict[str, object]:
    model = bundle["model"]
    return {
        "version": AUDIT_VERSION,
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "standardizer": {
            key: np.asarray(value, dtype=np.float64)
            for key, value in bundle["standardizer"].items()
        },
        "config": dict(bundle["config"]),
        "training_count": int(bundle["training_count"]),
        "input_dim": int(model.input_dim),
        "correction_max": float(model.correction_max),
    }


__all__ = [
    "AUDIT_VERSION",
    "TEXT_FEATURE_KEY",
    "HEAD_NAMES",
    "PRIMARY_STRATEGY",
    "ResidualHeadConfigV928",
    "ResidualTeamConfigV928",
    "BoundedResidualMLP",
    "summarize_sequence",
    "build_modality_summary_matrix",
    "append_anchor_features",
    "fit_residual_head",
    "predict_residual_head",
    "crossfit_residual_predictions",
    "fit_nonnegative_shrinkage",
    "strategy_definitions",
    "fit_strategy_weights",
    "apply_strategy",
    "prediction_metrics",
    "serializable_bundle",
]
