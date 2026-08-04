"""Utilities for the frozen DLF role-specialization and long-tail audit."""
from __future__ import annotations

import hashlib
import math
import re
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler


VERSION = "dlf_role_specialization_audit_v1"
METHOD = "DLF-FrozenRoleSpecializationAudit-v1"
OUTPUT_TAG = "dlf_role_specialization_audit_v1"
FORMAL_SEEDS = (1111, 1114)
MODES = ("LAV", "LA", "LV", "L")
REPRESENTATIONS = ("shared", "specific_present", "final_fusion")
NEUTRAL_TAU = 0.5
SENTIMENT_BINS = (-3, -2, -1, 0, 1, 2, 3)
SENTIMENT_EDGES = (-np.inf, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, np.inf)
INTENSITY_NAMES = ("neutral", "weak", "medium", "strong")
POLARITY_NAMES = ("negative", "neutral", "positive")
PROBE_C = 1.0
PROBE_MAX_ITER = 2000
PROBE_RANDOM_STATE = 0
ROLE_POSITIVE_COVERAGE = 0.75
ROLE_POLARITY_MEAN_ADVANTAGE = 0.01
ROLE_INTENSITY_MEAN_ADVANTAGE = 0.02
TAIL_IMBALANCE_RATIO = 3.0
WEIGHT_CAP = 3.0


def normalize_sample_id(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return "|".join(normalize_sample_id(item) for item in value)
    if torch.is_tensor(value):
        if value.numel() == 1:
            return str(value.item())
        return "|".join(str(item) for item in value.detach().cpu().view(-1).tolist())
    return str(value)


def parse_video_id(value) -> str:
    sample_id = normalize_sample_id(value)
    if "|" in sample_id:
        first = sample_id.split("|", 1)[0]
        if first:
            return first
    if "[" in sample_id:
        return sample_id.split("[", 1)[0]
    match = re.match(r"^(.*?)(?:[_-](?:seg)?\d+)$", sample_id, flags=re.IGNORECASE)
    if match and match.group(1):
        return match.group(1)
    return sample_id


def polarity_labels(values: Sequence[float], tau: float = NEUTRAL_TAU) -> np.ndarray:
    labels = np.asarray(values, dtype=np.float64).reshape(-1)
    result = np.ones(labels.size, dtype=np.int64)
    result[labels < -float(tau)] = 0
    result[labels > float(tau)] = 2
    return result


def intensity_labels(values: Sequence[float]) -> np.ndarray:
    absolute = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    return np.digitize(absolute, bins=np.asarray([0.5, 1.5, 2.5]), right=True).astype(np.int64)


def sentiment_bins(values: Sequence[float]) -> np.ndarray:
    labels = np.asarray(values, dtype=np.float64).reshape(-1)
    encoded = np.digitize(labels, bins=np.asarray(SENTIMENT_EDGES[1:-1]), right=False)
    return np.asarray(SENTIMENT_BINS, dtype=np.int64)[encoded]


def effective_number_weights(counts: Mapping[int, int]) -> pd.DataFrame:
    ordered = sorted(int(key) for key in counts)
    total = int(sum(int(counts[key]) for key in ordered))
    if total <= 1:
        raise ValueError("Effective-number weights require at least two samples.")
    beta = float(total - 1) / float(total)
    raw = []
    for key in ordered:
        count = int(counts[key])
        if count <= 0:
            raw.append(float("nan"))
        else:
            effective = (1.0 - beta ** count) / (1.0 - beta)
            raw.append(1.0 / effective)
    finite = np.asarray([value for value in raw if math.isfinite(value)], dtype=float)
    scale = 1.0 / float(finite.mean())
    normalized = [value * scale if math.isfinite(value) else value for value in raw]
    capped = [min(WEIGHT_CAP, value) if math.isfinite(value) else value for value in normalized]
    capped_finite = np.asarray([value for value in capped if math.isfinite(value)], dtype=float)
    capped_scale = 1.0 / float(capped_finite.mean())
    capped = [value * capped_scale if math.isfinite(value) else value for value in capped]
    return pd.DataFrame({
        "class": ordered,
        "count": [int(counts[key]) for key in ordered],
        "beta": beta,
        "effective_weight": normalized,
        "effective_weight_capped": capped,
        "weight_cap_before_renorm": WEIGHT_CAP,
    })


def tensor_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class RepresentationCapture(AbstractContextManager):
    """Capture fixed internal DLF representations with forward pre-hooks."""

    def __init__(self, model: torch.nn.Module):
        if not hasattr(model, "backbone"):
            raise ValueError("Representation audit expects MissingModalityWrapper.")
        self.model = model
        self.latest: MutableMapping[str, torch.Tensor] = {}
        self.handles = []
        modules = {
            "shared": model.backbone.proj1_c,
            "specific_l": model.backbone.proj1_l_high,
            "specific_v": model.backbone.proj1_v_high,
            "specific_a": model.backbone.proj1_a_high,
            "final_fusion": model.backbone.proj1,
        }
        for name, module in modules.items():
            self.handles.append(module.register_forward_pre_hook(self._hook(name)))

    def _hook(self, name):
        def callback(_module, inputs):
            if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
                raise RuntimeError("Unexpected representation hook input for {}.".format(name))
            self.latest[name] = inputs[0].detach().cpu().clone()
        return callback

    def take(self, mode: str) -> Dict[str, torch.Tensor]:
        required = {"shared", "specific_l", "specific_a", "specific_v", "final_fusion"}
        if set(self.latest) != required:
            raise RuntimeError("Incomplete representation capture: {}".format(sorted(self.latest)))
        present = {
            "LAV": ("specific_l", "specific_a", "specific_v"),
            "LA": ("specific_l", "specific_a"),
            "LV": ("specific_l", "specific_v"),
            "L": ("specific_l",),
        }
        if mode not in present:
            raise ValueError("Unknown modality mode: {}".format(mode))
        result = {name: value for name, value in self.latest.items()}
        result["specific_present"] = torch.cat(
            [self.latest[name] for name in present[mode]], dim=1
        )
        self.latest = {}
        return result

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        return False


@dataclass
class FixedPolarityProbe:
    scaler: StandardScaler
    classifier: LogisticRegression

    @classmethod
    def fit(cls, features: np.ndarray, labels: np.ndarray) -> "FixedPolarityProbe":
        scaler = StandardScaler()
        transformed = scaler.fit_transform(np.asarray(features, dtype=np.float64))
        labels = np.asarray(labels, dtype=np.int64)
        if np.unique(labels).size < 2:
            raise ValueError("Polarity probe requires at least two train classes.")
        classifier = LogisticRegression(
            C=PROBE_C,
            class_weight="balanced",
            max_iter=PROBE_MAX_ITER,
            random_state=PROBE_RANDOM_STATE,
            solver="lbfgs",
            multi_class="auto",
        )
        classifier.fit(transformed, labels)
        return cls(scaler=scaler, classifier=classifier)

    def predict(self, features: np.ndarray) -> np.ndarray:
        transformed = self.scaler.transform(np.asarray(features, dtype=np.float64))
        return self.classifier.predict(transformed).astype(np.int64)


@dataclass
class BinaryThresholdModel:
    constant: float | None
    classifier: LogisticRegression | None

    def probability(self, features: np.ndarray) -> np.ndarray:
        if self.classifier is None:
            return np.full(len(features), float(self.constant), dtype=np.float64)
        classes = list(self.classifier.classes_)
        probabilities = self.classifier.predict_proba(features)
        if 1 not in classes:
            return np.zeros(len(features), dtype=np.float64)
        return probabilities[:, classes.index(1)]


@dataclass
class FixedOrdinalProbe:
    scaler: StandardScaler
    thresholds: Tuple[BinaryThresholdModel, ...]

    @classmethod
    def fit(cls, features: np.ndarray, labels: np.ndarray) -> "FixedOrdinalProbe":
        scaler = StandardScaler()
        transformed = scaler.fit_transform(np.asarray(features, dtype=np.float64))
        labels = np.asarray(labels, dtype=np.int64)
        models: List[BinaryThresholdModel] = []
        for threshold in (0, 1, 2):
            binary = (labels > threshold).astype(np.int64)
            unique = np.unique(binary)
            if unique.size == 1:
                models.append(BinaryThresholdModel(constant=float(unique[0]), classifier=None))
                continue
            classifier = LogisticRegression(
                C=PROBE_C,
                class_weight="balanced",
                max_iter=PROBE_MAX_ITER,
                random_state=PROBE_RANDOM_STATE,
                solver="lbfgs",
            )
            classifier.fit(transformed, binary)
            models.append(BinaryThresholdModel(constant=None, classifier=classifier))
        return cls(scaler=scaler, thresholds=tuple(models))

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        transformed = self.scaler.transform(np.asarray(features, dtype=np.float64))
        probabilities = np.stack(
            [model.probability(transformed) for model in self.thresholds], axis=1
        )
        probabilities[:, 1] = np.minimum(probabilities[:, 0], probabilities[:, 1])
        probabilities[:, 2] = np.minimum(probabilities[:, 1], probabilities[:, 2])
        prediction = (probabilities >= 0.5).sum(axis=1).astype(np.int64)
        return prediction, probabilities


def polarity_probe_metrics(labels: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def ordinal_probe_metrics(labels: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    kappa = cohen_kappa_score(labels, predictions, weights="quadratic")
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "ordinal_mae": float(np.mean(np.abs(labels - predictions))),
        "quadratic_kappa": float(0.0 if not math.isfinite(float(kappa)) else kappa),
    }


def bin_risk_metrics(frame: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    required = {"label", "prediction", "sentiment_bin"}
    if not required.issubset(frame.columns):
        raise ValueError("Bin-risk frame lacks required columns.")
    local = frame.copy()
    local["absolute_error"] = np.abs(local.prediction.astype(float) - local.label.astype(float))
    true_polarity = polarity_labels(local.label.to_numpy(dtype=float))
    pred_polarity = polarity_labels(local.prediction.to_numpy(dtype=float))
    flip = ((true_polarity == 0) & (pred_polarity == 2)) | (
        (true_polarity == 2) & (pred_polarity == 0)
    )
    neutral_escape = (true_polarity == 1) & (pred_polarity != 1)
    local["polarity_flip"] = flip.astype(int)
    local["neutral_escape"] = neutral_escape.astype(int)
    rows = []
    for sentiment_bin in SENTIMENT_BINS:
        subset = local.loc[local.sentiment_bin.astype(int).eq(int(sentiment_bin))]
        if subset.empty:
            rows.append({
                "sentiment_bin": int(sentiment_bin),
                "count": 0,
                "mae": float("nan"),
                "polarity_flip_rate": float("nan"),
                "neutral_escape_rate": float("nan"),
            })
        else:
            rows.append({
                "sentiment_bin": int(sentiment_bin),
                "count": int(len(subset)),
                "mae": float(subset.absolute_error.mean()),
                "polarity_flip_rate": float(subset.polarity_flip.mean()),
                "neutral_escape_rate": float(subset.neutral_escape.mean()),
            })
    bins = pd.DataFrame(rows)
    finite_mae = bins.mae[np.isfinite(bins.mae.astype(float))].astype(float)
    summary = {
        "overall_mae": float(local.absolute_error.mean()),
        "macro_bin_mae": float(finite_mae.mean()),
        "worst_bin_mae": float(finite_mae.max()),
        "polarity_flip_rate": float(local.polarity_flip.mean()),
        "neutral_escape_rate": float(local.neutral_escape.mean()),
    }
    return bins, summary


def role_alignment_gate(comparisons: pd.DataFrame) -> Dict[str, object]:
    required = {"Seed", "Mode", "polarity_advantage", "intensity_advantage"}
    if not required.issubset(comparisons.columns):
        raise ValueError("Role-comparison table is incomplete.")
    values = comparisons.copy()
    polarity_positive = values.polarity_advantage.astype(float) > 0.0
    intensity_positive = values.intensity_advantage.astype(float) > 0.0
    lav = values.loc[values.Mode.astype(str).eq("LAV")]
    checks = {
        "both_lav_seeds_polarity_positive": bool(
            len(lav) == len(FORMAL_SEEDS)
            and (lav.polarity_advantage.astype(float) > 0.0).all()
        ),
        "both_lav_seeds_intensity_positive": bool(
            len(lav) == len(FORMAL_SEEDS)
            and (lav.intensity_advantage.astype(float) > 0.0).all()
        ),
        "polarity_positive_coverage_ge_0p75": bool(
            float(polarity_positive.mean()) >= ROLE_POSITIVE_COVERAGE
        ),
        "intensity_positive_coverage_ge_0p75": bool(
            float(intensity_positive.mean()) >= ROLE_POSITIVE_COVERAGE
        ),
        "mean_polarity_advantage_ge_0p01": bool(
            float(values.polarity_advantage.mean()) >= ROLE_POLARITY_MEAN_ADVANTAGE
        ),
        "mean_intensity_advantage_ge_0p02": bool(
            float(values.intensity_advantage.mean()) >= ROLE_INTENSITY_MEAN_ADVANTAGE
        ),
    }
    passed = bool(all(checks.values()))
    weak_positive = bool(
        float(values.polarity_advantage.mean()) > 0.0
        and float(values.intensity_advantage.mean()) > 0.0
    )
    if passed:
        status = "ROLE_ALIGNMENT_SUPPORTED"
    elif weak_positive:
        status = "PARTIAL_ROLE_ALIGNMENT_NEEDS_REVIEW"
    else:
        status = "ROLE_ALIGNMENT_NOT_SUPPORTED"
    return {
        "passed": passed,
        "status": status,
        "checks": checks,
        "mean_polarity_advantage": float(values.polarity_advantage.mean()),
        "mean_intensity_advantage": float(values.intensity_advantage.mean()),
        "polarity_positive_coverage": float(polarity_positive.mean()),
        "intensity_positive_coverage": float(intensity_positive.mean()),
        "comparison_count": int(len(values)),
        "required_positive_coverage": ROLE_POSITIVE_COVERAGE,
        "required_mean_polarity_advantage": ROLE_POLARITY_MEAN_ADVANTAGE,
        "required_mean_intensity_advantage": ROLE_INTENSITY_MEAN_ADVANTAGE,
    }


def long_tail_status(distribution: pd.DataFrame, dataset: str) -> Dict[str, object]:
    local = distribution.loc[
        distribution.Dataset.astype(str).eq(str(dataset))
        & distribution.Split.astype(str).eq("train")
        & distribution.LabelFamily.astype(str).eq("sentiment_7")
    ]
    positive = local.loc[local.Count.astype(int) > 0]
    if len(positive) < 2:
        raise ValueError("Long-tail status requires at least two occupied bins.")
    counts = positive.Count.astype(int).to_numpy()
    ratio = float(counts.max() / counts.min())
    occupied = int(len(positive))
    tail_present = bool(ratio >= TAIL_IMBALANCE_RATIO)
    return {
        "dataset": str(dataset),
        "tail_present": tail_present,
        "imbalance_ratio_max_over_min": ratio,
        "occupied_sentiment_bins": occupied,
        "required_ratio": TAIL_IMBALANCE_RATIO,
        "max_count": int(counts.max()),
        "min_nonzero_count": int(counts.min()),
    }
