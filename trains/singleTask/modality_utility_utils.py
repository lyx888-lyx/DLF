"""Deterministic helpers for the Stage 7A modality-utility audit.

These functions are analysis-only.  They create no optimizer and never mutate a
model.  The audit uses them for fixed derangements, paired bootstrap intervals,
input sensitivity, representation statistics, and the pre-registered A/B/C/D
classification.
"""
import hashlib
import json
import math
import random

import numpy as np
import torch


MODES = ("LAV", "LA", "LV", "L")
MODALITIES = ("A", "V", "AV")
UTILITY_MODALITIES = ("A", "V")
LABEL_BINS = ("[-3,-1)", "[-1,0)", "[0,1)", "[1,3]")
QUARTILE_NAMES = ("Q1_low", "Q2", "Q3", "Q4_high")


def stable_seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": [value.clone() for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def rng_states_equal(left, right):
    same = left["python"] == right["python"]
    same = same and np.array_equal(left["numpy"][1], right["numpy"][1])
    same = same and left["numpy"][2:] == right["numpy"][2:]
    same = same and torch.equal(left["torch"], right["torch"])
    if left["cuda"] is None or right["cuda"] is None:
        return same and left["cuda"] is right["cuda"]
    return same and len(left["cuda"]) == len(right["cuda"]) and all(
        torch.equal(a, b) for a, b in zip(left["cuda"], right["cuda"])
    )


def clone_parameters(model):
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters()}


def clone_buffers(model):
    return {name: value.detach().cpu().clone() for name, value in model.named_buffers()}


def tensor_maps_equal(left, right):
    return set(left) == set(right) and all(torch.equal(left[key], right[key]) for key in left)


def sattolo_derangement(size, seed):
    """Return a deterministic one-cycle derangement (therefore a fixed-point-free bijection)."""
    size = int(size)
    if size < 2:
        raise ValueError("A derangement requires at least two samples.")
    values = np.arange(size, dtype=np.int64)
    rng = np.random.RandomState(int(seed) % (2 ** 32))
    for index in range(size - 1, 0, -1):
        other = int(rng.randint(0, index))
        values[index], values[other] = values[other], values[index]
    validate_derangement(values, size)
    return values


def validate_derangement(mapping, size=None):
    mapping = np.asarray(mapping, dtype=np.int64)
    size = len(mapping) if size is None else int(size)
    if mapping.ndim != 1 or len(mapping) != size:
        raise ValueError("Derangement has the wrong shape.")
    if not np.array_equal(np.sort(mapping), np.arange(size)):
        raise ValueError("Derangement is not a bijection.")
    if np.any(mapping == np.arange(size)):
        raise ValueError("Derangement contains a fixed point.")
    return True


def permute_bound_fields(features, mapping, mask=None, lengths=None, extra_fields=None):
    """Apply one sample permutation to every field bound to a modality."""
    mapping = np.asarray(mapping, dtype=np.int64)
    validate_derangement(mapping, len(mapping))

    def take(value):
        if torch.is_tensor(value):
            return value.index_select(0, torch.as_tensor(mapping, device=value.device))
        return np.asarray(value)[mapping]

    result = {"features": take(features)}
    if mask is not None:
        result["mask"] = take(mask)
    if lengths is not None:
        result["lengths"] = take(lengths)
    for name, value in (extra_fields or {}).items():
        result[name] = take(value)
    return result


def bootstrap_summary(values, samples=2000, seed=270700):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "count": 0, "mean": float("nan"), "median": float("nan"),
            "std": float("nan"), "positive_fraction": float("nan"),
            "negative_fraction": float("nan"), "zero_fraction": float("nan"),
            "ci_low": float("nan"), "ci_high": float("nan"),
        }
    rng = np.random.RandomState(int(seed) % (2 ** 32))
    means = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        means[index] = x[rng.randint(0, len(x), len(x))].mean()
    return {
        "count": int(len(x)),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "std": float(x.std(ddof=0)),
        "positive_fraction": float((x > 0).mean()),
        "negative_fraction": float((x < 0).mean()),
        "zero_fraction": float((x == 0).mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def gain_values(pred_l, pred_with_modality, labels):
    return np.abs(np.asarray(pred_l) - np.asarray(labels)) - np.abs(
        np.asarray(pred_with_modality) - np.asarray(labels)
    )


def shuffle_damage(correct, shuffled, labels):
    return np.abs(np.asarray(shuffled) - np.asarray(labels)) - np.abs(
        np.asarray(correct) - np.asarray(labels)
    )


def infer_padding_mask(features, lengths=None):
    """Return [batch,time] validity; aligned MOSI uses non-zero feature rows."""
    if features.ndim != 3:
        raise ValueError("Expected [batch,time,feature] input.")
    if lengths is None:
        return features.detach().abs().sum(dim=-1).ne(0)
    lengths = torch.as_tensor(lengths, device=features.device).long().view(-1)
    if lengths.numel() != features.size(0):
        raise ValueError("Length count does not match batch size.")
    steps = torch.arange(features.size(1), device=features.device).view(1, -1)
    return steps < lengths.view(-1, 1)


def masked_input_sensitivity(features, gradients, valid_mask):
    if features.shape != gradients.shape or valid_mask.shape != features.shape[:2]:
        raise ValueError("Sensitivity tensor shapes are inconsistent.")
    mask = valid_mask.to(features).unsqueeze(-1)
    x = features * mask
    grad = gradients * mask
    reduce_dims = tuple(range(1, features.ndim))
    grad_norm = torch.sqrt(torch.sum(grad.double() ** 2, dim=reduce_dims))
    input_norm = torch.sqrt(torch.sum(x.double() ** 2, dim=reduce_dims))
    product_norm = torch.sqrt(torch.sum((x.double() * grad.double()) ** 2, dim=reduce_dims))
    sensitivity = product_norm / (input_norm + 1e-12)
    return grad_norm.detach().cpu().numpy(), sensitivity.detach().cpu().numpy()


def linear_cka(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    numerator = np.sum((x.T @ y) ** 2)
    denominator = math.sqrt(np.sum((x.T @ x) ** 2) * np.sum((y.T @ y) ** 2))
    return float(numerator / denominator) if denominator > 0 else float("nan")


def effective_rank(matrix):
    x = np.asarray(matrix, dtype=np.float64)
    singular = np.linalg.svd(x - x.mean(axis=0, keepdims=True), compute_uv=False)
    total = singular.sum()
    if total <= 0:
        return 0.0
    probability = singular[singular > 0] / total
    return float(np.exp(-np.sum(probability * np.log(probability))))


def representation_pair_summary(base, with_modality):
    base = np.asarray(base, dtype=np.float64)
    with_modality = np.asarray(with_modality, dtype=np.float64)
    if base.shape != with_modality.shape or base.ndim != 2:
        raise ValueError("Representation matrices must be matching 2-D arrays.")
    delta = with_modality - base
    delta_norm = np.linalg.norm(delta, axis=1)
    base_norm = np.linalg.norm(base, axis=1)
    denominator = base_norm * np.linalg.norm(with_modality, axis=1)
    cosine = np.divide(
        np.sum(base * with_modality, axis=1), denominator,
        out=np.full(len(base), np.nan), where=denominator > 0,
    )
    return {
        "Count": int(len(base)),
        "NormMean": float(delta_norm.mean()),
        "NormMedian": float(np.median(delta_norm)),
        "RelativeShiftMean": float(np.mean(delta_norm / (base_norm + 1e-12))),
        "RelativeShiftMedian": float(np.median(delta_norm / (base_norm + 1e-12))),
        "SameSampleCosineMean": float(np.nanmean(cosine)),
        "SameSampleCosineMedian": float(np.nanmedian(cosine)),
        "LinearCKA": linear_cka(base, with_modality),
        "BaseFeatureVariance": float(np.var(base, axis=0).mean()),
        "WithModalityFeatureVariance": float(np.var(with_modality, axis=0).mean()),
        "BaseEffectiveRank": effective_rank(base),
        "WithModalityEffectiveRank": effective_rank(with_modality),
    }


def fit_quartile_edges(values):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("Cannot fit quartiles to an empty array.")
    return np.quantile(x, [0.25, 0.50, 0.75]).astype(np.float64)


def assign_quartiles(values, edges):
    values = np.asarray(values, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if edges.shape != (3,):
        raise ValueError("Quartile edges must contain Q1/Q2/Q3.")
    indices = np.searchsorted(edges, values, side="right")
    return np.asarray([QUARTILE_NAMES[int(index)] for index in indices], dtype=object)


def label_bin(values):
    values = np.asarray(values, dtype=np.float64)
    result = np.empty(len(values), dtype=object)
    result[(values >= -3) & (values < -1)] = LABEL_BINS[0]
    result[(values >= -1) & (values < 0)] = LABEL_BINS[1]
    result[(values >= 0) & (values < 1)] = LABEL_BINS[2]
    result[(values >= 1) & (values <= 3)] = LABEL_BINS[3]
    outside = (values < -3) | (values > 3)
    if outside.any():
        raise ValueError("MOSI label outside the pre-registered [-3,3] bins.")
    return result


def classify_utility(gain_summary, damage_summary, prediction_shift_mean,
                     correct_prediction_std, relative_shift_mean):
    """Implement the pre-registered A/B/C/D decision tree exactly."""
    gain_supported = float(gain_summary["ci_low"]) > 0
    damage_supported = float(damage_summary["ci_low"]) > 0
    prediction_threshold = 0.10 * float(correct_prediction_std)
    representation_threshold = 0.10
    if gain_supported and damage_supported:
        return "A. Utility Supported"
    underuse = (
        not gain_supported
        and not damage_supported
        and float(prediction_shift_mean) < prediction_threshold
        and float(relative_shift_mean) < representation_threshold
    )
    if underuse:
        return "B. Underuse Supported"
    visibly_used = (
        float(prediction_shift_mean) >= prediction_threshold
        or float(relative_shift_mean) >= representation_threshold
    )
    unreliable = float(gain_summary["mean"]) <= 0 or not damage_supported
    if visibly_used and unreliable:
        return "C. Used but Unreliable"
    return "D. Mixed / Inconclusive"


def classify_change(gate_class, cf_class, delta_gain, delta_damage):
    if float(delta_gain["ci_low"]) > 0 and float(delta_damage["ci_low"]) > 0:
        return "Enhanced"
    if float(delta_gain["ci_high"]) < 0 and float(delta_damage["ci_high"]) < 0:
        return "Weakened"
    if gate_class == cf_class and (
        float(delta_gain["ci_low"]) <= 0 <= float(delta_gain["ci_high"])
        and float(delta_damage["ci_low"]) <= 0 <= float(delta_damage["ci_high"])
    ):
        return "Maintained / no consistent change supported"
    return "Mixed / Inconclusive"


def vector_to_json(vector):
    return json.dumps([float(value) for value in np.asarray(vector).reshape(-1)],
                      separators=(",", ":"), allow_nan=False)


def vector_from_json(value):
    return np.asarray(json.loads(value), dtype=np.float64)
