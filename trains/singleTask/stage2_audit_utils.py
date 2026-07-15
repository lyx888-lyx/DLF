"""Pure statistical helpers for the Stage 2.5 frozen-model audit."""

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .missing_utils import MISSING_MODES, regression_metrics

AUDIT_MODES = ("LAV", "LA", "LV", "L")
AUDIT_METHODS = ("gate3_directmask", "moddrop", "fixedkd")
METRIC_KEYS = ("acc_7", "acc_5", "acc_2", "F1_score", "Corr", "MAE", "Loss")
LABEL_BINS = (
    ("[-3,-1)", lambda values: (values >= -3.0) & (values < -1.0)),
    ("[-1,0)", lambda values: (values >= -1.0) & (values < 0.0)),
    ("{0}", lambda values: values == 0.0),
    ("(0,1]", lambda values: (values > 0.0) & (values <= 1.0)),
    ("(1,3]", lambda values: (values > 1.0) & (values <= 3.0)),
)


def checkpoint_metadata(path):
    """Return immutable checkpoint metadata without modifying its contents."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": int(path.stat().st_size),
    }


def assert_frozen_eval_model(model, name):
    if model.training:
        raise RuntimeError("{} must be in eval mode.".format(name))
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("{} has gradients during an inference-only audit.".format(name))


def _as_1d(values):
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.size == 0:
        raise ValueError("Audit statistics require at least one sample.")
    return result


def _quantile(values, percentile):
    return float(np.percentile(_as_1d(values), percentile))


def signed_fraction(values, predicate):
    return float(np.mean(predicate(_as_1d(values))))


def describe_abs_differences(lav_prediction, mode_prediction):
    """Describe abs(LAV - missing-mode) with fixed thresholds and sign rule."""
    lav = _as_1d(lav_prediction)
    mode = _as_1d(mode_prediction)
    if lav.shape != mode.shape:
        raise ValueError("Prediction arrays must have the same shape.")
    difference = np.abs(lav - mode)
    return {
        "mean": float(np.mean(difference)),
        "std": float(np.std(difference)),
        "median": _quantile(difference, 50),
        "p75": _quantile(difference, 75),
        "p90": _quantile(difference, 90),
        "p95": _quantile(difference, 95),
        "max": float(np.max(difference)),
        "fraction_lt_0.01": float(np.mean(difference < 0.01)),
        "fraction_lt_0.05": float(np.mean(difference < 0.05)),
        "fraction_lt_0.10": float(np.mean(difference < 0.10)),
        "fraction_gt_0.25": float(np.mean(difference > 0.25)),
        "fraction_gt_0.50": float(np.mean(difference > 0.50)),
        # Fixed convention: > 0 is positive; <= 0 is negative_or_zero.
        "sign_flip": float(np.mean((lav > 0.0) != (mode > 0.0))),
    }


def homogenization_label(ratio):
    if ratio < 0.75:
        return "compressed"
    if ratio <= 1.25:
        return "similar"
    return "amplified"


def error_change_statistics(lav_prediction, mode_prediction, label, unchanged_epsilon=1e-8):
    lav = _as_1d(lav_prediction)
    mode = _as_1d(mode_prediction)
    labels = _as_1d(label)
    if not (lav.shape == mode.shape == labels.shape):
        raise ValueError("Predictions and labels must have the same shape.")
    delta = np.abs(mode - labels) - np.abs(lav - labels)
    improved = delta < -unchanged_epsilon
    worsened = delta > unchanged_epsilon
    unchanged = np.abs(delta) <= unchanged_epsilon
    if not np.all(improved | worsened | unchanged):
        raise RuntimeError("Error-change categories must partition the samples.")
    return {
        "mean_delta_error": float(np.mean(delta)),
        "median_delta_error": _quantile(delta, 50),
        "improved_count": int(np.sum(improved)),
        "improved_fraction": float(np.mean(improved)),
        "worsened_count": int(np.sum(worsened)),
        "worsened_fraction": float(np.mean(worsened)),
        "unchanged_count": int(np.sum(unchanged)),
        "unchanged_fraction": float(np.mean(unchanged)),
        "mean_improvement_among_improved": float(np.mean(-delta[improved])) if np.any(improved) else np.nan,
        "mean_degradation_among_worsened": float(np.mean(delta[worsened])) if np.any(worsened) else np.nan,
        "p90_absolute_delta": _quantile(np.abs(delta), 90),
    }


def counterfactual_contributions(predictions):
    """Compute r_A, r_V, r_AV and all fixed missing-contribution identities."""
    values = {mode: _as_1d(predictions[mode]) for mode in AUDIT_MODES}
    shape = values["LAV"].shape
    if any(value.shape != shape for value in values.values()):
        raise ValueError("All four counterfactual predictions must align.")
    r_a = values["LA"] - values["L"]
    r_v = values["LV"] - values["L"]
    r_av = values["LAV"] - values["LA"] - values["LV"] + values["L"]
    result = {
        "r_A": r_a,
        "r_V": r_v,
        "r_AV": r_av,
        "delta_missing_LA": values["LAV"] - values["LA"],
        "delta_missing_LV": values["LAV"] - values["LV"],
        "delta_missing_L": values["LAV"] - values["L"],
    }
    result["identity_error"] = np.abs(values["LAV"] - (values["L"] + r_a + r_v + r_av))
    result["identity_error_LA"] = np.abs(result["delta_missing_LA"] - (r_v + r_av))
    result["identity_error_LV"] = np.abs(result["delta_missing_LV"] - (r_a + r_av))
    result["identity_error_L"] = np.abs(result["delta_missing_L"] - (r_a + r_v + r_av))
    return result


def validate_counterfactual_identities(contributions, tolerance=1e-6):
    identity_keys = ("identity_error", "identity_error_LA", "identity_error_LV", "identity_error_L")
    maxima = {key: float(np.max(_as_1d(contributions[key]))) for key in identity_keys}
    failures = {key: value for key, value in maxima.items() if value > tolerance}
    if failures:
        raise RuntimeError(
            "Counterfactual identity error exceeds {}: {}".format(tolerance, failures)
        )
    return maxima


def contribution_statistics(values):
    values = _as_1d(values)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "mean_abs": float(np.mean(np.abs(values))),
        "median": _quantile(values, 50),
        "median_abs": _quantile(np.abs(values), 50),
        "p75_abs": _quantile(np.abs(values), 75),
        "p90_abs": _quantile(np.abs(values), 90),
        "p95_abs": _quantile(np.abs(values), 95),
        "max_abs": float(np.max(np.abs(values))),
        "positive_fraction": signed_fraction(values, lambda item: item > 0.0),
        "negative_fraction": signed_fraction(values, lambda item: item < 0.0),
        "near_zero_fraction_0.01": float(np.mean(np.abs(values) <= 0.01)),
        "near_zero_fraction_0.05": float(np.mean(np.abs(values) <= 0.05)),
        "near_zero_fraction_0.10": float(np.mean(np.abs(values) <= 0.10)),
    }


def dominance_statistics(contributions):
    stacked = np.stack(
        [np.abs(_as_1d(contributions["r_A"])), np.abs(_as_1d(contributions["r_V"])), np.abs(_as_1d(contributions["r_AV"]))],
        axis=1,
    )
    dominant = np.argmax(stacked, axis=1)
    return {
        "dominant_r_A_fraction": float(np.mean(dominant == 0)),
        "dominant_r_V_fraction": float(np.mean(dominant == 1)),
        "dominant_r_AV_fraction": float(np.mean(dominant == 2)),
        "interaction_dominance_fraction": float(
            np.mean(stacked[:, 2] > np.maximum(stacked[:, 0], stacked[:, 1]))
        ),
    }


def pearson_or_nan(left, right):
    left = _as_1d(left)
    right = _as_1d(right)
    if left.shape != right.shape:
        raise ValueError("Correlation arrays must align.")
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def label_distribution(labels):
    labels = _as_1d(labels)
    if np.any((labels < -3.0) | (labels > 3.0)):
        raise ValueError("MOSI labels fall outside the fixed [-3, 3] audit range.")
    records = []
    covered = np.zeros(labels.shape, dtype=bool)
    for name, predicate in LABEL_BINS:
        membership = predicate(labels)
        covered |= membership
        records.append(
            {
                "bin": name,
                "count": int(np.sum(membership)),
                "proportion": float(np.mean(membership)),
            }
        )
    if not np.all(covered):
        raise RuntimeError("Fixed label bins did not cover all labels.")
    return {
        "count": int(labels.size),
        "mean": float(np.mean(labels)),
        "std": float(np.std(labels)),
        "median": _quantile(labels, 50),
        "min": float(np.min(labels)),
        "max": float(np.max(labels)),
        "negative_fraction": float(np.mean(labels < 0.0)),
        "zero_fraction": float(np.mean(labels == 0.0)),
        "positive_fraction": float(np.mean(labels > 0.0)),
        "bins": records,
    }


def total_variation_distance(left_distribution, right_distribution):
    left = _as_1d(left_distribution)
    right = _as_1d(right_distribution)
    if left.shape != right.shape:
        raise ValueError("Distribution vectors must align.")
    if not np.isclose(left.sum(), 1.0) or not np.isclose(right.sum(), 1.0):
        raise ValueError("TVD inputs must each sum to one.")
    return float(0.5 * np.sum(np.abs(left - right)))


def metrics_from_predictions(prediction, labels, batch_losses):
    """Match existing validation CSV semantics: global metrics plus mean batch L1."""
    prediction = _as_1d(prediction)
    labels = _as_1d(labels)
    if prediction.shape != labels.shape:
        raise ValueError("Predictions and labels must align.")
    import torch

    metrics = regression_metrics(torch.from_numpy(prediction), torch.from_numpy(labels))
    metrics["Loss"] = float(np.mean(_as_1d(batch_losses)))
    return metrics


def missing_macro_metrics(metrics_by_mode):
    result = {}
    for key in ("MAE", "Corr", "acc_2", "F1_score"):
        result["MissingMacro_{}".format(key)] = float(
            np.mean([metrics_by_mode[mode][key] for mode in MISSING_MODES])
        )
    result["J"] = float(
        0.5 * metrics_by_mode["LAV"]["MAE"] + 0.5 * result["MissingMacro_MAE"]
    )
    return result


def compare_metrics_to_reference(computed, reference_csv, seed, tolerance=1e-6):
    """Raise on any frozen validation CSV mismatch; never relax tolerance."""
    frame = pd.read_csv(reference_csv)
    matched = frame.loc[frame["Seed"] == int(seed)]
    if len(matched) != 1:
        raise RuntimeError("Expected exactly one seed={} row in {}".format(seed, reference_csv))
    row = matched.iloc[0]
    discrepancies = []
    for mode in AUDIT_MODES:
        for key in METRIC_KEYS:
            column = "{}_{}".format(mode, key)
            if column not in row:
                raise RuntimeError("Reference CSV lacks {}".format(column))
            observed = float(computed[mode][key])
            expected = float(row[column])
            difference = abs(observed - expected)
            if difference > tolerance:
                discrepancies.append(
                    {"field": column, "expected": expected, "observed": observed, "abs_difference": difference}
                )
    if discrepancies:
        raise RuntimeError("Validation reference mismatch: {}".format(discrepancies))
    return {"reference_csv": str(reference_csv), "tolerance": tolerance, "matched": True}


def dataframe_to_csv(frame, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.8f")
