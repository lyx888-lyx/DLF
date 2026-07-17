"""Frozen utilities for Stage 9A CFCompatKD five-seed prediction ensemble."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .cf_compat_kd_utils import compatibility_from_deltas
from .fixed_kd_utils import checkpoint_sha256
from .missing_utils import MISSING_MODES, regression_metrics


PE5_SEEDS = (1111, 1112, 1113, 1114, 1115)
PE5_MODES = ("LAV", "LA", "LV", "L")
PE5_SPLITS = ("valid", "test")
EXPECTED_J = {"valid": 0.669356, "test": 0.705777}
PREDICTION_COLUMNS = tuple("{}_pred".format(mode) for mode in PE5_MODES)
IDENTITY_COLUMNS = ("sample_index", "sample_id", "label")
METRICS = ("acc_7", "acc_5", "acc_2", "F1_score", "Corr", "MAE", "Loss")
LOWER_IS_BETTER = {"MAE", "Loss", "J"}
QUARTILE_LABELS = ("Q1_low", "Q2", "Q3", "Q4_high")


def canonical_identity_values(frame, column):
    if column == "sample_id":
        return frame[column].astype(str).to_numpy()
    if column == "label":
        # MOSI labels originate as float32 tensors. CSV decimal round-tripping
        # can produce a distinct float64 representation of the same label.
        return frame[column].to_numpy(dtype=np.float32)
    return frame[column].to_numpy(dtype=np.int64)


def require_locked_members(seeds):
    values = tuple(int(seed) for seed in seeds)
    if values != PE5_SEEDS:
        raise ValueError(
            "CFCompatKD-PE5 requires seeds 1111 1112 1113 1114 1115 in order."
        )
    return values


def stage8_directory(result_root, dataset):
    return (
        Path(result_root)
        / "missing_baseline"
        / "cfcompat_stability_v1"
        / dataset
    )


def stage9_directory(result_root, dataset):
    return (
        Path(result_root)
        / "missing_baseline"
        / "cfcompat_prediction_ensemble_v1"
        / dataset
    )


def load_locked_checkpoints(result_root, dataset, seeds=PE5_SEEDS):
    """Read only Stage8 validation-selected Online checkpoints."""
    seeds = require_locked_members(seeds)
    root = stage8_directory(result_root, dataset)
    manifest_path = root / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("AllOnlineReplaysPassed"):
        raise RuntimeError("Stage8 Online replay manifest is not complete.")
    if not manifest.get("NoTestSelectedCheckpoint"):
        raise RuntimeError("Stage8 permits a test-selected checkpoint.")
    online = [
        row for row in manifest["Methods"] if str(row.get("Method")) == "Online"
    ]
    if [int(row["Seed"]) for row in online] != list(seeds):
        raise RuntimeError("Stage8 Online checkpoint members/order differ.")
    records = []
    forbidden = ("diagnostic", "best_test", "ema_", "soup")
    for row in online:
        checkpoint = Path(str(row["Checkpoint"]))
        lowered = str(checkpoint).lower()
        if any(token in lowered for token in forbidden):
            raise RuntimeError("Forbidden ensemble checkpoint: {}".format(checkpoint))
        if checkpoint.name != "online_best_valid.pth":
            raise RuntimeError("Checkpoint is not Stage8 Online validation-best.")
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        actual_sha = checkpoint_sha256(checkpoint)
        if actual_sha != row["CheckpointSHA256"]:
            raise RuntimeError("Checkpoint SHA changed: {}".format(checkpoint))
        local = dict(row)
        local["Seed"] = int(local["Seed"])
        local["BestValidEpoch"] = int(local["BestValidEpoch"])
        local["Checkpoint"] = str(checkpoint)
        local["CheckpointSHA256"] = actual_sha
        local["SelectedBy"] = "validation_J"
        records.append(local)
    return records, manifest_path


def validate_prediction_frame(frame, split, seed=None, method="Online"):
    required = set(IDENTITY_COLUMNS + PREDICTION_COLUMNS)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("Prediction columns are missing: {}.".format(missing))
    local = frame.copy()
    if local.sample_index.duplicated().any():
        raise ValueError("Duplicate sample_index in prediction frame.")
    if "Split" in local and set(local.Split.astype(str)) != {str(split)}:
        raise ValueError("Validation/test predictions are mixed.")
    if seed is not None and "Seed" in local:
        if set(local.Seed.astype(int)) != {int(seed)}:
            raise ValueError("Prediction seed binding differs.")
    if method is not None and "Method" in local:
        if set(local.Method.astype(str)) != {str(method)}:
            raise ValueError("Prediction method binding differs.")
    numeric = local[
        ["sample_index", "label"] + list(PREDICTION_COLUMNS)
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise FloatingPointError("Prediction frame has NaN/Inf.")
    local["sample_index"] = local.sample_index.astype(int)
    return local.sort_values("sample_index", kind="mergesort").reset_index(drop=True)


def bind_prediction_frames(frames, split, seeds=PE5_SEEDS):
    seeds = require_locked_members(seeds)
    if len(frames) != len(seeds):
        raise ValueError("Exactly five prediction frames are required.")
    bound = [
        validate_prediction_frame(frame, split, seed, method="Online")
        for seed, frame in zip(seeds, frames)
    ]
    reference = bound[0].loc[:, IDENTITY_COLUMNS]
    for seed, frame in zip(seeds[1:], bound[1:]):
        identity = frame.loc[:, IDENTITY_COLUMNS]
        if not np.array_equal(
            identity.sample_index.to_numpy(), reference.sample_index.to_numpy()
        ):
            raise RuntimeError("sample_index set differs for seed{}.".format(seed))
        if not np.array_equal(
            identity.sample_id.astype(str).to_numpy(),
            reference.sample_id.astype(str).to_numpy(),
        ):
            raise RuntimeError("sample_id differs for seed{}.".format(seed))
        if not np.array_equal(
            canonical_identity_values(identity, "label"),
            canonical_identity_values(reference, "label"),
        ):
            raise RuntimeError("label differs for seed{}.".format(seed))
    return bound


def equal_prediction_ensemble(frames, split, seeds=PE5_SEEDS):
    bound = bind_prediction_frames(frames, split, seeds)
    return aligned_prediction_mean(bound, split, "CFCompatKD-PE5")


def aligned_prediction_mean(frames, split, method):
    """Mean already identified prediction frames without member selection."""
    if not frames:
        raise ValueError("At least one prediction frame is required.")
    bound = [
        validate_prediction_frame(frame, split, seed=None, method=None)
        for frame in frames
    ]
    reference = bound[0].loc[:, IDENTITY_COLUMNS]
    for frame in bound[1:]:
        for column in IDENTITY_COLUMNS:
            left = canonical_identity_values(reference, column)
            right = canonical_identity_values(frame, column)
            if not np.array_equal(left, right):
                raise RuntimeError("{} differs across ensemble members.".format(column))
    result = bound[0].loc[:, IDENTITY_COLUMNS].copy()
    for column in PREDICTION_COLUMNS:
        values = np.stack(
            [frame[column].to_numpy(dtype=np.float64) for frame in bound], axis=0
        )
        result[column] = values.mean(axis=0)
    result["Split"] = split
    result["Method"] = str(method)
    result["MemberCount"] = len(bound)
    result["EqualWeights"] = True
    return result


def metrics_from_predictions(frame):
    frame = validate_prediction_frame(
        frame, str(frame.Split.iloc[0]) if "Split" in frame else "unknown",
        seed=None, method=None,
    )
    labels = torch.tensor(frame.label.to_numpy(), dtype=torch.float32)
    by_mode = {}
    for mode in PE5_MODES:
        predictions = torch.tensor(
            frame["{}_pred".format(mode)].to_numpy(), dtype=torch.float32
        )
        values = regression_metrics(predictions, labels)
        values["Loss"] = values["MAE"]
        by_mode[mode] = values
    by_mode["MissingMacro"] = {
        metric: float(np.mean([by_mode[mode][metric] for mode in MISSING_MODES]))
        for metric in METRICS
    }
    j_value = 0.5 * by_mode["LAV"]["MAE"] + 0.5 * by_mode["MissingMacro"]["MAE"]
    return by_mode, float(j_value)


def metric_rows(frame, method, seed=None):
    metrics, j_value = metrics_from_predictions(frame)
    split = str(frame.Split.iloc[0]) if "Split" in frame else "unknown"
    rows = []
    for mode in PE5_MODES + ("MissingMacro",):
        row = {
            "Split": split,
            "Method": method,
            "Mode": mode,
            "J": j_value,
            **metrics[mode],
        }
        if seed is not None:
            row["Seed"] = int(seed)
        rows.append(row)
    return rows


def max_prediction_difference(left, right, split):
    left = validate_prediction_frame(left, split, method=None)
    right = validate_prediction_frame(right, split, method=None)
    for column in IDENTITY_COLUMNS:
        left_values = canonical_identity_values(left, column)
        right_values = canonical_identity_values(right, column)
        if not np.array_equal(left_values, right_values):
            raise RuntimeError("{} differs across prediction paths.".format(column))
    return float(
        max(
            np.max(
                np.abs(
                    left[column].to_numpy(dtype=np.float64)
                    - right[column].to_numpy(dtype=np.float64)
                )
            )
            for column in PREDICTION_COLUMNS
        )
    )


def metric_max_difference(left, right):
    left_metrics, left_j = metrics_from_predictions(left)
    right_metrics, right_j = metrics_from_predictions(right)
    differences = [abs(left_j - right_j)]
    for mode in PE5_MODES + ("MissingMacro",):
        differences.extend(
            abs(left_metrics[mode][metric] - right_metrics[mode][metric])
            for metric in METRICS
        )
    return float(max(differences))


def pearson(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or left.std() == 0 or right.std() == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def deterministic_quartiles(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or values.size < 4:
        raise ValueError("Quartiles require at least four finite values.")
    order = np.argsort(values, kind="mergesort")
    labels = np.empty(values.size, dtype=object)
    for rank, index in enumerate(order):
        group = min(3, int(4 * rank / values.size))
        labels[index] = QUARTILE_LABELS[group]
    return labels


def compatibility_quartiles(counterfactual_frame):
    """Split-local diagnostic compatibility from a frozen evaluator pass."""
    frame = counterfactual_frame.sort_values(
        "sample_index", kind="mergesort"
    ).reset_index(drop=True)
    required = {"sample_index", "sample_id", "label", "moddrop_LAV_pred"}
    required.update("moddrop_{}_pred".format(mode) for mode in MISSING_MODES)
    if not required.issubset(frame.columns) or frame.sample_index.duplicated().any():
        raise ValueError("Counterfactual compatibility source is malformed.")
    result = frame.loc[:, IDENTITY_COLUMNS].copy()
    for mode in MISSING_MODES:
        delta = np.abs(
            frame.moddrop_LAV_pred.to_numpy(dtype=np.float64)
            - frame["moddrop_{}_pred".format(mode)].to_numpy(dtype=np.float64)
        )
        _, _, compatibility = compatibility_from_deltas(delta)
        result["Compatibility_{}".format(mode)] = compatibility
        result["CompatibilityQuartile_{}".format(mode)] = deterministic_quartiles(
            compatibility
        )
    return result


def j_contribution(frame):
    labels = frame.label.to_numpy(dtype=np.float64)
    lav = np.abs(frame.LAV_pred.to_numpy(dtype=np.float64) - labels)
    missing = np.stack(
        [
            np.abs(frame["{}_pred".format(mode)].to_numpy(dtype=np.float64) - labels)
            for mode in MISSING_MODES
        ],
        axis=0,
    ).mean(axis=0)
    return 0.5 * lav + 0.5 * missing


def paired_bootstrap(left, right, samples=2000, seed=9092026):
    if int(samples) != 2000:
        raise ValueError("Stage9A fixes paired bootstrap at 2000 samples.")
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or not len(left):
        raise ValueError("Paired bootstrap vectors differ.")
    difference = left - right
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, len(difference), size=(samples, len(difference)))
    boot = difference[indices].mean(axis=1)
    return {
        "MeanDifference": float(difference.mean()),
        "MedianDifference": float(np.median(difference)),
        "Bootstrap95CILow": float(np.quantile(boot, 0.025)),
        "Bootstrap95CIHigh": float(np.quantile(boot, 0.975)),
        "EnsembleBetterFraction": float(np.mean(difference < 0)),
        "BootstrapSamples": int(samples),
        "BootstrapSeed": int(seed),
    }


def markdown_table(frame):
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        rendered = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                rendered.append("{:.6g}".format(float(value)))
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)
