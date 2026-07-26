"""Shared utilities for the source-disjoint Stage23D-A self-risk audit."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path("/code/DLF-mosei-arbiter-audit-v1")
OUT = ROOT / "result" / "expert_self_risk_v1" / "mosei"
RUNTIME = ROOT / "runtime" / "stage23d_a"
V1_SOURCE = SOURCE_ROOT / "result" / "arbiter_audit_v1" / "mosei"
V2_LOCAL = ROOT / "result" / "arbiter_audit_v2" / "mosei"
V2B_LOCAL = ROOT / "result" / "arbiter_audit_v2b" / "mosei"

EXPERTS = (
    "uniform_kd_seed1111",
    "moddrop_seed1111",
    "moddrop_seed1114",
    "cfcompat_seed1111",
    "cfcompat_seed1114",
)
FOLDS = (0, 1)
MODES = ("LAV", "LA", "LV", "L")
MISSING_MODES = ("LA", "LV", "L")
MODE_MASKS = {
    "LAV": (1, 1, 1),
    "LA": (1, 1, 0),
    "LV": (1, 0, 1),
    "L": (1, 0, 0),
}
HEADS = (
    "output_logit",
    "logits_c",
    "logits_l_hetero",
    "logits_a_hetero",
    "logits_v_hetero",
)
ACTIVE_HEADS = {
    "LAV": HEADS,
    "LA": ("output_logit", "logits_c", "logits_l_hetero", "logits_a_hetero"),
    "LV": ("output_logit", "logits_c", "logits_l_hetero", "logits_v_hetero"),
    "L": ("output_logit", "logits_c", "logits_l_hetero"),
}
LEGAL_SUBMODES = {
    "LAV": ("LAV", "LA", "LV", "L"),
    "LA": ("LA", "L"),
    "LV": ("LV", "L"),
    "L": ("L",),
}
SPLIT_SEED = 23170
SHUFFLE_SEEDS = (23071, 23072, 23073)
PCA_DIMS = (16, 32, 64)


def utc_now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value):
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.10g")
    os.replace(temporary, path)


def atomic_tsv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False, float_format="%.10g")
    os.replace(temporary, path)


def git_head(path=ROOT):
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(path), text=True
    ).strip()


def git_branch(path=ROOT):
    return subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=str(path), text=True
    ).strip()


def expert_directory(fold, expert):
    return (
        V1_SOURCE
        / "expert_oof"
        / f"outer_fold{fold}"
        / "experts"
        / expert
    )


def oof_path(fold):
    return V1_SOURCE / "expert_oof" / f"outer_fold{fold}" / "oof_predictions.csv"


def hierarchical_path(fold):
    return (
        V2_LOCAL
        / "features"
        / "hierarchical"
        / f"hierarchical_logits_fold{fold}.csv.gz"
    )


def summary_features(prefix, values):
    """Return fixed scalar summaries without retaining identifiers or labels."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{prefix} representation is empty or non-finite")
    near_zero = np.mean(np.abs(array) <= 1e-6)
    saturated = np.mean(np.abs(array) >= 0.95)
    return {
        f"{prefix}__l2": float(np.linalg.norm(array)),
        f"{prefix}__l1_mean": float(np.mean(np.abs(array))),
        f"{prefix}__mean": float(np.mean(array)),
        f"{prefix}__std": float(np.std(array)),
        f"{prefix}__max": float(np.max(array)),
        f"{prefix}__min": float(np.min(array)),
        f"{prefix}__near_zero_fraction": float(near_zero),
        f"{prefix}__saturated_fraction": float(saturated),
    }


def cosine_and_distance(prefix, left, right):
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError(f"{prefix} vectors have different shapes")
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    cosine = float(np.dot(left, right) / denominator) if denominator > 0 else 0.0
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return {
        f"{prefix}__cosine": cosine,
        f"{prefix}__distance": float(np.linalg.norm(left - right)),
        f"{prefix}__angle": float(np.arccos(cosine)),
    }


def active_head_features(mode, outputs):
    active_names = ACTIVE_HEADS[mode]
    values = np.asarray([float(outputs[name]) for name in active_names])
    pairwise = [
        abs(values[left] - values[right])
        for left in range(len(values))
        for right in range(left + 1, len(values))
    ]
    final_value = float(outputs["output_logit"])
    shared_value = float(outputs["logits_c"])
    result = {
        "prediction": final_value,
        "shared_prediction": shared_value,
        "final_shared_difference": final_value - shared_value,
        "active_head_mean": float(values.mean()),
        "active_head_std": float(values.std()),
        "active_head_range": float(values.max() - values.min()),
        "active_head_pairwise_abs_mean": float(np.mean(pairwise)),
        "active_head_pairwise_abs_max": float(np.max(pairwise)),
        "prediction_absolute_magnitude": abs(final_value),
        "prediction_sign": float(np.sign(final_value)),
        "prediction_distance_to_clip_boundary": float(
            min(abs(final_value + 3.0), abs(final_value - 3.0))
        ),
        "active_head_count": len(active_names),
        "active_head_names": "|".join(active_names),
    }
    for name in HEADS:
        result[f"head_active__{name}"] = int(name in active_names)
    specific = [
        outputs[name]
        for name in active_names
        if name.startswith("logits_") and name.endswith("_hetero")
    ]
    result["shared_specific_abs_mean"] = float(
        np.mean(np.abs(np.asarray(specific) - shared_value))
    )
    return result


class ReadOnlyActivationCapture:
    """Read-only forward-pre-hooks for representations used by scalar heads."""

    MODULES = {
        "final_fused": "out_layer",
        "shared_fused": "out_layer_c",
        "specific_l": "out_layer_l_high",
        "specific_a": "out_layer_a_high",
        "specific_v": "out_layer_v_high",
    }

    def __init__(self, wrapped_model):
        backbone = getattr(wrapped_model, "backbone", wrapped_model)
        self.values = {}
        self.handles = []
        for name, module_name in self.MODULES.items():
            module = getattr(backbone, module_name)
            handle = module.register_forward_pre_hook(self._hook(name))
            self.handles.append(handle)

    def _hook(self, name):
        def capture(_module, inputs):
            if not inputs or not torch.is_tensor(inputs[0]):
                raise RuntimeError(f"Hook {name} received no tensor input")
            self.values[name] = inputs[0].detach()

        return capture

    def clear(self):
        self.values.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def source_stratified_split(samples, seed=SPLIT_SEED):
    """Deterministically split sources 70/15/15 with coarse label/size balance."""
    required = {"sample_id", "video_id", "label"}
    if not required.issubset(samples.columns):
        raise ValueError(f"Missing split columns: {required - set(samples.columns)}")
    unique = samples.drop_duplicates("sample_id").copy()
    if unique.groupby("video_id")["label"].nunique().max() < 1:
        raise RuntimeError("Invalid source labels")
    sources = (
        unique.groupby("video_id")
        .agg(sample_count=("sample_id", "size"), label_mean=("label", "mean"))
        .reset_index()
    )
    label_edges = [-np.inf, -1.0, -0.1, 0.1, 1.0, np.inf]
    sources["label_bin"] = pd.cut(
        sources["label_mean"], label_edges, labels=False, include_lowest=True
    ).astype(int)
    rank = sources["sample_count"].rank(method="first")
    sources["size_bin"] = pd.qcut(
        rank, q=min(4, len(sources)), labels=False, duplicates="drop"
    ).astype(int)
    sources["stratum"] = (
        sources["label_bin"].astype(str) + "_" + sources["size_bin"].astype(str)
    )
    rng = np.random.RandomState(int(seed))
    assignments = []
    for _, group in sources.groupby("stratum", sort=True):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        count = len(indices)
        train_count = int(np.floor(0.70 * count))
        valid_count = int(np.floor(0.15 * count))
        if count >= 3:
            train_count = max(train_count, 1)
            valid_count = max(valid_count, 1)
        if train_count + valid_count >= count:
            train_count = max(1, count - 2)
            valid_count = 1 if count > 1 else 0
        roles = (
            ["inner_train"] * train_count
            + ["inner_valid"] * valid_count
            + ["outer"] * (count - train_count - valid_count)
        )
        for index, role in zip(indices, roles):
            assignments.append((index, role))
    role_by_index = dict(assignments)
    sources["self_risk_role"] = [
        role_by_index[index] for index in sources.index
    ]
    if set(sources["self_risk_role"]) != {
        "inner_train",
        "inner_valid",
        "outer",
    }:
        raise RuntimeError("Source split did not create all roles")
    if sources["video_id"].duplicated().any():
        raise RuntimeError("Duplicate source split row")
    return sources.sort_values("video_id", kind="mergesort").reset_index(drop=True)


def attach_source_roles(samples, source_roles):
    result = samples.merge(
        source_roles[["video_id", "self_risk_role"]],
        on="video_id",
        how="left",
        validate="many_to_one",
    )
    if result["self_risk_role"].isna().any():
        raise RuntimeError("Missing source role")
    source_sets = {
        role: set(result.loc[result["self_risk_role"] == role, "video_id"])
        for role in ("inner_train", "inner_valid", "outer")
    }
    if (
        source_sets["inner_train"] & source_sets["inner_valid"]
        or source_sets["inner_train"] & source_sets["outer"]
        or source_sets["inner_valid"] & source_sets["outer"]
    ):
        raise RuntimeError("Source leakage in self-risk split")
    return result


def risk_labels(frame, fixed_error_threshold=1.0):
    """Create labels using thresholds fitted on inner-train per mode only."""
    required = {"mode", "self_risk_role", "label", "prediction", "raw_uncertainty"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing risk label columns {required - set(frame.columns)}")
    result = frame.copy()
    result["abs_error"] = np.abs(result["label"] - result["prediction"])
    result["sq_error"] = np.square(result["label"] - result["prediction"])
    result["log_abs_error"] = np.log(result["abs_error"] + 1e-6)
    result["bad20"] = 0
    result["bad10"] = 0
    result["fixed_bad"] = (result["abs_error"] > fixed_error_threshold).astype(int)
    result["confident"] = 0
    result["confident_wrong"] = 0
    thresholds = []
    present_modes = [mode for mode in MODES if mode in set(result["mode"])]
    for mode in present_modes:
        train = result.loc[
            (result["mode"] == mode)
            & (result["self_risk_role"] == "inner_train")
        ]
        if len(train) < 20:
            raise RuntimeError(f"Too few inner-train rows for {mode}")
        error80 = float(train["abs_error"].quantile(0.80))
        error90 = float(train["abs_error"].quantile(0.90))
        uncertainty30 = float(train["raw_uncertainty"].quantile(0.30))
        mask = result["mode"] == mode
        result.loc[mask, "bad20"] = (
            result.loc[mask, "abs_error"] >= error80
        ).astype(int)
        result.loc[mask, "bad10"] = (
            result.loc[mask, "abs_error"] >= error90
        ).astype(int)
        result.loc[mask, "confident"] = (
            result.loc[mask, "raw_uncertainty"] <= uncertainty30
        ).astype(int)
        result.loc[mask, "confident_wrong"] = (
            (result.loc[mask, "abs_error"] >= error80)
            & (result.loc[mask, "raw_uncertainty"] <= uncertainty30)
        ).astype(int)
        thresholds.append(
            {
                "mode": mode,
                "error_top20_threshold": error80,
                "error_top10_threshold": error90,
                "uncertainty_bottom30_threshold": uncertainty30,
                "fixed_error_threshold": fixed_error_threshold,
                "fit_role": "inner_train",
            }
        )
    return result, pd.DataFrame(thresholds)
