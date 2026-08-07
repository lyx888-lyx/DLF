"""Train->Valid split-shift and failure diagnostic for CFCompatKD v5.1.

This script is diagnostic-only. It consumes the frozen gate CSV artifacts emitted
by the v5 gate-only pre-screen and never constructs a dataset loader, trains a
Student, selects checkpoints, or accesses Test.

Primary questions:
1) Which label-free prediction-geometry features shift from Train to Valid?
2) Which feature->Teacher-benefit relationships weaken or reverse?
3) Which sample x missing-mode events are high-confidence gate mistakes?
4) What shared categorical patterns are enriched among those failures?
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.metrics import roc_auc_score

from trains.singleTask.cfcompat_crossfit_transfer_risk_utils import (
    BENEFIT_MARGIN,
    FEATURE_COLUMNS,
)
from trains.singleTask.missing_utils import MISSING_MODES


VERSION = "cfcompat_split_shift_failure_diagnostic_v5p1"
DEFAULT_TRAIN_GATE = Path(
    "result/missing_baseline/cfcompat_crossfit_transfer_risk_v5/mosi/valid_screen/"
    "seed1113_dev/gate_only/crossfit_transfer_risk_v5_train_oof_gate.csv"
)
DEFAULT_VALID_GATE = Path(
    "result/missing_baseline/cfcompat_crossfit_transfer_risk_v5/mosi/valid_screen/"
    "seed1113_dev/gate_only/crossfit_transfer_risk_v5_valid_gate_diagnostic.csv"
)
DEFAULT_OUTPUT = Path(
    "result/missing_baseline/cfcompat_split_shift_failure_diagnostic_v5p1/"
    "mosi/seed1113_train_valid"
)
PROBABILITY_COLUMN = {
    "train": "oof_benefit_probability",
    "valid": "full_train_benefit_probability",
}
RULE_COLUMNS = (
    "mode",
    "baseline_difficulty_quartile",
    "teacher_gap_quartile",
    "full_missing_gap_quartile",
    "label_region",
    "teacher_baseline_sign_relation",
    "teacher_crosses_label_relative_baseline",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="CFCompatKD v5.1 Train->Valid split-shift / sample-failure diagnostic."
    )
    parser.add_argument("--train-gate-csv", default=str(DEFAULT_TRAIN_GATE))
    parser.add_argument("--valid-gate-csv", default=str(DEFAULT_VALID_GATE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--near-zero-label-threshold", type=float, default=0.5)
    parser.add_argument("--min-rule-support", type=int, default=20)
    parser.add_argument("--top-k-failures", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.near_zero_label_threshold < 0:
        parser.error("--near-zero-label-threshold must be non-negative.")
    if args.min_rule_support < 1:
        parser.error("--min-rule-support must be >= 1.")
    if args.top_k_failures < 1:
        parser.error("--top-k-failures must be >= 1.")
    return args


def _finite_numeric(frame: pd.DataFrame, columns: Sequence[str], context: str):
    values = frame.loc[:, columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise FloatingPointError(f"{context} contains NaN/Inf in required numeric columns.")


def validate_gate_frame(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    if split not in PROBABILITY_COLUMN:
        raise ValueError(f"Unsupported split: {split}")
    probability_column = PROBABILITY_COLUMN[split]
    required = {
        "sample_index",
        "sample_id",
        "mode",
        "split",
        "label",
        "baseline_missing_prediction",
        "baseline_full_prediction",
        "teacher_full_prediction",
        "initial_student_missing_prediction",
        "baseline_error",
        "teacher_error",
        "teacher_advantage_vs_baseline",
        "beneficial_label",
        probability_column,
        *FEATURE_COLUMNS,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{split} gate CSV lacks columns: {sorted(missing)}")
    result = frame.copy()
    split_values = set(result["split"].astype(str).str.lower().unique())
    if split_values != {split}:
        raise RuntimeError(f"Expected only split={split}; found {sorted(split_values)}")
    if result["split"].astype(str).str.lower().eq("test").any():
        raise RuntimeError("Test rows are forbidden in the v5.1 diagnostic.")
    if not set(result["mode"].astype(str)).issubset(set(MISSING_MODES)):
        raise RuntimeError(f"{split} contains unsupported missing modes.")
    if result.duplicated(["sample_index", "mode"]).any():
        raise RuntimeError(f"{split} contains duplicate sample_index x mode rows.")
    if result.empty:
        raise RuntimeError(f"{split} gate CSV is empty.")
    probability = result[probability_column].to_numpy(dtype=np.float64)
    if not np.isfinite(probability).all() or np.any(probability <= 0.0) or np.any(probability >= 1.0):
        raise FloatingPointError(f"{split} gate probabilities must be finite and strictly inside (0,1).")
    labels = result["beneficial_label"].to_numpy(dtype=np.int64)
    if not set(np.unique(labels)).issubset({0, 1}):
        raise RuntimeError(f"{split} beneficial_label must be binary.")
    _finite_numeric(
        result,
        list(FEATURE_COLUMNS)
        + [
            "label",
            "baseline_error",
            "teacher_error",
            "teacher_advantage_vs_baseline",
            probability_column,
        ],
        split,
    )
    return result


def load_gate_frames(train_path: Path, valid_path: Path):
    train = validate_gate_frame(pd.read_csv(train_path), "train")
    valid = validate_gate_frame(pd.read_csv(valid_path), "valid")
    return train, valid


def _rank_quartile(series: pd.Series) -> pd.Series:
    pct = series.rank(method="average", pct=True)
    return pd.cut(
        pct,
        bins=[0.0, 0.25, 0.50, 0.75, 1.0],
        labels=["Q1_low", "Q2", "Q3", "Q4_high"],
        include_lowest=True,
    ).astype(str)


def enrich_failure_columns(frame: pd.DataFrame, split: str, near_zero_label_threshold: float) -> pd.DataFrame:
    result = frame.copy()
    probability_column = PROBABILITY_COLUMN[split]
    result["gate_probability"] = result[probability_column].astype(float)
    result["gate_predicted_beneficial"] = result["gate_probability"] >= 0.5
    truth = result["beneficial_label"].astype(int).eq(1)
    predicted = result["gate_predicted_beneficial"].astype(bool)
    result["gate_misclassified"] = truth.ne(predicted)
    result["gate_error_type"] = np.select(
        [truth & predicted, truth & ~predicted, ~truth & predicted],
        ["true_positive", "false_negative", "false_positive"],
        default="true_negative",
    )
    result["gate_confidence"] = np.abs(result["gate_probability"] - 0.5) * 2.0
    result["gate_brier_each"] = (
        result["gate_probability"] - result["beneficial_label"].astype(float)
    ) ** 2

    advantage = result["teacher_advantage_vs_baseline"].astype(float)
    result["teacher_condition"] = np.select(
        [advantage <= -BENEFIT_MARGIN, advantage < 0.0, advantage < BENEFIT_MARGIN],
        ["strong_harmful", "mild_harmful", "ambiguous_positive"],
        default="beneficial",
    )

    y = result["label"].astype(float)
    b = result["baseline_missing_prediction"].astype(float)
    t = result["teacher_full_prediction"].astype(float)
    f = result["baseline_full_prediction"].astype(float)
    result["abs_teacher_baseline_gap"] = np.abs(t - b)
    result["abs_full_missing_gap"] = np.abs(f - b)
    result["teacher_crosses_label_relative_baseline"] = ((b - y) * (t - y) < 0.0).astype(str)
    sign_product = np.sign(t) * np.sign(b)
    result["teacher_baseline_sign_relation"] = np.select(
        [sign_product > 0, sign_product < 0],
        ["same_nonzero", "opposite_nonzero"],
        default="zero_involved",
    )
    result["label_region"] = np.select(
        [np.abs(y) <= near_zero_label_threshold, y < 0.0],
        ["near_zero", "negative"],
        default="positive",
    )

    result["baseline_difficulty_quartile"] = (
        result.groupby("mode", group_keys=False)["baseline_error"].transform(_rank_quartile)
    )
    result["teacher_gap_quartile"] = (
        result.groupby("mode", group_keys=False)["abs_teacher_baseline_gap"].transform(_rank_quartile)
    )
    result["full_missing_gap_quartile"] = (
        result.groupby("mode", group_keys=False)["abs_full_missing_gap"].transform(_rank_quartile)
    )
    return result


def _mode_slices(frame: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    yield "ALL", frame
    for mode in MISSING_MODES:
        yield str(mode), frame.loc[frame["mode"].astype(str).eq(mode)]


def _safe_smd(train_values: np.ndarray, valid_values: np.ndarray) -> float:
    train_sd = float(np.std(train_values, ddof=1)) if len(train_values) > 1 else 0.0
    valid_sd = float(np.std(valid_values, ddof=1)) if len(valid_values) > 1 else 0.0
    pooled = math.sqrt(max((train_sd * train_sd + valid_sd * valid_sd) / 2.0, 0.0))
    delta = float(np.mean(valid_values) - np.mean(train_values))
    if pooled <= 1e-12:
        return 0.0 if abs(delta) <= 1e-12 else float(np.sign(delta) * np.inf)
    return delta / pooled


def feature_shift_summary(train: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode, train_local in _mode_slices(train):
        valid_local = valid if mode == "ALL" else valid.loc[valid["mode"].astype(str).eq(mode)]
        for feature in FEATURE_COLUMNS:
            tv = train_local[feature].to_numpy(dtype=np.float64)
            vv = valid_local[feature].to_numpy(dtype=np.float64)
            if len(tv) == 0 or len(vv) == 0:
                continue
            train_sd = float(np.std(tv, ddof=1)) if len(tv) > 1 else 0.0
            valid_sd = float(np.std(vv, ddof=1)) if len(vv) > 1 else 0.0
            constant_both = train_sd <= 1e-12 and valid_sd <= 1e-12
            ks_stat, ks_p = (0.0, 1.0) if constant_both else ks_2samp(tv, vv)
            smd = _safe_smd(tv, vv)
            tq = np.quantile(tv, [0.10, 0.25, 0.50, 0.75, 0.90])
            vq = np.quantile(vv, [0.10, 0.25, 0.50, 0.75, 0.90])
            rows.append(
                {
                    "mode": mode,
                    "feature": feature,
                    "train_N": len(tv),
                    "valid_N": len(vv),
                    "train_mean": float(np.mean(tv)),
                    "valid_mean": float(np.mean(vv)),
                    "mean_delta_valid_minus_train": float(np.mean(vv) - np.mean(tv)),
                    "train_std": train_sd,
                    "valid_std": valid_sd,
                    "standardized_mean_difference": float(smd),
                    "abs_standardized_mean_difference": float(abs(smd)),
                    "ks_statistic": float(ks_stat),
                    "ks_pvalue": float(ks_p),
                    "train_q10": float(tq[0]),
                    "train_q25": float(tq[1]),
                    "train_q50": float(tq[2]),
                    "train_q75": float(tq[3]),
                    "train_q90": float(tq[4]),
                    "valid_q10": float(vq[0]),
                    "valid_q25": float(vq[1]),
                    "valid_q50": float(vq[2]),
                    "valid_q75": float(vq[3]),
                    "valid_q90": float(vq[4]),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["abs_standardized_mean_difference", "ks_statistic"], ascending=[False, False]
    ).reset_index(drop=True)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    value = spearmanr(x, y).correlation
    return float(value) if np.isfinite(value) else float("nan")


def _safe_auc(labels: np.ndarray, values: np.ndarray) -> float:
    if len(np.unique(labels)) != 2 or float(np.std(values)) <= 1e-12:
        return float("nan")
    return float(roc_auc_score(labels, values))


def relationship_shift_summary(train: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode, train_local in _mode_slices(train):
        valid_local = valid if mode == "ALL" else valid.loc[valid["mode"].astype(str).eq(mode)]
        for feature in FEATURE_COLUMNS:
            tx = train_local[feature].to_numpy(dtype=np.float64)
            vx = valid_local[feature].to_numpy(dtype=np.float64)
            ta = train_local["teacher_advantage_vs_baseline"].to_numpy(dtype=np.float64)
            va = valid_local["teacher_advantage_vs_baseline"].to_numpy(dtype=np.float64)
            ty = train_local["beneficial_label"].to_numpy(dtype=np.int64)
            vy = valid_local["beneficial_label"].to_numpy(dtype=np.int64)
            train_rho = _safe_spearman(tx, ta)
            valid_rho = _safe_spearman(vx, va)
            train_auc = _safe_auc(ty, tx)
            valid_auc = _safe_auc(vy, vx)
            rho_delta = (
                float(valid_rho - train_rho)
                if np.isfinite(train_rho) and np.isfinite(valid_rho)
                else float("nan")
            )
            train_signal = abs(train_rho) if np.isfinite(train_rho) else float("nan")
            valid_signal = abs(valid_rho) if np.isfinite(valid_rho) else float("nan")
            rows.append(
                {
                    "mode": mode,
                    "feature": feature,
                    "train_spearman_vs_teacher_advantage": train_rho,
                    "valid_spearman_vs_teacher_advantage": valid_rho,
                    "spearman_delta_valid_minus_train": rho_delta,
                    "spearman_sign_flip": bool(
                        np.isfinite(train_rho)
                        and np.isfinite(valid_rho)
                        and train_rho * valid_rho < 0.0
                    ),
                    "train_abs_spearman_signal": train_signal,
                    "valid_abs_spearman_signal": valid_signal,
                    "abs_spearman_signal_drop": (
                        float(train_signal - valid_signal)
                        if np.isfinite(train_signal) and np.isfinite(valid_signal)
                        else float("nan")
                    ),
                    "train_univariate_auc": train_auc,
                    "valid_univariate_auc": valid_auc,
                    "univariate_auc_delta_valid_minus_train": (
                        float(valid_auc - train_auc)
                        if np.isfinite(train_auc) and np.isfinite(valid_auc)
                        else float("nan")
                    ),
                    "auc_orientation_flip": bool(
                        np.isfinite(train_auc)
                        and np.isfinite(valid_auc)
                        and (train_auc - 0.5) * (valid_auc - 0.5) < 0.0
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result["relationship_instability_score"] = (
        result["abs_spearman_signal_drop"].fillna(0.0).clip(lower=0.0)
        + result["spearman_sign_flip"].astype(float)
        + result["auc_orientation_flip"].astype(float)
    )
    return result.sort_values(
        ["relationship_instability_score", "abs_spearman_signal_drop"],
        ascending=[False, False],
    ).reset_index(drop=True)


def relationship_bins(train: pd.DataFrame, valid: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    rows = []
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    for mode, train_local in _mode_slices(train):
        valid_local = valid if mode == "ALL" else valid.loc[valid["mode"].astype(str).eq(mode)]
        for feature in FEATURE_COLUMNS:
            train_values = train_local[feature].to_numpy(dtype=np.float64)
            edges = np.unique(np.quantile(train_values, quantiles))
            if len(edges) < 2:
                continue
            for split_name, local in (("train", train_local), ("valid", valid_local)):
                values = local[feature].to_numpy(dtype=np.float64)
                bin_index = np.digitize(values, edges[1:-1], right=True)
                for index in range(len(edges) - 1):
                    selected = local.loc[bin_index == index]
                    if selected.empty:
                        continue
                    rows.append(
                        {
                            "mode": mode,
                            "feature": feature,
                            "split": split_name,
                            "bin_index": int(index),
                            "train_edge_low": float(edges[index]),
                            "train_edge_high": float(edges[index + 1]),
                            "N": int(len(selected)),
                            "mean_feature": float(selected[feature].mean()),
                            "beneficial_prevalence": float(selected["beneficial_label"].mean()),
                            "mean_teacher_advantage": float(
                                selected["teacher_advantage_vs_baseline"].mean()
                            ),
                            "gate_misclassification_rate": float(selected["gate_misclassified"].mean()),
                        }
                    )
    return pd.DataFrame(rows)


def mode_failure_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    split = str(frame["split"].iloc[0]).lower()
    for mode, local in _mode_slices(frame):
        truth = local["beneficial_label"].astype(int).eq(1)
        fp = local["gate_error_type"].eq("false_positive")
        fn = local["gate_error_type"].eq("false_negative")
        nonbeneficial_count = int((~truth).sum())
        beneficial_count = int(truth.sum())
        rows.append(
            {
                "split": split,
                "mode": mode,
                "N": int(len(local)),
                "beneficial_prevalence": float(truth.mean()),
                "mean_teacher_advantage": float(local["teacher_advantage_vs_baseline"].mean()),
                "gate_misclassification_rate": float(local["gate_misclassified"].mean()),
                "false_positive_rate_among_nonbeneficial": (
                    float(fp.sum() / nonbeneficial_count) if nonbeneficial_count else float("nan")
                ),
                "false_negative_rate_among_beneficial": (
                    float(fn.sum() / beneficial_count) if beneficial_count else float("nan")
                ),
                "mean_gate_brier": float(local["gate_brier_each"].mean()),
                "mean_gate_confidence": float(local["gate_confidence"].mean()),
                "high_confidence_mistake_fraction": float(
                    (local["gate_misclassified"] & (local["gate_confidence"] >= 0.5)).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _rate(mask: pd.Series, denominator_mask: pd.Series) -> float:
    denominator = int(denominator_mask.sum())
    return float((mask & denominator_mask).sum() / denominator) if denominator else float("nan")


def failure_rules(frame: pd.DataFrame, min_support: int) -> pd.DataFrame:
    split = str(frame["split"].iloc[0]).lower()
    overall = float(frame["gate_misclassified"].mean())
    rows = []
    specs = [(column,) for column in RULE_COLUMNS] + list(combinations(RULE_COLUMNS, 2))
    for columns in specs:
        grouped = frame.groupby(list(columns), dropna=False, sort=False)
        for values, local in grouped:
            if len(local) < min_support:
                continue
            if not isinstance(values, tuple):
                values = (values,)
            truth = local["beneficial_label"].astype(int).eq(1)
            misclassified = local["gate_misclassified"].astype(bool)
            fp = local["gate_error_type"].eq("false_positive")
            fn = local["gate_error_type"].eq("false_negative")
            rule = " & ".join(f"{column}={value}" for column, value in zip(columns, values))
            error_rate = float(misclassified.mean())
            rows.append(
                {
                    "split": split,
                    "rule_order": len(columns),
                    "rule": rule,
                    "support": int(len(local)),
                    "support_fraction": float(len(local) / len(frame)),
                    "gate_misclassification_rate": error_rate,
                    "misclassification_lift_vs_split": (
                        float(error_rate / overall) if overall > 0 else float("nan")
                    ),
                    "false_positive_rate_among_nonbeneficial": _rate(fp, ~truth),
                    "false_negative_rate_among_beneficial": _rate(fn, truth),
                    "mean_gate_brier": float(local["gate_brier_each"].mean()),
                    "beneficial_prevalence": float(truth.mean()),
                    "mean_teacher_advantage": float(
                        local["teacher_advantage_vs_baseline"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def failure_rule_shift(train_rules: pd.DataFrame, valid_rules: pd.DataFrame) -> pd.DataFrame:
    train_cols = {
        column: f"train_{column}"
        for column in train_rules.columns
        if column not in {"split", "rule_order", "rule"}
    }
    valid_cols = {
        column: f"valid_{column}"
        for column in valid_rules.columns
        if column not in {"split", "rule_order", "rule"}
    }
    left = train_rules.drop(columns=["split"]).rename(columns=train_cols)
    right = valid_rules.drop(columns=["split"]).rename(columns=valid_cols)
    merged = left.merge(right, on=["rule_order", "rule"], how="outer")
    merged["valid_minus_train_misclassification_rate"] = (
        merged["valid_gate_misclassification_rate"]
        - merged["train_gate_misclassification_rate"]
    )
    merged["valid_minus_train_lift"] = (
        merged["valid_misclassification_lift_vs_split"]
        - merged["train_misclassification_lift_vs_split"]
    )
    return merged.sort_values(
        ["valid_minus_train_misclassification_rate", "valid_misclassification_lift_vs_split"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


def top_failure_samples(frame: pd.DataFrame, top_k: int) -> pd.DataFrame:
    columns = [
        "split",
        "sample_index",
        "sample_id",
        "mode",
        "label",
        "baseline_missing_prediction",
        "teacher_full_prediction",
        "initial_student_missing_prediction",
        "baseline_error",
        "teacher_error",
        "teacher_advantage_vs_baseline",
        "beneficial_label",
        "gate_probability",
        "gate_predicted_beneficial",
        "gate_error_type",
        "gate_confidence",
        "gate_brier_each",
        "teacher_condition",
        "baseline_difficulty_quartile",
        "teacher_gap_quartile",
        "full_missing_gap_quartile",
        "label_region",
        "teacher_baseline_sign_relation",
        "teacher_crosses_label_relative_baseline",
        "abs_teacher_baseline_gap",
        "abs_full_missing_gap",
    ]
    failures = frame.loc[frame["gate_misclassified"]].sort_values(
        ["gate_brier_each", "gate_confidence"], ascending=[False, False]
    )
    return failures.loc[:, columns].head(top_k).reset_index(drop=True)


def _records(frame: pd.DataFrame, columns: Sequence[str], n: int = 10):
    clean = frame.loc[:, columns].head(n).replace([np.inf, -np.inf], np.nan)
    return json.loads(clean.to_json(orient="records"))


def build_summary(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    shifts: pd.DataFrame,
    relationships: pd.DataFrame,
    rule_shift: pd.DataFrame,
    train_path: Path,
    valid_path: Path,
) -> dict:
    sign_flips = relationships.loc[
        relationships["spearman_sign_flip"] | relationships["auc_orientation_flip"]
    ].copy()
    valid_rule_candidates = rule_shift.loc[rule_shift["valid_support"].fillna(0) > 0].copy()
    valid_rule_candidates = valid_rule_candidates.sort_values(
        ["valid_misclassification_lift_vs_split", "valid_support"],
        ascending=[False, False],
        na_position="last",
    )
    return {
        "version": VERSION,
        "scope": "Train->Valid diagnostic only",
        "test_constructed": False,
        "test_accessed": False,
        "student_training_performed": False,
        "checkpoint_selection_performed": False,
        "benefit_margin": BENEFIT_MARGIN,
        "inputs": {
            "train_gate_csv": str(train_path),
            "valid_gate_csv": str(valid_path),
            "train_probability_source": PROBABILITY_COLUMN["train"],
            "valid_probability_source": PROBABILITY_COLUMN["valid"],
        },
        "counts": {
            "train_events": int(len(train)),
            "valid_events": int(len(valid)),
            "train_samples": int(train["sample_index"].nunique()),
            "valid_samples": int(valid["sample_index"].nunique()),
            "train_gate_mistakes": int(train["gate_misclassified"].sum()),
            "valid_gate_mistakes": int(valid["gate_misclassified"].sum()),
        },
        "gate_error_rates": {
            "train_oof": float(train["gate_misclassified"].mean()),
            "valid_full_train": float(valid["gate_misclassified"].mean()),
            "delta_valid_minus_train": float(
                valid["gate_misclassified"].mean() - train["gate_misclassified"].mean()
            ),
        },
        "top_feature_distribution_shifts": _records(
            shifts,
            [
                "mode",
                "feature",
                "standardized_mean_difference",
                "ks_statistic",
                "train_mean",
                "valid_mean",
            ],
        ),
        "top_relationship_instabilities": _records(
            relationships,
            [
                "mode",
                "feature",
                "train_spearman_vs_teacher_advantage",
                "valid_spearman_vs_teacher_advantage",
                "spearman_sign_flip",
                "train_univariate_auc",
                "valid_univariate_auc",
                "auc_orientation_flip",
                "relationship_instability_score",
            ],
        ),
        "relationship_sign_or_orientation_flips": _records(
            sign_flips,
            [
                "mode",
                "feature",
                "train_spearman_vs_teacher_advantage",
                "valid_spearman_vs_teacher_advantage",
                "train_univariate_auc",
                "valid_univariate_auc",
            ],
        ),
        "top_valid_failure_rules": _records(
            valid_rule_candidates,
            [
                "rule_order",
                "rule",
                "valid_support",
                "valid_support_fraction",
                "valid_gate_misclassification_rate",
                "valid_misclassification_lift_vs_split",
                "train_gate_misclassification_rate",
                "valid_minus_train_misclassification_rate",
            ],
        ),
        "interpretation_guardrails": [
            "This diagnostic identifies Train->Valid shift; it does not identify failing Test clips.",
            "High feature shift does not by itself prove causal failure.",
            "Relationship flips are prioritization signals for mechanism analysis, not a new gate selection criterion.",
            "Do not tune v5 frozen thresholds from these outputs and reinterpret the historical v5 result.",
        ],
    }


def run_diagnostic(
    train_path: Path,
    valid_path: Path,
    output_dir: Path,
    near_zero_label_threshold: float = 0.5,
    min_rule_support: int = 20,
    top_k_failures_count: int = 100,
    overwrite: bool = False,
):
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Diagnostic output already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_raw, valid_raw = load_gate_frames(train_path, valid_path)
    train = enrich_failure_columns(train_raw, "train", near_zero_label_threshold)
    valid = enrich_failure_columns(valid_raw, "valid", near_zero_label_threshold)

    sample_table = pd.concat([train, valid], ignore_index=True, sort=False)
    shifts = feature_shift_summary(train, valid)
    relationships = relationship_shift_summary(train, valid)
    bins = relationship_bins(train, valid)
    mode_summary = pd.concat(
        [mode_failure_summary(train), mode_failure_summary(valid)], ignore_index=True
    )
    train_rules = failure_rules(train, min_rule_support)
    valid_rules = failure_rules(valid, min_rule_support)
    rule_shift = failure_rule_shift(train_rules, valid_rules)
    top_failures = pd.concat(
        [
            top_failure_samples(train, top_k_failures_count),
            top_failure_samples(valid, top_k_failures_count),
        ],
        ignore_index=True,
    )

    sample_table.to_csv(output_dir / "sample_failure_table.csv", index=False)
    shifts.to_csv(output_dir / "feature_shift_summary.csv", index=False)
    relationships.to_csv(output_dir / "benefit_relationship_shift.csv", index=False)
    bins.to_csv(output_dir / "feature_benefit_relationship_bins.csv", index=False)
    mode_summary.to_csv(output_dir / "mode_failure_summary.csv", index=False)
    rule_shift.to_csv(output_dir / "failure_interaction_rules.csv", index=False)
    top_failures.to_csv(output_dir / "top_gate_failure_samples.csv", index=False)

    summary = build_summary(
        train,
        valid,
        shifts,
        relationships,
        rule_shift,
        train_path,
        valid_path,
    )
    (output_dir / "split_shift_failure_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main():
    args = parse_args()
    summary = run_diagnostic(
        Path(args.train_gate_csv),
        Path(args.valid_gate_csv),
        Path(args.output_dir),
        near_zero_label_threshold=float(args.near_zero_label_threshold),
        min_rule_support=int(args.min_rule_support),
        top_k_failures_count=int(args.top_k_failures),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
