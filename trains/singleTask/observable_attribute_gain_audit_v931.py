"""Post-hoc attribute-to-gain alignment utilities for V9.31.

This module never trains a router. It audits whether each V9.30 deployment-time
attribute is statistically aligned with the true post-hoc gain of its matching
expert. Labels are used only to compute diagnostic outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr

AUDIT_VERSION = "observable_attribute_gain_alignment_v931_v1"
EXPERT_NAMES = (
    "text_stable",
    "audio_informative",
    "vision_informative",
    "cross_modal_conflict",
)
BASELINE_NAMES = ("anchor", "old_v921_convex_shrinkage")


@dataclass(frozen=True)
class AlignmentConfigV931:
    top_fraction: float = 0.20
    quantile_bins: int = 5
    bootstrap_repetitions: int = 2000
    bootstrap_seed: int = 1131
    min_spearman: float = 0.05
    min_win_auc: float = 0.55
    min_top_gain: float = 0.005
    min_top_bottom_lift: float = 0.010
    required_positive_folds: int = 4

    def validate(self) -> None:
        if not 0.05 <= float(self.top_fraction) <= 0.45:
            raise ValueError("top_fraction must be in [0.05, 0.45]")
        if int(self.quantile_bins) < 3:
            raise ValueError("quantile_bins must be at least 3")
        if int(self.bootstrap_repetitions) < 100:
            raise ValueError("bootstrap_repetitions must be at least 100")
        if not -1.0 <= float(self.min_spearman) <= 1.0:
            raise ValueError("min_spearman must be in [-1, 1]")
        if not 0.5 <= float(self.min_win_auc) <= 1.0:
            raise ValueError("min_win_auc must be in [0.5, 1]")
        if int(self.required_positive_folds) < 1:
            raise ValueError("required_positive_folds must be positive")


def as_vector(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    return array


def posthoc_gain(baseline, expert, labels) -> np.ndarray:
    base = as_vector(baseline, "baseline")
    pred = as_vector(expert, "expert")
    y = as_vector(labels, "labels")
    if not (len(base) == len(pred) == len(y)):
        raise ValueError("gain inputs do not align")
    return np.abs(base - y) - np.abs(pred - y)


def spearman_or_nan(score, gain) -> float:
    x = as_vector(score, "score")
    y = as_vector(gain, "gain")
    if len(x) != len(y):
        raise ValueError("score/gain mismatch")
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    result = spearmanr(x, y)
    value = getattr(result, "statistic", getattr(result, "correlation", result[0]))
    return float(value) if np.isfinite(value) else float("nan")


def binary_auc(score, positive) -> float:
    """Compute tie-aware ROC AUC without a sklearn dependency."""
    x = as_vector(score, "score")
    y = np.asarray(positive, dtype=bool).reshape(-1)
    if len(x) != len(y):
        raise ValueError("score/positive mismatch")
    positive_count = int(y.sum())
    negative_count = int((~y).sum())
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    ranks = rankdata(x, method="average")
    rank_sum = float(ranks[y].sum())
    u_value = rank_sum - positive_count * (positive_count + 1) / 2.0
    return float(u_value / (positive_count * negative_count))


def extreme_masks(score, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    values = as_vector(score, "score")
    count = max(1, int(np.ceil(len(values) * float(fraction))))
    if 2 * count > len(values):
        count = max(1, len(values) // 2)
    order = np.argsort(values, kind="mergesort")
    bottom = np.zeros(len(values), dtype=bool)
    top = np.zeros(len(values), dtype=bool)
    bottom[order[:count]] = True
    top[order[-count:]] = True
    return top, bottom


def alignment_metrics(score, gain, top_fraction: float) -> Dict[str, float]:
    x = as_vector(score, "score")
    g = as_vector(gain, "gain")
    if len(x) != len(g):
        raise ValueError("score/gain mismatch")
    top, bottom = extreme_masks(x, top_fraction)
    top_gain = float(g[top].mean())
    bottom_gain = float(g[bottom].mean())
    return {
        "sample_count": int(len(g)),
        "attribute_mean": float(x.mean()),
        "gain_mean": float(g.mean()),
        "win_rate": float((g > 0.0).mean()),
        "spearman": spearman_or_nan(x, g),
        "win_auc": binary_auc(x, g > 0.0),
        "top_fraction": float(top_fraction),
        "top_count": int(top.sum()),
        "top_gain": top_gain,
        "top_win_rate": float((g[top] > 0.0).mean()),
        "bottom_count": int(bottom.sum()),
        "bottom_gain": bottom_gain,
        "bottom_win_rate": float((g[bottom] > 0.0).mean()),
        "top_bottom_gain_lift": float(top_gain - bottom_gain),
        "top_bottom_win_rate_lift": float(
            (g[top] > 0.0).mean() - (g[bottom] > 0.0).mean()
        ),
    }


def quantile_gain_rows(
    score,
    gain,
    bins: int,
    *,
    expert: str,
    baseline: str,
    outer_fold: object,
) -> list[Dict[str, object]]:
    x = as_vector(score, "score")
    g = as_vector(gain, "gain")
    if len(x) != len(g):
        raise ValueError("score/gain mismatch")
    order = np.argsort(x, kind="mergesort")
    split_indices = np.array_split(order, min(int(bins), len(order)))
    rows = []
    for bin_index, indices in enumerate(split_indices):
        if len(indices) == 0:
            continue
        rows.append(
            {
                "outer_fold": outer_fold,
                "expert": expert,
                "baseline": baseline,
                "quantile_bin": int(bin_index),
                "quantile_bins": int(len(split_indices)),
                "sample_count": int(len(indices)),
                "attribute_min": float(x[indices].min()),
                "attribute_max": float(x[indices].max()),
                "attribute_mean": float(x[indices].mean()),
                "mean_gain": float(g[indices].mean()),
                "win_rate": float((g[indices] > 0.0).mean()),
                "large_gain_rate_010": float((g[indices] > 0.10).mean()),
                "large_harm_rate_010": float((g[indices] < -0.10).mean()),
            }
        )
    return rows


def grouped_bootstrap_alignment(
    score,
    gain,
    group_ids: Sequence[object],
    config: AlignmentConfigV931,
    seed: int,
) -> Dict[str, float]:
    x = as_vector(score, "score")
    g = as_vector(gain, "gain")
    groups = np.asarray([str(value) for value in group_ids], dtype=object)
    if not (len(x) == len(g) == len(groups)):
        raise ValueError("bootstrap inputs do not align")
    unique = np.asarray(sorted(set(groups.tolist())), dtype=object)
    if unique.size < 2:
        raise ValueError("group bootstrap requires at least two groups")
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(int(seed))
    top_values = np.empty(int(config.bootstrap_repetitions), dtype=np.float64)
    lift_values = np.empty_like(top_values)
    spearman_values = np.empty_like(top_values)
    auc_values = np.empty_like(top_values)
    for repetition in range(int(config.bootstrap_repetitions)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_group[group] for group in sampled])
        local = alignment_metrics(
            x[indices], g[indices], config.top_fraction
        )
        top_values[repetition] = local["top_gain"]
        lift_values[repetition] = local["top_bottom_gain_lift"]
        spearman_values[repetition] = local["spearman"]
        auc_values[repetition] = local["win_auc"]

    def interval(values: np.ndarray, prefix: str) -> Dict[str, float]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return {
                f"{prefix}_ci_low": float("nan"),
                f"{prefix}_ci_high": float("nan"),
                f"{prefix}_positive_probability": float("nan"),
            }
        return {
            f"{prefix}_ci_low": float(np.quantile(finite, 0.025)),
            f"{prefix}_ci_high": float(np.quantile(finite, 0.975)),
            f"{prefix}_positive_probability": float((finite > 0.0).mean()),
        }

    return {
        **interval(top_values, "top_gain"),
        **interval(lift_values, "top_bottom_gain_lift"),
        "spearman_ci_low": float(np.nanquantile(spearman_values, 0.025)),
        "spearman_ci_high": float(np.nanquantile(spearman_values, 0.975)),
        "win_auc_ci_low": float(np.nanquantile(auc_values, 0.025)),
        "win_auc_ci_high": float(np.nanquantile(auc_values, 0.975)),
    }


def classify_expert_alignment(
    aggregate: Mapping[str, float],
    bootstrap: Mapping[str, float],
    positive_top_folds: int,
    config: AlignmentConfigV931,
) -> Dict[str, object]:
    criteria = {
        "spearman": bool(
            np.isfinite(float(aggregate["spearman"]))
            and float(aggregate["spearman"]) >= float(config.min_spearman)
        ),
        "win_auc": bool(
            np.isfinite(float(aggregate["win_auc"]))
            and float(aggregate["win_auc"]) >= float(config.min_win_auc)
        ),
        "top_gain": bool(
            float(aggregate["top_gain"]) >= float(config.min_top_gain)
        ),
        "top_bottom_lift": bool(
            float(aggregate["top_bottom_gain_lift"])
            >= float(config.min_top_bottom_lift)
        ),
        "fold_consistency": bool(
            int(positive_top_folds) >= int(config.required_positive_folds)
        ),
        "bootstrap_top_gain": bool(
            np.isfinite(float(bootstrap["top_gain_ci_low"]))
            and float(bootstrap["top_gain_ci_low"]) > 0.0
        ),
    }
    core = (
        criteria["top_gain"]
        and criteria["top_bottom_lift"]
        and criteria["fold_consistency"]
    )
    rank_signal = criteria["spearman"] or criteria["win_auc"]
    if core and rank_signal and criteria["bootstrap_top_gain"]:
        status = "supported"
    elif (
        float(aggregate["top_gain"]) > 0.0
        and float(aggregate["top_bottom_gain_lift"]) > 0.0
        and int(positive_top_folds) >= max(
            3, int(config.required_positive_folds) - 1
        )
        and sum(criteria.values()) >= 3
    ):
        status = "promising"
    else:
        status = "unsupported"
    return {
        "status": status,
        "criteria_passed": int(sum(criteria.values())),
        "criteria_total": int(len(criteria)),
        "criteria": criteria,
    }


def make_overall_verdict(
    anchor_status: Mapping[str, str],
    strong_baseline_rows: Mapping[str, Mapping[str, float]],
) -> Dict[str, object]:
    supported = sorted(
        expert for expert, status in anchor_status.items()
        if status == "supported"
    )
    promising = sorted(
        expert for expert, status in anchor_status.items()
        if status == "promising"
    )
    strong_positive = sorted(
        expert
        for expert, row in strong_baseline_rows.items()
        if float(row["top_gain"]) > 0.0
        and float(row["top_bottom_gain_lift"]) > 0.0
    )
    if len(supported) >= 2 and len(strong_positive) >= 2:
        verdict = "alignment_supports_targeted_diversity_training"
        negative_correlation_recommended = True
    elif len(supported) >= 1 or len(promising) >= 2:
        verdict = "weak_or_partial_alignment_retest_on_mosei_before_diversity_training"
        negative_correlation_recommended = False
    else:
        verdict = "attributes_not_aligned_redesign_before_diversity_training"
        negative_correlation_recommended = False
    return {
        "verdict": verdict,
        "supported_anchor_experts": supported,
        "promising_anchor_experts": promising,
        "positive_top_region_vs_v921": strong_positive,
        "negative_correlation_recommended": negative_correlation_recommended,
        "guardrails": {
            "no_router_was_trained": True,
            "labels_used_only_for_posthoc_diagnostics": True,
            "do_not_tune_attributes_on_outer_results": True,
        },
    }
