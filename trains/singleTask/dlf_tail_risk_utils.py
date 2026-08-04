"""Utilities for the frozen DLF long-tail and semantic-risk coupling audit."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .dlf_role_specialization_utils import (
    FORMAL_SEEDS,
    MODES,
    NEUTRAL_TAU,
    SENTIMENT_BINS,
    polarity_labels,
)


VERSION = "dlf_tail_risk_coupling_audit_v1_1"
METHOD = "DLF-FrozenTailRiskCouplingAudit-v1.1"
OUTPUT_TAG = "dlf_tail_risk_coupling_audit_v1"
SOURCE_OUTPUT_TAG = "dlf_role_specialization_audit_v1"
TAIL_BIN_COUNT = 3
HEAD_BIN_COUNT = 3
POSITIVE_COVERAGE_REQUIRED = 0.75
MAE_MEAN_GAP_REQUIRED = 0.05
MAE_MEAN_SPEARMAN_MAX = -0.25
RISK_MEAN_GAP_REQUIRED = 0.01
RISK_MEAN_SPEARMAN_MAX = -0.20
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260804
BOOTSTRAP_MAX_ATTEMPT_MULTIPLIER = 100
BOOTSTRAP_MIN_MAX_ATTEMPTS = 10000
CATASTROPHIC_ABS_ERROR = 2.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Ranks require finite non-empty values.")
    order = np.argsort(array, kind="mergesort")
    result = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        result[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return result


def spearman(first: Sequence[float], second: Sequence[float]) -> float:
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    finite = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite], right[finite]
    if left.size < 3:
        return float("nan")
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    if left_rank.std() == 0.0 or right_rank.std() == 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def tail_head_definition(distribution: pd.DataFrame) -> Dict[str, object]:
    required = {"Dataset", "Split", "LabelFamily", "Class", "Count"}
    if not required.issubset(distribution.columns):
        raise ValueError("Label distribution lacks required columns.")
    local = distribution.loc[
        distribution.Dataset.astype(str).eq("mosi")
        & distribution.Split.astype(str).eq("train")
        & distribution.LabelFamily.astype(str).eq("sentiment_7")
    ][["Class", "Count"]].copy()
    local["Class"] = local.Class.astype(int)
    local["Count"] = local.Count.astype(int)
    local = local.loc[local.Class.isin(SENTIMENT_BINS)]
    if tuple(sorted(local.Class.tolist())) != tuple(SENTIMENT_BINS):
        raise RuntimeError("MOSI train distribution must contain all seven bins.")
    if (local.Count <= 0).any():
        raise RuntimeError("Tail-risk audit requires all seven train bins occupied.")
    ordered = local.sort_values(["Count", "Class"], kind="mergesort")
    tail = tuple(int(value) for value in ordered.head(TAIL_BIN_COUNT).Class)
    head = tuple(int(value) for value in ordered.tail(HEAD_BIN_COUNT).Class)
    middle = tuple(
        int(value)
        for value in SENTIMENT_BINS
        if int(value) not in set(tail) | set(head)
    )
    counts = {
        int(row.Class): int(row.Count)
        for row in local.itertuples(index=False)
    }
    return {
        "tail_bins": list(tail),
        "head_bins": list(head),
        "middle_bins": list(middle),
        "train_counts": {str(key): value for key, value in sorted(counts.items())},
        "selection_rule": "three lowest and three highest train-frequency bins; stable ties by bin",
    }


def add_risk_events(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Seed", "Split", "Mode", "sample_index", "sample_id", "video_id",
        "label", "prediction", "sentiment_bin",
    }
    if not required.issubset(predictions.columns):
        raise ValueError("Baseline prediction table lacks required columns.")
    local = predictions.loc[predictions.Split.astype(str).eq("valid")].copy()
    if local.empty or set(local.Mode.astype(str)) != set(MODES):
        raise RuntimeError("Tail-risk audit requires all official Valid modes.")
    local["Seed"] = local.Seed.astype(int)
    local["sentiment_bin"] = local.sentiment_bin.astype(int)
    local["label"] = local.label.astype(float)
    local["prediction"] = local.prediction.astype(float)
    local["absolute_error"] = np.abs(local.prediction - local.label)
    true_polarity = polarity_labels(local.label.to_numpy(dtype=float))
    predicted_polarity = polarity_labels(local.prediction.to_numpy(dtype=float))
    direct_flip = ((true_polarity == 0) & (predicted_polarity == 2)) | (
        (true_polarity == 2) & (predicted_polarity == 0)
    )
    nonneutral_to_neutral = (true_polarity != 1) & (predicted_polarity == 1)
    neutral_escape = (true_polarity == 1) & (predicted_polarity != 1)
    catastrophic = local.absolute_error.to_numpy(dtype=float) >= CATASTROPHIC_ABS_ERROR
    severe_opposite = direct_flip & (np.abs(local.label.to_numpy(dtype=float)) > 1.5)
    high_cost = direct_flip | catastrophic
    local["true_polarity"] = true_polarity
    local["predicted_polarity"] = predicted_polarity
    local["direct_flip"] = direct_flip.astype(int)
    local["nonneutral_to_neutral"] = nonneutral_to_neutral.astype(int)
    local["neutral_escape"] = neutral_escape.astype(int)
    local["catastrophic_abs_error"] = catastrophic.astype(int)
    local["severe_opposite"] = severe_opposite.astype(int)
    local["high_cost_event"] = high_cost.astype(int)
    if not np.isfinite(local[["label", "prediction", "absolute_error"]]).all().all():
        raise FloatingPointError("Non-finite prediction audit values.")
    return local


def assert_complete_valid_grid(events: pd.DataFrame) -> None:
    expected = {(seed, mode) for seed in FORMAL_SEEDS for mode in MODES}
    observed = {
        (int(row.Seed), str(row.Mode))
        for row in events[["Seed", "Mode"]].drop_duplicates().itertuples(index=False)
    }
    if observed != expected:
        raise RuntimeError("Incomplete seed-view Valid grid: {}".format(sorted(observed)))
    reference = None
    key_columns = ["sample_index", "sample_id", "video_id", "label", "sentiment_bin"]
    for seed, mode in sorted(expected):
        local = events.loc[
            events.Seed.astype(int).eq(seed)
            & events.Mode.astype(str).eq(mode)
        ][key_columns].sort_values("sample_index").reset_index(drop=True)
        if local.sample_index.duplicated().any():
            raise RuntimeError("Duplicate Valid sample indices in seed-view run.")
        if reference is None:
            reference = local
        elif not local.equals(reference):
            raise RuntimeError("Official Valid sample binding differs across seed-view runs.")


def _group_name(sentiment_bin: int, definition: Mapping[str, object]) -> str:
    value = int(sentiment_bin)
    if value in set(int(item) for item in definition["tail_bins"]):
        return "tail"
    if value in set(int(item) for item in definition["head_bins"]):
        return "head"
    return "middle"


def compute_bin_and_run_metrics(
    events: pd.DataFrame,
    definition: Mapping[str, object],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    assert_complete_valid_grid(events)
    train_counts = {int(key): int(value) for key, value in definition["train_counts"].items()}
    max_count = max(train_counts.values())
    bin_rows = []
    run_rows = []
    for (seed, mode), run in events.groupby(["Seed", "Mode"], sort=True):
        run = run.copy()
        run["frequency_group"] = run.sentiment_bin.map(
            lambda value: _group_name(int(value), definition)
        )
        local_bin_rows = []
        for sentiment_bin in SENTIMENT_BINS:
            subset = run.loc[run.sentiment_bin.astype(int).eq(int(sentiment_bin))]
            if subset.empty:
                raise RuntimeError(
                    "Valid run seed={} mode={} lacks bin {}.".format(seed, mode, sentiment_bin)
                )
            row = {
                "Seed": int(seed),
                "Mode": str(mode),
                "sentiment_bin": int(sentiment_bin),
                "frequency_group": _group_name(int(sentiment_bin), definition),
                "train_count": int(train_counts[int(sentiment_bin)]),
                "rarity_score_log_max_over_count": float(
                    math.log(float(max_count) / float(train_counts[int(sentiment_bin)]))
                ),
                "valid_count": int(len(subset)),
                "valid_video_count": int(subset.video_id.astype(str).nunique()),
                "mae": float(subset.absolute_error.mean()),
                "direct_flip_rate": float(subset.direct_flip.mean()),
                "nonneutral_to_neutral_rate": float(subset.nonneutral_to_neutral.mean()),
                "neutral_escape_rate": float(subset.neutral_escape.mean()),
                "catastrophic_abs_error_rate": float(subset.catastrophic_abs_error.mean()),
                "severe_opposite_rate": float(subset.severe_opposite.mean()),
                "high_cost_event_rate": float(subset.high_cost_event.mean()),
            }
            local_bin_rows.append(row)
            bin_rows.append(row)
        bins = pd.DataFrame(local_bin_rows)
        tail_bins = bins.loc[bins.frequency_group.eq("tail")]
        head_bins = bins.loc[bins.frequency_group.eq("head")]
        tail_samples = run.loc[run.frequency_group.eq("tail")]
        head_samples = run.loc[run.frequency_group.eq("head")]
        if tail_bins.empty or head_bins.empty or tail_samples.empty or head_samples.empty:
            raise RuntimeError("Tail/head partition produced an empty group.")
        worst = bins.sort_values(["mae", "sentiment_bin"], ascending=[False, True]).iloc[0]
        run_rows.append({
            "Seed": int(seed),
            "Mode": str(mode),
            "count_vs_mae_spearman": spearman(bins.train_count, bins.mae),
            "count_vs_high_cost_spearman": spearman(
                bins.train_count, bins.high_cost_event_rate
            ),
            "rarity_vs_mae_spearman": spearman(bins.rarity_score_log_max_over_count, bins.mae),
            "rarity_vs_high_cost_spearman": spearman(
                bins.rarity_score_log_max_over_count, bins.high_cost_event_rate
            ),
            "tail_macro_mae": float(tail_bins.mae.mean()),
            "head_macro_mae": float(head_bins.mae.mean()),
            "tail_head_macro_mae_gap": float(tail_bins.mae.mean() - head_bins.mae.mean()),
            "tail_sample_mae": float(tail_samples.absolute_error.mean()),
            "head_sample_mae": float(head_samples.absolute_error.mean()),
            "tail_head_sample_mae_gap": float(
                tail_samples.absolute_error.mean() - head_samples.absolute_error.mean()
            ),
            "tail_high_cost_rate": float(tail_samples.high_cost_event.mean()),
            "head_high_cost_rate": float(head_samples.high_cost_event.mean()),
            "tail_head_high_cost_rate_gap": float(
                tail_samples.high_cost_event.mean() - head_samples.high_cost_event.mean()
            ),
            "tail_direct_flip_rate": float(tail_samples.direct_flip.mean()),
            "head_direct_flip_rate": float(head_samples.direct_flip.mean()),
            "tail_head_direct_flip_rate_gap": float(
                tail_samples.direct_flip.mean() - head_samples.direct_flip.mean()
            ),
            "tail_nonneutral_to_neutral_rate": float(
                tail_samples.nonneutral_to_neutral.mean()
            ),
            "head_nonneutral_to_neutral_rate": float(
                head_samples.nonneutral_to_neutral.mean()
            ),
            "tail_head_nonneutral_to_neutral_gap": float(
                tail_samples.nonneutral_to_neutral.mean()
                - head_samples.nonneutral_to_neutral.mean()
            ),
            "worst_mae_bin": int(worst.sentiment_bin),
            "worst_mae_bin_is_tail": bool(worst.frequency_group == "tail"),
            "valid_sample_count": int(len(run)),
            "valid_video_count": int(run.video_id.astype(str).nunique()),
        })
    return pd.DataFrame(bin_rows), pd.DataFrame(run_rows)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    denominator = float(weights.sum())
    if denominator <= 0.0:
        return float("nan")
    return float(np.sum(values * weights) / denominator)


def _bootstrap_run_gaps(
    run: pd.DataFrame,
    video_weights: Mapping[str, int],
    definition: Mapping[str, object],
) -> Tuple[float, float]:
    local = run.copy()
    local["bootstrap_weight"] = local.video_id.astype(str).map(
        lambda value: int(video_weights.get(str(value), 0))
    )
    local = local.loc[local.bootstrap_weight.astype(int) > 0]
    if local.empty:
        return float("nan"), float("nan")
    local["frequency_group"] = local.sentiment_bin.map(
        lambda value: _group_name(int(value), definition)
    )

    required_group_bins = {
        "tail": tuple(int(value) for value in definition["tail_bins"]),
        "head": tuple(int(value) for value in definition["head_bins"]),
    }
    observed_bins = set(local.sentiment_bin.astype(int))
    required_bins = set(required_group_bins["tail"]) | set(required_group_bins["head"])
    if not required_bins.issubset(observed_bins):
        return float("nan"), float("nan")

    macro = {}
    for group, sentiment_bins in required_group_bins.items():
        per_bin = []
        for sentiment_bin in sentiment_bins:
            subset = local.loc[
                local.sentiment_bin.astype(int).eq(int(sentiment_bin))
            ]
            value = _weighted_mean(
                subset.absolute_error.to_numpy(dtype=float),
                subset.bootstrap_weight.to_numpy(dtype=float),
            )
            if not math.isfinite(value):
                return float("nan"), float("nan")
            per_bin.append(value)
        macro[group] = float(np.mean(per_bin))

    tail = local.loc[local.frequency_group.eq("tail")]
    head = local.loc[local.frequency_group.eq("head")]
    tail_risk = _weighted_mean(
        tail.high_cost_event.to_numpy(dtype=float),
        tail.bootstrap_weight.to_numpy(dtype=float),
    )
    head_risk = _weighted_mean(
        head.high_cost_event.to_numpy(dtype=float),
        head.bootstrap_weight.to_numpy(dtype=float),
    )
    if not all(math.isfinite(value) for value in (tail_risk, head_risk)):
        return float("nan"), float("nan")
    return float(macro["tail"] - macro["head"]), float(tail_risk - head_risk)


def joint_video_bootstrap(
    events: pd.DataFrame,
    definition: Mapping[str, object],
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Collect a fixed number of valid joint video-cluster bootstrap replicates.

    A draw is invalid when it omits any train-defined Tail or Head sentiment
    bin, because the fixed six-bin macro statistic is then undefined. Invalid
    draws are rejected rather than encoded as zero or NaN.
    """
    assert_complete_valid_grid(events)
    replicates = int(replicates)
    if replicates < 100:
        raise ValueError("Formal video bootstrap requires at least 100 replicates.")
    videos = sorted(events.video_id.astype(str).unique())
    if len(videos) < 2:
        raise RuntimeError("Video bootstrap requires at least two Valid videos.")
    runs = {
        (int(seed_value), str(mode)): local.copy()
        for (seed_value, mode), local in events.groupby(["Seed", "Mode"], sort=True)
    }
    expected_run_count = len(FORMAL_SEEDS) * len(MODES)
    if len(runs) != expected_run_count:
        raise RuntimeError("Joint bootstrap requires exactly eight seed-view runs.")

    generator = np.random.default_rng(int(seed))
    rows = []
    attempts = 0
    invalid_draws = 0
    max_attempts = max(
        BOOTSTRAP_MIN_MAX_ATTEMPTS,
        replicates * BOOTSTRAP_MAX_ATTEMPT_MULTIPLIER,
    )
    while len(rows) < replicates and attempts < max_attempts:
        attempts += 1
        sampled = generator.choice(videos, size=len(videos), replace=True)
        counts = pd.Series(sampled).value_counts().to_dict()
        mae_gaps = []
        risk_gaps = []
        valid_draw = True
        for local in runs.values():
            mae_gap, risk_gap = _bootstrap_run_gaps(local, counts, definition)
            if not (math.isfinite(mae_gap) and math.isfinite(risk_gap)):
                valid_draw = False
                break
            mae_gaps.append(mae_gap)
            risk_gaps.append(risk_gap)
        if (
            not valid_draw
            or len(mae_gaps) != expected_run_count
            or len(risk_gaps) != expected_run_count
        ):
            invalid_draws += 1
            continue

        replicate = len(rows)
        rows.append({
            "Replicate": int(replicate),
            "DrawAttempt": int(attempts),
            "InvalidDrawsBeforeAcceptance": int(invalid_draws),
            "mean_tail_head_macro_mae_gap": float(np.mean(mae_gaps)),
            "mean_tail_head_high_cost_rate_gap": float(np.mean(risk_gaps)),
            "unique_videos_sampled": int(len(counts)),
        })

    if len(rows) != replicates:
        raise RuntimeError(
            "Unable to collect {} valid video-bootstrap replicates after {} "
            "draw attempts; {} draws omitted at least one required Tail/Head "
            "bin. The Valid video support is too sparse for this frozen "
            "bootstrap definition.".format(replicates, attempts, invalid_draws)
        )

    result = pd.DataFrame(rows)
    numeric = result[
        [
            "mean_tail_head_macro_mae_gap",
            "mean_tail_head_high_cost_rate_gap",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise FloatingPointError("Accepted video-bootstrap replicates must be finite.")
    return result


def bootstrap_diagnostics(bootstrap: pd.DataFrame) -> Dict[str, float]:
    required = {
        "Replicate",
        "DrawAttempt",
        "InvalidDrawsBeforeAcceptance",
        "mean_tail_head_macro_mae_gap",
        "mean_tail_head_high_cost_rate_gap",
    }
    if not required.issubset(bootstrap.columns) or bootstrap.empty:
        raise ValueError("Bootstrap diagnostics require the accepted-draw table.")
    expected = np.arange(len(bootstrap), dtype=int)
    if not np.array_equal(bootstrap.Replicate.to_numpy(dtype=int), expected):
        raise RuntimeError("Bootstrap replicate IDs must be contiguous from zero.")
    attempts = bootstrap.DrawAttempt.to_numpy(dtype=int)
    if (attempts <= 0).any() or (np.diff(attempts) <= 0).any():
        raise RuntimeError("Accepted bootstrap draw attempts must be strictly increasing.")
    total_attempts = int(attempts[-1])
    invalid_draws = int(total_attempts - len(bootstrap))
    recorded_invalid = bootstrap.InvalidDrawsBeforeAcceptance.to_numpy(dtype=int)
    if recorded_invalid[-1] != invalid_draws:
        raise RuntimeError("Bootstrap invalid-draw accounting is inconsistent.")
    invalid_fraction = (
        float(invalid_draws) / float(total_attempts)
        if total_attempts > 0 else 0.0
    )
    return {
        "valid_replicates": int(len(bootstrap)),
        "total_draw_attempts": total_attempts,
        "invalid_draw_count": invalid_draws,
        "invalid_draw_fraction": invalid_fraction,
        "acceptance_fraction": 1.0 - invalid_fraction,
    }


def _interval(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Bootstrap interval requires finite values.")
    low, high = np.quantile(array, [0.025, 0.975])
    return {
        "mean": float(array.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def coupling_gate(
    run_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    long_tail_present: bool,
) -> Dict[str, object]:
    expected_count = len(FORMAL_SEEDS) * len(MODES)
    if len(run_summary) != expected_count:
        raise RuntimeError("Coupling gate requires exactly eight seed-view runs.")
    diagnostics = bootstrap_diagnostics(bootstrap)
    mae_positive = run_summary.tail_head_macro_mae_gap.astype(float) > 0.0
    mae_negative_rho = run_summary.count_vs_mae_spearman.astype(float) < 0.0
    risk_positive = run_summary.tail_head_high_cost_rate_gap.astype(float) > 0.0
    risk_negative_rho = run_summary.count_vs_high_cost_spearman.astype(float) < 0.0
    lav = run_summary.loc[run_summary.Mode.astype(str).eq("LAV")]
    mae_bootstrap = _interval(bootstrap.mean_tail_head_macro_mae_gap)
    risk_bootstrap = _interval(bootstrap.mean_tail_head_high_cost_rate_gap)
    mae_checks = {
        "both_lav_seeds_tail_mae_higher": bool(
            len(lav) == len(FORMAL_SEEDS)
            and (lav.tail_head_macro_mae_gap.astype(float) > 0.0).all()
        ),
        "positive_gap_coverage_ge_0p75": bool(
            float(mae_positive.mean()) >= POSITIVE_COVERAGE_REQUIRED
        ),
        "negative_frequency_correlation_coverage_ge_0p75": bool(
            float(mae_negative_rho.mean()) >= POSITIVE_COVERAGE_REQUIRED
        ),
        "mean_macro_mae_gap_ge_0p05": bool(
            float(run_summary.tail_head_macro_mae_gap.mean()) >= MAE_MEAN_GAP_REQUIRED
        ),
        "mean_count_vs_mae_spearman_le_minus_0p25": bool(
            float(run_summary.count_vs_mae_spearman.mean()) <= MAE_MEAN_SPEARMAN_MAX
        ),
        "joint_video_bootstrap_ci_low_positive": bool(
            mae_bootstrap["ci95_low"] > 0.0
        ),
    }
    risk_checks = {
        "both_lav_seeds_tail_high_cost_higher": bool(
            len(lav) == len(FORMAL_SEEDS)
            and (lav.tail_head_high_cost_rate_gap.astype(float) > 0.0).all()
        ),
        "positive_gap_coverage_ge_0p75": bool(
            float(risk_positive.mean()) >= POSITIVE_COVERAGE_REQUIRED
        ),
        "negative_frequency_correlation_coverage_ge_0p75": bool(
            float(risk_negative_rho.mean()) >= POSITIVE_COVERAGE_REQUIRED
        ),
        "mean_high_cost_gap_ge_0p01": bool(
            float(run_summary.tail_head_high_cost_rate_gap.mean()) >= RISK_MEAN_GAP_REQUIRED
        ),
        "mean_count_vs_high_cost_spearman_le_minus_0p20": bool(
            float(run_summary.count_vs_high_cost_spearman.mean()) <= RISK_MEAN_SPEARMAN_MAX
        ),
        "joint_video_bootstrap_ci_low_positive": bool(
            risk_bootstrap["ci95_low"] > 0.0
        ),
    }
    mae_supported = bool(long_tail_present and all(mae_checks.values()))
    risk_supported = bool(long_tail_present and all(risk_checks.values()))
    weak_signal = bool(
        long_tail_present
        and (
            float(run_summary.tail_head_macro_mae_gap.mean()) > 0.0
            or float(run_summary.tail_head_high_cost_rate_gap.mean()) > 0.0
        )
    )
    if not long_tail_present:
        verdict = "STOP_LONG_TAIL_PREMISE_NOT_SUPPORTED"
    elif mae_supported and risk_supported:
        verdict = "PROMOTE_SEPARATE_FINAL_TAIL_AND_RISK_SCREENS"
    elif mae_supported:
        verdict = "PROMOTE_FINAL_TAIL_WEIGHT_SCREEN_ONLY"
    elif risk_supported:
        verdict = "PROMOTE_FINAL_SEMANTIC_RISK_SCREEN_ONLY"
    elif weak_signal:
        verdict = "PARTIAL_TAIL_RISK_COUPLING_DO_NOT_TRAIN"
    else:
        verdict = "STOP_TAIL_RISK_COUPLING_NOT_SUPPORTED"
    return {
        "verdict": verdict,
        "long_tail_present": bool(long_tail_present),
        "mae_coupling_supported": mae_supported,
        "semantic_risk_coupling_supported": risk_supported,
        "weak_signal": weak_signal,
        "mae_checks": mae_checks,
        "risk_checks": risk_checks,
        "mean_tail_head_macro_mae_gap": float(
            run_summary.tail_head_macro_mae_gap.mean()
        ),
        "mean_count_vs_mae_spearman": float(
            run_summary.count_vs_mae_spearman.mean()
        ),
        "mae_positive_gap_coverage": float(mae_positive.mean()),
        "mae_negative_correlation_coverage": float(mae_negative_rho.mean()),
        "mean_tail_head_high_cost_rate_gap": float(
            run_summary.tail_head_high_cost_rate_gap.mean()
        ),
        "mean_count_vs_high_cost_spearman": float(
            run_summary.count_vs_high_cost_spearman.mean()
        ),
        "risk_positive_gap_coverage": float(risk_positive.mean()),
        "risk_negative_correlation_coverage": float(risk_negative_rho.mean()),
        "mae_bootstrap": mae_bootstrap,
        "risk_bootstrap": risk_bootstrap,
        "bootstrap_diagnostics": diagnostics,
        "requirements": {
            "positive_coverage": POSITIVE_COVERAGE_REQUIRED,
            "mae_mean_gap": MAE_MEAN_GAP_REQUIRED,
            "mae_mean_spearman_max": MAE_MEAN_SPEARMAN_MAX,
            "risk_mean_gap": RISK_MEAN_GAP_REQUIRED,
            "risk_mean_spearman_max": RISK_MEAN_SPEARMAN_MAX,
            "bootstrap_replicates": int(len(bootstrap)),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_max_attempt_multiplier": BOOTSTRAP_MAX_ATTEMPT_MULTIPLIER,
        },
    }
