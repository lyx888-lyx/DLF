"""Hardened overrides for the MOSI CFCompat audit utilities."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import mosi_cfcompat_audit_utils as _base_utils
from .mosi_cfcompat_audit_utils import *  # noqa: F401,F403
from .mosi_cfcompat_audit_utils import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    FORMAL_SEEDS,
    MIN_GROUP_PER_SEED,
    MODES,
    MISSING_MODES,
    SENTIMENT_BINS,
    SENTIMENT_EDGES,
    group_mechanism_summary,
    interval,
    prediction_events as _prediction_events,
)


OPPORTUNITY_COLUMNS = (
    "GroupType",
    "GroupValue",
    "min_seed_N",
    "mean_cfcompat_gain",
    "min_seed_cfcompat_gain",
    "both_seeds_positive_gain",
    "mean_remaining_CFCompat_MAE",
    "mean_teacher_advantage",
    "mean_direction_correct_rate",
    "mean_compatibility",
    "recoverable_opportunity_score",
)


def sentiment_bin(labels) -> np.ndarray:
    """Return fixed sentiment bins for ndarray, Series, or Categorical input."""
    values = np.asarray(labels, dtype=np.float64).reshape(-1)
    result = pd.cut(
        values,
        bins=list(SENTIMENT_EDGES),
        labels=list(SENTIMENT_BINS),
        include_lowest=True,
        right=False,
    )
    if np.asarray(pd.isna(result), dtype=bool).any():
        raise ValueError("Sentiment values fall outside the fixed MOSI range.")
    return np.asarray(result.astype(np.int64), dtype=np.int64).reshape(-1)


def intensity_label(labels) -> np.ndarray:
    """Return absolute-intensity labels without pandas return-type assumptions."""
    values = np.abs(np.asarray(labels, dtype=np.float64).reshape(-1))
    result = pd.cut(
        values,
        bins=[-np.inf, 0.5, 1.5, 2.5, np.inf],
        labels=["neutral", "weak", "medium", "strong"],
        right=False,
    )
    if np.asarray(pd.isna(result), dtype=bool).any():
        raise ValueError("Intensity binning produced missing values.")
    return np.asarray(result.astype(str), dtype=str).reshape(-1)


# Imported functions retain the original module's global namespace. Patch these
# helpers there so dataset_sample_frame() and _prediction_events() also use the
# hardened conversions.
_base_utils.sentiment_bin = sentiment_bin
_base_utils.intensity_label = intensity_label


def prediction_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep compatibility deciles meaningful only for missing views."""
    result = _prediction_events(frame)
    result.loc[result.Mode.astype(str).eq("LAV"), "compatibility_decile"] = 0
    result["compatibility_decile"] = result.compatibility_decile.astype(int)
    return result


def opportunity_ranking(groups: pd.DataFrame) -> pd.DataFrame:
    candidates = groups.loc[
        groups.GroupType.isin(
            [
                "mode",
                "sentiment_bin",
                "polarity",
                "intensity",
                "compatibility_decile",
                "baseline_error_quartile",
                "teacher_condition",
                "video",
            ]
        )
        & ~groups.Seed.astype(str).eq("POOLED")
    ].copy()
    candidates = candidates.loc[
        ~(
            candidates.GroupType.eq("compatibility_decile")
            & candidates.GroupValue.astype(str).eq("0")
        )
    ]
    rows = []
    for (group_type, group_value), local in candidates.groupby(
        ["GroupType", "GroupValue"], sort=True
    ):
        if set(local.Seed.astype(int)) != set(FORMAL_SEEDS):
            continue
        if int(local.N.min()) < MIN_GROUP_PER_SEED:
            continue
        teacher_advantage = float(local.teacher_advantage.mean())
        direction = float(local.direction_correct_rate.mean())
        compatibility = float(local.mean_compatibility.mean())
        rows.append(
            {
                "GroupType": str(group_type),
                "GroupValue": str(group_value),
                "min_seed_N": int(local.N.min()),
                "mean_cfcompat_gain": float(local.cfcompat_gain.mean()),
                "min_seed_cfcompat_gain": float(local.cfcompat_gain.min()),
                "both_seeds_positive_gain": bool((local.cfcompat_gain > 0).all()),
                "mean_remaining_CFCompat_MAE": float(local.cfcompat_MAE.mean()),
                "mean_teacher_advantage": teacher_advantage,
                "mean_direction_correct_rate": direction,
                "mean_compatibility": compatibility,
                "recoverable_opportunity_score": (
                    max(0.0, teacher_advantage) * direction * compatibility
                ),
            }
        )
    frame = pd.DataFrame(rows, columns=OPPORTUNITY_COLUMNS)
    if frame.empty:
        return frame
    return frame.sort_values(
        [
            "both_seeds_positive_gain",
            "min_seed_cfcompat_gain",
            "recoverable_opportunity_score",
            "min_seed_N",
        ],
        ascending=[False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def joint_video_bootstrap(
    events: pd.DataFrame,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Collect a fixed number of finite joint video-cluster replicates."""
    if int(replicates) < 100:
        raise ValueError("Video bootstrap requires at least 100 replicates.")
    videos = sorted(events.video_id.astype(str).unique())
    if len(videos) < 2:
        raise RuntimeError("Video bootstrap requires at least two videos.")
    generator = np.random.default_rng(int(seed))
    rows = []
    attempts = 0
    max_attempts = max(10000, int(replicates) * 100)
    while len(rows) < int(replicates) and attempts < max_attempts:
        attempts += 1
        sampled = generator.choice(videos, size=len(videos), replace=True)
        counts = pd.Series(sampled).value_counts().to_dict()
        local = events.copy()
        local["bootstrap_weight"] = local.video_id.astype(str).map(
            lambda value: int(counts.get(value, 0))
        )
        local = local.loc[local.bootstrap_weight > 0]
        high = local.loc[
            local.Mode.astype(str).isin(MISSING_MODES)
            & (local.compatibility_decile >= 8)
        ]
        low = local.loc[
            local.Mode.astype(str).isin(MISSING_MODES)
            & local.compatibility_decile.between(1, 3)
        ]
        if high.empty or low.empty:
            continue
        seed_gains = []
        valid = True
        for _, seed_frame in local.groupby("Seed", sort=True):
            mode_gains = {}
            for mode, mode_frame in seed_frame.groupby("Mode", sort=True):
                weight = mode_frame.bootstrap_weight.to_numpy(dtype=float)
                if weight.sum() <= 0:
                    valid = False
                    break
                baseline = np.average(mode_frame.baseline_error, weights=weight)
                cfcompat = np.average(mode_frame.cfcompat_error, weights=weight)
                mode_gains[str(mode)] = float(baseline - cfcompat)
            if not valid or set(mode_gains) != set(MODES):
                valid = False
                break
            seed_gains.append(
                0.5 * mode_gains["LAV"]
                + 0.5 * np.mean([mode_gains[mode] for mode in MISSING_MODES])
            )
        if not valid or len(seed_gains) != len(FORMAL_SEEDS):
            continue
        high_gain = float(
            np.average(high.cfcompat_gain, weights=high.bootstrap_weight)
        )
        low_gain = float(
            np.average(low.cfcompat_gain, weights=low.bootstrap_weight)
        )
        if not np.isfinite([*seed_gains, high_gain, low_gain]).all():
            continue
        rows.append(
            {
                "Replicate": int(len(rows)),
                "DrawAttempt": int(attempts),
                "RejectedDrawsSoFar": int(attempts - len(rows) - 1),
                "mean_J_gain": float(np.mean(seed_gains)),
                "high_compat_gain": high_gain,
                "low_compat_gain": low_gain,
                "high_minus_low_gain": float(high_gain - low_gain),
                "unique_videos": int(len(counts)),
            }
        )
    if len(rows) != int(replicates):
        raise RuntimeError(
            "Unable to collect {} finite video-bootstrap replicates in {} attempts."
            .format(replicates, max_attempts)
        )
    return pd.DataFrame(rows)


def mechanism_assessment(
    events: pd.DataFrame,
    overall: pd.DataFrame,
    bootstrap: pd.DataFrame,
):
    """Assess compatibility only on LA/LV/L while keeping J over all views."""
    j_rows = overall.loc[overall.Mode.eq("J")].sort_values("Seed")
    both_positive = bool(
        len(j_rows) == len(FORMAL_SEEDS) and (j_rows.cfcompat_gain > 0).all()
    )
    pooled = group_mechanism_summary(events)
    condition = pooled.loc[
        pooled.Seed.astype(str).eq("POOLED")
        & pooled.GroupType.eq("teacher_condition")
    ].set_index("GroupValue")
    good_gain = (
        float(condition.loc["better_and_correct", "cfcompat_gain"])
        if "better_and_correct" in condition.index
        else float("nan")
    )
    bad_values = condition.loc[
        condition.index != "better_and_correct", "cfcompat_gain"
    ]
    bad_gain = float(bad_values.mean()) if len(bad_values) else float("nan")
    missing = events.loc[events.Mode.astype(str).isin(MISSING_MODES)]
    high = missing.loc[missing.compatibility_decile >= 8]
    low = missing.loc[missing.compatibility_decile.between(1, 3)]
    compatibility_gain_difference = float(
        high.cfcompat_gain.mean() - low.cfcompat_gain.mean()
    )
    compatibility_harm_difference = float(
        (high.transfer_quadrant == "Q2_harmful_imitation").mean()
        - (low.transfer_quadrant == "Q2_harmful_imitation").mean()
    )
    boot = interval(bootstrap.mean_J_gain)
    checks = {
        "cfcompat_J_gain_positive_both_seeds": both_positive,
        "joint_video_bootstrap_J_gain_ci_low_positive": bool(
            boot["ci95_low"] > 0.0
        ),
        "teacher_better_correct_subset_gain_exceeds_other_conditions": bool(
            math.isfinite(good_gain)
            and math.isfinite(bad_gain)
            and good_gain > bad_gain
        ),
        "high_compatibility_gain_exceeds_low_compatibility": bool(
            compatibility_gain_difference > 0.0
        ),
        "high_compatibility_not_more_harmful": bool(
            compatibility_harm_difference <= 0.0
        ),
    }
    if not both_positive:
        verdict = "CFCompat_GAIN_NOT_REPRODUCED"
    elif all(checks.values()):
        verdict = "CFCompat_IMPROVEMENT_MECHANISM_SUPPORTED"
    else:
        verdict = "CFCompat_GAIN_REPRODUCED_MECHANISM_PARTIAL"
    return {
        "verdict": verdict,
        "checks": checks,
        "mean_two_seed_J_gain": float(j_rows.cfcompat_gain.mean()),
        "per_seed_J_gain": {
            str(int(row.Seed)): float(row.cfcompat_gain)
            for row in j_rows.itertuples(index=False)
        },
        "video_bootstrap_J_gain": boot,
        "better_and_correct_gain": good_gain,
        "other_teacher_conditions_mean_gain": bad_gain,
        "high_minus_low_compatibility_gain": compatibility_gain_difference,
        "high_minus_low_harmful_imitation_rate": compatibility_harm_difference,
    }
