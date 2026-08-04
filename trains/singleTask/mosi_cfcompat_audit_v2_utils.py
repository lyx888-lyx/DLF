"""Hardened overrides for the MOSI CFCompat audit utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .mosi_cfcompat_audit_utils import *  # noqa: F401,F403
from .mosi_cfcompat_audit_utils import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    FORMAL_SEEDS,
    MIN_GROUP_PER_SEED,
    MODES,
    MISSING_MODES,
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
        high = local.loc[local.compatibility_decile >= 8]
        low = local.loc[local.compatibility_decile <= 3]
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
