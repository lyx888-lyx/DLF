"""Utilities for a locked held-out-seed Safe-CFCompatKD Valid screen."""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .missing_utils import MISSING_MODES


VERSION = "cfcompat_safe_projection_valid_screen_v1"
METHOD = "DLF-Safe-CFCompatKD-v1"
OUTPUT_TAG = "cfcompat_safe_projection_v1"
FORMAL_SEEDS = (1112, 1113, 1115)
RUNS = ("cfcompat_replay", "safe_uniform", "safe_cfcompat")
REPLAY_TOLERANCE = 1e-4
MEAN_GAIN_REQUIRED = 0.005
PER_MODE_MAX_DEGRADATION = 0.002
SUPPORTING_GAIN = 0.003
SUPPORTING_EPOCHS = 2
Q1_DAMAGE_REDUCTION_FRACTION = 0.50
Q4_GAIN_RETENTION_FRACTION = 0.80
GROUP_MAX_DEGRADATION = 0.002


def jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def safe_project_teacher(
    baseline_prediction: torch.Tensor,
    teacher_prediction: torch.Tensor,
    labels: torch.Tensor,
):
    """Project Teacher targets onto the closed baseline-to-label interval.

    The operation is training-only and differentiates neither through the
    frozen baseline nor the frozen Teacher.  It preserves Teacher values inside
    the safe interval, maps wrong-direction values to the baseline endpoint,
    and clips label-overshooting values to the label endpoint.
    """
    baseline = baseline_prediction.detach().view(-1, 1)
    teacher = teacher_prediction.detach().view(-1, 1)
    target = labels.detach().view(-1, 1)
    if not (baseline.shape == teacher.shape == target.shape):
        raise ValueError("Safe projection tensors must have identical shapes.")
    if not (
        torch.isfinite(baseline).all()
        and torch.isfinite(teacher).all()
        and torch.isfinite(target).all()
    ):
        raise FloatingPointError("Safe projection received NaN or Inf.")

    lower = torch.minimum(baseline, target)
    upper = torch.maximum(baseline, target)
    projected = torch.maximum(torch.minimum(teacher, upper), lower)

    baseline_to_label = target - baseline
    baseline_to_teacher = teacher - baseline
    directional_product = baseline_to_label * baseline_to_teacher
    wrong_direction = directional_product < 0.0
    overshoot = (
        (directional_product > 0.0)
        & (baseline_to_teacher.abs() > baseline_to_label.abs())
    )
    zero_width = baseline_to_label.abs() <= 1e-12
    unchanged = torch.isclose(projected, teacher, atol=1e-12, rtol=0.0)
    projected_to_baseline = torch.isclose(
        projected, baseline, atol=1e-12, rtol=0.0
    )
    projected_to_label = torch.isclose(
        projected, target, atol=1e-12, rtol=0.0
    )
    diagnostics = {
        "wrong_direction": wrong_direction.view(-1),
        "overshoot": overshoot.view(-1),
        "zero_width": zero_width.view(-1),
        "unchanged": unchanged.view(-1),
        "projected_to_baseline": projected_to_baseline.view(-1),
        "projected_to_label": projected_to_label.view(-1),
        "teacher_target_abs_shift": (projected - teacher).abs().view(-1),
        "safe_interval_width": (upper - lower).view(-1),
    }
    return projected, diagnostics


def projection_summary(records: Sequence[Mapping]) -> dict:
    if not records:
        raise ValueError("Projection summary requires records.")
    frame = pd.DataFrame(records)
    required = {
        "wrong_direction",
        "overshoot",
        "zero_width",
        "unchanged",
        "projected_to_baseline",
        "projected_to_label",
        "teacher_target_abs_shift",
        "safe_interval_width",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Projection records lack columns: {}".format(sorted(missing)))
    return {
        "sample_count": int(len(frame)),
        "wrong_direction_fraction": float(frame.wrong_direction.mean()),
        "overshoot_fraction": float(frame.overshoot.mean()),
        "zero_width_fraction": float(frame.zero_width.mean()),
        "unchanged_fraction": float(frame.unchanged.mean()),
        "projected_to_baseline_fraction": float(
            frame.projected_to_baseline.mean()
        ),
        "projected_to_label_fraction": float(frame.projected_to_label.mean()),
        "mean_teacher_target_abs_shift": float(
            frame.teacher_target_abs_shift.mean()
        ),
        "mean_safe_interval_width": float(frame.safe_interval_width.mean()),
    }


def missing_macro(row: Mapping, prefix: str = "valid") -> float:
    return float(
        np.mean(
            [float(row[f"{prefix}_{mode}_MAE"]) for mode in MISSING_MODES]
        )
    )


def replay_gate(replay: Mapping, reference: Mapping) -> dict:
    differences = {
        "J_valid": abs(
            float(replay["J_valid"]) - float(reference["J_valid"])
        )
    }
    for mode in ("LAV",) + MISSING_MODES:
        key = f"valid_{mode}_MAE"
        differences[key] = abs(float(replay[key]) - float(reference[key]))
    epoch_match = int(replay["BestValidEpoch"]) == int(
        reference["BestValidEpoch"]
    )
    return {
        "passed": bool(
            epoch_match
            and all(value <= REPLAY_TOLERANCE for value in differences.values())
        ),
        "epoch_match": bool(epoch_match),
        "tolerance": REPLAY_TOLERANCE,
        "differences": differences,
    }


def derive_valid_events(raw: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Seed",
        "Run",
        "Mode",
        "sample_index",
        "sample_id",
        "label",
        "baseline_prediction",
        "candidate_prediction",
        "teacher_prediction",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError("Raw Valid events lack columns: {}".format(sorted(missing)))
    events = raw.copy()
    events["baseline_error"] = np.abs(
        events.baseline_prediction - events.label
    )
    events["candidate_error"] = np.abs(
        events.candidate_prediction - events.label
    )
    events["teacher_error"] = np.abs(events.teacher_prediction - events.label)
    events["gain_vs_dlf"] = events.baseline_error - events.candidate_error
    events["teacher_advantage"] = events.baseline_error - events.teacher_error
    events["teacher_better"] = (events.teacher_advantage > 0.0).astype(int)
    events["direction_correct"] = (
        (events.teacher_prediction - events.baseline_prediction)
        * (events.label - events.baseline_prediction)
        > 0.0
    ).astype(int)
    events["teacher_approach"] = (
        np.abs(events.baseline_prediction - events.teacher_prediction)
        - np.abs(events.candidate_prediction - events.teacher_prediction)
    )
    toward = events.teacher_approach > 0.0
    improved = events.gain_vs_dlf > 0.0
    harmed = events.gain_vs_dlf < 0.0
    events["helpful_imitation"] = (toward & improved).astype(int)
    events["harmful_imitation"] = (toward & harmed).astype(int)
    events["teacher_condition"] = np.select(
        [
            (events.teacher_better == 1) & (events.direction_correct == 1),
            (events.teacher_better == 1) & (events.direction_correct == 0),
            (events.teacher_better == 0) & (events.direction_correct == 1),
        ],
        [
            "better_and_correct",
            "better_wrong_direction",
            "not_better_but_correct",
        ],
        default="not_better_wrong_direction",
    )

    unique = events[
        ["Seed", "Mode", "sample_index", "baseline_error"]
    ].drop_duplicates(["Seed", "Mode", "sample_index"])
    unique["baseline_error_quartile"] = unique.groupby(
        ["Seed", "Mode"], sort=False
    ).baseline_error.transform(
        lambda values: pd.qcut(
            values.rank(method="first"),
            4,
            labels=["Q1_easy", "Q2", "Q3", "Q4_hard"],
        )
    ).astype(str)
    events = events.merge(
        unique[["Seed", "Mode", "sample_index", "baseline_error_quartile"]],
        on=["Seed", "Mode", "sample_index"],
        how="left",
        validate="many_to_one",
    )
    if events.baseline_error_quartile.isna().any():
        raise RuntimeError("Baseline difficulty quartile binding failed.")
    return events


def overall_from_events(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, run), local_run in events.groupby(["Seed", "Run"], sort=True):
        mode_rows = {}
        for mode, local in local_run.groupby("Mode", sort=True):
            row = {
                "Seed": int(seed),
                "Run": str(run),
                "Mode": str(mode),
                "N": int(len(local)),
                "baseline_MAE": float(local.baseline_error.mean()),
                "candidate_MAE": float(local.candidate_error.mean()),
                "gain_vs_DLF": float(local.gain_vs_dlf.mean()),
                "win_rate": float((local.gain_vs_dlf > 0.0).mean()),
                "harmful_imitation_rate": float(local.harmful_imitation.mean()),
                "helpful_imitation_rate": float(local.helpful_imitation.mean()),
            }
            rows.append(row)
            mode_rows[str(mode)] = row
        if set(mode_rows) != {"LAV", *MISSING_MODES}:
            raise RuntimeError("A Valid run is missing one or more modality modes.")
        baseline_j = 0.5 * mode_rows["LAV"]["baseline_MAE"] + 0.5 * float(
            np.mean([mode_rows[mode]["baseline_MAE"] for mode in MISSING_MODES])
        )
        candidate_j = 0.5 * mode_rows["LAV"]["candidate_MAE"] + 0.5 * float(
            np.mean([mode_rows[mode]["candidate_MAE"] for mode in MISSING_MODES])
        )
        rows.append(
            {
                "Seed": int(seed),
                "Run": str(run),
                "Mode": "J",
                "N": int(len(local_run) // 4),
                "baseline_MAE": baseline_j,
                "candidate_MAE": candidate_j,
                "gain_vs_DLF": baseline_j - candidate_j,
                "win_rate": float((local_run.gain_vs_dlf > 0.0).mean()),
                "harmful_imitation_rate": float(
                    local_run.harmful_imitation.mean()
                ),
                "helpful_imitation_rate": float(
                    local_run.helpful_imitation.mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _group_row(frame: pd.DataFrame, seed, run, group_type, group_value):
    return {
        "Seed": seed,
        "Run": str(run),
        "GroupType": str(group_type),
        "GroupValue": str(group_value),
        "N": int(len(frame)),
        "baseline_MAE": float(frame.baseline_error.mean()),
        "candidate_MAE": float(frame.candidate_error.mean()),
        "gain_vs_DLF": float(frame.gain_vs_dlf.mean()),
        "win_rate": float((frame.gain_vs_dlf > 0.0).mean()),
        "teacher_advantage": float(frame.teacher_advantage.mean()),
        "teacher_better_rate": float(frame.teacher_better.mean()),
        "direction_correct_rate": float(frame.direction_correct.mean()),
        "helpful_imitation_rate": float(frame.helpful_imitation.mean()),
        "harmful_imitation_rate": float(frame.harmful_imitation.mean()),
        "teacher_approach_mean": float(frame.teacher_approach.mean()),
    }


def group_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_specs = (
        ("mode", "Mode"),
        ("baseline_error_quartile", "baseline_error_quartile"),
        ("teacher_condition", "teacher_condition"),
    )
    for (seed, run), local_run in events.groupby(["Seed", "Run"], sort=True):
        rows.append(_group_row(local_run, int(seed), run, "all", "ALL"))
        for group_type, column in group_specs:
            for value, local in local_run.groupby(column, sort=True):
                rows.append(
                    _group_row(local, int(seed), run, group_type, value)
                )
    for run, local_run in events.groupby("Run", sort=True):
        rows.append(_group_row(local_run, "POOLED", run, "all", "ALL"))
        for group_type, column in group_specs:
            for value, local in local_run.groupby(column, sort=True):
                rows.append(
                    _group_row(local, "POOLED", run, group_type, value)
                )
    return pd.DataFrame(rows)


def _group_value(groups, seed, run, group_type, group_value, metric):
    selected = groups.loc[
        groups.Seed.astype(str).eq(str(seed))
        & groups.Run.astype(str).eq(str(run))
        & groups.GroupType.astype(str).eq(str(group_type))
        & groups.GroupValue.astype(str).eq(str(group_value))
    ]
    if len(selected) != 1:
        raise RuntimeError(
            "Group metric is not unique: seed={} run={} {}={}".format(
                seed, run, group_type, group_value
            )
        )
    return float(selected.iloc[0][metric])


def per_seed_candidate_evidence(
    candidate: Mapping,
    baseline: Mapping,
    candidate_run: str,
    epoch_rows: Sequence[Mapping],
    groups: pd.DataFrame,
) -> dict:
    seed = int(candidate["Seed"])
    gain_j = float(baseline["J_valid"]) - float(candidate["J_valid"])
    mode_degradations = {
        mode: float(candidate[f"valid_{mode}_MAE"])
        - float(baseline[f"valid_{mode}_MAE"])
        for mode in ("LAV",) + MISSING_MODES
    }
    supporting = int(
        sum(
            str(row["Run"]) == candidate_run
            and int(row["Seed"]) == seed
            and float(row["J_valid"])
            <= float(baseline["J_valid"]) - SUPPORTING_GAIN
            for row in epoch_rows
        )
    )

    replay_q1 = _group_value(
        groups,
        seed,
        "cfcompat_replay",
        "baseline_error_quartile",
        "Q1_easy",
        "gain_vs_DLF",
    )
    candidate_q1 = _group_value(
        groups,
        seed,
        candidate_run,
        "baseline_error_quartile",
        "Q1_easy",
        "gain_vs_DLF",
    )
    q1_threshold = (
        replay_q1 * (1.0 - Q1_DAMAGE_REDUCTION_FRACTION)
        if replay_q1 < 0.0
        else replay_q1 - GROUP_MAX_DEGRADATION
    )

    replay_q4 = _group_value(
        groups,
        seed,
        "cfcompat_replay",
        "baseline_error_quartile",
        "Q4_hard",
        "gain_vs_DLF",
    )
    candidate_q4 = _group_value(
        groups,
        seed,
        candidate_run,
        "baseline_error_quartile",
        "Q4_hard",
        "gain_vs_DLF",
    )
    q4_threshold = (
        replay_q4 * Q4_GAIN_RETENTION_FRACTION
        if replay_q4 > 0.0
        else replay_q4 - GROUP_MAX_DEGRADATION
    )

    replay_harm = _group_value(
        groups,
        seed,
        "cfcompat_replay",
        "all",
        "ALL",
        "harmful_imitation_rate",
    )
    candidate_harm = _group_value(
        groups,
        seed,
        candidate_run,
        "all",
        "ALL",
        "harmful_imitation_rate",
    )

    replay_good = _group_value(
        groups,
        seed,
        "cfcompat_replay",
        "teacher_condition",
        "better_and_correct",
        "gain_vs_DLF",
    )
    candidate_good = _group_value(
        groups,
        seed,
        candidate_run,
        "teacher_condition",
        "better_and_correct",
        "gain_vs_DLF",
    )

    return {
        "gain_valid_J_vs_CFCompatKD": gain_j,
        "mode_degradations_vs_CFCompatKD": mode_degradations,
        "supporting_epoch_count": supporting,
        "positive_J_gain": bool(gain_j > 0.0),
        "no_mode_degradation_over_0p002": bool(
            max(mode_degradations.values()) <= PER_MODE_MAX_DEGRADATION
        ),
        "at_least_two_supporting_epochs": bool(
            supporting >= SUPPORTING_EPOCHS
        ),
        "Q1_CFCompat_gain_vs_DLF": replay_q1,
        "Q1_candidate_gain_vs_DLF": candidate_q1,
        "Q1_required_gain": q1_threshold,
        "Q1_damage_reduced_by_half": bool(candidate_q1 >= q1_threshold),
        "Q4_CFCompat_gain_vs_DLF": replay_q4,
        "Q4_candidate_gain_vs_DLF": candidate_q4,
        "Q4_required_gain": q4_threshold,
        "Q4_retains_80pct_gain": bool(candidate_q4 >= q4_threshold),
        "CFCompat_harmful_imitation_rate": replay_harm,
        "candidate_harmful_imitation_rate": candidate_harm,
        "harmful_imitation_rate_decreased": bool(
            candidate_harm < replay_harm - 1e-12
        ),
        "CFCompat_better_correct_gain_vs_DLF": replay_good,
        "candidate_better_correct_gain_vs_DLF": candidate_good,
        "better_correct_not_materially_degraded": bool(
            candidate_good >= replay_good - GROUP_MAX_DEGRADATION
        ),
    }


def aggregate_candidate_gate(
    candidate_run: str,
    grid_rows: Sequence[Mapping],
    epoch_rows: Sequence[Mapping],
    groups: pd.DataFrame,
) -> dict:
    grid = pd.DataFrame(grid_rows)
    if set(grid.Run.astype(str)) != set(RUNS):
        raise RuntimeError("Safe projection run grid is incomplete.")
    evidence = {}
    gains = []
    for seed in FORMAL_SEEDS:
        baseline_rows = grid.loc[
            grid.Seed.astype(int).eq(seed)
            & grid.Run.astype(str).eq("cfcompat_replay")
        ]
        candidate_rows = grid.loc[
            grid.Seed.astype(int).eq(seed)
            & grid.Run.astype(str).eq(candidate_run)
        ]
        if len(baseline_rows) != 1 or len(candidate_rows) != 1:
            raise RuntimeError("Candidate grid is not unique for seed {}.".format(seed))
        local = per_seed_candidate_evidence(
            candidate_rows.iloc[0].to_dict(),
            baseline_rows.iloc[0].to_dict(),
            candidate_run,
            epoch_rows,
            groups,
        )
        evidence[str(seed)] = local
        gains.append(local["gain_valid_J_vs_CFCompatKD"])

    primary_checks = {
        "all_three_seeds_positive_J_gain": all(
            item["positive_J_gain"] for item in evidence.values()
        ),
        "mean_J_gain_ge_0p005": float(np.mean(gains))
        >= MEAN_GAIN_REQUIRED,
        "no_seed_or_mode_degradation_over_0p002": all(
            item["no_mode_degradation_over_0p002"]
            for item in evidence.values()
        ),
        "two_supporting_epochs_each_seed": all(
            item["at_least_two_supporting_epochs"]
            for item in evidence.values()
        ),
    }
    mechanism_checks = {
        "Q1_damage_reduced_by_half_each_seed": all(
            item["Q1_damage_reduced_by_half"] for item in evidence.values()
        ),
        "Q4_retains_80pct_gain_each_seed": all(
            item["Q4_retains_80pct_gain"] for item in evidence.values()
        ),
        "harmful_imitation_rate_decreased_each_seed": all(
            item["harmful_imitation_rate_decreased"]
            for item in evidence.values()
        ),
        "better_correct_not_materially_degraded_each_seed": all(
            item["better_correct_not_materially_degraded"]
            for item in evidence.values()
        ),
    }
    checks = {**primary_checks, **mechanism_checks}
    return {
        "candidate_run": candidate_run,
        "passed": bool(all(checks.values())),
        "mean_gain_valid_J_vs_CFCompatKD": float(np.mean(gains)),
        "primary_checks": primary_checks,
        "mechanism_checks": mechanism_checks,
        "checks": checks,
        "per_seed": evidence,
        "thresholds": {
            "mean_gain_required": MEAN_GAIN_REQUIRED,
            "max_mode_degradation": PER_MODE_MAX_DEGRADATION,
            "supporting_gain": SUPPORTING_GAIN,
            "supporting_epochs": SUPPORTING_EPOCHS,
            "Q1_damage_reduction_fraction": Q1_DAMAGE_REDUCTION_FRACTION,
            "Q4_gain_retention_fraction": Q4_GAIN_RETENTION_FRACTION,
            "group_max_degradation": GROUP_MAX_DEGRADATION,
        },
        "official_test_authorized": False,
    }
