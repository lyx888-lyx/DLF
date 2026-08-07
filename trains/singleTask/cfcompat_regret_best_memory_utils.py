"""Best-So-Far Regret Memory utilities for CFCompatKD v4.2.

v4.2 keeps the successful v4 preservation idea and the v4.1 strong/weak
beneficial-Teacher distinction, but replaces the static frozen-baseline
preservation anchor with a per-Train-sample/per-missing-mode best-so-far
prediction memory.

For every Train event, the memory is first updated if the detached current
Student prediction has lower label error than the stored prediction. Routing is
then computed against the updated memory:

* STRONG_DISTILL: Teacher improves on memory by >= 0.02 and is current-Student
  safe.
* WEAK_DISTILL: Teacher improves on memory by (0, 0.02), moves in the correct
  memory-to-label direction, and is current-Student safe. Its KD contribution
  is 0.25x using the true two-tier normalization from v4.1.
* PRESERVE: Teacher does not improve on memory and current Student has regressed
  from memory by >= 0.02. The preservation target is the memory prediction
  clipped to the current-Student-to-label interval.
* ABSTAIN: otherwise.

The memory is initialized from the frozen validation-best ModDrop baseline and
is updated with Train labels only. Official Valid is used only for checkpoint
selection, and official Test is never constructed by this protocol.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .cfcompat_regret_preserve_guard_utils import tiered_kd_loss
from .cfcompat_regret_preserve_utils import (
    LAMBDA_PRESERVE,
    MILD_CFCOMPAT_BASE,
    MILD_CFCOMPAT_SCALE,
    NEGATIVE_TRANSFER_MARGIN,
    PRESERVE_MARGIN,
    SEVERE_NEGATIVE_TRANSFER_MARGIN,
    mild_cfcompat,
    negative_transfer_summary,
)
from .cfcompat_student_safe_abstain_utils import (
    student_projection_summary,
    student_safe_project_teacher,
)
from .missing_utils import MISSING_MODES


VERSION = "cfcompat_regret_best_memory_valid_screen_v4p2"
METHOD = "DLF-Regret-Aware-Best-So-Far-Memory-CFCompatKD-v4.2"
OUTPUT_TAG = "cfcompat_regret_best_memory_v4p2"
DEV_SEED = 1113
RUN = "regret_best_memory_cfcompat"
RUNS = (RUN,)

STRONG_DISTILL_MARGIN = 0.02
WEAK_DISTILL_SCALE = 0.25
MEMORY_UPDATE_EPS = 1e-12
PRESERVE_MARGIN_V4P2 = PRESERVE_MARGIN
LAMBDA_PRESERVE_V4P2 = LAMBDA_PRESERVE
MILD_CFCOMPAT_BASE_V4P2 = MILD_CFCOMPAT_BASE
MILD_CFCOMPAT_SCALE_V4P2 = MILD_CFCOMPAT_SCALE

# Frozen before the Seed-1113 v4.2 development run. v4.2 must retain the v4
# protection gains while preserving most of the v4.1 recovery on good Teacher
# events.
J_MAX_DEGRADATION_VS_V4 = 0.002
BETTER_CORRECT_MAX_GAIN_DEGRADATION_VS_V4P1 = 0.010
Q1_Q4_MAX_GAIN_DEGRADATION_VS_V4 = 0.005
NEGATIVE_TRANSFER_MAX_INCREASE_VS_V4 = 0.010
SEVERE_NEGATIVE_TRANSFER_MAX_INCREASE_VS_V4 = 0.010
POSITIVE_TRANSFER_MAX_DECREASE_VS_V4 = 0.010
HARMFUL_IMITATION_MAX_INCREASE_VS_V4 = 0.010
SUPPORTING_EPOCHS = 2


class BestSoFarPredictionMemory:
    """CPU best-prediction memory keyed by (sample_index, missing_mode)."""

    def __init__(self, baseline_frame: pd.DataFrame):
        required = {"sample_index", "sample_id", "label"} | {
            "baseline_{}_pred".format(mode) for mode in MISSING_MODES
        }
        missing = required.difference(baseline_frame.columns)
        if missing:
            raise ValueError("Best-memory baseline frame lacks: {}".format(sorted(missing)))
        if len(baseline_frame) != 1284 or baseline_frame.sample_index.nunique() != 1284:
            raise RuntimeError("Best-memory initialization requires 1284 unique Train samples.")
        self._state = {}
        for row in baseline_frame.itertuples(index=False):
            index = int(row.sample_index)
            label = float(row.label)
            sample_id = str(row.sample_id)
            for mode in MISSING_MODES:
                prediction = float(getattr(row, "baseline_{}_pred".format(mode)))
                error = abs(prediction - label)
                if not np.isfinite([prediction, error, label]).all():
                    raise FloatingPointError("Non-finite best-memory initialization.")
                self._state[(index, mode)] = {
                    "sample_index": index,
                    "sample_id": sample_id,
                    "label": label,
                    "mode": mode,
                    "initial_prediction": prediction,
                    "initial_error": error,
                    "best_prediction": prediction,
                    "best_error": error,
                    "update_count": 0,
                    "last_update_event": 0,
                }
        if len(self._state) != 1284 * len(MISSING_MODES):
            raise RuntimeError("Best-memory state cardinality is invalid.")

    def bind_and_update(
        self,
        indices,
        modes,
        current_prediction: torch.Tensor,
        labels: torch.Tensor,
        event_start_ordinal: int,
        device,
        dtype,
    ) -> dict:
        current = current_prediction.detach().view(-1).cpu().numpy().astype(np.float64)
        target = labels.detach().view(-1).cpu().numpy().astype(np.float64)
        if not (len(indices) == len(modes) == len(current) == len(target)):
            raise ValueError("Best-memory batch fields have different lengths.")
        before_pred, before_err = [], []
        after_pred, after_err, updated = [], [], []
        improvement = []
        update_count_after = []
        for offset, (index, mode, prediction, label) in enumerate(
            zip(indices, modes, current, target)
        ):
            key = (int(index), str(mode))
            if key not in self._state:
                raise KeyError("Unknown best-memory key: {}".format(key))
            state = self._state[key]
            if abs(float(state["label"]) - float(label)) > 1e-6:
                raise RuntimeError("Best-memory label binding changed for {}.".format(key))
            old_prediction = float(state["best_prediction"])
            old_error = float(state["best_error"])
            current_error = abs(float(prediction) - float(label))
            do_update = bool(current_error + MEMORY_UPDATE_EPS < old_error)
            if do_update:
                state["best_prediction"] = float(prediction)
                state["best_error"] = float(current_error)
                state["update_count"] = int(state["update_count"]) + 1
                state["last_update_event"] = int(event_start_ordinal) + offset
            new_prediction = float(state["best_prediction"])
            new_error = float(state["best_error"])
            if new_error > old_error + 1e-12:
                raise RuntimeError("Best-memory error increased.")
            before_pred.append(old_prediction)
            before_err.append(old_error)
            after_pred.append(new_prediction)
            after_err.append(new_error)
            updated.append(do_update)
            improvement.append(old_error - new_error)
            update_count_after.append(int(state["update_count"]))
        return {
            "memory_before_prediction": torch.as_tensor(before_pred, device=device, dtype=dtype).view(-1, 1),
            "memory_before_error": torch.as_tensor(before_err, device=device, dtype=dtype).view(-1),
            "memory_after_prediction": torch.as_tensor(after_pred, device=device, dtype=dtype).view(-1, 1),
            "memory_after_error": torch.as_tensor(after_err, device=device, dtype=dtype).view(-1),
            "memory_updated": torch.as_tensor(updated, device=device, dtype=torch.bool),
            "memory_improvement": torch.as_tensor(improvement, device=device, dtype=dtype).view(-1),
            "memory_update_count_after": torch.as_tensor(update_count_after, device=device, dtype=torch.int64).view(-1),
        }

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(list(self._state.values()))
        return frame.sort_values(["sample_index", "mode"], kind="mergesort").reset_index(drop=True)


def regret_best_memory_decision(
    student_prediction: torch.Tensor,
    teacher_prediction: torch.Tensor,
    memory_prediction: torch.Tensor,
    labels: torch.Tensor,
    compatibility: torch.Tensor,
) -> dict:
    """Compute detached v4.2 routing against the updated best-so-far memory."""
    student = student_prediction.detach().view(-1, 1)
    teacher = teacher_prediction.detach().view(-1, 1)
    memory = memory_prediction.detach().view(-1, 1)
    target = labels.detach().view(-1, 1)
    compat = compatibility.detach().view(-1)
    if not (student.shape == teacher.shape == memory.shape == target.shape):
        raise ValueError("v4.2 decision tensors must have identical shapes.")
    if len(compat) != len(student):
        raise ValueError("Compatibility length does not match v4.2 predictions.")
    if not all(torch.isfinite(value).all() for value in (student, teacher, memory, target, compat)):
        raise FloatingPointError("v4.2 decision received NaN/Inf.")

    teacher_safe, teacher_projection = student_safe_project_teacher(student, teacher, target)
    preserve_safe, _ = student_safe_project_teacher(student, memory, target)

    memory_error = torch.abs(memory - target).view(-1)
    teacher_error = torch.abs(teacher - target).view(-1)
    current_error = torch.abs(student - target).view(-1)
    teacher_advantage = memory_error - teacher_error
    current_regret = current_error - memory_error
    memory_direction_correct = (((teacher - memory) * (target - memory)).view(-1) > 0.0)
    safe_teacher_active = teacher_projection["active"].view(-1)

    teacher_better_any = teacher_advantage > 0.0
    strong_candidate = teacher_advantage >= STRONG_DISTILL_MARGIN
    weak_candidate = (
        teacher_better_any
        & (teacher_advantage < STRONG_DISTILL_MARGIN)
        & memory_direction_correct
    )
    strong_distill = strong_candidate & safe_teacher_active
    weak_distill = weak_candidate & safe_teacher_active
    current_regressed = current_regret >= PRESERVE_MARGIN_V4P2
    preserve = (~teacher_better_any) & current_regressed
    abstain = ~(strong_distill | weak_distill | preserve)

    stacked = torch.stack([strong_distill, weak_distill, preserve, abstain], dim=0).to(torch.int64)
    if not torch.all(stacked.sum(dim=0) == 1):
        raise AssertionError("v4.2 routing is not mutually exclusive/exhaustive.")

    mild = mild_cfcompat(compat)
    strong_gate = strong_distill.to(compat) * mild
    weak_gate = weak_distill.to(compat) * mild
    preserve_gate = preserve.to(compat)

    return {
        "teacher_safe_target": teacher_safe.detach().view(-1, 1),
        "preserve_safe_target": preserve_safe.detach().view(-1, 1),
        "strong_gate": strong_gate.detach(),
        "weak_gate": weak_gate.detach(),
        "preserve_gate": preserve_gate.detach(),
        "strong_distill": strong_distill.detach(),
        "weak_distill": weak_distill.detach(),
        "preserve": preserve.detach(),
        "abstain": abstain.detach(),
        "teacher_better_any": teacher_better_any.detach(),
        "strong_candidate": strong_candidate.detach(),
        "weak_candidate": weak_candidate.detach(),
        "memory_direction_correct": memory_direction_correct.detach(),
        "current_regressed": current_regressed.detach(),
        "memory_error": memory_error.detach(),
        "teacher_error": teacher_error.detach(),
        "current_error": current_error.detach(),
        "teacher_advantage_vs_memory": teacher_advantage.detach(),
        "current_regret_vs_memory": current_regret.detach(),
        "mild_compatibility": mild.detach(),
        "teacher_projection": teacher_projection,
    }


def best_memory_projection_summary(records: Sequence[Mapping]) -> dict:
    if not records:
        raise ValueError("v4.2 projection summary requires records.")
    summary = student_projection_summary(records)
    frame = pd.DataFrame(records)
    required = {
        "mode", "strong_distill", "weak_distill", "preserve", "memory_abstain",
        "memory_updated", "memory_before_error", "memory_after_error",
        "memory_improvement", "memory_error", "teacher_error", "current_error",
        "teacher_advantage_vs_memory", "current_regret_vs_memory",
        "compatibility", "mild_compatibility", "strong_gate", "weak_gate",
        "eligible_distill_mass", "effective_distill_gate", "preserve_gate",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("v4.2 records lack columns: {}".format(sorted(missing)))
    numeric_cols = [
        "memory_before_error", "memory_after_error", "memory_improvement",
        "memory_error", "teacher_error", "current_error",
        "teacher_advantage_vs_memory", "current_regret_vs_memory",
        "compatibility", "mild_compatibility", "strong_gate", "weak_gate",
        "eligible_distill_mass", "effective_distill_gate", "preserve_gate",
    ]
    if not np.isfinite(frame[numeric_cols].to_numpy(dtype=np.float64)).all():
        raise FloatingPointError("v4.2 records contain non-finite values.")
    strong = frame.strong_distill.astype(bool).to_numpy()
    weak = frame.weak_distill.astype(bool).to_numpy()
    preserve = frame.preserve.astype(bool).to_numpy()
    abstain = frame.memory_abstain.astype(bool).to_numpy()
    if np.any(strong.astype(int) + weak.astype(int) + preserve.astype(int) + abstain.astype(int) != 1):
        raise RuntimeError("v4.2 route accounting is invalid.")
    if np.any(frame.memory_after_error.to_numpy(float) > frame.memory_before_error.to_numpy(float) + 1e-10):
        raise RuntimeError("v4.2 best-memory error increased within an event.")

    summary.update({
        "strong_distill_fraction": float(strong.mean()),
        "weak_distill_fraction": float(weak.mean()),
        "preserve_fraction": float(preserve.mean()),
        "memory_abstain_fraction": float(abstain.mean()),
        "memory_update_fraction": float(frame.memory_updated.astype(bool).mean()),
        "mean_memory_improvement": float(frame.memory_improvement.mean()),
        "mean_memory_error": float(frame.memory_error.mean()),
        "mean_current_regret_vs_memory": float(frame.current_regret_vs_memory.mean()),
        "mean_teacher_advantage_vs_memory": float(frame.teacher_advantage_vs_memory.mean()),
        "mean_effective_distill_gate": float(frame.effective_distill_gate.mean()),
        "mean_preserve_gate": float(frame.preserve_gate.mean()),
    })
    if abs(
        summary["strong_distill_fraction"] + summary["weak_distill_fraction"]
        + summary["preserve_fraction"] + summary["memory_abstain_fraction"] - 1.0
    ) > 1e-12:
        raise RuntimeError("v4.2 route fractions do not sum to one.")
    for mode in MISSING_MODES:
        local = frame.loc[frame["mode"].astype(str).eq(mode)]
        if local.empty:
            raise RuntimeError("v4.2 records contain no {} events.".format(mode))
        summary["{}_memory_update_fraction".format(mode)] = float(local.memory_updated.astype(bool).mean())
        summary["{}_preserve_fraction".format(mode)] = float(local.preserve.astype(bool).mean())
    return summary


def _grid_row(frame: pd.DataFrame, run: str) -> Mapping:
    selected = frame.loc[
        frame.Seed.astype(int).eq(DEV_SEED) & frame.Run.astype(str).eq(str(run))
    ]
    if len(selected) != 1:
        raise RuntimeError("No unique Seed1113 grid row for {}.".format(run))
    return selected.iloc[0].to_dict()


def _group_value(groups, run, group_type, group_value, metric):
    selected = groups.loc[
        groups.Seed.astype(str).eq(str(DEV_SEED))
        & groups.Run.astype(str).eq(str(run))
        & groups.GroupType.astype(str).eq(str(group_type))
        & groups.GroupValue.astype(str).eq(str(group_value))
    ]
    if len(selected) != 1:
        raise RuntimeError("No unique group metric for {} {}={}.".format(run, group_type, group_value))
    return float(selected.iloc[0][metric])


def _transfer_value(frame, run, metric):
    selected = frame.loc[
        frame.Seed.astype(int).eq(DEV_SEED)
        & frame.Run.astype(str).eq(str(run))
        & frame.Mode.astype(str).eq("MISSING_ALL")
    ]
    if len(selected) != 1:
        raise RuntimeError("No unique MISSING_ALL transfer row for {}.".format(run))
    return float(selected.iloc[0][metric])


def dev_candidate_gate(candidate_row, v4_grid, v4p1_grid, groups, transfer, epoch_rows):
    """Frozen Seed1113 gate for v4.2 after observing v4/v4.1 failure modes."""
    v4 = _grid_row(v4_grid, "regret_preserve_cfcompat")
    v4p1 = _grid_row(v4p1_grid, "regret_preserve_guard_cfcompat")
    candidate_j = float(candidate_row["J_valid"])
    v4_j = float(v4["J_valid"])
    v4p1_j = float(v4p1["J_valid"])

    candidate_better = _group_value(groups, RUN, "teacher_condition", "better_and_correct", "gain_vs_DLF")
    v4p1_better = _group_value(groups, "regret_preserve_guard_cfcompat", "teacher_condition", "better_and_correct", "gain_vs_DLF")

    group_checks = {}
    group_values = {}
    for group_value, name in (("Q1_easy", "Q1"), ("Q4_hard", "Q4")):
        v4_gain = _group_value(groups, "regret_preserve_cfcompat", "baseline_error_quartile", group_value, "gain_vs_DLF")
        candidate_gain = _group_value(groups, RUN, "baseline_error_quartile", group_value, "gain_vs_DLF")
        minimum = v4_gain - Q1_Q4_MAX_GAIN_DEGRADATION_VS_V4
        group_values[name] = {"v4_gain": v4_gain, "candidate_gain": candidate_gain, "minimum_allowed": minimum}
        group_checks["{}_not_materially_degraded_vs_v4".format(name)] = bool(candidate_gain >= minimum)

    v4_harm = _group_value(groups, "regret_preserve_cfcompat", "all", "ALL", "harmful_imitation_rate")
    candidate_harm = _group_value(groups, RUN, "all", "ALL", "harmful_imitation_rate")
    v4_negative = _transfer_value(transfer, "regret_preserve_cfcompat", "negative_transfer_rate")
    candidate_negative = _transfer_value(transfer, RUN, "negative_transfer_rate")
    v4_severe = _transfer_value(transfer, "regret_preserve_cfcompat", "severe_negative_transfer_rate")
    candidate_severe = _transfer_value(transfer, RUN, "severe_negative_transfer_rate")
    v4_positive = _transfer_value(transfer, "regret_preserve_cfcompat", "positive_transfer_rate")
    candidate_positive = _transfer_value(transfer, RUN, "positive_transfer_rate")

    supporting = sum(
        int(row["Seed"]) == DEV_SEED
        and str(row["Run"]) == RUN
        and float(row["J_valid"]) <= v4_j + J_MAX_DEGRADATION_VS_V4
        for row in epoch_rows
    )
    fractions = {
        "strong_distill": float(candidate_row["projection_strong_distill_fraction"]),
        "weak_distill": float(candidate_row["projection_weak_distill_fraction"]),
        "preserve": float(candidate_row["projection_preserve_fraction"]),
        "abstain": float(candidate_row["projection_memory_abstain_fraction"]),
        "memory_update": float(candidate_row["projection_memory_update_fraction"]),
    }
    checks = {
        "J_not_materially_worse_than_v4": bool(candidate_j - v4_j <= J_MAX_DEGRADATION_VS_V4),
        "better_correct_retains_v4p1_within_0p01": bool(
            candidate_better >= v4p1_better - BETTER_CORRECT_MAX_GAIN_DEGRADATION_VS_V4P1
        ),
        "negative_transfer_not_worse_than_v4_by_0p01": bool(
            candidate_negative <= v4_negative + NEGATIVE_TRANSFER_MAX_INCREASE_VS_V4
        ),
        "severe_negative_transfer_not_worse_than_v4_by_0p01": bool(
            candidate_severe <= v4_severe + SEVERE_NEGATIVE_TRANSFER_MAX_INCREASE_VS_V4
        ),
        "positive_transfer_not_worse_than_v4_by_0p01": bool(
            candidate_positive >= v4_positive - POSITIVE_TRANSFER_MAX_DECREASE_VS_V4
        ),
        "harmful_imitation_not_worse_than_v4_by_0p01": bool(
            candidate_harm <= v4_harm + HARMFUL_IMITATION_MAX_INCREASE_VS_V4
        ),
        "memory_updates_exercised": bool(fractions["memory_update"] > 0.0),
        "preserve_exercised": bool(fractions["preserve"] > 0.0),
        "strong_and_weak_distill_exercised": bool(
            fractions["strong_distill"] > 0.0 and fractions["weak_distill"] > 0.0
        ),
        "two_noninferior_epochs_vs_v4": bool(supporting >= SUPPORTING_EPOCHS),
        **group_checks,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "candidate_J": candidate_j,
        "v4_J": v4_j,
        "v4p1_J": v4p1_j,
        "J_degradation_vs_v4": candidate_j - v4_j,
        "candidate_better_correct_gain": candidate_better,
        "v4p1_better_correct_gain": v4p1_better,
        "better_correct_change_vs_v4p1": candidate_better - v4p1_better,
        "candidate_negative_transfer_rate": candidate_negative,
        "v4_negative_transfer_rate": v4_negative,
        "candidate_severe_negative_transfer_rate": candidate_severe,
        "v4_severe_negative_transfer_rate": v4_severe,
        "candidate_positive_transfer_rate": candidate_positive,
        "v4_positive_transfer_rate": v4_positive,
        "candidate_harmful_imitation_rate": candidate_harm,
        "v4_harmful_imitation_rate": v4_harm,
        "supporting_epoch_count": int(supporting),
        "decision_fractions": fractions,
        "group_values": group_values,
    }


def frozen_thresholds() -> dict:
    return {
        "strong_distill_margin": STRONG_DISTILL_MARGIN,
        "weak_distill_scale": WEAK_DISTILL_SCALE,
        "memory_update_eps": MEMORY_UPDATE_EPS,
        "preserve_margin": PRESERVE_MARGIN_V4P2,
        "lambda_preserve": LAMBDA_PRESERVE_V4P2,
        "mild_cfcompat_base": MILD_CFCOMPAT_BASE_V4P2,
        "mild_cfcompat_scale": MILD_CFCOMPAT_SCALE_V4P2,
        "negative_transfer_margin": NEGATIVE_TRANSFER_MARGIN,
        "severe_negative_transfer_margin": SEVERE_NEGATIVE_TRANSFER_MARGIN,
        "J_max_degradation_vs_v4": J_MAX_DEGRADATION_VS_V4,
        "better_correct_max_gain_degradation_vs_v4p1": BETTER_CORRECT_MAX_GAIN_DEGRADATION_VS_V4P1,
        "Q1_Q4_max_gain_degradation_vs_v4": Q1_Q4_MAX_GAIN_DEGRADATION_VS_V4,
        "negative_transfer_max_increase_vs_v4": NEGATIVE_TRANSFER_MAX_INCREASE_VS_V4,
        "severe_negative_transfer_max_increase_vs_v4": SEVERE_NEGATIVE_TRANSFER_MAX_INCREASE_VS_V4,
        "positive_transfer_max_decrease_vs_v4": POSITIVE_TRANSFER_MAX_DECREASE_VS_V4,
        "harmful_imitation_max_increase_vs_v4": HARMFUL_IMITATION_MAX_INCREASE_VS_V4,
        "supporting_epochs": SUPPORTING_EPOCHS,
    }


__all__ = [
    "BestSoFarPredictionMemory", "DEV_SEED", "LAMBDA_PRESERVE_V4P2", "METHOD",
    "OUTPUT_TAG", "RUN", "RUNS", "VERSION", "WEAK_DISTILL_SCALE",
    "best_memory_projection_summary", "dev_candidate_gate", "frozen_thresholds",
    "negative_transfer_summary", "regret_best_memory_decision", "tiered_kd_loss",
]
