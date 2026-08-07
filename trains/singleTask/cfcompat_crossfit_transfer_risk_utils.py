"""Cross-fitted transfer-risk gating utilities for CFCompatKD v5.

v5 replaces sample-local/oracle Train routing with a label-free risk predictor
that can be evaluated on unseen samples.  Train labels are used only to create
a binary supervision target: whether the frozen full-modality Teacher improves
on the frozen missing-modality ModDrop baseline by at least 0.02 MAE.  The gate
features themselves never contain the label.

All three missing modes belonging to one Train sample are assigned to the same
cross-fit fold.  The Student trajectory uses only out-of-fold probabilities.
A full-Train model is fitted only for Valid gate diagnostics and is never used
to route Train events.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .cfcompat_regret_preserve_utils import negative_transfer_summary
from .cfcompat_student_safe_abstain_utils import student_projection_summary
from .missing_utils import MISSING_MODES


VERSION = "cfcompat_crossfit_transfer_risk_valid_screen_v5"
METHOD = "DLF-CrossFitted-TransferRisk-CFCompatKD-v5"
OUTPUT_TAG = "cfcompat_crossfit_transfer_risk_v5"
DEV_SEED = 1113
RUN = "crossfit_transfer_risk_cfcompat"
RUNS = (RUN,)

BENEFIT_MARGIN = 0.02
CROSSFIT_FOLDS = 5
CROSSFIT_SEED = 20260807
LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 2000
RISK_PROBABILITY_THRESHOLD = 0.5

# Cheap gate pre-screen.  Frozen before the first Seed1113 v5 run.
OOF_AUC_MIN = 0.58
VALID_AUC_MIN = 0.55
VALID_BRIER_MAX_EXCESS_VS_CONSTANT = 0.0

# Candidate promotion gate, also frozen before the first Seed1113 v5 run.
J_MAX_DEGRADATION_VS_REPLAY = 0.002
J_MAX_DEGRADATION_VS_V4 = 0.005
NEGATIVE_TRANSFER_REDUCTION_REQUIRED_VS_REPLAY = 0.02
SEVERE_NEGATIVE_TRANSFER_MAX_INCREASE_VS_REPLAY = 0.01
POSITIVE_TRANSFER_MAX_DECREASE_VS_REPLAY = 0.01
HARMFUL_IMITATION_MAX_INCREASE_VS_REPLAY = 0.01
Q1_Q4_MAX_GAIN_DEGRADATION_VS_REPLAY = 0.005
BETTER_CORRECT_MAX_GAIN_DEGRADATION_VS_V4 = 0.005
SUPPORTING_EPOCHS = 2

FEATURE_COLUMNS = (
    "baseline_missing_prediction",
    "baseline_full_prediction",
    "teacher_full_prediction",
    "initial_student_missing_prediction",
    "teacher_minus_baseline_missing",
    "student0_minus_baseline_missing",
    "teacher_minus_student0",
    "baseline_full_minus_missing",
    "teacher_minus_baseline_full",
    "abs_teacher_minus_baseline_missing",
    "abs_student0_minus_baseline_missing",
    "abs_teacher_minus_student0",
    "abs_baseline_full_minus_missing",
    "abs_teacher_minus_baseline_full",
    "abs_baseline_missing_prediction",
    "abs_teacher_full_prediction",
    "abs_initial_student_missing_prediction",
    "mode_LA",
    "mode_LV",
    "mode_L",
)


def add_transfer_risk_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the fixed label-free design matrix plus the Train-only target."""
    required = {
        "sample_index", "sample_id", "mode", "label",
        "baseline_missing_prediction", "baseline_full_prediction",
        "teacher_full_prediction", "initial_student_missing_prediction",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Transfer-risk raw frame lacks: {}".format(sorted(missing)))
    result = frame.copy()
    if not set(result.mode.astype(str)).issubset(set(MISSING_MODES)):
        raise ValueError("Transfer-risk rows may contain only LA/LV/L.")

    b = result.baseline_missing_prediction.to_numpy(dtype=np.float64)
    f = result.baseline_full_prediction.to_numpy(dtype=np.float64)
    t = result.teacher_full_prediction.to_numpy(dtype=np.float64)
    s0 = result.initial_student_missing_prediction.to_numpy(dtype=np.float64)
    y = result.label.to_numpy(dtype=np.float64)

    result["teacher_minus_baseline_missing"] = t - b
    result["student0_minus_baseline_missing"] = s0 - b
    result["teacher_minus_student0"] = t - s0
    result["baseline_full_minus_missing"] = f - b
    result["teacher_minus_baseline_full"] = t - f
    result["abs_teacher_minus_baseline_missing"] = np.abs(t - b)
    result["abs_student0_minus_baseline_missing"] = np.abs(s0 - b)
    result["abs_teacher_minus_student0"] = np.abs(t - s0)
    result["abs_baseline_full_minus_missing"] = np.abs(f - b)
    result["abs_teacher_minus_baseline_full"] = np.abs(t - f)
    result["abs_baseline_missing_prediction"] = np.abs(b)
    result["abs_teacher_full_prediction"] = np.abs(t)
    result["abs_initial_student_missing_prediction"] = np.abs(s0)
    for mode in MISSING_MODES:
        result["mode_{}".format(mode)] = result.mode.astype(str).eq(mode).astype(np.float64)

    baseline_error = np.abs(b - y)
    teacher_error = np.abs(t - y)
    advantage = baseline_error - teacher_error
    result["baseline_error"] = baseline_error
    result["teacher_error"] = teacher_error
    result["teacher_advantage_vs_baseline"] = advantage
    result["beneficial_label"] = (advantage >= BENEFIT_MARGIN).astype(np.int64)

    numeric = list(FEATURE_COLUMNS) + [
        "label", "baseline_error", "teacher_error", "teacher_advantage_vs_baseline"
    ]
    if not np.isfinite(result[numeric].to_numpy(dtype=np.float64)).all():
        raise FloatingPointError("Transfer-risk feature frame contains NaN/Inf.")
    return result


def _fold_map(sample_indices: Sequence[int]) -> dict:
    unique = np.asarray(sorted({int(value) for value in sample_indices}), dtype=np.int64)
    if len(unique) != 1284:
        raise RuntimeError("Cross-fit gate expects exactly 1284 Train sample groups.")
    rng = np.random.default_rng(CROSSFIT_SEED)
    shuffled = rng.permutation(unique)
    return {int(index): int(position % CROSSFIT_FOLDS) for position, index in enumerate(shuffled)}


def _design(frame: pd.DataFrame) -> np.ndarray:
    values = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise FloatingPointError("Cross-fit design matrix contains NaN/Inf.")
    return values


def _new_model() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=LOGISTIC_C,
                    solver="lbfgs",
                    max_iter=LOGISTIC_MAX_ITER,
                    random_state=CROSSFIT_SEED,
                ),
            ),
        ]
    )


def _metric_payload(labels, probabilities, constant_probability) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(np.unique(labels)) != 2:
        raise RuntimeError("Risk-gate metric requires both beneficial/non-beneficial classes.")
    if not np.isfinite(probabilities).all() or np.any(probabilities <= 0.0) or np.any(probabilities >= 1.0):
        raise FloatingPointError("Risk-gate probabilities must be finite and strictly inside (0,1).")
    constant = np.full(len(labels), float(constant_probability), dtype=np.float64)
    predicted = probabilities >= RISK_PROBABILITY_THRESHOLD
    return {
        "N": int(len(labels)),
        "beneficial_prevalence": float(labels.mean()),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "constant_brier": float(brier_score_loss(labels, constant)),
        "accuracy_at_0p5": float((predicted.astype(np.int64) == labels).mean()),
        "mean_probability": float(probabilities.mean()),
        "positive_probability_fraction": float((probabilities >= RISK_PROBABILITY_THRESHOLD).mean()),
    }


def fit_crossfit_transfer_risk_gate(train_features: pd.DataFrame, valid_features: pd.DataFrame):
    """Fit grouped 5-fold OOF gate and a full-Train diagnostic model for Valid."""
    train = add_transfer_risk_features(train_features).sort_values(
        ["sample_index", "mode"], kind="mergesort"
    ).reset_index(drop=True)
    valid = add_transfer_risk_features(valid_features).sort_values(
        ["sample_index", "mode"], kind="mergesort"
    ).reset_index(drop=True)
    if len(train) != 1284 * len(MISSING_MODES):
        raise RuntimeError("Cross-fit Train feature grid must contain 1284 x 3 events.")
    if train.duplicated(["sample_index", "mode"]).any():
        raise RuntimeError("Cross-fit Train feature grid has duplicate sample/mode rows.")
    if valid.duplicated(["sample_index", "mode"]).any():
        raise RuntimeError("Cross-fit Valid feature grid has duplicate sample/mode rows.")

    fold_by_sample = _fold_map(train.sample_index.astype(int).tolist())
    train["crossfit_fold"] = train.sample_index.astype(int).map(fold_by_sample).astype(int)
    oof = np.full(len(train), np.nan, dtype=np.float64)
    fold_rows = []
    y_all = train.beneficial_label.to_numpy(dtype=np.int64)

    for fold in range(CROSSFIT_FOLDS):
        held_mask = train.crossfit_fold.to_numpy(dtype=np.int64) == fold
        fit_mask = ~held_mask
        if not held_mask.any() or not fit_mask.any():
            raise RuntimeError("Cross-fit fold {} is empty.".format(fold))
        held_samples = set(train.loc[held_mask, "sample_index"].astype(int))
        fit_samples = set(train.loc[fit_mask, "sample_index"].astype(int))
        if held_samples.intersection(fit_samples):
            raise RuntimeError("Cross-fit sample leakage detected in fold {}.".format(fold))
        y_fit = y_all[fit_mask]
        y_held = y_all[held_mask]
        if len(np.unique(y_fit)) != 2 or len(np.unique(y_held)) != 2:
            raise RuntimeError("Cross-fit fold {} lacks both classes.".format(fold))
        model = _new_model()
        model.fit(_design(train.loc[fit_mask]), y_fit)
        probability = model.predict_proba(_design(train.loc[held_mask]))[:, 1]
        oof[held_mask] = probability
        fold_rows.append(
            {
                "fold": int(fold),
                "train_sample_count": int(len(fit_samples)),
                "heldout_sample_count": int(len(held_samples)),
                "heldout_event_count": int(held_mask.sum()),
                "heldout_prevalence": float(y_held.mean()),
                "heldout_auc": float(roc_auc_score(y_held, probability)),
            }
        )

    if not np.isfinite(oof).all():
        raise RuntimeError("OOF transfer-risk probabilities are incomplete.")
    train["oof_benefit_probability"] = oof

    train_prevalence = float(y_all.mean())
    oof_metrics = _metric_payload(y_all, oof, train_prevalence)

    final_model = _new_model()
    final_model.fit(_design(train), y_all)
    valid_probability = final_model.predict_proba(_design(valid))[:, 1]
    valid["full_train_benefit_probability"] = valid_probability
    valid_metrics = _metric_payload(
        valid.beneficial_label.to_numpy(dtype=np.int64),
        valid_probability,
        train_prevalence,
    )

    scaler = final_model.named_steps["scale"]
    logistic = final_model.named_steps["logistic"]
    model_snapshot = {
        "feature_columns": list(FEATURE_COLUMNS),
        "scaler_mean": [float(value) for value in scaler.mean_],
        "scaler_scale": [float(value) for value in scaler.scale_],
        "logistic_coef": [float(value) for value in logistic.coef_.reshape(-1)],
        "logistic_intercept": float(logistic.intercept_[0]),
    }
    summary = {
        "fold_count": CROSSFIT_FOLDS,
        "crossfit_seed": CROSSFIT_SEED,
        "benefit_margin": BENEFIT_MARGIN,
        "train_sample_group_count": int(train.sample_index.nunique()),
        "train_event_count": int(len(train)),
        "valid_event_count": int(len(valid)),
        "train_oof": oof_metrics,
        "valid_full_train_gate": valid_metrics,
        "folds": fold_rows,
        "full_train_model": model_snapshot,
    }
    summary["prescreen_checks"] = {
        "oof_auc_ge_0p58": bool(oof_metrics["roc_auc"] >= OOF_AUC_MIN),
        "valid_auc_ge_0p55": bool(valid_metrics["roc_auc"] >= VALID_AUC_MIN),
        "valid_brier_not_worse_than_constant": bool(
            valid_metrics["brier"]
            <= valid_metrics["constant_brier"] + VALID_BRIER_MAX_EXCESS_VS_CONSTANT
        ),
        "all_folds_have_both_classes": bool(len(fold_rows) == CROSSFIT_FOLDS),
    }
    summary["prescreen_passed"] = bool(all(summary["prescreen_checks"].values()))
    return train, valid, summary


def probability_to_distill_weight(probability: torch.Tensor) -> torch.Tensor:
    """Posterior-benefit weight: zero below 0.5, linear to one at probability one."""
    probability = probability.detach().view(-1)
    if not torch.isfinite(probability).all() or torch.any(probability <= 0.0) or torch.any(probability >= 1.0):
        raise FloatingPointError("OOF benefit probabilities must remain in (0,1).")
    return torch.clamp(2.0 * probability - 1.0, min=0.0, max=1.0).detach()


def transfer_risk_kd_loss(
    student_prediction: torch.Tensor,
    teacher_safe_target: torch.Tensor,
    active: torch.Tensor,
    benefit_probability: torch.Tensor,
):
    """True-strength KD: probabilities attenuate magnitude instead of renormalizing it away."""
    student = student_prediction.view(-1)
    target = teacher_safe_target.detach().view(-1)
    active_float = active.detach().view(-1).to(student)
    risk_weight = probability_to_distill_weight(benefit_probability).to(student)
    effective_gate = active_float * risk_weight
    each = F.smooth_l1_loss(student, target, reduction="none")
    loss = torch.sum(effective_gate * each) / (torch.sum(active_float) + 1e-8)
    if not torch.isfinite(loss):
        raise FloatingPointError("Cross-fit transfer-risk KD became non-finite.")
    return loss, each, effective_gate.detach(), active_float.detach(), risk_weight.detach()


def crossfit_projection_summary(records: Sequence[Mapping]) -> dict:
    if not records:
        raise ValueError("Cross-fit transfer-risk summary requires records.")
    summary = student_projection_summary(records)
    frame = pd.DataFrame(records)
    required = {
        "mode", "oof_benefit_probability", "risk_weight", "effective_gate",
        "safe_teacher_active", "distill_loss_each",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Cross-fit projection records lack: {}".format(sorted(missing)))
    numeric = frame[[
        "oof_benefit_probability", "risk_weight", "effective_gate", "distill_loss_each"
    ]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise FloatingPointError("Cross-fit projection records contain NaN/Inf.")
    active = frame.safe_teacher_active.astype(bool).to_numpy()
    summary.update(
        {
            "mean_oof_benefit_probability": float(frame.oof_benefit_probability.mean()),
            "mean_risk_weight": float(frame.risk_weight.mean()),
            "positive_risk_weight_fraction": float((frame.risk_weight.to_numpy(float) > 0.0).mean()),
            "safe_teacher_active_fraction": float(active.mean()),
            "mean_effective_gate": float(frame.effective_gate.mean()),
            "effective_distill_fraction": float((frame.effective_gate.to_numpy(float) > 0.0).mean()),
        }
    )
    for mode in MISSING_MODES:
        local = frame.loc[frame.mode.astype(str).eq(mode)]
        if local.empty:
            raise RuntimeError("No {} events in cross-fit projection records.".format(mode))
        summary["{}_effective_distill_fraction".format(mode)] = float(
            (local.effective_gate.to_numpy(float) > 0.0).mean()
        )
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
        raise RuntimeError("No unique group value for {} {}={}.".format(run, group_type, group_value))
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


def dev_candidate_gate(candidate_row, replay_grid, v4_grid, groups, transfer, epoch_rows, gate_summary):
    replay = _grid_row(replay_grid, "cfcompat_replay")
    v4 = _grid_row(v4_grid, "regret_preserve_cfcompat")
    candidate_j = float(candidate_row["J_valid"])
    replay_j = float(replay["J_valid"])
    v4_j = float(v4["J_valid"])

    replay_negative = _transfer_value(transfer, "cfcompat_replay", "negative_transfer_rate")
    candidate_negative = _transfer_value(transfer, RUN, "negative_transfer_rate")
    replay_severe = _transfer_value(transfer, "cfcompat_replay", "severe_negative_transfer_rate")
    candidate_severe = _transfer_value(transfer, RUN, "severe_negative_transfer_rate")
    replay_positive = _transfer_value(transfer, "cfcompat_replay", "positive_transfer_rate")
    candidate_positive = _transfer_value(transfer, RUN, "positive_transfer_rate")
    replay_harm = _group_value(groups, "cfcompat_replay", "all", "ALL", "harmful_imitation_rate")
    candidate_harm = _group_value(groups, RUN, "all", "ALL", "harmful_imitation_rate")

    group_checks = {}
    group_values = {}
    for group_value, name in (("Q1_easy", "Q1"), ("Q4_hard", "Q4")):
        replay_gain = _group_value(groups, "cfcompat_replay", "baseline_error_quartile", group_value, "gain_vs_DLF")
        candidate_gain = _group_value(groups, RUN, "baseline_error_quartile", group_value, "gain_vs_DLF")
        minimum = replay_gain - Q1_Q4_MAX_GAIN_DEGRADATION_VS_REPLAY
        group_values[name] = {"replay_gain": replay_gain, "candidate_gain": candidate_gain, "minimum_allowed": minimum}
        group_checks["{}_not_materially_degraded_vs_replay".format(name)] = bool(candidate_gain >= minimum)

    v4_better = _group_value(groups, "regret_preserve_cfcompat", "teacher_condition", "better_and_correct", "gain_vs_DLF")
    candidate_better = _group_value(groups, RUN, "teacher_condition", "better_and_correct", "gain_vs_DLF")
    supporting = sum(
        int(row["Seed"]) == DEV_SEED
        and str(row["Run"]) == RUN
        and float(row["J_valid"]) <= replay_j + J_MAX_DEGRADATION_VS_REPLAY
        for row in epoch_rows
    )

    checks = {
        "risk_gate_prescreen_passed": bool(gate_summary.get("prescreen_passed", False)),
        "J_not_materially_worse_than_replay": bool(candidate_j <= replay_j + J_MAX_DEGRADATION_VS_REPLAY),
        "J_not_materially_worse_than_v4": bool(candidate_j <= v4_j + J_MAX_DEGRADATION_VS_V4),
        "negative_transfer_reduction_ge_0p02_vs_replay": bool(
            replay_negative - candidate_negative >= NEGATIVE_TRANSFER_REDUCTION_REQUIRED_VS_REPLAY
        ),
        "severe_negative_transfer_not_worse_than_replay_by_0p01": bool(
            candidate_severe <= replay_severe + SEVERE_NEGATIVE_TRANSFER_MAX_INCREASE_VS_REPLAY
        ),
        "positive_transfer_not_worse_than_replay_by_0p01": bool(
            candidate_positive >= replay_positive - POSITIVE_TRANSFER_MAX_DECREASE_VS_REPLAY
        ),
        "harmful_imitation_not_worse_than_replay_by_0p01": bool(
            candidate_harm <= replay_harm + HARMFUL_IMITATION_MAX_INCREASE_VS_REPLAY
        ),
        "better_correct_retains_v4_within_0p005": bool(
            candidate_better >= v4_better - BETTER_CORRECT_MAX_GAIN_DEGRADATION_VS_V4
        ),
        "two_noninferior_epochs_vs_replay": bool(supporting >= SUPPORTING_EPOCHS),
        "risk_weight_not_collapsed": bool(
            float(candidate_row["projection_positive_risk_weight_fraction"]) > 0.01
            and float(candidate_row["projection_positive_risk_weight_fraction"]) < 0.95
        ),
        **group_checks,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "candidate_J": candidate_j,
        "replay_J": replay_j,
        "v4_J": v4_j,
        "candidate_negative_transfer_rate": candidate_negative,
        "replay_negative_transfer_rate": replay_negative,
        "negative_transfer_reduction_vs_replay": replay_negative - candidate_negative,
        "candidate_severe_negative_transfer_rate": candidate_severe,
        "replay_severe_negative_transfer_rate": replay_severe,
        "candidate_positive_transfer_rate": candidate_positive,
        "replay_positive_transfer_rate": replay_positive,
        "candidate_harmful_imitation_rate": candidate_harm,
        "replay_harmful_imitation_rate": replay_harm,
        "candidate_better_correct_gain": candidate_better,
        "v4_better_correct_gain": v4_better,
        "supporting_epoch_count": int(supporting),
        "group_values": group_values,
    }


def frozen_thresholds() -> dict:
    return {
        "benefit_margin": BENEFIT_MARGIN,
        "crossfit_folds": CROSSFIT_FOLDS,
        "crossfit_seed": CROSSFIT_SEED,
        "logistic_C": LOGISTIC_C,
        "risk_probability_threshold": RISK_PROBABILITY_THRESHOLD,
        "oof_auc_min": OOF_AUC_MIN,
        "valid_auc_min": VALID_AUC_MIN,
        "valid_brier_max_excess_vs_constant": VALID_BRIER_MAX_EXCESS_VS_CONSTANT,
        "J_max_degradation_vs_replay": J_MAX_DEGRADATION_VS_REPLAY,
        "J_max_degradation_vs_v4": J_MAX_DEGRADATION_VS_V4,
        "negative_transfer_reduction_required_vs_replay": NEGATIVE_TRANSFER_REDUCTION_REQUIRED_VS_REPLAY,
        "severe_negative_transfer_max_increase_vs_replay": SEVERE_NEGATIVE_TRANSFER_MAX_INCREASE_VS_REPLAY,
        "positive_transfer_max_decrease_vs_replay": POSITIVE_TRANSFER_MAX_DECREASE_VS_REPLAY,
        "harmful_imitation_max_increase_vs_replay": HARMFUL_IMITATION_MAX_INCREASE_VS_REPLAY,
        "Q1_Q4_max_gain_degradation_vs_replay": Q1_Q4_MAX_GAIN_DEGRADATION_VS_REPLAY,
        "better_correct_max_gain_degradation_vs_v4": BETTER_CORRECT_MAX_GAIN_DEGRADATION_VS_V4,
        "supporting_epochs": SUPPORTING_EPOCHS,
    }
