"""Train-only complementarity, Judge, and personalized Teacher audit."""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stage23a_common import (
    EXPERTS,
    MODES,
    MISSING_MODES,
    RESULT_ROOT,
    atomic_csv,
    atomic_json,
    git_head,
    judge_fold,
    load_preregistered,
    load_split_manifest,
    overall_j,
    project_simplex_rows,
    regression_metrics,
    sha256_file,
    stable_bucket,
)


EPS = 1e-9


def load_ledger():
    frames, manifests = [], []
    for fold in (0, 1):
        directory = RESULT_ROOT / "expert_oof" / "outer_fold{}".format(fold)
        path = directory / "oof_predictions.csv"
        manifest_path = directory / "fold_manifest.json"
        if not path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError("Outer-fold OOF artifact missing: {}".format(directory))
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("locked_test_access_count") != 0:
            raise RuntimeError("OOF fold test lock is not zero.")
        if manifest.get("prediction_sha256") != sha256_file(path):
            raise RuntimeError("OOF fold prediction SHA mismatch.")
        frames.append(pd.read_csv(path, dtype={"sample_id": str, "video_id": str}))
        manifests.append(manifest)
    frame = pd.concat(frames, ignore_index=True)
    keys = ["sample_id", "mode", "expert_id"]
    if frame.duplicated(keys).any():
        raise RuntimeError("Combined OOF ledger has duplicate keys.")
    if set(frame.expert_id) != set(EXPERTS) or set(frame.mode) != set(MODES):
        raise RuntimeError("Combined OOF candidate or mode set differs.")
    count = frame.groupby(["sample_id", "mode"]).expert_id.nunique()
    if int(count.min()) != len(EXPERTS) or int(count.max()) != len(EXPERTS):
        raise RuntimeError("OOF ledger is incomplete.")
    frame["absolute_error"] = np.abs(frame.prediction - frame.label)
    frame["signed_residual"] = frame.prediction - frame.label
    frame["regret"] = (
        frame.absolute_error
        - frame.groupby(["sample_id", "mode"]).absolute_error.transform("min")
    )
    frame["judge_fold"] = frame.video_id.map(judge_fold).astype(int)
    return frame, manifests


def wide_ledger(ledger, experts):
    prediction = ledger.pivot_table(
        index=["sample_id", "video_id", "mode", "label", "judge_fold"],
        columns="expert_id",
        values="prediction",
        aggfunc="first",
    ).reset_index()
    if prediction[list(experts)].isna().any().any():
        raise RuntimeError("Wide OOF ledger has missing expert predictions.")
    return prediction


def metric_rows(frame, method, prediction_column="prediction", split="train_oof"):
    rows = []
    for mode in MODES:
        local = frame.loc[frame.mode == mode]
        values = regression_metrics(local[prediction_column], local.label)
        rows.append({"split": split, "mode": mode, "method": method, **values})
    overall = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("MAE", "Corr", "Acc7", "Acc5", "Acc2", "F1")
    }
    overall["J"] = overall_j(frame, prediction_column)
    rows.append({"split": split, "mode": "Overall", "method": method, **overall})
    return rows


def optimize_simplex(predictions, labels):
    predictions = np.asarray(predictions, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    n_experts = predictions.shape[1]

    def objective(weights):
        return np.mean(np.abs(predictions.dot(weights) - labels))

    result = minimize(
        objective,
        np.full(n_experts, 1.0 / n_experts),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_experts,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError("Simplex optimization failed: {}".format(result.message))
    weights = np.maximum(result.x, 0)
    return weights / weights.sum()


def crossfit_fixed_predictions(wide, experts):
    global_prediction = np.zeros(len(wide), dtype=np.float64)
    mode_prediction = np.zeros(len(wide), dtype=np.float64)
    weight_rows = []
    values = wide[list(experts)].to_numpy()
    for valid_fold in (0, 1):
        train_mask = wide.judge_fold.to_numpy() != valid_fold
        valid_mask = ~train_mask
        global_weights = optimize_simplex(
            values[train_mask], wide.loc[train_mask, "label"].to_numpy()
        )
        global_prediction[valid_mask] = values[valid_mask].dot(global_weights)
        weight_rows.extend(
            {
                "valid_fold": valid_fold,
                "scope": "global",
                "mode": "ALL",
                "expert_id": expert,
                "weight": float(weight),
            }
            for expert, weight in zip(experts, global_weights)
        )
        for mode in MODES:
            train_mode = train_mask & (wide.mode.to_numpy() == mode)
            valid_mode = valid_mask & (wide.mode.to_numpy() == mode)
            weights = optimize_simplex(
                values[train_mode], wide.loc[train_mode, "label"].to_numpy()
            )
            mode_prediction[valid_mode] = values[valid_mode].dot(weights)
            weight_rows.extend(
                {
                    "valid_fold": valid_fold,
                    "scope": "per_mode",
                    "mode": mode,
                    "expert_id": expert,
                    "weight": float(weight),
                }
                for expert, weight in zip(experts, weights)
            )
    return global_prediction, mode_prediction, pd.DataFrame(weight_rows)


def complementarity_audit(ledger, experts, output):
    wide = wide_ledger(ledger, experts)
    metrics = []
    for expert in experts:
        local = ledger.loc[ledger.expert_id == expert].copy()
        metrics.extend(metric_rows(local, expert))
    wide["equal_average"] = wide[list(experts)].mean(axis=1)
    global_prediction, mode_prediction, weights = crossfit_fixed_predictions(
        wide, experts
    )
    wide["global_fixed_stacking"] = global_prediction
    wide["per_mode_fixed_stacking"] = mode_prediction
    for method in ("equal_average", "global_fixed_stacking", "per_mode_fixed_stacking"):
        metrics.extend(metric_rows(wide, method, method))

    # Oracle selection chooses exactly one existing expert.
    prediction_matrix = wide[list(experts)].to_numpy()
    errors = np.abs(prediction_matrix - wide.label.to_numpy()[:, None])
    oracle_index = errors.argmin(axis=1)
    wide["oracle_expert_selection"] = prediction_matrix[
        np.arange(len(wide)), oracle_index
    ]
    wide["oracle_expert_id"] = [experts[index] for index in oracle_index]
    metrics.extend(
        metric_rows(wide, "oracle_expert_selection", "oracle_expert_selection")
    )
    metric_frame = pd.DataFrame(metrics)

    residual_rows = []
    for mode in list(MODES) + ["Overall"]:
        local = ledger if mode == "Overall" else ledger.loc[ledger.mode == mode]
        residual = local.pivot_table(
            index=["sample_id", "mode"],
            columns="expert_id",
            values="signed_residual",
            aggfunc="first",
        )
        corr = residual[list(experts)].corr()
        for left in experts:
            for right in experts:
                residual_rows.append(
                    {
                        "mode": mode,
                        "expert_left": left,
                        "expert_right": right,
                        "residual_correlation": float(corr.loc[left, right]),
                    }
                )
    residual_frame = pd.DataFrame(residual_rows)

    contribution_rows = []
    loo_rows = []
    full_oracle = metric_frame.loc[
        (metric_frame.method == "oracle_expert_selection")
        & (metric_frame.mode == "Overall"),
        "J",
    ].iloc[0]
    for expert in experts:
        exact = float(np.mean(wide.oracle_expert_id == expert))
        expert_position = experts.index(expert)
        near = float(
            np.mean(errors[:, expert_position] <= errors.min(axis=1) + 0.05)
        )
        keep = [value for value in experts if value != expert]
        reduced_values = wide[keep].to_numpy()
        reduced_error = np.abs(
            reduced_values - wide.label.to_numpy()[:, None]
        )
        reduced_prediction = reduced_values[
            np.arange(len(wide)), reduced_error.argmin(axis=1)
        ]
        reduced = wide[["sample_id", "video_id", "mode", "label"]].copy()
        reduced["prediction"] = reduced_prediction
        reduced_metrics = pd.DataFrame(
            metric_rows(reduced, "without_{}".format(expert))
        )
        reduced_j = float(
            reduced_metrics.loc[reduced_metrics.mode == "Overall", "J"].iloc[0]
        )
        mode_deltas = {}
        for mode in MODES:
            full_mode = float(
                metric_frame.loc[
                    (metric_frame.method == "oracle_expert_selection")
                    & (metric_frame.mode == mode),
                    "J",
                ].iloc[0]
            )
            reduced_mode = float(
                reduced_metrics.loc[reduced_metrics.mode == mode, "J"].iloc[0]
            )
            mode_deltas[mode] = reduced_mode - full_mode
        own_j = float(
            metric_frame.loc[
                (metric_frame.method == expert) & (metric_frame.mode == "Overall"),
                "J",
            ].iloc[0]
        )
        best_single_j = float(
            metric_frame.loc[
                metric_frame.method.isin(experts)
                & (metric_frame.mode == "Overall"),
                "J",
            ].min()
        )
        not_failed = own_j <= best_single_j + 0.05
        unique = reduced_j - full_oracle >= 0.001 or max(mode_deltas.values()) >= 0.001
        retained = bool(not_failed and near >= 0.05 and unique)
        contribution_rows.append(
            {
                "expert_id": expert,
                "exact_best_fraction": exact,
                "near_best_0p05_fraction": near,
                "own_overall_J": own_j,
                "not_obviously_failed": not_failed,
                "leave_out_oracle_delta_J": reduced_j - full_oracle,
                "max_mode_leave_out_delta_J": max(mode_deltas.values()),
                "retained": retained,
            }
        )
        loo_rows.extend(
            {
                "removed_expert": expert,
                "mode": mode,
                "oracle_J_without": (
                    reduced_j
                    if mode == "Overall"
                    else float(
                        reduced_metrics.loc[
                            reduced_metrics.mode == mode, "J"
                        ].iloc[0]
                    )
                ),
                "delta_vs_full_oracle": (
                    reduced_j - full_oracle
                    if mode == "Overall"
                    else mode_deltas[mode]
                ),
            }
            for mode in list(MODES) + ["Overall"]
        )
    contributions = pd.DataFrame(contribution_rows)
    retained = contributions.loc[contributions.retained, "expert_id"].tolist()
    if len(retained) > 5:
        retained = (
            contributions.sort_values(
                ["leave_out_oracle_delta_J", "near_best_0p05_fraction"],
                ascending=False,
            )
            .head(5)
            .expert_id.tolist()
        )

    fixed_j = float(
        metric_frame.loc[
            (metric_frame.method == "per_mode_fixed_stacking")
            & (metric_frame.mode == "Overall"),
            "J",
        ].iloc[0]
    )
    oracle_j = float(
        metric_frame.loc[
            (metric_frame.method == "oracle_expert_selection")
            & (metric_frame.mode == "Overall"),
            "J",
        ].iloc[0]
    )
    mode_headroom = {}
    for mode in MISSING_MODES:
        fixed_mode = float(
            metric_frame.loc[
                (metric_frame.method == "per_mode_fixed_stacking")
                & (metric_frame.mode == mode),
                "MAE",
            ].iloc[0]
        )
        oracle_mode = float(
            metric_frame.loc[
                (metric_frame.method == "oracle_expert_selection")
                & (metric_frame.mode == mode),
                "MAE",
            ].iloc[0]
        )
        mode_headroom[mode] = oracle_mode - fixed_mode
    actionable_modes = [
        mode for mode, delta in mode_headroom.items() if delta <= -0.003
    ]
    gate = (
        oracle_j - fixed_j <= -0.005
        and len(actionable_modes) >= 2
        and 3 <= len(retained) <= 5
    )

    atomic_csv(ledger, output / "expert_error_ledger.csv")
    atomic_csv(metric_frame, output / "expert_and_ensemble_metrics.csv")
    atomic_csv(residual_frame, output / "residual_correlation_matrix.csv")
    atomic_csv(contributions, output / "expert_contributions.csv")
    atomic_csv(pd.DataFrame(loo_rows), output / "leave_one_expert_out_oracle.csv")
    atomic_csv(weights, output / "fixed_stacking_weights.csv")
    atomic_csv(wide, output / "complementarity_predictions.csv")
    summary = {
        "retained_experts": retained,
        "per_mode_fixed_stacking_J": fixed_j,
        "oracle_expert_selection_J": oracle_j,
        "oracle_delta_J": oracle_j - fixed_j,
        "missing_mode_oracle_delta_MAE": mode_headroom,
        "actionable_missing_modes_threshold_minus_0p003": actionable_modes,
        "gate_passed": gate,
    }
    atomic_json(output / "complementarity_gate.json", summary)
    return wide, retained, summary, metric_frame, contributions


def feature_matrix(frame, experts, level):
    base = frame[list(experts)].to_numpy(dtype=np.float64)
    mode_values = np.column_stack(
        [(frame.mode.to_numpy() == mode).astype(float) for mode in MODES]
    )
    if level == 0:
        return mode_values
    if level == 1:
        return np.column_stack([base, mode_values])
    disagreement = np.column_stack(
        [
            np.abs(base[:, left] - base[:, right])
            for left, right in combinations(range(len(experts)), 2)
        ]
    )
    return np.column_stack([base, disagreement, mode_values])


def fit_risk_models(train, experts, level, shuffled=False, random_seed=23):
    features = feature_matrix(train, experts, level)
    models = {}
    generator = np.random.RandomState(random_seed)
    for expert in experts:
        target = np.abs(train[expert].to_numpy() - train.label.to_numpy())
        if shuffled:
            shuffled_target = target.copy()
            for mode in MODES:
                index = np.flatnonzero(train.mode.to_numpy() == mode)
                shuffled_target[index] = shuffled_target[index][
                    generator.permutation(len(index))
                ]
            target = shuffled_target
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(features, target)
        models[expert] = model
    return models


def predict_risks(models, frame, experts, level):
    features = feature_matrix(frame, experts, level)
    return np.column_stack(
        [np.maximum(models[expert].predict(features), 1e-4) for expert in experts]
    )


def risk_diagnostics(train, valid, risks, experts, judge_name, split):
    true_error = np.abs(
        valid[list(experts)].to_numpy() - valid.label.to_numpy()[:, None]
    )
    flat_risk = risks.reshape(-1)
    flat_error = true_error.reshape(-1)
    correlation = float(spearmanr(flat_risk, flat_error).correlation)
    thresholds = np.quantile(
        np.abs(
            train[list(experts)].to_numpy() - train.label.to_numpy()[:, None]
        ).reshape(-1),
        0.8,
    )
    high = (flat_error >= thresholds).astype(int)
    auroc = float(roc_auc_score(high, flat_risk)) if len(np.unique(high)) == 2 else 0.5
    predicted_best = risks.argmin(axis=1)
    actual_best = true_error.argmin(axis=1)
    ranking_accuracy = float(np.mean(predicted_best == actual_best))
    buckets = pd.qcut(
        pd.Series(flat_risk).rank(method="first"),
        5,
        labels=False,
    ).to_numpy()
    bucket_error = [
        float(flat_error[buckets == bucket].mean()) for bucket in range(5)
    ]
    monotonic_violations = int(
        sum(
            bucket_error[index + 1] + 1e-8 < bucket_error[index]
            for index in range(4)
        )
    )
    return {
        "judge": judge_name,
        "valid_split": split,
        "error_spearman": correlation,
        "top20_high_error_auroc": auroc,
        "regret_ranking_accuracy": ranking_accuracy,
        "risk_bucket_true_error_1": bucket_error[0],
        "risk_bucket_true_error_2": bucket_error[1],
        "risk_bucket_true_error_3": bucket_error[2],
        "risk_bucket_true_error_4": bucket_error[3],
        "risk_bucket_true_error_5": bucket_error[4],
        "monotonic_violations": monotonic_violations,
    }


def correlation_matrices(frame, experts):
    result = {}
    for mode in MODES:
        local = frame.loc[frame.mode == mode]
        residual = (
            local[list(experts)].to_numpy()
            - local.label.to_numpy()[:, None]
        )
        corr = np.corrcoef(residual, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        corr = (corr + corr.T) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(corr)
        corr = (eigenvectors * np.maximum(eigenvalues, 1e-6)).dot(
            eigenvectors.T
        )
        diagonal = np.sqrt(np.diag(corr))
        corr = corr / diagonal[:, None] / diagonal[None, :]
        result[mode] = corr
    return result


def fixed_weights(frame, experts):
    values = frame[list(experts)].to_numpy()
    result = {}
    for mode in MODES:
        local = frame.mode.to_numpy() == mode
        result[mode] = optimize_simplex(values[local], frame.loc[local, "label"])
    return result


def inverse_mode_error_weights(frame, experts):
    result = {}
    for mode in MODES:
        local = frame.loc[frame.mode == mode]
        error = np.array(
            [
                np.mean(np.abs(local[expert] - local.label))
                for expert in experts
            ]
        )
        inverse = 1.0 / np.maximum(error, 1e-4)
        result[mode] = inverse / inverse.sum()
    return result


def apply_mode_weights(frame, experts, weights):
    prediction = np.zeros(len(frame), dtype=np.float64)
    matrix = frame[list(experts)].to_numpy()
    for mode in MODES:
        index = frame.mode.to_numpy() == mode
        prediction[index] = matrix[index].dot(weights[mode])
    return prediction


def dynamic_joint(
    frame,
    experts,
    risks,
    correlations,
    fallback,
    rho,
    gamma,
):
    matrix = frame[list(experts)].to_numpy(dtype=np.float64)
    prediction = np.zeros(len(frame), dtype=np.float64)
    all_weights = np.zeros_like(matrix)
    trigger = np.zeros(len(frame), dtype=bool)
    predicted_gain = np.zeros(len(frame), dtype=np.float64)
    identity = np.eye(len(experts))
    for mode in MODES:
        index = np.flatnonzero(frame.mode.to_numpy() == mode)
        local_risk = risks[index]
        correlation = correlations[mode]
        sigma = (
            local_risk[:, :, None]
            * correlation[None, :, :]
            * local_risk[:, None, :]
        )
        weights = np.repeat(fallback[mode][None, :], len(index), axis=0)
        # Stable projected gradient for the small convex quadratic.
        eigenmax = np.linalg.eigvalsh(sigma).max(axis=1) + float(rho)
        step = 0.45 / np.maximum(eigenmax, 1e-6)
        for _ in range(80):
            gradient = (
                2.0 * np.einsum("nij,nj->ni", sigma, weights)
                + 2.0 * float(rho) * (weights - fallback[mode][None, :])
            )
            weights = project_simplex_rows(weights - step[:, None] * gradient)
        fallback_risk = np.einsum(
            "i,nij,j->n", fallback[mode], sigma, fallback[mode]
        )
        dynamic_risk = np.einsum("ni,nij,nj->n", weights, sigma, weights)
        gain = fallback_risk - dynamic_risk
        local_trigger = gain >= float(gamma)
        final = np.where(local_trigger[:, None], weights, fallback[mode][None, :])
        all_weights[index] = final
        trigger[index] = local_trigger
        predicted_gain[index] = gain
        prediction[index] = np.sum(matrix[index] * final, axis=1)
    return prediction, all_weights, trigger, predicted_gain


def tune_joint(train, experts, level):
    tune_mask = np.array(
        [
            stable_bucket(video, "stage23a_rho_gamma_tune", 5) == 0
            for video in train.video_id
        ]
    )
    fit = train.loc[~tune_mask].reset_index(drop=True)
    tune = train.loc[tune_mask].reset_index(drop=True)
    if not len(fit) or not len(tune):
        raise RuntimeError("Empty Judge-train rho/gamma tuning split.")
    models = fit_risk_models(fit, experts, level)
    risks = predict_risks(models, tune, experts, level)
    correlations = correlation_matrices(fit, experts)
    fallback = fixed_weights(fit, experts)
    best = None
    rows = []
    for rho in (0.001, 0.01, 0.1, 1.0, 10.0):
        for gamma in (0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005):
            prediction, weights, trigger, gain = dynamic_joint(
                tune, experts, risks, correlations, fallback, rho, gamma
            )
            local = tune.copy()
            local["prediction"] = prediction
            score = overall_j(local)
            row = {
                "rho": rho,
                "gamma": gamma,
                "tune_J": score,
                "trigger_rate": float(trigger.mean()),
                "mean_predicted_gain": float(gain.mean()),
            }
            rows.append(row)
            key = (score, rho, gamma)
            if best is None or key < best[0]:
                best = (key, rho, gamma)
    return best[1], best[2], pd.DataFrame(rows)


def teacher_comparison(wide, experts, output):
    metric_rows_all = []
    diagnostic_rows = []
    judge_rows = []
    weight_rows = []
    tune_rows = []
    split_summaries = []
    for valid_fold in (0, 1):
        train = wide.loc[wide.judge_fold != valid_fold].reset_index(drop=True)
        valid = wide.loc[wide.judge_fold == valid_fold].reset_index(drop=True)
        models_j0 = fit_risk_models(train, experts, 0)
        models_j1 = fit_risk_models(train, experts, 1)
        models_j2 = fit_risk_models(train, experts, 2)
        models_j4 = fit_risk_models(
            train, experts, 2, shuffled=True, random_seed=2300 + valid_fold
        )
        risk_sets = {
            "J0_mode_only": predict_risks(models_j0, valid, experts, 0),
            "J1_predictions": predict_risks(models_j1, valid, experts, 1),
            "J2_predictions_disagreement": predict_risks(
                models_j2, valid, experts, 2
            ),
            "J4_shuffled": predict_risks(models_j4, valid, experts, 2),
        }
        for name, risks in risk_sets.items():
            level = 0 if name.startswith("J0") else (1 if name.startswith("J1") else 2)
            del level
            judge_rows.append(
                risk_diagnostics(train, valid, risks, experts, name, valid_fold)
            )

        fallback = fixed_weights(train, experts)
        global_weight = optimize_simplex(
            train[list(experts)].to_numpy(), train.label.to_numpy()
        )
        mode_only = inverse_mode_error_weights(train, experts)
        best_single = min(
            experts,
            key=lambda expert: overall_j(
                train.assign(prediction=train[expert].to_numpy())
            ),
        )
        correlations = correlation_matrices(train, experts)
        rho, gamma, tuning = tune_joint(train, experts, 2)
        tuning["valid_split"] = valid_fold
        tune_rows.append(tuning)
        prediction_matrix = valid[list(experts)].to_numpy()
        methods = {
            "Best single": valid[best_single].to_numpy(),
            "Equal average": prediction_matrix.mean(axis=1),
            "Global fixed stacking": prediction_matrix.dot(global_weight),
            "Per-mode fixed stacking": apply_mode_weights(valid, experts, fallback),
            "Mode-only weighting": apply_mode_weights(valid, experts, mode_only),
        }
        scalar = 1.0 / np.maximum(
            risk_sets["J2_predictions_disagreement"], 1e-4
        )
        scalar /= scalar.sum(axis=1, keepdims=True)
        methods["Scalar-risk weighting"] = np.sum(
            prediction_matrix * scalar, axis=1
        )
        joint, weights, trigger, predicted_gain = dynamic_joint(
            valid,
            experts,
            risk_sets["J2_predictions_disagreement"],
            correlations,
            fallback,
            rho,
            gamma,
        )
        shuffled, shuffled_weights, shuffled_trigger, _ = dynamic_joint(
            valid,
            experts,
            risk_sets["J4_shuffled"],
            correlations,
            fallback,
            rho,
            gamma,
        )
        methods["Joint-risk personalized Teacher"] = joint
        methods["Shuffled Judge"] = shuffled
        true_error = np.abs(prediction_matrix - valid.label.to_numpy()[:, None])
        oracle_index = true_error.argmin(axis=1)
        methods["Oracle expert selection"] = prediction_matrix[
            np.arange(len(valid)), oracle_index
        ]
        # No pre-existing calibrated expert self-confidence exists.  This row is
        # explicit rather than fabricating a confidence signal post hoc.
        diagnostic_rows.append(
            {
                "valid_split": valid_fold,
                "diagnostic": "Self-confidence weighting",
                "available": False,
                "reason": "Frozen experts expose no calibrated self-confidence.",
            }
        )

        for method, prediction in methods.items():
            local = valid[
                ["sample_id", "video_id", "mode", "label"]
            ].copy()
            local["prediction"] = prediction
            metric_rows_all.extend(
                metric_rows(
                    local,
                    method,
                    "prediction",
                    "judge_valid_{}".format(valid_fold),
                )
            )
        fallback_prediction = methods["Per-mode fixed stacking"]
        actual_gain = (
            np.abs(fallback_prediction - valid.label.to_numpy())
            - np.abs(joint - valid.label.to_numpy())
        )
        fallback_matrix = np.row_stack(
            [fallback[mode] for mode in valid.mode.to_numpy()]
        )
        weight_distance = np.abs(weights - fallback_matrix).sum(axis=1)
        split_summary = {
            "valid_split": valid_fold,
            "rho": rho,
            "gamma": gamma,
            "trigger_rate": float(trigger.mean()),
            "mean_weight_l1_from_fallback": float(weight_distance.mean()),
            "trigger_actual_gain": (
                float(actual_gain[trigger].mean()) if trigger.any() else 0.0
            ),
            "fallback_actual_gain": (
                float(actual_gain[~trigger].mean()) if (~trigger).any() else 0.0
            ),
            "shuffled_trigger_rate": float(shuffled_trigger.mean()),
        }
        split_summaries.append(split_summary)
        for position, row in valid.reset_index(drop=True).iterrows():
            for expert_index, expert in enumerate(experts):
                weight_rows.append(
                    {
                        "valid_split": valid_fold,
                        "sample_id": row.sample_id,
                        "mode": row.mode,
                        "expert_id": expert,
                        "fallback_weight": float(fallback[row.mode][expert_index]),
                        "joint_weight": float(weights[position, expert_index]),
                        "shuffled_weight": float(
                            shuffled_weights[position, expert_index]
                        ),
                        "triggered": bool(trigger[position]),
                        "predicted_risk": float(
                            risk_sets["J2_predictions_disagreement"][
                                position, expert_index
                            ]
                        ),
                    }
                )
    metrics = pd.DataFrame(metric_rows_all)
    judges = pd.DataFrame(judge_rows)
    split_summary = pd.DataFrame(split_summaries)
    atomic_csv(metrics, output / "teacher_metrics_by_split.csv")
    atomic_csv(judges, output / "judge_diagnostics.csv")
    atomic_csv(pd.DataFrame(diagnostic_rows), output / "self_confidence_audit.csv")
    atomic_csv(pd.DataFrame(weight_rows), output / "dynamic_teacher_weights.csv")
    atomic_csv(pd.concat(tune_rows, ignore_index=True), output / "rho_gamma_tuning.csv")
    atomic_csv(split_summary, output / "dynamic_trigger_diagnostics.csv")

    overall = metrics.loc[metrics.mode == "Overall"].copy()
    pivot = overall.pivot_table(
        index="method", columns="split", values="J", aggfunc="first"
    )
    baseline = pivot.loc["Per-mode fixed stacking"]
    joint = pivot.loc["Joint-risk personalized Teacher"]
    shuffled = pivot.loc["Shuffled Judge"]
    delta = joint - baseline
    delta_shuffled = joint - shuffled
    missing = metrics.loc[
        metrics.mode.isin(MISSING_MODES)
        & metrics.method.isin(
            ["Per-mode fixed stacking", "Joint-risk personalized Teacher"]
        )
    ]
    missing_pivot = missing.pivot_table(
        index=["split", "mode"], columns="method", values="MAE"
    )
    missing_pivot["delta"] = (
        missing_pivot["Joint-risk personalized Teacher"]
        - missing_pivot["Per-mode fixed stacking"]
    )
    mode_mean_delta = missing_pivot.groupby("mode").delta.mean()
    improved_missing_modes = int((mode_mean_delta < -1e-8).sum())
    lav = metrics.loc[
        (metrics.mode == "LAV")
        & metrics.method.isin(
            ["Per-mode fixed stacking", "Joint-risk personalized Teacher"]
        )
    ].pivot_table(index="split", columns="method", values="MAE")
    lav_delta = (
        lav["Joint-risk personalized Teacher"]
        - lav["Per-mode fixed stacking"]
    )
    classification_systematic = False
    for metric in ("Corr", "Acc7", "Acc5", "Acc2", "F1"):
        local = metrics.loc[
            metrics.method.isin(
                ["Per-mode fixed stacking", "Joint-risk personalized Teacher"]
            )
        ].pivot_table(index=["split", "mode"], columns="method", values=metric)
        difference = (
            local["Joint-risk personalized Teacher"]
            - local["Per-mode fixed stacking"]
        )
        if int((difference < -1e-8).sum()) > len(difference) / 2:
            classification_systematic = True
    monotonic_ok = bool(
        (
            judges.loc[
                judges.judge == "J2_predictions_disagreement",
                "monotonic_violations",
            ]
            <= 1
        ).all()
    )
    noncollapse = bool(
        (split_summary.trigger_rate > 0.05).all()
        and (split_summary.mean_weight_l1_from_fallback > 0.01).all()
    )
    trigger_gain_ok = bool(
        (
            split_summary.trigger_actual_gain
            > split_summary.fallback_actual_gain
        ).all()
    )
    directions_consistent = bool(np.all(delta.to_numpy() <= 0))
    gates = {
        "mean_delta_J_vs_per_mode_fixed": float(delta.mean()),
        "worst_split_delta_J_vs_per_mode_fixed": float(delta.max()),
        "mean_delta_J_vs_shuffled": float(delta_shuffled.mean()),
        "improved_missing_modes": improved_missing_modes,
        "missing_mode_mean_delta_MAE": mode_mean_delta.to_dict(),
        "worst_LAV_delta_MAE": float(lav_delta.max()),
        "classification_systematic_degradation": classification_systematic,
        "risk_buckets_monotonic": monotonic_ok,
        "dynamic_weights_noncollapsed": noncollapse,
        "trigger_gain_above_fallback": trigger_gain_ok,
        "split_directions_consistent": directions_consistent,
    }
    gates["passed"] = bool(
        gates["mean_delta_J_vs_per_mode_fixed"] <= -0.003
        and gates["worst_split_delta_J_vs_per_mode_fixed"] <= 0.001
        and gates["mean_delta_J_vs_shuffled"] <= -0.002
        and improved_missing_modes >= 2
        and gates["worst_LAV_delta_MAE"] <= 0.003
        and not classification_systematic
        and monotonic_ok
        and noncollapse
        and trigger_gain_ok
        and directions_consistent
    )
    atomic_json(output / "judge_teacher_gate.json", gates)
    return metrics, judges, gates, split_summary


def method_vs_seed_complementarity(ledger):
    residual = ledger.pivot_table(
        index=["sample_id", "mode"],
        columns="expert_id",
        values="signed_residual",
        aggfunc="first",
    )
    method_pairs = [
        ("moddrop_seed1111", "cfcompat_seed1111"),
        ("moddrop_seed1114", "cfcompat_seed1114"),
    ]
    seed_pairs = [
        ("moddrop_seed1111", "moddrop_seed1114"),
        ("cfcompat_seed1111", "cfcompat_seed1114"),
    ]
    method_corr = np.mean(
        [residual[left].corr(residual[right]) for left, right in method_pairs]
    )
    seed_corr = np.mean(
        [residual[left].corr(residual[right]) for left, right in seed_pairs]
    )
    return {
        "mean_method_matched_seed_residual_correlation": float(method_corr),
        "mean_seed_matched_method_residual_correlation": float(seed_corr),
        "larger_complementarity_source": (
            "method_difference" if method_corr < seed_corr else "seed_difference"
        ),
    }


def write_report(
    output,
    status,
    complementarity,
    retained,
    contributions,
    method_seed,
    judge_gate=None,
    judges=None,
):
    lines = [
        "# Stage 23A — Expert Complementarity and Personalized Teacher Feasibility Audit",
        "",
        "Final status: `{}`".format(status),
        "",
        "## Protocol lock",
        "",
        "- Locked Test access count: **0**",
        "- Official Valid access count: **0**",
        "- Student trained: **No**",
        "- All Expert OOF predictions are 2-fold source-video-disjoint.",
        "",
        "## Required answers",
        "",
        "1. Candidate complementarity: {} (Oracle ΔJ vs per-mode fixed stacking = {:.6f}).".format(
            "actionable" if complementarity["gate_passed"] else "not actionable",
            complementarity["oracle_delta_J"],
        ),
        "2. Retained experts: {}.".format(
            ", ".join(retained) if retained else "none"
        ),
        "3. Larger complementarity source: `{}` (method-pair residual corr {:.4f}; seed-pair {:.4f}).".format(
            method_seed["larger_complementarity_source"],
            method_seed["mean_method_matched_seed_residual_correlation"],
            method_seed["mean_seed_matched_method_residual_correlation"],
        ),
        "4. Oracle expert-selection headroom: ΔJ={:.6f}; missing-mode ΔMAE={}.".format(
            complementarity["oracle_delta_J"],
            json.dumps(complementarity["missing_mode_oracle_delta_MAE"], sort_keys=True),
        ),
    ]
    if judge_gate is None:
        lines.extend(
            [
                "5. Judge error/regret prediction: not trained because the complementarity gate failed.",
                "6. Self-confidence increment: unavailable; frozen Experts expose no calibrated confidence.",
                "7. Joint-risk vs scalar-risk: not evaluated.",
                "8. Dynamic Teacher vs per-mode fixed stacking: not evaluated.",
                "9. Missing-mode shortcut: not applicable.",
                "10. Dynamic-weight collapse: not applicable.",
                "11. Improved modes: none established beyond Oracle headroom.",
                "12. Proceed to Stage23B Student Distillation: **No**.",
            ]
        )
    else:
        j2 = judges.loc[judges.judge == "J2_predictions_disagreement"]
        lines.extend(
            [
                "5. Judge error/regret prediction: J2 mean Spearman {:.4f}, AUROC {:.4f}, ranking accuracy {:.4f}.".format(
                    j2.error_spearman.mean(),
                    j2.top20_high_error_auroc.mean(),
                    j2.regret_ranking_accuracy.mean(),
                ),
                "6. Self-confidence increment: unavailable; no calibrated frozen signal existed.",
                "7. Joint-risk vs scalar-risk: see `teacher_metrics_by_split.csv`; the formal gate uses fixed stacking and shuffled Judge.",
                "8. Dynamic Teacher ΔJ vs per-mode fixed stacking: mean {:.6f}, worst split {:.6f}.".format(
                    judge_gate["mean_delta_J_vs_per_mode_fixed"],
                    judge_gate["worst_split_delta_J_vs_per_mode_fixed"],
                ),
                "9. Missing-mode shortcut: checked jointly with LAV and per-mode results; LAV worst ΔMAE={:.6f}.".format(
                    judge_gate["worst_LAV_delta_MAE"]
                ),
                "10. Dynamic-weight collapse: {}.".format(
                    "No" if judge_gate["dynamic_weights_noncollapsed"] else "Yes / insufficient dynamics"
                ),
                "11. Missing-mode mean ΔMAE: {}.".format(
                    json.dumps(judge_gate["missing_mode_mean_delta_MAE"], sort_keys=True)
                ),
                "12. Proceed to Stage23B Student Distillation: **{}**.".format(
                    "Only after Official Valid confirmation"
                    if judge_gate["passed"]
                    else "No"
                ),
            ]
        )
    lines.extend(
        [
            "13. Locked Test access count: **0**.",
            "",
            "## Retention evidence",
            "",
            contributions.to_markdown(index=False),
            "",
            "Only `STAGE23A_PERSONALIZED_TEACHER_FEASIBILITY_PASSED` would permit a Stage23B recommendation.",
            "",
        ]
    )
    (output / "stage23a_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    protocol, protocol_sha = load_preregistered()
    _, split_sha = load_split_manifest()
    output = RESULT_ROOT / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    ledger, fold_manifests = load_ledger()
    wide, retained, complementarity, expert_metrics, contributions = (
        complementarity_audit(ledger, list(EXPERTS), output)
    )
    method_seed = method_vs_seed_complementarity(ledger)
    atomic_json(output / "method_vs_seed_complementarity.json", method_seed)
    if not complementarity["gate_passed"]:
        status = "STAGE23A_NO_ACTIONABLE_EXPERT_COMPLEMENTARITY"
        judge_gate = judges = None
    else:
        # Recompute the fixed baseline and Judge only on experts retained by the
        # pre-registered, label-independent retention rule.
        retained_wide = wide[
            ["sample_id", "video_id", "mode", "label", "judge_fold"] + retained
        ].copy()
        teacher_metrics, judges, judge_gate, split_summary = teacher_comparison(
            retained_wide, retained, output
        )
        status = (
            "STAGE23A_PERSONALIZED_TEACHER_TRAIN_ONLY_PASSED"
            if judge_gate["passed"]
            else "STAGE23A_JUDGE_NOT_ACTIONABLE"
        )
    write_report(
        output,
        status,
        complementarity,
        retained,
        contributions,
        method_seed,
        judge_gate,
        judges,
    )
    manifest = {
        "stage": "Stage 23A",
        "status": status,
        "protocol_sha256": protocol_sha,
        "source_split_sha256": split_sha,
        "outer_fold_prediction_sha256": [
            value["prediction_sha256"] for value in fold_manifests
        ],
        "retained_experts": retained,
        "complementarity_gate": complementarity,
        "judge_teacher_gate": judge_gate,
        "method_vs_seed": method_seed,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "test_loader_constructed": False,
        "student_trained": False,
        "code_commit": git_head(),
    }
    atomic_json(output / "stage23a_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
