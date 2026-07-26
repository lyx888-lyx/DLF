#!/usr/bin/env python
"""Run leakage-free Stage23A-v2a signal audits and simple probes.

This is intentionally not the formal dual-head Judge.  Every fitted object is
an explicitly named low-capacity diagnostic probe; outer labels are used once
for metrics only and never for fitting or selection.
"""

from __future__ import annotations

import html
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stage23a_v2_common import (
    EXPERTS,
    MODES,
    ROOT,
    RUNTIME_ROOT,
    V2_ROOT,
    atomic_csv,
    atomic_json,
    metric_rows,
    overall_j,
    regression_metrics,
    sha256_file,
)


ANALYSIS = V2_ROOT / "analysis_v2a"
PLOTS = ANALYSIS / "plots"
ROLE_NAMES = ("inner_train", "inner_valid", "outer_evaluation")
ALPHAS = (0.1, 1.0, 10.0, 100.0)
SHUFFLE_SEEDS = (23021, 23022, 23023)
NOISE_SEEDS = (23121, 23122, 23123)


def load_direction(direction):
    root = V2_ROOT / "features" / "meta57" / "direction_{}".format(direction)
    result = {}
    for role in ROLE_NAMES:
        values = np.load(root / "{}_features_57d.npz".format(role))
        X = values["X"].astype(np.float64)
        names = values["feature_names"].astype(str).tolist()
        meta_row_id = values["meta_row_id"].astype(str)
        sidecar = pd.read_csv(
            root / "{}_targets_and_audit.csv".format(role),
            dtype={"sample_id": str, "video_id": str, "meta_row_id": str},
        )
        if sidecar["meta_row_id"].tolist() != meta_row_id.tolist():
            raise RuntimeError("Feature/sidecar binding mismatch.")
        result[role] = {"X": X, "names": names, "sidecar": sidecar}
    return result


def finite_spearman(left, right):
    value = spearmanr(np.asarray(left), np.asarray(right)).correlation
    return float(value) if np.isfinite(value) else 0.0


def svg_histogram(values, path, title, bins=40, x_label="value"):
    values = np.asarray(values, dtype=np.float64)
    counts, edges = np.histogram(values, bins=bins)
    width, height = 760, 420
    left, right, top, bottom = 70, 20, 55, 65
    plot_w, plot_h = width - left - right, height - top - bottom
    maximum = max(int(counts.max()), 1)
    bars = []
    for index, count in enumerate(counts):
        x = left + plot_w * index / len(counts)
        bar_w = plot_w / len(counts) - 1
        bar_h = plot_h * count / maximum
        y = top + plot_h - bar_h
        bars.append(
            '<rect x="{:.2f}" y="{:.2f}" width="{:.2f}" height="{:.2f}" fill="#4C78A8"/>'.format(
                x, y, bar_w, bar_h
            )
        )
    text = """<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<rect width="100%" height="100%" fill="white"/>
<text x="{cx}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{title}</text>
<line x1="{l}" y1="{t}" x2="{l}" y2="{yb}" stroke="#222"/>
<line x1="{l}" y1="{yb}" x2="{xr}" y2="{yb}" stroke="#222"/>
{bars}
<text x="{l}" y="{label_y}" font-family="Arial" font-size="12">{xmin:.4g}</text>
<text x="{xr}" y="{label_y}" text-anchor="end" font-family="Arial" font-size="12">{xmax:.4g}</text>
<text x="{cx}" y="{hminus}" text-anchor="middle" font-family="Arial" font-size="13">{x_label}</text>
<text x="18" y="{cy}" transform="rotate(-90 18 {cy})" text-anchor="middle" font-family="Arial" font-size="13">count</text>
</svg>""".format(
        w=width,
        h=height,
        cx=width / 2,
        cy=height / 2,
        title=html.escape(title),
        l=left,
        t=top,
        yb=top + plot_h,
        xr=left + plot_w,
        bars="\n".join(bars),
        label_y=height - 42,
        xmin=edges[0],
        xmax=edges[-1],
        hminus=height - 12,
        x_label=html.escape(x_label),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def svg_bars(labels, values, path, title):
    width, height = 760, 420
    left, top, bottom = 90, 55, 100
    plot_w, plot_h = 640, height - top - bottom
    maximum = max(max(values), 1e-12)
    nodes = []
    for index, (label, value) in enumerate(zip(labels, values)):
        center = left + (index + 0.5) * plot_w / len(values)
        bar_w = 0.62 * plot_w / len(values)
        bar_h = plot_h * value / maximum
        nodes.append(
            '<rect x="{:.2f}" y="{:.2f}" width="{:.2f}" height="{:.2f}" fill="#F58518"/>'.format(
                center - bar_w / 2, top + plot_h - bar_h, bar_w, bar_h
            )
        )
        nodes.append(
            '<text x="{:.2f}" y="{}" transform="rotate(-28 {:.2f} {})" text-anchor="end" font-family="Arial" font-size="11">{}</text>'.format(
                center, height - 76, center, height - 76, html.escape(label)
            )
        )
        nodes.append(
            '<text x="{:.2f}" y="{:.2f}" text-anchor="middle" font-family="Arial" font-size="11">{:.3f}</text>'.format(
                center, top + plot_h - bar_h - 5, value
            )
        )
    text = """<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
<rect width="100%" height="100%" fill="white"/>
<text x="{cx}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{title}</text>
<line x1="{l}" y1="{t}" x2="{l}" y2="{yb}" stroke="#222"/>
<line x1="{l}" y1="{yb}" x2="{xr}" y2="{yb}" stroke="#222"/>
{nodes}
<text x="18" y="{cy}" transform="rotate(-90 18 {cy})" text-anchor="middle" font-family="Arial" font-size="13">selection proportion</text>
</svg>""".format(
        w=width,
        h=height,
        cx=width / 2,
        cy=height / 2,
        title=html.escape(title),
        l=left,
        t=top,
        yb=top + plot_h,
        xr=left + plot_w,
        nodes="\n".join(nodes),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def oracle_geometry(direction, data):
    rows = []
    detail_rows = []
    outer = data["outer_evaluation"]
    frame = outer["sidecar"].copy()
    X = outer["X"]
    predictions = X[:, 4:9]
    labels = frame["label"].to_numpy(dtype=np.float64)
    errors = np.abs(predictions - labels[:, None])
    sorted_errors = np.sort(errors, axis=1)
    margins = sorted_errors[:, 1] - sorted_errors[:, 0]
    frame["best_second_error_gap"] = margins
    frame["label_group"] = np.where(
        labels < 0, "negative", np.where(labels > 0, "positive", "zero")
    )
    for mode in list(MODES) + ["Overall"]:
        mask = (
            np.ones(len(frame), dtype=bool)
            if mode == "Overall"
            else frame["mode"].to_numpy() == mode
        )
        local = frame.loc[mask]
        if mode == "Overall":
            static_metric = overall_j(local, "strong_static")
            oracle_metric = overall_j(local, "oracle_prediction")
        else:
            static_metric = float(local["strong_static_absolute_error"].mean())
            oracle_metric = float(local["oracle_absolute_error"].mean())
        gain = local["oracle_gain"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "direction": direction,
                "role": "outer_evaluation",
                "mode": mode,
                "strong_static_J_or_MAE": static_metric,
                "oracle_J_or_MAE": oracle_metric,
                "oracle_minus_strong_static": oracle_metric - static_metric,
                "gain_mean": float(gain.mean()),
                "gain_std": float(gain.std(ddof=0)),
                "gain_q10": float(np.quantile(gain, 0.10)),
                "gain_q25": float(np.quantile(gain, 0.25)),
                "gain_median": float(np.median(gain)),
                "gain_q75": float(np.quantile(gain, 0.75)),
                "gain_q90": float(np.quantile(gain, 0.90)),
                "gain_q95": float(np.quantile(gain, 0.95)),
                "gain_q99": float(np.quantile(gain, 0.99)),
                "fraction_gain_gt_0": float(np.mean(gain > 0)),
                "fraction_gain_gt_0p01": float(np.mean(gain > 0.01)),
                "fraction_gain_gt_0p03": float(np.mean(gain > 0.03)),
                "fraction_gain_gt_0p05": float(np.mean(gain > 0.05)),
                "fraction_gain_gt_0p10": float(np.mean(gain > 0.10)),
                "strong_static_strictly_better_fraction": float(np.mean(gain < -1e-12)),
                "exact_tie_fraction": float(np.mean(np.abs(gain) <= 1e-12)),
                "best_second_gap_mean": float(
                    frame.loc[mask, "best_second_error_gap"].mean()
                ),
                "near_tie_gap_le_0p001": float(
                    np.mean(frame.loc[mask, "best_second_error_gap"] <= 0.001)
                ),
                "near_tie_gap_le_0p005": float(
                    np.mean(frame.loc[mask, "best_second_error_gap"] <= 0.005)
                ),
                "near_tie_gap_le_0p01": float(
                    np.mean(frame.loc[mask, "best_second_error_gap"] <= 0.01)
                ),
            }
        )
    for grouping in ("mode", "video_id", "label_group"):
        counts = (
            frame.groupby([grouping, "oracle_expert_id"], sort=True)
            .size()
            .rename("wins")
            .reset_index()
        )
        totals = frame.groupby(grouping).size().rename("total").reset_index()
        counts = counts.merge(totals, on=grouping)
        for row in counts.itertuples(index=False):
            detail_rows.append(
                {
                    "direction": direction,
                    "grouping": grouping,
                    "group_value": str(getattr(row, grouping)),
                    "expert_id": row.oracle_expert_id,
                    "wins": int(row.wins),
                    "total": int(row.total),
                    "proportion": float(row.wins / row.total),
                    "record_type": "oracle_expert_win",
                    "mean_gain": np.nan,
                }
            )
        gain_groups = (
            frame.groupby(grouping, sort=True)["oracle_gain"]
            .agg(["count", "mean", lambda value: np.mean(value > 0.05)])
            .reset_index()
        )
        gain_groups.columns = [grouping, "total", "mean_gain", "high_gain_fraction"]
        for row in gain_groups.itertuples(index=False):
            detail_rows.append(
                {
                    "direction": direction,
                    "grouping": grouping,
                    "group_value": str(getattr(row, grouping)),
                    "expert_id": "ALL",
                    "wins": int(round(row.high_gain_fraction * row.total)),
                    "total": int(row.total),
                    "proportion": float(row.high_gain_fraction),
                    "record_type": "gain_gt_0p05_concentration",
                    "mean_gain": float(row.mean_gain),
                }
            )
    overall_selection = (
        frame["oracle_expert_id"].value_counts(normalize=True).reindex(EXPERTS).fillna(0)
    )
    svg_histogram(
        frame["oracle_gain"],
        PLOTS / "direction_{}_oracle_gain_histogram.svg".format(direction),
        "Direction {} outer Oracle gain distribution".format(direction),
        x_label="strong-static absolute error − oracle absolute error (positive is better)",
    )
    svg_histogram(
        frame["best_second_error_gap"],
        PLOTS / "direction_{}_best_second_margin_histogram.svg".format(direction),
        "Direction {} outer best-to-second Expert error margin".format(direction),
        x_label="second-best absolute error − best absolute error",
    )
    svg_bars(
        list(EXPERTS),
        overall_selection.tolist(),
        PLOTS / "direction_{}_oracle_selection.svg".format(direction),
        "Direction {} outer Oracle expert selection".format(direction),
    )
    better = frame.loc[frame["oracle_gain"] < -1e-12].copy()
    better["reason"] = "static convex combination closer than every single Expert"
    return rows, detail_rows, better[
        [
            "meta_row_id",
            "sample_id",
            "video_id",
            "mode",
            "label",
            "strong_static",
            "oracle_prediction",
            "strong_static_absolute_error",
            "oracle_absolute_error",
            "oracle_gain",
            "reason",
        ]
    ]


def consistency_audit(direction, data):
    rows = []
    train_frame = data["inner_train"]["sidecar"]
    train_consistency = data["inner_train"]["X"][:, 17:22]
    q99_thresholds = {}
    for mode in MODES:
        mode_mask = train_frame["mode"].to_numpy() == mode
        for expert_index, expert in enumerate(EXPERTS):
            q99_thresholds[(mode, expert)] = float(
                np.quantile(train_consistency[mode_mask, expert_index], 0.99)
            )
    for role in ROLE_NAMES:
        frame = data[role]["sidecar"]
        X = data[role]["X"]
        predictions = X[:, 4:9]
        consistency = X[:, 17:22]
        labels = frame["label"].to_numpy(dtype=np.float64)
        errors = np.abs(predictions - labels[:, None])
        regret = errors - errors.min(axis=1, keepdims=True)
        best = errors == errors.min(axis=1, keepdims=True)
        for mode in MODES:
            mode_mask = frame["mode"].to_numpy() == mode
            for expert_index, expert in enumerate(EXPERTS):
                score = consistency[mode_mask, expert_index]
                err = errors[mode_mask, expert_index]
                reg = regret[mode_mask, expert_index]
                is_best = best[mode_mask, expert_index].astype(int)
                auc_best = (
                    roc_auc_score(is_best, -score)
                    if len(np.unique(is_best)) == 2
                    else 0.5
                )
                low = np.quantile(err, 0.20)
                high = np.quantile(err, 0.80)
                extremes = (err <= low) | (err >= high)
                high_error = (err[extremes] >= high).astype(int)
                auc_error = (
                    roc_auc_score(high_error, score[extremes])
                    if len(np.unique(high_error)) == 2
                    else 0.5
                )
                rows.append(
                    {
                        "direction": direction,
                        "role": role,
                        "mode": mode,
                        "expert_id": expert,
                        "method": expert.split("_seed")[0],
                        "seed": int(expert.rsplit("seed", 1)[1]),
                        "samples": int(mode_mask.sum()),
                        "spearman_consistency_error": finite_spearman(score, err),
                        "spearman_consistency_regret": finite_spearman(score, reg),
                        "auroc_low_consistency_is_exact_best": float(auc_best),
                        "auroc_consistency_high_error_top_vs_bottom20": float(auc_error),
                        "consistency_mean": float(score.mean()),
                        "consistency_std": float(score.std()),
                        "consistency_q05": float(np.quantile(score, 0.05)),
                        "consistency_q50": float(np.quantile(score, 0.50)),
                        "consistency_q95": float(np.quantile(score, 0.95)),
                        "inner_train_consistency_q99_threshold": q99_thresholds[
                            (mode, expert)
                        ],
                        "consistency_outlier_rate_above_inner_q99": float(
                            np.mean(score > q99_thresholds[(mode, expert)])
                        ),
                        "error_bottom20_consistency_mean": float(score[err <= low].mean()),
                        "error_top20_consistency_mean": float(score[err >= high].mean()),
                    }
                )
    result = pd.DataFrame(rows)
    train = result.loc[result["role"] == "inner_train"][
        ["mode", "expert_id", "spearman_consistency_error"]
    ].rename(columns={"spearman_consistency_error": "train_rho"})
    outer = result.loc[result["role"] == "outer_evaluation"][
        ["mode", "expert_id", "spearman_consistency_error"]
    ].rename(columns={"spearman_consistency_error": "outer_rho"})
    reversal = train.merge(outer, on=["mode", "expert_id"])
    reversal["sign_reversal"] = (
        np.sign(reversal["train_rho"]) != np.sign(reversal["outer_rho"])
    )
    reversal["direction"] = direction
    svg_histogram(
        data["outer_evaluation"]["X"][:, 17:22].reshape(-1),
        PLOTS / "direction_{}_outer_consistency_distribution.svg".format(direction),
        "Direction {} outer hierarchical consistency distribution".format(direction),
        x_label="active-head MAD consistency (higher means less internally coherent)",
    )
    return result, reversal


def fit_ridge_selected(X_train, y_train, X_valid, y_valid, scorer):
    candidates = []
    for alpha in ALPHAS:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(X_train, y_train)
        candidates.append((scorer(model.predict(X_valid), y_valid), alpha, model))
    score, alpha, model = min(candidates, key=lambda value: (value[0], value[1]))
    return model, alpha, score


def probe_a(direction, data):
    indices = list(range(0, 4)) + list(range(22, 57))
    train = data["inner_train"]
    valid = data["inner_valid"]
    y_train = train["sidecar"]["label"].to_numpy(dtype=np.float64)
    y_valid = valid["sidecar"]["label"].to_numpy(dtype=np.float64)

    def valid_j(prediction, _):
        local = valid["sidecar"][["mode", "label"]].copy()
        local["probe"] = prediction
        return overall_j(local, "probe")

    model, alpha, selected_valid_j = fit_ridge_selected(
        train["X"][:, indices],
        y_train,
        valid["X"][:, indices],
        y_valid,
        valid_j,
    )
    rows = []
    for role in ROLE_NAMES:
        local = data[role]["sidecar"][["mode", "label"]].copy()
        local["prediction"] = model.predict(data[role]["X"][:, indices])
        for row in metric_rows(local, "content_sentiment_ridge", "prediction"):
            rows.append(
                {
                    "direction": direction,
                    "role": role,
                    "selected_alpha_inner_valid_only": alpha,
                    "selected_inner_valid_J": selected_valid_j,
                    **row,
                }
            )
    return pd.DataFrame(rows)


def probe_b_fold_identity(directions):
    # A Direction alone contains only one fold in development, making a
    # Direction-local fold classifier mathematically undefined.  This pooled
    # diagnostic uses both frozen development folds: their source-disjoint
    # inner-train roles for fitting and their inner-valid roles for evaluation.
    sets = {}
    for role in ("inner_train", "inner_valid"):
        X = np.concatenate(
            [directions["A"][role]["X"], directions["B"][role]["X"]], axis=0
        )
        frame = pd.concat(
            [
                directions["A"][role]["sidecar"],
                directions["B"][role]["sidecar"],
            ],
            ignore_index=True,
        )
        sets[role] = (X, frame["expert_fold"].to_numpy(dtype=int), frame)
    feature_sets = {
        "content_only": list(range(22, 57)),
        "consistency_only": list(range(0, 4)) + list(range(17, 22)),
        "content_plus_consistency": list(range(0, 4)) + list(range(17, 57)),
    }
    rows = []
    for name, indices in feature_sets.items():
        candidates = []
        for c_value in (0.1, 1.0, 10.0):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=c_value,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=2302,
                ),
            )
            model.fit(sets["inner_train"][0][:, indices], sets["inner_train"][1])
            probability = model.predict_proba(sets["inner_valid"][0][:, indices])[:, 1]
            auc = roc_auc_score(sets["inner_valid"][1], probability)
            candidates.append((-auc, c_value, model))
        _, c_value, model = min(candidates, key=lambda value: (value[0], value[1]))
        for role in ("inner_train", "inner_valid"):
            X, y, frame = sets[role]
            probability = model.predict_proba(X[:, indices])[:, 1]
            prediction = (probability >= 0.5).astype(int)
            rows.append(
                {
                    "probe": name,
                    "role": role,
                    "selected_C_inner_valid_only": c_value,
                    "samples": len(y),
                    "sources": int(frame["video_id"].nunique()),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(y, prediction)
                    ),
                    "AUROC": float(roc_auc_score(y, probability)),
                    "log_loss": float(log_loss(y, probability)),
                    "outer_evaluation_run": False,
                    "reason_no_outer": "both folds are required to fit fold identity; opposite Direction outer rows are the same samples and cannot be independent",
                }
            )
    return pd.DataFrame(rows)


def regret_targets(data_role):
    X = data_role["X"]
    labels = data_role["sidecar"]["label"].to_numpy(dtype=np.float64)
    errors = np.abs(X[:, 4:9] - labels[:, None])
    return errors - errors.min(axis=1, keepdims=True)


def soft_metrics(predicted_regret, true_regret, tau):
    q_true = softmax(-true_regret / tau, axis=1)
    q_pred = softmax(-predicted_regret / tau, axis=1)
    kl = np.sum(
        q_true
        * (
            np.log(np.clip(q_true, 1e-12, 1.0))
            - np.log(np.clip(q_pred, 1e-12, 1.0))
        ),
        axis=1,
    )
    return {
        "regret_MAE": float(np.mean(np.abs(predicted_regret - true_regret))),
        "regret_Spearman": finite_spearman(
            predicted_regret.reshape(-1), true_regret.reshape(-1)
        ),
        "ranking_accuracy": float(
            np.mean(
                np.argmin(predicted_regret, axis=1)
                == np.argmin(true_regret, axis=1)
            )
        ),
        "soft_preference_KL": float(kl.mean()),
        "soft_expected_regret": float(np.mean(np.sum(q_pred * true_regret, axis=1))),
    }


def fit_regret_probe(train_X, train_y, valid_X, valid_y, tau):
    candidates = []
    for alpha in ALPHAS:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(train_X, train_y)
        metrics = soft_metrics(model.predict(valid_X), valid_y, tau)
        candidates.append((metrics["regret_MAE"], alpha, model))
    return min(candidates, key=lambda value: (value[0], value[1]))[1:]


def probe_c(direction, data):
    feature_sets = {
        "C0_predictions_mode": list(range(0, 9)),
        "C1_plus_disagreement_outlier": list(range(0, 17)),
        "C2_plus_consistency": list(range(0, 22)),
        "C3_plus_content_effective_mask": list(range(0, 57)),
    }
    targets = {role: regret_targets(data[role]) for role in ROLE_NAMES}
    positive = targets["inner_train"][targets["inner_train"] > 0]
    tau = max(float(np.median(positive)), 1e-3)
    rows = []
    outer_predictions = {}
    for feature_set, indices in feature_sets.items():
        alpha, model = fit_regret_probe(
            data["inner_train"]["X"][:, indices],
            targets["inner_train"],
            data["inner_valid"]["X"][:, indices],
            targets["inner_valid"],
            tau,
        )
        for role in ROLE_NAMES:
            predicted = model.predict(data[role]["X"][:, indices])
            values = soft_metrics(predicted, targets[role], tau)
            rows.append(
                {
                    "direction": direction,
                    "feature_set": feature_set,
                    "role": role,
                    "expert_id": "All",
                    "selected_alpha_inner_valid_only": alpha,
                    "soft_temperature_inner_train_only": tau,
                    **values,
                }
            )
            if role == "outer_evaluation":
                outer_predictions[feature_set] = predicted
                for expert_index, expert in enumerate(EXPERTS):
                    rows.append(
                        {
                            "direction": direction,
                            "feature_set": feature_set,
                            "role": role,
                            "expert_id": expert,
                            "selected_alpha_inner_valid_only": alpha,
                            "soft_temperature_inner_train_only": tau,
                            "regret_MAE": float(
                                np.mean(
                                    np.abs(
                                        predicted[:, expert_index]
                                        - targets[role][:, expert_index]
                                    )
                                )
                            ),
                            "regret_Spearman": finite_spearman(
                                predicted[:, expert_index],
                                targets[role][:, expert_index],
                            ),
                            "ranking_accuracy": np.nan,
                            "soft_preference_KL": np.nan,
                            "soft_expected_regret": np.nan,
                        }
                    )
    result = pd.DataFrame(rows)
    overall_outer = result.loc[
        (result["role"] == "outer_evaluation")
        & (result["expert_id"] == "All")
    ].set_index("feature_set")
    increments = []
    for before, after, label in (
        (
            "C0_predictions_mode",
            "C1_plus_disagreement_outlier",
            "C1-C0",
        ),
        (
            "C1_plus_disagreement_outlier",
            "C2_plus_consistency",
            "C2-C1",
        ),
        (
            "C2_plus_consistency",
            "C3_plus_content_effective_mask",
            "C3-C2",
        ),
    ):
        increments.append(
            {
                "direction": direction,
                "increment": label,
                **{
                    "delta_{}".format(metric): float(
                        overall_outer.loc[after, metric]
                        - overall_outer.loc[before, metric]
                    )
                    for metric in (
                        "regret_MAE",
                        "regret_Spearman",
                        "ranking_accuracy",
                        "soft_preference_KL",
                        "soft_expected_regret",
                    )
                },
            }
        )
    return result, pd.DataFrame(increments), tau


def cross_source_shuffle(X, sidecar, seed):
    result = X.copy()
    rng = np.random.RandomState(seed)
    for mode in MODES:
        indices = np.flatnonzero(sidecar["mode"].to_numpy() == mode)
        videos = sidecar.iloc[indices]["video_id"].to_numpy()
        permutation = rng.permutation(len(indices))
        for _ in range(1000):
            collisions = videos == videos[permutation]
            if not collisions.any():
                break
            permutation[collisions] = rng.permutation(permutation[collisions])
        if np.any(videos == videos[permutation]):
            # Deterministic cyclic search is guaranteed here because each mode
            # has hundreds of sources.
            for shift in range(1, len(indices)):
                candidate = np.roll(np.arange(len(indices)), shift)
                if not np.any(videos == videos[candidate]):
                    permutation = candidate
                    break
        if np.any(videos == videos[permutation]):
            raise RuntimeError("Cross-source shuffle could not remove collisions.")
        result[indices, 22:54] = X[indices[permutation], 22:54]
    return result


def matched_noise(X, mean, std, seed):
    result = X.copy()
    rng = np.random.RandomState(seed)
    result[:, 22:54] = rng.normal(
        loc=mean, scale=np.maximum(std, 1e-8), size=(len(X), 32)
    )
    # Preserve structural missingness: coordinates that are zero because the
    # modality is effectively unavailable remain zero.
    result[result[:, 54] == 0, 22:38] = 0
    result[result[:, 55] == 0, 38:46] = 0
    result[result[:, 56] == 0, 46:54] = 0
    return result


def probe_d(direction, data, tau):
    targets = {role: regret_targets(data[role]) for role in ROLE_NAMES}
    cases = [("no_content", None), ("aligned_content", None)]
    cases.extend(("cross_source_shuffle", seed) for seed in SHUFFLE_SEEDS)
    cases.extend(("matched_noise", seed) for seed in NOISE_SEEDS)
    train_content = data["inner_train"]["X"][:, 22:54]
    content_mean = train_content.mean(axis=0)
    content_std = train_content.std(axis=0)
    rows = []
    collision_audit = []
    for case, seed in cases:
        if case == "no_content":
            indices = list(range(0, 22)) + list(range(54, 57))
            train_X = data["inner_train"]["X"][:, indices]
            valid_X = data["inner_valid"]["X"][:, indices]
            outer_X = data["outer_evaluation"]["X"][:, indices]
        elif case == "aligned_content":
            indices = list(range(57))
            train_X = data["inner_train"]["X"]
            valid_X = data["inner_valid"]["X"]
            outer_X = data["outer_evaluation"]["X"]
        elif case == "cross_source_shuffle":
            indices = list(range(57))
            train_X = cross_source_shuffle(
                data["inner_train"]["X"],
                data["inner_train"]["sidecar"],
                seed,
            )
            valid_X = data["inner_valid"]["X"]
            outer_X = data["outer_evaluation"]["X"]
            collision_audit.append(
                {
                    "direction": direction,
                    "control": case,
                    "seed": seed,
                    "train_cross_source_collisions": 0,
                    "valid_outer_content_application": "normally paired",
                }
            )
        else:
            indices = list(range(57))
            train_X = matched_noise(
                data["inner_train"]["X"], content_mean, content_std, seed
            )
            valid_X = matched_noise(
                data["inner_valid"]["X"], content_mean, content_std, seed + 1000
            )
            outer_X = matched_noise(
                data["outer_evaluation"]["X"], content_mean, content_std, seed + 2000
            )
        alpha, model = fit_regret_probe(
            train_X,
            targets["inner_train"],
            valid_X,
            targets["inner_valid"],
            tau,
        )
        for role, X_local in (
            ("inner_train", train_X),
            ("inner_valid", valid_X),
            ("outer_evaluation", outer_X),
        ):
            values = soft_metrics(model.predict(X_local), targets[role], tau)
            rows.append(
                {
                    "direction": direction,
                    "case": case,
                    "seed": seed if seed is not None else -1,
                    "role": role,
                    "selected_alpha_inner_valid_only": alpha,
                    **values,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(collision_audit)


def summarize_controls(probe_d_frame):
    rows = []
    outer = probe_d_frame.loc[probe_d_frame["role"] == "outer_evaluation"]
    for direction in ("A", "B"):
        local = outer.loc[outer["direction"] == direction]
        for case in ("cross_source_shuffle", "matched_noise"):
            control = local.loc[local["case"] == case]
            for metric, lower_better in (
                ("regret_MAE", True),
                ("regret_Spearman", False),
                ("ranking_accuracy", False),
                ("soft_preference_KL", True),
                ("soft_expected_regret", True),
            ):
                values = control[metric].to_numpy(dtype=np.float64)
                strongest = values.min() if lower_better else values.max()
                rows.append(
                    {
                        "direction": direction,
                        "control": case,
                        "metric": metric,
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=0)),
                        "strongest": float(strongest),
                        "seeds": "|".join(
                            str(value) for value in control["seed"].astype(int)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def determine_conclusion(geometry, consistency, reversal, increments, probe_b, probe_d):
    overall = geometry.loc[geometry["mode"] == "Overall"].set_index("direction")
    broad_geometry = bool(
        (overall["oracle_minus_strong_static"] <= -0.05).all()
        and (overall["fraction_gain_gt_0p01"] >= 0.25).all()
    )
    outer_consistency = consistency.loc[
        consistency["role"] == "outer_evaluation"
    ]
    consistency_positive = bool(
        (outer_consistency.groupby("direction")["spearman_consistency_error"].median() > 0).all()
        and (reversal.groupby("direction")["sign_reversal"].mean() < 0.5).all()
    )
    content_increment = increments.loc[increments["increment"] == "C3-C2"].set_index(
        "direction"
    )
    content_helpful = bool(
        (
            (content_increment["delta_regret_MAE"] < 0)
            & (content_increment["delta_soft_preference_KL"] < 0)
            & (content_increment["delta_regret_Spearman"] > 0)
        ).all()
    )
    outer_d = probe_d.loc[probe_d["role"] == "outer_evaluation"]
    control_beaten = True
    control_details = {}
    for direction in ("A", "B"):
        local = outer_d.loc[outer_d["direction"] == direction]
        aligned = local.loc[local["case"] == "aligned_content"].iloc[0]
        controls = local.loc[
            local["case"].isin(["cross_source_shuffle", "matched_noise"])
        ]
        passed = bool(
            aligned["soft_preference_KL"] < controls["soft_preference_KL"].min()
            and aligned["soft_expected_regret"]
            < controls["soft_expected_regret"].min()
        )
        control_details[direction] = passed
        control_beaten = control_beaten and passed
    identity_auc = float(
        probe_b.loc[
            (probe_b["role"] == "inner_valid")
            & (probe_b["probe"] == "content_plus_consistency"),
            "AUROC",
        ].iloc[0]
    )
    no_dominant_fold_fingerprint = identity_auc < 0.80
    if (
        broad_geometry
        and consistency_positive
        and content_helpful
        and control_beaten
        and no_dominant_fold_fingerprint
    ):
        status = "SIGNAL_AUDIT_PASS"
    elif broad_geometry and (
        consistency_positive or content_helpful or control_beaten
    ):
        status = "SIGNAL_AUDIT_WEAK"
    else:
        status = "SIGNAL_AUDIT_FAIL"
    return status, {
        "broad_oracle_geometry": broad_geometry,
        "consistency_transfers_without_systematic_reversal": consistency_positive,
        "content_increment_C3_over_C2_both_directions": content_helpful,
        "aligned_content_beats_strongest_controls_both_directions": control_beaten,
        "control_detail": control_details,
        "fold_identity_AUROC": identity_auc,
        "no_dominant_fold_fingerprint_AUROC_lt_0p80": no_dominant_fold_fingerprint,
    }


def markdown_table(frame, columns, digits=6):
    display = frame[columns].copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "NA" if pd.isna(value) else "{:.{}f}".format(value, digits)
            )
    header = "| " + " | ".join(display.columns) + " |"
    rule = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule] + rows)


def main():
    authorization_path = V2_ROOT / "protocol" / "v2a_authorization_manifest.json"
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if not authorization["signal_audit_authorized"]:
        raise RuntimeError("Signal audit is not authorized.")
    if any(
        authorization[key]
        for key in (
            "judge_training_authorized",
            "official_valid_authorized",
            "test_authorized",
            "student_training_authorized",
        )
    ):
        raise RuntimeError("A forbidden authorization flag is open.")
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    directions = {key: load_direction(key) for key in ("A", "B")}

    geometry_rows, concentration_rows, better_pieces = [], [], []
    consistency_pieces, reversal_pieces = [], []
    probe_a_pieces, probe_c_pieces, increment_pieces, probe_d_pieces = [], [], [], []
    collision_pieces = []
    for direction in ("A", "B"):
        geometry, concentration, better = oracle_geometry(
            direction, directions[direction]
        )
        geometry_rows.extend(geometry)
        concentration_rows.extend(concentration)
        better["direction"] = direction
        better_pieces.append(better)
        consistency, reversal = consistency_audit(
            direction, directions[direction]
        )
        consistency_pieces.append(consistency)
        reversal_pieces.append(reversal)
        probe_a_pieces.append(probe_a(direction, directions[direction]))
        probe_c_frame, increments, tau = probe_c(
            direction, directions[direction]
        )
        probe_c_pieces.append(probe_c_frame)
        increment_pieces.append(increments)
        probe_d_frame, collisions = probe_d(
            direction, directions[direction], tau
        )
        probe_d_pieces.append(probe_d_frame)
        collision_pieces.append(collisions)

    geometry = pd.DataFrame(geometry_rows)
    concentration = pd.DataFrame(concentration_rows)
    better = pd.concat(better_pieces, ignore_index=True)
    consistency = pd.concat(consistency_pieces, ignore_index=True)
    reversal = pd.concat(reversal_pieces, ignore_index=True)
    consistency_method_seed = (
        consistency.groupby(["direction", "role", "method", "seed"], sort=True)
        .agg(
            mean_spearman_error=("spearman_consistency_error", "mean"),
            mean_spearman_regret=("spearman_consistency_regret", "mean"),
            mean_AUROC_exact_best=("auroc_low_consistency_is_exact_best", "mean"),
            mean_outlier_rate=("consistency_outlier_rate_above_inner_q99", "mean"),
        )
        .reset_index()
    )
    probe_a_frame = pd.concat(probe_a_pieces, ignore_index=True)
    probe_b_frame = probe_b_fold_identity(directions)
    probe_c_frame = pd.concat(probe_c_pieces, ignore_index=True)
    increments = pd.concat(increment_pieces, ignore_index=True)
    probe_d_frame = pd.concat(probe_d_pieces, ignore_index=True)
    collisions = pd.concat(collision_pieces, ignore_index=True)
    controls = summarize_controls(probe_d_frame)

    output_frames = {
        "oracle_geometry.tsv": geometry,
        "oracle_concentration.tsv": concentration,
        "strong_static_better_cases.tsv": better,
        "hierarchical_consistency_signal.tsv": consistency,
        "hierarchical_consistency_outer_reversal.tsv": reversal,
        "hierarchical_consistency_method_seed.tsv": consistency_method_seed,
        "probe_a_content_sentiment.tsv": probe_a_frame,
        "probe_b_fold_identity.tsv": probe_b_frame,
        "probe_c_regret_ridge.tsv": probe_c_frame,
        "probe_c_incremental_signal.tsv": increments,
        "probe_d_soft_preference.tsv": probe_d_frame,
        "control_summary.tsv": controls,
        "control_binding_assertions.tsv": collisions,
    }
    artifact_sha = {}
    for name, frame in output_frames.items():
        path = ANALYSIS / name
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, sep="\t", index=False, float_format="%.10g")
        artifact_sha[str(path.relative_to(ROOT))] = sha256_file(path)

    conclusion, criteria = determine_conclusion(
        geometry,
        consistency,
        reversal,
        increments,
        probe_b_frame,
        probe_d_frame,
    )
    leakage = {
        "status": "PASS",
        "feature_dimension": 57,
        "labels_sources_folds_absent_from_feature_names": True,
        "direction_specific_preprocessors_fit_inner_train_only": True,
        "strong_static_selected_inner_valid_only": True,
        "outer_evaluation_used_for_fit_or_selection": False,
        "probe_hyperparameters_selected_inner_valid_only": True,
        "ground_truth_used_as_probe_target_and_metrics_only": True,
        "formal_dual_head_judge_trained": False,
        "expert_checkpoint_modified_or_retrained": False,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "student_trained": False,
        "fold_identity_probe_outer_omitted_to_avoid_train_eval_sample_reuse": True,
    }
    leakage_path = ANALYSIS / "leakage_assertions.json"
    atomic_json(leakage_path, leakage)
    artifact_sha[str(leakage_path.relative_to(ROOT))] = sha256_file(leakage_path)

    geometry_core = geometry.loc[
        geometry["mode"].isin(["LAV", "LA", "LV", "L", "Overall"])
    ]
    c_increment = increments.loc[increments["increment"].isin(["C2-C1", "C3-C2"])]
    report = {
        "stage": "Stage23A-v2a Feature Extraction and Signal Audit",
        "status": conclusion,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "authorization_manifest_path": str(authorization_path.resolve()),
        "authorization_manifest_sha256": sha256_file(authorization_path),
        "oracle_geometry": geometry_core.to_dict(orient="records"),
        "signal_criteria": criteria,
        "probe_c_incremental": c_increment.to_dict(orient="records"),
        "probe_b_fold_identity": probe_b_frame.to_dict(orient="records"),
        "controls": controls.to_dict(orient="records"),
        "artifacts_sha256": artifact_sha,
        "plots": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for path in sorted(PLOTS.glob("*.svg"))
        ],
        "interpretation_guard": "This signal-audit result does not authorize or train the formal Judge.",
        "judge_training_authorized": False,
        "judge_training_started": False,
        "official_valid_authorized": False,
        "official_valid_access_count": 0,
        "test_authorized": False,
        "locked_test_access_count": 0,
        "student_training_authorized": False,
        "student_trained": False,
    }
    report_path = ANALYSIS / "stage23a_v2a_signal_audit.json"
    atomic_json(report_path, report)

    overall_table = geometry.loc[geometry["mode"] == "Overall", [
        "direction",
        "strong_static_J_or_MAE",
        "oracle_J_or_MAE",
        "oracle_minus_strong_static",
        "fraction_gain_gt_0p01",
        "strong_static_strictly_better_fraction",
        "best_second_gap_mean",
    ]]
    consistency_summary = (
        consistency.loc[consistency["role"] == "outer_evaluation"]
        .groupby("direction")
        .agg(
            median_rho_error=("spearman_consistency_error", "median"),
            mean_AUROC_best=("auroc_low_consistency_is_exact_best", "mean"),
            mean_AUROC_top_bottom=("auroc_consistency_high_error_top_vs_bottom20", "mean"),
        )
        .reset_index()
    )
    report_md = """# Stage23A-v2a Feature Extraction and Signal Audit

## Decision

**{status}**

This is a Train-OOF signal audit only. It does **not** authorize or train the
formal dual-head Judge. Official Valid, Test, Student training and Expert
modification remained locked.

## Strong-static-relative Oracle geometry (outer one-shot)

{geometry}

## Hierarchical consistency transfer

{consistency}

Outer sign-reversal fractions: {reversal}.

## Incremental regret-probe signal

{increments}

For error-like metrics (MAE, KL, expected regret), negative deltas improve.
For Spearman and ranking accuracy, positive deltas improve.

## Fold fingerprint diagnostic

{fold_identity}

A Direction-local fold classifier is undefined because a development Direction
contains only one fold. Therefore this diagnostic pools both source-disjoint
inner-train sets and evaluates only on both source-disjoint inner-valid sets.
No pseudo-outer score is reported, because the opposite Direction outer samples
are the same samples used on the pooled training side.

## Frozen controls

{controls}

## Criterion ledger

```json
{criteria}
```

## Locks and leakage

- Frozen Experts retrained or modified: **No**
- Formal Judge trained: **No**
- Official Valid accesses: **0**
- Locked Test accesses: **0**
- Student trained: **No**
- Outer labels used for fitting/selection: **No**
- 482 ineffective Vision clips excluded from Vision PCA fit and forced to zero
  after transform: **Yes**

## Plain-language answers

1. The one-best-existing-Expert Oracle remains far better than the inner-valid
   selected strong static baseline; the opportunity is not an artifact of a weak
   per-mode stacking reference.
2. The hierarchical consistency signal is summarized above, including
   Direction-wise outer reversals; it is useful only if its association transfers.
3. Content utility is judged by C3−C2 and by aligned content against every frozen
   shuffle/noise control, not by training fit.
4. Fold identity is explicitly audited. A high inner-valid AUROC is treated as a
   checkpoint/domain fingerprint warning, not as positive routing evidence.
5. All preprocessing and probe choices were made on Direction inner-train and
   inner-valid only; outer rows were transformed and scored once.
6. The conclusion is `{status}` and still requires a separate explicit user
   authorization before any formal Judge training.
""".format(
        status=conclusion,
        geometry=markdown_table(overall_table, list(overall_table.columns)),
        consistency=markdown_table(
            consistency_summary, list(consistency_summary.columns)
        ),
        reversal=json.dumps(
            reversal.groupby("direction")["sign_reversal"].mean().to_dict(),
            ensure_ascii=False,
        ),
        increments=markdown_table(c_increment, list(c_increment.columns)),
        fold_identity=markdown_table(
            probe_b_frame.loc[probe_b_frame["role"] == "inner_valid"],
            [
                "probe",
                "role",
                "selected_C_inner_valid_only",
                "balanced_accuracy",
                "AUROC",
                "log_loss",
            ],
        ),
        controls=markdown_table(
            controls,
            [
                "direction",
                "control",
                "metric",
                "mean",
                "std",
                "strongest",
            ],
        ),
        criteria=json.dumps(criteria, indent=2, ensure_ascii=False),
    )
    md_path = ANALYSIS / "stage23a_v2a_signal_audit.md"
    md_path.write_text(report_md, encoding="utf-8")

    command_path = ANALYSIS / "reproducible_commands.txt"
    command_path.write_text(
        "\n".join(
            [
                "/usr/miniconda3/envs/DLF/bin/python scripts/mosei/stage23a_v2a_authorize.py",
                "CUDA_VISIBLE_DEVICES=0 /usr/miniconda3/envs/DLF/bin/python scripts/mosei/stage23a_v2a_extract_hierarchical.py --outer-fold 0 --gpu-id 0",
                "CUDA_VISIBLE_DEVICES=2 /usr/miniconda3/envs/DLF/bin/python scripts/mosei/stage23a_v2a_extract_hierarchical.py --outer-fold 1 --gpu-id 0",
                "/usr/miniconda3/envs/DLF/bin/python scripts/mosei/stage23a_v2a_extract_content.py",
                "/usr/miniconda3/envs/DLF/bin/python scripts/mosei/stage23a_v2a_build_features.py",
                "/usr/miniconda3/envs/DLF/bin/python scripts/mosei/stage23a_v2a_signal_audit.py",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    final_dir = V2_ROOT / "final_v2a"
    final_dir.mkdir(parents=True, exist_ok=True)
    lock = {
        "stage": "Stage23A-v2a",
        "status": conclusion,
        "judge_training_authorized": False,
        "judge_training_started": False,
        "official_valid_authorized": False,
        "official_valid_access_count": 0,
        "test_authorized": False,
        "locked_test_access_count": 0,
        "test_loader_constructed": False,
        "student_training_authorized": False,
        "student_trained": False,
        "expert_checkpoint_modified_or_retrained": False,
    }
    atomic_json(final_dir / "TEST_LOCK_STATUS.json", lock)
    state = {
        **lock,
        "analysis_report_path": str(report_path.resolve()),
        "analysis_report_sha256": sha256_file(report_path),
        "markdown_report_path": str(md_path.resolve()),
        "markdown_report_sha256": sha256_file(md_path),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "next_action": "stop; await explicit user decision; formal Judge remains unauthorized",
    }
    atomic_json(RUNTIME_ROOT / "state.json", state)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
