"""Finalize Stage23A from frozen MOSEI Expert OOF artifacts only.

This command never constructs dataset loaders.  It validates the completed
Train OOF ledgers, packages the complementarity/Judge evidence produced by
``stage23a_analyze.py``, adds robustness diagnostics, and closes the Test lock.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from stage23a_common import (
    EXPERTS,
    MODES,
    RESULT_ROOT,
    atomic_json,
    git_head,
    ordered_id_sha,
    overall_j,
    regression_metrics,
    sha256_file,
    source_video,
)


COMPONENTS = ("clean_seed1111", "clean_seed1114") + EXPERTS
ANALYSIS = RESULT_ROOT / "analysis"
FINAL = RESULT_ROOT / "final"
RUNTIME = Path(__file__).resolve().parents[2] / "runtime" / "stage23a"


def atomic_tsv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    display = frame.fillna("NA").replace("", "NA")
    display.to_csv(temporary, sep="\t", index=False, float_format="%.10g")
    temporary.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_inputs():
    protocol_path = RESULT_ROOT / "protocol" / "preregistered_protocol.json"
    split_path = RESULT_ROOT / "protocol" / "source_splits.csv"
    protocol = load_json(protocol_path)
    splits = pd.read_csv(
        split_path, dtype={"sample_id": str, "video_id": str}
    ).sort_values("train_index", kind="mergesort")
    if tuple(protocol["expert_ids"]) != EXPERTS:
        raise RuntimeError("Frozen Expert pool differs from source code.")
    if protocol["train_sample_count"] != 16326:
        raise RuntimeError("Frozen MOSEI Train count is not 16,326.")
    if protocol["source_split_sha256"] != sha256_file(split_path):
        raise RuntimeError("Source split SHA differs from preregistration.")
    if ordered_id_sha(splits.sample_id.tolist()) != protocol[
        "train_ordered_sample_id_sha256"
    ]:
        raise RuntimeError("Full Train ordered sample SHA mismatch.")

    frames = []
    fold_audits = []
    component_audits = []
    for fold in (0, 1):
        directory = RESULT_ROOT / "expert_oof" / "outer_fold{}".format(fold)
        prediction_path = directory / "oof_predictions.csv"
        fold_manifest_path = directory / "fold_manifest.json"
        fold_manifest = load_json(fold_manifest_path)
        prediction = pd.read_csv(
            prediction_path, dtype={"sample_id": str, "video_id": str}
        )
        expected = splits.loc[splits.outer_fold.astype(int) == fold].copy()
        expected_sources = set(expected.video_id)
        training_sources = set(
            splits.loc[splits.outer_fold.astype(int) != fold, "video_id"]
        )
        component_dirs = sorted(
            [value.name for value in (directory / "components").iterdir() if value.is_dir()]
            + [value.name for value in (directory / "experts").iterdir() if value.is_dir()]
        )
        if set(component_dirs) != set(COMPONENTS):
            raise RuntimeError("Fold {} component set differs.".format(fold))

        key = ["sample_id", "mode", "expert_id"]
        expected_rows = len(expected) * len(MODES) * len(EXPERTS)
        checks = {
            "fold": fold,
            "fold_manifest_sha256": sha256_file(fold_manifest_path),
            "rows": int(len(prediction)),
            "expected_rows": int(expected_rows),
            "samples": int(prediction.sample_id.nunique()),
            "expected_samples": int(len(expected)),
            "sources": int(prediction.video_id.nunique()),
            "duplicate_keys": int(prediction.duplicated(key).sum()),
            "missing_cells": int(expected_rows - len(prediction)),
            "source_overlap_count": int(len(expected_sources & training_sources)),
            "sample_set_matches": set(prediction.sample_id) == set(expected.sample_id),
            "train_index_set_matches": set(prediction.train_index)
            == set(expected.train_index),
            "video_mapping_matches": bool(
                (prediction.video_id == prediction.sample_id.map(source_video)).all()
            ),
            "modes_match": set(prediction["mode"]) == set(MODES),
            "experts_match": set(prediction.expert_id) == set(EXPERTS),
            "prediction_sha_matches": sha256_file(prediction_path)
            == fold_manifest["prediction_sha256"],
            "ordered_holdout_sample_sha_matches": ordered_id_sha(
                expected.sort_values("train_index").sample_id.tolist()
            )
            == fold_manifest["ordered_holdout_sample_id_sha256"],
            "protocol_sha_matches": sha256_file(protocol_path)
            == fold_manifest["protocol_sha256"],
            "split_sha_matches": sha256_file(split_path)
            == fold_manifest["source_split_sha256"],
            "component_set_matches": True,
            "locks_closed": (
                fold_manifest["official_valid_access_count"] == 0
                and fold_manifest["locked_test_access_count"] == 0
                and not fold_manifest["test_loader_constructed"]
                and not fold_manifest["student_trained"]
            ),
        }
        boolean_checks = [
            checks["rows"] == checks["expected_rows"],
            checks["samples"] == checks["expected_samples"],
            checks["duplicate_keys"] == 0,
            checks["missing_cells"] == 0,
            checks["source_overlap_count"] == 0,
            *[
                value
                for key_name, value in checks.items()
                if key_name.endswith(("matches", "closed"))
            ],
        ]
        if not all(boolean_checks):
            raise RuntimeError("Fold {} OOF validation failed: {}".format(fold, checks))

        for component in COMPONENTS:
            group = "components" if component.startswith("clean") else "experts"
            component_dir = directory / group / component
            run_manifest_path = component_dir / "run_manifest.json"
            run_manifest = load_json(run_manifest_path)
            epochs = pd.read_csv(component_dir / "epoch_metrics.csv")
            best = epochs.loc[epochs.inner_J.idxmin()]
            component_check = {
                "fold": fold,
                "component": component,
                "run_manifest_sha256": sha256_file(run_manifest_path),
                "role": "base_component" if component.startswith("clean") else "committee_expert",
                "best_epoch": int(run_manifest["best_epoch"]),
                "argmin_inner_valid_epoch": int(best.epoch),
                "best_inner_J": float(run_manifest["best_inner_train_only_J"]),
                "argmin_inner_J": float(best.inner_J),
                "checkpoint_sha_matches": sha256_file(run_manifest["checkpoint"])
                == run_manifest["checkpoint_sha256"],
                "selection_source": "train-side source-disjoint inner-valid only",
                "outer_holdout_used_for_selection": False,
                "locked_test_access_count": int(
                    run_manifest["locked_test_access_count"]
                ),
                "official_valid_access_count": int(
                    run_manifest["official_valid_access_count"]
                ),
            }
            if (
                component_check["best_epoch"]
                != component_check["argmin_inner_valid_epoch"]
                or abs(
                    component_check["best_inner_J"]
                    - component_check["argmin_inner_J"]
                )
                > 1e-8
                or not component_check["checkpoint_sha_matches"]
                or component_check["locked_test_access_count"] != 0
                or component_check["official_valid_access_count"] != 0
            ):
                raise RuntimeError(
                    "Component selection validation failed: {}".format(component_check)
                )
            component_audits.append(component_check)

        frames.append(prediction)
        fold_audits.append(checks)

    ledger = pd.concat(frames, ignore_index=True)
    expected_total = protocol["train_sample_count"] * len(MODES) * len(EXPERTS)
    combined = {
        "rows": int(len(ledger)),
        "expected_rows": int(expected_total),
        "samples": int(ledger.sample_id.nunique()),
        "expected_samples": int(protocol["train_sample_count"]),
        "sources": int(ledger.video_id.nunique()),
        "expected_sources": int(protocol["train_source_count"]),
        "duplicate_keys": int(
            ledger.duplicated(["sample_id", "mode", "expert_id"]).sum()
        ),
        "missing_cells": int(expected_total - len(ledger)),
        "fold_sample_overlap": int(
            len(set(frames[0].sample_id) & set(frames[1].sample_id))
        ),
        "sample_union_matches_train": set(ledger.sample_id) == set(splits.sample_id),
        "finite_predictions": bool(np.isfinite(ledger.prediction).all()),
        "finite_labels": bool(np.isfinite(ledger.label).all()),
        "label_consistency": bool(
            (ledger.groupby("sample_id").label.nunique() == 1).all()
        ),
    }
    if not (
        combined["rows"] == combined["expected_rows"]
        and combined["samples"] == combined["expected_samples"]
        and combined["sources"] == combined["expected_sources"]
        and combined["duplicate_keys"] == 0
        and combined["missing_cells"] == 0
        and combined["fold_sample_overlap"] == 0
        and combined["sample_union_matches_train"]
        and combined["finite_predictions"]
        and combined["finite_labels"]
        and combined["label_consistency"]
    ):
        raise RuntimeError("Combined OOF validation failed: {}".format(combined))
    return protocol, splits, ledger, {
        "folds": fold_audits,
        "components": component_audits,
        "combined": combined,
    }


def to_long_expert(ledger):
    metrics = pd.read_csv(ANALYSIS / "expert_and_ensemble_metrics.csv")
    correlations = pd.read_csv(ANALYSIS / "residual_correlation_matrix.csv")
    contributions = pd.read_csv(ANALYSIS / "expert_contributions.csv")
    leave_one_out = pd.read_csv(ANALYSIS / "leave_one_expert_out_oracle.csv")
    fixed_weights = pd.read_csv(ANALYSIS / "fixed_stacking_weights.csv")
    method_seed = load_json(ANALYSIS / "method_vs_seed_complementarity.json")
    rows = []

    def add(
        section,
        metric,
        value,
        fold="all",
        mode="Overall",
        method="",
        expert_id="",
        expert_left="",
        expert_right="",
        reference="",
        delta=np.nan,
        notes="",
    ):
        rows.append(
            {
                "section": section,
                "fold": fold,
                "mode": mode,
                "method": method,
                "expert_id": expert_id,
                "expert_left": expert_left,
                "expert_right": expert_right,
                "metric": metric,
                "value": value,
                "reference": reference,
                "delta": delta,
                "notes": notes,
            }
        )

    metric_names = ("J", "MAE", "Corr", "Acc7", "Acc5", "Acc2", "F1")
    for row in metrics.itertuples(index=False):
        for metric in metric_names:
            add(
                "combined_oof_metric",
                metric,
                getattr(row, metric),
                mode=row.mode,
                method=row.method,
                expert_id=row.method if row.method in EXPERTS else "",
            )
    for fold in (0, 1):
        local_fold = ledger.loc[ledger.outer_fold.astype(int) == fold]
        for expert in EXPERTS:
            local_expert = local_fold.loc[local_fold.expert_id == expert]
            for mode in MODES:
                local = local_expert.loc[local_expert["mode"] == mode]
                values = regression_metrics(local.prediction, local.label)
                for metric, value in values.items():
                    add(
                        "expert_metric_by_outer_fold",
                        metric,
                        value,
                        fold=fold,
                        mode=mode,
                        method=expert,
                        expert_id=expert,
                    )
    for row in correlations.itertuples(index=False):
        add(
            "residual_correlation",
            "residual_correlation",
            row.residual_correlation,
            mode=row.mode,
            expert_left=row.expert_left,
            expert_right=row.expert_right,
        )
    prediction_wide = ledger.pivot_table(
        index=["sample_id", "mode"],
        columns="expert_id",
        values="prediction",
        aggfunc="first",
    )
    for mode in list(MODES) + ["Overall"]:
        local = (
            prediction_wide
            if mode == "Overall"
            else prediction_wide.loc[
                prediction_wide.index.get_level_values("mode") == mode
            ]
        )
        for left_index, left in enumerate(EXPERTS):
            for right in EXPERTS[left_index + 1 :]:
                difference = np.abs(local[left] - local[right])
                add(
                    "prediction_disagreement",
                    "mean_absolute_disagreement",
                    float(difference.mean()),
                    mode=mode,
                    expert_left=left,
                    expert_right=right,
                )
                add(
                    "prediction_disagreement",
                    "p90_absolute_disagreement",
                    float(difference.quantile(0.9)),
                    mode=mode,
                    expert_left=left,
                    expert_right=right,
                )
    for row in contributions.itertuples(index=False):
        for metric in (
            "exact_best_fraction",
            "near_best_0p05_fraction",
            "own_overall_J",
            "leave_out_oracle_delta_J",
            "max_mode_leave_out_delta_J",
        ):
            add(
                "expert_contribution",
                metric,
                getattr(row, metric),
                expert_id=row.expert_id,
                notes="retained={}".format(bool(row.retained)),
            )
    for row in leave_one_out.itertuples(index=False):
        add(
            "leave_one_expert_out_oracle",
            "oracle_J_without",
            row.oracle_J_without,
            mode=row.mode,
            expert_id=row.removed_expert,
            reference="full_oracle",
            delta=row.delta_vs_full_oracle,
        )
    for row in fixed_weights.itertuples(index=False):
        add(
            "fixed_simplex_weight",
            "weight",
            row.weight,
            fold=row.valid_fold,
            mode=row.mode,
            method=row.scope,
            expert_id=row.expert_id,
        )
    for metric, value in method_seed.items():
        add(
            "method_vs_seed",
            metric,
            value if isinstance(value, (int, float)) else np.nan,
            notes="" if isinstance(value, (int, float)) else str(value),
        )
    best_single = (
        metrics.loc[
            metrics.method.isin(EXPERTS) & (metrics["mode"] == "Overall")
        ]
        .sort_values("J")
        .iloc[0]
    )
    add(
        "best_single",
        "J",
        best_single.J,
        method=best_single.method,
        expert_id=best_single.method,
    )
    return pd.DataFrame(rows)


def robustness_diagnostics(ledger):
    wide = ledger.pivot_table(
        index=["sample_id", "video_id", "mode", "label", "outer_fold"],
        columns="expert_id",
        values="prediction",
        aggfunc="first",
    ).reset_index()
    prediction = wide[list(EXPERTS)].to_numpy()
    error = np.abs(prediction - wide.label.to_numpy()[:, None])
    ordered_error = np.sort(error, axis=1)
    gap = ordered_error[:, 1] - ordered_error[:, 0]
    gap_summary = {}
    for mode in list(MODES) + ["Overall"]:
        mask = (
            np.ones(len(wide), dtype=bool)
            if mode == "Overall"
            else wide["mode"].to_numpy() == mode
        )
        gap_summary[mode] = {
            "mean_best_second_error_gap": float(gap[mask].mean()),
            "median_best_second_error_gap": float(np.median(gap[mask])),
            "p90_best_second_error_gap": float(np.quantile(gap[mask], 0.9)),
        }

    complementarity = pd.read_csv(ANALYSIS / "complementarity_predictions.csv")
    complementarity["fixed_absolute_error"] = np.abs(
        complementarity.per_mode_fixed_stacking - complementarity.label
    )
    complementarity["oracle_absolute_error"] = np.abs(
        complementarity.oracle_expert_selection - complementarity.label
    )
    trimmed = {}
    for fraction in (0.0, 0.01, 0.05):
        keep_parts = []
        for mode in MODES:
            local = complementarity.loc[
                complementarity["mode"] == mode
            ].copy()
            threshold = local.fixed_absolute_error.quantile(1.0 - fraction)
            keep_parts.append(local.loc[local.fixed_absolute_error <= threshold])
        kept = pd.concat(keep_parts, ignore_index=True)
        fixed_j = overall_j(kept, "per_mode_fixed_stacking")
        oracle_j = overall_j(kept, "oracle_expert_selection")
        trimmed[str(fraction)] = {
            "removed_top_fixed_error_fraction_per_mode": fraction,
            "retained_rows": int(len(kept)),
            "per_mode_fixed_J": float(fixed_j),
            "oracle_J": float(oracle_j),
            "oracle_delta_J": float(oracle_j - fixed_j),
        }

    weights = pd.read_csv(
        ANALYSIS / "dynamic_teacher_weights.csv",
        dtype={"sample_id": str},
    )
    key = ["valid_split", "sample_id", "mode"]
    per_sample = (
        weights.assign(
            distance=np.abs(weights.joint_weight - weights.fallback_weight)
        )
        .groupby(key, as_index=False)
        .agg(
            distance_to_fallback_l1=("distance", "sum"),
            triggered=("triggered", "first"),
        )
    )
    dynamic = {}
    dominance = {}
    for split in (0, 1):
        dynamic[str(split)] = {}
        dominance[str(split)] = {}
        for mode in MODES:
            local = weights.loc[
                (weights.valid_split.astype(int) == split)
                & (weights["mode"] == mode)
            ]
            local_sample = per_sample.loc[
                (per_sample.valid_split.astype(int) == split)
                & (per_sample["mode"] == mode)
            ]
            variance = local.groupby("expert_id").joint_weight.var(ddof=0)
            dynamic[str(split)][mode] = {
                "mean_expert_weight_variance": float(variance.mean()),
                "max_expert_weight_variance": float(variance.max()),
                "mean_distance_to_fallback_l1": float(
                    local_sample.distance_to_fallback_l1.mean()
                ),
                "p90_distance_to_fallback_l1": float(
                    local_sample.distance_to_fallback_l1.quantile(0.9)
                ),
                "trigger_rate": float(local_sample.triggered.astype(bool).mean()),
                "fallback_rate": float(
                    1.0 - local_sample.triggered.astype(bool).mean()
                ),
            }
            pivot = local.pivot_table(
                index=["sample_id", "mode"],
                columns="expert_id",
                values="joint_weight",
                aggfunc="first",
            )[list(EXPERTS)]
            winner = pivot.idxmax(axis=1)
            dominance[str(split)][mode] = {
                expert: float((winner == expert).mean()) for expert in EXPERTS
            }

    risk = weights.pivot_table(
        index=["valid_split", "sample_id", "mode"],
        columns="expert_id",
        values="predicted_risk",
        aggfunc="first",
    )[list(EXPERTS)].reset_index()
    actual = wide[
        ["sample_id", "mode", "label"] + list(EXPERTS)
    ].copy()
    top2 = risk.merge(actual, on=["sample_id", "mode"], validate="one_to_one")
    predicted_risk = top2[["{}_x".format(expert) for expert in EXPERTS]].to_numpy()
    actual_prediction = top2[["{}_y".format(expert) for expert in EXPERTS]].to_numpy()
    actual_error = np.abs(actual_prediction - top2.label.to_numpy()[:, None])
    chosen = predicted_risk.argmin(axis=1)
    actual_order = np.argsort(actual_error, axis=1)
    chosen_error = actual_error[np.arange(len(top2)), chosen]
    oracle_error = actual_error.min(axis=1)
    selected_regret = chosen_error - oracle_error
    top2_hit = np.any(actual_order[:, :2] == chosen[:, None], axis=1)
    top2_summary = {}
    for split in (0, 1):
        mask = top2.valid_split.astype(int).to_numpy() == split
        top2_summary[str(split)] = {
            "predicted_best_in_true_top2_rate": float(top2_hit[mask].mean()),
            "mean_selected_expert_regret": float(selected_regret[mask].mean()),
            "median_selected_expert_regret": float(
                np.median(selected_regret[mask])
            ),
            "p90_selected_expert_regret": float(
                np.quantile(selected_regret[mask], 0.9)
            ),
        }
    return {
        "best_second_error_gap": gap_summary,
        "trimmed_oracle": trimmed,
        "judge_top2_regret": top2_summary,
        "dynamic_weight_diagnostics": dynamic,
        "expert_dominance_fraction": dominance,
    }


def judge_tsv(additional, gate):
    diagnostics = pd.read_csv(ANALYSIS / "judge_diagnostics.csv")
    rows = []
    for row in diagnostics.itertuples(index=False):
        for metric in diagnostics.columns:
            if metric in ("judge", "valid_split"):
                continue
            rows.append(
                {
                    "section": "judge_oof_metric",
                    "split": row.valid_split,
                    "judge": row.judge,
                    "metric": metric,
                    "value": getattr(row, metric),
                    "threshold": "",
                    "passed": "",
                }
            )
    thresholds = {
        "mean_delta_J_vs_per_mode_fixed": "<= -0.003",
        "worst_split_delta_J_vs_per_mode_fixed": "<= +0.001",
        "mean_delta_J_vs_shuffled": "<= -0.002",
        "improved_missing_modes": ">= 2",
        "classification_systematic_degradation": "False",
        "risk_buckets_monotonic": "True",
        "dynamic_weights_noncollapsed": "True",
        "trigger_gain_above_fallback": "True",
        "split_directions_consistent": "True",
    }
    for metric, threshold in thresholds.items():
        rows.append(
            {
                "section": "frozen_judge_gate",
                "split": "mean_or_worst",
                "judge": "J2_predictions_disagreement",
                "metric": metric,
                "value": gate[metric],
                "threshold": threshold,
                "passed": gate["passed"],
            }
        )
    for split, values in additional["judge_top2_regret"].items():
        for metric, value in values.items():
            rows.append(
                {
                    "section": "top2_regret_diagnostic",
                    "split": split,
                    "judge": "J2_predictions_disagreement",
                    "metric": metric,
                    "value": value,
                    "threshold": "diagnostic_only",
                    "passed": "",
                }
            )
    return pd.DataFrame(rows)


def teacher_tsv(additional, gate):
    metrics = pd.read_csv(ANALYSIS / "teacher_metrics_by_split.csv")
    rows = []
    for row in metrics.itertuples(index=False):
        for metric in ("J", "MAE", "Corr", "Acc7", "Acc5", "Acc2", "F1"):
            rows.append(
                {
                    "section": "teacher_oof_metric",
                    "split": row.split,
                    "mode": row.mode,
                    "method": row.method,
                    "expert_id": "",
                    "metric": metric,
                    "value": getattr(row, metric),
                    "reference": "Per-mode fixed stacking",
                    "notes": "",
                }
            )
    for split, by_mode in additional["dynamic_weight_diagnostics"].items():
        for mode, values in by_mode.items():
            for metric, value in values.items():
                rows.append(
                    {
                        "section": "dynamic_weight_diagnostic",
                        "split": split,
                        "mode": mode,
                        "method": "Joint-risk personalized Teacher",
                        "expert_id": "",
                        "metric": metric,
                        "value": value,
                        "reference": "Per-mode fixed stacking",
                        "notes": "diagnostic_only",
                    }
                )
    for split, by_mode in additional["expert_dominance_fraction"].items():
        for mode, values in by_mode.items():
            for expert, value in values.items():
                rows.append(
                    {
                        "section": "expert_dominance",
                        "split": split,
                        "mode": mode,
                        "method": "Joint-risk personalized Teacher",
                        "expert_id": expert,
                        "metric": "dominance_fraction",
                        "value": value,
                        "reference": "",
                        "notes": "diagnostic_only",
                    }
                )
    for metric, value in gate.items():
        if isinstance(value, dict):
            continue
        rows.append(
            {
                "section": "frozen_teacher_gate",
                "split": "mean_or_worst",
                "mode": "Overall",
                "method": "Joint-risk personalized Teacher",
                "expert_id": "",
                "metric": metric,
                "value": value,
                "reference": "Per-mode fixed stacking",
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def write_report(status, protocol, validation, complementarity, gate, method_seed, additional):
    combined = validation["combined"]
    top2 = additional["judge_top2_regret"]
    lines = [
        "# Stage 23A-MOSEI — Expert Complementarity and Personalized Teacher Audit",
        "",
        "Final status: `{}`".format(status),
        "",
        "## Frozen committee",
        "",
        "- Seven trained components per fold: `{}`.".format("`, `".join(COMPONENTS)),
        "- Committee Experts: `{}`.".format("`, `".join(EXPERTS)),
        "- `clean_seed1111/1114` are base components only and never enter the committee.",
        "- `uniform_kd_seed1114` remains excluded exactly as preregistered; no Expert was added or retrained.",
        "",
        "## OOF integrity",
        "",
        "- Train samples: {}/{}; sources: {}/{}.".format(
            combined["samples"],
            combined["expected_samples"],
            combined["sources"],
            combined["expected_sources"],
        ),
        "- OOF rows: {}/{} = 16,326 × 4 modes × 5 Experts.".format(
            combined["rows"], combined["expected_rows"]
        ),
        "- duplicate keys: {}; missing cells: {}; fold sample overlap: 0; source overlap: 0.".format(
            combined["duplicate_keys"], combined["missing_cells"]
        ),
        "- Fold/run-manifest SHA values are recorded; all referenced prediction/sample/protocol/split/checkpoint SHA checks passed.",
        "- Every best epoch equals the training-side source-disjoint inner-valid argmin; outer OOF was not used for checkpoint selection.",
        "",
        "## Expert complementarity",
        "",
        "- Per-mode fixed simplex stacking J: {:.6f}.".format(
            complementarity["per_mode_fixed_stacking_J"]
        ),
        "- Oracle single-existing-Expert selection J: {:.6f}.".format(
            complementarity["oracle_expert_selection_J"]
        ),
        "- Oracle ΔJ: {:.6f}; frozen threshold ≤ -0.005: **passed**.".format(
            complementarity["oracle_delta_J"]
        ),
        "- Missing-mode Oracle ΔMAE: `{}`; all three modes pass the dynamic-space threshold.".format(
            json.dumps(
                complementarity["missing_mode_oracle_delta_MAE"],
                sort_keys=True,
            )
        ),
        "- Larger complementarity source: `{}`.".format(
            method_seed["larger_complementarity_source"]
        ),
        "",
        "## Judge and Personalized Teacher",
        "",
        "- Joint-risk vs per-mode fixed mean ΔJ: {:.8f} (required ≤ -0.003).".format(
            gate["mean_delta_J_vs_per_mode_fixed"]
        ),
        "- Worst split ΔJ: {:.8f} (required ≤ +0.001).".format(
            gate["worst_split_delta_J_vs_per_mode_fixed"]
        ),
        "- Joint-risk vs shuffled mean ΔJ: {:.8f} (required ≤ -0.002).".format(
            gate["mean_delta_J_vs_shuffled"]
        ),
        "- Improved missing modes: {}/3; split directions consistent: {}.".format(
            gate["improved_missing_modes"],
            gate["split_directions_consistent"],
        ),
        "- Dynamic weights noncollapsed: {}; systematic classification degradation: {}.".format(
            gate["dynamic_weights_noncollapsed"],
            gate["classification_systematic_degradation"],
        ),
        "- J2 selected Expert is in the true top-2 on {:.2%}/{:.2%} of the two Judge folds; mean selected regrets are {:.6f}/{:.6f}.".format(
            top2["0"]["predicted_best_in_true_top2_rate"],
            top2["1"]["predicted_best_in_true_top2_rate"],
            top2["0"]["mean_selected_expert_regret"],
            top2["1"]["mean_selected_expert_regret"],
        ),
        "- Result: the Oracle space is real, but the frozen Judge does not reliably realize it. Student training is not recommended.",
        "",
        "## Additional diagnostics",
        "",
        "- Best/second-best error gaps, top-1%/5%-trimmed Oracle, top-2 regret, dynamic-weight variance, fallback distance/rate, and Expert dominance are recorded in `stage23a_mosei_audit.json` and the TSV files.",
        "- Removing high fixed-error tails is diagnostic only and does not change any gate.",
        "",
        "## Data locks",
        "",
        "- Official Valid access count: **0** (Judge gate failed, so no one-shot confirmation was permitted).",
        "- Locked Test access count: **0**.",
        "- Test loader constructed: **No**.",
        "- Student trained: **No**.",
        "",
    ]
    (ANALYSIS / "stage23a_mosei_audit.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    protocol, splits, ledger, validation = validate_inputs()
    del splits
    complementarity = load_json(ANALYSIS / "complementarity_gate.json")
    gate = load_json(ANALYSIS / "judge_teacher_gate.json")
    method_seed = load_json(ANALYSIS / "method_vs_seed_complementarity.json")
    if not complementarity["gate_passed"]:
        status = "STAGE23A_NO_ACTIONABLE_EXPERT_COMPLEMENTARITY"
        if (ANALYSIS / "judge_teacher_gate.json").exists():
            raise RuntimeError("Judge artifacts exist despite failed Expert gate.")
    else:
        status = (
            "STAGE23A_TRAIN_ONLY_PASSED"
            if gate["passed"]
            else "STAGE23A_JUDGE_NOT_ACTIONABLE"
        )
    if gate["passed"]:
        raise RuntimeError(
            "Train-only passed: run a separately audited one-shot Official Valid command."
        )

    additional = robustness_diagnostics(ledger)
    expert_output = to_long_expert(ledger)
    judge_output = judge_tsv(additional, gate)
    teacher_output = teacher_tsv(additional, gate)
    atomic_tsv(expert_output, ANALYSIS / "expert_complementarity.tsv")
    atomic_tsv(judge_output, ANALYSIS / "judge_metrics.tsv")
    atomic_tsv(
        teacher_output, ANALYSIS / "personalized_teacher_metrics.tsv"
    )
    write_report(
        status,
        protocol,
        validation,
        complementarity,
        gate,
        method_seed,
        additional,
    )

    lock = {
        "stage": "Stage23A-MOSEI",
        "status": "LOCKED",
        "final_status": status,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "test_loader_constructed": False,
        "test_samples_read": 0,
        "test_predictions_read": 0,
        "test_labels_read": 0,
        "test_metrics_read": 0,
        "student_trained": False,
        "reason": "Judge/Personalized Teacher frozen gate failed before Official Valid.",
    }
    atomic_json(FINAL / "TEST_LOCK_STATUS.json", lock)
    state = {
        "stage": "Stage23A-MOSEI",
        "phase": "FINALIZED",
        "status": status,
        "expert_oof_retrained": False,
        "frozen_expert_pool_modified": False,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "test_loader_constructed": False,
        "student_trained": False,
        "analysis_code_commit": git_head(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(RUNTIME / "state.json", state)

    artifact_paths = [
        ANALYSIS / "expert_complementarity.tsv",
        ANALYSIS / "judge_metrics.tsv",
        ANALYSIS / "personalized_teacher_metrics.tsv",
        ANALYSIS / "stage23a_mosei_audit.md",
        FINAL / "TEST_LOCK_STATUS.json",
        RUNTIME / "state.json",
    ]
    audit = {
        "stage": "Stage23A-MOSEI",
        "status": status,
        "frozen_components": list(COMPONENTS),
        "committee_experts": list(EXPERTS),
        "clean_components_enter_committee": False,
        "uniform_kd_seed1114_added": False,
        "expert_oof_retrained": False,
        "validation": validation,
        "expert_complementarity_gate": complementarity,
        "judge_personalized_teacher_gate": gate,
        "method_vs_seed": method_seed,
        "additional_diagnostics": additional,
        "official_valid": {
            "access_count": 0,
            "executed": False,
            "reason": "Train-only Judge/Teacher gate failed.",
        },
        "locked_test_access_count": 0,
        "test_loader_constructed": False,
        "student_trained": False,
        "input_sha256": {
            "protocol": sha256_file(
                RESULT_ROOT / "protocol" / "preregistered_protocol.json"
            ),
            "source_splits": sha256_file(
                RESULT_ROOT / "protocol" / "source_splits.csv"
            ),
            "outer_fold0_predictions": sha256_file(
                RESULT_ROOT
                / "expert_oof"
                / "outer_fold0"
                / "oof_predictions.csv"
            ),
            "outer_fold1_predictions": sha256_file(
                RESULT_ROOT
                / "expert_oof"
                / "outer_fold1"
                / "oof_predictions.csv"
            ),
        },
        "artifact_sha256": {
            str(path.relative_to(Path(__file__).resolve().parents[2])): sha256_file(path)
            for path in artifact_paths
        },
        "analysis_code_commit": git_head(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(ANALYSIS / "stage23a_mosei_audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
