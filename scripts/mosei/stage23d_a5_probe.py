#!/usr/bin/env python3
"""Conditional R3 static-plus-A5 pilot selection and one-shot evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from stage23d_phase1 import (
    atomic_gzip_csv,
    choose_classifier,
    choose_regression,
    fixed_precision_recall,
    predict_model,
    predict_probability,
    safe_auc,
    safe_auprc,
    safe_pearson,
    safe_spearman,
)
from stage23d_self_risk_common import (
    EXPERTS,
    MODES,
    OUT,
    atomic_json,
    atomic_tsv,
    sha256_file,
    utc_now,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("select", "evaluate"), required=True)
    parser.add_argument("--checkpoint-fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--expert-id", choices=EXPERTS, required=True)
    return parser.parse_args()


def directories(fold, expert):
    static = OUT / "phase1" / f"checkpoint_fold{fold}" / expert
    a5 = OUT / "a5" / "features" / f"checkpoint_fold{fold}" / expert
    output = OUT / "a5" / "probes" / f"checkpoint_fold{fold}" / expert
    return static, a5, output


def load_features(fold, expert, mode):
    static, a5, _ = directories(fold, expert)
    selection = json.loads(
        (static / "frozen_inner_valid_selection.json").read_text()
    )
    payload = selection["modes"][mode]
    static_path = Path(payload["frozen_static_feature_path"])
    if sha256_file(static_path) != payload["frozen_static_feature_sha256"]:
        raise RuntimeError("Frozen static feature SHA mismatch")
    static_frame = pd.read_csv(
        static_path, dtype={"sample_id": str, "video_id": str}
    )
    a5_manifest = json.loads((a5 / "a5_extraction_manifest.json").read_text())
    a5_path = Path(a5_manifest["feature_path"])
    if sha256_file(a5_path) != a5_manifest["feature_sha256"]:
        raise RuntimeError("A5 feature SHA mismatch")
    a5_frame = pd.read_csv(
        a5_path, dtype={"sample_id": str, "video_id": str}
    )
    a5_frame = a5_frame.loc[a5_frame["mode"] == mode]
    joined = static_frame.merge(
        a5_frame[
            [
                "sample_id",
                *[
                    column
                    for column in a5_frame.columns
                    if column.startswith("a5_")
                ],
            ]
        ],
        on="sample_id",
        validate="one_to_one",
    )
    return joined, static, a5, payload


def targets(frame, static, fold, expert, mode):
    development = pd.read_csv(
        OUT
        / "features"
        / f"checkpoint_fold{fold}"
        / expert
        / f"development_risk_labels_{mode}.csv.gz",
        dtype={"sample_id": str},
    )[["sample_id", "label"]]
    local = frame.merge(development, on="sample_id", how="left", validate="one_to_one")
    threshold = pd.read_csv(
        static / mode / "risk_label_thresholds.tsv", sep="\t"
    ).iloc[0]
    local["abs_error"] = np.abs(local["label"] - local["prediction"])
    local["log_abs_error"] = np.log(local["abs_error"] + 1e-6)
    local["bad20"] = (
        local["abs_error"] >= float(threshold["error_top20_threshold"])
    ).astype(int)
    local["confident_wrong"] = (
        (local["raw_uncertainty"] <= float(threshold["uncertainty_bottom30_threshold"]))
        & (local["bad20"] == 1)
    ).astype(int)
    return local, threshold


def select(cli):
    gate = json.loads((OUT / "analysis" / "phase1_gate.json").read_text())
    if gate["decision"] not in ("WEAK", "PASS"):
        raise RuntimeError("R3 is not authorized after Phase-1 FAIL")
    _, _, output = directories(cli.checkpoint_fold, cli.expert_id)
    selection_path = output / "frozen_r3_inner_valid_selection.json"
    if selection_path.exists():
        print(selection_path.read_text())
        return
    mode_payloads = {}
    for mode in MODES:
        frame, static, _, static_payload = load_features(
            cli.checkpoint_fold, cli.expert_id, mode
        )
        target, _ = targets(
            frame, static, cli.checkpoint_fold, cli.expert_id, mode
        )
        train = frame["self_risk_role"].eq("inner_train").to_numpy()
        valid = frame["self_risk_role"].eq("inner_valid").to_numpy()
        outer = frame["self_risk_role"].eq("outer").to_numpy()
        identifiers = {
            "sample_id",
            "video_id",
            "mode",
            "self_risk_role",
            "row_binding_sha256",
            "checkpoint_fold",
        }
        feature_columns = [
            column
            for column in frame.select_dtypes(include=[np.number]).columns
            if column not in identifiers
        ]
        values = frame[feature_columns].to_numpy(dtype=float)
        regression, regression_table, regression_selected = choose_regression(
            values,
            target["log_abs_error"].fillna(0.0).to_numpy(),
            train,
            valid,
            [
                ("ridge", 0.1),
                ("ridge", 1.0),
                ("ridge", 10.0),
                ("hgb", "fixed"),
            ],
        )
        bad, bad_table, bad_selected = choose_classifier(
            values,
            target["bad20"].to_numpy(),
            train,
            valid,
        )
        cw, cw_table, cw_selected = choose_classifier(
            values,
            target["confident_wrong"].to_numpy(),
            train,
            valid,
        )
        prediction = frame[
            ["sample_id", "video_id", "mode", "row_binding_sha256"]
        ].loc[outer].copy()
        prediction["R3_expected_abs_error"] = np.maximum(
            np.exp(predict_model(regression, values[outer])) - 1e-6, 0.0
        )
        prediction["R3_P_bad20"] = predict_probability(bad, values[outer])
        prediction["R3_P_confident_wrong"] = predict_probability(
            cw, values[outer]
        )
        mode_dir = output / mode
        prediction_path = mode_dir / "outer_r3_predictions_label_free.csv.gz"
        atomic_gzip_csv(prediction, prediction_path)
        atomic_tsv(regression_table, mode_dir / "r3_regression_candidates.tsv")
        atomic_tsv(bad_table, mode_dir / "r3_bad20_candidates.tsv")
        atomic_tsv(cw_table, mode_dir / "r3_confident_wrong_candidates.tsv")
        bundle_path = mode_dir / "frozen_r3_bundle.joblib"
        mode_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "feature_columns": feature_columns,
                "regression": regression,
                "bad20": bad,
                "confident_wrong": cw,
            },
            bundle_path,
        )
        mode_payloads[mode] = {
            "feature_columns": feature_columns,
            "selected_regression": regression_selected,
            "selected_bad20": bad_selected,
            "selected_confident_wrong": cw_selected,
            "outer_prediction_path": str(prediction_path.resolve()),
            "outer_prediction_sha256": sha256_file(prediction_path),
            "static_selection_sha256": sha256_file(
                static / "frozen_inner_valid_selection.json"
            ),
            "static_feature_sha256": static_payload[
                "frozen_static_feature_sha256"
            ],
            "bundle_path": str(bundle_path.resolve()),
            "bundle_sha256": sha256_file(bundle_path),
        }
    selection = {
        "stage": "Stage23D-A frozen R3 inner-valid selection",
        "status": "FROZEN_OUTER_LABELS_NOT_OPENED",
        "checkpoint_fold": cli.checkpoint_fold,
        "expert_id": cli.expert_id,
        "phase1_decision": gate["decision"],
        "modes": mode_payloads,
        "outer_evaluation_access_count": 0,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "frozen_at": utc_now(),
    }
    atomic_json(selection_path, selection)
    print(json.dumps(selection, indent=2, sort_keys=True))


def evaluate(cli):
    static, _, output = directories(cli.checkpoint_fold, cli.expert_id)
    selection_path = output / "frozen_r3_inner_valid_selection.json"
    result_path = output / "r3_outer_evaluation_manifest.json"
    lock_path = output / "r3_outer_access_lock.json"
    if result_path.exists():
        print(result_path.read_text())
        return
    if lock_path.exists():
        raise RuntimeError("R3 outer access already claimed")
    selection = json.loads(selection_path.read_text())
    atomic_json(
        lock_path,
        {
            "status": "CLAIMED_BEFORE_LABEL_FILE_OPEN",
            "outer_evaluation_access_count": 1,
            "claimed_at": utc_now(),
        },
    )
    rows = []
    for mode in MODES:
        prediction_path = Path(
            selection["modes"][mode]["outer_prediction_path"]
        )
        if sha256_file(prediction_path) != selection["modes"][mode][
            "outer_prediction_sha256"
        ]:
            raise RuntimeError("R3 outer prediction SHA mismatch")
        r3 = pd.read_csv(prediction_path, dtype={"sample_id": str})
        r2 = pd.read_csv(
            static / mode / "outer_risk_predictions_label_free.csv.gz",
            dtype={"sample_id": str},
        )
        labels = pd.read_csv(
            OUT
            / "features"
            / f"checkpoint_fold{cli.checkpoint_fold}"
            / cli.expert_id
            / "sealed_outer_labels"
            / f"outer_labels_{mode}.csv.gz",
            dtype={"sample_id": str},
        )
        static_frame = pd.read_csv(
            OUT
            / "features"
            / f"checkpoint_fold{cli.checkpoint_fold}"
            / cli.expert_id
            / f"static_internal_features_{mode}.csv.gz",
            usecols=["sample_id", "prediction"],
            dtype={"sample_id": str},
        )
        joined = (
            r3.merge(r2, on=["sample_id", "video_id", "mode", "row_binding_sha256"])
            .merge(labels[["sample_id", "label"]], on="sample_id", validate="one_to_one")
            .merge(static_frame, on="sample_id", validate="one_to_one")
        )
        actual = np.abs(joined["label"] - joined["prediction"]).to_numpy()
        threshold = pd.read_csv(
            static / mode / "risk_label_thresholds.tsv", sep="\t"
        ).iloc[0]
        bad20 = (
            actual >= float(threshold["error_top20_threshold"])
        ).astype(int)
        confident_wrong = (
            (
                joined["raw_uncertainty"]
                <= float(threshold["uncertainty_bottom30_threshold"])
            )
            & (bad20 == 1)
        ).to_numpy(dtype=int)
        for model in ("R2", "R3"):
            risk = joined[f"{model}_expected_abs_error"].to_numpy()
            bad_probability = joined[f"{model}_P_bad20"].to_numpy()
            cw_probability = joined[
                f"{model}_P_confident_wrong"
            ].to_numpy()
            rows.append(
                {
                    "checkpoint_fold": cli.checkpoint_fold,
                    "expert_id": cli.expert_id,
                    "mode": mode,
                    "model": model,
                    "Error_Spearman": safe_spearman(risk, actual),
                    "Error_Pearson": safe_pearson(risk, actual),
                    "Risk_MAE": mean_absolute_error(actual, risk),
                    "Risk_RMSE": mean_squared_error(
                        actual, risk, squared=False
                    ),
                    "bad20_AUROC": safe_auc(bad20, bad_probability),
                    "bad20_AUPRC": safe_auprc(bad20, bad_probability),
                    "confident_wrong_AUROC": safe_auc(
                        confident_wrong, cw_probability
                    ),
                    "confident_wrong_AUPRC": safe_auprc(
                        confident_wrong, cw_probability
                    ),
                    "confident_wrong_recall_at_precision_0p8": fixed_precision_recall(
                        confident_wrong, cw_probability
                    ),
                }
            )
    metrics = pd.DataFrame(rows)
    atomic_tsv(metrics, output / "r3_outer_metrics.tsv")
    result = {
        "stage": "Stage23D-A one-shot R3 outer evaluation",
        "status": "COMPLETED",
        "checkpoint_fold": cli.checkpoint_fold,
        "expert_id": cli.expert_id,
        "metrics_path": str((output / "r3_outer_metrics.tsv").resolve()),
        "metrics_sha256": sha256_file(output / "r3_outer_metrics.tsv"),
        "outer_access_lock_sha256": sha256_file(lock_path),
        "outer_evaluation_access_count": 1,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "completed_at": utc_now(),
    }
    atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    cli = parse_args()
    if cli.phase == "select":
        select(cli)
    else:
        evaluate(cli)


if __name__ == "__main__":
    main()
