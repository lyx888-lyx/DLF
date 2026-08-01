"""Engineering and leakage audit for V9.15 absolute-target distillation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trains.singleTask.model.AbsoluteTargetExpertStudentV915 import (  # noqa: E402
    STUDENT_VERSION,
)
from trains.singleTask.strict_oof_absolute_target_distillation_v915 import (  # noqa: E402
    TEACHER_VERSION,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=(
            "result/strict_oof_absolute_target_distillation_v915/"
            "mosi/seed_1111"
        ),
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(left: float, right: float, atol: float = 1e-6) -> bool:
    return abs(float(left) - float(right)) <= atol


def main():
    cli = parse_args()
    root = Path(cli.root)
    paths = {
        "teacher": root / "strict_oof_soft_teacher_v915.pth",
        "teacher_csv": root / "v915_strict_oof_teacher_diagnostic.csv",
        "history": root / "v915_student_training_history.csv",
        "candidates": root / "v915_validation_candidates.csv",
        "selection": root / "v915_validation_selection.json",
        "test_predictions": root / "v915_test_predictions.csv",
        "test_summary": root / "v915_test_summary.csv",
        "summary": root / "strict_oof_absolute_target_distillation_v915_summary.json",
    }
    for path in paths.values():
        require(path.is_file(), f"missing V9.15 output: {path}")

    teacher = torch.load(paths["teacher"], map_location="cpu")
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    history = pd.read_csv(paths["history"])
    candidates = pd.read_csv(paths["candidates"])
    predictions = pd.read_csv(paths["test_predictions"])
    test_summary = pd.read_csv(paths["test_summary"])

    require(summary["version"] == STUDENT_VERSION, "unexpected Student version")
    require(selection["version"] == STUDENT_VERSION, "selection version mismatch")
    require(summary["teacher_version"] == TEACHER_VERSION, "teacher version mismatch")
    require(teacher["version"] == TEACHER_VERSION, "stored teacher version mismatch")

    provenance = summary["provenance"]
    required_true = (
        "strict_oof_experts_used_for_train_targets",
        "each_train_target_excludes_its_sample",
        "train_labels_used_for_student_training",
        "absolute_teacher_predictions_used_for_distillation",
        "official_validation_used_for_early_stopping",
        "official_validation_used_for_beta_calibration",
        "official_validation_used_for_candidate_selection",
        "test_dataset_constructed_after_selection_artifact_written",
        "sample_id_alignment_checked",
        "student_deploys_without_expert_pool",
        "backbone_frozen",
    )
    for key in required_true:
        require(provenance.get(key) is True, f"provenance flag not true: {key}")
    require(
        provenance.get("full_train_expert_outputs_used_for_student_training") is False,
        "full-train expert outputs entered Student training",
    )
    require(
        provenance.get("oof_relative_corrections_used_for_unified_distillation") is False,
        "V9.15 unexpectedly distilled OOF-relative corrections",
    )
    require(
        provenance.get("official_test_used_for_training_or_selection") is False,
        "Test entered training or selection",
    )
    require(selection["selected_by_validation_only"] is True, "selection leakage flag")
    require(selection["test_dataset_constructed"] is False, "selection artifact saw Test")
    require(selection["test_evaluated"] is False, "selection artifact evaluated Test")

    teacher_provenance = teacher["provenance"]
    require(teacher_provenance["strict_oof_experts"] is True, "teacher is not strict OOF")
    require(
        teacher_provenance["each_training_target_excludes_its_sample"] is True,
        "teacher target includes its own sample",
    )
    require(
        teacher_provenance["validation_labels_used_to_construct_teacher"] is False,
        "Validation labels entered teacher construction",
    )
    require(
        teacher_provenance["test_labels_used_to_construct_teacher"] is False,
        "Test labels entered teacher construction",
    )

    beta_grid = [float(value) for value in summary["beta_grid"]]
    require(beta_grid and 0.0 in beta_grid, "beta grid must contain zero")
    require(len(candidates) == int(summary["candidate_count"]), "candidate count mismatch")
    require(len(history) > len(candidates), "training history is unexpectedly empty")
    for column in (
        "candidate_id",
        "distill_weight",
        "best_epoch",
        "selected_beta",
        "validation_raw_mae",
        "validation_mae",
        "validation_gain",
    ):
        require(column in candidates.columns, f"missing candidate column: {column}")
    require(
        all(any(close(value, grid) for grid in beta_grid) for value in candidates["selected_beta"]),
        "candidate beta outside preregistered grid",
    )

    selected = selection["selected"]
    require(
        selected["candidate_id"] == summary["selected_candidate_id"],
        "selected candidate mismatch",
    )
    require(close(selected["selected_beta"], summary["selected_beta"]), "selected beta mismatch")
    require(any(close(summary["selected_beta"], value) for value in beta_grid), "selected beta outside grid")

    checkpoint_path = Path(summary["checkpoint_paths"]["selected_student"])
    require(checkpoint_path.is_file(), f"missing selected checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    require(checkpoint["version"] == STUDENT_VERSION, "checkpoint version mismatch")
    require(close(checkpoint["selected_beta"], summary["selected_beta"]), "checkpoint beta mismatch")

    required_prediction_columns = (
        "sample_id",
        "label",
        "anchor",
        "raw_student_prediction",
        "raw_student_correction",
        "selected_beta",
        "selected_prediction",
        "anchor_abs_error",
        "selected_abs_error",
    )
    for column in required_prediction_columns:
        require(column in predictions.columns, f"missing prediction column: {column}")
    require(len(predictions) > 0, "empty Test predictions")
    require(
        predictions["sample_id"].astype(str).nunique() == len(predictions),
        "duplicate Test sample IDs",
    )
    calculated_raw = predictions["anchor"] + predictions["raw_student_correction"]
    require(
        bool((calculated_raw - predictions["raw_student_prediction"]).abs().max() < 1e-5),
        "raw prediction is not anchor plus correction",
    )
    require(
        bool((predictions["selected_beta"] - float(summary["selected_beta"])).abs().max() < 1e-7),
        "per-row selected beta mismatch",
    )
    calculated_selected = (
        predictions["anchor"]
        + float(summary["selected_beta"]) * predictions["raw_student_correction"]
    )
    require(
        bool((calculated_selected - predictions["selected_prediction"]).abs().max() < 1e-5),
        "selected prediction does not apply calibrated beta",
    )
    anchor_mae = float((predictions["anchor"] - predictions["label"]).abs().mean())
    selected_mae = float(
        (predictions["selected_prediction"] - predictions["label"]).abs().mean()
    )
    require(close(anchor_mae, summary["test_anchor_mae"], 2e-6), "Test Anchor MAE mismatch")
    require(close(selected_mae, summary["test_selected_mae"], 2e-6), "Test selected MAE mismatch")
    require(
        close(anchor_mae - selected_mae, summary["test_selected_gain"], 2e-6),
        "Test gain mismatch",
    )
    require(
        "validation_selected_deployable" in set(test_summary["model"]),
        "deployable Test summary row missing",
    )

    if summary["selected_candidate_id"] == "anchor_fallback":
        require(close(summary["selected_beta"], 0.0), "Anchor fallback beta must be zero")
        require(close(summary["test_selected_gain"], 0.0, 2e-6), "Anchor fallback changed Test")

    print("V9.15 ENGINEERING AUDIT PASSED")
    print("protocol: strict OOF absolute targets; Validation epoch/beta selection")
    print("relative OOF correction distillation: False")
    print("candidate count:", summary["candidate_count"])
    print("selected candidate:", summary["selected_candidate_id"])
    print("selected epoch/beta:", summary["selected_epoch"], summary["selected_beta"])
    print(
        "Validation anchor/raw/calibrated/gain:",
        f"{summary['validation_anchor_mae']:.6f}",
        f"{summary['validation_selected_raw_mae']:.6f}",
        f"{summary['validation_selected_mae']:.6f}",
        f"{summary['validation_selected_gain']:+.6f}",
    )
    print(
        "Test anchor/selected/gain:",
        f"{summary['test_anchor_mae']:.6f}",
        f"{summary['test_selected_mae']:.6f}",
        f"{summary['test_selected_gain']:+.6f}",
    )
    print(
        "Test harm_over_010_rate:",
        f"{summary['test_selected_harm_over_010_rate']:.4f}",
    )


if __name__ == "__main__":
    main()
