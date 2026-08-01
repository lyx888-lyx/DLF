"""Engineering and leakage audit for V9.16 small-scale distillation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trains.singleTask.model.SmallScaleExpertStudentV916 import (  # noqa: E402
    STUDENT_VERSION,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=(
            "result/strict_oof_small_scale_distillation_v916/"
            "mosi/seed_1111"
        ),
    )
    return parser.parse_args()


def require(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def close(left: float, right: float, atol: float = 1e-6):
    return abs(float(left) - float(right)) <= atol


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    cli = parse_args()
    root = Path(cli.root)
    paths = {
        "teacher": root / "strict_oof_soft_teacher_v916.pth",
        "history": root / "v916_student_training_history.csv",
        "candidates": root / "v916_validation_candidates.csv",
        "selection": root / "v916_validation_selection.json",
        "predictions": root / "v916_test_predictions.csv",
        "test_summary": root / "v916_test_summary.csv",
        "summary": root / "strict_oof_small_scale_distillation_v916_summary.json",
    }
    for path in paths.values():
        require(path.is_file(), f"missing V9.16 output: {path}")

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    candidates = pd.read_csv(paths["candidates"])
    history = pd.read_csv(paths["history"])
    predictions = pd.read_csv(paths["predictions"])
    test_summary = pd.read_csv(paths["test_summary"])
    teacher = torch.load(paths["teacher"], map_location="cpu")

    require(summary["version"] == STUDENT_VERSION, "summary version mismatch")
    require(selection["version"] == STUDENT_VERSION, "selection version mismatch")
    require(summary["candidate_count"] == 6, "expected six pre-registered candidates")
    require(len(candidates) == 6, "candidate CSV row count mismatch")
    require(
        sorted(round(float(value), 3) for value in summary["residual_max_grid"])
        == [0.015, 0.03, 0.045],
        "unexpected residual-max grid",
    )
    require(
        sorted(float(value) for value in summary["distill_weight_grid"])
        == [0.0, 1.0],
        "unexpected distillation grid",
    )
    require(
        set(round(float(value), 3) for value in candidates["residual_max"])
        == {0.015, 0.03, 0.045},
        "candidate residual scales mismatch",
    )
    require(
        set(float(value) for value in candidates["distill_weight"])
        == {0.0, 1.0},
        "candidate distillation weights mismatch",
    )
    require(
        bool((candidates["deployment_beta"].astype(float) == 1.0).all()),
        "candidate uses post-hoc beta",
    )
    require(
        bool((history["deployment_beta"].astype(float) == 1.0).all()),
        "training history uses post-hoc beta",
    )

    provenance = summary["provenance"]
    require(provenance["strict_oof_experts_used_for_train_targets"] is True, "OOF targets missing")
    require(provenance["each_train_target_excludes_its_sample"] is True, "sample isolation missing")
    require(provenance["full_train_expert_outputs_used_for_student_training"] is False, "full experts leaked")
    require(provenance["teacher_targets_projected_to_deployment_scale"] is True, "targets not projected")
    require(provenance["training_prediction_equals_deployment_prediction"] is True, "train/deploy mismatch")
    require(provenance["posthoc_beta_calibration_used"] is False, "post-hoc beta was used")
    require(provenance["deployment_beta_fixed_to_one"] is True, "deployment beta not fixed")
    require(provenance["official_test_used_for_training_or_selection"] is False, "Test entered selection")
    require(provenance["test_dataset_constructed_after_selection_artifact_written"] is True, "Test isolation missing")
    require(provenance["student_deploys_without_expert_pool"] is True, "expert pool required at inference")
    require(provenance["backbone_frozen"] is True, "Anchor backbone was trained")
    require(provenance["neural_router_trained"] is False, "router was trained")

    teacher_provenance = teacher["provenance"]
    require(teacher_provenance["strict_oof_experts"] is True, "teacher is not strict OOF")
    require(teacher_provenance["each_training_target_excludes_its_sample"] is True, "teacher holdout isolation missing")
    require(teacher_provenance["test_labels_used_to_construct_teacher"] is False, "Test labels entered teacher")

    selected_id = summary["selected_candidate_id"]
    require(selected_id == selection["selected"]["candidate_id"], "selected ID mismatch")
    if selected_id != "anchor_fallback":
        require(selected_id in set(candidates["candidate_id"]), "selected candidate absent")
        selected_scale = float(summary["selected_residual_max"])
        require(selected_scale in {0.015, 0.03, 0.045}, "invalid selected scale")
        require(close(summary["deployment_beta"], 1.0), "summary beta mismatch")
        require(selection["posthoc_beta_calibration_used"] is False, "selection beta leakage")
        matched = summary["matched_control"]
        require(matched is not None, "matched control missing")
        require(
            close(matched["residual_max"], selected_scale),
            "matched control scale mismatch",
        )
        require(
            close(
                float(matched["distill_weight"])
                + float(summary["selected_distill_weight"]),
                1.0,
            ),
            "matched control is not the opposite distillation condition",
        )
    else:
        selected_scale = 0.0

    require(len(predictions) > 0, "empty Test predictions")
    required_columns = {
        "sample_id",
        "label",
        "anchor",
        "selected_prediction",
        "selected_correction",
        "anchor_abs_error",
        "selected_abs_error",
    }
    require(required_columns.issubset(predictions.columns), "prediction columns missing")
    require(
        not predictions["sample_id"].astype(str).duplicated().any(),
        "duplicate Test sample IDs",
    )
    require(
        torch.allclose(
            torch.tensor(predictions["selected_prediction"].to_numpy()).float(),
            torch.tensor(predictions["anchor"].to_numpy()).float()
            + torch.tensor(predictions["selected_correction"].to_numpy()).float(),
            atol=1e-6,
        ),
        "stored selected prediction is not anchor plus correction",
    )

    anchor_error = torch.tensor(predictions["anchor_abs_error"].to_numpy()).float()
    selected_error = torch.tensor(predictions["selected_abs_error"].to_numpy()).float()
    test_anchor_mae = float(anchor_error.mean().item())
    test_selected_mae = float(selected_error.mean().item())
    test_gain = test_anchor_mae - test_selected_mae
    test_harm = float(
        ((selected_error - anchor_error) > 0.10).float().mean().item()
    )
    require(close(test_anchor_mae, summary["test_anchor_mae"]), "Test Anchor MAE mismatch")
    require(close(test_selected_mae, summary["test_selected_mae"]), "Test selected MAE mismatch")
    require(close(test_gain, summary["test_selected_gain"]), "Test gain mismatch")
    require(close(test_harm, summary["test_selected_harm_over_010_rate"]), "Test harm mismatch")

    max_abs_correction = float(
        predictions["selected_correction"].abs().max()
    )
    if selected_id == "anchor_fallback":
        require(max_abs_correction <= 1e-7, "fallback has nonzero correction")
    else:
        require(
            max_abs_correction <= selected_scale + 1e-6,
            "deployment correction exceeds trained scale",
        )
    require(
        close(max_abs_correction, summary["test_selected_max_abs_correction"], atol=2e-6),
        "stored max correction mismatch",
    )

    require(
        "validation_selected_deployable" in set(test_summary["model"]),
        "selected Test row missing",
    )
    if selected_id != "anchor_fallback":
        require(
            "predeclared_matched_control" in set(test_summary["model"]),
            "matched Test row missing",
        )
        require("matched_prediction" in predictions.columns, "matched prediction column missing")
        require("matched_abs_error" in predictions.columns, "matched error column missing")

    for name, value in summary["checkpoint_paths"].items():
        if value is None:
            continue
        path = Path(value)
        require(path.is_file(), f"missing checkpoint {name}: {path}")
        expected = summary["checkpoint_sha256"][name]
        require(sha256(path) == expected, f"checkpoint hash mismatch: {name}")

    print("V9.16 ENGINEERING AUDIT PASSED")
    print("protocol: strict OOF absolute targets projected to final correction scale")
    print("post-hoc beta calibration: False; deployment beta: 1.0")
    print("candidate count:", summary["candidate_count"])
    print("selected candidate:", selected_id)
    print("selected residual_max:", f"{selected_scale:.3f}")
    print(
        "Validation anchor/selected/gain:",
        f"{summary['validation_anchor_mae']:.6f}",
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
        "Test correction mean/max/saturation:",
        f"{summary['test_selected_mean_abs_correction']:.6f}",
        f"{summary['test_selected_max_abs_correction']:.6f}",
        f"{summary['test_selected_saturation_rate']:.4f}",
    )
    if summary["matched_control_test"] is not None:
        matched = summary["matched_control_test"]
        print(
            "Matched control Test MAE/gain:",
            f"{matched['mae']:.6f}",
            f"{matched['gain_vs_anchor']:+.6f}",
        )


if __name__ == "__main__":
    main()
