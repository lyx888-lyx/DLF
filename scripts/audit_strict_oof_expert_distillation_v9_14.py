"""Engineering audit for V9.14 strict OOF expert distillation."""

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

from trains.singleTask.model.StrictOOFExpertStudentV914 import (  # noqa: E402
    STUDENT_VERSION,
)
from trains.singleTask.strict_oof_expert_distillation_v914 import (  # noqa: E402
    TEACHER_VERSION,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="result/strict_oof_expert_distillation_v914/mosi/seed_1111",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main():
    cli = parse_args()
    root = Path(cli.root)
    teacher_path = root / "strict_oof_soft_teacher_v914.pth"
    teacher_csv = root / "v914_strict_oof_teacher_diagnostic.csv"
    history_path = root / "v914_student_training_history.csv"
    candidates_path = root / "v914_validation_candidates.csv"
    selection_path = root / "v914_validation_selection.json"
    prediction_path = root / "v914_test_predictions.csv"
    test_summary_path = root / "v914_test_summary.csv"
    summary_path = root / "strict_oof_expert_distillation_v914_summary.json"

    for path in (
        teacher_path,
        teacher_csv,
        history_path,
        candidates_path,
        selection_path,
        prediction_path,
        test_summary_path,
        summary_path,
    ):
        require(path.is_file(), f"missing V9.14 output: {path}")

    teacher = torch.load(teacher_path, map_location="cpu")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidates = pd.read_csv(candidates_path)
    predictions = pd.read_csv(prediction_path)
    test_summary = pd.read_csv(test_summary_path)

    require(teacher.get("version") == TEACHER_VERSION, "teacher version mismatch")
    require(summary.get("version") == STUDENT_VERSION, "student version mismatch")
    require(selection.get("version") == STUDENT_VERSION, "selection version mismatch")
    require(
        teacher["provenance"].get("strict_oof_experts") is True,
        "teacher is not strict OOF",
    )
    require(
        teacher["provenance"].get(
            "each_training_target_excludes_its_sample"
        ) is True,
        "a training target may include its own sample",
    )
    require(
        teacher["provenance"].get(
            "validation_labels_used_to_construct_teacher"
        ) is False,
        "Validation labels entered teacher construction",
    )
    require(
        teacher["provenance"].get(
            "test_labels_used_to_construct_teacher"
        ) is False,
        "Test labels entered teacher construction",
    )
    require(
        selection.get("selected_by_validation_only") is True,
        "selection was not Validation-only",
    )
    require(
        selection.get("test_dataset_constructed") is False,
        "selection artifact was written after Test construction",
    )
    require(
        selection.get("test_evaluated") is False,
        "selection artifact was written after Test evaluation",
    )

    provenance = summary.get("provenance", {})
    require(
        provenance.get("strict_oof_experts_used_for_train_targets") is True,
        "strict OOF experts were not used",
    )
    require(
        provenance.get("each_train_target_excludes_its_sample") is True,
        "train target isolation missing",
    )
    require(
        provenance.get("full_train_expert_outputs_used_for_student_training")
        is False,
        "full-train expert outputs entered Student training",
    )
    require(
        provenance.get("official_test_used_for_training_or_selection") is False,
        "Test entered training or selection",
    )
    require(
        provenance.get(
            "test_dataset_constructed_after_selection_artifact_written"
        ) is True,
        "Test was constructed before selection was frozen",
    )
    require(
        provenance.get("sample_id_alignment_checked") is True,
        "sample ID alignment was not checked",
    )
    require(
        provenance.get("student_deploys_without_expert_pool") is True,
        "deployment still depends on the expert pool",
    )
    require(
        provenance.get("backbone_frozen") is True,
        "V9.14 unexpectedly fine-tuned the Anchor backbone",
    )
    require(
        provenance.get("neural_router_trained") is False,
        "V9.14 unexpectedly trained a router",
    )

    n = len(teacher["sample_ids"])
    require(n > 0, "empty strict OOF teacher")
    require(len(set(teacher["sample_ids"])) == n, "duplicate teacher sample IDs")
    require(teacher["labels"].shape == (n, 1), "teacher label shape mismatch")
    require(teacher["anchor"].shape == (n, 1), "teacher anchor shape mismatch")
    require(
        teacher["expert_predictions"].shape == (n, 4),
        "teacher expert prediction shape mismatch",
    )
    require(
        teacher["expert_corrections"].shape == (n, 4),
        "teacher expert correction shape mismatch",
    )
    require(
        teacher["expert_relevance"].shape == (n, 4),
        "teacher relevance shape mismatch",
    )
    require(
        bool(torch.isfinite(teacher["teacher_prediction"]).all()),
        "non-finite teacher prediction",
    )
    require(
        bool(torch.isfinite(teacher["expert_relevance"]).all()),
        "non-finite expert relevance",
    )
    require(
        torch.allclose(
            teacher["expert_relevance"].sum(dim=1),
            torch.ones(n),
            atol=1e-5,
        ),
        "expert relevance does not sum to one",
    )
    max_alpha = float(teacher["config"]["max_alpha"])
    require(
        float(teacher["alpha"].max()) <= max_alpha + 1e-6,
        "teacher alpha exceeds configured bound",
    )
    anchor_error = torch.abs(teacher["anchor"] - teacher["labels"])
    teacher_error = torch.abs(
        teacher["teacher_prediction"] - teacher["labels"]
    )
    require(
        bool((teacher_error <= anchor_error + 1e-5).all()),
        "bounded teacher is worse than OOF Anchor on a training sample",
    )

    require(len(candidates) >= 2, "too few Student candidates")
    require(
        set(candidates["distill_weight"].round(4)) >= {0.0, 0.25},
        "label-only or distilled control is missing",
    )
    selected_checkpoint = Path(
        summary["checkpoint_paths"]["selected_student"]
    )
    require(selected_checkpoint.is_file(), "selected Student checkpoint missing")
    selected_payload = torch.load(selected_checkpoint, map_location="cpu")
    require(
        selected_payload.get("version") == STUDENT_VERSION,
        "selected checkpoint version mismatch",
    )
    require(
        isinstance(selected_payload.get("student_state_dict"), dict),
        "selected checkpoint has no Student state",
    )
    require(
        all(
            not key.startswith("backbone.")
            for key in selected_payload["student_state_dict"]
        ),
        "selected checkpoint unexpectedly stores a trainable backbone",
    )

    required_prediction_columns = {
        "sample_id",
        "label",
        "anchor",
        "selected_prediction",
        "selected_correction",
        "anchor_abs_error",
        "selected_abs_error",
    }
    require(
        required_prediction_columns.issubset(predictions.columns),
        "test prediction columns missing",
    )
    require(len(predictions) > 0, "empty Test predictions")
    require(
        len(set(predictions["sample_id"])) == len(predictions),
        "duplicate Test prediction IDs",
    )
    require(
        {
            "anchor",
            "validation_selected_deployable",
        }.issubset(set(test_summary["model"])),
        "Test summary rows missing",
    )

    print("V9.14 ENGINEERING AUDIT PASSED")
    print("protocol: strict OOF expert targets; one frozen-Anchor Student")
    print("strict OOF training samples:", n)
    print("Student candidates:", len(candidates))
    print("selected candidate:", summary["selected_candidate_id"])
    print("selected distill weight:", summary["selected_distill_weight"])
    print("selected epoch:", summary["selected_epoch"])
    print(
        "OOF Anchor/teacher/gain:",
        f"{summary['teacher_diagnostic']['anchor_oof_mae']:.6f}",
        f"{summary['teacher_diagnostic']['teacher_oof_mae']:.6f}",
        f"{summary['teacher_diagnostic']['teacher_gain']:+.6f}",
    )
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
        "Test harm_over_010_rate:",
        f"{summary['test_selected_harm_over_010_rate']:.4f}",
    )
    if summary["selected_candidate_id"] == "anchor_fallback":
        print("WARNING: Validation rejected all Student variants")
    elif summary["test_selected_gain"] <= 0.0:
        print("WARNING: selected Student did not beat Anchor on Test")


if __name__ == "__main__":
    main()
