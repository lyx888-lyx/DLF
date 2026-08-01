"""Train V9.15 strict-OOF absolute-target residual Student."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Mapping

import pandas as pd
import torch

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.model.AbsoluteTargetExpertStudentV915 import STUDENT_VERSION
from trains.singleTask.model.SemanticCostCoachV99 import SPECIALIST_NAMES
from trains.singleTask.semantic_expert_pool_v99 import anchor_checkpoint_from_summary
from trains.singleTask.strict_oof_absolute_target_distillation_v915 import (
    TEACHER_VERSION,
    StudentConfigV915,
    TeacherConfigV914,
    align_teacher_to_dataset,
    build_soft_teacher_v914,
    evaluate_student,
    load_student_checkpoint,
    new_student,
    train_student_variant,
    trainable_state_dict,
)
from utils import assign_gpu, setup_seed

logger = logging.getLogger("MMSA")


def parse_float_grid(value: str):
    result = tuple(
        float(item.strip()) for item in value.split(",") if item.strip()
    )
    if not result:
        raise argparse.ArgumentTypeError("grid cannot be empty")
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "V9.15 strict OOF absolute-target distillation into one "
            "conservative frozen-anchor residual Student"
        )
    )
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", default="./config/config.json")
    parser.add_argument(
        "--v92-root",
        default="./result/oof_tail_residual_experts_v92/mosi/seed_1111",
    )
    parser.add_argument(
        "--v98-root",
        default="./result/strict_nested_frontier_coach_v98/mosi/seed_1111",
    )
    parser.add_argument("--strict-oof-pool", default="")
    parser.add_argument("--anchor-checkpoint", default="")
    parser.add_argument(
        "--save-root",
        default="./result/strict_oof_absolute_target_distillation_v915",
    )
    parser.add_argument("--num-workers", type=int, default=1)

    parser.add_argument("--teacher-temperature", type=float, default=0.15)
    parser.add_argument("--teacher-gain-margin", type=float, default=0.05)
    parser.add_argument("--teacher-gain-scale", type=float, default=0.15)
    parser.add_argument("--teacher-max-alpha", type=float, default=0.50)

    parser.add_argument("--student-hidden-dim", type=int, default=64)
    parser.add_argument("--student-dropout", type=float, default=0.10)
    parser.add_argument("--student-residual-max", type=float, default=0.15)
    parser.add_argument(
        "--student-auxiliary-residual-max", type=float, default=1.50
    )
    parser.add_argument("--student-max-epochs", type=int, default=60)
    parser.add_argument("--student-early-stop", type=int, default=10)
    parser.add_argument("--student-learning-rate", type=float, default=3e-5)
    parser.add_argument("--student-weight-decay", type=float, default=1e-3)
    parser.add_argument("--student-batch-size", type=int, default=32)
    parser.add_argument("--student-auxiliary-weight", type=float, default=0.05)
    parser.add_argument("--student-correction-penalty", type=float, default=0.10)
    parser.add_argument("--student-gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--distill-weight-grid",
        type=parse_float_grid,
        default=parse_float_grid("0.00,0.25,0.50,1.00"),
    )
    parser.add_argument(
        "--beta-grid",
        type=parse_float_grid,
        default=parse_float_grid(
            "0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,"
            "0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00"
        ),
    )
    parser.add_argument("--minimum-validation-gain", type=float, default=0.0005)
    parser.add_argument("--maximum-validation-harm", type=float, default=0.05)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def resolve_paths(cli):
    v92_root = Path(
        str(cli.v92_root).replace("seed_1111", f"seed_{cli.seed}")
    )
    v98_root = Path(
        str(cli.v98_root).replace("seed_1111", f"seed_{cli.seed}")
    )
    strict_pool = (
        Path(cli.strict_oof_pool)
        if cli.strict_oof_pool
        else v98_root
        / "strict_nested_frontier_pool"
        / "strict_nested_v93_frontier_pool_v98.pth"
    )
    anchor = (
        Path(cli.anchor_checkpoint)
        if cli.anchor_checkpoint
        else anchor_checkpoint_from_summary(v92_root)
    )
    missing = [
        str(path)
        for path in (strict_pool, anchor)
        if not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "V9.15 requires the completed V9.8 strict nested pool and the "
            f"full deployment Anchor checkpoint. Missing: {missing}"
        )
    return strict_pool, anchor


def write_teacher_diagnostics(path: Path, teacher) -> None:
    frame = pd.DataFrame(
        {
            "sample_id": teacher["sample_ids"],
            "group_id": teacher["group_ids"],
            "outer_fold": teacher["fold_index"].view(-1).tolist(),
            "label": teacher["labels"].view(-1).tolist(),
            "oof_anchor": teacher["anchor"].view(-1).tolist(),
            "proposal": teacher["proposal"].view(-1).tolist(),
            "alpha": teacher["alpha"].view(-1).tolist(),
            "teacher_prediction": teacher[
                "teacher_prediction"
            ].view(-1).tolist(),
            "teacher_gain": teacher["teacher_gain"].view(-1).tolist(),
            "primary_expert": [
                SPECIALIST_NAMES[int(index)]
                for index in teacher["primary_expert_index"].view(-1).tolist()
            ],
        }
    )
    for index, name in enumerate(SPECIALIST_NAMES):
        frame[f"{name}_prediction"] = teacher[
            "expert_predictions"
        ][:, index].tolist()
        frame[f"{name}_relevance"] = teacher[
            "expert_relevance"
        ][:, index].tolist()
    frame.to_csv(path, index=False)


def save_anchor_fallback(
    args,
    valid_dataset,
    anchor_checkpoint: Path,
    student_config: StudentConfigV915,
    output_path: Path,
    num_workers: int,
):
    model = new_student(args, anchor_checkpoint, student_config)
    validation = evaluate_student(
        model,
        valid_dataset,
        args.device,
        student_config.batch_size,
        num_workers,
        beta=0.0,
    )
    torch.save(
        {
            "version": STUDENT_VERSION,
            "student_state_dict": trainable_state_dict(model),
            "anchor_checkpoint": str(anchor_checkpoint),
            "student_config": student_config.__dict__,
            "teacher_config": {},
            "distill_weight": 0.0,
            "epoch": 0,
            "selected_beta": 0.0,
            "validation_anchor_mae": validation["anchor_mae"],
            "validation_raw_mae": validation["anchor_mae"],
            "validation_mae": validation["mae"],
            "validation_gain": validation["gain_vs_anchor"],
            "validation_harm_over_010_rate": validation[
                "harm_over_010_rate"
            ],
            "anchor_fallback": True,
        },
        output_path,
    )
    return validation


def main():
    cli = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    setup_seed(cli.seed)
    args = get_config_regression("DLF", cli.dataset, Path(cli.config))
    args["device"] = assign_gpu([cli.gpu])
    args["mode"] = "train"
    args["train_mode"] = "regression"
    args["feature_T"] = ""
    args["feature_A"] = ""
    args["feature_V"] = ""
    args["seed"] = int(cli.seed)
    args["cur_seed"] = int(cli.seed)
    args["batch_size"] = int(cli.student_batch_size)

    strict_pool_path, anchor_checkpoint = resolve_paths(cli)
    train_dataset = MMDataset(args, mode="train")
    valid_dataset = MMDataset(args, mode="valid")
    if "seq_lens" in args:
        args["seq_lens"] = train_dataset.get_seq_len()

    save_dir = Path(cli.save_root) / cli.dataset / f"seed_{cli.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    teacher_path = save_dir / "strict_oof_soft_teacher_v915.pth"
    teacher_csv_path = save_dir / "v915_strict_oof_teacher_diagnostic.csv"
    history_path = save_dir / "v915_student_training_history.csv"
    candidates_path = save_dir / "v915_validation_candidates.csv"
    selection_path = save_dir / "v915_validation_selection.json"
    prediction_path = save_dir / "v915_test_predictions.csv"
    test_summary_path = save_dir / "v915_test_summary.csv"
    summary_path = save_dir / "strict_oof_absolute_target_distillation_v915_summary.json"
    anchor_fallback_path = save_dir / "anchor_fallback_v915.pth"

    strict_pool = torch.load(strict_pool_path, map_location="cpu")
    teacher_config = TeacherConfigV914(
        temperature=cli.teacher_temperature,
        gain_margin=cli.teacher_gain_margin,
        gain_scale=cli.teacher_gain_scale,
        max_alpha=cli.teacher_max_alpha,
    )
    teacher = align_teacher_to_dataset(
        build_soft_teacher_v914(strict_pool, teacher_config),
        train_dataset,
    )
    torch.save(teacher, teacher_path)
    write_teacher_diagnostics(teacher_csv_path, teacher)

    student_config = StudentConfigV915(
        hidden_dim=cli.student_hidden_dim,
        dropout=cli.student_dropout,
        residual_max=cli.student_residual_max,
        auxiliary_residual_max=cli.student_auxiliary_residual_max,
        max_epochs=cli.student_max_epochs,
        early_stop=cli.student_early_stop,
        learning_rate=cli.student_learning_rate,
        weight_decay=cli.student_weight_decay,
        batch_size=cli.student_batch_size,
        auxiliary_weight=cli.student_auxiliary_weight,
        correction_penalty=cli.student_correction_penalty,
        gradient_clip=cli.student_gradient_clip,
    )

    anchor_validation = save_anchor_fallback(
        args,
        valid_dataset,
        anchor_checkpoint,
        student_config,
        anchor_fallback_path,
        cli.num_workers,
    )

    candidate_rows = []
    all_history = []
    for distill_weight in cli.distill_weight_grid:
        checkpoint_path = (
            save_dir
            / f"student_absolute_distill_{float(distill_weight):.4f}_best_v915.pth"
        )
        result = train_student_variant(
            args=args,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            teacher=teacher,
            anchor_checkpoint=anchor_checkpoint,
            output_path=checkpoint_path,
            student_config=student_config,
            distill_weight=float(distill_weight),
            beta_grid=cli.beta_grid,
            seed=int(cli.seed) + 915101,
            num_workers=cli.num_workers,
        )
        all_history.extend(result.pop("history"))
        candidate_rows.append(result)
        logger.info(
            "V9.15 candidate %s valid=%.6f raw=%.6f gain=%+.6f "
            "epoch=%d beta=%.2f",
            result["candidate_id"],
            result["validation_mae"],
            result["validation_raw_mae"],
            result["validation_gain"],
            result["best_epoch"],
            result["selected_beta"],
        )
    pd.DataFrame(all_history).to_csv(history_path, index=False)
    pd.DataFrame(candidate_rows).to_csv(candidates_path, index=False)

    best = min(
        candidate_rows,
        key=lambda row: (
            float(row["validation_mae"]),
            float(row["validation_harm_over_010_rate"]),
            float(row["selected_beta"]),
            float(row["distill_weight"]),
        ),
    )
    passes_gain = (
        float(anchor_validation["anchor_mae"])
        - float(best["validation_mae"])
        >= float(cli.minimum_validation_gain)
    )
    passes_harm = (
        float(best["validation_harm_over_010_rate"])
        <= float(cli.maximum_validation_harm)
    )
    if passes_gain and passes_harm and float(best["selected_beta"]) > 0.0:
        selected = dict(best)
        selected["family"] = "student"
    else:
        selected = {
            "candidate_id": "anchor_fallback",
            "family": "anchor_fallback",
            "distill_weight": 0.0,
            "checkpoint": str(anchor_fallback_path),
            "best_epoch": 0,
            "selected_beta": 0.0,
            "validation_anchor_mae": anchor_validation["anchor_mae"],
            "validation_raw_mae": anchor_validation["anchor_mae"],
            "validation_mae": anchor_validation["mae"],
            "validation_gain": anchor_validation["gain_vs_anchor"],
            "validation_harm_over_010_rate": anchor_validation[
                "harm_over_010_rate"
            ],
        }

    selection_payload = {
        "version": STUDENT_VERSION,
        "teacher_version": TEACHER_VERSION,
        "selected_by_validation_only": True,
        "test_dataset_constructed": False,
        "test_evaluated": False,
        "selected": selected,
        "candidates": candidate_rows,
        "anchor_validation": {
            key: value
            for key, value in anchor_validation.items()
            if not isinstance(value, (torch.Tensor, list))
        },
        "beta_grid": list(cli.beta_grid),
        "minimum_validation_gain": float(cli.minimum_validation_gain),
        "maximum_validation_harm": float(cli.maximum_validation_harm),
        "teacher_diagnostic": teacher["diagnostic"],
        "strict_oof_pool": str(strict_pool_path),
        "strict_oof_pool_sha256": sha256(strict_pool_path),
        "anchor_checkpoint": str(anchor_checkpoint),
        "anchor_checkpoint_sha256": sha256(anchor_checkpoint),
    }
    selection_path.write_text(
        json.dumps(jsonable(selection_payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info(
        "V9.15 Validation selected %s: MAE %.6f gain %+.6f beta=%.2f",
        selected["candidate_id"],
        selected["validation_mae"],
        selected["validation_gain"],
        selected["selected_beta"],
    )

    # Test is not constructed until the complete selection artifact exists.
    test_dataset = MMDataset(args, mode="test")
    selected_model, selected_checkpoint = load_student_checkpoint(
        args,
        Path(selected["checkpoint"]),
        anchor_checkpoint,
        student_config,
    )
    test = evaluate_student(
        selected_model,
        test_dataset,
        args.device,
        student_config.batch_size,
        cli.num_workers,
        beta=float(selected["selected_beta"]),
    )
    valid_ids = set(
        evaluate_student(
            selected_model,
            valid_dataset,
            args.device,
            student_config.batch_size,
            cli.num_workers,
            beta=float(selected["selected_beta"]),
        )["sample_ids"]
    )
    test_ids = set(test["sample_ids"])
    if valid_ids & test_ids:
        raise RuntimeError("Validation/Test sample ID overlap")

    pd.DataFrame(
        {
            "sample_id": test["sample_ids"],
            "label": test["labels"].view(-1).tolist(),
            "anchor": test["anchor"].view(-1).tolist(),
            "raw_student_prediction": test[
                "raw_prediction"
            ].view(-1).tolist(),
            "raw_student_correction": test["correction"].view(-1).tolist(),
            "selected_beta": [float(selected["selected_beta"])] * len(test["labels"]),
            "selected_prediction": test["prediction"].view(-1).tolist(),
            "anchor_abs_error": torch.abs(
                test["anchor"] - test["labels"]
            ).view(-1).tolist(),
            "selected_abs_error": torch.abs(
                test["prediction"] - test["labels"]
            ).view(-1).tolist(),
        }
    ).to_csv(prediction_path, index=False)
    pd.DataFrame(
        [
            {
                "model": "anchor",
                "validation_mae": anchor_validation["anchor_mae"],
                "test_mae": test["anchor_mae"],
                "test_gain_vs_anchor": 0.0,
                "test_harm_over_010_rate": 0.0,
            },
            {
                "model": "validation_selected_deployable",
                "candidate_id": selected["candidate_id"],
                "distill_weight": selected["distill_weight"],
                "best_epoch": selected["best_epoch"],
                "selected_beta": selected["selected_beta"],
                "validation_raw_mae": selected["validation_raw_mae"],
                "validation_mae": selected["validation_mae"],
                "validation_gain": selected["validation_gain"],
                "test_mae": test["mae"],
                "test_gain_vs_anchor": test["gain_vs_anchor"],
                "test_harm_over_010_rate": test["harm_over_010_rate"],
            },
        ]
    ).to_csv(test_summary_path, index=False)

    summary = {
        "version": STUDENT_VERSION,
        "teacher_version": TEACHER_VERSION,
        "method": "strict_oof_absolute_target_residual_student_v9_15",
        "dataset": cli.dataset,
        "seed": int(cli.seed),
        "selected_candidate_id": selected["candidate_id"],
        "selected_distill_weight": selected["distill_weight"],
        "selected_epoch": selected["best_epoch"],
        "selected_beta": selected["selected_beta"],
        "validation_anchor_mae": anchor_validation["anchor_mae"],
        "validation_selected_raw_mae": selected["validation_raw_mae"],
        "validation_selected_mae": selected["validation_mae"],
        "validation_selected_gain": selected["validation_gain"],
        "test_anchor_mae": test["anchor_mae"],
        "test_selected_mae": test["mae"],
        "test_selected_gain": test["gain_vs_anchor"],
        "test_selected_harm_over_010_rate": test["harm_over_010_rate"],
        "teacher_config": teacher["config"],
        "teacher_diagnostic": teacher["diagnostic"],
        "student_config": student_config.__dict__,
        "beta_grid": list(cli.beta_grid),
        "candidate_count": len(candidate_rows),
        "candidates": candidate_rows,
        "checkpoint_paths": {
            "strict_oof_pool": str(strict_pool_path),
            "anchor": str(anchor_checkpoint),
            "selected_student": str(selected["checkpoint"]),
        },
        "checkpoint_sha256": {
            "strict_oof_pool": sha256(strict_pool_path),
            "anchor": sha256(anchor_checkpoint),
            "selected_student": sha256(Path(selected["checkpoint"])),
        },
        "selected_checkpoint_metadata": {
            key: value
            for key, value in selected_checkpoint.items()
            if key != "student_state_dict"
        },
        "outputs": {
            "teacher": str(teacher_path),
            "teacher_diagnostic": str(teacher_csv_path),
            "training_history": str(history_path),
            "validation_candidates": str(candidates_path),
            "validation_selection": str(selection_path),
            "test_predictions": str(prediction_path),
            "test_summary": str(test_summary_path),
        },
        "provenance": {
            "strict_oof_experts_used_for_train_targets": True,
            "each_train_target_excludes_its_sample": True,
            "full_train_expert_outputs_used_for_student_training": False,
            "train_labels_used_for_student_training": True,
            "absolute_teacher_predictions_used_for_distillation": True,
            "oof_relative_corrections_used_for_unified_distillation": False,
            "official_validation_used_for_early_stopping": True,
            "official_validation_used_for_beta_calibration": True,
            "official_validation_used_for_candidate_selection": True,
            "official_test_used_for_training_or_selection": False,
            "test_dataset_constructed_after_selection_artifact_written": True,
            "sample_id_alignment_checked": True,
            "student_deploys_without_expert_pool": True,
            "backbone_frozen": True,
            "neural_router_trained": False,
        },
    }
    summary_path.write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info(
        "V9.15 TEST anchor=%.6f selected=%.6f gain=%+.6f "
        "candidate=%s beta=%.2f",
        test["anchor_mae"],
        test["mae"],
        test["gain_vs_anchor"],
        selected["candidate_id"],
        selected["selected_beta"],
    )


if __name__ == "__main__":
    main()
