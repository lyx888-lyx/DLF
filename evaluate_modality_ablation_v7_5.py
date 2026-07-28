"""Evaluate the frozen V7.1 pipeline under controlled modality availability.

This script answers two distinct questions without retraining:
1. How does the existing full-modality-trained method behave when only a subset
   of modalities is available at inference time?
2. After recalibrating only ensemble weights and shrinkage on the validation
   split under the same availability condition, which single modality is best?

All condition-specific choices are fitted on Valid. Test labels never influence
anchor/committee/student/hybrid policy selection.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from config import get_config_regression
from data_loader import MMDataLoader
from trains.singleTask.complementarity_v71 import (
    apply_global_committee,
    apply_region_committee,
    fit_committee_cv,
    selection_stats,
)
from trains.singleTask.expert_analysis import normalize_batch_ids
from trains.singleTask.model.CPFD_DLF import ComplementarityResidualStudent
from trains.singleTask.model.FSC_DLF import FrozenTeacherDLF
from trains.utils import MetricsTop
from utils import assign_gpu, setup_seed


LOGGER = logging.getLogger("MMSA")
VALID_MODALITIES = frozenset("tav")
DEFAULT_CONDITIONS = ("t", "a", "v", "ta", "tv", "av", "tav")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate V7.1 under t/a/v/ta/tv/av/tav availability with "
            "condition-specific Valid-only calibration."
        )
    )
    parser.add_argument("--dataset", choices=["mosi", "mosei"], default="mosi")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=str, default="./config/config.json")
    parser.add_argument(
        "--source-run-dir",
        type=str,
        default="./result/complementarity_v71/mosi/seed_1111",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default="./result/modality_ablation_v75",
    )
    parser.add_argument("--teacher-cache", type=str, default="")
    parser.add_argument("--source-summary", type=str, default="")
    parser.add_argument("--source-checkpoint", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.18)
    parser.add_argument("--residual-max", type=float, default=0.35)
    parser.add_argument("--region-temperature", type=float, default=0.55)
    parser.add_argument("--committee-steps", type=int, default=600)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(DEFAULT_CONDITIONS),
        help="Any subset of: t a v ta tv av tav",
    )
    return parser.parse_args()


def _load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _normalize_condition(value: str) -> str:
    value = "".join(character for character in "tav" if character in value.lower())
    if not value or not set(value).issubset(VALID_MODALITIES):
        raise ValueError(f"Invalid modality condition: {value!r}")
    return value


def _mask_modalities(
    text: torch.Tensor,
    audio: torch.Tensor,
    vision: torch.Tensor,
    condition: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero unavailable modality tensors while preserving input shapes."""
    condition = _normalize_condition(condition)
    if "t" not in condition:
        text = torch.zeros_like(text)
    if "a" not in condition:
        audio = torch.zeros_like(audio)
    if "v" not in condition:
        vision = torch.zeros_like(vision)
    return text, audio, vision


def _sort_collected(
    predictions: List[torch.Tensor],
    labels: List[torch.Tensor],
    sample_ids: List[object],
) -> Dict[str, object]:
    prediction = torch.cat(predictions, dim=0).float()
    target = torch.cat(labels, dim=0).float()
    order = sorted(range(len(sample_ids)), key=lambda index: str(sample_ids[index]))
    indices = torch.tensor(order, dtype=torch.long)
    return {
        "prediction": prediction[indices],
        "labels": target[indices],
        "sample_ids": [sample_ids[index] for index in order],
    }


@torch.no_grad()
def _collect_teacher(
    model: FrozenTeacherDLF,
    dataloader,
    device,
    condition: str,
) -> Dict[str, object]:
    predictions: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    sample_ids: List[object] = []
    model.eval()
    for batch in tqdm(dataloader, leave=False):
        text, audio, vision = _mask_modalities(
            batch["text"].to(device),
            batch["audio"].to(device),
            batch["vision"].to(device),
            condition,
        )
        output = model(text, audio, vision)
        predictions.append(output["prediction"].detach().cpu())
        labels.append(batch["labels"]["M"].view(-1, 1).cpu())
        sample_ids.extend(normalize_batch_ids(batch.get("id")))
    return _sort_collected(predictions, labels, sample_ids)


@torch.no_grad()
def _collect_student(
    model: ComplementarityResidualStudent,
    dataloader,
    device,
    condition: str,
) -> Dict[str, object]:
    buffers: Dict[str, List[torch.Tensor]] = {
        "base_prediction": [],
        "prediction": [],
    }
    labels: List[torch.Tensor] = []
    sample_ids: List[object] = []
    model.eval()
    for batch in tqdm(dataloader, leave=False):
        text, audio, vision = _mask_modalities(
            batch["text"].to(device),
            batch["audio"].to(device),
            batch["vision"].to(device),
            condition,
        )
        output = model(text, audio, vision)
        for key in buffers:
            buffers[key].append(output[key].detach().cpu())
        labels.append(batch["labels"]["M"].view(-1, 1).cpu())
        sample_ids.extend(normalize_batch_ids(batch.get("id")))

    order = sorted(range(len(sample_ids)), key=lambda index: str(sample_ids[index]))
    indices = torch.tensor(order, dtype=torch.long)
    result: Dict[str, object] = {
        "labels": torch.cat(labels, dim=0).float()[indices],
        "sample_ids": [sample_ids[index] for index in order],
    }
    for key, values in buffers.items():
        result[key] = torch.cat(values, dim=0).float()[indices]
    return result


def _assert_alignment(reference: Mapping[str, object], other: Mapping[str, object], name: str):
    if reference["sample_ids"] != other["sample_ids"]:
        raise RuntimeError(f"Sample ID mismatch for {name}.")
    if not torch.allclose(reference["labels"], other["labels"], atol=1e-6, rtol=0.0):
        raise RuntimeError(f"Label mismatch for {name}.")


def _metric_dict(metrics_fn, prediction: torch.Tensor, labels: torch.Tensor):
    return {
        key: float(value)
        for key, value in metrics_fn(
            prediction.detach().cpu(), labels.detach().cpu()
        ).items()
    }


def _select_student_alpha(base, raw, labels):
    rows = []
    best = None
    for alpha in (0.0, 0.25, 0.50, 0.75, 1.0):
        prediction = base + float(alpha) * (raw - base)
        stats = selection_stats(base, prediction, labels)
        row = {
            "alpha": float(alpha),
            "objective": float(stats["mae"] + 0.05 * stats["harm_over_010_rate"]),
            **stats,
        }
        rows.append(row)
        if best is None or (row["objective"], row["mae"]) < (
            best["objective"], best["mae"]
        ):
            best = row
    return best, rows


def _committee_predictions(predictions, anchor, fitted):
    uniform = predictions.mean(dim=1)
    global_simplex = apply_global_committee(
        predictions, fitted["global_weights"]
    )
    region_simplex = apply_region_committee(
        predictions,
        anchor,
        fitted["region_weights"],
        0.55,
    )
    selected = (
        region_simplex
        if fitted["selected"] == "region_simplex"
        else global_simplex
    )
    return {
        "uniform": uniform,
        "global_simplex": global_simplex,
        "region_simplex": region_simplex,
        "selected": selected,
    }


def _select_hybrid(base, student, committees, labels):
    rows = []
    best = None
    for committee_name in (
        "uniform",
        "global_simplex",
        "region_simplex",
        "selected",
    ):
        for beta in (0.0, 0.25, 0.50, 0.75, 1.0):
            prediction = (
                float(beta) * committees[committee_name]
                + (1.0 - float(beta)) * student
            )
            stats = selection_stats(base, prediction, labels)
            row = {
                "committee": committee_name,
                "beta": float(beta),
                "objective": float(stats["mae"] + 0.02 * stats["harm_over_010_rate"]),
                **stats,
            }
            rows.append(row)
            if best is None or (row["objective"], row["mae"]) < (
                best["objective"], best["mae"]
            ):
                best = row
    return best, rows


def _evaluate_condition(
    condition: str,
    teacher_paths: Sequence[Path],
    init_index: int,
    student_model: ComplementarityResidualStudent,
    dataloaders,
    args,
    metrics_fn,
    committee_steps: int,
):
    split_teacher_predictions: Dict[str, List[torch.Tensor]] = {
        "valid": [],
        "test": [],
    }
    split_reference: Dict[str, Dict[str, object]] = {}
    teacher_metric_rows = []

    for teacher_index, path in enumerate(teacher_paths):
        LOGGER.info(
            "Condition=%s teacher=%d/%d %s",
            condition,
            teacher_index + 1,
            len(teacher_paths),
            path,
        )
        teacher = FrozenTeacherDLF(args).to(args.device)
        incompatible = teacher.load_checkpoint(path, map_location=args.device)
        LOGGER.info(
            "Loaded teacher (missing=%d unexpected=%d)",
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )
        teacher.freeze()
        for split_name in ("valid", "test"):
            collected = _collect_teacher(
                teacher,
                dataloaders[split_name],
                args.device,
                condition,
            )
            if split_name not in split_reference:
                split_reference[split_name] = collected
            else:
                _assert_alignment(
                    split_reference[split_name],
                    collected,
                    f"{condition}/{split_name}/teacher_{teacher_index}",
                )
            split_teacher_predictions[split_name].append(collected["prediction"])
            if split_name == "test":
                teacher_metric_rows.append({
                    "condition": condition,
                    "teacher_index": teacher_index,
                    "checkpoint": str(path),
                    **_metric_dict(metrics_fn, collected["prediction"], collected["labels"]),
                })
        del teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    teacher_splits = {}
    for split_name in ("valid", "test"):
        teacher_splits[split_name] = {
            "predictions": torch.stack(
                split_teacher_predictions[split_name], dim=1
            ),
            "labels": split_reference[split_name]["labels"],
            "sample_ids": split_reference[split_name]["sample_ids"],
        }

    student = {}
    for split_name in ("valid", "test"):
        student[split_name] = _collect_student(
            student_model,
            dataloaders[split_name],
            args.device,
            condition,
        )
        _assert_alignment(
            teacher_splits[split_name],
            student[split_name],
            f"{condition}/{split_name}/student",
        )

    valid_teacher = teacher_splits["valid"]
    test_teacher = teacher_splits["test"]
    valid_anchor = valid_teacher["predictions"][:, init_index]
    test_anchor = test_teacher["predictions"][:, init_index]

    fitted = fit_committee_cv(
        valid_teacher["predictions"],
        valid_teacher["labels"],
        valid_anchor,
        valid_teacher["sample_ids"],
        temperature=0.55,
        steps=int(committee_steps),
    )
    valid_committees = _committee_predictions(
        valid_teacher["predictions"], valid_anchor, fitted
    )
    test_committees = _committee_predictions(
        test_teacher["predictions"], test_anchor, fitted
    )

    student_policy, student_rows = _select_student_alpha(
        student["valid"]["base_prediction"],
        student["valid"]["prediction"],
        valid_teacher["labels"],
    )
    alpha = float(student_policy["alpha"])
    valid_student = student["valid"]["base_prediction"] + alpha * (
        student["valid"]["prediction"] - student["valid"]["base_prediction"]
    )
    test_student = student["test"]["base_prediction"] + alpha * (
        student["test"]["prediction"] - student["test"]["base_prediction"]
    )

    hybrid_policy, hybrid_rows = _select_hybrid(
        student["valid"]["base_prediction"],
        valid_student,
        valid_committees,
        valid_teacher["labels"],
    )
    beta = float(hybrid_policy["beta"])
    committee_name = str(hybrid_policy["committee"])
    test_hybrid = beta * test_committees[committee_name] + (1.0 - beta) * test_student

    named_predictions = {
        "anchor": test_anchor,
        "uniform": test_committees["uniform"],
        "global_simplex": test_committees["global_simplex"],
        "region_simplex": test_committees["region_simplex"],
        "student_calibrated": test_student,
        "hybrid_valid_selected": test_hybrid,
    }
    result_rows = []
    results = {}
    for model_name, prediction in named_predictions.items():
        metrics = _metric_dict(metrics_fn, prediction, test_teacher["labels"])
        results[model_name] = metrics
        result_rows.append({
            "condition": condition,
            "model": model_name,
            **metrics,
        })

    prediction_frame = {
        "sample_id": test_teacher["sample_ids"],
        "label": test_teacher["labels"].view(-1).tolist(),
        "hybrid_valid_selected": test_hybrid.view(-1).tolist(),
    }
    return {
        "condition": condition,
        "results": results,
        "result_rows": result_rows,
        "teacher_rows": teacher_metric_rows,
        "prediction_frame": prediction_frame,
        "committee": {
            "selected": fitted["selected"],
            "global_cv_score": fitted["global_cv_score"],
            "region_cv_score": fitted["region_cv_score"],
            "global_weights": fitted["global_weights"].tolist(),
            "region_weights": fitted["region_weights"].tolist(),
        },
        "student_policy": student_policy,
        "hybrid_policy": hybrid_policy,
        "student_rows": student_rows,
        "hybrid_rows": hybrid_rows,
        "cv_rows": fitted["cv_rows"],
    }


def main():
    cli = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    setup_seed(cli.seed)
    device = assign_gpu([cli.gpu])

    conditions = []
    for raw in cli.conditions:
        condition = _normalize_condition(raw)
        if condition not in conditions:
            conditions.append(condition)

    source_dir = Path(cli.source_run_dir)
    summary_path = (
        Path(cli.source_summary)
        if cli.source_summary
        else source_dir / "complementarity_v71_summary.json"
    )
    checkpoint_path = (
        Path(cli.source_checkpoint)
        if cli.source_checkpoint
        else source_dir / "complementarity_v71_best.pth"
    )
    cache_path = (
        Path(cli.teacher_cache)
        if cli.teacher_cache
        else source_dir / "complementarity_v71_teacher_cache.pth"
    )
    for path in (summary_path, checkpoint_path, cache_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    teacher_paths = [Path(value) for value in source_summary["teacher_paths"]]
    missing = [path for path in teacher_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing Teacher checkpoints:\n" + "\n".join(f"  - {path}" for path in missing)
        )
    init_index = int(source_summary["base_teacher_index"])

    args = get_config_regression("DLF", cli.dataset, Path(cli.config))
    args["device"] = device
    args["train_mode"] = "regression"
    args["feature_T"] = ""
    args["feature_A"] = ""
    args["feature_V"] = ""
    args["seed"] = cli.seed
    args["cur_seed"] = 1
    args["batch_size"] = cli.batch_size
    dataloaders = MMDataLoader(args, cli.num_workers)

    student_model = ComplementarityResidualStudent(
        args,
        hidden_dim=cli.hidden_dim,
        dropout=cli.dropout,
        residual_max=cli.residual_max,
        region_temperature=cli.region_temperature,
    ).to(device)
    best = _load_torch(checkpoint_path)
    state = best.get("state", best) if isinstance(best, dict) else best
    incompatible = student_model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "V7.1 Student checkpoint mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    student_model.eval()

    metrics_fn = MetricsTop("regression").getMetics(cli.dataset)
    save_dir = Path(cli.save_root) / cli.dataset / f"seed_{cli.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_teachers = []
    all_student_rows = []
    all_hybrid_rows = []
    all_cv_rows = []
    condition_summaries = {}

    for condition in conditions:
        result = _evaluate_condition(
            condition=condition,
            teacher_paths=teacher_paths,
            init_index=init_index,
            student_model=student_model,
            dataloaders=dataloaders,
            args=args,
            metrics_fn=metrics_fn,
            committee_steps=cli.committee_steps,
        )
        all_results.extend(result["result_rows"])
        all_teachers.extend(result["teacher_rows"])
        all_student_rows.extend(
            {"condition": condition, **row} for row in result["student_rows"]
        )
        all_hybrid_rows.extend(
            {"condition": condition, **row} for row in result["hybrid_rows"]
        )
        all_cv_rows.extend(
            {"condition": condition, **row} for row in result["cv_rows"]
        )
        pd.DataFrame(result["prediction_frame"]).to_csv(
            save_dir / f"v75_{condition}_test_predictions.csv", index=False
        )
        condition_summaries[condition] = {
            "results": result["results"],
            "committee": result["committee"],
            "student_policy": result["student_policy"],
            "hybrid_policy": result["hybrid_policy"],
        }
        selected = result["results"]["hybrid_valid_selected"]
        LOGGER.info(
            "Condition=%s final MAE=%.4f Acc7=%.4f Acc5=%.4f Corr=%.4f",
            condition,
            selected["MAE"],
            selected["acc_7"],
            selected["acc_5"],
            selected["Corr"],
        )

    pd.DataFrame(all_results).to_csv(
        save_dir / "v75_modality_comparison.csv", index=False
    )
    pd.DataFrame(all_teachers).to_csv(
        save_dir / "v75_single_teacher_metrics.csv", index=False
    )
    pd.DataFrame(all_student_rows).to_csv(
        save_dir / "v75_student_calibration.csv", index=False
    )
    pd.DataFrame(all_hybrid_rows).to_csv(
        save_dir / "v75_hybrid_calibration.csv", index=False
    )
    pd.DataFrame(all_cv_rows).to_csv(
        save_dir / "v75_committee_cv.csv", index=False
    )

    single_conditions = [value for value in ("t", "a", "v") if value in condition_summaries]
    best_single_by_mae = None
    if single_conditions:
        best_single_by_mae = min(
            single_conditions,
            key=lambda condition: condition_summaries[condition]["results"]
            ["hybrid_valid_selected"]["MAE"],
        )

    summary = {
        "method": "condition_calibrated_modality_ablation_v7_5",
        "protocol": (
            "The underlying V7.1 models are never retrained. Missing modalities are "
            "zeroed at inference. For each condition, committee weights, Student alpha "
            "and Hybrid beta are selected only on Valid under that same condition, then "
            "applied once to Test."
        ),
        "dataset": cli.dataset,
        "seed": cli.seed,
        "conditions": conditions,
        "best_single_modality_by_hybrid_mae": best_single_by_mae,
        "condition_summaries": condition_summaries,
        "source_run_dir": str(source_dir),
        "source_summary": str(summary_path),
        "source_checkpoint": str(checkpoint_path),
    }
    (save_dir / "modality_ablation_v75_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if "t" in condition_summaries:
        text_metrics = condition_summaries["t"]["results"]["hybrid_valid_selected"]
        LOGGER.info(
            "TEXT-ONLY: MAE=%.4f Acc7=%.4f Acc5=%.4f Acc2=%.4f F1=%.4f Corr=%.4f",
            text_metrics["MAE"],
            text_metrics["acc_7"],
            text_metrics["acc_5"],
            text_metrics["acc_2"],
            text_metrics["F1_score"],
            text_metrics["Corr"],
        )
    LOGGER.info("Best single modality by Hybrid MAE: %s", best_single_by_mae)


if __name__ == "__main__":
    main()
