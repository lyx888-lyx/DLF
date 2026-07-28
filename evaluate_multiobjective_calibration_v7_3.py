"""Valid-only multi-objective calibration for MOSI V7.3.

This evaluator selects a single calibration policy on the validation split and
applies it once to the test split. Test metrics never participate in policy
selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import pandas as pd
import torch

from config import get_config_regression
from data_loader import MMDataLoader
from trains.singleTask.complementarity_system_v71 import (
    ComplementarityTrainerV71,
)
from trains.singleTask.model.CPFD_DLF import ComplementarityResidualStudent
from trains.utils import MetricsTop
from utils import assign_gpu, setup_seed


LOGGER = logging.getLogger("MMSA")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select an affine/hybrid calibration on Valid that improves Acc-7 "
            "and Acc-5 while constraining validation MAE, then evaluate exactly "
            "one frozen policy on Test."
        )
    )
    parser.add_argument("--dataset", choices=["mosi", "mosei"], default="mosi")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=str, default="./config/config.json")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="./result/complementarity_v71/mosi/seed_1111",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default="./result/multiobjective_calibration_v73",
    )
    parser.add_argument("--teacher-cache", type=str, default="")
    parser.add_argument("--best-checkpoint", type=str, default="")
    parser.add_argument("--summary", type=str, default="")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.18)
    parser.add_argument("--residual-max", type=float, default=0.35)
    parser.add_argument("--region-temperature", type=float, default=0.55)
    parser.add_argument("--committee-steps", type=int, default=600)

    parser.add_argument("--beta-step", type=float, default=0.05)
    parser.add_argument("--scale-min", type=float, default=0.90)
    parser.add_argument("--scale-max", type=float, default=1.15)
    parser.add_argument("--scale-step", type=float, default=0.01)
    parser.add_argument("--bias-min", type=float, default=-0.10)
    parser.add_argument("--bias-max", type=float, default=0.10)
    parser.add_argument("--bias-step", type=float, default=0.01)

    parser.add_argument(
        "--valid-mae-tolerance",
        type=float,
        default=0.0010,
        help=(
            "Maximum allowed validation MAE increase relative to the original "
            "V7.1 valid-selected hybrid."
        ),
    )
    parser.add_argument(
        "--fold-mae-tolerance",
        type=float,
        default=0.0100,
        help=(
            "Maximum allowed worst-fold validation MAE increase relative to "
            "the original hybrid."
        ),
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--acc7-weight", type=float, default=1.0)
    parser.add_argument("--acc5-weight", type=float, default=1.0)
    parser.add_argument("--stability-weight", type=float, default=0.25)
    parser.add_argument("--identity-penalty", type=float, default=0.002)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument(
        "--top-grid-rows",
        type=int,
        default=1000,
        help="Number of highest-ranked validation candidates saved to CSV.",
    )
    return parser.parse_args()


def _load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _float_grid(start: float, stop: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("Grid step must be positive.")
    if stop < start:
        raise ValueError("Grid stop must be >= start.")
    count = int(math.floor((stop - start) / step + 1e-9))
    values = [start + index * step for index in range(count + 1)]
    if not values or values[-1] < stop - 1e-9:
        values.append(stop)
    return [round(float(value), 10) for value in values]


def _metric_dict(metrics_fn, prediction, labels) -> Dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics_fn(
            prediction.detach().cpu(), labels.detach().cpu()
        ).items()
    }


def _acc_round(
    prediction: torch.Tensor,
    labels: torch.Tensor,
    minimum: float,
    maximum: float,
) -> float:
    pred = prediction.view(-1).clamp(minimum, maximum)
    truth = labels.view(-1).clamp(minimum, maximum)
    return float((torch.round(pred) == torch.round(truth)).float().mean().item())


def _fast_metrics(
    prediction: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    return {
        "mae": float(
            torch.abs(prediction.view(-1) - labels.view(-1)).mean().item()
        ),
        "acc_7": _acc_round(prediction, labels, -3.0, 3.0),
        "acc_5": _acc_round(prediction, labels, -2.0, 2.0),
    }


def _video_group_key(sample_id: object) -> str:
    if isinstance(sample_id, (tuple, list)) and sample_id:
        return str(sample_id[0])

    value = str(sample_id)
    for separator in ("$_$", "::", "##", "#"):
        if separator in value:
            return value.split(separator, 1)[0]

    match = re.match(r"^(.*?)[_-](\d+)$", value)
    if match and match.group(1):
        return match.group(1)
    return value


def _fold_assignments(
    sample_ids: Sequence[object],
    folds: int,
    seed: int,
) -> torch.Tensor:
    if folds < 2:
        raise ValueError("--folds must be at least 2.")
    assignments = []
    for value in sample_ids:
        key = f"{seed}|{_video_group_key(value)}"
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        assignments.append(int(digest, 16) % folds)
    result = torch.tensor(assignments, dtype=torch.long)
    if any(int((result == index).sum().item()) == 0 for index in range(folds)):
        raise RuntimeError(
            "At least one validation fold is empty. Reduce --folds."
        )
    return result


def _fold_summary(
    prediction: torch.Tensor,
    labels: torch.Tensor,
    assignments: torch.Tensor,
    folds: int,
) -> Dict[str, object]:
    maes = []
    acc7s = []
    acc5s = []
    for index in range(folds):
        mask = assignments == index
        metrics = _fast_metrics(prediction[mask], labels[mask])
        maes.append(metrics["mae"])
        acc7s.append(metrics["acc_7"])
        acc5s.append(metrics["acc_5"])
    return {
        "fold_maes": maes,
        "fold_acc7s": acc7s,
        "fold_acc5s": acc5s,
        "fold_mae_mean": float(sum(maes) / len(maes)),
        "fold_acc7_mean": float(sum(acc7s) / len(acc7s)),
        "fold_acc5_mean": float(sum(acc5s) / len(acc5s)),
        "fold_acc7_std": float(torch.tensor(acc7s).std(unbiased=False).item()),
        "fold_acc5_std": float(torch.tensor(acc5s).std(unbiased=False).item()),
    }


def _candidate_prediction(
    student_prediction: torch.Tensor,
    committee_prediction: torch.Tensor,
    beta: float,
    scale: float,
    bias: float,
) -> torch.Tensor:
    hybrid = float(beta) * committee_prediction + (
        1.0 - float(beta)
    ) * student_prediction
    return float(scale) * hybrid + float(bias)


def _select_policy(
    valid_student: torch.Tensor,
    committees: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    sample_ids: Sequence[object],
    reference_policy: Mapping[str, object],
    args,
) -> Tuple[Dict[str, object], List[Dict[str, object]], torch.Tensor]:
    reference_name = str(reference_policy["committee"])
    reference_beta = float(reference_policy["beta"])
    if reference_name not in committees:
        raise KeyError(
            f"Reference committee {reference_name!r} is unavailable."
        )

    reference_prediction = _candidate_prediction(
        valid_student,
        committees[reference_name],
        reference_beta,
        1.0,
        0.0,
    )
    reference_metrics = _fast_metrics(reference_prediction, labels)
    assignments = _fold_assignments(sample_ids, args.folds, args.seed)
    reference_folds = _fold_summary(
        reference_prediction, labels, assignments, args.folds
    )

    mae_limit = reference_metrics["mae"] + float(args.valid_mae_tolerance)
    fold_mae_limits = [
        value + float(args.fold_mae_tolerance)
        for value in reference_folds["fold_maes"]
    ]

    rows: List[Dict[str, object]] = []
    feasible: List[Dict[str, object]] = []
    beta_values = _float_grid(0.0, 1.0, args.beta_step)
    scale_values = _float_grid(
        args.scale_min, args.scale_max, args.scale_step
    )
    bias_values = _float_grid(args.bias_min, args.bias_max, args.bias_step)

    source_names = (
        "uniform",
        "global_simplex",
        "region_simplex",
        "selected",
    )
    for committee_name in source_names:
        committee = committees[committee_name]
        for beta in beta_values:
            raw = float(beta) * committee + (
                1.0 - float(beta)
            ) * valid_student
            for scale in scale_values:
                scaled = float(scale) * raw
                for bias in bias_values:
                    prediction = scaled + float(bias)
                    metrics = _fast_metrics(prediction, labels)
                    folds = _fold_summary(
                        prediction, labels, assignments, args.folds
                    )
                    worst_fold_mae_delta = max(
                        folds["fold_maes"][index]
                        - reference_folds["fold_maes"][index]
                        for index in range(args.folds)
                    )
                    accuracy_score = (
                        float(args.acc7_weight) * metrics["acc_7"]
                        + float(args.acc5_weight) * metrics["acc_5"]
                    )
                    stability_cost = float(args.stability_weight) * (
                        folds["fold_acc7_std"] + folds["fold_acc5_std"]
                    )
                    identity_cost = float(args.identity_penalty) * (
                        abs(float(scale) - 1.0) + abs(float(bias))
                    )
                    score = accuracy_score - stability_cost - identity_cost
                    is_feasible = (
                        metrics["mae"] <= mae_limit + 1e-12
                        and all(
                            folds["fold_maes"][index]
                            <= fold_mae_limits[index] + 1e-12
                            for index in range(args.folds)
                        )
                    )
                    row = {
                        "committee": committee_name,
                        "beta": float(beta),
                        "scale": float(scale),
                        "bias": float(bias),
                        "score": float(score),
                        "accuracy_score": float(accuracy_score),
                        "stability_cost": float(stability_cost),
                        "identity_cost": float(identity_cost),
                        "feasible": bool(is_feasible),
                        "worst_fold_mae_delta": float(worst_fold_mae_delta),
                        **metrics,
                        "fold_mae_mean": folds["fold_mae_mean"],
                        "fold_acc7_mean": folds["fold_acc7_mean"],
                        "fold_acc5_mean": folds["fold_acc5_mean"],
                        "fold_acc7_std": folds["fold_acc7_std"],
                        "fold_acc5_std": folds["fold_acc5_std"],
                    }
                    rows.append(row)
                    if is_feasible:
                        feasible.append(row)

    if not feasible:
        raise RuntimeError(
            "No calibration candidate satisfies the validation MAE constraints. "
            "Increase --valid-mae-tolerance or --fold-mae-tolerance."
        )

    selected = max(
        feasible,
        key=lambda row: (
            row["score"],
            row["acc_7"] + row["acc_5"],
            min(row["acc_7"], row["acc_5"]),
            -row["mae"],
            -abs(row["scale"] - 1.0),
            -abs(row["bias"]),
        ),
    )
    selected = {
        **selected,
        "reference_committee": reference_name,
        "reference_beta": reference_beta,
        "reference_valid_mae": reference_metrics["mae"],
        "reference_valid_acc_7": reference_metrics["acc_7"],
        "reference_valid_acc_5": reference_metrics["acc_5"],
        "valid_mae_limit": mae_limit,
        "fold_mae_limits": fold_mae_limits,
    }
    return selected, rows, reference_prediction


def _bootstrap_comparison(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    labels: torch.Tensor,
    samples: int,
    seed: int,
) -> Dict[str, object]:
    if samples <= 0:
        return {}
    reference = reference.view(-1).cpu()
    candidate = candidate.view(-1).cpu()
    labels = labels.view(-1).cpu()
    generator = torch.Generator().manual_seed(int(seed))
    n = labels.numel()

    mae_gains = torch.empty(samples)
    acc7_gains = torch.empty(samples)
    acc5_gains = torch.empty(samples)
    for index in range(samples):
        draw = torch.randint(0, n, (n,), generator=generator)
        y = labels[draw]
        ref = reference[draw]
        cand = candidate[draw]
        ref_metrics = _fast_metrics(ref, y)
        cand_metrics = _fast_metrics(cand, y)
        mae_gains[index] = ref_metrics["mae"] - cand_metrics["mae"]
        acc7_gains[index] = cand_metrics["acc_7"] - ref_metrics["acc_7"]
        acc5_gains[index] = cand_metrics["acc_5"] - ref_metrics["acc_5"]

    def summarize(values: torch.Tensor) -> Dict[str, float]:
        ordered = values.sort().values
        low_index = max(0, int(math.floor(0.025 * samples)))
        high_index = min(samples - 1, int(math.ceil(0.975 * samples)) - 1)
        return {
            "mean": float(values.mean().item()),
            "ci95_low": float(ordered[low_index].item()),
            "ci95_high": float(ordered[high_index].item()),
            "probability_positive": float(
                (values > 0).float().mean().item()
            ),
        }

    return {
        "mae_gain_positive_is_better": summarize(mae_gains),
        "acc7_gain_positive_is_better": summarize(acc7_gains),
        "acc5_gain_positive_is_better": summarize(acc5_gains),
    }


def main():
    cli = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    setup_seed(cli.seed)
    device = assign_gpu([cli.gpu])

    run_dir = Path(cli.run_dir)
    cache_path = (
        Path(cli.teacher_cache)
        if cli.teacher_cache
        else run_dir / "complementarity_v71_teacher_cache.pth"
    )
    checkpoint_path = (
        Path(cli.best_checkpoint)
        if cli.best_checkpoint
        else run_dir / "complementarity_v71_best.pth"
    )
    summary_path = (
        Path(cli.summary)
        if cli.summary
        else run_dir / "complementarity_v71_summary.json"
    )
    for path in (cache_path, checkpoint_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    teacher_paths = [Path(value) for value in source_summary["teacher_paths"]]
    init_index = int(source_summary["base_teacher_index"])
    original_hybrid_policy = source_summary["hybrid_policy"]
    student_alpha = float(source_summary["student_policy"]["alpha"])

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
    cache = _load_torch(cache_path)
    best = _load_torch(checkpoint_path)

    model = ComplementarityResidualStudent(
        args,
        hidden_dim=cli.hidden_dim,
        dropout=cli.dropout,
        residual_max=cli.residual_max,
        region_temperature=cli.region_temperature,
    ).to(device)
    incompatible = model.load_state_dict(best["state"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "V7.1 checkpoint architecture mismatch. "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

    save_dir = Path(cli.save_root) / cli.dataset / f"seed_{cli.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_fn = MetricsTop("regression").getMetics(cli.dataset)
    trainer = ComplementarityTrainerV71(
        args=args,
        metrics_fn=metrics_fn,
        save_dir=save_dir,
        teacher_cache=cache,
        teacher_paths=teacher_paths,
        init_index=init_index,
        region_temperature=cli.region_temperature,
        committee_steps=cli.committee_steps,
        max_epochs=1,
    )

    valid = trainer.collect(model, dataloaders["valid"], "valid")
    test = trainer.collect(model, dataloaders["test"], "test")
    valid_committees = trainer._committee_predictions("valid")
    test_committees = trainer._committee_predictions("test")

    valid_student = valid["base_prediction"] + student_alpha * (
        valid["prediction"] - valid["base_prediction"]
    )
    test_student = test["base_prediction"] + student_alpha * (
        test["prediction"] - test["base_prediction"]
    )

    selected, grid_rows, reference_valid = _select_policy(
        valid_student,
        valid_committees,
        valid["labels"],
        valid["sample_ids"],
        original_hybrid_policy,
        cli,
    )

    reference_name = str(original_hybrid_policy["committee"])
    reference_beta = float(original_hybrid_policy["beta"])
    reference_test = _candidate_prediction(
        test_student,
        test_committees[reference_name],
        reference_beta,
        1.0,
        0.0,
    )
    calibrated_test = _candidate_prediction(
        test_student,
        test_committees[str(selected["committee"])],
        float(selected["beta"]),
        float(selected["scale"]),
        float(selected["bias"]),
    )

    named_predictions = {
        "anchor": test["base_prediction"],
        "global_simplex": test_committees["global_simplex"],
        "original_v71_hybrid": reference_test,
        "multiobjective_calibrated_v73": calibrated_test,
    }
    comparison_rows = []
    results = {}
    for name, prediction in named_predictions.items():
        metrics = _metric_dict(metrics_fn, prediction, test["labels"])
        results[name] = metrics
        comparison_rows.append({"model": name, **metrics})

    ranked_rows = sorted(
        grid_rows,
        key=lambda row: (
            not row["feasible"],
            -row["score"],
            row["mae"],
        ),
    )
    pd.DataFrame(ranked_rows[: max(1, cli.top_grid_rows)]).to_csv(
        save_dir / "v73_valid_calibration_top_grid.csv", index=False
    )
    pd.DataFrame(comparison_rows).to_csv(
        save_dir / "v73_test_comparison.csv", index=False
    )
    pd.DataFrame({
        "sample_id": test["sample_ids"],
        "label": test["labels"].view(-1).tolist(),
        "original_v71_hybrid": reference_test.view(-1).tolist(),
        "multiobjective_calibrated_v73": calibrated_test.view(-1).tolist(),
    }).to_csv(save_dir / "v73_test_predictions.csv", index=False)

    bootstrap = _bootstrap_comparison(
        reference_test,
        calibrated_test,
        test["labels"],
        cli.bootstrap_samples,
        cli.seed + 73,
    )
    summary = {
        "method": "valid_only_multiobjective_calibration_v7_3",
        "selection_protocol": (
            "All beta/scale/bias/source choices are selected exclusively on "
            "Valid. Test is evaluated once after freezing the selected policy."
        ),
        "dataset": cli.dataset,
        "seed": cli.seed,
        "source_run_dir": str(run_dir),
        "source_summary": str(summary_path),
        "student_alpha": student_alpha,
        "original_hybrid_policy": original_hybrid_policy,
        "selected_policy": selected,
        "valid_reference_metrics": _fast_metrics(
            reference_valid, valid["labels"]
        ),
        "valid_selected_metrics": {
            key: selected[key] for key in ("mae", "acc_7", "acc_5")
        },
        "test_results": results,
        "test_mae_below_070": bool(
            results["multiobjective_calibrated_v73"]["MAE"] < 0.70
        ),
        "bootstrap_vs_original_hybrid": bootstrap,
    }
    (save_dir / "multiobjective_calibration_v73_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    LOGGER.info(
        "V7.3 selected committee=%s beta=%.2f scale=%.2f bias=%+.2f "
        "Valid(MAE=%.4f Acc7=%.4f Acc5=%.4f) "
        "Test(MAE=%.4f Acc7=%.4f Acc5=%.4f)",
        selected["committee"],
        selected["beta"],
        selected["scale"],
        selected["bias"],
        selected["mae"],
        selected["acc_7"],
        selected["acc_5"],
        results["multiobjective_calibrated_v73"]["MAE"],
        results["multiobjective_calibrated_v73"]["acc_7"],
        results["multiobjective_calibrated_v73"]["acc_5"],
    )


if __name__ == "__main__":
    main()
