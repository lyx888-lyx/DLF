"""Evaluate robust anchored teacher committees without training a student."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import pandas as pd
import torch

from config import get_config_regression
from data_loader import MMDataLoader
from train_complementarity_v7_1 import discover_teacher_paths
from trains.singleTask.complementarity_v71 import (
    REGION_NAMES,
    apply_global_committee,
    apply_region_committee,
    committee_dispersion,
    fit_committee_cv,
    region_index,
    selection_stats,
)
from trains.singleTask.function_consensus_system_v7 import build_teacher_cache
from trains.utils import MetricsTop
from utils import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit robust global/region teacher committees on Valid, then select "
            "a conservative anchor-to-committee shrinkage coefficient. No student "
            "training is performed."
        )
    )
    parser.add_argument("--dataset", choices=["mosi", "mosei"], default="mosei")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=str, default="./config/config.json")
    parser.add_argument(
        "--save-root",
        type=str,
        default="./result/anchored_committee_v72",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--teacher-checkpoint", action="append", default=[])
    parser.add_argument("--teacher-glob", action="append", default=[])
    parser.add_argument("--max-teachers", type=int, default=9)
    parser.add_argument("--teacher-cache", type=str, default="")
    parser.add_argument("--rebuild-teacher-cache", action="store_true")
    parser.add_argument("--region-temperature", type=float, default=0.55)
    parser.add_argument("--committee-steps", type=int, default=600)
    parser.add_argument("--beta-step", type=float, default=0.05)
    parser.add_argument("--harm-weight", type=float, default=0.02)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def _metrics(metrics_fn, prediction, labels):
    return {
        key: float(value)
        for key, value in metrics_fn(prediction.cpu(), labels.cpu()).items()
    }


def _committee_predictions(split, fitted, init_index, temperature):
    predictions = split["predictions"].float()
    anchor = predictions[:, init_index]
    uniform = predictions.mean(dim=1)
    global_simplex = apply_global_committee(
        predictions, fitted["global_weights"]
    )
    region_simplex = apply_region_committee(
        predictions,
        anchor,
        fitted["region_weights"],
        temperature,
    )
    selected = (
        region_simplex
        if fitted["selected"] == "region_simplex"
        else global_simplex
    )
    return {
        "anchor": anchor,
        "uniform": uniform,
        "global_simplex": global_simplex,
        "region_simplex": region_simplex,
        "selected": selected,
        "dispersion": committee_dispersion(predictions),
    }


def _beta_grid(step):
    if not 0.0 < step <= 1.0:
        raise ValueError("--beta-step must be in (0, 1].")
    count = int(round(1.0 / step))
    values = [min(1.0, index * step) for index in range(count + 1)]
    if values[-1] < 1.0 - 1e-8:
        values.append(1.0)
    return sorted(set(round(value, 10) for value in values))


def _calibrate_shrinkage(valid, labels, beta_step, harm_weight):
    rows = []
    best = None
    anchor = valid["anchor"]
    for committee_name in (
        "uniform",
        "global_simplex",
        "region_simplex",
        "selected",
    ):
        committee = valid[committee_name]
        for beta in _beta_grid(beta_step):
            prediction = anchor + float(beta) * (committee - anchor)
            stats = selection_stats(anchor, prediction, labels)
            objective = stats["mae"] + float(harm_weight) * stats[
                "harm_over_010_rate"
            ]
            row = {
                "committee": committee_name,
                "beta": float(beta),
                "objective": float(objective),
                **stats,
            }
            rows.append(row)
            if best is None or (
                row["objective"], row["mae"], row["beta"]
            ) < (best["objective"], best["mae"], best["beta"]):
                best = row
    return best, rows


def _bootstrap_gain(
    reference,
    candidate,
    labels,
    samples,
    seed,
):
    if samples <= 0:
        return None
    generator = torch.Generator().manual_seed(int(seed))
    n = labels.size(0)
    gains = torch.empty(samples, dtype=torch.float32)
    reference_error = torch.abs(reference - labels).view(-1)
    candidate_error = torch.abs(candidate - labels).view(-1)
    per_sample_gain = reference_error - candidate_error
    for index in range(samples):
        draw = torch.randint(0, n, (n,), generator=generator)
        gains[index] = per_sample_gain[draw].mean()
    ordered = gains.sort().values
    low = ordered[max(0, int(0.025 * samples) - 1)]
    high = ordered[min(samples - 1, int(0.975 * samples))]
    return {
        "mean_gain": float(gains.mean().item()),
        "ci95_low": float(low.item()),
        "ci95_high": float(high.item()),
        "probability_gain_positive": float((gains > 0).float().mean().item()),
    }


def _region_rows(name, anchor, prediction, labels):
    regions = region_index(labels)
    rows = []
    for index, region_name in enumerate(REGION_NAMES):
        mask = regions == index
        if mask.any():
            rows.append({
                "model": name,
                "region": region_name,
                "count": int(mask.sum().item()),
                "anchor_mae": float(
                    torch.abs(anchor[mask] - labels[mask]).mean().item()
                ),
                "final_mae": float(
                    torch.abs(prediction[mask] - labels[mask]).mean().item()
                ),
                "anchor_bias": float(
                    (labels[mask] - anchor[mask]).mean().item()
                ),
                "final_bias": float(
                    (labels[mask] - prediction[mask]).mean().item()
                ),
            })
    return rows


def main():
    cli = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("MMSA")
    setup_seed(cli.seed)
    device = assign_gpu([cli.gpu])

    args = get_config_regression("DLF", cli.dataset, Path(cli.config))
    args["device"] = device
    args["train_mode"] = "regression"
    args["feature_T"] = ""
    args["feature_A"] = ""
    args["feature_V"] = ""
    args["seed"] = cli.seed
    args["cur_seed"] = 1
    args["batch_size"] = cli.batch_size

    save_dir = Path(cli.save_root) / cli.dataset / f"seed_{cli.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    teacher_paths = discover_teacher_paths(cli)
    logger.info("Anchored committee teachers (%d): %s", len(teacher_paths), teacher_paths)

    dataloaders = MMDataLoader(args, cli.num_workers)
    cache_path = (
        Path(cli.teacher_cache)
        if cli.teacher_cache
        else save_dir / "anchored_committee_teacher_cache.pth"
    )
    cache = build_teacher_cache(
        args,
        dataloaders,
        teacher_paths,
        device,
        cache_path,
        rebuild=cli.rebuild_teacher_cache,
    )

    valid_split = cache["splits"]["valid"]
    valid_maes = torch.abs(
        valid_split["predictions"] - valid_split["labels"].unsqueeze(1)
    ).mean(dim=(0, 2))
    init_index = int(torch.argmin(valid_maes).item())

    fitted = fit_committee_cv(
        valid_split["predictions"],
        valid_split["labels"],
        valid_split["predictions"][:, init_index],
        valid_split["sample_ids"],
        temperature=cli.region_temperature,
        steps=cli.committee_steps,
    )

    valid = _committee_predictions(
        valid_split, fitted, init_index, cli.region_temperature
    )
    policy, policy_rows = _calibrate_shrinkage(
        valid,
        valid_split["labels"],
        cli.beta_step,
        cli.harm_weight,
    )

    test_split = cache["splits"]["test"]
    test = _committee_predictions(
        test_split, fitted, init_index, cli.region_temperature
    )
    final = test["anchor"] + float(policy["beta"]) * (
        test[policy["committee"]] - test["anchor"]
    )

    named: Dict[str, torch.Tensor] = {
        "anchor": test["anchor"],
        "committee_uniform": test["uniform"],
        "committee_global_simplex": test["global_simplex"],
        "committee_region_simplex": test["region_simplex"],
        "committee_cv_selected": test["selected"],
        "anchored_shrinkage_valid_selected": final,
    }
    metrics_fn = MetricsTop("regression").getMetics(cli.dataset)
    rows = []
    regions = []
    results = {}
    for name, prediction in named.items():
        metric = _metrics(metrics_fn, prediction, test_split["labels"])
        stats = selection_stats(
            test["anchor"], prediction, test_split["labels"]
        )
        results[name] = {"metrics": metric, "selection": stats}
        rows.append({"model": name, **metric, **stats})
        regions.extend(
            _region_rows(
                name,
                test["anchor"],
                prediction,
                test_split["labels"],
            )
        )

    bootstrap = {
        "final_vs_anchor": _bootstrap_gain(
            test["anchor"], final, test_split["labels"],
            cli.bootstrap_samples, cli.seed + 11,
        ),
        "final_vs_global_simplex": _bootstrap_gain(
            test["global_simplex"], final, test_split["labels"],
            cli.bootstrap_samples, cli.seed + 17,
        ),
        "final_vs_region_simplex": _bootstrap_gain(
            test["region_simplex"], final, test_split["labels"],
            cli.bootstrap_samples, cli.seed + 23,
        ),
    }

    pd.DataFrame(fitted["cv_rows"]).to_csv(
        save_dir / "v72_committee_cv.csv", index=False
    )
    pd.DataFrame(policy_rows).to_csv(
        save_dir / "v72_shrinkage_calibration.csv", index=False
    )
    pd.DataFrame(rows).to_csv(
        save_dir / "v72_test_comparison.csv", index=False
    )
    pd.DataFrame(regions).to_csv(
        save_dir / "v72_test_region_diagnostics.csv", index=False
    )

    prediction_frame = {
        "sample_id": test_split["sample_ids"],
        "label": test_split["labels"].view(-1).tolist(),
    }
    for name, prediction in named.items():
        prediction_frame[name] = prediction.view(-1).tolist()
    pd.DataFrame(prediction_frame).to_csv(
        save_dir / "anchored_committee_v72_predictions.csv", index=False
    )

    summary = {
        "method": "anchored_complementarity_shrinkage_v7_2",
        "dataset": cli.dataset,
        "seed": cli.seed,
        "teacher_count": len(teacher_paths),
        "teacher_paths": [str(value) for value in teacher_paths],
        "base_teacher_index": init_index,
        "base_teacher_valid_mae": float(valid_maes[init_index].item()),
        "committee": {
            "selected": fitted["selected"],
            "global_weights": fitted["global_weights"].tolist(),
            "region_weights": fitted["region_weights"].tolist(),
            "selected_regularization": fitted["selected_regularization"],
            "global_cv_score": fitted["global_cv_score"],
            "region_cv_score": fitted["region_cv_score"],
        },
        "shrinkage_policy": policy,
        "results": results,
        "bootstrap": bootstrap,
    }
    (save_dir / "anchored_committee_v72_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info(
        "Anchored committee selected=%s beta=%.2f test_MAE=%.6f",
        policy["committee"],
        policy["beta"],
        results["anchored_shrinkage_valid_selected"]["metrics"]["MAE"],
    )


if __name__ == "__main__":
    main()
