"""Run V9.13 validation-only fusion on the frozen full deployment expert pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Mapping

import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.model.SemanticCostCoachV99 import ACTION_NAMES, SPECIALIST_NAMES
from trains.singleTask.position_aware_residual_fusion_v913 import (
    FUSION_VERSION,
    apply_config,
    evaluate_prediction,
    fit_validation_fusion,
    pool_tensors,
    selected_family_configs,
)
from trains.singleTask.semantic_expert_pool_v99 import (
    FrozenSemanticExpertPoolV99,
    anchor_checkpoint_from_summary,
    default_expert_paths,
)
from utils import assign_gpu, setup_seed

logger = logging.getLogger("MMSA")


def parse_float_grid(value: str):
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("grid cannot be empty")
    return result


def parse_str_grid(value: str):
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("grid cannot be empty")
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "V9.13 frozen deployment experts plus Validation-only global and "
            "position-aware residual fusion"
        )
    )
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", default="./config/config.json")
    parser.add_argument(
        "--v9-root",
        default="./result/role_conditioned_experts_v9/mosi/seed_1111",
    )
    parser.add_argument(
        "--v92-root",
        default="./result/oof_tail_residual_experts_v92/mosi/seed_1111",
    )
    parser.add_argument(
        "--save-root",
        default="./result/position_aware_residual_fusion_v913",
    )
    parser.add_argument("--anchor-checkpoint", default="")
    parser.add_argument("--boundary-checkpoint", default="")
    parser.add_argument("--positive-checkpoint", default="")
    parser.add_argument("--strong-negative-checkpoint", default="")
    parser.add_argument("--strong-positive-checkpoint", default="")
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--role-residual-max", type=float, default=0.45)
    parser.add_argument("--tail-residual-max", type=float, default=1.50)

    parser.add_argument(
        "--beta-grid",
        type=parse_float_grid,
        default=parse_float_grid(
            "0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,"
            "0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00"
        ),
    )
    parser.add_argument("--simplex-step", type=float, default=0.05)
    parser.add_argument(
        "--position-sources",
        type=parse_str_grid,
        default=parse_str_grid("anchor,median"),
    )
    parser.add_argument(
        "--tau-grid",
        type=parse_float_grid,
        default=parse_float_grid("0.25,0.50,0.75,1.00,1.50"),
    )
    parser.add_argument(
        "--rho-grid",
        type=parse_float_grid,
        default=parse_float_grid("0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50"),
    )
    parser.add_argument(
        "--confidence-powers",
        type=parse_float_grid,
        default=parse_float_grid("0.0,1.0"),
    )
    parser.add_argument("--minimum-validation-gain", type=float, default=0.0005)
    parser.add_argument("--maximum-validation-harm", type=float, default=0.05)
    return parser.parse_args()


def build_loader(dataset, batch_size: int, num_workers: int):
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_paths(cli):
    v9_root = Path(str(cli.v9_root).replace("seed_1111", f"seed_{cli.seed}"))
    v92_root = Path(str(cli.v92_root).replace("seed_1111", f"seed_{cli.seed}"))
    experts = default_expert_paths(v9_root, v92_root)
    overrides = {
        "boundary": cli.boundary_checkpoint,
        "positive": cli.positive_checkpoint,
        "strong_negative": cli.strong_negative_checkpoint,
        "strong_positive": cli.strong_positive_checkpoint,
    }
    for name, value in overrides.items():
        if value:
            experts[name] = Path(value)
    anchor = (
        Path(cli.anchor_checkpoint)
        if cli.anchor_checkpoint
        else anchor_checkpoint_from_summary(v92_root)
    )
    all_paths = {"anchor": anchor, **experts}
    missing = [str(path) for path in all_paths.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            "V9.13 requires the frozen historical deployment checkpoints. "
            f"Missing: {missing}"
        )
    return anchor, experts


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


def row_for_csv(row):
    result = {key: value for key, value in row.items() if key != "config"}
    result["config_json"] = json.dumps(row["config"], sort_keys=True)
    return result


def metrics_for_config(pool, config):
    values = pool_tensors(pool)
    prediction, extras = apply_config(pool, config)
    metrics = evaluate_prediction(prediction, values["actions"], values["labels"])
    return prediction, extras, metrics


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
    args["batch_size"] = int(cli.feature_batch_size)

    anchor_checkpoint, expert_paths = resolve_paths(cli)
    checkpoint_paths = {"anchor": anchor_checkpoint, **expert_paths}
    checkpoint_hashes = {
        name: sha256(path) for name, path in checkpoint_paths.items()
    }

    datasets = {
        split: MMDataset(args, mode=split)
        for split in ("train", "valid", "test")
    }
    if "seq_lens" in args:
        args["seq_lens"] = datasets["train"].get_seq_len()

    save_dir = Path(cli.save_root) / cli.dataset / f"seed_{cli.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    valid_pool_path = save_dir / "valid_expert_pool_v913.pth"
    test_pool_path = save_dir / "test_expert_pool_v913.pth"
    candidates_path = save_dir / "v913_validation_candidates.csv"
    selection_path = save_dir / "v913_validation_selection.json"
    test_summary_path = save_dir / "v913_test_summary.csv"
    prediction_path = save_dir / "v913_test_predictions.csv"
    summary_path = save_dir / "position_aware_residual_fusion_v913_summary.json"

    frozen_pool = FrozenSemanticExpertPoolV99(
        args,
        anchor_checkpoint=anchor_checkpoint,
        expert_paths=expert_paths,
        role_residual_max=cli.role_residual_max,
        tail_residual_max=cli.tail_residual_max,
    )

    # Selection stage: only the official Validation split is collected and used.
    valid_pool = frozen_pool.collect(
        build_loader(datasets["valid"], cli.feature_batch_size, cli.num_workers)
    )
    torch.save(valid_pool, valid_pool_path)
    fit = fit_validation_fusion(
        valid_pool,
        beta_grid=cli.beta_grid,
        simplex_step=cli.simplex_step,
        position_sources=cli.position_sources,
        tau_grid=cli.tau_grid,
        rho_grid=cli.rho_grid,
        confidence_powers=cli.confidence_powers,
        minimum_validation_gain=cli.minimum_validation_gain,
        maximum_validation_harm=cli.maximum_validation_harm,
    )
    pd.DataFrame([row_for_csv(row) for row in fit["candidate_rows"]]).to_csv(
        candidates_path, index=False
    )
    selection_payload = {
        "version": FUSION_VERSION,
        "selected_by_validation_only": True,
        "test_pool_collected": False,
        "selected": fit["selected"],
        "anchor": fit["anchor"],
        "best_single": fit["best_single"],
        "best_pairwise": fit["best_pairwise"],
        "best_pairwise_by_expert": fit["best_pairwise_by_expert"],
        "best_simplex": fit["best_simplex"],
        "best_position_aware": fit["best_position_aware"],
        "simplex_candidate_count": fit["simplex_candidate_count"],
        "minimum_validation_gain": fit["minimum_validation_gain"],
        "maximum_validation_harm": fit["maximum_validation_harm"],
        "checkpoint_sha256": checkpoint_hashes,
    }
    selection_path.write_text(
        json.dumps(jsonable(selection_payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info(
        "V9.13 Validation selected %s: MAE %.6f gain %+.6f harm %.4f",
        fit["selected"]["candidate_id"],
        fit["selected"]["mae"],
        fit["selected"]["gain_vs_anchor"],
        fit["selected"]["harm_over_010_rate"],
    )

    # Test is collected only after the complete selection artifact is written.
    test_pool = frozen_pool.collect(
        build_loader(datasets["test"], cli.feature_batch_size, cli.num_workers)
    )
    torch.save(test_pool, test_pool_path)
    valid_ids = set(valid_pool["sample_ids"])
    test_ids = set(test_pool["sample_ids"])
    if valid_ids & test_ids:
        raise RuntimeError("Validation/Test sample ID overlap")

    configs = selected_family_configs(fit)
    test_rows = []
    selected_prediction = None
    selected_extras = None
    for model_name, config in configs.items():
        _, _, valid_metrics = metrics_for_config(valid_pool, config)
        prediction, extras, test_metrics = metrics_for_config(test_pool, config)
        row = {
            "model": model_name,
            "config_json": json.dumps(config, sort_keys=True),
            **{f"validation_{key}": value for key, value in valid_metrics.items()},
            **{f"test_{key}": value for key, value in test_metrics.items()},
        }
        test_rows.append(row)
        if model_name == "validation_selected_deployable":
            selected_prediction = prediction
            selected_extras = extras
    pd.DataFrame(test_rows).to_csv(test_summary_path, index=False)
    if selected_prediction is None:
        raise RuntimeError("selected deployable prediction missing")

    test_values = pool_tensors(test_pool)
    prediction_frame = pd.DataFrame(
        {
            "sample_id": list(test_pool["sample_ids"]),
            "label": test_values["labels"].view(-1).numpy(),
            **{
                name: test_values["actions"][:, index].numpy()
                for index, name in enumerate(ACTION_NAMES)
            },
            "selected_prediction": selected_prediction.view(-1).numpy(),
        }
    )
    prediction_frame["anchor_abs_error"] = (
        prediction_frame["anchor"] - prediction_frame["label"]
    ).abs()
    prediction_frame["selected_abs_error"] = (
        prediction_frame["selected_prediction"] - prediction_frame["label"]
    ).abs()
    prediction_frame["selected_minus_anchor_abs_error"] = (
        prediction_frame["selected_abs_error"] - prediction_frame["anchor_abs_error"]
    )
    if "specialist_weights" in selected_extras:
        weights = selected_extras["specialist_weights"]
        for index, name in enumerate(SPECIALIST_NAMES):
            prediction_frame[f"weight_{name}"] = weights[:, index].numpy()
        prediction_frame["total_specialist_weight"] = weights.sum(dim=1).numpy()
    if "action_weights" in selected_extras:
        weights = selected_extras["action_weights"]
        for index, name in enumerate(ACTION_NAMES):
            prediction_frame[f"weight_{name}"] = weights[:, index].numpy()
    prediction_frame.to_csv(prediction_path, index=False)

    selected_test_row = next(
        row for row in test_rows if row["model"] == "validation_selected_deployable"
    )
    summary = {
        "version": FUSION_VERSION,
        "method": "frozen_expert_validation_only_low_dimensional_fusion_v9_13",
        "dataset": cli.dataset,
        "seed": int(cli.seed),
        "selected_candidate_id": fit["selected"]["candidate_id"],
        "selected_config": fit["selected"]["config"],
        "validation_anchor_mae": fit["anchor"]["mae"],
        "validation_selected_mae": fit["selected"]["mae"],
        "validation_selected_gain": fit["selected"]["gain_vs_anchor"],
        "validation_selected_harm_over_010_rate": fit["selected"][
            "harm_over_010_rate"
        ],
        "test_anchor_mae": selected_test_row["test_anchor_mae"],
        "test_selected_mae": selected_test_row["test_mae"],
        "test_selected_gain": selected_test_row["test_gain_vs_anchor"],
        "test_selected_harm_over_010_rate": selected_test_row[
            "test_harm_over_010_rate"
        ],
        "best_validation_single": fit["best_single"],
        "best_validation_pairwise": fit["best_pairwise"],
        "best_validation_simplex": fit["best_simplex"],
        "best_validation_position_aware": fit["best_position_aware"],
        "simplex_candidate_count": fit["simplex_candidate_count"],
        "checkpoint_paths": {name: str(path) for name, path in checkpoint_paths.items()},
        "checkpoint_sha256": checkpoint_hashes,
        "outputs": {
            "validation_pool": str(valid_pool_path),
            "test_pool": str(test_pool_path),
            "validation_candidates": str(candidates_path),
            "validation_selection": str(selection_path),
            "test_summary": str(test_summary_path),
            "test_predictions": str(prediction_path),
        },
        "provenance": {
            "experts_frozen": True,
            "neural_router_trained": False,
            "router_train_split_created": False,
            "official_validation_used_for_selection": True,
            "official_validation_used_for_expert_training": False,
            "test_labels_used_for_training_or_selection": False,
            "test_pool_collected_after_selection_artifact_written": True,
            "selected_by_validation_only": True,
            "sample_id_alignment_checked": True,
        },
    }
    summary_path.write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info(
        "V9.13 TEST anchor=%.6f selected=%.6f gain=%+.6f method=%s",
        summary["test_anchor_mae"],
        summary["test_selected_mae"],
        summary["test_selected_gain"],
        summary["selected_candidate_id"],
    )


if __name__ == "__main__":
    main()
