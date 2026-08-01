"""Run the V9.17 fixed expert-by-region capability audit."""

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
from trains.singleTask.fixed_expert_region_audit_v917 import (
    ACTION_NAMES,
    AUDIT_VERSION,
    DESIGNATED_ACTION_BY_REGION,
    REGION_NAMES,
    build_region_audit,
    normalize_expert_pool,
    specialization_rows,
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
        raise argparse.ArgumentTypeError("threshold grid cannot be empty")
    if any(item <= 0.0 for item in result):
        raise argparse.ArgumentTypeError("thresholds must be positive")
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "V9.17 fixed expert capability matrix on strict OOF Train and "
            "frozen deployment Validation/Test pools"
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
        "--v98-root",
        default="./result/strict_nested_frontier_coach_v98/mosi/seed_1111",
    )
    parser.add_argument("--strict-oof-pool", default="")
    parser.add_argument("--anchor-checkpoint", default="")
    parser.add_argument("--boundary-checkpoint", default="")
    parser.add_argument("--positive-checkpoint", default="")
    parser.add_argument("--strong-negative-checkpoint", default="")
    parser.add_argument("--strong-positive-checkpoint", default="")
    parser.add_argument(
        "--save-root",
        default="./result/fixed_expert_region_audit_v917",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--role-residual-max", type=float, default=0.45)
    parser.add_argument("--tail-residual-max", type=float, default=1.50)
    parser.add_argument(
        "--advantage-thresholds",
        type=parse_float_grid,
        default=parse_float_grid("0.05,0.10"),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
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
    v9_root = Path(str(cli.v9_root).replace("seed_1111", f"seed_{cli.seed}"))
    v92_root = Path(str(cli.v92_root).replace("seed_1111", f"seed_{cli.seed}"))
    v98_root = Path(str(cli.v98_root).replace("seed_1111", f"seed_{cli.seed}"))
    strict_pool = (
        Path(cli.strict_oof_pool)
        if cli.strict_oof_pool
        else v98_root
        / "strict_nested_frontier_pool"
        / "strict_nested_v93_frontier_pool_v98.pth"
    )
    expert_paths = default_expert_paths(v9_root, v92_root)
    overrides = {
        "boundary": cli.boundary_checkpoint,
        "positive": cli.positive_checkpoint,
        "strong_negative": cli.strong_negative_checkpoint,
        "strong_positive": cli.strong_positive_checkpoint,
    }
    for name, value in overrides.items():
        if value:
            expert_paths[name] = Path(value)
    anchor = (
        Path(cli.anchor_checkpoint)
        if cli.anchor_checkpoint
        else anchor_checkpoint_from_summary(v92_root)
    )
    all_paths = {
        "strict_oof_pool": strict_pool,
        "anchor": anchor,
        **expert_paths,
    }
    missing = [str(path) for path in all_paths.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            "V9.17 requires the completed V9.8 strict OOF pool and all frozen "
            f"deployment checkpoints. Missing: {missing}"
        )
    return strict_pool, anchor, expert_paths


def write_matrix(
    metrics: pd.DataFrame,
    split: str,
    value_column: str,
    output_path: Path,
):
    subset = metrics[
        (metrics["split"] == split) & (metrics["region"].isin(REGION_NAMES))
    ]
    pivot = subset.pivot(index="region", columns="expert", values=value_column)
    pivot = pivot.reindex(index=list(REGION_NAMES), columns=list(ACTION_NAMES))
    counts = (
        subset[["region", "sample_count"]]
        .drop_duplicates()
        .set_index("region")
        .reindex(list(REGION_NAMES))
    )
    result = counts.join(pivot).reset_index()
    result.to_csv(output_path, index=False)
    return result


def markdown_table(frame: pd.DataFrame, float_digits: int = 6) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(str(value) for value in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.{int(float_digits)}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def split_summary(
    split: str,
    metrics: pd.DataFrame,
    designated: pd.DataFrame,
    oracle: pd.DataFrame,
):
    global_rows = metrics[
        (metrics["split"] == split) & (metrics["region"] == "all")
    ]
    region_rows = metrics[
        (metrics["split"] == split) & (metrics["region"].isin(REGION_NAMES))
    ]
    best_by_region = {}
    for region in REGION_NAMES:
        local = region_rows[region_rows["region"] == region].sort_values(
            ["expert_mae", "expert_index"]
        )
        best = local.iloc[0]
        best_by_region[region] = {
            "expert": str(best["expert"]),
            "mae": float(best["expert_mae"]),
            "gain_vs_anchor": float(best["gain_vs_anchor"]),
            "win_rate": float(best["win_rate"]),
            "gain_ci_low": float(best["gain_ci_low"]),
            "gain_ci_high": float(best["gain_ci_high"]),
        }
    return {
        "sample_count": int(
            designated[designated["split"] == split]["sample_count"].sum()
        ),
        "global_metrics": {
            str(row["expert"]): {
                "mae": float(row["expert_mae"]),
                "gain_vs_anchor": float(row["gain_vs_anchor"]),
                "gain_ci_low": float(row["gain_ci_low"]),
                "gain_ci_high": float(row["gain_ci_high"]),
                "bootstrap_positive_rate": float(
                    row["bootstrap_positive_rate"]
                ),
                "win_rate": float(row["win_rate"]),
                "large_gain_rate_010": float(row["large_gain_rate_010"]),
                "large_harm_rate_010": float(row["large_harm_rate_010"]),
            }
            for _, row in global_rows.iterrows()
        },
        "designated_region_metrics": (
            designated[designated["split"] == split]
            .sort_values("region_index")
            .to_dict(orient="records")
        ),
        "best_expert_by_true_region": best_by_region,
        "sample_oracle_by_region": (
            oracle[oracle["split"] == split]
            .sort_values("region_index")
            .to_dict(orient="records")
        ),
    }


def main():
    cli = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    required_thresholds = {0.05, 0.10}
    configured_thresholds = {
        round(float(value), 2) for value in cli.advantage_thresholds
    }
    if not required_thresholds.issubset(configured_thresholds):
        raise ValueError(
            "V9.17 requires advantage thresholds 0.05 and 0.10 "
            "for the standard capability tables"
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
    args["batch_size"] = int(cli.batch_size)

    strict_pool_path, anchor_checkpoint, expert_paths = resolve_paths(cli)
    checkpoint_paths = {
        "anchor": anchor_checkpoint,
        **expert_paths,
        "strict_oof_pool": strict_pool_path,
    }
    checkpoint_hashes = {
        name: sha256(path) for name, path in checkpoint_paths.items()
    }

    save_dir = Path(cli.save_root) / cli.dataset / f"seed_{cli.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    valid_pool_path = save_dir / "valid_fixed_expert_pool_v917.pth"
    test_pool_path = save_dir / "test_fixed_expert_pool_v917.pth"
    metrics_path = save_dir / "v917_region_expert_metrics_long.csv"
    global_path = save_dir / "v917_global_expert_metrics.csv"
    designated_path = save_dir / "v917_designated_expert_summary.csv"
    test_designated_path = save_dir / "v917_test_designated_expert_table.csv"
    test_designated_md_path = save_dir / "v917_test_designated_expert_table.md"
    samples_path = save_dir / "v917_sample_level_advantages.csv"
    oracle_path = save_dir / "v917_region_oracle_summary.csv"
    specialization_path = save_dir / "v917_expert_specialization_summary.csv"
    summary_path = save_dir / "fixed_expert_region_audit_v917_summary.json"

    strict_pool = torch.load(strict_pool_path, map_location="cpu")
    strict_provenance = strict_pool.get("provenance", {})
    if strict_provenance.get("is_fully_nested_teacher_stack") is not True:
        raise ValueError("V9.17 Train audit requires the strict nested V9.8 pool")
    if strict_provenance.get("holdout_label_isolation") is not True:
        raise ValueError("strict OOF Train pool does not guarantee holdout isolation")

    datasets = {
        split: MMDataset(args, mode=split)
        for split in ("valid", "test")
    }
    if "seq_lens" in args:
        args["seq_lens"] = datasets["valid"].get_seq_len()

    frozen_pool = FrozenSemanticExpertPoolV99(
        args,
        anchor_checkpoint=anchor_checkpoint,
        expert_paths=expert_paths,
        role_residual_max=cli.role_residual_max,
        tail_residual_max=cli.tail_residual_max,
    )
    valid_pool = frozen_pool.collect(
        build_loader(datasets["valid"], cli.batch_size, cli.num_workers)
    )
    torch.save(valid_pool, valid_pool_path)
    test_pool = frozen_pool.collect(
        build_loader(datasets["test"], cli.batch_size, cli.num_workers)
    )
    torch.save(test_pool, test_pool_path)

    normalized = {
        "train_oof": normalize_expert_pool(
            strict_pool,
            "strict_nested_outer_fold_oof_shadow_experts",
        ),
        "valid": normalize_expert_pool(
            valid_pool,
            "frozen_full_train_deployment_experts",
        ),
        "test": normalize_expert_pool(
            test_pool,
            "frozen_full_train_deployment_experts",
        ),
    }
    split_names = list(normalized)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            overlap = set(normalized[left]["sample_ids"]) & set(
                normalized[right]["sample_ids"]
            )
            if overlap:
                raise RuntimeError(f"{left}/{right} sample ID overlap")

    metric_rows = []
    global_rows = []
    designated_rows = []
    sample_rows = []
    oracle_rows = []
    for offset, (split, pool) in enumerate(normalized.items()):
        result = build_region_audit(
            pool,
            split=split,
            thresholds=cli.advantage_thresholds,
            bootstrap_repeats=cli.bootstrap_repeats,
            bootstrap_seed=int(cli.seed) + 917000 + 10000 * offset,
        )
        metric_rows.extend(result["metrics_rows"])
        global_rows.extend(result["global_rows"])
        designated_rows.extend(result["designated_rows"])
        sample_rows.extend(result["sample_rows"])
        oracle_rows.extend(result["oracle_region_rows"])

    metrics = pd.DataFrame(metric_rows + global_rows)
    global_metrics = pd.DataFrame(global_rows)
    designated = pd.DataFrame(designated_rows)
    samples = pd.DataFrame(sample_rows)
    oracle = pd.DataFrame(oracle_rows)
    specialization = pd.DataFrame(
        specialization_rows(metric_rows, global_rows)
    )

    metrics.to_csv(metrics_path, index=False)
    global_metrics.to_csv(global_path, index=False)
    designated.to_csv(designated_path, index=False)
    samples.to_csv(samples_path, index=False)
    oracle.to_csv(oracle_path, index=False)
    specialization.to_csv(specialization_path, index=False)

    matrix_outputs = {}
    matrix_specs = {
        "mae": "expert_mae",
        "gain": "gain_vs_anchor",
        "win_rate": "win_rate",
        "large_gain_rate_010": "large_gain_rate_010",
        "large_harm_rate_010": "large_harm_rate_010",
        "oracle_best_rate": "oracle_best_rate",
    }
    for split in normalized:
        matrix_outputs[split] = {}
        for label, column in matrix_specs.items():
            path = save_dir / f"v917_{split}_{label}_matrix.csv"
            write_matrix(metrics, split, column, path)
            matrix_outputs[split][label] = str(path)

    test_designated = (
        designated[designated["split"] == "test"]
        .sort_values("region_index")
        .reset_index(drop=True)
    )
    test_designated.to_csv(test_designated_path, index=False)
    test_display_columns = [
        "region",
        "sample_count",
        "designated_expert",
        "anchor_mae",
        "designated_expert_mae",
        "improvement",
        "win_rate",
        "large_gain_rate_010",
        "large_harm_rate_010",
        "gain_ci_low",
        "gain_ci_high",
    ]
    test_designated_md_path.write_text(
        "# V9.17 Test designated-expert capability table\n\n"
        + markdown_table(test_designated[test_display_columns]),
        encoding="utf-8",
    )

    split_summaries = {
        split: split_summary(split, metrics, designated, oracle)
        for split in normalized
    }
    summary = {
        "version": AUDIT_VERSION,
        "method": "fixed_expert_true_region_capability_audit_v9_17",
        "dataset": cli.dataset,
        "seed": int(cli.seed),
        "action_names": list(ACTION_NAMES),
        "region_names": list(REGION_NAMES),
        "region_definition": {
            "strong_negative": "label < -1.5",
            "negative": "-1.5 <= label < -0.5",
            "boundary": "-0.5 <= label <= 0.5",
            "positive": "0.5 < label <= 1.5",
            "strong_positive": "label > 1.5",
        },
        "designated_action_by_region": DESIGNATED_ACTION_BY_REGION,
        "advantage_definition": (
            "abs(anchor_prediction-label) - abs(expert_prediction-label)"
        ),
        "advantage_thresholds": list(cli.advantage_thresholds),
        "bootstrap": {
            "unit": "conversation_group",
            "repeats": int(cli.bootstrap_repeats),
            "interval": "2.5% to 97.5% empirical quantiles",
        },
        "split_summaries": split_summaries,
        "checkpoint_paths": {
            name: str(path) for name, path in checkpoint_paths.items()
        },
        "checkpoint_sha256": checkpoint_hashes,
        "outputs": {
            "valid_pool": str(valid_pool_path),
            "test_pool": str(test_pool_path),
            "metrics_long": str(metrics_path),
            "global_metrics": str(global_path),
            "designated_summary": str(designated_path),
            "test_designated_table": str(test_designated_path),
            "test_designated_markdown": str(test_designated_md_path),
            "sample_advantages": str(samples_path),
            "region_oracle_summary": str(oracle_path),
            "expert_specialization_summary": str(specialization_path),
            "matrices": matrix_outputs,
        },
        "provenance": {
            "models_trained": False,
            "router_trained": False,
            "fusion_selected": False,
            "train_statistics_use_strict_oof_shadow_experts": True,
            "each_train_prediction_excludes_its_sample": True,
            "validation_uses_frozen_full_train_experts": True,
            "test_uses_frozen_full_train_experts": True,
            "test_labels_used_for_reporting_only": True,
            "test_labels_used_for_training_or_selection": False,
            "region_definitions_reused_from_v9_training": True,
            "sample_id_alignment_checked": True,
        },
    }
    summary_path.write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    logger.info("V9.17 fixed expert region audit complete: %s", save_dir)
    print("\nV9.17 TEST DESIGNATED-EXPERT TABLE")
    print(
        test_designated[
            [
                "region",
                "sample_count",
                "designated_expert",
                "anchor_mae",
                "designated_expert_mae",
                "improvement",
                "win_rate",
                "large_gain_rate_010",
                "large_harm_rate_010",
                "gain_ci_low",
                "gain_ci_high",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
