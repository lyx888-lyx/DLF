"""Engineering audit for V9.23 no-training gradient-conflict artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from trains.singleTask.gradient_conflict_audit_v923 import (
    AUDIT_VERSION,
    REGION_TASK_NAMES,
    TASK_NAMES,
)
from trains.singleTask.same_stack_expert_factory_v919 import sha256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v919-root",
        default="result/full_same_stack_nested_crossfit_v919/mosi/seed_1111",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(
    left: float,
    right: float,
    atol: float = 2e-6,
) -> bool:
    return abs(float(left) - float(right)) <= float(atol)


def as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def main():
    cli = parse_args()
    root = Path(cli.v919_root)
    output = (
        Path(cli.output_dir)
        if cli.output_dir
        else root / "v923_gradient_space_conflict_audit"
    )
    paths = {
        "summary": output / "v923_gradient_conflict_summary.json",
        "tasks": output / "v923_task_gradient_stats.csv",
        "pairwise": (
            output / "v923_pairwise_gradient_geometry.csv"
        ),
        "layerwise": (
            output / "v923_layerwise_gradient_geometry.csv"
        ),
        "consistency": output / "v923_pairwise_consistency.csv",
        "direction_tasks": (
            output / "v923_direction_task_effects.csv"
        ),
        "direction_summary": (
            output / "v923_direction_summary_by_fold.csv"
        ),
        "mgda": output / "v923_mgda_weights.csv",
        "folds": output / "v923_fold_summary.csv",
        "inventory": output / "v923_parameter_inventory.csv",
        "integrity": output / "v923_checkpoint_integrity.csv",
        "report": output / "v923_gradient_conflict_report.md",
    }
    for path in paths.values():
        require(path.is_file(), f"missing V9.23 artifact: {path}")

    summary = json.loads(
        paths["summary"].read_text(encoding="utf-8")
    )
    tasks = pd.read_csv(paths["tasks"])
    pairwise = pd.read_csv(paths["pairwise"])
    layerwise = pd.read_csv(paths["layerwise"])
    direction_tasks = pd.read_csv(paths["direction_tasks"])
    direction_summary = pd.read_csv(paths["direction_summary"])
    mgda = pd.read_csv(paths["mgda"])
    folds = pd.read_csv(paths["folds"])
    inventory = pd.read_csv(paths["inventory"])
    integrity = pd.read_csv(paths["integrity"])

    require(
        summary["version"] == AUDIT_VERSION,
        "summary version mismatch",
    )
    provenance = summary["provenance"]
    for key in (
        "autograd_grad_only",
        "frozen_v919_cfcompat_anchors_loaded",
        "development_partitions_only",
    ):
        require(
            provenance.get(key) is True,
            f"provenance flag not true: {key}",
        )
    for key in (
        "models_trained",
        "optimizer_created",
        "backward_called",
        "parameters_updated",
        "outer_holdout_samples_entered_gradient_computation",
        "outer_holdout_predictions_or_labels_loaded",
        "official_validation_loaded",
        "official_test_loaded",
        "router_or_fusion_head_used",
        "hyperparameters_selected_from_audit_results",
    ):
        require(
            provenance.get(key) is False,
            f"provenance flag not false: {key}",
        )

    expected_folds = set(range(int(cli.outer_folds)))
    require(
        set(folds["outer_fold"].astype(int)) == expected_folds,
        "fold set mismatch",
    )
    require(
        len(folds) == int(cli.outer_folds),
        "fold summary count mismatch",
    )
    require(
        set(tasks["outer_fold"].astype(int)) == expected_folds,
        "task fold mismatch",
    )

    for fold in expected_folds:
        local_tasks = tasks[tasks["outer_fold"] == fold]
        require(
            set(local_tasks["task"].astype(str)) == set(TASK_NAMES),
            "task names mismatch",
        )
        require(
            len(local_tasks) == len(TASK_NAMES),
            "duplicate task rows",
        )
        global_count = int(
            local_tasks[
                local_tasks["task"] == "global"
            ]["sample_count"].iloc[0]
        )
        region_count = int(
            local_tasks[
                local_tasks["task"].isin(REGION_TASK_NAMES)
            ]["sample_count"].sum()
        )
        require(
            global_count == region_count,
            "region counts do not sum to global",
        )
        require(
            bool((local_tasks["gradient_norm"] > 0.0).all()),
            "non-positive gradient norm",
        )

        local_pairwise = pairwise[
            pairwise["outer_fold"] == fold
        ]
        require(
            len(local_pairwise) == len(TASK_NAMES) ** 2,
            "pairwise matrix shape mismatch",
        )
        indexed = local_pairwise.set_index(
            ["left_task", "right_task"]
        )
        for left in TASK_NAMES:
            diagonal = indexed.loc[(left, left)]
            require(
                close(diagonal["cosine"], 1.0, 5e-5),
                "pairwise diagonal not one",
            )
            require(
                not as_bool(diagonal["conflict"]),
                "diagonal marked conflict",
            )
            for right in TASK_NAMES:
                first = indexed.loc[(left, right)]
                second = indexed.loc[(right, left)]
                require(
                    close(first["dot"], second["dot"], 2e-5),
                    "dot matrix not symmetric",
                )
                require(
                    close(
                        first["cosine"],
                        second["cosine"],
                        2e-5,
                    ),
                    "cosine matrix not symmetric",
                )
                require(
                    as_bool(first["conflict"])
                    == (float(first["dot"]) < 0.0),
                    "conflict flag disagrees with dot sign",
                )

        local_effects = direction_tasks[
            direction_tasks["outer_fold"] == fold
        ]
        local_summaries = direction_summary[
            direction_summary["outer_fold"] == fold
        ]
        for row in local_summaries.itertuples(index=False):
            effects = local_effects[
                local_effects["direction"] == row.direction
            ]
            require(
                len(effects) == len(TASK_NAMES),
                "direction task row count mismatch",
            )
            regions = effects[
                effects["task"].isin(REGION_TASK_NAMES)
            ]
            global_row = effects[
                effects["task"] == "global"
            ].iloc[0]
            improved = int(
                sum(
                    as_bool(value)
                    for value in regions["would_improve"]
                )
            )
            require(
                improved == int(row.region_improved_count),
                "region improvement count mismatch",
            )
            require(
                as_bool(row.all_regions_improve)
                == (improved == len(REGION_TASK_NAMES)),
                "all-regions flag mismatch",
            )
            require(
                as_bool(row.global_improves)
                == as_bool(global_row["would_improve"]),
                "global improvement flag mismatch",
            )
            require(
                close(
                    row.minimum_region_cosine,
                    regions["cosine"].min(),
                ),
                "minimum region cosine mismatch",
            )

        local_mgda = mgda[mgda["outer_fold"] == fold]
        require(
            set(local_mgda["task"].astype(str))
            == set(REGION_TASK_NAMES),
            "MGDA task set mismatch",
        )
        require(
            bool((local_mgda["weight"] >= -1e-8).all()),
            "negative MGDA weight",
        )
        require(
            close(local_mgda["weight"].sum(), 1.0, 1e-6),
            "MGDA weights do not sum to one",
        )

        fold_row = folds[
            folds["outer_fold"] == fold
        ].iloc[0]
        mgda_summary = local_summaries[
            local_summaries["direction"]
            == "mgda_normalized_regions"
        ].iloc[0]
        config = summary["config"]
        expected_feasible = bool(
            as_bool(mgda_summary["all_regions_improve"])
            and as_bool(mgda_summary["global_improves"])
            and float(mgda_summary["minimum_region_cosine"])
            >= float(config["common_cosine_margin"])
            and float(fold_row["mgda_direction_norm"])
            >= float(config["minimum_mgda_norm"])
        )
        require(
            as_bool(fold_row["fold_geometry_feasible"])
            == expected_feasible,
            "fold feasibility flag mismatch",
        )
        require(
            float(
                fold_row[
                    "global_region_decomposition_relative_error"
                ]
            )
            < 5e-5,
            "gradient decomposition error too large",
        )

    require(not layerwise.empty, "empty layerwise geometry")
    require(not inventory.empty, "empty parameter inventory")
    require(
        bool((inventory["parameter_elements"] > 0).all()),
        "empty parameter group",
    )

    for row in integrity.itertuples(index=False):
        path = Path(row.checkpoint)
        require(
            path.is_file(),
            f"missing audited checkpoint: {path}",
        )
        current = sha256(path)
        require(
            current == row.checkpoint_sha256_before,
            "checkpoint hash changed from before",
        )
        require(
            current == row.checkpoint_sha256_after,
            "checkpoint hash changed from after",
        )
        require(
            row.selected_parameter_sha256_before
            == row.selected_parameter_sha256_after,
            "selected parameter fingerprint changed",
        )
        require(
            as_bool(row.unchanged),
            "integrity row not marked unchanged",
        )

    feasible = int(
        sum(
            as_bool(value)
            for value in folds["fold_geometry_feasible"]
        )
    )
    require(
        feasible == int(summary["feasible_outer_folds"]),
        "feasible fold count mismatch",
    )
    expected_verdict = feasible >= int(
        summary["required_feasible_outer_folds"]
    )
    require(
        bool(summary["gradient_space_training_recommended"])
        == expected_verdict,
        "aggregate recommendation mismatch",
    )

    print(
        "V9.23 NO-TRAINING GRADIENT CONFLICT "
        "ENGINEERING AUDIT PASSED"
    )
    print("models trained: False")
    print("optimizer created: False")
    print("parameters updated: False")
    print("parameter scope:", summary["parameter_scope"])
    print(
        "feasible folds:",
        f"{feasible}/{int(cli.outer_folds)}",
    )
    print(
        "gradient-space training recommended:",
        summary["gradient_space_training_recommended"],
    )


if __name__ == "__main__":
    main()
