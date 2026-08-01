"""Engineering audit for V9.12 single-holdout soft gating."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="result/holdout_soft_gate_v912/mosi/seed_1111",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main():
    cli = parse_args()
    root = Path(cli.root)
    pool_path = root / "holdout_expert_stack" / "holdout_expert_pool_v912.pth"
    summary_path = root / "holdout_soft_gate_v912_summary.json"
    test_path = root / "v912_test_summary.csv"
    prediction_path = root / "holdout_soft_gate_v912_predictions.csv"

    for path in (pool_path, summary_path, test_path, prediction_path):
        require(path.is_file(), f"missing V9.12 output: {path}")

    pool = torch.load(pool_path, map_location="cpu")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    provenance = pool.get("provenance", {})

    require(
        pool.get("version") == "v912_single_holdout_same_expert_stack",
        "unexpected V9.12 expert-pool version",
    )
    require(
        provenance.get("router_labels_used_for_expert_training") is False,
        "Router labels entered expert training",
    )
    require(
        provenance.get("router_groups_disjoint_from_expert_train") is True,
        "Router groups are not isolated from Expert-Train",
    )
    require(
        provenance.get("same_expert_models_router_valid_test") is True,
        "Router/Validation/Test do not share one unchanged expert stack",
    )
    require(
        provenance.get("official_validation_used_for_expert_training") is False,
        "official Validation entered expert training",
    )
    require(
        provenance.get("official_test_used_for_training_or_selection") is False,
        "Test entered training or selection",
    )
    require(
        provenance.get("deep_nested_coach_oof") is False,
        "V9.12 unexpectedly uses deep nested coach OOF",
    )

    expert_groups = set(pool.get("expert_train_groups", []))
    router_groups = set(pool.get("router_train_groups", []))
    require(expert_groups, "empty Expert-Train groups")
    require(router_groups, "empty Router-Train groups")
    require(
        not (expert_groups & router_groups),
        "Expert-Train and Router-Train group overlap",
    )

    for split in ("router_train", "valid", "test"):
        values = pool["splits"][split]
        n = len(values["labels"])
        require(n > 0, f"empty split: {split}")
        require(
            values["anchor"].shape == (n, 1),
            f"invalid Anchor shape in {split}",
        )
        require(
            values["function_space"].shape[0] == n,
            f"invalid function-space length in {split}",
        )
        for name in (
            "strong_negative",
            "boundary",
            "positive",
            "strong_positive",
        ):
            expert = values["experts"][name]
            require(
                expert["prediction"].shape == (n, 1),
                f"{split}/{name} prediction shape mismatch",
            )
            require(
                expert["signature"].shape == (n, 21),
                f"{split}/{name} signature shape mismatch",
            )
            require(
                bool(torch.isfinite(expert["signature"]).all()),
                f"{split}/{name} has non-finite signatures",
            )

    checkpoint_paths = [
        *pool.get("role_expert_checkpoints", {}).values(),
        *pool.get("tail_expert_checkpoints", {}).values(),
    ]
    require(len(checkpoint_paths) == 4, "expected four expert checkpoints")
    for value in checkpoint_paths:
        require(Path(value).is_file(), f"missing expert checkpoint: {value}")

    gate_checkpoints = summary.get("gate_ensemble_checkpoints", [])
    require(
        len(gate_checkpoints) == 3,
        f"expected three gate checkpoints, got {len(gate_checkpoints)}",
    )
    for value in gate_checkpoints:
        require(Path(value).is_file(), f"missing gate checkpoint: {value}")

    require(
        summary.get("same_expert_models_router_valid_test") is True,
        "summary lost same-stack guarantee",
    )
    require(
        summary.get("deep_nested_coach_oof") is False,
        "summary reports deep nested OOF",
    )
    require(
        summary.get("selected_by_validation_only") is True,
        "gate beta was not selected by Validation only",
    )
    require(
        float(summary["selected_beta"]) in (0.0, 0.25, 0.5, 0.75, 1.0),
        "selected beta is outside the preregistered grid",
    )
    require(
        summary["provenance"].get(
            "test_labels_used_for_training_or_selection"
        ) is False,
        "Test labels entered training or selection",
    )

    test = pd.read_csv(test_path)
    require(
        "holdout_soft_gate_valid_selected" in set(test["model"]),
        "deployable Test row missing",
    )
    require("anchor" in set(test["model"]), "Anchor Test row missing")

    print("ENGINEERING AUDIT PASSED")
    print("protocol: one fixed Router-Train holdout")
    print("source outer fold:", pool["outer_fold"])
    print("Expert-Train samples:", pool["expert_train_count"])
    print("Router-Train samples:", pool["router_train_count"])
    print("same experts Router/Validation/Test: True")
    print("deep nested coach OOF: False")
    print("gate selected epochs:", summary["gate_selected_epochs"])
    print("selected beta:", summary["selected_beta"])
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
    if summary["selected_beta"] == 0.0:
        print("WARNING: Validation selected Anchor fallback")
    elif summary["test_selected_gain"] <= 0.0:
        print("WARNING: selected holdout soft gate did not beat Anchor on Test")


if __name__ == "__main__":
    main()
