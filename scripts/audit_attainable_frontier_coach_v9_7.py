"""Engineering and scientific audit for V9.7."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=str(
            REPO_ROOT
            / "result"
            / "attainable_frontier_coach_v97"
            / "mosi"
            / "seed_1111"
        ),
    )
    parser.add_argument("--require-deployable-gain", type=float, default=None)
    return parser.parse_args()


def require(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main():
    cli = parse_args()
    root = Path(cli.root)
    summary = json.loads(
        require(
            root / "attainable_frontier_coach_v97_summary.json"
        ).read_text(encoding="utf-8")
    )
    pool_path = require(
        root
        / "frontier_expert_pool"
        / "crossfit_v93_frontier_pool_v97.pth"
    )
    pool = torch.load(pool_path, map_location="cpu")
    required_pool = {
        "sample_ids",
        "group_ids",
        "labels",
        "anchor",
        "function_space",
        "fold_index",
        "expert_predictions",
        "expert_confidences",
        "oracle_value",
        "oracle_gain",
        "oracle_cost_margin",
        "provenance",
    }
    if not required_pool.issubset(pool):
        raise RuntimeError(
            f"V9.7 pool missing {sorted(required_pool - set(pool))}"
        )
    if (
        len(pool["sample_ids"]) != 1284
        or len(set(pool["sample_ids"])) != 1284
    ):
        raise RuntimeError(
            "V9.7 frontier pool must contain 1284 unique samples"
        )
    if pool["expert_predictions"].shape != (1284, 4, 1):
        raise RuntimeError("V9.7 expert prediction shape is invalid")
    for key in (
        "labels",
        "anchor",
        "function_space",
        "expert_predictions",
        "expert_confidences",
        "oracle_value",
        "oracle_gain",
        "oracle_cost_margin",
    ):
        if not torch.isfinite(pool[key]).all():
            raise RuntimeError(f"V9.7 pool tensor is non-finite: {key}")
    if not bool(pool["provenance"].get("holdout_label_isolation")):
        raise RuntimeError(
            "V9.7 pool does not assert holdout-label isolation"
        )

    coach_oof = pd.read_csv(
        require(root / "v97_frontier_coach_oof_predictions.csv")
    )
    if len(coach_oof) != 1284 or coach_oof.sample_id.nunique() != 1284:
        raise RuntimeError("V9.7 coach OOF predictions are incomplete")
    numeric = coach_oof.select_dtypes(
        include=[np.number]
    ).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("V9.7 coach OOF contains non-finite values")

    profiles = pd.read_csv(
        require(root / "v97_oof_policy_profiles.csv")
    )
    if set(profiles["profile"]) != {
        "conservative",
        "balanced",
        "broad",
    }:
        raise RuntimeError(
            "V9.7 must evaluate exactly three pre-registered profiles"
        )
    validation = pd.read_csv(
        require(root / "v97_validation_profile_selection.csv")
    )
    if len(validation) > 4 or "anchor" not in set(validation["profile"]):
        raise RuntimeError(
            "V9.7 Validation selection is not pre-registered"
        )

    predictions = pd.read_csv(
        require(root / "attainable_frontier_coach_v97_predictions.csv")
    )
    if len(predictions) != 686 or predictions.sample_id.nunique() != 686:
        raise RuntimeError(
            "V9.7 Test predictions must contain 686 unique samples"
        )
    test_summary = pd.read_csv(
        require(root / "v97_test_summary.csv")
    ).set_index("model")
    required_models = {
        "anchor",
        "frontier_value_direct",
        "nearest_candidate_to_frontier_all",
        "attainable_frontier_valid_selected",
        "true_region_expert_policy",
        "sample_oracle_upper_bound",
    }
    if not required_models.issubset(test_summary.index):
        raise RuntimeError("V9.7 Test summary is incomplete")

    anchor = float(test_summary.loc["anchor", "MAE"])
    coach = float(
        test_summary.loc["attainable_frontier_valid_selected", "MAE"]
    )
    gain = anchor - coach
    pool_summary = json.loads(
        require(
            root
            / "frontier_expert_pool"
            / "v97_crossfit_frontier_pool_summary.json"
        ).read_text(encoding="utf-8")
    )
    print("ENGINEERING AUDIT PASSED")
    print("frontier OOF samples:", len(pool["sample_ids"]))
    print("frontier OOF anchor MAE:", pool_summary["anchor_oof_mae"])
    print(
        "frontier OOF sample-oracle MAE:",
        pool_summary["sample_oracle_oof_mae"],
    )
    print(
        "frontier OOF action counts:",
        pool_summary["oracle_action_counts"],
    )
    print(
        "teacher stack fully nested:",
        pool["provenance"]["is_fully_nested_teacher_stack"],
    )
    print(
        "holdout-label isolation:",
        pool["provenance"]["holdout_label_isolation"],
    )
    print("coach OOF diagnostics:", summary["frontier_coach"])
    print("OOF policy profiles:")
    for row in summary["oof_policy_profiles"]:
        print(
            " ",
            row["profile"],
            "gain=%.6f" % row["oof_gain"],
            "lower=%.6f" % row["bootstrap_gain_lower"],
            "activation=%.4f" % row["activation_rate"],
            "eligible=%s" % row["oof_eligible"],
        )
    print("validation selected profile:", summary["selected_profile"])
    print("validation selected policy:", summary["selected_policy"])
    print("test activation rate:", summary["test_activation_rate"])
    print(
        "test proposed actions:",
        summary["test_proposed_action_counts"],
    )
    print(
        "test deployed actions:",
        summary["test_deployed_action_counts"],
    )
    print(f"test anchor MAE: {anchor:.6f}")
    print(f"test frontier coach MAE: {coach:.6f}")
    print(f"test deployable gain: {gain:+.6f}")
    print(
        "test true-region MAE:",
        float(test_summary.loc["true_region_expert_policy", "MAE"]),
    )
    print(
        "test sample-oracle MAE:",
        float(test_summary.loc["sample_oracle_upper_bound", "MAE"]),
    )
    if (
        cli.require_deployable_gain is not None
        and gain < cli.require_deployable_gain
    ):
        raise RuntimeError(
            f"deployable gain {gain:.6f} is below required "
            f"{cli.require_deployable_gain:.6f}"
        )
    if gain <= 0:
        print(
            "WARNING: V9.7 deployable coach did not beat Anchor on Test"
        )
    if not bool(
        pool["provenance"]["is_fully_nested_teacher_stack"]
    ):
        print(
            "NOTE: V9.7 excludes holdout labels from fold-local specialists, "
            "but reuses the frozen V9 teacher stack rather than nesting all "
            "teacher training."
        )


if __name__ == "__main__":
    main()
