"""Engineering and scientific audit for V9.6."""

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
            / "distributional_target_nearest_expert_v96"
            / "mosi"
            / "seed_1111"
        ),
    )
    parser.add_argument(
        "--require-deployable-gain", type=float, default=None
    )
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
            root
            / "distributional_target_nearest_expert_v96_summary.json"
        ).read_text()
    )
    oof = pd.read_csv(
        require(root / "v96_target_oof_predictions.csv")
    )
    if len(oof) != 1284 or oof.sample_id.nunique() != 1284:
        raise RuntimeError(
            "V9.6 target OOF must contain 1284 unique samples"
        )
    quantiles = oof[
        ["q10", "q25", "q50", "q75", "q90"]
    ].to_numpy(dtype=float)
    if not np.isfinite(quantiles).all():
        raise RuntimeError(
            "V9.6 OOF quantiles contain non-finite values"
        )
    if not np.all(
        quantiles[:, 1:] >= quantiles[:, :-1] - 1e-7
    ):
        raise RuntimeError(
            "V9.6 OOF quantiles are not monotone"
        )
    checkpoint = torch.load(
        require(summary["target_coach"]["checkpoint"]),
        map_location="cpu",
    )
    if not isinstance(checkpoint.get("state_dict"), dict):
        raise RuntimeError(
            "invalid V9.6 target checkpoint"
        )
    calibration = pd.read_csv(
        require(root / "v96_route_calibration.csv")
    )
    if not {
        "anchor",
        "distributional_nearest_expert",
    }.issubset(set(calibration["mode"])):
        raise RuntimeError(
            "V9.6 calibration is incomplete"
        )
    predictions = pd.read_csv(
        require(
            root
            / "distributional_target_nearest_expert_v96_predictions.csv"
        )
    )
    test_summary = pd.read_csv(
        require(root / "v96_test_summary.csv")
    ).set_index("model")
    required_models = {
        "anchor",
        "target_median_direct",
        "nearest_expert_to_target_median_all",
        "minimum_quantile_risk_expert_all",
        "distributional_target_nearest_expert_valid_selected",
        "true_region_expert_policy",
        "sample_oracle_upper_bound",
    }
    if not required_models.issubset(test_summary.index):
        raise RuntimeError(
            "V9.6 test summary is incomplete"
        )
    if (
        len(predictions) != 686
        or predictions.sample_id.nunique() != 686
    ):
        raise RuntimeError(
            "V9.6 prediction file must contain 686 unique Test samples"
        )
    if set(summary.get("action_schema", [])) != {
        "anchor",
        "strong_negative",
        "boundary",
        "positive",
        "strong_positive",
    }:
        raise RuntimeError(
            "V9.6 action schema is invalid"
        )
    anchor = float(test_summary.loc["anchor", "MAE"])
    coach = float(
        test_summary.loc[
            "distributional_target_nearest_expert_valid_selected",
            "MAE",
        ]
    )
    gain = anchor - coach
    print("ENGINEERING AUDIT PASSED")
    print("target OOF samples:", len(oof))
    print(
        "target OOF metrics:",
        summary["target_coach"]["oof_metrics"],
    )
    print(
        "target Validation metrics:",
        summary["target_coach"]["valid_metrics"],
    )
    print(
        "validation-selected route:",
        summary["route_policy"],
    )
    print(
        "test activation rate:",
        summary["test_activation_rate"],
    )
    print(
        "test deployed actions:",
        summary["test_deployed_action_counts"],
    )
    print(f"test anchor MAE: {anchor:.6f}")
    print(
        "test target-median MAE:",
        float(
            test_summary.loc[
                "target_median_direct", "MAE"
            ]
        ),
    )
    print(f"test deployable coach MAE: {coach:.6f}")
    print(f"test deployable gain: {gain:+.6f}")
    print(
        "test true-region MAE:",
        float(
            test_summary.loc[
                "true_region_expert_policy", "MAE"
            ]
        ),
    )
    print(
        "test sample-oracle MAE:",
        float(
            test_summary.loc[
                "sample_oracle_upper_bound", "MAE"
            ]
        ),
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
            "WARNING: V9.6 deployable coach did not beat Anchor on Test"
        )


if __name__ == "__main__":
    main()
