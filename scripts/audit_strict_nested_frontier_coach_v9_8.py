"""Engineering and scientific audit for V9.8 strict nested frontier coaching."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
POOL_VERSION = "v98_strict_nested_v93_architecture_frontier_pool"
ACTION_NAMES = (
    "anchor",
    "strong_negative",
    "boundary",
    "positive",
    "strong_positive",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=str(
            REPO_ROOT
            / "result"
            / "strict_nested_frontier_coach_v98"
            / "mosi"
            / "seed_1111"
        ),
    )
    parser.add_argument(
        "--minimum-plausible-oof-oracle-mae", type=float, default=0.20
    )
    parser.add_argument("--require-deployable-gain", type=float, default=None)
    return parser.parse_args()


def require(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    cli = parse_args()
    root = Path(cli.root)
    pool_root = root / "strict_nested_frontier_pool"
    pool_path = require(
        pool_root / "strict_nested_v93_frontier_pool_v98.pth"
    )
    pool = torch.load(pool_path, map_location="cpu")
    pool_summary = json.loads(
        require(
            pool_root / "v98_strict_nested_frontier_pool_summary.json"
        ).read_text()
    )
    summary = json.loads(
        require(
            root / "strict_nested_frontier_coach_v98_summary.json"
        ).read_text()
    )
    oof_frame = pd.read_csv(
        require(pool_root / "v98_strict_nested_frontier_pool.csv")
    )
    test_summary = pd.read_csv(
        require(root / "v98_test_summary.csv")
    ).set_index("model")
    predictions = pd.read_csv(
        require(root / "strict_nested_frontier_coach_v98_predictions.csv")
    )

    if pool.get("version") != POOL_VERSION:
        raise RuntimeError(
            f"unexpected V9.8 pool version: {pool.get('version')}"
        )
    provenance = pool.get("provenance", {})
    if provenance.get("is_fully_nested_teacher_stack") is not True:
        raise RuntimeError("V9.8 pool is not marked fully nested")
    if provenance.get("holdout_label_isolation") is not True:
        raise RuntimeError(
            "V9.8 pool does not guarantee holdout-label isolation"
        )
    if provenance.get("historical_full_train_teacher_reuse") is not False:
        raise RuntimeError("V9.8 reused a historical full-Train teacher")

    if len(oof_frame) != 1284 or oof_frame.sample_id.nunique() != 1284:
        raise RuntimeError(
            "V9.8 frontier OOF must contain 1284 unique samples"
        )
    if len(predictions) != 686 or predictions.sample_id.nunique() != 686:
        raise RuntimeError(
            "V9.8 Test predictions must contain 686 unique samples"
        )
    required_models = {
        "anchor",
        "attainable_frontier_valid_selected",
        "true_region_expert_policy",
        "sample_oracle_upper_bound",
    }
    if not required_models.issubset(test_summary.index):
        raise RuntimeError("V9.8 Test summary is incomplete")

    tensors = (
        pool["anchor"],
        pool["function_space"],
        pool["expert_predictions"],
        pool["expert_confidences"],
        pool["labels"],
    )
    if any(not torch.isfinite(value).all() for value in tensors):
        raise RuntimeError("V9.8 pool contains non-finite tensors")

    all_holdout_groups = []
    for row in pool.get("fold_metadata", []):
        fold = int(row["outer_fold"])
        inner_train = set(row["inner_train_groups"])
        inner_valid = set(row["inner_valid_groups"])
        holdout = set(row["holdout_groups"])
        if (
            inner_train & inner_valid
            or inner_train & holdout
            or inner_valid & holdout
        ):
            raise RuntimeError(f"group leakage in V9.8 outer fold {fold}")
        all_holdout_groups.extend(holdout)
        checkpoints = row["upstream_checkpoints"]
        hashes = row["checkpoint_sha256"]
        for name in ("clean", "moddrop", "cfcompat"):
            path = require(checkpoints[name])
            expected_fragment = f"outer_fold_{fold}"
            if expected_fragment not in str(path):
                raise RuntimeError(
                    f"fold {fold} {name} checkpoint is not fold-local: {path}"
                )
            if sha256(path) != hashes[name]:
                raise RuntimeError(
                    f"fold {fold} {name} checkpoint hash changed"
                )
    if len(all_holdout_groups) != len(set(all_holdout_groups)):
        raise RuntimeError(
            "a source-video group appears in multiple outer holdouts"
        )

    anchor_oof = float(pool_summary["anchor_oof_mae"])
    oracle_oof = float(pool_summary["sample_oracle_oof_mae"])
    if oracle_oof < float(cli.minimum_plausible_oof_oracle_mae):
        raise RuntimeError(
            "V9.8 OOF oracle is implausibly low "
            f"({oracle_oof:.6f}); inspect leakage before using this pool"
        )

    anchor_test = float(test_summary.loc["anchor", "MAE"])
    coach_test = float(
        test_summary.loc[
            "attainable_frontier_valid_selected", "MAE"
        ]
    )
    test_oracle = float(
        test_summary.loc["sample_oracle_upper_bound", "MAE"]
    )
    gain = anchor_test - coach_test
    oracle_gap = abs(test_oracle - oracle_oof)

    labels = pool["labels"].view(-1)
    expert_mae = {
        name: float(
            torch.abs(
                pool["expert_predictions"][:, index, 0] - labels
            )
            .mean()
            .item()
        )
        for index, name in enumerate(ACTION_NAMES[1:])
    }
    action_counts = {
        name: int(
            (pool["oracle_action_index"].view(-1) == index)
            .sum()
            .item()
        )
        for index, name in enumerate(ACTION_NAMES)
    }

    print("ENGINEERING AUDIT PASSED")
    print("strict nested frontier OOF samples:", len(oof_frame))
    print(
        "teacher stack fully nested:",
        provenance["is_fully_nested_teacher_stack"],
    )
    print(
        "historical full-Train teacher reuse:",
        provenance["historical_full_train_teacher_reuse"],
    )
    print(
        "holdout-label isolation:",
        provenance["holdout_label_isolation"],
    )
    print(f"frontier OOF anchor MAE: {anchor_oof:.6f}")
    print(f"frontier OOF sample-oracle MAE: {oracle_oof:.6f}")
    print("frontier OOF expert global MAE:", expert_mae)
    print("frontier OOF oracle action counts:", action_counts)
    print("coach OOF diagnostics:", summary.get("frontier_coach"))
    print(
        "validation selected profile:",
        summary.get("selected_profile"),
    )
    print(
        "validation selected policy:",
        summary.get("selected_policy"),
    )
    print("test activation rate:", summary.get("test_activation_rate"))
    print(
        "test proposed actions:",
        summary.get("test_proposed_action_counts"),
    )
    print(
        "test deployed actions:",
        summary.get("test_deployed_action_counts"),
    )
    print(f"test anchor MAE: {anchor_test:.6f}")
    print(f"test strict frontier coach MAE: {coach_test:.6f}")
    print(f"test deployable gain: {gain:+.6f}")
    print(f"test sample-oracle MAE: {test_oracle:.6f}")
    print(f"OOF/Test oracle absolute gap: {oracle_gap:.6f}")
    if oracle_gap > 0.25:
        print(
            "WARNING: OOF/Test oracle gap remains large; "
            "shadow/final expert mismatch is substantial"
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
            "WARNING: V9.8 deployable coach did not beat Anchor on Test"
        )


if __name__ == "__main__":
    main()
