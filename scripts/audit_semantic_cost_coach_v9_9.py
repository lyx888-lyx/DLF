"""Engineering and scientific audit for V9.9 semantic cost coaching."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
POOL_VERSION = "v99_strict_nested_semantic_expert_pool"
SIGNATURE_VERSION = "semantic_expert_signature_v1"
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
            / "semantic_cost_coach_v99"
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
    pool_root = root / "strict_semantic_expert_pool"
    pool_path = require(pool_root / "strict_semantic_expert_pool_v99.pth")
    pool = torch.load(pool_path, map_location="cpu")
    pool_summary = json.loads(
        require(
            pool_root / "v99_strict_semantic_expert_pool_summary.json"
        ).read_text(encoding="utf-8")
    )
    summary = json.loads(
        require(root / "semantic_cost_coach_v99_summary.json").read_text(
            encoding="utf-8"
        )
    )
    oof_frame = pd.read_csv(
        require(root / "v99_semantic_cost_oof_predictions.csv")
    )
    test_summary = pd.read_csv(
        require(root / "v99_test_summary.csv")
    ).set_index("model")
    predictions = pd.read_csv(
        require(root / "semantic_cost_coach_v99_predictions.csv")
    )

    if pool.get("version") != POOL_VERSION:
        raise RuntimeError(f"unexpected V9.9 pool version: {pool.get('version')}")
    if pool.get("signature_version") != SIGNATURE_VERSION:
        raise RuntimeError("V9.9 signature version mismatch")
    fields = list(pool.get("signature_fields", []))
    if len(fields) != 21 or len(set(fields)) != len(fields):
        raise RuntimeError("V9.9 signature schema is missing or duplicated")
    provenance = pool.get("provenance", {})
    if provenance.get("is_fully_nested_teacher_stack") is not True:
        raise RuntimeError("V9.9 semantic pool is not fully nested")
    if provenance.get("holdout_label_isolation") is not True:
        raise RuntimeError("V9.9 semantic pool lacks holdout isolation")
    if provenance.get("historical_full_train_teacher_reuse") is not False:
        raise RuntimeError("V9.9 reused a historical full-Train teacher")

    if len(oof_frame) != 1284 or oof_frame.sample_id.nunique() != 1284:
        raise RuntimeError("V9.9 OOF predictions must contain 1284 unique samples")
    if len(predictions) != 686 or predictions.sample_id.nunique() != 686:
        raise RuntimeError("V9.9 Test predictions must contain 686 unique samples")
    required_models = {
        "anchor",
        "minimum_predicted_cost_action_all",
        "semantic_cost_soft_mixture_all",
        "semantic_cost_valid_selected",
        "true_region_expert_policy",
        "sample_oracle_upper_bound",
    }
    if not required_models.issubset(test_summary.index):
        raise RuntimeError("V9.9 Test summary is incomplete")

    tensors = (
        pool["anchor"],
        pool["function_space"],
        pool["expert_predictions"],
        pool["expert_signatures"],
        pool["labels"],
        pool["action_costs"],
    )
    if any(not torch.isfinite(value).all() for value in tensors):
        raise RuntimeError("V9.9 pool contains non-finite tensors")
    if pool["expert_signatures"].shape != (1284, 4, 21):
        raise RuntimeError("V9.9 semantic signature tensor has wrong shape")
    if not torch.allclose(
        pool["expert_signatures"][:, :, 0],
        pool["expert_predictions"].squeeze(-1),
        atol=1e-6,
    ):
        raise RuntimeError("signature prediction field differs from expert output")
    role_available = pool["expert_signatures"][:, :, 9]
    tail_available = pool["expert_signatures"][:, :, 10]
    risk_available = pool["expert_signatures"][:, :, 11]
    if not role_available[:, 1:3].eq(1).all() or not role_available[:, (0, 3)].eq(0).all():
        raise RuntimeError("role availability mask is inconsistent")
    if not tail_available[:, (0, 3)].eq(1).all() or not tail_available[:, 1:3].eq(0).all():
        raise RuntimeError("tail availability mask is inconsistent")
    if not risk_available[:, 1:3].eq(1).all() or not risk_available[:, (0, 3)].eq(0).all():
        raise RuntimeError("risk availability mask is inconsistent")

    all_holdout_groups = []
    for row in pool.get("fold_metadata", []):
        fold = int(row["outer_fold"])
        inner_train = set(row["inner_train_groups"])
        inner_valid = set(row["inner_valid_groups"])
        holdout = set(row["holdout_groups"])
        if inner_train & inner_valid or inner_train & holdout or inner_valid & holdout:
            raise RuntimeError(f"group leakage in V9.9 outer fold {fold}")
        all_holdout_groups.extend(holdout)
        checkpoints = row["upstream_checkpoints"]
        hashes = row["checkpoint_sha256"]
        for name in ("clean", "moddrop", "cfcompat"):
            path = require(checkpoints[name])
            if f"outer_fold_{fold}" not in str(path):
                raise RuntimeError(
                    f"fold {fold} {name} checkpoint is not fold-local: {path}"
                )
            if sha256(path) != hashes[name]:
                raise RuntimeError(f"fold {fold} {name} checkpoint hash changed")
    if len(all_holdout_groups) != len(set(all_holdout_groups)):
        raise RuntimeError("a source-video group occurs in multiple holdouts")

    anchor_oof = float(pool_summary["anchor_oof_mae"])
    oracle_oof = float(pool_summary["sample_oracle_oof_mae"])
    if oracle_oof < float(cli.minimum_plausible_oof_oracle_mae):
        raise RuntimeError(
            f"V9.9 OOF oracle is implausibly low ({oracle_oof:.6f})"
        )

    coach = summary.get("semantic_cost_coach", {})
    required_coach = {
        "oof_per_action_cost_mae",
        "oof_selected_action_accuracy",
        "oof_cost_selected_mae",
        "oof_anchor_mae",
        "oof_selected_gain_mae",
        "oof_scale_coverage_1x",
        "oof_scale_coverage_2x",
    }
    if not required_coach.issubset(coach):
        raise RuntimeError("V9.9 coach diagnostics are incomplete")

    anchor_test = float(test_summary.loc["anchor", "MAE"])
    deploy_test = float(test_summary.loc["semantic_cost_valid_selected", "MAE"])
    hard_test = float(
        test_summary.loc["minimum_predicted_cost_action_all", "MAE"]
    )
    soft_test = float(test_summary.loc["semantic_cost_soft_mixture_all", "MAE"])
    oracle_test = float(test_summary.loc["sample_oracle_upper_bound", "MAE"])
    gain = anchor_test - deploy_test

    print("ENGINEERING AUDIT PASSED")
    print("strict semantic OOF samples:", len(oof_frame))
    print("teacher stack fully nested:", provenance["is_fully_nested_teacher_stack"])
    print("historical full-Train teacher reuse:", provenance["historical_full_train_teacher_reuse"])
    print("holdout-label isolation:", provenance["holdout_label_isolation"])
    print("signature version:", pool["signature_version"])
    print("signature dimension:", len(fields))
    print(f"semantic OOF anchor MAE: {anchor_oof:.6f}")
    print(f"semantic OOF sample-oracle MAE: {oracle_oof:.6f}")
    print("semantic OOF expert global MAE:", pool_summary.get("expert_global_mae"))
    print("semantic OOF oracle action counts:", pool_summary.get("oracle_action_counts"))
    print("cost coach OOF diagnostics:", coach)
    for row in summary.get("oof_policy_profiles", []):
        print(
            "  profile=%s gain=%+.6f lower=%+.6f harm=%.4f activation=%.4f eligible=%s"
            % (
                row["profile"],
                row["oof_gain"],
                row["bootstrap_gain_lower"],
                row["harm_over_010_rate"],
                row["activation_rate"],
                row["oof_eligible"],
            )
        )
    print("validation selected profile:", summary.get("selected_profile"))
    print("validation selected policy:", summary.get("selected_policy"))
    print("test activation rate:", summary.get("test_activation_rate"))
    print("test proposed actions:", summary.get("test_proposed_action_counts"))
    print("test deployed actions:", summary.get("test_deployed_action_counts"))
    print(f"test anchor MAE: {anchor_test:.6f}")
    print(f"test hard cost-selected MAE: {hard_test:.6f}")
    print(f"test soft cost-mixture MAE: {soft_test:.6f}")
    print(f"test deployable semantic coach MAE: {deploy_test:.6f}")
    print(f"test deployable gain: {gain:+.6f}")
    print(f"test sample-oracle MAE: {oracle_test:.6f}")
    if cli.require_deployable_gain is not None and gain < cli.require_deployable_gain:
        raise RuntimeError(
            f"deployable gain {gain:.6f} below required {cli.require_deployable_gain:.6f}"
        )
    if gain <= 0:
        print("WARNING: V9.9 deployable coach did not beat Anchor on Test")


if __name__ == "__main__":
    main()
