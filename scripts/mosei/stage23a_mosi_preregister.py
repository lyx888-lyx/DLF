"""Freeze the MOSEI Stage23A protocol and create MOSI 5-fold source OOF."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from stage23a_mosi_common import (
    DATASET_PATH,
    EXPERTS,
    FROZEN_MOSEI_ROOT,
    N_FOLDS,
    RESULT_ROOT,
    SPLIT_SEED,
    atomic_csv,
    atomic_json,
    git_head,
    inner_valid,
    judge_fold,
    ordered_id_sha,
    sha256_file,
    source_video,
)


FROZEN_FILES = (
    "result/arbiter_audit_v1/mosei/protocol/preregistered_protocol.json",
    "scripts/mosei/stage23a_common.py",
    "scripts/mosei/stage23a_train_fold.py",
    "scripts/mosei/stage23a_analyze.py",
)

ASSETS = {
    "uniform_kd_seed1111": (
        "/code/DLF-mosi-cfcompat-evidence-v1/runtime/cfcompat_evidence_v1/"
        "checkpoints/mosi/stage18d/uniform/best_valid.pth"
    ),
    "moddrop_seed1111": (
        "/code/DLF/pt/missing_baseline/moddrop/DLF_mosi_seed1111_best.pth"
    ),
    "moddrop_seed1114": (
        "/code/DLF/pt/missing_baseline/moddrop_benchmark_multiseed_v1/"
        "seed1114/DLF_mosi_seed1114_best_valid.pth"
    ),
    "cfcompat_seed1111": (
        "/code/DLF/pt/missing_baseline/cf_compat_kd_v1/"
        "DLF_mosi_seed1111_best_valid.pth"
    ),
    "cfcompat_seed1114": (
        "/code/DLF/pt/missing_baseline/cf_compat_kd_v1/"
        "benchmark_multiseed/seed1114/DLF_mosi_seed1114_best_valid.pth"
    ),
}


def main():
    frozen_path = RESULT_ROOT / "protocol" / "preregistered_protocol.json"
    if frozen_path.exists():
        raise RuntimeError("MOSI protocol is already frozen: {}".format(frozen_path))
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(DATASET_PATH)
    frozen_sha = {}
    for relative in FROZEN_FILES:
        path = FROZEN_MOSEI_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(
                "Frozen MOSEI protocol input absent: {}".format(path)
            )
        frozen_sha[relative] = sha256_file(path)
    mosei_protocol = json.loads(
        (FROZEN_MOSEI_ROOT / FROZEN_FILES[0]).read_text()
    )
    if tuple(mosei_protocol["expert_ids"]) != tuple(EXPERTS):
        raise RuntimeError("Code and frozen MOSEI candidate pools differ.")
    if mosei_protocol.get("locked_test_access_count") != 0:
        raise RuntimeError("Frozen MOSEI manifest test lock is not zero.")
    for expert, path in ASSETS.items():
        if not Path(path).is_file():
            raise FileNotFoundError("{} asset absent: {}".format(expert, path))

    # Only the train key is selected. No Valid/Test sample is inspected.
    import pickle

    with DATASET_PATH.open("rb") as handle:
        train = pickle.load(handle)["train"]
    ids = [str(value) for value in train["id"]]
    videos = [source_video(value) for value in ids]
    sources = np.array(sorted(set(videos)), dtype=object)
    splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SPLIT_SEED)
    source_to_fold = {}
    for fold, (_, held_out) in enumerate(splitter.split(sources)):
        for position in held_out:
            source_to_fold[str(sources[position])] = int(fold)
    rows = []
    for index, (sample_id, video_id) in enumerate(zip(ids, videos)):
        row = {
            "train_index": index,
            "sample_id": sample_id,
            "video_id": video_id,
            "outer_fold": source_to_fold[video_id],
            "judge_fold": judge_fold(video_id),
        }
        row.update(
            {
                "inner_valid_outer{}".format(fold): int(
                    inner_valid(video_id, fold)
                )
                for fold in range(N_FOLDS)
            }
        )
        rows.append(row)
    split = pd.DataFrame(rows)
    if split.groupby("video_id").outer_fold.nunique().max() != 1:
        raise RuntimeError("MOSI outer folds leak source videos.")
    split_path = RESULT_ROOT / "protocol" / "source_splits.csv"
    atomic_csv(split, split_path)
    fold_summary = []
    for fold in range(N_FOLDS):
        local = split.loc[split.outer_fold == fold]
        fold_summary.append(
            {
                "fold": fold,
                "sample_count": int(len(local)),
                "source_count": int(local.video_id.nunique()),
                "ordered_sample_id_sha256": ordered_id_sha(local.sample_id.tolist()),
                "source_id_sha256": hashlib.sha256(
                    json.dumps(
                        sorted(local.video_id.unique().tolist()),
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
            }
        )
    assets = {
        expert: {
            "path": path,
            "sha256": sha256_file(path),
            "bytes": Path(path).stat().st_size,
        }
        for expert, path in ASSETS.items()
    }
    protocol = {
        "stage": "Stage 23A-MOSI Frozen-Protocol Parallel Transfer Audit",
        "status": "PREREGISTERED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_head(),
        "frozen_mosei_root": str(FROZEN_MOSEI_ROOT),
        "frozen_mosei_protocol_sha256": frozen_sha[FROZEN_FILES[0]],
        "frozen_mosei_file_sha256": frozen_sha,
        "dataset": "mosi",
        "dataset_path": str(DATASET_PATH),
        "train_sample_count": len(ids),
        "train_source_count": len(sources),
        "train_ordered_sample_id_sha256": ordered_id_sha(ids),
        "n_folds": N_FOLDS,
        "split_seed": SPLIT_SEED,
        "source_split_path": str(split_path),
        "source_split_sha256": sha256_file(split_path),
        "fold_summary": fold_summary,
        "expert_ids": list(EXPERTS),
        "component_ids": [
            "clean_seed1111",
            "clean_seed1114",
            *list(EXPERTS),
        ],
        "candidate_assets": assets,
        "training_budget": dict(mosei_protocol["training_budget"]),
        "metric_J": "0.5*MAE_LAV + 0.5*mean(MAE_LA,MAE_LV,MAE_L)",
        "expert_quality_gate": {
            "not_failed": "own_J <= best_single_J + 0.05",
            "near_best": "abs_error <= min_expert_error + 0.05 on >=5% samples",
            "unique": "leave-one-out Oracle delta J >=0.001 or any mode delta >=0.001",
            "retained_count": "3..5",
            "fold_robustness": "near-best contribution occurs in at least two outer folds",
        },
        "oracle_gate": {
            "delta_J_vs_per_mode_fixed": -0.005,
            "missing_mode_delta_MAE": -0.003,
            "minimum_actionable_missing_modes": 2,
            "minimum_improving_outer_folds": 3,
        },
        "judge": {
            "splits": "two fixed source-disjoint folds",
            "target": "per-expert absolute error (regret ranking derived)",
            "J0": "mode-only",
            "J1": "expert scalar predictions + mode",
            "J2": "predictions + pairwise absolute disagreement + mode",
            "J3": "unavailable; no frozen calibrated self-confidence",
            "J4": "J2 with error labels shuffled within mode",
            "model": "StandardScaler + Ridge(alpha=10)",
            "forbidden_inputs": ["ground truth", "high-dimensional hidden state"],
        },
        "teacher": {
            "scalar_risk": "normalize inverse predicted absolute risks",
            "fallback": "per-mode MAE-minimizing simplex stacking",
            "joint_risk": "Sigma_i=D(r_i) C_mode D(r_i)",
            "objective": "w^T Sigma_i w + rho*||w-w_fallback||^2",
            "rho_grid": [0.001, 0.01, 0.1, 1.0, 10.0],
            "gamma_grid": [0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005],
            "rho_gamma_selection": "Judge-train-only deterministic source tune subset",
            "fallback_trigger": "use dynamic weights only if predicted risk gain >= gamma",
            "shuffled_control": "same joint construction with J4 risk",
        },
        "train_only_gate": {
            "mean_delta_J_vs_fixed": -0.003,
            "worst_split_delta_J_vs_fixed": 0.001,
            "mean_delta_J_vs_shuffled": -0.002,
            "minimum_improved_missing_modes": 2,
            "worst_LAV_delta_MAE": 0.003,
            "no_systematic_classification_degradation": True,
            "risk_bucket_max_violations": 1,
            "dynamic_trigger_min_rate": 0.05,
            "dynamic_weight_min_l1": 0.01,
            "two_split_direction_consistency": True,
        },
        "official_valid_gate": (
            "single confirmation only after train-only pass; must beat per-mode "
            "fixed stacking, shuffled Judge, and best single without retuning"
        ),
        "parallel_policy": {
            "gpu_count": 1,
            "maximum_training_workers": 1,
            "OMP_NUM_THREADS": 2,
            "dataloader_workers": 2,
            "mosei_epoch_slowdown_stop_threshold": 0.15,
        },
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "test_loader_constructed": False,
        "student_trained": False,
        "dependencies_upgraded": False,
    }
    atomic_json(frozen_path, protocol)
    print(json.dumps(protocol, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
