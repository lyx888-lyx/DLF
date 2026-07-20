"""Freeze the Stage 20 protocol after audits and before candidate training."""

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trains.singleTask.mgrd_utils import sha256_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--stage19-root", required=True)
    return parser.parse_args()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(str(temporary), str(path))


def main():
    cli = parse_args()
    result = Path(cli.result_root)
    stage19 = Path(cli.stage19_root)
    cache_manifest = stage19 / "protocol/cache/seed1111/cache_manifest.json"
    uniform_manifest = stage19 / "baseline/uniform_seed1111_fp32/run_manifest.json"
    uniform_metrics = stage19 / "baseline/uniform_seed1111_fp32/epoch_metrics.csv"
    uniform_pointer = (
        stage19
        / "baseline/uniform_seed1111_fp32/checkpoints/best_checkpoint.json"
    )
    cache = json.loads(cache_manifest.read_text())
    baseline = json.loads(uniform_manifest.read_text())
    pointer = json.loads(uniform_pointer.read_text())
    checkpoint = Path(pointer["path"])
    required = {
        "cache_manifest": cache_manifest,
        "uniform_manifest": uniform_manifest,
        "uniform_epoch_metrics": uniform_metrics,
        "uniform_checkpoint": checkpoint,
    }
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError("{}: {}".format(name, path))
    if cache["locked_test_access_count"] != 0:
        raise RuntimeError("Stage 19 cache violated the Test lock.")
    if baseline["locked_test_access_count"] != 0:
        raise RuntimeError("Stage 19 Uniform manifest violated the Test lock.")
    retro = json.loads(
        (result / "retro/stage19_retro_best_checkpoint_audit.json").read_text()
    )
    ghost = json.loads((result / "audit/ghost_modality_audit.json").read_text())
    tests = json.loads((result / "tests/test_results.json").read_text())
    if tests["tests_failed"]:
        raise RuntimeError("Implementation gates have not passed.")
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=str(ROOT), text=True
    ).strip()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
    ).strip()
    baseline_payload = {
        "baseline_id": "stage19_uniform_seed1111_fp32",
        "reused": True,
        "reuse_reason": "All frozen optimizer/data/cache/initialization/loss-normalization fields match Stage 19.",
        "manifest_path": str(uniform_manifest),
        "manifest_sha256": sha256_file(uniform_manifest),
        "epoch_metrics_path": str(uniform_metrics),
        "epoch_metrics_sha256": sha256_file(uniform_metrics),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "best_epoch": baseline["best_epoch"],
        "best_J_valid": baseline["best_J_valid"],
        "locked_test_access_count": 0,
    }
    protocol = {
        "protocol": "SAFE-DLF v1",
        "base_commit": "56a7a47de5fd48c4044a6fd1790d26033aa037bd",
        "implementation_commit": commit,
        "branch": branch,
        "dataset": "mosei",
        "seed_order": [1111, 1114],
        "methods": ["sao_pm", "safe_full"],
        "amp_mode": "off",
        "batch_size": 16,
        "update_epochs": 10,
        "accumulation_semantics": "sum_not_mean",
        "learning_rate": 0.0001,
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau(mode=min,factor=0.5,patience=5)",
        "early_stop": 10,
        "maximum_epochs": 30,
        "screen_epoch_1": 4,
        "screen_epoch_2": 8,
        "teacher_cache": {
            "path": str(cache_manifest),
            "sha256": sha256_file(cache_manifest),
            "train_order_sha256": cache["entries"]["train"][
                "ordered_sample_id_sha256"
            ],
            "valid_order_sha256": cache["entries"]["valid"][
                "ordered_sample_id_sha256"
            ],
        },
        "uniform_baseline": baseline_payload,
        "selection": "minimum Official Valid J; best-so-far comparison",
        "seed1_gate": {
            "delta_J_max": -0.003,
            "missing_macro_mae_must_improve": True,
            "missing_modes_mae_improved_min": 2,
            "lav_mae_degradation_max": 0.003,
            "lav_corr_degradation_max": 0.002,
            "missing_macro_corr_degradation_max": 0.002,
            "classification_degradation_max": 0.003,
        },
        "seed2_gate": {
            "mean_delta_J_max": -0.003,
            "worst_delta_J_max": 0.001,
        },
        "phase_r_status": retro["status"],
        "phase_a_status": ghost["status"],
        "implementation_gate_status": tests["status"],
        "locked_test_access_count": 0,
        "dependencies_upgraded": False,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(result / "baseline/uniform_seed1111_manifest.json", baseline_payload)
    atomic_json(result / "protocol/frozen_protocol.json", protocol)
    atomic_json(
        result / "protocol/initial_state.json",
        {
            "branch": branch,
            "head": commit,
            "source_worktree_head": "dc8536f338c7e310a73447e4185c38be52800f48",
            "stage19_worktree_head": "56a7a47de5fd48c4044a6fd1790d26033aa037bd",
            "running_training_processes": 0,
            "gpus_initially_free": [0, 1, 2, 3],
            "locked_test_access_count": 0,
        },
    )
    print(json.dumps({"status": "STAGE20_PROTOCOL_FROZEN", "commit": commit}, indent=2))


if __name__ == "__main__":
    main()
