"""Freeze the Stage 21A audit protocol before inspecting probe outcomes."""

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mosei.stage21a_common import atomic_json, sha256_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--uniform-checkpoint", required=True)
    parser.add_argument("--teacher-cache-manifest", required=True)
    parser.add_argument("--compatibility-cache", required=True)
    parser.add_argument("--gpu-id", type=int, default=3)
    return parser.parse_args()


def command(*args):
    return subprocess.check_output(list(args), cwd=str(ROOT), text=True).strip()


def main():
    cli = parse_args()
    branch = command("git", "branch", "--show-current")
    head = command("git", "rev-parse", "HEAD")
    gpu_state = command(
        "nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv,noheader"
    ).splitlines()
    process_state = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"],
        text=True,
    ).strip().splitlines()
    protocol = {
        "protocol": "Stage 21A Source-Relative Learning Feasibility Audit v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "branch": branch,
        "implementation_commit": head,
        "base_commit": cli.base_commit,
        "dataset": "mosei",
        "allowed_splits": ["train", "valid"],
        "locked_test_access_count": 0,
        "gpu_id": cli.gpu_id,
        "full_dlf_training": False,
        "official_valid_probe_requires_train_gate": True,
        "delta_candidates": [1.0, 0.75, 0.5],
        "delta_selection": "first descending Train-only delta with >=60% sample and >=50% multi-clip source coverage",
        "max_pairs_per_source": 64,
        "pair_seed": 2100,
        "audit_split_seeds": [2101, 2102],
        "primary_representation": "actual single-clip tensor entering backbone.proj1 after ModDrop mask residual",
        "secondary_representation": "concatenated c_l_sim/c_v_sim/c_a_sim; diagnostic only",
        "shared_head": True,
        "mode_specific_head_used_for_gate": False,
        "ridge_alpha": 0.001,
        "relative_mass": 0.5,
        "permutations": 2000,
        "cluster_bootstraps": 2000,
        "gradient_batches_max": 20,
        "optimizer_steps": 0,
        "uniform_checkpoint": {
            "path": cli.uniform_checkpoint,
            "sha256": sha256_file(cli.uniform_checkpoint),
        },
        "teacher_cache_manifest": {
            "path": cli.teacher_cache_manifest,
            "sha256": sha256_file(cli.teacher_cache_manifest),
        },
        "compatibility_cache": {
            "path": cli.compatibility_cache,
            "sha256": sha256_file(cli.compatibility_cache),
        },
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "dependencies_upgraded": False,
    }
    output = Path(cli.output_root)
    atomic_json(output / "protocol/frozen_protocol.json", protocol)
    atomic_json(
        output / "protocol/initial_state.json",
        {
            "created_at": protocol["created_at"],
            "branch": branch,
            "head": head,
            "base_commit": cli.base_commit,
            "git_status": command("git", "status", "--porcelain"),
            "gpu_state": gpu_state,
            "compute_process_state": process_state,
            "approved_gpu": cli.gpu_id,
            "external_compute_processes_present": bool(process_state),
            "locked_test_access_count": 0,
        },
    )
    print(json.dumps({"branch": branch, "commit": head, "gpu": cli.gpu_id}, indent=2))


if __name__ == "__main__":
    main()
