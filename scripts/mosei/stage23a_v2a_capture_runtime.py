#!/usr/bin/env python
"""Capture the reproducibility/runtime snapshot without touching model data."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone

import numpy
import pandas
import scipy
import sklearn
import torch

from stage23a_v2_common import ROOT, V2_ROOT, atomic_json, git_head, sha256_file


def command(args):
    return subprocess.check_output(args, cwd=str(ROOT), text=True).strip()


def main():
    analysis = V2_ROOT / "analysis_v2a"
    diff_text = subprocess.check_output(
        [
            "git",
            "diff",
            "c889cf6220c51fdec44407cc469471b4d6b99386",
            "--",
            "scripts/mosei",
            "tests",
        ],
        cwd=str(ROOT),
        text=True,
    )
    diff_path = analysis / "git_diff_from_v2_protocol.patch"
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    diff_path.write_text(diff_text, encoding="utf-8")
    logs = {}
    runtime_dir = ROOT / "runtime" / "stage23a_v2a"
    for path in sorted(runtime_dir.glob("*.log")):
        logs[path.name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "modified_at_utc": datetime.fromtimestamp(
                path.stat().st_mtime, timezone.utc
            ).isoformat(),
        }
    snapshot = {
        "stage": "Stage23A-v2a runtime/environment snapshot",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "git_branch": command(["git", "branch", "--show-current"]),
        "git_status_short": command(["git", "status", "--short"]),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "nvidia_smi_gpu_snapshot": command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            ]
        ).splitlines(),
        "gpu_allocation": {
            "hierarchical_fold0": "physical GPU 0 via CUDA_VISIBLE_DEVICES=0",
            "hierarchical_fold1": "physical GPU 2 via CUDA_VISIBLE_DEVICES=2",
            "physical_GPU_1": "not used; external process present",
            "physical_GPU_3": "not used; external process present",
            "content_feature_and_probes": "CPU",
        },
        "runtime_logs": logs,
        "git_diff_path": str(diff_path.resolve()),
        "git_diff_sha256": sha256_file(diff_path),
        "dependency_install_or_upgrade_performed": False,
        "expert_training_performed": False,
        "judge_training_performed": False,
        "student_training_performed": False,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
    }
    path = analysis / "runtime_environment.json"
    atomic_json(path, snapshot)
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
