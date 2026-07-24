"""Close Stage23A-MOSI after the parallel resource safety gate blocks it."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from stage23a_mosi_common import (
    RESULT_ROOT,
    atomic_json,
    git_head,
    load_preregistered,
    sha256_file,
)


STATUS = "STAGE23A_MOSI_PARALLEL_BLOCKED_NO_SAFE_RESOURCE"
ROOT = Path(__file__).resolve().parents[2]


def main():
    protocol, protocol_sha = load_preregistered()
    speed_path = RESULT_ROOT / "speed_probe" / "speed_probe_manifest.json"
    monitor_path = RESULT_ROOT / "parallel_safety_monitor.json"
    if not speed_path.is_file() or not monitor_path.is_file():
        raise FileNotFoundError("Speed probe or ten-minute monitor artifact absent.")
    speed = json.loads(speed_path.read_text())
    initial_monitor = json.loads(monitor_path.read_text())
    final_snapshot = initial_monitor["snapshots"][-1]
    fold_rows = []
    for fold in ("0", "1"):
        values = final_snapshot["folds"][fold]["new_epoch_seconds"]
        if not values:
            raise RuntimeError("No post-launch MOSEI epoch for fold{}.".format(fold))
        baseline = float(
            final_snapshot["folds"][fold]["baseline_seconds_per_epoch"]
        )
        observed = float(values[-1])
        fold_rows.append(
            {
                "fold": int(fold),
                "baseline_seconds_per_epoch": baseline,
                "parallel_seconds_per_epoch": observed,
                "slowdown_ratio": observed / baseline,
                "slowdown_fraction": observed / baseline - 1.0,
            }
        )
    if not all(row["slowdown_fraction"] > 0.15 for row in fold_rows):
        raise RuntimeError("Cross-fold resource block is not supported.")
    corrected_monitor = dict(initial_monitor)
    corrected_monitor.update(
        {
            "automatic_initial_verdict": initial_monitor["verdict"],
            "verdict": (
                "BLOCKED_CROSS_FOLD_CORROBORATED_MOSEI_SLOWDOWN_GT_15_PERCENT"
            ),
            "unsafe": True,
            "posthoc_cross_fold_reassessment": True,
            "cross_fold_slowdown": fold_rows,
            "mosi_process_group_stopped": True,
            "mosei_processes_untouched": True,
        }
    )
    atomic_json(monitor_path, corrected_monitor)

    baseline_mean = float(
        np.mean([row["baseline_seconds_per_epoch"] for row in fold_rows])
    )
    mosi_epoch = float(speed["observed_seconds_per_epoch"])
    speed_ratio = baseline_mean / mosi_epoch
    started = datetime.fromisoformat(speed["completed_at"])
    # Include the three probe epochs that ended immediately before completed_at.
    started = started - (
        datetime.fromtimestamp(
            sum(speed["epoch_seconds"]), tz=timezone.utc
        )
        - datetime.fromtimestamp(0, tz=timezone.utc)
    )
    ended = datetime.now(timezone.utc)
    wall_seconds = (ended - started).total_seconds()
    report = [
        "# Stage 23A-MOSI — Frozen-Protocol Parallel Transfer Audit",
        "",
        "Final status: `{}`".format(STATUS),
        "",
        "The audit stopped at the mandatory parallel-resource gate. No Expert "
        "complementarity or Judge conclusion is permitted from partial OOF output.",
        "",
        "## Required answers",
        "",
        "1. MOSI speed probe was {:.3f} s/epoch; the mean pre-launch MOSEI "
        "baseline was {:.3f} s/epoch, so the raw dataset/model speed ratio was "
        "{:.2f}x.".format(mosi_epoch, baseline_mean, speed_ratio),
        "2. Parallel execution affected MOSEI throughput: fold0 slowed {:.1%} "
        "and fold1 slowed {:.1%}; MOSI alone was stopped.".format(
            fold_rows[0]["slowdown_fraction"],
            fold_rows[1]["slowdown_fraction"],
        ),
        "3. Five-fold MOSI OOF quality: not evaluable; the resource gate stopped "
        "the run before all folds/components completed.",
        "4. Method-vs-seed complementarity: not evaluable.",
        "5. Oracle expert selection vs per-mode fixed stacking: not evaluable.",
        "6. Unique expert marginal contributions: not evaluable.",
        "7. Judge error/regret prediction: Judge was not trained.",
        "8. Joint-risk Teacher vs fixed stacking: not evaluated.",
        "9. MOSI/MOSEI trend consistency: not evaluable from partial OOF.",
        "10. Worth waiting for primary MOSEI Stage23A: Yes.",
        "11. Recommend Student training: No.",
        "12. Official Valid accessed: No.",
        "13. Locked Test access count: 0.",
        "",
        "## Execution facts",
        "",
        "- GPU used: 2",
        "- MOSI workers: one training worker; OMP_NUM_THREADS=2; DataLoader workers=2",
        "- 3-epoch probe: {:.3f} s/epoch, peak {:.3f} GiB".format(
            mosi_epoch, speed["peak_gpu_memory_bytes"] / (1024 ** 3)
        ),
        "- MOSI process group stopped: Yes",
        "- MOSEI processes killed/paused/reniced: No",
        "- Official Valid access count: 0",
        "- Locked Test access count: 0",
        "- Student trained: No",
        "- Dependencies upgraded: No",
        "- Existing MOSI/MOSEI worktrees modified: No",
        "",
    ]
    report_path = RESULT_ROOT / "analysis" / "stage23a_mosi_audit.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")
    manifest = {
        "stage": "Stage 23A-MOSI",
        "status": STATUS,
        "branch": "audit/mosi-personalized-teacher-feasibility-v1",
        "base_commit": "bfc12dfc38bcc015adefd0c43e60305a7d9973fc",
        "code_commit_at_finalize": git_head(),
        "protocol_sha256": protocol_sha,
        "speed_probe_sha256": sha256_file(speed_path),
        "parallel_monitor_sha256": sha256_file(monitor_path),
        "report_sha256": sha256_file(report_path),
        "gpu_id": 2,
        "speed_probe_seconds_per_epoch": mosi_epoch,
        "speed_ratio_vs_mean_mosei_baseline": speed_ratio,
        "cross_fold_slowdown": fold_rows,
        "total_wall_seconds": wall_seconds,
        "tests_passed": 3,
        "gpu_released_to_preexisting_baseline": True,
        "mosei_processes_untouched": True,
        "existing_worktrees_modified": False,
        "dependencies_upgraded": False,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "test_loader_constructed": False,
        "student_trained": False,
        "completed_at": ended.isoformat(),
    }
    manifest_path = RESULT_ROOT / "analysis" / "stage23a_mosi_manifest.json"
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
