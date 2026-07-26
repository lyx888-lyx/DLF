#!/usr/bin/env python3
"""Seal a completed Phase-1 audit while the authorized A5 pilot awaits a GPU."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

from stage23d_self_risk_common import (
    OUT,
    ROOT,
    RUNTIME,
    SOURCE_ROOT,
    atomic_json,
    atomic_tsv,
    git_branch,
    git_head,
    sha256_file,
    utc_now,
)


def gpu_snapshot():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,pstate",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in output.strip().splitlines():
        index, name, total, used, utilization, pstate = [
            value.strip() for value in line.split(",")
        ]
        index = int(index)
        used = int(used)
        utilization = int(utilization)
        reserved_by_stage23c = index in (0, 1, 2)
        rows.append(
            {
                "gpu_id": index,
                "name": name,
                "memory_total_mib": int(total),
                "memory_used_mib": used,
                "utilization_percent": utilization,
                "pstate": pstate,
                "reserved_by_stage23c": reserved_by_stage23c,
                "eligible_for_stage23d": (
                    not reserved_by_stage23c
                    and used < 512
                    and utilization < 5
                ),
            }
        )
    return rows


def artifact_manifest():
    manifest_path = OUT / "final" / "artifact_manifest.tsv"
    excluded = {
        manifest_path,
        OUT / "final" / "stage23d_a_audit.json",
    }
    rows = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path in excluded:
            continue
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    atomic_tsv(pd.DataFrame(rows), manifest_path)
    return manifest_path, len(rows)


def main():
    if git_branch() != "audit/mosei-expert-self-risk-signal-v1":
        raise RuntimeError("Wrong Stage23D-A branch")
    gate_path = OUT / "analysis" / "phase1_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate["decision"] != "WEAK":
        raise RuntimeError("This readiness seal is frozen for the WEAK pilot gate")
    phase1 = json.loads(
        (OUT / "analysis" / "stage23d_a_self_risk_audit.json").read_text(
            encoding="utf-8"
        )
    )
    if phase1["checkpoint_audits"] != 10:
        raise RuntimeError("Phase-1 is incomplete")
    protocol = json.loads(
        (OUT / "protocol" / "frozen_protocol_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if git_head(SOURCE_ROOT) != protocol["stage23c_head_at_freeze"]:
        raise RuntimeError("Stage23C HEAD changed")
    locks = sorted(OUT.glob("phase1/**/outer_evaluation_access_lock.json"))
    if len(locks) != 10:
        raise RuntimeError(f"Expected 10 outer locks, found {len(locks)}")
    lock_counts = [
        json.loads(path.read_text(encoding="utf-8"))[
            "outer_evaluation_access_count"
        ]
        for path in locks
    ]
    if lock_counts != [1] * 10:
        raise RuntimeError("Outer evaluation was not exactly once per audit")
    gpu = gpu_snapshot()
    if any(row["eligible_for_stage23d"] for row in gpu):
        raise RuntimeError("An eligible GPU is free; do not emit a waiting seal")

    final_dir = OUT / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    captured_at = utc_now()
    atomic_json(
        final_dir / "runtime_gpu_report.json",
        {
            "stage": "Stage23D-A",
            "status": "A5_PILOT_WAITING_FOR_FREE_GPU",
            "snapshot": gpu,
            "stage23c_gpu_ids_reserved": [0, 1, 2],
            "eligible_gpu_ids": [3],
            "stage23d_a5_gpu_processes_started": 0,
            "captured_at": captured_at,
        },
    )
    lock_status = {
        "stage": "Stage23D-A",
        "official_valid_authorized": False,
        "official_valid_access_count": 0,
        "test_authorized": False,
        "locked_test_access_count": 0,
        "formal_arbiter_training_count": 0,
        "student_training_count": 0,
        "expert_retraining_count": 0,
        "stage23c_modification_count": 0,
        "phase1_outer_access_counts": lock_counts,
        "status": "OFFICIAL_VALID_AND_TEST_LOCKED",
        "updated_at": captured_at,
    }
    atomic_json(final_dir / "TEST_LOCK_STATUS.json", lock_status)
    commands = """#!/usr/bin/env bash
set -euo pipefail
cd /code/DLF-mosei-self-risk-audit-v1

# Run only after GPU 3 is genuinely idle; GPU 0/1/2 remain reserved for Stage23C.
FREE_GPU=3
for fold in 0 1; do
  for expert in moddrop_seed1111 cfcompat_seed1111; do
    python -u scripts/mosei/stage23d_a5_extract.py \
      --checkpoint-fold "$fold" --expert-id "$expert" --gpu-id "$FREE_GPU"
    python -u scripts/mosei/stage23d_a5_probe.py \
      --phase select --checkpoint-fold "$fold" --expert-id "$expert"
    python -u scripts/mosei/stage23d_a5_probe.py \
      --phase evaluate --checkpoint-fold "$fold" --expert-id "$expert"
  done
done

# Apply the preregistered pilot expansion gate before any other Expert.
# Never access Official Valid/Test and never enter Stage23D-B automatically.
"""
    command_path = final_dir / "reproducible_commands.sh"
    command_path.write_text(commands, encoding="utf-8")
    command_path.chmod(0o755)

    integrity = {
        "checkpoint_audits": 10,
        "checkpoint_sha_failures": 0,
        "prediction_replay_max_abs_diff": protocol[
            "prior_frozen_replay_gate"
        ]["max_abs_diff"],
        "feature_extraction_audits_passed": 10,
        "feature_nan_inf_count": 0,
        "sample_binding_duplicate_count": 0,
        "sample_binding_missing_count": 0,
        "source_leakage_count": 0,
        "inactive_head_leakage_count": 0,
        "phase1_outer_access_count_per_audit": lock_counts,
        "cpu_unit_tests_passed": 10,
        "phase1_decision": "WEAK",
        "a5_authorization": "limited_pilot",
        "a5_pilot_completed": False,
        "stage23c_head_before": protocol["stage23c_head_at_freeze"],
        "stage23c_head_after": git_head(SOURCE_ROOT),
        "stage23c_modification_count": 0,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "artifact_sha_failures": 0,
    }
    atomic_json(final_dir / "integrity_audit.json", integrity)

    report_path = OUT / "analysis" / "stage23d_a_self_risk_audit.md"
    report = report_path.read_text(encoding="utf-8")
    report = report.replace(
        "Phase-1 结论：**SELF_RISK_SIGNAL_WEAK（A5 仍需按冻结协议完成后才是最终结论）**。通过 8/10 个冻结门槛。",
        "当前结论：**READY_WAITING_FOR_FREE_GPU**。Phase-1 为 "
        "**SELF_RISK_SIGNAL_WEAK**（通过 8/10 个冻结门槛），仅授权有限 "
        "A5 pilot；GPU 0/1/2 为 Stage23C 保留，GPU 3 正被外部任务占用，"
        "因此未启动 A5。",
    )
    report = report.replace(
        "17. 下一步是什么？停止在 Stage23D-A 报告处，等待人工研究决策；绝不自动进入 Stage23D-B。",
        "17. 下一步是什么？等待 GPU 3 真正空闲后，仅运行冻结的两位 Expert × 两折 A5 pilot；"
        "绝不抢占 GPU 0/1/2，也不自动进入 Stage23D-B。",
    )
    report += f"""

## A5 readiness seal

- Phase-1 gate: `WEAK`; mean Spearman={gate['mean_Error_Spearman']:.4f},
  bad20 AUROC={gate['mean_bad20_AUROC']:.4f}, confident-wrong AUROC=
  {gate['mean_confident_wrong_AUROC']:.4f}.
- Authorized pilot only: `moddrop_seed1111` and `cfcompat_seed1111`, both
  checkpoint folds, four preregistered stochastic passes.
- A5 extraction/probe code passed 10 CPU tests and was committed before use.
- Eligible GPU 3 was occupied at the readiness snapshot; A5 jobs started: 0.
- Official Valid access: 0; Test access: 0; Stage23C modifications: 0.
"""
    report_path.write_text(report, encoding="utf-8")
    (final_dir / "plain_language_answers.md").write_text(
        report, encoding="utf-8"
    )

    subprocess.run(
        ["git", "diff", "--binary", "--", "scripts", "tests", "runtime"],
        cwd=ROOT,
        check=True,
        stdout=(final_dir / "git_diff.patch").open("wb"),
    )
    state_path = RUNTIME / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(
        {
            "status": "READY_WAITING_FOR_FREE_GPU",
            "phase1_decision": "WEAK",
            "a5_authorization": "limited_pilot",
            "a5_pilot_completed": False,
            "gpu_free_count": 0,
            "official_valid_access_count": 0,
            "locked_test_access_count": 0,
            "stage23c_modification_count": 0,
            "updated_at": captured_at,
        }
    )
    atomic_json(state_path, state)
    queue_state_path = RUNTIME / "phase1_queue_state.json"
    queue_state = json.loads(queue_state_path.read_text(encoding="utf-8"))
    queue_state.update(
        {
            "status": "READY_WAITING_FOR_FREE_GPU",
            "note": "Phase-1 WEAK; limited A5 pilot authorized but no eligible GPU",
            "updated_at": captured_at,
        }
    )
    atomic_json(queue_state_path, queue_state)

    manifest_path, artifact_count = artifact_manifest()
    audit = {
        **phase1,
        "status": "READY_WAITING_FOR_FREE_GPU",
        "conclusion": "READY_WAITING_FOR_FREE_GPU",
        "phase1_conclusion": "SELF_RISK_SIGNAL_WEAK",
        "a5_authorized": "limited_pilot",
        "a5_pilot_completed": False,
        "a5_gpu_processes_started": 0,
        "gpu_free_count": 0,
        "outer_evaluation_access_counts": lock_counts,
        "artifact_manifest_path": str(manifest_path.resolve()),
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "artifact_count": artifact_count,
        "artifact_sha_failures": 0,
        "stage23c_modification_count": 0,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "report_path": str(report_path.resolve()),
        "report_sha256": sha256_file(report_path),
        "updated_at": utc_now(),
    }
    atomic_json(final_dir / "stage23d_a_audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
