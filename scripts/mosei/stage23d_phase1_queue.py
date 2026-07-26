#!/usr/bin/env python3
"""Run Phase-1 serially only after extraction and Stage23C training finish."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from stage23d_self_risk_common import EXPERTS, FOLDS, OUT, ROOT, RUNTIME, atomic_json, utc_now


def extraction_complete():
    path = RUNTIME / "extraction_queue_state.json"
    if not path.exists():
        return False
    if json.loads(path.read_text()).get("status") != "COMPLETED":
        return False
    for fold in FOLDS:
        for expert in EXPERTS:
            manifest = (
                OUT
                / "features"
                / f"checkpoint_fold{fold}"
                / expert
                / "extraction_manifest.json"
            )
            try:
                payload = json.loads(manifest.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                return False
            if payload.get("feature_schema_version") != 2:
                return False
    return True


def stage23c_active_processes():
    result = subprocess.run(
        ["ps", "-eo", "args="],
        check=False,
        capture_output=True,
        text=True,
    )
    return [
        line
        for line in result.stdout.splitlines()
        if any(
            token in line
            for token in (
                "stage23c_train.py",
                "stage23c_queue.py",
                "stage23c_pipeline.py",
            )
        )
        and "stage23d_phase1_queue.py" not in line
    ]


def completed(fold, expert):
    manifest = OUT / "phase1" / f"checkpoint_fold{fold}" / expert / "outer_evaluation_manifest.json"
    try:
        payload = json.loads(manifest.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "COMPLETED"
        and payload.get("outer_evaluation_access_count") == 1
        and payload.get("official_valid_access_count") == 0
        and payload.get("locked_test_access_count") == 0
    )


def write_state(status, current=None, note=None):
    done = [
        f"fold{fold}/{expert}"
        for fold in FOLDS
        for expert in EXPERTS
        if completed(fold, expert)
    ]
    atomic_json(
        RUNTIME / "phase1_queue_state.json",
        {
            "stage": "Stage23D-A Phase-1 static self-risk probes",
            "status": status,
            "current": current,
            "completed": done,
            "pending_count": 10 - len(done),
            "note": note,
            "official_valid_access_count": 0,
            "locked_test_access_count": 0,
            "updated_at": utc_now(),
        },
    )


def wait_for_safe_start():
    last_status = None
    while True:
        if not extraction_complete():
            status = "WAITING_FOR_EXTRACTION"
            note = "GPU3 extraction queue has not completed"
        else:
            processes = stage23c_active_processes()
            if processes:
                status = "WAITING_FOR_STAGE23C_FULL_RELEASE"
                note = f"{len(processes)} Stage23C train/queue/pipeline entries remain"
            else:
                return
        if status != last_status:
            write_state(status, note=note)
            print(f"[{utc_now()}] {status}: {note}", flush=True)
            last_status = status
        time.sleep(30)


def wait_for_extraction():
    while not extraction_complete():
        write_state(
            "WAITING_FOR_EXTRACTION",
            note="All ten schema-v2 extraction manifests are required",
        )
        print(f"[{utc_now()}] waiting for schema-v2 extraction", flush=True)
        time.sleep(30)


def run(command):
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    result = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {command}")


def main():
    wait_for_extraction()
    wait_for_safe_start()
    validation_path = OUT / "audit" / "feature_extraction_audit.json"
    validation_ok = False
    if validation_path.exists():
        try:
            validation_ok = json.loads(validation_path.read_text()).get(
                "status"
            ) == "PASS"
        except json.JSONDecodeError:
            validation_ok = False
    if not validation_ok:
        write_state("VALIDATING_SCHEMA_V2_FEATURES")
        run(
            [
                sys.executable,
                "-u",
                str(
                    ROOT
                    / "scripts"
                    / "mosei"
                    / "stage23d_validate_features.py"
                ),
            ]
        )
    for fold in FOLDS:
        for expert in EXPERTS:
            key = f"fold{fold}/{expert}"
            if completed(fold, expert):
                print(f"[{utc_now()}] SKIP complete {key}", flush=True)
                continue
            # Re-check at every boundary in case Stage23C launched a new train.
            wait_for_safe_start()
            write_state("RUNNING", key)
            for phase in ("select", "evaluate"):
                print(f"[{utc_now()}] START {key} {phase}", flush=True)
                run(
                    [
                        sys.executable,
                        "-u",
                        str(ROOT / "scripts" / "mosei" / "stage23d_phase1.py"),
                        "--phase",
                        phase,
                        "--checkpoint-fold",
                        str(fold),
                        "--expert-id",
                        expert,
                    ]
                )
            if not completed(fold, expert):
                raise RuntimeError(f"Completion manifest missing for {key}")
            print(f"[{utc_now()}] DONE {key}", flush=True)
    write_state("AGGREGATING")
    run(
        [
            sys.executable,
            "-u",
            str(ROOT / "scripts" / "mosei" / "stage23d_aggregate.py"),
            "--require-complete",
        ]
    )
    gate = json.loads((OUT / "analysis" / "phase1_gate.json").read_text())
    status = (
        "COMPLETED_NO_A5_AUTHORIZED"
        if gate["decision"] == "FAIL"
        else "COMPLETED_A5_ACTION_REQUIRED"
    )
    write_state(status, note=f"Phase-1 decision: {gate['decision']}")
    print(f"[{utc_now()}] {status}", flush=True)


if __name__ == "__main__":
    main()
