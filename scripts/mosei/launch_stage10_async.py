"""Validate and detach the immutable Stage 10 MOSEI supervisor."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mosei.stage10_common import (
    FROZEN_METHOD_COMMIT,
    SEEDS,
    atomic_json,
    git_head,
    sha256,
    utc_now,
    write_state,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mosei",), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--max-mosei-gpus", type=int, default=2)
    parser.add_argument("--reserve-free-gpus", type=int, default=1)
    parser.add_argument(
        "--frozen-method-commit", required=True, default=FROZEN_METHOD_COMMIT
    )
    parser.add_argument("--mosei-gpus", nargs="+", type=int, default=[3])
    parser.add_argument("--test-once", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument(
        "--config-file", default="config/config.json"
    )
    parser.add_argument(
        "--data-file",
        default="/data4t/lyx/datasets/MOSEI/Processed/aligned_50.pkl",
    )
    parser.add_argument(
        "--tests-passed-marker",
        default="runtime/stage10_preflight/TESTS_PASSED.json",
    )
    return parser.parse_args()


def gpu_snapshot():
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    gpus = []
    for line in query.splitlines():
        index, uuid, name, total, used, utilization = [
            value.strip() for value in line.split(",", 5)
        ]
        gpus.append(
            {
                "Index": int(index),
                "UUID": uuid,
                "Name": name,
                "MemoryTotalMiB": int(total),
                "MemoryUsedMiB": int(used),
                "UtilizationPercent": int(utilization),
            }
        )
    processes = []
    process_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    for line in process_query.stdout.splitlines():
        if not line.strip():
            continue
        uuid, pid, name, memory = [value.strip() for value in line.split(",", 3)]
        processes.append(
            {
                "GPUUUID": uuid,
                "PID": int(pid),
                "Process": name,
                "MemoryMiB": int(memory),
            }
        )
    return gpus, processes


def main():
    args = parse_args()
    if tuple(args.seeds) != SEEDS:
        raise RuntimeError("Formal Stage 10 seeds/order must be 1111..1115.")
    if args.frozen_method_commit != FROZEN_METHOD_COMMIT:
        raise RuntimeError("Frozen method commit differs.")
    if not args.test_once or not args.detach:
        raise RuntimeError("Formal launch requires --test-once and --detach.")
    if len(args.mosei_gpus) > args.max_mosei_gpus:
        raise RuntimeError("MOSEI GPU count exceeds --max-mosei-gpus.")
    if args.mosei_gpus != [3]:
        raise RuntimeError("This approved launch is restricted to physical GPU 3.")
    head = git_head(ROOT)
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", FROZEN_METHOD_COMMIT, head],
        cwd=str(ROOT),
        check=False,
    ).returncode:
        raise RuntimeError("Launch commit does not descend from frozen method.")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=str(ROOT), text=True
    ).strip():
        raise RuntimeError("Worktree must be clean before formal launch.")
    data_file = Path(args.data_file).resolve()
    config_file = Path(args.config_file).resolve()
    tests_marker = Path(args.tests_passed_marker).resolve()
    for path in (data_file, config_file, tests_marker):
        if not path.is_file():
            raise FileNotFoundError(path)
    tests = json.loads(tests_marker.read_text())
    if not tests.get("Passed") or tests.get("Commit") != head:
        raise RuntimeError("Tests-passed marker is absent or belongs to another commit.")
    gpus, processes = gpu_snapshot()
    by_uuid = {row["UUID"]: row["Index"] for row in gpus}
    active_on = {
        by_uuid[row["GPUUUID"]]
        for row in processes
        if row["GPUUUID"] in by_uuid
    }
    if any(gpu in active_on for gpu in args.mosei_gpus):
        raise RuntimeError("Approved MOSEI GPU has an active compute process.")
    reserved = [
        row["Index"] for row in gpus if row["Index"] not in args.mosei_gpus
    ]
    if len(reserved) < args.reserve_free_gpus:
        raise RuntimeError("Insufficient RESERVED_FOR_MOSI GPUs.")
    run_dir = Path(args.run_dir).resolve()
    result_root = Path(args.result_root).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("Run directory already contains state; use resume.")
    run_dir.mkdir(parents=True, exist_ok=True)
    queues = [{"GPU": 3, "Seeds": list(SEEDS)}]
    allocation = {
        "QueriedAt": utc_now(),
        "AllGPUs": gpus,
        "ActiveComputeProcesses": processes,
        "MOSEIGPUs": [3],
        "RESERVED_FOR_MOSI": reserved,
        "WorkerQueues": queues,
    }
    atomic_json(run_dir / "gpu_allocation.json", allocation)
    config = {
        "Dataset": "mosei",
        "Seeds": list(SEEDS),
        "FrozenMethodCommit": FROZEN_METHOD_COMMIT,
        "LaunchCommit": head,
        "Worktree": str(ROOT),
        "RunDir": str(run_dir),
        "ResultRoot": str(result_root),
        "ConfigFile": str(config_file),
        "ConfigSHA256": sha256(config_file),
        "DataFile": str(data_file),
        "DataSHA256": sha256(data_file),
        "TestOnce": True,
        "TestsPassed": tests,
        "WorkerQueues": queues,
        "RESERVED_FOR_MOSI": reserved,
        "CreatedAt": utc_now(),
    }
    atomic_json(run_dir / "config.json", config)
    atomic_json(run_dir / "completed_stages.json", {})
    write_state(run_dir, "PREPARING", Error=None)
    log_path = run_dir / "supervisor.log"
    command = [
        sys.executable,
        str(ROOT / "scripts/mosei/stage10_supervisor.py"),
        "--run-dir",
        str(run_dir),
    ]
    with log_path.open("a") as log:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (run_dir / "supervisor.pid").write_text(str(process.pid) + "\n")
    print(json.dumps({"SupervisorPID": process.pid, **allocation}, sort_keys=True))


if __name__ == "__main__":
    main()
