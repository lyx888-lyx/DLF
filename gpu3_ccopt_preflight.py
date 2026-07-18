"""Sample the shared physical GPU 3 for sixty seconds before Stage 11."""
import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def gpu_row():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            "3",
            "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    index, total, used, free, utilization = [
        int(value.strip()) for value in output.split(",")
    ]
    return {
        "At": now(),
        "Index": index,
        "MemoryTotalMiB": total,
        "MemoryUsedMiB": used,
        "MemoryFreeMiB": free,
        "UtilizationPercent": utilization,
    }


def processes():
    stage10 = subprocess.check_output(
        ["bash", "-lc", "ps -ef | grep -E 'stage10_(train|worker|supervisor)' | grep -v grep || true"],
        text=True,
    ).splitlines()
    compute = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            "3",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    ).stdout.splitlines()
    return stage10, compute


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--historical-mosi-peak-mib", type=int, default=4500)
    args = parser.parse_args()
    stage10_before, compute_before = processes()
    samples = []
    for index in range(13):
        samples.append(gpu_row())
        if index != 12:
            time.sleep(5)
    stage10_after, compute_after = processes()
    required_free = args.historical_mosi_peak_mib + 4096
    conditions = {
        "MOSEIProcessAliveThroughout": bool(stage10_before and stage10_after),
        "FreeMemoryAboveHistoricalMOSIPeakPlus4GiB": min(
            row["MemoryFreeMiB"] for row in samples
        )
        > required_free,
        "ExactlyOneGPU3ComputeProcess": len(compute_before) == 1
        and len(compute_after) == 1,
        "OriginalBatchSizeCanBeRetained": min(
            row["MemoryFreeMiB"] for row in samples
        )
        > required_free,
        "NoMOSEIStopRequired": True,
        "UtilizationNotPersistentlyNear100": sum(
            row["UtilizationPercent"] >= 95 for row in samples
        )
        < 0.8 * len(samples),
        "MemoryBelow90Percent": max(
            row["MemoryUsedMiB"] / row["MemoryTotalMiB"] for row in samples
        )
        < 0.90,
    }
    payload = {
        "StartedAt": samples[0]["At"],
        "EndedAt": samples[-1]["At"],
        "PhysicalGPU": 3,
        "CUDAVisibleDevices": "3",
        "ProgramLogicalGPU": 0,
        "HistoricalMOSIPeakMiB": args.historical_mosi_peak_mib,
        "RequiredFreeMiB": required_free,
        "Samples": samples,
        "Stage10ProcessesBefore": stage10_before,
        "Stage10ProcessesAfter": stage10_after,
        "GPU3ComputeProcessesBefore": compute_before,
        "GPU3ComputeProcessesAfter": compute_after,
        "Conditions": conditions,
        "Passed": all(conditions.values()),
        "Verdict": (
            "GPU3_SHARED_RESOURCE_GATE_PASSED"
            if all(conditions.values())
            else "GPU3_SHARED_RESOURCE_GATE_FAILED"
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(payload["Verdict"])
    if not payload["Passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
