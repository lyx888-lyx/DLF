"""Persistent Stage 10 supervisor."""
import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mosei.stage10_common import (
    append_jsonl,
    atomic_json,
    git_head,
    utc_now,
    write_state,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    config = json.loads((run_dir / "config.json").read_text())
    if git_head(ROOT) != config["LaunchCommit"]:
        raise RuntimeError("Supervisor checkout moved after launch.")
    write_state(run_dir, "PREPARING", Error=None)
    workers = []
    write_state(run_dir, "RUNNING_WORKERS")
    for allocation in config["WorkerQueues"]:
        gpu = int(allocation["GPU"])
        log_path = run_dir / "worker_gpu_{}.log".format(gpu)
        command = [
            sys.executable,
            str(ROOT / "scripts/mosei/stage10_worker.py"),
            "--run-dir",
            str(run_dir),
            "--result-root",
            config["ResultRoot"],
            "--config-file",
            config["ConfigFile"],
            "--physical-gpu",
            str(gpu),
            "--seeds",
            *[str(value) for value in allocation["Seeds"]],
            "--commit",
            config["LaunchCommit"],
        ]
        append_jsonl(
            run_dir / "commands.jsonl",
            {"Type": "worker", "GPU": gpu, "Command": command, "At": utc_now()},
        )
        with log_path.open("a") as log:
            process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        (run_dir / "worker_gpu_{}.pid".format(gpu)).write_text(
            str(process.pid) + "\n"
        )
        workers.append((gpu, process))
    failures = []
    for gpu, process in workers:
        code = process.wait()
        if code:
            failures.append({"GPU": gpu, "ExitCode": code})
    if failures:
        raise RuntimeError("One or more workers failed: {}".format(failures))
    write_state(run_dir, "TEST_UNLOCK_CHECK")
    coordinator_log = run_dir / "coordinator.log"
    command = [
        sys.executable,
        str(ROOT / "scripts/mosei/stage10_coordinator.py"),
        "--run-dir",
        str(run_dir),
    ]
    append_jsonl(
        run_dir / "commands.jsonl",
        {"Type": "coordinator", "Command": command, "At": utc_now()},
    )
    with coordinator_log.open("a") as log:
        environment = dict(__import__("os").environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(config["WorkerQueues"][0]["GPU"])
        coordinator = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (run_dir / "coordinator.pid").write_text(str(coordinator.pid) + "\n")
    code = coordinator.wait()
    if code:
        raise RuntimeError("Coordinator exited {}.".format(code))


if __name__ == "__main__":
    arguments = parse_args()
    try:
        main()
    except Exception as error:
        run = Path(arguments.run_dir)
        append_jsonl(
            run / "failures.jsonl",
            {
                "Component": "supervisor",
                "Error": repr(error),
                "Traceback": traceback.format_exc(),
                "At": utc_now(),
            },
        )
        write_state(run, "FAILED", Error=repr(error))
        raise
