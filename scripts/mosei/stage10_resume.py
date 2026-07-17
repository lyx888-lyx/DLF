"""Detach the same supervisor configuration after a verified interruption."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mosei.stage10_common import SEEDS, git_head, pid_is_alive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    config = json.loads((run / "config.json").read_text())
    if tuple(config["Seeds"]) != SEEDS or git_head(ROOT) != config["LaunchCommit"]:
        raise RuntimeError("Resume configuration/commit differs from original launch.")
    pid_path = run / "supervisor.pid"
    if pid_path.is_file() and pid_is_alive(int(pid_path.read_text())):
        raise RuntimeError("Supervisor is already alive.")
    command = [
        sys.executable,
        str(ROOT / "scripts/mosei/stage10_supervisor.py"),
        "--run-dir",
        str(run),
    ]
    with (run / "supervisor.log").open("a") as log:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path.write_text(str(process.pid) + "\n")
    print("Resumed Stage 10 supervisor PID {}".format(process.pid))


if __name__ == "__main__":
    main()
