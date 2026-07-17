"""Safely terminate only PIDs owned by this Stage 10 run."""
import argparse
import os
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mosei.stage10_common import load_json, owned_pid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    config = load_json(run / "config.json")
    stopped = []
    for path in sorted(run.glob("*.pid"), reverse=True):
        pid = int(path.read_text().strip())
        if owned_pid(pid, config["Worktree"], run):
            os.kill(pid, signal.SIGTERM)
            stopped.append(pid)
    print("Terminated owned Stage 10 PIDs: {}".format(stopped))


if __name__ == "__main__":
    main()
