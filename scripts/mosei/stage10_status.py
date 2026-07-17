"""Print one non-mutating Stage 10 status snapshot."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mosei.stage10_common import load_json, pid_is_alive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    result = {
        "RunDir": str(run),
        "State": load_json(run / "state.json"),
        "Allocation": load_json(run / "gpu_allocation.json"),
        "PIDs": {},
    }
    for path in sorted(run.glob("*.pid")):
        pid = int(path.read_text().strip())
        result["PIDs"][path.name] = {"PID": pid, "Alive": pid_is_alive(pid)}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
