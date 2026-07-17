"""Shared contracts for the asynchronous Stage 10 MOSEI pipeline."""
import hashlib
import json
import os
import subprocess
import signal
from datetime import datetime, timezone
from pathlib import Path


SEEDS = (1111, 1112, 1113, 1114, 1115)
STAGES = ("clean", "moddrop", "compatibility", "cfcompat")
STATES = (
    "PREPARING",
    "RUNNING_WORKERS",
    "WAITING_FOR_WORKERS",
    "TEST_UNLOCK_CHECK",
    "RUNNING_LOCKED_TEST",
    "BUILDING_PE5",
    "BUILDING_ADPEP",
    "AGGREGATING",
    "COMPLETED",
    "FAILED",
)
FROZEN_METHOD_COMMIT = "d3c2d62166c272ff16d210af8a6941efdb485108"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(str(temporary), str(path))


def append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def git_head(cwd="."):
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(cwd), text=True
    ).strip()


def stage_directory(result_root, stage, seed):
    mapping = {
        "clean": "clean_dlf",
        "moddrop": "moddrop",
        "compatibility": "compatibility_cache",
        "cfcompat": "cfcompat",
    }
    return Path(result_root) / mapping[stage] / "seed{}".format(int(seed))


def stage_manifest_path(result_root, stage, seed):
    return stage_directory(result_root, stage, seed) / "stage_manifest.json"


def sentinel_path(run_dir, seed, stage):
    return (
        Path(run_dir)
        / "sentinels"
        / "seed_{}".format(int(seed))
        / "{}.done".format(stage)
    )


def load_json(path):
    return json.loads(Path(path).read_text())


def write_state(run_dir, state, **fields):
    if state not in STATES:
        raise ValueError("Unknown Stage 10 state: {}".format(state))
    path = Path(run_dir) / "state.json"
    previous = load_json(path) if path.is_file() else {}
    payload = dict(previous)
    payload.update(fields)
    payload.update({"State": state, "UpdatedAt": utc_now()})
    atomic_json(path, payload)
    return payload


def pid_is_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def owned_pid(pid, worktree, run_dir):
    """Return True only for a live Stage 10 process belonging to this run."""
    if not pid_is_alive(pid):
        return False
    cmdline = Path("/proc") / str(int(pid)) / "cmdline"
    try:
        command = cmdline.read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
    except OSError:
        return False
    return (
        str(Path(worktree).resolve()) in command
        and str(Path(run_dir).resolve()) in command
        and "stage10_" in command
    )


def validate_stage_manifest(path, seed, stage, commit):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = load_json(path)
    if (
        int(manifest.get("Seed", -1)) != int(seed)
        or manifest.get("Stage") != stage
        or manifest.get("Commit") != commit
        or manifest.get("Dataset") != "mosei"
        or not manifest.get("ValidationSelected")
        or not manifest.get("NoTestAccess")
    ):
        raise RuntimeError("Invalid stage manifest: {}.".format(path))
    for output in manifest.get("Outputs", []):
        output_path = Path(output["Path"])
        if not output_path.is_file() or sha256(output_path) != output["SHA256"]:
            raise RuntimeError("Stage output SHA mismatch: {}.".format(output_path))
    return manifest


def validate_sentinel(path, result_root, seed, stage, commit):
    path = Path(path)
    if not path.is_file():
        return None
    sentinel = load_json(path)
    if (
        int(sentinel.get("Seed", -1)) != int(seed)
        or sentinel.get("Stage") != stage
        or sentinel.get("Commit") != commit
        or int(sentinel.get("ExitCode", -1)) != 0
    ):
        raise RuntimeError("Invalid completion sentinel: {}.".format(path))
    validate_stage_manifest(
        stage_manifest_path(result_root, stage, seed), seed, stage, commit
    )
    return sentinel
