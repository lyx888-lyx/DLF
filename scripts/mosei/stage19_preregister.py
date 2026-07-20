"""Freeze Stage 19 screen and training protocol before candidate results."""

import argparse
import json
import math
import os
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def main():
    cli = parse_args()
    root = Path(cli.artifact_root)
    epochs = []
    sources = []
    for seed in (1111, 1112, 1113, 1114, 1115):
        path = root / "cfcompat" / "seed{}".format(seed) / "stage_manifest.json"
        manifest = json.loads(path.read_text())
        epoch = int(manifest["BestValidEpoch"])
        epochs.append(epoch)
        sources.append({"seed": seed, "path": str(path), "selected_epoch": epoch})
    median = float(statistics.median(epochs))
    screen_1 = max(2, int(math.ceil(0.25 * median)))
    screen_2 = max(screen_1 + 1, int(math.ceil(0.50 * median)))
    maximum = 30
    if screen_2 > math.floor(0.60 * maximum):
        raise RuntimeError("Frozen screen_2 exceeds 60% of the conventional budget.")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    payload = {
        "protocol": "MOSEI Multi-Granular Recoverable Distillation v1",
        "code_commit": head,
        "selected_epoch_source": "historical MOSEI proposed CFCompat validation-selected epochs; no MOSEI Uniform trajectory existed",
        "historical_sources": sources,
        "selected_epochs": epochs,
        "selected_epoch_median": median,
        "screen_epoch_1": screen_1,
        "screen_epoch_2": screen_2,
        "maximum_regular_training_epochs": maximum,
        "batch_size": 16,
        "update_epochs": 10,
        "accumulation_semantics": "sum_not_mean",
        "learning_rate": 0.0001,
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau(mode=min,factor=0.5,patience=5)",
        "early_stop": 10,
        "tau": 0.5,
        "lambda_kd": 1.0,
        "mgd_weights": {"continuous": 0.5, "ordinal_total": 0.5, "granularity": "equal"},
        "amp_candidate_order": ["bf16_if_supported", "fp16_grad_scaler", "off_fallback"],
        "bf16_supported_by_runtime": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "headroom_gate": {
            "aggregate_coarse_only_min": 0.10,
            "at_least_modes": 2,
            "per_mode_coarse_only_min": 0.05,
            "c_cv_min": 0.10,
            "per_mode_granularity_nonzero_coverage_min": 0.10
        },
        "locked_test_access_count": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(cli.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
