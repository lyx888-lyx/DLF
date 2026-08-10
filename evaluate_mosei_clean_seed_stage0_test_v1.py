"""Aggregate-only MOSEI Test reproduction check for a frozen clean DLF checkpoint.

This is a baseline reproduction evaluation, not a tuning entry point.
The checkpoint must match the SHA recorded by the Stage-0 training manifest.
Only the Test split is constructed and only aggregate metrics are persisted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from config import get_config_regression
from data_loader import MMDataLoader
from trains import ATIO
from trains.singleTask.fixed_kd_utils import checkpoint_sha256
from trains.singleTask.model.DLF import DLF
from utils.functions import assign_gpu, setup_seed


PAPER_DLF = {
    "acc_7": 0.5390,
    "acc_5": 0.5570,
    "acc_2": 0.8542,
    "F1_score": 0.8527,
    "Corr": 0.7640,
    "MAE": 0.5360,
}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate frozen clean DLF on MOSEI Test once, aggregate only")
    p.add_argument("--seed", type=int, default=1111)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--config-file", default="config/config.json")
    p.add_argument("--model-root", default="pt")
    p.add_argument("--result-root", default="result")
    p.add_argument("--matmul-precision", choices=("highest", "high", "medium"), default="high")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.num_workers != 0:
        p.error("Windows reproduction runner fixes --num-workers=0.")
    return args


def main():
    cli = parse_args()
    seed = int(cli.seed)
    setup_seed(seed)
    torch.set_float32_matmul_precision(cli.matmul_precision)

    checkpoint = Path(cli.model_root) / f"DLF_mosei_seed{seed}_best.pth"
    manifest_path = (
        Path(cli.result_root)
        / "missing_baseline"
        / "mosei_clean_stage0_v1"
        / f"seed{seed}"
        / "run_manifest.json"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Stage-0 manifest required before Test: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_sha = str(manifest.get("checkpoint_sha256", ""))
    actual_sha = checkpoint_sha256(checkpoint)
    if not expected_sha or actual_sha != expected_sha:
        raise RuntimeError(
            f"Checkpoint SHA mismatch: manifest={expected_sha!r} actual={actual_sha!r}"
        )
    if str(manifest.get("selected_by")) != "validation_Loss":
        raise RuntimeError(f"Unexpected checkpoint selection rule: {manifest.get('selected_by')!r}")

    args = get_config_regression("DLF", "mosei", cli.config_file)
    args.mode = "test"
    args.is_training = False
    args.train_mode = "regression"
    args.seed = seed
    args.cur_seed = 1
    args.feature_T = args.feature_A = args.feature_V = ""
    args.device = assign_gpu([int(cli.gpu_id)])

    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"test"}:
        raise RuntimeError(f"Test-only loader expected, got {sorted(loaders)}")

    model = DLF(args).to(args.device)
    state = torch.load(checkpoint, map_location=args.device)
    model.load_state_dict(state, strict=True)
    model.eval()

    trainer = ATIO().getTrain(args)
    metrics = trainer.do_test(model, loaders["test"], mode="TEST")
    result = {key: float(metrics[key]) for key in PAPER_DLF}

    output = (
        Path(cli.result_root)
        / "missing_baseline"
        / "mosei_clean_stage0_v1"
        / f"seed{seed}"
        / "test_reference"
    )
    if output.exists() and not cli.overwrite:
        raise FileExistsError(f"Output exists; inspect it or use --overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)

    deltas = {}
    for metric, paper in PAPER_DLF.items():
        value = result[metric]
        # Raw delta: positive is better for all except MAE.
        deltas[metric] = value - paper

    summary = {
        "protocol": "MOSEI_CLEAN_DLF_BASELINE_TEST_REPRODUCTION",
        "dataset": "mosei",
        "split": "test",
        "seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_sha,
        "checkpoint_selected_by": "validation_Loss",
        "test_used_for_tuning": False,
        "sample_level_test_output_written": False,
        "matmul_precision": cli.matmul_precision,
        "metrics": result,
        "paper_DLF_reference_from_user_table": PAPER_DLF,
        "raw_delta_ours_minus_paper": deltas,
    }
    summary_path = output / "clean_dlf_test_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("================ MOSEI clean DLF Test reproduction ================")
    print(f"seed:       {seed}")
    print(f"checkpoint: {checkpoint}")
    print(f"sha256:     {actual_sha}")
    print("")
    print("Metric        Ours            Paper DLF       Delta ours-paper")
    print("----------------------------------------------------------------")
    for metric in ("acc_7", "acc_5", "acc_2", "F1_score", "Corr", "MAE"):
        print(f"{metric:10s} {result[metric]:.9f}    {PAPER_DLF[metric]:.9f}    {deltas[metric]:+.9f}")
    print("")
    print("TEST USED FOR TUNING: False")
    print("SAMPLE-LEVEL TEST OUTPUT WRITTEN: False")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
