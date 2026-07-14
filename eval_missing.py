"""Evaluation entrypoint for Stage 1 DLF-DirectMask and DLF-ModDrop.

The test split is deliberately blocked unless the caller passes the explicit
--confirm-test-once acknowledgement.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from config import get_config_regression
from trains.singleTask.fixed_kd_utils import fixed_kd_checkpoint_path
from trains.singleTask.missing_utils import (
    MissingModalityWrapper,
    build_single_split_loader,
    clean_checkpoint_path,
    evaluate_all_modes,
    flatten_mode_metrics,
    missing_checkpoint_path,
    write_result_csvs,
)
from trains.singleTask.model.DLF import DLF
from utils.functions import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a Stage 1 missing-modality baseline.")
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1111])
    parser.add_argument("--method", choices=("directmask", "moddrop", "fixedkd"), required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--confirm-test-once", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-dir", default="result/missing_baseline")
    parser.add_argument("--config-file", default="config/config.json")
    parsed = parser.parse_args()
    if parsed.split == "test" and not parsed.confirm_test_once:
        parser.error("Refusing test evaluation without --confirm-test-once.")
    return parsed


def build_config(cli_args, seed):
    args = get_config_regression("DLF", cli_args.dataset, cli_args.config_file)
    args.feature_T = ""
    args.feature_A = ""
    args.feature_V = ""
    args.mode = cli_args.split
    args.is_training = False
    args.train_mode = "regression"
    args.seed = int(seed)
    args.cur_seed = int(seed)
    args.device = assign_gpu(list(cli_args.gpu_ids))
    return args


def load_model(cli_args, args, seed):
    if cli_args.method == "directmask":
        checkpoint = clean_checkpoint_path(cli_args.model_save_dir, args.dataset_name, seed)
        if not checkpoint.is_file():
            raise FileNotFoundError("Gate 3 validation-best checkpoint not found: {}".format(checkpoint))
        model = DLF(args).to(args.device)
        model.load_state_dict(torch.load(checkpoint, map_location=args.device), strict=True)
        return model, checkpoint

    if cli_args.method == "moddrop":
        checkpoint = missing_checkpoint_path(cli_args.model_save_dir, args.dataset_name, seed)
    elif cli_args.method == "fixedkd":
        checkpoint = fixed_kd_checkpoint_path(cli_args.model_save_dir, args.dataset_name, seed)
    else:
        raise ValueError("Unsupported student-only evaluation method: {}".format(cli_args.method))
    if not checkpoint.is_file():
        raise FileNotFoundError("{} student checkpoint not found: {}".format(cli_args.method, checkpoint))

    backbone = DLF(args).to(args.device)
    model = MissingModalityWrapper(backbone, args.feature_dims[1], args.feature_dims[2]).to(args.device)
    model.load_state_dict(torch.load(checkpoint, map_location=args.device), strict=True)
    return model, checkpoint


def main():
    cli_args = parse_args()
    rows = []
    for seed in cli_args.seeds:
        setup_seed(seed)
        args = build_config(cli_args, seed)
        dataloader = build_single_split_loader(args, cli_args.split, cli_args.num_workers)
        model, checkpoint = load_model(cli_args, args, seed)
        metrics = evaluate_all_modes(model, dataloader, args.device, "moddrop" if cli_args.method == "fixedkd" else cli_args.method, nn.L1Loss())
        row = {"Seed": int(seed), "Checkpoint": str(checkpoint)}
        row.update(flatten_mode_metrics(metrics))
        rows.append(row)
        print(
            "seed={} method={} split={} checkpoint={}".format(
                seed, cli_args.method, cli_args.split, checkpoint
            )
        )
        for mode, values in metrics.items():
            print(
                "{} acc_7={:.4f} acc_5={:.4f} acc_2={:.4f} F1={:.4f} Corr={:.4f} MAE={:.4f} Loss={:.4f}".format(
                    mode,
                    values["acc_7"],
                    values["acc_5"],
                    values["acc_2"],
                    values["F1_score"],
                    values["Corr"],
                    values["MAE"],
                    values["Loss"],
                )
            )
        del model
        torch.cuda.empty_cache()

    output_dir = Path(cli_args.result_dir) / cli_args.method / cli_args.split
    write_result_csvs(rows, output_dir, cli_args.dataset)
    print("results={}".format(output_dir))


if __name__ == "__main__":
    main()
