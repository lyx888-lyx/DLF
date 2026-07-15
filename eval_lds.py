"""Validation-only evaluator for DLF-LDS-ModDrop-v1.

The split is intentionally fixed to validation; this program has no held-out
split option and builds only the train labels needed for its fixed density map.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.lds_utils import (
    assert_fixed_lds_config,
    evaluate_lds_validation,
    lds_checkpoint_path,
    lds_v1_config,
    prepare_train_label_weights,
)
from trains.singleTask.missing_utils import MissingModalityWrapper, build_single_split_loader, flatten_mode_metrics, clean_checkpoint_path, write_result_csvs
from trains.singleTask.model.DLF import DLF
from utils.functions import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DLF-LDS-ModDrop-v1 on validation only.")
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1111])
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-dir", default="result/missing_baseline/lds_moddrop_v1/valid")
    parser.add_argument("--config-file", default="config/config.json")
    parsed = parser.parse_args()
    assert_fixed_lds_config(lds_v1_config())
    return parsed


def build_config(cli_args, seed):
    args = get_config_regression("DLF", cli_args.dataset, cli_args.config_file)
    args.feature_T = args.feature_A = args.feature_V = ""
    args.mode = "valid"
    args.is_training = False
    args.train_mode = "regression"
    args.seed = int(seed)
    args.cur_seed = int(seed)
    args.device = assign_gpu(list(cli_args.gpu_ids))
    return args


def load_model(cli_args, args, seed):
    clean = clean_checkpoint_path(cli_args.model_save_dir, args.dataset_name, seed)
    checkpoint = lds_checkpoint_path(cli_args.model_save_dir, args.dataset_name, seed)
    if not clean.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("Required clean or LDS validation-best checkpoint is missing.")
    backbone = DLF(args).to(args.device)
    backbone.load_state_dict(torch.load(clean, map_location=args.device), strict=True)
    model = MissingModalityWrapper(backbone, args.feature_dims[1], args.feature_dims[2]).to(args.device)
    model.load_state_dict(torch.load(checkpoint, map_location=args.device), strict=True)
    return model, checkpoint


def main():
    cli_args = parse_args()
    rows, bin_rows, group_rows = [], [], []
    for seed in cli_args.seeds:
        setup_seed(seed)
        args = build_config(cli_args, seed)
        train_dataset = MMDataset(args, mode="train")
        artifacts = prepare_train_label_weights(train_dataset.labels["M"])
        valid_loader = build_single_split_loader(args, "valid", cli_args.num_workers)
        model, checkpoint = load_model(cli_args, args, seed)
        metrics, bins, groups = evaluate_lds_validation(model, valid_loader, args.device, nn.L1Loss(), artifacts, seed, 0)
        row = {"Seed": int(seed), "Checkpoint": str(checkpoint)}
        row.update(flatten_mode_metrics(metrics))
        rows.append(row)
        bin_rows.extend(bins)
        group_rows.extend(groups)
        print("seed={} split=valid checkpoint={}".format(seed, checkpoint))
        for mode, values in metrics.items():
            print("{} MAE={:.4f} Corr={:.4f} Acc2={:.4f} MacroBinMAE={:.4f}".format(mode, values["MAE"], values["Corr"], values["acc_2"], values["MacroBinMAE"]))
        del model
        torch.cuda.empty_cache()
    output_dir = Path(cli_args.result_dir)
    write_result_csvs(rows, output_dir, cli_args.dataset)
    pd.DataFrame(bin_rows).to_csv(output_dir / "{}_bin_metrics.csv".format(cli_args.dataset), index=False)
    pd.DataFrame(group_rows).to_csv(output_dir / "{}_density_group_metrics.csv".format(cli_args.dataset), index=False)
    print("results={}".format(output_dir))


if __name__ == "__main__":
    main()
