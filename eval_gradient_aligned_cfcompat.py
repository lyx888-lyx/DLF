"""Student-only Stage 6B checkpoint re-evaluation."""
import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from train_gradient_aligned_cfcompat import (
    GRADIENT_POLICIES, _flatten, build_config, method_paths, prediction_rows,
)
from trains.singleTask.missing_utils import (
    MissingModalityWrapper, build_single_split_loader, evaluate_all_modes,
    validation_objective,
)
from trains.singleTask.model.DLF import DLF


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Stage 6B Student checkpoint.")
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--gradient-policy", choices=GRADIENT_POLICIES, required=True)
    parser.add_argument("--checkpoint-kind", choices=("valid", "diagnostic"), default="valid")
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--config-file", default="config/config.json")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main():
    cli = parse_args(); args = build_config(cli, cli.seed)
    result_dir, main_template, diagnostic_template = method_paths(cli, cli.dataset)
    template = main_template if cli.checkpoint_kind == "valid" else diagnostic_template
    checkpoint = Path(str(template).format(cli.seed))
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    model = MissingModalityWrapper(DLF(args).to(args.device), args.feature_dims[1], args.feature_dims[2]).to(args.device)
    model.load_state_dict(torch.load(checkpoint, map_location=args.device), strict=True)
    criterion = nn.L1Loss()
    valid_loader = build_single_split_loader(args, "valid", cli.num_workers)
    test_loader = build_single_split_loader(args, "test", cli.num_workers)
    valid = evaluate_all_modes(model, valid_loader, args.device, "moddrop", criterion)
    test = evaluate_all_modes(model, test_loader, args.device, "moddrop", criterion)
    predictions = prediction_rows(model, valid_loader if cli.checkpoint_kind == "valid" else test_loader, args.device)
    if cli.checkpoint_kind == "diagnostic":
        predictions["selected_by"] = "test"; predictions["diagnostic_only"] = True; predictions["not_main_result"] = True
    row = {"Seed": cli.seed, "GradientPolicy": cli.gradient_policy, "Checkpoint": str(checkpoint),
           "CheckpointKind": cli.checkpoint_kind, "StudentOnly": True,
           "J_valid": validation_objective(valid), "J_test": validation_objective(test),
           **_flatten(valid, "valid"), **_flatten(test, "test")}
    result_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(result_dir / "mosi_reeval_{}_seed{}.csv".format(cli.checkpoint_kind, cli.seed), index=False)
    predictions.to_csv(result_dir / "mosi_reeval_{}_seed{}_predictions.csv".format(cli.checkpoint_kind, cli.seed), index=False)
    print("checkpoint_kind={} J_valid={:.9f} J_test={:.9f}".format(cli.checkpoint_kind, row["J_valid"], row["J_test"]))


if __name__ == "__main__":
    main()
