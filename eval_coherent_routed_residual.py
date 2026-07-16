"""Student-only CCRRD evaluation without Teacher, evaluator, or train caches."""
import argparse
import json
from pathlib import Path

import torch

from config import get_config_regression
from trains.singleTask.cf_residual_utils import (
    evaluate_residual_modes,
    evaluation_objectives,
    load_residual_student_for_eval,
)
from trains.singleTask.coherent_routed_residual_utils import ROUTE_VARIANTS
from trains.singleTask.missing_utils import build_single_split_loader
from utils.functions import assign_gpu


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Independent Stage 4A.1 Student evaluation.")
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--route-variant", choices=tuple(ROUTE_VARIANTS), required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--confirm-test-once", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--config-file", default="config/config.json")
    cli = parser.parse_args(argv)
    if cli.split == "test" and not cli.confirm_test_once:
        parser.error("Refusing test evaluation without --confirm-test-once.")
    return cli


def build_config(cli):
    args = get_config_regression("DLF", cli.dataset, cli.config_file)
    args.mode = cli.split; args.feature_T = args.feature_A = args.feature_V = ""
    args.is_training = False; args.train_mode = "regression"
    args.seed = args.cur_seed = 1111; args.device = assign_gpu(list(cli.gpu_ids))
    return args


def resolve_checkpoint(cli):
    if cli.checkpoint:
        return Path(cli.checkpoint)
    version = ROUTE_VARIANTS[cli.route_variant]["version"]
    return Path(cli.model_save_dir) / "missing_baseline" / version / "DLF_mosi_seed1111_best_valid.pth"


def main(argv=None):
    cli = parse_args(argv); args = build_config(cli); checkpoint = resolve_checkpoint(cli)
    if not checkpoint.is_file():
        raise FileNotFoundError("Stage 4A.1 Student checkpoint absent: {}".format(checkpoint))
    model = load_residual_student_for_eval(args, checkpoint)
    loader = build_single_split_loader(args, cli.split, cli.num_workers)
    evaluation = evaluate_residual_modes(model, loader, args.device, torch.nn.L1Loss())
    corrected_j, base_j = evaluation_objectives(evaluation)
    print(json.dumps({"checkpoint": str(checkpoint), "route_variant": cli.route_variant,
                      "split": cli.split, "J_corrected": corrected_j, "J_base": base_j,
                      "metrics": evaluation}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
