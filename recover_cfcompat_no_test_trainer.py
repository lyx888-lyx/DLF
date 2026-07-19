"""CLI for one immutable Stage18A no-Test recovery run."""

import argparse
from pathlib import Path

from trains.singleTask.cfcompat_fair_trainer import train_no_test


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", choices=("moddrop", "cfcompat"), required=True
    )
    parser.add_argument("--run-label", choices=("A", "B"), required=True)
    parser.add_argument("--seed", type=int, choices=(1114,), default=1114)
    parser.add_argument("--num-workers", type=int, choices=(1,), default=1)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--config-file", default="config/config.json")
    return parser.parse_args()


def main():
    cli = parse_args()
    if cli.gpu_ids != [0]:
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must expose physical GPU 3 as internal GPU 0."
        )
    root = Path("result/missing_baseline/cfcompat_evidence_v1/mosi")
    run = (
        root
        / "stage18a_training_recovery"
        / "{}_run{}".format(cli.method, cli.run_label)
    )
    checkpoint = (
        Path("runtime/cfcompat_evidence_v1/checkpoints/mosi/stage18a")
        / "{}_run{}".format(cli.method, cli.run_label)
    )
    train_no_test(cli, cli.method, cli.run_label, run, checkpoint)


if __name__ == "__main__":
    main()
