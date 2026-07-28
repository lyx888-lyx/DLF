"""Rebuild the V7.1 teacher cache from the frozen run summary.

This helper is intentionally separate from the V7.3 evaluator so an existing
cache remains untouched, while a missing cache can be regenerated from the
five teacher checkpoints recorded in complementarity_v71_summary.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# When this file is executed as ``python3 scripts/rebuild_v71_teacher_cache.py``,
# Python places ``scripts/`` rather than the repository root on sys.path.
# Add the root explicitly so project modules such as config.py are importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import get_config_regression
from data_loader import MMDataLoader
from trains.singleTask.function_consensus_system_v7 import build_teacher_cache
from utils import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild a missing V7.1 teacher cache.")
    parser.add_argument("--dataset", choices=["mosi", "mosei"], default="mosi")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=str, default="./config/config.json")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="./result/complementarity_v71/mosi/seed_1111",
    )
    parser.add_argument("--summary", type=str, default="")
    parser.add_argument("--teacher-cache", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    cli = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("MMSA")
    setup_seed(cli.seed)
    device = assign_gpu([cli.gpu])

    run_dir = Path(cli.run_dir)
    summary_path = Path(cli.summary) if cli.summary else run_dir / "complementarity_v71_summary.json"
    cache_path = Path(cli.teacher_cache) if cli.teacher_cache else run_dir / "complementarity_v71_teacher_cache.pth"

    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    teacher_paths = [Path(value) for value in summary.get("teacher_paths", [])]
    if len(teacher_paths) < 3:
        raise RuntimeError(
            "The V7.1 summary does not contain at least three teacher checkpoints. "
            f"Found: {teacher_paths}"
        )

    missing = [path for path in teacher_paths if not path.is_file()]
    if missing:
        lines = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Teacher checkpoints recorded in the summary are missing. "
            "Pass a corrected summary or restore these files:\n" + lines
        )

    if cache_path.is_file() and not cli.force:
        logger.info("Teacher cache already exists: %s", cache_path)
        return

    args = get_config_regression("DLF", cli.dataset, Path(cli.config))
    args["device"] = device
    args["train_mode"] = "regression"
    args["feature_T"] = ""
    args["feature_A"] = ""
    args["feature_V"] = ""
    args["seed"] = cli.seed
    args["cur_seed"] = 1
    args["batch_size"] = cli.batch_size

    dataloaders = MMDataLoader(args, cli.num_workers)
    build_teacher_cache(
        args,
        dataloaders,
        teacher_paths,
        device,
        cache_path,
        rebuild=True,
    )
    logger.info("Rebuilt V7.1 teacher cache: %s", cache_path)


if __name__ == "__main__":
    main()
