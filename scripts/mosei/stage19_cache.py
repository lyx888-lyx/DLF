"""Build Stage 19 train/valid frozen prediction caches.

The accepted split set is deliberately closed and cannot include test.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.cf_compat_kd_utils import evaluator_prediction
from trains.singleTask.fixed_kd_utils import build_frozen_teacher, teacher_lav_prediction
from trains.singleTask.mgrd_utils import (
    canonical_ids,
    ordered_id_sha,
    sha256_file,
    unordered_id_sha,
)
from trains.singleTask.missing_utils import MissingModalityWrapper
from trains.singleTask.model.DLF import DLF
from utils.functions import assign_gpu, setup_seed


ALLOWED_SPLITS = ("train", "valid")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int, choices=(1111, 1114))
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def git_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()


def build_args(cli):
    args = get_config_regression("DLF", "mosei", cli.config_file)
    args.mode = "train"
    args.feature_T = args.feature_A = args.feature_V = ""
    args.is_training = True
    args.train_mode = "regression"
    args.seed = args.cur_seed = int(cli.seed)
    args.device = assign_gpu([int(cli.gpu_id)])
    return args


def locked_loader(args, split, num_workers):
    if split not in ALLOWED_SPLITS:
        raise RuntimeError("Stage 19 cache permits train/valid only.")
    dataset = MMDataset(args, mode=split)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )


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


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def load_input_manifests(artifact_root, seed):
    root = Path(artifact_root)
    clean = json.loads((root / "clean_dlf" / "seed{}".format(seed) / "stage_manifest.json").read_text())
    moddrop = json.loads((root / "moddrop" / "seed{}".format(seed) / "stage_manifest.json").read_text())
    clean_path = Path(clean["Checkpoint"])
    moddrop_path = Path(moddrop["Checkpoint"])
    if sha256_file(clean_path) != clean["CheckpointSHA256"]:
        raise RuntimeError("Teacher checkpoint SHA mismatch.")
    if sha256_file(moddrop_path) != moddrop["CheckpointSHA256"]:
        raise RuntimeError("ModDrop checkpoint SHA mismatch.")
    return clean, moddrop, clean_path, moddrop_path


def main():
    cli = parse_args()
    setup_seed(cli.seed)
    args = build_args(cli)
    clean_manifest, moddrop_manifest, clean_path, moddrop_path = load_input_manifests(
        cli.artifact_root, cli.seed
    )
    teacher = build_frozen_teacher(DLF, args, clean_path)
    reference_backbone = DLF(args).to(args.device)
    reference = MissingModalityWrapper(
        reference_backbone, args.feature_dims[1], args.feature_dims[2]
    ).to(args.device)
    reference.load_state_dict(torch.load(moddrop_path, map_location=args.device), strict=True)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in teacher.parameters()) or any(
        parameter.requires_grad for parameter in reference.parameters()
    ):
        raise RuntimeError("Frozen cache models have trainable parameters.")

    output_root = Path(cli.output_root) / "seed{}".format(cli.seed)
    entries = {}
    for split in ALLOWED_SPLITS:
        loader = locked_loader(args, split, cli.num_workers)
        ids = []
        teacher_values = []
        reference_values = {"LA": [], "LV": [], "L": []}
        for batch in loader:
            batch_ids = canonical_ids(batch["id"])
            text = batch["text"].to(args.device)
            audio = batch["audio"].to(args.device)
            vision = batch["vision"].to(args.device)
            ids.extend(batch_ids)
            teacher_values.append(
                teacher_lav_prediction(teacher, text, audio, vision).view(-1).cpu().numpy()
            )
            for mode in reference_values:
                reference_values[mode].append(
                    evaluator_prediction(reference, text, audio, vision, mode).view(-1).cpu().numpy()
                )
        ids = canonical_ids(ids)
        if ids != canonical_ids(loader.dataset.ids):
            raise RuntimeError("Cache loader order differs from dataset order.")
        arrays = {
            "sample_id": np.asarray(ids, dtype=np.unicode_),
            "teacher_lav": np.concatenate(teacher_values).astype(np.float32),
        }
        arrays.update(
            {
                "moddrop_{}".format(mode): np.concatenate(values).astype(np.float32)
                for mode, values in reference_values.items()
            }
        )
        if any(value.shape != (len(ids),) for key, value in arrays.items() if key != "sample_id"):
            raise RuntimeError("Cache tensor shape mismatch.")
        cache_path = output_root / "{}_predictions.npz".format(split)
        atomic_npz(cache_path, **arrays)
        entry = {
            "dataset": "mosei",
            "split": split,
            "sample_count": len(ids),
            "ordered_sample_id_sha256": ordered_id_sha(ids),
            "unordered_sample_id_sha256": unordered_id_sha(ids),
            "teacher_checkpoint": str(clean_path),
            "teacher_checkpoint_sha256": clean_manifest["CheckpointSHA256"],
            "moddrop_checkpoint": str(moddrop_path),
            "moddrop_checkpoint_sha256": moddrop_manifest["CheckpointSHA256"],
            "code_commit": git_head(),
            "dtype": "float32",
            "tensor_shapes": {
                "teacher_lav": [len(ids)],
                "moddrop_LA": [len(ids)],
                "moddrop_LV": [len(ids)],
                "moddrop_L": [len(ids)],
            },
            "creation_timestamp": utc_now(),
            "path": str(cache_path),
            "sha256": sha256_file(cache_path),
            "locked_test_access_count": 0,
        }
        entry_path = output_root / "{}_manifest.json".format(split)
        atomic_json(entry_path, entry)
        entry["manifest_path"] = str(entry_path)
        entry["manifest_sha256"] = sha256_file(entry_path)
        entries[split] = entry
        print(json.dumps({"Seed": cli.seed, "Split": split, "Rows": len(ids)}, sort_keys=True))

    aggregate = {
        "dataset": "mosei",
        "seed": cli.seed,
        "entries": entries,
        "code_commit": git_head(),
        "created_at": utc_now(),
        "locked_test_access_count": 0,
    }
    atomic_json(output_root / "cache_manifest.json", aggregate)


if __name__ == "__main__":
    main()
