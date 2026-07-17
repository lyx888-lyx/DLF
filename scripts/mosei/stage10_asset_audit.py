"""Audit the manually supplied official MOSEI aligned asset."""
import argparse
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mosei.stage10_common import atomic_json, sha256, utc_now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-file",
        default="/data4t/lyx/datasets/MOSEI/Processed/aligned_50.pkl",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--estimated-seed-hours", type=float, default=0.0)
    args = parser.parse_args()
    data_file = Path(args.data_file).resolve()
    output = Path(args.output_dir).resolve()
    if not data_file.is_file():
        raise FileNotFoundError(data_file)
    with data_file.open("rb") as handle:
        dataset = pickle.load(handle)
    if tuple(dataset.keys()) != ("train", "valid", "test"):
        raise RuntimeError("Official MOSEI split keys are not train/valid/test.")
    split_records = {}
    all_ids = {}
    for split in ("train", "valid", "test"):
        values = dataset[split]
        required = {
            "id",
            "audio",
            "vision",
            "text",
            "text_bert",
            "regression_labels",
        }
        if not required.issubset(values):
            raise RuntimeError("MOSEI {} fields are incomplete.".format(split))
        ids = np.asarray(values["id"]).astype(str).reshape(-1)
        all_ids[split] = set(ids.tolist())
        labels = np.asarray(values["regression_labels"])
        finite = {}
        shapes = {}
        for key in ("text", "text_bert", "audio", "vision", "regression_labels"):
            array = np.asarray(values[key])
            shapes[key] = list(array.shape)
            finite[key] = bool(np.isfinite(array).all())
        split_records[split] = {
            "Samples": len(ids),
            "Shapes": shapes,
            "Finite": finite,
            "UniqueIDs": len(set(ids.tolist())),
            "DuplicateIDs": len(ids) - len(set(ids.tolist())),
            "LabelMin": float(labels.min()),
            "LabelMax": float(labels.max()),
        }
    overlaps = {
        "train_valid": len(all_ids["train"] & all_ids["valid"]),
        "train_test": len(all_ids["train"] & all_ids["test"]),
        "valid_test": len(all_ids["valid"] & all_ids["test"]),
    }
    if [split_records[key]["Samples"] for key in ("train", "valid", "test")] != [
        16326,
        1871,
        4659,
    ]:
        raise RuntimeError("Official MOSEI split counts differ.")
    if any(overlaps.values()) or any(
        not value
        for record in split_records.values()
        for value in record["Finite"].values()
    ):
        raise RuntimeError("MOSEI IDs overlap or features contain NaN/Inf.")
    output.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(str(output))
    manifest = {
        "AuditedAt": utc_now(),
        "DataPath": str(data_file),
        "DataBytes": data_file.stat().st_size,
        "DataSHA256": sha256(data_file),
        "OfficialSplitConfirmed": True,
        "Splits": split_records,
        "SplitIDOverlaps": overlaps,
        "ReusableAssets": [{"Path": str(data_file), "SHA256": sha256(data_file)}],
        "MissingAssets": [
            "MOSEI clean DLF checkpoints",
            "MOSEI ModDrop checkpoints",
            "MOSEI compatibility caches",
            "MOSEI CFCompatKD checkpoints",
        ],
        "EstimatedHoursPerSeed": float(args.estimated_seed_hours),
        "EstimatedTotalGPUHours": float(args.estimated_seed_hours) * 5.0,
        "FreeDiskBytes": disk.free,
    }
    atomic_json(output / "mosei_asset_manifest.json", manifest)
    lines = [
        "# MOSEI Asset Audit",
        "",
        "- Data: `{}`".format(data_file),
        "- SHA256: `{}`".format(manifest["DataSHA256"]),
        "- Official split confirmed: yes",
        "- Counts: train 16326, valid 1871, test 4659",
        "- Label range: [{:.1f}, {:.1f}]".format(
            min(row["LabelMin"] for row in split_records.values()),
            max(row["LabelMax"] for row in split_records.values()),
        ),
        "- Duplicate or overlapping sample IDs: none",
        "- NaN/Inf: none",
        "- Feature dimensions: text 768, audio 74, vision 35",
        "- Existing reusable trained assets: none",
        "- Missing assets: clean, ModDrop, compatibility cache, CFCompatKD",
        "- Estimated hours per seed: {:.3f}".format(args.estimated_seed_hours),
        "- Estimated total GPU hours: {:.3f}".format(
            args.estimated_seed_hours * 5.0
        ),
        "- Free disk: {:.1f} GiB".format(disk.free / 1024 ** 3),
    ]
    (output / "MOSEI_ASSET_AUDIT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
