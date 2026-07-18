"""Five-seed train/valid-only replay of frozen Stage 8 Online CFCompatKD assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import get_config_regression
from trains.singleTask.fixed_kd_utils import checkpoint_sha256
from trains.singleTask.missing_utils import (
    MissingModalityWrapper,
    build_single_split_loader,
    mode_to_mask,
    regression_metrics,
    validation_objective,
)
from trains.singleTask.model.DLF import DLF
from utils.functions import setup_seed


SEEDS = (1111, 1112, 1113, 1114, 1115)
MODES = ("LAV", "LA", "LV", "L")
ROOT = Path("result/missing_baseline/mcao_v1/mosi")
SOURCE_MANIFEST = Path(
    "/code/DLF-mosi-dcrc-v1/result/missing_baseline/dcrc_v1/"
    "mosi/baseline_asset_manifest.json"
)
STAGE8_ROOT = Path(
    "/code/DLF/result/missing_baseline/cfcompat_stability_v1/mosi"
)
DATA_PATH = Path("/code/DLF/dataset/MOSI/Processed/aligned_50.pkl")
BERT_PATH = "/code/DLF/bert-base-uncased"


def array_sha256(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def build_args(seed, device):
    args = get_config_regression("DLF", "mosi", "config/config.json")
    args.mode = "train"
    args.feature_T = args.feature_A = args.feature_V = ""
    args.is_training = True
    args.train_mode = "regression"
    args.seed = args.cur_seed = int(seed)
    args.device = device
    args.pretrained = BERT_PATH
    args.featurePath = str(DATA_PATH)
    return args


def source_row(seed):
    payload = json.loads(SOURCE_MANIFEST.read_text())
    rows = [
        row for row in payload["CheckpointAndEvaluatorAssets"]
        if int(row["Seed"]) == int(seed)
    ]
    if len(rows) != 1:
        raise RuntimeError("No unique frozen asset record for seed{}.".format(seed))
    return rows[0]


def predict_split(model, loader, device):
    model.eval()
    predictions = {mode: [] for mode in MODES}
    labels, indices, sample_ids = [], [], []
    with torch.inference_mode():
        for batch in loader:
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            truth = batch["labels"]["M"].to(device).view(-1)
            for mode in MODES:
                mask = mode_to_mask(
                    mode, len(truth), device=device, dtype=audio.dtype
                )
                output = model(text, audio, vision, mask)["output_logit"].view(-1)
                predictions[mode].append(output.cpu())
            labels.append(truth.cpu())
            indices.extend(batch["index"].view(-1).cpu().numpy().astype(int).tolist())
            sample_ids.extend(map(str, list(batch["id"])))
    order = np.argsort(indices, kind="mergesort")
    matrix = np.column_stack(
        [torch.cat(predictions[mode]).numpy() for mode in MODES]
    ).astype(np.float32)[order]
    truth = torch.cat(labels).numpy().astype(np.float32)[order]
    indices = np.asarray(indices, dtype=np.int64)[order]
    sample_ids = np.asarray(sample_ids, dtype=str)[order]
    if len(np.unique(indices)) != len(indices):
        raise RuntimeError("Duplicate sample_index in split replay.")
    metrics = {}
    for column, mode in enumerate(MODES):
        row = regression_metrics(
            torch.from_numpy(matrix[:, column]), torch.from_numpy(truth)
        )
        row["Loss"] = row["MAE"]
        metrics[mode] = row
    return {
        "Predictions": matrix,
        "Labels": truth,
        "Indices": indices,
        "SampleIds": sample_ids,
        "Metrics": metrics,
        "J": float(validation_objective(metrics)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args_cli = parser.parse_args()
    resource = json.loads((ROOT / "gpu3_resource_gate_raw.json").read_text())
    if not resource["Passed"]:
        raise RuntimeError("Stage17 GPU3 resource gate did not pass.")
    setup_seed(args_cli.seed)
    device = torch.device("cuda:0")
    args = build_args(args_cli.seed, device)
    source = source_row(args_cli.seed)
    checkpoint = Path(source["CheckpointPath"])
    if checkpoint_sha256(checkpoint) != source["CheckpointSHA256"]:
        raise RuntimeError("Online checkpoint SHA mismatch.")
    if checkpoint_sha256(source["EvaluatorPath"]) != source["EvaluatorSHA256"]:
        raise RuntimeError("ModDrop/evaluator SHA mismatch.")
    stage8 = json.loads(
        (STAGE8_ROOT / "seed{}".format(args_cli.seed) / "seed_manifest.json").read_text()
    )
    clean = Path("/code/DLF/pt/DLF_mosi_seed{}_best.pth".format(args_cli.seed))
    if checkpoint_sha256(clean) != stage8["TeacherSHA256"]:
        raise RuntimeError("Clean teacher SHA mismatch.")

    model = MissingModalityWrapper(
        DLF(args), args.feature_dims[1], args.feature_dims[2]
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    replay = {}
    for split in ("train", "valid"):
        loader = build_single_split_loader(args, split, 1)
        value = predict_split(model, loader, device)
        replay[split] = {
            "SampleCount": len(value["Indices"]),
            "SampleIndexSHA256": array_sha256(value["Indices"]),
            "SampleIdSHA256": hashlib.sha256(
                "\n".join(value["SampleIds"]).encode("utf-8")
            ).hexdigest(),
            "LabelSHA256": array_sha256(value["Labels"]),
            "PredictionSHA256": array_sha256(value["Predictions"]),
            "Metrics": value["Metrics"],
            "J": value["J"],
        }
        if split == "valid":
            reference = pd.read_csv(source["ValidPredictionPath"]).sort_values(
                "sample_index", kind="mergesort"
            )
            if not np.array_equal(
                reference.sample_index.to_numpy(dtype=np.int64), value["Indices"]
            ):
                raise RuntimeError("Valid sample binding mismatch.")
            expected = reference[
                ["LAV_pred", "LA_pred", "LV_pred", "L_pred"]
            ].to_numpy(dtype=np.float32)
            prediction_diff = float(np.max(np.abs(expected - value["Predictions"])))
            expected_j = float(stage8["Methods"]["Online"]["J_valid"])
            j_diff = abs(value["J"] - expected_j)
            replay[split]["ReferencePath"] = source["ValidPredictionPath"]
            replay[split]["ReferenceSHA256"] = source["ValidPredictionSHA256"]
            replay[split]["PredictionMaxDiff"] = prediction_diff
            replay[split]["ExpectedJ"] = expected_j
            replay[split]["JDiff"] = j_diff
            if prediction_diff > 1e-7 or j_diff > 1e-8:
                raise RuntimeError("Frozen valid replay tolerance exceeded.")

    seed_manifest = {
        "Seed": args_cli.seed,
        "CheckpointPath": str(checkpoint),
        "CheckpointSHA256": source["CheckpointSHA256"],
        "CleanTeacherPath": str(clean),
        "CleanTeacherSHA256": stage8["TeacherSHA256"],
        "ModDropEvaluatorPath": source["EvaluatorPath"],
        "ModDropEvaluatorSHA256": source["EvaluatorSHA256"],
        "CompatibilityCacheSHA256": stage8["CompatibilityCacheSHA256"],
        "MissingSequenceSHA256": stage8["MissingSequenceSHA256"],
        "ValidationBestEpoch": stage8["Methods"]["Online"]["BestValidEpoch"],
        "Replay": replay,
        "Passed": True,
        "TestLoaderConstructed": False,
        "TestFeaturesRead": False,
        "TestLabelsRead": False,
        "TestPredictionsRead": False,
        "TestEvaluationPerformed": False,
        "LockedTestAccessCount": 0,
    }
    out = ROOT / "baseline_replay" / "seed{}".format(args_cli.seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(
        json.dumps(seed_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        "seed={} valid_J={:.12f} max_diff={:.3g} train_sha={}".format(
            args_cli.seed,
            replay["valid"]["J"],
            replay["valid"]["PredictionMaxDiff"],
            replay["train"]["PredictionSHA256"],
        )
    )


if __name__ == "__main__":
    main()
