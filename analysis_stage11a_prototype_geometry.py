"""Stage 11A: train-prototype/validation-signal audit with no test access."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_config_regression
from trains.singleTask.cfcompat_prediction_ensemble_utils import (
    max_prediction_difference,
    metric_max_difference,
)
from trains.singleTask.missing_utils import (
    MissingModalityWrapper,
    build_single_split_loader,
    mode_to_mask,
    regression_metrics,
)
from trains.singleTask.model.DLF import DLF
from trains.singleTask.ordered_prototype_geometry import (
    LEVELS,
    MODES,
    MISSING_MODES,
    FusionRepresentationCapture,
    compute_prototypes,
    cross_mode_retrieval,
    displacement,
    fit_sentiment_axis,
    geometry_summary,
    ordinal_spearman,
    prototype_estimate,
    prototype_temperature,
    spearman,
    stage11a_gate,
)
from utils.functions import assign_gpu, setup_seed

SEEDS = (1111, 1112, 1113, 1114, 1115)
SOURCE_ROOT = Path("/code/DLF")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument(
        "--output-root",
        default="result/missing_baseline/ccopt_v1/mosi",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    if tuple(args.seeds) != SEEDS:
        parser.error("Stage 11A requires fixed seeds 1111..1115.")
    return args


def model_args(seed):
    args = get_config_regression(
        "DLF", "mosi", str(ROOT / "config/config.json")
    )
    args.mode = "train"
    args.featurePath = str(
        SOURCE_ROOT / "dataset/MOSI/Processed/aligned_50.pkl"
    )
    args.pretrained = str(SOURCE_ROOT / "bert-base-uncased")
    args.feature_T = args.feature_A = args.feature_V = ""
    args.is_training = False
    args.train_mode = "regression"
    args.seed = args.cur_seed = int(seed)
    args.device = assign_gpu([0])
    return args


def online_records():
    path = (
        SOURCE_ROOT
        / "result/missing_baseline/cfcompat_stability_v1/mosi/checkpoint_manifest.json"
    )
    manifest = json.loads(path.read_text())
    records = [
        row
        for row in manifest["Methods"]
        if row["Method"] == "Online" and int(row["Seed"]) in SEEDS
    ]
    records.sort(key=lambda row: int(row["Seed"]))
    if [int(row["Seed"]) for row in records] != list(SEEDS):
        raise RuntimeError("Five Online baseline checkpoints are incomplete.")
    return records, manifest, path


def compatibility_path(seed):
    if int(seed) == 1111:
        return (
            SOURCE_ROOT
            / "result/counterfactual_compatibility/cf_compat_v1/mosi/train_counterfactual_compatibility.csv"
        )
    return (
        SOURCE_ROOT
        / "result/counterfactual_compatibility/cf_compat_v1_multiseed/mosi"
        / "seed{}".format(seed)
        / "train_counterfactual_compatibility.csv"
    )


def evaluator_path(seed):
    if int(seed) == 1111:
        return (
            SOURCE_ROOT
            / "pt/missing_baseline/moddrop/DLF_mosi_seed1111_best.pth"
        )
    return (
        SOURCE_ROOT
        / "pt/missing_baseline/moddrop_benchmark_multiseed_v1"
        / "seed{}".format(seed)
        / "DLF_mosi_seed{}_best_valid.pth".format(seed)
    )


def saved_valid_path(seed):
    return (
        SOURCE_ROOT
        / "result/missing_baseline/cfcompat_prediction_ensemble_v1/mosi"
        / "online_seed{}_valid_predictions.csv".format(seed)
    )


def load_model(args, record):
    checkpoint = SOURCE_ROOT / record["Checkpoint"]
    if not checkpoint.is_file() or sha256(checkpoint) != record["CheckpointSHA256"]:
        raise RuntimeError("Baseline checkpoint SHA mismatch: {}".format(checkpoint))
    backbone = DLF(args).to(args.device)
    model = MissingModalityWrapper(
        backbone, args.feature_dims[1], args.feature_dims[2]
    ).to(args.device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=args.device), strict=True
    )
    model.eval()
    return model, checkpoint


def extract_split(model, args, split, num_workers):
    if split not in ("train", "valid"):
        raise RuntimeError("Stage 11A cannot construct a test loader.")
    loader = build_single_split_loader(args, split, num_workers)
    capture = FusionRepresentationCapture(model)
    representations = {mode: [] for mode in MODES}
    predictions = {mode: [] for mode in MODES}
    labels, indices, ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            text = batch["text"].to(args.device)
            audio = batch["audio"].to(args.device)
            vision = batch["vision"].to(args.device)
            target = batch["labels"]["M"].view(-1).cpu()
            labels.append(target)
            indices.extend(batch["index"].view(-1).tolist())
            ids.extend(str(value) for value in list(batch["id"]))
            for mode in MODES:
                mask = mode_to_mask(
                    mode,
                    target.numel(),
                    args.device,
                    audio.dtype,
                )
                output = model(text, audio, vision, mask)
                hidden = capture.pop()
                representations[mode].append(hidden.detach().cpu())
                predictions[mode].append(
                    output["output_logit"].detach().view(-1).cpu()
                )
    capture.close()
    return {
        "representations": {
            mode: torch.cat(values).numpy()
            for mode, values in representations.items()
        },
        "predictions": {
            mode: torch.cat(values).numpy() for mode, values in predictions.items()
        },
        "labels": torch.cat(labels).numpy(),
        "sample_index": np.asarray(indices, dtype=np.int64),
        "sample_id": np.asarray(ids, dtype=object),
    }


def prediction_frame(split_data, split, seed, predictions=None):
    values = predictions or split_data["predictions"]
    frame = pd.DataFrame(
        {
            "sample_index": split_data["sample_index"],
            "sample_id": split_data["sample_id"],
            "label": split_data["labels"],
            **{
                "{}_pred".format(mode): values[mode]
                for mode in MODES
            },
        }
    )
    frame["Split"] = split
    frame["Method"] = "Online"
    frame["Seed"] = int(seed)
    return frame.sort_values("sample_index", kind="mergesort")


def all_mode_metrics(labels, predictions):
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    result = {}
    for mode in MODES:
        result[mode] = regression_metrics(
            torch.tensor(predictions[mode], dtype=torch.float32),
            labels_tensor,
        )
        result[mode]["Loss"] = result[mode]["MAE"]
    result["MissingMacro"] = {
        metric: float(np.mean([result[mode][metric] for mode in MISSING_MODES]))
        for metric in result["LAV"]
    }
    result["J"] = 0.5 * result["LAV"]["MAE"] + 0.5 * result[
        "MissingMacro"
    ]["MAE"]
    return result


def baseline_asset_records(records, stability, stability_path):
    seed_manifests = {
        int(row["Seed"]): row for row in stability["SeedManifests"]
    }
    assets = []
    for record in records:
        seed = int(record["Seed"])
        seed_manifest = seed_manifests[seed]
        checkpoint = SOURCE_ROOT / record["Checkpoint"]
        cache = compatibility_path(seed)
        evaluator = evaluator_path(seed)
        saved_valid = saved_valid_path(seed)
        expected = (
            (checkpoint, record["CheckpointSHA256"], "validation_best_checkpoint"),
            (
                cache,
                seed_manifest["CompatibilityCacheSHA256"],
                "train_only_compatibility_cache",
            ),
            (evaluator, seed_manifest["EvaluatorSHA256"], "moddrop_evaluator"),
            (saved_valid, None, "frozen_valid_predictions"),
        )
        seed_assets = []
        for path, expected_sha, role in expected:
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = sha256(path)
            if expected_sha and actual != expected_sha:
                raise RuntimeError("Baseline asset SHA mismatch: {}".format(path))
            seed_assets.append(
                {
                    "Role": role,
                    "Path": str(path),
                    "SHA256": actual,
                    "Bytes": path.stat().st_size,
                }
            )
        assets.append(
            {
                "Seed": seed,
                "BestValidEpoch": int(record["BestValidEpoch"]),
                "JValid": float(record["J_valid"]),
                "Assets": seed_assets,
                "MissingSequenceSHA256": seed_manifest[
                    "MissingSequenceSHA256"
                ],
                "MissingSequenceReplayPassed": bool(
                    seed_manifest["Replay"]["MissingSequenceMatch"]
                ),
                "SelectedBy": "validation_J",
            }
        )
    return {
        "Dataset": "mosi",
        "Seeds": list(SEEDS),
        "SourceManifest": str(stability_path),
        "SourceManifestSHA256": sha256(stability_path),
        "Members": assets,
        "NoTestAssetUsedByStage11A": True,
    }


def write_gate_outputs(output, gate, baseline_replay_passed):
    gate_path = output / "stage11a_gate.json"
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    report = [
        "# Stage 11A Ordered Prototype Geometry Audit",
        "",
        "- Dataset: MOSI",
        "- Seeds: 1111, 1112, 1113, 1114, 1115",
        "- Prototype source: train labels and train representations only",
        "- Evaluation split: valid only",
        "- Test loader constructed: false",
        "- Diagnostic step: 0.10 (fixed)",
        "- Temperature: train-prototype pairwise-distance median",
        "- Hidden representation: shared 300-D fusion tensor entering `backbone.proj1`",
        "- Baseline replay passed: {}".format(baseline_replay_passed),
        "",
        "## Evidence gate",
        "",
    ]
    for name, passed in gate["Conditions"].items():
        report.append("- {}: {}".format(name, "PASS" if passed else "FAIL"))
    report.extend(
        [
            "",
            "- Mean Delta J_valid: {:.9f}".format(gate["Means"]["DeltaJ"]),
            "- Mean Delta LAV MAE: {:.9f}".format(
                gate["Means"]["DeltaLAVMAE"]
            ),
            "- Mean Delta MissingMacro MAE: {:.9f}".format(
                gate["Means"]["DeltaMissingMacroMAE"]
            ),
            "- Mean Delta Acc7: {:.9f}".format(
                gate["Means"]["DeltaMeanAcc7"]
            ),
            "- Mean Delta Acc5: {:.9f}".format(
                gate["Means"]["DeltaMeanAcc5"]
            ),
            "- Improved J seeds: {}/5".format(
                gate["Counts"]["ImprovedJSeeds"]
            ),
            "- Verdict: **{}**".format(gate["Verdict"]),
        ]
    )
    (output / "stage11a_geometry_audit.md").write_text(
        "\n".join(report) + "\n"
    )


def main():
    cli = parse_args()
    output_root = Path(cli.output_root)
    output = output_root / "stage11a_geometry"
    output.mkdir(parents=True, exist_ok=True)
    if cli.finalize_existing:
        geometry = pd.read_csv(output / "stage11a_geometry_per_seed.csv")
        diagnostics = pd.read_csv(
            output / "stage11a_direction_diagnostic.csv"
        )
        replay = pd.read_csv(output / "stage11a_baseline_replay.csv")
        gate = stage11a_gate(
            geometry.to_dict("records"), diagnostics.to_dict("records")
        )
        replay_passed = bool(
            replay.PredictionIdentityPassed.astype(bool).all()
            and replay.MetricIdentityPassed.astype(bool).all()
        )
        write_gate_outputs(output, gate, replay_passed)
        print(gate["Verdict"], flush=True)
        raise SystemExit(0 if gate["Passed"] else 3)
    records, stability, stability_path = online_records()
    asset_manifest = baseline_asset_records(records, stability, stability_path)
    prototype_rows, geometry_rows, diagnostic_rows = [], [], []
    replay_rows, generated_assets = [], []
    for record in records:
        seed = int(record["Seed"])
        setup_seed(seed)
        args = model_args(seed)
        model, checkpoint = load_model(args, record)
        state_before = sha256(checkpoint)
        train = extract_split(model, args, "train", cli.num_workers)
        valid = extract_split(model, args, "valid", cli.num_workers)
        prototypes = {}
        for mode in MODES:
            prototypes[mode], counts, variances = compute_prototypes(
                train["representations"][mode], train["labels"]
            )
            for level in LEVELS:
                prototype = prototypes[mode][int(level)]
                prototype_rows.append(
                    {
                        "Seed": seed,
                        "Mode": mode,
                        "Level": int(level),
                        "SampleCount": counts[int(level)],
                        "EmptyBin": prototype is None,
                        "WithinClassVariance": variances[int(level)],
                        "PrototypeNorm": (
                            float(np.linalg.norm(prototype))
                            if prototype is not None
                            else np.nan
                        ),
                        "PrototypeVector": (
                            json.dumps(prototype.tolist())
                            if prototype is not None
                            else ""
                        ),
                    }
                )
        axis = fit_sentiment_axis(prototypes["LAV"])
        baseline_metrics = all_mode_metrics(
            valid["labels"], valid["predictions"]
        )
        diagnostic_predictions = {}
        for mode in MODES:
            summary = geometry_summary(
                train["representations"][mode],
                train["labels"],
                prototypes[mode],
            )
            retrieval = (
                {
                    "SameBinRetrievalCount": 7,
                    "ValidBinCount": 7,
                    "SameBinRetrievalRate": 1.0,
                    "OrderedRetrievalError": 0.0,
                    "Rows": [],
                }
                if mode == "LAV"
                else cross_mode_retrieval(
                    prototypes[mode], prototypes["LAV"]
                )
            )
            displacement_values = displacement(
                valid["representations"][mode],
                valid["labels"],
                prototypes[mode],
            )
            absolute_error = np.abs(
                valid["predictions"][mode] - valid["labels"]
            )
            temperature = prototype_temperature(prototypes[mode])
            estimate = prototype_estimate(
                valid["representations"][mode],
                prototypes[mode],
                temperature,
            )
            diagnostic_predictions[mode] = np.clip(
                valid["predictions"][mode]
                + 0.10 * (estimate - valid["predictions"][mode]),
                -3.0,
                3.0,
            )
            geometry_rows.append(
                {
                    "Seed": seed,
                    "Mode": mode,
                    **summary,
                    "OrdinalSpearman": ordinal_spearman(
                        prototypes[mode], axis
                    ),
                    "SameBinRetrievalCount": retrieval[
                        "SameBinRetrievalCount"
                    ],
                    "ValidBinCount": retrieval["ValidBinCount"],
                    "SameBinRetrievalRate": retrieval[
                        "SameBinRetrievalRate"
                    ],
                    "OrderedRetrievalError": retrieval[
                        "OrderedRetrievalError"
                    ],
                    "ErrorAssociationSpearman": spearman(
                        displacement_values, absolute_error
                    ),
                    "PrototypeTemperature": temperature,
                    "EmptyBinCount": int(
                        sum(
                            prototypes[mode][int(level)] is None
                            for level in LEVELS
                        )
                    ),
                }
            )
        diagnostic_metrics = all_mode_metrics(
            valid["labels"], diagnostic_predictions
        )
        delta_j = diagnostic_metrics["J"] - baseline_metrics["J"]
        for mode in MODES:
            diagnostic_rows.append(
                {
                    "Seed": seed,
                    "Mode": mode,
                    "Step": 0.10,
                    "JBaseline": baseline_metrics["J"],
                    "JDiagnostic": diagnostic_metrics["J"],
                    "DeltaJ": delta_j,
                    "BaselineMAE": baseline_metrics[mode]["MAE"],
                    "DiagnosticMAE": diagnostic_metrics[mode]["MAE"],
                    "DeltaMAE": diagnostic_metrics[mode]["MAE"]
                    - baseline_metrics[mode]["MAE"],
                    "DeltaLAVMAE": diagnostic_metrics["LAV"]["MAE"]
                    - baseline_metrics["LAV"]["MAE"],
                    "DeltaMissingMacroMAE": diagnostic_metrics[
                        "MissingMacro"
                    ]["MAE"]
                    - baseline_metrics["MissingMacro"]["MAE"],
                    "DeltaMeanAcc7": np.mean(
                        [
                            diagnostic_metrics[value]["acc_7"]
                            - baseline_metrics[value]["acc_7"]
                            for value in MODES
                        ]
                    ),
                    "DeltaMeanAcc5": np.mean(
                        [
                            diagnostic_metrics[value]["acc_5"]
                            - baseline_metrics[value]["acc_5"]
                            for value in MODES
                        ]
                    ),
                }
            )
        train_path = output / "seed{}_train_predictions.csv".format(seed)
        valid_path = output / "seed{}_valid_predictions.csv".format(seed)
        generated_valid = prediction_frame(valid, "valid", seed)
        prediction_frame(train, "train", seed).to_csv(train_path, index=False)
        generated_valid.to_csv(valid_path, index=False)
        saved_valid = pd.read_csv(saved_valid_path(seed))
        max_prediction = max_prediction_difference(
            generated_valid, saved_valid, "valid"
        )
        max_metric = metric_max_difference(generated_valid, saved_valid)
        if max_prediction > 1e-7 or max_metric > 1e-8:
            raise RuntimeError("BASELINE_ASSET_REPLAY_FAILED seed{}".format(seed))
        if sha256(checkpoint) != state_before:
            raise RuntimeError("Checkpoint changed during read-only extraction.")
        replay_rows.append(
            {
                "Seed": seed,
                "PredictionMaxDifference": max_prediction,
                "MetricMaxDifference": max_metric,
                "PredictionIdentityPassed": max_prediction <= 1e-7,
                "MetricIdentityPassed": max_metric <= 1e-8,
                "CheckpointUnchanged": True,
                "TestLoaderConstructed": False,
            }
        )
        generated_assets.extend(
            [
                {
                    "Seed": seed,
                    "Role": "generated_train_predictions",
                    "Path": str(train_path),
                    "SHA256": sha256(train_path),
                },
                {
                    "Seed": seed,
                    "Role": "replayed_valid_predictions",
                    "Path": str(valid_path),
                    "SHA256": sha256(valid_path),
                },
            ]
        )
        del model
        torch.cuda.empty_cache()
        print("Stage11A seed{} complete".format(seed), flush=True)
    geometry = pd.DataFrame(geometry_rows)
    prototypes_frame = pd.DataFrame(prototype_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    replay = pd.DataFrame(replay_rows)
    gate = stage11a_gate(geometry_rows, diagnostic_rows)
    geometry.to_csv(output / "stage11a_geometry_per_seed.csv", index=False)
    prototypes_frame.to_csv(
        output / "stage11a_prototype_metrics.csv", index=False
    )
    diagnostics.to_csv(
        output / "stage11a_direction_diagnostic.csv", index=False
    )
    replay.to_csv(output / "stage11a_baseline_replay.csv", index=False)
    asset_manifest["GeneratedTrainValidAssets"] = generated_assets
    asset_manifest["AllReplayPassed"] = bool(
        replay.PredictionIdentityPassed.all()
        and replay.MetricIdentityPassed.all()
    )
    asset_manifest_path = output_root / "baseline_asset_manifest.json"
    asset_manifest_path.write_text(
        json.dumps(asset_manifest, indent=2, sort_keys=True) + "\n"
    )
    write_gate_outputs(
        output, gate, asset_manifest["AllReplayPassed"]
    )
    print(gate["Verdict"], flush=True)
    if not gate["Passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
