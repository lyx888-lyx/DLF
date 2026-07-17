"""Stage 7A Modality Utility and Conditional Contribution Audit (MUCCA).

The program is deliberately analysis-only: it constructs only MOSI train/valid
datasets, evaluates two frozen validation-selected states, and never creates an
optimizer or saves a checkpoint.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.cf_compat_kd_utils import (
    build_frozen_evaluator, cache_paths, evaluator_prediction, locate_stage1_evaluator,
)
from trains.singleTask.fixed_kd_utils import checkpoint_sha256, teacher_grad_count
from trains.singleTask.missing_utils import (
    MissingModalityWrapper, mode_to_mask, regression_metrics,
)
from trains.singleTask.model.DLF import DLF
from trains.singleTask.modality_utility_utils import (
    LABEL_BINS, MODALITIES, MODES, UTILITY_MODALITIES, assign_quartiles,
    bootstrap_summary, capture_rng_state, classify_change, classify_utility,
    clone_buffers, clone_parameters, fit_quartile_edges, gain_values,
    infer_padding_mask, label_bin, masked_input_sensitivity,
    representation_pair_summary, restore_rng_state, rng_states_equal,
    sattolo_derangement, sha256_file, shuffle_damage, stable_seed,
    tensor_maps_equal, validate_derangement, vector_from_json, vector_to_json,
)
from utils.functions import assign_gpu, setup_seed


BASE_BRANCH = "analysis/mode-gradient-conflict-v1"
BASE_COMMIT = "13a6eb7d7e9708e6480033f80be7f48312739c04"
STATES = ("gate3_init", "cfcompat_best_valid")
SPLITS = ("train", "valid")
MODE_FOR_MODALITY = {"A": "LA", "V": "LV", "AV": "LAV"}
COMPAT_FOR_MODALITY = {"A": "compat_LV", "V": "compat_LA", "AV": "compat_L"}
RAW_FILES = (
    "sample_manifest.csv", "shuffle_manifest.csv", "sample_predictions.csv",
    "sample_modality_gains.csv", "shuffle_sample_metrics.csv",
    "input_sensitivity_samples.csv", "representation_contribution_samples.csv",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Pure train/valid Stage 7A modality audit.")
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--shuffle-repeats", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=270700)
    parser.add_argument("--derangement-seed", type=int, default=270710)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[2])
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--config-file", default="config/config.json")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--verify-existing-artifacts", action="store_true")
    args = parser.parse_args()
    if args.seed != 1111:
        parser.error("Stage 7A is locked to seed1111.")
    if args.shuffle_repeats != 10 and not args.summary_only:
        parser.error("Stage 7A fixes shuffle repeats at 10.")
    if args.bootstrap_samples != 2000 and not args.summary_only:
        parser.error("Stage 7A fixes paired bootstrap resamples at 2000.")
    if args.num_workers != 0:
        parser.error("Stage 7A fixes num_workers=0.")
    if args.summary_only and not args.verify_existing_artifacts:
        parser.error("--summary-only requires --verify-existing-artifacts.")
    return args


def git(*arguments):
    return subprocess.check_output(["git"] + list(arguments), text=True).strip()


def output_directory(cli):
    return (
        Path(cli.result_root) / "analysis" / "modality_utility_v1"
        / cli.dataset / "seed{}".format(cli.seed)
    )


def build_config(cli):
    args = get_config_regression("DLF", cli.dataset, cli.config_file)
    args.mode = "train"
    args.feature_T = args.feature_A = args.feature_V = ""
    args.is_training = False
    args.train_mode = "regression"
    args.seed = args.cur_seed = cli.seed
    args.device = assign_gpu(list(cli.gpu_ids))
    return args


def build_audit_datasets(args, workers):
    datasets = {split: MMDataset(args, mode=split) for split in SPLITS}
    loaders = {
        split: DataLoader(
            datasets[split], batch_size=args.batch_size, num_workers=workers,
            shuffle=False, drop_last=False,
        )
        for split in SPLITS
    }
    for split, loader in loaders.items():
        if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
            raise RuntimeError("{} audit loader is not deterministic.".format(split))
    return datasets, loaders


def locate_validation_selected_states(result_root, seed):
    root = Path(result_root) / "missing_baseline" / "cf_compat_kd_v1" / "benchmark_multiseed"
    gate_manifest = root / "gate3_checkpoint_manifest.csv"
    cf_manifest = root / "checkpoint_manifest.csv"
    if not gate_manifest.is_file() or not cf_manifest.is_file():
        raise FileNotFoundError("Stage3 checkpoint manifests are required.")
    gate_rows = pd.read_csv(gate_manifest)
    cf_rows = pd.read_csv(cf_manifest)
    gate = gate_rows.loc[gate_rows.Seed.astype(int).eq(int(seed))]
    cf = cf_rows.loc[cf_rows.Seed.astype(int).eq(int(seed))]
    if len(gate) != 1 or len(cf) != 1:
        raise ValueError("Stage3 manifests must contain one seed1111 row.")
    gate_row, cf_row = gate.iloc[0], cf.iloc[0]
    if str(gate_row.Verified).lower() != "true" or "validation-best" not in str(gate_row.Protocol):
        raise ValueError("Gate3 checkpoint is not manifest-verified validation-best.")
    gate_path = Path(str(gate_row.Checkpoint))
    cf_path = Path(str(cf_row.CFCompatCheckpoint))
    forbidden = ("diagnostic", "best_test")
    if any(token in str(cf_path).lower() for token in forbidden) or "best_valid" not in str(cf_path):
        raise ValueError("CFCompat checkpoint is not validation-selected.")
    if not gate_path.is_file() or not cf_path.is_file():
        raise FileNotFoundError("A manifest-selected checkpoint is absent.")
    if checkpoint_sha256(gate_path) != str(gate_row.SHA256):
        raise ValueError("Gate3 checkpoint SHA does not match its manifest.")
    if checkpoint_sha256(cf_path) != str(cf_row.CFCompatSHA256):
        raise ValueError("CFCompat checkpoint SHA does not match its manifest.")
    return {
        "gate3": gate_path, "cfcompat": cf_path,
        "gate_manifest": gate_manifest, "cf_manifest": cf_manifest,
        "gate_epoch": int(gate_row.BestEpoch),
    }


def initialize_state(name, args, paths):
    rng = capture_rng_state()
    try:
        if name == "gate3_init":
            backbone = DLF(args).to(args.device)
            backbone.load_state_dict(
                torch.load(paths["gate3"], map_location=args.device), strict=True
            )
            return MissingModalityWrapper(
                backbone, args.feature_dims[1], args.feature_dims[2]
            ).to(args.device)
        model = MissingModalityWrapper(
            DLF(args).to(args.device), args.feature_dims[1], args.feature_dims[2]
        ).to(args.device)
        model.load_state_dict(
            torch.load(paths["cfcompat"], map_location=args.device), strict=True
        )
        return model
    finally:
        restore_rng_state(rng)


def load_train_compatibility(result_root, dataset):
    paths = cache_paths(result_root, dataset)
    if not paths["csv"].is_file() or not paths["config"].is_file():
        raise FileNotFoundError("Locked Stage3 train compatibility cache is absent.")
    frame = pd.read_csv(paths["csv"])
    config = json.loads(paths["config"].read_text())
    if (
        config.get("source") != "train_only"
        or int(config.get("train_sample_count", -1)) != 1284
        or config.get("version") != "cf_compat_v1"
        or len(frame) != 1284
        or frame.sample_index.nunique() != 1284
    ):
        raise ValueError("Stage3 compatibility cache is not the locked train-only artifact.")
    return frame, paths, config


def empirical_compatibility(train_delta, query_delta):
    train = np.sort(np.asarray(train_delta, dtype=np.float64))
    query = np.asarray(query_delta, dtype=np.float64)
    left = np.searchsorted(train, query, side="left")
    right = np.searchsorted(train, query, side="right")
    q = (left + 0.5 * (right - left)) / float(len(train))
    q = np.clip(q, 0.5 / len(train), 1.0 - 0.5 / len(train))
    return 1.0 - q


def build_valid_compatibility(args, loader, train_cache, evaluator_checkpoint):
    evaluator = build_frozen_evaluator(DLF, args, evaluator_checkpoint)
    rows = []
    for batch in loader:
        text = batch["text"].to(args.device)
        audio = batch["audio"].to(args.device)
        vision = batch["vision"].to(args.device)
        predictions = {
            mode: evaluator_prediction(evaluator, text, audio, vision, mode).view(-1).cpu().numpy()
            for mode in MODES
        }
        labels = batch["labels"]["M"].view(-1).cpu().numpy()
        for position, index in enumerate(batch["index"].view(-1).tolist()):
            row = {
                "sample_index": int(index), "sample_id": str(batch["id"][position]),
                "label": float(labels[position]),
            }
            for mode in MODES:
                row["evaluator_{}_pred".format(mode)] = float(predictions[mode][position])
            rows.append(row)
    if teacher_grad_count(evaluator) != 0:
        raise RuntimeError("Frozen compatibility evaluator received gradients.")
    frame = pd.DataFrame(rows).sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    if frame.sample_index.duplicated().any() or not np.array_equal(
        frame.sample_index.to_numpy(), np.arange(len(frame))
    ):
        raise RuntimeError("Valid compatibility sample binding is incomplete.")
    for mode in ("LA", "LV", "L"):
        delta = np.abs(frame.evaluator_LAV_pred - frame["evaluator_{}_pred".format(mode)])
        frame["delta_{}".format(mode)] = delta
        frame["compat_{}".format(mode)] = empirical_compatibility(
            train_cache["delta_{}".format(mode)], delta
        )
    del evaluator
    torch.cuda.empty_cache()
    return frame


def build_sample_manifest(datasets, train_cache, valid_compat):
    compatibility = {"train": train_cache, "valid": valid_compat}
    rows = []
    for split in SPLITS:
        dataset = datasets[split]
        compat = compatibility[split].set_index("sample_index")
        for index in range(len(dataset)):
            sample = dataset[index]
            if index not in compat.index:
                raise KeyError("Missing compatibility for {} index {}.".format(split, index))
            record = compat.loc[index]
            if str(sample["id"]) != str(record.sample_id):
                raise RuntimeError("Compatibility sample-id binding differs.")
            label = float(sample["labels"]["M"].view(-1)[0])
            if not np.isclose(label, float(record.label), rtol=0, atol=1e-6):
                raise RuntimeError("Compatibility label binding differs.")
            audio_mask = infer_padding_mask(sample["audio"].unsqueeze(0))[0]
            vision_mask = infer_padding_mask(sample["vision"].unsqueeze(0))[0]
            rows.append({
                "Split": split, "SampleIndex": index, "SampleID": str(sample["id"]),
                "Label": label, "LabelBin": str(label_bin([label])[0]),
                "AudioValidSteps": int(audio_mask.sum()), "VisionValidSteps": int(vision_mask.sum()),
                "compat_LA": float(record.compat_LA), "compat_LV": float(record.compat_LV),
                "compat_L": float(record.compat_L),
                "CompatibilitySource": "locked_train_cache" if split == "train"
                else "valid_evaluator_delta_mapped_by_train_empirical_cdf",
            })
    frame = pd.DataFrame(rows)
    for modality, column in COMPAT_FOR_MODALITY.items():
        train = frame.loc[frame.Split.eq("train"), column].to_numpy()
        edges = fit_quartile_edges(train)
        frame["CompatQuartile_{}".format(modality)] = assign_quartiles(frame[column], edges)
    return frame


def build_derangements(datasets, cli):
    mappings, rows = {}, []
    for split_index, split in enumerate(SPLITS):
        mappings[split] = {}
        size = len(datasets[split])
        for repeat in range(cli.shuffle_repeats):
            seed = cli.derangement_seed + split_index * 1000 + repeat
            mapping = sattolo_derangement(size, seed)
            validate_derangement(mapping, size)
            mappings[split][repeat] = mapping
            for source, target in enumerate(mapping):
                rows.append({
                    "Split": split, "Repeat": repeat, "Seed": seed,
                    "SourceIndex": source, "TargetIndex": int(target),
                    "AudioTargetIndex": int(target), "VisionTargetIndex": int(target),
                    "IsBijection": True, "HasFixedPoint": False, "CrossSplit": False,
                })
    return mappings, pd.DataFrame(rows)


def batch_arrays(dataset, indices, modality):
    array = dataset.audio if modality == "audio" else dataset.vision
    return torch.from_numpy(np.asarray(array[np.asarray(indices, dtype=np.int64)])).float()


def audit_one_state(state_name, model, datasets, loaders, mappings, args):
    prediction_rows, gain_rows, shuffle_rows = [], [], []
    sensitivity_rows, representation_rows = [], []
    parameters_before = clone_parameters(model)
    buffers_before = clone_buffers(model)
    rng_before = capture_rng_state()
    training_before = model.training
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Student parameters have pre-existing gradients.")
    model.eval()
    for split in SPLITS:
        dataset = datasets[split]
        captured = []
        capture_enabled = [True]

        def representation_hook(module, inputs):
            if capture_enabled[0]:
                captured.append(inputs[0].detach().cpu().numpy())

        handle = model.backbone.proj1.register_forward_pre_hook(representation_hook)
        for batch_index, batch in enumerate(loaders[split]):
            text = batch["text"].to(args.device)
            audio = batch["audio"].to(args.device)
            vision = batch["vision"].to(args.device)
            labels = batch["labels"]["M"].to(args.device).view(-1)
            indices = batch["index"].view(-1).cpu().numpy().astype(int)
            identifiers = [str(value) for value in batch["id"]]
            predictions, representations = {}, {}
            for mode in MODES:
                before = len(captured)
                mask = mode_to_mask(mode, len(indices), args.device, audio.dtype)
                if mode == "LAV":
                    audio_grad = audio.detach().clone().requires_grad_(True)
                    vision_grad = vision.detach().clone().requires_grad_(True)
                    output = model(text, audio_grad, vision_grad, mask)
                    gradient_a, gradient_v = torch.autograd.grad(
                        output["output_logit"].sum(), (audio_grad, vision_grad),
                        retain_graph=False, create_graph=False,
                    )
                    mask_a = infer_padding_mask(audio_grad)
                    mask_v = infer_padding_mask(vision_grad)
                    gradnorm_a, sensitivity_a = masked_input_sensitivity(
                        audio_grad, gradient_a, mask_a
                    )
                    gradnorm_v, sensitivity_v = masked_input_sensitivity(
                        vision_grad, gradient_v, mask_v
                    )
                    for position, sample_index in enumerate(indices):
                        sensitivity_rows.append({
                            "State": state_name, "Split": split,
                            "SampleIndex": int(sample_index), "SampleID": identifiers[position],
                            "Label": float(labels[position].detach().cpu()),
                            "GradNorm_A": float(gradnorm_a[position]),
                            "GradNorm_V": float(gradnorm_v[position]),
                            "Sensitivity_A": float(sensitivity_a[position]),
                            "Sensitivity_V": float(sensitivity_v[position]),
                            "AudioValidSteps": int(mask_a[position].sum().detach().cpu()),
                            "VisionValidSteps": int(mask_v[position].sum().detach().cpu()),
                        })
                else:
                    with torch.no_grad():
                        output = model(text, audio, vision, mask)
                if len(captured) != before + 1:
                    raise RuntimeError("Stage6A shared-representation hook did not fire once.")
                predictions[mode] = output["output_logit"].detach().view(-1).cpu().numpy()
                representations[mode] = captured[-1]
                del output
            for position, sample_index in enumerate(indices):
                row = {
                    "State": state_name, "Split": split,
                    "SampleIndex": int(sample_index), "SampleID": identifiers[position],
                    "Label": float(labels[position].detach().cpu()),
                }
                row.update({
                    "F_{}".format(mode): float(predictions[mode][position]) for mode in MODES
                })
                prediction_rows.append(row)
                h = {mode: representations[mode][position] for mode in MODES}
                base_norm = float(np.linalg.norm(h["L"]))
                rep_row = dict(row)
                for mode in MODES:
                    rep_row["H_{}".format(mode)] = vector_to_json(h[mode])
                for modality, mode in MODE_FOR_MODALITY.items():
                    delta_norm = float(np.linalg.norm(h[mode] - h["L"]))
                    rep_row["NormDelta_{}".format(modality)] = delta_norm
                    rep_row["RelativeShift_{}".format(modality)] = delta_norm / (base_norm + 1e-12)
                representation_rows.append(rep_row)
                for modality, mode in MODE_FOR_MODALITY.items():
                    gain_rows.append({
                        "State": state_name, "Split": split,
                        "SampleIndex": int(sample_index), "SampleID": identifiers[position],
                        "Label": row["Label"], "Modality": modality,
                        "Gain": float(gain_values(
                            [row["F_L"]], [row["F_{}".format(mode)]], [row["Label"]]
                        )[0]),
                    })
            correct = predictions["LAV"]
            label_numpy = labels.detach().cpu().numpy()
            capture_enabled[0] = False
            for repeat, mapping in mappings[split].items():
                mapped = mapping[indices]
                shuffled_audio = batch_arrays(dataset, mapped, "audio").to(args.device)
                shuffled_vision = batch_arrays(dataset, mapped, "vision").to(args.device)
                lav_mask = mode_to_mask("LAV", len(indices), args.device, audio.dtype)
                with torch.no_grad():
                    shuffled_a = model(
                        text, shuffled_audio, vision, lav_mask
                    )["output_logit"].view(-1).cpu().numpy()
                    shuffled_v = model(
                        text, audio, shuffled_vision, lav_mask
                    )["output_logit"].view(-1).cpu().numpy()
                    shuffled_av = model(
                        text, shuffled_audio, shuffled_vision, lav_mask
                    )["output_logit"].view(-1).cpu().numpy()
                for position, sample_index in enumerate(indices):
                    row = {
                        "State": state_name, "Split": split, "Repeat": repeat,
                        "SampleIndex": int(sample_index), "SampleID": identifiers[position],
                        "Label": float(label_numpy[position]),
                        "F_correct": float(correct[position]),
                        "F_shuffled_A": float(shuffled_a[position]),
                        "F_shuffled_V": float(shuffled_v[position]),
                        "F_shuffled_AV": float(shuffled_av[position]),
                    }
                    for modality in MODALITIES:
                        shuffled = row["F_shuffled_{}".format(modality)]
                        row["ShuffleDamage_{}".format(modality)] = float(
                            shuffle_damage([row["F_correct"]], [shuffled], [row["Label"]])[0]
                        )
                        row["PredictionShift_{}".format(modality)] = abs(
                            row["F_correct"] - shuffled
                        )
                    shuffle_rows.append(row)
            capture_enabled[0] = True
            if (batch_index + 1) % 20 == 0:
                print(
                    "state={} split={} batches={}/{}".format(
                        state_name, split, batch_index + 1, len(loaders[split])
                    ), flush=True,
                )
        handle.remove()
    model.train(training_before)
    if not tensor_maps_equal(parameters_before, clone_parameters(model)):
        raise RuntimeError("Student parameters changed during Stage7A audit.")
    if not tensor_maps_equal(buffers_before, clone_buffers(model)):
        raise RuntimeError("Student buffers changed during Stage7A audit.")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Input sensitivity populated a parameter gradient.")
    # DataLoader iterator construction consumes a Torch base seed even with
    # num_workers=0.  Stage7A therefore restores the complete pre-audit state
    # explicitly, then verifies the restoration byte-for-byte.
    restore_rng_state(rng_before)
    if not rng_states_equal(rng_before, capture_rng_state()):
        raise RuntimeError("Audit RNG state changed.")
    return {
        "predictions": prediction_rows, "gains": gain_rows,
        "shuffle": shuffle_rows, "sensitivity": sensitivity_rows,
        "representation": representation_rows,
    }


def metrics_from_arrays(predictions, labels):
    raw = regression_metrics(
        torch.tensor(np.asarray(predictions)), torch.tensor(np.asarray(labels))
    )
    return {
        "MAE": raw["MAE"], "Corr": raw["Corr"], "Acc2": raw["acc_2"],
        "F1": raw["F1_score"], "Acc5": raw["acc_5"], "Acc7": raw["acc_7"],
    }


def summary_rows(frame, value_column, keys, cli):
    rows = []
    for group, local in frame.groupby(keys, sort=True):
        group = group if isinstance(group, tuple) else (group,)
        stats = bootstrap_summary(
            local[value_column], cli.bootstrap_samples,
            stable_seed(cli.bootstrap_seed, value_column, *group),
        )
        rows.append(dict(zip(keys, group), Metric=value_column, **stats))
    return rows


def conditional_rows(base, grouping, cli):
    rows = []
    measures = ("Gain", "ShuffleDamage", "Sensitivity", "RepresentationShift")
    keys = ["State", "Split", "Modality", grouping]
    for group, local in base.groupby(keys, sort=True):
        for measure in measures:
            stats = bootstrap_summary(
                local[measure], cli.bootstrap_samples,
                stable_seed(cli.bootstrap_seed, grouping, measure, *group),
            )
            rows.append(dict(zip(keys, group), Measure=measure, **stats))
    return rows


def load_raw(directory):
    return {name: pd.read_csv(directory / name) for name in RAW_FILES}


def regenerate_summaries(directory, cli):
    raw = load_raw(directory)
    predictions = raw["sample_predictions.csv"]
    gains = raw["sample_modality_gains.csv"]
    shuffle = raw["shuffle_sample_metrics.csv"]
    sensitivity = raw["input_sensitivity_samples.csv"]
    representation = raw["representation_contribution_samples.csv"]
    sample_manifest = raw["sample_manifest.csv"]

    standard_rows = []
    for (state, split), local in predictions.groupby(["State", "Split"], sort=True):
        for mode in MODES:
            standard_rows.append({
                "State": state, "Split": split, "Mode": mode, "Count": len(local),
                **metrics_from_arrays(local["F_{}".format(mode)], local.Label),
            })
    standard = pd.DataFrame(standard_rows)
    standard.to_csv(directory / "standard_mode_metrics.csv", index=False)

    gain_summary = pd.DataFrame(summary_rows(
        gains, "Gain", ["State", "Split", "Modality"], cli
    ))
    gain_summary.to_csv(directory / "modality_gain_summary.csv", index=False)

    shuffle_aggregate = shuffle.groupby(
        ["State", "Split", "SampleIndex", "SampleID", "Label"], as_index=False, sort=True
    ).mean(numeric_only=True)
    shuffle_rows = []
    for (state, split), local in shuffle_aggregate.groupby(["State", "Split"], sort=True):
        correct_std = float(local.F_correct.std(ddof=0))
        for modality in MODALITIES:
            stats = bootstrap_summary(
                local["ShuffleDamage_{}".format(modality)], cli.bootstrap_samples,
                stable_seed(cli.bootstrap_seed, "shuffle", state, split, modality),
            )
            shuffle_rows.append({
                "State": state, "Split": split, "Modality": modality,
                **stats,
                "PredictionShiftMean": float(local["PredictionShift_{}".format(modality)].mean()),
                "PredictionShiftMedian": float(local["PredictionShift_{}".format(modality)].median()),
                "CorrectPredictionStd": correct_std,
                "PredictionUnderuseThreshold": 0.10 * correct_std,
            })
    shuffle_summary = pd.DataFrame(shuffle_rows)
    shuffle_summary.to_csv(directory / "shuffle_summary.csv", index=False)

    sensitivity_long = []
    for modality in UTILITY_MODALITIES:
        local = sensitivity.rename(columns={
            "GradNorm_{}".format(modality): "GradNorm",
            "Sensitivity_{}".format(modality): "Sensitivity",
        }).copy()
        local["Modality"] = modality
        sensitivity_long.append(local[[
            "State", "Split", "SampleIndex", "SampleID", "Label",
            "Modality", "GradNorm", "Sensitivity",
        ]])
    sensitivity_long = pd.concat(sensitivity_long, ignore_index=True)
    sensitivity_summary = []
    for (state, split, modality), local in sensitivity_long.groupby(
        ["State", "Split", "Modality"], sort=True
    ):
        grad_stats = bootstrap_summary(
            local.GradNorm, cli.bootstrap_samples,
            stable_seed(cli.bootstrap_seed, "gradnorm", state, split, modality),
        )
        sens_stats = bootstrap_summary(
            local.Sensitivity, cli.bootstrap_samples,
            stable_seed(cli.bootstrap_seed, "sensitivity", state, split, modality),
        )
        sensitivity_summary.append({
            "State": state, "Split": split, "Modality": modality,
            **{"GradNorm_{}".format(key): value for key, value in grad_stats.items()},
            **{"Sensitivity_{}".format(key): value for key, value in sens_stats.items()},
        })
    sensitivity_summary = pd.DataFrame(sensitivity_summary)
    sensitivity_summary.to_csv(directory / "input_sensitivity_summary.csv", index=False)

    representation_summary = []
    for (state, split), local in representation.groupby(["State", "Split"], sort=True):
        matrices = {
            mode: np.stack([vector_from_json(value) for value in local["H_{}".format(mode)]])
            for mode in MODES
        }
        for modality, mode in MODE_FOR_MODALITY.items():
            representation_summary.append({
                "State": state, "Split": split, "Modality": modality,
                "BaseMode": "L", "WithModalityMode": mode,
                **representation_pair_summary(matrices["L"], matrices[mode]),
            })
    representation_summary = pd.DataFrame(representation_summary)
    representation_summary.to_csv(
        directory / "representation_contribution_summary.csv", index=False
    )

    manifest_columns = [
        "Split", "SampleIndex", "Label", "LabelBin",
        "CompatQuartile_A", "CompatQuartile_V", "CompatQuartile_AV",
    ]
    base = gains.merge(
        shuffle_aggregate[[
            "State", "Split", "SampleIndex",
            "ShuffleDamage_A", "ShuffleDamage_V", "ShuffleDamage_AV",
        ]], on=["State", "Split", "SampleIndex"], validate="many_to_one",
    )
    base = base.merge(
        sensitivity_long[[
            "State", "Split", "SampleIndex", "Modality", "Sensitivity",
        ]], on=["State", "Split", "SampleIndex", "Modality"], how="left",
        validate="one_to_one",
    )
    relative = []
    for modality in MODALITIES:
        local = representation[[
            "State", "Split", "SampleIndex", "RelativeShift_{}".format(modality)
        ]].rename(columns={"RelativeShift_{}".format(modality): "RepresentationShift"})
        local["Modality"] = modality
        relative.append(local)
    relative = pd.concat(relative, ignore_index=True)
    base = base.merge(
        relative, on=["State", "Split", "SampleIndex", "Modality"], validate="one_to_one"
    )
    base["ShuffleDamage"] = [
        row["ShuffleDamage_{}".format(row.Modality)] for _, row in base.iterrows()
    ]
    base = base.merge(
        sample_manifest[manifest_columns], on=["Split", "SampleIndex", "Label"],
        validate="many_to_one",
    )
    base["CompatibilityQuartile"] = [
        row["CompatQuartile_{}".format(row.Modality)] for _, row in base.iterrows()
    ]
    pd.DataFrame(conditional_rows(base, "LabelBin", cli)).to_csv(
        directory / "conditional_label_bins.csv", index=False
    )
    pd.DataFrame(conditional_rows(base, "CompatibilityQuartile", cli)).to_csv(
        directory / "conditional_compatibility_quartiles.csv", index=False
    )

    predictions_with_error = predictions.copy()
    predictions_with_error["TextError"] = np.abs(
        predictions_with_error.F_L - predictions_with_error.Label
    )
    text_edges = {}
    text_groups = []
    for state in STATES:
        train_values = predictions_with_error.loc[
            predictions_with_error.State.eq(state) & predictions_with_error.Split.eq("train"),
            "TextError",
        ]
        edges = fit_quartile_edges(train_values)
        text_edges[state] = [float(value) for value in edges]
        local = predictions_with_error.loc[predictions_with_error.State.eq(state)].copy()
        local["TextErrorQuartile"] = assign_quartiles(local.TextError, edges)
        text_groups.append(local[["State", "Split", "SampleIndex", "TextErrorQuartile"]])
    text_groups = pd.concat(text_groups, ignore_index=True)
    base = base.merge(
        text_groups, on=["State", "Split", "SampleIndex"], validate="many_to_one"
    )
    pd.DataFrame(conditional_rows(base, "TextErrorQuartile", cli)).to_csv(
        directory / "conditional_text_error_quartiles.csv", index=False
    )

    comparison_rows = []
    paired_sources = {
        "Gain": gains,
        "ShuffleDamage": pd.concat([
            shuffle_aggregate.assign(
                Modality=modality,
                Value=shuffle_aggregate["ShuffleDamage_{}".format(modality)],
            )[["State", "Split", "SampleIndex", "Modality", "Value"]]
            for modality in MODALITIES
        ], ignore_index=True),
        "Sensitivity": sensitivity_long.rename(columns={"Sensitivity": "Value"}),
        "RelativeShift": relative.rename(columns={"RepresentationShift": "Value"}),
    }
    paired_sources["Gain"] = paired_sources["Gain"].rename(columns={"Gain": "Value"})
    for metric, source in paired_sources.items():
        for (split, modality), local in source.groupby(["Split", "Modality"], sort=True):
            pivot = local.pivot(index="SampleIndex", columns="State", values="Value")
            if set(STATES) - set(pivot.columns):
                continue
            difference = pivot["cfcompat_best_valid"] - pivot["gate3_init"]
            stats = bootstrap_summary(
                difference, cli.bootstrap_samples,
                stable_seed(cli.bootstrap_seed, "comparison", metric, split, modality),
            )
            comparison_rows.append({
                "Split": split, "Modality": modality, "Metric": metric,
                "Gate3Mean": float(pivot["gate3_init"].mean()),
                "CFCompatMean": float(pivot["cfcompat_best_valid"].mean()),
                "DeltaCFMinusGate3": float(difference.mean()),
                "DeltaCILow": stats["ci_low"], "DeltaCIHigh": stats["ci_high"],
            })
    for (split, modality), local in representation_summary.groupby(
        ["Split", "Modality"], sort=True
    ):
        values = local.set_index("State").WithModalityEffectiveRank
        comparison_rows.append({
            "Split": split, "Modality": modality, "Metric": "EffectiveRank",
            "Gate3Mean": float(values["gate3_init"]),
            "CFCompatMean": float(values["cfcompat_best_valid"]),
            "DeltaCFMinusGate3": float(
                values["cfcompat_best_valid"] - values["gate3_init"]
            ),
            "DeltaCILow": np.nan, "DeltaCIHigh": np.nan,
        })
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(directory / "model_state_comparison.csv", index=False)

    classifications = {state: {} for state in STATES}
    for state in STATES:
        for modality in UTILITY_MODALITIES:
            gain_row = gain_summary[
                gain_summary.State.eq(state) & gain_summary.Split.eq("valid")
                & gain_summary.Modality.eq(modality)
            ].iloc[0]
            damage_row = shuffle_summary[
                shuffle_summary.State.eq(state) & shuffle_summary.Split.eq("valid")
                & shuffle_summary.Modality.eq(modality)
            ].iloc[0]
            rep_row = representation_summary[
                representation_summary.State.eq(state)
                & representation_summary.Split.eq("valid")
                & representation_summary.Modality.eq(modality)
            ].iloc[0]
            classifications[state][modality] = classify_utility(
                gain_row, damage_row, damage_row.PredictionShiftMean,
                damage_row.CorrectPredictionStd, rep_row.RelativeShiftMean,
            )
    changes = {}
    for modality in UTILITY_MODALITIES:
        gain_delta = comparison[
            comparison.Split.eq("valid") & comparison.Modality.eq(modality)
            & comparison.Metric.eq("Gain")
        ].iloc[0]
        damage_delta = comparison[
            comparison.Split.eq("valid") & comparison.Modality.eq(modality)
            & comparison.Metric.eq("ShuffleDamage")
        ].iloc[0]
        changes[modality] = classify_change(
            classifications["gate3_init"][modality],
            classifications["cfcompat_best_valid"][modality],
            {"ci_low": gain_delta.DeltaCILow, "ci_high": gain_delta.DeltaCIHigh},
            {"ci_low": damage_delta.DeltaCILow, "ci_high": damage_delta.DeltaCIHigh},
        )
    if any("Underuse" in value for values in classifications.values() for value in values.values()):
        next_direction = (
            "Manual review should prioritize why the affected non-text modality is weakly "
            "coupled to prediction and shared representation; no Stage7B method is authorized."
        )
    elif any("Unreliable" in value for values in classifications.values() for value in values.values()):
        next_direction = (
            "Manual review should prioritize conditional reliability/calibration of the used "
            "non-text modality; do not add a method before reviewing the conditional tables."
        )
    elif all("Utility Supported" in value for values in classifications.values() for value in values.values()):
        next_direction = (
            "Both modalities show generalizable utility; preserve their contribution and "
            "review conditional weak bins before considering any future method."
        )
    else:
        next_direction = (
            "Evidence is mixed; inspect label, compatibility, and text-error strata before "
            "authorizing any follow-up method."
        )

    config = json.loads((directory / "audit_config.json").read_text())
    summary = {
        "Dataset": cli.dataset, "Seed": cli.seed,
        "TrainAccessed": True, "ValidAccessed": True, "TestAccessed": False,
        "OptimizerStepCount": 0, "ParameterChanged": False, "BufferChanged": False,
        "RNGPreserved": True, "TeacherGradientCount": 0,
        "TrainSampleCount": int(sample_manifest.Split.eq("train").sum()),
        "ValidSampleCount": int(sample_manifest.Split.eq("valid").sum()),
        "ModelStates": list(STATES), "ShuffleRepeats": int(config["shuffle_repeats"]),
        "BootstrapSamples": int(config["bootstrap_samples"]),
        "AudioUtilityClassification": {
            state: classifications[state]["A"] for state in STATES
        },
        "VisionUtilityClassification": {
            state: classifications[state]["V"] for state in STATES
        },
        "CFCompatChangedAudioUse": changes["A"],
        "CFCompatChangedVisionUse": changes["V"],
        "EvidenceBasedNextDirection": next_direction,
        "TextErrorTrainQuartileEdges": text_edges,
        "SummaryOnlyReproducible": True,
        "StopDeclaration": "Stage7A complete; no Stage7B branch or method was created.",
    }
    (directory / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(
        directory, summary, gain_summary, shuffle_summary,
        sensitivity_summary, representation_summary, comparison,
    )


def markdown_table(frame, columns):
    local = frame.loc[:, columns].copy()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in local.itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value):
                values.append("NA")
            elif isinstance(value, (float, np.floating)):
                values.append("{:.6f}".format(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(directory, summary, gains, shuffles, sensitivity, representations, comparison):
    valid_gain = gains[gains.Split.eq("valid")]
    valid_shuffle = shuffles[shuffles.Split.eq("valid")]
    valid_rep = representations[
        representations.Split.eq("valid") & representations.Modality.isin(UTILITY_MODALITIES)
    ]
    valid_sensitivity = sensitivity[sensitivity.Split.eq("valid")]
    valid_comparison = comparison[
        comparison.Split.eq("valid") & comparison.Modality.isin(UTILITY_MODALITIES)
    ]
    lines = [
        "# Stage 7A Modality Utility and Conditional Contribution Audit", "",
        "This is a frozen MOSI train/valid audit. Test was not accessed, no optimizer "
        "was created, no checkpoint was written, and no model parameter or buffer changed.", "",
        "## Registered classifications", "",
        "- Gate3 audio: **{}**".format(summary["AudioUtilityClassification"]["gate3_init"]),
        "- CFCompat audio: **{}**".format(summary["AudioUtilityClassification"]["cfcompat_best_valid"]),
        "- Gate3 vision: **{}**".format(summary["VisionUtilityClassification"]["gate3_init"]),
        "- CFCompat vision: **{}**".format(summary["VisionUtilityClassification"]["cfcompat_best_valid"]),
        "- CFCompat change in audio use: **{}**".format(summary["CFCompatChangedAudioUse"]),
        "- CFCompat change in vision use: **{}**".format(summary["CFCompatChangedVisionUse"]), "",
        "## Valid sample-level gain", "",
        markdown_table(valid_gain, [
            "State", "Modality", "count", "mean", "median", "std",
            "positive_fraction", "negative_fraction", "ci_low", "ci_high",
        ]), "",
        "## Valid derangement damage", "",
        markdown_table(valid_shuffle, [
            "State", "Modality", "count", "mean", "median", "ci_low", "ci_high",
            "PredictionShiftMean", "CorrectPredictionStd",
        ]), "",
        "## Valid input sensitivity", "",
        markdown_table(valid_sensitivity, [
            "State", "Modality", "Sensitivity_count", "Sensitivity_mean",
            "Sensitivity_median", "Sensitivity_ci_low", "Sensitivity_ci_high",
        ]), "",
        "## Valid shared-representation contribution", "",
        markdown_table(valid_rep, [
            "State", "Modality", "NormMean", "RelativeShiftMean",
            "SameSampleCosineMean", "LinearCKA", "WithModalityFeatureVariance",
            "WithModalityEffectiveRank",
        ]), "",
        "## CFCompat minus Gate3 on valid", "",
        markdown_table(valid_comparison, [
            "Modality", "Metric", "Gate3Mean", "CFCompatMean",
            "DeltaCFMinusGate3", "DeltaCILow", "DeltaCIHigh",
        ]), "",
        "## Evidence-based next direction", "", summary["EvidenceBasedNextDirection"], "",
        "## Isolation checks", "",
        "- TrainAccessed={}; ValidAccessed={}; TestAccessed={}".format(
            summary["TrainAccessed"], summary["ValidAccessed"], summary["TestAccessed"]
        ),
        "- OptimizerStepCount={}; ParameterChanged={}; BufferChanged={}; RNGPreserved={}".format(
            summary["OptimizerStepCount"], summary["ParameterChanged"],
            summary["BufferChanged"], summary["RNGPreserved"],
        ),
        "- All ten split-local derangements were fixed-point-free bijections and shared "
        "by both model states.",
        "- Shared representation is exactly the Stage6A `backbone.proj1` forward-pre-hook input.",
        "- Compatibility quartiles and L-only error quartiles use train-fitted boundaries "
        "for both train and valid.", "",
        "## Stop declaration", "", summary["StopDeclaration"], "",
    ]
    (directory / "stage7a_modality_utility_audit.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def refresh_manifest(directory):
    artifacts = [
        path for path in directory.iterdir()
        if path.is_file() and path.name != "audit_manifest.json"
    ]
    manifest = {
        "artifacts": {path.name: sha256_file(path) for path in sorted(artifacts)},
        "artifact_count": len(artifacts),
        "train_accessed": True, "valid_accessed": True, "test_accessed": False,
        "checkpoint_created": False, "optimizer_step_count": 0,
    }
    (directory / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def run_full(cli):
    setup_seed(cli.seed)
    args = build_config(cli)
    directory = output_directory(cli)
    directory.mkdir(parents=True, exist_ok=True)
    datasets, loaders = build_audit_datasets(args, cli.num_workers)
    if len(datasets["train"]) != 1284:
        raise RuntimeError("Locked MOSI train sample count changed.")
    paths = locate_validation_selected_states(cli.result_root, cli.seed)
    train_cache, cache_artifacts, cache_config = load_train_compatibility(
        cli.result_root, cli.dataset
    )
    evaluator_checkpoint, evaluator_epoch, evaluator_source = locate_stage1_evaluator(
        cli.result_root, cli.dataset, cli.seed
    )
    if cache_config.get("evaluator_sha256") != checkpoint_sha256(evaluator_checkpoint):
        raise ValueError("Compatibility cache/evaluator SHA binding failed.")
    valid_compat = build_valid_compatibility(
        args, loaders["valid"], train_cache, evaluator_checkpoint
    )
    sample_manifest = build_sample_manifest(datasets, train_cache, valid_compat)
    mappings, shuffle_manifest = build_derangements(datasets, cli)
    sample_manifest.to_csv(directory / "sample_manifest.csv", index=False)
    shuffle_manifest.to_csv(directory / "shuffle_manifest.csv", index=False)
    compatibility_edges = {
        modality: [
            float(value) for value in fit_quartile_edges(
                sample_manifest.loc[
                    sample_manifest.Split.eq("train"), COMPAT_FOR_MODALITY[modality]
                ]
            )
        ]
        for modality in MODALITIES
    }
    config = {
        "dataset": cli.dataset, "seed": cli.seed,
        "allowed_splits": list(SPLITS), "train_accessed": True,
        "valid_accessed": True, "test_accessed": False,
        "shuffle_repeats": cli.shuffle_repeats,
        "derangement_seed_base": cli.derangement_seed,
        "derangement_seeds": {
            split: [cli.derangement_seed + index * 1000 + repeat
                    for repeat in range(cli.shuffle_repeats)]
            for index, split in enumerate(SPLITS)
        },
        "bootstrap_samples": cli.bootstrap_samples,
        "bootstrap_seed": cli.bootstrap_seed, "num_workers": cli.num_workers,
        "rng_restore_policy": (
            "restore full Python/NumPy/Torch/CUDA state after each frozen-state audit; "
            "required because DataLoader iterators consume a Torch base seed"
        ),
        "optimizer_created": False, "optimizer_step_count": 0,
        "aligned_data": bool(args.need_data_aligned),
        "explicit_audio_lengths_present": hasattr(datasets["train"], "audio_lengths"),
        "explicit_vision_lengths_present": hasattr(datasets["train"], "vision_lengths"),
        "padding_mask_rule": "nonzero feature rows for aligned MOSI",
        "shuffle_bound_fields": (
            "complete modality feature tensor; inferred padding mask and valid-step count "
            "are functions of and therefore move with that tensor"
        ),
        "same_derangement_for_audio_and_vision_within_repeat": True,
        "same_derangements_for_both_model_states": True,
        "compatibility_valid_mapping": (
            "frozen Stage1 evaluator delta mapped through train empirical CDF"
        ),
        "compatibility_train_quartile_edges": compatibility_edges,
        "label_bins": list(LABEL_BINS),
        "representation_feature_key": "MissingModalityWrapper.backbone.proj1 forward-pre-hook input",
        "base_branch": BASE_BRANCH, "base_commit": BASE_COMMIT,
        "implementation_head_at_audit": git("rev-parse", "HEAD"),
        "gate3_checkpoint": str(paths["gate3"]),
        "gate3_checkpoint_sha256": checkpoint_sha256(paths["gate3"]),
        "gate3_best_valid_epoch": paths["gate_epoch"],
        "cfcompat_checkpoint": str(paths["cfcompat"]),
        "cfcompat_checkpoint_sha256": checkpoint_sha256(paths["cfcompat"]),
        "gate3_manifest": str(paths["gate_manifest"]),
        "cfcompat_manifest": str(paths["cf_manifest"]),
        "compatibility_cache": str(cache_artifacts["csv"]),
        "compatibility_cache_sha256": checkpoint_sha256(cache_artifacts["csv"]),
        "evaluator_checkpoint": str(evaluator_checkpoint),
        "evaluator_checkpoint_sha256": checkpoint_sha256(evaluator_checkpoint),
        "evaluator_best_valid_epoch": int(evaluator_epoch),
        "evaluator_manifest_source": str(evaluator_source),
    }
    (directory / "audit_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    combined = {
        "predictions": [], "gains": [], "shuffle": [],
        "sensitivity": [], "representation": [],
    }
    global_rng = capture_rng_state()
    for state_name in STATES:
        model = initialize_state(state_name, args, paths)
        outcome = audit_one_state(
            state_name, model, datasets, loaders, mappings, args
        )
        for key in combined:
            combined[key].extend(outcome[key])
        del model
        torch.cuda.empty_cache()
    if not rng_states_equal(global_rng, capture_rng_state()):
        restore_rng_state(global_rng)
        raise RuntimeError("Global Stage7A audit RNG was not preserved.")
    pd.DataFrame(combined["predictions"]).to_csv(
        directory / "sample_predictions.csv", index=False
    )
    pd.DataFrame(combined["gains"]).to_csv(
        directory / "sample_modality_gains.csv", index=False
    )
    pd.DataFrame(combined["shuffle"]).to_csv(
        directory / "shuffle_sample_metrics.csv", index=False
    )
    pd.DataFrame(combined["sensitivity"]).to_csv(
        directory / "input_sensitivity_samples.csv", index=False
    )
    pd.DataFrame(combined["representation"]).to_csv(
        directory / "representation_contribution_samples.csv", index=False
    )
    regenerate_summaries(directory, cli)
    manifest = refresh_manifest(directory)
    print(json.dumps({
        "complete": True, "directory": str(directory),
        "artifact_count": manifest["artifact_count"],
        "states": list(STATES), "splits": list(SPLITS),
    }, indent=2, sort_keys=True))


def verify_existing(cli):
    directory = output_directory(cli)
    old_manifest = json.loads((directory / "audit_manifest.json").read_text())
    for name, expected in old_manifest["artifacts"].items():
        path = directory / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError("Artifact SHA mismatch before summary-only: {}".format(name))
    for required in RAW_FILES:
        if not (directory / required).is_file():
            raise FileNotFoundError(required)
    regenerate_summaries(directory, cli)
    new_manifest = refresh_manifest(directory)
    if old_manifest != new_manifest:
        changed = sorted(
            name for name, digest in new_manifest["artifacts"].items()
            if old_manifest["artifacts"].get(name) != digest
        )
        raise ValueError("summary-only was not byte-reproducible: {}".format(changed))
    print(json.dumps({
        "verified": True, "summary_only_reproducible": True,
        "artifact_count": new_manifest["artifact_count"],
        "test_accessed": False,
    }, indent=2, sort_keys=True))


def main():
    cli = parse_args()
    if cli.summary_only:
        verify_existing(cli)
    else:
        run_full(cli)


if __name__ == "__main__":
    main()
