"""Stage 2.5 frozen-model, sample-level, and one-time milestone audit."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional

from config import get_config_regression
from trains.singleTask.fixed_kd_utils import fixed_kd_checkpoint_path
from trains.singleTask.missing_utils import (
    MissingModalityWrapper,
    apply_direct_mask,
    build_single_split_loader,
    clean_checkpoint_path,
    missing_checkpoint_path,
    mode_to_mask,
)
from trains.singleTask.model.DLF import DLF
from trains.singleTask.stage2_audit_utils import (
    AUDIT_METHODS,
    AUDIT_MODES,
    METRIC_KEYS,
    assert_frozen_eval_model,
    checkpoint_metadata,
    compare_metrics_to_reference,
    contribution_statistics,
    counterfactual_contributions,
    dataframe_to_csv,
    describe_abs_differences,
    dominance_statistics,
    error_change_statistics,
    homogenization_label,
    label_distribution,
    metrics_from_predictions,
    missing_macro_metrics,
    pearson_or_nan,
    total_variation_distance,
    validate_counterfactual_identities,
)
from utils.functions import assign_gpu, setup_seed

REFERENCE_VALIDATION_CSVS = {
    "gate3_directmask": Path("result/missing_baseline/directmask/valid/mosi_per_seed.csv"),
    "moddrop": Path("result/missing_baseline/moddrop/train/mosi_per_seed.csv"),
    "fixedkd": Path("result/missing_baseline/fixed_kd/train/mosi_per_seed.csv"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Frozen Stage 2.5 MOSI audit.")
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--split", choices=("valid", "test"), required=True)
    parser.add_argument("--confirm-stage2-milestone-test", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-root", default="result/milestone_audit/stage2")
    parser.add_argument("--config-file", default="config/config.json")
    parsed = parser.parse_args()
    if parsed.split == "test" and not parsed.confirm_stage2_milestone_test:
        parser.error(
            "Refusing test audit without --confirm-stage2-milestone-test."
        )
    return parsed


def build_config(cli_args):
    args = get_config_regression("DLF", cli_args.dataset, cli_args.config_file)
    args.feature_T = ""
    args.feature_A = ""
    args.feature_V = ""
    args.mode = cli_args.split
    args.is_training = False
    args.train_mode = "regression"
    args.seed = int(cli_args.seed)
    args.cur_seed = int(cli_args.seed)
    args.device = assign_gpu(list(cli_args.gpu_ids))
    return args


def checkpoint_paths(cli_args, args):
    paths = {
        "gate3_directmask": clean_checkpoint_path(
            cli_args.model_save_dir, args.dataset_name, cli_args.seed
        ),
        "moddrop": missing_checkpoint_path(
            cli_args.model_save_dir, args.dataset_name, cli_args.seed
        ),
        "fixedkd": fixed_kd_checkpoint_path(
            cli_args.model_save_dir, args.dataset_name, cli_args.seed
        ),
    }
    if len(set(paths.values())) != len(paths):
        raise RuntimeError("Gate 3, ModDrop, and FixedKD checkpoints must differ.")
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError("{} checkpoint not found: {}".format(name, path))
    return paths


def _freeze_for_audit(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_frozen_models(cli_args, args):
    paths = checkpoint_paths(cli_args, args)
    metadata = {}
    models = {}
    for name, path in paths.items():
        state_dict = torch.load(path, map_location=args.device)
        metadata[name] = checkpoint_metadata(path)
        metadata[name]["state_dict_key_count"] = int(len(state_dict))
        if name == "gate3_directmask":
            model = DLF(args).to(args.device)
        else:
            model = MissingModalityWrapper(
                DLF(args).to(args.device),
                args.feature_dims[1],
                args.feature_dims[2],
            ).to(args.device)
        model.load_state_dict(state_dict, strict=True)
        models[name] = _freeze_for_audit(model)
        assert_frozen_eval_model(models[name], name)
    return models, paths, metadata


def _batch_sample_ids(batch_data, expected_start):
    indices = batch_data["index"]
    if torch.is_tensor(indices):
        indices = indices.detach().cpu().view(-1).tolist()
    else:
        indices = list(indices)
    expected = list(range(expected_start, expected_start + len(indices)))
    if [int(item) for item in indices] != expected:
        raise RuntimeError("Non-shuffled sample index sequence is not stable.")
    ids = batch_data.get("id")
    if ids is None:
        return [str(index) for index in indices], [int(index) for index in indices]
    if torch.is_tensor(ids):
        ids = ids.detach().cpu().view(-1).tolist()
    else:
        ids = list(ids)
    return [str(item) for item in ids], [int(index) for index in indices]


def collect_predictions(models, dataloader, device):
    """One unified, no-gradient forward audit over all three frozen models."""
    storage = {
        method: {mode: {"prediction": [], "batch_loss": []} for mode in AUDIT_MODES}
        for method in AUDIT_METHODS
    }
    labels = []
    sample_ids = []
    sample_indices = []
    expected_start = 0
    for model in models.values():
        assert_frozen_eval_model(model, "pre-forward model")

    with torch.inference_mode():
        for batch_data in dataloader:
            batch_ids, batch_indices = _batch_sample_ids(batch_data, expected_start)
            expected_start += len(batch_indices)
            text = batch_data["text"].to(device)
            audio = batch_data["audio"].to(device)
            vision = batch_data["vision"].to(device)
            label = batch_data["labels"]["M"].to(device).view(-1, 1)
            labels.append(label.detach().cpu().numpy().reshape(-1).copy())
            sample_ids.extend(batch_ids)
            sample_indices.extend(batch_indices)

            for mode in AUDIT_MODES:
                mask = mode_to_mask(
                    mode, batch_size=label.size(0), device=device, dtype=audio.dtype
                )
                direct_audio, direct_vision = apply_direct_mask(audio, vision, mask)
                gate_prediction = models["gate3_directmask"](
                    text, direct_audio, direct_vision
                )["output_logit"]
                moddrop_prediction = models["moddrop"](
                    text, audio, vision, mask
                )["output_logit"]
                fixedkd_prediction = models["fixedkd"](
                    text, audio, vision, mask
                )["output_logit"]
                for method, prediction in (
                    ("gate3_directmask", gate_prediction),
                    ("moddrop", moddrop_prediction),
                    ("fixedkd", fixedkd_prediction),
                ):
                    storage[method][mode]["prediction"].append(
                        prediction.detach().cpu().numpy().reshape(-1).copy()
                    )
                    storage[method][mode]["batch_loss"].append(
                        float(functional.l1_loss(prediction, label).item())
                    )

    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Original sample ids must be unique in a split audit.")
    if sample_indices != list(range(len(sample_indices))):
        raise RuntimeError("Sample indices are not stable zero-based indices.")
    labels = np.concatenate(labels)
    output = {}
    metrics = {}
    for method in AUDIT_METHODS:
        output[method] = {}
        metrics[method] = {}
        for mode in AUDIT_MODES:
            prediction = np.concatenate(storage[method][mode]["prediction"])
            if prediction.size != labels.size:
                raise RuntimeError("{} {} sample count differs from labels.".format(method, mode))
            output[method][mode] = prediction
            metrics[method][mode] = metrics_from_predictions(
                prediction, labels, storage[method][mode]["batch_loss"]
            )
    for name, model in models.items():
        assert_frozen_eval_model(model, name)
    return {
        "sample_id": sample_ids,
        "sample_index": sample_indices,
        "label": labels,
        "prediction": output,
        "metrics": metrics,
    }


def prediction_frame(collected):
    frame = pd.DataFrame(
        {
            "sample_id": collected["sample_id"],
            "sample_index": collected["sample_index"],
            "label": collected["label"],
        }
    )
    for method in AUDIT_METHODS:
        for mode in AUDIT_MODES:
            frame["{}_{}_pred".format(method, mode)] = collected["prediction"][method][mode]
    return frame


def overall_metric_frame(metrics_by_method):
    records = []
    for method in AUDIT_METHODS:
        macro = missing_macro_metrics(metrics_by_method[method])
        for mode in AUDIT_MODES:
            record = {"method": method, "mode": mode}
            record.update(metrics_by_method[method][mode])
            record.update(macro)
            records.append(record)
    return pd.DataFrame(records)


def mode_difference_frame(predictions):
    records = []
    base = {}
    for method in ("moddrop", "fixedkd"):
        base[method] = {}
        for mode in ("LA", "LV", "L"):
            stats = describe_abs_differences(
                predictions[method]["LAV"], predictions[method][mode]
            )
            base[method][mode] = stats
    for method in ("moddrop", "fixedkd"):
        for mode in ("LA", "LV", "L"):
            record = {"method": method, "mode": mode}
            record.update(base[method][mode])
            if method == "fixedkd":
                denominator = base["moddrop"][mode]["mean"]
                ratio = np.nan if denominator == 0.0 else base["fixedkd"][mode]["mean"] / denominator
                record["homogenization_ratio_mode"] = ratio
                record["homogenization_interpretation"] = (
                    "undefined_zero_moddrop_mean" if np.isnan(ratio) else homogenization_label(ratio)
                )
            else:
                record["homogenization_ratio_mode"] = 1.0
                record["homogenization_interpretation"] = "reference_moddrop"
            records.append(record)
    return pd.DataFrame(records)


def error_change_frame(predictions, labels):
    records = []
    for method in ("moddrop", "fixedkd"):
        for mode in ("LA", "LV", "L"):
            record = {"method": method, "mode": mode}
            record.update(
                error_change_statistics(
                    predictions[method]["LAV"], predictions[method][mode], labels
                )
            )
            records.append(record)
    return pd.DataFrame(records)


def contribution_frames(collected):
    contributions = counterfactual_contributions(collected["prediction"]["moddrop"])
    identity_maxima = validate_counterfactual_identities(contributions)
    frame = pd.DataFrame(
        {
            "sample_id": collected["sample_id"],
            "sample_index": collected["sample_index"],
            "label": collected["label"],
        }
    )
    for mode in AUDIT_MODES:
        frame["moddrop_{}_pred".format(mode)] = collected["prediction"]["moddrop"][mode]
    for key, values in contributions.items():
        frame[key] = values

    summary = []
    for variable in (
        "r_A",
        "r_V",
        "r_AV",
        "delta_missing_LA",
        "delta_missing_LV",
        "delta_missing_L",
    ):
        record = {"record_type": "contribution", "variable": variable}
        record.update(contribution_statistics(contributions[variable]))
        summary.append(record)
    dominant = {"record_type": "dominance", "variable": "r_A_r_V_r_AV"}
    dominant.update(dominance_statistics(contributions))
    summary.append(dominant)
    identity = {"record_type": "identity", "variable": "counterfactual_identities"}
    identity.update(identity_maxima)
    summary.append(identity)
    return frame, pd.DataFrame(summary), contributions, identity_maxima


def contribution_association_frame(collected, contributions):
    labels = collected["label"]
    predictions = collected["prediction"]
    records = []
    for method in ("moddrop", "fixedkd"):
        for mode in ("LA", "LV", "L"):
            contribution_key = "delta_missing_{}".format(mode)
            delta_missing = contributions[contribution_key]
            missing_error = np.abs(predictions[method][mode] - labels)
            error_delta = missing_error - np.abs(predictions[method]["LAV"] - labels)
            records.append(
                {
                    "record_type": "correlation",
                    "method": method,
                    "mode": mode,
                    "correlation": pearson_or_nan(np.abs(delta_missing), missing_error),
                    "correlation_name": "abs_delta_missing_vs_abs_missing_error",
                }
            )
            records.append(
                {
                    "record_type": "correlation",
                    "method": method,
                    "mode": mode,
                    "correlation": pearson_or_nan(np.abs(delta_missing), error_delta),
                    "correlation_name": "abs_delta_missing_vs_delta_error",
                }
            )
            magnitude = np.abs(delta_missing)
            boundaries = np.quantile(magnitude, [0.0, 0.25, 0.5, 0.75, 1.0])
            for quartile in range(4):
                low, high = boundaries[quartile], boundaries[quartile + 1]
                if quartile == 0:
                    membership = magnitude <= high
                elif quartile == 3:
                    membership = magnitude > low
                else:
                    membership = (magnitude > low) & (magnitude <= high)
                moddrop_mae = np.mean(
                    np.abs(predictions["moddrop"][mode][membership] - labels[membership])
                ) if np.any(membership) else np.nan
                fixedkd_mae = np.mean(
                    np.abs(predictions["fixedkd"][mode][membership] - labels[membership])
                ) if np.any(membership) else np.nan
                fixed_better = np.mean(
                    np.abs(predictions["fixedkd"][mode][membership] - labels[membership])
                    < np.abs(predictions["moddrop"][mode][membership] - labels[membership])
                ) if np.any(membership) else np.nan
                records.append(
                    {
                        "record_type": "quartile",
                        "method": "moddrop_vs_fixedkd",
                        "mode": mode,
                        "quartile": quartile + 1,
                        "lower_bound": float(low),
                        "upper_bound": float(high),
                        "sample_count": int(np.sum(membership)),
                        "moddrop_mae": float(moddrop_mae),
                        "fixedkd_mae": float(fixedkd_mae),
                        "fixedkd_minus_moddrop_mae": float(fixedkd_mae - moddrop_mae),
                        "fixedkd_better_fraction": float(fixed_better),
                    }
                )
    return pd.DataFrame(records)


def collect_labels_for_split(args, split, num_workers):
    loader = build_single_split_loader(args, split, num_workers)
    values = []
    for batch_data in loader:
        values.append(batch_data["labels"]["M"].view(-1).cpu().numpy().copy())
    return np.concatenate(values)


def label_distribution_frame(args, cli_args, current_labels):
    observed = {"train": collect_labels_for_split(args, "train", cli_args.num_workers)}
    if cli_args.split == "valid":
        observed["valid"] = current_labels
    else:
        observed["valid"] = collect_labels_for_split(args, "valid", cli_args.num_workers)
        observed["test"] = current_labels
    distributions = {split: label_distribution(values) for split, values in observed.items()}
    records = []
    for split, distribution in distributions.items():
        summary = {"record_type": "summary", "split": split}
        summary.update({key: value for key, value in distribution.items() if key != "bins"})
        records.append(summary)
        for bin_record in distribution["bins"]:
            records.append({"record_type": "bin", "split": split, **bin_record})
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if left in distributions and right in distributions:
            left_bins = [item["proportion"] for item in distributions[left]["bins"]]
            right_bins = [item["proportion"] for item in distributions[right]["bins"]]
            records.append(
                {
                    "record_type": "tvd",
                    "comparison": "{}-{}".format(left, right),
                    "total_variation_distance": total_variation_distance(left_bins, right_bins),
                }
            )
    return pd.DataFrame(records), distributions


def validation_reference_audit(metrics, cli_args):
    results = {}
    for method, path in REFERENCE_VALIDATION_CSVS.items():
        if not path.is_file():
            raise FileNotFoundError("Validation reference CSV not found: {}".format(path))
        results[method] = compare_metrics_to_reference(
            metrics[method], path, cli_args.seed, tolerance=1e-6
        )
    return results


def git_metadata():
    return {
        "git_branch": subprocess.check_output(
            ["git", "branch", "--show-current"], text=True
        ).strip(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }


def write_json_atomically(path, payload):
    path = Path(path)
    temporary = path.with_name("{}.tmp".format(path.name))
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_audit_outputs(output_dir, collected, checkpoint_info, label_frame, reference):
    output_dir = Path(output_dir)
    dataframe_to_csv(prediction_frame(collected), output_dir / "mosi_seed1111_predictions.csv")
    dataframe_to_csv(overall_metric_frame(collected["metrics"]), output_dir / "overall_metrics.csv")
    dataframe_to_csv(mode_difference_frame(collected["prediction"]), output_dir / "mode_difference_summary.csv")
    dataframe_to_csv(error_change_frame(collected["prediction"], collected["label"]), output_dir / "error_change_summary.csv")
    contribution_frame, summary_frame, contributions, identity_maxima = contribution_frames(collected)
    dataframe_to_csv(contribution_frame, output_dir / "counterfactual_contributions.csv")
    dataframe_to_csv(summary_frame, output_dir / "counterfactual_summary.csv")
    dataframe_to_csv(
        contribution_association_frame(collected, contributions),
        output_dir / "contribution_error_association.csv",
    )
    dataframe_to_csv(label_frame, output_dir / "label_distribution.csv")
    summary = {
        "audit_name": "stage2_counterfactual_and_milestone_audit",
        "dataset": "mosi",
        "seed": 1111,
        "split": str(output_dir.name),
        "sample_count": int(len(collected["label"])),
        "checkpoint_metadata": checkpoint_info,
        "identity_max_errors": identity_maxima,
        "validation_reference": reference,
        "metrics": {
            method: {
                "modes": collected["metrics"][method],
                "missing_macro": missing_macro_metrics(collected["metrics"][method]),
            }
            for method in AUDIT_METHODS
        },
    }
    write_json_atomically(output_dir / "audit_summary.json", summary)
    return summary


def assert_test_not_previously_completed(marker_path, test_output_dir):
    marker_path = Path(marker_path)
    test_output_dir = Path(test_output_dir)
    if marker_path.exists():
        previous = marker_path.read_text(encoding="utf-8")
        raise RuntimeError("Test milestone already completed; refusing rerun: {}".format(previous))
    if test_output_dir.exists():
        raise RuntimeError(
            "Test output directory already exists without a completion marker; refusing rerun: {}".format(test_output_dir)
        )


def run_audit(cli_args, output_dir):
    setup_seed(cli_args.seed)
    args = build_config(cli_args)
    models, paths, metadata = load_frozen_models(cli_args, args)
    loader = build_single_split_loader(args, cli_args.split, cli_args.num_workers)
    collected = collect_predictions(models, loader, args.device)
    reference = validation_reference_audit(collected["metrics"], cli_args) if cli_args.split == "valid" else {}
    labels_frame, _ = label_distribution_frame(args, cli_args, collected["label"])
    summary = write_audit_outputs(output_dir, collected, metadata, labels_frame, reference)
    del models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, metadata


def main():
    cli_args = parse_args()
    root = Path(cli_args.result_root)
    output_dir = root / cli_args.split
    if cli_args.split == "valid":
        summary, _ = run_audit(cli_args, output_dir)
        print("validation_audit_passed samples={} output={}".format(summary["sample_count"], output_dir))
        return

    marker = output_dir / "AUDIT_COMPLETED.json"
    assert_test_not_previously_completed(marker, output_dir)
    root.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=".stage2-test-audit-", dir=root))
    try:
        summary, metadata = run_audit(cli_args, temporary_dir)
        os.replace(temporary_dir, output_dir)
        completion = {
            "audit_name": "stage2_counterfactual_and_milestone_audit",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **git_metadata(),
            "seed": int(cli_args.seed),
            "dataset": cli_args.dataset,
            "split": "test",
            "checkpoint_metadata": metadata,
            "sample_count": int(summary["sample_count"]),
            "command": list(sys.argv),
            "success": True,
        }
        write_json_atomically(marker, completion)
        print("test_milestone_audit_completed samples={} output={}".format(summary["sample_count"], output_dir))
    except Exception as error:
        failure_log = temporary_dir / "AUDIT_FAILED.txt"
        failure_log.write_text("{}\n".format(error), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
