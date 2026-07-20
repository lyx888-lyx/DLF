"""Audit Ghost Modality behavior of the frozen Stage 19 Uniform checkpoint."""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.missing_utils import (
    MISSING_MODES,
    MissingModalityWrapper,
    mode_to_mask,
    regression_metrics,
)
from trains.singleTask.model.DLF import DLF
from trains.singleTask.safe_dlf_utils import gradient_norm
from utils.functions import assign_gpu, setup_seed


BRANCHES = {"audio": 1, "vision": 2}
MODE_MISSING = {"LA": ("vision",), "LV": ("audio",), "L": ("audio", "vision")}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--uniform-checkpoint", required=True)
    parser.add_argument("--uniform-epoch-metrics", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gpu-id", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=1)
    return parser.parse_args()


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(str(temporary), str(path))


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_frame(path, rows, sep="\t"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, sep=sep, index=False)
    os.replace(str(temporary), str(path))


def build_args(cli):
    args = get_config_regression("DLF", "mosei", cli.config_file)
    args.mode = "train"
    args.feature_T = args.feature_A = args.feature_V = ""
    args.is_training = True
    args.train_mode = "regression"
    args.seed = args.cur_seed = 1111
    args.device = assign_gpu([cli.gpu_id])
    return args


def feature_statistics(loader):
    sums = {}
    squares = {}
    counts = {}
    for batch in loader:
        for name in ("audio", "vision"):
            values = batch[name].float().reshape(-1, batch[name].size(-1))
            sums[name] = sums.get(name, 0) + values.sum(dim=0)
            squares[name] = squares.get(name, 0) + values.square().sum(dim=0)
            counts[name] = counts.get(name, 0) + values.size(0)
    result = {}
    for name in ("audio", "vision"):
        mean = sums[name] / counts[name]
        variance = squares[name] / counts[name] - mean.square()
        result[name] = (
            mean.view(1, 1, -1),
            variance.clamp_min(1e-12).sqrt().view(1, 1, -1),
        )
    return result


def fill_inputs(name, audio, vision, statistics, generator):
    if name == "zero":
        return torch.zeros_like(audio), torch.zeros_like(vision)
    if name == "permutation":
        index = torch.arange(audio.size(0) - 1, -1, -1, device=audio.device)
        return audio.index_select(0, index), vision.index_select(0, index)
    if name == "high":
        return torch.full_like(audio, 100.0), torch.full_like(vision, 100.0)
    if name != "gaussian":
        raise ValueError(name)
    values = []
    for source, modality in ((audio, "audio"), (vision, "vision")):
        mean, std = statistics[modality]
        noise = torch.randn(source.shape, generator=generator)
        values.append((noise * std + mean).to(source.device, source.dtype))
    return tuple(values)


def branch_modules(backbone, branch):
    suffix = "a" if branch == "audio" else "v"
    names = [
        "proj_{}".format(suffix),
        "encoder_s_{}".format(suffix),
        "decoder_{}".format(suffix),
        "align_c_{}".format(suffix),
        "self_attentions_c_{}".format(suffix),
        "trans_l_with_{}".format(suffix),
        "trans_{}_mem".format(suffix),
        "proj1_{}_low".format(suffix),
        "proj2_{}_low".format(suffix),
        "out_layer_{}_low".format(suffix),
        "proj1_{}_high".format(suffix),
        "proj2_{}_high".format(suffix),
        "out_layer_{}_high".format(suffix),
        "projector_{}".format(suffix),
    ]
    parameters = []
    for name in names:
        parameters.extend(getattr(backbone, name).parameters())
    return list(dict.fromkeys(parameters))


def activation_tensors(output, branch, positional):
    suffix = "a" if branch == "audio" else "v"
    fusion_width = output["lfa_l"].size(1)
    fusion_index = 2 if branch == "audio" else 1
    start = fusion_index * fusion_width
    end = start + fusion_width
    return {
        "shared_encoder_output": output["c_{}".format(suffix)],
        "specific_encoder_output": output["s_{}".format(suffix)],
        "positional_dropout_output": positional[
            "specific_{}".format(suffix)
        ],
        "transformer_ffn_output": output["lfa_ffn_{}".format(suffix)],
        "lfa_cross_attention_output": output[
            "lfa_cross_{}".format(suffix)
        ],
        "projector_output": output["lfa_{}".format(suffix)],
        "specific_prediction_hidden": output[
            "specific_hidden_{}".format(suffix)
        ],
        "final_fusion_input": output["fusion_input"][:, start:end],
    }


class Capture:
    def __init__(self, backbone):
        self.values = defaultdict(list)
        self.handles = []
        for suffix in ("a", "v"):
            layer = getattr(backbone, "encoder_s_{}".format(suffix)).layers[0]
            self.handles.append(
                layer.register_forward_pre_hook(
                    self._pre("specific_{}".format(suffix))
                )
            )
        shared_layer = backbone.encoder_c.layers[0]
        self.handles.append(shared_layer.register_forward_pre_hook(self._pre("shared")))
        self.handles.append(
            backbone.trans_l_with_a.layers[0].self_attn.register_forward_hook(
                self._attention("audio")
            )
        )
        self.handles.append(
            backbone.trans_l_with_v.layers[0].self_attn.register_forward_hook(
                self._attention("vision")
            )
        )

    def _pre(self, name):
        def hook(module, inputs):
            del module
            self.values[name].append(inputs[0].detach())
        return hook

    def _attention(self, name):
        def hook(module, inputs, output):
            del module, inputs
            self.values["attention_{}".format(name)].append(output[1].detach())
        return hook

    def reset(self):
        self.values.clear()

    def positional(self):
        return {
            "specific_a": self.values["specific_a"][0],
            "specific_v": self.values["specific_v"][0],
        }

    def remove(self):
        for handle in self.handles:
            handle.remove()


def main():
    cli = parse_args()
    setup_seed(20)
    args = build_args(cli)
    train = MMDataset(args, mode="train")
    valid = MMDataset(args, mode="valid")
    args.seq_lens = train.get_seq_len()
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cli.num_workers,
    )
    valid_loader = DataLoader(
        valid,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cli.num_workers,
    )
    statistics = feature_statistics(train_loader)
    backbone = DLF(args).to(args.device)
    model = MissingModalityWrapper(
        backbone, args.feature_dims[1], args.feature_dims[2]
    ).to(args.device)
    model.load_state_dict(
        torch.load(cli.uniform_checkpoint, map_location=args.device), strict=True
    )
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(2020)
    capture = Capture(backbone)

    first = next(iter(valid_loader))
    text = first["text"].to(args.device)
    audio = first["audio"].to(args.device)
    vision = first["vision"].to(args.device)
    labels = first["labels"]["M"].to(args.device).view(-1, 1)
    batch_size = labels.size(0)
    layer_rows = []
    attention_rows = []
    with torch.no_grad():
        lav_mask = mode_to_mask(
            "LAV", batch_size, args.device, audio.dtype
        )
        capture.reset()
        lav_output = model(text, audio, vision, lav_mask)
        lav_positional = capture.positional()
        for mode, missing_branches in MODE_MISSING.items():
            mask = mode_to_mask(mode, batch_size, args.device, audio.dtype)
            fill_audio, fill_vision = fill_inputs(
                "zero", audio.cpu(), vision.cpu(), statistics, generator
            )
            input_audio = torch.where(
                mask[:, 1, None, None].cpu().bool(), audio.cpu(), fill_audio
            ).to(args.device)
            input_vision = torch.where(
                mask[:, 2, None, None].cpu().bool(), vision.cpu(), fill_vision
            ).to(args.device)
            capture.reset()
            output = model(text, input_audio, input_vision, mask)
            positional = capture.positional()
            for branch in missing_branches:
                absent_values = activation_tensors(output, branch, positional)
                present_values = activation_tensors(
                    lav_output, branch, lav_positional
                )
                for layer in absent_values:
                    absent_norm = float(absent_values[layer].float().norm().cpu())
                    present_norm = float(
                        present_values[layer].float().norm().cpu()
                    )
                    layer_rows.append(
                        {
                            "Mode": mode,
                            "AbsentBranch": branch,
                            "Layer": layer,
                            "AbsentNorm": absent_norm,
                            "PresentNorm": present_norm,
                            "GhostActivationRatio": absent_norm
                            / (present_norm + 1e-12),
                        }
                    )
                weights = capture.values["attention_{}".format(branch)][0]
                attention_rows.append(
                    {
                        "Mode": mode,
                        "AbsentBranch": branch,
                        "MeanAttentionMassToAbsentKV": float(
                            weights.sum(dim=-1).mean().cpu()
                        ),
                        "MaxAttentionWeightToAbsentKV": float(
                            weights.max().cpu()
                        ),
                    }
                )

    filler_names = ("zero", "permutation", "gaussian", "high")
    predictions = {
        mode: {filler: [] for filler in filler_names}
        for mode in MISSING_MODES
    }
    representations = {
        mode: {filler: [] for filler in filler_names}
        for mode in MISSING_MODES
    }
    all_labels = []
    with torch.no_grad():
        for batch in valid_loader:
            text = batch["text"].to(args.device)
            audio = batch["audio"].to(args.device)
            vision = batch["vision"].to(args.device)
            current_labels = batch["labels"]["M"].view(-1, 1)
            all_labels.append(current_labels)
            for mode in MISSING_MODES:
                mask = mode_to_mask(
                    mode, current_labels.size(0), args.device, audio.dtype
                )
                for filler in filler_names:
                    fill_audio, fill_vision = fill_inputs(
                        filler,
                        audio.cpu(),
                        vision.cpu(),
                        statistics,
                        generator,
                    )
                    input_audio = torch.where(
                        mask[:, 1, None, None].cpu().bool(),
                        audio.cpu(),
                        fill_audio,
                    ).to(args.device)
                    input_vision = torch.where(
                        mask[:, 2, None, None].cpu().bool(),
                        vision.cpu(),
                        fill_vision,
                    ).to(args.device)
                    output = model(text, input_audio, input_vision, mask)
                    predictions[mode][filler].append(
                        output["output_logit"].cpu()
                    )
                    representations[mode][filler].append(
                        output["fusion_input"].cpu()
                    )
    all_labels = torch.cat(all_labels)
    baseline_frame = pd.read_csv(cli.uniform_epoch_metrics)
    baseline_row = baseline_frame.loc[baseline_frame["JValid"].idxmin()]
    lav_mae = float(baseline_row["LAV_MAE"])
    filler_rows = []
    filler_metrics = {}
    for mode in MISSING_MODES:
        reference_prediction = torch.cat(predictions[mode]["zero"])
        reference_representation = torch.cat(representations[mode]["zero"])
        filler_metrics[mode] = {}
        for filler in filler_names:
            prediction = torch.cat(predictions[mode][filler])
            representation = torch.cat(representations[mode][filler])
            metrics = regression_metrics(prediction, all_labels)
            filler_metrics[mode][filler] = metrics
            filler_rows.append(
                {
                    "Mode": mode,
                    "Filler": filler,
                    "PredictionMaxAbsDifference": float(
                        (prediction - reference_prediction).abs().max()
                    ),
                    "PredictionMeanAbsDifference": float(
                        (prediction - reference_prediction).abs().mean()
                    ),
                    "RepresentationMaxAbsDifference": float(
                        (representation - reference_representation).abs().max()
                    ),
                    "RepresentationMeanAbsDifference": float(
                        (representation - reference_representation).abs().mean()
                    ),
                    "MAE": metrics["MAE"],
                    "DeltaMAEVsZero": metrics["MAE"]
                    - filler_metrics[mode]["zero"]["MAE"],
                    "Corr": metrics["Corr"],
                }
            )
    j_by_filler = {}
    for filler in filler_names:
        macro = float(
            np.mean(
                [filler_metrics[mode][filler]["MAE"] for mode in MISSING_MODES]
            )
        )
        j_by_filler[filler] = 0.5 * lav_mae + 0.5 * macro
    for row in filler_rows:
        row["J"] = j_by_filler[row["Filler"]]
        row["DeltaJVsZero"] = (
            j_by_filler[row["Filler"]] - j_by_filler["zero"]
        )

    gradient_rows = []
    unsupported_rows = []
    first_train = next(iter(train_loader))
    text = first_train["text"].to(args.device)
    base_audio = first_train["audio"].to(args.device)
    base_vision = first_train["vision"].to(args.device)
    labels = first_train["labels"]["M"].to(args.device).view(-1, 1)
    all_parameters = list(model.parameters())
    for mode in MISSING_MODES:
        mask = mode_to_mask(mode, labels.size(0), args.device, base_audio.dtype)
        audio = base_audio.detach().requires_grad_(True)
        vision = base_vision.detach().requires_grad_(True)
        output = model(text, audio, vision, mask)
        scalar = output["output_logit"].sum()
        input_gradients = torch.autograd.grad(
            scalar, (audio, vision), retain_graph=True, allow_unused=True
        )
        absent_norms = []
        present_norms = []
        for branch, gradient in zip(("audio", "vision"), input_gradients):
            value = 0.0 if gradient is None else float(gradient.float().norm())
            present = bool(mask[0, BRANCHES[branch]])
            (present_norms if present else absent_norms).append(value)
        absent_parameters = []
        for branch in MODE_MISSING[mode]:
            absent_parameters.extend(branch_modules(backbone, branch))
        absent_parameters = list(dict.fromkeys(absent_parameters))
        absent_parameter_ids = {id(value) for value in absent_parameters}
        present_parameters = [
            value
            for value in all_parameters
            if id(value) not in absent_parameter_ids
        ]
        absent_parameter_norm = gradient_norm(
            scalar, absent_parameters, retain_graph=True
        )
        present_parameter_norm = gradient_norm(
            scalar, present_parameters, retain_graph=True
        )
        gradient_rows.append(
            {
                "Mode": mode,
                "AbsentInputGradientNorm": float(
                    math.sqrt(sum(value * value for value in absent_norms))
                ),
                "PresentInputGradientNorm": float(
                    math.sqrt(sum(value * value for value in present_norms))
                ),
                "AbsentBranchParameterGradientNorm": absent_parameter_norm,
                "PresentBranchParameterGradientNorm": present_parameter_norm,
                "AbsentGradientRatio": absent_parameter_norm
                / (present_parameter_norm + 1e-12),
            }
        )
        weighted = {
            "final_supervised": torch.abs(
                output["output_logit"] - labels
            ).mean(),
            "shared_prediction": torch.abs(output["logits_c"] - labels).mean(),
            "language_specific_prediction": 3.0
            * torch.abs(output["logits_l_hetero"] - labels).mean(),
            "audio_specific_prediction": torch.abs(
                output["logits_a_hetero"] - labels
            ).mean(),
            "vision_specific_prediction": torch.abs(
                output["logits_v_hetero"] - labels
            ).mean(),
        }
        norms = {
            name: gradient_norm(value, all_parameters, retain_graph=True)
            for name, value in weighted.items()
        }
        denominator = sum(norms.values()) + 1e-12
        for name, norm in norms.items():
            modality = (
                "audio"
                if name.startswith("audio")
                else "vision"
                if name.startswith("vision")
                else None
            )
            unsupported = modality in MODE_MISSING[mode]
            unsupported_rows.append(
                {
                    "Mode": mode,
                    "Objective": name,
                    "ActiveInFrozenUniform": True,
                    "Unsupported": unsupported,
                    "WeightedGradientNorm": norm,
                    "WeightedGradientShare": norm / denominator,
                }
            )
        for name in (
            "absent_reconstruction",
            "absent_specific_consistency",
            "absent_orthogonality",
            "absent_triplet",
        ):
            unsupported_rows.append(
                {
                    "Mode": mode,
                    "Objective": name,
                    "ActiveInFrozenUniform": False,
                    "Unsupported": True,
                    "WeightedGradientNorm": 0.0,
                    "WeightedGradientShare": 0.0,
                }
            )
    capture.remove()
    aggregate_unsupported_share = float(
        np.mean(
            [
                sum(
                    row["WeightedGradientShare"]
                    for row in unsupported_rows
                    if row["Mode"] == mode and row["Unsupported"]
                )
                for mode in MISSING_MODES
            ]
        )
    )
    maximum_null_sensitivity = max(
        row["PredictionMaxAbsDifference"] for row in filler_rows
    )
    maximum_ghost_ratio = max(
        row["GhostActivationRatio"] for row in layer_rows
    )
    payload = {
        "status": "STAGE20_GHOST_AUDIT_COMPLETED",
        "checkpoint": str(Path(cli.uniform_checkpoint).resolve()),
        "locked_test_access_count": 0,
        "sample_counts": {"train": len(train), "valid": len(valid)},
        "fillers": list(filler_names),
        "gaussian_statistics": "per-feature mean/std from the complete MOSEI train split",
        "maximum_ghost_activation_ratio": maximum_ghost_ratio,
        "maximum_prediction_filler_sensitivity": maximum_null_sensitivity,
        "j_by_filler": j_by_filler,
        "aggregate_unsupported_gradient_share": aggregate_unsupported_share,
        "layerwise_ghost_activation": layer_rows,
        "attention_leakage": attention_rows,
        "filler_sensitivity": filler_rows,
        "absent_gradients": gradient_rows,
        "unsupported_gradients": unsupported_rows,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output = Path(cli.output_root) / "audit"
    atomic_json(output / "ghost_modality_audit.json", payload)
    atomic_frame(output / "layerwise_ghost_activation.tsv", layer_rows)
    atomic_frame(output / "filler_sensitivity.tsv", filler_rows)
    atomic_frame(output / "unsupported_gradient.tsv", unsupported_rows)
    lines = [
        "# Ghost Modality audit",
        "",
        "- Maximum GhostActivationRatio: `{:.9f}`".format(maximum_ghost_ratio),
        "- Maximum prediction filler sensitivity: `{:.3e}`".format(
            maximum_null_sensitivity
        ),
        "- Aggregate unsupported gradient share: `{:.6f}`".format(
            aggregate_unsupported_share
        ),
        "- Locked Test access count: `0`",
        "",
        "The filler audit uses zero, other-sample permutation, Gaussian values "
        "from train feature mean/std, and a constant magnitude of 100.",
        "",
        "Detailed per-layer, per-filler and per-objective values are in the TSV files.",
    ]
    atomic_text(output / "ghost_modality_audit.md", "\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "maximum_ghost_activation_ratio": maximum_ghost_ratio,
                "maximum_prediction_filler_sensitivity": maximum_null_sensitivity,
                "aggregate_unsupported_gradient_share": aggregate_unsupported_share,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
