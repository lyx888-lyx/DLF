"""Runtime scalar and true autograd audit for the actual CFCompatKD objective."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import get_config_regression
from run_cfcompat_stability_multiseed import load_locked_cache
from trains.singleTask.HingeLoss import HingeLoss
from trains.singleTask.cf_compat_kd_utils import (
    compatibility_for_modes,
    gate_weights,
    gated_kd_loss,
)
from trains.singleTask.fixed_kd_utils import (
    build_frozen_teacher,
    checkpoint_sha256,
    teacher_lav_prediction,
)
from trains.singleTask.loss_dependency_registry import LOSS_BY_NAME, semantic_active
from trains.singleTask.missing_utils import (
    MissingModalityWrapper,
    build_single_split_loader,
    mode_to_mask,
)
from trains.singleTask.modality_presence import validate_presence_binding
from trains.singleTask.model.DLF import DLF
from utils.functions import setup_seed


AUDIT_SEEDS = (1114, 1111)
MODES = ("LAV", "LA", "LV", "L")
ROOT = Path("result/missing_baseline/mcao_v1/mosi")
AUDIT = ROOT / "stage17a_loss_audit"
SOURCE_MANIFEST = Path(
    "/code/DLF-mosi-dcrc-v1/result/missing_baseline/dcrc_v1/"
    "mosi/baseline_asset_manifest.json"
)
DATA_PATH = Path("/code/DLF/dataset/MOSI/Processed/aligned_50.pkl")
BERT_PATH = "/code/DLF/bert-base-uncased"


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
        raise RuntimeError("No unique source row.")
    return rows[0]


def full_components(output, labels, criterion, cosine, hinge):
    feature_dim = output["s_l"].shape[-1]
    target = torch.full(
        (output["s_l"].reshape(-1, feature_dim).size(0),),
        -1.0,
        dtype=output["s_l"].dtype,
        device=output["s_l"].device,
    )
    shared = (output["c_l_sim"], output["c_v_sim"], output["c_a_sim"])
    features = torch.cat(
        [
            feature[index].view(1, -1)
            for index in range(labels.size(0))
            for feature in shared
        ],
        dim=0,
    )
    ids = torch.cat(
        [labels[index].view(1, -1).repeat(3, 1) for index in range(labels.size(0))],
        dim=0,
    )
    return {
        "full_final_task": criterion(output["output_logit"], labels),
        "full_common_task": criterion(output["logits_c"], labels),
        "full_text_specific_task": criterion(output["logits_l_hetero"], labels),
        "full_audio_specific_task": criterion(output["logits_a_hetero"], labels),
        "full_visual_specific_task": criterion(output["logits_v_hetero"], labels),
        "full_text_reconstruction": F.mse_loss(output["recon_l"], output["origin_l"]),
        "full_audio_reconstruction": F.mse_loss(output["recon_a"], output["origin_a"]),
        "full_visual_reconstruction": F.mse_loss(output["recon_v"], output["origin_v"]),
        "full_text_consistency": F.mse_loss(
            output["s_l"].permute(1, 2, 0), output["s_l_r"]
        ),
        "full_audio_consistency": F.mse_loss(
            output["s_a"].permute(1, 2, 0), output["s_a_r"]
        ),
        "full_visual_consistency": F.mse_loss(
            output["s_v"].permute(1, 2, 0), output["s_v_r"]
        ),
        "full_text_orthogonality": cosine(
            output["s_l"].reshape(-1, feature_dim),
            output["c_l"].reshape(-1, feature_dim),
            target,
        ),
        "full_audio_orthogonality": cosine(
            output["s_a"].reshape(-1, feature_dim),
            output["c_a"].reshape(-1, feature_dim),
            target,
        ),
        "full_visual_orthogonality": cosine(
            output["s_v"].reshape(-1, feature_dim),
            output["c_v"].reshape(-1, feature_dim),
            target,
        ),
        "full_similarity_triplet": hinge(ids, features),
    }


def missing_components(output, labels, criterion, kd_loss):
    return {
        "missing_final_task": criterion(output["output_logit"], labels),
        "missing_common_task": criterion(output["logits_c"], labels),
        "missing_text_specific_task": criterion(output["logits_l_hetero"], labels),
        "missing_audio_specific_task": criterion(output["logits_a_hetero"], labels),
        "missing_visual_specific_task": criterion(output["logits_v_hetero"], labels),
        "missing_cfcompat_kd": kd_loss,
    }


def weighted(components):
    return {
        name: value * LOSS_BY_NAME[name].coefficient
        for name, value in components.items()
    }


def parameter_group(name):
    if name in {"missing_audio_token", "missing_vision_token"} or name.startswith(
        "mask_adapter."
    ):
        return "cfcompat_added_parameters"
    if name == "backbone.out_layer.weight" or name == "backbone.out_layer.bias":
        return "final_output_head"
    short = name.removeprefix("backbone.")
    text_prefixes = (
        "text_model.", "proj_l.", "encoder_s_l.", "decoder_l.", "proj_cosine_l.",
        "proj1_l_low.", "proj2_l_low.", "out_layer_l_low.",
        "proj1_l_high.", "proj2_l_high.", "out_layer_l_high.",
    )
    audio_prefixes = (
        "proj_a.", "encoder_s_a.", "decoder_a.", "proj_cosine_a.",
        "proj1_a_low.", "proj2_a_low.", "out_layer_a_low.",
        "proj1_a_high.", "proj2_a_high.", "out_layer_a_high.",
    )
    visual_prefixes = (
        "proj_v.", "encoder_s_v.", "decoder_v.", "proj_cosine_v.",
        "proj1_v_low.", "proj2_v_low.", "out_layer_v_low.",
        "proj1_v_high.", "proj2_v_high.", "out_layer_v_high.",
    )
    shared_prefixes = (
        "encoder_c.", "align_c_l.", "align_c_a.", "align_c_v.",
        "self_attentions_c_l.", "self_attentions_c_a.", "self_attentions_c_v.",
        "proj1_c.", "proj2_c.", "out_layer_c.",
    )
    fusion_prefixes = (
        "trans_", "projector_l.", "projector_a.", "projector_v.", "projector_c.",
    )
    if short.startswith(text_prefixes):
        return "text_specific_encoder_head"
    if short.startswith(audio_prefixes):
        return "audio_specific_encoder_head"
    if short.startswith(visual_prefixes):
        return "visual_specific_encoder_head"
    if short.startswith(shared_prefixes):
        return "shared_common_encoder"
    if short.startswith(fusion_prefixes) or short in {
        "proj1.weight", "proj1.bias", "proj2.weight", "proj2.bias"
    }:
        return "cross_modal_fusion"
    raise RuntimeError("Unclassified trainable parameter: {}".format(name))


def grad(loss, parameters, retain_graph=True):
    return torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
        create_graph=False,
    )


def selected_indices(groups, group):
    if group == "all_student_parameters":
        return range(len(groups))
    return [index for index, value in enumerate(groups) if value == group]


def gradient_stats(values, indices):
    square, maximum, tensors, elements = 0.0, 0.0, 0, 0
    for index in indices:
        value = values[index]
        if value is None:
            continue
        square += float(value.detach().double().pow(2).sum())
        maximum = max(maximum, float(value.detach().abs().max()))
        nonzero = int(torch.count_nonzero(value.detach()).item())
        tensors += int(nonzero > 0)
        elements += nonzero
    return math.sqrt(square), maximum, tensors, elements


def gradient_cosine(left, right, indices):
    dot, left_square, right_square = 0.0, 0.0, 0.0
    for index in indices:
        first, second = left[index], right[index]
        if first is None or second is None:
            continue
        first = first.detach().double()
        second = second.detach().double()
        dot += float((first * second).sum())
        left_square += float(first.pow(2).sum())
        right_square += float(second.pow(2).sum())
    denominator = math.sqrt(left_square * right_square)
    return dot / denominator if denominator else 0.0


def fixed_dropout_seed(seed, batch_index):
    value = int(seed) * 100003 + int(batch_index) * 97 + 17
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def audit_mode(
    seed,
    batch_index,
    mode,
    model,
    teacher,
    batch,
    device,
    cache_by_index,
    criterion,
    cosine,
    hinge,
    parameters,
    parameter_names,
    groups,
):
    text = batch["text"].to(device)
    audio = batch["audio"].to(device)
    vision = batch["vision"].to(device)
    labels = batch["labels"]["M"].to(device).view(-1, 1)
    indices = batch["index"].view(-1).cpu().numpy().astype(int).tolist()
    mask = mode_to_mask(mode, len(labels), device, audio.dtype)
    binding = validate_presence_binding(mode, mask)
    if not binding["Passed"]:
        raise RuntimeError("Presence/sample binding failure.")

    fixed_dropout_seed(seed, batch_index)
    full_mask = mode_to_mask("LAV", len(labels), device, audio.dtype)
    full_output = model(text, audio, vision, full_mask)
    full_raw = full_components(full_output, labels, criterion, cosine, hinge)
    full_weighted = weighted(full_raw)

    if mode == "LAV":
        current_raw = full_raw
        current_weighted = full_weighted
        total = sum(current_weighted.values())
        final_name = "full_final_task"
    else:
        fixed_dropout_seed(seed, batch_index)
        missing_output = model(text, audio, vision, mask)
        with torch.no_grad():
            teacher_prediction = teacher_lav_prediction(
                teacher, text, audio, vision
            )
        compatibility = compatibility_for_modes(
            cache_by_index,
            indices,
            [mode] * len(indices),
            device,
            labels.dtype,
        )
        gate, _ = gate_weights(
            compatibility, teacher_prediction, labels, "compat"
        )
        kd_loss, _ = gated_kd_loss(
            missing_output["output_logit"], teacher_prediction, gate
        )
        current_raw = missing_components(
            missing_output, labels, criterion, kd_loss
        )
        current_weighted = weighted(current_raw)
        total = sum(full_weighted.values()) + sum(current_weighted.values())
        final_name = "missing_final_task"

    total_gradient = grad(total, parameters)
    final_gradient = grad(current_weighted[final_name], parameters)
    group_names = sorted(set(groups)) + ["all_student_parameters"]
    scalar_rows, gradient_rows, cosine_rows = [], [], []
    available = ",".join(
        modality for modality, value in zip(("L", "A", "V"), mask[0].tolist())
        if value
    )
    for name, raw_value in current_raw.items():
        entry = LOSS_BY_NAME[name]
        expected = semantic_active(entry, mode)
        value = current_weighted[name]
        scalar_rows.append(
            {
                "Seed": seed,
                "BatchIndex": batch_index,
                "Mode": mode,
                "LossName": name,
                "RawScalar": float(raw_value.detach()),
                "EffectiveCoefficient": entry.coefficient,
                "WeightedScalar": float(value.detach()),
                "EnteredTotalLoss": True,
                "RequiredModalities": ",".join(entry.required_modalities) or "always",
                "AvailableModalities": available,
                "ExpectedActive": expected,
                "ActualActive": bool(abs(float(value.detach())) > 1e-7),
                "PresenceMaskValue": 1.0,
                "CurrentCodePresenceMasked": False,
                "PresenceMismatchCount": binding["MismatchCount"],
            }
        )
        current_gradient = grad(value, parameters)
        for group in group_names:
            selected = list(selected_indices(groups, group))
            norm, maximum, tensor_count, element_count = gradient_stats(
                current_gradient, selected
            )
            gradient_rows.append(
                {
                    "Seed": seed,
                    "BatchIndex": batch_index,
                    "Mode": mode,
                    "LossName": name,
                    "ParameterGroup": group,
                    "GradientL2Norm": norm,
                    "GradientMaxAbs": maximum,
                    "NonzeroParameterTensorCount": tensor_count,
                    "NonzeroElementCount": element_count,
                    "RequiredModalities": ",".join(entry.required_modalities) or "always",
                    "ExpectedActive": expected,
                    "ActuallyUnmasked": True,
                }
            )
            cosine_rows.append(
                {
                    "Seed": seed,
                    "BatchIndex": batch_index,
                    "Mode": mode,
                    "LossName": name,
                    "ParameterGroup": group,
                    "CosineWithFinalTask": gradient_cosine(
                        current_gradient, final_gradient, selected
                    ),
                    "CosineWithCompleteTotal": gradient_cosine(
                        current_gradient, total_gradient, selected
                    ),
                }
            )
        del current_gradient

    abnormal = [
        current_weighted[name]
        for name in current_weighted
        if not semantic_active(LOSS_BY_NAME[name], mode)
    ]
    auxiliary_names = [
        name for name in current_weighted
        if LOSS_BY_NAME[name].family in {
            "common_task", "specific_task", "reconstruction",
            "consistency", "orthogonality", "similarity_triplet",
        }
    ]
    if abnormal:
        abnormal_gradient = grad(sum(abnormal), parameters)
        auxiliary_gradient = grad(
            sum(current_weighted[name] for name in auxiliary_names), parameters
        )
        selected = list(range(len(parameters)))
        abnormal_norm = gradient_stats(abnormal_gradient, selected)[0]
        auxiliary_norm = gradient_stats(auxiliary_gradient, selected)[0]
    else:
        abnormal_norm, auxiliary_norm = 0.0, gradient_stats(
            grad(sum(current_weighted[name] for name in auxiliary_names), parameters),
            list(range(len(parameters))),
        )[0]
    ratio_row = {
        "Seed": seed,
        "BatchIndex": batch_index,
        "Mode": mode,
        "InvalidLosses": ",".join(
            name for name in current_weighted
            if not semantic_active(LOSS_BY_NAME[name], mode)
        ),
        "AbnormalAuxiliaryGradientNorm": abnormal_norm,
        "OriginalAuxiliaryGradientNorm": auxiliary_norm,
        "AbnormalGradientRatio": abnormal_norm / max(auxiliary_norm, 1e-12),
    }
    del total_gradient, final_gradient, full_output
    return scalar_rows, gradient_rows, cosine_rows, ratio_row, binding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=AUDIT_SEEDS, required=True)
    parser.add_argument("--batches", type=int, default=8, choices=(8,))
    cli = parser.parse_args()
    resource = json.loads((ROOT / "gpu3_resource_gate_raw.json").read_text())
    if not resource["Passed"]:
        raise RuntimeError("Stage17 GPU3 resource gate did not pass.")
    setup_seed(cli.seed)
    device = torch.device("cuda:0")
    args = build_args(cli.seed, device)
    source = source_row(cli.seed)
    checkpoint = Path(source["CheckpointPath"])
    if checkpoint_sha256(checkpoint) != source["CheckpointSHA256"]:
        raise RuntimeError("Selected online checkpoint SHA mismatch.")
    clean = Path("/code/DLF/pt/DLF_mosi_seed{}_best.pth".format(cli.seed))
    teacher = build_frozen_teacher(DLF, args, clean)
    model = MissingModalityWrapper(
        DLF(args), args.feature_dims[1], args.feature_dims[2]
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    model.train()
    teacher.eval()

    compatibility_cli = SimpleNamespace(
        result_root="/code/DLF/result", dataset="mosi", seed=cli.seed
    )
    _, cache_by_index, cache_path, cache_sha, _, _ = load_locked_cache(
        compatibility_cli, source["EvaluatorSHA256"]
    )
    loader = build_single_split_loader(args, "train", 1)
    criterion, cosine, hinge = nn.L1Loss(), nn.CosineEmbeddingLoss(), HingeLoss()
    named = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    parameter_names = [name for name, _ in named]
    parameters = [value for _, value in named]
    groups = [parameter_group(name) for name in parameter_names]
    group_counts = pd.Series(groups).value_counts().to_dict()

    scalars, gradients, cosines, ratios, bindings = [], [], [], [], []
    processed_batches = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= cli.batches:
            break
        for mode in MODES:
            rows = audit_mode(
                cli.seed, batch_index, mode, model, teacher, batch, device,
                cache_by_index, criterion, cosine, hinge, parameters,
                parameter_names, groups,
            )
            scalar, gradient, cosine_rows, ratio, binding = rows
            scalars.extend(scalar)
            gradients.extend(gradient)
            cosines.extend(cosine_rows)
            ratios.append(ratio)
            bindings.append(binding)
            torch.cuda.empty_cache()
        processed_batches += 1
        print("seed={} batch={}/{} complete".format(
            cli.seed, batch_index + 1, cli.batches
        ), flush=True)
    if processed_batches != cli.batches:
        raise RuntimeError("Fewer than eight train batches.")

    seed_dir = AUDIT / "runtime_seed{}".format(cli.seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scalars).to_csv(seed_dir / "loss_activation.csv", index=False)
    pd.DataFrame(gradients).to_csv(seed_dir / "per_loss_gradients.csv", index=False)
    pd.DataFrame(cosines).to_csv(seed_dir / "gradient_cosines.csv", index=False)
    pd.DataFrame(ratios).to_csv(seed_dir / "abnormal_gradient_ratios.csv", index=False)
    manifest = {
        "Seed": cli.seed,
        "Batches": cli.batches,
        "Modes": list(MODES),
        "StudentState": "validation-selected Stage8 Online CFCompatKD checkpoint",
        "StudentCheckpoint": str(checkpoint),
        "StudentCheckpointSHA256": source["CheckpointSHA256"],
        "TeacherCheckpoint": str(clean),
        "TeacherCheckpointSHA256": checkpoint_sha256(clean),
        "CompatibilityCache": str(cache_path),
        "CompatibilityCacheSHA256": cache_sha,
        "ModelTrainMode": True,
        "DropoutEnabled": True,
        "OptimizerStepPerformed": False,
        "PresenceBindingMismatchCount": sum(
            row["MismatchCount"] for row in bindings
        ),
        "ParameterGroupTensorCounts": group_counts,
        "TestLoaderConstructed": False,
        "TestFeaturesRead": False,
        "TestLabelsRead": False,
        "TestPredictionsRead": False,
        "TestEvaluationPerformed": False,
        "LockedTestAccessCount": 0,
    }
    (seed_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
