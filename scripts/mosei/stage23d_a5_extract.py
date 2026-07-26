#!/usr/bin/env python3
"""Conditionally authorized four-pass A5 perturbation pilot extraction."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stage23d_extract import (
    DLF,
    MMDataset,
    MissingModalityWrapper,
    ReadOnlyActivationCapture,
    atomic_gzip_csv,
    batch_flat,
    build_args,
    mode_to_mask,
)
from stage23d_self_risk_common import (
    ACTIVE_HEADS,
    EXPERTS,
    HEADS,
    MODE_MASKS,
    MODES,
    OUT,
    ROOT,
    atomic_json,
    enable_dropout_only,
    expert_directory,
    sha256_file,
    utc_now,
)


PILOT_EXPERTS = ("moddrop_seed1111", "cfcompat_seed1111")
PASS_NAMES = (
    "mc_dropout",
    "time_mask_5pct",
    "visible_feature_noise_1pct",
    "visible_local_dropout_5pct",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--expert-id", choices=EXPERTS, required=True)
    parser.add_argument("--gpu-id", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def perturb_continuous(value, profile, generator):
    result = value.clone()
    valid_time = torch.linalg.vector_norm(result.float(), dim=-1) > 0
    if profile == "time_mask_5pct":
        mask = (
            torch.rand(
                valid_time.shape,
                generator=generator,
                device=value.device,
            )
            < 0.05
        ) & valid_time
        result[mask] = 0
    elif profile == "visible_feature_noise_1pct":
        valid_values = result[valid_time]
        scale = (
            valid_values.float().std().clamp_min(1e-6) * 0.01
            if valid_values.numel()
            else torch.tensor(0.0, device=value.device)
        )
        noise = torch.randn(
            result.shape,
            generator=generator,
            device=value.device,
            dtype=result.dtype,
        )
        result = result + noise * scale * valid_time.unsqueeze(-1)
    elif profile == "visible_local_dropout_5pct":
        keep = torch.rand(
            result.shape,
            generator=generator,
            device=value.device,
        ) >= 0.05
        result = result * keep.to(result.dtype)
    return result


def perturbed_inputs(text, audio, vision, mode, profile, generator):
    # BERT token ids/masks are not continuous features and are deliberately
    # left unchanged. A/V perturbations are applied only when currently visible.
    mask = MODE_MASKS[mode]
    local_audio = (
        perturb_continuous(audio, profile, generator) if mask[1] else audio
    )
    local_vision = (
        perturb_continuous(vision, profile, generator) if mask[2] else vision
    )
    return text, local_audio, local_vision


def main():
    cli = parse_args()
    gate_path = OUT / "analysis" / "phase1_gate.json"
    if not gate_path.exists():
        raise RuntimeError("A5 forbidden before the frozen Phase-1 gate")
    gate = json.loads(gate_path.read_text())
    if gate["decision"] not in ("WEAK", "PASS"):
        raise RuntimeError("A5 is not authorized for a Phase-1 FAIL")
    expansion = OUT / "protocol" / "a5_expansion_authorization.json"
    if cli.expert_id not in PILOT_EXPERTS and not expansion.exists():
        raise RuntimeError("Only the two frozen pilot Experts are authorized")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    args = build_args(cli)
    dataset = MMDataset(args, mode="train")
    args.seq_lens = dataset.get_seq_len()
    split = pd.read_csv(
        OUT / "protocol" / "sample_split_manifest.csv",
        dtype={"sample_id": str, "video_id": str},
    )
    holdout = split.loc[
        split["checkpoint_fold"].astype(int) == cli.checkpoint_fold
    ]
    role = holdout.set_index("sample_id")["self_risk_role"].to_dict()
    source = holdout.set_index("sample_id")["video_id"].to_dict()
    indices = holdout["train_index"].astype(int).tolist()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=cli.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cli.num_workers,
    )
    directory = expert_directory(cli.checkpoint_fold, cli.expert_id)
    run_manifest = json.loads((directory / "run_manifest.json").read_text())
    checkpoint = Path(run_manifest["checkpoint"])
    if sha256_file(checkpoint) != run_manifest["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint SHA mismatch")
    model = MissingModalityWrapper(
        DLF(args), args.feature_dims[1], args.feature_dims[2]
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    model.to(args.device).eval()
    expected = pd.read_csv(directory.parent.parent / "oof_predictions.csv")
    expected = expected.loc[expected["expert_id"] == cli.expert_id].set_index(
        ["sample_id", "mode"]
    )["prediction"]
    rows = []
    replay = []
    dropout_modules = enable_dropout_only(model)
    with ReadOnlyActivationCapture(model) as capture, torch.no_grad():
        for batch_number, batch in enumerate(loader):
            text = batch["text"].to(args.device)
            audio = batch["audio"].to(args.device)
            vision = batch["vision"].to(args.device)
            identifiers = [str(value) for value in batch["id"]]
            batch_size = len(identifiers)
            for mode_index, mode in enumerate(MODES):
                model.eval()
                capture.clear()
                mask = mode_to_mask(mode, batch_size, args.device, audio.dtype)
                baseline = model(text, audio, vision, mask)
                baseline_prediction = (
                    baseline["output_logit"].view(-1).detach().cpu().numpy()
                )
                baseline_representation = batch_flat(
                    capture.values["final_fused"], batch_size
                )
                pass_predictions = []
                pass_representations = []
                pass_heads = {name: [] for name in ACTIVE_HEADS[mode]}
                for pass_index, profile in enumerate(PASS_NAMES):
                    seed = (
                        23900
                        + 100 * cli.checkpoint_fold
                        + 10 * EXPERTS.index(cli.expert_id)
                        + pass_index
                    )
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    generator = torch.Generator(device=args.device)
                    generator.manual_seed(seed + batch_number * 1009 + mode_index)
                    local_text, local_audio, local_vision = perturbed_inputs(
                        text, audio, vision, mode, profile, generator
                    )
                    enable_dropout_only(model)
                    capture.clear()
                    output = model(
                        local_text, local_audio, local_vision, mask
                    )
                    pass_predictions.append(
                        output["output_logit"].view(-1).detach().cpu().numpy()
                    )
                    pass_representations.append(
                        batch_flat(capture.values["final_fused"], batch_size)
                    )
                    for name in pass_heads:
                        pass_heads[name].append(
                            output[name].view(-1).detach().cpu().numpy()
                        )
                prediction_matrix = np.stack(pass_predictions, axis=1)
                representation_stack = np.stack(pass_representations, axis=1)
                drift = np.linalg.norm(
                    representation_stack
                    - baseline_representation[:, np.newaxis, :],
                    axis=2,
                )
                for position, sample_id in enumerate(identifiers):
                    old = float(expected.loc[(sample_id, mode)])
                    replay.append(abs(baseline_prediction[position] - old))
                    values = prediction_matrix[position]
                    head_variance = np.mean(
                        [
                            np.var(
                                np.stack(pass_heads[name], axis=1)[position]
                            )
                            for name in pass_heads
                        ]
                    )
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "video_id": source[sample_id],
                            "checkpoint_fold": cli.checkpoint_fold,
                            "expert_id": cli.expert_id,
                            "mode": mode,
                            "self_risk_role": role[sample_id],
                            "a5_prediction_variance": float(np.var(values)),
                            "a5_prediction_MAD": float(
                                np.mean(np.abs(values - np.mean(values)))
                            ),
                            "a5_prediction_max_deviation": float(
                                np.max(
                                    np.abs(
                                        values
                                        - baseline_prediction[position]
                                    )
                                )
                            ),
                            "a5_prediction_sign_flip_rate": float(
                                np.mean(
                                    np.sign(values)
                                    != np.sign(
                                        baseline_prediction[position]
                                    )
                                )
                            ),
                            "a5_hierarchical_head_variance": float(
                                head_variance
                            ),
                            "a5_representation_drift_mean": float(
                                drift[position].mean()
                            ),
                            "a5_representation_drift_max": float(
                                drift[position].max()
                            ),
                        }
                    )
    frame = pd.DataFrame(rows)
    if (
        len(frame) != len(indices) * len(MODES)
        or frame.duplicated(["sample_id", "mode"]).any()
        or not np.isfinite(
            frame.select_dtypes(include=[np.number]).to_numpy()
        ).all()
    ):
        raise RuntimeError("A5 feature binding/finite gate failed")
    if max(replay) > 1e-6:
        raise RuntimeError("A5 baseline replay gate failed")
    output_dir = (
        OUT
        / "a5"
        / "features"
        / f"checkpoint_fold{cli.checkpoint_fold}"
        / cli.expert_id
    )
    feature_path = output_dir / "a5_features_label_free.csv.gz"
    atomic_gzip_csv(frame, feature_path)
    manifest = {
        "stage": "Stage23D-A conditional A5 four-pass extraction",
        "status": "COMPLETED",
        "phase1_decision": gate["decision"],
        "checkpoint_fold": cli.checkpoint_fold,
        "expert_id": cli.expert_id,
        "pilot": cli.expert_id in PILOT_EXPERTS,
        "pass_names": list(PASS_NAMES),
        "stochastic_passes": 4,
        "dropout_modules": dropout_modules,
        "feature_path": str(feature_path.resolve()),
        "feature_sha256": sha256_file(feature_path),
        "rows": len(frame),
        "duplicates": 0,
        "missing": 0,
        "nan_inf_count": 0,
        "max_abs_baseline_replay_diff": float(max(replay)),
        "missing_modality_generated_or_imputed": False,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "expert_retrained": False,
        "completed_at": utc_now(),
    }
    atomic_json(output_dir / "a5_extraction_manifest.json", manifest)
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
