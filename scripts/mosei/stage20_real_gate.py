"""Run Stage 20 implementation gates on a real MOSEI checkpoint and batch."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_config_regression
from data_loader import MMDataset
from scripts.mosei.run_stage20_safe_dlf import PredictionCache, locked_dataset
from trains.singleTask.missing_utils import (
    MissingModalityWrapper,
    compute_full_dlf_loss,
    mode_to_mask,
)
from trains.singleTask.model.DLF import DLF
from trains.singleTask.safe_dlf_utils import (
    SafeMissingModalityWrapper,
    support_aligned_task_loss,
    supported_triplet_inputs,
)
from utils.functions import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--uniform-checkpoint", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--resume-report", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--output-file", default="tests/test_results.json"
    )
    parser.add_argument("--gpu-id", type=int, default=2)
    return parser.parse_args()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
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


def main():
    cli = parse_args()
    setup_seed(20)
    args = build_args(cli)
    valid = locked_dataset(args, "valid")
    args.seq_lens = valid.get_seq_len()
    batch = next(iter(DataLoader(valid, batch_size=4, shuffle=False)))
    text = batch["text"].to(args.device)
    audio = batch["audio"].to(args.device)
    vision = batch["vision"].to(args.device)
    labels = batch["labels"]["M"].to(args.device).view(-1, 1)
    state = torch.load(cli.uniform_checkpoint, map_location=args.device)

    def model(wrapper):
        value = wrapper(
            DLF(args).to(args.device),
            args.feature_dims[1],
            args.feature_dims[2],
        ).to(args.device)
        value.load_state_dict(state, strict=True)
        value.eval()
        return value

    original = model(MissingModalityWrapper)
    safe = model(SafeMissingModalityWrapper)
    criterion = torch.nn.L1Loss()
    checks = {}
    evidence = {}

    lav = mode_to_mask("LAV", 4, args.device, audio.dtype)
    with torch.no_grad():
        original_lav = original(text, audio, vision, lav)
        safe_lav = safe(text, audio, vision, lav)
        original_loss, original_components = compute_full_dlf_loss(
            original_lav, labels, criterion
        )
        safe_loss, safe_components = compute_full_dlf_loss(
            safe_lav, labels, criterion
        )
    evidence["lav_output_max_abs_diff"] = float(
        (original_lav["output_logit"] - safe_lav["output_logit"])
        .abs()
        .max()
    )
    evidence["lav_total_loss_abs_diff"] = float(
        (original_loss - safe_loss).abs()
    )
    evidence["lav_component_loss_max_abs_diff"] = max(
        float((original_components[key] - safe_components[key]).abs())
        for key in original_components
    )
    checks["lav_output_parity"] = evidence["lav_output_max_abs_diff"] <= 1e-6
    checks["lav_total_loss_parity"] = (
        evidence["lav_total_loss_abs_diff"] <= 1e-6
    )
    checks["lav_component_loss_parity"] = (
        evidence["lav_component_loss_max_abs_diff"] <= 1e-6
    )

    fillers = {
        "zero": (torch.zeros_like(audio), torch.zeros_like(vision)),
        "permutation": (audio.flip(0), vision.flip(0)),
        "gaussian": (torch.randn_like(audio), torch.randn_like(vision)),
        "high": (
            torch.full_like(audio, 100.0),
            torch.full_like(vision, 100.0),
        ),
    }
    maximum_prediction_difference = 0.0
    maximum_fusion_difference = 0.0
    maximum_absent_input_gradient = 0.0
    minimum_present_input_gradient = float("inf")
    maximum_absent_representation = 0.0
    unsupported_gradient_terms = []
    for mode in ("LA", "LV", "L"):
        mask = mode_to_mask(mode, 4, args.device, audio.dtype)
        outputs = []
        inputs = []
        for fill_audio, fill_vision in fillers.values():
            current_audio = torch.where(
                mask[:, 1, None, None].bool(), audio, fill_audio
            ).detach().requires_grad_(True)
            current_vision = torch.where(
                mask[:, 2, None, None].bool(), vision, fill_vision
            ).detach().requires_grad_(True)
            outputs.append(safe(text, current_audio, current_vision, mask))
            inputs.append((current_audio, current_vision))
        reference = outputs[0]
        for output in outputs[1:]:
            maximum_prediction_difference = max(
                maximum_prediction_difference,
                float(
                    (output["output_logit"] - reference["output_logit"])
                    .abs()
                    .max()
                ),
            )
            maximum_fusion_difference = max(
                maximum_fusion_difference,
                float(
                    (output["fusion_input"] - reference["fusion_input"])
                    .abs()
                    .max()
                ),
            )
        gradient_audio, gradient_vision = torch.autograd.grad(
            reference["output_logit"].sum(),
            inputs[0],
            retain_graph=True,
            allow_unused=True,
        )
        for present, gradient in zip(mask[0, 1:].tolist(), (gradient_audio, gradient_vision)):
            value = 0.0 if gradient is None else float(gradient.abs().max())
            if present:
                minimum_present_input_gradient = min(
                    minimum_present_input_gradient, value
                )
            else:
                maximum_absent_input_gradient = max(
                    maximum_absent_input_gradient, value
                )
        if not bool(mask[0, 1]):
            for key in ("origin_a", "s_a", "c_a", "lfa_a", "specific_hidden_a"):
                maximum_absent_representation = max(
                    maximum_absent_representation,
                    float(reference[key].abs().max()),
                )
        if not bool(mask[0, 2]):
            for key in ("origin_v", "s_v", "c_v", "lfa_v", "specific_hidden_v"):
                maximum_absent_representation = max(
                    maximum_absent_representation,
                    float(reference[key].abs().max()),
                )
        _, details = support_aligned_task_loss(reference, labels, mask)
        unsupported_gradient_terms.extend(
            float(details[key].abs())
            for key in (
                "absent_reconstruction",
                "absent_specific_consistency",
                "absent_orthogonality",
                "absent_triplet",
            )
        )
    evidence["filler_output_max_abs_diff"] = maximum_prediction_difference
    evidence["filler_fusion_max_abs_diff"] = maximum_fusion_difference
    evidence["absent_input_gradient_max"] = maximum_absent_input_gradient
    evidence["present_input_gradient_min"] = minimum_present_input_gradient
    evidence["absent_representation_max"] = maximum_absent_representation
    evidence["unsupported_loss_max"] = max(unsupported_gradient_terms)
    checks["filler_prediction_invariance"] = maximum_prediction_difference <= 1e-6
    checks["filler_fusion_invariance"] = maximum_fusion_difference <= 1e-6
    checks["zero_absent_input_gradient"] = maximum_absent_input_gradient <= 1e-8
    checks["present_input_gradient"] = minimum_present_input_gradient > 1e-8
    checks["zero_absent_representation"] = maximum_absent_representation <= 1e-8
    checks["unsupported_loss_zero"] = max(unsupported_gradient_terms) == 0.0

    l_mask = mode_to_mask("L", 4, args.device, audio.dtype)
    lav_ids, lav_features, lav_support = supported_triplet_inputs(
        labels, safe_lav, lav
    )
    l_output = safe(text, audio, vision, l_mask)
    l_ids, l_features, l_support = supported_triplet_inputs(
        labels, l_output, l_mask
    )
    evidence["lav_triplet_entries"] = len(lav_support)
    evidence["l_triplet_entries"] = len(l_support)
    checks["triplet_lav_unchanged"] = (
        len(lav_support) == 12
        and lav_support == [
            (sample, modality)
            for sample in range(4)
            for modality in range(3)
        ]
    )
    checks["triplet_l_only_has_language"] = (
        len(l_support) == 4
        and all(modality == 0 for _, modality in l_support)
    )

    mixed = torch.stack(
        [
            mode_to_mask(mode, device=args.device, dtype=audio.dtype)
            for mode in ("LA", "LV", "L", "LAV")
        ]
    )
    mixed_output = safe(text, audio, vision, mixed)
    mixed_correct = True
    for sample in range(4):
        if not bool(mixed[sample, 1]):
            mixed_correct &= float(mixed_output["lfa_a"][sample].abs().max()) <= 1e-8
        if not bool(mixed[sample, 2]):
            mixed_correct &= float(mixed_output["lfa_v"][sample].abs().max()) <= 1e-8
    checks["mixed_batch_projection"] = bool(mixed_correct)

    cache = PredictionCache(cli.cache_root, 1111)
    cache.validate_dataset("valid", valid)
    class Reversed:
        ids = list(reversed(valid.ids))
    cache_hard_failed = False
    try:
        cache.validate_dataset("valid", Reversed())
    except RuntimeError:
        cache_hard_failed = True
    checks["cache_binding_valid"] = True
    checks["cache_order_mismatch_hard_fail"] = cache_hard_failed

    resume = json.loads(Path(cli.resume_report).read_text())
    evidence["resume_max_numeric_difference"] = resume["max_numeric_difference"]
    checks["resume_integrity"] = (
        resume["status"] == "STAGE19_RESUME_INTEGRITY_PASSED"
        and resume["max_numeric_difference"] == 0.0
    )
    test_locked = False
    try:
        locked_dataset(args, "test")
    except RuntimeError:
        test_locked = True
    checks["test_lock"] = test_locked
    runner_source = (
        ROOT / "scripts/mosei/run_stage20_safe_dlf.py"
    ).read_text()
    checks["runner_has_no_allow_test_flag"] = "--allow-test" not in runner_source
    checks["runner_has_no_test_evaluation_import"] = (
        "build_single_split_loader" not in runner_source
        and 'locked_dataset(args, "test")' not in runner_source
    )

    failed = [name for name, accepted in checks.items() if not accepted]
    payload = {
        "status": (
            "STAGE20_IMPLEMENTATION_AUDIT_PASSED"
            if not failed
            else "STAGE20_IMPLEMENTATION_AUDIT_FAILED"
        ),
        "tests_run": len(checks),
        "tests_passed": len(checks) - len(failed),
        "tests_failed": len(failed),
        "failed_tests": failed,
        "checks": checks,
        "evidence": evidence,
        "locked_test_access_count": 0,
    }
    atomic_json(Path(cli.output_root) / cli.output_file, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
