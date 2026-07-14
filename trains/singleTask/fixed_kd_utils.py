"""Stage 2 fixed-weight prediction-KD helpers.

Only the final regression prediction is distilled.  This module intentionally
contains no feature, attention, auxiliary-head, or adaptive distillation.
"""

import hashlib
import random
from pathlib import Path

import numpy as np
import torch

from .missing_utils import MISSING_MODES, mode_to_mask


def checkpoint_sha256(checkpoint_path):
    """Return the SHA256 of an immutable initialization checkpoint."""
    digest = hashlib.sha256()
    with open(checkpoint_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_kd_checkpoint_path(root, dataset_name, seed):
    return (
        Path(root)
        / "missing_baseline"
        / "fixed_kd"
        / "DLF_{}_seed{}_best.pth".format(dataset_name, seed)
    )


def capture_rng_state():
    """Capture all RNGs that teacher construction could otherwise perturb."""
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def freeze_teacher(teacher):
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher.eval()
    if teacher.training:
        raise RuntimeError("Teacher must remain in eval mode.")
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("Teacher parameters were not fully frozen.")
    return teacher


def teacher_grad_count(teacher):
    return sum(parameter.grad is not None for parameter in teacher.parameters())


def teacher_parameter_ids(teacher):
    return {id(parameter) for parameter in teacher.parameters()}


def assert_teacher_not_in_optimizer(teacher, optimizer):
    teacher_ids = teacher_parameter_ids(teacher)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if teacher_ids & optimizer_ids:
        raise RuntimeError("Teacher parameters must not appear in student optimizer groups.")


def build_frozen_teacher(model_factory, args, checkpoint_path):
    """Load a teacher without changing the RNG sequence used by the student."""
    rng_state = capture_rng_state()
    try:
        teacher = model_factory(args).to(args.device)
        teacher.load_state_dict(torch.load(checkpoint_path, map_location=args.device), strict=True)
        return freeze_teacher(teacher)
    finally:
        restore_rng_state(rng_state)


def _restore_normal_position_cache():
    """DLF's shared sinusoidal-position cache must not retain inference tensors."""
    from ..subNets.transformers_encoder.position_embedding import make_positions

    # ``make_positions`` keeps one cache per device as ``range_buf_<device>``.
    # The cache created under inference_mode cannot be updated by the following
    # ordinary student forward. Removing only those transient caches lets the
    # student recreate regular tensors without changing the DLF implementation.
    for name in list(vars(make_positions)):
        if name.startswith("range_buf_"):
            delattr(make_positions, name)


def teacher_lav_prediction(teacher, text, audio, vision):
    """Produce a detached normal tensor from a frozen teacher LAV forward."""
    with torch.inference_mode():
        prediction = teacher(text, audio, vision)["output_logit"]
    normal_prediction = prediction.detach().clone()
    _restore_normal_position_cache()
    return normal_prediction


def prediction_kd_loss(kd_criterion, student_missing_output, teacher_full_output):
    """Fixed Stage 2 KD on output_logit only."""
    return kd_criterion(
        student_missing_output["output_logit"],
        teacher_full_output["output_logit"].detach().clone(),
    )


def assert_initial_lav_equivalence(teacher, student, text, audio, vision, rtol=1e-5, atol=1e-6):
    """Verify equal clean initialization before any student update."""
    teacher_prediction = teacher_lav_prediction(teacher, text, audio, vision)
    lav_mask = mode_to_mask(
        "LAV",
        batch_size=text.size(0),
        device=audio.device,
        dtype=audio.dtype,
    )
    with torch.no_grad():
        student_output = student(text, audio, vision, lav_mask)
    torch.testing.assert_close(
        teacher_prediction,
        student_output["output_logit"],
        rtol=rtol,
        atol=atol,
    )


def compute_validation_gaps(teacher, student, dataloader, device):
    """Diagnostic teacher-LAV versus student-missing prediction gaps only."""
    teacher.eval()
    student.eval()
    totals = {mode: 0.0 for mode in MISSING_MODES}
    sample_count = 0
    for batch_data in dataloader:
        text = batch_data["text"].to(device)
        audio = batch_data["audio"].to(device)
        vision = batch_data["vision"].to(device)
        teacher_prediction = teacher_lav_prediction(teacher, text, audio, vision)
        with torch.no_grad():
            for mode in MISSING_MODES:
                mask = mode_to_mask(
                    mode,
                    batch_size=text.size(0),
                    device=device,
                    dtype=audio.dtype,
                )
                student_prediction = student(text, audio, vision, mask)["output_logit"]
                totals[mode] += float(
                    torch.abs(teacher_prediction - student_prediction).sum().item()
                )
        sample_count += int(text.size(0))
    if sample_count == 0:
        raise RuntimeError("Validation loader is empty.")
    return {"Gap_{}".format(mode): totals[mode] / sample_count for mode in MISSING_MODES}


def assert_student_state_dict_has_no_teacher(student_state_dict):
    if any(key.startswith("teacher.") for key in student_state_dict):
        raise RuntimeError("Student checkpoint must not contain teacher parameters.")
