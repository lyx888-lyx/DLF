"""Frozen dependency registry for the actual Stage 8 CFCompatKD objective."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LossDependency:
    name: str
    view: str
    family: str
    coefficient: float
    required_modalities: tuple
    always_on: bool
    formula: str
    tensors: str
    gradient_path: str
    currently_masked: bool

    def record(self):
        value = asdict(self)
        value["required_modalities"] = ",".join(self.required_modalities)
        return value


def _entry(
    name,
    view,
    family,
    coefficient,
    required,
    always,
    formula,
    tensors,
    path,
):
    return LossDependency(
        name=name,
        view=view,
        family=family,
        coefficient=float(coefficient),
        required_modalities=tuple(required),
        always_on=bool(always),
        formula=formula,
        tensors=tensors,
        gradient_path=path,
        currently_masked=False,
    )


LOSS_DEPENDENCIES = (
    _entry("full_final_task", "full_LAV", "final_task", 1, (), True,
           "L1(output_logit, y)", "output_logit, labels", "fusion and final head"),
    _entry("full_common_task", "full_LAV", "common_task", 1, (), True,
           "L1(logits_c, y)", "logits_c, labels", "shared/common encoder and common head"),
    _entry("full_text_specific_task", "full_LAV", "specific_task", 3, ("L",), False,
           "L1(logits_l_hetero, y)", "logits_l_hetero, labels", "text-specific and cross-modal text branch"),
    _entry("full_audio_specific_task", "full_LAV", "specific_task", 1, ("A",), False,
           "L1(logits_a_hetero, y)", "logits_a_hetero, labels", "audio-specific and cross-modal audio branch"),
    _entry("full_visual_specific_task", "full_LAV", "specific_task", 1, ("V",), False,
           "L1(logits_v_hetero, y)", "logits_v_hetero, labels", "visual-specific and cross-modal visual branch"),
    _entry("full_text_reconstruction", "full_LAV", "reconstruction", 0.1, ("L",), False,
           "MSE(recon_l, origin_l)", "recon_l, origin_l", "text projection/shared/specific encoder and decoder"),
    _entry("full_audio_reconstruction", "full_LAV", "reconstruction", 0.1, ("A",), False,
           "MSE(recon_a, origin_a)", "recon_a, origin_a", "audio projection/shared/specific encoder and decoder"),
    _entry("full_visual_reconstruction", "full_LAV", "reconstruction", 0.1, ("V",), False,
           "MSE(recon_v, origin_v)", "recon_v, origin_v", "visual projection/shared/specific encoder and decoder"),
    _entry("full_text_consistency", "full_LAV", "consistency", 0.1, ("L",), False,
           "MSE(s_l, s_l_r)", "s_l, s_l_r", "text-specific encoder and decoder"),
    _entry("full_audio_consistency", "full_LAV", "consistency", 0.1, ("A",), False,
           "MSE(s_a, s_a_r)", "s_a, s_a_r", "audio-specific encoder and decoder"),
    _entry("full_visual_consistency", "full_LAV", "consistency", 0.1, ("V",), False,
           "MSE(s_v, s_v_r)", "s_v, s_v_r", "visual-specific encoder and decoder"),
    _entry("full_text_orthogonality", "full_LAV", "orthogonality", 0.01, ("L",), False,
           "CosineEmbedding(s_l, c_l, -1)", "s_l, c_l", "text-specific and shared encoder"),
    _entry("full_audio_orthogonality", "full_LAV", "orthogonality", 0.01, ("A",), False,
           "CosineEmbedding(s_a, c_a, -1)", "s_a, c_a", "audio-specific and shared encoder"),
    _entry("full_visual_orthogonality", "full_LAV", "orthogonality", 0.01, ("V",), False,
           "CosineEmbedding(s_v, c_v, -1)", "s_v, c_v", "visual-specific and shared encoder"),
    _entry("full_similarity_triplet", "full_LAV", "similarity_triplet", 0.01, ("L", "A", "V"), False,
           "HingeLoss(y, concat(c_l_sim,c_a_sim,c_v_sim))",
           "c_l_sim, c_a_sim, c_v_sim, labels", "shared encoder and three alignment heads"),
    _entry("missing_final_task", "sampled_missing", "final_task", 1, (), True,
           "L1(output_logit, y)", "missing output_logit, labels", "student fusion and final head"),
    _entry("missing_common_task", "sampled_missing", "common_task", 1, (), True,
           "L1(logits_c, y)", "missing logits_c, labels", "student shared/common encoder and head"),
    _entry("missing_text_specific_task", "sampled_missing", "specific_task", 3, ("L",), False,
           "L1(logits_l_hetero, y)", "missing logits_l_hetero, labels", "text-specific and cross-modal text branch"),
    _entry("missing_audio_specific_task", "sampled_missing", "specific_task", 1, ("A",), False,
           "L1(logits_a_hetero, y)", "missing logits_a_hetero, labels", "audio-specific and cross-modal audio branch"),
    _entry("missing_visual_specific_task", "sampled_missing", "specific_task", 1, ("V",), False,
           "L1(logits_v_hetero, y)", "missing logits_v_hetero, labels", "visual-specific and cross-modal visual branch"),
    _entry("missing_cfcompat_kd", "sampled_missing", "cfcompat_kd", 1, (), True,
           "sum(gate_i*abs(student_i-teacher_i))/sum(gate_i)",
           "missing output_logit, frozen teacher prediction, compatibility gate",
           "student fusion/final path only; teacher and gate detached"),
)

LOSS_BY_NAME = {entry.name: entry for entry in LOSS_DEPENDENCIES}
if len(LOSS_BY_NAME) != len(LOSS_DEPENDENCIES):
    raise RuntimeError("Duplicate loss registry names.")


def semantic_active(entry, mode):
    if mode not in ("LAV", "LA", "LV", "L"):
        raise ValueError("Unknown mode: {}".format(mode))
    available = set(mode)
    return entry.always_on or set(entry.required_modalities).issubset(available)


def actual_view_active(entry, mode):
    return (entry.view == "full_LAV" and mode == "LAV") or (
        entry.view == "sampled_missing" and mode in ("LA", "LV", "L")
    )
