"""Synthetic smoke test for CFCompatKD v8 residual isolation."""
from __future__ import annotations

import torch
import torch.nn as nn

from trains.singleTask.cfcompat_sample_residual_utils import (
    MAX_ABS_RESIDUAL,
    FrozenS0SampleResidual,
    assert_s0_no_gradients,
    module_state_sha256,
    residual_parameter_summary,
)


class ToyS0(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.d_l = 4
        self.backbone.linear = nn.Linear(2, 1)
        self.aux = nn.Linear(2, 4)

    def forward(self, text, audio, vision, modality_mask):
        x = text[:, :2]
        base = self.backbone.linear(x)
        aux = self.aux(x)
        return {
            "c_l_sim": aux[:, 0:4],
            "c_v_sim": aux[:, 0:4] * 0.5,
            "c_a_sim": aux[:, 0:4] * -0.25,
            "logits_c": base + 0.1,
            "logits_l_hetero": base + 0.2,
            "logits_v_hetero": base - 0.1,
            "logits_a_hetero": base - 0.2,
            "output_logit": base,
            "origin_l": x.unsqueeze(0),
            "origin_v": x.unsqueeze(0),
            "origin_a": x.unsqueeze(0),
            "s_l": x.unsqueeze(0),
            "s_v": x.unsqueeze(0),
            "s_a": x.unsqueeze(0),
            "c_l": x.unsqueeze(0),
            "c_v": x.unsqueeze(0),
            "c_a": x.unsqueeze(0),
            "s_l_r": x.unsqueeze(0),
            "s_v_r": x.unsqueeze(0),
            "s_a_r": x.unsqueeze(0),
            "recon_l": x.unsqueeze(0),
            "recon_v": x.unsqueeze(0),
            "recon_a": x.unsqueeze(0),
        }


def main():
    torch.manual_seed(7)
    s0 = ToyS0()
    model = FrozenS0SampleResidual(s0)
    summary = residual_parameter_summary(model)
    before = module_state_sha256(model.s0)

    text = torch.randn(4, 3)
    audio = torch.zeros(4, 1)
    vision = torch.zeros(4, 1)
    masks = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )
    with torch.no_grad():
        baseline = model.s0(text, audio, vision, masks)["output_logit"].clone()
    initial = model(text, audio, vision, masks)
    if not torch.equal(initial["output_logit"], baseline):
        raise RuntimeError("Zero-initialized residual does not reproduce S0 exactly.")
    if not torch.equal(initial["residual_delta"], torch.zeros_like(baseline)):
        raise RuntimeError("Initial residual is not exactly zero.")

    with torch.no_grad():
        model.residual_heads["LA"][-1].bias.fill_(0.4)
    changed = model(text, audio, vision, masks)
    if abs(float((changed["output_logit"][0] - baseline[0]).detach())) <= 0.1:
        raise RuntimeError("LA residual did not change the LA event.")
    if not torch.equal(changed["output_logit"][1:].detach(), baseline[1:]):
        raise RuntimeError("Mode-specific residual leaked into LV/L/LAV.")
    if float(changed["residual_delta"].detach().abs().max()) > MAX_ABS_RESIDUAL + 1e-7:
        raise RuntimeError("Residual bound was violated.")

    target = changed["output_logit"].sum()
    target.backward()
    assert_s0_no_gradients(model)
    if not any(
        p.grad is not None and float(p.grad.detach().abs().sum()) > 0.0
        for p in model.residual_heads["LA"].parameters()
    ):
        raise RuntimeError("Residual head did not receive gradients.")
    if module_state_sha256(model.s0) != before:
        raise RuntimeError("Frozen S0 state changed during smoke test.")

    print("v8 sample-conditioned residual smoke test passed")
    print("feature_dim:", summary["feature_dim"])
    print("trainable_parameter_count:", summary["trainable_parameter_count"])
    print("max_abs_residual:", summary["max_abs_residual"])


if __name__ == "__main__":
    main()
