"""Synthetic smoke tests for CFCompatKD v9 cross-fit residual consensus."""
from __future__ import annotations

import torch
import torch.nn as nn

from trains.singleTask.cfcompat_crossfit_residual_consensus_utils import (
    CONSENSUS_MIN_AGREE,
    N_FOLDS,
    FrozenS0CrossfitConsensus,
    FrozenS0FoldResidual,
    bank_state_cpu,
    deterministic_video_group_folds,
)


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_l = 2
        self.anchor = nn.Parameter(torch.tensor([0.25]))


class DummyS0(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = DummyBackbone()

    def forward(self, text, audio, vision, modality_mask):
        batch = text.size(0)
        z2 = torch.stack(
            [
                text[:, 0] + 0.0 * self.backbone.anchor,
                text[:, 1] + 0.0 * self.backbone.anchor,
            ],
            dim=1,
        )
        scalar = (text[:, :1] * 0.0) + 0.1 + 0.0 * self.backbone.anchor
        return {
            "c_l_sim": z2,
            "c_v_sim": z2 * 0.5,
            "c_a_sim": z2 * -0.25,
            "logits_c": scalar,
            "logits_l_hetero": scalar + 0.01,
            "logits_v_hetero": scalar + 0.02,
            "logits_a_hetero": scalar + 0.03,
            "output_logit": scalar + 0.04,
        }


def make_states(s0):
    template = FrozenS0FoldResidual(s0, hidden_dim=4, max_abs_residual=1.0)
    states = []
    # LA: 4 positive / 1 negative -> apply positive median.
    # LV: 3 positive / 2 negative -> abstain.
    # L: 5 negative -> apply negative median.
    la = [0.20, 0.15, 0.10, 0.05, -0.30]
    lv = [0.20, 0.10, 0.05, -0.10, -0.20]
    l = [-0.05, -0.10, -0.15, -0.20, -0.25]
    for fold in range(N_FOLDS):
        state = bank_state_cpu(template)
        state["residual_heads.LA.2.bias"] = torch.tensor([la[fold]])
        state["residual_heads.LV.2.bias"] = torch.tensor([lv[fold]])
        state["residual_heads.L.2.bias"] = torch.tensor([l[fold]])
        states.append(state)
    return states


def main():
    ids = [
        "vidA$_$0", "vidA$_$1", "vidB$_$0", "vidC$_$0", "vidD$_$0",
        "vidE$_$0", "vidF$_$0", "vidG$_$0", "vidH$_$0", "vidI$_$0",
    ]
    folds = deterministic_video_group_folds(ids, N_FOLDS)
    assert folds.groupby("video_id").fold.nunique().max() == 1
    assert set(folds.fold.astype(int)) == set(range(N_FOLDS))

    s0 = DummyS0()
    fold_model = FrozenS0FoldResidual(s0, hidden_dim=4, max_abs_residual=1.0)
    text = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    audio = torch.zeros(2, 1)
    vision = torch.zeros(2, 1)
    la_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    out = fold_model(text, audio, vision, la_mask)
    out["output_logit"].sum().backward()
    assert s0.backbone.anchor.grad is None
    assert any(p.grad is not None for p in fold_model.bank.parameters())

    ensemble = FrozenS0CrossfitConsensus(
        s0, make_states(s0), hidden_dim=4, max_abs_residual=1.0,
        min_agree=CONSENSUS_MIN_AGREE,
    )
    la_out = ensemble(text, audio, vision, la_mask)
    assert bool(la_out["consensus_applied"].all())
    assert bool((la_out["residual_delta"] > 0).all())

    lv_mask = torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    lv_out = ensemble(text, audio, vision, lv_mask)
    assert not bool(lv_out["consensus_applied"].any())
    assert torch.allclose(lv_out["residual_delta"], torch.zeros_like(lv_out["residual_delta"]))

    l_mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    l_out = ensemble(text, audio, vision, l_mask)
    assert bool(l_out["consensus_applied"].all())
    assert bool((l_out["residual_delta"] < 0).all())

    lav_mask = torch.ones(2, 3)
    lav_out = ensemble(text, audio, vision, lav_mask)
    assert torch.allclose(lav_out["residual_delta"], torch.zeros_like(lav_out["residual_delta"]))
    assert torch.allclose(lav_out["output_logit"], lav_out["s0_output_logit"])

    print("v9 cross-fit residual consensus smoke test passed")


if __name__ == "__main__":
    main()
