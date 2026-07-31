"""CPU-only structural smoke test for V9.2 grouped OOF and tail objectives."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from trains.singleTask.function_space_features_v92 import (
    FEATURE_KEYS,
    function_space_feature,
)
from trains.singleTask.oof_group_splits_v92 import (
    build_nested_group_folds,
    conversation_group_id,
)
from trains.singleTask.model.CachedTailResidualHeadV92 import (
    CachedTailResidualHeadV92,
)
from trains.singleTask.oof_tail_residual_v92 import (
    OOFTailLossWeights,
    TAIL_ROLE_NAMES,
    oof_tail_residual_loss,
)


def main():
    sample_ids = []
    labels = []
    for group in range(18):
        for segment in range(4):
            sample_ids.append(f"video{group:02d}$_${segment}")
            if segment == 0:
                labels.append(-2.2 - 0.02 * (group % 3))
            elif segment == 1:
                labels.append(2.1 + 0.03 * (group % 3))
            elif segment == 2:
                labels.append(-0.8)
            else:
                labels.append(0.7)
    specs, manifest = build_nested_group_folds(
        sample_ids, labels, outer_folds=3, inner_valid_fraction=0.20, seed=1111
    )
    assert len(specs) == 3
    assert len(manifest) == 3 * len(sample_ids)
    assert conversation_group_id("abc$_$12") == "abc"
    holdout = [index for spec in specs for index in spec.outer_holdout_indices]
    assert sorted(holdout) == list(range(len(sample_ids)))
    for spec in specs:
        assert not set(spec.inner_train_groups) & set(spec.inner_valid_groups)
        assert not set(spec.inner_train_groups) & set(spec.outer_holdout_groups)
        assert not set(spec.inner_valid_groups) & set(spec.outer_holdout_groups)

    fake_output = {
        key: torch.randn(32, 1) for key in FEATURE_KEYS
    }
    aligned_feature = function_space_feature(fake_output)
    assert aligned_feature.shape == (32, len(FEATURE_KEYS))
    assert torch.isfinite(aligned_feature).all()

    torch.manual_seed(7)
    feature = torch.randn(32, len(FEATURE_KEYS))
    anchor = torch.linspace(-2.5, 2.5, 32).view(-1, 1)
    labels_tensor = anchor + 0.3 * torch.tanh(torch.randn_like(anchor))
    model = CachedTailResidualHeadV92(
        feature_dim=len(FEATURE_KEYS), hidden_dim=16
    )
    for role in TAIL_ROLE_NAMES:
        outputs = model(feature, anchor)
        losses = oof_tail_residual_loss(
            outputs,
            labels_tensor,
            role=role,
            membership_temperature=0.25,
            gain_margin=0.08,
            gain_fraction=0.20,
            weights=OOFTailLossWeights(),
        )
        assert torch.isfinite(losses["total"])
        model.zero_grad(set_to_none=True)
        losses["total"].backward()
        assert model.residual_head.weight.grad is not None
        assert model.applicability_head.weight.grad is not None
        assert model.mechanism_head.weight.grad is not None

    print("V9.2 GROUPED OOF TAIL RESIDUAL SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
