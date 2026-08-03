"""Synthetic smoke test for V9.27 paired perturbation utilities."""

from __future__ import annotations

import numpy as np
import torch

from trains.singleTask.paired_perturbation_relative_stability_v927 import (
    PairedPerturbationConfigV927,
    apply_paired_perturbation,
    build_relative_stability_features,
    perturbation_specs,
)


def main():
    config = PairedPerturbationConfigV927(
        batch_size=2,
        num_workers=0,
        noise_scale=0.01,
        temporal_mask_fraction=0.10,
        base_seed=1111,
    )
    specs = perturbation_specs(config)
    assert len(specs) == 13
    assert specs[0].name == "identity"
    assert len({spec.name for spec in specs}) == len(specs)

    audio = torch.arange(2 * 10 * 3, dtype=torch.float32).reshape(2, 10, 3)
    vision = torch.arange(2 * 10 * 2, dtype=torch.float32).reshape(2, 10, 2)
    audio[:, 8:] = 0
    vision[:, 9:] = 0
    sample_indices = [7, 19]

    identity_audio, identity_vision = apply_paired_perturbation(
        audio,
        vision,
        sample_indices,
        specs[0],
        config,
    )
    assert torch.equal(identity_audio, audio)
    assert torch.equal(identity_vision, vision)

    gain = next(spec for spec in specs if spec.name == "audio_gain_095")
    gain_audio, gain_vision = apply_paired_perturbation(
        audio, vision, sample_indices, gain, config
    )
    assert torch.allclose(gain_audio, audio * 0.95)
    assert torch.equal(gain_vision, vision)

    noise = next(spec for spec in specs if spec.name == "audio_noise_a")
    noisy_a, unchanged_v = apply_paired_perturbation(
        audio, vision, sample_indices, noise, config
    )
    noisy_b, _ = apply_paired_perturbation(
        audio, vision, sample_indices, noise, config
    )
    assert torch.equal(noisy_a, noisy_b)
    assert torch.equal(unchanged_v, vision)
    assert torch.equal(noisy_a[:, 8:], audio[:, 8:])
    assert not torch.equal(noisy_a[:, :8], audio[:, :8])

    mask = next(spec for spec in specs if spec.name == "vision_mask_025")
    unchanged_a, masked_v = apply_paired_perturbation(
        audio, vision, sample_indices, mask, config
    )
    assert torch.equal(unchanged_a, audio)
    assert torch.equal(masked_v[:, 9:], vision[:, 9:])
    assert int((masked_v[:, :9] == 0).all(dim=2).sum()) >= 2

    rng = np.random.default_rng(1111)
    variants, samples, actions = 13, 24, 5
    action_values = rng.normal(size=(variants, samples, actions))
    action_values[0] = rng.normal(size=(samples, actions))
    confidence = rng.uniform(size=(variants, samples, 4))
    baseline = np.einsum(
        "vna,a->vn",
        action_values,
        np.asarray([0.30, 0.15, 0.20, 0.20, 0.15]),
    )
    payload = {
        "actions": torch.tensor(action_values, dtype=torch.float32),
        "expert_confidences": torch.tensor(
            confidence, dtype=torch.float32
        ),
    }
    result = build_relative_stability_features(
        payload, baseline, "positive", config
    )
    assert result["matrix"].shape == (samples, 8)
    assert len(result["feature_names"]) == 8
    assert np.isfinite(result["matrix"]).all()
    assert np.all(result["expert_std"] >= 0)
    assert np.all(result["baseline_std"] >= 0)
    assert np.all(
        (result["relative_direction_flip_rate"] >= 0)
        & (result["relative_direction_flip_rate"] <= 1)
    )
    assert np.all(
        (result["applicability_flip_rate"] >= 0)
        & (result["applicability_flip_rate"] <= 1)
    )

    repeated_actions = np.repeat(action_values[0:1], variants, axis=0)
    repeated_confidence = np.repeat(confidence[0:1], variants, axis=0)
    repeated_baseline = np.repeat(baseline[0:1], variants, axis=0)
    stable = build_relative_stability_features(
        {
            "actions": torch.tensor(
                repeated_actions, dtype=torch.float32
            ),
            "expert_confidences": torch.tensor(
                repeated_confidence, dtype=torch.float32
            ),
        },
        repeated_baseline,
        "boundary",
        config,
    )
    assert np.max(stable["expert_std"]) < 1e-7
    assert np.max(stable["baseline_std"]) < 1e-7
    assert np.max(stable["relative_gap_std"]) < 1e-7
    assert np.max(stable["relative_direction_flip_rate"]) == 0
    assert np.max(stable["applicability_flip_rate"]) == 0
    assert np.max(stable["expert_max_change"]) == 0
    assert np.max(stable["baseline_max_change"]) == 0

    print("V9.27 PAIRED PERTURBATION SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
