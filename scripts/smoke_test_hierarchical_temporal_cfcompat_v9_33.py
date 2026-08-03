"""CPU smoke tests for V9.33 hierarchical temporal components."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from trains.singleTask.hierarchical_temporal_cfcompat_v933 import (
    PRIMARY_VARIANT,
    HierarchicalTemporalConfigV933,
    HierarchicalTemporalDatasetV933,
    HierarchicalTemporalModelV933,
    build_partition_context_bindings,
    make_temporal_loader,
    predict_hierarchical_temporal_model,
    reverse_valid_steps,
    success_gate,
    train_hierarchical_temporal_model,
    valid_step_mask,
)


def tail_state(dim: int):
    proj1 = nn.Linear(dim, dim)
    proj2 = nn.Linear(dim, dim)
    out = nn.Linear(dim, 1)
    return {
        "proj1": proj1.state_dict(),
        "proj2": proj2.state_dict(),
        "out_layer": out.state_dict(),
        "output_dropout": 0.0,
        "feature_dim": dim,
    }


def main():
    torch.manual_seed(7)
    sample_ids = [
        "videoA$_$0",
        "videoA$_$1",
        "videoA$_$2",
        "videoA$_$3",
        "videoB$_$0",
        "videoB$_$1",
        "videoB$_$2",
        "videoB$_$3",
    ]
    bindings = build_partition_context_bindings(sample_ids, range(8), 3)
    assert bindings[3]["ordered_indices"] == (0, 1, 2)
    assert bindings[0]["ordered_indices"] == (-1, -1, -1)
    assert all(
        sample_ids[index].startswith("videoB")
        for index in bindings[3]["wrong_indices"]
    )

    audio = np.zeros((8, 12, 3), dtype=np.float32)
    vision = np.zeros((8, 10, 4), dtype=np.float32)
    for index in range(8):
        audio[index, : 4 + index % 3] = np.linspace(
            0.1, 1.0, (4 + index % 3) * 3
        ).reshape(-1, 3)
        vision[index, : 3 + index % 2] = np.linspace(
            -0.5, 0.5, (3 + index % 2) * 4
        ).reshape(-1, 4)

    class Unaligned:
        def __init__(self):
            self.audio = audio
            self.vision = vision
            self.ids = np.asarray(sample_ids, dtype=object)

        def __len__(self):
            return len(self.ids)

    unaligned = Unaligned()
    dim = 12
    rows = {
        index: {
            "sample_index": index,
            "sample_id": sample_ids[index],
            "label": float(np.sin(index)),
            "feature": torch.randn(dim),
            "anchor_prediction": float(np.cos(index) * 0.1),
        }
        for index in range(8)
    }
    dataset = HierarchicalTemporalDatasetV933(
        unaligned, range(8), rows, bindings, 3
    )
    batch = next(iter(make_temporal_loader(dataset, 4, 0, False, 7)))
    mask = valid_step_mask(batch["audio"])
    reversed_audio = reverse_valid_steps(batch["audio"], mask)
    positions = torch.nonzero(mask[0]).view(-1)
    assert torch.allclose(
        reversed_audio[0, positions],
        batch["audio"][0, positions.flip(0)],
    )

    config = HierarchicalTemporalConfigV933(
        temporal_hidden_dim=16,
        context_hidden_dim=16,
        branch_dropout=0.0,
        batch_size=4,
        num_workers=0,
        max_epochs=2,
        early_stop=2,
        use_amp=False,
    )
    state = tail_state(dim)
    model = HierarchicalTemporalModelV933(
        state, 3, 4, PRIMARY_VARIANT, config
    )
    model.eval()
    with torch.no_grad():
        output = model(
            batch["current_feature"],
            batch["audio"],
            batch["vision"],
            batch["ordered_context"],
            batch["ordered_context_mask"],
        )
    assert torch.allclose(
        output["prediction"], output["current_prediction"], atol=1e-7
    )

    train_indices = [0, 1, 4, 5]
    valid_indices = [2, 6]
    test_indices = [3, 7]
    train_dataset = HierarchicalTemporalDatasetV933(
        unaligned,
        train_indices,
        rows,
        build_partition_context_bindings(sample_ids, train_indices, 3),
        3,
    )
    valid_dataset = HierarchicalTemporalDatasetV933(
        unaligned,
        valid_indices,
        rows,
        build_partition_context_bindings(sample_ids, valid_indices, 3),
        3,
    )
    test_dataset = HierarchicalTemporalDatasetV933(
        unaligned,
        test_indices,
        rows,
        build_partition_context_bindings(sample_ids, test_indices, 3),
        3,
    )
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "best.pth"
        fitted = train_hierarchical_temporal_model(
            state,
            3,
            4,
            PRIMARY_VARIANT,
            config,
            make_temporal_loader(train_dataset, 4, 0, True, 9),
            make_temporal_loader(valid_dataset, 2, 0, False, 9),
            torch.device("cpu"),
            checkpoint,
            {"synthetic": True},
            9,
            resume=False,
        )
        assert checkpoint.is_file()
        predicted = predict_hierarchical_temporal_model(
            fitted["model"],
            make_temporal_loader(test_dataset, 2, 0, False, 9),
            torch.device("cpu"),
            diagnostics=True,
        )
        assert predicted["prediction"].shape == (2,)
        assert np.isfinite(predicted["prediction"]).all()
        assert np.isfinite(predicted["wrong_context_prediction"]).all()
        assert np.isfinite(predicted["reversed_time_prediction"]).all()

    gate = success_gate(
        [0.01, 0.008, 0.007, 0.006, -0.001],
        0.006,
        0.005,
        config,
    )
    assert gate["passed"] is True
    rejected = success_gate(
        [0.01, -0.01, 0.0, 0.0, 0.0],
        0.001,
        0.005,
        config,
    )
    assert rejected["passed"] is False
    print("V9.33 HIERARCHICAL TEMPORAL SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
