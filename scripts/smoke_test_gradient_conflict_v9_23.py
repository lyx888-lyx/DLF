"""Synthetic smoke test for V9.23 gradient-space conflict geometry."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from trains.singleTask.gradient_conflict_audit_v923 import (
    GradientAuditConfigV923,
    build_candidate_directions,
    collect_mean_mae_gradients,
    direction_effect_rows,
    fold_geometry_summary,
    global_region_decomposition_error,
    pairwise_geometry_rows,
    select_parameter_records,
    selected_parameter_fingerprint,
)


class ToyDataset(Dataset):
    def __init__(self):
        self.x = torch.linspace(0.4, 1.6, 25).view(-1, 1)
        self.y = torch.tensor(
            [-2.6, -2.3, -2.0, -1.8, -1.6,
             -1.4, -1.2, -1.0, -0.8, -0.6,
             -0.4, -0.2, 0.0, 0.25, 0.45,
             0.6, 0.8, 1.0, 1.2, 1.4,
             1.6, 1.8, 2.0, 2.3, 2.6],
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return {"x": self.x[index], "label": self.y[index]}


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.projector_l = nn.Linear(1, 1, bias=False)
        self.projector_a = nn.Linear(1, 1, bias=False)
        self.projector_v = nn.Linear(1, 1, bias=False)
        self.projector_c = nn.Linear(1, 1, bias=False)
        self.proj1 = nn.Linear(1, 1, bias=False)
        self.proj2 = nn.Linear(1, 1, bias=False)
        self.out_layer = nn.Linear(1, 1, bias=True)
        for parameter in self.parameters():
            nn.init.constant_(parameter, 0.12)

    def forward(self, x):
        value = (
            self.projector_l(x)
            + self.projector_a(x)
            + self.projector_v(x)
            + self.projector_c(x)
        )
        value = torch.tanh(self.proj1(value))
        value = self.proj2(value)
        return self.out_layer(value)


class ToyWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ToyBackbone()

    def forward(self, x):
        return self.backbone(x)


def main():
    model = ToyWrapper()
    loader = DataLoader(ToyDataset(), batch_size=7, shuffle=False)
    records = select_parameter_records(model, "fusion_tail")
    before = selected_parameter_fingerprint(records)

    def forward_batch(local_model, batch):
        return local_model(batch["x"]).view(-1), batch["label"].view(-1)

    collected = collect_mean_mae_gradients(
        model, loader, records, forward_batch
    )
    after = selected_parameter_fingerprint(records)
    assert before == after
    assert all(record.parameter.grad is None for record in records)
    assert collected["sample_counts"]["global"] == 25
    assert sum(
        collected["sample_counts"][name]
        for name in (
            "strong_negative",
            "negative",
            "boundary",
            "positive",
            "strong_positive",
        )
    ) == 25

    gradients = collected["gradients"]
    error = global_region_decomposition_error(
        gradients, collected["sample_counts"]
    )
    assert error < 1e-5
    pairwise, layerwise = pairwise_geometry_rows(gradients, records)
    assert len(pairwise) == 36
    assert layerwise
    assert any(bool(row["conflict"]) for row in pairwise)

    config = GradientAuditConfigV923()
    built = build_candidate_directions(gradients, config)
    assert abs(float(built["mgda"]["weights"].sum()) - 1.0) < 1e-6
    task_effects, direction_summary = direction_effect_rows(
        gradients, built["directions"], 1e-12
    )
    assert len(task_effects) == 24
    assert len(direction_summary) == 4
    summary = fold_geometry_summary(
        pairwise,
        direction_summary,
        built["mgda"],
        error,
        config,
    )
    assert 0.0 <= summary["region_pair_conflict_fraction"] <= 1.0
    assert summary["global_region_decomposition_relative_error"] < 1e-5
    print("V9.23 NO-TRAINING GRADIENT CONFLICT SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
