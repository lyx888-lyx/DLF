"""Synthetic smoke test for V9.24 region-gradient consolidation."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from trains.singleTask.gradient_conflict_audit_v923 import (
    collect_mean_mae_gradients,
)
from trains.singleTask.region_gradient_consolidation_v924 import (
    STRATEGIES,
    RegionGradientTrainingConfigV924,
    build_training_direction,
    evaluate_model,
    freeze_to_parameter_scope,
    regression_metrics,
    train_one_strategy,
)


class ToyDataset(Dataset):
    def __init__(self):
        self.x = torch.linspace(-1.5, 1.5, 60).view(-1, 1)
        labels = torch.tensor(
            [
                -2.6, -2.3, -2.0, -1.8, -1.6,
                -1.4, -1.2, -1.0, -0.8, -0.6,
                -0.45, -0.35, -0.25, -0.15, -0.05,
                0.05, 0.15, 0.25, 0.35, 0.45,
                0.6, 0.8, 1.0, 1.2, 1.4,
                1.6, 1.8, 2.0, 2.3, 2.6,
            ],
            dtype=torch.float32,
        )
        self.y = torch.cat([labels, labels], dim=0)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return {
            "x": self.x[index],
            "label": self.y[index],
            "index": torch.tensor(index),
        }


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(1, 1)
        self.projector_l = nn.Linear(1, 2)
        self.projector_a = nn.Linear(1, 2)
        self.projector_v = nn.Linear(1, 2)
        self.projector_c = nn.Linear(1, 2)
        self.proj1 = nn.Linear(2, 2)
        self.proj2 = nn.Linear(2, 2)
        self.out_layer = nn.Linear(2, 1)

    def forward(self, x):
        base = self.encoder(x)
        fused = (
            self.projector_l(base)
            + self.projector_a(base)
            + self.projector_v(base)
            + self.projector_c(base)
        ) / 4.0
        fused = torch.tanh(self.proj1(fused))
        fused = self.proj2(fused)
        return self.out_layer(fused)


class ToyWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ToyBackbone()

    def forward(self, x):
        return self.backbone(x)


def forward_batch(model, batch):
    return (
        model(batch["x"]).view(-1),
        batch["label"].view(-1),
    )


def main():
    torch.manual_seed(24)
    dataset = ToyDataset()
    train_loader = DataLoader(
        dataset,
        batch_size=9,
        shuffle=False,
    )
    valid_loader = DataLoader(
        dataset,
        batch_size=11,
        shuffle=False,
    )
    initial_model = ToyWrapper()
    initial_state = copy.deepcopy(initial_model.state_dict())
    frozen_encoder = initial_state[
        "backbone.encoder.weight"
    ].clone()

    probe = ToyWrapper()
    probe.load_state_dict(initial_state)
    records = freeze_to_parameter_scope(probe, "fusion_tail")
    collected = collect_mean_mae_gradients(
        probe,
        train_loader,
        records,
        forward_batch,
    )
    for strategy in STRATEGIES:
        built = build_training_direction(
            strategy,
            collected["gradients"],
            RegionGradientTrainingConfigV924(
                max_epochs=2,
                early_stop=2,
                minimum_mgda_direction_norm=1e-4,
            ),
        )
        assert built["applied_direction_norm"] > 0.0
        if strategy == "normalized_mgda":
            assert (
                abs(sum(built["task_weights"].values()) - 1.0)
                < 1e-5
            )
            assert all(
                value >= -1e-8
                for value in built["task_weights"].values()
            )

    for strategy in STRATEGIES:
        model = ToyWrapper()
        model.load_state_dict(initial_state)
        result = train_one_strategy(
            model,
            strategy,
            train_loader,
            valid_loader,
            forward_batch,
            RegionGradientTrainingConfigV924(
                learning_rate=1e-3,
                max_epochs=2,
                early_stop=2,
                gradient_clip_norm=1.0,
                minimum_mgda_direction_norm=1e-4,
            ),
        )
        assert result["strategy"] == strategy
        assert result["selected_parameter_count"] > 0
        assert result["best_epoch"] in {0, 1, 2}
        assert torch.equal(
            model.state_dict()["backbone.encoder.weight"],
            frozen_encoder,
        )
        evaluated = evaluate_model(
            model,
            valid_loader,
            forward_batch,
        )
        metrics = regression_metrics(
            evaluated["prediction"],
            evaluated["labels"],
        )
        assert torch.isfinite(torch.tensor(metrics["mae"]))
        assert len(metrics["region_mae"]) == 5

    print(
        "V9.24 REGION-GRADIENT CONSOLIDATION "
        "SMOKE TEST PASSED"
    )


if __name__ == "__main__":
    main()
