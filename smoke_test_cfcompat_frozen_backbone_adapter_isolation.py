"""Lightweight v7 utility smoke test; no dataset, checkpoints, or Test split."""
import torch
import torch.nn as nn
import pandas as pd

from trains.singleTask.cfcompat_adapter_isolation_utils import (
    EXPECTED_TRAINABLE_NAMES,
    build_epoch_event_trajectory,
    clip_failure_epoch_summary,
    enforce_isolation_train_mode,
    failure_onset_table,
    freeze_shared_backbone,
    module_state_sha256,
    prediction_frame_to_long,
)


class ToyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
        self.missing_audio_token = nn.Parameter(torch.zeros(1, 1, 2))
        self.missing_vision_token = nn.Parameter(torch.zeros(1, 1, 3))
        self.mask_adapter = nn.Linear(3, 4, bias=False)


def prediction_frame(delta):
    return pd.DataFrame(
        [
            {"sample_index": 0, "sample_id": "a", "label": 1.0,
             "LAV_pred": 0.90 + delta, "LA_pred": 0.80 + delta,
             "LV_pred": 0.75 + delta, "L_pred": 0.70 + delta},
            {"sample_index": 1, "sample_id": "b", "label": -1.0,
             "LAV_pred": -0.90 + delta, "LA_pred": -0.80 + delta,
             "LV_pred": -0.75 + delta, "L_pred": -0.70 + delta},
        ]
    )


def main():
    student = ToyStudent()
    isolation = freeze_shared_backbone(student)
    assert tuple(isolation["trainable_names"]) == EXPECTED_TRAINABLE_NAMES
    enforce_isolation_train_mode(student)
    assert not student.backbone.training
    before = module_state_sha256(student.backbone)
    optimizer = torch.optim.SGD(
        [p for p in student.parameters() if p.requires_grad], lr=0.1
    )
    optimizer.zero_grad()
    loss = student.mask_adapter(torch.ones(2, 3)).pow(2).mean()
    loss = loss + student.missing_audio_token.sum() + student.missing_vision_token.sum()
    loss.backward(); optimizer.step()
    after = module_state_sha256(student.backbone)
    assert before == after

    ref = pd.DataFrame(
        [
            {"sample_index": 0, "sample_id": "a", "label": 1.0,
             "teacher_prediction": 0.95, "baseline_LAV_pred": 0.4,
             "baseline_LA_pred": 0.4, "baseline_LV_pred": 0.4, "baseline_L_pred": 0.4},
            {"sample_index": 1, "sample_id": "b", "label": -1.0,
             "teacher_prediction": -0.95, "baseline_LAV_pred": -0.4,
             "baseline_LA_pred": -0.4, "baseline_LV_pred": -0.4, "baseline_L_pred": -0.4},
        ]
    )
    epochs = pd.concat(
        [
            prediction_frame_to_long(prediction_frame(0.0), 0),
            prediction_frame_to_long(prediction_frame(-0.1), 1),
            prediction_frame_to_long(prediction_frame(-1.3), 2),
        ],
        ignore_index=True,
    )
    trajectory = build_epoch_event_trajectory(epochs, ref, selected_best_epoch=1)
    onset = failure_onset_table(trajectory)
    clips = clip_failure_epoch_summary(trajectory)
    assert len(trajectory) == 2 * 4 * 3
    assert len(onset) == 2 * 3
    assert set(clips.Epoch.astype(int)) == {1, 2}
    print("v7 frozen-backbone adapter-isolation utility smoke test: PASS")


if __name__ == "__main__":
    main()
