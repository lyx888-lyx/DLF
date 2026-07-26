import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "mosei"))

from stage23d_self_risk_common import (  # noqa: E402
    ACTIVE_HEADS,
    ReadOnlyActivationCapture,
    active_head_features,
    attach_source_roles,
    risk_labels,
    source_stratified_split,
)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_layer = nn.Linear(4, 1)
        self.out_layer_c = nn.Linear(3, 1)
        self.out_layer_l_high = nn.Linear(2, 1)
        self.out_layer_a_high = nn.Linear(2, 1)
        self.out_layer_v_high = nn.Linear(2, 1)

    def forward(self, values):
        final = values[:, :4]
        shared = values[:, :3]
        specific = values[:, :2]
        return {
            "output_logit": self.out_layer(final),
            "logits_c": self.out_layer_c(shared),
            "logits_l_hetero": self.out_layer_l_high(specific),
            "logits_a_hetero": self.out_layer_a_high(specific),
            "logits_v_hetero": self.out_layer_v_high(specific),
        }


class Stage23DSelfRiskTests(unittest.TestCase):
    def test_read_only_hook_preserves_prediction(self):
        torch.manual_seed(7)
        model = TinyBackbone().eval()
        values = torch.randn(5, 4)
        expected = model(values)["output_logit"].detach().clone()
        with ReadOnlyActivationCapture(model) as capture:
            actual = model(values)["output_logit"].detach().clone()
            self.assertEqual(set(capture.values), set(capture.MODULES))
        self.assertTrue(torch.equal(expected, actual))

    def test_inactive_heads_are_not_emitted_as_values(self):
        outputs = {
            "output_logit": 0.5,
            "logits_c": 0.4,
            "logits_l_hetero": 0.3,
            "logits_a_hetero": 0.2,
            "logits_v_hetero": 9.9,
        }
        result = active_head_features("LA", outputs)
        self.assertEqual(result["head_active__logits_v_hetero"], 0)
        self.assertNotIn("active__logits_v_hetero", result)
        self.assertEqual(
            result["active_head_names"], "|".join(ACTIVE_HEADS["LA"])
        )

    def test_source_split_has_no_leakage(self):
        rows = []
        for source in range(120):
            for clip in range(1 + source % 4):
                rows.append(
                    {
                        "sample_id": f"s{source}_{clip}",
                        "video_id": f"s{source}",
                        "label": float((source % 7) - 3),
                    }
                )
        samples = pd.DataFrame(rows)
        roles = source_stratified_split(samples, 23170)
        joined = attach_source_roles(samples, roles)
        by_source = joined.groupby("video_id")["self_risk_role"].nunique()
        self.assertEqual(int(by_source.max()), 1)
        self.assertEqual(
            set(joined["self_risk_role"]),
            {"inner_train", "inner_valid", "outer"},
        )

    def test_risk_thresholds_use_inner_train_only(self):
        rows = []
        for mode in ("LAV", "LA", "LV", "L"):
            for index in range(40):
                role = (
                    "inner_train"
                    if index < 30
                    else "inner_valid"
                    if index < 35
                    else "outer"
                )
                rows.append(
                    {
                        "mode": mode,
                        "self_risk_role": role,
                        "label": float(index) / 10.0,
                        "prediction": 0.0,
                        "raw_uncertainty": float(index) / 100.0,
                    }
                )
        frame = pd.DataFrame(rows)
        _, first = risk_labels(frame)
        changed = frame.copy()
        changed.loc[changed["self_risk_role"] == "outer", "label"] = 10000.0
        _, second = risk_labels(changed)
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(np.isfinite(first.select_dtypes(include=[np.number])).all().all())


if __name__ == "__main__":
    unittest.main()
