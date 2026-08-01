"""Boundary-safe V9.5 router: soft probabilities never redefine semantic regions."""

from __future__ import annotations

import torch

from .model.SelectiveCategoryCoachV95 import (
    REGION_TO_SPECIALIST,
    boundary_distance,
    region_index,
    semantic_region_probabilities,
    specialist_center_progress,
)
from .selective_category_coach_system_v95 import (
    SelectiveCategoryCoachTrainerV95 as _BaseTrainerV95,
)


class SelectiveCategoryCoachTrainerV95(_BaseTrainerV95):
    """Use fixed MOSI boundaries for category identity and smooth scores for confidence."""

    def _route(self, pool, region_output, policy):
        anchor = pool["anchor"].float().view(-1)
        anchor_probs = semantic_region_probabilities(anchor, policy["temperature"])
        calibrated_score = region_output["score"].float().view(-1)
        calibrated_probs = semantic_region_probabilities(
            calibrated_score, policy["temperature"]
        )

        # The semantic category is defined only by the fixed label thresholds.
        # Smooth probabilities are confidence estimates and must not move them.
        anchor_region = region_index(anchor)
        calibrated_region = region_index(calibrated_score)
        source = policy["source"]
        if source == "anchor_score":
            proposed = anchor_region
            probabilities = anchor_probs
            agreement = torch.ones_like(proposed, dtype=torch.bool)
            score = anchor
        elif source == "calibrated":
            proposed = calibrated_region
            probabilities = calibrated_probs
            agreement = torch.ones_like(proposed, dtype=torch.bool)
            score = calibrated_score
        elif source == "agreement":
            proposed = calibrated_region
            probabilities = calibrated_probs
            agreement = calibrated_region == anchor_region
            score = calibrated_score
        else:
            raise ValueError(f"unknown category source: {source}")

        region_probability = probabilities.gather(1, proposed.view(-1, 1)).view(-1)
        distance = boundary_distance(score)
        chosen = anchor.clone()
        activated = torch.zeros(len(anchor), dtype=torch.bool)
        selected_names = ["anchor"] * len(anchor)
        for region, name in REGION_TO_SPECIALIST.items():
            prediction = self._expert(pool, name, "prediction")
            confidence = self._expert(pool, name, "confidence")
            mask = proposed == int(region)
            mask &= agreement
            mask &= region_probability >= float(policy["min_region_probability"])
            mask &= distance >= float(policy["min_boundary_distance"])
            mask &= confidence >= float(policy["min_expert_confidence"])
            if bool(policy["require_center_progress"]):
                mask &= specialist_center_progress(anchor, prediction, int(region))
            if region == 0:
                mask &= prediction <= anchor
            elif region == 2:
                mask &= prediction.abs() <= anchor.abs()
            else:
                mask &= prediction >= anchor
            chosen[mask] = prediction[mask]
            activated |= mask
            for index in torch.nonzero(mask, as_tuple=False).view(-1).tolist():
                selected_names[index] = name

        beta = float(policy["beta"])
        return {
            "prediction": (anchor + beta * (chosen - anchor)).view(-1, 1),
            "activated": activated,
            "selected_names": selected_names,
            "anchor_region": anchor_region,
            "calibrated_region": calibrated_region,
            "proposed_region": proposed,
            "region_probability": region_probability,
            "boundary_distance": distance,
        }
