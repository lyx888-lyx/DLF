"""Safe V9.3 router selection: baselines never masquerade as the deployable route."""

from __future__ import annotations

import pandas as pd

from .coach_routing_v93 import CoachRouterV93 as _BaseCoachRouterV93
from .coach_utils_v93 import json_scalar


class CoachRouterV93(_BaseCoachRouterV93):
    """Select the deployable policy only from Anchor or advantage-aware routing.

    Category-only routes remain reported baselines and are selected separately
    during final evaluation. This prevents a winning hard-route baseline from
    being interpreted through the advantage router's threshold fields.
    """

    def calibrate(self, advantage_results, valid_region):
        super().calibrate(advantage_results, valid_region)
        frame = pd.read_csv(self.save_dir / "v93_route_calibration.csv")
        deployable = frame.loc[
            frame["mode"].isin(("anchor", "ordinal_advantage"))
        ].sort_values(
            ["objective", "mae", "harm_over_010_rate"],
            kind="mergesort",
        )
        if deployable.empty:
            raise RuntimeError("calibration generated no deployable policy")
        return {
            key: json_scalar(value)
            for key, value in deployable.iloc[0].to_dict().items()
        }
