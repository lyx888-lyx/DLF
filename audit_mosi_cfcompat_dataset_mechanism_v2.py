"""Hardened independent-audit entry point."""
from __future__ import annotations

import audit_mosi_cfcompat_dataset_mechanism as base
from trains.singleTask.mosi_cfcompat_audit_v2_utils import (
    joint_video_bootstrap,
    mechanism_assessment,
    opportunity_ranking,
    prediction_events,
)


def main():
    base.prediction_events = prediction_events
    base.joint_video_bootstrap = joint_video_bootstrap
    base.opportunity_ranking = opportunity_ranking
    base.mechanism_assessment = mechanism_assessment
    base.main()


if __name__ == "__main__":
    main()
