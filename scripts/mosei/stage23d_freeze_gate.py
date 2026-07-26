#!/usr/bin/env python3
"""Freeze previously qualitative Phase-1 gate operators before outer access."""

from __future__ import annotations

import json

from stage23d_self_risk_common import OUT, atomic_json, sha256_file, utc_now


def main():
    path = OUT / "protocol" / "gate_operationalization.json"
    if path.exists():
        print(path.read_text(encoding="utf-8"))
        return
    gate = {
        "stage": "Stage23D-A preregistered gate operationalization",
        "frozen_before_any_outer_evaluation": True,
        "primary_aggregation": (
            "unweighted mean across the 10 checkpoint audits and four modes"
        ),
        "phase1_conditions": {
            "G1": {"metric": "R2 mean Error_Spearman", "operator": ">=", "value": 0.25},
            "G2": {
                "metric": "worst checkpoint-fold R2 mean Error_Spearman",
                "operator": ">=",
                "value": 0.15,
            },
            "G3": {
                "metric": "Expert configs with positive R2 mean Spearman in both folds",
                "operator": ">=",
                "value": 4,
            },
            "G4": {"metric": "R2 mean bad20_AUROC", "operator": ">=", "value": 0.63},
            "G5": {
                "metric": "R2 mean confident_wrong_AUROC",
                "operator": ">=",
                "value": 0.63,
            },
            "G6": {
                "metric": "R2-minus-R1 mean Error_Spearman",
                "operator": ">=",
                "value": 0.02,
                "secondary_requirement": (
                    "R2-minus-R1 bad20 or confident-wrong mean AUROC >= 0.01, "
                    "and neither event AUROC delta < -0.01"
                ),
            },
            "G7": {
                "metric": "R2-minus-strongest-N0 mean Error_Spearman",
                "operator": ">=",
                "value": 0.03,
                "secondary_requirement": "R2-minus-strongest-N0 bad20 AUROC >= 0.02",
            },
            "G8": {
                "metric": "missing modes with positive R2 mean Error_Spearman",
                "operator": ">=",
                "value": 2,
            },
            "G9": {
                "metric": "risk-coverage basic monotonicity",
                "definition": (
                    "for R2, retained actual MAE ordered by coverage may have at "
                    "most one downward adjacent step per mode; mean improvement "
                    "versus random over coverage<1 must be positive; at least "
                    "2/3 missing modes and the all-mode aggregate must pass"
                ),
            },
            "G10": {
                "metric": "not mode/length/prediction-magnitude only",
                "definition": (
                    "R2 mean Error_Spearman exceeds both N2 length/mask-only and "
                    "N5 magnitude/sign/mode-prior by >=0.02"
                ),
            },
        },
        "strongest_shuffle_definition": (
            "the N0 fixed shuffle seed with the highest aggregate outer metric; "
            "choosing the strongest negative control is conservative and does "
            "not tune R2"
        ),
        "phase1_decision": {
            "PASS": "all ten G1-G10 conditions pass",
            "WEAK": (
                "at least 7/10 conditions pass, R2 mean Spearman >=0.20, "
                "worst-fold mean >=0.10, both event AUROCs >=0.58, and R2 "
                "Spearman deltas versus R1 and strongest N0 are positive"
            ),
            "FAIL": "neither PASS nor WEAK",
        },
        "a5_authorization": {
            "PASS": "full A5, beginning with the fixed two-Expert pilot",
            "WEAK": "only the fixed two-Expert pilot",
            "FAIL": "no A5 execution",
        },
        "a5_pilot_expansion": (
            "expand only if R3-minus-R2 mean Spearman >=0.05 OR bad20 AUROC "
            ">=0.03 OR confident-wrong AUROC >=0.03"
        ),
        "final_pass_gate": {
            "mean_Error_Spearman": 0.30,
            "each_expert_two_fold_mean_Error_Spearman": 0.20,
            "mean_bad20_AUROC": 0.65,
            "mean_confident_wrong_AUROC": 0.65,
            "R2_or_R3_beats_R1": "same explicit margins as G6",
            "R2_or_R3_beats_strongest_shuffle": "same explicit margins as G7",
            "missing_modes_pass": "at least 2/3",
            "fold_direction_consistent": True,
            "risk_coverage_pass": True,
            "calibrated_absolute_error_unit": True,
            "official_valid_test_access_count": 0,
            "arbiter_student_expert_training_count": 0,
        },
        "created_at": utc_now(),
    }
    atomic_json(path, gate)
    print(json.dumps({"path": str(path), "sha256": sha256_file(path)}, indent=2))


if __name__ == "__main__":
    main()
