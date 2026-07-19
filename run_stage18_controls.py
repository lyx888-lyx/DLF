"""CLI for Stage18C/D control training and two-seed aggregation."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from trains.singleTask.control_training import train_control
from trains.singleTask.kd_control_gates import CONTROL_METHODS


ROOT = Path("result/missing_baseline/cfcompat_evidence_v1/mosi")
STAGE_C = ROOT / "stage18c_seed1114_controls"
STAGE_D = ROOT / "stage18d_seed1111_replication"
CHECKPOINTS = Path("runtime/cfcompat_evidence_v1/checkpoints/mosi")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action", choices=("train", "aggregate-c", "aggregate-d"), required=True
    )
    parser.add_argument("--seed", type=int, choices=(1111, 1114))
    parser.add_argument("--method", choices=CONTROL_METHODS)
    parser.add_argument("--physical-gpu", type=int, choices=(2,), default=2)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--num-workers", type=int, choices=(1,), default=1)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--config-file", default="config/config.json")
    cli = parser.parse_args()
    if cli.action == "train" and (cli.seed is None or cli.method is None):
        parser.error("Training requires --seed and --method.")
    if cli.gpu_ids != [0]:
        parser.error("Physical GPU 2 must be internal GPU 0.")
    return cli


def source_metric_path(seed, method):
    if seed == 1114 and method == "moddrop":
        return (
            ROOT
            / "stage18a_training_recovery/moddrop_runA/run_metrics.csv"
        )
    if seed == 1114 and method == "cfcompat":
        return (
            ROOT
            / "stage18a_training_recovery/cfcompat_runA/run_metrics.csv"
        )
    if seed == 1111 and method == "moddrop":
        return ROOT / "stage18b_student_learnability/seed1111/unified/run_metrics.csv"
    stage = STAGE_C if seed == 1114 else STAGE_D
    return stage / method / "run_metrics.csv"


def aggregate_seed(seed):
    stage = STAGE_C if seed == 1114 else STAGE_D
    rows = []
    for method in CONTROL_METHODS:
        row = pd.read_csv(source_metric_path(seed, method)).iloc[0].to_dict()
        row["Method"] = method
        rows.append(row)
    frame = pd.DataFrame(rows)
    columns = [
        "Seed",
        "Method",
        "BestValidEpoch",
        "LastEpoch",
        "J_valid",
        "valid_LAV_MAE",
        "valid_MissingMacro_MAE",
        "valid_LA_MAE",
        "valid_LV_MAE",
        "valid_L_MAE",
        "valid_LAV_Corr",
        "valid_MissingMacro_Corr",
        "valid_LAV_acc_7",
        "valid_LAV_acc_5",
        "valid_LAV_acc_2",
        "valid_LAV_F1_score",
        "TotalKDMass",
        "KDMass_LA",
        "KDMass_LV",
        "KDMass_L",
        "TrainingSeconds",
        "CheckpointSHA256",
    ]
    for column in columns:
        if column not in frame:
            frame[column] = 0.0 if column.startswith("KDMass") or column in {
                "TotalKDMass"
            } else np.nan
    destination = (
        stage / "stage18c_seed1114_metrics.csv"
        if seed == 1114
        else stage / "stage18d_seed1111_metrics.csv"
    )
    frame[columns].to_csv(destination, index=False)
    return frame


def aggregate_two_seed():
    first = aggregate_seed(1114)
    second = aggregate_seed(1111)
    combined = pd.concat([first, second], ignore_index=True)
    questions = [
        ("Q1", "uniform", "moddrop"),
        ("Q2", "cfcompat", "uniform"),
        ("Q3", "cfcompat", "equal_mass"),
        ("Q4", "cfcompat", "mode_mean"),
        ("Q5", "cfcompat", "shuffled_gate"),
        ("Q6", "cfcompat", "shuffled_teacher"),
        ("Q7", "oracle", "uniform"),
        ("Q8", "cfcompat", "oracle"),
    ]
    rows = []
    for question, method, reference in questions:
        for seed in (1114, 1111):
            m = combined.loc[
                combined.Seed.astype(int).eq(seed)
                & combined.Method.eq(method)
            ].iloc[0]
            r = combined.loc[
                combined.Seed.astype(int).eq(seed)
                & combined.Method.eq(reference)
            ].iloc[0]
            rows.append(
                {
                    "Question": question,
                    "Seed": seed,
                    "Method": method,
                    "Reference": reference,
                    "Delta_J": float(m.J_valid - r.J_valid),
                    "Delta_LAV_MAE": float(
                        m.valid_LAV_MAE - r.valid_LAV_MAE
                    ),
                    "Delta_MissingMacro_MAE": float(
                        m.valid_MissingMacro_MAE
                        - r.valid_MissingMacro_MAE
                    ),
                    "Delta_LAV_Corr": float(
                        m.valid_LAV_Corr - r.valid_LAV_Corr
                    ),
                    "Delta_MissingMacro_Corr": float(
                        m.valid_MissingMacro_Corr
                        - r.valid_MissingMacro_Corr
                    ),
                    "Delta_Acc7": float(
                        m.valid_LAV_acc_7 - r.valid_LAV_acc_7
                    ),
                    "Delta_Acc5": float(
                        m.valid_LAV_acc_5 - r.valid_LAV_acc_5
                    ),
                    "Delta_Acc2": float(
                        m.valid_LAV_acc_2 - r.valid_LAV_acc_2
                    ),
                    "Delta_F1": float(
                        m.valid_LAV_F1_score - r.valid_LAV_F1_score
                    ),
                }
            )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        STAGE_D / "stage18d_two_seed_comparison.csv", index=False
    )
    means = comparison.groupby(
        ["Question", "Method", "Reference"]
    ).mean(numeric_only=True)
    (STAGE_D / "stage18d_two_seed_audit.md").write_text(
        "# Stage 18D Two-Seed Control Audit\n\n"
        "Negative Delta J/MAE is improvement; positive Delta Corr/classification "
        "is improvement.\n\n```\n{}\n```\n".format(
            means.to_string(float_format=lambda x: "{:.9f}".format(x))
        )
    )
    print(means.to_string())


def main():
    cli = parse_args()
    if cli.action == "train":
        stage = STAGE_C if cli.seed == 1114 else STAGE_D
        train_control(
            cli,
            cli.method,
            stage / cli.method,
            CHECKPOINTS
            / ("stage18c" if cli.seed == 1114 else "stage18d")
            / cli.method,
        )
    elif cli.action == "aggregate-c":
        frame = aggregate_seed(1114)
        (STAGE_C / "stage18c_control_matrix_audit.md").write_text(
            "# Stage 18C Seed1114 Control Matrix\n\n"
            "All eight preregistered methods are reported without selection.\n\n"
            "```\n{}\n```\n".format(
                frame[
                    ["Method", "J_valid", "valid_LAV_MAE", "valid_MissingMacro_MAE"]
                ].to_string(index=False)
            )
        )
        print(frame[["Method", "J_valid"]].to_string(index=False))
    else:
        aggregate_two_seed()


if __name__ == "__main__":
    main()
