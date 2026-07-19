"""Run or aggregate Stage18B unified and fixed-mode specialist evidence."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from trains.singleTask.cfcompat_fair_trainer import train_no_test
from trains.singleTask.specialist_training import specialist_gap, train_specialist


ROOT = Path("result/missing_baseline/cfcompat_evidence_v1/mosi")
STAGE = ROOT / "stage18b_student_learnability"
CHECKPOINTS = Path("runtime/cfcompat_evidence_v1/checkpoints/mosi/stage18b")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action", choices=("unified", "specialist", "aggregate"), required=True
    )
    parser.add_argument("--seed", type=int, choices=(1111, 1114))
    parser.add_argument("--mode", choices=("LA", "LV", "L"))
    parser.add_argument("--num-workers", type=int, choices=(1,), default=1)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--physical-gpu", type=int, choices=(2,), default=2)
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--config-file", default="config/config.json")
    cli = parser.parse_args()
    if cli.action != "aggregate" and cli.seed is None:
        parser.error("--seed is required for training.")
    if cli.action == "specialist" and cli.mode is None:
        parser.error("--mode is required for specialist training.")
    if cli.action == "unified" and cli.seed == 1114:
        parser.error("seed1114 unified reuses Stage18A ModDrop Run A.")
    if cli.gpu_ids != [0]:
        parser.error("Only physical GPU 3 exposed as internal GPU 0 is allowed.")
    return cli


def unified_path(seed):
    if int(seed) == 1114:
        return (
            ROOT
            / "stage18a_training_recovery"
            / "moddrop_runA"
            / "run_metrics.csv"
        )
    return STAGE / "seed1111" / "unified" / "run_metrics.csv"


def aggregate():
    metric_rows, gaps = [], []
    for seed in (1114, 1111):
        unified = pd.read_csv(unified_path(seed)).iloc[0]
        teacher_errors = []
        for mode in ("LA", "LV", "L"):
            specialist_path = (
                STAGE
                / "seed{}".format(seed)
                / "specialist_{}".format(mode)
                / "run_metrics.csv"
            )
            specialist = pd.read_csv(specialist_path).iloc[0]
            e_unified = float(unified["valid_{}_MAE".format(mode)])
            e_specialist = float(specialist["valid_{}_MAE".format(mode)])
            e_teacher = float(specialist["teacher_LAV_MAE"])
            teacher_errors.append(e_teacher)
            g_opt, g_info, g_total = specialist_gap(
                e_unified, e_specialist, e_teacher
            )
            gaps.append(
                {
                    "Seed": seed,
                    "Mode": mode,
                    "E_unified": e_unified,
                    "E_specialist": e_specialist,
                    "E_teacher": e_teacher,
                    "G_opt": g_opt,
                    "G_info": g_info,
                    "G_total": g_total,
                    "IdentityResidual": g_total - (g_opt + g_info),
                }
            )
            metric_rows.append(
                {
                    "Seed": seed,
                    "Method": "UnifiedModDrop",
                    "Mode": mode,
                    "BestValidEpoch": int(unified.BestValidEpoch),
                    **{
                        metric: float(
                            unified["valid_{}_{}".format(mode, metric)]
                        )
                        for metric in (
                            "acc_7",
                            "acc_5",
                            "acc_2",
                            "F1_score",
                            "Corr",
                            "MAE",
                            "Loss",
                        )
                    },
                }
            )
            metric_rows.append(
                {
                    "Seed": seed,
                    "Method": "ModeSpecialist",
                    "Mode": mode,
                    "BestValidEpoch": int(specialist.BestValidEpoch),
                    **{
                        metric: float(
                            specialist["valid_{}_{}".format(mode, metric)]
                        )
                        for metric in (
                            "acc_7",
                            "acc_5",
                            "acc_2",
                            "F1_score",
                            "Corr",
                            "MAE",
                            "Loss",
                        )
                    },
                }
            )
        if max(teacher_errors) - min(teacher_errors) > 1e-12:
            raise RuntimeError("Teacher evaluation differs among specialists.")
        metric_rows.append(
            {
                "Seed": seed,
                "Method": "CleanLAVTeacher",
                "Mode": "LAV",
                "BestValidEpoch": np.nan,
                **{
                    metric: float(
                        pd.read_csv(
                            STAGE
                            / "seed{}".format(seed)
                            / "specialist_LA"
                            / "run_metrics.csv"
                        ).iloc[0]["teacher_LAV_{}".format(metric)]
                    )
                    for metric in (
                        "acc_7",
                        "acc_5",
                        "acc_2",
                        "F1_score",
                        "Corr",
                        "MAE",
                        "Loss",
                    )
                },
            }
        )
    metrics = pd.DataFrame(metric_rows)
    gap_frame = pd.DataFrame(gaps)
    if gap_frame.IdentityResidual.abs().max() > 1e-8:
        raise RuntimeError("Gap identity exceeded tolerance.")
    metrics.to_csv(STAGE / "stage18b_specialist_metrics.csv", index=False)
    gap_frame.to_csv(STAGE / "stage18b_gap_decomposition.csv", index=False)
    means = gap_frame.groupby("Mode")[["G_opt", "G_info", "G_total"]].mean()
    audit = """# Stage 18B Student Learnability Audit

Fixed-mode specialists used the same DLF student initialization family,
optimizer, scheduler, maximum budget, no-clipping rule, official validation
data, and early-stop patience as the unified trainer. They saw every train
segment exactly once per epoch, used one fixed missing mode, used labels only,
and used neither a Teacher nor KD. Specialist checkpoints were selected by the
target mode's official validation MAE; an all-mode J is not meaningful for a
single-mode model.

The decomposition identity held within 1e-8 for every seed and mode.

## Two-seed mean gaps

```
{}
```

Positive G_opt means the specialist reduced error relative to the unified
student; positive G_info is the remaining missing-view error relative to the
clean LAV Teacher. Conclusions must use both seeds and retain per-mode
heterogeneity.
""".format(means.to_string(float_format=lambda value: "{:.9f}".format(value)))
    (STAGE / "stage18b_student_learnability_audit.md").write_text(audit)
    print(gap_frame.to_string(index=False))


def main():
    cli = parse_args()
    if cli.action == "aggregate":
        aggregate()
    elif cli.action == "unified":
        train_no_test(
            cli,
            "moddrop",
            "Stage18B",
            STAGE / "seed{}".format(cli.seed) / "unified",
            CHECKPOINTS / "seed{}".format(cli.seed) / "unified",
        )
    else:
        train_specialist(
            cli,
            cli.mode,
            STAGE / "seed{}".format(cli.seed) / "specialist_{}".format(cli.mode),
            CHECKPOINTS
            / "seed{}".format(cli.seed)
            / "specialist_{}".format(cli.mode),
        )


if __name__ == "__main__":
    main()
