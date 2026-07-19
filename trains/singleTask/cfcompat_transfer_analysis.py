"""Train-only transfer diagnostics for the preregistered Stage 18 controls."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


KEYS = ["Seed", "sample_index", "mode"]
METHODS = (
    "uniform",
    "equal_mass",
    "mode_mean",
    "shuffled_gate",
    "shuffled_teacher",
    "oracle",
    "cfcompat",
)


def _safe_binary_metrics(score, target):
    score = np.asarray(score, dtype=float)
    target = np.asarray(target, dtype=int)
    if len(np.unique(target)) < 2:
        return {
            "AUROC": np.nan,
            "AUPRC": np.nan,
            "PositiveRate": float(target.mean()),
            "Spearman": np.nan,
            "PointBiserial": np.nan,
        }
    return {
        "AUROC": float(roc_auc_score(target, score)),
        "AUPRC": float(average_precision_score(target, score)),
        "PositiveRate": float(target.mean()),
        "Spearman": float(spearmanr(score, target).correlation),
        "PointBiserial": float(pointbiserialr(target, score).correlation),
    }


def bind_transfer_rows(baseline, method):
    """Bind a method to its same-seed ModDrop baseline and derive outcomes."""
    required = {
        *KEYS,
        "prediction",
        "teacher_prediction",
        "label",
        "compatibility",
        "oracle_gate",
        "Method",
    }
    for name, frame in (("baseline", baseline), ("method", method)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError("{} missing columns {}".format(name, sorted(missing)))
        if frame.duplicated(KEYS).any():
            raise ValueError("{} has duplicate sample bindings".format(name))
    left = baseline.rename(
        columns={
            "prediction": "prediction_before",
            "teacher_prediction": "teacher_before",
            "label": "label_before",
            "compatibility": "compatibility_before",
            "oracle_gate": "oracle_gate_before",
        }
    )
    right = method.rename(
        columns={
            "prediction": "prediction_after",
            "teacher_prediction": "teacher_after",
            "label": "label_after",
            "compatibility": "compatibility_after",
            "oracle_gate": "oracle_gate_after",
        }
    )
    bound = left.merge(right, on=KEYS, how="inner", validate="one_to_one")
    if len(bound) != len(left) or len(bound) != len(right):
        raise ValueError("Method and ModDrop sample bindings are incomplete")
    for stem in ("teacher", "label", "compatibility", "oracle_gate"):
        before = bound["{}_before".format(stem)].to_numpy(dtype=float)
        after = bound["{}_after".format(stem)].to_numpy(dtype=float)
        if not np.allclose(before, after, rtol=0.0, atol=1e-12):
            raise ValueError("{} changed across frozen predictions".format(stem))
    bound["Method"] = bound["Method_y"]
    bound["teacher_prediction"] = bound["teacher_before"]
    bound["label"] = bound["label_before"]
    bound["compatibility"] = bound["compatibility_before"]
    bound["oracle_gate"] = bound["oracle_gate_before"].astype(int)
    bound["Delta_T"] = (
        (bound.prediction_after - bound.teacher_prediction).abs()
        - (bound.prediction_before - bound.teacher_prediction).abs()
    )
    bound["Delta_Y"] = (
        (bound.prediction_after - bound.label).abs()
        - (bound.prediction_before - bound.label).abs()
    )
    toward = bound.Delta_T < 0
    improved = bound.Delta_Y < 0
    worsened_t = bound.Delta_T > 0
    worsened_y = bound.Delta_Y > 0
    bound["Quadrant"] = np.select(
        [
            toward & improved,
            toward & worsened_y,
            worsened_t & improved,
            worsened_t & worsened_y,
        ],
        ["Q1", "Q2", "Q3", "Q4"],
        default="Boundary",
    )
    bound["TeacherBetter"] = (
        (bound.teacher_prediction - bound.label).abs()
        < (bound.prediction_before - bound.label).abs()
    ).astype(int)
    bound["DirectionCorrect"] = (
        (bound.teacher_prediction - bound.prediction_before)
        * (bound.label - bound.prediction_before)
        > 0
    ).astype(int)
    return bound


def _summary_row(frame, seed, method, mode):
    quadrants = frame.Quadrant.value_counts()
    total = len(frame)
    return {
        "Seed": seed,
        "Method": method,
        "Mode": mode,
        "N": total,
        "TeacherApproachRate": float((frame.Delta_T < 0).mean()),
        "HelpfulTransferRate_Q1": float(quadrants.get("Q1", 0) / total),
        "HarmfulImitationRate_Q2": float(quadrants.get("Q2", 0) / total),
        "ImprovementWithoutImitationRate_Q3": float(
            quadrants.get("Q3", 0) / total
        ),
        "DoubleFailureRate_Q4": float(quadrants.get("Q4", 0) / total),
        "BoundaryRate": float(quadrants.get("Boundary", 0) / total),
        "MeanDeltaT": float(frame.Delta_T.mean()),
        "MeanDeltaY": float(frame.Delta_Y.mean()),
    }


def transfer_summaries(bound):
    rows = []
    for (seed, method), frame in bound.groupby(["Seed", "Method"], sort=True):
        rows.append(_summary_row(frame, int(seed), method, "ALL"))
        for mode, local in frame.groupby("mode", sort=True):
            rows.append(_summary_row(local, int(seed), method, mode))
    for method, frame in bound.groupby("Method", sort=True):
        rows.append(_summary_row(frame, "POOLED", method, "ALL"))
        for mode, local in frame.groupby("mode", sort=True):
            rows.append(_summary_row(local, "POOLED", method, mode))
    return pd.DataFrame(rows)


def teacher_benefit_summaries(bound):
    base = bound.drop_duplicates(KEYS).copy()
    base["LabelIntensity"] = pd.cut(
        base.label.abs(),
        bins=[-np.inf, 1.0, 2.0, 3.0, np.inf],
        labels=["abs<1", "1<=abs<2", "2<=abs<3", "abs>=3"],
        right=False,
    ).astype(str)
    base["PredictionPolarity"] = np.where(
        base.prediction_before > 0,
        "positive",
        np.where(base.prediction_before < 0, "negative", "zero"),
    )
    rows = []

    def append(group_type, group_value, frame, seed):
        rows.append(
            {
                "Seed": seed,
                "GroupType": group_type,
                "GroupValue": group_value,
                "N": len(frame),
                "TeacherBetterRate": float(frame.TeacherBetter.mean()),
                "DirectionCorrectRate": float(frame.DirectionCorrect.mean()),
                "OracleGateRate": float(frame.oracle_gate.mean()),
                "MeanCompatibility": float(frame.compatibility.mean()),
            }
        )

    for seed, frame in base.groupby("Seed", sort=True):
        append("all", "ALL", frame, int(seed))
        for column, kind in (
            ("mode", "mode"),
            ("LabelIntensity", "label_intensity"),
            ("PredictionPolarity", "prediction_polarity"),
        ):
            for value, local in frame.groupby(column, sort=True):
                append(kind, value, local, int(seed))
    append("all", "ALL", base, "POOLED")
    for column, kind in (
        ("mode", "mode"),
        ("LabelIntensity", "label_intensity"),
        ("PredictionPolarity", "prediction_polarity"),
        ("Seed", "seed"),
    ):
        for value, local in base.groupby(column, sort=True):
            append(kind, value, local, "POOLED")
    return pd.DataFrame(rows)


def compatibility_summaries(bound):
    base = bound.drop_duplicates(KEYS).copy()
    targets = (
        ("TeacherBetter", "TeacherBetter"),
        ("DirectionCorrect", "DirectionCorrect"),
        ("OracleGate", "oracle_gate"),
    )
    rows = []
    groups = [("POOLED", "ALL", base)]
    for seed, frame in base.groupby("Seed", sort=True):
        groups.append((int(seed), "ALL", frame))
        for mode, local in frame.groupby("mode", sort=True):
            groups.append((int(seed), mode, local))
    for seed, mode, frame in groups:
        ordered = frame.sort_values(
            ["compatibility", "sample_index", "mode"], kind="mergesort"
        )
        size = max(1, len(ordered) // 10)
        low = ordered.iloc[:size]
        high = ordered.iloc[-size:]
        for target_name, column in targets:
            metrics = _safe_binary_metrics(
                frame.compatibility.to_numpy(), frame[column].to_numpy()
            )
            rows.append(
                {
                    "Seed": seed,
                    "Mode": mode,
                    "Target": target_name,
                    "N": len(frame),
                    **metrics,
                    "LowDecilePositiveRate": float(low[column].mean()),
                    "HighDecilePositiveRate": float(high[column].mean()),
                    "HighMinusLowPositiveRate": float(
                        high[column].mean() - low[column].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def compatibility_deciles(bound):
    rows = []
    for seed, seed_frame in bound.groupby("Seed", sort=True):
        baseline = seed_frame.drop_duplicates(KEYS).copy()
        baseline["Decile"] = (
            pd.qcut(
                baseline.compatibility.rank(method="first"),
                10,
                labels=False,
            )
            + 1
        )
        lookup = baseline[KEYS + ["Decile"]]
        local = seed_frame.merge(lookup, on=KEYS, validate="many_to_one")
        for decile, frame in local.groupby("Decile", sort=True):
            unique = frame.drop_duplicates(KEYS)
            uniform = frame.loc[frame.Method.eq("uniform")]
            cfcompat = frame.loc[frame.Method.eq("cfcompat")]
            mode_counts = unique["mode"].value_counts()
            rows.append(
                {
                    "Seed": int(seed),
                    "Decile": int(decile),
                    "N": len(unique),
                    "MeanCompatibility": float(unique.compatibility.mean()),
                    "TeacherBetterRate": float(unique.TeacherBetter.mean()),
                    "DirectionCorrectRate": float(unique.DirectionCorrect.mean()),
                    "UniformMeanDeltaY": float(uniform.Delta_Y.mean()),
                    "CFCompatMeanDeltaY": float(cfcompat.Delta_Y.mean()),
                    "UniformHarmfulImitationRate": float(
                        (uniform.Quadrant == "Q2").mean()
                    ),
                    "CFCompatHarmfulImitationRate": float(
                        (cfcompat.Quadrant == "Q2").mean()
                    ),
                    "N_LA": int(mode_counts.get("LA", 0)),
                    "N_LV": int(mode_counts.get("LV", 0)),
                    "N_L": int(mode_counts.get("L", 0)),
                }
            )
    return pd.DataFrame(rows)


def continuation_gate(metric_frame, harmful_frame, tolerance=0.001):
    def j(seed, method):
        return float(
            metric_frame.loc[
                metric_frame.Seed.astype(int).eq(seed)
                & metric_frame.Method.eq(method),
                "J_valid",
            ].iloc[0]
        )

    seeds = (1114, 1111)
    cf_equal = [j(seed, "cfcompat") - j(seed, "equal_mass") for seed in seeds]
    cf_shuffle = [
        j(seed, "cfcompat") - j(seed, "shuffled_gate") for seed in seeds
    ]
    cf_shuffled_teacher = [
        j(seed, "cfcompat") - j(seed, "shuffled_teacher") for seed in seeds
    ]
    pooled = harmful_frame.loc[
        harmful_frame.Seed.astype(str).eq("POOLED")
        & harmful_frame.Mode.eq("ALL")
    ].set_index("Method")
    harmful_cf = float(pooled.loc["cfcompat", "HarmfulImitationRate_Q2"])
    harmful_uniform = float(pooled.loc["uniform", "HarmfulImitationRate_Q2"])
    shuffled_teacher_equivalent = bool(
        abs(float(np.mean(cf_shuffled_teacher))) <= tolerance
        and max(abs(value) for value in cf_shuffled_teacher) <= tolerance
    )
    checks = [
        {
            "Criterion": "CFCompatVsEqualMassMeanDeltaJ<0",
            "Value": float(np.mean(cf_equal)),
            "Threshold": "<0",
            "Passed": bool(np.mean(cf_equal) < 0),
        },
        {
            "Criterion": "CFCompatVsShuffledGateMeanDeltaJ<0",
            "Value": float(np.mean(cf_shuffle)),
            "Threshold": "<0",
            "Passed": bool(np.mean(cf_shuffle) < 0),
        },
        {
            "Criterion": "WorstTwoSeedComparisonDegradation<=0.001",
            "Value": float(max(cf_equal + cf_shuffle)),
            "Threshold": "<=0.001",
            "Passed": bool(max(cf_equal + cf_shuffle) <= tolerance),
        },
        {
            "Criterion": "CFCompatHarmfulImitation<Uniform",
            "Value": harmful_cf - harmful_uniform,
            "Threshold": "<0",
            "Passed": bool(harmful_cf < harmful_uniform),
        },
        {
            "Criterion": "ShuffledTeacherNotEquivalentToCFCompat",
            "Value": float(np.mean(cf_shuffled_teacher)),
            "Threshold": "not equivalent within abs(delta)<=0.001",
            "Passed": not shuffled_teacher_equivalent,
        },
    ]
    frame = pd.DataFrame(checks)
    return frame, bool(frame.Passed.all())


def write_mass_audit(stage_c):
    stage_c = Path(stage_c)
    controlled = {
        "equal_mass": ("total",),
        "mode_mean": ("total", "LA", "LV", "L"),
        "shuffled_gate": ("total", "LA", "LV", "L"),
        "shuffled_teacher": ("total", "LA", "LV", "L"),
    }
    rows = []
    for method, scopes in controlled.items():
        frame = pd.read_csv(stage_c / method / "kd_mass_by_epoch.csv")
        for _, row in frame.iterrows():
            for scope in scopes:
                actual = (
                    float(row.TotalKDMass)
                    if scope == "total"
                    else float(row["KDMass_{}".format(scope)])
                )
                reference = (
                    float(row.ReferenceCFMass)
                    if scope == "total"
                    else float(row["ReferenceCFMass_{}".format(scope)])
                )
                rows.append(
                    {
                        "Seed": int(row.Seed),
                        "Method": method,
                        "Epoch": int(row.Epoch),
                        "Scope": scope,
                        "ActualMass": actual,
                        "ReferenceCFMass": reference,
                        "AbsDelta": abs(actual - reference),
                        "Passed1e-10": abs(actual - reference) <= 1e-10,
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(stage_c / "stage18c_kd_mass_audit.csv", index=False)
    return result
