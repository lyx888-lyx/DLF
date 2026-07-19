import pandas as pd

from trains.singleTask.cfcompat_transfer_analysis import continuation_gate


def _metrics():
    values = {
        1114: {
            "equal_mass": 0.68,
            "shuffled_gate": 0.68,
            "shuffled_teacher": 0.6704,
            "cfcompat": 0.67,
        },
        1111: {
            "equal_mass": 0.69,
            "shuffled_gate": 0.69,
            "shuffled_teacher": 0.6804,
            "cfcompat": 0.68,
        },
    }
    return pd.DataFrame(
        [
            {"Seed": seed, "Method": method, "J_valid": value}
            for seed, methods in values.items()
            for method, value in methods.items()
        ]
    )


def _harmful(cf=0.1, uniform=0.2):
    return pd.DataFrame(
        [
            {
                "Seed": "POOLED",
                "Mode": "ALL",
                "Method": "cfcompat",
                "HarmfulImitationRate_Q2": cf,
            },
            {
                "Seed": "POOLED",
                "Mode": "ALL",
                "Method": "uniform",
                "HarmfulImitationRate_Q2": uniform,
            },
        ]
    )


def test_equivalent_shuffled_teacher_blocks_gate():
    checks, passed = continuation_gate(_metrics(), _harmful())
    assert not passed
    row = checks.loc[
        checks.Criterion.eq("ShuffledTeacherNotEquivalentToCFCompat")
    ].iloc[0]
    assert not bool(row.Passed)


def test_harmful_imitation_is_a_hard_gate():
    metrics = _metrics()
    metrics.loc[
        metrics.Method.eq("shuffled_teacher"), "J_valid"
    ] += 0.01
    checks, passed = continuation_gate(metrics, _harmful(cf=0.3, uniform=0.2))
    assert not passed
    row = checks.loc[
        checks.Criterion.eq("CFCompatHarmfulImitation<Uniform")
    ].iloc[0]
    assert not bool(row.Passed)
