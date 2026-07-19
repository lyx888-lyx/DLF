from trains.singleTask.kd_control_gates import (
    build_epoch_bindings,
    deterministic_permutation,
    oracle_direction,
)


def fixture():
    indices = [0, 1, 2, 3, 4, 5]
    modes = ["LA", "LA", "LV", "LV", "L", "L"]
    cache = {}
    for index in indices:
        cache[index] = {
            "compat_LA": 0.1 + index * 0.01,
            "compat_LV": 0.2 + index * 0.01,
            "compat_L": 0.3 + index * 0.01,
            "evaluator_LA_pred": 0.0,
            "evaluator_LV_pred": 0.0,
            "evaluator_L_pred": 0.0,
        }
    teacher = {index: 1.0 for index in indices}
    labels = {index: (-1.0 if index % 2 else 1.0) for index in indices}
    return indices, modes, cache, teacher, labels


def test_mass_controls_exact():
    values = fixture()
    for method in ("equal_mass", "mode_mean", "shuffled_gate"):
        result = build_epoch_bindings(method, 1114, 1, *values)
        if method == "equal_mass":
            assert abs(
                result["mass"]["total"] - result["reference_mass"]["total"]
            ) <= 1e-8
        else:
            for mode in ("LA", "LV", "L"):
                assert abs(
                    result["mass"][mode]
                    - result["reference_mass"][mode]
                ) <= 1e-8


def test_deterministic_shuffles():
    first = deterministic_permutation([1, 2, 3], 1114, 1, "LA", "gate")
    second = deterministic_permutation([1, 2, 3], 1114, 1, "LA", "gate")
    assert first == second


def test_oracle_formula():
    assert oracle_direction(1.0, 0.0, 2.0) == 1.0
    assert oracle_direction(-1.0, 0.0, 2.0) == 0.0
