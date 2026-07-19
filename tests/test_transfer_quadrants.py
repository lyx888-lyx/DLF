import pandas as pd

from trains.singleTask.cfcompat_transfer_analysis import bind_transfer_rows


def _frame(method, predictions):
    return pd.DataFrame(
        {
            "Seed": [1114] * 4,
            "Method": [method] * 4,
            "sample_index": list(range(4)),
            "mode": ["LA"] * 4,
            "prediction": predictions,
            "teacher_prediction": [1.0] * 4,
            "label": [1.0, -1.0, -1.0, 1.0],
            "compatibility": [0.1, 0.2, 0.3, 0.4],
            "oracle_gate": [1, 0, 0, 1],
        }
    )


def test_four_quadrants_and_transfer_definitions():
    baseline = _frame("moddrop", [0.0, 0.0, 0.0, 0.0])
    method = _frame("uniform", [0.5, 0.5, -0.5, -0.5])
    bound = bind_transfer_rows(baseline, method)
    assert bound.Quadrant.tolist() == ["Q1", "Q2", "Q3", "Q4"]
    assert bound.TeacherBetter.tolist() == [1, 0, 0, 1]
    assert bound.DirectionCorrect.tolist() == [1, 0, 0, 1]


def test_binding_rejects_changed_labels():
    baseline = _frame("moddrop", [0.0] * 4)
    method = _frame("uniform", [0.1] * 4)
    method.loc[0, "label"] = 2.0
    try:
        bind_transfer_rows(baseline, method)
    except ValueError as error:
        assert "label changed" in str(error)
    else:
        raise AssertionError("Changed labels must fail sample binding")
