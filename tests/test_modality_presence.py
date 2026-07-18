import pytest
import torch

from trains.singleTask.modality_presence import (
    modes_from_presence,
    presence_from_mode,
    required_modalities_active,
    validate_presence_binding,
)


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("LAV", [1.0, 1.0, 1.0]),
        ("LA", [1.0, 1.0, 0.0]),
        ("LV", [1.0, 0.0, 1.0]),
        ("L", [1.0, 0.0, 0.0]),
    ],
)
def test_authoritative_mode_presence(mode, expected):
    value = presence_from_mode(mode, 2)
    assert value.shape == (2, 3)
    assert value[0].tolist() == expected
    assert modes_from_presence(value) == [mode, mode]
    assert validate_presence_binding(mode, value)["MismatchCount"] == 0


def test_presence_rejects_noncanonical_rows_and_never_reads_features():
    with pytest.raises(ValueError):
        modes_from_presence(torch.tensor([[1.0, 0.5, 0.0]]))
    assert "feature" not in presence_from_mode.__code__.co_varnames


def test_required_modality_activation():
    values = torch.stack(
        [presence_from_mode(mode) for mode in ("LAV", "LA", "LV", "L")]
    )
    assert required_modalities_active(("L",), values).tolist() == [1, 1, 1, 1]
    assert required_modalities_active(("A",), values).tolist() == [1, 1, 0, 0]
    assert required_modalities_active(("V",), values).tolist() == [1, 0, 1, 0]
    assert required_modalities_active(("A", "V"), values).tolist() == [1, 0, 0, 0]
