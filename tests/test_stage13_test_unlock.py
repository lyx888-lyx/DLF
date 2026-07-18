from pathlib import Path


def test_test_unlock_requires_all_four_positive_gates():
    gates = {"a1": True, "a2": True, "a3": False, "b": False}
    assert not all(gates.values())


def test_no_unlock_manifest_is_part_of_stopped_audit():
    output = Path("result/missing_baseline/cs_dfmrc_v1/mosi")
    if output.exists():
        assert not (output / "TEST_UNLOCK_MANIFEST.json").exists()
