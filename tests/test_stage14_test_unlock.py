from pathlib import Path


def test_lomo_failure_blocks_stage14_test_unlock():
    gates = {"a0": True, "a1": True, "a2": True, "a3": False}
    assert not all(gates.values())


def test_stopped_run_has_no_test_unlock_manifest():
    output = Path("result/missing_baseline/dcrc_v1/mosi")
    if output.exists():
        assert not (output / "TEST_UNLOCK_MANIFEST.json").exists()
        assert not (
            output
            / "frozen_method/DCRC_FROZEN_METHOD_MANIFEST.json"
        ).exists()
