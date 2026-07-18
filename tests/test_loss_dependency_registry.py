from pathlib import Path

from trains.singleTask.loss_dependency_registry import (
    LOSS_BY_NAME,
    actual_view_active,
    semantic_active,
)


def test_actual_training_path_has_two_unmasked_missing_specific_losses():
    audio = LOSS_BY_NAME["missing_audio_specific_task"]
    visual = LOSS_BY_NAME["missing_visual_specific_task"]
    assert audio.currently_masked is False
    assert visual.currently_masked is False
    assert actual_view_active(audio, "LV") and not semantic_active(audio, "LV")
    assert actual_view_active(visual, "LA") and not semantic_active(visual, "LA")
    assert actual_view_active(audio, "L") and not semantic_active(audio, "L")
    assert actual_view_active(visual, "L") and not semantic_active(visual, "L")


def test_always_on_objectives_and_original_coefficients_are_frozen():
    assert LOSS_BY_NAME["missing_final_task"].always_on
    assert LOSS_BY_NAME["missing_common_task"].always_on
    assert LOSS_BY_NAME["missing_cfcompat_kd"].always_on
    assert LOSS_BY_NAME["missing_text_specific_task"].coefficient == 3.0
    assert LOSS_BY_NAME["missing_audio_specific_task"].coefficient == 1.0
    assert LOSS_BY_NAME["missing_visual_specific_task"].coefficient == 1.0


def test_full_auxiliary_families_are_only_computed_on_actual_lav_view():
    for name in (
        "full_audio_reconstruction",
        "full_visual_consistency",
        "full_text_orthogonality",
        "full_similarity_triplet",
    ):
        entry = LOSS_BY_NAME[name]
        assert actual_view_active(entry, "LAV")
        assert not actual_view_active(entry, "LA")


def test_historical_stage8_loop_is_identified_as_test_unsafe():
    source = Path("run_cfcompat_stability_multiseed.py").read_text()
    assert 'build_single_split_loader(\n        args, "test"' in source
    assert "test = evaluate_all_modes(" in source
