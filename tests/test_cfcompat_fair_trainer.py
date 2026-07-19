import inspect

from trains.singleTask import cfcompat_fair_trainer as fair


def test_no_test_symbols_or_loader_path():
    source = inspect.getsource(fair)
    assert "build_single_split_loader" not in source
    assert 'loaders["test"]' not in source
    assert fair.TEST_ISOLATION == {
        "test_loader_constructed": False,
        "test_features_read": False,
        "test_labels_read": False,
        "test_predictions_read": False,
        "test_evaluation_performed": False,
        "locked_test_access_count": 0,
    }


def test_stage18a_methods_are_frozen():
    assert fair.ALLOWED_STAGE18A_METHODS == ("moddrop", "cfcompat")


def test_state_hash_is_order_stable():
    import torch

    first = {"b": torch.tensor([2.0]), "a": torch.tensor([1.0])}
    second = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
    assert fair.state_dict_sha(first) == fair.state_dict_sha(second)
