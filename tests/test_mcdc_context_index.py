import pandas as pd
import pytest

from trains.singleTask.causal_context_index import (
    CONTEXT_LENGTH,
    build_context_index,
    canonical_context_sha,
    parse_mosi_sample_id,
    parse_split,
    train_inner_video_split,
    validate_context_index,
)


def frame(split="train"):
    return pd.DataFrame(
        {
            "sample_index": [3, 0, 2, 1, 4],
            "sample_id": [
                "videoA$_$4",
                "videoA$_$1",
                "videoA$_$3",
                "videoA$_$2",
                "videoB$_$1",
            ],
        }
    )


def test_observed_id_parser():
    parsed = parse_mosi_sample_id("03bSnISJMiM$_$11")
    assert parsed.video_id == "03bSnISJMiM"
    assert parsed.segment_index == 11


@pytest.mark.parametrize("value", ["bad", "$_$1", "video$_$x", "video$_$-1"])
def test_bad_ids_are_rejected(value):
    with pytest.raises(ValueError):
        parse_mosi_sample_id(value)


def test_duplicate_video_segment_is_rejected():
    duplicate = pd.DataFrame(
        {"sample_index": [0, 1], "sample_id": ["video$_$1", "video$_$1"]}
    )
    with pytest.raises(ValueError):
        parse_split(duplicate, "train")


def test_context_is_past_same_video_left_padded_and_k3():
    parsed = parse_split(frame(), "train")
    context = build_context_index(parsed)
    first = context.loc[context.sample_id == "videoA$_$1"].iloc[0]
    fourth = context.loc[context.sample_id == "videoA$_$4"].iloc[0]
    other = context.loc[context.sample_id == "videoB$_$1"].iloc[0]
    assert CONTEXT_LENGTH == 3
    assert first.context_mask == "000" and first.context_length == 0
    assert other.context_mask == "000" and other.context_length == 0
    assert list(
        fourth[
            ["context_sample_id_1", "context_sample_id_2", "context_sample_id_3"]
        ]
    ) == ["videoA$_$1", "videoA$_$2", "videoA$_$3"]
    assert fourth.context_mask == "111"
    assert sum(validate_context_index(context, parsed).values()) == 0


def test_non_frozen_k_is_rejected():
    with pytest.raises(ValueError):
        build_context_index(parse_split(frame(), "train"), k=2)


def test_context_index_sha_is_order_invariant_and_deterministic():
    parsed = parse_split(frame(), "train")
    first = build_context_index(parsed)
    second = build_context_index(parsed.sample(frac=1, random_state=7))
    assert canonical_context_sha(first) == canonical_context_sha(second)


def test_train_valid_video_isolation_can_be_enforced():
    train = parse_split(frame(), "train")
    valid = parse_split(
        pd.DataFrame({"sample_index": [0], "sample_id": ["videoA$_$8"]}), "valid"
    )
    assert set(train.video_id) & set(valid.video_id) == {"videoA"}


def test_hash_inner_split_is_deterministic_and_video_grouped():
    ids = ["v3", "v1", "v2", "v1"]
    assert train_inner_video_split(ids) == train_inner_video_split(reversed(ids))
    assert set(train_inner_video_split(ids)) == {"v1", "v2", "v3"}
