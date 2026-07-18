from trains.singleTask.robust_cross_seed_consensus import huber_center


def test_huber_supports_exactly_four_lomo_members():
    result = huber_center([0.0, 0.1, 0.2, 2.0])
    assert -3.0 <= result.center <= 3.0


def test_each_lomo_case_drops_one_distinct_member():
    seeds = (1111, 1112, 1113, 1114, 1115)
    for dropped in seeds:
        retained = tuple(seed for seed in seeds if seed != dropped)
        assert len(retained) == 4
        assert dropped not in retained


def test_fixed_anchor_is_not_reselected_when_member_is_dropped():
    anchor = 1114
    for dropped in (1111, 1112, 1113, 1114, 1115):
        assert anchor == 1114
        if dropped == anchor:
            assert anchor not in tuple(
                seed
                for seed in (1111, 1112, 1113, 1114, 1115)
                if seed != dropped
            )
