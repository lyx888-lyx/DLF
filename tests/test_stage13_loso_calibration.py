from trains.singleTask.cross_seed_median_residual import minimum_group_count


def test_loso_uses_four_training_seeds():
    seeds = {1111, 1112, 1113, 1114, 1115}
    for held in seeds:
        training = seeds - {held}
        assert len(training) == 4
        assert held not in training


def test_loso_n_min_does_not_depend_on_valid():
    assert minimum_group_count(1284) == 8
