from trains.singleTask.specialist_training import specialist_gap


def test_gap_identity():
    g_opt, g_info, g_total = specialist_gap(0.7, 0.6, 0.5)
    assert abs(g_opt - 0.1) < 1e-12
    assert abs(g_info - 0.1) < 1e-12
    assert abs(g_total - 0.2) < 1e-12
    assert abs(g_total - g_opt - g_info) <= 1e-8
