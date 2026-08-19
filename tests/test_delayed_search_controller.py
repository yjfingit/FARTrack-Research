from lib.test.tracker.fartrack_sparse_delayed_search import DelayedSearchController


def test_delayed_alarm_expands_only_the_next_frame_then_resets():
    controller = DelayedSearchController(4.0, 1.2, 0.5)
    assert controller.factor_for_current_frame() == 4.0
    controller.observe_disagreement(0.5)
    assert controller.factor_for_current_frame() == 4.8
    assert controller.factor_for_current_frame() == 4.0


def test_low_alarm_clears_a_pending_action():
    controller = DelayedSearchController(4.0, 1.3, 0.5)
    controller.observe_disagreement(0.9)
    controller.observe_disagreement(0.1)
    assert controller.factor_for_current_frame() == 4.0


def test_disabled_controller_is_exact_normal_factor():
    controller = DelayedSearchController(4.0, 1.3, 0.0, enabled=False)
    controller.observe_disagreement(99.0)
    assert controller.factor_for_current_frame() == 4.0
