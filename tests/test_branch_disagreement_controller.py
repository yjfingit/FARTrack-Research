import numpy as np

from lib.test.tracker.branch_disagreement_controller import (
    branch_l1_disagreement,
    constant_velocity_prior,
    mix_xywh,
    should_mix,
)


def test_branch_disagreement_uses_only_four_normalized_coordinates():
    value = branch_l1_disagreement([0.0, 0.2, 0.4, 0.6], [0.1, 0.0, 0.3, 0.9])
    assert value == 0.17500000000000004


def test_threshold_boundary_does_not_mix():
    threshold = 0.17500000000000004
    value = branch_l1_disagreement([0.0, 0.2, 0.4, 0.6], [0.1, 0.0, 0.3, 0.9])
    assert not should_mix(value, threshold)
    assert should_mix(value, threshold - 1e-6)


def test_constant_velocity_prior_is_causal_and_mix_preserves_alpha_zero():
    prior = constant_velocity_prior([11, 18, 22, 31], [8, 14, 20, 30])
    np.testing.assert_allclose(prior, [14, 22, 24, 32])
    np.testing.assert_allclose(mix_xywh([10, 10, 20, 20], prior, 0.0), [10, 10, 20, 20])
    np.testing.assert_allclose(mix_xywh([10, 10, 20, 20], prior, 0.25), [11, 13, 21, 23])
