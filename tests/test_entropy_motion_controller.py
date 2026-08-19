import numpy as np
import torch

from lib.test.tracker.entropy_motion_controller import (
    constant_velocity_prior,
    mix_xywh,
    normalized_probability_entropy,
)


def test_constant_velocity_prior_uses_only_previous_states():
    prior = constant_velocity_prior([11, 18, 22, 31], [8, 14, 20, 30])
    np.testing.assert_allclose(prior, [14, 22, 24, 32])


def test_mix_is_disabled_by_zero_alpha_and_active_for_high_weight():
    model = [10, 10, 20, 20]
    motion = [14, 6, 24, 16]
    np.testing.assert_allclose(mix_xywh(model, motion, 0.0), model)
    np.testing.assert_allclose(mix_xywh(model, motion, 0.5), [12, 8, 22, 18])


def test_entropy_orders_peaked_below_uniform_distributions():
    peaked = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]).repeat(4, 1, 1)
    uniform = torch.full_like(peaked, 0.25)
    low = normalized_probability_entropy(peaked)
    high = normalized_probability_entropy(uniform)
    assert float(low) < 0.01
    assert float(high) > 0.99
