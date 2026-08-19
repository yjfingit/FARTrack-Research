import torch

from lib.test.tracker.posterior_shape_controller import (
    adaptive_projection,
    normalized_mean_mode_disagreement,
    project_posterior,
)


def test_projection_endpoints_and_baseline_midpoint():
    mode = torch.tensor([[0.0, 0.2, 0.4, 0.6]])
    mean = torch.tensor([[0.2, 0.4, 0.6, 0.8]])
    assert torch.equal(project_posterior(mode, mean, 0.0), mode)
    assert torch.equal(project_posterior(mode, mean, 1.0), mean)
    assert torch.equal(project_posterior(mode, mean, 0.5), (mode + mean) / 2)


def test_normalized_disagreement_and_strict_gate():
    mode = torch.zeros(1, 4)
    mean = torch.full((1, 4), 0.2)
    disagreement = normalized_mean_mode_disagreement(mode, mean, 2.0)
    assert torch.isclose(disagreement, torch.tensor(0.1))
    assert torch.equal(adaptive_projection(mode, mean, disagreement, 0.1, 1.0), (mode + mean) / 2)
    assert torch.equal(adaptive_projection(mode, mean, disagreement, 0.09, 1.0), mean)
