import torch

from lib.test.tracker.cspp_controller import (
    normalized_cx_mode_mean_disagreement,
    project_cx_only,
)


def test_cx_disagreement_uses_midpoints_not_corners():
    mode = torch.tensor([[0.0, 0.1, 0.4, 0.6]])
    mean = torch.tensor([[0.1, 0.8, 0.7, 1.2]])
    # Centers differ by 0.2 and the midpoint predicted width is 0.5.
    assert torch.isclose(normalized_cx_mode_mean_disagreement(mode, mean), torch.tensor(0.4))


def test_cx_only_projection_preserves_midpoint_geometry():
    mode = torch.tensor([[0.0, 0.2, 0.4, 0.8]])
    mean = torch.tensor([[0.2, -0.5, 0.8, 1.6]])
    midpoint = (mode + mean) / 2
    output = project_cx_only(midpoint, mean, torch.tensor(0.2), 0.1)
    assert torch.equal(output[..., 1], midpoint[..., 1])
    assert torch.equal(output[..., 3], midpoint[..., 3])
    assert torch.equal(output[..., 2] - output[..., 0], midpoint[..., 2] - midpoint[..., 0])
    assert torch.equal((output[..., 0] + output[..., 2]) / 2,
                       (mean[..., 0] + mean[..., 2]) / 2)


def test_threshold_is_inclusive_and_no_trigger_is_exact_midpoint():
    midpoint = torch.tensor([[0.0, 0.2, 0.4, 0.8]])
    mean = torch.tensor([[0.2, -0.5, 0.8, 1.6]])
    exact = project_cx_only(midpoint, mean, torch.tensor(0.1), 0.1)
    below = project_cx_only(midpoint, mean, torch.tensor(0.099), 0.1)
    assert torch.equal(below, midpoint)
    assert not torch.equal(exact, midpoint)
