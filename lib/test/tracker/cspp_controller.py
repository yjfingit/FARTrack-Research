"""CUDA-resident coordinate-selective posterior projection primitives."""

import torch


def normalized_cx_mode_mean_disagreement(mode_corners, mean_corners):
    """Return |cx_mode-cx_mean| normalized by the midpoint predicted width.

    This is exactly the calibration statistic used to freeze node 17's q95.
    Both numerator and denominator have the same crop-to-image scale, so it
    can be evaluated directly in the released normalized crop coordinates.
    """
    mode_cx = (mode_corners[..., 0] + mode_corners[..., 2]) * 0.5
    mean_cx = (mean_corners[..., 0] + mean_corners[..., 2]) * 0.5
    midpoint_width = ((mode_corners[..., 2] + mean_corners[..., 2])
                      - (mode_corners[..., 0] + mean_corners[..., 0])) * 0.5
    # Native corner predictions are ordered. Clamp only protects the
    # degenerate numerical edge case while remaining entirely CUDA-resident.
    midpoint_width = midpoint_width.clamp_min(torch.finfo(midpoint_width.dtype).eps)
    return ((mode_cx - mean_cx).abs() / midpoint_width).mean()


def project_cx_only(midpoint_corners, mean_corners, disagreement, threshold):
    """Replace only ``cx`` with the posterior expectation above a fixed gate.

    The same horizontal offset is added to left and right corners. Therefore
    width, vertical center, and height are bitwise the midpoint values on both
    branches; the no-trigger branch returns the midpoint tensor directly.
    """
    if not torch.is_tensor(disagreement):
        raise TypeError("disagreement must be a tensor to keep gating on-device")
    midpoint_cx = (midpoint_corners[..., 0] + midpoint_corners[..., 2]) * 0.5
    mean_cx = (mean_corners[..., 0] + mean_corners[..., 2]) * 0.5
    dx = mean_cx - midpoint_cx
    zero = torch.zeros_like(dx)
    dx = torch.where(disagreement >= float(threshold), dx, zero)
    output = midpoint_corners.clone()
    output[..., 0] = midpoint_corners[..., 0] + dx.unsqueeze(-1)
    output[..., 2] = midpoint_corners[..., 2] + dx.unsqueeze(-1)
    return output
