"""Pure tensor primitives for posterior-shape adaptive projection (PSAP)."""

import torch


def normalized_mean_mode_disagreement(mode, mean, coordinate_range):
    """Return mean per-coordinate mean--mode distance normalized by support.

    ``mode`` is FARTrack's autoregressive discrete coordinate branch and
    ``mean`` is the expectation of its already-computed bin posterior.  The
    result remains a scalar CUDA tensor and deliberately has no host read.
    """
    if coordinate_range <= 0:
        raise ValueError("coordinate_range must be positive")
    return (mean - mode).abs().mean() / float(coordinate_range)


def project_posterior(mode, mean, alpha):
    """Convex posterior projection; alpha=0/0.5/1 gives mode/base/mean."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    # Preserve the released arithmetic for the baseline projection exactly.
    if alpha == 0.5:
        return (mode + mean) / 2
    if alpha == 0.0:
        return mode
    if alpha == 1.0:
        return mean
    return mode * (1.0 - alpha) + mean * alpha


def adaptive_projection(mode, mean, disagreement, threshold, high_alpha,
                        low_alpha=0.5):
    """Select an alpha on-device. Strict thresholding makes q-gates precise."""
    high = project_posterior(mode, mean, high_alpha)
    low = project_posterior(mode, mean, low_alpha)
    return torch.where(disagreement > float(threshold), high, low)
