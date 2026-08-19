"""CUDA-only selection primitives for disagreement-conditioned template quality."""

import torch


PRUNE_LEVELS = (25, 50, 75, 90)


def validate_mask_levels(masks):
    """Check the four native masks are nested 49-token boolean blocks."""
    if len(masks) != 4:
        raise ValueError("FARTrackSparse must provide four pruning masks")
    reference_shape = masks[0].shape
    if reference_shape[-1] != 49:
        raise ValueError("template mask must contain one 7x7 token block")
    for mask in masks:
        if mask.dtype != torch.bool or mask.shape != reference_shape:
            raise ValueError("all pruning masks must be same-shape boolean tensors")
    for looser, stronger in zip(masks, masks[1:]):
        if torch.any(stronger & ~looser):
            raise ValueError("stronger pruning must be a subset of weaker retention")


def select_mask(masks, sequence_branch, feature_branch, threshold, level,
                enabled=True, static=False):
    """Return a selected native mask and an on-device trigger tensor.

    No posterior is recomputed and no scalar crosses the CUDA boundary.  In
    adaptive mode, high branch disagreement suppresses more low-attention
    tokens in the just-written template; otherwise exact 25% retention is
    used.  Static mode is a diagnostic with the same storage semantics.
    """
    if not 0 <= int(level) < len(PRUNE_LEVELS):
        raise ValueError("level must index one of the four native masks")
    if not enabled or int(level) == 0:
        return masks[0], torch.zeros((), dtype=torch.bool, device=masks[0].device)
    if static:
        return masks[int(level)], torch.ones((), dtype=torch.bool, device=masks[0].device)
    disagreement = (sequence_branch - feature_branch).abs().mean()
    trigger = disagreement > float(threshold)
    return torch.where(trigger, masks[int(level)], masks[0]), trigger
