"""One-frame-delayed, branch-disagreement-controlled search expansion.

The action is deliberately a sensing action only.  An alarm from frame ``t``
changes the crop factor used to obtain the fixed-size input on frame ``t+1``;
it never edits frame ``t``'s prediction, state, templates, or mask.
"""

import torch

from lib.test.tracker.fartrack_sparse import FARTrackSparse


class DelayedSearchController:
    """Single-use next-frame action state, kept separate for unit testing."""

    def __init__(self, normal_factor, expansion_factor, threshold, enabled=True):
        self.normal_factor = float(normal_factor)
        self.expansion_factor = float(expansion_factor)
        self.threshold = float(threshold)
        self.enabled = bool(enabled)
        self._expand_next = False

    def factor_for_current_frame(self):
        """Consume a pending alarm so expansion cannot persist beyond one frame."""
        expand = self.enabled and self._expand_next
        self._expand_next = False
        return self.normal_factor * self.expansion_factor if expand else self.normal_factor

    def observe_disagreement(self, disagreement):
        """Set only the next-frame action after a completed current forward pass."""
        self._expand_next = self.enabled and float(disagreement) >= self.threshold


class FARTrackSparseDelayedSearch(FARTrackSparse):
    def __init__(self, params, dataset_name):
        super().__init__(params, dataset_name)
        self._delayed_search = DelayedSearchController(
            normal_factor=params.search_factor,
            expansion_factor=params.delayed_search_expansion_factor,
            threshold=params.delayed_search_disagreement_threshold,
            enabled=params.delayed_search_enabled,
        )

    def initialize(self, image, info: dict, name: str):
        result = super().initialize(image, info, name)
        # Explicitly reconstruct per sequence: no alarm may cross sequences.
        self._delayed_search = DelayedSearchController(
            normal_factor=self.params.search_factor,
            expansion_factor=self.params.delayed_search_expansion_factor,
            threshold=self.params.delayed_search_disagreement_threshold,
            enabled=self.params.delayed_search_enabled,
        )
        return result

    def _select_search_factor(self):
        return self._delayed_search.factor_for_current_frame()

    def _after_tracking_forward(self, out_dict):
        if not self.params.delayed_search_enabled:
            return
        logits = out_dict["feat"][0:4, :, 0:self.bins * self.range]
        probabilities = logits.softmax(dim=-1)
        seq_branch = (out_dict["seqs"][:, 0:4] + 0.5) / (self.bins - 1) - 0.5
        coordinates = torch.arange(
            self.bins * self.range, device=logits.device, dtype=logits.dtype
        ) * (2.0 / (self.bins * self.range)) + (-self.range * 0.5 + 0.5)
        feat_branch = (probabilities * coordinates).sum(dim=-1).permute(1, 0)
        disagreement = (seq_branch - feat_branch).abs().mean()
        self._delayed_search.observe_disagreement(disagreement.item())


def get_tracker_class():
    return FARTrackSparseDelayedSearch
