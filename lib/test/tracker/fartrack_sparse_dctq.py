"""Disagreement-Conditioned Template Quality for frozen FARTrackSparse."""

from lib.test.tracker.dctq_controller import select_mask, validate_mask_levels
from lib.test.tracker.fartrack_sparse import FARTrackSparse


class FARTrackSparseDCTQ(FARTrackSparse):
    """Select only the native mask attached to the current template write."""

    def __init__(self, params, dataset_name):
        super().__init__(params, dataset_name)
        self.dctq_enabled = bool(params.dctq_enabled)
        self.dctq_threshold = float(params.dctq_threshold)
        self.dctq_level = int(params.dctq_level)
        self.dctq_static = bool(params.dctq_static)
        self._dctq_mask = None
        self.last_dctq_trigger = None

    def _prepare_template_mask(self, masks, sequence_branch, feature_branch):
        # This is called after the existing forward and feature softmax.  The
        # tensor gate remains on CUDA until the normal next-frame model use.
        validate_mask_levels(masks)
        self._dctq_mask, self.last_dctq_trigger = select_mask(
            masks, sequence_branch, feature_branch,
            threshold=self.dctq_threshold,
            level=self.dctq_level,
            enabled=self.dctq_enabled,
            static=self.dctq_static,
        )

    def _select_template_mask(self, masks):
        # Disabled mode intentionally returns the released first native mask.
        if not self.dctq_enabled or self._dctq_mask is None:
            return masks[0]
        return self._dctq_mask


def get_tracker_class():
    return FARTrackSparseDCTQ
