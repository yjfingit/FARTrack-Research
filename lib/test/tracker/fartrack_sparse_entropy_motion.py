"""Entropy-conditioned causal motion mixing for frozen FARTrackSparse.

This tracker leaves the network, template update, and low-entropy tracking
path unchanged.  A one-step constant-velocity state prior is mixed into the
current model box only when the frozen model's coordinate-bin entropy exceeds
the predeclared threshold.  It uses one existing forward per frame.
"""

import numpy as np
import torch

from lib.test.tracker.entropy_motion_controller import (
    constant_velocity_prior,
    normalized_probability_entropy,
)
from lib.test.tracker.fartrack_sparse import FARTrackSparse


class FARTrackSparseEntropyMotion(FARTrackSparse):
    def __init__(self, params, dataset_name):
        super().__init__(params, dataset_name)
        self.entropy_motion_enabled = bool(getattr(params, "entropy_motion_enabled", True))
        self.entropy_motion_threshold = float(params.entropy_motion_threshold)
        self.entropy_motion_alpha = float(params.entropy_motion_alpha)
        if not 0.0 <= self.entropy_motion_alpha <= 1.0:
            raise ValueError("entropy_motion_alpha must be in [0, 1]")
        self._pending_entropy = None
        self._motion_previous_state = None
        self._motion_last_state = None

    def initialize(self, image, info: dict, name: str):
        result = super().initialize(image, info, name)
        initial = np.asarray(info["init_bbox"], dtype=np.float64).copy()
        self._motion_previous_state = initial.copy()
        self._motion_last_state = initial.copy()
        return result

    def _observe_coordinate_probabilities(self, probabilities):
        """Receive the released path's already-computed coordinate softmax."""
        self._pending_entropy = normalized_probability_entropy(probabilities)

    def _adjust_predicted_box(self, pred_new, resize_factor: float):
        """Mix on GPU, avoiding a second synchronization before ``tolist``.

        ``pred_new`` is the released normalized search-crop cxcywh box.  The
        causal prior is converted from image xywh into that same coordinate
        system.  The affine conversion makes mixing exactly equivalent to
        mixing the two boxes after mapping to image coordinates.
        """
        if self._pending_entropy is None:
            return pred_new
        motion_box = constant_velocity_prior(
            self._motion_last_state, self._motion_previous_state
        )
        prev_center = self._motion_last_state[:2] + 0.5 * self._motion_last_state[2:]
        half_side = 0.5 * self.params.search_size / resize_factor
        motion_center = motion_box[:2] + 0.5 * motion_box[2:]
        motion_cxcywh = np.concatenate((
            (motion_center - (prev_center - half_side)) * resize_factor / self.params.search_size,
            motion_box[2:] * resize_factor / self.params.search_size,
        ))
        prior = pred_new.new_tensor(motion_cxcywh)
        mixed = (1.0 - self.entropy_motion_alpha) * pred_new + self.entropy_motion_alpha * prior
        return torch.where(self._pending_entropy > self.entropy_motion_threshold, mixed, pred_new)

    def track(self, image, info: dict = None):
        # Exact base delegation is a reproducibility requirement, not merely a
        # zero-alpha approximation.
        if not self.entropy_motion_enabled:
            return super().track(image, info)

        self._motion_last_state = np.asarray(self.state, dtype=np.float64).copy()
        self._pending_entropy = None
        result = super().track(image, info)

        # The next frame's prior uses only states accepted at earlier frames.
        self._motion_previous_state = self._motion_last_state.copy()
        return result


def get_tracker_class():
    return FARTrackSparseEntropyMotion
