"""Branch-disagreement-gated causal box mixing for frozen FARTrackSparse.

The model forward and template update are inherited unchanged.  When the two
native coordinate branches disagree, the final box is cautiously mixed with a
one-step constant-velocity prior computed only from prior accepted states.
"""

import numpy as np
import torch

from lib.test.tracker.branch_disagreement_controller import (
    branch_l1_disagreement,
    constant_velocity_prior,
    mix_xywh,
    should_mix,
)
from lib.test.tracker.fartrack_sparse import FARTrackSparse
from lib.utils.box_ops import clip_box


class FARTrackSparseBranchMotion(FARTrackSparse):
    def __init__(self, params, dataset_name):
        super().__init__(params, dataset_name)
        self.branch_motion_enabled = bool(getattr(params, "branch_motion_enabled", True))
        self.branch_motion_threshold = float(params.branch_motion_threshold)
        self.branch_motion_alpha = float(params.branch_motion_alpha)
        if not 0.0 <= self.branch_motion_alpha <= 1.0:
            raise ValueError("branch_motion_alpha must be in [0, 1]")
        self._motion_previous_state = None
        self._motion_last_state = None
        self.last_branch_disagreement = None
        self.last_branch_mixed = False

    def initialize(self, image, info: dict, name: str):
        result = super().initialize(image, info, name)
        initial = np.asarray(info["init_bbox"], dtype=np.float64).copy()
        self._motion_previous_state = initial.copy()
        self._motion_last_state = initial.copy()
        return result

    def track(self, image, info: dict = None):
        # Exact delegation is required for disabled byte-parity.
        if not self.branch_motion_enabled:
            return super().track(image, info)
        self._motion_last_state = np.asarray(self.state, dtype=np.float64).copy()
        self.last_branch_disagreement = None
        self.last_branch_mixed = False
        result = super().track(image, info)
        self._motion_previous_state = self._motion_last_state.copy()
        return result

    def _materialize_predicted_box(self, pred_new, seq_branch, feat_branch, resize_factor):
        """Use the released path's sole host transfer for all needed values.

        Concatenating the 4-D model crop box and the two 4-D native branch
        estimates produces one ``tolist`` call, rather than a separate scalar
        ``item`` synchronization for the disagreement gate.
        """
        if not self.branch_motion_enabled:
            # This is intentionally the released source expression.  The base
            # class sees the method on the subclass even when track delegates,
            # so this guard preserves disabled output byte-for-byte.
            return (pred_new * self.params.search_size / resize_factor).tolist()
        model_crop = pred_new * self.params.search_size / resize_factor
        sequence_mean = seq_branch.view(-1, 4).mean(dim=0)
        feature_mean = feat_branch.view(-1, 4).mean(dim=0)
        payload = torch.cat((model_crop, sequence_mean, feature_mean)).tolist()
        model_crop, sequence_box, feature_box = payload[:4], payload[4:8], payload[8:12]
        disagreement = branch_l1_disagreement(sequence_box, feature_box)
        self.last_branch_disagreement = disagreement
        if not should_mix(disagreement, self.branch_motion_threshold):
            return model_crop

        # map_box_back uses the previous accepted self.state, which is still
        # intact at this point.  The prior never observes the current model box.
        model_image = np.asarray(self.map_box_back(model_crop, resize_factor), dtype=np.float64)
        prior = constant_velocity_prior(self._motion_last_state, self._motion_previous_state)
        mixed_image = mix_xywh(model_image, prior, self.branch_motion_alpha)
        self.last_branch_mixed = True

        # Return an equivalent search-crop cxcywh box to the inherited released
        # path, which maps and clips it in the usual location.
        previous = self._motion_last_state
        previous_center = previous[:2] + 0.5 * previous[2:]
        half_side = 0.5 * self.params.search_size / resize_factor
        mixed_center = mixed_image[:2] + 0.5 * mixed_image[2:]
        return [
            mixed_center[0] - (previous_center[0] - half_side),
            mixed_center[1] - (previous_center[1] - half_side),
            mixed_image[2],
            mixed_image[3],
        ]


def get_tracker_class():
    return FARTrackSparseBranchMotion
