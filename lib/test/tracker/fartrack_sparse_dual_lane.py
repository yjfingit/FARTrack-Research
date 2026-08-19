"""Entropy-conditioned dual-lane memory for frozen FARTrackSparse.

The newest localization crop is always available at the next frame.  Entropy
only controls admission into the three older, stable slots, so this is not a
binary template-write hold.
"""

import math

import torch

from lib.test.tracker.fartrack_sparse import FARTrackSparse


class FARTrackSparseDualLane(FARTrackSparse):
    def __init__(self, params, dataset_name):
        super().__init__(params, dataset_name)
        self.dual_lane_enabled = bool(getattr(params, "dual_lane_enabled", True))
        self.stable_entropy_threshold = float(params.stable_entropy_threshold)

    def initialize(self, image, info: dict, name: str):
        result = super().initialize(image, info, name)
        self.stable_templates = []
        self.stable_masks = []
        return result

    def _coordinate_entropy(self, out_dict):
        logits = out_dict["feat"][0:4, :, 0:self.bins * self.range]
        probabilities = logits.softmax(dim=-1)
        log_bins = math.log(float(logits.shape[-1]))
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        return float((entropy / log_bins).mean().item())

    @staticmethod
    def _mask_blocks(mask):
        return [mask[index].detach().clone() for index in range(4)]

    def _assemble_dual_lane(self, newest_template, newest_mask):
        """Build [anchor, three risk-certified stable slots, newest]."""
        anchor = self.z_dict1[0]
        anchor_mask = torch.ones([1, 49], dtype=torch.bool, device=anchor.device)
        stable_count = len(self.stable_templates)

        # Spread stable evidence over accepted history.  During warm-up, the
        # anchor fills unavailable stable positions without admitting risk.
        if stable_count:
            stable_indices = [
                int(round(index * (stable_count - 1) / 2.0)) for index in range(3)
            ]
        else:
            stable_indices = []

        templates = [anchor]
        masks = [anchor_mask]
        for slot in range(3):
            if stable_indices:
                accepted_index = stable_indices[slot]
                templates.append(self.stable_templates[accepted_index])
                masks.append(self.stable_masks[accepted_index])
            else:
                templates.append(anchor)
                masks.append(anchor_mask)
        templates.append(newest_template)
        masks.append(newest_mask)

        self.z_dict1 = templates
        mask_temp = torch.ones([1, 445], dtype=torch.bool, device=anchor.device)
        for slot, block in enumerate(masks):
            mask_temp[:, 49 * slot:49 * (slot + 1)] = block
        self.mask = mask_temp.unsqueeze(-1).expand(-1, -1, 445).permute(0, 2, 1)

    def _dual_lane_update(self, new_z, mask, entropy):
        newest_mask = self._mask_blocks(mask)[0]
        if entropy <= self.stable_entropy_threshold:
            self.stable_templates.append(new_z)
            self.stable_masks.append(newest_mask)
        self._assemble_dual_lane(new_z, newest_mask)

    def track(self, image, info: dict = None):
        """Inline the released path only at its post-forward update junction."""
        # The base method does not expose its out_dict.  The one-frame hook
        # captures entropy while preserving every localization operation.
        self._pending_entropy = None
        original_forward = self.network.forward

        def observing_forward(*args, **kwargs):
            out_dict = original_forward(*args, **kwargs)
            self._pending_entropy = self._coordinate_entropy(out_dict)
            return out_dict

        self.network.forward = observing_forward
        try:
            if not self.dual_lane_enabled:
                return super().track(image, info)

            original_update = self.template_update_sampling

            def dual_lane_update(new_z, sampling_method="exponential", mask=None):
                self._dual_lane_update(new_z, mask, self._pending_entropy)

            self.template_update_sampling = dual_lane_update
            try:
                return super().track(image, info)
            finally:
                self.template_update_sampling = original_update
        finally:
            self.network.forward = original_forward


def get_tracker_class():
    return FARTrackSparseDualLane
