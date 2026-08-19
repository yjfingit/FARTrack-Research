"""Entropy-and-age conditioned one-frame FARTrack template reader.

EACR never changes the current-frame localization or template write path.  A
high-entropy, old-memory event after frame t only changes the five templates
read at frame t+1, then the inherited exponential reader restores itself.
"""

import math

import torch

from lib.test.tracker.fartrack_sparse import FARTrackSparse


class EACRController:
    """Small state holder kept separate to make trigger/reset behavior testable."""

    def __init__(self, entropy_threshold, min_mean_age, enabled=True):
        self.entropy_threshold = float(entropy_threshold)
        self.min_mean_age = float(min_mean_age)
        self.enabled = bool(enabled)
        self.next_alarm = None
        self.last_read_indices = None

    def set_alarm(self, entropy, mean_age):
        if not self.enabled:
            self.next_alarm = torch.zeros((), dtype=torch.bool, device=entropy.device)
            return
        age_ok = torch.tensor(mean_age >= self.min_mean_age, dtype=torch.bool, device=entropy.device)
        self.next_alarm = (entropy >= self.entropy_threshold) & age_ok

    def consume_alarm(self, device):
        if self.next_alarm is None:
            return torch.zeros((), dtype=torch.bool, device=device)
        alarm = self.next_alarm
        self.next_alarm = torch.zeros((), dtype=torch.bool, device=device)
        return alarm


class FARTrackSparseEACR(FARTrackSparse):
    """One-frame recent-template reader controlled by frozen localization entropy."""

    def __init__(self, params, dataset_name):
        super().__init__(params, dataset_name)
        self.eacr = EACRController(
            params.eacr_entropy_threshold,
            params.eacr_min_mean_age,
            params.eacr_enabled,
        )

    def _default_exponential_indices(self):
        """The exact index formula used by the inherited exponential sampler."""
        current_frame_count = self.frame_id + 1
        if current_frame_count < self.num_template:
            return None
        indices = [0]
        for i in range(1, self.num_template - 1):
            indices.append(int((current_frame_count - 1) * (1 - 0.7 ** i)))
        indices.append(current_frame_count - 1)
        return indices

    def _mean_default_template_age(self):
        indices = self._default_exponential_indices()
        if indices is None:
            return 0.0
        return sum(self.frame_id - index for index in indices) / len(indices)

    def _observe_coordinate_distribution(self, probabilities):
        # probabilities is the released predictor's softmax result.  All work
        # remains on-device; the scalar is only consumed through torch.where
        # on the following frame, so there is no per-frame GPU synchronization.
        if not self.eacr.enabled:
            return
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        entropy = (entropy / math.log(probabilities.shape[-1])).mean()
        self.eacr.set_alarm(entropy, self._mean_default_template_age())

    def _prepare_template_read(self):
        if not self.eacr.enabled or not hasattr(self, "stored_templates") or len(self.stored_templates) < 5:
            return
        gate = self.eacr.consume_alarm(self.z_dict1[0].device)
        last = len(self.stored_templates) - 1
        read_indices = [0, last - 3, last - 2, last - 1, last]
        default_templates = list(self.z_dict1)
        default_mask = self.mask
        alternate_templates = [self.stored_templates[index] for index in read_indices]

        # Select on GPU so an inactive alarm is an exact template/mask pass
        # through, without materializing a Python bool from a CUDA tensor.
        self.z_dict1 = [
            torch.where(gate, alternate, default)
            for alternate, default in zip(alternate_templates, default_templates)
        ]
        alternate_mask = torch.ones_like(default_mask)
        for position, index in enumerate(read_indices):
            if index < len(self.store_mask):
                alternate_mask[:, :, 49 * position:49 * (position + 1)] = self.store_mask[index].to(
                    device=alternate_mask.device
                ).unsqueeze(1)
        self.mask = torch.where(gate.to(device=default_mask.device), alternate_mask, default_mask)
        self.eacr.last_read_indices = read_indices


def get_tracker_class():
    return FARTrackSparseEACR
