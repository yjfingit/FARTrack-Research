"""Training-free appearance-diverse template reservoir for FARTrackSparse."""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import cv2
import torch
import torch.nn.functional as F


@dataclass
class TemplateRecord:
    template: torch.Tensor
    token_mask: torch.Tensor
    descriptor: torch.Tensor
    frame_id: int


class AppearanceDiverseReservoir:
    """A constant-size every-frame template store with chronological rebinding.

    The original anchor is immutable.  All later crops are accepted; when the
    capacity is reached the record nearest in untrained RGB descriptor space to
    another historical record is replaced.  Binding always retains anchor and
    newest, then uses the three other cached historical records in temporal
    order. Dynamic capacity is exactly four: the
    active FARTrack set is anchor + three historical views + newest.
    """

    def __init__(self, num_templates: int = 5, capacity: int = 4, token_count: int = 49):
        if num_templates != 5:
            raise ValueError("FARTrackSparse research configuration uses five template slots.")
        if capacity != num_templates - 1:
            raise ValueError("dynamic capacity must equal the four non-anchor FARTrack slots")
        self.num_templates = num_templates
        self.capacity = capacity
        self.token_count = token_count
        self.reset()

    def reset(self):
        self.anchor = None
        self.records: List[TemplateRecord] = []
        self.newest = None
        self.writes = 0
        self.replacements = 0

    @staticmethod
    def descriptor(template: torch.Tensor) -> torch.Tensor:
        """Fixed RGB 4x4 pooled descriptor; no learned feature or model pass.

        Selection deliberately lives on CPU after one tiny 48-value transfer.
        Keeping dozens of tiny distance kernels on the tracking GPU caused
        synchronization overhead larger than the intended controller budget.
        """
        rgb_grid = F.adaptive_avg_pool2d(template.detach().float(), output_size=(4, 4)).flatten(1).cpu()
        return F.normalize(rgb_grid, dim=1).squeeze(0)

    @staticmethod
    def descriptor_from_rgb(rgb_crop: np.ndarray) -> torch.Tensor:
        """Vectorized fixed 4x4 RGB descriptor from FARTrack's CPU crop.

        `sample_target` already creates this 112x112 RGB crop every frame.
        Reusing it avoids adding a GPU pooling kernel and a synchronizing
        device-to-host transfer to the tracker critical path.
        """
        height, width, channels = rgb_crop.shape
        if channels != 3 or height % 4 or width % 4:
            raise ValueError("expected an RGB crop with dimensions divisible by four")
        grid = cv2.resize(rgb_crop, (4, 4), interpolation=cv2.INTER_AREA)
        descriptor = torch.from_numpy(grid.astype(np.float32, copy=False).transpose(2, 0, 1).reshape(-1).copy())
        return F.normalize(descriptor, dim=0)

    def initialize(self, anchor_template: torch.Tensor, descriptor: torch.Tensor = None):
        dense = torch.ones((1, self.token_count), dtype=torch.bool, device=anchor_template.device)
        self.anchor = TemplateRecord(anchor_template.detach(), dense,
                                     self.descriptor(anchor_template) if descriptor is None else descriptor, frame_id=0)
        self.records = []
        self.newest = self.anchor
        self.writes = 0
        self.replacements = 0

    @staticmethod
    def _closest_redundant_index(records: Sequence[TemplateRecord]) -> int:
        descriptors = torch.stack([item.descriptor for item in records])
        distances = torch.cdist(descriptors, descriptors, p=2)
        distances.fill_diagonal_(float("inf"))
        # Prefer discarding the newer member of the closest pair only on ties;
        # older views provide a longer temporal support horizon.
        nearest_distance, _ = distances.min(dim=1)
        return min(range(len(records)), key=lambda i: (float(nearest_distance[i]), -records[i].frame_id))

    def write(self, template: torch.Tensor, token_mask: torch.Tensor, frame_id: int,
              descriptor: torch.Tensor = None):
        if self.anchor is None:
            raise RuntimeError("initialize must be called before write")
        record = TemplateRecord(template.detach(), token_mask.detach().bool().reshape(1, self.token_count),
                                self.descriptor(template) if descriptor is None else descriptor, int(frame_id))
        self.writes += 1
        self.newest = record
        if len(self.records) < self.capacity:
            self.records.append(record)
            return
        # The current observation is unconditionally admitted. Replace one old
        # record using one tiny 5x5 vectorized descriptor matrix; binding is O(1).
        candidates = self.records + [record]
        descriptors = torch.stack([item.descriptor for item in candidates])
        distances = torch.cdist(descriptors, descriptors, p=2)
        distances.fill_diagonal_(float("inf"))
        old_nearest = distances[:-1].min(dim=1).values
        # Retain a broad chronological range on ties; never evict newest.
        ages = torch.tensor([-item.frame_id for item in self.records], dtype=old_nearest.dtype)
        evict = int(torch.argmin(old_nearest + ages * 1e-8).item())
        self.records[evict] = record
        self.replacements += 1

    @staticmethod
    def _max_min_indices(records: Sequence[TemplateRecord], anchors: Sequence[TemplateRecord], count: int) -> List[int]:
        available = list(range(len(records)))
        selected: List[int] = []
        support = [entry.descriptor for entry in anchors]
        while available and len(selected) < count:
            candidate_descriptors = torch.stack([records[index].descriptor for index in available])
            support_descriptors = torch.stack(support)
            min_distances = torch.cdist(candidate_descriptors, support_descriptors, p=2).min(dim=1).values
            # A single deterministic tie break prefers the earliest view;
            # avoid per-candidate scalar extraction, which would synchronize
            # the GPU hundreds of times on a long sequence.
            earliest = torch.tensor([-records[index].frame_id for index in available], dtype=min_distances.dtype)
            chosen_position = int(torch.argmax(min_distances + earliest * 1e-8).item())
            chosen = available[chosen_position]
            selected.append(chosen)
            support.append(records[chosen].descriptor)
            available.remove(chosen)
        return selected

    def bind(self) -> Tuple[List[torch.Tensor], torch.Tensor, List[int]]:
        """Return FARTrack's five active templates, compatibility mask, and frame ids."""
        if self.anchor is None:
            raise RuntimeError("initialize must be called before bind")
        # Reservoir replacement already enforces diversity. Binding only
        # reorders the four cached records by time, with no descriptor work.
        historical = [record for record in self.records if record.frame_id != self.newest.frame_id]
        middle = sorted(historical, key=lambda item: item.frame_id)
        active = [self.anchor] + middle + [self.newest]
        # Early frames retain the baseline's repeated-template behavior while
        # exposing the same fixed five-slot shape to the frozen transformer.
        while len(active) < self.num_templates:
            active.insert(-1, self.anchor)
        active = active[:self.num_templates - 1] + [self.newest]
        valid = torch.cat([entry.token_mask for entry in active] + [
            torch.ones((1, 200), dtype=torch.bool, device=self.anchor.template.device)
        ], dim=1)
        mask = valid.unsqueeze(-1).expand(-1, -1, valid.shape[1]).permute(0, 2, 1)
        return [entry.template for entry in active], mask, [entry.frame_id for entry in active]
