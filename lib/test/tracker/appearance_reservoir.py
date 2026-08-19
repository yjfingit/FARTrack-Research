"""Training-free appearance-diverse template reservoir for FARTrackSparse."""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass
class TemplateRecord:
    template: torch.Tensor
    token_mask: torch.Tensor
    descriptor: torch.Tensor
    frame_id: int


class AppearanceDiverseReservoir:
    """A bounded every-frame template store with deterministic max-min rebinding.

    The original anchor is immutable.  All later crops are accepted; when the
    capacity is reached the record nearest in untrained RGB descriptor space to
    another historical record is replaced.  Binding always retains anchor and
    newest, then selects three maximally separated historical records and
    places those slots in temporal order.
    """

    def __init__(self, num_templates: int = 5, capacity: int = 64, token_count: int = 49):
        if num_templates != 5:
            raise ValueError("FARTrackSparse research configuration uses five template slots.")
        if capacity < num_templates:
            raise ValueError("capacity must retain at least one complete active set")
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

    def initialize(self, anchor_template: torch.Tensor):
        dense = torch.ones((1, self.token_count), dtype=torch.bool, device=anchor_template.device)
        self.anchor = TemplateRecord(anchor_template.detach().clone(), dense,
                                     self.descriptor(anchor_template), frame_id=0)
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

    def write(self, template: torch.Tensor, token_mask: torch.Tensor, frame_id: int):
        if self.anchor is None:
            raise RuntimeError("initialize must be called before write")
        record = TemplateRecord(template.detach().clone(), token_mask.detach().bool().reshape(1, self.token_count).clone(),
                                self.descriptor(template), int(frame_id))
        self.writes += 1
        self.newest = record
        if len(self.records) < self.capacity:
            self.records.append(record)
            return
        # The current observation is unconditionally admitted.  Select its
        # eviction target after insertion, never reject the write itself.
        candidates = self.records + [record]
        evict = self._closest_redundant_index(candidates)
        if evict == len(self.records):
            # Preserve current-frame adaptation even if it resembles history.
            evict = self._closest_redundant_index(self.records)
            self.records[evict] = record
        else:
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
        historical = [record for record in self.records if record.frame_id != self.newest.frame_id]
        selected = self._max_min_indices(historical, [self.anchor, self.newest], count=3)
        middle = sorted((historical[index] for index in selected), key=lambda item: item.frame_id)
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
