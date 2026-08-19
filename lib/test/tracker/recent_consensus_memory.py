"""Recent-consensus template rebinding for frozen FARTrackSparse.

Every predicted crop is written exactly once into a fixed-length FIFO window.
The model still runs once per video frame; this module only rebinds its five
already-supported template slots after that write.
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass
class RecentTemplate:
    template: torch.Tensor
    token_mask: torch.Tensor
    descriptor: torch.Tensor
    frame_id: int


class RecentConsensusMemory:
    """Immutable anchor plus temporal RGB-medoid summaries of recent writes."""

    def __init__(self, num_templates: int = 5, window_size: int = 30, token_count: int = 49):
        if num_templates != 5:
            raise ValueError("FARTrackSparse research tracker has exactly five template slots")
        if window_size < 4:
            raise ValueError("window_size must support newest plus three temporal medoids")
        self.num_templates = num_templates
        self.window_size = window_size
        self.token_count = token_count
        self.reset()

    def reset(self):
        self.anchor = None
        self.recent: List[RecentTemplate] = []
        self.writes = 0

    @staticmethod
    def descriptor(template: torch.Tensor) -> torch.Tensor:
        """A fixed, no-forward-pass RGB descriptor used only for rebinding."""
        pooled = F.adaptive_avg_pool2d(template.detach().float(), (4, 4)).flatten(1)
        return F.normalize(pooled, dim=1).squeeze(0)

    def initialize(self, anchor_template: torch.Tensor):
        dense = torch.ones((1, self.token_count), dtype=torch.bool, device=anchor_template.device)
        self.anchor = RecentTemplate(anchor_template.detach().clone(), dense,
                                     self.descriptor(anchor_template), frame_id=0)
        self.recent = []
        self.writes = 0
        return self.bind()

    def write(self, template: torch.Tensor, token_mask: torch.Tensor, frame_id: int):
        """Store every candidate exactly once; eviction is FIFO only."""
        if self.anchor is None:
            raise RuntimeError("initialize must be called before write")
        record = RecentTemplate(template.detach().clone(),
                                token_mask.detach().bool().reshape(1, self.token_count).clone(),
                                self.descriptor(template), int(frame_id))
        self.recent.append(record)
        if len(self.recent) > self.window_size:
            self.recent.pop(0)
        self.writes += 1

    @staticmethod
    def _medoid(records: Sequence[RecentTemplate]) -> RecentTemplate:
        if not records:
            raise ValueError("medoid needs at least one record")
        if len(records) == 1:
            return records[0]
        descriptors = torch.stack([record.descriptor for record in records])
        total_distances = torch.cdist(descriptors, descriptors, p=2).sum(dim=1)
        # Older is deterministic tie-breaker, preserving chronological support.
        index = min(range(len(records)), key=lambda i: (float(total_distances[i]), records[i].frame_id))
        return records[index]

    def _chronological_medoids(self) -> List[RecentTemplate]:
        if not self.recent:
            return []
        # Newest is explicitly bound in the final slot. Exclude it here so the
        # other three slots summarize preceding evidence rather than duplicate it.
        history = self.recent[:-1]
        if not history:
            return []
        length = len(history)
        edges = [round(index * length / 3) for index in range(4)]
        medoids = []
        for start, end in zip(edges[:-1], edges[1:]):
            if end > start:
                medoids.append(self._medoid(history[start:end]))
        return medoids

    def bind(self) -> Tuple[List[torch.Tensor], torch.Tensor, List[int]]:
        """Build `anchor + chronological medoids + newest` and a valid mask."""
        if self.anchor is None:
            raise RuntimeError("initialize must be called before bind")
        newest = self.recent[-1] if self.recent else self.anchor
        active = [self.anchor] + self._chronological_medoids() + [newest]
        # During warm-up, retain FARTrack's fixed five template slots by filling
        # absent consensus positions with the immutable anchor.
        while len(active) < self.num_templates:
            active.insert(-1, self.anchor)
        active = active[:self.num_templates - 1] + [newest]
        template_mask = torch.cat([record.token_mask for record in active], dim=1)
        search_mask = torch.ones((1, 200), dtype=torch.bool, device=template_mask.device)
        valid = torch.cat((template_mask, search_mask), dim=1)
        attention_mask = valid.unsqueeze(-1).expand(-1, -1, valid.shape[1]).permute(0, 2, 1)
        return [record.template for record in active], attention_mask, [record.frame_id for record in active]
