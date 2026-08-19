"""Bounded, training-free stable-anchor recovery for FARTrackSparse.

The scheduler owns only template selection.  It never changes frozen model
weights, search crops, or the benchmark harness.  A failure run freezes
template writes and briefly replays the immutable first-frame anchor together
with the most recently accepted view, then enters a cooldown before another
recovery can be armed.
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch


@dataclass
class AnchorSnapshot:
    template: torch.Tensor
    token_mask: torch.Tensor
    box: np.ndarray
    frame_id: int


def _iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    right = max(0.0, min(ax + max(0.0, aw), bx + max(0.0, bw)) - max(ax, bx))
    bottom = max(0.0, min(ay + max(0.0, ah), by + max(0.0, bh)) - max(ay, by))
    inter = right * bottom
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    return inter / union if union > 1e-8 else 0.0


class StableAnchorRecoveryScheduler:
    """A bounded recovery controller with an immutable first-frame anchor."""

    def __init__(self, num_templates: int, token_count: int = 49,
                 failure_patience: int = 3, recovery_frames: int = 2,
                 cooldown_frames: int = 8, history_size: int = 8):
        if num_templates < 2:
            raise ValueError("Stable-anchor recovery requires at least two templates")
        if min(failure_patience, recovery_frames, cooldown_frames) < 1:
            raise ValueError("Scheduler durations must be positive")
        self.num_templates = num_templates
        self.token_count = token_count
        self.failure_patience = failure_patience
        self.recovery_frames = recovery_frames
        self.cooldown_frames = cooldown_frames
        self.history_size = history_size
        self.reset()

    def reset(self):
        self.anchor = None
        self.accepted: List[AnchorSnapshot] = []
        self.accepted_boxes: List[np.ndarray] = []
        self.failure_run = 0
        self.recovery_left = 0
        self.cooldown_left = 0
        self.last_action = "initialize"
        self.stats = {"commit": 0, "freeze": 0, "recover": 0}

    @staticmethod
    def confidence_from_logits(coord_logits: torch.Tensor) -> float:
        logits = coord_logits.detach().float()
        probs = logits.softmax(dim=-1).clamp_min(1e-12)
        entropy = -(probs * probs.log()).sum(dim=-1)
        return float((1.0 - entropy / np.log(probs.shape[-1])).mean().clamp(0, 1).cpu())

    def initialize(self, template: torch.Tensor):
        mask = torch.ones((1, self.token_count), dtype=torch.bool, device=template.device)
        self.anchor = AnchorSnapshot(template.detach().clone(), mask, np.zeros(4, dtype=np.float32), 0)
        self.accepted = [self.anchor]
        self.accepted_boxes = []
        return self._pack(self.accepted)

    def _motion_error(self, box: np.ndarray) -> float:
        if len(self.accepted_boxes) < 2:
            return 0.0
        last, previous = self.accepted_boxes[-1], self.accepted_boxes[-2]
        prediction = last.copy()
        prediction[:2] += last[:2] - previous[:2]
        return 1.0 - _iou_xywh(box, prediction)

    def _failure(self, box: np.ndarray, coord_logits: torch.Tensor) -> bool:
        # Low normalized certainty is the primary causal evidence.  A severe
        # motion jump is only trusted once an accepted velocity is available.
        low_confidence = self.confidence_from_logits(coord_logits) < 0.035
        high_motion = len(self.accepted_boxes) >= 2 and self._motion_error(box) > 0.92
        return low_confidence or high_motion

    def _pack(self, snapshots: List[AnchorSnapshot]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        if self.anchor is None:
            raise RuntimeError("initialize must be called before scheduling")
        active = list(snapshots)
        while len(active) < self.num_templates:
            active.append(active[-1] if len(active) > 1 else self.anchor)
        active = active[:self.num_templates]
        token_masks = torch.cat([snapshot.token_mask for snapshot in active], dim=1)
        search_mask = torch.ones((1, 200), dtype=torch.bool, device=token_masks.device)
        valid = torch.cat((token_masks, search_mask), dim=1)
        return [snapshot.template for snapshot in active], valid.unsqueeze(-1).expand(-1, -1, valid.shape[1]).permute(0, 2, 1)

    def _recovery_views(self) -> List[AnchorSnapshot]:
        assert self.anchor is not None
        latest = self.accepted[-1] if self.accepted else self.anchor
        # Replaying only these two sources explicitly removes contaminated
        # short-term candidates from the forward pass.
        return [self.anchor, latest]

    def step(self, candidate_template: torch.Tensor, candidate_mask: torch.Tensor,
             box: Sequence[float], coord_logits: torch.Tensor, frame_id: int,
             force_failure: bool = None) -> Tuple[List[torch.Tensor], torch.Tensor, str]:
        if self.anchor is None:
            raise RuntimeError("initialize must be called before step")
        current_box = np.asarray(box, dtype=np.float32).copy()
        failed = self._failure(current_box, coord_logits) if force_failure is None else force_failure

        if self.recovery_left:
            self.recovery_left -= 1
            self.cooldown_left = max(self.cooldown_left, self.cooldown_frames)
            self.failure_run = 0
            self.last_action = "recover"
            self.stats["recover"] += 1
            return (*self._pack(self._recovery_views()), self.last_action)

        if self.cooldown_left:
            self.cooldown_left -= 1
        self.failure_run = self.failure_run + 1 if failed else 0
        if self.failure_run >= self.failure_patience and self.cooldown_left == 0:
            # Schedule the current and next recovery views. No candidate from
            # the failure run is committed before anchor replay.
            self.recovery_left = self.recovery_frames - 1
            self.failure_run = 0
            self.cooldown_left = self.cooldown_frames
            self.last_action = "recover"
            self.stats["recover"] += 1
            return (*self._pack(self._recovery_views()), self.last_action)

        if failed:
            self.last_action = "freeze"
            self.stats["freeze"] += 1
            return (*self._pack(self.accepted), self.last_action)

        proposed_mask = candidate_mask.detach().bool().reshape(1, self.token_count).clone()
        snapshot = AnchorSnapshot(candidate_template.detach().clone(), proposed_mask, current_box, frame_id)
        self.accepted.append(snapshot)
        if len(self.accepted) > self.num_templates:
            # Preserve anchor and a compact chronological accepted history.
            self.accepted.pop(1)
        self.accepted_boxes.append(current_box)
        self.accepted_boxes = self.accepted_boxes[-self.history_size:]
        self.last_action = "commit"
        self.stats["commit"] += 1
        return (*self._pack(self.accepted), self.last_action)
