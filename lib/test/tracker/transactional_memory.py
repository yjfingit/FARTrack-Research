"""Causal, training-free template-memory control for FARTrackSparse.

The ledger deliberately contains no learned component.  It keeps an immutable
anchor plus a bounded set of accepted template snapshots and exposes only the
template list and attention mask needed by the frozen FARTrack forward pass.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class MemorySnapshot:
    template: torch.Tensor
    token_mask: torch.Tensor
    box: np.ndarray
    frame_id: int
    certainty: float


def _iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, aw, ah = [float(v) for v in a]
    bx1, by1, bw, bh = [float(v) for v in b]
    ax2, ay2 = ax1 + max(0.0, aw), ay1 + max(0.0, ah)
    bx2, by2 = bx1 + max(0.0, bw), by1 + max(0.0, bh)
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = max(0.0, aw * ah + bw * bh - inter)
    return inter / union if union > 1e-8 else 0.0


class TransactionalTemplateLedger:
    """Finite-state transactional memory for a fixed-size FARTrack template set."""

    STABLE = "stable"
    SUSPECT = "suspect"
    LOST = "lost"

    def __init__(self, num_templates: int, token_count: int = 49, history: int = 12,
                 suspect_patience: int = 2, lost_patience: int = 4, min_commits: int = 3):
        if num_templates < 2:
            raise ValueError("Transactional memory needs an anchor and one dynamic slot.")
        self.num_templates = num_templates
        self.token_count = token_count
        self.history_size = history
        self.suspect_patience = suspect_patience
        self.lost_patience = lost_patience
        self.min_commits = min_commits
        self.reset()

    def reset(self):
        self.anchor: Optional[MemorySnapshot] = None
        self.snapshots: List[MemorySnapshot] = []
        self.state = self.STABLE
        self.suspect_frames = 0
        self.clean_frames = 0
        self.metric_history = {"entropy": [], "motion": [], "mask": []}
        self.accepted_boxes: List[np.ndarray] = []
        self.last_action = "initialize"
        self.stats = {"commit": 0, "hold": 0, "rollback": 0}

    @staticmethod
    def certainty_from_logits(coord_logits: torch.Tensor) -> float:
        """Normalized coordinate-token certainty, robust to an arbitrary vocabulary size."""
        logits = coord_logits.detach().float()
        probabilities = logits.softmax(dim=-1).clamp_min(1e-12)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1)
        certainty = 1.0 - entropy / float(np.log(probabilities.shape[-1]))
        return float(certainty.mean().clamp(0.0, 1.0).cpu())

    @staticmethod
    def mask_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        a = a.detach().bool().reshape(-1)
        b = b.detach().bool().reshape(-1)
        union = (a | b).sum().item()
        return float((a & b).sum().item() / union) if union else 1.0

    def initialize(self, template: torch.Tensor, device: torch.device):
        dense = torch.ones((1, self.token_count), dtype=torch.bool, device=device)
        self.anchor = MemorySnapshot(template.detach().clone(), dense, np.zeros(4), 0, 1.0)
        self.snapshots = [self.anchor]
        self.state = self.STABLE
        self.last_action = "initialize"

    def _robust_anomaly(self, name: str, value: float, direction: str) -> bool:
        previous = self.metric_history[name]
        if len(previous) < self.min_commits:
            return False
        median = float(np.median(previous))
        mad = float(np.median(np.abs(np.asarray(previous) - median)))
        scale = max(1e-3, 1.4826 * mad)
        z = (value - median) / scale
        # Entropy and motion are anomalous upwards; mask consistency downwards.
        return z > 3.5 if direction == "high" else z < -3.5

    def _append_metric(self, name: str, value: float):
        values = self.metric_history[name]
        values.append(float(value))
        if len(values) > self.history_size:
            del values[0]

    def _motion_error(self, box: np.ndarray) -> float:
        if len(self.accepted_boxes) < 2:
            return 0.0
        previous, before_previous = self.accepted_boxes[-1], self.accepted_boxes[-2]
        prediction = previous.copy()
        prediction[:2] += previous[:2] - before_previous[:2]
        return 1.0 - _iou_xywh(box, prediction)

    def _transition(self, anomaly_count: int):
        if anomaly_count >= 2:
            self.suspect_frames += 1
            self.clean_frames = 0
            if self.suspect_frames >= self.lost_patience:
                self.state = self.LOST
            elif self.suspect_frames >= self.suspect_patience:
                self.state = self.SUSPECT
        else:
            self.suspect_frames = 0
            self.clean_frames += 1
            if self.state == self.SUSPECT and self.clean_frames >= self.suspect_patience:
                self.state = self.STABLE
            elif self.state == self.LOST and self.clean_frames >= self.lost_patience:
                self.state = self.SUSPECT

    def _active_snapshots(self) -> List[MemorySnapshot]:
        assert self.anchor is not None
        if self.state == self.LOST:
            return [self.anchor, self.snapshots[-1]]
        # Snapshot reliability is represented by certainty.  A deterministic
        # recency tie-break gives fixed behavior across repeated evaluations.
        ranked = sorted(self.snapshots[1:], key=lambda s: (s.certainty, s.frame_id), reverse=True)
        return [self.anchor] + ranked

    def _pack_active(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        active = self._active_snapshots()
        assert self.anchor is not None
        while len(active) < self.num_templates:
            active.append(active[-1] if len(active) > 1 else self.anchor)
        active = active[:self.num_templates]
        template_masks = torch.cat([s.token_mask for s in active], dim=1)
        extra = torch.ones((1, 200), dtype=torch.bool, device=template_masks.device)
        valid = torch.cat((template_masks, extra), dim=1)
        mask = valid.unsqueeze(-1).expand(-1, -1, valid.shape[1]).permute(0, 2, 1)
        return [s.template for s in active], mask

    def update(self, candidate_template: torch.Tensor, candidate_mask: torch.Tensor, box: Sequence[float],
               coord_logits: torch.Tensor, frame_id: int) -> Tuple[List[torch.Tensor], torch.Tensor, str]:
        """Transactionally decide whether a newly predicted crop may enter memory."""
        if self.anchor is None:
            raise RuntimeError("initialize must be called before update")
        proposed_mask = candidate_mask.detach().bool().reshape(1, self.token_count)
        current_box = np.asarray(box, dtype=np.float32).copy()
        certainty = self.certainty_from_logits(coord_logits)
        entropy = 1.0 - certainty
        motion = self._motion_error(current_box)
        agreement = self.mask_similarity(proposed_mask, self.snapshots[-1].token_mask)
        anomalies = sum((
            self._robust_anomaly("entropy", entropy, "high"),
            self._robust_anomaly("motion", motion, "high"),
            self._robust_anomaly("mask", agreement, "low"),
        ))
        self._transition(anomalies)

        if self.state == self.STABLE:
            snapshot = MemorySnapshot(candidate_template.detach().clone(), proposed_mask.clone(), current_box,
                                      frame_id, certainty)
            self.snapshots.append(snapshot)
            if len(self.snapshots) > self.num_templates:
                # Anchor is immutable; evict the weakest non-anchor transaction.
                weakest = min(range(1, len(self.snapshots)), key=lambda i: (self.snapshots[i].certainty,
                                                                            self.snapshots[i].frame_id))
                del self.snapshots[weakest]
            self.accepted_boxes.append(current_box)
            self.last_action = "commit"
            self.stats["commit"] += 1
            self._append_metric("entropy", entropy)
            self._append_metric("motion", motion)
            self._append_metric("mask", agreement)
        elif self.state == self.SUSPECT:
            self.last_action = "hold"
            self.stats["hold"] += 1
        else:
            self.last_action = "rollback"
            self.stats["rollback"] += 1
        templates, mask = self._pack_active()
        return templates, mask, self.last_action
