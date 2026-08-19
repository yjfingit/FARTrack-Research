"""Reliability-gated sparse-flow proposal for a frozen FARTrack checkpoint.

This tracker intentionally leaves the network box untouched unless both the
frozen branch-disagreement alarm and conservative Lucas--Kanade checks pass.
"""

import cv2
import numpy as np

from lib.test.tracker.fartrack_sparse import FARTrackSparse
from lib.utils.box_ops import clip_box


def sparse_flow_displacement(previous_rgb, current_rgb, previous_box, min_inliers=6,
                             fb_threshold=1.0, max_corners=80):
    """Return a robust RGB-frame displacement or ``None`` after any gate fails."""
    if previous_rgb is None or current_rgb is None:
        return None
    if previous_rgb.ndim != 3 or current_rgb.ndim != 3:
        return None
    if previous_rgb.shape[:2] != current_rgb.shape[:2]:
        return None
    height, width = previous_rgb.shape[:2]
    x, y, bw, bh = [float(value) for value in previous_box]
    x0, y0 = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
    x1, y1 = min(width, int(np.ceil(x + bw))), min(height, int(np.ceil(y + bh)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    previous_gray = cv2.cvtColor(previous_rgb, cv2.COLOR_RGB2GRAY)
    current_gray = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    points0 = cv2.goodFeaturesToTrack(previous_gray, maxCorners=max_corners,
                                      qualityLevel=0.01, minDistance=3,
                                      blockSize=5, mask=mask)
    if points0 is None or len(points0) < min_inliers:
        return None
    points1, status01, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, points0, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    if points1 is None or status01 is None:
        return None
    points0b, status10, _ = cv2.calcOpticalFlowPyrLK(
        current_gray, previous_gray, points1, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    if points0b is None or status10 is None:
        return None
    forward_backward = np.linalg.norm(points0.reshape(-1, 2) - points0b.reshape(-1, 2), axis=1)
    valid = (status01.reshape(-1) > 0) & (status10.reshape(-1) > 0) & (forward_backward <= fb_threshold)
    if int(valid.sum()) < min_inliers:
        return None
    motion = points1.reshape(-1, 2)[valid] - points0.reshape(-1, 2)[valid]
    displacement = np.median(motion, axis=0)
    if not np.isfinite(displacement).all():
        return None
    # A single-frame translation beyond the full target extent is implausible
    # under this local correction model and is safer to reject.
    if np.linalg.norm(displacement) > max(float(bw), float(bh), 1.0):
        return None
    return displacement.astype(np.float64), int(valid.sum())


class FARTrackSparseRBSFA(FARTrackSparse):
    """Rare Branch-disagreement Sparse-Flow Arbitration."""

    def __init__(self, params, dataset_name):
        super().__init__(params, dataset_name)
        self.rbsfa_enabled = bool(getattr(params, "rbsfa_enabled", True))
        self.rbsfa_threshold = float(getattr(params, "rbsfa_threshold", 0.006562486290931702))
        self.rbsfa_blend = float(getattr(params, "rbsfa_blend", 0.25))
        self.rbsfa_min_inliers = int(getattr(params, "rbsfa_min_inliers", 6))
        self.rbsfa_fb_threshold = float(getattr(params, "rbsfa_fb_threshold", 1.0))
        self._previous_rgb = None
        self.rbsfa_stats = {"alarms": 0, "accepted": 0, "rejected": 0}

    def initialize(self, image, info: dict, name: str):
        output = super().initialize(image, info, name)
        self._previous_rgb = image.copy()
        return output

    def adjust_candidate_state(self, image, candidate_state, branch_disagreement):
        network_state = np.asarray(candidate_state, dtype=np.float64)
        if not self.rbsfa_enabled or branch_disagreement < self.rbsfa_threshold:
            return network_state.tolist()
        self.rbsfa_stats["alarms"] += 1
        displacement_result = sparse_flow_displacement(
            self._previous_rgb, image, self.state, self.rbsfa_min_inliers, self.rbsfa_fb_threshold)
        if displacement_result is None:
            self.rbsfa_stats["rejected"] += 1
            return network_state.tolist()
        displacement, _ = displacement_result
        flow_state = self.state.copy()
        flow_state[0] += float(displacement[0])
        flow_state[1] += float(displacement[1])
        # Size stays network-derived. Only interpolate the independent center proposal.
        alpha = self.rbsfa_blend
        corrected = network_state.copy()
        corrected[:2] = (1.0 - alpha) * network_state[:2] + alpha * np.asarray(flow_state[:2])
        self.rbsfa_stats["accepted"] += 1
        return clip_box(corrected.tolist(), image.shape[0], image.shape[1], margin=10)

    def after_track_frame(self, image):
        self._previous_rgb = image.copy()


def get_tracker_class():
    return FARTrackSparseRBSFA
