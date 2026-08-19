import unittest
from unittest.mock import patch

import torch

from lib.test.tracker.fartrack_sparse_dual_lane import FARTrackSparseDualLane
from lib.test.tracker.fartrack_sparse import FARTrackSparse


def _controller():
    tracker = FARTrackSparseDualLane.__new__(FARTrackSparseDualLane)
    tracker.num_template = 5
    tracker.frame_id = 1
    tracker.z_dict1 = [torch.tensor([0.0])] * 5
    tracker.stable_templates = []
    tracker.stable_masks = []
    tracker.stable_entropy_threshold = 0.4
    return tracker


def _mask(value):
    return torch.full((4, 1, 49), value, dtype=torch.bool)


class DualLaneControllerTest(unittest.TestCase):
    def test_never_drops_newest_and_rejects_risky_stable_write(self):
        tracker = _controller()
        newest = torch.tensor([1.0])
        tracker._dual_lane_update(newest, _mask(True), entropy=0.8)
        self.assertTrue(all(torch.equal(slot, torch.tensor([0.0])) for slot in tracker.z_dict1[:4]))
        self.assertTrue(torch.equal(tracker.z_dict1[4], newest))
        self.assertEqual(tracker.stable_templates, [])

    def test_stable_slots_only_reference_certified_history(self):
        tracker = _controller()
        accepted_one = torch.tensor([1.0])
        accepted_two = torch.tensor([2.0])
        tracker._dual_lane_update(accepted_one, _mask(True), entropy=0.3)
        tracker._dual_lane_update(accepted_two, _mask(False), entropy=0.2)
        stable_values = [float(slot.item()) for slot in tracker.z_dict1[1:4]]
        self.assertTrue(set(stable_values).issubset({1.0, 2.0}))
        self.assertEqual(float(tracker.z_dict1[4].item()), 2.0)

    def test_exact_threshold_is_inclusive(self):
        tracker = _controller()
        candidate = torch.tensor([3.0])
        tracker._dual_lane_update(candidate, _mask(True), entropy=0.4)
        self.assertEqual(len(tracker.stable_templates), 1)

    def test_disabled_policy_delegates_exactly_to_base_tracking_path(self):
        tracker = _controller()
        tracker.dual_lane_enabled = False
        tracker.network = type("Network", (), {"forward": object()})()
        sentinel = {"target_bbox": [1, 2, 3, 4]}
        with patch.object(FARTrackSparse, "track", return_value=sentinel) as base_track:
            self.assertIs(tracker.track("image", {"frame": 1}), sentinel)
        base_track.assert_called_once_with("image", {"frame": 1})


if __name__ == "__main__":
    unittest.main()
