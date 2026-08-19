"""CPU-only invariants for rare two-view arbitration."""

import unittest
from unittest.mock import patch

import numpy as np

from lib.test.tracker.fartrack_sparse import FARTrackSparse
from lib.test.tracker.fartrack_sparse_two_view import FARTrackSparseTwoView


class TwoViewArbitrationTests(unittest.TestCase):
    def _tracker(self, threshold=0.5):
        tracker = object.__new__(FARTrackSparseTwoView)
        tracker.two_view_enabled = True
        tracker.two_view_threshold = threshold
        tracker.two_view_expansion = 1.2
        tracker.two_view_stats = {"frames": 0, "alarms": 0, "expanded_chosen": 0}
        tracker.two_view_log_path = None
        tracker.frame_id = 0
        tracker.state = [10.0, 10.0, 20.0, 20.0]
        tracker.params = type("Params", (), {"search_factor": 4.0})()
        return tracker

    def test_choice_and_tie_break(self):
        nominal = {"disagreement": 0.4}
        expanded = {"disagreement": 0.3}
        self.assertIs(FARTrackSparseTwoView._choose_candidate(nominal, expanded)[0], expanded)
        tied = {"disagreement": 0.4}
        self.assertIs(FARTrackSparseTwoView._choose_candidate(nominal, tied)[0], nominal)

    def test_nonalarm_performs_one_forward_and_one_commit(self):
        tracker = self._tracker()
        forwards, commits = [], []
        tracker._forward_candidate = lambda image, state, factor: forwards.append(factor) or {"disagreement": 0.1, "state": state}
        tracker._commit_candidate = lambda image, candidate: commits.append(candidate)
        out = tracker._track_enabled(np.zeros((40, 40, 3), dtype=np.uint8))
        self.assertEqual(forwards, [4.0])
        self.assertEqual(len(commits), 1)
        self.assertEqual(out["target_bbox"], tracker.state)
        self.assertEqual(tracker.two_view_stats["alarms"], 0)

    def test_alarm_performs_two_forwards_but_one_commit(self):
        tracker = self._tracker()
        forwards, commits = [], []
        candidates = iter(({"disagreement": 0.7, "state": [1]}, {"disagreement": 0.2, "state": [2]}))
        tracker._forward_candidate = lambda image, state, factor: forwards.append(factor) or next(candidates)
        tracker._commit_candidate = lambda image, candidate: commits.append(candidate)
        tracker._track_enabled(np.zeros((40, 40, 3), dtype=np.uint8))
        self.assertEqual(forwards, [4.0, 4.8])
        self.assertEqual(commits, [{"disagreement": 0.2, "state": [2]}])
        self.assertEqual(tracker.two_view_stats["expanded_chosen"], 1)

    def test_disabled_delegates_exactly_to_base_tracker(self):
        tracker = self._tracker()
        tracker.two_view_enabled = False
        sentinel = {"target_bbox": [9, 8, 7, 6]}
        with patch.object(FARTrackSparse, "track", return_value=sentinel) as base_track:
            self.assertIs(tracker.track(np.zeros((1, 1, 3), dtype=np.uint8), {"x": 1}), sentinel)
        base_track.assert_called_once()


if __name__ == "__main__":
    unittest.main()
