"""Matched-control tracker entrypoint reusing the audited sparse tracker."""

from lib.test.tracker.fartrack_sparse import FARTrackSparse


def get_tracker_class():
    return FARTrackSparse
