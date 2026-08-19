"""Research harness alias for the unmodified FARTrackSparse behavior."""

from lib.test.tracker.fartrack_sparse import FARTrackSparse


def get_tracker_class():
    return FARTrackSparse
