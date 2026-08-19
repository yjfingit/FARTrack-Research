"""Immutable released FARTrackSparse baseline under the research parameter name."""

from lib.test.tracker.fartrack_sparse import FARTrackSparse


def get_tracker_class():
    return FARTrackSparse
