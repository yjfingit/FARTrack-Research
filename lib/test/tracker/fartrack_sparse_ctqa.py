"""CTQA tracker entrypoint reusing the audited FARTrackSparse tracker."""

from lib.test.tracker.fartrack_sparse import FARTrackSparse


def get_tracker_class():
    return FARTrackSparse
