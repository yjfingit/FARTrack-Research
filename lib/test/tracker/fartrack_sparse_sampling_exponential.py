from lib.test.tracker.template_sampling_sweep import FARTrackSparseSamplingSweep


class FARTrackSparseExponential(FARTrackSparseSamplingSweep):
    sampling_method = "exponential"


def get_tracker_class():
    return FARTrackSparseExponential
