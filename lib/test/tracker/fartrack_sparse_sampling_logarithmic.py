from lib.test.tracker.template_sampling_sweep import FARTrackSparseSamplingSweep


class FARTrackSparseLogarithmic(FARTrackSparseSamplingSweep):
    sampling_method = "logarithmic"


def get_tracker_class():
    return FARTrackSparseLogarithmic
