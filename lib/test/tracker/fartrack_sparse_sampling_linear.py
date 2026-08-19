from lib.test.tracker.template_sampling_sweep import FARTrackSparseSamplingSweep


class FARTrackSparseLinear(FARTrackSparseSamplingSweep):
    sampling_method = "linear"


def get_tracker_class():
    return FARTrackSparseLinear
