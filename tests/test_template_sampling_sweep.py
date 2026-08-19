from lib.test.tracker.template_sampling_sweep import VALID_SCHEDULES
from lib.test.tracker.fartrack_sparse_sampling_exponential import FARTrackSparseExponential
from lib.test.tracker.fartrack_sparse_sampling_linear import FARTrackSparseLinear
from lib.test.tracker.fartrack_sparse_sampling_logarithmic import FARTrackSparseLogarithmic


def test_supported_schedules_are_exactly_the_preexisting_baseline_modes():
    assert VALID_SCHEDULES == ("exponential", "linear", "logarithmic")


def test_independent_tracker_entries_select_only_their_sampling_method():
    assert FARTrackSparseExponential.sampling_method == "exponential"
    assert FARTrackSparseLinear.sampling_method == "linear"
    assert FARTrackSparseLogarithmic.sampling_method == "logarithmic"
