"""Zero-overhead sampling-policy switch for frozen FARTrackSparse.

This module deliberately delegates all template writes, masks, and forward
calls to FARTrackSparse. It changes only the already-supported sampling string
given to the baseline's `template_update_sampling` method.
"""

from lib.test.tracker.fartrack_sparse import FARTrackSparse


VALID_SCHEDULES = ("exponential", "linear", "logarithmic")


class FARTrackSparseSamplingSweep(FARTrackSparse):
    """Independent tracker whose only behavioral degree of freedom is sampling."""

    sampling_method = "exponential"

    def __init__(self, params, dataset_name):
        super().__init__(params, dataset_name)
        self.sampling_method = getattr(params, "template_sampling_method", self.sampling_method)
        if self.sampling_method not in VALID_SCHEDULES:
            raise ValueError("Unsupported template sampling method: {}".format(self.sampling_method))

    def template_update_sampling(self, new_z, sampling_method="exponential", mask=None):
        # `track` invokes this once after the unchanged primary network forward.
        # Keep the baseline write and mask packing implementation verbatim.
        return super().template_update_sampling(new_z, self.sampling_method, mask)
