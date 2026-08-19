"""FARTrackSparse with appearance-diverse, every-frame template rebinding."""

from lib.test.tracker.appearance_reservoir import AppearanceDiverseReservoir
from lib.test.tracker.fartrack_sparse import FARTrackSparse


class FARTrackSparseADR(FARTrackSparse):
    def initialize(self, image, info: dict, name: str):
        output = super().initialize(image, info, name)
        self.appearance_reservoir = AppearanceDiverseReservoir(num_templates=self.num_template)
        self.appearance_reservoir.initialize(self.z_dict1[0])
        self.z_dict1, self.mask, self.active_template_frames = self.appearance_reservoir.bind()
        return output

    def template_update_sampling(self, new_z, sampling_method="exponential", mask=None):
        """Accept each baseline candidate then only rebind its five existing slots."""
        self.appearance_reservoir.write(new_z, mask[0], self.frame_id)
        self.z_dict1, self.mask, self.active_template_frames = self.appearance_reservoir.bind()


def get_tracker_class():
    return FARTrackSparseADR
