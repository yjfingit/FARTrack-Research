"""FARTrackSparse with recent-consensus template rebinding."""

from lib.test.tracker.fartrack_sparse import FARTrackSparse
from lib.test.tracker.recent_consensus_memory import RecentConsensusMemory


class FARTrackSparseRCM(FARTrackSparse):
    """Preserves baseline's one forward and per-frame candidate write path."""

    def initialize(self, image, info: dict, name: str):
        output = super().initialize(image, info, name)
        self.recent_consensus = RecentConsensusMemory(
            num_templates=self.num_template,
            window_size=getattr(self.params, "recent_consensus_window", 30))
        self.z_dict1, self.mask, self.active_template_frames = self.recent_consensus.initialize(self.z_dict1[0])
        return output

    def template_update_sampling(self, new_z, sampling_method="exponential", mask=None):
        # Called by the baseline once after every tracking forward. The write is
        # unconditional and exact; only slot selection differs on the next frame.
        self.recent_consensus.write(new_z, mask[0], self.frame_id)
        self.z_dict1, self.mask, self.active_template_frames = self.recent_consensus.bind()


def get_tracker_class():
    return FARTrackSparseRCM
