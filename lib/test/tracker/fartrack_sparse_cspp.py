"""CSPP: fixed-gate, horizontal-center-only posterior projection."""

from lib.test.tracker.cspp_controller import (
    normalized_cx_mode_mean_disagreement,
    project_cx_only,
)
from lib.test.tracker.fartrack_sparse import FARTrackSparse


class FARTrackSparseCSPP(FARTrackSparse):
    def __init__(self, params, dataset_name):
        super().__init__(params, dataset_name)
        self.cspp_enabled = bool(params.cspp_enabled)
        self.cspp_threshold = float(params.cspp_threshold)

    def _materialize_predicted_box(self, pred_new, seq_branch, feat_branch,
                                   resize_factor):
        if not self.cspp_enabled:
            return (pred_new * self.params.search_size / resize_factor).tolist()

        # Branches are raw normalized [left, top, right, bottom] coordinates.
        # Use the existing midpoint first, then substitute only its cx.
        midpoint = (seq_branch + feat_branch) / 2
        disagreement = normalized_cx_mode_mean_disagreement(seq_branch, feat_branch)
        projected = project_cx_only(midpoint, feat_branch, disagreement,
                                    self.cspp_threshold)
        projected = projected.view(-1, 4).mean(dim=0)
        output_xywh = projected.clone()
        output_xywh[2] = projected[2] - projected[0]
        output_xywh[3] = projected[3] - projected[1]
        output_xywh[0] = projected[0] + output_xywh[2] / 2
        output_xywh[1] = projected[1] + output_xywh[3] / 2
        return (output_xywh * self.params.search_size / resize_factor).tolist()


def get_tracker_class():
    return FARTrackSparseCSPP
