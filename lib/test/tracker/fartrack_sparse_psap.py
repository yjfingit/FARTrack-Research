"""Posterior-shape adaptive output projection for frozen FARTrackSparse.

PSAP changes only conversion of two native coordinate estimates into the
current box.  It does not modify the model, search crop, accepted state before
projection, template reader/writer, or network-forward count.
"""

import json
from pathlib import Path

import torch

from lib.test.tracker.fartrack_sparse import FARTrackSparse
from lib.test.tracker.posterior_shape_controller import (
    adaptive_projection,
    normalized_mean_mode_disagreement,
)


class FARTrackSparsePSAP(FARTrackSparse):
    def __init__(self, params, dataset_name):
        super().__init__(params, dataset_name)
        self.psap_enabled = bool(params.psap_enabled)
        self.psap_threshold = float(params.psap_threshold)
        self.psap_high_alpha = float(params.psap_high_alpha)
        self.psap_low_alpha = float(params.psap_low_alpha)
        self.psap_log_dir = getattr(params, "psap_log_dir", None)
        self.last_psap_disagreement = None
        self._psap_log_file = None

    def initialize(self, image, info: dict, name: str):
        if self.psap_log_dir:
            directory = Path(self.psap_log_dir)
            directory.mkdir(parents=True, exist_ok=True)
            self._psap_log_file = (directory / f"{name}.jsonl").open("w")
        return super().initialize(image, info, name)

    def track(self, image, info: dict = None):
        # Exact released delegation is the disabled-mode parity contract.
        if not self.psap_enabled:
            return super().track(image, info)
        return super().track(image, info)

    def _materialize_predicted_box(self, pred_new, seq_branch, feat_branch,
                                   resize_factor):
        if not self.psap_enabled:
            return (pred_new * self.params.search_size / resize_factor).tolist()
        mode = seq_branch
        mean = feat_branch
        disagreement = normalized_mean_mode_disagreement(mode, mean, self.range)
        projected = adaptive_projection(
            mode, mean, disagreement, self.psap_threshold,
            self.psap_high_alpha, self.psap_low_alpha,
        )
        projected = projected.view(-1, 4).mean(dim=0)
        pred_new = projected.clone()
        pred_new[2] = projected[2] - projected[0]
        pred_new[3] = projected[3] - projected[1]
        pred_new[0] = projected[0] + pred_new[2] / 2
        pred_new[1] = projected[1] + pred_new[3] / 2
        self.last_psap_disagreement = disagreement.detach()
        crop_box = pred_new * self.params.search_size / resize_factor
        if self._psap_log_file is None:
            return crop_box.tolist()
        # The diagnostic scalar joins the released output transfer rather than
        # causing a scalar CUDA synchronization.  Logging is calibration-only.
        payload = torch.cat((crop_box, disagreement.reshape(1))).tolist()
        self._psap_log_file.write(json.dumps({"normalized_mean_mode_disagreement": payload[4]}) + "\n")
        return payload[:4]


def get_tracker_class():
    return FARTrackSparsePSAP
