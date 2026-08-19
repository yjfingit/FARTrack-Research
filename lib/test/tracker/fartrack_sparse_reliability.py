"""Frozen FARTrackSparse tracker with passive per-frame reliability logging.

This tracker intentionally reuses FARTrackSparse.track unchanged apart from a
post-forward observer hook.  No logged signal feeds into localization, state,
or template updates.
"""

import json
from pathlib import Path

import torch

from lib.test.tracker.fartrack_sparse import FARTrackSparse


class FARTrackSparseReliability(FARTrackSparse):
    def __init__(self, params, dataset_name):
        super().__init__(params, dataset_name)
        self.reliability_log_dir = Path(params.reliability_log_dir)
        self.reliability_log_path = None

    def initialize(self, image, info: dict, name: str):
        result = super().initialize(image, info, name)
        self.reliability_log_dir.mkdir(parents=True, exist_ok=True)
        self.reliability_log_path = self.reliability_log_dir / f"{name}.jsonl"
        self.reliability_log_path.unlink(missing_ok=True)
        return result

    def _record_reliability(self, out_dict):
        """Write zero-extra-forward localization diagnostics for this frame."""
        logits = out_dict["feat"][0:4, :, 0:self.bins * self.range]
        probabilities = logits.softmax(dim=-1)
        log_bins = torch.log(torch.tensor(float(logits.shape[-1]), device=logits.device))
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        entropy = (entropy / log_bins).mean()
        top2 = probabilities.topk(k=2, dim=-1).values
        margin = (top2[..., 0] - top2[..., 1]).mean()

        # Both terms use the exact coordinate representations that the frozen
        # tracker averages below: sequence-token coordinates and feature-bin
        # expectations in the normalized search crop.
        seq_branch = (out_dict["seqs"][:, 0:4] + 0.5) / (self.bins - 1) - 0.5
        coordinates = torch.arange(
            self.bins * self.range, device=logits.device, dtype=logits.dtype
        ) * (2.0 / (self.bins * self.range)) + (-self.range * 0.5 + 0.5)
        feat_branch = (probabilities * coordinates).sum(dim=-1).permute(1, 0)
        disagreement = (seq_branch - feat_branch).abs().mean()

        row = {
            "frame_index": int(self.frame_id),
            "coord_entropy": float(entropy.item()),
            "coord_margin": float(margin.item()),
            "branch_disagreement_l1": float(disagreement.item()),
        }
        with self.reliability_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def get_tracker_class():
    return FARTrackSparseReliability
