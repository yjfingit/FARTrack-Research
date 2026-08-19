"""Rare two-view localization arbitration for frozen FARTrackSparse.

The nominal view always runs.  On a pre-calibrated high-disagreement alarm a
single expanded search crop is evaluated from the same pre-frame state and
template history.  Both forwards are side-effect free; the selected candidate
is committed exactly once.
"""

import json
import os
from pathlib import Path

import torch

from lib.test.tracker.fartrack_sparse import FARTrackSparse
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
from lib.utils.box_ops import clip_box


class FARTrackSparseTwoView(FARTrackSparse):
    def __init__(self, params, dataset_name):
        super().__init__(params, dataset_name)
        self.two_view_enabled = bool(getattr(params, "two_view_enabled", True))
        self.two_view_threshold = float(getattr(params, "two_view_disagreement_threshold", 0.0))
        self.two_view_expansion = float(getattr(params, "two_view_expansion", 1.1))
        self.two_view_stats = {"frames": 0, "alarms": 0, "expanded_chosen": 0}
        log_dir = getattr(params, "two_view_log_dir", "")
        self.two_view_log_dir = Path(log_dir) if log_dir else None
        self.two_view_log_path = None

    def initialize(self, image, info: dict, name: str):
        result = super().initialize(image, info, name)
        if self.two_view_enabled and self.two_view_log_dir is not None:
            self.two_view_log_dir.mkdir(parents=True, exist_ok=True)
            self.two_view_log_path = self.two_view_log_dir / f"{name}.jsonl"
            self.two_view_log_path.unlink(missing_ok=True)
        return result

    def _history_seq_input(self, prior_state, resize_factor):
        """Encode prior states in the coordinate system of one search crop."""
        for index, history_box in enumerate(self.store_result):
            box_out = transform_image_to_crop(
                torch.Tensor(history_box), torch.Tensor(prior_state), resize_factor,
                torch.Tensor([self.cfg.TEST.SEARCH_SIZE, self.cfg.TEST.SEARCH_SIZE]), normalize=True,
            )
            box_out[2] += box_out[0]
            box_out[3] += box_out[1]
            box_out = box_out.clamp(min=-0.5, max=1.5)
            box_out = (box_out + 0.5) * (self.bins - 1)
            seqs_out = box_out if index == 0 else torch.cat((seqs_out, box_out), dim=-1)
        return seqs_out.unsqueeze(0)

    def _decode_candidate(self, out_dict, resize_factor, prior_state, image_height, image_width):
        """Decode a forward result without modifying tracker state."""
        seq_branch = (out_dict["seqs"][:, 0:4] + 0.5) / (self.bins - 1) - 0.5
        logits = out_dict["feat"][0:4, :, 0:self.bins * self.range]
        probabilities = logits.softmax(dim=-1)
        coordinates = torch.arange(
            self.bins * self.range, device=logits.device, dtype=logits.dtype
        ) * (2.0 / (self.bins * self.range)) + (-self.range * 0.5 + 0.5)
        feat_branch = (probabilities * coordinates).sum(dim=-1).permute(1, 0)
        disagreement = float((seq_branch - feat_branch).abs().mean().item())

        pred_boxes = ((seq_branch + feat_branch) / 2).view(-1, 4).mean(dim=0)
        center_size = pred_boxes.clone()
        center_size[2] = pred_boxes[2] - pred_boxes[0]
        center_size[3] = pred_boxes[3] - pred_boxes[1]
        center_size[0] = pred_boxes[0] + center_size[2] / 2
        center_size[1] = pred_boxes[1] + center_size[3] / 2
        crop_box = (center_size * self.params.search_size / resize_factor).tolist()

        prior_cx = prior_state[0] + 0.5 * prior_state[2]
        prior_cy = prior_state[1] + 0.5 * prior_state[3]
        half_side = 0.5 * self.params.search_size / resize_factor
        cx, cy, width, height = crop_box
        mapped = [cx + prior_cx - half_side - 0.5 * width,
                  cy + prior_cy - half_side - 0.5 * height, width, height]
        return {
            "state": clip_box(mapped, image_height, image_width, margin=10),
            "mask": out_dict["mask"],
            "disagreement": disagreement,
            "crop_box": crop_box,
            "resize_factor": resize_factor,
        }

    def _forward_candidate(self, image, prior_state, search_factor):
        """Run one candidate forward; no state, history, or template mutation."""
        image_height, image_width, _ = image.shape
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image, prior_state, search_factor, output_sz=self.params.search_size
        )
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        seqs_out = self._history_seq_input(prior_state, resize_factor)
        with torch.no_grad():
            out_dict = self.network.forward(
                template=self.z_dict1, search=search.tensors, ce_template_mask=self.box_mask_z,
                seq_input=seqs_out, stage="inference", search_feature=None, mask=self.mask,
            )
        return self._decode_candidate(out_dict, resize_factor, prior_state, image_height, image_width)

    def _commit_candidate(self, image, candidate):
        """Make the sole state/template/history mutation for a frame."""
        self.state = candidate["state"]
        z_patch_arr, _, z_amask_arr = sample_target(
            image, self.state, self.params.template_factor, output_sz=self.params.template_size
        )
        new_z = self.preprocessor.process(z_patch_arr, z_amask_arr).tensors
        self.template_update_sampling(new_z, "exponential", mask=candidate["mask"])
        if len(self.store_result) < self.prenum:
            self.store_result.append(self.state.copy())
        else:
            self.store_result[:-1] = self.store_result[1:]
            self.store_result[-1] = self.state.copy()

    @staticmethod
    def _choose_candidate(nominal, expanded):
        """Nominal wins exact ties, as pre-registered."""
        if expanded["disagreement"] < nominal["disagreement"]:
            return expanded, True
        return nominal, False

    def _track_enabled(self, image):
        prior_state = self.state.copy()
        self.frame_id += 1
        self.two_view_stats["frames"] += 1
        nominal = self._forward_candidate(image, prior_state, self.params.search_factor)
        chosen = nominal
        expanded_chosen = False
        if nominal["disagreement"] >= self.two_view_threshold:
            self.two_view_stats["alarms"] += 1
            expanded = self._forward_candidate(
                image, prior_state, self.params.search_factor * self.two_view_expansion
            )
            chosen, expanded_chosen = self._choose_candidate(nominal, expanded)
            self.two_view_stats["expanded_chosen"] += int(expanded_chosen)
        self._commit_candidate(image, chosen)
        if self.two_view_log_path is not None:
            with self.two_view_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "frame_index": self.frame_id,
                    "nominal_disagreement": nominal["disagreement"],
                    "alarm": nominal["disagreement"] >= self.two_view_threshold,
                    "expanded_chosen": expanded_chosen,
                }, sort_keys=True) + "\n")
        return {"target_bbox": self.state}

    def track(self, image, info: dict = None):
        # This exact delegation keeps the disabled path byte-identical to the
        # released tracker, including its original operation ordering.
        if not self.two_view_enabled:
            return super().track(image, info)
        return self._track_enabled(image)


def get_tracker_class():
    return FARTrackSparseTwoView
