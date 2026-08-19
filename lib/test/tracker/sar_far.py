"""Research-only FARTrackSparse tracker with bounded stable-anchor replay."""

import torch

from lib.test.tracker.fartrack_sparse import FARTrackSparse
from lib.test.tracker.stable_anchor_recovery import StableAnchorRecoveryScheduler
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
from lib.utils.box_ops import clip_box


class StableAnchorRecoveryFARTrackSparse(FARTrackSparse):
    """Frozen FARTrackSparse plus an independent template-selection controller."""

    def initialize(self, image, info: dict, name: str):
        output = super().initialize(image, info, name)
        self.anchor_scheduler = StableAnchorRecoveryScheduler(self.num_template)
        self.z_dict1, self.mask = self.anchor_scheduler.initialize(self.z_dict1[0])
        self.store_result = [info['init_bbox'].copy() for _ in range(self.prenum)]
        return output

    def track(self, image, info: dict = None):
        height, width, _ = image.shape
        self.frame_id += 1
        x_patch, resize_factor, x_mask = sample_target(
            image, self.state, self.params.search_factor, output_sz=self.params.search_size)
        sequence_boxes = []
        for previous_box in self.store_result:
            box = transform_image_to_crop(torch.Tensor(previous_box), torch.Tensor(self.state), resize_factor,
                                          torch.Tensor([self.cfg.TEST.SEARCH_SIZE, self.cfg.TEST.SEARCH_SIZE]), normalize=True)
            box[2:] += box[:2]
            sequence_boxes.append((box.clamp(min=-0.5, max=1.5) + 0.5) * (self.bins - 1))
        sequence_input = torch.cat(sequence_boxes, dim=-1).unsqueeze(0)
        search = self.preprocessor.process(x_patch, x_mask)
        with torch.no_grad():
            out = self.network.forward(template=self.z_dict1, search=search.tensors,
                                       ce_template_mask=self.box_mask_z, seq_input=sequence_input,
                                       stage="inference", search_feature=None, mask=self.mask)
        sequence_prediction = (out['seqs'][:, :4] + 0.5) / (self.bins - 1) - 0.5
        logits = out['feat'][:4, :, :self.bins * self.range]
        values = torch.linspace(-self.range * 0.5 + 0.5, self.range * 0.5 + 0.5 - 1 / (self.bins * self.range),
                                self.bins * self.range, device=logits.device, dtype=logits.dtype)
        decoded = (logits.softmax(-1) * values).sum(dim=-1).permute(1, 0)
        corners = ((decoded + sequence_prediction) / 2).view(-1, 4).mean(dim=0)
        prediction = corners.clone()
        prediction[2:] = corners[2:] - corners[:2]
        prediction[:2] = corners[:2] + prediction[2:] / 2
        search_box = (prediction * self.params.search_size / resize_factor).tolist()
        self.state = clip_box(self.map_box_back(search_box, resize_factor), height, width, margin=10)

        z_patch, _, z_mask = sample_target(image, self.state, self.params.template_factor, output_sz=self.params.template_size)
        candidate = self.preprocessor.process(z_patch, z_mask).tensors
        self.z_dict1, self.mask, self.memory_action = self.anchor_scheduler.step(
            candidate, out['mask'][0], self.state, logits, self.frame_id)
        self.store_result = (self.store_result + [self.state.copy()])[-self.prenum:]
        if self.save_all_boxes:
            return {"target_bbox": self.state, "all_boxes": self.state * self.cfg.MODEL.NUM_OBJECT_QUERIES}
        return {"target_bbox": self.state}


def get_tracker_class():
    return StableAnchorRecoveryFARTrackSparse
