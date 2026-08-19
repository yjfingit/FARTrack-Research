"""FARTrackSparse with constant-size appearance-diverse rebinding."""

import torch

from lib.test.tracker.appearance_reservoir import AppearanceDiverseReservoir
from lib.test.tracker.fartrack_sparse import FARTrackSparse
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
from lib.utils.box_ops import clip_box


class FARTrackSparseADR(FARTrackSparse):
    def initialize(self, image, info: dict, name: str):
        output = super().initialize(image, info, name)
        self.appearance_reservoir = AppearanceDiverseReservoir(num_templates=self.num_template, capacity=4)
        self.appearance_reservoir.initialize(
            self.z_dict1[0], self.appearance_reservoir.descriptor_from_rgb(self.z_patch_arr))
        self.z_dict1, self.mask, self.active_template_frames = self.appearance_reservoir.bind()
        return output

    def track(self, image, info: dict = None):
        """Baseline inference with only the final template-rebinding operation replaced."""
        height, width, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image, self.state, self.params.search_factor, output_sz=self.params.search_size)
        sequence_tokens = []
        for previous_box in self.store_result:
            box = transform_image_to_crop(torch.Tensor(previous_box), torch.Tensor(self.state), resize_factor,
                                          torch.Tensor([self.cfg.TEST.SEARCH_SIZE, self.cfg.TEST.SEARCH_SIZE]),
                                          normalize=True)
            box[2:] += box[:2]
            sequence_tokens.append((box.clamp(min=-0.5, max=1.5) + 0.5) * (self.bins - 1))
        seqs_out = torch.cat(sequence_tokens, dim=-1).unsqueeze(0)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        with torch.no_grad():
            out_dict = self.network.forward(template=self.z_dict1, search=search.tensors,
                                            ce_template_mask=self.box_mask_z, seq_input=seqs_out,
                                            stage="inference", search_feature=None, mask=self.mask)
        predicted = (out_dict['seqs'][:, :4] + 0.5) / (self.bins - 1) - 0.5
        logits = out_dict['feat'][:4, :, :self.bins * self.range]
        values = torch.arange(self.bins * self.range, device=logits.device, dtype=logits.dtype)
        values = -self.range * 0.5 + 0.5 + values * (2.0 / (self.bins * self.range))
        refinement = (logits.softmax(-1) * values).sum(dim=-1).permute(1, 0)
        corners = ((refinement + predicted) / 2).view(-1, 4).mean(dim=0)
        box = corners.clone()
        box[2:] = corners[2:] - corners[:2]
        box[:2] = corners[:2] + box[2:] / 2
        self.state = clip_box(self.map_box_back((box * self.params.search_size / resize_factor).tolist(), resize_factor),
                              height, width, margin=10)
        z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                     output_sz=self.params.template_size)
        candidate = self.preprocessor.process(z_patch_arr, z_amask_arr).tensors
        self.appearance_reservoir.write(candidate, out_dict['mask'][0], self.frame_id,
                                        self.appearance_reservoir.descriptor_from_rgb(z_patch_arr))
        self.z_dict1, self.mask, self.active_template_frames = self.appearance_reservoir.bind()
        self.store_result = (self.store_result + [self.state.copy()])[-self.prenum:]
        if self.save_all_boxes:
            return {"target_bbox": self.state, "all_boxes": self.state * self.cfg.MODEL.NUM_OBJECT_QUERIES}
        return {"target_bbox": self.state}


def get_tracker_class():
    return FARTrackSparseADR
