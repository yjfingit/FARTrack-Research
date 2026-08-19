"""Research tracker: FARTrackSparse with a transactional template ledger."""

import numpy as np
import torch

from lib.test.tracker.fartrack_sparse import FARTrackSparse
from lib.test.tracker.transactional_memory import TransactionalTemplateLedger
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box


class TRMFARTrackSparse(FARTrackSparse):
    """Keeps the released FARTrackSparse network frozen and baseline untouched."""

    def initialize(self, image, info: dict, name: str):
        output = super().initialize(image, info, name)
        self.ledger = TransactionalTemplateLedger(self.num_template)
        self.ledger.initialize(self.z_dict1[0], self.z_dict1[0].device)
        self.z_dict1, self.mask = self.ledger._pack_active()
        self.store_result = [info['init_bbox'].copy() for _ in range(self.prenum)]
        return output

    def track(self, image, info: dict = None):
        h, w, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image, self.state, self.params.search_factor, output_sz=self.params.search_size)
        seqs = []
        for previous_box in self.store_result:
            box = transform_image_to_crop(torch.Tensor(previous_box), torch.Tensor(self.state), resize_factor,
                                          torch.Tensor([self.cfg.TEST.SEARCH_SIZE, self.cfg.TEST.SEARCH_SIZE]),
                                          normalize=True)
            box[2:] += box[:2]
            seqs.append(((box.clamp(min=-0.5, max=1.5) + 0.5) * (self.bins - 1)))
        seqs_out = torch.cat(seqs, dim=-1).unsqueeze(0)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        with torch.no_grad():
            out_dict = self.network.forward(template=self.z_dict1, search=search.tensors,
                                            ce_template_mask=self.box_mask_z, seq_input=seqs_out,
                                            stage="inference", search_feature=None, mask=self.mask)
        seq_boxes = (out_dict['seqs'][:, :4] + 0.5) / (self.bins - 1) - 0.5
        coord_logits = out_dict['feat'][:4, :, :self.bins * self.range]
        distribution = coord_logits.softmax(-1)
        values = torch.linspace(-self.range * 0.5 + 0.5,
                                self.range * 0.5 + 0.5 - 1 / (self.bins * self.range),
                                self.bins * self.range, device=coord_logits.device, dtype=coord_logits.dtype)
        decoded = (distribution * values).sum(dim=-1).permute(1, 0)
        corners = ((decoded + seq_boxes) / 2).view(-1, 4).mean(dim=0)
        pred = corners.clone()
        pred[2:] = corners[2:] - corners[:2]
        pred[:2] = corners[:2] + pred[2:] / 2
        search_box = (pred * self.params.search_size / resize_factor).tolist()
        self.state = clip_box(self.map_box_back(search_box, resize_factor), h, w, margin=10)

        z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                     output_sz=self.params.template_size)
        candidate = self.preprocessor.process(z_patch_arr, z_amask_arr).tensors
        # FARTrack emits four pruning ratios; preserve its baseline 25% candidate
        # while making its write reversible through the ledger.
        self.z_dict1, self.mask, self.memory_action = self.ledger.update(
            candidate, out_dict['mask'][0], self.state, coord_logits, self.frame_id)
        self.store_result = (self.store_result + [self.state.copy()])[-self.prenum:]
        if self.save_all_boxes:
            return {"target_bbox": self.state, "all_boxes": self.state * self.cfg.MODEL.NUM_OBJECT_QUERIES}
        return {"target_bbox": self.state}


def get_tracker_class():
    return TRMFARTrackSparse
