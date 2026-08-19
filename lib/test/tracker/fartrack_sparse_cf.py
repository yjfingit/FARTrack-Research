"""Counterfactual template-pool verifier for frozen FARTrackSparse.

The primary forward pass is unchanged.  Before a dynamic template is written,
two bounded counterfactual passes retain only the oldest or newest accepted
template views.  Their disagreement is used solely as a write gate.
"""

from __future__ import annotations

import torch

from lib.test.tracker.fartrack_sparse import FARTrackSparse
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
from lib.utils.box_ops import clip_box


class CounterfactualAgreementVerifier:
    """Build template-key masks and measure coordinate disagreement."""

    def __init__(self, num_template: int, template_tokens: int = 49):
        if num_template < 2:
            raise ValueError("counterfactual verification needs at least two templates")
        self.num_template = num_template
        self.template_tokens = template_tokens

    def retain_only(self, base_mask: torch.Tensor, retained_slots: tuple[int, ...]) -> torch.Tensor:
        """Mask all template key columns except retained slots, preserving base sparsity."""
        mask = base_mask.clone()
        retained = set(retained_slots)
        for slot in range(self.num_template):
            if slot not in retained:
                start = slot * self.template_tokens
                mask[:, :, start:start + self.template_tokens] = False
        return mask

    @staticmethod
    def normalized_l1(old_box: torch.Tensor, recent_box: torch.Tensor) -> torch.Tensor:
        """Mean L1 difference in normalized search coordinates, batch preserving."""
        return (old_box - recent_box).abs().mean(dim=-1)


class AcceptedTemplatePool:
    """A fixed-width FARTrack template pool indexed by accepted writes only."""

    def __init__(self, anchor: torch.Tensor, num_template: int, template_tokens: int = 49):
        self.anchor = anchor
        self.num_template = num_template
        self.template_tokens = template_tokens
        self.anchor_mask = torch.ones(template_tokens, dtype=torch.bool, device=anchor.device)
        self.dynamic: list[tuple[torch.Tensor, torch.Tensor]] = []

    @property
    def capacity(self) -> int:
        # The anchor must remain available to both the tracker and verifier.
        return self.num_template - 1

    def accept(self, template: torch.Tensor, pruning_masks) -> None:
        """Store one candidate and retain only the newest bounded dynamic views."""
        token_mask = pruning_masks[0][0].to(device=template.device, dtype=torch.bool).clone()
        if token_mask.numel() != self.template_tokens:
            raise ValueError(f"expected {self.template_tokens} template-mask tokens, got {token_mask.numel()}")
        self.dynamic.append((template, token_mask))
        self.dynamic = self.dynamic[-self.capacity:]

    def slots(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        # Right-alignment matches FARTrack's early-frame anchor padding while
        # keeping every dynamic slot tied to a successful write.
        padding = self.num_template - len(self.dynamic)
        return [(self.anchor, self.anchor_mask)] * padding + self.dynamic

    def fartrack_inputs(self) -> tuple[list[torch.Tensor], torch.Tensor]:
        slots = self.slots()
        templates = [template for template, _ in slots]
        key_mask = torch.ones(
            (1, self.template_tokens * self.num_template + 200),
            dtype=torch.bool, device=self.anchor.device)
        for slot, (_, token_mask) in enumerate(slots):
            start = slot * self.template_tokens
            key_mask[:, start:start + self.template_tokens] = token_mask
        # FARTrack's attention accepts a full query-by-key boolean mask.
        return templates, key_mask.unsqueeze(-1).expand(-1, -1, key_mask.shape[-1]).permute(0, 2, 1)


class FARTrackSparseCounterfactual(FARTrackSparse):
    """FARTrackSparse with a label-free counterfactual template-write gate."""

    def __init__(self, params, dataset_name):
        super().__init__(params, dataset_name)
        self.cf_group_size = int(getattr(params, "cf_group_size", 2))
        self.cf_min_accepted = int(getattr(params, "cf_min_accepted", self.num_template))
        self.cf_max_disagreement = float(getattr(params, "cf_max_disagreement", 0.075))
        self.cf_verifier = CounterfactualAgreementVerifier(self.num_template)
        self.cf_stats = {"checked": 0, "accepted": 0, "rejected": 0, "last_disagreement": None}

    def initialize(self, image, info: dict, name: str):
        output = super().initialize(image, info, name)
        # Slots start as copies of the anchor.  A counterfactual view is only
        # meaningful after independent accepted updates fill the pool.
        self.cf_accepted_writes = 0
        self.cf_template_pool = AcceptedTemplatePool(self.z_dict1[0], self.num_template)
        self.cf_stats = {"checked": 0, "accepted": 0, "rejected": 0, "last_disagreement": None}
        return output

    def _accept_template(self, template: torch.Tensor, pruning_masks) -> None:
        """Update FARTrack inputs from accepted-write state, never frame index."""
        self.cf_template_pool.accept(template, pruning_masks)
        self.z_dict1, self.mask = self.cf_template_pool.fartrack_inputs()
        self.cf_accepted_writes += 1
        self.cf_stats["accepted"] += 1

    def _sequence_input(self, resize_factor: float) -> torch.Tensor:
        history = []
        for stored_box in self.store_result:
            box = transform_image_to_crop(
                torch.Tensor(stored_box), torch.Tensor(self.state), resize_factor,
                torch.Tensor([self.cfg.TEST.SEARCH_SIZE, self.cfg.TEST.SEARCH_SIZE]), normalize=True)
            box[2] = box[2] + box[0]
            box[3] = box[3] + box[1]
            history.append(((box.clamp(min=-0.5, max=1.5) + 0.5) * (self.bins - 1)))
        return torch.cat(history, dim=-1).unsqueeze(0)

    def _decode_box(self, out_dict: dict) -> torch.Tensor:
        pred_boxes = (out_dict["seqs"][:, 0:4] + 0.5) / (self.bins - 1) - 0.5
        pred = out_dict["feat"][0:4, :, 0:self.bins * self.range]
        distribution = pred.softmax(-1)
        bin_values = torch.linspace(
            -self.range * 0.5 + 0.5,
            self.range * 0.5 + 0.5 - 1 / (self.bins * self.range),
            self.bins * self.range, device=pred.device, dtype=pred.dtype)
        expectation = (distribution * bin_values).sum(dim=-1).permute(1, 0)
        return ((expectation + pred_boxes) / 2).view(-1, 4).mean(dim=0)

    def _counterfactual_agreement(self, search_tensor: torch.Tensor, seq_input: torch.Tensor) -> torch.Tensor:
        old_slots = tuple(range(self.cf_group_size))
        recent_slots = tuple(range(self.num_template - self.cf_group_size, self.num_template))
        old_mask = self.cf_verifier.retain_only(self.mask, old_slots)
        recent_mask = self.cf_verifier.retain_only(self.mask, recent_slots)
        old_out = self.network.forward(template=self.z_dict1, search=search_tensor,
                                       ce_template_mask=self.box_mask_z, seq_input=seq_input,
                                       stage="inference", search_feature=None, mask=old_mask)
        recent_out = self.network.forward(template=self.z_dict1, search=search_tensor,
                                          ce_template_mask=self.box_mask_z, seq_input=seq_input,
                                          stage="inference", search_feature=None, mask=recent_mask)
        return self.cf_verifier.normalized_l1(self._decode_box(old_out), self._decode_box(recent_out))

    def _update_history(self) -> None:
        if len(self.store_result) < self.prenum:
            self.store_result.append(self.state.copy())
        else:
            self.store_result = self.store_result[1:] + [self.state.copy()]

    def track(self, image, info: dict = None):
        height, width, _ = image.shape
        self.frame_id += 1
        search_patch, resize_factor, search_mask = sample_target(
            image, self.state, self.params.search_factor, output_sz=self.params.search_size)
        search = self.preprocessor.process(search_patch, search_mask)
        seq_input = self._sequence_input(resize_factor)

        with torch.no_grad():
            main_out = self.network.forward(template=self.z_dict1, search=search.tensors,
                                            ce_template_mask=self.box_mask_z, seq_input=seq_input,
                                            stage="inference", search_feature=None, mask=self.mask)
            normalized_box = self._decode_box(main_out)
            should_write = True
            disagreement = None
            if self.cf_accepted_writes >= self.cf_min_accepted:
                disagreement = self._counterfactual_agreement(search.tensors, seq_input)
                should_write = bool(disagreement.item() <= self.cf_max_disagreement)
                self.cf_stats["checked"] += 1
                self.cf_stats["last_disagreement"] = float(disagreement.item())

        predicted = normalized_box.clone()
        predicted[2] = normalized_box[2] - normalized_box[0]
        predicted[3] = normalized_box[3] - normalized_box[1]
        predicted[0] = normalized_box[0] + predicted[2] / 2
        predicted[1] = normalized_box[1] + predicted[3] / 2
        crop_box = (predicted * self.params.search_size / resize_factor).tolist()
        self.state = clip_box(self.map_box_back(crop_box, resize_factor), height, width, margin=10)

        if should_write:
            template_patch, _, template_mask = sample_target(
                image, self.state, self.params.template_factor, output_sz=self.params.template_size)
            new_template = self.preprocessor.process(template_patch, template_mask).tensors
            self._accept_template(new_template, main_out["mask"])
        elif disagreement is not None:
            self.cf_stats["rejected"] += 1
        self._update_history()

        result = {"target_bbox": self.state}
        if self.save_all_boxes:
            result["all_boxes"] = self.map_box_back_batch(
                predicted.unsqueeze(0) * self.params.search_size / resize_factor, resize_factor).view(-1).tolist()
        return result


def get_tracker_class():
    return FARTrackSparseCounterfactual
