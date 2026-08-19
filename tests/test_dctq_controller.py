import torch

from lib.test.tracker.dctq_controller import PRUNE_LEVELS, select_mask, validate_mask_levels


def native_masks():
    # Retained tokens count: 37, 25, 13, 5, mirroring 25/50/75/90% pruning.
    return [torch.tensor([[i < count for i in range(49)]], dtype=torch.bool)
            for count in (37, 25, 13, 5)]


def test_native_mask_shapes_and_nested_retention():
    masks = native_masks()
    validate_mask_levels(masks)
    assert PRUNE_LEVELS == (25, 50, 75, 90)
    assert [int(mask.sum()) for mask in masks] == [37, 25, 13, 5]


def test_disabled_and_low_disagreement_retain_released_25_mask():
    masks = native_masks()
    seq = torch.zeros(1, 4)
    feat = torch.full((1, 4), 0.001)
    disabled, disabled_alarm = select_mask(masks, seq, feat, 0.0, 3, enabled=False)
    low, low_alarm = select_mask(masks, seq, feat, 0.01, 2)
    assert torch.equal(disabled, masks[0]) and not bool(disabled_alarm)
    assert torch.equal(low, masks[0]) and not bool(low_alarm)


def test_high_disagreement_and_static_select_requested_level():
    masks = native_masks()
    seq = torch.zeros(1, 4)
    feat = torch.full((1, 4), 0.2)
    adaptive, alarm = select_mask(masks, seq, feat, 0.1, 2)
    static, static_alarm = select_mask(masks, seq, feat, 99.0, 3, static=True)
    assert torch.equal(adaptive, masks[2]) and bool(alarm)
    assert torch.equal(static, masks[3]) and bool(static_alarm)
