import torch

from lib.test.tracker.appearance_reservoir import AppearanceDiverseReservoir


def _template(value):
    return torch.full((1, 3, 112, 112), float(value))


def test_every_frame_write_and_five_slot_binding():
    reservoir = AppearanceDiverseReservoir(capacity=4)
    reservoir.initialize(_template(0))
    for frame in range(1, 9):
        reservoir.write(_template(frame), torch.ones(1, 49, dtype=torch.bool), frame)
    templates, mask, frame_ids = reservoir.bind()
    assert reservoir.writes == 8
    assert reservoir.replacements == 4
    assert len(templates) == 5
    assert mask.shape == (1, 445, 445)
    assert frame_ids[0] == 0 and frame_ids[-1] == 8


def test_middle_slots_follow_chronology_after_max_min_selection():
    reservoir = AppearanceDiverseReservoir(capacity=4)
    reservoir.initialize(_template(0))
    for frame, value in enumerate((0.1, 0.9, 0.2, 0.8, 0.4, 0.7), start=1):
        reservoir.write(_template(value), torch.ones(1, 49, dtype=torch.bool), frame)
    _, _, frame_ids = reservoir.bind()
    assert frame_ids[0] == 0 and frame_ids[-1] == 6
    assert frame_ids[1:4] == sorted(frame_ids[1:4])
    assert len(set(frame_ids[1:4])) == 3


def test_anchor_mask_stays_dense_and_dynamic_mask_is_carried():
    reservoir = AppearanceDiverseReservoir(capacity=4)
    reservoir.initialize(_template(0))
    sparse = torch.zeros(1, 49, dtype=torch.bool)
    sparse[:, ::2] = True
    reservoir.write(_template(1), sparse, 1)
    _, mask, frame_ids = reservoir.bind()
    valid = mask[0, 0]
    assert valid[:49].all()
    assert torch.equal(valid[4 * 49:5 * 49], sparse[0])
    assert frame_ids[-1] == 1
