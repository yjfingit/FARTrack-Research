import torch

from lib.test.tracker.recent_consensus_memory import RecentConsensusMemory


def _template(value):
    return torch.full((1, 3, 112, 112), float(value))


def _colour_template(red):
    template = torch.empty((1, 3, 112, 112))
    template[:, 0].fill_(float(red))
    template[:, 1].fill_(float(1.0 - red))
    template[:, 2].fill_(0.2)
    return template


def _mask(value=True):
    return torch.full((1, 49), value, dtype=torch.bool)


def test_writes_are_exact_and_window_is_fifo_bounded():
    memory = RecentConsensusMemory(window_size=4)
    memory.initialize(_colour_template(0))
    for frame in range(1, 8):
        memory.write(_template(frame), _mask(), frame)
    assert memory.writes == 7
    assert [record.frame_id for record in memory.recent] == [4, 5, 6, 7]


def test_bind_is_anchor_three_temporal_medoids_and_newest():
    memory = RecentConsensusMemory(window_size=12)
    memory.initialize(_template(0))
    # Three clusters split over chronological thirds. Their middle values are medoids.
    for frame, value in enumerate((0.0, 0.1, 0.2, 0.7, 0.8, 0.9, 0.3, 0.4, 0.5, 0.6), start=1):
        memory.write(_colour_template(value), _mask(), frame)
    templates, attention_mask, frame_ids = memory.bind()
    assert len(templates) == 5
    assert frame_ids == [0, 2, 5, 8, 10]
    assert attention_mask.shape == (1, 445, 445)
    assert attention_mask.dtype == torch.bool


def test_mask_alignment_and_warmup_slots_are_valid():
    memory = RecentConsensusMemory(window_size=8)
    memory.initialize(_template(0))
    sparse = _mask(False)
    sparse[:, ::2] = True
    memory.write(_template(1), sparse, 1)
    _, attention_mask, frame_ids = memory.bind()
    valid = attention_mask[0, 0]
    assert frame_ids[-1] == 1
    assert valid[:49].all()
    assert torch.equal(valid[4 * 49:5 * 49], sparse[0])
    assert valid[-200:].all()
