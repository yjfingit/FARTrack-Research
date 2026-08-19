import torch

from lib.test.tracker.stable_anchor_recovery import StableAnchorRecoveryScheduler


def _candidate(value):
    return torch.full((1, 3, 2, 2), float(value))


def _mask():
    return torch.ones(49, dtype=torch.bool)


def _logits(confident=True):
    logits = torch.zeros(4, 1, 16)
    if confident:
        logits[..., 0] = 12.0
    return logits


def test_stable_commits_preserve_immutable_anchor_and_mask_shape():
    scheduler = StableAnchorRecoveryScheduler(5)
    templates, mask = scheduler.initialize(_candidate(0))
    assert len(templates) == 5 and mask.shape == (1, 445, 445)
    scheduler.step(_candidate(1), _mask(), [0, 0, 10, 10], _logits(), 1)
    templates, mask, action = scheduler.step(_candidate(2), _mask(), [1, 0, 10, 10], _logits(), 2)
    assert action == "commit"
    assert torch.equal(scheduler.anchor.template, _candidate(0))
    assert torch.equal(templates[0], _candidate(0))
    assert mask.dtype == torch.bool


def test_persistent_failure_replays_anchor_and_last_accepted_without_writes():
    scheduler = StableAnchorRecoveryScheduler(5, failure_patience=3, recovery_frames=2, cooldown_frames=4)
    scheduler.initialize(_candidate(0))
    scheduler.step(_candidate(1), _mask(), [0, 0, 10, 10], _logits(), 1)
    for frame in (2, 3):
        _, _, action = scheduler.step(_candidate(99), _mask(), [100, 100, 10, 10], _logits(False), frame)
        assert action == "freeze"
    templates, _, action = scheduler.step(_candidate(99), _mask(), [100, 100, 10, 10], _logits(False), 4)
    assert action == "recover"
    assert torch.equal(templates[0], _candidate(0))
    assert any(torch.equal(template, _candidate(1)) for template in templates[1:])
    assert not any(torch.equal(snapshot.template, _candidate(99)) for snapshot in scheduler.accepted)
    _, _, action = scheduler.step(_candidate(100), _mask(), [100, 100, 10, 10], _logits(False), 5)
    assert action == "recover"


def test_recovery_is_bounded_by_cooldown():
    scheduler = StableAnchorRecoveryScheduler(3, failure_patience=2, recovery_frames=1, cooldown_frames=3)
    scheduler.initialize(_candidate(0))
    scheduler.step(_candidate(1), _mask(), [0, 0, 10, 10], _logits(), 1)
    scheduler.step(_candidate(2), _mask(), [50, 50, 10, 10], _logits(False), 2)
    _, _, action = scheduler.step(_candidate(3), _mask(), [50, 50, 10, 10], _logits(False), 3)
    assert action == "recover"
    _, _, action = scheduler.step(_candidate(4), _mask(), [50, 50, 10, 10], _logits(False), 4)
    assert action == "freeze"
