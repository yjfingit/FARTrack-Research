import numpy as np
import torch

from lib.test.tracker.transactional_memory import TransactionalTemplateLedger


def _logits(certainty):
    logits = torch.zeros(4, 1, 16)
    if certainty:
        logits[:, :, 0] = 20.0
    return logits


def test_commits_and_packs_five_templates():
    ledger = TransactionalTemplateLedger(5, history=4, min_commits=2)
    anchor = torch.zeros(1, 3, 112, 112)
    ledger.initialize(anchor, anchor.device)
    templates, mask, action = ledger.update(torch.ones_like(anchor), torch.ones(1, 49, dtype=torch.bool),
                                            [10, 10, 20, 20], _logits(True), 1)
    assert action == "commit"
    assert len(templates) == 5
    assert mask.shape == (1, 445, 445)
    assert ledger.stats["commit"] == 1


def test_sustained_multisignal_failure_holds_then_rolls_back():
    ledger = TransactionalTemplateLedger(3, history=4, min_commits=2, suspect_patience=1, lost_patience=2)
    anchor = torch.zeros(1, 3, 112, 112)
    ledger.initialize(anchor, anchor.device)
    good_mask = torch.ones(1, 49, dtype=torch.bool)
    for frame in range(1, 4):
        ledger.update(torch.full_like(anchor, frame), good_mask, [frame, 0, 10, 10], _logits(True), frame)
    bad_mask = torch.zeros(1, 49, dtype=torch.bool)
    _, _, action = ledger.update(torch.ones_like(anchor), bad_mask, [200, 200, 2, 2], _logits(False), 4)
    assert action == "hold"
    _, _, action = ledger.update(torch.ones_like(anchor), bad_mask, [300, 300, 2, 2], _logits(False), 5)
    assert action == "rollback"
    assert ledger.state == ledger.LOST
    assert ledger.stats["rollback"] == 1


def test_anchor_never_evicted():
    ledger = TransactionalTemplateLedger(2, min_commits=1)
    anchor = torch.zeros(1, 3, 112, 112)
    ledger.initialize(anchor, anchor.device)
    for frame in range(1, 6):
        ledger.update(torch.full_like(anchor, frame), torch.ones(1, 49, dtype=torch.bool),
                      [frame, 0, 10, 10], _logits(True), frame)
    assert ledger.snapshots[0].template.data_ptr() == anchor.data_ptr() or torch.equal(ledger.snapshots[0].template, anchor)
