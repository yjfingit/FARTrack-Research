#!/usr/bin/env python3
"""Verify frozen-checkpoint parity and the first CTQA optimizer update.

This synthetic test does not access tracking datasets.  It is intended to run
before any training launch and proves both checkpoint compatibility and that a
zero-initialized adapter is not a dead branch.
"""

import copy
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from lib.config.fartrack_sparse.config import cfg, update_config_from_file
from lib.models.fartrack_sparse import build_fartrack_sparse


CONFIG = os.path.join(ROOT, "experiments", "fartrack_sparse", "fartrack_sparse_224_ctqa_got10k_train.yaml")


def build(enabled):
    local_cfg = copy.deepcopy(cfg)
    update_config_from_file(CONFIG, base_cfg=local_cfg)
    local_cfg.MODEL.TRAJECTORY_QUERY_ADAPTER.ENABLED = enabled
    return build_fartrack_sparse(local_cfg, training=True).eval()


def main():
    torch.manual_seed(18)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline = build(False).to(device)
    ctqa = build(True).to(device)
    baseline_state = baseline.state_dict()
    for name, value in ctqa.state_dict().items():
        if "trajectory_query_adapter" not in name and not torch.equal(value.cpu(), baseline_state[name].cpu()):
            raise RuntimeError(f"Unexpected non-CTQA checkpoint mismatch: {name}")
    # Both builds load the same immutable checkpoint.  The only missing keys
    # in the CTQA build must be the newly introduced adapter parameters.
    batch = 1
    template = tuple(torch.randn(batch, 3, 112, 112, device=device) for _ in range(5))
    search = torch.randn(batch, 3, 224, 224, device=device)
    history = torch.randint(0, 600, (batch, 12), device=device)
    mask = torch.ones(batch, 445, 445, dtype=torch.bool, device=device)
    common = dict(template=template, search=search, seq_input=history, mask=mask, stage="train")

    with torch.no_grad():
        reference = baseline(**common)["feat"]
        repeat_reference = baseline(**common)["feat"]
        initial = ctqa(**common)["feat"]
    torch.testing.assert_close(repeat_reference, reference, rtol=0, atol=0)
    torch.testing.assert_close(initial, reference, rtol=0, atol=0)

    ctqa.train()
    adapter = ctqa.backbone.trajectory_query_adapter
    optimizer = torch.optim.SGD(adapter.parameters(), lr=1e-2)
    optimizer.zero_grad()
    loss = ctqa(**common)["feat"].square().mean()
    loss.backward()
    if adapter.output.weight.grad is None or not torch.count_nonzero(adapter.output.weight.grad):
        raise RuntimeError("CTQA final projection received no first-step gradient")
    optimizer.step()
    ctqa.eval()
    with torch.no_grad():
        updated = ctqa(**common)["feat"]
    if torch.equal(initial, updated):
        raise RuntimeError("CTQA output did not change after an optimizer update")
    print("PASS: exact frozen-checkpoint parity at initialization; CTQA diverges after one update")


if __name__ == "__main__":
    main()
