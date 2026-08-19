#!/usr/bin/env python3
"""Smoke-test CTQA-trained and released baseline checkpoint loading.

This verifier creates only synthetic tensors.  It never constructs a dataset,
reads B_dev, or invokes the official evaluator.
"""

import argparse
import copy
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from lib.config.fartrack_sparse.config import cfg, update_config_from_file
from lib.models.fartrack_sparse import build_fartrack_sparse


CTQA_CONFIG = os.path.join(ROOT, "experiments", "fartrack_sparse", "fartrack_sparse_224_ctqa_bdev.yaml")
RELEASED_CHECKPOINT = "/root/autodl-tmp/experiment/.research-assets/checkpoints/FARTrackSparse_ep0015.pth.tar"


def make_config(enabled):
    local_cfg = copy.deepcopy(cfg)
    update_config_from_file(CTQA_CONFIG, base_cfg=local_cfg)
    local_cfg.MODEL.TRAJECTORY_QUERY_ADAPTER.ENABLED = enabled
    return local_cfg


def strict_load(cfg_obj, checkpoint):
    model = build_fartrack_sparse(cfg_obj, training=False)
    state = torch.load(checkpoint, map_location="cpu")["net"]
    model.load_state_dict(state, strict=True)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="trained CTQA checkpoint containing adapter weights")
    parser.add_argument("--released-checkpoint", default=RELEASED_CHECKPOINT)
    args = parser.parse_args()
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)
    if not os.path.isfile(args.released_checkpoint):
        raise FileNotFoundError(args.released_checkpoint)

    ctqa = strict_load(make_config(True), args.checkpoint).eval()
    baseline = strict_load(make_config(False), args.released_checkpoint).eval()
    if ctqa.backbone.trajectory_query_adapter is None:
        raise RuntimeError("CTQA configuration did not instantiate its adapter")
    if baseline.backbone.trajectory_query_adapter is not None:
        raise RuntimeError("Released-checkpoint baseline unexpectedly enabled CTQA")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctqa.to(device)
    with torch.no_grad():
        template = tuple(torch.randn(1, 3, 112, 112, device=device) for _ in range(5))
        output = ctqa(
            template=template,
            search=torch.randn(1, 3, 224, 224, device=device),
            seq_input=torch.randint(0, 600, (1, 12), device=device),
            mask=torch.ones(1, 445, 445, dtype=torch.bool, device=device),
            stage="inference",
        )
    if output["seqs"].shape != (1, 4):
        raise RuntimeError(f"Unexpected CTQA output shape: {tuple(output['seqs'].shape)}")
    print("PASS: trained CTQA loads strictly with adapter enabled; released baseline loads strictly with CTQA disabled")


if __name__ == "__main__":
    main()
