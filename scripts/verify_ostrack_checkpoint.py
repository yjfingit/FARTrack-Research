#!/usr/bin/env python3
"""Strictly validate an OSTrack checkpoint without accessing a dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lib.config.ostrack.config import cfg, update_config_from_file
from lib.models.ostrack import build_ostrack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    update_config_from_file(args.config)
    model = build_ostrack(cfg, training=False)
    state = torch.load(args.checkpoint, map_location="cpu")["net"]
    model.load_state_dict(state, strict=True)
    model = model.to(args.device).eval()
    with torch.no_grad():
        result = model(
            template=torch.zeros(1, 3, cfg.TEST.TEMPLATE_SIZE, cfg.TEST.TEMPLATE_SIZE, device=args.device),
            search=torch.zeros(1, 3, cfg.TEST.SEARCH_SIZE, cfg.TEST.SEARCH_SIZE, device=args.device),
        )
    output_shapes = {key: list(value.shape) for key, value in result.items() if hasattr(value, "shape")}
    print(
        json.dumps(
            {
                "strict_load": True,
                "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else args.device,
                "config": str(args.config),
                "checkpoint": str(args.checkpoint),
                "outputs": output_shapes,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
