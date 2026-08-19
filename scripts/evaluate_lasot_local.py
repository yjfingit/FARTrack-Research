#!/usr/bin/env python3
"""Read-only LaSOT Protocol-II summary for an existing tracker result folder."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.test.analysis.extract_results import calc_seq_err_robust
from lib.test.evaluation import get_dataset
from lib.test.utils.load_text import load_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = get_dataset("lasot")
    overlap_curves, precision_curves, norm_precision_curves, per_sequence = [], [], [], {}
    overlap_thresholds = torch.arange(0.0, 1.05, 0.05, dtype=torch.float64)
    center_thresholds = torch.arange(0, 51, dtype=torch.float64)
    norm_thresholds = torch.arange(0, 0.51, 0.01, dtype=torch.float64)
    for sequence in dataset:
        result_path = args.results_root / f"{sequence.name}.txt"
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        prediction = torch.tensor(load_text(str(result_path), delimiter=("\t", ","), dtype=np.float64))
        ground_truth = torch.tensor(sequence.ground_truth_rect)
        if prediction.shape[0] != ground_truth.shape[0]:
            raise ValueError(f"length mismatch for {sequence.name}: {prediction.shape[0]} != {ground_truth.shape[0]}")
        visible = torch.tensor(sequence.target_visible, dtype=torch.uint8)
        overlap, center, norm_center, valid = calc_seq_err_robust(prediction, ground_truth, "lasot", visible)
        overlap_curve = (overlap[:, None] > overlap_thresholds[None, :]).float().mean(0)
        precision_curve = (center[:, None] <= center_thresholds[None, :]).float().mean(0)
        norm_precision_curve = (norm_center[:, None] <= norm_thresholds[None, :]).float().mean(0)
        overlap_curves.append(overlap_curve)
        precision_curves.append(precision_curve)
        norm_precision_curves.append(norm_precision_curve)
        per_sequence[sequence.name] = float(overlap[valid.bool()].mean())

    success_curve = torch.stack(overlap_curves).mean(0)
    precision_curve = torch.stack(precision_curves).mean(0)
    norm_precision_curve = torch.stack(norm_precision_curves).mean(0)
    result = {
        "protocol": "LaSOT Protocol-II local success/precision implementation",
        "num_sequences": len(dataset),
        "success_auc": float(success_curve.mean()),
        "precision_at_20px": float(precision_curve[20]),
        "normalized_precision_auc": float(norm_precision_curve.mean()),
        "per_sequence_visible_frame_iou": per_sequence,
        "result_root": str(args.results_root),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "per_sequence_visible_frame_iou"}, indent=2))


if __name__ == "__main__":
    main()
