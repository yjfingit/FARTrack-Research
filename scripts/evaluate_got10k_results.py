#!/usr/bin/env python3
"""Read-only GOT-10k validation AO and coverage evaluator.

This script consumes files emitted by an upstream tracker.  It never imports
or changes the tracker, annotations, benchmark list, or official harness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def iou_xywh(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction_end = prediction[:, :2] + np.maximum(prediction[:, 2:], 0.0)
    target_end = target[:, :2] + np.maximum(target[:, 2:], 0.0)
    intersection = np.maximum(0.0, np.minimum(prediction_end, target_end) - np.maximum(prediction[:, :2], target[:, :2])).prod(axis=1)
    union = np.maximum(prediction[:, 2:], 0.0).prod(axis=1) + np.maximum(target[:, 2:], 0.0).prod(axis=1) - intersection
    return intersection / np.maximum(union, 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    list_path = args.data_root / "list.txt"
    # Official GOT-10k releases occur both with a list.txt manifest and as a
    # val/ directory containing sequence folders.  In the latter layout the
    # directory names are the immutable public sequence manifest.
    names = (
        list_path.read_text().splitlines()
        if list_path.is_file()
        else sorted(path.name for path in args.data_root.iterdir() if (path / "groundtruth.txt").is_file())
    )
    if not names:
        raise ValueError(f"no GOT-10k sequences found under {args.data_root}")
    per_sequence_ao: dict[str, float] = {}
    all_overlaps: list[np.ndarray] = []
    all_times: list[np.ndarray] = []
    for name in names:
        prediction_path = args.results_root / f"{name}.txt"
        target_path = args.data_root / name / "groundtruth.txt"
        time_path = args.results_root / f"{name}_time.txt"
        if not prediction_path.is_file() or not time_path.is_file():
            raise FileNotFoundError(f"missing tracker output for {name}")
        prediction = np.loadtxt(prediction_path, delimiter="\t", ndmin=2).astype(np.float64)
        target = np.loadtxt(target_path, delimiter=",", ndmin=2).astype(np.float64)
        elapsed = np.loadtxt(time_path, delimiter="\t", ndmin=1).astype(np.float64).reshape(-1)
        if len(prediction) != len(target) or len(elapsed) != len(target):
            raise ValueError(f"{name}: predictions={len(prediction)}, times={len(elapsed)}, annotations={len(target)}")
        overlaps = iou_xywh(prediction, target)
        per_sequence_ao[name] = float(overlaps.mean())
        all_overlaps.append(overlaps)
        all_times.append(elapsed)

    frame_count = sum(len(item) for item in all_overlaps)
    elapsed_seconds = float(np.concatenate(all_times).sum())
    report = {
        "metric": "GOT-10k validation AO (mean frame IoU)",
        "num_sequences": len(per_sequence_ao),
        "num_frames": int(frame_count),
        "ao": float(np.concatenate(all_overlaps).mean()),
        "elapsed_seconds": elapsed_seconds,
        "fps": float(frame_count / elapsed_seconds) if elapsed_seconds > 0 else None,
        "per_sequence_ao": per_sequence_ao,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in report if key != "per_sequence_ao"}, indent=2))


if __name__ == "__main__":
    main()
