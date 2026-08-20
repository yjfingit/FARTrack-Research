#!/usr/bin/env python3
"""Render fixed-rule qualitative cases from retained GOT-10k B_dev results.

This utility is deliberately read-only: it consumes public validation images,
ground truth, and existing result files.  For each supplied case it selects the
frame with the smallest candidate-minus-baseline IoU, then draws GT (green),
baseline (blue), and candidate (red).  It neither invokes a tracker nor writes
into a benchmark result directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle


@dataclass(frozen=True)
class Case:
    method: str
    sequence: str
    candidate_dir: Path


def parse_box(line: str) -> np.ndarray:
    values = [float(value) for value in line.replace("\t", ",").split(",") if value]
    if len(values) != 4:
        raise ValueError(f"Expected four box values, got {line!r}")
    return np.asarray(values, dtype=np.float64)


def read_boxes(path: Path) -> np.ndarray:
    return np.asarray([parse_box(line) for line in path.read_text().splitlines() if line.strip()])


def iou_xywh(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    ax1, ay1 = boxes_a[:, 0], boxes_a[:, 1]
    ax2, ay2 = ax1 + np.maximum(boxes_a[:, 2], 0), ay1 + np.maximum(boxes_a[:, 3], 0)
    bx1, by1 = boxes_b[:, 0], boxes_b[:, 1]
    bx2, by2 = bx1 + np.maximum(boxes_b[:, 2], 0), by1 + np.maximum(boxes_b[:, 3], 0)
    inter_w = np.maximum(0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
    inter_h = np.maximum(0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    intersection = inter_w * inter_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def draw_box(axis: plt.Axes, box: np.ndarray, color: str) -> None:
    axis.add_patch(
        Rectangle((box[0], box[1]), box[2], box[3], fill=False, edgecolor=color, linewidth=2.2)
    )


def resolve_case(spec: str, output_root: Path) -> Case:
    method, sequence, tracker_name = spec.split(":")
    return Case(method, sequence, output_root / tracker_name / "fartrack_sparse_224_full" / "got10k")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help="METHOD:SEQUENCE:TRACKER_DIRECTORY_NAME; may be repeated.",
    )
    args = parser.parse_args()
    cases = [resolve_case(spec, args.output_root) for spec in args.case]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, case in zip(axes.flat, cases):
        sequence_dir = args.data_root / case.sequence
        ground_truth = read_boxes(sequence_dir / "groundtruth.txt")
        baseline = read_boxes(args.baseline_dir / f"{case.sequence}.txt")
        candidate = read_boxes(case.candidate_dir / f"{case.sequence}.txt")
        frame_count = min(len(ground_truth), len(baseline), len(candidate))
        if frame_count == 0:
            raise ValueError(f"No aligned frames for {case.method}/{case.sequence}")
        ground_truth, baseline, candidate = (array[:frame_count] for array in (ground_truth, baseline, candidate))
        base_iou = iou_xywh(baseline, ground_truth)
        candidate_iou = iou_xywh(candidate, ground_truth)
        delta = candidate_iou - base_iou
        frame_index = int(np.argmin(delta))
        image_path = sequence_dir / f"{frame_index + 1:08d}.jpg"
        image = Image.open(image_path).convert("RGB")
        axis.imshow(image)
        draw_box(axis, ground_truth[frame_index], "#30a14e")
        draw_box(axis, baseline[frame_index], "#0969da")
        draw_box(axis, candidate[frame_index], "#cf222e")
        axis.set_title(
            f"{case.method} | {case.sequence} | frame {frame_index + 1}\n"
            f"IoU: GT/base {base_iou[frame_index]:.3f}, GT/candidate {candidate_iou[frame_index]:.3f}, "
            f"delta {delta[frame_index]:+.3f}",
            fontsize=9,
        )
        axis.axis("off")
        records.append(
            {
                "method": case.method,
                "sequence": case.sequence,
                "selection_rule": "argmin_frame(candidate_iou - baseline_iou)",
                "frame_index_zero_based": frame_index,
                "frame_file": image_path.name,
                "baseline_iou": float(base_iou[frame_index]),
                "candidate_iou": float(candidate_iou[frame_index]),
                "delta_iou": float(delta[frame_index]),
            }
        )
    for axis in axes.flat[len(cases) :]:
        axis.axis("off")
    figure.legend(
        handles=[
            Rectangle((0, 0), 1, 1, fill=False, edgecolor="#30a14e", label="Ground truth"),
            Rectangle((0, 0), 1, 1, fill=False, edgecolor="#0969da", label="Immutable baseline"),
            Rectangle((0, 0), 1, 1, fill=False, edgecolor="#cf222e", label="Candidate"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    figure.savefig(args.output_dir / "failure_cases.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    (args.output_dir / "failure_cases.json").write_text(json.dumps(records, indent=2) + "\n")
    with (args.output_dir / "failure_cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
