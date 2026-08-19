"""Read-only GOT-10k validation AO evaluator for FARTrack result files.

The official test harness produces integer xywh tracks. This script reads those
tracks and the unmodified public validation annotations; it does not alter
results, annotations, or the harness. AO is the mean frame IoU, matching the
GOT-10k validation protocol's primary overlap metric.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def iou_xywh(prediction, target):
    p1 = prediction[:, :2]
    p2 = p1 + np.maximum(prediction[:, 2:], 0.0)
    t1 = target[:, :2]
    t2 = t1 + np.maximum(target[:, 2:], 0.0)
    intersection = np.maximum(0.0, np.minimum(p2, t2) - np.maximum(p1, t1)).prod(axis=1)
    union = np.maximum(prediction[:, 2:], 0.0).prod(axis=1) + target[:, 2:].prod(axis=1) - intersection
    return intersection / np.maximum(union, 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data_root, results_root = Path(args.data_root), Path(args.results_root)
    names = (data_root / "list.txt").read_text().splitlines()
    per_sequence = {}
    all_overlaps = []
    for name in names:
        prediction_path = results_root / "got10k" / f"{name}.txt"
        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)
        target = np.loadtxt(data_root / name / "groundtruth.txt", delimiter=",", ndmin=2)
        prediction = np.loadtxt(prediction_path, delimiter="\t", ndmin=2)
        if len(prediction) != len(target):
            raise ValueError(f"{name}: {len(prediction)} predictions != {len(target)} annotations")
        overlaps = iou_xywh(prediction.astype(np.float64), target.astype(np.float64))
        per_sequence[name] = float(overlaps.mean())
        all_overlaps.append(overlaps)
    report = {
        "metric": "GOT-10k validation AO (mean frame IoU)",
        "num_sequences": len(per_sequence),
        "num_frames": int(sum(len(item) for item in all_overlaps)),
        "ao": float(np.concatenate(all_overlaps).mean()),
        "per_sequence_ao": per_sequence,
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in report if key != "per_sequence_ao"}, indent=2))


if __name__ == "__main__":
    main()
