"""Read-only frame-weighted AO evaluator for deterministic node-10 partitions."""

import argparse
import json
from pathlib import Path

import numpy as np


def iou_xywh(prediction, target):
    left = np.maximum(prediction[:, 0], target[:, 0])
    top = np.maximum(prediction[:, 1], target[:, 1])
    right = np.minimum(prediction[:, 0] + prediction[:, 2], target[:, 0] + target[:, 2])
    bottom = np.minimum(prediction[:, 1] + prediction[:, 3], target[:, 1] + target[:, 3])
    intersection = np.maximum(right - left, 0) * np.maximum(bottom - top, 0)
    union = prediction[:, 2] * prediction[:, 3] + target[:, 2] * target[:, 3] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--remainder", required=True, type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    names = [name for name in (args.data_root / "list.txt").read_text().splitlines()
             if int(name.rsplit("_", 1)[1]) % 3 == args.remainder]
    if len(names) != 60:
        raise RuntimeError(f"Expected 60 sequences, found {len(names)}")
    sequence_ao, all_overlaps = {}, []
    for name in names:
        prediction = np.loadtxt(args.results_root / "got10k" / f"{name}.txt", delimiter="\t", ndmin=2)
        target = np.loadtxt(args.data_root / name / "groundtruth.txt", delimiter=",", ndmin=2)
        if len(prediction) != len(target):
            raise ValueError(f"{name}: prediction/annotation lengths differ")
        overlap = iou_xywh(prediction.astype(np.float64), target.astype(np.float64))
        sequence_ao[name] = float(overlap.mean())
        all_overlaps.append(overlap)
    report = {
        "metric": "GOT-10k validation AO (mean frame IoU)",
        "partition": {"id_modulo": 3, "id_remainder": args.remainder},
        "num_sequences": len(names),
        "num_frames": int(sum(len(values) for values in all_overlaps)),
        "ao": float(np.concatenate(all_overlaps).mean()),
        "per_sequence_ao": sequence_ao,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "per_sequence_ao"}, indent=2))


if __name__ == "__main__":
    main()
