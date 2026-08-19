#!/usr/bin/env python3
"""Associate measured B_dev AO deltas with pre-existing GOT-10k metadata."""

import argparse
import configparser
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def load_labels(path: Path, count: int) -> np.ndarray:
    if not path.is_file():
        return np.zeros(count, dtype=float)
    values = np.loadtxt(path, ndmin=1, dtype=float)
    if values.size != count:
        raise ValueError(f"{path}: {values.size} labels for {count} frames")
    return values


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    # scipy handles the frequent ties in GOT-10k ordinal annotations correctly.
    return float(spearmanr(a, b).statistic)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text())["per_sequence_ao"]
    candidate = json.loads(args.candidate.read_text())["per_sequence_ao"]
    rows = []
    for sequence in sorted(baseline):
        root = args.data_root / sequence
        boxes = np.loadtxt(root / "groundtruth.txt", delimiter=",", ndmin=2)
        count = boxes.shape[0]
        centers = boxes[:, :2] + boxes[:, 2:] / 2
        scale = np.sqrt(np.maximum(boxes[:, 2] * boxes[:, 3], 1.0))
        motion = np.linalg.norm(np.diff(centers, axis=0), axis=1) / np.maximum(scale[:-1], 1.0)
        meta = configparser.ConfigParser()
        meta.read(root / "meta_info.ini")
        rows.append({
            "sequence": sequence,
            "delta_ao": candidate[sequence] - baseline[sequence],
            "frames": count,
            "absence_fraction": float(load_labels(root / "absence.label", count).mean()),
            # GOT-10k cover.label is an ordinal cover annotation, not a binary
            # visibility indicator; retain only its mean label as descriptive.
            "cover_label_mean": float(load_labels(root / "cover.label", count).mean()),
            "cut_fraction": float(load_labels(root / "cut_by_image.label", count).mean()),
            "relative_motion_median": float(np.median(motion)) if motion.size else 0.0,
            "log_scale_variation": float(np.std(np.log(scale))),
            "object_class": meta.get("METAINFO", "object_class", fallback="unknown"),
            "motion_class": meta.get("METAINFO", "motion_class", fallback="unknown"),
        })

    numeric = ["frames", "absence_fraction", "cover_label_mean", "cut_fraction", "relative_motion_median", "log_scale_variation"]
    delta = np.asarray([row["delta_ao"] for row in rows])
    correlations = {name: rank_correlation(delta, np.asarray([row[name] for row in rows])) for name in numeric}
    median_delta = float(np.median(delta))
    output = {
        "dataset": "GOT-10k validation B_dev",
        "num_sequences": len(rows),
        "candidate": str(args.candidate),
        "baseline": str(args.baseline),
        "median_delta_ao": median_delta,
        "spearman_rank_correlation_with_delta_ao": correlations,
        "top_improvements": sorted(rows, key=lambda row: row["delta_ao"], reverse=True)[:10],
        "top_degradations": sorted(rows, key=lambda row: row["delta_ao"])[:10],
        "per_sequence": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in output if key != "per_sequence"}, indent=2))


if __name__ == "__main__":
    main()
