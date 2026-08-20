#!/usr/bin/env python3
"""Read-only analysis of opt-in OSTrack response-risk records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def iou_xywh(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction_end = prediction[:, :2] + np.maximum(prediction[:, 2:], 0.0)
    target_end = target[:, :2] + np.maximum(target[:, 2:], 0.0)
    intersection = np.maximum(0.0, np.minimum(prediction_end, target_end) - np.maximum(prediction[:, :2], target[:, :2])).prod(axis=1)
    union = np.maximum(prediction[:, 2:], 0.0).prod(axis=1) + np.maximum(target[:, 2:], 0.0).prod(axis=1) - intersection
    return intersection / np.maximum(union, 1e-12)


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float('nan')
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float((ranks[labels].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return float('nan')
    order = np.argsort(-scores, kind='mergesort')
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / positives)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_segments(paths: list[Path]) -> list[list[dict]]:
    records: list[dict] = []
    for path in paths:
        records.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    segments: list[list[dict]] = []
    current: list[dict] = []
    for record in records:
        if record['frame'] == 1 and current:
            segments.append(current)
            current = []
        expected = len(current) + 1
        if record['frame'] != expected:
            raise ValueError(f"nonconsecutive risk log frame {record['frame']} expected {expected}")
        current.append(record)
    if current:
        segments.append(current)
    return segments


def summary(sequence_rows: list[dict]) -> dict[str, float]:
    labels = np.concatenate([row['low_iou'] for row in sequence_rows])
    risk = np.concatenate([row['risk'] for row in sequence_rows])
    ious = np.concatenate([row['iou'] for row in sequence_rows])
    return {
        'auroc': auroc(labels, risk),
        'average_precision': average_precision(labels, risk),
        'spearman_risk_iou': float(spearmanr(risk, ious).statistic),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--baseline-results', required=True, type=Path)
    parser.add_argument('--risk-results', required=True, type=Path)
    parser.add_argument('--risk-log', required=True, action='append', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=20260820)
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()

    names = sorted(path.name for path in args.data_root.iterdir() if (path / 'groundtruth.txt').is_file())
    segments = read_segments(args.risk_log)
    if len(segments) != len(names):
        raise ValueError(f"risk segments={len(segments)} but sequences={len(names)}")

    rows: list[dict] = []
    parity_mismatches: list[str] = []
    for name, records in zip(names, segments):
        baseline_path = args.baseline_results / f'{name}.txt'
        risk_path = args.risk_results / f'{name}.txt'
        if not baseline_path.is_file() or not risk_path.is_file():
            raise FileNotFoundError(name)
        if sha256(baseline_path) != sha256(risk_path):
            parity_mismatches.append(name)
        target = np.loadtxt(args.data_root / name / 'groundtruth.txt', delimiter=',', ndmin=2).astype(np.float64)
        prediction = np.loadtxt(baseline_path, delimiter='\t', ndmin=2).astype(np.float64)
        if len(prediction) != len(target) or len(records) != len(target) - 1:
            raise ValueError(f'{name}: prediction={len(prediction)}, target={len(target)}, risk={len(records)}')
        ious = iou_xywh(prediction[1:], target[1:])
        # Larger entropy is the pre-registered high-risk direction.
        risk = np.asarray([record['response_entropy'] for record in records], dtype=np.float64)
        rows.append({'name': name, 'iou': ious, 'risk': risk, 'low_iou': ious < 0.2})

    measured = summary(rows)
    rng = np.random.default_rng(args.seed)
    bootstrap = {key: [] for key in measured}
    for _ in range(args.bootstrap):
        sample = [rows[index] for index in rng.integers(0, len(rows), size=len(rows))]
        value = summary(sample)
        for key, metric in value.items():
            if np.isfinite(metric):
                bootstrap[key].append(metric)
    ci = {key: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] for key, values in bootstrap.items()}
    report = {
        'metric': 'passive low-IoU risk prediction; low IoU is IoU < 0.2',
        'num_sequences': len(rows),
        'num_tracked_frames': int(sum(len(row['iou']) for row in rows)),
        'low_iou_prevalence': float(np.concatenate([row['low_iou'] for row in rows]).mean()),
        'response_entropy_high_risk': measured,
        'sequence_bootstrap_95_ci': ci,
        'parity': {'baseline_vs_risk_tracker_sha256_mismatches': parity_mismatches, 'match_all_sequences': not parity_mismatches},
        'bootstrap_seed': args.seed,
        'bootstrap_replicates': args.bootstrap,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
