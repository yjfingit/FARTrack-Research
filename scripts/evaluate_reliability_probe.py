"""Evaluate passive FARTrack reliability logs against public GOT-10k val GT."""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_got10k_val import iou_xywh


SIGNALS = ("coord_entropy", "coord_margin", "branch_disagreement_l1")


def average_precision(labels, scores):
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    precision = np.cumsum(labels) / (np.arange(len(labels)) + 1)
    return float((precision * labels).sum() / positives)


def auroc(labels, scores):
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return float((ranks[labels].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def rank_correlation(x, y):
    def ranks(values):
        order = np.argsort(values, kind="mergesort")
        out = np.empty(len(values), dtype=np.float64)
        sorted_values = values[order]
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and sorted_values[end] == sorted_values[start]:
                end += 1
            out[order[start:end]] = (start + end - 1) / 2.0
            start = end
        return out
    rx, ry = ranks(x), ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def metric_bundle(rows, signal):
    labels = np.array([row["low_iou"] for row in rows], dtype=bool)
    iou = np.array([row["iou"] for row in rows], dtype=np.float64)
    raw = np.array([row[signal] for row in rows], dtype=np.float64)
    risk = -raw if signal == "coord_margin" else raw
    return {
        "spearman_with_iou": rank_correlation(raw, iou),
        "low_iou_auroc": auroc(labels, risk),
        "low_iou_average_precision": average_precision(labels, risk),
    }


def sequence_bootstrap(per_sequence_rows, signal, samples, seed):
    """Sequence-resampled 95% intervals, preserving within-video dependence."""
    names = sorted(per_sequence_rows)
    rng = np.random.default_rng(seed)
    values = {key: [] for key in ("spearman_with_iou", "low_iou_auroc", "low_iou_average_precision")}
    for _ in range(samples):
        sampled_rows = []
        for index in rng.integers(0, len(names), size=len(names)):
            sampled_rows.extend(per_sequence_rows[names[index]])
        for key, value in metric_bundle(sampled_rows, signal).items():
            values[key].append(value)
    return {
        key: [float(np.nanquantile(value, 0.025)), float(np.nanquantile(value, 0.975))]
        for key, value in values.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--log-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--low-iou", type=float, default=0.2)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=20260820)
    args = parser.parse_args()

    all_rows = []
    per_sequence = {}
    per_sequence_rows = {}
    for name in (args.data_root / "list.txt").read_text().splitlines():
        log_path = args.log_root / f"{name}.jsonl"
        prediction_path = args.prediction_root / "got10k" / f"{name}.txt"
        if not log_path.exists() or not prediction_path.exists():
            continue
        logs = [json.loads(line) for line in log_path.read_text().splitlines() if line]
        gt = np.loadtxt(args.data_root / name / "groundtruth.txt", delimiter=",", ndmin=2)
        prediction = np.loadtxt(prediction_path, delimiter="\t", ndmin=2)
        # Initial GT box is not logged by track(); all remaining frames must align exactly.
        if len(logs) != len(gt) - 1 or len(prediction) != len(gt):
            raise ValueError(f"{name}: logs={len(logs)}, predictions={len(prediction)}, gt={len(gt)}")
        iou = iou_xywh(prediction[1:].astype(np.float64), gt[1:].astype(np.float64))
        rows = []
        for log, value in zip(logs, iou):
            row = dict(log)
            row["sequence"] = name
            row["iou"] = float(value)
            row["low_iou"] = bool(value < args.low_iou)
            rows.append(row)
        all_rows.extend(rows)
        per_sequence_rows[name] = rows
        per_sequence[name] = {"frames": len(rows), "mean_iou": float(iou.mean())}

    if not all_rows:
        raise ValueError("No aligned logs/predictions found")
    labels = np.array([row["low_iou"] for row in all_rows], dtype=bool)
    iou = np.array([row["iou"] for row in all_rows], dtype=np.float64)
    metrics = {}
    for signal in SIGNALS:
        metrics[signal] = metric_bundle(all_rows, signal)
        metrics[signal]["sequence_bootstrap_95ci"] = sequence_bootstrap(
            per_sequence_rows, signal, args.bootstrap_samples, args.bootstrap_seed
        )
    report = {
        "protocol": "Public GOT-10k validation only; passive logging after the frozen forward pass.",
        "low_iou_definition": f"IoU < {args.low_iou}",
        "bootstrap": {
            "unit": "sequence",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
        "num_sequences": len(per_sequence),
        "num_logged_frames": len(all_rows),
        "low_iou_prevalence": float(labels.mean()),
        "metrics": metrics,
        "per_sequence": per_sequence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in report if key != "per_sequence"}, indent=2))


if __name__ == "__main__":
    main()
