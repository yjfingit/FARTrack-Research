"""Read-only conditional action-advantage analysis for retained B_dev outputs.

This script does not run a tracker or modify data/evaluation code. It aligns
the byte-parity Node 7 baseline prediction files, Node 7 reliability logs, the
valid Node 17 CSPP selection outputs, and public GOT-10k validation ground
truth. Risk thresholds are estimated only from the calibration partition.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def sequence_number(name):
    return int(name.rsplit("_", 1)[1])


def iou_xywh(first, second):
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    first_right = first[:, 0] + np.maximum(0.0, first[:, 2])
    first_bottom = first[:, 1] + np.maximum(0.0, first[:, 3])
    second_right = second[:, 0] + np.maximum(0.0, second[:, 2])
    second_bottom = second[:, 1] + np.maximum(0.0, second[:, 3])
    left = np.maximum(first[:, 0], second[:, 0])
    top = np.maximum(first[:, 1], second[:, 1])
    right = np.minimum(first_right, second_right)
    bottom = np.minimum(first_bottom, second_bottom)
    intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
    union = np.maximum(0.0, first[:, 2]) * np.maximum(0.0, first[:, 3])
    union += np.maximum(0.0, second[:, 2]) * np.maximum(0.0, second[:, 3]) - intersection
    return intersection / np.maximum(union, 1e-12)


def load_boxes(path):
    return np.loadtxt(path, delimiter="\t", ndmin=2, dtype=np.float64)


def load_rows(data_root, baseline_root, candidate_root, log_root, selector):
    rows = {}
    for name in (data_root / "list.txt").read_text(encoding="utf-8").splitlines():
        if not selector(sequence_number(name)):
            continue
        paths = {
            "baseline": baseline_root / f"{name}.txt",
            "candidate": candidate_root / f"{name}.txt",
            "log": log_root / f"{name}.jsonl",
            "groundtruth": data_root / name / "groundtruth.txt",
        }
        missing = [key for key, value in paths.items() if not value.exists()]
        if missing:
            raise FileNotFoundError(f"{name}: missing {','.join(missing)}")
        baseline = load_boxes(paths["baseline"])
        candidate = load_boxes(paths["candidate"])
        groundtruth = np.loadtxt(paths["groundtruth"], delimiter=",", ndmin=2, dtype=np.float64)
        logs = [json.loads(line) for line in paths["log"].read_text(encoding="utf-8").splitlines() if line]
        if not (len(baseline) == len(candidate) == len(groundtruth)):
            raise ValueError(f"{name}: prediction/GT length mismatch")
        if len(logs) != len(groundtruth) - 1:
            raise ValueError(f"{name}: logs={len(logs)}, expected={len(groundtruth) - 1}")
        base_iou = iou_xywh(baseline[1:], groundtruth[1:])
        candidate_iou = iou_xywh(candidate[1:], groundtruth[1:])
        entropy = np.array([row["coord_entropy"] for row in logs], dtype=np.float64)
        disagreement = np.array([row["branch_disagreement_l1"] for row in logs], dtype=np.float64)
        changed = np.any(np.abs(candidate[1:] - baseline[1:]) > 1e-9, axis=1)
        rows[name] = {
            "delta_iou": candidate_iou - base_iou,
            "entropy": entropy,
            "disagreement": disagreement,
            "changed": changed,
            "baseline_iou": base_iou,
            "candidate_iou": candidate_iou,
        }
    return rows


def load_calibration_risk(data_root, log_root):
    entropy, disagreement, names = [], [], []
    for name in (data_root / "list.txt").read_text(encoding="utf-8").splitlines():
        if sequence_number(name) % 3 != 1:
            continue
        path = log_root / f"{name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{name}: missing reliability log")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        entropy.extend(row["coord_entropy"] for row in rows)
        disagreement.extend(row["branch_disagreement_l1"] for row in rows)
        names.append(name)
    return np.asarray(entropy, dtype=np.float64), np.asarray(disagreement, dtype=np.float64), names


def flatten(sequence_rows, mask_name=None, threshold=None):
    values = []
    for name in sorted(sequence_rows):
        rows = sequence_rows[name]
        mask = np.ones(len(rows["delta_iou"]), dtype=bool)
        if mask_name == "high_entropy":
            mask = rows["entropy"] >= threshold
        elif mask_name == "low_entropy":
            mask = rows["entropy"] < threshold
        elif mask_name == "high_disagreement":
            mask = rows["disagreement"] >= threshold
        elif mask_name == "low_disagreement":
            mask = rows["disagreement"] < threshold
        elif mask_name == "changed":
            mask = rows["changed"]
        elif mask_name == "unchanged":
            mask = ~rows["changed"]
        values.append(rows["delta_iou"][mask])
    return np.concatenate(values) if values else np.empty(0, dtype=np.float64)


def bootstrap_sequence_mean(sequence_rows, mask_name, threshold, samples, seed):
    names = sorted(sequence_rows)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        sampled = [sequence_rows[names[index]] for index in rng.integers(0, len(names), len(names))]
        values = flatten({str(index): value for index, value in enumerate(sampled)}, mask_name, threshold)
        if len(values):
            estimates.append(float(values.mean()))
    if not estimates:
        return [float("nan"), float("nan")]
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def summarize(sequence_rows, mask_name, threshold, samples, seed):
    values = flatten(sequence_rows, mask_name, threshold)
    coverage = int(len(values))
    return {
        "frames": coverage,
        "mean_delta_iou": float(values.mean()) if coverage else float("nan"),
        "median_delta_iou": float(np.median(values)) if coverage else float("nan"),
        "positive_fraction": float((values > 0).mean()) if coverage else float("nan"),
        "sequence_bootstrap_95_ci": bootstrap_sequence_mean(
            sequence_rows, mask_name, threshold, samples, seed
        ),
    }


def write_csv(path, summaries):
    fields = ["stratum", "frames", "mean_delta_iou", "median_delta_iou", "positive_fraction", "ci_low", "ci_high"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name, result in summaries.items():
            writer.writerow({
                "stratum": name,
                "frames": result["frames"],
                "mean_delta_iou": result["mean_delta_iou"],
                "median_delta_iou": result["median_delta_iou"],
                "positive_fraction": result["positive_fraction"],
                "ci_low": result["sequence_bootstrap_95_ci"][0],
                "ci_high": result["sequence_bootstrap_95_ci"][1],
            })


def plot(path, summaries):
    names = list(summaries)
    labels = {
        "all_selection_frames": "All frames",
        "high_entropy_q85": "High entropy\n(q85)",
        "low_entropy_q85": "Low entropy\n(<q85)",
        "high_disagreement_q85": "High disagreement\n(q85)",
        "low_disagreement_q85": "Low disagreement\n(<q85)",
        "CSPP_changed_frames": "CSPP changed\nbox",
        "CSPP_unchanged_frames": "CSPP unchanged\nbox",
    }
    means = [summaries[name]["mean_delta_iou"] for name in names]
    lows = [summaries[name]["sequence_bootstrap_95_ci"][0] for name in names]
    highs = [summaries[name]["sequence_bootstrap_95_ci"][1] for name in names]
    errors = np.array([[mean - low for mean, low in zip(means, lows)], [high - mean for mean, high in zip(means, highs)]])
    figure, axis = plt.subplots(figsize=(8.2, 3.2))
    colors = ["#b54c3b" if value < 0 else "#2f7e70" for value in means]
    axis.bar(range(len(names)), means, yerr=errors, capsize=3, color=colors)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(range(len(names)), [labels[name] for name in names], rotation=22, ha="right")
    axis.tick_params(axis="x", labelsize=8)
    axis.set_ylabel("CSPP minus baseline IoU")
    axis.set_title("Selection-partition conditional action advantage")
    axis.grid(axis="y", color="#d0d0d0", linewidth=0.5)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--log-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260820)
    args = parser.parse_args()

    entropy_values, disagreement_values, calibration_names = load_calibration_risk(
        args.data_root, args.log_root
    )
    selection = load_rows(
        args.data_root, args.baseline_root, args.candidate_root, args.log_root,
        lambda number: number % 3 == 2,
    )
    thresholds = {
        "entropy_q85_calibration": float(np.quantile(entropy_values, 0.85)),
        "disagreement_q85_calibration": float(np.quantile(disagreement_values, 0.85)),
    }
    strata = {
        "all_selection_frames": (None, None),
        "high_entropy_q85": ("high_entropy", thresholds["entropy_q85_calibration"]),
        "low_entropy_q85": ("low_entropy", thresholds["entropy_q85_calibration"]),
        "high_disagreement_q85": ("high_disagreement", thresholds["disagreement_q85_calibration"]),
        "low_disagreement_q85": ("low_disagreement", thresholds["disagreement_q85_calibration"]),
        "CSPP_changed_frames": ("changed", None),
        "CSPP_unchanged_frames": ("unchanged", None),
    }
    summaries = {
        name: summarize(selection, mask, threshold, args.bootstrap_samples, args.bootstrap_seed)
        for name, (mask, threshold) in strata.items()
    }
    per_sequence = {}
    for name, values in selection.items():
        per_sequence[name] = {
            "frames": int(len(values["delta_iou"])),
            "changed_frames": int(values["changed"].sum()),
            "mean_delta_iou": float(values["delta_iou"].mean()),
            "high_entropy_frames": int((values["entropy"] >= thresholds["entropy_q85_calibration"]).sum()),
            "high_disagreement_frames": int((values["disagreement"] >= thresholds["disagreement_q85_calibration"]).sum()),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "Read-only analysis of public GOT-10k validation only. No tracker/evaluator/data modifications and no B_test access.",
        "candidate": "valid Node 17 CSPP selection run 1705",
        "baseline": "Node 7 reliability tracker, previously verified byte-identical to immutable baseline",
        "calibration_partition": "sequence ID modulo 3 == 1",
        "selection_partition": "sequence ID modulo 3 == 2",
        "thresholds": thresholds,
        "coverage": {
            "calibration_sequences": len(calibration_names),
            "selection_sequences": len(selection),
            "selection_tracked_frames": int(sum(len(value["delta_iou"]) for value in selection.values())),
        },
        "bootstrap": {"unit": "sequence", "samples": args.bootstrap_samples, "seed": args.bootstrap_seed},
        "strata": summaries,
        "per_sequence": per_sequence,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "strata.csv", summaries)
    plot(args.output_dir / "strata.png", summaries)
    print(json.dumps({key: payload[key] for key in ("protocol", "thresholds", "coverage", "strata")}, indent=2))


if __name__ == "__main__":
    main()
