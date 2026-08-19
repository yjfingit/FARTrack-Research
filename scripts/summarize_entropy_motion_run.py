"""Summarize a node-9 partition run against a read-only baseline summary."""

import argparse
import json
from pathlib import Path

import numpy as np


def aggregate_fps(results_root, names):
    elapsed, frames = 0.0, 0
    for name in names:
        values = np.loadtxt(results_root / "got10k" / f"{name}_time.txt", ndmin=1)
        elapsed += float(np.sum(values))
        frames += len(values)
    return float(frames / elapsed)


def paired_bootstrap(delta, samples=10000, seed=20260820):
    rng = np.random.default_rng(seed)
    values = []
    for indices in rng.integers(0, len(delta), size=(samples, len(delta))):
        values.append(float(delta[indices].mean()))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--candidate-results-root", required=True, type=Path)
    parser.add_argument("--baseline-results-root", required=True, type=Path)
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    if baseline["partition"] != candidate["partition"]:
        raise ValueError("baseline and candidate partitions differ")
    names = sorted(candidate["per_sequence_ao"])
    delta = np.array([candidate["per_sequence_ao"][name] - baseline["per_sequence_ao"][name] for name in names])
    baseline_fps = aggregate_fps(args.baseline_results_root, names)
    candidate_fps = aggregate_fps(args.candidate_results_root, names)
    report = {
        "alpha": args.alpha,
        "partition": candidate["partition"],
        "baseline_ao": baseline["ao"],
        "candidate_ao": candidate["ao"],
        "delta_ao": candidate["ao"] - baseline["ao"],
        "mean_per_sequence_delta": float(delta.mean()),
        "median_per_sequence_delta": float(np.median(delta)),
        "improved_sequences": int((delta > 0).sum()),
        "degraded_sequences": int((delta < 0).sum()),
        "tied_sequences": int((delta == 0).sum()),
        "paired_bootstrap": {"resamples": 10000, "seed": 20260820, "ci95": paired_bootstrap(delta)},
        "aggregate_fps": {"baseline_read_only": baseline_fps, "candidate": candidate_fps,
                          "relative_change": candidate_fps / baseline_fps - 1.0},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
