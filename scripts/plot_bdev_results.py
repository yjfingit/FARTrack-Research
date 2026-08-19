#!/usr/bin/env python3
"""Plot measured B_dev AO deltas and aggregate tracker throughput."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_ao(path: Path) -> float:
    with path.open() as handle:
        return float(json.load(handle)["ao"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--assets", default=Path("/root/autodl-tmp/experiment/.research-assets/output"), type=Path)
    args = parser.parse_args()

    filenames = {
        "Baseline": "got10k_val_baseline_ao.json",
        "TRM-FAR": "got10k_val_node1_trm_ao.json",
        "Counterfactual": "got10k_val_node2_cf_ao.json",
        "Anchor recovery": "got10k_val_node3_sar_ao.json",
        "Log sampler": "got10k_val_node6_logarithmic_ao.json",
    }
    ao = {name: load_ao(args.assets / filename) for name, filename in filenames.items()}
    fps = {"Baseline": 40.08210429921829, "TRM-FAR": 42.160655418265875,
           "Counterfactual": 23.467610737504696, "Anchor recovery": 40.902096149190406,
           "Log sampler": 31.315525840882607}
    names = list(filenames)
    colors = ["#4C78A8", "#D95F02", "#D95F02", "#D95F02", "#D95F02"]

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    delta = np.array([ao[name] - ao["Baseline"] for name in names]) * 100
    axes[0].bar(names, delta, color=colors)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("AO delta (percentage points)")
    axes[0].set_title("GOT-10k validation: change from baseline")
    axes[0].tick_params(axis="x", rotation=22)
    for index, value in enumerate(delta):
        axes[0].text(index, value - 0.15 if value < 0 else value + 0.05, f"{value:+.2f}", ha="center", va="top" if value < 0 else "bottom", fontsize=8)

    values = [fps[name] for name in names]
    axes[1].bar(names, values, color=colors)
    axes[1].set_ylabel("Aggregate FPS")
    axes[1].set_ylim(0, max(values) * 1.18)
    axes[1].set_title("Measured tracker throughput")
    axes[1].tick_params(axis="x", rotation=22)
    for index, value in enumerate(values):
        axes[1].text(index, value + 0.6, f"{value:.1f}", ha="center", fontsize=8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)


if __name__ == "__main__":
    main()
