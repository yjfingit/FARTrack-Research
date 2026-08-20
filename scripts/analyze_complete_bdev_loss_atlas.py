"""Read-only effect-distribution analysis for complete GOT-10k B_dev runs."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = {
    "TRM-FAR": "got10k_val_node1_trm_ao.json",
    "Counterfactual agreement": "got10k_val_node2_cf_ao.json",
    "Stable-anchor recovery": "got10k_val_node3_sar_ao.json",
    "Logarithmic sampler": "got10k_val_node6_logarithmic_ao.json",
}


def bootstrap(delta, samples, seed):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(samples, len(delta)))
    means = delta[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summary(method, baseline, candidate, samples, seed):
    keys = sorted(baseline)
    if keys != sorted(candidate):
        raise ValueError(f"{method}: sequence key mismatch")
    delta = np.array([candidate[key] - baseline[key] for key in keys], dtype=np.float64)
    loss = np.maximum(-delta, 0.0)
    sorted_loss = np.sort(loss)[::-1]
    total_loss = float(loss.sum())
    top5_share = float(sorted_loss[:5].sum() / total_loss) if total_loss else 0.0
    worst = int(np.argmin(delta))
    best = int(np.argmax(delta))
    return {
        "method": method,
        "sequences": len(keys),
        "mean_delta_ao": float(delta.mean()),
        "median_delta_ao": float(np.median(delta)),
        "mean_bootstrap_95_ci": bootstrap(delta, samples, seed),
        "improved_sequences": int((delta > 0).sum()),
        "degraded_sequences": int((delta < 0).sum()),
        "unchanged_sequences": int((delta == 0).sum()),
        "top5_loss_share": top5_share,
        "worst_sequence": keys[worst],
        "worst_delta_ao": float(delta[worst]),
        "best_sequence": keys[best],
        "best_delta_ao": float(delta[best]),
        "per_sequence_delta": {key: float(value) for key, value in zip(keys, delta)},
    }


def write_csv(path, results):
    fields = [
        "method", "sequences", "mean_delta_ao", "median_delta_ao", "ci_low", "ci_high",
        "improved_sequences", "degraded_sequences", "unchanged_sequences", "top5_loss_share",
        "worst_sequence", "worst_delta_ao", "best_sequence", "best_delta_ao",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for result in results:
            writer.writerow({
                **{field: result[field] for field in fields if field in result},
                "ci_low": result["mean_bootstrap_95_ci"][0],
                "ci_high": result["mean_bootstrap_95_ci"][1],
            })


def plot(path, results):
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), gridspec_kw={"width_ratios": [1.5, 1]})
    deltas = [list(result["per_sequence_delta"].values()) for result in results]
    violin = axes[0].violinplot(deltas, showmeans=True, showmedians=True)
    for body in violin["bodies"]:
        body.set_facecolor("#b54c3b")
        body.set_edgecolor("#5f1f18")
        body.set_alpha(0.72)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xticks(range(1, len(results) + 1), ["TRM", "CF", "SAR", "Log"])
    axes[0].set_ylabel("Per-sequence AO delta")
    axes[0].set_title("Complete Bdev effect distributions")
    axes[0].grid(axis="y", color="#d0d0d0", linewidth=0.5)

    shares = [100.0 * result["top5_loss_share"] for result in results]
    axes[1].bar(range(len(results)), shares, color="#3b7a8c")
    axes[1].set_xticks(range(len(results)), ["TRM", "CF", "SAR", "Log"])
    axes[1].set_ylabel("Top-5 share of total loss (%)")
    axes[1].set_title("Loss concentration")
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis="y", color="#d0d0d0", linewidth=0.5)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260820)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))["per_sequence_ao"]
    results = []
    for offset, (method, filename) in enumerate(METHODS.items()):
        candidate = json.loads((args.candidate_dir / filename).read_text(encoding="utf-8"))["per_sequence_ao"]
        results.append(summary(method, baseline, candidate, args.bootstrap_samples, args.bootstrap_seed + offset))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "Read-only analysis of retained complete public GOT-10k B_dev AO JSON files. No tracker/evaluator/data changes and no B_test access.",
        "bootstrap": {"unit": "sequence", "samples": args.bootstrap_samples, "seed_base": args.bootstrap_seed},
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "summary.csv", results)
    plot(args.output_dir / "loss_atlas.png", results)
    print(json.dumps({key: payload[key] for key in ("protocol", "bootstrap", "results")}, indent=2))


if __name__ == "__main__":
    main()
