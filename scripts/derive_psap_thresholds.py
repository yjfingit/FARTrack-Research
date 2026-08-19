"""Derive pre-registered PSAP gates from calibration-only diagnostic logs."""

import argparse
import json
from pathlib import Path

import numpy as np


def sequence_id(name):
    return int(name.rsplit("_", 1)[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    names = [name for name in (args.data_root / "list.txt").read_text().splitlines()
             if sequence_id(name) % 3 == 1]
    values = []
    for name in names:
        rows = [json.loads(line) for line in (args.log_root / f"{name}.jsonl").read_text().splitlines() if line]
        values.extend(row["normalized_mean_mode_disagreement"] for row in rows)
    report = {"source": "node14 calibration IDs only (id % 3 == 1)",
              "sequences": len(names), "frames": len(values),
              "quantiles": {f"q{q:.2f}": float(np.quantile(values, q)) for q in (0.85, 0.95)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
