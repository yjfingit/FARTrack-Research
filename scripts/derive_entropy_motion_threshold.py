"""Derive entropy threshold candidates from node-7 calibration logs only."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    names = [name for name in (args.data_root / "list.txt").read_text().splitlines()
             if int(name.rsplit("_", 1)[1]) % 3 == 1]
    values = []
    for name in names:
        rows = [json.loads(line) for line in (args.log_root / f"{name}.jsonl").read_text().splitlines() if line]
        values.extend(row["coord_entropy"] for row in rows)
    report = {
        "source": "node7 passive coordinate-entropy logs; calibration IDs only (id % 3 == 1)",
        "sequences": len(names),
        "frames": len(values),
        "quantiles": {f"q{quantile:.2f}": float(np.quantile(values, quantile))
                      for quantile in (0.80, 0.85, 0.90)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
