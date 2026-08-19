"""Derive the pre-registered q95 alarm threshold from Node 7 calibration logs."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def sequence_id(path):
    match = re.search(r"(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"cannot read numeric sequence id from {path.name}")
    return int(match.group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--preselection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(path for path in args.log_dir.glob("*.jsonl") if sequence_id(path) % 3 == 1)
    values = []
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            values.append(json.loads(line)["branch_disagreement_l1"])
    if not values:
        raise ValueError("no calibration log values")
    threshold = float(np.quantile(np.asarray(values), 0.95, method="higher"))
    alarm_count = sum(value >= threshold for value in values)
    payload = {
        "node_id": 12,
        "status": "threshold-frozen-before-candidate-calibration",
        "threshold": threshold,
        "comparison": ">= threshold",
        "source": "Node 7 passive branch_disagreement_l1 logs",
        "calibration_split": "numeric sequence id modulo 3 equals 1",
        "num_sequences": len(files),
        "num_frames": len(values),
        "calibration_alarm_count": alarm_count,
        "calibration_alarm_rate": alarm_count / len(values),
        "preselection_sha256": hashlib.sha256(args.preselection.read_bytes()).hexdigest(),
        "log_file_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in payload if key != "log_file_sha256"}, indent=2))


if __name__ == "__main__":
    main()
