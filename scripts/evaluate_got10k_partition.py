"""Read-only AO summary for a prespecified GOT-10k validation partition."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_got10k_val import iou_xywh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--remainder", required=True, type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    data_root, results_root = Path(args.data_root), Path(args.results_root)
    names = [
        name for name in (data_root / "list.txt").read_text().splitlines()
        if int(name.rsplit("_", 1)[1]) % 3 == args.remainder
    ]
    if len(names) != 60:
        raise RuntimeError(f"Expected 60 sequences, found {len(names)}")
    per_sequence, all_overlaps = {}, []
    for name in names:
        prediction = np.loadtxt(results_root / "got10k" / f"{name}.txt", delimiter="\t", ndmin=2)
        target = np.loadtxt(data_root / name / "groundtruth.txt", delimiter=",", ndmin=2)
        if len(prediction) != len(target):
            raise ValueError(f"{name}: prediction/annotation lengths differ")
        overlaps = iou_xywh(prediction.astype(np.float64), target.astype(np.float64))
        per_sequence[name] = float(overlaps.mean())
        all_overlaps.append(overlaps)
    report = {
        "metric": "GOT-10k validation AO (mean frame IoU)",
        "partition": {"id_modulo": 3, "id_remainder": args.remainder},
        "num_sequences": len(names),
        "num_frames": int(sum(len(item) for item in all_overlaps)),
        "ao": float(np.concatenate(all_overlaps).mean()),
        "per_sequence_ao": per_sequence,
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in report if key != "per_sequence_ao"}, indent=2))


if __name__ == "__main__":
    main()
