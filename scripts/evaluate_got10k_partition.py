"""Read-only AO and timing summary for a fixed GOT-10k validation partition."""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_got10k_val import iou_xywh


def sequence_id(name):
    return int(name.rsplit("_", 1)[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--partition-mod", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    names = [name for name in (args.data_root / "list.txt").read_text().splitlines()
             if sequence_id(name) % 3 == args.partition_mod]
    if len(names) != 60:
        raise ValueError(f"Expected 60 sequences, found {len(names)}")
    overlaps, times, per_sequence = [], [], {}
    for name in names:
        prediction = np.loadtxt(args.results_root / "got10k" / f"{name}.txt", delimiter="\t", ndmin=2)
        time = np.loadtxt(args.results_root / "got10k" / f"{name}_time.txt", delimiter="\t", ndmin=1)
        target = np.loadtxt(args.data_root / name / "groundtruth.txt", delimiter=",", ndmin=2)
        if len(prediction) != len(target) or len(time) != len(target):
            raise ValueError(f"{name}: prediction/time/GT length mismatch")
        overlap = iou_xywh(prediction.astype(np.float64), target.astype(np.float64))
        overlaps.append(overlap)
        times.append(time)
        per_sequence[name] = float(overlap.mean())
    all_overlaps, all_times = np.concatenate(overlaps), np.concatenate(times)
    report = {"protocol": "Read-only public GOT-10k validation partition; frame AO includes initial frame.",
              "partition": f"sequence ID modulo 3 == {args.partition_mod}",
              "num_sequences": len(names), "num_frames": int(len(all_overlaps)),
              "ao": float(all_overlaps.mean()),
              "mean_seconds_per_frame": float(all_times.mean()),
              "aggregate_fps": float(len(all_times) / all_times.sum()),
              "per_sequence_ao": per_sequence}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "per_sequence_ao"}, indent=2))


if __name__ == "__main__":
    main()
