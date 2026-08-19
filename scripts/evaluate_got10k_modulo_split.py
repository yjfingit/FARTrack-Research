"""Read-only AO/alarm evaluator for a fixed GOT-10k validation modulo split."""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from evaluate_got10k_val import iou_xywh


def sequence_id(name):
    match = re.search(r"(\d+)$", name)
    if match is None:
        raise ValueError(f"cannot parse numeric sequence id from {name}")
    return int(match.group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--remainder", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path)
    parser.add_argument("--alarm-log-dir", type=Path)
    args = parser.parse_args()
    names = [name for name in args.data_root.joinpath("list.txt").read_text().splitlines() if sequence_id(name) % 3 == args.remainder]
    sequence_scores, overlaps, elapsed_seconds = {}, [], 0.0
    for name in names:
        prediction = np.loadtxt(args.results_root / "got10k" / f"{name}.txt", delimiter="\t", ndmin=2)
        target = np.loadtxt(args.data_root / name / "groundtruth.txt", delimiter=",", ndmin=2)
        if len(prediction) != len(target):
            raise ValueError(f"{name}: prediction/target length mismatch")
        score = iou_xywh(prediction.astype(np.float64), target.astype(np.float64))
        overlaps.append(score)
        sequence_scores[name] = float(score.mean())
        time_path = args.results_root / "got10k" / f"{name}_time.txt"
        if not time_path.exists():
            raise FileNotFoundError(time_path)
        elapsed_seconds += float(np.loadtxt(time_path, ndmin=1).sum())
    num_frames = int(sum(map(len, overlaps)))
    report = {"metric": "GOT-10k validation AO (mean frame IoU)", "remainder": args.remainder, "num_sequences": len(names), "num_frames": num_frames, "ao": float(np.concatenate(overlaps).mean()), "aggregate_fps": num_frames / elapsed_seconds, "per_sequence_ao": sequence_scores}
    if args.stats_path is not None:
        stats = json.loads(args.stats_path.read_text())
        report["two_view_stats"] = stats
    if args.alarm_log_dir is not None:
        alarms, chosen, frames = 0, 0, 0
        for name in names:
            log_path = args.alarm_log_dir / f"{name}.jsonl"
            if not log_path.exists():
                raise FileNotFoundError(log_path)
            for line in log_path.read_text().splitlines():
                row = json.loads(line)
                alarms += int(row["alarm"])
                chosen += int(row["expanded_chosen"])
                frames += 1
        report["two_view_stats"] = {"frames": frames, "alarms": alarms, "expanded_chosen": chosen, "alarm_rate": alarms / max(frames, 1)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "per_sequence_ao"}, indent=2))


if __name__ == "__main__":
    main()
