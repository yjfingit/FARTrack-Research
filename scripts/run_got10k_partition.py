"""Run a deterministic public GOT-10k validation partition without data edits."""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker


def sequence_id(sequence):
    return int(sequence.name.rsplit("_", 1)[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tracker_name")
    parser.add_argument("tracker_param")
    parser.add_argument("--remainder", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--num-gpus", type=int, default=1)
    args = parser.parse_args()
    dataset = [sequence for sequence in get_dataset("got10k_val") if sequence_id(sequence) % 3 == args.remainder]
    if len(dataset) != 60:
        raise RuntimeError(f"Expected 60 sequences, found {len(dataset)}")
    print("partition", args.remainder, "sequence_ids", [sequence_id(sequence) for sequence in dataset])
    run_dataset(dataset, [Tracker(args.tracker_name, args.tracker_param, "got10k_val")],
                debug=False, threads=args.threads, num_gpus=args.num_gpus)


if __name__ == "__main__":
    main()
