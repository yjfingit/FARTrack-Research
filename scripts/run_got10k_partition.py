"""Run a fixed public GOT-10k validation partition without altering data."""

import argparse
import os
import sys
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(key, "1")

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
    parser.add_argument("--runid", required=True, type=int)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=1)
    args = parser.parse_args()
    dataset = [item for item in get_dataset("got10k_val")
               if sequence_id(item) % 3 == args.remainder]
    if len(dataset) != 60:
        raise RuntimeError(f"Expected 60 sequences, found {len(dataset)}")
    print("partition", args.remainder, "sequence_ids", [sequence_id(item) for item in dataset])
    run_dataset(dataset, [Tracker(args.tracker_name, args.tracker_param,
                                  "got10k_val", args.runid)],
                debug=False, threads=args.threads, num_gpus=args.num_gpus)


if __name__ == "__main__":
    main()
