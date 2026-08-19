"""Run a deterministic public GOT-10k validation ID partition unchanged."""

import argparse
import os
import sys

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(name, "1")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker


def sequence_id(name):
    return int(name.rsplit("_", 1)[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tracker_name")
    parser.add_argument("tracker_param")
    parser.add_argument("--runid", required=True, type=int)
    parser.add_argument("--partition-mod", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=1)
    args = parser.parse_args()
    selected = [seq for seq in get_dataset("got10k_val") if sequence_id(seq.name) % 3 == args.partition_mod]
    if len(selected) != 60:
        raise ValueError(f"Expected 60 sequences, found {len(selected)}")
    run_dataset(selected, [Tracker(args.tracker_name, args.tracker_param, "got10k_val", args.runid)],
                debug=0, threads=args.threads, num_gpus=args.num_gpus)


if __name__ == "__main__":
    main()
