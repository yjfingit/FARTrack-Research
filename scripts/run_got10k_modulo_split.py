"""Run a tracker on a GOT-10k validation modulo split with bounded workers."""

import argparse
import concurrent.futures
import os
import re
import subprocess
from pathlib import Path


def sequence_id(name):
    match = re.search(r"(\d+)$", name)
    if match is None:
        raise ValueError(f"cannot parse numeric sequence id from {name}")
    return int(match.group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tracker")
    parser.add_argument("parameter")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--remainder", type=int, required=True)
    parser.add_argument("--runid", type=int, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--extra-env", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    names = args.data_root.joinpath("list.txt").read_text(encoding="utf-8").splitlines()
    selected = [(index, name) for index, name in enumerate(names) if sequence_id(name) % 3 == args.remainder]
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"})
    for item in args.extra_env:
        key, value = item.split("=", 1)
        environment[key] = value

    def run_one(pair):
        index, name = pair
        command = ["python", "tracking/test.py", args.tracker, args.parameter, "--dataset_name", "got10k_val", "--sequence", str(index), "--threads", "0", "--num_gpus", "1", "--runid", str(args.runid)]
        completed = subprocess.run(command, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return name, completed.returncode, completed.stdout

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for name, returncode, output in executor.map(run_one, selected):
            print(f"[{name}] returncode={returncode}", flush=True)
            if returncode:
                failures.append((name, output))
    if failures:
        for name, output in failures:
            print(f"\n--- {name} ---\n{output}")
        raise SystemExit(f"{len(failures)} tracker runs failed")
    print(f"completed {len(selected)} sequences for remainder {args.remainder}")


if __name__ == "__main__":
    main()
