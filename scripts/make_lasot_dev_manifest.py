"""Create a fixed, category-stratified LaSOT development manifest.

This is a protocol artifact, not an evaluator. The list is derived from the
official `testing_set.txt`, fixed before candidate scores are inspected, and
kept separate from the remaining held-out LaSOT test sequences.
"""

from pathlib import Path


DEV_PER_CATEGORY = 1


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("testing_set")
    parser.add_argument("output")
    args = parser.parse_args()
    sequences = [line.strip() for line in Path(args.testing_set).read_text().splitlines() if line.strip()]
    by_category = {}
    for sequence in sequences:
        by_category.setdefault(sequence.split("-", 1)[0], []).append(sequence)
    selected = [sorted(items)[0] for _, items in sorted(by_category.items())][:]
    if DEV_PER_CATEGORY != 1:
        selected = [sequence for _, items in sorted(by_category.items()) for sequence in sorted(items)[:DEV_PER_CATEGORY]]
    Path(args.output).write_text("\n".join(selected) + "\n")
    print(f"wrote {len(selected)} fixed development sequences")


if __name__ == "__main__":
    main()
