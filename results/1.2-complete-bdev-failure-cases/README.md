# Complete B_dev Failure Cases

`failure_cases.png` is a read-only qualitative supplement for the four methods
with complete GOT-10k validation results. Each panel is selected mechanically:
within its named, pre-selected lowest-sequence-AO case, it is the frame with
the smallest candidate-minus-immutable-baseline IoU. Green is ground truth,
blue is the immutable baseline, and red is the candidate.

This rendering does not run a tracker, alter a prediction file, alter the
benchmark protocol, or establish a semantic cause of a failure. The exact frame
and IoUs are retained in `failure_cases.json` and `failure_cases.csv`.
