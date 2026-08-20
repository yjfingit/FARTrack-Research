# OSTrack Risk-Triggered Search-Area Action

**Status: preregistered before calibration metric computation and before any
action run.**

## Fixed Candidate

- **Signal:** prior-frame post-Hann response entropy, as recorded by the
  byte-parity passive probe.
- **Direction:** high entropy means risk.
- **Threshold:** the 85th percentile of all calibration (`suffix % 3 == 1`)
  passive-probe entropy values.  It is computed once, written to a frozen JSON
  artifact, and never re-estimated from selection or confirmation data.
- **Action:** on the frame *after* a high-risk observation, use OSTrack's
  existing crop routine with only the search-area factor multiplied by `1.10`.
  The model, checkpoint, template, box head, score aggregation, and current
  frame's localization stay unchanged.  There is no extra model forward,
  template update, optimizer step, or training-data access.
- **Reset:** the multiplier applies for exactly one next frame and then returns
  to the released search factor, unless the immediately preceding observation
  independently triggers it again.

## Information Control

Use exactly the same one-frame `1.10` multiplier on deterministic periodic
frames (`frame_id % 5 == 0`), approximately the same 20% action rate as q85.
It consumes no uncertainty signal.  It tests whether any observed change is
specific to information-conditioned intervention rather than merely altered
search geometry.

## Evaluation Sequence

1. Run candidate and information control only on calibration sequences.
2. Freeze the q85 threshold and record calibration AO/FPS/action rate.
3. Run candidate exactly once on selection sequences.
4. Enter confirmation only when selection has delta AO >= +0.003, aggregate
   FPS loss <= 10%, and a paired sequence-bootstrap 95% interval not clearly
   adverse.  Otherwise stop both candidate and control; do not run B_test.

The immutable OSTrack B_dev baseline is AO `0.8651649115`, FPS `40.008971`.
All action comparisons use its corresponding sequence subset and the split
definition in `../../protocol/SPLIT_DEFINITION.md`.
