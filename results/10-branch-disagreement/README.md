# Branch-Disagreement-Gated Causal Mixing

Node 10 evaluates a frozen FARTrackSparse state-estimation intervention on the
public GOT-10k validation split only. The tracker performs its released single
network forward and released template update. After the released localization
branches have been computed, it transfers the usual model box plus the two
already-computed four-coordinate branch estimates in one host materialization.
The host computes their mean L1 disagreement and, only above the frozen
threshold, mixes the image-space model box with a causal one-step
constant-velocity box prior.

The deterministic partitions are numeric sequence IDs modulo three:
calibration `== 1`, selection `== 2`, and confirmation `== 0`. Threshold
`q0.85 = 0.0037756040692329407` was derived only from node-7 passive logs on
calibration. The finite calibration schedule was alpha `{0.05, 0.10, 0.25}`;
all three runs are retained. `preselection.json` freezes alpha `0.05` under
the predeclared largest-calibration-AO rule and its SHA256 is recorded before
selection. No B_test data is used.

Disabled mode was compared to a fresh immutable-baseline run on sequence 1 and
the prediction text files were byte-identical. A repeated enabled timing pair
on the same sequence showed median relative FPS change `-1.82%`, within the
`-5%` precondition for calibration. The selection gate requires both at least
`+0.003` frame AO and no more than `5%` aggregate FPS loss; confirmation runs
only if that gate passes.

Calibration AO deltas for alpha `0.05`, `0.10`, and `0.25` were `-0.011307`,
`-0.013602`, and `-0.020232`, respectively. The frozen alpha `0.05` result on
untouched selection was `0.824604` AO versus `0.826292` baseline (`-0.001688`),
which fails the predeclared AO gate. No confirmation partition or B_test run
was performed.
