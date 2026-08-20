# OSTrack Evidence

## Immutable B_dev baseline

Official upstream commit `33b5e125` was evaluated with the released
`vitb_384_mae_ce_32x4_ep300` configuration and ModelScope checkpoint.  The
unmodified official test entrypoint produced all 180 GOT-10k validation result
files.  A separate read-only evaluator checked every prediction and timing row
against the public annotations.

| metric | value |
| --- | ---: |
| AO (mean frame IoU) | 0.8651649115 |
| aggregate FPS | 40.008971 |
| sequences / frames | 180 / 21,007 |

Raw tracks and the machine-readable score remain on the data disk under
`/root/autodl-tmp/experiment/.research-assets/output/cross_model/ostrack/`.
They are evidence artifacts, not versioned source files.  This is a baseline
only: no risk signal or corrective action has been evaluated yet.
