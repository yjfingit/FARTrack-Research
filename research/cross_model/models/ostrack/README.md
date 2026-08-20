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
only: no corrective action has been evaluated yet.

## Passive risk probe

An opt-in observer records response-map entropy but preserves all 180 baseline
result files byte-for-byte.  On 20,827 non-initialization frames, entropy
predicts IoU below 0.2 with AUROC `0.867915` (sequence-bootstrap 95% CI
`[0.812824, 0.912209]`) and AP `0.270585`.  This is correlational diagnostic
evidence, not evidence that any response to the signal improves tracking.
