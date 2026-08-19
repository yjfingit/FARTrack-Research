# Node 17: Coordinate-Selective Posterior Projection (CSPP)

## Idea

Frozen FARTrackSparse emits two native normalized crop-coordinate estimates in
raw `[left, top, right, bottom]` form: an autoregressive sequence estimate
(`mode`) and a feature-posterior expectation (`mean`). The released tracker
uses their midpoint. CSPP applies the calibration-frozen gate

`abs(cx_mode - cx_mean) / width_midpoint >= 0.025943091585107998`.

Only when this inclusive q95 gate fires, CSPP replaces the midpoint horizontal
center with `cx_mean`. It applies the resulting scalar offset to both left and
right endpoints, so the released midpoint width, vertical center, and height
remain unchanged. It adds no forward pass, history, template, search, or
masking change.

## Integrity Checks

- The threshold was already derived from passive calibration IDs modulo three
  equal to one. It was not adjusted in this experiment.
- The controller has three CPU tensor tests: midpoint-size normalization,
  `cx`-only geometry preservation, and inclusive threshold/exact no-trigger
  behavior (`3 passed`).
- Enabled CUDA smoke completed on `GOT-10k_Val_000001` and altered predictions,
  proving that the frozen q95 gate is reachable.
- With `FARTRACK_CSPP_ENABLED=0`, a fresh GPU run produced a byte-identical
  prediction text file to `fartrack_sparse_research` on that sequence.
- No data, checkpoint, official evaluator, or testing protocol was modified.

## Pre-registered Protocol

The public GOT-10k validation split is partitioned by numeric sequence ID:
calibration `% 3 == 1`, selection `% 3 == 2`, and confirmation `% 3 == 0`.
The selection gate requires frame AO improvement at least `+0.003` and relative
aggregate-FPS loss no worse than 5%. Confirmation and B_test are allowed only
after that gate passes.

An initial run used an incorrect `MODEL.RANGE` normalization. Audit of the
calibration provenance detected the mismatch before any conclusion: the
pre-registered statistic is normalized by midpoint predicted width. That run
(`runid 1702`) is retained only as an invalid configuration audit and is not
used for screening. The implementation was corrected, re-tested, and the
valid frozen-policy evaluation below used `runid 1705`.

## Independent Selection Result

Read-only scoring of the 60 sequence, 8,015 frame selection partition gives:

| Tracker | Frame AO | Aggregate FPS | AO delta | Relative FPS delta |
| --- | ---: | ---: | ---: | ---: |
| Frozen released baseline | 0.826292 | 51.94 | 0.000000 | 0.00% |
| CSPP, frozen q95 policy | 0.825130 | 60.83 | -0.001162 | +17.12% |

The policy fails the AO gate. A sequence-mean paired bootstrap is also
inconclusive (`mean=-0.001715`, 95% CI `[-0.021908, 0.017693]`; 25 positive,
22 negative, and 13 exact-zero sequence deltas). The speed estimate is not a
performance claim because independent run timing is variable; it only confirms
that CSPP did not introduce an obvious latency penalty.

Therefore CSPP is rejected before confirmation and B_test. The passive
same-state center-x evidence was insufficient to establish a beneficial causal
closed-loop intervention, consistent with node 14's failure of whole-box mean
projection. This negative candidate must not be presented as a contribution.

## Reproduction

```bash
VENV=/root/autodl-tmp/experiment/.venvs/fartrack-research/bin/python
DATA=/root/autodl-tmp/experiment/.research-assets/data/got10k/val
OUT=/root/autodl-tmp/experiment/.research-assets/output/node17_cspp

# Read-only frozen baseline score.
$VENV scripts/evaluate_got10k_partition.py --data-root "$DATA" \
  --results-root /root/autodl-tmp/experiment/.research-assets/output/test/tracking_results/fartrack_sparse_research/fartrack_sparse_224_full \
  --partition-mod 2 --output "$OUT/selection_baseline_frozen.json"

# Valid CSPP selection run; all internal CPU thread pools are capped.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
FARTRACK_CSPP_ENABLED=1 FARTRACK_CSPP_THRESHOLD=0.0259431 \
$VENV scripts/run_got10k_partition.py fartrack_sparse_cspp fartrack_sparse_224_full \
  --runid 1705 --partition-mod 2 --threads 0 --num-gpus 1

$VENV scripts/evaluate_got10k_partition.py --data-root "$DATA" \
  --results-root /root/autodl-tmp/experiment/.research-assets/output/test/tracking_results/fartrack_sparse_cspp/fartrack_sparse_224_full_1705 \
  --partition-mod 2 --output "$OUT/selection_cspp_valid.json"
```
