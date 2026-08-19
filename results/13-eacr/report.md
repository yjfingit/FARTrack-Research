# Node 13: Entropy-Age Conditioned Ephemeral Reader (EACR)

## Idea

EACR is a training-free, one-frame template-reader intervention.  The frozen
coordinate posterior from frame `t` queues a read override only when normalized
coordinate entropy is at least `0.47618` and the inherited exponential reader's
mean template age is at least `40`.  At frame `t+1`, the tracker reads
`[initial, t-3, t-2, t-1, t]`, writes the normal new template, and returns to
the released exponential reader on frame `t+2`.

The threshold and age gate were pre-registered from the node-7 calibration
split (`ID % 3 == 1`); they were not changed during this experiment.

## Implementation and checks

- `FARTrackSparseEACR` is an independent tracker/parameter module.
- The base tracker receives inert optional hooks only.  The released tracker
  does not define them and retains its original control path.
- The control decision uses the already-computed softmax distribution: no
  extra network forward or GPU-to-CPU scalar synchronization occurs per frame.
- The candidate does not alter the current-frame box or template write path.
- Controller trigger/reset tests: `2 passed`.
- GPU representative-sequence smoke passed.
- With `FARTRACK_EACR_ENABLED=0`, the candidate output file for
  `GOT-10k_Val_000001` was byte-identical to the immutable baseline output.

## Calibration result (B_dev only)

Public GOT-10k validation partition `ID % 3 == 1`, 60 sequences, 6,591 frames:

| Method | Frame AO | Aggregate FPS |
| --- | ---: | ---: |
| Immutable baseline | 0.847678241 | 29.39145 |
| EACR | 0.846992904 | 32.55659 |

EACR's AO delta is `-0.000685336` (candidate minus baseline).  Its aggregate
FPS is 10.77% higher in this run, but timing is secondary to the accuracy gate.
The calibration outcome is slightly negative, so it does not meet the
pre-registered condition to advance: calibration AO must be at least baseline
before selection (`ID % 3 == 2`).  No selection, confirmation, full B_dev, or
B_test run was performed.

Raw result files remain on the data disk:

- `/root/autodl-tmp/experiment/.research-assets/output/node13_eacr/calibration_eacr.json`
- `/root/autodl-tmp/experiment/.research-assets/output/node13_eacr/calibration_baseline.json`

## Reproduction

```bash
VENV=/root/autodl-tmp/experiment/.venvs/fartrack-research/bin/python
DATA=/root/autodl-tmp/experiment/.research-assets/data/got10k/val
OUT=/root/autodl-tmp/experiment/.research-assets/output/node13_eacr

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\
$VENV scripts/run_got10k_partition.py fartrack_sparse_eacr fartrack_sparse_224_full \\
  --runid 13 --partition-mod 1 --threads 0 --num-gpus 1

$VENV scripts/evaluate_got10k_partition.py --data-root "$DATA" \\
  --results-root /root/autodl-tmp/experiment/.research-assets/output/test/tracking_results/fartrack_sparse_eacr/fartrack_sparse_224_full_013 \\
  --partition-mod 1 --output "$OUT/calibration_eacr.json"
```
