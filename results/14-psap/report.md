# Node 14: Posterior-Shape Adaptive Projection (PSAP)

## Idea

FARTrackSparse produces two native normalized-coordinate estimates on each
frozen forward pass: a discrete autoregressive sequence estimate (called
`mode`) and the expectation of the feature-bin posterior (called `mean`). The
released output is their midpoint. PSAP computes

`d = mean(abs(mean - mode)) / MODEL.RANGE`

on CUDA and, only above a calibration-fixed quantile, uses either pure `mode`
or pure `mean`; otherwise it retains the released midpoint. This is strictly an
output-projection experiment, not the motion/state correction of nodes 9--10.

## Integrity and implementation checks

- `FARTrackSparsePSAP` is an independent tracker and parameter module.
- The base tracker adds only an optional output-materialization hook. With no
  hook the released path is unchanged; with PSAP the gate and projection remain
  on GPU and use the existing output-coordinate transfer.
- No template, crop, state-history, model, checkpoint, annotation, evaluator,
  or test protocol was modified. Network forwards remain one per frame.
- Controller arithmetic and strict gate tests: `2 passed`.
- CUDA smoke completed on GOT-10k validation sequence `000001`.
- With `FARTRACK_PSAP_ENABLED=0`, a fresh GPU run was byte-identical to the
  immutable baseline prediction text for sequence `000001`.
- With `alpha=0.5`, calibration AO exactly equaled the immutable baseline and
  the first sequence text was byte-identical.

## Pre-registered calibration

The public B_dev calibration partition is GOT-10k validation numeric sequence
ID modulo three equal to one: 60 sequences, 6,591 frames. First, the endpoint
sanity checks ran `alpha in {0, 0.5, 1}`. The diagnostic pass at `alpha=0.5`
derived `q85=0.001887807622551918` and
`q95=0.0032803118228912354` using calibration frames only. Before gated runs,
`preselection.json` froze the four policies `{q85,q95} x {mode,mean}` and the
selection rule: highest calibration AO with FPS at least 95% of the released
midpoint. Ties would prefer q95 then the lower high alpha.

| Policy | Frame AO | Aggregate FPS | AO delta vs midpoint |
| --- | ---: | ---: | ---: |
| Pure mode (`alpha=0`) | 0.847602436 | 31.7227 | -0.000075805 |
| Released midpoint (`alpha=0.5`) | 0.847678241 | 31.0835 | 0.000000000 |
| Pure mean (`alpha=1`) | 0.834773809 | 31.9439 | -0.012904432 |
| q85 -> mode | 0.835438917 | 31.2359 | -0.012239323 |
| q85 -> mean | 0.829126267 | 31.5631 | -0.018551973 |
| q95 -> mode | 0.834246392 | 31.7677 | -0.013431849 |
| q95 -> mean | 0.837691226 | 31.0583 | -0.009987015 |

All four gated policies satisfy the speed constraint but fail the calibration
AO criterion. The released midpoint is therefore retained as the calibration
winner, not a new method. Per the pre-registered policy, no selection,
confirmation, full B_dev, or B_test evaluation was run for PSAP.

## Interpretation and related work

The failure is informative: mode--mean disagreement is not a direction signal
for choosing either endpoint in this frozen coordinate posterior. At the rare
large-disagreement frames, discarding one component produces a much larger
causal tracking error. This does not establish a general impossibility result.

Related-work review found prior probabilistic tracking output distributions
(Danelljan et al., CVPR 2020) and uncertainty-guided learned tracking
(Zhou et al., AAAI 2021). Since PSAP has no positive independent-split result,
it must not be claimed as novel or included as a contribution.

## Artifacts and reproduction

Raw outputs and read-only metric reports are on the data disk:

- `/root/autodl-tmp/experiment/.research-assets/output/node14_psap/`
- `/root/autodl-tmp/experiment/.research-assets/output/test/tracking_results/fartrack_sparse_psap/`

```bash
VENV=/root/autodl-tmp/experiment/.venvs/fartrack-research/bin/python
DATA=/root/autodl-tmp/experiment/.research-assets/data/got10k/val
OUT=/root/autodl-tmp/experiment/.research-assets/output/node14_psap

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
FARTRACK_PSAP_THRESHOLD=0.001887807622551918 \
FARTRACK_PSAP_HIGH_ALPHA=0 FARTRACK_PSAP_LOW_ALPHA=0.5 \
$VENV scripts/run_got10k_partition.py fartrack_sparse_psap fartrack_sparse_224_full \
  --runid 1485 --partition-mod 1 --threads 0 --num-gpus 1

$VENV scripts/evaluate_got10k_partition.py --data-root "$DATA" \
  --results-root /root/autodl-tmp/experiment/.research-assets/output/test/tracking_results/fartrack_sparse_psap/fartrack_sparse_224_full_1485 \
  --partition-mod 1 --output "$OUT/calibration_q85_mode.json"
```
