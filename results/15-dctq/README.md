# Node 15: Disagreement-Conditioned Template Quality (DCTQ)

DCTQ is a training-free template-write experiment for frozen FARTrackSparse.
Each model forward already returns four attention-ranked boolean masks for the
just-written 7x7 template block. The released implementation stores all four
but its reader always consumes the 25% pruning mask. DCTQ preserves template
identity, every write, exponential chronological sampler, current-frame box,
and one network forward. It only changes the mask attached to a write when the
existing sequence/feature coordinate branches disagree.

The native levels prune exactly `12/49`, `24/49`, `36/49`, and `44/49` tokens,
respectively (nominally 25%, 50%, 75%, 90%), retaining `37`, `25`, `13`, and
`5` tokens. They are nested: every retained token at a stronger level is also
retained at the weaker level. A high disagreement therefore selects stronger
*pruning*, not greater retention.

`preselection.json` was created before calibration. It fixes the q85 threshold
to `0.0037756040692329407`, inherited from Node 7's calibration-only passive
logs, and limits the policy family to trigger levels 50%, 75%, and 90%.
The public GOT-10k validation calibration partition is ID modulo 3 equal to
one (60 sequences / 6,591 frames). IDs modulo 2 and zero remain untouched.

## Calibration outcome

| Policy | AO | Delta vs 25% | Aggregate FPS | Relative FPS |
| --- | ---: | ---: | ---: | ---: |
| Released 25% baseline | 0.847678241 | 0.000000000 | 32.2449 | 0.00% |
| Static 50% diagnostic | 0.828578080 | -0.019100160 | 32.2463 | +0.00% |
| Static 75% diagnostic | 0.808817822 | -0.038860419 | 32.9810 | +2.28% |
| Static 90% diagnostic | 0.765953647 | -0.081724593 | 29.1395 | -9.63% |
| q85 -> 50% | 0.836996629 | -0.010681611 | 31.7911 | -1.41% |
| q85 -> 75% | 0.838668434 | -0.009009807 | 31.8838 | -1.12% |
| q85 -> 90% | 0.834802924 | -0.012875317 | 31.4125 | -2.58% |

All adaptive policies met the <=5% FPS-loss constraint but none matched the
baseline AO, so none is selected. Per the preregistered gate, no selection,
confirmation, full B_dev, or B_test was run.

## Integrity checks

- `tests/test_dctq_controller.py`: three unit tests cover level selection,
  disabled/low-gate baseline behavior, static selection, nested mask shape,
  and token counts.
- A CUDA smoke run on `GOT-10k_Val_000001` completed with active 90% DCTQ.
- A fresh disabled DCTQ GPU run was byte-identical to a fresh baseline run for
  that sequence (`sha256=29d3813815dbd7edde898dd31bc2cc9f7eb77d2f0b2851731701983d5528fd2d`).
- The code appends every `new_z` exactly as the released tracker does and keeps
  all four raw mask lists; only `store_mask`, which the released reader already
  uses, changes for an active DCTQ write.
- No data, official evaluator, annotation, checkpoint, or test protocol was
  modified. Outputs and metrics reside under the data disk.

## Related-work distinction

Adaptive template updating and uncertainty-aware tracking are established
themes (e.g., Yang et al., *Learning the Model Update for Siamese Trackers*,
ICCV 2019; Zhou et al., *Model Uncertainty Guides Visual Object Tracking*,
AAAI 2021). DCTQ would have been narrower: conditional use of pre-existing
attention-pruning masks in a frozen transformer's template reader, with no
learned updater or added observation. Its negative independent calibration
result means it is not a paper contribution.

## Reproduction

```bash
VENV=/root/autodl-tmp/experiment/.venvs/fartrack-research/bin/python
DATA=/root/autodl-tmp/experiment/.research-assets/data/got10k/val
OUT=/root/autodl-tmp/experiment/.research-assets/output/node15_dctq

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
FARTRACK_DCTQ_LEVEL=2 FARTRACK_DCTQ_THRESHOLD=0.0037756040692329407 \
$VENV scripts/run_got10k_partition.py fartrack_sparse_dctq fartrack_sparse_224_full \
  --remainder 1 --runid 1515 --threads 0 --num-gpus 1

$VENV scripts/evaluate_got10k_partition.py --data-root "$DATA" \
  --results-root /root/autodl-tmp/experiment/.research-assets/output/test/tracking_results/fartrack_sparse_dctq/fartrack_sparse_224_full_1515 \
  --remainder 1 --output "$OUT/calibration_q85_75.json"
```
