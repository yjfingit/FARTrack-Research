# Node 2: Immutable OSTrack B_dev Baseline

## Status

Completed.  This node establishes a frozen, official OSTrack baseline; it
does not test a candidate method and makes no improvement claim.

## Protocol

- Upstream: `botaoye/OSTrack`, commit `33b5e125` (MIT).
- Configuration: `vitb_384_mae_ce_32x4_ep300.yaml`.
- Checkpoint: official ModelScope package `damo/cv_vitb_video-single-object-tracking_ostrack`, SHA256 `8e6de3c6f10cbc21eaf1c34532358e23618f78a1a2bb9f0d9a2e55adc7af4894`.
- Dataset: existing public GOT-10k validation only; no train split was read.
- Process: one GPU process with all CPU numerical thread pools capped at one.
- Compatibility: `scripts/run_legacy_ostrack.py` supplies the removed
  `torch._six.string_classes` symbol in-process only.  Upstream tracker,
  checkpoint, annotations, and official test script are unchanged.

## Reproduction

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
/root/autodl-tmp/experiment/.venvs/fartrack-research/bin/python \
scripts/run_legacy_ostrack.py UPSTREAM/tracking/test.py ostrack \
vitb_384_mae_ce_32x4_ep300 --dataset_name got10k_val --threads 0 --num_gpus 1

/root/autodl-tmp/experiment/.venvs/fartrack-research/bin/python \
scripts/evaluate_got10k_results.py --data-root DATA/got10k/val \
--results-root RESULTS/got10k --output METRICS.json
```

## Measured Result

| metric | value |
| --- | ---: |
| AO | 0.8651649114760162 |
| FPS | 40.00897098375996 |
| sequences | 180 |
| frames | 21,007 |
| elapsed seconds | 525.057243 |

The read-only score JSON is
`/root/autodl-tmp/experiment/.research-assets/output/cross_model/ostrack/metrics/immutable_bdev_baseline.json`
with SHA256 `f7006d95bd7f8a99fb71073868faa7513957e1b8043866a170b8c3c84b860b8d`.
Raw predictions are under the adjacent `test/tracking_results` directory.

## Decision

Admit this immutable output as the OSTrack B_dev baseline.  The next node may
only observe its internal state passively before any pre-registered action is
considered.  B_test was not accessed.
