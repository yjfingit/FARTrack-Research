# Node 7: Frozen-model localization reliability diagnostic

This is a diagnostic only. `fartrack_sparse_reliability` observes tensors from
the released `FARTrackSparse` forward pass after it has returned; its signals
never enter localization, state updates, masks, or template sampling.

## Public-development result

Dataset: public GOT-10k validation (`B_dev`), 180 sequences and 20,827 tracked
frames. The public set contains 21,007 annotations; the difference is exactly
one initialization annotation per sequence, which is supplied to the tracker
and deliberately has no `track()`-time diagnostic row.

Low-risk task definition: identify frames with IoU `< 0.2` (prevalence 2.9049%).
All intervals are deterministic 500-replicate sequence bootstrap 95% intervals
(seed `20260820`).

| passive signal | Spearman with IoU | AUROC low-IoU [95% CI] | AP low-IoU [95% CI] |
| --- | ---: | ---: | ---: |
| coordinate-bin entropy | -0.6593 | 0.9067 [0.8531, 0.9451] | 0.3686 [0.1704, 0.5573] |
| coordinate-bin margin | 0.5134 | 0.8309 [0.7360, 0.8962] | 0.2139 [0.0598, 0.4265] |
| token/feature branch L1 disagreement | -0.4569 | 0.8064 [0.7507, 0.8591] | 0.2072 [0.1028, 0.3298] |

The coordinate entropy result validates the node's *diagnostic* hypothesis:
the frozen model exposes a useful localization-risk signal. It is not an AO
improvement and no tracker policy is proposed or evaluated here.

## Reproduction

```bash
source /root/autodl-tmp/experiment/.venvs/fartrack-research/bin/activate
cd /tmp/arbor-worktrees-0/coordinator__n7-mechanism-frozen-model-localizat-52e8d36a

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
FARTRACK_RELIABILITY_LOG_DIR=/root/autodl-tmp/experiment/.research-assets/output/node7_reliability/logs \
python tracking/test.py fartrack_sparse_reliability fartrack_sparse_224_full \
  --dataset_name got10k_val --threads 2 --num_gpus 1

python scripts/evaluate_reliability_probe.py \
  --data-root /root/autodl-tmp/experiment/.research-assets/data/got10k/val \
  --prediction-root /root/autodl-tmp/experiment/.research-assets/output/test/tracking_results/fartrack_sparse_reliability/fartrack_sparse_224_full \
  --log-root /root/autodl-tmp/experiment/.research-assets/output/node7_reliability/logs \
  --output /root/autodl-tmp/experiment/.research-assets/output/node7_reliability/full_bdev_metrics.json \
  --bootstrap-samples 500
```

The result JSON is stored outside the repository at
`/root/autodl-tmp/experiment/.research-assets/output/node7_reliability/full_bdev_metrics.json`.

## Prediction invariance

The normal baseline class is unchanged except for a disabled observer lookup:
it calls no new code unless a subclass defines `_record_reliability`. Two
checks were run:

1. A rerun of `fartrack_sparse_research` on `GOT-10k_Val_000002` with a fresh
   result directory was byte-identical to the historical immutable baseline.
2. Every one of the 180 `fartrack_sparse_reliability` result files was
   byte-identical to the corresponding immutable baseline result. Mismatches:
   `0 / 180`.

Therefore logging is observational with respect to the released integer result
files, including all template and state updates that produced later frames.
