# Node 5: Recent-consensus template rebinding

`fartrack_sparse_rcm` is an independent tracker that reuses the released
FARTrackSparse checkpoint and original one-forward tracking loop. Each frame's
candidate crop and 25-percent token mask are written exactly once into a
fixed FIFO window. Before the next forward, it binds the immutable initial
anchor, three chronological RGB-descriptor medoids from prior recent records,
and the newest exact crop, then rebuilds FARTrack's boolean `[1,445,445]`
attention mask.

This artifact is implementation and smoke-test provenance only. No score is
claimed until matched B_dev evaluation is complete.

Focused public-development screen (not a full B_dev result):

```bash
CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/experiment/.venvs/fartrack-research/bin/python \
  tracking/test.py fartrack_sparse_rcm fartrack_sparse_224_full \
  --dataset_name got10k_val --sequence 0 --threads 0 --num_gpus 1
```

The unmodified evaluator input was `GOT-10k_Val_000001` (60 frames). Its AO
was `0.9078298880`; the immutable baseline result for the same sequence was
`0.9098032749`. This single-sequence screen is intentionally reported without
an improvement claim. Results live under
`/root/autodl-tmp/experiment/.research-assets/output/test/tracking_results/`.
