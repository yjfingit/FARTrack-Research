# Training-Free FARTrack Research Log

## Protocol

- Checkpoint: immutable released `FARTrackSparse_ep0015.pth.tar`.
- B_dev: public GOT-10k validation, 180 sequences and 21,007 frames.
- Metric: mean frame IoU (AO), computed by `scripts/evaluate_got10k_val.py`.
- B_test: full official LaSOT Testing.  It is checksum-verified but was not
  extracted, read, or used for selection while this log was written.
- All tracker outputs use the repository's existing result writer; data and
  official evaluation code are not modified.

## Completed B_dev Results

| Tracker | AO | Delta to baseline | 95% paired bootstrap CI | Aggregate FPS | Decision |
| --- | ---: | ---: | --- | ---: | --- |
| `fartrack_sparse_research` | 0.832741 | 0.000000 | -- | 40.07 | immutable baseline |
| `trm_far` | 0.822471 | -0.008228 | [-0.019499, 0.004104] | 42.16 | rejected |
| `fartrack_sparse_cf` | 0.806470 | -0.019995 | [-0.037073, -0.003256] | 23.47 | rejected |
| `sar_far` | 0.797473 | -0.032038 | [-0.050311, -0.015034] | 40.90 | rejected |

The bootstrap resamples the 180 per-sequence AO deltas with seed `20260819`
for 10,000 draws.  No rejected method was run on B_test or tuned after its
full B_dev result.

## Reproduction

```bash
VENV=/root/autodl-tmp/experiment/.venvs/fartrack-research/bin/python
DATA=/root/autodl-tmp/experiment/.research-assets/data/got10k/val

$VENV tracking/test.py fartrack_sparse_research fartrack_sparse_224_full \
  --dataset_name got10k_val --threads 0 --num_gpus 1
$VENV scripts/evaluate_got10k_val.py --data-root "$DATA" \
  --results-root /root/autodl-tmp/experiment/.research-assets/output/test/tracking_results/fartrack_sparse_research/fartrack_sparse_224_full \
  --output /root/autodl-tmp/experiment/.research-assets/output/got10k_val_baseline_ao.json
```

Candidate commands use their own isolated worktree and tracker name.  For
parallel development screening, each process is an independent sequence
tracker and receives `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, and
`VECLIB_MAXIMUM_THREADS=1`; a final selected method will be rerun in the
single-process command above before B_test.

## Interpretation

Three hard interventions (write holding, rollback/replay, and hard
counterfactual rejection) all reduced AO.  The next hypotheses preserve the
baseline's per-frame template-write cadence and one-forward inference budget;
they only rebind the existing fixed template slots.  This is a constraint from
measured evidence, not a claimed improvement.
