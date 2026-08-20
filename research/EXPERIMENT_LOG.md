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
| logarithmic sampler | 0.830650 | -0.002730 | [-0.010716, 0.003984] | 31.32 | rejected |

## Split-Controlled Reliability and Decision Studies

After the complete-development results above were frozen, a separate
calibration/selection/confirmation partition of GOT-10k validation was used
for diagnostics and narrowly pre-registered interventions. These runs are not
comparable to the full-development table and none reached its selection gate or
accessed B_test.

| Node | Mechanism | Development evidence | Decision |
| --- | --- | --- | --- |
| 7 | Passive localization reliability probe | Coordinate entropy: low-IoU AUROC 0.906718; branch disagreement AUROC 0.806355; outputs byte-identical on 180/180 sequences | diagnostic only |
| 8 | Entropy-conditioned dual-lane memory | selection AO -0.003999; FPS -15.6% | rejected |
| 9 | Entropy-conditioned state mixing | selection AO +0.000644, 95% CI [-0.019301, 0.020574]; FPS -58.6% | rejected |
| 10 | Branch-disagreement state mixing | selection AO -0.001688, 95% CI [-0.023716, 0.017422] | rejected |
| 11 | Delayed search expansion | selection AO -0.003339 | rejected |
| 12 | Rare two-view arbitration | calibration AO -0.000046 at 1.20x; FPS -11.64% | rejected before selection |
| 13 | Entropy-age conditional recent read | calibration AO -0.000685; FPS +10.77% | rejected before selection |
| 14 | Posterior-shape adaptive projection | best gated calibration AO 0.837691 versus 0.847678 baseline | rejected before selection |
| 15 | Disagreement-conditioned token quarantine | best q85 adaptive 75% pruning AO -0.009010 | rejected before selection |
| 16 | Bidirectional sparse-flow arbitration | focused screen AO -0.002827 and triggered FPS -27.94% | rejected before calibration |
| 17 | Coordinate-selective posterior projection | frozen selection AO -0.001162 (0.825130 versus 0.826292); FPS +17.12% | rejected before confirmation |
| 18 | Causal trajectory-query adapter (CTQA) | source-only checkpoint-compatibility screen; no optimizer step, trained checkpoint, B_dev candidate run, or B_test access | abandoned: outside the final training-free scope |

Node 7 established that the frozen coordinate posterior is strongly predictive
of failure but did not improve any box and did not alter the base tracker.
Nodes 8--14 test distinct actions rather than treating this predictive
association as causal evidence. Their reports, exact frozen split definitions,
commands, and raw-result paths are in `.arbor/sessions/rgb_tracking_training_free_20260819/experiments/7`
through `14` in the coordinator worktree. Raw outputs remain on the data disk
under `/root/autodl-tmp/experiment/.research-assets/output/`.

Node 17 applied a single pre-registered, deployable horizontal-center correction
only when coordinate-branch disagreement was high. Its earlier passive analysis
did not translate into a causal AO gain on the frozen selection partition, so it
was not confirmed or evaluated on B_test. Node 18 identified a real architectural
fact--the sparse backbone receives a trajectory tensor but replaces it with
command tokens--and implemented a zero-initialized causal trajectory-query
adapter only to verify checkpoint compatibility. The user then reaffirmed the
training-free objective. CTQA is therefore excluded from the research route:
there were zero optimizer updates, no trained CTQA checkpoint, no CTQA B_dev
candidate score, and no CTQA B_test access.

Node 15 verified that the native mask levels retain 37/25/13/5 of 49 template
tokens for 25/50/75/90% pruning respectively. Even rare q85 adaptive stronger
pruning was harmful, so the low-attention tokens cannot be treated as safely
discardable for this checkpoint. Node 16 was an explicitly prior-art-aware
feasibility ablation using bidirectionally validated sparse optical flow. It
failed both direction and real-time gates on two fixed screens and was stopped
before calibration.

The bootstrap resamples the 180 per-sequence AO deltas with seed `20260819`
for 10,000 draws.  No rejected method was run on B_test or tuned after its
full B_dev result.

## Scope Correction: Unused Training Mirror

An archive containing GOT-10k training data was downloaded to the data disk
while CTQA was being prepared as a contingency. Only its `train/` directory was
structurally inspected after archive validation; its `test/` directory was never
extracted, read, or used. No training process imported the data: two launcher
attempts stopped during missing-package imports before an optimizer was created.
The archive and extracted training directory are retained solely as unused,
auditable artifacts on the data disk. They are not a source of any reported
metric, model, checkpoint, or paper claim.

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

Three hard interventions, a soft rebinding screen, and a zero-extra-forward
logarithmic temporal sampler all failed their complete-development or prespecified
screen gates.  No candidate is claimed to improve the frozen baseline.

## Held-Out Baseline Reference

After candidate selection was frozen, the immutable baseline was run once on
the verified official LaSOT Testing archive (Protocol-II, 280 sequences). The
separate local, read-only summary returned success AUC `0.6147875189781189`,
precision at 20 pixels `0.6446002125740051`, and normalized precision AUC
`0.6399275660514832`. Results are in
`/root/autodl-tmp/experiment/.research-assets/output/lasot_btest_baseline_metrics.json`.
This is an external baseline reference, not a new-method result. No rejected
candidate was run on LaSOT.
