# Node 3: OSTrack Passive Risk Probe

## Idea

Without changing an OSTrack decision, test whether uncertainty already present
in the post-Hann response map predicts future low localization overlap.

## Changes

The isolated OSTrack worktree adds an opt-in `ostrack_risk` tracker entrypoint.
When `OSTRACK_RISK_LOG_PATH` is set, it records response entropy, peak, and
peak margin *after* the baseline response is formed and before the unchanged
box-head call.  With the hook disabled, the added branch does not execute.
`scripts/analyze_ostrack_passive_risk.py` is a separate read-only analysis of
the output tracks, GT, and logs.

## Protocol

- Official OSTrack ViT-B/384 checkpoint and unmodified GOT-10k validation
  images/annotations; no train split was read.
- Risk target: a non-initialization frame with IoU below 0.2.
- Risk direction: larger response entropy means greater risk.
- 2,000 sequence-resampling bootstrap replicates, seed `20260820`.
- The passive risk tracker was evaluated on all 180 validation sequences.

## Reproduction

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
OSTRACK_RISK_LOG_PATH=PATH.jsonl PYTHONPATH=UPSTREAM \
python RUN_LEGACY_OSTRACK.py UPSTREAM/tracking/test.py ostrack_risk \
vitb_384_mae_ce_32x4_ep300 --dataset_name got10k_val --threads 0 --num_gpus 1

python scripts/analyze_ostrack_passive_risk.py --data-root DATA/got10k/val \
--baseline-results BASELINE/got10k --risk-results PASSIVE/got10k \
--risk-log SEQ1.jsonl --risk-log SEQ2.jsonl --risk-log REMAINING.jsonl \
--output passive_risk_bdev.json
```

## Measured Result

| quantity | value |
| --- | ---: |
| sequences / non-init frames | 180 / 20,827 |
| low-IoU prevalence | 0.021703 |
| entropy AUROC | 0.867915 |
| entropy AP | 0.270585 |
| entropy vs IoU Spearman | -0.587119 |
| AUROC bootstrap 95% CI | [0.812824, 0.912209] |
| AP bootstrap 95% CI | [0.114976, 0.442848] |
| Spearman bootstrap 95% CI | [-0.640270, -0.522730] |
| output SHA256 mismatches | 0 / 180 |

Result JSON:
`/root/autodl-tmp/experiment/.research-assets/output/cross_model/ostrack_risk/metrics/passive_risk_bdev.json`.

## Score

There is no candidate AO score because this is a passive probe.  The immutable
OSTrack baseline remains AO `0.8651649115`, FPS `40.008971`.

## Insight

OSTrack response entropy strongly ranks low-IoU frames while leaving all
baseline predictions byte-identical.  This validates a risk-prediction signal,
not a corrective action.  The next node must pre-register one action and use
the fixed calibration/selection split before it can make a causal claim.
