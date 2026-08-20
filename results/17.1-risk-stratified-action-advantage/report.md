# Experiment 17.1: Risk-Stratified Conditional Action Advantage

## Purpose

This is a read-only follow-up to Node 17, not a new tracker or a tuning run.
It tests whether Node 7 reliability signals identify frames on which the valid
Node 17 CSPP action has a reproducibly positive signed IoU advantage over the
released baseline. The question is distinct from whether a signal identifies a
baseline failure.

## Protocol and Integrity

- Public GOT-10k validation only; B_test was not read or accessed.
- Baseline prediction files are from the Node 7 reliability tracker. The 60
  selection files were checked byte-for-byte equal to immutable baseline files.
- Candidate predictions are the valid CSPP run `1705`; the invalid `1702` run
  is not read by this analysis.
- Risk thresholds are q85 values estimated only from the calibration partition
  (sequence ID modulo 3 equals 1): entropy `0.4000612944364548` and branch
  disagreement `0.0037756040692329407`.
- The held selection partition is sequence ID modulo 3 equals 2: 60 sequences,
  8,015 result rows, and 7,955 non-initialization frames that have reliability
  logs.
- The script reads prediction files, GT, and JSONL logs only. It neither imports
  tracking code nor invokes the official evaluator.

## Result

| Selection stratum | Frames | Mean CSPP - baseline IoU | 95% sequence-bootstrap CI | Positive-frame fraction |
| --- | ---: | ---: | --- | ---: |
| All tracked frames | 7,955 | -0.001171 | [-0.014158, 0.011442] | 0.2683 |
| Entropy q85 high | 1,339 | +0.009409 | [-0.017325, 0.040717] | 0.3854 |
| Entropy q85 low | 6,616 | -0.003312 | [-0.014908, 0.006163] | 0.2446 |
| Disagreement q85 high | 1,272 | +0.008231 | [-0.016408, 0.033060] | 0.3797 |
| Disagreement q85 low | 6,683 | -0.002960 | [-0.014953, 0.007677] | 0.2470 |
| Frames whose CSPP box changed | 4,487 | -0.002076 | [-0.025566, 0.020478] | 0.4756 |

The high-risk strata have positive point estimates, which is compatible with a
possible conditional benefit. However, both sequence-bootstrap intervals cross
zero substantially, and the actual changed-frame estimate is negative with a
wide interval. The selection result therefore does not establish a reproducible
positive action advantage. It strengthens the paper's bounded conclusion: a
failure-predictive score may correlate with an attractive point estimate yet be
insufficient to justify a deployed replacement action at this sample size and
protocol. This is descriptive post-hoc analysis of a frozen candidate; it is
not a fresh confirmation test and must not be used to retune CSPP.

## Reproduction

```bash
VENV=/root/autodl-tmp/experiment/.venvs/fartrack-research/bin/python
$VENV scripts/analyze_cspp_action_advantage.py \
  --data-root /root/autodl-tmp/experiment/.research-assets/data/got10k/val \
  --baseline-root /root/autodl-tmp/experiment/.research-assets/output/test/tracking_results/fartrack_sparse_reliability/fartrack_sparse_224_full/got10k \
  --candidate-root /root/autodl-tmp/experiment/.research-assets/output/test/tracking_results/fartrack_sparse_cspp/fartrack_sparse_224_full_1705/got10k \
  --log-root /root/autodl-tmp/experiment/.research-assets/output/node7_reliability/logs \
  --output-dir results/17.1-risk-stratified-action-advantage \
  --bootstrap-samples 10000 --bootstrap-seed 20260820
```

Generated artifacts are `summary.json`, `strata.csv`, and `strata.png` in this
directory. The PNG is generated exclusively from retained public B_dev results.
