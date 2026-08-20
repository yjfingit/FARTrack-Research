# Node 1.2: Deterministic Complete-B_dev Failure-Case Rendering

## Question and Boundary

Can the completed negative results be supplemented with auditable qualitative
examples without executing a new tracker run or using B_test? Yes. This node
is a read-only documentation analysis, not a candidate mechanism.

## Fixed Selection Rule

The sequences were fixed from the retained complete-B_dev loss atlas: TRM-FAR
uses `GOT-10k_Val_000042`, counterfactual agreement and stable-anchor recovery
use `GOT-10k_Val_000108`, and logarithmic sampling uses
`GOT-10k_Val_000071`. Within every one of these named sequences, the rendered
frame is the deterministic `argmin(candidate_IoU - immutable_baseline_IoU)`
over aligned frames. GT is green, immutable baseline blue, and candidate red.

## Reproduction

```bash
/root/autodl-tmp/experiment/.venvs/fartrack-research/bin/python \
  scripts/render_complete_bdev_failure_cases.py \
  --data-root /root/autodl-tmp/experiment/.research-assets/data/got10k/val \
  --baseline-dir /root/autodl-tmp/experiment/.research-assets/output/test/tracking_results/fartrack_sparse_research/fartrack_sparse_224_full/got10k \
  --output-root /root/autodl-tmp/experiment/.research-assets/output/test/tracking_results \
  --output-dir results/1.2-complete-bdev-failure-cases \
  --case TRM-FAR:GOT-10k_Val_000042:trm_far \
  --case Counterfactual:GOT-10k_Val_000108:fartrack_sparse_cf \
  --case Stable-anchor:GOT-10k_Val_000108:sar_far \
  --case Logarithmic:GOT-10k_Val_000071:fartrack_sparse_sampling_logarithmic
```

## Observed Frames

| Method | Sequence / frame | Baseline IoU | Candidate IoU | Delta |
| --- | --- | ---: | ---: | ---: |
| TRM-FAR | `000042` / 74 | 0.951 | 0.000 | -0.951 |
| Counterfactual agreement | `000108` / 162 | 0.987 | 0.028 | -0.959 |
| Stable-anchor recovery | `000108` / 162 | 0.987 | 0.028 | -0.959 |
| Logarithmic sampler | `000071` / 50 | 0.943 | 0.000 | -0.943 |

The exact numbers are in `failure_cases.json` and `failure_cases.csv`. The
image is not representative-sampling evidence and makes no semantic-cause
claim. It only makes the retained per-frame geometric discrepancy inspectable.
