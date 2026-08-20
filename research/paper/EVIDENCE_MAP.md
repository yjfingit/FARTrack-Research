# Draft Evidence Map

This map binds the manuscript's quantitative claims to local, immutable-or-
read-only artifacts. It is a draft ledger, not a human verification record.
The source manifest deliberately marks every external reference `unverified`
until an accountable author checks the source, citation metadata, and relevant
claim. The manuscript is not submission-ready.

| Claim group | Local evidence | Scope | Status |
| --- | --- | --- | --- |
| Complete B_dev baseline and Nodes 1--3/6 AO, CIs, wins/losses, FPS | `.arbor/sessions/rgb_tracking_training_free_20260819/experiments/{baseline,1,2,3,6}/metrics.json`; `/root/autodl-tmp/experiment/.research-assets/output/got10k_val_*_ao.json` | 180 GOT-10k validation sequences | Raw AO JSON retained; human audit pending |
| Passive entropy/disagreement reliability | `.arbor/.../experiments/7/metrics.json`; `/root/autodl-tmp/experiment/.research-assets/output/node7_reliability/full_bdev_metrics.json` | 180 sequences; 20,827 tracked frames | Logs and metric summary retained |
| Split-controlled mechanism decisions | `.arbor/.../experiments/8` through `17` reports and metrics | Calibration/selection/focused as stated per row | Do not pool with complete B_dev results |
| CSPP action-advantage audit | `results/17.1-risk-stratified-action-advantage/{summary.json,strata.csv,strata.png}` | 60 calibration plus 60 held selection sequences | Read-only; no tracker execution or B_test access |
| Baseline-only LaSOT reference | `/root/autodl-tmp/experiment/.research-assets/output/lasot_btest_baseline_metrics.json` | 280 Protocol-II sequences | Baseline only; not a candidate generalization result |
| Venue format | `SUBMISSION_TARGET.md`; source E008 | ICASSP 2027 | Official template kit still required |

The relevant reproducibility commands are in `research/EXPERIMENT_LOG.md`,
`research/NEGATIVE_STUDY_AUDIT.md`, and each Arbor experiment report. No local
data, checkpoint, raw result tree, or official evaluator is committed to Git.
