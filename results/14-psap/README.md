# Posterior-Shape Adaptive Projection (PSAP)

Node 14 evaluates a training-free output-only intervention for frozen
FARTrackSparse. The released tracker averages its discrete autoregressive
coordinate (`mode`) and its feature-bin posterior expectation (`mean`). PSAP
uses their already-computed normalized mean absolute disagreement to decide
whether the current output should instead use `mode` or `mean`. It changes no
network forward, crop, state prior, template read, or template write.

Development calibration uses public GOT-10k validation IDs with `id % 3 == 1`.
The untouched selection partition is `id % 3 == 2`, and confirmation is
`id % 3 == 0`. `preselection.json` fixes the finite policy family and all gates
before its gated calibration runs. Calibration did not select any candidate,
so neither selection nor held-out data was accessed.

The outcome is negative: every high-disagreement endpoint switch degraded AO.
This is consistent with the frozen FARTrack midpoint being a deliberately
calibrated fusion, rather than an uncertainty-dependent interpolation problem.
PSAP is therefore rejected and is not a paper contribution.

## Related-work check

Probabilistic visual tracking has long used predictive distributions and point
estimates, e.g. Danelljan et al., *Probabilistic Regression for Visual
Tracking*, CVPR 2020, https://openaccess.thecvf.com/content_CVPR_2020/papers/Danelljan_Probabilistic_Regression_for_Visual_Tracking_CVPR_2020_paper.pdf.
Uncertainty-guided tracking was also studied with learned online sampling by
Zhou et al., *Model Uncertainty Guides Visual Object Tracking*, AAAI 2021,
https://doi.org/10.1609/aaai.v35i4.16473. Thus PSAP would require positive
evidence and a FARTrack-specific claim about its native mode/mean branches;
this experiment supplies neither.
