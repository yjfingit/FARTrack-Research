# Audit: Training-Free FARTrackSparse Negative Study

Audit date: 2026-08-20. This document separates directly reproducible complete
development evidence from split-controlled screening evidence. It does not claim
that all training-free trackers, or all post-hoc adaptations, are ineffective.

## Scope and Immutable Boundary

- **Model:** released `FARTrackSparse_ep0015.pth.tar` checkpoint.
- **B_dev:** public GOT-10k validation, 180 sequences and 21,007 frames.
- **B_test:** LaSOT Testing, 280 sequences. Candidate methods did not access it.
- **Metric:** frame-weighted mean IoU (AO), calculated from normal tracker result
  files by `scripts/evaluate_got10k_val.py`.
- **Training:** excluded. CTQA is not an experiment in this study: it has zero
  optimizer updates, no trained checkpoint, no candidate B_dev score, and no
  B_test access.

The B_dev evaluator currently has SHA-256
`383b5a04340109b6dc1b84c1863740532c610f9e4913230a88e8a0f6b78b1382`, equal
to its committed read-only evaluator version. `tracking/test.py` has no working
tree modification. Node 7 recorded byte-identical result files to the immutable
baseline for all 180 sequences, establishing that passive reliability logging
does not change prediction files. The audit does not treat those checks as a
proof that every independent candidate branch is correct; each candidate's
separate smoke/parity checks and report remain the applicable evidence.

## Evidence Ledger

`Complete B_dev` means one candidate and the immutable baseline were compared
over all 180 validation sequences. `Calibration` and `selection` are the fixed
ID-modulo-three partitions (60 sequences each) and must never be compared as
complete-development scores. `Focused` is a fixed small screen and is not a
generalization estimate.

| Node | Intervention family and action | Information used / independent evidence? | Evaluation tier | Observed result | Decision and evidence status |
| --- | --- | --- | --- | --- | --- |
| Baseline | Released frozen tracker | Current RGB, existing templates and logits | Complete B_dev | AO 0.832741; 40.07 FPS | Immutable reference; raw AO JSON retained |
| 1 | Transactional ledger: hold/rollback template writes | Internal anomaly signals; no independent evidence | Complete B_dev | AO 0.822471; delta -0.008228; 95% CI [-0.019499, 0.004104]; 42.16 FPS | Rejected; raw AO JSON retained |
| 2 | Counterfactual old/recent template agreement gate | Extra model forwards on same image/template evidence | Complete B_dev | AO 0.806470; delta -0.019995; 95% CI [-0.037073, -0.003256]; 23.47 FPS | Rejected; raw AO JSON retained |
| 3 | Stable-anchor replay after persistent alarm | Existing template/history evidence | Complete B_dev | AO 0.797473; delta -0.032038; 95% CI [-0.050311, -0.015034]; 40.90 FPS | Rejected; raw AO JSON retained |
| 4 / 4.1 | Appearance-diverse descriptor reservoir | Untrained crop RGB descriptors; no independent evidence | Focused 80-frame screen | +0.000838 AO; -16.57% FPS in final paired screen | Efficiency gate failed; not a full result |
| 5 | Recent-consensus temporal rebinding | Untrained crop RGB descriptors | Focused 60-frame screen | delta -0.001973 | Directionality failed; not a full result |
| 6 | Existing sampler switched to logarithmic history | Existing template chronology | Complete B_dev | AO 0.830650; delta -0.002730; 95% CI [-0.010716, 0.003984]; 31.32 FPS | Rejected; raw AO JSON retained |
| 7 | Passive entropy/margin/branch-disagreement probe | Frozen coordinate posterior; no action | Complete B_dev diagnostic | entropy low-IoU AUROC 0.906718, AP 0.368611; 180/180 output parity | Positive diagnostic only; full logs retained |
| 8 | Dual-lane entropy-gated memory | Node 7 entropy; no independent evidence | Held selection | delta -0.003999; -15.6% FPS | Rejected before confirmation; session metric retained |
| 9 | Entropy-gated motion-state mixing | Node 7 entropy plus past boxes | Held selection | delta +0.000644; CI [-0.019301, 0.020574]; -58.6% FPS | Effect and efficiency gate failed |
| 10 | Branch-disagreement motion mixing | Existing coordinate branches and past boxes | Held selection | delta -0.001688; repeated timing median -1.82% | Rejected before confirmation |
| 11 | One-frame delayed search expansion | Branch disagreement; enlarged next-frame crop | Held selection | delta -0.003339; +18.3% FPS | Rejected before confirmation |
| 12 | q95 two-view crop arbitration | Additional forward/crop from same RGB frame | Calibration | best delta -0.000046; -11.64% FPS | Calibration gate failed |
| 13 | Entropy-age recent-window reader | Entropy, template age, existing history | Calibration | delta -0.000685; +10.77% FPS | Calibration gate failed; summaries retained |
| 14 | Posterior-shape mean/mode projection | Existing per-coordinate posterior | Calibration | best gated AO 0.837691 vs 0.847678 baseline | Calibration gate failed; summaries retained |
| 15 | Disagreement-conditioned token pruning | Existing attention masks and branch disagreement | Calibration | best delta -0.009010; about -1.1% FPS | Calibration gate failed; summaries retained |
| 16 | Forward-backward sparse-flow arbitration | Independent algorithm, but same RGB frames | Focused two-sequence screen | changed screen delta -0.002827; -27.94% FPS | Feasibility gate failed; not a full result |
| 17 | Horizontal-only posterior expectation correction | Existing coordinate branches | Held selection | delta -0.001162; +17.12% FPS | Rejected before confirmation; valid selection JSON retained |
| 17.1 | Read-only CSPP conditional action-advantage audit | Node 7 passive risk logs plus retained Node 17 predictions | Held selection, descriptive | entropy-q85 +0.009409 and disagreement-q85 +0.008231 frame-IoU deltas, but both sequence-bootstrap CIs cross zero; changed-frame delta -0.002076 | No tuning or new run; supports bounded risk/action distinction |
| 1.1 | Read-only complete-B_dev loss atlas | Retained baseline and Nodes 1/2/3/6 per-sequence AO JSONs | Complete B_dev, descriptive | all four have mixed sequence effects; three memory controls have negative medians; logarithmic top-five loss share 55.89% | No new method; documents failure heterogeneity |
| 18 | CTQA learned trajectory adapter | Would require training data and optimizer | Not evaluated | no efficacy result | Excluded by scope |

`node8`, `node9`, `node10`, `node11`, and `node16` do not currently retain
their complete raw tracker-result trees in the data-disk output root; their
session `metrics.json`, pre-registration, commands, and executor report remain.
They can support transparent screening tables, but not new unplanned
per-frame/post-hoc analyses without a pre-registered rerun. This is an artifact
retention limitation, not evidence of a positive or negative effect.

## Split and Test Audit

1. The complete-development table contains only baseline, Nodes 1--3, and Node
   6. It is valid to compare these five rows directly because each has 180
   sequences and 21,007 frames.
2. Nodes 8--17 used a calibration/selection/confirmation protocol. Their
   calibration and selection scores are intentionally not pooled with complete
   B_dev AO. No node reached its stated selection criterion.
3. LaSOT Testing was accessed once for the immutable baseline only: success AUC
   0.614788, precision@20 0.644600, normalized-precision AUC 0.639928. No
   candidate result exists on B_test.
4. All candidate reports state that the original result writer and development
   dataset were left untouched. The coordinator worktree does not modify
   `tracking/test.py` or `scripts/evaluate_got10k_val.py`; branch-level diffs
   and smoke records must remain part of each artifact bundle.

## What the Existing Evidence Supports

For this checkpoint and protocol, the following is supported:

1. Several qualitatively distinct post-hoc controls of template memory and
   sampling failed to improve complete B_dev AO.
2. Frozen posterior entropy is a strong predictor of a low-IoU event, but the
   tested risk-conditioned actions did not meet their pre-registered effect and
   efficiency gates.
3. Altering coordinate decoding, template read/write behavior, attention-mask
   pruning, local search, or a sparse-flow correction did not provide a verified
   deployment benefit under their stated tiers.

Node 17.1 makes the key diagnostic distinction explicit using only retained
outputs. It estimates q85 entropy and branch-disagreement thresholds on the 60
calibration sequences, then aligns Node 7 byte-parity baseline files and the
valid CSPP selection run over 60 held selection sequences (7,955 non-initial
frames). High-risk strata have positive mean deltas (+0.009409 for entropy and
+0.008231 for disagreement), but their sequence-bootstrap 95% intervals are
[-0.017325, 0.040717] and [-0.016408, 0.033060]. The 4,487 frames whose CSPP
box actually changed have mean delta -0.002076 with interval [-0.025566,
0.020478]. This descriptive, post-hoc analysis does not establish a deployable
positive action advantage and cannot be used to retune CSPP.

Node 1.1 separately audits all 180 per-sequence AO deltas for the four complete
B_dev candidates. It finds mixed sequence effects rather than uniform decline:
TRM, counterfactual agreement, and stable-anchor recovery have medians
-0.003931, -0.004659, and -0.007263, while their largest five losses account
for 33.76%, 35.26%, and 33.54% of total negative delta. Logarithmic sampling has
a near-zero median (-0.000534) but 55.89% of total loss in its five worst
sequences. This supports a heterogeneous-failure interpretation, not an
unsupported semantic attribution to any one sequence.

The following is **not** supported: a universal impossibility theorem, a claim
that the baseline is Bayes-optimal, a comparison of calibration/focused results
to the complete B_dev table, or a claim about candidate generalization to LaSOT.

## Paper Outline

Provisional title: **When Confidence Is Not Correction: A Controlled Negative
Study of Training-Free Post-hoc Adaptation for RGB Single-Object Tracking**.

1. **Introduction.** State the practical question and the narrow study scope;
   distinguish useful negative evidence from a universal claim.
2. **Problem formulation.** Let \(Z_t\) be the frozen tracker information,
   \(a_0(Z_t)\) the released action, and \(a_g(Z_t)\) a deterministic post-hoc
   action. Define conditional action advantage
   \(A_g(Z_t)=\mathbb{E}[\operatorname{IoU}(a_g(Z_t),B_t^*)-
   \operatorname{IoU}(a_0(Z_t),B_t^*)\mid Z_t]\).
3. **Why risk is not correction.** Explain that a risk score estimates an
   error event, whereas a useful gate needs the sign of \(A_g\). Give the
   local quadratic argument, stale-template/contamination trade-off, motion
   residual covariance condition, joint-IoU versus marginal-coordinate issue,
   and the independent-observation requirement. The data-processing inequality
   is presented only as a limitation of post-processing, not optimality proof.
4. **Protocol and integrity controls.** Frozen checkpoint, B_dev/B_test
   separation, result writer/evaluator integrity, parity checks, predefined
   stopping gates, statistics, and artifact retention limits.
5. **Intervention taxonomy.** Memory, motion/state, sensing/search, posterior
   decoding, token control, and sparse flow, each with its available evidence
   tier.
6. **Results.** First present the complete B_dev table. Then a clearly separate
   split-controlled table and a clearly separate focused-screen table. Report
   AO, delta, paired CI where available, wins/losses, FPS, and stop point.
7. **Failure analysis.** Passive reliability result; quantitative conditional
   action-advantage analysis when existing retained outputs allow it; qualitative
   failure cases; limits from missing retained raw trees.
8. **Related work and recommendations.** Contrast training-free post-hoc
   controls with training/test-time adaptation and evidence-adding methods.
   Recommend proving a positive conditional action advantage or introducing
   independent evidence before deploying a risk gate.
9. **Limitations, reproducibility, and conclusion.** State one checkpoint,
   public development split, no candidate B_test result, no universal theorem,
   AI-assistance disclosure, exact commands/commits/output locations.

## Pre-GPU Evidence Gap Plan

No new GPU B_dev run is justified until the retained Node 7 logs and paired
candidate results can answer a pre-registered question: does a risk-stratified
candidate have a reproducible signed advantage over baseline rather than merely
a high error-prediction score? If the answer cannot be recovered from retained
artifacts, the paper will disclose that boundary rather than retrying existing
template, motion, threshold, or marginal-posterior mechanisms.
