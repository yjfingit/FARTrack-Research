# Final Report: Training-Free FARTrackSparse Negative Study

## Status

This is a completed **negative-evidence research package** for the released,
frozen FARTrackSparse epoch-15 checkpoint. It is not a claimed new tracker.
The paper target is IEEE ICASSP 2027; the present spconf layout is a four-page
working proxy and must be replaced with the official author kit before
submission.

## Research Question and Answer

**Question.** Can post-hoc, training-free controls of a frozen RGB
single-object tracker use its own confidence, memory, history, and posterior
signals to improve tracking reliably?

**Answer supported by this package.** Not for the tested fixed FARTrackSparse
implementations and protocol. Four complete public GOT-10k validation
evaluations were all negative in mean frame IoU. A passive uncertainty signal
was strongly predictive of low IoU, but the risk-conditioned actions tested
did not establish a positive conditional action advantage. This is a bounded
empirical conclusion, not an impossibility theorem.

The mathematical distinction is:

\[
 A_g(Z_t)=\mathbb{E}[\mathrm{IoU}(a_g(Z_t),B_t^*)-
 \mathrm{IoU}(a_0(Z_t),B_t^*)\mid Z_t].
\]

A score that predicts a low-IoU event can have high AUROC without determining
the sign of the action advantage. A threshold action therefore needs signed,
causal action advantage, rather than failure-prediction accuracy alone.

## Valid Complete B_dev Results

B_dev is public GOT-10k validation: 180 sequences and 21,007 frames. AO is
mean frame IoU from normal tracker result files. Intervals are paired
per-sequence bootstrap percentile 95% intervals, 10,000 resamples with seed
20260819.

| Method | AO | Delta AO | 95% paired interval | FPS | Decision |
| --- | ---: | ---: | --- | ---: | --- |
| Immutable FARTrackSparse | 0.832741 | 0.000000 | -- | 40.07 | reference |
| Transactional template ledger (TRM-FAR) | 0.822471 | -0.008228 | [-0.019499, 0.004104] | 42.16 | reject |
| Counterfactual template agreement | 0.806470 | -0.019995 | [-0.037073, -0.003256] | 23.47 | reject |
| Stable-anchor recovery | 0.797473 | -0.032038 | [-0.050311, -0.015034] | 40.90 | reject |
| Existing logarithmic sampler | 0.830650 | -0.002730 | [-0.010716, 0.003984] | 31.32 | reject |

The complete-effect audit is in results/1.1-complete-bdev-loss-atlas/report.md.
It shows mixed per-sequence outcomes, but negative means and medians for the
three memory controls. The logarithmic sampler is closest to neutral yet has
its top five losses accounting for 55.89% of total negative sequence delta.

## Diagnostic and Split-Controlled Results

The passive Node 7 probe left all 180 baseline result files byte-identical. Its
coordinate entropy predicts an IoU below 0.2 with AUROC 0.906718, AP 0.368611,
and Spearman(IoU) -0.659318. This is diagnostic evidence, not an improvement.

Pre-registered calibration/selection screens rejected the following actions
before confirmation: entropy-conditioned dual-lane memory, causal state
mixing, branch-disagreement state mixing, delayed search expansion, rare
two-view arbitration, entropy-age recent reading, posterior projection,
disagreement-conditioned token pruning, sparse-flow arbitration, and a
horizontal-only coordinate correction. Their exact evaluation tier, result,
stop point, and artifact-retention status are in NEGATIVE_STUDY_AUDIT.md.

The read-only conditional-action audit of the final valid coordinate-correction
selection outputs is in results/17.1-risk-stratified-action-advantage/report.md.
High-risk point estimates are positive, but their 95% sequence-bootstrap
intervals cross zero; frames whose boxes changed are negative on average.
This supports the risk-versus-action distinction but cannot justify retuning.

## Qualitative Evidence

results/1.2-complete-bdev-failure-cases/failure_cases.png is generated only
from existing result files, public B_dev frames, and GT. For each complete
candidate it selects a fixed lowest-sequence-AO sequence, then the frame
minimizing candidate-minus-baseline IoU. Exact frame selections and overlaps
are retained in JSON and CSV alongside the image. The panels show geometric
discrepancies only; they are not a semantic or causal attribution.

## Development/Test Separation

- Candidate decisions use B_dev only.
- Full official LaSOT Testing is B_test. No rejected candidate ran on B_test.
- After decisions were frozen, the immutable baseline was evaluated once on
  LaSOT Protocol-II (280 sequences): success AUC 0.614788, precision@20
  0.644600, normalized precision AUC 0.639928. This is a baseline reference,
  not a new-method generalization result.
- The final study contains no training. CTQA source experiments were abandoned:
  zero optimizer updates, no trained CTQA checkpoint, no CTQA candidate B_dev
  score, and no CTQA B_test access.

## Reproduction

Environment and immutable paths:

    VENV=/root/autodl-tmp/experiment/.venvs/fartrack-research/bin/python
    REPO=/tmp/arbor-rgb-tracking-research-20260819
    DATA=/root/autodl-tmp/experiment/.research-assets/data/got10k/val
    CKPT=/root/autodl-tmp/experiment/.research-assets/checkpoints/FARTrackSparse_ep0015.pth.tar
    OUT=/root/autodl-tmp/experiment/.research-assets/output
    cd "$REPO"

Immutable baseline tracking and read-only B_dev scoring:

    $VENV tracking/test.py fartrack_sparse_research fartrack_sparse_224_full \
      --dataset_name got10k_val --threads 0 --num_gpus 1
    $VENV scripts/evaluate_got10k_val.py --data-root "$DATA" \
      --results-root "$OUT/test/tracking_results/fartrack_sparse_research/fartrack_sparse_224_full" \
      --output "$OUT/got10k_val_baseline_ao.json"

All historical candidate commands, isolated branches, result paths, and
stopping decisions are retained in EXPERIMENT_LOG.md, NEGATIVE_STUDY_AUDIT.md,
and .arbor/sessions/rgb_tracking_training_free_20260819/experiments/. The
two new read-only analyses reproduce without tracker execution:

    $VENV scripts/analyze_complete_bdev_loss_atlas.py --help
    $VENV scripts/analyze_cspp_action_advantage.py --help
    $VENV scripts/render_complete_bdev_failure_cases.py --help

The fully specified failure-case rendering command is in
results/1.2-complete-bdev-failure-cases/report.md. All data, checkpoints, and
raw outputs remain on the data disk under
/root/autodl-tmp/experiment/.research-assets/; none are committed to Git.

To compile the manuscript:

    cd "$REPO/research/paper"
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

## Paper and Positioning

- LaTeX: research/paper/main.tex
- PDF: research/paper/main.pdf
- Evidence map: research/paper/EVIDENCE_MAP.md
- Venue/format checklist: research/paper/SUBMISSION_TARGET.md
- Related-work boundary: uncertainty-aware tracking is acknowledged as prior
  work; this paper claims neither uncertainty gating nor optical-flow
  arbitration as a conceptual innovation.

The contribution is the controlled negative finding and its evidence
discipline: frozen checkpoint, unchanged evaluator/result writer, B_dev/B_test
separation, parity checks, pre-specified stopping gates, effect-size intervals,
retained qualitative provenance, and a formal risk-versus-action criterion.

## Remaining Work Before Submission

1. Download and use the official ICASSP 2027 author kit once released or linked
   by the venue; the current spconf file is only a proxy.
2. Have accountable authors verify every external citation and every local
   artifact claim in the evidence map. The source manifest deliberately marks
   external sources as unverified until that review.
3. Complete author list, affiliation, funding, conflict, data/code
   availability, and AI-use declarations under current IEEE policy.
4. Obtain venue-appropriate human peer review. This package reports one
   checkpoint and one full public development split, so the paper must retain
   its bounded claim and limitations.
5. Preserve future-run lifecycle metadata from the start. The current session
   has report and metrics coverage, but some early executor prompts plus the
   global events.jsonl and run_stats.json were not retained; strict Arbor
   artifact validation therefore remains unavailable for those historical
   nodes. No missing event history has been reconstructed or fabricated.
