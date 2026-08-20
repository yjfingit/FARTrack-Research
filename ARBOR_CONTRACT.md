# Arbor Research Contract: Cross-Model Risk Is Not Action

- **Target:** isolated branch 'arbor/cross-model-risk-action-20260820'. The
  original FARTrack worktree, its uncommitted changes, external tracker caches,
  data, checkpoints, and official evaluators are protected.
- **Task:** conduct an honest, training-free cross-model RGB single-object
  tracking study of whether internal risk prediction supplies a positive
  conditional corrective-action advantage.
- **Models:** SiamRPN++, OSTrack, MixFormer-Online, ODTrack, and
  FARTrackSparse. FARTrackSparse reuses frozen prior evidence only.
- **Metric:** maximize mean frame IoU (AO) on public GOT-10k validation
  (B_dev). Every candidate comparison uses per-sequence AO deltas and a
  10,000-resample paired bootstrap interval.
- **Baseline anchor:** FARTrackSparse complete B_dev AO 0.832741, FPS 40.07;
  the other frozen baselines are unknown until their official checkpoint
  reproduction and coverage audit pass.
- **Development/test discipline:** B_dev is GOT-10k validation. B_test is
  LaSOT Testing. No candidate accesses B_test unless it passes the frozen
  selection gate in research/cross_model/protocol/STOPPING_RULES.md.
- **Hard constraints:** no training data, optimizer steps, fine-tuning,
  test-time optimization, adapters, or pseudo-labeling; no data, GT, official
  evaluator, result writer, or protocol modification; no fabricated results.
- **Edit surface:** only new research docs, independent worktree adaptations,
  configs, read-only evaluators, scripts, tables, figures, and paper files.
  Every model adaptation runs in its own data-disk worktree.
- **Resources:** downloads, pip caches, virtual environments, checkpoints,
  source mirrors, and outputs live under
  /root/autodl-tmp/experiment/.research-assets/. The upstream cache
  /root/autodl-tmp/experiment/external-trackers/ is read-only.
- **Budget:** real B_dev experiments only after source/checkpoint audit and
  baseline parity are established. Five models, each with one passive probe,
  one conservative action, and one information-control action at most.
