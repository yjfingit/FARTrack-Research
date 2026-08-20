# Cross-Model Risk-Action Study for ICASSP 2027

This directory extends, but does not overwrite, the prior FARTrackSparse
negative study. It tests a narrower cross-model question: whether a frozen
RGB tracker's own risk signal predicts a **better action**, rather than merely
a likely error.

The model registry, split contract, stopping rules, evidence ledger, and paper
assets are all versioned here. Upstream source caches, data, checkpoints, and
raw result trees remain on the data disk and are never committed.

## Scope

- **Task:** training-free RGB single-object tracking.
- **Target venue:** ICASSP 2027.
- **Question:** can an internal risk signal identify frames where a prescribed,
  low-cost action causally improves tracking quality?
- **Protocol:** calibration, selection, and confirmation use disjoint public
  GOT-10k validation partitions. LaSOT Testing is reserved for an externally
  confirmed configuration only.
- **Models:** SiamRPN++, OSTrack, MixFormer-Online (official MixFormerV2
  implementation), ODTrack, and FARTrackSparse.

The study is deliberately falsifiable: risk separability without a positive
action advantage is recorded as a negative result rather than promoted as a
method improvement.
