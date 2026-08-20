# Cross-Model Risk-Action Study

This directory extends, but does not overwrite, the prior FARTrackSparse
negative study. It tests a narrower cross-model question: whether a frozen
RGB tracker's own risk signal predicts a **better action**, rather than merely
a likely error.

The model registry, split contract, stopping rules, evidence ledger, and paper
assets are all versioned here. Upstream source caches, data, checkpoints, and
raw result trees remain on the data disk and are never committed.
