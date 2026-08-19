# Entropy-Conditioned Causal Motion Mixing

This directory records node 9 under a three-way public GOT-10k validation
development protocol. Numeric sequence IDs with `id % 3 == 1` form the
calibration partition, `== 2` the untouched selection partition, and `== 0`
the confirmation partition. No B_test data was used.

`calibration_signal_summary.json` derives entropy quantiles exclusively from
node-7 passive logs on the calibration partition. Before selection,
`preselection.json` freezes threshold `q0.85 = 0.4000612944364548`, the three
calibration weights, selection rule, and selection gates. Its SHA256 is in
`preselection.sha256`.

The predeclared calibration schedule selected `alpha=0.10` as the least
negative candidate: AO deltas for `0.10`, `0.25`, and `0.50` were `-0.005446`,
`-0.018927`, and `-0.052148`. On untouched selection, the frozen `alpha=0.10`
policy obtained `+0.000644` AO, below the `+0.003` gate, and a `-58.6%` FPS
change, below the `-5%` efficiency budget. It is rejected without
confirmation or B_test.

The `obsolete_pre*` directories are intentionally untracked local audit
artifacts from two pre-optimization implementations. They document why no
metric from a version that introduced an avoidable host synchronization or a
redundant softmax was used in the decision.
