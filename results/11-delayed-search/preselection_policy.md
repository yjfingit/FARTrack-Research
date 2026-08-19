# Node 11 preselection policy

This document is frozen before inspecting any Node 11 tracking result.

- Development split: public GOT-10k validation; calibration sequence numeric ID modulo 3 equals 1; selection equals 2; confirmation equals 0.
- Alarm: after the completed forward pass at frame `t`, `branch_disagreement_l1 >= 0.0037756040692329407`, where the threshold is the empirical q85 over node7's calibration-only logs (60 sequences, 6,531 tracked frames).
- Action: the alarm can multiply only the next frame crop factor, reset immediately after that crop, preserve `SEARCH_SIZE=224`, and does not alter current localization/state/template logic.
- Calibration candidates: multipliers `1.10`, `1.20`, and `1.30`; each uses the same frozen checkpoint and one forward per frame.
- Selection rule: choose the multiplier with the highest calibration frame AO among candidates whose aggregate FPS is no more than 5% below the immutable baseline on the same partition. Ties break toward the smaller multiplier. If no candidate satisfies the FPS condition, no selection run is permitted.
- Held selection gate: selected multiplier must improve frame AO by at least `+0.003` and lose no more than 5% aggregate FPS against the immutable baseline on ID modulo 3 equals 2.
- Confirmation: run only when the held selection gate passes, on ID modulo 3 equals 0. B_test is prohibited for this node.
