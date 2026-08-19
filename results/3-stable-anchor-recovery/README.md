# Node 3: Stable-anchor recovery scheduler

`sar_far` is an independent research tracker. It keeps the released FARTrackSparse checkpoint and the official `fartrack_sparse` implementation untouched.

On a persistent causal failure run, the scheduler freezes new writes and replays only the initial anchor plus the last accepted view for two frames. An eight-frame cooldown bounds repeated replay. This artifact contains implementation and smoke-test provenance only; no LaSOT score is claimed until the pre-registered B_dev data are available.

Validation used `/root/autodl-tmp/experiment/.venvs/fartrack-research/bin/python`:

```bash
python - <<'PY'
from tests.test_stable_anchor_recovery import *
for test in (test_stable_commits_preserve_immutable_anchor_and_mask_shape,
             test_persistent_failure_replays_anchor_and_last_accepted_without_writes,
             test_recovery_is_bounded_by_cooldown):
    test()
PY
```

All three tests passed. A CUDA smoke also loaded the frozen 84 MB
`FARTrackSparse_ep0015.pth.tar` and tracked three synthetic RGB frames through
`sar_far`; every forward used five templates and a boolean `[1, 445, 445]`
mask. The synthetic run committed all three frames, as expected for its stable
input; recovery activation is verified deterministically by the unit test.
