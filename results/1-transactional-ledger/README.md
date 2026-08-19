# Node 1 smoke evidence

No benchmark score is stored here: the fixed LaSOT development data were not
available in this worktree.  The implementation was checked with the released
`FARTrackSparse_ep0015.pth.tar` checkpoint and synthetic RGB frames.

```bash
CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/experiment/.venvs/fartrack-research/bin/python \
  -m pytest -q tests/test_transactional_memory.py
```

`pytest` is not installed in the shared environment, so the three test
functions were directly executed with the same interpreter.  They covered
commit/packing, sustained-anomaly hold-to-rollback, and anchor preservation.
The frozen checkpoint also completed one and three frame GPU smoke sequences.
