# Node 18: Causal Trajectory Query Adapter (CTQA)

## Hypothesis

FARTrackSparse accepts the last three predicted boxes as twelve discrete
coordinate tokens, but its sparse backbone replaces them with four fixed
command tokens before decoding.  A small causal adapter can recover that
unused temporal information by converting the historical coordinate embeddings
into an additive bias on the four coordinate queries.

## Compatibility invariant

The adapter is behind `MODEL.TRAJECTORY_QUERY_ADAPTER.ENABLED` (default
`False`).  Its final projection is zero initialized, so a released checkpoint
loaded with `strict=False` has exact output parity when CTQA is enabled but
untrained.  The adapter reads the shared word embedding through functional
lookup to avoid the legacy embedding's `max_norm` in-place mutation.

## Pre-training checks

```bash
cd /tmp/arbor-worktrees-0/coordinator__n18-mechanism-causal-trajectory-quer-438901eb
/root/autodl-tmp/experiment/.venvs/fartrack-research/bin/python -m pytest -q tests/test_ctqa.py
CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/experiment/.venvs/fartrack-research/bin/python scripts/verify_ctqa_checkpoint.py
```

Both checks passed before this report was written.  The second check loads
`FARTrackSparse_ep0015.pth.tar`, proves exact feature equality to the disabled
baseline at initialization, then proves a synthetic optimizer step produces a
nonzero change.

## Training isolation

`experiments/fartrack_sparse/fartrack_sparse_224_ctqa_got10k_train.yaml` names
only `GOT10K_train_full`; validation is explicitly empty.  The dedicated
environment is `research/ctqa_got10k_train_environment.py` and contains only
the expected training root:
`/root/autodl-tmp/experiment/.research-assets/data/got10k/train`.

No real-dataset training has started.  It is blocked until the coordinator
confirms verified extraction and checksum evidence for that training root.
The public GOT-10k validation set and LaSOT Testing remain evaluation-only.

## Novelty boundary

Temporal transformer tracking and motion-conditioned query methods are prior
art.  CTQA makes no broad claim of inventing temporal tracking.  The narrow,
testable contribution is a checkpoint-compatible way to activate a specific
otherwise-discarded trajectory interface in frozen FARTrackSparse.  Any
efficacy or novelty claim is deferred until a split-isolated trained experiment
and literature comparison are complete.

## Pre-registered matched training screen

The first learning screen uses one epoch of 2,048 sampled examples, batch size
four, hence exactly 512 optimizer steps per seed.  It compares three fixed
seeds (`1008`, `2026`, `3407`) for each arm, for 3,072 total updates:

| Arm | Config | CTQA flag | Checkpoint | Training data |
| --- | --- | --- | --- | --- |
| Continued-training control | `fartrack_sparse_224_continue_got10k_screen` | false | `FARTrackSparse_ep0015.pth.tar` | `GOT10K_train_full` only |
| CTQA | `fartrack_sparse_224_ctqa_got10k_screen` | true | `FARTrackSparse_ep0015.pth.tar` | `GOT10K_train_full` only |

All optimizer, sampling, augmentation, batch, worker, and seed settings are
identical.  The paired-config test prevents drift outside the feature flag.
The launcher requires explicit `CTQA_DATA_VERIFIED=1`, installs a temporary
train-only local environment, restores it on exit, and limits the parent and
four DataLoader processes to one internal CPU thread each.

```bash
cd /tmp/arbor-worktrees-0/coordinator__n18-mechanism-causal-trajectory-quer-438901eb
for seed in 1008 2026 3407; do
  CTQA_DATA_VERIFIED=1 bash scripts/launch_ctqa_screen.sh control "$seed"
  CTQA_DATA_VERIFIED=1 bash scripts/launch_ctqa_screen.sh ctqa "$seed"
done
```

Expected artifacts for arm `A` and seed `S` are:

```text
/root/autodl-tmp/experiment/.research-assets/runs/ctqa_screen/A/seed_S/stdout.log
/root/autodl-tmp/experiment/.research-assets/runs/ctqa_screen/A/seed_S/logs/fartrack_sparse-<config>.log
/root/autodl-tmp/experiment/.research-assets/runs/ctqa_screen/A/seed_S/checkpoints/train/fartrack_sparse/<config>/FARTrackSparse_ep0001.pth.tar
```
