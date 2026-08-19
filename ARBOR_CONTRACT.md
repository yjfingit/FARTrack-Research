# Arbor Research Contract: Transactional AR Memory for FARTrack

- **Target**: isolated branch `arbor/rgb-tracking-research-20260819`; the original `main` worktree and its existing changes are protected.
- **Task**: develop a causal, training-free controller for frozen FARTrackSparse template memory.
- **Metric**: maximize public GOT-10k validation AO during development. Full LaSOT and GOT-10k test are held out until configuration freeze.
- **Baseline**: to be measured using FARTrackSparse ep0015 with identical checkpoint and protocol on GOT-10k validation.
- **Permitted edits**: new tracker/controller, parameter/config files, experiment scripts, results, visualizations, and paper source in this worktree only.
- **Protected paths**: datasets, checkpoints, and official evaluation code/protocol. No score-affecting benchmark modifications.
- **Budget**: real GPU experiments; use GOT-10k validation for all candidate selection. No model training.
- **Resources**: all downloads, environments, caches, datasets, outputs, and paper artifacts live under `/root/autodl-tmp/experiment` data disk.
