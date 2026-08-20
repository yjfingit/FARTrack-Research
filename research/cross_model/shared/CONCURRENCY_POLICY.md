# Inference Concurrency Policy

Each full tracking result is mathematically independent across sequences, but
its process scheduling and output ownership must remain explicit.  We use
process-level sharding only when a model-specific micro-benchmark proves
trajectory parity and higher aggregate throughput.

## Required Invariants

- One sequence is claimed by exactly one process.
- Each process has a unique result root or `runid`; no result or timing file is
  shared between workers.
- `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `NUMEXPR_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` equal one in every
  worker.
- The tracker has no cross-sequence mutable state and loads its own checkpoint.
- Each completed shard must pass prediction/GT row-count validation before it
  is admitted to the combined evaluation.
- A concurrency choice is a scheduling decision, never a model-selection
  decision; AO always uses the same complete B_dev sequence manifest.

## OSTrack Evidence

On the single RTX 4090 D, two simultaneous OSTrack processes (different
sequences and `runid`s) completed in 8.888 seconds wall clock.  Their output
track SHA256 values exactly matched the corresponding single-process outputs:

| sequence | single and concurrent SHA256 |
| --- | --- |
| GOT-10k_Val_000001 | `12038ebcd76d55582df061a627065918830d50696787a7f04f0fb90603fcaf44` |
| GOT-10k_Val_000002 | `0e3557f04efd1194de3e9d1de1654a346ada8c33c81658be6e4eda2470f97873` |

Therefore later OSTrack runs may use at most two workers per GPU.  A third
worker is not authorized without a fresh parity and throughput benchmark.
