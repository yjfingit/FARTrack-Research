# Node 6: Zero-overhead template sampling sweep

Independent tracker entries select only FARTrackSparse's preexisting
`exponential`, `linear`, or `logarithmic` template sampling method. They retain
the released checkpoint, exact baseline one-forward `track` path, every-frame
candidate writes, token masks, result format, and read-only GOT-10k AO
evaluator. The fixed public development screen is sequences 1, 11, 44, and
100 (zero-based harness indices 0, 10, 43, 99).

## Fixed screen result

| schedule | frame-weighted AO | aggregate FPS | AO delta vs exponential |
| --- | ---: | ---: | ---: |
| exponential (baseline-equivalent) | 0.681366 | 20.92 | 0.000000 |
| linear | 0.683710 | 19.66 | +0.002345 |
| logarithmic | 0.685696 | 19.92 | +0.004331 |

Each schedule used the same 351 public validation frames. Exponential
predictions were exactly identical to the pre-existing immutable baseline on
all four sequences, validating the independent switch path. The measured FPS
values came from sequential, single-process, thread-capped jobs but shared the
GPU with an unrelated running LaSOT job; they are useful for confirming no
extra-forward overhead, not a clean standalone throughput comparison.

The only recommended full-B_dev candidate is `logarithmic`: it has the highest
small-screen AO, though it gains substantially on `000044` while degrading
`000011`. No final performance claim is justified before the pre-registered
180-sequence GOT-10k validation evaluation.
