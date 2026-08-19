# Node 4 fixed B_dev screen

The screen uses the fifth sequence in the public GOT-10k validation split,
`GOT-10k_Val_000005` (80 frames).  Results were generated with the immutable
released `FARTrackSparse_ep0015.pth.tar` checkpoint on the same GPU and read
by `scripts/evaluate_got10k_val.py`'s IoU function.

| Tracker | AO | FPS |
| --- | ---: | ---: |
| FARTrackSparse baseline | 0.955114093 | 25.371486 |
| Appearance-diverse reservoir | 0.957341090 | 21.921622 |

The screen is diagnostic only, not a valid selection score.  Although AO rose
by 0.002227, throughput fell 13.6%, exceeding the predeclared 5% overhead
constraint.  No full B_dev or B_test evaluation was run.
