# Node 4.1 constant-size ADR screen

The exact fixed screen is `GOT-10k_Val_000005` (80 public GOT-10k validation
frames) using the immutable `FARTrackSparse_ep0015.pth.tar` checkpoint and
identical environment variables (`CUDA_VISIBLE_DEVICES=0`, one CPU thread for
OpenMP/MKL/OpenBLAS).

| Tracker | AO | FPS | Total tracking time |
| --- | ---: | ---: | ---: |
| FARTrackSparse baseline | 0.955114093 | 26.279923 | 3.044149 s |
| Constant-size ADR | 0.955951846 | 21.926074 | 3.648624 s |

The raw AO delta is +0.000838.  Throughput is 16.57% lower, exceeding the
predeclared 5% budget.  A prior independent ADR repeat reported 27.211972 FPS
while another paired repeat reported 20.525757 FPS, so this short screen has
substantial timing variation; neither selective best-case timing nor this
small AO change justify a full B_dev run.  B_test was not run.
