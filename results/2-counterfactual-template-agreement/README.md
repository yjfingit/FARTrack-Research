# Node 2 smoke evidence

This directory contains only implementation validation. It contains no
benchmark score because the pre-registered LaSOT development data are absent.

## Commands run

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/root/autodl-tmp/experiment/.venvs/fartrack-research/bin/python \
  -c '<CounterfactualAgreementVerifier mask and zero-disagreement assertions>'
```

Result: `PASS`.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/root/autodl-tmp/experiment/.venvs/fartrack-research/bin/python \
  -c '<synthetic 256x320 RGB tracker initialization and three track calls>'
```

Result: `PASS`; the verifier executed two counterfactual checks, with final
normalized L1 disagreement `0.04007465` under the fixed development default.

A separate forced-rejection smoke (`cf_max_disagreement=0.0`) produced one
checked frame, one rejection, and no additional stored template. This confirms
the disagreement changes the write decision rather than the primary predicted
box.

The template pool is indexed exclusively by accepted writes. A regression
check exercised twenty rejected frames followed by a later acceptance and
verified a bounded five-slot template list and FARTrack-compatible
`1 x 445 x 445` mask throughout.
