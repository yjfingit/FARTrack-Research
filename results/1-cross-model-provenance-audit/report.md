# Node 1: Cross-Model Frozen-Inference Qualification Audit

## Scope

This is a read-only audit of the user-provided source cache and data-disk
checkpoint directory. It ran no tracker, installed no package, used no
training data, and did not access B_test.

## Result

| Model | Official cached remote | Cached HEAD | License | Released weight visible | Eligibility |
| --- | --- | --- | --- | --- | --- |
| SiamRPN++ / PySOT | STVIR/pysot | d04028f8 | Apache-2.0 | no | fetch official checkpoint |
| OSTrack | botaoye/OSTrack | 33b5e125 | MIT | no | fetch official checkpoint |
| MixFormerV2 online config | mcg-nju/MixFormerV2 | 10414383 | MIT | no | fetch official checkpoint |
| ODTrack | GXNU-ZhongLab/ODTrack | 88c0a8e4 | MIT | no | fetch official checkpoint |

The local checkpoint directory contains the retained FARTrackSparse epoch-15
weight only. The cached source repositories are sufficient for source
provenance, but not for a frozen-checkpoint baseline. Each README/MODEL_ZOO
contains an official model-download location and a normal testing entry point.

## Decision

All four external models are provisionally eligible to fetch their published
weights into the data-disk asset area. No model is yet eligible for a B_dev
baseline. The next execution must obtain one official weight, record its URL
and SHA-256, and perform a checkpoint-load smoke before any full evaluation.
