# Node 11 frozen selection policy

Written before reading or running ID modulo 3 equals 2.

The calibration policy (`preselection_policy.md`, SHA-256 `1f68465a18f3187069e96e18b64663044e4ce931ec1bc77f8485af06a0960b09`) selected multiplier `1.10`: it achieved the highest calibration AO (`0.8388122348518865`) among the three predeclared candidates, and its aggregate FPS (`30.858572861749042`) did not exceed the allowed 5% loss relative to the recorded calibration baseline (`29.39145297617692`).

The fixed selection run uses only:

- public GOT-10k validation sequence numeric ID modulo 3 equals 2;
- branch-disagreement q85 threshold `0.0037756040692329407`;
- expansion factor `1.10`;
- identical 224-pixel search output and exactly one forward pass per frame.

Pass condition is both frame AO delta at least `+0.003` and aggregate FPS loss at most 5% against the frozen baseline on this same held selection partition. Failure prohibits confirmation and B_test.
