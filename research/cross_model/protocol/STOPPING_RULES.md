# Stopping Rules

Before seeing selection metrics, every candidate must declare its risk signal,
threshold rule, action, expected information source, and speed measurement.

A candidate can enter confirmation only if its frozen selection result has all
of the following:

1. delta_AO >= +0.003;
2. the paired 95% bootstrap interval is not clearly adverse;
3. aggregate FPS loss is at most 10% versus its immutable baseline.

Any calibration or selection failure stops that candidate. It may not receive a
new threshold, replacement action, repeated selection run, or B_test run.
Only a confirmed B_dev winner may receive one final B_test run.
