# Development Split Definition

B_dev is the 180-sequence public GOT-10k validation directory:

/root/autodl-tmp/experiment/.research-assets/data/got10k/val

For action experiments only, sort sequence identifiers lexicographically and
assign the numeric suffix modulo three:

- calibration: suffix % 3 == 1;
- selection: suffix % 3 == 2;
- confirmation: suffix % 3 == 0.

The passive probe and immutable baseline may use all 180 B_dev sequences.
Thresholds and action configuration are frozen after calibration. Selection is
executed once. Confirmation is permitted only after selection succeeds. This
split must not be changed per model.
