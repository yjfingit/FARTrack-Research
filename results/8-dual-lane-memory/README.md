# Entropy-Conditioned Dual-Lane Memory

`preselection.json` freezes the entropy threshold, slot schedule, split, and
selection gate before the selection partition is executed. It is derived only
from passive node-7 entropy logs on calibration IDs whose numeric suffix is
congruent to one modulo three.

The separate partition runner and evaluator only read the official GOT-10k
validation data and standard tracker result files.
