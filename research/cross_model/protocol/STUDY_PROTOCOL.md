# Study Protocol

For a frozen tracker with state Z_t, a passive score may estimate an event
such as P(IoU(a_0(Z_t), B*_t) < 0.2 | Z_t). A deployed intervention instead
needs a positive conditional action advantage:

A_g(Z_t) = E[IoU(a_g(Z_t), B*_t) - IoU(a_0(Z_t), B*_t) | Z_t].

Each eligible model follows the same ordered procedure:

1. reproduce an immutable official-checkpoint GOT-10k validation baseline;
2. attach an output-inert passive risk probe and prove byte parity when it is
   disabled;
3. run one pre-registered low-cost, model-semantic conservative action;
4. run one pre-registered information-control action that explicitly discloses
   extra forwards and whether it adds an observation;
5. stop according to the fixed rules.

No step uses training data or B_test. Complete B_dev scores, calibration,
selection, and focused screens are separate evidence tiers and are never
pooled.
