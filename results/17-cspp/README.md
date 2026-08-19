# Node 17: Coordinate-Selective Posterior Projection (CSPP)

This directory contains the preregistration and honest result summary for a
training-free FARTrackSparse output-projection candidate. Raw predictions and
read-only metrics remain on the data disk, not in Git:

- `/root/autodl-tmp/experiment/.research-assets/output/node17_cspp/selection_baseline_frozen.json`
- `/root/autodl-tmp/experiment/.research-assets/output/node17_cspp/selection_cspp_valid.json`
- `/root/autodl-tmp/experiment/.research-assets/output/test/tracking_results/fartrack_sparse_cspp/fartrack_sparse_224_full_1705/`

The valid selection evaluation uses IDs modulo three equal to two. It failed
the pre-registered AO gate, so confirmation and B_test are prohibited.
