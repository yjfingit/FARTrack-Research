# Node 11: delayed branch-disagreement search expansion

This isolated experiment keeps FARTrackSparse's frozen checkpoint, output search resolution (224), network forward count (one/frame), current-frame localization, state update, and template update unchanged.  After frame `t` has completed, a branch-disagreement alarm can expand only frame `t+1`'s crop; the pending action is consumed on that crop and immediately resets.

The q85 disagreement threshold comes from Node 7's calibration-only passive logs. `preselection_policy.md` was hashed before the three calibration factors were evaluated. `selection_policy.md` was then hashed before reading/running the selection partition.

Raw tracks and timing files are stored on the data disk under `/root/autodl-tmp/experiment/.research-assets/output/test/tracking_results/fartrack_sparse_delayed_search/`; the JSON summaries in this directory are derived read-only from those files and the unchanged public annotations.

The held selection gate failed (`-0.003339` frame AO, required `+0.003`), so the confirmation partition and B_test were not run.
