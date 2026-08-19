# CTQA Novelty Boundary

CTQA is under empirical investigation, not an established contribution. The
following directly relevant work prevents a broad novelty claim.

| Prior work | What it already establishes | Consequence for CTQA |
| --- | --- | --- |
| Xie et al., AQATrack, CVPR 2024 | Learnable autoregressive queries and a spatio-temporal transformer for single-object tracking | CTQA must not claim to introduce temporal or autoregressive queries for SOT. |
| Geng et al., TGTrack, CVPR 2026 | Temporal generative supervision, autoregressive prediction, and explicit time embeddings for unified SOT | CTQA must not claim generic temporal-supervision novelty. |
| Zhou et al., Global Tracking Transformers, CVPR 2022 | Trajectory queries over a temporal feature buffer for multi-object tracking | CTQA must distinguish SOT use and the underlying interface-specific intervention. |

The only candidate contribution under evaluation is narrow: the released
FARTrackSparse implementation already receives three prior quantized boxes, but
overwrites that tensor with fixed command tokens before it reaches its query
decoder. CTQA activates that specific existing interface by a zero-initialized,
checkpoint-compatible adapter, then compares it with matched continued training
from the same released checkpoint.

This is an implementation- and evidence-dependent contribution, not a claim of
inventing temporal tracking. It can be retained in a paper only if the
pre-registered matched training and split-isolated B_dev results demonstrate a
reproducible benefit. Otherwise it remains a documented negative structural
probe.

## Primary Sources

- J. Xie et al., "Autoregressive Queries for Adaptive Tracking with
  Spatio-Temporal Transformers," CVPR 2024, pp. 19300--19309,
  https://openaccess.thecvf.com/content/CVPR2024/papers/Xie_Autoregressive_Queries_for_Adaptive_Tracking_with_Spatio-Temporal_Transformers_CVPR_2024_paper.pdf
- W. Geng et al., "TGTrack: Temporal Generative Learning for Unified Single
  Object Tracking," CVPR 2026, pp. 28134--28144,
  https://openaccess.thecvf.com/content/CVPR2026/html/Geng_TGTrack_Temporal_Generative_Learning_for_Unified_Single_Object_Tracking_CVPR_2026_paper.html
- X. Zhou et al., "Global Tracking Transformers," CVPR 2022, pp. 8771--8780,
  https://openaccess.thecvf.com/content/CVPR2022/html/Zhou_Global_Tracking_Transformers_CVPR_2022_paper.html
