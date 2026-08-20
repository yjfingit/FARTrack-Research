# Node 1.1: OSTrack Official-Repository Compatibility Smoke

## Scope

This is a checkpoint-load and synthetic CUDA-forward smoke only. It used no
dataset frame, training data, optimizer, B_dev metric, or B_test data.

## Weight Provenance

- Source: the ModelScope OSTrack release linked by the official OSTrack README.
- Model identifier: damo/cv_vitb_video-single-object-tracking_ostrack.
- Local data-disk path:
  /root/autodl-tmp/experiment/.research-assets/checkpoints/cross_model/ostrack_modelscope/pytorch_model.bin.
- SHA-256:
  8e6de3c6f10cbc21eaf1c34532358e23618f78a1a2bb9f0d9a2e55adc7af4894.
- Cached official source: botaoye/OSTrack commit
  33b5e12586216b7fd0e95d255bd01ba44cbec759.

## Compatibility Result

The weight contains a net state dict. The 256 configuration rejected it only
on positional embeddings (weight 144 template and 576 search tokens versus
the config's 64 and 256), identifying the model as 192/384 rather than an
incompatible architecture. With the upstream 384 configuration,
vitb_384_mae_ce_32x4_ep300.yaml, load_state_dict(..., strict=True) passed.

On an RTX 4090 D, one synthetic CUDA forward emitted:

- score_map: [1, 1, 24, 24];
- size_map: [1, 2, 24, 24];
- offset_map: [1, 2, 24, 24];
- pred_boxes: [1, 1, 4].

## Decision

This checkpoint is eligible for an OSTrack immutable GOT-10k validation
baseline under the explicitly recorded 192/384 configuration. No AO, FPS, or
candidate claim is made by this smoke.
