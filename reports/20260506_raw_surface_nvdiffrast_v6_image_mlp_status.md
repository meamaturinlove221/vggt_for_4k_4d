# Raw Surface nvdiffrast v6 Image-MLP Status

Status: `image_conditioned_residual_smoke_complete_visual_fail`

This report records the first image-conditioned residual decoder smoke on the
nvdiffrast full-mesh renderer path. It does not claim mentor success, does not
create a teacher, does not create a candidate, and does not allow cloud
execution.

## Current Gate Truth

```text
strict_candidate_passes = 0
strict_teacher_passes = 0
cloud = blocked
```

## What v6 Added

The raw-surface optimizer now supports:

```text
--offset-mode image_mlp
```

This appends per-vertex raw-image features to the surface decoder input:

```text
multi-view RGB mean
multi-view RGB variance
view support
```

These are sampled from the initial connected surface under the calibrated
cameras and human masks. This is the first image-conditioned residual decoder
smoke in the renderer-backend branch.

## Smoke Output

```text
output/normal_line_multiview_20260506/raw_surface_nvdiffrast_v6_image_mlp_t96_step80
```

Run:

```text
target_size = 96
views = 0,10,20,30,40,50
steps = 80
offset_mode = image_mlp
mlp_hidden = 96
photometric_variance_weight = 0.05
```

Metrics:

```text
loss = 0.374080 -> 0.365165
avg_initial_iou = 0.759479
avg_final_iou = 0.763399
avg_iou_delta = +0.003920
photometric_variance = 0.032740 -> 0.032749
vertices_with_two_view_support = 34,392
mean_view_support = 4.2675
max_vertex_delta = 0.004085
mean_vertex_delta = 0.000543
```

The image-conditioned decoder gets slightly better mask numbers than v5, but
the photometric variance does not decrease and the surface remains visually
template-like.

## Open3D Visual Review

Review outputs:

```text
output/normal_line_multiview_20260506/raw_surface_nvdiffrast_v6_image_mlp_t96_step80/open3d_review_full
output/normal_line_multiview_20260506/raw_surface_nvdiffrast_v6_image_mlp_t96_step80/open3d_review_head
output/normal_line_multiview_20260506/raw_surface_nvdiffrast_v6_image_mlp_t96_step80/open3d_review_face
output/normal_line_multiview_20260506/raw_surface_nvdiffrast_v6_image_mlp_t96_step80/open3d_review_hands
```

Visual decision:

```text
fail
```

Reasons:

```text
full body remains SMPL-X-template-like
face remains non-personalized
mouth/nose relief is not recovered
hair remains a coarse cap
hands remain weak/template-like
```

## Decision

```text
Do not export v6 as a teacher.
Do not cloud train from v6.
Do not continue by tuning image_mlp hidden size / LR / steps / scalar weights.
```

The evidence now says the renderer backend works, but the current carrier and
vertex residual decoders still do not have enough geometry capacity to create
mentor-level head/face/hair/hand detail from weak 2D signals.

## Required Unblocker

Continue only with one of:

```text
1. a strict-passing same-frame dense surface teacher
2. a richer learned local surface-token backend with visibility-aware image
   feature aggregation and rendered normal/depth regularization
```

Do not return to:

```text
semantic top-k/radius/weight tuning
offset/support/threshold loops
VGGT shell recycling
numeric-only pass
cloud execution
```
