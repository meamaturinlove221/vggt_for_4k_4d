# Raw Surface nvdiffrast v5 MLP Status

Status: `learned_residual_smoke_complete_visual_fail`

This report records the first minimal learned residual decoder smoke on the
nvdiffrast full-mesh renderer path. It does not claim mentor success, does not
create a teacher, does not create a candidate, and does not allow cloud
execution.

## Current Gate Truth

```text
strict_candidate_passes = 0
strict_teacher_passes = 0
cloud = blocked
```

## What v5 Added

The raw-surface optimizer now supports:

```text
--offset-mode mlp
```

This replaces free per-vertex residuals with a tiny shared residual decoder over
surface features:

```text
centered xyz
vertex normal
body-part one-hot
bounded part-aware residual limits
```

The purpose was to test the first learned-surface-backend direction without
returning to semantic top-k / offset tuning.

## Smoke Output

```text
output/normal_line_multiview_20260506/raw_surface_nvdiffrast_v5_mlp_photometric_t96_step50
```

Run:

```text
target_size = 96
views = 0,10,20,30,40,50
steps = 50
offset_mode = mlp
mlp_hidden = 64
photometric_variance_weight = 0.05
```

Metrics:

```text
loss = 0.373907 -> 0.369405
avg_initial_iou = 0.759479
avg_final_iou = 0.761166
avg_iou_delta = +0.001687
photometric_variance = 0.032740 -> 0.032745
max_vertex_delta = 0.001576
mean_vertex_delta = 0.000329
```

The learned residual decoder trains and lowers the aggregate objective, but it
does not reduce the multi-view photometric variance and does not materially
change the surface geometry.

## Open3D Visual Review

Review outputs:

```text
output/normal_line_multiview_20260506/raw_surface_nvdiffrast_v5_mlp_photometric_t96_step50/open3d_review_full
output/normal_line_multiview_20260506/raw_surface_nvdiffrast_v5_mlp_photometric_t96_step50/open3d_review_head
output/normal_line_multiview_20260506/raw_surface_nvdiffrast_v5_mlp_photometric_t96_step50/open3d_review_face
output/normal_line_multiview_20260506/raw_surface_nvdiffrast_v5_mlp_photometric_t96_step50/open3d_review_hands
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
Do not export v5 as a teacher.
Do not cloud train from v5.
Do not tune the small MLP hidden size / LR / step count as a main route.
```

The next useful step is not another scalar tweak. A mentor-level result needs a
surface representation with actual local image-conditioned detail capacity:

```text
surface tokens / local displacement fields
visibility-aware feature aggregation
face/hair/hand specialized residual heads
rendered normal/depth consistency at the nvdiffrast level
```

The strict gate remains the final judge.
