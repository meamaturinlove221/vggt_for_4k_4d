# Raw Surface nvdiffrast v4 Photometric Status

Status: `photometric_smoke_complete_visual_fail`

This report records the first multi-view raw-image photometric smoke on the
new nvdiffrast full-mesh renderer path. It does not claim mentor success, does
not create a teacher, does not create a candidate, and does not allow cloud
execution.

## Current Gate Truth

```text
strict_candidate_passes = 0
strict_teacher_passes = 0
cloud = blocked
```

## What v4 Added

v4 extends:

```text
tools/optimize_raw_surface_nvdiffrast.py
```

with:

```text
multi-view photometric color variance loss
raw RGB sampling through differentiable projection / grid_sample
mask-aware view support
full connected mesh nvdiffrast rendering
```

This is the first raw-image signal beyond silhouette on the new renderer
backend. It still uses the same connected carrier and does not use VGGT
depth/point/normal as a teacher.

## Smoke Output

```text
output/normal_line_multiview_20260506/raw_surface_nvdiffrast_v4_photometric_t96_step20
```

Run:

```text
target_size = 96
views = 0,10,20,30,40,50
steps = 20
photometric_variance_weight = 0.05
```

Metrics:

```text
avg_initial_iou = 0.759479
avg_final_iou = 0.762925
avg_iou_delta = +0.003446
photometric_variance = 0.032740 -> 0.032624
photometric_valid_vertices = 34,392 -> 34,396
photometric_mean_support = 4.5656 -> 4.5659
max_vertex_delta = 0.003981
mean_vertex_delta = 0.0000668
```

The photometric term is live and differentiable, but it produces only a small
local numerical change at this scale.

## Open3D Visual Review

Review outputs:

```text
output/normal_line_multiview_20260506/raw_surface_nvdiffrast_v4_photometric_t96_step20/open3d_review_full
output/normal_line_multiview_20260506/raw_surface_nvdiffrast_v4_photometric_t96_step20/open3d_review_head
output/normal_line_multiview_20260506/raw_surface_nvdiffrast_v4_photometric_t96_step20/open3d_review_face
output/normal_line_multiview_20260506/raw_surface_nvdiffrast_v4_photometric_t96_step20/open3d_review_hands
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
Do not export v4 as a teacher.
Do not run strict teacher gate from v4.
Do not cloud train from v4.
```

The renderer backend is real and usable, but the current connected carrier plus
mask/photometric vertex-offset objective is still not a mentor-level surface.

## Next Local Direction

The next step should not tune v4 weights. It should add a representation that
can actually express non-template surface detail:

```text
image-conditioned residual surface tokens or local displacement fields
part-local face/hair/hand refinement with visibility-aware photometric support
rendered surface normal/depth regularization at the nvdiffrast level
Open3D visual precheck before teacher export
```

Still prohibited:

```text
offset/support/threshold loops
semantic carrier top-k/radius/weight tuning
VGGT shell recycling
numeric-only pass
cloud execution
```
