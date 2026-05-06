# Renderer Backend Branch Status

Status: `renderer_backend_unblocked_surface_not_solved`

This branch intentionally moved away from the frozen handcrafted connected
surface carrier route. It does not claim mentor success, does not create a
teacher, does not create a candidate, and does not allow cloud execution.

## Current Gate Truth

```text
strict_candidate_passes = 0
strict_teacher_passes = 0
cloud = blocked
```

## Branch

```text
codex/renderer-backend-preflight
```

Pushed commits:

```text
3322073 Add renderer backend preflight
e40ae2e Add nvdiffrast raw surface smoke
7953fd4 Add learned residual renderer smoke
38f5ae7 Add image conditioned surface residual smoke
```

## Positive Result

The local Windows/GPU environment now has a working nvdiffrast backend:

```text
torch = 2.9.1+cu130
GPU = NVIDIA GeForce RTX 5080
nvdiffrast = 0.4.0
```

The backend preflight passed:

```text
output/normal_line_multiview_20260506/renderer_backend_preflight_nvdiffrast_v1
```

Key result:

```text
full mesh = 39,962 vertices / 80,569 faces
views = 0,10,20,30,40,50
min_mask_iou_vs_hard_zbuffer = 0.8864
max_median_abs_depth_residual = 0.00890
max_p90_abs_depth_residual = 0.03476
backward_ran = true
grad_finite = true
grad_nonzero_vertices = 9,347
```

This removes the old CPU/sampled-renderer blocker. It is not a geometry pass.

## Negative Surface Results

### v3 mask-only nvdiffrast smoke

```text
output/normal_line_multiview_20260506/raw_surface_nvdiffrast_v3_smoke_t96_step10
avg_iou_delta = +0.001945
visual = fail
```

### v4 multi-view photometric variance smoke

```text
output/normal_line_multiview_20260506/raw_surface_nvdiffrast_v4_photometric_t96_step20
avg_iou_delta = +0.003446
photometric_variance = 0.032740 -> 0.032624
visual = fail
```

### v5 geometry MLP residual smoke

```text
output/normal_line_multiview_20260506/raw_surface_nvdiffrast_v5_mlp_photometric_t96_step50
avg_iou_delta = +0.001687
photometric_variance = 0.032740 -> 0.032745
visual = fail
```

### v6 image-conditioned MLP residual smoke

```text
output/normal_line_multiview_20260506/raw_surface_nvdiffrast_v6_image_mlp_t96_step80
avg_iou_delta = +0.003920
photometric_variance = 0.032740 -> 0.032749
visual = fail
```

All Open3D reviews show the same core failure:

```text
full body remains SMPL-X-template-like
face remains non-personalized
mouth/nose relief is not recovered
hair remains a coarse cap
hands remain weak/template-like
```

## Interpretation

The renderer/backend problem is solved locally. The surface problem is not.

The evidence now blocks these local continuation loops:

```text
mask-only optimization
plain photometric variance on current carrier
per-vertex bounded offset tuning
small geometry MLP tuning
small image-conditioned MLP tuning
semantic top-k/radius/weight tuning
```

These routes move small numeric values but do not create mentor-level
head/face/hair/hand/fullbody geometry.

## Required Next Unblocker

Proceed only with one of:

```text
1. strict-passing same-frame dense target surface teacher
2. real learned local surface-token backend with visibility-aware image feature
   aggregation, part-specialized residual heads, and rendered normal/depth
   regularization
```

Still prohibited:

```text
cloud execution
teacher export from visual-fail meshes
candidate claim from numeric deltas
VGGT depth/point/normal shell recycling
handcrafted carrier scalar tuning
```
