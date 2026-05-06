# Raw Surface nvdiffrast v3 Status

Status: `backend_integration_smoke_complete_visual_fail`

This report records the first local raw-surface optimization smoke using the
validated nvdiffrast full-mesh renderer. It does not claim mentor success, does
not create a teacher, does not create a candidate, and does not allow cloud
execution.

## Current Gate Truth

```text
strict_candidate_passes = 0
strict_teacher_passes = 0
cloud = blocked
```

## What Changed

Implemented:

```text
tools/optimize_raw_surface_nvdiffrast.py
```

This script avoids the old CPU/surfel optimizer path and uses the validated
nvdiffrast renderer directly:

```text
full connected mesh = 39,962 vertices / 80,569 faces
views = 0,10,20,30,40,50
target_size = 96
steps = 10
renderer = nvdiffrast full mesh
loss = raw mask BCE + target recall + overfill + offset/edge regularization
```

## Smoke Output

```text
output/normal_line_multiview_20260506/raw_surface_nvdiffrast_v3_smoke_t96_step10
```

Key metrics:

```text
avg_initial_iou = 0.759479
avg_final_iou = 0.761424
avg_iou_delta = +0.001945
max_vertex_delta = 0.002101
mean_vertex_delta = 0.000010
```

This proves the nvdiffrast optimization loop is connected, fast, and stable.
It does not prove a useful human surface.

## Open3D Visual Review

Review outputs:

```text
output/normal_line_multiview_20260506/raw_surface_nvdiffrast_v3_smoke_t96_step10/open3d_review_full
output/normal_line_multiview_20260506/raw_surface_nvdiffrast_v3_smoke_t96_step10/open3d_review_head
output/normal_line_multiview_20260506/raw_surface_nvdiffrast_v3_smoke_t96_step10/open3d_review_face
output/normal_line_multiview_20260506/raw_surface_nvdiffrast_v3_smoke_t96_step10/open3d_review_hands
```

Visual decision:

```text
fail
```

Reasons:

```text
full body remains an SMPL-X-like template surface
face is still non-personalized
mouth/nose relief is not modeled from the real person
hair/head remains a coarse cap
hands are connected but remain template-like and weak
Open3D does not satisfy mentor-pass normal-human detail
```

## Decision

```text
Do not export v3 as a teacher.
Do not run strict teacher gate from v3.
Do not cloud train from v3.
```

The backend integration is useful, but mask-only optimization is not enough.

## Next Work

Continue within the renderer-backend route, but upgrade the objective:

```text
multi-view photometric color consistency
view visibility handling
face / hair / hand part-local photometric weighting
surface-level normal/depth regularization from rendered outputs
Open3D visual precheck before any teacher export
```

Do not return to:

```text
semantic top-k / radius / weight tuning
offset/support/threshold loops
VGGT shell recycling
numeric-only pass
cloud execution
```
