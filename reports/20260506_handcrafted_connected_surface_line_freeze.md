# Handcrafted Connected Surface Line Freeze

Status: `frozen_negative`

This report freezes the handcrafted connected-surface carrier line after the
fixed `v28_semantic_detail_layer_falsification` smoke. It does not claim mentor
success, does not create a teacher, does not create a candidate, and does not
allow cloud execution.

## Current Gate Truth

```text
strict_candidate_passes = 0
strict_teacher_passes = 0
cloud = blocked
```

## What v28 Tested

The v28 test added a payload-driven connected semantic detail layer to the
existing carrier:

```text
output/normal_line_multiview_20260506/connected_surface_template_v28_semantic_detail_mouth_nose_fingers
```

The layer was restricted to audited failure groups:

```text
face.mouth
face.central_nose
left/right thumb, index, middle, ring, pinky
```

It added welded local topology only:

```text
semantic detail seed vertices = 43
semantic detail used vertices = 268
semantic detail new vertices = 268
semantic detail duplicated faces = 235
semantic detail stitch faces = 462
```

It did not use VGGT depth / point / normal as a teacher. It did not create a
floating patch. It did not create a strict teacher or candidate.

## Fixed Smoke

Run:

```text
output/normal_line_multiview_20260506/connected_surface_v28_semantic_detail_falsification_smoke2_t96_step10
```

Metrics:

```text
v24 outer+triangle iou_delta = +0.005383
v27 semantic-control iou_delta = +0.004345
v28 semantic-detail iou_delta = +0.003490

v24 target_recall_delta = +0.001859
v27 target_recall_delta = -0.000889
v28 target_recall_delta = -0.001308
```

The detail layer did not improve the fixed smoke relative to v24/v27.

## Open3D Review

Explicit visual review:

```text
output/normal_line_multiview_20260506/connected_surface_v28_semantic_detail_falsification_smoke2_t96_step10/open3d_review_full
output/normal_line_multiview_20260506/connected_surface_v28_semantic_detail_falsification_smoke2_t96_step10/open3d_review_head
output/normal_line_multiview_20260506/connected_surface_v28_semantic_detail_falsification_smoke2_t96_step10/open3d_review_face
output/normal_line_multiview_20260506/connected_surface_v28_semantic_detail_falsification_smoke2_t96_step10/open3d_review_hands
```

Review JSON:

```text
output/normal_line_multiview_20260506/connected_surface_v28_semantic_detail_falsification_smoke2_t96_step10/visual_review_codex_fail_or_pass.json
```

Visual decision:

```text
fail
```

Reasons:

```text
full body remains a SMPL-X-like template surface
face remains non-personalized and lacks modeled mouth/nose relief
head/hair remains a coarse cap-like carrier with artifacts
hands remain weak template/finger strands
Open3D does not look like mentor-pass normal human geometry
```

## Frozen Routes

The following handcrafted connected-carrier extensions are now frozen as a main
route:

```text
global/normal offsets
part-free offsets
outer hair/clothing layer
image-edge SDF
triangle RGB/gradient smoke
broad face/hand landmark Chamfer
unique-side hand landmark assignment
semantic landmark correspondence loss
semantic group offset basis
semantic top-k control offsets
payload-driven semantic detail layer
```

Do not continue by:

```text
increasing steps
increasing view count
increasing landmark weights
tuning semantic top-k / radius / offsets
tuning confidence/support/thresholds
exporting teacher targets from this visual-fail surface
claiming strict pass from numeric deltas
```

## Next Route

Switch to:

```text
codex/renderer-backend-preflight
```

Goal:

```text
Build or validate a real differentiable mesh rasterizer backend before further
surface optimization.
```

Minimum backend requirements:

```text
render full connected mesh, not sampled random faces
output mask / depth / normal / RGB / visibility
match existing NumPy z-buffer on a calibration smoke
support 96 or 128 resolution, 6 views, stable backward pass
support face / hair / hands / clothing local losses
avoid OOM without cloud training
```

Candidate backends:

```text
nvdiffrast
PyTorch3D
Kaolin
WSL2 / Linux conda / Docker / lab Linux if Windows wheels fail
```

This is backend environment work, not cloud training.
