# Renderer Backend Preflight

Status: `backend_preflight_pass`

This report records the first non-handcrafted-carrier progress after freezing
`v28_semantic_detail_layer_falsification`. It validates the renderer backend
only. It does not claim mentor success, does not create a teacher, does not
create a candidate, and does not allow cloud execution.

## Current Gate Truth

```text
strict_candidate_passes = 0
strict_teacher_passes = 0
cloud = blocked
```

## Why This Is a New Route

The handcrafted connected-surface carrier line is frozen in:

```text
reports/20260506_handcrafted_connected_surface_line_freeze.md
```

The next blocker was not another semantic offset, threshold, or view-count
loop. The next blocker was whether the local machine can run a real
differentiable full-mesh renderer instead of the earlier CPU toy renderer or
sampled surfel smoke.

## Backend Install

`nvdiffrast` was installed locally into the `g3splat` environment from source:

```text
D:\anaconda\envs\g3splat\python.exe
torch = 2.9.1+cu130
cuda = 13.0
gpu = NVIDIA GeForce RTX 5080
nvdiffrast = 0.4.0
```

Important Windows build detail:

```text
pip install nvdiffrast
```

had no wheel. Source build succeeded only after:

```text
vcvars64.bat
CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
DISTUTILS_USE_SDK=1
PYTHONUTF8=1
PYTHONIOENCODING=utf-8
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git
```

## Preflight Script

Implemented:

```text
tools/preflight_differentiable_renderer_backend.py
```

The script:

```text
loads the full connected mesh payload
renders all 80,569 faces with nvdiffrast, not sampled faces
outputs mask / depth / normal / RGB / visibility
compares against the existing NumPy z-buffer rasterizer
checks a stable backward pass to mesh vertices
avoids importing the old soft-surface optimizer because that import path
mutates CUDA device discovery on this Windows setup
```

## Fixed Preflight Run

Output:

```text
output/normal_line_multiview_20260506/renderer_backend_preflight_nvdiffrast_v1
```

Command:

```text
D:\anaconda\envs\g3splat\python.exe tools/preflight_differentiable_renderer_backend.py ^
  --scene-dir output/4k4d_preprocessed_scene_variants/0012_11_frame0000_60views_human_crop ^
  --template-payload output/normal_line_multiview_20260506/connected_surface_template_v28_semantic_detail_mouth_nose_fingers/connected_human_surface_template_payload.npz ^
  --output-dir output/normal_line_multiview_20260506/renderer_backend_preflight_nvdiffrast_v1 ^
  --target-size 96 ^
  --view-indices 0,10,20,30,40,50 ^
  --backend nvdiffrast ^
  --overwrite
```

Result:

```text
status = backend_preflight_pass
views_tested = 6
full_mesh_vertices = 39,962
full_mesh_faces = 80,569
elapsed_seconds = 16.85
min_mask_iou_vs_hard_zbuffer = 0.8864
max_median_abs_depth_residual = 0.00890
max_p90_abs_depth_residual = 0.03476
backward_ran = true
grad_finite = true
grad_nonzero_vertices = 9,347
```

The preflight threshold is intentionally limited to backend alignment:

```text
min_mask_iou_vs_hard_zbuffer >= 0.88
max_median_abs_depth_residual <= 0.02
max_p90_abs_depth_residual <= 0.05
min_views = 6
```

The mask IoU is not expected to be exactly 1.0 because nvdiffrast and the
existing NumPy rasterizer use different triangle edge and boundary-fill rules.
The depth residual confirms the camera and z ordering are aligned.

## Outputs

The preflight writes per-view:

```text
*_mask.png
*_hard_mask.png
*_depth.png
*_hard_depth.png
*_normal.png
*_rgb.png
*_visibility.png
renderer_backend_preflight_summary.json
renderer_backend_preflight_summary.md
```

## Decision

```text
nvdiffrast is now a usable local differentiable full-mesh renderer backend for
the next raw-image surface work.
```

This does not unblock cloud or mentor pass. It only replaces the previous
CPU/sampled-surface renderer blocker with a real local backend.

## Next Local Work

Do not return to:

```text
semantic top-k / radius / weight tuning
offset / support / threshold loops
VGGT depth/point/normal shell recycling
teacher export from visual-fail carriers
cloud execution
```

Next implementation target:

```text
integrate nvdiffrast into a raw-image surface optimization v3 path with
full-mesh mask/depth/normal/RGB/visibility rendering, part-local losses, and
strict teacher-gate export only after Open3D visual precheck improves.
```
