# Raw-Image Surface Upper-Bound Status

Date: 2026-05-05

Branch:

```text
codex/raw-image-surface-upperbound
```

## Current Truth

This stage intentionally stops recycling VGGT depth / point / normal shell
observations. It starts the long-route method rebuild requested by the mentor:

```text
raw RGB / mask / calibrated camera / SMPL-X scaffold
  -> differentiable human surface optimization
  -> learned visibility-aware human surface backend
  -> existing strict candidate gate
```

No mentor pass has been achieved. No cloud upload/run is allowed.

Current cloud guard remains:

```text
reports/20260504_strict_gate_registry.json
strict_candidate_passes = 0
strict_teacher_passes = 0
cloud_allowed = false
```

## New Local Tools

Two new local-only tools were added:

```text
tools/raw_image_surface_upperbound_preflight.py
tools/optimize_raw_smplx_silhouette_torch.py
```

They are not candidate generators. They do not write VGGT-format
`predictions.npz`, do not bypass strict gate, and do not unblock cloud.

## Raw Asset Preflight

Run:

```text
output/normal_line_multiview_20260505/raw_image_surface_upperbound_preflight_60v_humancrop
```

Inputs:

```text
scene: output/4k4d_preprocessed_scene_variants/0012_11_frame0000_60views_human_crop
views: 60 / 60
camera source: rgb_cams_smc
SMPL-X model: G:/.../datasets/smplx/SMPLX_NEUTRAL.npz
```

Result:

```text
truthful_status = assets_ok_but_blocked_missing_differentiable_renderer
asset_preflight_pass = true
true_stage_a_ready = false
mean full IoU = 0.7783
mean full target recall = 0.8932
head IoU mean = 0.9315
face IoU mean = 0.9292
```

Interpretation:

- Raw RGB/mask/camera/SMPL-X assets are present and coherent enough to start a
  raw-image upper-bound.
- The blocker is no longer "maybe data is missing".
- The blocker is the missing differentiable surface renderer/backend.

Dependency state:

```text
D:/anaconda/python.exe:
  torch = 2.9.0+cu126
  CUDA reported available, but runtime kernel is not usable on this GPU
  smplx/cv2/h5py/scipy available
  open3d/trimesh/pytorch3d/nvdiffrast/kaolin missing

D:/anaconda/envs/g3splat/python.exe:
  open3d available
  smplx/pytorch3d/nvdiffrast/kaolin missing
```

## Differentiable Silhouette Smoke

Run 1:

```text
output/normal_line_multiview_20260505/raw_smplx_silhouette_torch_smoke12_global
```

This optimizes only global scale/translation using raw masks and projected
SMPL-X vertices. It runs on CPU because the local CUDA runtime reports:

```text
CUDA error: no kernel image is available for execution on the device
```

Result:

```text
truthful_status = raw_silhouette_differentiable_smoke_complete_not_surface_backend
initial mean IoU = 0.7712
optimized mean IoU = 0.8076
IoU delta = +0.0365
initial target recall = 0.8955
optimized target recall = 0.8705
target recall delta = -0.0250
```

Interpretation:

- The raw-mask differentiable loop is alive.
- Global-only fitting tightens the contour, improving IoU but losing recall.
- This is not sufficient for clothing, hair, hands, or face detail.

Run 2:

```text
output/normal_line_multiview_20260505/raw_smplx_silhouette_torch_smoke12_normaloffset
```

This adds bounded normal-direction residual offsets to the SMPL-X vertices.

Result:

```text
truthful_status = raw_silhouette_differentiable_smoke_complete_not_surface_backend
initial mean IoU = 0.7712
optimized mean IoU = 0.8280
IoU delta = +0.0569
initial target recall = 0.8955
optimized target recall = 0.9030
target recall delta = +0.0075
```

Interpretation:

- SMPL-X residual displacement has a real raw-image silhouette signal.
- This is the first positive non-wall evidence for the long-route surface
  backend.
- It is still only a 2D silhouette smoke. It is not a 3D modeled face/hair/hand
  surface, not a teacher, and not a candidate.

## Why This Is Different From The Frozen Routes

This route does not:

- use VGGT 60v depth/point/normal observations as teacher;
- run TSDF / Poisson / visual hull over VGGT shells;
- tune p40 / fixed thresholds / confidence;
- replace the camera head with HART-style PnP;
- make a numeric-only pass claim.

It uses:

- raw images;
- raw masks / soft silhouette;
- calibrated cameras;
- SMPL-X only as scaffold / topology / correspondence;
- differentiable torch optimization of silhouette signals.

## Current Blocker

The current implementation still lacks the part required for mentor-level
geometry:

```text
true differentiable triangle/soft rasterization
photometric masked RGB consistency
body-part-aware displacement and normal residuals
visibility-aware multi-view aggregation
surface-to-view depth/world_points/normal rasterization
Open3D full/head/face/hands strict gate
```

The local environment does not currently provide `pytorch3d`, `nvdiffrast`, or
`kaolin`, and the current torch CUDA build cannot execute kernels on this GPU.
CPU smoke is enough for proof-of-loop, not for full 60v surface optimization.

## Next Non-Wall Action

Proceed with a local learned/differentiable surface backend v0, not another r
candidate:

1. Add a differentiable soft triangle/silhouette renderer or install/provide
   `nvdiffrast` / `pytorch3d` / equivalent.
2. Extend the current silhouette smoke from projected vertices to triangle
   surface coverage.
3. Add part-aware residual displacement:
   - torso/limbs: stronger scaffold regularization;
   - hands: compactness + attached support;
   - face: weak SMPL-X, image/edge/detail stronger;
   - hair/clothing: mask boundary and image residual, not SMPL-X teacher.
4. Add masked RGB/edge photometric consistency.
5. Rasterize the optimized surface back to depth/world_points/normal/confidence.
6. Only then run the existing full strict candidate gate.

Until that full gate passes, cloud remains blocked.
