# Raw-Image Soft Surfel Surface v1 Status

Date: 2026-05-05

Branch:

```text
codex/raw-image-surface-upperbound
```

## Current Truth

No mentor pass has been achieved. No cloud upload/run is allowed.

```text
strict_candidate_passes = 0
strict_teacher_passes = 0
cloud_allowed = false
```

This stage remains local-only and does not create a VGGT candidate.

## What Changed

Added:

```text
tools/optimize_raw_smplx_softsurfel_torch.py
```

This is the first raw-image v1 surface smoke. It intentionally does not use:

```text
VGGT depth
VGGT point maps
VGGT normals
VGGT confidence
r-candidate outputs
```

It uses:

```text
raw RGB crop PNGs
raw masks
calibrated 4K4D cameras
SMPL-X scaffold
pure Torch CPU soft surfel rendering
multi-view RGB consistency
part-aware normal-offset limits
soft target-recall guard
hard raster export to depth/world/normal/mask NPZ
```

## Important Local Fix

The 60-view human-crop manifest stores `crop_bbox_xyxy` in the native exported
518-frame. Passing `target_size < 518` directly into the shared intrinsics
alignment helper projects SMPL-X into negative image coordinates.

The new script therefore aligns intrinsics at the native exported crop size and
then scales the already-cropped square view to the CPU smoke size. This fixed the
zero-projection debug failure.

## Main 6-View v1 Smoke

Run:

```text
output/normal_line_multiview_20260505/raw_softsurfel_surface_smoke6_t126_export6v
```

Command class:

```text
target_size = 126
views = 6
steps = 6
surfels = 600
render = pure Torch CPU soft surfel
export target protocol = 6views_sparseproto_headshoulder_crop at 518
```

Result:

```text
truthful_status = raw_softsurfel_surface_smoke_complete_not_teacher_or_candidate
initial mean IoU = 0.7653
optimized mean IoU = 0.8434
IoU delta = +0.0781
initial target recall = 0.8835
optimized target recall = 0.8734
target recall delta = -0.0101
```

Interpretation:

- The raw-image soft surfel renderer has a real optimization signal.
- The recall guard reduced the shrink-to-fit failure seen in the previous
  6-view smoke.
- This still only proves a local raw-image surface optimization loop. It is not
  a modeled face/hair/hand surface and not a strict-passing teacher.

Outputs:

```text
optimized_softsurfel_surface_mesh.ply
initial_overlay_contact_sheet.png
optimized_overlay_contact_sheet.png
soft_render_overlay_contact_sheet.png
rasterized_surface_targets/rasterized_surface_targets.npz
rasterized_surface_targets/debug_images/*_{mask,depth,normal}.png
```

## Teacher Gate Diagnostics

Exported target:

```text
output/normal_line_multiview_20260505/raw_softsurfel_surface_smoke6_t126_export6v/rasterized_surface_targets/rasterized_surface_targets.npz
```

### A. Strict VGGT/reference-depth gate

Reference:

```text
output/local_inference_results/r44_real_camera_oracle_keepworld_on6v_headshoulder/predictions.npz
```

Result:

```text
numeric_pass = false
visual_pass = false
pass = false
```

Aggregate diagnostic:

| ROI | raw visible coverage | depth-compatible coverage | raw median depth residual |
| --- | ---: | ---: | ---: |
| face_core | 0.8508 | 0.0000 | 1.9944 |
| head_face | 0.8596 | 0.0000 | 2.0050 |
| hairline | 0.6304 | 0.0000 | 1.9924 |
| head | 0.8596 | 0.0000 | 2.0050 |

Meaning:

The raw-camera optimized surface projects into the 6-view image protocol with
good visible coverage, but it is not in the VGGT/reference prediction depth
space. This is a coordinate/depth-protocol blocker, not a threshold problem.

### B. Self-protocol sanity gate

Reference:

```text
rasterized_surface_targets.npz
```

Result:

```text
numeric_pass = false
visual_pass = false
pass = false
```

Aggregate diagnostic:

| ROI | depth-compatible coverage | per-ROI numeric pass |
| --- | ---: | ---: |
| face_core | 0.8508 | 6 / 6 |
| head_face | 0.8596 | 6 / 6 |
| hairline | 0.6304 | 0 / 6 |
| head | 0.8596 | 6 / 6 |

Meaning:

The exported NPZ format and raw-camera self-protocol are internally coherent for
face/head. Hairline remains under-covered or too holey, and visual review is
still missing, so it is not a strict teacher.

## Current Blockers

1. Raw-camera surface is not yet bridged into the VGGT/reference prediction
   depth/world protocol used by the current strict gate.
2. Hairline coverage is still below the strict teacher threshold.
3. The renderer is a soft surfel smoke, not a true visibility/depth-ordered
   soft triangle renderer.
4. No explicit Open3D visual review has passed.
5. Full-body and hand strict candidate gates have not been run on this surface.

## Next Non-Wall Actions

Do not return to r-candidate threshold/confidence loops.

Next actions should be:

1. Add a truthful raw-camera to VGGT/reference protocol bridge diagnostic:
   estimate whether a single similarity transform can align raw SMPL-X surface
   depth to the chosen VGGT prediction protocol without destroying projection.
2. Improve hairline/head-top surface support using image boundary / mask-edge
   residuals rather than SMPL-X face/hair hard teacher.
3. Add depth-ordered soft visibility or a soft triangle renderer, then rerun the
   same 6-view and 60-view raw-image checks.
4. Only after raw surface, protocol bridge, hairline, Open3D visual, full-body,
   and hands pass strict gates should any learned backend or cloud run be
   considered.
