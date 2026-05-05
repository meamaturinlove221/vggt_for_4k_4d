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

1. Hairline coverage is still below the strict teacher threshold.
2. The current global bridge is close but not strict enough: several views still
   fail median depth residual (`>0.025m`) and view 3 has head/head_face coverage
   and p90 depth residual failures.
3. The renderer is a soft surfel smoke, not a true visibility/depth-ordered
   soft triangle renderer.
4. No explicit Open3D visual review has passed.
5. Full-body and hand strict candidate gates have not been run on this surface.

## Protocol Bridge Diagnostic

New tool:

```text
tools/audit_raw_surface_vggt_protocol_bridge.py
```

Run:

```text
output/normal_line_multiview_20260505/raw_softsurfel_surface_smoke6_t126_export6v/raw_to_vggt_protocol_bridge_headface
```

It estimates a single global similarity transform from the raw-camera surface
target to the VGGT/reference prediction protocol using the `head_face` ROI. This
is a diagnostic only; it is not a teacher pass.

Estimated transform:

```text
scale = 0.5886290099
det(rotation) ~= 1.0
translation = [-0.0110, 0.1325, 1.0808]
```

Aggregate after bridge:

| ROI | valid coverage | compatible coverage | residual p50 | residual p90 |
| --- | ---: | ---: | ---: | ---: |
| face_core | 0.8508 | 0.8385 | 0.0280 | 0.0414 |
| head_face | 0.8596 | 0.7591 | 0.0313 | 0.0563 |
| hairline | 0.6304 | 0.6255 | 0.0309 | 0.0388 |
| head | 0.8596 | 0.7591 | 0.0313 | 0.0563 |

Gate on transformed NPZ:

```text
numeric_pass = false
visual_pass = false
pass = false
```

Per-ROI transformed gate:

```text
face_core: 2 / 6 views pass
head_face: 2 / 6 views pass
hairline: 0 / 6 views pass
head: 2 / 6 views pass
```

Meaning:

- The raw/VGGT protocol mismatch is not a dead end: a global similarity transform
  removes the previous roughly `2m` residual and brings most residuals close to
  the current gate thresholds.
- It is still not strict-passing. Hairline remains the hardest blocker, and the
  median residual threshold is too tight for several views.
- This supports continuing the raw-image surface backend route, but does not
  permit cloud or teacher-supervised training.

Extra fit comparison:

```text
fit_roi = face_core
face_core compatible coverage = 0.8315
head_face compatible coverage = 0.7077
head_face residual p90 = 0.0657
```

Face-core-only bridge improves nothing overall and makes head/head_face worse.
The current best non-cheating bridge remains a single global `head_face`-fit
similarity. Per-view offsets or per-view threshold tuning should remain blocked
because they would turn this into a protocol-fitting shortcut rather than a
defensible surface bridge.

## Hairline Flex Smoke

Run:

```text
output/normal_line_multiview_20260505/raw_softsurfel_surface_smoke6_t126_hairflex_export6v
```

This loosened head/head-top residual limits and increased boundary pressure to
test whether hairline failure is just an offset-limit issue.

Result:

```text
initial mean IoU = 0.7653
optimized mean IoU = 0.7729
IoU delta = +0.0076
initial target recall = 0.8835
optimized target recall = 0.7892
target recall delta = -0.0944
```

Conclusion:

This is negative. Simply allowing larger head/hairline offsets causes a
shrink-to-fit failure and does not create a better surface. Do not continue by
increasing hairline offset limits. The next hairline attempt needs actual
head-top/hair support from image boundary / mask-edge surface construction, not
looser SMPL-X residuals.

## Next Non-Wall Actions

Do not return to r-candidate threshold/confidence loops.

Next actions should be:

1. Refine the bridge beyond a single global similarity only if it remains
   geometrically meaningful and does not become a per-view cheat; test whether a
   robust head/face weighted similarity or bounded affine scale explains the
   remaining residuals.
2. Improve hairline/head-top surface support using image boundary / mask-edge
   residuals rather than SMPL-X face/hair hard teacher.
3. Add depth-ordered soft visibility or a soft triangle renderer, then rerun the
   same 6-view and 60-view raw-image checks.
4. Only after raw surface, protocol bridge, hairline, Open3D visual, full-body,
   and hands pass strict gates should any learned backend or cloud run be
   considered.
