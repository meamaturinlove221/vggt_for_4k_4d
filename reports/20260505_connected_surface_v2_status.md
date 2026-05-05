# Connected Surface v2 Local Status

Status: `not_passed_not_teacher_not_candidate`

Cloud status: blocked. This report does not change
`reports/20260504_strict_gate_registry.json`: strict candidate passes and strict
teacher passes remain zero.

## Why This Route Exists

Earlier raw-image v1 checks showed that silhouette and hairline signals exist,
but free hairline surfels and pure SMPL-X head offsets still produced floating
points, template head shells, or target-recall regressions. Continuing offset,
support, threshold, or view-count loops would repeat the same failure mode.

This v2 step therefore changes the representation: it builds a connected,
part-aware human surface carrier before any training or cloud work.

## Implemented

- Added `tools/build_connected_human_surface_template.py`.
- Added optional `--connected-template-payload` support to
  `tools/optimize_raw_smplx_softsurfel_torch.py`.
- The raw optimizer can now use a connected hybrid mesh instead of plain SMPL-X.
- No VGGT depth, point, normal, confidence, or r-candidate output is used as a
  teacher.

## Template Output

Template directory:

```text
output/normal_line_multiview_20260505/connected_surface_template_v2_0012_11_frame0000_smoothcap
```

Key files:

```text
connected_human_surface_template_payload.npz
connected_human_surface_template_hybrid.ply
connected_head_hair_cap_template.ply
smplx_part_template_full.ply
open3d_hybrid_template_review/
open3d_hair_cap_template_review/
```

Counts:

```text
base vertices = 10475
base faces = 20908
hybrid vertices = 10764
hybrid faces = 21580
hair seam vertices = 96
hair cap new vertices = 289
hair cap faces = 672
```

The first cap attempt produced crossing spike sheets. That was rejected and the
generator was corrected to use a smoothed inner ring, explicit scalp-anchor weld
faces, an outer ring, and a top cap ring. The resulting scaffold is connected,
but it is still only a carrier and remains template-like.

## Connected Optimizer Smoke

Smoke output:

```text
output/normal_line_multiview_20260505/connected_surface_template_v2_0012_11_frame0000_smoothcap_opt_smoke3_t96
```

Configuration:

```text
views = 3
target_size = 96
steps = 8
surfel_samples = 1800
connected_template_payload = connected_surface_template_v2_0012_11_frame0000_smoothcap
uses_vggt_depth_point_normal = false
creates_teacher_targets = false
creates_candidate_predictions = false
```

Metrics:

```text
initial mean IoU = 0.7690008282661438
optimized mean IoU = 0.7914394736289978
IoU delta = +0.022438645362854004
initial target recall = 0.8903587460517883
optimized target recall = 0.8473384976387024
target recall delta = -0.04302024841308594
```

Visual review:

```text
open3d_optimized_connected_template_review/solid_mesh/iso.png
open3d_optimized_connected_template_review/solid_mesh/head_close.png
optimized_overlay_contact_sheet.png
soft_render_overlay_contact_sheet.png
```

Interpretation:

The connected carrier and optimizer are now wired, and the mask objective has a
valid local gradient. However, target recall regresses and Open3D still shows a
template body/head with a crude connected cap, not a modeled face, hairline, or
normal human surface. This is a useful implementation step, not a mentor pass.

## Original 6-View Teacher Gate

The smoke was also rasterized back to:

```text
output/4k4d_preprocessed_scene_variants/0012_11_frame0000_6views_sparseproto_headshoulder_crop
```

Initial audit against the signfix protocol exposed a bridge-tool bug: the dense
target bridge transformed `world_points` but preserved source `depth` and source
cameras. That produced misleading ~2m residuals. The bridge helper was fixed to:

```text
transform world_points
rotate normals
overwrite intrinsic/extrinsic with the target protocol cameras
recompute depth/depths from transformed world_points in the target camera frame
```

Fixed bridge output:

```text
output/normal_line_multiview_20260505/connected_surface_template_v2_0012_11_frame0000_smoothcap_opt_smoke3_t96_export6v/raw_export_to_signfix_bridge_fixed/teacher_targets.npz
```

Fixed strict teacher-gate output:

```text
output/normal_line_multiview_20260505/connected_surface_template_v2_0012_11_frame0000_smoothcap_opt_smoke3_t96_export6v/teacher_gate_after_raw_export_bridge_fixed
```

Numeric result after the fixed bridge:

```text
overall numeric pass = false
visual pass = false
face_core pass = 1 / 6
head_face pass = 2 / 6
hairline pass = 0 / 6
head pass = 2 / 6
```

Representative depth-compatible results:

```text
view01 head_face: coverage 0.9069, p50 depth residual 0.0161, pass true
view05 head_face: coverage 0.8839, p50 depth residual 0.0099, pass true
view05 face_core: coverage 0.9620, p50 depth residual 0.0087, pass true
view00 face_core: coverage 0.1181, p50 depth residual 0.0563, pass false
view00 hairline: coverage 0.0131, p50 depth residual 0.0539, pass false
hairline total: 0 / 6 pass
```

Open3D visual review:

```text
teacher_gate_after_raw_export_bridge_fixed/open3d_teacher_head_face/iso.png
teacher_gate_after_raw_export_bridge_fixed/open3d_teacher_face_core/face_close.png
teacher_gate_after_raw_export_bridge_fixed/open3d_teacher_hairline/iso.png
```

The fixed bridge improves coordinate compatibility, but the visible geometry is
still a template-like head/body shell with a crude cap and missing/fragmented
hairline/face support. It is not a normal human Open3D surface and must remain
blocked.

## Soft Depth-Ordering Smoke

The soft surfel renderer was extended with an optional `--depth-softness`
parameter. When enabled, the renderer keeps the differentiable spatial alpha
mask, but computes depth/normal maps with front-surface-biased weights. This is
a renderer-layer diagnostic, not another threshold/support loop.

Depth-ordered smoke:

```text
output/normal_line_multiview_20260505/connected_surface_template_v2_0012_11_frame0000_smoothcap_depthsoft_smoke3_t96
```

Comparison against the previous connected-template smoke:

```text
baseline connected smoke:
  IoU delta = +0.022438645362854004
  target recall delta = -0.04302024841308594
  photo loss first -> last = 0.09653176367282867 -> 0.09178324043750763

soft depth-ordering smoke:
  depth_softness = 0.035
  IoU delta = +0.021018385887145996
  target recall delta = -0.04438692331314087
  photo loss first -> last = 0.0676174983382225 -> 0.053195662796497345
```

Interpretation:

Soft depth ordering improves the photometric visibility signal, but it does not
fix the surface problem: recall still drops, and this smoke still has no strict
teacher pass. The next useful step is therefore not more renderer temperature
tuning; it is adding an actual connected surface objective that can move the
cap/face/hands toward raw-image boundaries without shrinking the body.

## Decision

Do not:

```text
claim success
cloud upload
cloud train
turn this into an r-candidate
continue tuning offsets/support/threshold/view count
```

Next non-redundant step:

```text
replace pure soft-splat masking with depth-ordered connected surface rendering,
then add raw-image surface losses that can actually shape the connected cap:
multi-view photometric consistency, boundary/edge terms, face weak reprojection,
hand connectivity, and full-body visual review.
```

Only if the optimized 60-view connected surface looks like a normal human and
rasterizes back to the original 6-view protocol under the strict teacher gate
can it become a teacher for a later 6-view learned backend.
