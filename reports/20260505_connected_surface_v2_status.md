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

## Part-Aware Coverage Smoke

The optimizer was extended with an optional `--part-recall-weight` guard and
coarse raw-mask part proxies:

```text
head_upper: top raw-mask region, rendered by head/face + hairline surfels
hairline_top: top raw-mask strip, rendered by hairline surfels
hands_side: left/right raw-mask side regions, rendered by both hand surfels
```

This first exposed a representation problem: with area-only sampling at 1800
surfels, the connected surface had almost no differentiable support for the
mentor-critical parts:

```text
torso_limbs = 557
left_hand = 24
right_hand = 25
head_face = 9
head_top_hairline_proxy = 139
lower_clothing_proxy = 1046
```

A part-balanced sampler was therefore added as a representation diagnostic,
not as a pass gate. The balanced smoke used:

```text
output/normal_line_multiview_20260505/connected_surface_template_v2_0012_11_frame0000_smoothcap_balanced_partrecall_smoke3_t96
```

Balanced surfel support:

```text
torso_limbs = 395
left_hand = 126
right_hand = 127
head_face = 222
head_top_hairline_proxy = 365
lower_clothing_proxy = 565
```

Metrics:

```text
initial mean IoU = 0.7690008282661438
optimized mean IoU = 0.7772732377052307
IoU delta = +0.008272409439086914
initial target recall = 0.8903587460517883
optimized target recall = 0.8394091725349426
target recall delta = -0.0509495735168457
```

Final part recall losses were still high:

```text
head_upper = 0.6917787194252014
hairline_top = 0.28859564661979675
hands_side = 0.7944923043251038
```

Interpretation:

Balanced sampling fixes the obvious lack of head/hand/hairline surfel support,
but the coarse image-space part proxies still do not produce a valid connected
surface. The hard rasterized full-body target recall drops further, so this is
another negative diagnostic. Do not tune the part proxy fractions or weight as
another loop. The next non-redundant step must improve the surface objective or
renderer itself, rather than weighting the same soft-splat mask losses harder.

## Mesh-Level Hair Boundary Smoke

The connected cap payload stores the scalp seam and cap vertex order, so the
optimizer was extended with an optional `--hair-boundary-weight`. This is a
mesh-level diagnostic: it pulls the connected hair/head cap outer ring toward
the raw human-mask silhouette using the image SDF. It does not use VGGT
depth/point/normal and does not create a candidate.

Smoke output:

```text
output/normal_line_multiview_20260505/connected_surface_template_v2_0012_11_frame0000_smoothcap_hairboundary_smoke3_t96_export6v
```

Small-smoke metrics:

```text
initial mean IoU = 0.7690008282661438
optimized mean IoU = 0.8019518852233887
IoU delta = +0.03295105695724487
initial target recall = 0.8903587460517883
optimized target recall = 0.8553047180175781
target recall delta = -0.035054028034210205
```

This is better than the proxy-only part recall diagnostic, and the target recall
drop is smaller than the previous soft-depth smoke, but it still loses full-mask
coverage and is not a teacher.

After rasterizing to the original 6-view headshoulder protocol, bridging into
the signfix VGGT world, and running the strict teacher gate:

```text
overall numeric pass = false
visual pass = false
face_core pass = 1 / 6
head_face pass = 2 / 6
hairline pass = 0 / 6
head pass = 2 / 6
```

Passing subviews were limited:

```text
face_core: view05 only, coverage 0.9650, p50 residual 0.0084, p90 residual 0.0181
head_face: view01 and view05 only
head: view01 and view05 only
hairline: 0 / 6
```

Interpretation:

The mesh-level hair-boundary term is the first non-redundant positive signal in
v2, because it acts on connected cap vertices rather than free points or coarse
part proxies. However, the strict gate remains blocked: coverage is view-local,
hairline still fails every view, and the Open3D result still cannot be called a
normal human head/face/hair surface.

## 6-View / 4000-Surfel Scale Smoke

To check whether the previous result was only a tiny 3-view / 1800-surfel smoke,
the same connected hair-boundary setup was scaled locally to 6 spaced views and
4000 balanced surfels:

```text
output/normal_line_multiview_20260505/connected_surface_template_v2_0012_11_frame0000_smoothcap_hairboundary_smoke6_t96_s4000
```

Metrics:

```text
initial mean IoU = 0.7700232863426208
optimized mean IoU = 0.7995951771736145
IoU delta = +0.029571890830993652
initial target recall = 0.8914699554443359
optimized target recall = 0.8558440208435059
target recall delta = -0.03562593460083008
```

Interpretation:

More views and more balanced surfels preserve the same basic pattern as the
3-view hair-boundary smoke: mask IoU improves, but hard target recall still
drops by about 3.5 points. This means the current soft-splat connected carrier
still prefers a tighter template shell and does not form a mentor-valid raw
surface. Do not continue scaling view count or surfel count as the next move.

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
move beyond soft-splat/proxy-mask objectives toward mesh-level connected
visibility: a depth-ordered triangle or connected-surfel renderer with
surface-aware boundary/edge, face weak reprojection, hand-wrist connectivity,
and full-body visual review. Balanced surfel sampling should stay as a debug
guard, not as the main optimization signal.
```

Only if the optimized 60-view connected surface looks like a normal human and
rasterizes back to the original 6-view protocol under the strict teacher gate
can it become a teacher for a later 6-view learned backend.
