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

## Freeze-Global Transform Diagnostic

The 6-view / 4000-surfel hair-boundary smoke ended with a global scale of about
`0.953` and a non-trivial translation. This showed that the apparent IoU
improvement was largely coming from shrinking/sliding the whole template shell,
which directly conflicts with the full-body and hairline coverage requirement.

The optimizer therefore gained an explicit `--freeze-global-transform` switch.
With the same 6-view / 4000-surfel setup:

```text
output/normal_line_multiview_20260505/connected_surface_template_v2_0012_11_frame0000_smoothcap_hairboundary_freezeglobal_smoke6_t96_s4000
```

Metrics:

```text
initial mean IoU = 0.7700232863426208
optimized mean IoU = 0.7720444798469543
IoU delta = +0.002021193504333496
initial target recall = 0.8914699554443359
optimized target recall = 0.8888117671012878
target recall delta = -0.0026581883430480957
```

Interpretation:

Freezing the global transform prevents the bad hard-recall collapse, but it also
removes nearly all of the previous IoU gain. That confirms the current
soft-splat objective was mostly exploiting global template shrinkage rather than
learning a better local human surface.

The freeze-global export was also rasterized to original 6-view headshoulder and
audited under a short output path to avoid an Open3D Windows long-path failure:

```text
output/surface_gate_freezeglobal_tmp
```

Strict teacher gate:

```text
overall numeric pass = false
visual pass = false
face_core pass = 2 / 6
head_face pass = 2 / 6
hairline pass = 0 / 6
head pass = 2 / 6
```

Open3D still shows a template-like shell/cap rather than a normal human
head/face/hair surface. This is not a teacher and does not unblock training or
cloud.

The same freeze-global setup was then run for 40 local optimization steps:

```text
output/normal_line_multiview_20260505/connected_surface_template_v2_0012_11_frame0000_smoothcap_hairboundary_freezeglobal_40step6_t96_s4000
```

Metrics:

```text
initial mean IoU = 0.7700232863426208
optimized mean IoU = 0.7779861092567444
IoU delta = +0.007962822914123535
initial target recall = 0.8914699554443359
optimized target recall = 0.8838089108467102
target recall delta = -0.007661044597625732
```

Local residuals did move more than the 12-step smoke:

```text
hairline normal-offset mean = 0.004240455571562052
hairline free-offset mean = 0.005870182067155838
left/right hand mean offsets ~= 0.0027 - 0.0028
```

But after export + bridge + strict teacher gate:

```text
output/surface_40step_gate_tmp

overall numeric pass = false
visual pass = false
face_core pass = 2 / 6
head_face pass = 2 / 6
hairline pass = 0 / 6
head pass = 2 / 6
```

Interpretation:

More local steps allow the connected cap and hands to move by a few millimeters,
but the optimized result is still a template shell with an incomplete hairline.
This confirms the current raw-image v2 carrier can be optimized locally, but it
still lacks the surface representation/objective needed for a strict-passing
head/face/hair teacher.

## Connected Face Landmark Weak-Loss Diagnostic

To avoid reviving the previously frozen floating MediaPipe face-patch route, the
optimizer was extended with an optional weak 2D landmark reprojection loss:

```text
--face-landmarker-task
--face-landmark-weight
--face-landmark-bidir-weight
--face-landmark-min-points
```

Important constraint:

```text
MediaPipe landmarks are not triangulated and no face patch is created.
They only weakly constrain the already-connected SMPL-X / hybrid face vertices.
```

The base Python 3.13 environment does not provide `mediapipe`; the local
`g3splat` environment does, so the landmark smoke was run with:

```text
D:\anaconda\envs\g3splat\python.exe
```

Short detection smoke:

```text
output/normal_line_multiview_20260505/landmark_loss_detect_stride10_smoke6_t96_step1
```

Detected face landmarks:

```text
detected views = 4 / 6
mean inside-mask ratio = 0.8922594142259415
face landmark vertex count = 943
```

This proves the 2D signal can be attached to the connected face mesh without
constructing a floating teacher patch.

The actual 40-step freeze-global diagnostic was:

```text
output/normal_line_multiview_20260505/smoothcap_face_landmark_freezeglobal_40step6_stride10_t96_s4000
```

Metrics:

```text
initial mean IoU = 0.7700232863426208
optimized mean IoU = 0.7612533569335938
IoU delta = -0.0087699294090271
initial target recall = 0.8914699554443359
optimized target recall = 0.8518168330192566
target recall delta = -0.039653122425079346
```

Losses did move:

```text
face landmark loss: 0.10886478424072266 -> 0.10491538792848587
photometric consistency: 0.12081477046012878 -> 0.11680746078491211
hair boundary loss: 0.023661160841584206 -> 0.018935473635792732
```

But the hard rasterized metrics regressed, and the visual overlays remain a
template-like connected shell rather than a modeled human face/head/hair
surface. Therefore this is a negative diagnostic, not a teacher and not a
candidate. Do not continue by tuning the landmark weight: the failure mode is
representation/objective mismatch, not a missing scalar weight.

## Connected Hand Landmark Weak-Loss Diagnostic

Because the mentor explicitly requires full-body and hand detail as hard
bottom-line checks, the same weak-landmark idea was applied to hands, again
without constructing a floating patch:

```text
--hand-landmarker-task
--hand-landmark-weight
--hand-landmark-bidir-weight
--hand-landmark-min-points
```

Important constraint:

```text
MediaPipe hand landmarks are only 2D weak constraints on connected SMPL-X hand
vertices. They are not triangulated and do not create a hand teacher patch.
```

Detection smoke:

```text
output/normal_line_multiview_20260505/hand_landmark_detect_stride10_smoke6_t96_step1
```

Detected hand landmarks:

```text
detected views = 6 / 6
detected hands = 9
mean inside-mask ratio = 0.9722222222222222
left/right hand vertex counts = 629 / 628
```

This confirms there is usable 2D hand evidence in the selected 60-view raw
images.

The 40-step freeze-global hand-landmark diagnostic was:

```text
output/normal_line_multiview_20260505/smoothcap_hand_landmark_freezeglobal_40step6_stride10_t96_s4000
```

Metrics:

```text
initial mean IoU = 0.7700232863426208
optimized mean IoU = 0.762291431427002
IoU delta = -0.0077318549156188965
initial target recall = 0.8914699554443359
optimized target recall = 0.8523262143135071
target recall delta = -0.03914374113082886
```

Losses did move:

```text
hand landmark loss: 0.05577178671956062 -> 0.05188523605465889
photometric consistency: 0.12127858400344849 -> 0.11710040271282196
hair boundary loss: 0.023661160841584206 -> 0.018925823271274567
```

Hand vertices moved more than in the no-landmark run:

```text
left hand mean abs offset = 0.008744870312511921
right hand mean abs offset = 0.009468389675021172
```

However, the hard rasterized body metrics still regress and the visual overlay
remains a connected template shell, not a normal-human hand/full-body surface.
Therefore the hand landmark signal is useful evidence, but the current carrier
and objective still cannot satisfy the full-body/hands strict gate. Do not
continue by tuning hand landmark weights.

## Soft Triangle Renderer Diagnostic

The repeated negative landmark / soft-splat results indicate that the renderer
itself is part of the failure mode: Gaussian surfel splats can improve proxy
losses without representing connected mesh visibility. The optimizer therefore
now has an explicit renderer switch:

```text
--renderer surfel
--renderer triangle
```

`surfel` remains the default. `triangle` is a CPU-only diagnostic renderer using
a sampled set of connected mesh triangles and a soft barycentric inside test. It
is not a production rasterizer, not a teacher, and not a cloud unblocker.

Tiny 1-view smoke:

```text
output/normal_line_multiview_20260505/triangle_renderer_smoke1_t64_s2000
sampled render faces = 1771
mask loss = 0.745597779750824
soft recall loss = 0.7477416396141052
```

The 5-step 1-view optimization smoke was:

```text
output/normal_line_multiview_20260505/triangle_renderer_opt_smoke1_t64_s2000_step5
```

Metrics:

```text
initial mean IoU = 0.7706708312034607
optimized mean IoU = 0.7737909555435181
IoU delta = +0.003120124340057373
initial target recall = 0.8502581715583801
optimized target recall = 0.8537005186080933
target recall delta = +0.0034423470497131348
```

Loss decreased monotonically:

```text
loss: 1.0841139554977417 -> 1.0075054168701172
mask loss: 0.745597779750824 -> 0.6873644590377808
soft recall loss: 0.7477416396141052 -> 0.7071121335029602
```

The 5-step 3-view optimization smoke was:

```text
output/normal_line_multiview_20260505/triangle_renderer_opt_smoke3_t64_s2000_step5
```

Metrics:

```text
initial mean IoU = 0.7706115245819092
optimized mean IoU = 0.7750326991081238
IoU delta = +0.0044211745262146
initial target recall = 0.8578081130981445
optimized target recall = 0.8643603324890137
target recall delta = +0.006552219390869141
```

Loss again decreased monotonically:

```text
loss: 0.8999999165534973 -> 0.8428559303283691
mask loss: 0.5696393847465515 -> 0.5290072560310364
soft recall loss: 0.7117570042610168 -> 0.6752879619598389
photometric consistency: 0.09462591260671616 -> 0.09386990964412689
```

The renderer was then fixed so diagnostic contact sheets follow the selected
renderer. Before this fix, `--renderer triangle` still saved surfel debug
overlays, which could mislead visual review. The optimizer also gained:

```text
--triangle-render-face-budget
```

where `-1` renders all connected mesh faces at low resolution instead of only
sampled surfel faces.

All-face CUDA smoke:

```text
output/normal_line_multiview_20260505/triangle_renderer_allfaces_cuda_smoke1_t64_step1
renderer = triangle
triangle_render_face_budget = -1
sampled render faces = 21580
device = cuda in D:\anaconda\envs\g3splat
```

Metrics:

```text
initial mean IoU = 0.7706708312034607
optimized mean IoU = 0.772230863571167
IoU delta = +0.0015600323677062988
initial target recall = 0.8502581715583801
optimized target recall = 0.8519793748855591
target recall delta = +0.001721203327178955
```

Interpretation:

All-face triangle rendering removes the sampled-face speckle in debug overlays
and is locally executable on RTX 5080 through the `g3splat` Python/Torch stack.
It is still only a `64px`, `1-view`, `1-step` smoke and is not a teacher, but it
confirms that connected mesh visibility can be tested locally without falling
back to Gaussian surfel splats.

The same all-face triangle renderer was then run for a slightly larger local
diagnostic:

```text
output/normal_line_multiview_20260505/triangle_renderer_allfaces_cuda_smoke3_t64_step5
renderer = triangle
triangle_render_face_budget = -1
sampled render faces = 21580
views = 3
steps = 5
target size = 64
global transform frozen = true
```

Metrics:

```text
initial mean IoU = 0.7706115245819092
optimized mean IoU = 0.7802132964134216
IoU delta = +0.009601771831512451
initial target recall = 0.8578081130981445
optimized target recall = 0.8672575950622559
target recall delta = +0.009449481964111328
```

Loss decreased monotonically:

```text
loss: 0.3032742142677307 -> 0.29087352752685547
mask loss: 0.18814441561698914 -> 0.1802668273448944
soft recall loss: 0.23325559496879578 -> 0.22342583537101746
photometric consistency: 0.09475450962781906 -> 0.09383748471736908
```

This is the strongest local raw-surface signal so far because hard raster IoU
and target recall improve together while the global transform is frozen and all
connected mesh faces participate in the soft mask. The limitation is equally
important: the visual result is still a low-resolution SMPL-X-like template
surface, not a normal-human Open3D surface with modeled face/hair/hands. This is
therefore a renderer/backend direction signal, not a teacher pass.

Attempting full all-face triangle rendering at `96px` with the current local GPU
memory state failed with CUDA OOM, even with smaller chunks:

```text
output/normal_line_multiview_20260505/triangle_renderer_allfaces_cuda_smoke3_t96_step5_fchunk1024
output/normal_line_multiview_20260505/triangle_renderer_allfaces_cuda_smoke3_t96_step5_chunk256
```

To avoid killing unrelated user GPU processes or brute-forcing memory, the next
smoke used a deterministic `8000`-face budget:

```text
output/normal_line_multiview_20260505/triangle_renderer_budget8000_cuda_smoke3_t96_step5
renderer = triangle
triangle_render_face_budget = 8000
views = 3
steps = 5
target size = 96
global transform frozen = true
```

Metrics:

```text
initial mean IoU = 0.7690008282661438
optimized mean IoU = 0.7756476402282715
IoU delta = +0.0066468119621276855
initial target recall = 0.8903587460517883
optimized target recall = 0.8972101211547852
target recall delta = +0.006851375102996826
```

Loss decreased monotonically:

```text
loss: 0.6194117069244385 -> 0.5563229918479919
mask loss: 0.35480600595474243 -> 0.3121137320995331
soft recall loss: 0.5641481876373291 -> 0.5190099477767944
photometric consistency: 0.09914322942495346 -> 0.09848229587078094
```

Interpretation:

The `96px` budgeted triangle smoke preserves the same non-collapsing direction,
but sampled-face soft overlays remain speckled and the hard overlay is still
only a coarse template silhouette. This is not enough for any teacher/candidate
gate, but it is a stronger argument that future work should build or install a
proper connected mesh rasterizer instead of continuing soft-splat proxy tuning.

Interpretation:

This is the first raw-surface diagnostic in this sequence where the optimized
hard-raster IoU and target recall both improve without allowing global
scale/translation shrinkage. The effect is still small and low-resolution
(`64px`, `3 views`, `5 steps`, sampled triangles only), and the visual result is
not a normal-human Open3D surface. It should not be scaled by brute-force CPU
loops. The useful conclusion is narrower: the next non-redundant route is a real
connected-mesh rasterizer/visibility backend, preferably accelerated, rather
than further tuning surfel splat weights, landmark weights, support thresholds,
or r-candidates.

## Local Accelerated Rasterizer Feasibility

Local environment check:

```text
base python: Python 3.13.5, torch 2.9.0+cu126
base cuda warning: RTX 5080 / sm_120 is not supported by that torch build

g3splat python: Python 3.10.19, torch 2.9.1+cu130
g3splat cuda_available = true
GPU = NVIDIA GeForce RTX 5080
```

The `g3splat` environment is the only reasonable local GPU Python for this
surface work. However:

```text
nvdiffrast installed = false
pytorch3d installed = false
kaolin installed = false
pip index nvdiffrast = no matching distribution
pip index pytorch3d = no matching distribution
pip index kaolin = only kaolin 0.1
cl compiler in PATH = false
nvcc = CUDA 13.1
```

Interpretation:

The next method-level step should use an actual accelerated differentiable
rasterizer, but this local Windows environment does not currently expose a
ready wheel or compiler path for `nvdiffrast` / `pytorch3d` / modern `kaolin`.
Therefore it is valid to keep the CPU triangle renderer as a small correctness
smoke, but it should not be brute-force scaled into 60-view training loops. A
proper next implementation needs either:

```text
1. a prepared environment with nvdiffrast / PyTorch3D / Kaolin working on RTX 5080;
2. a small custom CUDA extension compiled in a verified Visual Studio + CUDA path;
3. or a cloud/local Linux environment after strict local design gates justify it.
```

This does not unblock cloud training; it only identifies the missing renderer
backend required to move beyond proxy soft-splat objectives.

## Decision

Do not:

```text
claim success
cloud upload
cloud train
turn this into an r-candidate
continue tuning offsets/support/threshold/view count
continue tuning landmark loss weights
continue tuning hand landmark loss weights
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

## Existing Mesh Export Helper And Strict Gate Refresh

After disk cleanup, the current best positive-signal triangle diagnostic was
not re-optimized. Instead, it was exported from the already existing mesh to
avoid another parameter loop:

```text
tools/export_raw_surface_mesh_targets.py
```

Input mesh:

```text
output/normal_line_multiview_20260505/triangle_renderer_budget8000_cuda_smoke3_t96_step5/optimized_softsurfel_surface_mesh.ply
```

Export output:

```text
output/normal_line_multiview_20260505/triangle_renderer_budget8000_cuda_smoke3_t96_step5_export6v/rasterized_surface_targets/rasterized_surface_targets.npz
```

The helper only reads an ASCII triangle PLY and writes raw-camera rasterized
debug targets. It is explicitly marked:

```text
truthful_status = raw_surface_mesh_export_complete_not_teacher_or_candidate
uses_vggt_depth_point_normal_as_teacher = false
creates_candidate_predictions = false
allows_cloud = false
```

The export was bridged to the signfix VGGT protocol with the fixed bridge:

```text
output/normal_line_multiview_20260505/triangle_renderer_budget8000_cuda_smoke3_t96_step5_export6v/raw_export_to_signfix_bridge_fixed/teacher_targets.npz
```

Strict teacher gate with explicit visual fail:

```text
output/normal_line_multiview_20260505/triangle_renderer_budget8000_cuda_smoke3_t96_step5_export6v/teacher_gate_after_raw_export_bridge_manual_visual_fail_v2
```

Numeric gate:

```text
overall numeric pass = false
visual pass = false
overall pass = false
face_core pass = 2 / 6
head_face pass = 2 / 6
hairline pass = 0 / 6
head pass = 2 / 6
```

Aggregate compatible coverage:

```text
face_core mean compatible coverage = 0.4228
head_face mean compatible coverage = 0.6460
hairline mean compatible coverage = 0.3330
head mean compatible coverage = 0.6460
```

Explicit local visual review:

```text
output/normal_line_multiview_20260505/triangle_renderer_budget8000_cuda_smoke3_t96_step5/manual_visual_review_fail.json
```

Review conclusion:

```text
Open3D review shows a template-like SMPL-X head/body shell with missing modeled
face detail, fragmented hairline/head-top support, and non-human sparse surface
artifacts; not a normal human surface.
```

Representative reviewed images:

```text
teacher_gate_after_raw_export_bridge_fixed/open3d_teacher_head_face/iso.png
teacher_gate_after_raw_export_bridge_fixed/open3d_teacher_face_core/face_close.png
teacher_gate_after_raw_export_bridge_fixed/open3d_teacher_hairline/iso.png
```

Interpretation:

The connected triangle renderer is still the right implementation direction
because it improved hard-raster IoU and target recall without global shrink.
However, the current optimized surface is still a coarse SMPL-X-like shell. It
does not contain modeled face, hairline, or normal-human head surface geometry,
and it cannot become a teacher. Do not tune its thresholds, landmark weights,
or support counts. The next non-redundant step remains a real connected,
visibility-aware mesh/surface backend with enough expressive capacity for
face/hairline/hands, not another raw export or r-candidate.

## Refreshed Strict Registry After Cleanup

The strict registry was refreshed after removing old output directories:

```text
reports/20260504_strict_gate_registry.json
```

Cloud gate result:

```text
cloud_allowed = false
strict_candidate_passes = 0
strict_teacher_passes = 0
registry_age_hours ~= 0
```

This keeps the cloud guard current. Local cleanup removed stale artifacts but
did not change the truthful result: no candidate or teacher currently satisfies
the mentor strict gate.

## Full-Body And Hand Hard Gate On Raw Surface Export

After adding confidence maps to the raw surface raster export, the same
triangle diagnostic mesh was also projected to the original 6-view human-crop
full-body protocol:

```text
output/normal_line_multiview_20260505/triangle_renderer_budget8000_cuda_smoke3_t96_step5_export6v_fullbody/rasterized_surface_targets/rasterized_surface_targets.npz
```

Full-body / hand audit:

```text
output/normal_line_multiview_20260505/triangle_renderer_budget8000_cuda_smoke3_t96_step5_export6v_fullbody/fullbody_hand_audit/fullbody_hand_integrity_summary.json
```

Numeric result:

```text
points_after_conf = 165760
largest_component_ratio = 1.0000
full_body_gate.pass = true
hand_gate.pass = false
views_passing_hand_kept_ratio = 0
views_with_compact_3d_hand_boxes = 0
implausible_hand_boxes = 2
per-view body gate = 5 / 6
per-view hand gate = 0 / 6
```

Explicit Open3D / overlay review outputs:

```text
output/normal_line_multiview_20260505/triangle_renderer_budget8000_cuda_smoke3_t96_step5_export6v_fullbody/teacher_surface_fullbody_review/iso.png
output/normal_line_multiview_20260505/triangle_renderer_budget8000_cuda_smoke3_t96_step5_export6v_fullbody/teacher_surface_hands_review/iso.png
output/normal_line_multiview_20260505/triangle_renderer_budget8000_cuda_smoke3_t96_step5_export6v_fullbody/teacher_surface_hands_review/solid_mesh/iso.png
output/normal_line_multiview_20260505/triangle_renderer_budget8000_cuda_smoke3_t96_step5_export6v_fullbody/fullbody_hand_audit/view_02_fullbody_hand_overlay.png
```

Manual visual conclusion:

```text
The full-body surface is connected enough to pass the numeric body continuity
screen, but it is still a coarse SMPL-X-like template shell. The hands are not
mentor-pass geometry: MediaPipe-visible hand views fail compact 3D hand-box
checks, and Open3D still shows template fingers / hand support rather than
reconstructed personal hand detail. The head top remains an artificial cap and
the face remains template-like. This export is therefore not a strict teacher,
not a candidate pass, and not cloud-eligible.
```

Interpretation:

The raw-image connected surface route has produced a useful negative: body
continuity can be made much less broken than old point-cloud candidates, but
this alone is insufficient. The remaining blocker is not thresholding or point
count; it is the lack of a learned / optimizable surface representation that
can express true face, hairline, clothing, and hand details while staying
attached to the body. Do not tune the current mesh to chase the hand gate. The
next non-redundant method step is a part-aware connected surface backend with
real raw-image photometric and boundary evidence, not another fullbody export
or confidence change.

## Connected Surface v2.1 Local Subdivision And Part-Free Residual

To avoid repeating the same coarse-template carrier, v2.1 adds local conforming
triangle subdivision to the connected template builder:

```text
tools/build_connected_human_surface_template.py
```

New local-only carrier:

```text
output/normal_line_multiview_20260506/connected_surface_template_v21_conforming_subdiv_facehairhandscloth/connected_human_surface_template_payload.npz
```

Carrier counts:

```text
base vertices = 10475
base faces = 20908
hybrid vertices = 28185
hybrid faces = 56424
subdivision selected parts = left_hand, right_hand, head_face, head_top_hairline, lower_clothing_proxy
subdivision levels = 1
selected faces = 11496
new midpoint vertices = 17421
```

The subdivision is conforming across selected/unselected face boundaries, so it
does not intentionally create T-junctions. Payload invariants were checked:

```text
part_ids shape = 28185
face_front_vertex_mask shape = 28185
head_vertex_mask shape = 28185
hairline_vertex_mask shape = 28185
left/right hand masks shape = 28185
```

The raw optimizer now also supports optional bounded connected 3D part-free
residuals for face, hands, hairline, and clothing:

```text
--part-free-offset-limit-face
--part-free-offset-limit-hands
--part-free-offset-limit-hairline
--part-free-offset-limit-clothing
--part-free-offset-reg
--part-free-smooth-reg
```

These are disabled by default and remain connected mesh residuals, not floating
patches.

Smoke runs:

```text
output/normal_line_multiview_20260506/connected_surface_template_v21_conforming_softsurfel_smoke6_t96_step20
output/normal_line_multiview_20260506/connected_surface_template_v21_conforming_landmark_softsurfel_smoke6_t96_step20
output/normal_line_multiview_20260506/connected_surface_template_v21_partfree_partrecall_landmark_smoke6_t96_step20
```

Best part-free / part-recall / landmark smoke metrics:

```text
initial_iou.mean = 0.7700
optimized_iou.mean = 0.7727
iou_delta = +0.0027
initial_target_recall.mean = 0.8915
optimized_target_recall.mean = 0.8885
target_recall_delta = -0.0029
face landmarks detected = 4 / 6 views
hand landmarks detected = 6 / 6 views, 9 hands
```

Open3D review:

```text
output/normal_line_multiview_20260506/connected_surface_template_v21_partfree_partrecall_landmark_smoke6_t96_step20/open3d_review_full/iso.png
output/normal_line_multiview_20260506/connected_surface_template_v21_partfree_partrecall_landmark_smoke6_t96_step20/open3d_review_full/solid_mesh/face_close.png
output/normal_line_multiview_20260506/connected_surface_template_v21_partfree_partrecall_landmark_smoke6_t96_step20/open3d_review_hands/solid_mesh/iso.png
output/normal_line_multiview_20260506/connected_surface_template_v21_partfree_partrecall_landmark_smoke6_t96_step20/open3d_review_hands/solid_mesh/head_close.png
```

Conclusion:

```text
v2.1 is an implementation improvement, not a mentor pass. It increases local
connected surface capacity and verifies that face/hand landmark detectors can
supervise connected vertices, but the optimized result remains a template-like
SMPL-X shell. The face is not modeled, the head cap is artificial, clothing is
not reconstructed as personal geometry, and hands remain template/detail-poor.
No strict teacher gate was run because the visual precheck already fails.
```

Next non-redundant implication:

The bottleneck has moved beyond carrier density and bounded residual variables.
The current raw-image objective is still too weak to recover a normal-human
surface from RGB/mask constraints alone at this local smoke scale. Further
progress needs a stronger learned or optimized surface backend with real
visibility-aware photometric/normal objectives and possibly a strict-passing
dense target-frame teacher; do not continue by increasing v2.1 steps or weights
without adding new information or a stronger objective.

## Connected Surface v2.2 Raw RGB Edge-SDF Smoke

To avoid another parameter-only loop, v2.2 adds an optional raw-image edge
distance objective to the connected surface optimizer:

```text
tools/optimize_raw_smplx_softsurfel_torch.py
```

New local-only arguments:

```text
--image-edge-weight
--image-edge-part-ids
--image-edge-canny-low
--image-edge-canny-high
--image-edge-mask-dilate
--image-edge-max-distance
```

This loss computes Canny edges from the raw RGB crop inside a dilated human
mask, converts them to a normalized distance field, and samples that field only
at selected connected mesh vertices. It uses no VGGT depth, point, normal, or
confidence output; it also creates no floating face/hand/hair patch and is
disabled by default.

Smoke run:

```text
output/normal_line_multiview_20260506/connected_surface_v21_imageedge_partfree_landmark_smoke6_t96_step20
```

Metrics:

```text
initial_iou.mean = 0.7700
optimized_iou.mean = 0.7765
iou_delta = +0.0065
initial_target_recall.mean = 0.8915
optimized_target_recall.mean = 0.8907
target_recall_delta = -0.0008
image_edge_weight = 0.12
image_edge usable views = 6 / 6
mean raw edge pixels = 340.33
face landmarks detected = 4 / 6 views
hand landmarks detected = 6 / 6 views, 9 hands
```

Open3D review:

```text
output/normal_line_multiview_20260506/connected_surface_v21_imageedge_partfree_landmark_smoke6_t96_step20/open3d_review_full/solid_mesh/iso.png
output/normal_line_multiview_20260506/connected_surface_v21_imageedge_partfree_landmark_smoke6_t96_step20/open3d_review_full/solid_mesh/face_close.png
output/normal_line_multiview_20260506/connected_surface_v21_imageedge_partfree_landmark_smoke6_t96_step20/open3d_review_hands/solid_mesh/iso.png
```

Conclusion:

```text
The edge-SDF term gives a stronger 2D raw-image signal than the previous
v2.1 smoke, and the IoU delta improves from roughly +0.0027 to +0.0065. However
the visual failure mode is unchanged: the mesh remains a connected SMPL-X-like
template shell. The face is not a personalized modeled face, the head cap is
still artificial, clothing remains template-like, and hands are still not
mentor-pass hand geometry. Therefore no strict teacher gate was run and the
cloud remains blocked.
```

Next non-redundant implication:

The raw RGB edge distance objective helps local alignment but does not solve the
surface representation/objective gap. The next distinct local step should make
the connected mesh explain raw image appearance under its own visibility:
visibility-aware triangle RGB/gradient rendering with a fixed per-vertex
appearance estimate. That is a stronger objective than silhouette, landmark,
edge-SDF, or surfel color variance, and it still avoids VGGT shell recycling.

## Connected Surface v2.3 Triangle RGB / Gradient Render Smoke

v2.3 implements the next non-redundant objective proposed after v2.2: the
connected mesh now bakes fixed per-vertex RGB appearance from raw views and the
triangle renderer can output a barycentric color map. The optimizer can penalize
foreground RGB residual and Sobel-gradient residual between the rendered mesh
and raw images:

```text
--triangle-rgb-weight
--triangle-gradient-weight
--triangle-rgb-depth-tolerance
--triangle-rgb-mask-threshold
```

The color bake and render loss use raw RGB/masks/cameras only. They do not use
VGGT depth, point, normal, confidence, or any teacher mesh.

Initial limited-face smoke:

```text
output/normal_line_multiview_20260506/connected_surface_v22_triangle_rgbgrad_smoke3_t64_step5
```

This revealed a bad experimental setup rather than a useful signal:

```text
triangle color bake coverage = 0.0067
triangle RGB pixels per view = 21.67
```

The depth-gated color bake was too conservative when rendering only a small
face subset, so the bake was fixed to allow `--triangle-rgb-depth-tolerance 0`
to mean raw-mask/in-image color baking without sampled depth gating.

No-depth-gate limited-face smoke:

```text
output/normal_line_multiview_20260506/connected_surface_v22_triangle_rgbgrad_nodgate_smoke3_t64_step5
```

This fixed vertex color coverage but the face-budget triangle renderer was
still too sparse:

```text
triangle color bake coverage = 1.0000
triangle RGB pixels per view = 21.33
iou_delta = +0.0007
target_recall_delta = +0.0013
```

All-face renderer smoke:

```text
output/normal_line_multiview_20260506/connected_surface_v22_triangle_rgbgrad_allfaces_smoke2_t64_step5
```

This is the first valid triangle RGB/gradient smoke:

```text
rendered faces = 56424
triangle color bake coverage = 0.9135
triangle RGB pixels per view = 467.0
triangle_rgb_loss = 0.0995
triangle_gradient_loss = 0.3461
iou_delta = +0.0042
target_recall_delta = +0.0019
```

Open3D review:

```text
output/normal_line_multiview_20260506/connected_surface_v22_triangle_rgbgrad_allfaces_smoke2_t64_step5/open3d_review_full/solid_mesh/iso.png
```

Conclusion:

```text
The all-face triangle RGB/gradient objective is a real raw-image signal and is
more meaningful than the limited-face smoke. It improves 2D alignment without
shrinking recall. However, the optimized mesh remains a connected SMPL-X-style
template body with artificial head cap, template face, weak clothing geometry,
and non-personalized hands. This is not a strict teacher, not a candidate pass,
and not cloud-eligible.
```

Next non-redundant implication:

Triangle RGB/gradient makes the objective stronger, but the connected carrier
still lacks enough expressive geometry for hair, clothing, face, and hands. A
plain SMPL-X scaffold plus small bounded offsets cannot become the mentor-level
surface just by adding appearance losses. The next method-level change must
increase the surface representation itself, for example explicit connected
outer garment / hair surface layers or a learned surface-token decoder. Do not
spend more cycles only increasing triangle steps, weights, or face budgets.

## Connected Surface v2.4 Outer Hair / Clothing Layer

v2.4 implements the first representation-level change after the RGB/gradient
objective: optional connected outer surface layers in the template builder.

New builder arguments:

```text
--outer-layer-parts
--outer-layer-offset
```

The outer layer duplicates selected connected part faces, offsets them along
vertex normals, and welds the duplicate sheet back to the original surface with
boundary side faces. It is carrier geometry only: no VGGT geometry is used and
no teacher/candidate is produced.

Template:

```text
output/normal_line_multiview_20260506/connected_surface_template_v24_outer_hair_clothing_subdiv_facehairhandscloth
```

Carrier counts:

```text
base vertices = 10475
base faces = 20908
hybrid vertices = 39694
hybrid faces = 79872
outer layer parts = head_top_hairline, lower_clothing_proxy
outer layer new vertices = 2798
outer layer duplicated faces = 5580
outer layer stitch faces = 282
subdivision levels = 1
subdivision parts = left_hand, right_hand, head_face, head_top_hairline, lower_clothing_proxy
```

Smoke run:

```text
output/normal_line_multiview_20260506/connected_surface_v24_outer_imageedge_landmark_smoke6_t96_step20
```

Metrics:

```text
initial_iou.mean = 0.7711
optimized_iou.mean = 0.7837
iou_delta = +0.0126
initial_target_recall.mean = 0.9495
optimized_target_recall.mean = 0.9488
target_recall_delta = -0.0008
```

Compared with v2.2, the outer layer gives much higher target coverage and the
largest 2D IoU gain so far in this raw-image surface route. The Open3D result
is also less broken as a full connected body carrier:

```text
output/normal_line_multiview_20260506/connected_surface_v24_outer_imageedge_landmark_smoke6_t96_step20/open3d_review_full/solid_mesh/iso.png
output/normal_line_multiview_20260506/connected_surface_v24_outer_imageedge_landmark_smoke6_t96_step20/open3d_review_full/solid_mesh/face_close.png
output/normal_line_multiview_20260506/connected_surface_v24_outer_imageedge_landmark_smoke6_t96_step20/open3d_review_hands/solid_mesh/iso.png
```

Strict visual conclusion:

```text
v2.4 is still not a mentor-pass teacher. It improves the connected carrier and
begins to express a head/hair and outer-clothing layer, but the surface remains
template-driven. The face is not personalized enough, the hair layer is still a
coarse cap, and hands remain weak/detail-poor. No original-6v strict teacher
gate was run because Open3D visual precheck still fails.
```

Next non-redundant implication:

The first outer-layer representation change is useful and should be preserved,
but it is still hand-built geometry. The next step should not be another
outer-layer offset/weight loop; it should either add a more targeted connected
face/hair/hand surface parameterization or move to a learned surface-token
decoder trained against a strict-passing dense target-frame surface. Cloud
remains blocked.

## v2.4 + Triangle RGB/Gradient Combined Smoke

The v2.4 outer-layer carrier was also tested with the all-face triangle
RGB/gradient objective, to check whether the new geometry carrier and stronger
raw-image objective complement each other.

Run:

```text
output/normal_line_multiview_20260506/connected_surface_v24_outer_triangle_rgbgrad_allfaces_smoke2_t64_step5
```

Key metrics:

```text
triangle color bake coverage = 0.9279
triangle RGB pixels per view = 500.5
initial_iou.mean = 0.7872
optimized_iou.mean = 0.7926
iou_delta = +0.0054
initial_target_recall.mean = 0.9194
optimized_target_recall.mean = 0.9212
target_recall_delta = +0.0019
triangle_rgb_loss = 0.1015
triangle_gradient_loss = 0.3076
```

Open3D review:

```text
output/normal_line_multiview_20260506/connected_surface_v24_outer_triangle_rgbgrad_allfaces_smoke2_t64_step5/open3d_review_full/solid_mesh/iso.png
```

Strict visual conclusion:

```text
The combined objective is valid and gives positive 2D alignment without recall
collapse. But Open3D still shows a template-like body, coarse cap-like hair,
weak hands, and non-personalized face geometry. This is not a strict teacher
and not a candidate. It should not be escalated by simply adding more steps,
larger RGB weights, or larger face budgets.
```

Next non-redundant implication:

The remaining blocker is not merely renderer sparsity or lack of raw RGB
signal. The current connected geometry has no precise semantic correspondence
for face and hand details: landmark losses are computed, but they are broad
Chamfer terms over large vertex sets and therefore weak. Before any more long
optimization runs, audit whether the face/hand landmark losses are pulling the
right connected vertices and add explicit diagnostics or correspondence-aware
constraints if possible.

## Landmark Pull Audit

To test whether the weak MediaPipe terms are actually pulling the intended
connected vertices, a diagnostic-only audit tool was added:

```text
tools/audit_connected_surface_landmark_pull.py
```

It projects the optimized connected mesh into the raw views, runs the same face
and hand landmark detectors, and reports nearest connected surface vertices and
pixel distances. It creates no teacher, no candidate, and no cloud unlock.

Audit run:

```text
output/normal_line_multiview_20260506/connected_surface_v24_outer_triangle_rgbgrad_allfaces_smoke2_t64_step5/landmark_pull_audit
```

Results:

```text
face candidate vertices = 7352
left/right hand vertices = 2564 / 2560
face lm->mesh mean px = 8.83
face lm->mesh p50 px = 3.81
face lm->mesh p90 px = 19.28
face lm->mesh max px = 25.64
hand lm->mesh mean px = 4.65
hand lm->mesh p50 px = 1.74
hand lm->mesh p90 px = 10.31
hand lm->mesh max px = 18.82
hand matches = 9
duplicate-side hand views = 2
```

Representative overlays:

```text
output/normal_line_multiview_20260506/connected_surface_v24_outer_triangle_rgbgrad_allfaces_smoke2_t64_step5/landmark_pull_audit/overlays/view_00_cam00_face_pull.png
output/normal_line_multiview_20260506/connected_surface_v24_outer_triangle_rgbgrad_allfaces_smoke2_t64_step5/landmark_pull_audit/overlays/view_00_cam00_hand0_pull.png
output/normal_line_multiview_20260506/connected_surface_v24_outer_triangle_rgbgrad_allfaces_smoke2_t64_step5/landmark_pull_audit/overlays/view_20_cam20_face_pull.png
```

Conclusion:

```text
The landmark losses have usable local signal in several views, but they are too
broad to guarantee correct detailed geometry. Face pull is good in some views
but fails badly in at least one front view. Hand landmarks are often close to
connected hand vertices, but two views match multiple detected hands to the
same side, so the loss can reinforce ambiguous or wrong hand-side alignment.
This explains why the current landmark terms do not turn the template hand/face
surface into mentor-pass geometry.
```

Next non-redundant implication:

Do not increase landmark weights blindly. The next method change should make
landmark supervision correspondence-aware: enforce per-view left/right hand
assignment uniqueness and split face landmarks into contour / central face /
hairline regions before applying losses. Otherwise stronger weights will only
pull broad vertex sets and can worsen template artifacts.

## Unique-Side Hand Landmark Constraint

The optimizer now supports an optional correspondence-aware hand term:

```text
--hand-landmark-unique-side
```

When multiple hands are detected in one view, this assigns at most one
detection to `left_hand` and at most one to `right_hand` before accumulating the
hand landmark loss. This directly addresses the landmark-pull audit failure
where two views matched multiple detections to the same side.

Smoke run:

```text
output/normal_line_multiview_20260506/connected_surface_v24_outer_uniquehand_landmark_smoke6_t96_step12
```

Results:

```text
initial_iou.mean = 0.7711
optimized_iou.mean = 0.7803
iou_delta = +0.0092
initial_target_recall.mean = 0.9495
optimized_target_recall.mean = 0.9492
target_recall_delta = -0.0004
hand landmark views = 6
hand landmark detections used = 9
unique-side views = 3
duplicate-side fallback views = 0
```

Conclusion:

```text
The unique-side hand constraint fixes a real ambiguity in the loss path and
should remain available. It is still not sufficient for mentor-pass hands:
Open3D-quality hand geometry remains governed by the underlying connected
surface carrier, and the current carrier still produces template/detail-poor
hands. This smoke is diagnostic only and not a candidate.
```

Next non-redundant implication:

The hand-side ambiguity is no longer the main blocker. The next issue is lack of
semantic correspondence inside each hand and face region. A stronger method
would need per-finger / central-face correspondences or a learned surface-token
decoder; simply increasing the unique-side landmark weight is a loop.
