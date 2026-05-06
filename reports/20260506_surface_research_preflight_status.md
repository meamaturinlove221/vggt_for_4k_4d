# Surface Research Preflight Status

Status: `modal_research_preflight_plumbing_complete_not_pass`

This report records the first implementation pass after switching from
handcrafted carrier tuning to the parallel unblocker matrix. It does not claim
mentor success, does not create a teacher, does not create a candidate, and does
not unblock formal cloud train/infer/export.

## Disk Cleanup

Low-risk cleanup was performed before new work:

```text
deleted_items = 52
freed_space = 6.977 GB
skipped = output/_tmp_tests because one zero-value temp dir was permission denied
```

Only corrupted downloads, top-level temporary/probe output folders, old empty
worktree cleanup archives, and `predictions_chunks_*` caches with a sibling
`predictions.npz` were removed.

## Current Gate Truth

```text
strict_candidate_passes = 0
strict_teacher_passes = 0
formal cloud train/infer/export = blocked
research-preflight cloud = allowed only through isolated research entrypoint
```

The existing formal Modal train/infer guards remain intact.

## New Research-Only Entry Point

Added:

```text
modal_surface_research_preflight.py
```

Purpose:

```text
run A-line dense-teacher readiness or B0 surface-token smoke
emit artifacts/reports only
never write strict pass
never export teacher
never export candidate
never call formal VGGT train/infer
```

GPU selection is not hard-coded per lane. The research Modal app uses:

```text
VGGT_MODAL_RESEARCH_GPU
```

and records both expected and actual import-time GPU specs in the launch guard.

## Modal Debug Outcome

The first Modal attempts did not count as successful runs. The server-side
sequence was:

```text
nvdiffrast build failed: missing wheel
nvdiffrast build failed: missing clang++
nvdiffrast build failed: empty CUDA arch list during image build
A-readiness failed: remote scene lacked rgb_cams.smc
ping_scene crash-looped: missing return after ping summary
A-readiness CPU failed: unnecessary torch import via renderer helper
```

Fixes applied:

```text
CUDA devel image
wheel/setuptools/ninja
clang
TORCH_CUDA_ARCH_LIST=8.0
portable camera_params_sidecar.npz
CPU/GPU research lane split
ping_scene early return
A-readiness torch-free parse_view_indices
correct crop-scene intrinsics downscale for 64/96 research sizes
```

After these fixes, isolated research runs completed on Modal.

## A-Line Modal Readiness

Remote output:

```text
output/surface_research_preflight/A_readiness_60v_humancrop_t96_cpu_v3_intrinsics
```

Key result:

```text
status = completed
camera_source = camera_params_sidecar
view_count_total = 60
selected_views = 0,10,20,30,40,50
target_size = 96
all_selected_cameras_available = true
camera_numeric_ok = true
asset_ready_for_research_preflight = true
strict_candidate_passes = 0
strict_teacher_passes = 0
```

This only proves the raw 60-view human-crop RGB/mask/camera assets are portable
and ready for A-line dense reconstruction research. It is not a teacher and not
a candidate.

## B0 Surface-Token Backend Smoke

Added:

```text
tools/optimize_surface_token_backend_b0.py
```

This is intentionally not `image_mlp++`. It builds:

```text
part-aware occupied spatial surface tokens
visibility-aware multi-view RGB mean/variance/support features
part-specific token heads
nvdiffrast mask/depth rendering
photometric variance and rendered depth smoothness proxy losses
```

Local smoke output:

```text
output/surface_research_preflight_local/B0_surface_tokens_t96_step20
```

Run:

```text
target_size = 96
views = 0,10,20,30,40,50
steps = 20
token_grid = 5
token_hidden = 64
```

Key metrics:

```text
avg_initial_iou = 0.7594788682
avg_final_iou = 0.7605119822
avg_iou_delta = +0.0010331140
vertices_with_two_view_support = 34392
mean_support = 4.2675266266
max_vertex_delta = 0.0017279357
mean_vertex_delta = 0.0002690158
```

Open3D review outputs:

```text
output/surface_research_preflight_local/B0_surface_tokens_t96_step20/open3d_review_full
output/surface_research_preflight_local/B0_surface_tokens_t96_step20/open3d_review_head
output/surface_research_preflight_local/B0_surface_tokens_t96_step20/open3d_review_face
output/surface_research_preflight_local/B0_surface_tokens_t96_step20/open3d_review_hands
```

Visual decision:

```text
fail
```

Reason:

```text
This smoke proves the B0 surface-token plumbing runs, but the short local run
does not yet create mentor-level non-template face/hair/hand geometry. Numeric
delta is small and cannot be used as a pass signal.
```

Modal B0 GPU smoke also completed:

```text
output/surface_research_preflight/B0_surface_tokens_t64_step2_gpu
gpu_name = NVIDIA A100-SXM4-40GB
views = 0,30
target_size = 64
steps = 2
token_count = 246
vertices_with_two_view_support = 19611
avg_initial_iou = 0.7658801503
avg_final_iou = 0.7658801503
avg_iou_delta = 0.0
```

This proves the Modal GPU/nvdiffrast B0 execution path runs. It does not prove
geometry improvement.

## A3 Visual Hull Init

Added:

```text
tools/preflight_visual_hull_init.py
```

Remote output:

```text
output/surface_research_preflight/A3_visual_hull_init_t96_g56_s4
```

Key result:

```text
research_only = true
camera_source = camera_params_sidecar
selected_views = 0,10,20,30,40,50
target_size = 96
grid_resolution = 56
grid_points = 175616
support_threshold = 4
occupied_points = 31727
occupied_fraction = 0.1806612154
support_histogram[4] = 9187
support_histogram[5] = 6819
support_histogram[6] = 15721
```

Decision:

```text
A3 visual hull is only a coarse initialization/readiness diagnostic for dense
surface reconstruction. It is not continuous/person-specific enough to be a
teacher and must not enter training or strict pass accounting.
```

## Next Actions

Continue only with the unblocker matrix:

```text
A-line: implement actual dense reconstruction preflights from raw masks/RGB/cameras,
        starting from visual-hull initialization, but export no teacher until
        Open3D full/head/face/hairline/hands and strict teacher gate pass
B-line: stronger learned local surface-token backend, not scalar tuning
C-line: weak landmark/edge/hair/hand constraints only as B-line inputs
D-line: strict gate and Open3D visual review
```

Do not return to:

```text
v6 hidden/step/weight tuning
offset/support/threshold loops
VGGT shell recycling
teacher export from visual-fail meshes
formal cloud train/infer/export while strict passes are zero
```

## Modal Log Audit And A3 Mesh Seed Update

Direct Modal CLI audit confirms that the screenshot-level `stopped` state is not by itself a failure. Earlier apps did contain real server-side failures, and the current branch now records them explicitly:

```text
ap-3Itoiv9BNSrLOo97znokS6: A-readiness failed because CPU lane imported torch through renderer helper
ap-Jz8e0NunSfqa3r4EIFeAYO: ping_scene local entrypoint hit KeyError(output_subdir) after remote write
ap-WOUhqSOrVE1Tje4WX8N4YK / ap-niGcEwOX6Ot6AhHjJvv7Xq: remote import failed because /workspace/vggt was not inserted before tools imports
```

The latest audited research apps completed normally:

```text
ap-RLFc9YeP5uxJl5X2wJGIeM: Stopping app - local entrypoint completed
ap-mzSEBa2ycWirqzkcIVGIZG: Stopping app - local entrypoint completed
```

Volume summaries and downloaded local summaries confirm `status=completed` and `returncode=0` for the B0 GPU smoke and A3 visual-hull runs. These are only research-preflight successes and do not change strict pass accounting.

A3 has now been upgraded from point-only support diagnostics to a continuous marching-cubes mesh seed:

```text
output/surface_research_preflight/A3_visual_hull_mesh_t96_g56_s4
mesh_status = extracted
mesh_vertices = 14208
mesh_faces = 28586
occupied_points = 31727
occupied_fraction = 0.1806612154
```

Decision:

```text
This mesh is a raw-mask visual-hull initialization seed for A-line dense reconstruction. It is still not a strict teacher: it has no proof of person-specific face/hair/hand detail and must not enter teacher-supervised training or pass accounting.
```

## A3 Mesh Projection Self-Check

A3 now emits a connected marching-cubes mesh and per-view projection overlays. The local 3-view smoke validated the renderer path:

```text
output/surface_research_preflight_local/A3_visual_hull_mesh_project_t64_g32_s3
mesh_vertices = 3116
mesh_faces = 6260
mesh_projection_mean_iou = 0.9201100368
mesh_projection_mean_recall = 0.9941385520
mesh_projection_mean_precision = 0.9252155355
```

The 6-view Modal run also completed:

```text
output/surface_research_preflight/A3_visual_hull_mesh_project_t96_g56_s4
status = completed
returncode = 0
mesh_vertices = 14208
mesh_faces = 28586
mesh_projection_mean_iou = 0.6611346666
mesh_projection_mean_recall = 0.9996233214
mesh_projection_mean_precision = 0.6613312293
```

Interpretation:

```text
The visual hull mesh has very high recall but low precision in the 6-view setting,
so it is an over-covering hull. This is useful as a continuous initialization seed
for A-line dense reconstruction, but it is explicitly not a strict teacher and not
a mentor-level surface.
```
