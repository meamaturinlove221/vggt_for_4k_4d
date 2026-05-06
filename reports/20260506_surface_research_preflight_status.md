# Surface Research Preflight Status

Status: `research_preflight_local_b0_smoke_complete_not_pass`

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

## Next Actions

Continue only with the unblocker matrix:

```text
A-line: dense teacher reconstruction readiness and research-preflight
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
