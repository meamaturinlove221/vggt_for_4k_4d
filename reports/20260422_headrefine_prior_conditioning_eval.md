# 2026-04-22 Head-Refined Prior Conditioning Eval

## Bottom line

The `detail_normal_refiner -> scene prior_maps -> sparse-view inference` path is now technically runnable for a one-case experiment, but the current result is **near-zero gain**.

- The offline stitch-back tool works.
- The patched 6-view scene runs through the existing prior-enabled smoke checkpoint.
- The final geometry is almost unchanged relative to the same checkpoint on the original scene.

So the current truthful read is:

> `detail_normal_refiner` is a valid local normal-improvement module, but simply patching its refined head ROI normals back into dense `prior_maps` is not enough, by itself, to materially improve the final 6-view point cloud.

## What was added

- stitch-back tool:
  - [patch_scene_prior_with_refined_normals.py](D:/vggt/vggt-main/tools/patch_scene_prior_with_refined_normals.py)

This tool:

- loads a trained `detail_normal_refiner` checkpoint
- applies it to an ROI dataset export with `roi_box_xyxy` and `view_index`
- copies the source scene directory
- replaces only the dense `smplx_cam_nx/ny/nz` channels inside `prior_maps.npz`
- keeps summary tokens untouched

This is intentionally a one-case offline conditioning experiment, not a trainer-wide integration.

## Patch run

- checkpoint:
  - [best_model.pt](D:/vggt/vggt-main/output/detail_normal_refiner_20260421/remote_head_60to6v_e50/best_model.pt)
- ROI dataset:
  - [head_samples.npz](D:/vggt/vggt-main/output/detail_normal_refiner_20260421/dataset_export_6v_teacher60/head_roi/head_samples.npz)
- source scene:
  - [0012_11_frame0000_6views_sparseproto](D:/vggt/vggt-main/output/4k4d_scenes/0012_11_frame0000_6views_sparseproto)
- patched scene:
  - [0012_11_frame0000_6views_sparseproto_headrefine](D:/vggt/vggt-main/output/detail_normal_refiner_20260422/0012_11_frame0000_6views_sparseproto_headrefine)
- patch summary:
  - [refined_prior_patch_summary.json](D:/vggt/vggt-main/output/detail_normal_refiner_20260422/0012_11_frame0000_6views_sparseproto_headrefine/refined_prior_patch_summary.json)

Patch statistics:

- `6` ROI samples patched
- changed pixels per view: `4,977` to `7,434`
- mean per-pixel normal delta inside patched ROI: about `0.338` to `0.389`

This confirms the scene prior was genuinely changed, not left untouched.

## Inference comparison

Checkpoint used for both runs:

- `vggt_4k4d_train/20260421_6view_focus_smoke_r1/inference_model.pt`

Runs:

- baseline:
  - [20260421_6views_sparseproto_from_6view_focus_smoke_r1](D:/vggt/vggt-main/output/modal_results/20260421_6views_sparseproto_from_6view_focus_smoke_r1)
- head-refined prior:
  - [20260422_6v_focussmoke_headrefine_compare](D:/vggt/vggt-main/output/modal_results/20260422_6v_focussmoke_headrefine_compare)

Direct masked comparison:

- compare summary:
  - [direct_compare_summary.json](D:/vggt/vggt-main/output/modal_results/20260422_6v_headrefine_vs_smokebaseline_compare/direct_compare_summary.json)

Key numbers:

| Metric | Value |
| --- | ---: |
| compare pixel count | `68,130` |
| depth MAE | `0.00115` |
| world-point L2 mean | `0.00142` |
| normal angle mean | `0.013 deg` |
| normal angle p95 | `0.048 deg` |
| translation L2 mean | `0.00425` |

These are all tiny deltas. The final prediction barely changes.

## Open3D comparison

- baseline vs patched Open3D renders:
  - [headrefine_compare_open3d](D:/vggt/vggt-main/output/detail_normal_refiner_20260422/headrefine_compare_open3d)
- face close comparison:
  - [compare_face_close_baseline_vs_headrefine.png](D:/vggt/vggt-main/output/detail_normal_refiner_20260422/headrefine_compare_open3d/compare_panels/compare_face_close_baseline_vs_headrefine.png)
- head close comparison:
  - [compare_head_close_baseline_vs_headrefine.png](D:/vggt/vggt-main/output/detail_normal_refiner_20260422/headrefine_compare_open3d/compare_panels/compare_head_close_baseline_vs_headrefine.png)

ROI counts:

| Variant | Full-body points | Head ROI points | Face ROI points |
| --- | ---: | ---: | ---: |
| baseline smoke | `40,878` | `8,993` | `4,018` |
| head-refined prior | `40,878` | `8,993` | `4,014` |

The Open3D close-up panels also show almost no usable gain.

## Interpretation

What this experiment proves:

- we can safely patch refined ROI normals back into the dense scene prior bundle
- the current `detail_normal_refiner` output is not being ignored at the file level
- the sparse-view backbone still does not convert that local prior change into a meaningful geometry gain

What it does **not** prove:

- that `detail_normal_refiner` is useless
- that normal refinement is the wrong direction

More likely, it means the current influence path is still too weak:

- summary tokens stay unchanged
- `prior_normals` supervision stays mesh-derived
- the backbone only sees a local dense-channel perturbation, not a stronger coarse-to-fine geometry objective

## Current decision

Keep:

- `coarse prior normal + ROI-first detail_normal_refiner` as the strongest normal-design direction
- the new stitch-back tool as a valid one-case research utility

Do not claim yet:

- that patched refined normals already improve final sparse-view point clouds
- that the refiner has been successfully integrated into the geometry main line
