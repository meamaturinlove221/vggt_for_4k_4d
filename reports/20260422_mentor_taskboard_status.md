# 2026-04-22 Mentor Taskboard Status

## Overall

Current normal evidence is `SMPL-X view-aligned coarse prior normal`; the next method step is local high-resolution detail refinement. The ROI, crop, sparse-view, Open3D, overfit, and targetpatch sections below are taskboard/supplement evidence, not the main mentor-facing conclusion.

This project is **not at the mentor-final bar yet**. As of April 22, 2026, the work is in a stronger intermediate state:

- `face ROI / head ROI` visualization is now a real evaluation path, not just a suggestion.
- `full / human_crop / human_crop_hardmask / human_crop_softmatte` preprocessing ablation is now runnable, cloud-inferred, downloaded, compared, and partially integrated into training-case prep.
- `human_crop` is no longer only a preprocessing ablation; it now has completed `focus6` and `family` crop-trained cloud checkpoints.
- `coarse prior normal + ROI-first detail_normal_refiner` is the current viable geometry-improvement direction.
- Even the stronger `projected summary-token targetpatch` integration is still below the 6-view smoke baseline on `face ROI`.
- The original global sparse-view `normal head` still does **not** meet the mentor target.

Latest crop-mainline ROI deltas on the same `6-view human_crop` scene:

| Variant | Full-body points | Head ROI points | Face ROI points |
| --- | ---: | ---: | ---: |
| `crop baseline` | `111,082` | `24,438` | `9,634` |
| `crop focus6 train` | `111,078` | `24,437` | `11,774` |
| `crop family train` | `111,078` | `24,437` | `11,913` |

This is real progress, but still not enough to close the mentor-final claim without stronger qualitative face/head evidence.

## 1. Face ROI / Head ROI

Status: **completed as an evaluation and visualization path**

Implemented:

- Open3D point-cloud rendering for `full`, `head`, and `face` ROI:
  - [render_open3d_pointcloud.py](D:/vggt/vggt-main/tools/render_open3d_pointcloud.py)
- ROI renders for preprocessing ablation:
  - [compare_head_roi_variants.png](D:/vggt/vggt-main/output/preprocess_ablation_20260421/open3d_compare/compare_head_roi_variants.png)
  - [compare_face_roi_variants.png](D:/vggt/vggt-main/output/preprocess_ablation_20260421/open3d_compare/compare_face_roi_variants.png)
  - [compare_fullbody_faceclose_variants.png](D:/vggt/vggt-main/output/preprocess_ablation_20260421/open3d_compare/compare_fullbody_faceclose_variants.png)

Current ROI point counts after confidence filtering:

| Variant | Full-body points | Head ROI points | Face ROI points |
| --- | ---: | ---: | ---: |
| `full` | 40,882 | 8,994 | 4,177 |
| `human_crop` | 111,094 | 24,441 | 11,523 |
| `human_crop_hardmask` | 111,078 | 24,437 | 10,712 |
| `human_crop_softmatte` | 151,734 | 33,382 | 15,127 |

Interpretation:

- `ROI` has become the main acceptance view for head/face geometry.
- `human_crop` already gives a large sparse-view gain over `full`.
- `human_crop_softmatte` currently produces the densest head/face ROI clouds in this 6-view case.

## 2. Crop / Segmentation / Matting

Status:

- `human_crop`: **completed**
- `human_crop_hardmask`: **completed**
- `human_crop_softmatte`: **completed as a first-pass matting variant**
- separate learned segmentation model: **not added**

Implemented:

- Variant generator:
  - [build_preprocessed_scene_variants.py](D:/vggt/vggt-main/tools/build_preprocessed_scene_variants.py)
- Comparison utility:
  - [summarize_preprocess_ablation.py](D:/vggt/vggt-main/tools/summarize_preprocess_ablation.py)

Generated scene variants:

- [0012_11_frame0000_6views_sparseproto_full](D:/vggt/vggt-main/output/preprocess_ablation_20260421/0012_11_frame0000_6views_sparseproto_full)
- [0012_11_frame0000_6views_sparseproto_human_crop](D:/vggt/vggt-main/output/preprocess_ablation_20260421/0012_11_frame0000_6views_sparseproto_human_crop)
- [0012_11_frame0000_6views_sparseproto_human_crop_hardmask](D:/vggt/vggt-main/output/preprocess_ablation_20260421/0012_11_frame0000_6views_sparseproto_human_crop_hardmask)
- [0012_11_frame0000_6views_sparseproto_human_crop_softmatte](D:/vggt/vggt-main/output/preprocess_ablation_20260421/0012_11_frame0000_6views_sparseproto_human_crop_softmatte)

Cloud inference outputs:

- [20260421_6views_preprocess_full_b40](D:/vggt/vggt-main/output/modal_results/20260421_6views_preprocess_full_b40)
- [20260421_6views_preprocess_crop_b40](D:/vggt/vggt-main/output/modal_results/20260421_6views_preprocess_crop_b40)
- [20260421_6views_preprocess_crop_hardmask_b40](D:/vggt/vggt-main/output/modal_results/20260421_6views_preprocess_crop_hardmask_b40)
- [20260421_6views_preprocess_crop_softmatte_v2_b40](D:/vggt/vggt-main/output/modal_results/20260421_6views_preprocess_crop_softmatte_v2_b40)

Comparison outputs:

- [20260421_6views_preprocess_ablation_compare_b40](D:/vggt/vggt-main/output/modal_results/20260421_6views_preprocess_ablation_compare_b40)
- [20260421_6views_preprocess_ablation_compare_b40_softmatte](D:/vggt/vggt-main/output/modal_results/20260421_6views_preprocess_ablation_compare_b40_softmatte)
- [20260421_6views_preprocess_ablation_comparison.md](D:/vggt/vggt-main/reports/20260421_6views_preprocess_ablation_comparison.md)

Relative-to-`full` aligned deltas on the same 6-view case:

| Variant vs `full` | Depth MAE | Mean world-point L2 | Mean normal angle (deg) | Mean translation L2 |
| --- | ---: | ---: | ---: | ---: |
| `human_crop` | 0.0400 | 0.0453 | 0.3762 | 0.0554 |
| `human_crop_hardmask` | 0.0481 | 0.0504 | 0.4963 | 0.0966 |
| `human_crop_softmatte` | 0.0586 | 0.0480 | 0.4832 | 0.1160 |

Interpretation:

- `human_crop` is still the most stable and least disruptive preprocessing variant relative to `full`.
- `human_crop_hardmask` improves occupancy, but shifts camera/geometry more than `human_crop`.
- `human_crop_softmatte` gives the largest ROI point density in this case, but its global aligned stability is currently worse than plain `human_crop`.

## 3. Sparse-View Protocol

Status: **completed as a protocol baseline, not yet closed as a final benchmark**

Existing subset outputs:

- [20260420_local_subsets](D:/vggt/runs/sparse_preprocess/20260420_local_subsets)
- [0012_11_frame0000_60views_subset_summary.json](D:/vggt/runs/sparse_preprocess/20260420_local_subsets/0012_11_frame0000_60views_subset_summary.json)

Supported sparse counts already exercised in the code/results tree:

- `6`
- `7`
- `8`
- `12`
- `13`
- `20`
- `60`

Related script:

- [run_sparse_view_protocol.py](D:/vggt/vggt-main/tools/run_sparse_view_protocol.py)

What is still missing:

- A mentor-ready unified sparse-view benchmark sheet that ties `baseline / crop / normal / refinement` together under the same metric panel.

## 4. Open3D Visualization Chain

Status: **completed**

Implemented:

- [open3d_view_pointcloud.py](D:/vggt/vggt-main/tools/open3d_view_pointcloud.py)
- [render_open3d_pointcloud.py](D:/vggt/vggt-main/tools/render_open3d_pointcloud.py)

This now covers:

- full-body point cloud renders
- head ROI renders
- face ROI renders
- per-variant comparison panels

This task is no longer blocked on MeshLab.

## 5. Normal Direction

Status: **partially completed, not mentor-final**

Completed:

- `coarse prior normal` chain established
- advisor pack / coarse-prior reporting path established
- ROI-first `detail_normal_refiner` trained on cloud and downloaded to local

Key status report:

- [20260421_sparse_view_detail_normal_status.md](D:/vggt/vggt-main/reports/20260421_sparse_view_detail_normal_status.md)

Refiner training/eval outputs:

- [remote_head_60to6v_e50](D:/vggt/vggt-main/output/detail_normal_refiner_20260421/remote_head_60to6v_e50)
- [remote_face_60to6v_e50](D:/vggt/vggt-main/output/detail_normal_refiner_20260421/remote_face_60to6v_e50)
- [eval_head_6v_teacher60](D:/vggt/vggt-main/output/detail_normal_refiner_20260421/eval_head_6v_teacher60)
- [eval_head_12v_teacher60](D:/vggt/vggt-main/output/detail_normal_refiner_20260421/eval_head_12v_teacher60)
- [eval_head_20v_teacher60](D:/vggt/vggt-main/output/detail_normal_refiner_20260421/eval_head_20v_teacher60)

Current truthful reading:

- Global sparse-view end-to-end `normal head` is still below target.
- `coarse prior normal + local ROI refinement` is the currently validated path.
- This is still **not** the mentor-final “6-view face point cloud quality is competitive” endpoint.

## 5.5 Refined-Normal Stitch-Back To Scene Prior

Status: **completed as a one-case conditioning experiment, but not yet effective**

Implemented:

- [patch_scene_prior_with_refined_normals.py](D:/vggt/vggt-main/tools/patch_scene_prior_with_refined_normals.py)

One-case patched scene:

- [0012_11_frame0000_6views_sparseproto_headrefine](D:/vggt/vggt-main/output/detail_normal_refiner_20260422/0012_11_frame0000_6views_sparseproto_headrefine)

Evaluation report:

- [20260422_headrefine_prior_conditioning_eval.md](D:/vggt/vggt-main/reports/20260422_headrefine_prior_conditioning_eval.md)

What this proves:

- We can now take `detail_normal_refiner` outputs and stitch them back into scene-level `prior_maps.npz`.
- The patched scene runs through the current 30-channel prior-enabled smoke checkpoint without breaking the inference path.

What it does **not** prove:

- That the refined normals already improve final sparse-view geometry.

Observed result:

- Direct masked compare against the same smoke checkpoint on the original scene is near-zero:
  - `depth_mae_fg = 0.00115`
  - `world_points_l2_mean_fg = 0.00142`
  - `normal_angle_mean_deg_fg = 0.013`
- Open3D ROI counts are essentially unchanged:
  - baseline smoke: full `40,878`, head `8,993`, face `4,018`
  - head-refined prior: full `40,878`, head `8,993`, face `4,014`

Interpretation:

- The `refiner -> dense prior normal channels` path is technically viable.
- But in the current setup, changing only dense normal channels while leaving summary tokens and mesh-derived `prior_normals` targets unchanged is too weak to move the final point cloud.

## 5.6 Projected Summary-Token TargetPatch

Status: **completed and still negative**

Implemented:

- projected token-sample update inside:
  - [patch_training_case_with_refined_normals.py](D:/vggt/vggt-main/tools/patch_training_case_with_refined_normals.py)
  - [patch_scene_prior_with_refined_normals.py](D:/vggt/vggt-main/tools/patch_scene_prior_with_refined_normals.py)

Evaluation report:

- [20260422_projected_targetpatch_eval.md](D:/vggt/vggt-main/reports/20260422_projected_targetpatch_eval.md)

What changed relative to `5.5`:

- patched `inputs.prior_maps`
- patched `targets.prior_normals`
- patched `prior_summary_tokens` by projecting token positions back into the ROI and sampling refined normals there

Completed 6-view runs:

- teacher upper bound:
  - remote checkpoint: `20260422_6view_headteacherproj_targetpatch_r1/inference_model.pt`
  - remote inference: `vggt_4k4d_infer/20260422_6views_headteacherproj_targetpatch_infer_r1`
- refined deployable path:
  - remote checkpoint: `20260422_6view_headrefineproj_targetpatch_r1/inference_model.pt`
  - remote inference: `vggt_4k4d_infer/20260422_6views_headrefineproj_targetpatch_infer_r1`

ROI counts:

| Variant | Full-body points | Head ROI points | Face ROI points |
| --- | ---: | ---: | ---: |
| smoke baseline | `40,878` | `8,993` | `4,018` |
| teacher projected targetpatch | `40,883` | `8,995` | `3,546` |
| refined projected targetpatch | `40,892` | `8,997` | `3,852` |

Interpretation:

- This is stronger than the earlier dense-only stitch-back path.
- `refined projected targetpatch` is better than the earlier `3,637` refined targetpatch result, but it still does not beat the smoke baseline.
- `teacher projected targetpatch` is also below the smoke baseline, so this path is not currently showing a useful upper bound either.

Current decision:

- Keep the tooling because the integration path is now technically closed.
- Do **not** promote projected targetpatch as a quality win.
- Treat this as a stronger negative result: the current detail-normal integration still does not lift final 6-view face geometry above the baseline.

## 6. SMPL-X Pose Source

Status: **partially completed**

What is completed:

- Dataset-stage pose prior is fully available from 4K4D / DNA-style annotations.
- External SMPL-X parameter import/normalization utility exists:
  - [import_external_smplx_params.py](D:/vggt/vggt-main/tools/import_external_smplx_params.py)

What is not completed:

- A fully integrated real-data SMPL-X regressor/fitter pipeline is **not yet validated end-to-end** in the main sparse-view experiments.
- So the mentor requirement “real capture later can obtain SMPL pose automatically” is only partially addressed today.

## 7. Training-Case Integration

Status: **completed for preprocess variants**

Implemented:

- [prepare_4k4d_prior_training_case.py](D:/vggt/vggt-main/tools/prepare_4k4d_prior_training_case.py)

Verified outputs:

- [0012_11_frame0000_6views_preprocess_full_b40](D:/vggt/vggt-main/output/training_cases/0012_11_frame0000_6views_preprocess_full_b40)
- [0012_11_frame0000_6views_preprocess_crop_b40](D:/vggt/vggt-main/output/training_cases/0012_11_frame0000_6views_preprocess_crop_b40)
- [0012_11_frame0000_6views_preprocess_crop_hardmask_b40](D:/vggt/vggt-main/output/training_cases/0012_11_frame0000_6views_preprocess_crop_hardmask_b40)
- [0012_11_frame0000_6views_preprocess_crop_softmatte_v2_b40](D:/vggt/vggt-main/output/training_cases/0012_11_frame0000_6views_preprocess_crop_softmatte_v2_b40)
- [0012_11_frame0000_6views_sparseproto_recheck_b40](D:/vggt/vggt-main/output/training_cases/0012_11_frame0000_6views_sparseproto_recheck_b40)

Important behavior now verified:

- preprocess-variant scenes load `scene-local prior_maps.npz`
- preprocess variants skip misaligned geometry-prior regeneration
- original non-variant scenes still keep real-camera SMPL-X geometry-prior behavior

## 7.5 Single-Case Overfit Eval On Preprocess Variants

Status: **completed and negative**

Cloud runs completed:

- trained crop checkpoint:
  - remote checkpoint: `vggt_4k4d_train/20260422_6view_singlecase_crop_b40_a10080_r1/inference_model.pt`
  - local eval: [20260422_crop_trained_eval](D:/vggt/vggt-main/output/modal_results/20260422_crop_trained_eval)
- trained softmatte checkpoint:
  - remote checkpoint: `vggt_4k4d_train/20260422_6view_singlecase_softmatte_b40_a10080_r1/inference_model.pt`
  - local eval: [20260422_softmatte_trained_eval](D:/vggt/vggt-main/output/modal_results/20260422_softmatte_trained_eval)

Trained-vs-untrained comparisons:

- crop:
  - [20260422_crop_trained_vs_untrained_compare](D:/vggt/vggt-main/output/modal_results/20260422_crop_trained_vs_untrained_compare)
- softmatte:
  - [20260422_softmatte_trained_vs_untrained_compare](D:/vggt/vggt-main/output/modal_results/20260422_softmatte_trained_vs_untrained_compare)

New Open3D renders:

- crop trained:
  - [crop_trained](D:/vggt/vggt-main/output/overfit_trained_eval_20260422/open3d_compare/crop_trained)
- softmatte trained:
  - [softmatte_trained](D:/vggt/vggt-main/output/overfit_trained_eval_20260422/open3d_compare/softmatte_trained)

Observed read:

- `crop` trained did **not** improve sparse-view face ROI quality.
- `softmatte` trained did **not** improve sparse-view face ROI quality.
- Both trained checkpoints keep roughly the same full/head occupancy as the untrained variants, but the `face ROI` becomes less usable.

ROI point counts:

| Variant | Full-body points | Head ROI points | Face ROI points |
| --- | ---: | ---: | ---: |
| `crop` untrained | 111,094 | 24,441 | 11,523 |
| `crop` trained | 111,078 | 24,437 | 10,952 |
| `softmatte` untrained | 151,734 | 33,382 | 15,127 |
| `softmatte` trained | 151,726 | 33,380 | 14,068 |

Qualitative failure mode:

- In `face_close` Open3D views, both trained checkpoints collapse toward a flatter, silhouette-like slab instead of a clearer 3D facial surface.
- This means the current `single-case preprocess overfit` line is **not** a mentor-ready improvement line.
- It should be treated as an internal negative result, not as the next main method branch.

## 8. What Is Still Missing For Mentor-Final

The following items are **not complete yet**:

1. A final sparse-view result that clearly meets the mentor bar on `6-view` face/head point-cloud quality.
2. End-to-end proof that refined normal cues feed back into the sparse geometry branch and raise final point-cloud quality, not only normal visualization quality.
3. A validated real-data SMPL-X regression/fitting path plugged into the main inference/training loop.
4. A consolidated mentor-ready benchmark page that compares:
   - original sparse baseline
   - crop
   - crop + softmatte
   - coarse prior normal
   - detail refinement
   - final point cloud quality on face/head ROI
5. A replacement for the negative `single-case preprocess overfit` result, since that line currently regresses `face ROI` geometry instead of improving it.
6. A replacement for the now-negative `projected summary-token targetpatch` line, since even the stronger teacher/refined targetpatch variants remain below the smoke baseline on `face ROI`.

## 9. Current Best Practical Read

If we had to choose the strongest next-step stack **today**, it would be:

1. `human_crop` as the default stable sparse-view preprocessing.
2. `human_crop_softmatte` as a promising high-occupancy branch to keep testing, but not yet the default.
3. `coarse prior normal + ROI-first detail_normal_refiner` as the main normal/geometry direction.
4. `Open3D head/face ROI` as the hard visualization gate for acceptance.

That is meaningful progress, but it is still one step short of the mentor-final version.
