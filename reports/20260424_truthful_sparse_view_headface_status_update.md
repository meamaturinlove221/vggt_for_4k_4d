# 2026-04-24 Truthful Sparse-View Head/Face Status
## Verdict
- Mentor-final bar is **not reached**. This report intentionally does not mark the work as passed.
- Quantitative gains on the targetcam30 front-face protocol do **not** transfer to the original `6views_sparseproto_headshoulder_crop` protocol.
- Visual evidence remains below the requested HumanRAM/PSHuman-like face/head point quality: face regions are still blocky, holey, or multi-view inconsistent.

## ROI Summary
| Protocol | Run | Face ROI points | Truthful interpretation |
|---|---:|---:|---|
| `original_6v_headshoulder` | `signfix_ckpt4_ref_best` | 16825 | current reference best; visual still not mentor-final |
| `original_6v_headshoulder` | `headshoulder6v_r3_ckpt0` | 16608 | below reference |
| `original_6v_headshoulder` | `headshoulder6v_r3_ckpt1_or_inference` | 16525 | below reference |
| `original_6v_headshoulder` | `teachergeom_roi_combo_from_ckpt4_ckpt0` | 16336 | below reference |
| `original_6v_headshoulder` | `targetcam30_signfix_teacher_roi_masks_e3` | 13804 | does not generalize to original protocol |
| `original_6v_headshoulder` | `targetcam30_continue_e6` | 14626 | still below reference |
| `targetcam30_raw_headface` | `ckpt4` | 25020 | front-face target protocol; better visual interpretability but still blocky/holes |
| `targetcam30_raw_headface` | `roi_masks_from60vteacher_e3` | 26283 | higher count; Open3D/3D visual still sparse/holes |
| `targetcam30_raw_headface` | `signfix_teacher_roi_masks_e3` | 26832 | highest targetcam30 count; not mentor-final visual |
| `targetcam30_raw_headface` | `continue_e6` | 23143 | regressed after longer low-lr continuation |
| `targetcam30_raw_headface` | `60v_refined_prior_ckpt4` | 25051 | refined-normal prior patch did not improve geometry |
| `targetcam30_raw_headface` | `normalbae_refined_prior_ckpt4` | 25049 | external NormalBae aligned refiner did not improve geometry |

## New Work Completed This Session
- Verified ROI masks are consumed by the dataset/loss path; training logs showed `prior_normal_weight_mean=7.0477`, confirming ROI weighting was active.
- Ran targetcam30 ROI-mask cloud training from current best ckpt4 with both 60v-teacher and signfix-teacher cases. Best targetcam30 face ROI reached `26832`, but visual close-ups did not pass.
- Ran low-lr continuation from the targetcam30 ROI-mask checkpoint; it regressed to `23143` on targetcam30 and `14626` on original 6v headshoulder.
- Completed the interrupted external SMPL-X bundle bridge smoke: `output\smoke_external_bundle_case\scene_with_external_prior_bridge` successfully entered Modal prior-enabled inference with ckpt4.
- Built and tested `modal_external_normal_teacher.py` for cloud NormalBae external normal teacher generation.
- Exported NormalBae head/face ROI teacher data, found and fixed a coordinate sign mismatch by negating xyz, trained aligned head/face `detail_normal_refiner` models, and patched scene priors. The patched priors still did not improve 6v geometry (`25049` face ROI, visually same failure mode).

## Visual Evidence Paths
- camera projection face compare: `D:/vggt/vggt-main/output/detail_normal_refiner_20260424/camera_projected_pointcloud_compare/camera_projected_face_compare.png`
- target-view-only face compare: `D:/vggt/vggt-main/output/detail_normal_refiner_20260424/camera_projected_pointcloud_compare_target_view_only/target_view_only_face_compare.png`
- consistency filter face compare: `D:/vggt/vggt-main/output/detail_normal_refiner_20260424/target_camera_consistency_filter/target_consistency_filtered_face_compare.png`
- conf ablation p20: `D:/vggt/vggt-main/output/detail_normal_refiner_20260424/target_view_conf_ablation/target_view_face_confp20.png`
- NormalBae teacher: `D:/vggt/vggt-main/output/detail_normal_refiner_20260424/external_normal_teacher/targetcam30_6v_normalbae/00_00_tgt_cam30_normalbae.png`
- aligned NormalBae face refiner strip: `D:/vggt/vggt-main/output/detail_normal_refiner_20260424/runs/targetcam30_face_normalbae_aligned_e120/best_train/visuals/00_00_tgt_cam30_summary_strip.png`

## Diagnosis
- Target-view RGB/low-threshold projections can look face-complete, but that is essentially 2.5D target-view visibility, not validated 3D multi-view geometry.
- Camera-projected multi-view clouds show blocky seams and face/hair ghosting. Consistency filtering removes many inconsistent points but tears the face apart, proving the problem is not just visualization.
- 60v teacher and NormalBae teacher are both useful engineering links, but neither currently supplies a high-quality, geometry-consistent face-detail teacher for sparse-view VGGT training.
- Summary-token/projected-targetpatch remains a rejected path and was not used as a mainline result.

## Completed Checklist Items
- Coarse-prior advisor pack was already converted to the truthful `SMPL-X view-aligned coarse prior normal` wording, with failed 4v probe isolated as `failed_predicted_normal_probe`.
- Design mouthpiece: `detail_normal_refiner`, alias `pifuhd_style_normal_refine`, is image-aligned residual refinement on `RGB crop + coarse prior normal crop + human mask`; it does not replace VGGT.
- External bundle -> scene-level `prior_maps.npz` -> Modal prior-enabled inference bridge is now end-to-end smoke-tested.
- Open3D/projection visual evidence exists for head/face ROI; projection-only fallback avoids Open3D crashes but is not used to claim success.
- `detail_normal_refiner` branch is operational: RGB crop + coarse prior normal crop + mask -> refined normal/residual, with cosine/edge/mask/ROI metrics and visual strips.

## Remaining Non-Pass Items
- Original `6views_sparseproto_headshoulder_crop` face ROI did not exceed the `16825` signfix ckpt4 reference in any new training that also preserved visual quality.
- targetcam30 protocol can produce higher counts (`26832`) but still lacks clean eyes/nose/facial surface detail and does not transfer to original protocol.
- External normal teacher is not yet a high-resolution geometry teacher; NormalBae is smooth and image-aligned, not PIFuHD/PSHuman-level detailed surface geometry.

## Next Route
- Do not continue current same-checkpoint micro-tuning without a better teacher.
- Next credible step is to build a stronger high-quality pseudo-GT teacher: either multi-view local surface fitting in head/face ROI, or a stronger external normal/depth/mesh estimator with verified coordinate alignment and visible-region masking.
- Only after a teacher shows clean head/face ROI normal/geometry on one frame should it be used for VGGT sparse-view training.

## 2026-04-25 Addendum: Multi-View Surface Teacher Attempt
### Verdict
- Mentor-final bar is still **not reached**. The new surface-teacher experiments produced either lower ROI counts or higher counts with collapsed confidence/poorer Open3D visuals.
- The result must remain a negative/diagnostic branch, not a pass claim.

### New Tools / Artifacts
- Added `tools/build_multiview_surface_prior_teacher.py` to fuse high-confidence 60v VGGT points into a local Open3D/PCA surface teacher and patch sparse-scene head/face prior positions/normals.
- Added `tools/build_surface_teacher_training_case.py` to convert that surface teacher into ROI target supervision (`depths`, `cam_points`, `world_points`, `teacher_normals`, ROI masks) for 6v training.
- Targetcam30 surface prior scene: `output/4k4d_preprocessed_scene_variants/0012_11_frame0000_6views_targetcam30_multiview_surface_head_face_prior`.
- Original-protocol surface prior scene: `output/4k4d_preprocessed_scene_variants/0012_11_frame0000_6views_headshoulder_multiview_surface_head_face_prior`.
- Original-protocol surface-teacher training case: `output/training_cases/0012_11_frame0000_6views_headshoulder_multiview_surface_teacher_r1`.

### ROI Results
| Protocol | Run | Face ROI points | Truthful interpretation |
|---|---:|---:|---|
| `targetcam30_raw_headface` | `multiview_surface_prior_ckpt4` | 23526 | lower than ckpt4/ROI-mask references |
| `targetcam30_raw_headface` | `multiview_surface_prior_denseonly_ckpt4` | 26193 | below `signfix_teacher_roi_masks_e3=26832`; visual still not pass |
| `original_6v_headshoulder` | `multiview_surface_prior_ckpt4` | 16785 | below `signfix_ckpt4_ref_best=16825` |
| `original_6v_headshoulder` | `multiview_surface_prior_denseonly_ckpt4` | 15658 | below reference |
| `original_6v_headshoulder` | `surface_teacher_e2_inference` | 26156 | numeric pseudo-gain, but confidence threshold collapsed to `1.0` and Open3D visual is worse; must be treated as pseudo-positive |

### Visual Evidence
- Surface prior diagnostic sheet: `D:/vggt/vggt-main/output/detail_normal_refiner_20260425/open3d_surface_prior_diagnostics/surface_prior_truthful_evidence_sheet.png`.
- Surface-teacher training comparison: `D:/vggt/vggt-main/output/detail_normal_refiner_20260425/open3d_surface_teacher_training_e2/signfix_vs_surface_train_e2_sheet.png`.
- The `surface_teacher_e2` close-up shows smeared/noisy head-face geometry despite higher ROI count, so it fails the “quantitative and visual both true” rule.

### Diagnosis Update
- Directly patching 60v-fused surface prior into inference does not improve original 6v sparse-view geometry.
- Removing stale summary tokens (`denseonly`) does not fix the failure.
- Training for two epochs on surface-teacher ROI targets increases point count only by lowering/collapsing confidence; it worsens face/head close-up structure.
- Current bottleneck remains teacher quality and geometry consistency, not just ROI weighting, summary-token conflicts, or Open3D visualization.

### Next Route
- Stop expanding this surface-teacher branch unless the teacher itself is upgraded and visually validated first.
- A credible next attempt requires a genuinely higher-quality human-specific teacher (PIFuHD/PSHuman-style mesh/normal estimator or robust multi-view TSDF/mesh fit with clean head/face details) before any further VGGT sparse-view training.

## 2026-04-25 Addendum: Sapiens Human Normal Teacher Attempt
### Verdict
- Mentor-final bar is still **not reached**.
- Sapiens 0.3B produces a visibly sharper human normal map than NormalBae on the targetcam30 head/face crop, but patching the refined normal prior into VGGT inference did **not** improve sparse-view geometry.

### New Tools / Artifacts
- Added `modal_sapiens_normal_teacher.py` for Modal-based Sapiens normal inference using `facebook/sapiens-normal-0.3b-torchscript`.
- Added `tools/export_external_normal_refiner_dataset.py` to convert external normal maps into the existing `detail_normal_refiner` ROI dataset format.
- Sapiens teacher output: `output/detail_normal_refiner_20260425/external_sapiens_normal_teacher/targetcam30_6v_sapiens03b`.
- Sapiens ROI dataset: `output/detail_normal_refiner_20260425/datasets/targetcam30_6v_sapiens_teacher_flipyz`.
- Sapiens refiner runs:
  - `output/detail_normal_refiner_20260425/runs/targetcam30_face_sapiens_flipyz_e160`
  - `output/detail_normal_refiner_20260425/runs/targetcam30_head_sapiens_flipyz_e160`
- Patched scene: `output/4k4d_preprocessed_scene_variants/0012_11_frame0000_6views_targetcam30_sapiens_head_face_refined_prior`.

### Teacher Quality Check
- Visual compare sheet: `D:/vggt/vggt-main/output/detail_normal_refiner_20260425/external_sapiens_normal_teacher/targetcam30_6v_sapiens03b/sapiens_teacher_compare_sheet.png`.
- The best coordinate transform against the coarse prior was `flip-yz` with mean dot `0.8033` and positive-dot fraction `0.9986`.
- Sapiens captures clearer face/hair/head normal structure than NormalBae, so it is a better detail-normal teacher candidate.

### Geometry Result
| Protocol | Run | Face ROI points | Truthful interpretation |
|---|---:|---:|---|
| `targetcam30_raw_headface` | `sapiens_refined_prior_ckpt4` | 25007 | slightly below ckpt4 targetcam30 reference (`25020`) and far below `signfix_teacher_roi_masks_e3=26832`; no geometry win |

### Diagnosis Update
- Better 2D human normal quality alone is insufficient when only the dense normal prior is patched at inference time.
- This confirms the next required step is not just “find a prettier normal map”; the teacher must influence depth/point geometry through training or through a geometry-consistent position/depth teacher.
- Sapiens remains useful as a high-quality normal teacher source, but the current normal-only prior patch is not a pass result.

## 2026-04-25 Addendum: Sapiens Direct VGGT Supervision Attempt
### Verdict
- Mentor-final bar is still **not reached**.
- Directly using Sapiens normals as VGGT `teacher_normals` supervision raises face ROI counts but collapses confidence to `1.0` and does not produce a clean HumanRAM/PSHuman-like head/face point cloud.

### New Tools / Configs
- Added `tools/build_external_normal_training_case.py` to copy a 4K4D training case and inject external normal maps as `teacher_normals`, `teacher_mask`, and ROI masks.
- Added conservative config `training/config/4k4d_prior_case_sparseproto_humancrop_pointnormal_r3_sapiens_conservative.yaml` to reduce normal-teacher dominance while preserving point-normal consistency.
- Original-protocol Sapiens teacher output: `output/detail_normal_refiner_20260425/external_sapiens_normal_teacher/original6v_headshoulder_sapiens03b`.
- Original-protocol Sapiens training case: `output/training_cases/0012_11_frame0000_6views_headshoulder_sapiens_normal_teacher_r1`.

### ROI Results
| Protocol | Run | Face ROI points | Truthful interpretation |
|---|---:|---:|---|
| `original_6v_headshoulder` | `sapiens_normal_teacher_e2` | 26879 | numeric pseudo-gain; confidence threshold collapsed to `1.0`, visual has ghost/double-head artifacts |
| `original_6v_headshoulder` | `sapiens_conservative_e1` | 27090 | numeric pseudo-gain; confidence threshold still `1.0`, visual not clean enough |

### Visual Evidence
- Sapiens e2 evidence sheet: `D:/vggt/vggt-main/output/detail_normal_refiner_20260425/open3d_sapiens_normal_teacher_training_e2/signfix_surface_sapiens_evidence_sheet.png`.
- Sapiens conservative evidence sheet: `D:/vggt/vggt-main/output/detail_normal_refiner_20260425/open3d_sapiens_conservative_e1/signfix_vs_sapiens_evidence_sheet.png`.

### Diagnosis Update
- Sapiens normals are a better detail-normal teacher than NormalBae, but normal-only supervision still does not anchor sparse-view depth/point geometry.
- The confidence-collapse pattern matches prior pseudo-positive failures, so these higher counts are not valid pass evidence.
- Next credible route requires coupling the Sapiens/detail normal teacher to a geometry-consistent depth/position teacher, or using a stronger full human reconstruction teacher that supplies aligned surface points, not normals alone.

## 2026-04-25 Addendum: Normal-Guided Depth Teacher Attempt
### Verdict
- Mentor-final bar is still **not reached**.
- Converting Sapiens normals into a local normal-guided depth/point pseudo teacher did not fix the confidence-collapse problem and worsened the Open3D head/face visual.

### New Tool / Artifacts
- Added `tools/build_normal_guided_depth_training_case.py`, which anchors depth to `signfix ckpt4`, applies a clipped normal-guided local depth refinement, and writes a 4K4D training case with updated `depths`, `cam_points`, `world_points`, and `teacher_normals`.
- Final safer head/face-only teacher case: `output/training_cases/0012_11_frame0000_6views_headshoulder_sapiens_normal_guided_depth_headface_r2`.
- Teacher diagnostics: `output/detail_normal_refiner_20260425/normal_guided_depth_teacher_original6v_sapiens_headface_r2`.
- Trained model output: `output/modal_training_results/20260425_original6v_sapiens_normal_guided_depth_hf_r2_lr2e7_e1`.

### ROI / Visual Result
| Protocol | Run | Face ROI points | Truthful interpretation |
|---|---:|---:|---|
| `original_6v_headshoulder` | `sapiens_normal_guided_depth_r2` | 28552 | numeric pseudo-gain; confidence threshold is still `1.0`; Open3D head/face visual is worse |

### Evidence
- Visual comparison: `D:/vggt/vggt-main/output/detail_normal_refiner_20260425/open3d_sapiens_normal_guided_depth_r2/signfix_sapiens_depth_evidence_sheet.png`.
- The head view becomes visibly broken/noisy compared with `signfix ckpt4`, so this is not a valid pass despite the higher ROI count.

### Diagnosis Update
- Normal-guided depth from a monocular normal map is not geometry-consistent enough for sparse multi-view VGGT training in this setup.
- The repeated failure mode is now clear: normal-only or normal-derived pseudo-depth can inflate point counts while destroying confidence calibration and 3D structure.
- The next credible step must use a full surface/mesh/depth teacher with multi-view consistency, not a locally integrated monocular normal map.

## 2026-04-25 Addendum: Confidence-Guard / Geometry-Only Teacher Attempts
### Verdict
- Mentor-final bar is still **not reached**.
- The confidence-collapse diagnosis is now confirmed: teacher branches that supervise confidence can create pseudo-positive ROI counts at `conf_threshold=1.0`; disabling confidence supervision avoids that collapse but does not improve the original 6v face ROI or Open3D quality.

### New Tools / Configs
- Added `supervise_conf` switches in `training/loss.py` for point/depth/human-prior losses so auxiliary teacher geometry can be trained without pushing `world_points_conf` toward the `expp1` lower bound.
- Added `training/config/4k4d_prior_case_sparseproto_humancrop_pointnormal_r4_confguard.yaml` as a calibrated confidence-guard attempt.
- Added `training/config/4k4d_prior_case_sparseproto_humancrop_pointnormal_r5_geoonly.yaml` with human-prior depth/point/normal/point-normal confidence supervision disabled.
- Re-downloaded `output/modal_results/20260426_original6v_surface_teacher_geoonly_inference_on6v_headshoulder/predictions.npz` via RPC chunks after the first local npz had a CRC failure.

### ROI / Visual Results
| Protocol | Run | Conf threshold | Face ROI points | Truthful interpretation |
|---|---:|---:|---:|---|
| `original_6v_headshoulder` | `mesh_raycast_facecore_r4` | 1.0 | 30808 | numeric pseudo-gain; noisy/failed Open3D visual |
| `original_6v_headshoulder` | `mesh_raycast_facecore_r4_confguard` | n/a | 12630 | confidence guarded but underperforms reference |
| `original_6v_headshoulder` | `surface_teacher_confguard` | 1.0 | 26550 | pseudo-positive; visual is not a clear win |
| `original_6v_headshoulder` | `surface_teacher_geoonly` | 45.916 | 15535 | confidence collapse fixed, but below signfix reference `16825` |

### Evidence
- Mesh facecore r4 evidence: `D:/vggt/vggt-main/output/detail_normal_refiner_20260425/open3d_mesh_raycast_facecore_r4/signfix_vs_facecore_r4_evidence_sheet.png`.
- Surface confguard evidence: `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_surface_confguard_e1/signfix_vs_surface_confguard_evidence_sheet.png`.
- Surface geoonly evidence: `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_surface_geoonly_e1/signfix_vs_surface_geoonly_evidence_sheet.png`.

### Diagnosis Update
- `surface_teacher_geoonly` proves the confidence-collapse fix works mechanically: its 40th-percentile confidence threshold is `45.916`, not `1.0`.
- The fixed-confidence result is still worse than `signfix ckpt4` on the same original 6v headshoulder protocol: face ROI `15535` vs `16825`, and the Open3D face close-up is visibly more hollow.
- Therefore the remaining blocker is teacher geometry quality/alignment, not just confidence calibration.

## 2026-04-25 Addendum: Sapiens Depth Teacher Attempt
### Verdict
- Mentor-final bar is still **not reached**.
- Sapiens depth is not reliable enough as a geometry teacher after per-view affine alignment to the signfix anchor.

### New Tool / Artifacts
- Added `modal_sapiens_depth_teacher.py` for Modal-based `facebook/sapiens-depth-0.3b-torchscript` inference.
- Added `tools/build_external_depth_training_case.py` to align external relative depth to the signfix anchor and write patched `depths`, `cam_points`, and `world_points`.
- Sapiens depth output: `output/detail_normal_refiner_20260425/external_sapiens_depth_teacher/original6v_headshoulder_sapiens03b/sapiens_depths.npz`.
- Conservative training case: `output/training_cases/0012_11_frame0000_6views_headshoulder_sapiens_depth_facecore_r1`.
- Diagnostics: `output/detail_normal_refiner_20260425/sapiens_depth_teacher_original6v_facecore_r1/preview_sheet.png`.

### Diagnosis Update
- Per-view Sapiens depth correlation with the signfix anchor is low, roughly `0.12–0.43`, so it is not a trustworthy multi-view geometry teacher.
- This branch should remain diagnostic only unless a stronger alignment/teacher source is introduced.

## 2026-04-25 Addendum: External Mesh Raycast Adapter
### Verdict
- The engineering bridge is now real, but it is **not** a quality pass by itself.
- Reusing the existing Poisson mesh teacher through the new adapter confirms the adapter can write a valid training case, but the underlying mesh is still too coarse/noisy for mentor-final head/face quality.

### New Tool / Smoke
- Added `tools/build_external_mesh_raycast_training_case.py`.
- Intended input: an external full-human mesh from an ECON/PSHuman/PIFuHD-style estimator, plus the sparse scene cameras and a VGGT anchor prediction.
- It can optionally Sim(3)-align the external mesh to anchor points with `--align-mode umeyama_icp`, then raycast the mesh into the 6 sparse views and patch `depths`, `cam_points`, `world_points`, `teacher_normals`, `teacher_mask`, and ROI masks.
- Smoke output case: `output/training_cases/0012_11_frame0000_6views_headshoulder_external_mesh_adapter_smoke`.
- Smoke diagnostics: `output/detail_normal_refiner_20260426/external_mesh_adapter_smoke/preview_sheet.png`.

### Smoke Result
- Smoke mesh source: `output/detail_normal_refiner_20260425/mesh_raycast_teacher_original6v_facecore_r4/teacher_mesh_poisson.ply`.
- Raycast hit pixels: `27396` total, per view `[3523, 4149, 5388, 5365, 4812, 4159]`.
- The preview confirms the adapter projects mesh hits into the expected head/face regions, but the hit map is sparse and fragmented because the source Poisson mesh is already weak.

### Diagnosis Update
- This closes the missing repo-side adapter for `external full-human mesh -> 6-view camera raycast -> training case`.
- It does **not** remove the main blocker: a high-quality external clothed-human/face mesh is still required before another cloud training run is justified.

## 2026-04-26 Addendum: 60v SurfacePose Point-Cloud Mesh via External Adapter
### Verdict
- Mentor-final bar is still **not reached**.
- A cleaner 60v-derived facecore mesh teacher still did not improve original 6v head/face geometry after low-lr geometry-only training.

### New Case / Training
- Built a 60v SurfacePose point-cloud Poisson teacher through the external mesh adapter:
  `output/training_cases/0012_11_frame0000_6views_headshoulder_external_60v_surfacepose_mesh_facecore_r1`.
- Teacher source:
  `output/modal_results/0012_11_frame0000_60views_smplxsurfacepose_a10080_e2_r2/pointcloud_depth_unprojection_dense_p40_hires/fused_pointcloud_masked.ply`.
- Teacher diagnostics:
  `output/detail_normal_refiner_20260426/external_60v_surfacepose_mesh_facecore_r1/preview_sheet.png`.
- Training output:
  `output/modal_training_results/20260426_original6v_external60v_surfacepose_facecore_geoonly_lr5e8_e1`.
- Inference output:
  `output/modal_results/20260426_original6v_external60v_surfacepose_facecore_geoonly_inference_on6v_headshoulder`.

### ROI / Visual Result
| Protocol | Run | Conf threshold | Face ROI points | Truthful interpretation |
|---|---:|---:|---:|---|
| `original_6v_headshoulder` | `signfix_ckpt4_ref` | 38.507 | 16825 | current valid reference |
| `original_6v_headshoulder` | `external60v_surfacepose_facecore_direct_noboost` | 38.507 | 16695 | direct fusion keeps threshold but is below reference; not pass |
| `original_6v_headshoulder` | `external60v_surfacepose_facecore_geoonly` | 81.000 | 11729 | confidence is high but face ROI is much lower and Open3D is visibly worse |

### Evidence
- Open3D evidence sheet:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_external60v_surfacepose_facecore_geoonly_e1/signfix_vs_external60v_surfacepose_facecore_geoonly_evidence_sheet.png`.
- Direct no-boost Open3D evidence:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_external60v_surfacepose_facecore_direct_noboost/face/face_close.png`,
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_external60v_surfacepose_facecore_direct_noboost/head/head_close.png`.

### Diagnosis Update
- The 60v SurfacePose-derived mesh is cleaner as a teacher preview than older noisy Poisson variants, but it is still too incomplete/small in the face-core ROI.
- Direct no-boost fusion confirms the 60v SurfacePose teacher does not improve the same original 6v headshoulder protocol even before training.
- Training on this teacher at `lr=5e-8` and with `supervise_conf=false` avoids confidence collapse, yet damages sparse-view point placement.
- This reinforces that continuing to recycle 60v VGGT/Poisson meshes is not enough; the next real teacher must be an external high-quality clothed-human mesh/normal source.

## 2026-04-26 Addendum: R3 Faceboost and PIFuHD Mesh Teacher Attempts
### Verdict
- Mentor-final bar is still **not reached**.
- The 2026-04-24 r3 faceboost / teacher-combo cloud runs are confirmed negative under the same original 6v headshoulder ROI protocol.
- PIFuHD 256/512 single-view mesh teachers can be generated and raycast into the sparse cameras, but low-lr geometry-only training from `signfix ckpt4` still damages the head/face point cloud.

### R3 / Teacher-Combo ROI Results
| Protocol | Run | Conf threshold | Face ROI points | Truthful interpretation |
|---|---:|---:|---:|---|
| `original_6v_headshoulder` | `signfix_ckpt4_ref` | 38.507 | 16825 | current valid reference |
| `original_6v_headshoulder` | `r3mixed_ckpt3` | 48.433 | 16461 | below reference |
| `original_6v_headshoulder` | `r3mixed_inference_model` | 48.433 | 16461 | below reference |
| `original_6v_headshoulder` | `r3_6v_ckpt2` | 98.725 | 16463 | below reference |
| `original_6v_headshoulder` | `r3_6v_inference_model` | 98.725 | 16463 | below reference |
| `original_6v_headshoulder` | `teachercombo_ckpt3` | 54.380 | 16368 | below reference |
| `original_6v_headshoulder` | `teachercombo_inference_model` | 54.380 | 16368 | below reference |
| `original_6v_headshoulder` | `teachergeom_roi_combo_from_ckpt4_ckpt0` | 21.259 | 16336 | below reference |

### PIFuHD Teacher Results
| Protocol | Run | Conf threshold | Face ROI points | Truthful interpretation |
|---|---:|---:|---:|---|
| `original_6v_headshoulder` | `pifuhd256_facecore_geoonly` | 68.781 | 10611 | much lower than reference; Open3D face/head collapse into a flattened slab |
| `original_6v_headshoulder` | `pifuhd512_facecore_geoonly_from_infer` | 64.053 | 10863 | much lower than reference; 512 mesh is denser but the trained point cloud remains flattened |

### New Artifacts
- PIFuHD 256 teacher mesh:
  `output/detail_normal_refiner_20260426/pifuhd_mesh_teacher_smoke/result_30_src_cam30_256.ply`.
- PIFuHD 512 teacher mesh:
  `output/detail_normal_refiner_20260426/pifuhd_mesh_teacher_512/result_30_src_cam30_512.ply`.
- PIFuHD 256 raycast case:
  `output/training_cases/0012_11_frame0000_6views_headshoulder_pifuhd256_mesh_facecore_r1`.
- PIFuHD 512 raycast case:
  `output/training_cases/0012_11_frame0000_6views_headshoulder_pifuhd512_mesh_facecore_r1`.
- PIFuHD 256 Open3D evidence:
  `output/detail_normal_refiner_20260426/open3d_pifuhd256_facecore_geoonly_e1/signfix_vs_pifuhd256_facecore_geoonly_evidence_sheet.png`.
- PIFuHD 512 Open3D evidence:
  `output/detail_normal_refiner_20260426/open3d_pifuhd512_facecore_geoonly_from_infer_e1/signfix_vs_pifuhd512_facecore_geoonly_evidence_sheet.png`.

### Diagnosis Update
- Increasing the PIFuHD mesh resolution from 256 to 512 raises mesh density but does not materially improve raycast coverage: `25610` vs `25352` face-core hit pixels.
- The repeated PIFuHD failure is not just point density; after alignment/raycast and geoonly fine-tuning, the predicted sparse-view point cloud shifts into a flattened head/face geometry.
- There is still no local ECON/PSHuman-quality external mesh asset available in the repo. PIFuHD is now a validated smoke teacher source, but not a mentor-final reconstruction result.

## 2026-04-26 Addendum: No-Boost PIFuHD512 Residual Fusion
### Verdict
- Mentor-final bar is still **not reached**.
- Unlike the failed geoonly training runs, post-hoc residual fusion can improve the original 6v face ROI count without changing confidence, but the Open3D close-ups still show seams/noise and do not yet show clear eyes/nose/facial detail.
- This is a useful diagnostic direction, not a pass claim.

### New Tool / Engineering Updates
- Added `tools/fuse_external_teacher_into_predictions.py` for single external teacher residual fusion.
- Added `tools/fuse_multi_external_teachers_into_predictions.py` for multi-teacher fusion with `mean/median`, `min_votes`, and `--max-agreement-distance`.
- The multi-teacher tool keeps `world_points_conf` unchanged, so any ROI change is not caused by confidence inflation.
- Added `estimator-command` mode to `tools/run_realdata_smplx_driver.py`; it can launch an external SMPL-X estimator command, then import and validate its output as a canonical prior bundle.
- Rewrote `docs/realdata_smplx_driver.md` with the current truthful real-data route and the estimator-command smoke command.

### Original 6v Headshoulder ROI Results
| Protocol | Run | Conf threshold | Face ROI points | Truthful interpretation |
|---|---:|---:|---:|---|
| `original_6v_headshoulder` | `signfix_ckpt4_ref` | 38.507 | 16825 | current valid reference |
| `original_6v_headshoulder` | `pifuhd512_cam00_single_residual_noboost` | 38.507 | 17113 | above reference, but visual not final |
| `original_6v_headshoulder` | `pifuhd512_cam30_single_residual_noboost` | 38.507 | 17300 | above reference, but still has seams/noise |
| `original_6v_headshoulder` | `pifuhd512_cam45_single_residual_noboost` | 38.507 | 17126 | above reference, but visual not final |
| `original_6v_headshoulder` | `pifuhd512_multi3_mean_v1_noboost` | 38.507 | 17315 | best no-boost count; visual still not mentor-final |
| `original_6v_headshoulder` | `pifuhd512_multi3_mean_v2_ag020_noboost` | 38.507 | 17123 | cleaner/conservative, still not clear face detail |
| `original_6v_headshoulder` | `pifuhd512_multi3_mean_v3_noboost` | 38.507 | 17052 | strict all-teacher consensus; still not final |

### Evidence
- Main no-boost evidence sheet:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_multi3_truthful_evidence/signfix_vs_pifuhd512_multi3_truthful_evidence_sheet.png`.
- Best local ROI summary:
  `D:/vggt/vggt-main/output/modal_results/20260426_signfix_pifuhd512_multi3_mean_v1_xz050_y000_d004_noboost/local_roi_summary.json`.
- Agreement-gated ROI summary:
  `D:/vggt/vggt-main/output/modal_results/20260426_signfix_pifuhd512_multi3_mean_v2_ag020_xz050_y000_d004_noboost/local_roi_summary.json`.

### Targetcam30 Positive-Face Check
| Protocol | Run | Conf threshold | Face ROI points | Truthful interpretation |
|---|---:|---:|---:|---|
| `targetcam30_raw_headface` | `signfix_ckpt4` | 31.246 | 23331 | front/side face crop, but still sparse/holes |
| `targetcam30_raw_headface` | `pifuhd512_residual_noboost` | 31.246 | 23316 | slightly below baseline; not useful |

### Real-Data SMPL-X Bridge Smoke
- `build_scene_prior_from_external_bundle.py` smoke remains valid.
- The previously interrupted Modal inference smoke is now complete:
  `output/modal_results/20260424_smoke_external_prior_scene_bridge_ckpt4`.
- Modal summary confirms `prior_tensor_shape=[2, 30, 518, 518]` and `prior_summary_tensor_shape=[2, 16, 27]`.
- `run_realdata_smplx_driver.py --mode estimator-command --skip-estimator-run` smoke succeeded at:
  `output/smoke_external_bundle_case/bundle_via_estimator_command_skip`.

### Diagnosis Update
- Residual fusion is the first current branch to exceed `16825` while preserving the original confidence threshold (`38.507`), so it is more credible than prior pseudo-positive confidence-collapse runs.
- The improvement is still not enough: Open3D face/head views show more side/head structure but also seam artifacts and no reliable eye/nose/detail recovery.
- The targetcam30 check confirms PIFuHD512 coverage is not strong enough as a front-face detail teacher in the current alignment/raycast form.
- Next credible route remains a stronger external human-specific mesh/normal teacher, ideally PSHuman/ECON-quality, before another large sparse-view VGGT training run.

## 2026-04-26 Addendum: Camera-Aligned Preprocess Recheck
### Verdict
- Mentor-final bar is still **not reached**.
- Head/face crops increase retained point counts, but the camera-aligned Open3D face views expose large holes and fragmented facial surfaces.
- These variants are therefore numeric-positive but visual-negative, and must not be promoted as a final sparse-view result.

### Original 6v / Cam30-Aligned ROI Results
| Protocol | Variant | Conf threshold | Head ROI points | Face ROI points | Truthful interpretation |
|---|---:|---:|---:|---:|---|
| `original_6v_headshoulder` | `headshoulder_ref / signfix_ckpt4` | 38.507 | 40527 | 16825 | current valid reference; lower density but fewer crop-induced holes |
| `6v_headface_crop` | `signfix_ckpt4` | 5.012 | 75937 | 21126 | higher point count, but face view has large missing regions; not pass |
| `6v_raw_headface_crop` | `signfix_ckpt4` | 22.505 | 76060 | 20187 | higher point count, but face/head surface has stripes and holes; not pass |
| `6v_raw_headface_hardmask` | `signfix_ckpt4` | 24.706 | 76060 | 20433 | higher point count, but hardmask worsens face holes; not pass |

### Evidence
- Camera-aligned head/face comparison sheet:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_camera_view_cam30_preprocess_compare/preprocess_camera_view_head_face_truthful_sheet.png`.
- No-downsample ROI table:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_camera_view_cam30_preprocess_compare/preprocess_local_roi_no_downsample.json`.
- Per-render Open3D ROI table:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_camera_view_cam30_preprocess_compare/preprocess_open3d_roi_table.json`.

### Diagnosis Update
- The crop variants mainly increase pixel/point coverage by zooming into the head/face, but this does not translate into reliable facial geometry.
- The face ROI renders show that the extra points are not organized into clear eyes, nose, mouth, or continuous cheek/forehead surfaces.
- This confirms the current bottleneck is not just input crop scale; it is still a missing high-quality human surface/detail teacher or refinement signal.

## 2026-04-26 Addendum: Official PSHuman Mesh Teacher Check
### Verdict
- Mentor-final bar is still **not reached**.
- The self-hosted official PSHuman route is now technically unblocked and can generate a mesh for the target head/face crop, align it to the sparse-view prediction, raycast it into the six cameras, and fuse it without confidence boosting.
- The same-protocol raw-headface ROI count improves slightly for the direct fusion variant, but Open3D face/head close-ups show interior fragments and no reliable eye/nose/mouth recovery. This is a useful teacher bridge, not a sparse-view geometry pass.

### Raw-Headface ROI Results
| Protocol | Run | Conf percentile | Head ROI points | Face ROI points | Truthful interpretation |
|---|---:|---:|---:|---:|---|
| `targetcam30_raw_headface` | `signfix_ckpt4_ref` | 40 | 70758 | 25020 | current raw-headface reference; still holey |
| `targetcam30_raw_headface` | `pshuman_official_direct_a100_noboost` | 40 | 70758 | 26046 | numeric gain, but visual adds fragmented interior surfaces; not pass |
| `targetcam30_raw_headface` | `pshuman_official_xz050_y000_d004_noboost` | 40 | 70758 | 25121 | conservative fusion is nearly baseline and still not clear enough |

### Official PSHuman Artifacts
- Mesh output:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/pshuman_official_mesh_targetcam30_raw_headface_assets_r10_tmpsymlink/result_clr_scale4_00_tgt_cam30.obj`.
- Raycast teacher case:
  `D:/vggt/vggt-main/output/training_cases/0012_11_frame0000_6views_targetcam30_pshuman_official_mesh_headface_r1`.
- Direct fused prediction:
  `D:/vggt/vggt-main/output/modal_results/20260426_pshuman_official_mesh_headface_r1_direct_a100_noboost`.
- Conservative fused prediction:
  `D:/vggt/vggt-main/output/modal_results/20260426_pshuman_official_mesh_headface_r1_xz050_y000_d004_noboost`.
- Open3D comparison sheets:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_pshuman_official_mesh_compare/contact_sheets/pshuman_official_face_close_compare.png`,
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_pshuman_official_mesh_compare/contact_sheets/pshuman_official_head_close_compare.png`,
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_pshuman_official_mesh_compare/contact_sheets/pshuman_official_camera_view00_crop_compare.png`.

### Diagnosis Update
- The successful PSHuman mesh run proves the external human-specific teacher path can be executed end-to-end inside the repository workflow.
- A single target-view PSHuman mesh is not yet a sufficient geometric teacher: alignment/raycast/fusion increases local point count but introduces inconsistent internal face surfaces.
- The next credible branch is not another large VGGT training run on this noisy single-view teacher; it is a stronger multi-view/consensus teacher stage before training or final promotion.

## 2026-04-26 Addendum: Official PSHuman Multi-View Consensus Check
### Verdict
- Mentor-final bar is still **not reached**.
- Official PSHuman meshes were generated for all six raw-headface input views, then each mesh was Sim(3)+ICP aligned, raycast into the same six sparse cameras, and fused without confidence boosting.
- Multi-view consensus reduces some single-view noise but does not recover clear eye/nose/mouth geometry. The best multi6 face ROI count is still only a small numeric change over the raw-headface reference and the Open3D close-ups remain holey/fragmented.

### Multi-View Teacher Coverage
| Teacher view | Mesh vertices | Mesh faces | Raycast hit pixels | Alignment final median residual |
|---|---:|---:|---:|---:|
| `00_tgt_cam30` | 77517 | 155038 | 34901 | 0.0109 |
| `01_src_cam00` | 84208 | 168472 | 39487 | 0.0108 |
| `02_src_cam11` | 74052 | 148100 | 38841 | 0.0130 |
| `03_src_cam22` | 83644 | 167312 | 38090 | 0.0087 |
| `04_src_cam34` | 78397 | 156870 | 40831 | 0.0108 |
| `05_src_cam45` | 83425 | 166876 | 38278 | 0.0109 |

### Multi6 ROI Results
| Protocol | Run | Conf percentile | Head ROI points | Face ROI points | Truthful interpretation |
|---|---:|---:|---:|---:|---|
| `targetcam30_raw_headface` | `signfix_ckpt4_ref` | 40 | 70758 | 25020 | raw-headface reference |
| `targetcam30_raw_headface` | `pshuman_official_single_direct` | 40 | 70758 | 26046 | more points but visible interior fragments; not pass |
| `targetcam30_raw_headface` | `pshuman_official_multi3_mean_min2` | 40 | 70758 | 25079 | nearly baseline; not pass |
| `targetcam30_raw_headface` | `pshuman_official_multi6_mean_min3` | 40 | 70758 | 25342 | best multi6 count, still small and visually not clear |
| `targetcam30_raw_headface` | `pshuman_official_multi6_median_min3` | 40 | 70758 | 25280 | still fragmented |
| `targetcam30_raw_headface` | `pshuman_official_multi6_xz_ag` | 40 | 70758 | 25069 | conservative and near baseline |
| `targetcam30_raw_headface` | `pshuman_official_multi6_min4_ag` | 40 | 70758 | 25241 | still not final |

### Evidence
- Multi3 comparison sheets:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_pshuman_official_mesh_multi3_compare/contact_sheets/pshuman_official_multi3_face_close_compare.png`,
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_pshuman_official_mesh_multi3_compare/contact_sheets/pshuman_official_multi3_camera_view00_crop_compare.png`.
- Multi6 comparison sheets:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_pshuman_official_mesh_multi6_compare/contact_sheets/pshuman_official_multi6_face_close_compare.png`,
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_pshuman_official_mesh_multi6_compare/contact_sheets/pshuman_official_multi6_camera_view00_crop_compare.png`.

### Diagnosis Update
- The official PSHuman bridge is real and reusable, but the fast 512-resolution single-image mesh setting is not a mentor-final teacher.
- Multi-view consensus helps reject some view-specific hallucination; it does not add missing facial detail.
- The next credible check is a higher-quality official PSHuman run before any new sparse-view VGGT training on PSHuman-derived targets.

## 2026-04-26 Addendum: Official PSHuman HQ1024 / Strict Face-Core Check
### Verdict
- Mentor-final bar is still **not reached**.
- The higher-resolution official PSHuman run is technically successful and produces a much denser mesh, but the fused sparse-view point cloud still fails the visual requirement: face/head Open3D close-ups show extra fragments rather than reliable eyes, nose, mouth, or continuous facial surfaces.
- The direct teacher-mesh Open3D render confirms the current HQ1024 PSHuman teacher itself is not PSHuman-paper-quality for this crop, so it should not be used as a final VGGT training target without a stronger validation/refinement stage.

### HQ1024 ROI Results
| Protocol | Run | Conf percentile | Head ROI points | Face ROI points | Truthful interpretation |
|---|---:|---:|---:|---:|---|
| `targetcam30_raw_headface` | `signfix_ckpt4_ref` | 40 | 70758 | 25020 | raw-headface reference; still holey |
| `targetcam30_raw_headface` | `pshuman_official_hq1024_direct` | 40 | 70758 | 25567 | small numeric gain, but more fragmented facial interior; not pass |
| `targetcam30_raw_headface` | `pshuman_official_hq1024_xz050` | 40 | 70758 | 25122 | near baseline; not pass |
| `targetcam30_raw_headface` | `pshuman_official_hq1024_facecore_strict_direct` | 40 | 70758 | 24964 | stricter face-core is below baseline; not pass |

### HQ1024 Teacher / Fusion Artifacts
- HQ1024 mesh output:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/pshuman_official_mesh_targetcam30_raw_headface_hq1024_s40_r700_c200/result_clr_scale4_00_tgt_cam30.obj`.
- HQ1024 mesh stats:
  `358023` vertices and `716114` faces.
- HQ1024 raycast teacher case:
  `D:/vggt/vggt-main/output/training_cases/0012_11_frame0000_6views_targetcam30_pshuman_official_mesh_hq1024_headface_r1`.
- Strict face-core raycast teacher case:
  `D:/vggt/vggt-main/output/training_cases/0012_11_frame0000_6views_targetcam30_pshuman_official_mesh_hq1024_facecore_strict_r1`.
- HQ1024 Open3D comparison sheets:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_pshuman_official_mesh_hq1024_compare/contact_sheets/pshuman_official_hq1024_face_close_compare.png`,
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_pshuman_official_mesh_hq1024_compare/contact_sheets/pshuman_official_hq1024_head_close_compare.png`,
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_pshuman_official_mesh_hq1024_compare/contact_sheets/pshuman_official_hq1024_camera_view00_crop_compare.png`.
- Strict face-core Open3D evidence:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_pshuman_official_mesh_hq1024_facecore_strict_compare/facecore_direct_face/face_close.png`.
- Direct teacher-mesh Open3D evidence:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_pshuman_hq1024_teacher_mesh_direct/face/face_close.png`.

### Diagnosis Update
- Increasing PSHuman resolution alone does not fix the sparse-view face/head geometry bottleneck.
- Tightening to a strict face-core teacher reduces coverage and drops below the baseline face ROI count, so ROI/depth tolerance alone is not the missing ingredient.
- The current external-teacher path remains useful infrastructure, but the next credible route needs a stronger geometry-consistent multi-view surface teacher or a higher-quality human reconstruction source before promoting another sparse-view training run.

## 2026-04-26 Addendum: True-Highres Raw Head/Face Crop Check
### Verdict
- Mentor-final bar is still **not reached**.
- The new high-resolution crop path is real: it crops from the original `2048x2448` / `3000x4096` RGB pixels before resizing to `518x518`.
- This improves the input crop provenance, but the current ckpt4 geometry still has large face holes in Open3D and does not beat the existing raw-headface hardmask protocol.

### ROI Results
| Protocol | Run | Conf percentile | Head ROI points | Face ROI points | Truthful interpretation |
|---|---:|---:|---:|---:|---|
| `6v_raw_headface_hardmask` | `signfix_ckpt4` | 40 | 76060 | 20433 | previous raw-headface hardmask reference; still visual-negative |
| `6v_truehighres_headface_hardmask` | `signfix_ckpt4` | 40 | 78755 | 20142 | true raw-pixel crop, but face ROI is lower and Open3D remains holey; not pass |
| `6v_truehighres_headface_hardmask_reprojprior` | `signfix_ckpt4` | 40 | 78755 | 20319 | reprojected SMPL-X prior recovers only a small count and preserves the same large face hole; not pass |
| `targetcam30_truehighres_headface_hardmask_reprojprior` | `signfix_ckpt4` | 40 | 73275 | 25788 | front-face target protocol gains points, but Open3D still has a large central face hole; not pass |
| `targetcam30_truehighres_headshoulder_reprojprior` | `signfix_ckpt4` | 40 | 51269 | 19718 | wider crop with RGB background/context drops face count and keeps severe holes; not pass |

### Evidence
- True-highres scene:
  `D:/vggt/vggt-main/output/4k4d_preprocessed_scene_variants/0012_11_frame0000_6views_sparseproto_truehighres_headface_hardmask`.
- Modal inference:
  `D:/vggt/vggt-main/output/modal_results/20260426_6v_truehighres_headface_hardmask_signfix_ckpt4`.
- Reprojected-prior Modal inference:
  `D:/vggt/vggt-main/output/modal_results/20260426_6v_truehighres_headface_hardmask_reprojprior_signfix_ckpt4`.
- Targetcam30 true-highres Modal inference:
  `D:/vggt/vggt-main/output/modal_results/20260426_6v_targetcam30_truehighres_headface_hardmask_reprojprior_signfix_ckpt4`.
- Targetcam30 true-highres headshoulder Modal inference:
  `D:/vggt/vggt-main/output/modal_results/20260426_6v_targetcam30_truehighres_headshoulder_reprojprior_signfix_ckpt4`.
- Open3D face/head renders:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_truehighres_headface_hardmask_ckpt4_face/face_close.png`,
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_truehighres_headface_hardmask_ckpt4_head/head_close.png`.
- Reprojected-prior Open3D face/head renders:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_truehighres_headface_hardmask_reprojprior_ckpt4_face/face_close.png`,
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_truehighres_headface_hardmask_reprojprior_ckpt4_head/head_close.png`.
- Targetcam30 true-highres Open3D face/head renders:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_targetcam30_truehighres_headface_hardmask_reprojprior_ckpt4_face/face_close.png`,
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_targetcam30_truehighres_headface_hardmask_reprojprior_ckpt4_head/head_close.png`.
- Targetcam30 true-highres headshoulder Open3D face/head renders:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_targetcam30_truehighres_headshoulder_reprojprior_ckpt4_face/face_close.png`,
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_targetcam30_truehighres_headshoulder_reprojprior_ckpt4_head/head_close.png`.

### Diagnosis Update
- The previous concern that crop preprocessing did not exploit raw DNA image resolution is now directly tested.
- Raw-pixel crop alone does not solve the geometry bottleneck; exact SMPL-X prior reprojection after crop also does not remove the large face hole.
- Even the easier front-face targetcam30 protocol remains visual-negative: the count increase comes with a large central face hole rather than eyes/nose/mouth detail.
- A wider headshoulder crop with real RGB background/context is worse, so the blocker is not simply over-tight hardmasking.
- The model still needs a stronger geometry/detail signal, not only a sharper crop or better-aligned coarse prior.

## 2026-04-26 Addendum: 60-View Targetcam30 Teacher Viability Check
### Verdict
- Mentor-final bar is still **not reached**.
- The corrupted local `60v_targetcam30_raw_headface_hardmask_reproj` prediction was recovered through chunked Modal redownload and reassembly.
- The 60-view run has a very high all-view ROI count, but Open3D target face/head close-ups are still sparse and fragmented; it does not reliably show eyes, nose, mouth, or continuous facial surfaces.
- Therefore this 60-view VGGT output should **not** be promoted as a detail teacher for sparse-view distillation without additional surface cleanup or a stronger external teacher.

### Recovery / ROI Results
| Protocol | Run | Conf percentile | Head ROI points | Face ROI points | Truthful interpretation |
|---|---:|---:|---:|---:|---|
| `targetcam30_raw_headface_hardmask_reproj` | `60v_signfix_ckpt4` | 40 | 687030 | 254907 | all-view remote ROI is high, but visual target face remains fragmented; not pass as teacher |
| `targetcam30_raw_headface_hardmask_reproj` | `60v_signfix_ckpt4_open3d_local` | 40 | 110000 | 24427 | Open3D render after target/ROI filtering confirms sparse/holey face; not a reliable detail teacher |

### Evidence
- Recovered prediction:
  `D:/vggt/vggt-main/output/modal_results/20260424_60v_targetcam30_raw_headface_hardmask_reproj_signfix_ckpt4_redownload/predictions.npz`.
- Chunk resume tool update:
  `D:/vggt/vggt-main/modal_4k4d_vggt_infer.py` now skips valid local chunks, re-downloads invalid/missing chunks, retries transient chunk failures, and reassembles `predictions.npz`.
- 60v Open3D face render:
  `D:/vggt/vggt-main/output/open3d_audit/20260426_60v_targetcam30_raw_headface_signfix_face/face_close.png`.
- 60v Open3D head render:
  `D:/vggt/vggt-main/output/open3d_audit/20260426_60v_targetcam30_raw_headface_signfix_head/head_close.png`.

### Diagnosis Update
- More source views alone do not produce the PSHuman-level point quality requested by the mentor.
- Current VGGT 60-view outputs are useful as a consistency diagnostic, but not yet a high-quality normal/surface teacher.
- The next valid direction should stay within `detail_normal_refiner` / `pifuhd_style_normal_refine`: image-aligned residual refinement from RGB crop + coarse prior normal + mask, trained first on local head/neck ROI with a teacher that is visually validated before sparse-view end-to-end training.

## 2026-04-26 Addendum: Same-Protocol Sapiens Numeric False Positives
### Verdict
- Mentor-final bar is still **not reached**.
- Three existing same-protocol `6views_sparseproto_headshoulder_crop` Sapiens-trained/fused checkpoints produce much higher face ROI counts, but all have `conf_threshold=1.0`, meaning the confidence gate is not discriminating.
- Open3D face/head close-ups show fragmented or distorted point clouds instead of clearer eyes, nose, mouth, or continuous face surfaces, so these are numeric false positives.

### Same-Protocol ROI Results
| Protocol | Run | Conf percentile | Head ROI points | Face ROI points | Truthful interpretation |
|---|---:|---:|---:|---:|---|
| `6views_sparseproto_headshoulder_crop` | `signfix_ckpt4_ref` | 40 | 40527 | 16825 | current reference |
| `6views_sparseproto_headshoulder_crop` | `detail_refined_prior` | 40 | 40527 | 16816 | no improvement |
| `6views_sparseproto_headshoulder_crop` | `detail_refined_prior_full60_denseonly` | 40 | 40527 | 16828 | +3 only; visual-equivalent |
| `6views_sparseproto_headshoulder_crop` | `sapiens_normal_teacher_e2` | 40 | 67545 | 26879 | high count, but `conf_threshold=1.0` and face remains fragmented |
| `6views_sparseproto_headshoulder_crop` | `sapiens_conservative_e1` | 40 | 67545 | 27090 | high count, but still fragmented/distorted |
| `6views_sparseproto_headshoulder_crop` | `sapiens_normal_guided_depth_r2` | 40 | 67545 | 28552 | highest count, but Open3D head/face is visually worse; not pass |

### Evidence
- Same-protocol candidate comparison sheet:
  `D:/vggt/vggt-main/output/open3d_audit/20260426_sameprotocol_face_candidate_compare.png`.
- Sapiens normal-guided face render:
  `D:/vggt/vggt-main/output/open3d_audit/20260426_20260425_original6v_sapiens_normal_guided_depth_r2_inference_on6v_headshoulder_face/face_close.png`.
- Sapiens normal-guided head render:
  `D:/vggt/vggt-main/output/open3d_audit/20260426_20260425_original6v_sapiens_normal_guided_depth_r2_inference_on6v_headshoulder_head/head_close.png`.

### Diagnosis Update
- The current Sapiens-derived sparse-view checkpoints cannot be used as final evidence despite larger point counts.
- The acceptance gate must continue to require both quantitative improvement and clear Open3D face/head visual quality.
- The next experiment should not simply raise confidence-free point counts; it must improve the geometry source or enforce a stricter visible-surface/normal consistency target.

## 2026-04-26 Addendum: Clean PSHuman HQ1024 Multi-View Consensus Gate
### Verdict
- Mentor-final bar is still **not reached**.
- The clean-rebased PSHuman official HQ1024 six-mesh consensus path is now real and reproducible: all six meshes were recovered/validated, raycast from the clean `signfix` raw-headface case, fused conservatively, and rendered with Open3D.
- It does **not** pass the gate. The face ROI is slightly below the raw-headface reference and the Open3D face/head close-ups remain visually equivalent and holey, without reliable eye/nose/mouth detail.
- Because the Open3D gate failed, this teacher should **not** be used for another sparse-view training run.

### ROI Results
| Protocol | Run | Conf percentile | Head ROI points | Face ROI points | Truthful interpretation |
|---|---:|---:|---:|---:|---|
| `targetcam30_raw_headface_hardmask_reproj` | `signfix_ckpt4_ref` | 40 | 70758 | 25020 | current raw-headface reference |
| `targetcam30_raw_headface_hardmask_reproj` | `pshuman_hq1024_clean_multi6_median_min3_ag020_xz035_y000_d003_noboost` | 40 | 70758 | 25003 | conservative confidence preserved, but face count is `-17` and visual structure is not clearer; not pass |

### Evidence
- Fused prediction:
  `D:/vggt/vggt-main/output/modal_results/20260426_pshuman_hq1024_clean_multi6_median_min3_ag020_xz035_y000_d003_noboost/predictions.npz`.
- Fusion summary:
  `D:/vggt/vggt-main/output/modal_results/20260426_pshuman_hq1024_clean_multi6_median_min3_ag020_xz035_y000_d003_noboost/fusion_summary.json`.
- Open3D face render:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_pshuman_hq1024_clean_multi6_face/face_close.png`.
- Open3D head render:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_pshuman_hq1024_clean_multi6_head/head_close.png`.
- Baseline-vs-candidate comparison sheet:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_pshuman_hq1024_clean_multi6_compare_sheet.png`.
- Clean raycast cases:
  `D:/vggt/vggt-main/output/training_cases/0012_11_frame0000_6views_targetcam30_pshuman_hq1024_clean_tgt_cam30_headface_r1`,
  `D:/vggt/vggt-main/output/training_cases/0012_11_frame0000_6views_targetcam30_pshuman_hq1024_clean_src_cam00_headface_r1`,
  `D:/vggt/vggt-main/output/training_cases/0012_11_frame0000_6views_targetcam30_pshuman_hq1024_clean_src_cam11_headface_r1`,
  `D:/vggt/vggt-main/output/training_cases/0012_11_frame0000_6views_targetcam30_pshuman_hq1024_clean_src_cam22_headface_r1`,
  `D:/vggt/vggt-main/output/training_cases/0012_11_frame0000_6views_targetcam30_pshuman_hq1024_clean_src_cam34_headface_r1`,
  `D:/vggt/vggt-main/output/training_cases/0012_11_frame0000_6views_targetcam30_pshuman_hq1024_clean_src_cam45_headface_r1`.

### Diagnosis Update
- Clean multi-view PSHuman consensus is not enough when fused only where teachers agree; the agreement filter leaves only `8252` fused pixels and does not repair the central facial holes.
- The failure is not a confidence-threshold artifact: the candidate keeps the reference `conf_threshold=31.2460`, unlike previous `conf_threshold=1.0` pseudo-wins.
- The next credible step should target the rendering/evidence side first: improve Open3D point filtering/visibility diagnostics and seek a teacher that produces continuous target-view face surfaces before any new end-to-end training.

## 2026-04-26 Addendum: Targetcam30 Normal-Confidence False Positives
### Verdict
- Mentor-final bar is still **not reached**.
- A full output scan found a non-`conf_threshold=1.0` Sapiens-depth candidate with a high targetcam30 raw-headface face count, but the Open3D close-up is visually worse: the face becomes striped/distorted rather than clearer.
- This confirms the gate still must require visual continuity, not only normal confidence and a higher ROI count.

### ROI Results
| Protocol | Run | Conf percentile | Head ROI points | Face ROI points | Truthful interpretation |
|---|---:|---:|---:|---:|---|
| `targetcam30_raw_headface_hardmask_reproj` | `signfix_ckpt4_ref` | 40 | 70758 | 25020 | current raw-headface reference |
| `targetcam30_raw_headface_hardmask_reproj` | `sapiens_depth_headface_r1_noboost` | 40 | 70758 | 28125 | high count with normal confidence, but Open3D face/head are distorted and not mentor-quality |
| `targetcam30_raw_headface_hardmask_reproj` | `pshuman_hq1024_clean_multi6_median_min3_ag020_xz035_y000_d003_noboost` | 40 | 70758 | 25003 | conservative PSHuman consensus; no visual repair |

### Evidence
- Sapiens-depth face render:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_sapiens_depth_headface_r1_noboost_face/face_close.png`.
- Sapiens-depth head render:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_sapiens_depth_headface_r1_noboost_head/head_close.png`.
- False-positive comparison sheet:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_targetcam30_falsepositive_compare_sheet.png`.
- Depth-unprojection check:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_baseline_raw_headface_ckpt4_depthunproj_face/open3d_summary.json`,
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_pshuman_hq1024_clean_multi6_depthunproj_face/open3d_summary.json`.

### Diagnosis Update
- The Sapiens-depth candidate increases point count, but not usable face geometry. It should not be promoted or trained as a teacher without a stronger visible-surface consistency filter.
- Re-rendering the same predictions from `depth_unprojection` does not rescue the result; it produces lower face ROI counts and the same lack of reliable facial details.
- Current blockers are teacher geometry quality and surface continuity, not only the point-cloud renderer choice.

## 2026-04-26 Addendum: Sapiens Depth Face-Core Strict No-Boost Grid
### Verdict
- Mentor-final bar is still **not reached**.
- A stricter Sapiens-depth face-core grid was tested without confidence boost. It preserves the normal confidence threshold, but still does not remove the face/head holes in Open3D.
- The best strict grid result is `facecore_d005` with face ROI `25667`, only `+647` over the raw-headface reference and below the pre-declared `>=26500` gate. The close-up remains holey, so no training was launched.

### ROI Results
| Protocol | Run | Conf percentile | Head ROI points | Face ROI points | Truthful interpretation |
|---|---:|---:|---:|---:|---|
| `targetcam30_raw_headface_hardmask_reproj` | `signfix_ckpt4_ref` | 40 | 70758 | 25020 | current raw-headface reference |
| `targetcam30_raw_headface_hardmask_reproj` | `sapiens_depth_facecore_d005_noboost_fusion` | 40 | 70758 | 25667 | best strict count, but visual hole remains; not pass |
| `targetcam30_raw_headface_hardmask_reproj` | `sapiens_depth_facecore_d010_noboost_fusion` | 40 | 70758 | 25502 | small numeric gain, visual not pass |
| `targetcam30_raw_headface_hardmask_reproj` | `sapiens_depth_facecore_d015_noboost_fusion` | 40 | 70758 | 25157 | near baseline, visual not pass |
| `targetcam30_raw_headface_hardmask_reproj` | `sapiens_depth_facecore_d020_noboost_fusion` | 40 | 70758 | 25024 | effectively baseline, visual not pass |
| `targetcam30_raw_headface_hardmask_reproj` | `sapiens_depth_face_d010_noboost_fusion` | 40 | 70758 | 25479 | broader face patch, still no visual repair |
| `targetcam30_raw_headface_hardmask_reproj` | `sapiens_depth_face_d015_noboost_fusion` | 40 | 70758 | 24888 | regresses count |

### Evidence
- Tool compatibility fix:
  `D:/vggt/vggt-main/tools/build_external_depth_training_case.py` now writes `teacher_mask` so the generated depth-teacher case can be consumed by `tools/fuse_external_teacher_into_predictions.py`.
- Best strict face render:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_sapiens_depth_facecore_d005_noboost_face/face_close.png`.
- Best strict head render:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_sapiens_depth_facecore_d005_noboost_head/head_close.png`.
- Grid comparison sheet:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_sapiens_depth_facecore_grid_compare_sheet.png`.
- Best strict fused prediction:
  `D:/vggt/vggt-main/output/modal_results/20260426_sapiens_depth_facecore_d005_noboost_fusion/predictions.npz`.

### Diagnosis Update
- Restricting Sapiens depth to face-core and reducing the maximum depth delta avoids the worst striping, but it only nudges existing points. It does not create a continuous facial surface.
- Since the direct teacher-fusion Open3D gate did not pass, a geoonly training run from this teacher would be premature and was intentionally not launched.
- The next route should focus on generating or validating a continuous target-view head/face surface, not on more low-amplitude residual warps of the same sparse point cloud.

## 2026-04-26 Addendum: Targetcam30 Raw-Headface `image_mode=crop` Check
### Verdict
- Mentor-final bar is still **not reached**.
- The existing targetcam30 raw-headface scene images are already square `518x518` crops, so Modal `image_mode=crop` gives the same Open3D ROI counts as the `pad` reference.
- This rules out a simple inference-time `pad` vs `crop` preprocessing mismatch as the source of the face hole.

### ROI Results
| Protocol | Run | Image mode | Conf percentile | Head ROI points | Face ROI points | Truthful interpretation |
|---|---:|---:|---:|---:|---:|---|
| `targetcam30_raw_headface_hardmask_reproj` | `signfix_ckpt4_ref` | `pad` | 40 | 70758 | 25020 | current reference |
| `targetcam30_raw_headface_hardmask_reproj` | `signfix_ckpt4_cropmode` | `crop` | 40 | 70758 | 25020 | identical ROI; not pass |

### Evidence
- Modal output:
  `D:/vggt/vggt-main/output/modal_results/20260426_6v_targetcam30_raw_headface_hardmask_reproj_signfix_ckpt4_cropmode/predictions.npz`.
- Open3D face render:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_targetcam30_raw_headface_ckpt4_cropmode_face/face_close.png`.
- Open3D head render:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_targetcam30_raw_headface_ckpt4_cropmode_head/head_close.png`.
- RGB/mask input audit:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/targetcam30_raw_headface_scene_rgb_mask_audit.png`.

### Diagnosis Update
- The targetcam30 input mask covers the visible human region and does not obviously cut out the central face. The failure is therefore not explained by a simple segmentation hole.
- The key blocker remains sparse-view geometry/surface continuity under the current VGGT confidence and point prediction, not the `image_mode` switch.

## 2026-04-27 Addendum: True-1024 Full-Human PSHuman Gate
### Verdict
- Mentor-final bar is still **not reached**.
- A true high-resolution full-human PSHuman input was built from raw human crop/mask pixels at `1024x1024`; the PSHuman mesh generation itself is therefore real, not the earlier accidental `518x518` resize path.
- Fusing the true-1024 PSHuman mesh into the targetcam30 sparse-view prediction gives only tiny face-ROI count changes and does not visibly repair the Open3D face/head holes. The direct variant adds fragments, while conservative variants are visually equivalent to the baseline.
- Because the visual gate failed, no sparse-view VGGT training should be launched from this teacher as-is.

### Bug Record
- Earlier run `pshuman_official_targetcam30_truehighres_human_hq1024_s40_r700_c200` was mislabeled as high-resolution but still used `518x518` inputs due to a resize helper bug.
- `tools/build_highres_headface_scene.py` was fixed so `--target-size` is honored, and the corrected scene `output/4k4d_preprocessed_scene_variants/0012_11_frame0000_60views_targetcam30_truehighres_human_hardmask_1024` was verified with `1024x1024` images and masks.

### Teacher / Raycast Facts
- Corrected PSHuman output:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/pshuman_official_targetcam30_true1024_human_hq1024_s40_r700_c200`.
- PSHuman mesh:
  `result_clr_scale4_00_tgt_cam30.obj`, `198387` vertices and `396798` faces.
- Raycasted training case:
  `D:/vggt/vggt-main/output/training_cases/0012_11_frame0000_6views_targetcam30_pshuman_true1024_human_headface_r1`.
- Raycast diagnostics:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/pshuman_true1024_human_headface_raycast_r1`.
- Raycast hit pixels were `25280`, only modestly above the old 518 full-body PSHuman teacher hit count (`24040`); this is not enough by itself to prove a better geometry teacher.

### ROI Results
| Protocol | Run | Conf percentile | Head ROI points | Face ROI points | Truthful interpretation |
|---|---:|---:|---:|---:|---|
| `targetcam30_raw_headface_hardmask_reproj` | `signfix_ckpt4_ref_open3d_local` | 40 | 66000 | 23331 | local Open3D baseline for this render/gate setup |
| `targetcam30_raw_headface_hardmask_reproj` | `pshuman_true1024_human_direct_a100_d008_noboost` | 40 | 66000 | 23629 | `+298` count, but visual adds fragments rather than clean eyes/nose/mouth structure; not pass |
| `targetcam30_raw_headface_hardmask_reproj` | `pshuman_true1024_human_half_a050_d006_noboost` | 40 | 66000 | 23566 | small count gain; visual remains baseline-like and holey |
| `targetcam30_raw_headface_hardmask_reproj` | `pshuman_true1024_human_xz050_y000_d006_noboost` | 40 | 66000 | 23509 | small count gain; no visible quality win |
| `targetcam30_raw_headface_hardmask_reproj` | `pshuman_true1024_human_xz035_y000_d004_noboost` | 40 | 66000 | 23431 | near-baseline; no visible quality win |

### Evidence
- Face comparison sheet:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/pshuman_true1024_human_face_close_compare_sheet.png`.
- Head comparison sheet:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/pshuman_true1024_human_head_close_compare_sheet.png`.
- Camera-view crop comparison sheet:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/pshuman_true1024_human_camera_view_00_crop_compare_sheet.png`.
- Best direct face render:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/open3d_20260427_pshuman_true1024_direct_a100_d008_face/face_close.png`.
- Best direct head render:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/open3d_20260427_pshuman_true1024_direct_a100_d008_head/head_close.png`.
- Baseline face render:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/open3d_20260427_pshuman_baseline_signfix_ckpt4_face/face_close.png`.
- Baseline head render:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/open3d_20260427_pshuman_baseline_signfix_ckpt4_head/head_close.png`.

### Diagnosis Update
- The true-1024 full-human PSHuman route is a cleaner teacher-generation attempt than the earlier accidental 518 route, but it still does not supply a continuous, aligned head/face surface after raycast/fusion.
- The failure is not a confidence-collapse artifact: all variants preserve the baseline `conf_threshold=31.246`.
- The next credible route should diagnose teacher visibility/alignment and continuous target-view surface coverage before more cloud training. A teacher that cannot pass the direct Open3D face/head gate should not be used to claim sparse-view geometry improvement.

## 2026-04-27 Addendum: External Prior Scene Bridge Smoke And 2D-ROI Open3D Audit
### Verdict
- Mentor-final sparse-view head/face bar is still **not reached**.
- The real-data bridge is now end-to-end smoke-tested: an external SMPL-X prior bundle can be converted into a scene-level `prior_maps.npz`, uploaded to Modal, and consumed by prior-enabled VGGT inference.
- This closes an engineering gap for the real-data route, but it is not a geometry-quality pass and does not prove 6-view face detail.
- A new Open3D `--roi-source 2d` path gives clearer camera-aligned face/head visualization, but it is a visualization/ROI correction only; it must not be counted as model-quality improvement.

### External Prior Bridge Evidence
- Bridged scene:
  `D:/vggt/vggt-main/output/smoke_external_bundle_case/scene_with_external_prior_bridge`.
- Modal inference output:
  `D:/vggt/vggt-main/output/modal_results/20260424_smoke_external_prior_scene_bridge_ckpt4`.
- Modal summary:
  `D:/vggt/vggt-main/output/modal_results/20260424_smoke_external_prior_scene_bridge_ckpt4/summary.json`.
- Inference confirmed `num_images=2`, `input_tensor_shape=[2,3,518,518]`, `prior_tensor_shape=[2,30,518,518]`, and `prior_summary_tensor_shape=[2,16,27]`.

### Visible-Surface Teacher Gate
- Batch audit file:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/visible_surface_audit_batch_summary.json`.
- All audited existing teacher candidates failed the visible-surface gate.
- Representative failures:
  - `pshuman_src34_targetcam30`: `roi=14524`, `compat=230`, `hole=0.984`, `median_residual=0.0472m`.
  - `pshuman_src00_targetcam30`: `roi=14524`, `compat=420`, `hole=0.971`, `median_residual=0.0440m`.
  - `pshuman_hq1024_targetcam30`: `roi=14524`, `compat=166`, `hole=0.989`, `median_residual=0.0376m`.
  - `pifuhd512_targetcam30`: `roi=14524`, `compat=3214`, `hole=0.779`, `median_residual=0.0136m`.
- Gate interpretation: existing external mesh teachers do not provide a continuous, depth-compatible target-view face-core surface. Training on them would likely optimize toward broken or partial supervision.

### 2D ROI Open3D Visualization Audit
- Renderer change:
  `D:/vggt/vggt-main/tools/render_open3d_pointcloud.py` now supports `--roi-source 2d`.
- Original 6-view signfix ckpt4 2D face ROI:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/open3d_original6v_signfix_ckpt4_2droi_face/open3d_summary.json`.
  - `roi_source=2d_mask`, `points_after_roi=43841`, `conf_threshold=43.857`.
- Targetcam30 signfix ckpt4 2D face ROI:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/open3d_ckpt4_2droi_face_conf40/open3d_summary.json`.
  - `roi_source=2d_mask`, `points_after_roi=64046`, `conf_threshold=28.349`.
- Original 6-view comparison sheet:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/original6v_2droi_face_compare_sheet.png`.
- Visual conclusion: the 2D ROI render is more interpretable and better aligned with the visible human face, but the baseline and PIFuHD residual variants remain effectively the same; there is still no clear eyes/nose/mouth surface repair.

### Next Gate
- Do not launch another large sparse-view end-to-end run until a candidate teacher first passes the visible-surface gate.
- Required teacher pre-gate before training:
  - face-core hit pixels `>=11000`;
  - largest connected component `>=0.80`;
  - hole ratio `<=0.15`;
  - median depth residual `<=0.012m`.
- After teacher gate pass, run only a small `detail_normal_refiner` ROI overfit first, then verify with Open3D head/face close-ups before any multi-case or sparse-view end-to-end training.

## 2026-04-27 Addendum: Depth-Anything Prior Negative Check
### Verdict
- Mentor-final sparse-view head/face bar is still **not reached**.
- A pre-existing Depth-Anything-V2-Small prior scene and ckpt4 Modal inference were rendered through the new 2D ROI Open3D path.
- The route does not improve face/head point-cloud quality. It keeps the same 2D face ROI point count as the targetcam30 baseline and visually makes facial detail smoother/more ambiguous rather than clearer.

### Evidence
- Depth-Anything external teacher:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260425/external_depth_teacher/targetcam30_6v_depthanything_v2`.
- Depth-Anything prior scene:
  `D:/vggt/vggt-main/output/4k4d_preprocessed_scene_variants/0012_11_frame0000_6views_targetcam30_depthanything_head_face_prior`.
- Modal inference:
  `D:/vggt/vggt-main/output/modal_results/20260425_6v_targetcam30_depthanything_prior_ckpt4`.
- Open3D face render:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/open3d_depthanything_prior_ckpt4_2droi_face/face_close.png`.
- Open3D head render:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/open3d_depthanything_prior_ckpt4_2droi_head/head_close.png`.
- Face comparison sheet:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/depthanything_prior_2droi_face_compare_sheet.png`.

### ROI Results
| Protocol | Run | Conf percentile | Face ROI points | Conf threshold | Truthful interpretation |
|---|---:|---:|---:|---:|---|
| `targetcam30_raw_headface_hardmask_reproj` | `signfix_ckpt4_ref_2droi` | 40 | 64046 | 28.349 | current 2D ROI visualization baseline |
| `targetcam30_depthanything_head_face_prior` | `depthanything_prior_ckpt4_2droi` | 40 | 64046 | 33.539 | same point count; visual face is smoother/less detailed, not pass |

### Diagnosis Update
- Depth-Anything aligned prior is not a usable shortcut for mentor-quality face geometry in this setup.
- It should not be promoted to a teacher-training branch unless a stronger visible-surface/normal consistency test first shows a real head/face detail gain.

## 2026-04-27 Addendum: Local Surface-Completion Patch Negative Check
### Verdict
- Mentor-final sparse-view head/face bar is still **not reached**.
- A reproducible diagnostic post-process was added to test whether fast target-view surface completion can repair the Open3D face without launching another large training run.
- The tested variants do not pass the visual gate. Low-confidence VGGT inpainting adds floating/noisy fragments around the face/hair region; SMPL-prior surface patching is too coarse and sparse after visibility/alignment.
- Therefore this route cannot be promoted as a sparse-view geometry solution.

### Tool
- `D:/vggt/vggt-main/tools/patch_predictions_surface_completion.py`.
- The tool writes a patched `predictions.npz` plus `surface_completion_summary.json` and explicitly labels the result as a diagnostic post-process, not raw VGGT output.

### Tested Variants
| Protocol | Variant | Patch source | Confidence mode | 2D face ROI points | Conf threshold | Truthful interpretation |
|---|---:|---:|---:|---:|---:|---|
| `targetcam30_raw_headface_hardmask_reproj` | `signfix_ckpt4_ref_2droi` | raw VGGT | raw | 64046 | 28.349 | reference visualization baseline |
| `targetcam30_raw_headface_hardmask_reproj` | `surface_completion_face_lowconf_keep` | VGGT inpaint | keep | 64046 | 28.349 | adds fragments/noise; not pass |
| `targetcam30_raw_headface_hardmask_reproj` | `surface_completion_face_lowconf_floor` | VGGT inpaint | floor synthetic patch conf | 64046 | 34.847 | still noisy; confidence floor is synthetic, not a real quality pass |
| `targetcam30_raw_headface_hardmask_reproj` | `surface_completion_face_allroi_floor` | VGGT inpaint | floor synthetic patch conf | 64046 | 34.847 | smoother but loses reliable face detail; not pass |
| `targetcam30_raw_headface_hardmask_reproj` | `surface_completion_face_smpl_allroi_affine_floor` | affine-aligned SMPL prior | floor synthetic patch conf | 64046 | 28.376 | coarse/sparse prior patch; not pass |

### Evidence
- VGGT inpaint comparison sheet:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/surface_completion_targetcam30_face_2droi_compare_sheet.png`.
- SMPL affine prior face render:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/open3d_surface_completion_targetcam30_face_smpl_allroi_affine_floor_2droi_face/face_close.png`.
- Surface completion summaries:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/surface_completion_targetcam30_face_lowconf_keep/surface_completion_summary.json`,
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/surface_completion_targetcam30_face_lowconf_floor/surface_completion_summary.json`,
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/surface_completion_targetcam30_face_allroi_floor/surface_completion_summary.json`,
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/surface_completion_targetcam30_face_smpl_allroi_affine_floor/surface_completion_summary.json`.

### Diagnosis Update
- Simple local surface completion can fill or move points, but it does not produce reliable eyes/nose/mouth geometry.
- Synthetic confidence flooring can change thresholds and must not be treated as a real model confidence improvement.
- The next credible route is still to find or produce a continuous target-view teacher that first passes the visible-surface gate, then use it for a small ROI `detail_normal_refiner` overfit before any large sparse-view end-to-end training.

## 2026-04-27 Addendum: DepthPro Same-Protocol Negative Check
### Verdict
- Mentor-final sparse-view head/face bar is still **not reached**.
- DepthPro was tested on the original `6views_sparseproto_headshoulder_crop` protocol, not only the easier targetcam30 view.
- Conservative x/z-only fusion preserves the baseline confidence threshold but produces only tiny ROI count jitter and no visible Open3D improvement.
- Direct full fusion is visually worse: the face/head surface twists sideways and introduces clear shape artifacts.

### Evidence
- DepthPro teacher:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/original6v_headshoulder_depthpro_teacher`.
- Training-case diagnostics:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/original6v_headshoulder_depthpro_gate/depthpro_headface_d012_diagnostics/external_depth_training_summary.json`.
- Conservative comparison sheet:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/original6v_headshoulder_depthpro_gate/original6v_depthpro_conservative_3droi_face_comparison_sheet.png`.
- Direct fusion comparison sheet:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/original6v_headshoulder_depthpro_gate/original6v_depthpro_direct_3droi_face_comparison_sheet.png`.

### ROI Results
| Protocol | Run | Conf percentile | Face ROI points | Head ROI points | Conf threshold | Truthful interpretation |
|---|---:|---:|---:|---:|---:|---|
| `original_6v_headshoulder` | `signfix_ckpt4_ref_3droi` | 40 | 16825 | 40527 | 38.507 | current same-protocol reference; visual still not mentor-final |
| `original_6v_headshoulder` | `depthpro_xz020_y000_d020` | 40 | 16828 | 40527 | 38.507 | +3 points only; visually same as baseline |
| `original_6v_headshoulder` | `depthpro_xz035_y000_d020` | 40 | 16840 | 40527 | 38.507 | +15 points only; visually same as baseline |
| `original_6v_headshoulder` | `depthpro_xz050_y000_d035` | 40 | 16808 | 40527 | 38.507 | below baseline |
| `original_6v_headshoulder` | `depthpro_direct_full` | 40 | n/a | n/a | 38.507 | visibly distorted side/face geometry; not pass |

### Diagnosis Update
- The DepthPro teacher alignment is not stable enough for a geometry-quality pass on the original sparse-view protocol.
- Small ROI count changes at unchanged confidence threshold are measurement jitter unless the Open3D close-up visibly improves; here it does not.
- DepthPro should remain a diagnostic branch and should not receive large cloud training unless a future teacher pre-gate first shows continuous, depth-compatible head/face surface coverage.

## 2026-04-27 Addendum: Local Photometric Depth-Sweep Negative Check
### Verdict
- Mentor-final sparse-view head/face bar is still **not reached**.
- A same-protocol local multi-view photometric depth-sweep diagnostic was added to test whether the current signfix ckpt4 depth can be locally refined in the target face/head ROI without another large training run.
- The tool is useful as a gate because it preserves raw confidence and only updates accepted ROI pixels, but the tested configurations do not create a measurable or visible Open3D improvement.
- The best rendered same-protocol variant keeps the exact same `face=16825`, `head=40527`, and `conf_threshold=38.507` as the reference; visually the face close-up remains the same weak surface without clear eyes/nose/mouth detail.

### Tool
- `D:/vggt/vggt-main/tools/patch_predictions_photometric_depth_refine.py`.
- Camera convention is `world->cam`; the tool projects candidate target-view depths into source views, scores RGB/gradient consistency, and writes a diagnostic `predictions.npz`.
- Confidence policy is deliberately `unchanged`.
- World-points policy after the implementation fix is `only_accepted_target_roi_pixels_recomputed; all other world_points preserved`.

### Evidence
- Strict face-core gate:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/photometric_depth_refine_original6v_facecore_r1/photometric_depth_refine_summary.json`.
- Loose face-core gate:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/photometric_depth_refine_original6v_facecore_loose_nodepth/photometric_depth_refine_summary.json`.
- Loose face gate with corrected world-point preservation:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/photometric_depth_refine_original6v_face_loose_v2/photometric_depth_refine_summary.json`.
- Open3D face render for corrected loose face gate:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/photometric_depth_refine_original6v_face_loose_v2/open3d_face/face_close.png`.
- Open3D head render for corrected loose face gate:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/photometric_depth_refine_original6v_face_loose_v2/open3d_head/head_close.png`.

### ROI Results
| Protocol | Run | Accepted coverage | Face ROI points | Head ROI points | Conf threshold | Truthful interpretation |
|---|---:|---:|---:|---:|---:|---|
| `original_6v_headshoulder` | `signfix_ckpt4_ref_3droi` | n/a | 16825 | 40527 | 38.507 | current same-protocol reference; visual still not mentor-final |
| `original_6v_headshoulder` | `photometric_facecore_r1` | 0.160 | 16728 | 40527 | 38.507 | too little accepted coverage and face ROI decreases; not pass |
| `original_6v_headshoulder` | `photometric_facecore_loose_nodepth` | 0.338 | not promoted | not promoted | raw confidence unchanged | still far below gate coverage; not promoted to visual claim |
| `original_6v_headshoulder` | `photometric_face_loose_v2` | 0.341 | 16825 | 40527 | 38.507 | exact ROI tie with baseline and visual same; not pass |

### Diagnosis Update
- Local RGB consistency finds a shallow optimum near the existing ckpt4 surface, but it does not add reliable missing facial geometry.
- This makes the photometric sweep useful as a rejection gate: it says the current sparse-view image evidence alone is not enough to repair face detail by small depth perturbations.
- Do not launch large training from this photometric teacher unless a future version first passes both accepted-coverage thresholds and Open3D visual improvement.
- The next credible route remains a stronger continuous target-view teacher: high-quality mesh/surface fitting or an external human-specific method that passes the visible-surface gate before training.

## 2026-04-27 Addendum: LHM-MINI Mesh Teacher Gate
### Verdict
- Mentor-final sparse-view head/face bar is still **not reached**.
- LHM-MINI can now run end-to-end on Modal and export local PLY meshes, but the exported meshes do **not** pass the visible-surface gate for the original 6-view headshoulder protocol.
- The result is a teacher-candidate infrastructure success only, not a geometry-quality pass.

### New Tooling / Fixes
- `D:/vggt/vggt-main/modal_lhm_mesh_teacher.py` now runs LHM-MINI on Modal with cached LHM assets, PyTorch3D CUDA build support, and a runtime parsing fallback for missing SAM2/parsingnet.
- `D:/vggt/vggt-main/tools/build_external_mesh_raycast_training_case.py` now supports `--align-roi-kind`, so full-body/head-face alignment can be separated from the smaller patch ROI.

### Evidence
- Back-view LHM output:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/lhm_mini_mesh_teacher_00_tgt_cam00_v15/lhm_mesh_teacher_summary.json`.
- Front/side LHM output:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/lhm_mini_mesh_teacher_30_src_cam30_v01/lhm_mesh_teacher_summary.json`.
- Back-view visible-surface audit:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/lhm_mini_mesh_gate_00_tgt_cam00_v15/visible_surface_audit/visible_surface_teacher_audit_summary.json`.
- Front/side visible-surface audit:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/lhm_mini_mesh_gate_30_src_cam30_v01/visible_surface_audit/visible_surface_teacher_audit_summary.json`.

### Gate Results
| Candidate | Patch ROI | Align ROI | Audit view | Depth-compatible hits | Hole ratio | Largest component | Median residual | Truthful interpretation |
|---|---|---|---:|---:|---:|---:|---:|---|
| `00_tgt_cam00_v15` | `face_core` | `face_core` | 0 | 531 | 0.927 | 0.944 | 0.0046m | fails hit/hole thresholds; back-view face coverage is insufficient |
| `30_src_cam30_v01` | `face_core` | `face_core` | 3 | 505 | 0.937 | 0.778 | 0.0069m | fails hit/hole/component thresholds; projected face teacher is a thin strip |
| `30_src_cam30_alignall_v01` | `face_core` | `all` | n/a | n/a | n/a | n/a | n/a | raycast produced only 159 total face-core hits across all views |
| `30_src_cam30_alignheadface_v01` | `face_core` | `head_face` | n/a | n/a | n/a | n/a | n/a | raycast produced only 849 total face-core hits across all views |
| `LHM-MINI 15_src_cam15_sweep01` | `face_core` | `face_core` | 2 | 13 | 0.999 | n/a | 0.0064m | fails coverage catastrophically |
| `LHM-MINI 45_src_cam45_sweep01` | `face_core` | `face_core` | 4 | 1018 | 0.863 | n/a | 0.0062m | best MINI sweep still far below coverage gate |
| `LHM-MINI 59_src_cam59_sweep01` | `face_core` | `face_core` | 5 | 907 | 0.848 | n/a | 0.0057m | far below coverage gate |
| `LHM-500M-HF 30_src_cam30_v01` | `face_core` | `face_core` | 3 | 484 | 0.940 | 0.785 | 0.0056m | stronger LHM model still fails hit/hole/component thresholds |
| `LHM-1B-HF 30_src_cam30_probe01` | `face_core` | `head_face` | all views | 647 total raycast hits | n/a | n/a | n/a | stronger model still too sparse for face-core teacher |
| `LHM-1B-HF 30_src_cam30_probe01` | `face_core` | `all` | all views | 1171 total raycast hits | n/a | n/a | n/a | best LHM-1B alignment remains far below gate scale |

### Diagnosis Update
- LHM-MINI produces a mesh file, but after alignment its face/head surface is not continuous enough in the target sparse-view cameras.
- The `--align-roi-kind` split rules out the earlier failure being only a script artifact: face-core, head-face, and all-human alignment still fail to provide sufficient depth-compatible face coverage.
- A sweep over the visible sparse views and stronger `LHM-500M-HF` / `LHM-1B-HF` runs did not fix coverage. Do not fuse or train from these LHM targets unless a future mesh teacher first passes the visible-surface gate.

## 2026-04-27 Addendum: Original-Protocol True-1024 PSHuman Check
### Verdict
- Mentor-final sparse-view head/face bar is still **not reached**.
- A true `1024x1024` full-human PSHuman input was built from the original 6-view sparse scene, not from the targetcam30 variant, and PSHuman successfully generated a dense mesh for `30_src_cam30.png`.
- The resulting teacher is stronger than the LHM candidates in raw coverage, but it still fails the same visible-surface gate and should not be used as a final training target.

### Artifacts
- High-resolution input scene:
  `D:/vggt/vggt-main/output/4k4d_preprocessed_scene_variants/0012_11_frame0000_6views_sparseproto_truehighres_human_hardmask_1024`.
- PSHuman mesh output:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/pshuman_original6v_true1024_human_cam30_s40_r700_c200/result_clr_scale4_30_src_cam30.obj`.
- All-human aligned raycast teacher:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/pshuman_original6v_true1024_cam30_alignall_v01/diagnostics/external_mesh_raycast_teacher_summary.json`.
- Visible-surface audit:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/pshuman_original6v_true1024_cam30_alignall_v01/visible_surface_audit_view3/visible_surface_teacher_audit_summary.json`.

### Gate Result
| Candidate | Patch ROI | Align ROI | Audit view | Raw hits | Depth-compatible hits | Hole ratio | Largest component | Median residual | Truthful interpretation |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `pshuman_original6v_true1024_cam30` | `face_core` | `all` | 3 | 4017 | 1435 | 0.822 | 0.580 | 0.0058m | fails hit/hole/component thresholds; teacher covers face as disconnected fragments |

### Diagnosis Update
- The better PSHuman mesh still does not provide a continuous depth-compatible face-core surface in the original sparse-view cameras.
- The overlay shows disconnected left/right face fragments rather than a stable full face surface. This is not a valid teacher for claiming sparse-view geometry improvement.
- The next credible route must either improve the external mesh alignment/surface quality before fusion, or use a different teacher source that passes this gate first.

## 2026-04-27 Addendum: PSHuman Translation Refinement Still Fails Gate
### Verdict
- Mentor-final sparse-view head/face bar is still **not reached**.
- The best true-1024 PSHuman cam30 mesh was given an additional camera-coordinate translation search around the previous optimum.
- Coverage improved compared with the first alignment, but the teacher still fails the depth-compatible visible-surface gate by a wide margin; it must not be fused or used as a pass claim.

### Evidence
- Refined summary:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/pshuman_original6v_true1024_cam30_alignall_translation_refine_v02/mesh_translation_refine_summary.json`.
- Refined overlay:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/pshuman_original6v_true1024_cam30_alignall_translation_refine_v02/mesh_translation_refined_overlay.png`.

### Gate Result
| Candidate | ROI pixels | Best camera delta | Raw hits | Depth-compatible hits | Depth coverage | Depth hole ratio | Largest component | Median residual | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `pshuman_true1024_cam30_translation_v01` | 8058 | `[0.0, -0.04, -0.03]` | n/a | 2143 | 0.266 | 0.734 | 0.965 | 0.0052m | fail |
| `pshuman_true1024_cam30_translation_v02` | 8058 | `[0.01, -0.08, -0.05]` | 6158 | 2706 | 0.336 | 0.664 | 1.000 | 0.0053m | fail |

### Diagnosis Update
- The refinement reduced disconnectedness and increased raw coverage, but depth-compatible face-core coverage is still only about one third of the ROI.
- The hard gate remains: at least `5000` depth-compatible pixels and hole ratio `<=0.15`. The best result has `2706` pixels and hole ratio `0.664`.
- This confirms PSHuman alignment is a useful diagnostic direction but not yet a valid teacher for final sparse-view geometry.

## 2026-04-27 Addendum: 20260424 Long-Run Training Evaluation Negative
### Verdict
- Mentor-final sparse-view head/face bar is still **not reached**.
- The three 20260424 cloud training runs were fully evaluated on the same `6views_sparseproto_headshoulder_crop` protocol.
- None beats the reference `signfix ckpt4` under the official same-protocol ROI summary.
- Several local fixed-threshold Open3D checks showed that apparently high point counts are not real face detail; they are low-confidence or fragmented surface artifacts.

### Official Same-Protocol Reference
| Run | Conf percentile | Face ROI points | Head ROI points | Conf threshold | Truthful interpretation |
|---|---:|---:|---:|---:|---|
| `20260424_signfix_ckpt4_on6v_headshoulder` | 40 | 16825 | 40527 | 38.507 | current same-protocol reference; still below mentor-final visual bar |

### Official ROI Results
| Group | Checkpoint | Face ROI points | Head ROI points | Conf threshold | Truthful interpretation |
|---|---:|---:|---:|---:|---|
| `pointnormal_r3_mixed` | `ckpt0` | 16262 | 40527 | 16.855 | below reference |
| `pointnormal_r3_mixed` | `ckpt1` | 16461 | 40527 | 29.114 | below reference |
| `pointnormal_r3_mixed` | `ckpt2` | 16462 | 40527 | 41.374 | below reference |
| `pointnormal_r3_mixed` | `ckpt3` / `inference` | 16461 | 40527 | 48.433 | below reference |
| `pointnormal_r3_6vonly` | `ckpt0` | 16546 | 40527 | 50.791 | below reference |
| `pointnormal_r3_6vonly` | `ckpt1` | 16418 | 40527 | 81.069 | below reference |
| `pointnormal_r3_6vonly` | `ckpt2` / `inference` | 16463 | 40527 | 98.725 | below reference |
| `teachergeom_roi_combo` | `ckpt0` | 16336 | 40527 | 21.259 | below reference |
| `teachergeom_roi_combo` | `ckpt1` | 16369 | 40527 | 31.586 | below reference |
| `teachergeom_roi_combo` | `ckpt2` | 16350 | 40527 | 46.044 | below reference |
| `teachergeom_roi_combo` | `ckpt3` / `inference` | 16368 | 40527 | 54.380 | below reference |

### Evidence
- Aggregate JSON:
  `D:/vggt/vggt-main/reports/20260427_6views_sparseproto_headshoulder_crop_roi_summary_conf40.json`.
- Aggregate CSV:
  `D:/vggt/vggt-main/reports/20260427_6views_sparseproto_headshoulder_crop_roi_summary_conf40.csv`.
- Local quick Open3D ROI table:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/quick_roi_20260427_training_eval_table.json`.
- Fixed absolute-confidence ROI table:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/fixed_conf_roi_20260427_training_eval_table.json`.
- Reference fixed-threshold face visual:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/fixedconf385_open3d_20260424_signfix_ckpt4_on6v_headshoulder_face/camera_view_03_crop.png`.
- `pointnormal_r3_6vonly_ckpt2` fixed-threshold face visual:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/fixedconf385_open3d_cam3_20260427_pointnormal_r3_6vonly_ckpt2_on6v_headshoulder_face/camera_view_03_crop.png`.
- `teachergeom_roi_combo_ckpt3` fixed-threshold face visual:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/fixedconf385_open3d_20260427_teachergeom_roi_combo_ckpt3_on6v_headshoulder_face/camera_view_03_crop.png`.

### Diagnosis Update
- The cloud training variants did not improve the official same-protocol face ROI count.
- Local Open3D fixed-threshold checks are useful for exposing false positives: some variants keep more points at an absolute confidence threshold, but the camera-aligned face render becomes noisier or more fragmented rather than clearer.
- These runs should be treated as negative evidence, not as a mentor-ready improvement.

## 2026-04-27 Addendum: 60v Surface-Teacher Candidate Recheck
### Verdict
- Mentor-final sparse-view head/face bar is still **not reached**.
- Historical 60v surface-teacher / detail-prior candidates were rechecked under the original 6-view headshoulder scene.
- Some candidates show inflated p40 point counts because their confidence distribution collapses to `conf=1`; fixed absolute confidence and camera-aligned Open3D renders do not show mentor-grade face detail.

### Evidence
- Percentile ROI table:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/quick_roi_60vteacher_candidates_20260427.json`.
- Fixed absolute-confidence ROI table:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/fixed_conf_roi_60vteacher_candidates_20260427.json`.
- `external60v_surfacepose_facecore_geoonly` fixed-threshold face visual:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/fixedconf385_open3d_cam3_20260426_original6v_external60v_surfacepose_facecore_geoonly_inference_on6v_headshoulder_face/camera_view_03_crop.png`.
- `multiview_surface_prior_denseonly` fixed-threshold face visual:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/fixedconf385_open3d_20260425_original6v_headshoulder_multiview_surface_prior_denseonly_ckpt4_face/camera_view_00_crop.png`.

### Diagnosis Update
- The best weak signal remains `20260426_original6v_external60v_surfacepose_facecore_geoonly_inference_on6v_headshoulder`, which has more fixed-threshold points but still lacks clear eyes/nose/mouth structure and shows fragmented facial artifacts.
- It may justify a tightly controlled continuation experiment, but it does not justify a pass claim.
- Any continuation must still be judged by same-protocol fixed-threshold Open3D close-ups, not just point count.

## 2026-04-27 Addendum: PSHuman Alternate-View Teacher Gate Negative
### Verdict
- Mentor-final sparse-view head/face bar is still **not reached**.
- PSHuman official teacher was run on alternate original sparse-view source images, including `45_src_cam45.png`, `59_src_cam59.png`, and `15_src_cam15.png`.
- None passes the visible-surface teacher gate; therefore none should be fused or used as final sparse-view evidence.

### Evidence
- Aggregate gate summary:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/pshuman_original6v_true1024_altviews_gate_summary.json`.
- `cam45` mesh:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/pshuman_original6v_true1024_altviews_cam45_s40_r700_c200/result_clr_scale4_45_src_cam45.obj`.
- `cam59` mesh:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/pshuman_original6v_true1024_altviews_cam59_s40_r700_c200/result_clr_scale4_59_src_cam59.obj`.
- `cam15` mesh:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/pshuman_original6v_true1024_altviews_cam15_s40_r700_c200/result_clr_scale4_15_src_cam15.obj`.

### Gate Results
| Candidate | Align ROI | Audit view | Depth-compatible hits | Hole ratio | Largest component | Median residual | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| `cam45` | `all` | 3 | 1597 | 0.795 | 0.591 | 0.0058m | fail |
| `cam15` | `all` | 3 | 1351 | n/a | n/a | n/a | fail |
| `cam59` | `all` | 3 | 152 | n/a | n/a | n/a | fail |
| alternate views | `head_face` | 3 | ~0 | ~1.0 | n/a | n/a | fail |

### Diagnosis Update
- Alternate-view PSHuman meshes are not enough; even the best `cam45` case is far below the `5000` depth-compatible pixel requirement and has a disconnected largest component.
- This reinforces the gate rule: PSHuman is currently useful for probing alignment, but not yet a reliable sparse-view face teacher.

## 2026-04-27 Addendum: PSHuman Similarity Gate Close but Still Negative
### Verdict
- Mentor-final sparse-view head/face bar is still **not reached**.
- A stronger similarity search was added for the best PSHuman true-1024 cam30 mesh, including translation, uniform scale, yaw, pitch, and roll.
- The best result is much closer than pure translation, but it still fails the hard visible-surface teacher gate because the depth-compatible hole ratio remains above the allowed limit.
- A diagnostic no-boost fusion from the best similarity-refined mesh did not improve same-protocol Open3D face/head evidence.

### Tool
- `D:/vggt/vggt-main/tools/refine_mesh_similarity_for_visible_surface.py`.

### Evidence
- Final summary:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/pshuman_original6v_true1024_cam30_similarity_refine_v01/mesh_similarity_refine_summary.json`.
- Final gate text:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/pshuman_original6v_true1024_cam30_similarity_refine_v01/mesh_similarity_refine_gate_result.txt`.
- Expanded-search summary:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/pshuman_original6v_true1024_cam30_similarity_refine_v01/_similarity_pose_expanded/mesh_similarity_refine_summary.json`.
- Diagnostic no-boost fusion:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/pshuman_similarity_expanded_fuse_a100_d040_noboost/fusion_summary.json`.

### Gate Result
| Candidate | ROI pixels | Best transform | Raw hits | Depth-compatible hits | Depth hole ratio | Largest component | Median residual | Gate |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `similarity_pose_expanded` | 8058 | scale `1.05`, yaw `30`, pitch `-15`, roll `30` | 7894 | 5881 | 0.270 | 1.000 | 0.0050m | fail |
| `similarity_final` | 8058 | scale `1.30`, yaw `57`, pitch `-10`, roll `45` | 7996 | 6129 | 0.239 | 1.000 | 0.0064m | fail |

### Diagnostic Fusion Result
| Variant | Max distance | Alpha | Fixed-conf face ROI | Fixed-conf head ROI | Truthful interpretation |
|---|---:|---:|---:|---:|---|
| `a050_d012_noboost` | 0.012m | 0.50 | 16924 | 40742 | essentially unchanged from reference |
| `a100_d012_noboost` | 0.012m | 1.00 | 16924 | 40742 | essentially unchanged from reference |
| `a035_d040_noboost` | 0.040m | 0.35 | 16916 | 40742 | face ROI decreases; no visual pass |
| `a050_d040_noboost` | 0.040m | 0.50 | 16886 | 40742 | face ROI decreases; no visual pass |
| `a100_d040_noboost` | 0.040m | 1.00 | 16722 | 40742 | face ROI decreases; no visual pass |

### Diagnosis Update
- Similarity alignment is the strongest PSHuman teacher-gate signal so far, but it still leaves about one quarter of the face-core ROI without depth-compatible support.
- The no-boost fusion confirms that even the closer teacher alignment does not automatically improve final sparse-view point clouds.
- Continue only if the next teacher/alignment method can reduce the depth hole ratio below `0.15` without destroying semantic face alignment.

## 2026-04-27 Addendum: Geoonly Continuation Training Negative
### Verdict
- Mentor-final sparse-view head/face bar is still **not reached**.
- The weak-signal `external60v_surfacepose_facecore_geoonly` branch was continued for four more epochs and evaluated on the original `6views_sparseproto_headshoulder_crop` protocol.
- It produces many more retained points under low p40 confidence, but the Open3D face/head geometry is visibly worse: the face is torn, filled by coarse sheet-like surfaces, and still lacks reliable eyes/nose/mouth structure.
- Therefore this branch is a negative result, not a mentor-ready improvement.

### Evidence
- Training summary:
  `D:/vggt/vggt-main/output/modal_training_results/20260427_original6v_external60v_surfacepose_facecore_geoonly_continue_lr2e8_e4/run_summary.json`.
- Inference/eval status:
  `D:/vggt/vggt-main/output/logs/20260427_geoonly_continue_eval/eval_status.json`.
- Fixed-confidence ROI table:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/geoonly_continue_roi_eval_20260427.json`.
- P40 ROI table:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/geoonly_continue_p40_roi_eval_20260427.json`.
- `ckpt3` fixed-confidence face Open3D:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/fixedconf385_open3d_20260427_geoonly_continue_ckpt3_on6v_headshoulder_face/camera_view_03.png`.
- `ckpt3` fixed-confidence face close-up:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/fixedconf385_open3d_20260427_geoonly_continue_ckpt3_on6v_headshoulder_face/face_close.png`.

### ROI Results
| Run | P40 face ROI | P40 head ROI | P40 conf | Fixed-conf face ROI | Fixed-conf head ROI | Truthful interpretation |
|---|---:|---:|---:|---:|---:|---|
| `geoonly_continue_ckpt0` | 84639 | 212513 | 1.059 | 23956 | 66212 | more points but degraded geometry |
| `geoonly_continue_ckpt1` | 86398 | 212513 | 1.192 | 30637 | 74230 | low-confidence point inflation; not pass |
| `geoonly_continue_ckpt2` | 86536 | 212513 | 1.297 | 34308 | 79093 | sheet-like face artifacts; not pass |
| `geoonly_continue_ckpt3` / `inference` | 86543 | 212513 | 1.323 | 34904 | 81200 | highest count, visually worse than reference |

### Diagnosis Update
- This branch demonstrates the danger of optimizing toward an imperfect 60v/external surface teacher: the model can increase point retention while worsening face topology.
- The visual gate correctly rejects it. Do not continue this exact geoonly branch without a better teacher mask/surface gate.

## 2026-04-27 Addendum: Kinect True-Depth Gate Useful but Direct Fusion Negative
### Verdict
- Mentor-final sparse-view head/face bar is still **not reached**.
- The local 4K4D Kinect SMC is a real metric depth source and is now verified as a useful teacher-gate candidate.
- `Calibration/Kinect/*/RT` is consistent with `camera_to_world`, and depth values are millimeters.
- After projecting frame-0 Kinect depth into the original 6-view headshoulder scene and aligning the real target-camera coordinates to VGGT prediction coordinates, the gate coverage is good, but direct point fusion does **not** produce clearer eyes/nose/mouth or mentor-grade face detail.
- Therefore Kinect should be treated as a real-depth geometry/scale gate, not as the final high-detail face teacher.

### New Tool
- Kinect smoke exporter:
  `D:/vggt/vggt-main/tools/export_kinect_depth_smoke.py`.
- Kinect-to-scene teacher target builder:
  `D:/vggt/vggt-main/tools/build_kinect_depth_teacher_targets.py`.

### Evidence
- Compact SMC summary:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/kinect_smc_compact_summary.json`.
- Kinect smoke summary:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/kinect_depth_smoke/summary.json`.
- Similarity-aligned teacher gate:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/kinect_teacher_original6v_headface_similarity_gate/kinect_teacher_summary.json`.
- Axis-affine teacher gate:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/kinect_teacher_original6v_headface_axisaffine_gate/kinect_teacher_summary.json`.
- Direct-fusion 2D ROI visual comparison:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/kinect_direct_fuse_visual_compare/kinect_axis_fuse_face_close_comparison.png`.
- Direct-fusion 3D ROI visual comparison:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/kinect_direct_fuse_visual_compare/kinect_3droi_face_close_comparison.png`.

### Gate Results
| Gate | ROI | Per-view hit ratio | Target alignment median residual | p95 residual | Interpretation |
|---|---|---:|---:|---:|---|
| `similarity` | `head_face` | `0.657–0.787` | `0.0228` | `0.1034` | usable real-depth coverage, but coarse |
| `axis_affine` | `head_face` | `0.657–0.787` | `0.0165` | `0.0672` | tighter numeric alignment, still low-res |

### Direct-Fusion Result
| Variant | ROI protocol | Fixed-conf face ROI | Fixed-conf head ROI | Visual result |
|---|---|---:|---:|---|
| `signfix ckpt4` reference | 3D ROI, conf `38.5067` | `16825` | `40527` | current reference; still not final pass |
| `kinect_axis_a020` | 3D ROI, conf `38.5067` | `16908` | `40527` | tiny count increase; no face-detail breakthrough |
| `kinect_axis_a035` | 3D ROI, conf `38.5067` | `16992` | `40527` | more noisy/coarse surface; no eyes/nose/mouth clarity |
| `kinect_sim_a020` | 3D ROI, conf `38.5067` | `16917` | `40527` | tiny count increase; visually not pass |
| `kinect_sim_a035` | 3D ROI, conf `38.5067` | `16998` | `40527` | noisy coarse shell; visually not pass |

### Diagnosis Update
- Kinect confirms a truthful path to real geometry supervision, but its `576x640` depth is too coarse to directly supply PSHuman/HumanRAM-level face detail.
- Direct fusion tends to add a more complete but lower-detail head/shoulder shell, and it can wash out the sparse-view face structure instead of making it clearer.
- Keep Kinect as a depth/scale/visibility gate for higher-resolution 60v RGB/MVS or visual-hull teachers.
- The next credible path is a 60v RGB/mask/MVS or visual-hull teacher validated by Kinect depth, not another confidence or point-count trick.

## 2026-04-27 Addendum: Visual-Hull and COLMAP Teacher Gates Are Not Final Passes
### Verdict
- Mentor-final sparse-view head/face bar is still **not reached**.
- A 60-view mask visual hull and a 60-view known-camera COLMAP MVS teacher were both made into real candidate teacher artifacts.
- Both are useful diagnostic gates, but neither currently supplies PSHuman-level continuous, clean target-view face geometry.
- Direct fusion from these teachers either fails same-protocol ROI or introduces halo/noise/sheet artifacts in Open3D close-ups.

### New Tools
- Visual-hull smoke builder:
  `D:/vggt/vggt-main/tools/build_visual_hull_teacher_smoke.py`.
- COLMAP known-camera scene exporter:
  `D:/vggt/vggt-main/tools/export_colmap_known_camera_scene.py`.
- COLMAP fused point cloud mask-vote filter:
  `D:/vggt/vggt-main/tools/filter_colmap_fused_pointcloud_by_scene_masks.py`.
- COLMAP PatchMatch depth to sparse-scene teacher target bridge:
  `D:/vggt/vggt-main/tools/build_colmap_depth_teacher_targets.py`.

### Evidence
- Visual hull summary:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/visual_hull_smoke_r96_v086/visual_hull_summary.json`.
- Visual hull aligned raycast diagnostics:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/visual_hull_r96_headface_aligned_raycast_diag/external_mesh_raycast_teacher_summary.json`.
- Visual hull direct-fusion comparison:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/visual_hull_direct_fuse_visual_compare/visual_hull_3d_face_close_comparison.png`.
- COLMAP workspace:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/colmap_known_camera_60v_headshoulder_518_masked`.
- COLMAP projection gate:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/colmap_mvs_teacher_gate/colmap_mvs_projection_face_head_sheet.png`.
- COLMAP depth-map teacher gate:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/colmap_depthmap_teacher_gate/colmap_depthmap_face_head_sheet_6cams.png`.
- COLMAP direct-fusion Open3D comparison:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/colmap_depth_direct_fuse_open3d/colmap_depth_direct_fuse_face_head_comparison.png`.

### Quantitative Results
| Candidate | Protocol | Face ROI | Head ROI | Visual gate |
|---|---|---:|---:|---|
| `signfix ckpt4` reference | original 6v headshoulder, fixed conf `38.5067` | `16825` | `40527` | current best, still not final pass |
| `visual_hull_a020` | direct fusion, 3D ROI | `16889` | `40527` | outer shell/ring artifacts |
| `visual_hull_a035` | direct fusion, 3D ROI | `17028` | `40527` | silhouette shell, no face detail |
| `visual_hull_a050` | direct fusion, 3D ROI | `17170` | `40527` | stronger artifact, not pass |
| `colmap_depth_xz020` | direct fusion, 3D ROI | `16732` | `40527` | lower than reference |
| `colmap_depth_xz035` | direct fusion, 3D ROI | `16739` | `40527` | lower than reference |
| `colmap_depth_xz050` | direct fusion, 3D ROI | `16783` | `40527` | lower than reference |
| `colmap_depth_xyz020` | direct fusion, 3D ROI | `16723` | `40527` | lower than reference |

### Diagnosis Update
- Visual hull is a good coverage/silhouette constraint, but it cannot recover eyes, nose, mouth, or hairline detail.
- COLMAP PatchMatch depth maps have broad mask coverage, but the target-view depth contains holes, boundary tearing, and black halo artifacts.
- COLMAP fused point clouds can reach millions of points, yet projection shows noisy halos around the person and no reliable face-detail teacher.
- These gates should remain as validation/auxiliary teachers, not as main proof of sparse-view head/face quality.

## 2026-04-27 Addendum: Local Surface Completion and Existing 60v Surface-Prior Results
### Verdict
- Mentor-final sparse-view head/face bar is still **not reached**.
- Local ROI surface completion and existing 60v VGGT surface-prior variants can increase point counts, but the Open3D visual gate still rejects them.
- The dense-only surface-prior variant is the strongest numeric candidate found in existing outputs, but its face/head close-ups remain visually close to the reference rather than a mentor-level breakthrough.

### Evidence
- Local surface-completion comparison:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/surface_completion_open3d/surface_completion_face_head_3d_comparison.png`.
- Camera-aligned local surface-completion comparison:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/surface_completion_open3d_2droi/surface_completion_2droi_cam3_face_comparison.png`.
- Existing 60v surface-prior comparison:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/existing_60v_surface_prior_open3d/existing_60v_surface_prior_face_head_comparison.png`.
- Existing 60v surface-prior summary:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/existing_60v_surface_prior_open3d/existing_60v_surface_prior_summary.json`.

### Quantitative Results
| Candidate | Protocol | Face ROI | Head ROI | Visual gate |
|---|---|---:|---:|---|
| `signfix ckpt4` reference | original 6v headshoulder, fixed conf `38.5067` | `16825` | `40527` | current best, still not final pass |
| `surface_face_lowconf` | diagnostic post-process | `17216` | `40527` | fragmented camera-aligned face, not pass |
| `surface_face_all` | diagnostic post-process | `17053` | `40527` | more sheet/halo artifacts, not pass |
| `surface_head_lowconf` | diagnostic post-process | `17262` | `40527` | fragmented head/face, not pass |
| `surface_head_all` | diagnostic post-process | `16960` | `40527` | no clear facial structure gain |
| `60v_surface_prior` | existing prior-enabled inference | `16732` | `40421` | below reference |
| `60v_surface_prior_denseonly` | existing prior-enabled inference | `18465` | `46991` | numeric gain but visually same-level, not final pass |

### Diagnosis Update
- Point-count gains alone are not trustworthy; they can come from dense shells, ROI jitter, or low-detail surface filling.
- The visual gate remains decisive: the target face still lacks stable, readable eyes/nose/mouth geometry.
- The next route should not be another confidence/ROI trick. It needs a cleaner, continuous, view-aligned head/face teacher or a local multiview optimization that demonstrably improves target-view surface quality.

## 2026-04-27 Addendum: True-Highres / Targetcam30 Existing Runs Checked
### Verdict
- Mentor-final sparse-view head/face bar is still **not reached**.
- Existing `truehighres` and `targetcam30` runs were re-rendered with the same fixed confidence threshold (`38.5067`) to avoid p40 confidence-collapse artifacts.
- Some variants increase face/head ROI counts, but Open3D close-ups show severe broken sheets, holes, and disconnected head/face surfaces.
- These outputs are therefore negative or diagnostic only; they cannot be used as the mentor-final sparse-view result.

### Evidence
- Fixed-confidence comparison sheet:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/highres_fixedconf_open3d/highres_fixedconf_face_head_comparison.png`.
- Fixed-confidence summary:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/highres_fixedconf_open3d/highres_fixedconf_summary.json`.

### Quantitative Results
| Candidate | Scene/protocol note | Face ROI | Head ROI | Visual gate |
|---|---|---:|---:|---|
| `truehr_headface` | original 6v cams, truehighres headface hardmask | `16930` | `61606` | more points, but large holes/broken sheets |
| `truehr_headface_reproj` | original 6v cams, reproj prior | `13390` | `37474` | below reference and fragmented |
| `target30_truehr_headface_reproj` | targetcam30 protocol, not same as original baseline | `17045` | `45266` | broken target-view face/head |
| `target30_truehr_headshoulder_reproj` | targetcam30 protocol, not same as original baseline | `20467` | `53298` | numeric gain, but visibly hollow/torn |

### Diagnosis Update
- `truehighres` preprocessing alone is not sufficient because the model output is still `518x518`, and hardmask/reprojection priors can create discontinuities.
- Targetcam30 variants are useful as diagnostic views but are not directly comparable to the original `6views_sparseproto_headshoulder_crop` reference.
- Do not present these as a pass. The visual failure is obvious even when point counts rise.

## 2026-04-27 Addendum: Multi-View Consistent ROI Pointcloud Post-Process
### Verdict
- Mentor-final sparse-view head/face bar is still **not reached**.
- The new multi-view consistency filter can produce cleaner diagnostic head/face ROI point clouds, but the rendered close-ups are mostly back/side head and shoulder surfaces rather than readable front-face detail.
- This is a post-process artifact, not raw VGGT output, and it cannot be reported as a sparse-view reconstruction breakthrough.

### Evidence
- Comparison sheet:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/mv_consistent_roi/mv_consistent_roi_face_close_comparison.png`.
- Summary:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/mv_consistent_roi/mv_consistent_roi_comparison_summary.json`.
- Tool:
  `D:/vggt/vggt-main/tools/fuse_multiview_consistent_roi_pointcloud.py`.

### Quantitative Results
| Candidate | ROI source | Points after consistency | Points after postprocess | Visual gate |
|---|---|---:|---:|---|
| `face2d_v2_d025` | 2D face mask | `32844` | `31403` | mostly back/side head; no front-face detail |
| `face2d_v3_d035` | 2D face mask | `29282` | `28135` | cleaner but still no readable eyes/nose/mouth |
| `head2d_v2_d025` | 2D head mask | `56954` | `54035` | head/shoulder shape, not face-quality proof |
| `head2d_v3_d035` | 2D head mask | `51078` | `49146` | similar head/shoulder shape, not pass |
| `face2d_v2_d035_poisson` | 2D face mask + Poisson sampling | `37249` | `35665` / `22000` sampled | over-smoothed/noisy surface; not pass |

### Diagnosis Update
- Multi-view consistency is useful as a cleanup/gating diagnostic, but it does not create missing target-view face geometry.
- The method should stay in the evidence toolbox for filtering artifacts, not as the main model-quality claim.
- The next mainline still needs a stronger continuous, view-aligned head/face teacher or an optimization that improves the actual same-protocol target face surface.

## 2026-04-27 Addendum: Humancrop6v `+17` Re-Audit With Fixed Confidence and Camera-Aligned Views
### Verdict
- The earlier progress-pack figure `03_humancrop6v_plus17_visual_same_not_breakthrough.png` is **insufficient by itself** because it used the generic Open3D `face_close` / `head_close` presets and a 3D percentile ROI; that view is not a semantic target-camera face crop.
- After re-rendering with the same absolute confidence threshold as `signfix ckpt4` (`38.506744384765625`), the humancrop6v 3D face ROI is `16809`, not `16842`; the apparent `+17` came from using each run's own p40 threshold.
- Camera-aligned 2D face/head ROI renders do show a visible face from camera index `3`, but `humancrop6v ckpt0` is still almost identical to `signfix ckpt4`: the cheek/nose region remains sparse/noisy, the mouth/eye structure is not readable, and the face is not a continuous modeled surface.
- Therefore the previous negative verdict remains correct, but the evidence should be described more carefully: **not a mentor-level pass because the camera-aligned face remains unformed**, not merely because the old 3D close-up looked “same morphology.”

### Evidence
- Re-audit sheet:
  `D:/vggt/vggt-main/output/comparisons/20260427_humancrop6v_reaudit/humancrop6v_reaudit_fixedconf_camera_aligned_sheet.png`.
- Signfix camera-aligned face crop:
  `D:/vggt/vggt-main/output/comparisons/20260427_humancrop6v_reaudit/signfix_face_2d_fixed/camera_view_03_crop.png`.
- Humancrop camera-aligned face crop:
  `D:/vggt/vggt-main/output/comparisons/20260427_humancrop6v_reaudit/humancrop_face_2d_fixed/camera_view_03_crop.png`.
- Summary:
  `D:/vggt/vggt-main/output/comparisons/20260427_humancrop6v_reaudit/humancrop6v_reaudit_fixedconf_summary.json`.

### Quantitative Results
| Candidate | ROI source / view | Conf threshold | Points after ROI | Visual gate |
|---|---|---:|---:|---|
| `signfix ckpt4` | 3D face ROI | `38.5067` | `16825` | current reference, still not final pass |
| `humancrop6v ckpt0` | 3D face ROI | `38.5067` | `16809` | no longer a numeric improvement under fixed threshold |
| `signfix ckpt4` | 2D face ROI | `38.5067` | `44963` | camera03 shows face, but sparse/noisy |
| `humancrop6v ckpt0` | 2D face ROI | `38.5067` | `45489` | slightly denser but still no readable eyes/nose/mouth surface |
| `signfix ckpt4` | 2D head ROI | `38.5067` | `82138` | current reference head evidence |
| `humancrop6v ckpt0` | 2D head ROI | `38.5067` | `82854` | small density gain, no qualitative face breakthrough |

### Diagnosis Update
- Future visual pass/fail checks must include camera-aligned target/source face views, not only generic Open3D 3D percentile close-ups.
- The pass criterion should be “formed, continuous, readable face geometry” rather than “slightly more points in a face ROI.”
- The humancrop6v branch remains useful as a diagnostic but should not be the main route unless it can improve camera-aligned face continuity, not just ROI count.

## 2026-04-27 Addendum: Kinect Direct-Fusion Re-Audit With Camera-Aligned Views
### Verdict
- Mentor-final sparse-view head/face bar is still **not reached**.
- Kinect is still valuable as a metric real-depth gate, but direct point fusion into the 6-view prediction does not improve the same-protocol camera-aligned face.
- Under fixed confidence `38.506744384765625`, 2D face/head ROI point counts remain unchanged because the fusion only changes point coordinates, not confidence. The visual gate rejects the result: conservative Kinect variants add speckle/noise around hair, cheeks, shirt, and arm without forming eyes/nose/mouth or a continuous face surface.

### Evidence
- Camera-aligned re-audit sheet:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/kinect_camera_aligned_reaudit/kinect_camera03_face_head_reaudit_sheet.png`.
- Summary:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/kinect_camera_aligned_reaudit/kinect_camera03_face_head_reaudit_summary.json`.
- Existing 3D ROI comparison:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/kinect_direct_fuse_visual_compare/kinect_axis_fuse_face_close_comparison.png`.

### Quantitative Results
| Candidate | ROI source / view | Conf threshold | Points after ROI | Visual gate |
|---|---|---:|---:|---|
| `signfix ckpt4` | 2D face ROI, camera03 | `38.5067` | `44963` | sparse/noisy reference |
| `kinect_axis_a020` | 2D face ROI, camera03 | `38.5067` | `44963` | more speckle; no formed face |
| `kinect_axis_a035` | 2D face ROI, camera03 | `38.5067` | `44963` | stronger noise; not pass |
| `kinect_xz050_y025` | 2D face ROI, camera03 | `38.5067` | `44963` | visibly worse/noisier |
| `signfix ckpt4` | 2D head ROI, camera03 | `38.5067` | `82138` | sparse reference |
| `kinect_axis_a020/a035/xz050` | 2D head ROI, camera03 | `38.5067` | `82138` | no qualitative gain; artifacts increase |

### Diagnosis Update
- Kinect should not be used as a direct point-coordinate patch in this form.
- If reused, it should be a gated supervision source for a small ROI refiner, with strict visible-surface masks and a pass/fail gate before any training.
- Direct fusion is now negative under both legacy 3D close-up and corrected camera-aligned visual criteria.

## 2026-04-27 Addendum: PIFuHD512 Residual Re-Audit With Camera-Aligned Views
### Verdict
- Mentor-final sparse-view head/face bar is still **not reached**.
- PIFuHD512 residual variants remain useful as an external-teacher diagnostic, but they do not yet provide a continuous, depth-compatible target-view face surface.
- The same-protocol 3D face ROI numbers can rise to `17152–17315`, but under fixed-confidence 2D camera-aligned rendering the face/head point counts stay unchanged and the visual face still lacks a readable eye/nose/mouth surface.

### Evidence
- Camera-aligned re-audit sheet:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/pifuhd_camera_aligned_reaudit/pifuhd_camera03_face_head_reaudit_sheet.png`.
- Summary:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/pifuhd_camera_aligned_reaudit/pifuhd_camera03_face_head_reaudit_summary.json`.
- Earlier 3D evidence sheet:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260426/open3d_multi3_truthful_evidence/signfix_vs_pifuhd512_multi3_truthful_evidence_sheet.png`.

### Quantitative Results
| Candidate | Metric / view | Value | Visual gate |
|---|---|---:|---|
| `signfix ckpt4` | 2D face ROI, camera03, fixed conf | `44963` | sparse/noisy reference |
| `pifuhd512_multi3_mean_v1` | 2D face ROI, camera03, fixed conf | `44963` | changed coordinates, but no formed face |
| `pifuhd512_single30` | 2D face ROI, camera03, fixed conf | `44963` | similar/no clear face continuity gain |
| `pifuhd512_multi3_mean_v2` | 2D face ROI, camera03, fixed conf | `44963` | similar/no clear face continuity gain |
| `signfix ckpt4` | 2D head ROI, camera03, fixed conf | `82138` | sparse reference |
| `pifuhd512 variants` | 2D head ROI, camera03, fixed conf | `82138` | no clear head/face breakthrough |

### Diagnosis Update
- The PIFuHD route should not be counted as a pass just because 3D percentile face ROI increases.
- Its main failure mode is teacher/geometry compatibility: local residuals can perturb the surface but do not create a stable, continuous target-view face model.
- Future use should be gated by camera-aligned face continuity first, then used as a teacher/refiner target only if the visible-surface gate passes.
