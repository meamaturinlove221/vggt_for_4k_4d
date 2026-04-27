# 20260424 Truthful Sparse-View Head/Face Status

Final status: **NOT_MENTOR_FINAL_PASS**

Quantitative and visual sparse-view head/face point-cloud quality still do not meet the mentor bar.

## Completed Evidence Checks
- `coarse_prior_pack_checklist`: checked and usable as a mentor-facing coarse-prior-normal pack; canonical and legacy advisor packs are machine-checked, the `4v probe` is isolated, 60/13/7 comparisons exist, and wording is fixed.
- `realdata_external_prior_bridge`: checked as repo infrastructure; external bundle -> scene `prior_maps.npz` -> Modal prior-enabled inference smoke completed.
- `detail_normal_refiner_roi_chain`: checked at normal-map ROI level; 60v -> 13v -> 7v head/face ROI refined-normal training/apply improves loss vs coarse prior, but this is not a sparse-view geometry pass.
- `open3d_visualization_chain`: checked as evidence infrastructure; Open3D head/face renders exist for reference, refined-prior, and headface crop variants.

## Not Enough / Failed
- `prior_only_refined_normals`: {"summary_token_variant_face_roi": 16816, "dense_only_variant_face_roi": 16828, "reference_best_face_roi": 16825, "verdict": "dense-only is only +3 and Open3D is visually same-shaped; not mentor-level."}
- `end_to_end_refinednormal_training`: {"ckpt0_face_roi": 16333, "ckpt1_face_roi": 16481, "ckpt2_or_inference_face_roi": 16500, "reference_best_face_roi": 16825, "verdict": "clear regression; cannot use."}
- `headface_crop`: {"face_roi_points": 21126, "verdict": "different crop protocol; Open3D has larger face scale but still missing/washed face geometry; not final pass.", "important_note": "current crop tool resizes source images to 518 before cropping, so it does not yet exploit original DNA high-resolution images."}

## Key Outputs
- `coarseprior_pack`: `output/normal_advisor_pack_20260421_coarseprior`
- `legacy_synced_pack`: `output/normal_advisor_pack_20260421`
- `detail_refiner_root`: `output/detail_normal_refiner_20260424`
- `roi_quant_report`: `output/detail_normal_refiner_20260424/roi_refinement_quant_report.json`
- `open3d_comparisons`: `output/detail_normal_refiner_20260424/open3d_compare`
- `headface_scene`: `output/4k4d_preprocessed_scene_variants/0012_11_frame0000_6views_sparseproto_headface_crop`

## Next Required
- Implement true high-resolution crop from original DNA RGB/mask before VGGT resize; current crop-after-518 is insufficient.
- Regenerate prior maps consistently for that high-res crop or reproject SMPL-X prior directly in cropped coordinates.
- Use detail_normal_refiner outputs as teacher/conditioning only after high-res crop is geometrically consistent.
- Then rerun 6-view inference and require both ROI increase and Open3D face/head close-up visual improvement before claiming pass.
