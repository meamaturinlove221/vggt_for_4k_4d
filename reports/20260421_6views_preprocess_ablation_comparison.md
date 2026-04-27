# 2026-04-21 Sparse-View Preprocess Ablation Comparison

Compared three modal output variants for the same 6-view case:

- `full`: `output/modal_results/20260421_6views_preprocess_full_b40`
- `human_crop`: `output/modal_results/20260421_6views_preprocess_crop_b40`
- `human_crop_hardmask`: `output/modal_results/20260421_6views_preprocess_crop_hardmask_b40`

Generated comparison artifacts:

- `output/modal_results/20260421_6views_preprocess_ablation_compare_b40/variant_summary.csv`
- `output/modal_results/20260421_6views_preprocess_ablation_compare_b40/per_view_metrics.csv`
- `output/modal_results/20260421_6views_preprocess_ablation_compare_b40/aligned_diff_summary_vs_baseline.csv`
- `output/modal_results/20260421_6views_preprocess_ablation_compare_b40/aligned_diff_per_view_vs_baseline.csv`
- `output/modal_results/20260421_6views_preprocess_ablation_compare_b40/preview_sheets/*.png`

## Variant-local summary

| Variant | Mean mask coverage | Mean crop area ratio | Foreground depth mean | Foreground point-conf mean | Foreground normal-conf mean | Mean fx | Mean translation norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full` | 0.0423 | 1.0000 | 0.9532 | 1.0242 | 0.45819 | 529.67 | 0.7185 |
| `human_crop` | 0.1150 | 0.1426 | 0.9700 | 1.0234 | 0.45838 | 766.69 | 0.7326 |
| `human_crop_hardmask` | 0.1150 | 0.1426 | 0.9988 | 1.0546 | 0.45834 | 840.61 | 0.7418 |

## Aligned relative-to-full deltas

Dense comparisons below were computed after inverse-mapping cropped outputs back into the `full` image coordinate frame and evaluating on the `full` human mask.

| Variant vs `full` | Depth MAE | Mean world-point L2 | Mean normal angle (deg) | Mean translation L2 |
| --- | ---: | ---: | ---: | ---: |
| `human_crop` | 0.0400 | 0.0453 | 0.3762 | 0.0554 |
| `human_crop_hardmask` | 0.0481 | 0.0504 | 0.4963 | 0.0966 |

Within this single case, `human_crop` stayed closer to `full` than `human_crop_hardmask` on the aligned depth, world-point, normal, and pose deltas, while both crop variants increased effective foreground occupancy and predicted focal length.

## Limitations

- This is a relative comparison across preprocess variants for one case, not an accuracy evaluation against ground truth.
- The crop variants were resized during preprocessing; aligned dense deltas depend on an inverse resize back to the `full` frame and are therefore approximate.
- Foreground stats are measured on each variant's own input mask, so they are useful for comparison but not identical to a shared canonical region.

## Softmatte Generator Validation

- Changed files: `tools/build_preprocessed_scene_variants.py`, `reports/20260421_6views_preprocess_ablation_comparison.md`
- Validation command: `python D:\vggt\vggt-main\tools\build_preprocessed_scene_variants.py --scene-dir D:\vggt\vggt-main\output\4k4d_scenes\0012_11_frame0000_6views_sparseproto --output-root D:\vggt\vggt-main\output\4k4d_preprocessed_scene_variants --variants human_crop_softmatte --softmatte-radius 4.0 --overwrite`
- Output directory: `D:\vggt\vggt-main\output\4k4d_preprocessed_scene_variants\0012_11_frame0000_6views_sparseproto_human_crop_softmatte`
- Key outputs: `scene_manifest.json`, `prior_maps.npz`, `rgb_contact_sheet.png`, `mask_contact_sheet.png`
- Sanity check: `masks/00_tgt_cam00.png` and `prior_maps.npz` view-0 silhouette channel both contained 24,020 intermediate alpha pixels, confirming a non-binary soft edge matte.
