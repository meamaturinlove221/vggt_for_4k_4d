# 2026-04-22 Single-Case Overfit Eval Results

## Bottom line

The `6-view preprocess single-case overfit` line is **not** a mentor-final improvement path.

- `crop` overfit does not improve the `face ROI` point cloud.
- `softmatte` overfit does not improve the `face ROI` point cloud.
- In both cases, the trained checkpoint keeps similar overall/head occupancy but visually collapses the face crop toward a flatter, more silhouette-like slab.

This branch should be kept as an internal negative result, not promoted as the next main method direction.

## Remote checkpoints used

- crop:
  - `vggt_4k4d_train/20260422_6view_singlecase_crop_b40_a10080_r1/inference_model.pt`
- softmatte:
  - `vggt_4k4d_train/20260422_6view_singlecase_softmatte_b40_a10080_r1/inference_model.pt`

## Local eval outputs

- crop trained eval:
  - [20260422_crop_trained_eval](D:/vggt/vggt-main/output/modal_results/20260422_crop_trained_eval)
- softmatte trained eval:
  - [20260422_softmatte_trained_eval](D:/vggt/vggt-main/output/modal_results/20260422_softmatte_trained_eval)

## Trained-vs-untrained comparisons

- crop:
  - [20260422_crop_trained_vs_untrained_compare](D:/vggt/vggt-main/output/modal_results/20260422_crop_trained_vs_untrained_compare)
- softmatte:
  - [20260422_softmatte_trained_vs_untrained_compare](D:/vggt/vggt-main/output/modal_results/20260422_softmatte_trained_vs_untrained_compare)

Aligned diff summary against the matching untrained variant:

| Variant | Depth MAE | World-point L2 mean | Normal angle mean | Translation L2 mean |
| --- | ---: | ---: | ---: | ---: |
| `crop trained vs crop untrained` | `0.0470` | `0.0986` | `59.57 deg` | `0.0110` |
| `softmatte trained vs softmatte untrained` | `0.0361` | `0.0857` | `59.27 deg` | `0.0136` |

These deltas confirm that the trained checkpoints change the geometry significantly, but they do **not** show a quality gain on the region we actually care about.

## Open3D ROI renders

- trained renders:
  - [open3d_compare](D:/vggt/vggt-main/output/overfit_trained_eval_20260422/open3d_compare)
- face comparison sheet:
  - [compare_face_close_untrained_vs_trained.png](D:/vggt/vggt-main/output/overfit_trained_eval_20260422/compare_panels/compare_face_close_untrained_vs_trained.png)
- head comparison sheet:
  - [compare_head_close_untrained_vs_trained.png](D:/vggt/vggt-main/output/overfit_trained_eval_20260422/compare_panels/compare_head_close_untrained_vs_trained.png)

## ROI point counts

| Variant | Full-body points | Head ROI points | Face ROI points |
| --- | ---: | ---: | ---: |
| `crop` untrained | `111,094` | `24,441` | `11,523` |
| `crop` trained | `111,078` | `24,437` | `10,952` |
| `softmatte` untrained | `151,734` | `33,382` | `15,127` |
| `softmatte` trained | `151,726` | `33,380` | `14,068` |

The trained checkpoints do not increase usable head/face occupancy. `face ROI` goes down for both branches.

## Visual read

- `crop` untrained still preserves a clearer volumetric front-face slab than `crop` trained.
- `softmatte` untrained still preserves a clearer facial depth gradient than `softmatte` trained.
- Both trained variants flatten the `face ROI`, especially in the `face_close` renders.

## Decision

Keep:

- `human_crop` as the stable preprocess default.
- `human_crop_softmatte` as a high-occupancy branch worth keeping for ablation.
- `coarse prior normal + ROI-first detail_normal_refiner` as the stronger geometry direction.

Do not keep as main line:

- `single-case preprocess overfit` as a supposed sparse-view quality improvement branch.
