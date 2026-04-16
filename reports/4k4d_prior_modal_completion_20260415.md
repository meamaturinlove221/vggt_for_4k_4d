# 4K4D Prior-Guided VGGT Completion Report

Date: 2026-04-15

## Completed items

- Human-prior feature injection code is landed in the VGGT model and training stack.
- 4K4D pseudo-training case preparation is landed and verified for both:
  - `0012_11_frame0000_7views_depth_prior`
  - `0012_11_frame0000_13views_depth_prior`
- Modal cloud training pipeline is landed and verified on A100.
- High-resolution fused point-cloud renders and labeled comparison sheets are generated locally.
- Modal apps were checked after completion and no active app remained.

## Final cloud training run

- Experiment: `4k4d_prior_dualcase_a10080_e2_b12`
- GPU: `NVIDIA A100 80GB PCIe`
- Cases:
  - `training_cases/0012_11_frame0000_7views_depth_prior`
  - `training_cases/0012_11_frame0000_13views_depth_prior`
- Epochs: `2`
- Train batches per epoch: `12`
- Val batches per epoch: `2`
- Local downloaded output:
  - `output/modal_training_results/dualcase_13fixed_a10080_e2_b12`

## Key local outputs

### Training

- `output/modal_training_results/dualcase_13fixed_a10080_e2_b12/run_summary.json`
- `output/modal_training_results/dualcase_13fixed_a10080_e2_b12/logs/log.txt`
- `output/modal_training_results/dualcase_13fixed_a10080_e2_b12/logs/tensorboard`

### High-resolution fused point clouds

- 7views world-points:
  - `output/modal_results/0012_11_frame0000_7views/pointcloud_dense_p40_hires`
- 13views world-points:
  - `output/modal_results/0012_11_frame0000_13views/pointcloud_dense_p40_hires`
- 7views depth-unprojection:
  - `output/modal_results/0012_11_frame0000_7views/pointcloud_depth_unprojection_dense_p40_hires`
- 13views depth-unprojection:
  - `output/modal_results/0012_11_frame0000_13views/pointcloud_depth_unprojection_dense_p40_hires`

### Labeled comparison sheets

- Masked comparison:
  - `output/comparisons/0012_11_frame0000_pointcloud_hires/masked_comparison_hires.png`
- Raw comparison:
  - `output/comparisons/0012_11_frame0000_pointcloud_hires/raw_comparison_hires.png`

## Point-count summary

- 7views, world-points, masked kept: `52,709`
- 13views, world-points, masked kept: `95,608`
- 7views, depth-unprojection, masked kept: `52,709`
- 13views, depth-unprojection, masked kept: `95,608`

These counts are the fused totals after stacking all selected views, not per-view counts.

## Important technical note

This landed variant uses projected image-space human priors:

- silhouette prior
- 2D keypoint heatmap prior

These priors are injected into the token stream through the `HumanPriorAdapter`, so the model does train with human-prior features present.

However, the current exported 4K4D prior case does **not** include SMPL-derived `prior_depths` or `prior_points`, so the explicit auxiliary `loss_human_prior / loss_prior_depth / loss_prior_point` terms remain zero in this run.

That means the completed implementation is:

- **done**: prior-feature fusion into VGGT + Modal A100 training pipeline
- **not yet added**: direct SMPL 3D prior supervision loss or explicit hole-filling by SMPL geometry

## Expected effect boundary

- This setup is suitable for improving body completeness and reducing some missing-body regions.
- Hair and very fine face detail remain hard and are not expected to be fully solved by this feature-only prior path.
