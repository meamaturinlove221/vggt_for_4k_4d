# 4K4D Original VGGT Status

## What is complete

- The local `data_used_in_4K4D` root is now bridge-ready.
- `0012_11` manifest is green:
  - [0012_11_manifest.json](/D:/vggt/vggt-main/reports/dna_case_probe/0012_11_manifest.json)
  - `ready_for_rgb_bridge = true`
  - `ready_for_mask_bridge = true`
- Contact sheets exist for `frame 0`:
  - [rgb_contact_sheet.png](/D:/vggt/vggt-main/output/4k4d_scenes/0012_11_frame0000_7views/rgb_contact_sheet.png)
  - [mask_contact_sheet.png](/D:/vggt/vggt-main/output/4k4d_scenes/0012_11_frame0000_7views/mask_contact_sheet.png)
- Official VGGT inference has completed on Modal for both source-count baselines:
  - `6src` case: [0012_11_frame0000_7views](/D:/vggt/vggt-main/output/modal_results/0012_11_frame0000_7views)
  - `12src` case: [0012_11_frame0000_13views](/D:/vggt/vggt-main/output/modal_results/0012_11_frame0000_13views)
- Fused point-cloud exports now exist for both baselines:
  - `6src` pointcloud: [pointcloud](/D:/vggt/vggt-main/output/modal_results/0012_11_frame0000_7views/pointcloud)
  - `12src` pointcloud: [pointcloud](/D:/vggt/vggt-main/output/modal_results/0012_11_frame0000_13views/pointcloud)

## Current baseline runs

### 6src

- Scene package:
  - [scene_manifest.json](/D:/vggt/vggt-main/output/4k4d_scenes/0012_11_frame0000_7views/scene_manifest.json)
- Cloud output:
  - [summary.json](/D:/vggt/vggt-main/output/modal_results/0012_11_frame0000_7views/summary.json)
  - [predictions.npz](/D:/vggt/vggt-main/output/modal_results/0012_11_frame0000_7views/predictions.npz)
  - [pointcloud_summary.json](/D:/vggt/vggt-main/output/modal_results/0012_11_frame0000_7views/pointcloud/pointcloud_summary.json)
  - [fused_pointcloud_raw.ply](/D:/vggt/vggt-main/output/modal_results/0012_11_frame0000_7views/pointcloud/fused_pointcloud_raw.ply)
  - [fused_pointcloud_masked.ply](/D:/vggt/vggt-main/output/modal_results/0012_11_frame0000_7views/pointcloud/fused_pointcloud_masked.ply)
  - [fused_pointcloud_raw_views.png](/D:/vggt/vggt-main/output/modal_results/0012_11_frame0000_7views/pointcloud/fused_pointcloud_raw_views.png)
  - [fused_pointcloud_masked_views.png](/D:/vggt/vggt-main/output/modal_results/0012_11_frame0000_7views/pointcloud/fused_pointcloud_masked_views.png)
- Modal summary:
  - `num_images = 7`
  - `gpu = NVIDIA A100-SXM4-40GB`
  - `elapsed_seconds = 47.764`
  - output tensors include `extrinsic`, `intrinsic`, `depth`, `depth_conf`, `world_points`, `world_points_conf`
- Point-cloud summary:
  - `raw.valid_points_before_conf = 1,878,268`
  - `raw.points_written = 180,000`
  - `masked.valid_points_before_conf = 87,848`
  - `masked.points_written = 26,355`

### 12src

- Scene package:
  - [scene_manifest.json](/D:/vggt/vggt-main/output/4k4d_scenes/0012_11_frame0000_13views/scene_manifest.json)
- Cloud output:
  - [summary.json](/D:/vggt/vggt-main/output/modal_results/0012_11_frame0000_13views/summary.json)
  - [predictions.npz](/D:/vggt/vggt-main/output/modal_results/0012_11_frame0000_13views/predictions.npz)
  - [pointcloud_summary.json](/D:/vggt/vggt-main/output/modal_results/0012_11_frame0000_13views/pointcloud/pointcloud_summary.json)
  - [fused_pointcloud_raw.ply](/D:/vggt/vggt-main/output/modal_results/0012_11_frame0000_13views/pointcloud/fused_pointcloud_raw.ply)
  - [fused_pointcloud_masked.ply](/D:/vggt/vggt-main/output/modal_results/0012_11_frame0000_13views/pointcloud/fused_pointcloud_masked.ply)
  - [fused_pointcloud_raw_views.png](/D:/vggt/vggt-main/output/modal_results/0012_11_frame0000_13views/pointcloud/fused_pointcloud_raw_views.png)
  - [fused_pointcloud_masked_views.png](/D:/vggt/vggt-main/output/modal_results/0012_11_frame0000_13views/pointcloud/fused_pointcloud_masked_views.png)
- Modal summary:
  - `num_images = 13`
  - `gpu = NVIDIA A100-SXM4-40GB`
  - `elapsed_seconds = 54.931`
  - output tensors include `extrinsic`, `intrinsic`, `depth`, `depth_conf`, `world_points`, `world_points_conf`
- Local verification:
  - `predictions.npz` opens successfully and exposes readable `world_points`, `world_points_conf`, and `extrinsic`
- Point-cloud summary:
  - `raw.valid_points_before_conf = 3,488,212`
  - `raw.points_written = 180,000`
  - `masked.valid_points_before_conf = 159,346`
  - `masked.points_written = 47,804`

## Important note

- The original-VGGT forward baseline is complete for `0012_11 / frame 0` at both `6src` and `12src`.
- Fused point-cloud export is also complete for both baselines.
- The old `render_raw_compare.py` / `ghost_score` stack is not present in the current repo and is also not available in the referenced `F:` repo anymore.
- So the current stopping point is:
  - official forward baseline: done
  - fused `.ply` and static point-cloud renders: done
  - target-view reprojection render: not yet implemented here
  - old legacy `weight_native / pred_native / ghost` compare stack: not yet reintroduced here

## Visualization labels

- [previews](/D:/vggt/vggt-main/output/modal_results/0012_11_frame0000_7views/previews) and [previews](/D:/vggt/vggt-main/output/modal_results/0012_11_frame0000_13views/previews) are per-view prediction previews only:
  - `*_depth.png`: per-view depth visualization
  - `*_depth_conf.png`: per-view depth-confidence visualization
  - `*_point_conf.png`: per-view point-confidence visualization
- [fused_pointcloud_raw_views.png](/D:/vggt/vggt-main/output/modal_results/0012_11_frame0000_7views/pointcloud/fused_pointcloud_raw_views.png) and [fused_pointcloud_masked_views.png](/D:/vggt/vggt-main/output/modal_results/0012_11_frame0000_7views/pointcloud/fused_pointcloud_masked_views.png) are fused point-cloud static renders:
  - each image is a 3-panel comparison of `Front (X/Y)`, `Side (Z/Y)`, and `Top (X/Z)`
  - `raw` means all valid points after confidence filtering
  - `masked` means points additionally filtered by the exported 4K4D foreground masks
- The intended process-comparison set should be labeled explicitly as:
  - input RGB contact sheet
  - input mask contact sheet
  - per-view `depth / depth_conf / point_conf` previews
  - fused point cloud `raw`
  - fused point cloud `masked`
  - target-view reprojection result
  - legacy `weight_native / pred_native / ghost` compare images when that stack is restored

## Re-run commands

Run `6src`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_4k4d_vggt_modal_case.ps1 `
  -Seq 0012_11 `
  -Frame 0 `
  -TargetCamera 00 `
  -AutoSources 6 `
  -OverwriteScene
```

Run `12src`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_4k4d_vggt_modal_case.ps1 `
  -Seq 0012_11 `
  -Frame 0 `
  -TargetCamera 00 `
  -AutoSources 12 `
  -OverwriteScene
```

Render fused point cloud for `6src`:

```powershell
python .\tools\render_vggt_pointcloud.py `
  --predictions-npz .\output\modal_results\0012_11_frame0000_7views\predictions.npz `
  --scene-dir .\output\4k4d_scenes\0012_11_frame0000_7views `
  --output-dir .\output\modal_results\0012_11_frame0000_7views\pointcloud
```

Render fused point cloud for `12src`:

```powershell
python .\tools\render_vggt_pointcloud.py `
  --predictions-npz .\output\modal_results\0012_11_frame0000_13views\predictions.npz `
  --scene-dir .\output\4k4d_scenes\0012_11_frame0000_13views `
  --output-dir .\output\modal_results\0012_11_frame0000_13views\pointcloud
```
