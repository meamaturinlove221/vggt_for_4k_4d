# 2026-04-21 Sparse-View Detail Normal Status

## Bottom Line

- `6-view` 全图 end-to-end `normal head` 仍未达到导师目标。
- 当前更有效的方向已经切换为 `coarse prior normal + ROI-first detail_normal_refiner`。
- 这条 ROI residual refinement 链已经完成：
  - 本地正式训练脚本
  - Modal 云端训练脚本
  - checkpoint 应用/跨 view 评估脚本
  - `60v train -> 6v val` 的 A100 实际训练与本地回收

## What Failed

### 1. Global 6-view normal head still collapses

`output/modal_results/20260421_6views_sparseproto_from_singlecase_overfit_b40/predictions.npz`

- `normal_mean = [0.2653, 0.7361, -0.6232]`
- `normal_std = [0.0592, 0.0132, 0.0452]`
- `normal_conf_mean = 0.4587`
- `normal_conf_std = 0.0010`

Interpretation:

- 法向图几乎是常值场。
- 头部 ROI 点云比 smoke 更连贯，但离“脸部高细节点云”仍很远。

`output/modal_results/20260421_6views_sparseproto_from_singlecase_overfit_b200/predictions.npz`

- `normal_mean = [0.0120, 0.3952, -0.9087]`
- `normal_std = [0.0203, 0.0248, 0.0201]`
- `normal_conf_mean = 0.0060`
- `normal_conf_std = 0.0233`

Interpretation:

- `b200` 没有把几何再往导师目标推进，反而进一步退化。
- 继续沿“全图 sparse-view normal head 硬加训”烧云算力，不是当前最优路径。

## What Now Works

### 1. Formal ROI refiner training pipeline

New files:

- `tools/train_detail_normal_refiner.py`
- `modal_detail_normal_refiner.py`
- `tools/apply_detail_normal_refiner.py`

This pipeline fixes the current technical positioning:

- not replacing VGGT
- not replacing coarse prior normal
- refining image-aligned local detail on top of coarse prior normal

### 2. A100 cloud training completed

Train:

- dataset: `output/detail_normal_refiner_20260421/dataset_export_60v/head_roi/head_samples.npz`

Val:

- dataset: `output/detail_normal_refiner_20260421/dataset_export_6v_teacher60/head_roi/head_samples.npz`

Remote run:

- local mirror: `output/detail_normal_refiner_20260421/remote_head_60to6v_e50`

Best metrics:

- `best_epoch = 50`
- `best_val_loss = 0.08249`
- `best_val_cosine = 0.04210`
- `best_val_edge = 0.04677`
- `best_val_mask_restricted = 0.28705`
- `hairline_cosine = 0.39411`
- `ear_band_cosine = 0.19635`

Qualitative read:

- refined normal is visibly closer to teacher than coarse prior
- head boundary and neck/shoulder transition are cleaner
- side/back views also stay stable after download to local

Representative local outputs:

- `output/detail_normal_refiner_20260421/remote_head_60to6v_e50/best_val/visuals/00_00_tgt_cam00_summary_strip.png`
- `output/detail_normal_refiner_20260421/remote_head_60to6v_e50/best_val/visuals/02_15_src_cam15_summary_strip.png`
- `output/detail_normal_refiner_20260421/remote_head_60to6v_e50/best_val/visuals/03_30_src_cam30_summary_strip.png`

### 3. Face ROI cloud training also completed

Train:

- dataset: `output/detail_normal_refiner_20260421/dataset_export_60v_face/face_roi/face_samples.npz`

Val:

- dataset: `output/detail_normal_refiner_20260421/dataset_export_6v_face_teacher60/face_roi/face_samples.npz`

Remote run:

- local mirror: `output/detail_normal_refiner_20260421/remote_face_60to6v_e50`

Best metrics:

- `best_epoch = 42`
- `best_val_loss = 0.11291`
- `best_val_cosine = 0.06359`
- `best_val_edge = 0.03287`
- `best_val_mask_restricted = 0.41093`

Qualitative read:

- face ROI 任务明显更难，数值不如 head ROI 低。
- 但 side/front face views 仍然可以看到 refined normal 向 teacher 收敛。
- 这说明 `face-only refinement` 不是死路，但它还没有单独突破到导师最终要求。

Representative local outputs:

- `output/detail_normal_refiner_20260421/remote_face_60to6v_e50/best_val/visuals/00_00_tgt_cam00_summary_strip.png`
- `output/detail_normal_refiner_20260421/remote_face_60to6v_e50/best_val/visuals/02_15_src_cam15_summary_strip.png`
- `output/detail_normal_refiner_20260421/remote_face_60to6v_e50/best_val/visuals/03_30_src_cam30_summary_strip.png`

## Cross-View Evaluation With The Same Checkpoint

Checkpoint:

- `output/detail_normal_refiner_20260421/remote_head_60to6v_e50/best_model.pt`

### 7 views

- output: `output/detail_normal_refiner_20260421/eval_head_7v_teacherlocal`
- `loss_detail_normal_total = 0.08629`
- `loss_detail_normal_cosine = 0.04122`

### 13 views

- output: `output/detail_normal_refiner_20260421/eval_head_13v_teacherlocal`
- `loss_detail_normal_total = 0.09891`
- `loss_detail_normal_cosine = 0.04884`

### 6 views

- output: `output/detail_normal_refiner_20260421/eval_head_6v_teacher60`
- `loss_detail_normal_total = 0.08249`
- `loss_detail_normal_cosine = 0.04209`

### 12 views

- output: `output/detail_normal_refiner_20260421/eval_head_12v_teacher60`
- `loss_detail_normal_total = 0.08854`
- `loss_detail_normal_cosine = 0.04600`

### 20 views

- output: `output/detail_normal_refiner_20260421/eval_head_20v_teacher60`
- `loss_detail_normal_total = 0.09349`
- `loss_detail_normal_cosine = 0.04887`

Interpretation:

- 同一个 refiner checkpoint 在 `7 / 13 / 6 / 12 / 20` view 的 head ROI 上都保持稳定。
- view 数增加后没有出现“突然大幅变清晰”的跃迁。
- 这进一步支持当前判断：
  - 当前瓶颈更接近 `coarse prior + teacher` 的表达上限
  - 而不是简单把 view 数从 `6 -> 12 -> 20` 就能自动补齐脸部细节

Note:

- `7v / 13v` 这一组来自旧 sparse 目录里的 teacher predictions，不是新 `teacher60subset` 方案。
- 因此数值可以作为稳定性参考，但不适合和 `6v / 12v / 20v` 做完全严格的一一比较。

## What Is Still Not Mentor-Final

- 还没有达到“`6-view` 脸部点云细节足够清楚、方法有竞争力”的最终标准。
- refined normal 目前主要证明了：
  - `normal` 方向是对的
  - 但更适合走 `PIFuHD-style coarse-to-fine local refinement`
  - 而不是当前这版全图 sparse-view global normal head
- refined normal 还没有正式回接到 sparse-view 点云几何主链里，所以当前最强证据仍是：
  - normal refinement 成立
  - 但最终高质量 face point cloud 还没闭环到导师最终要求
- `face ROI` refiner 已经能训练和泛化，但还没有把脸部三维细节拉到类似 PSHuman / 导师预期的清晰等级。

## Recommended Next Steps

1. Keep the current main line on `detail_normal_refiner`, not the collapsed global `normal head`.
2. Add a tighter `face ROI` export/eval pack in addition to `head ROI`.
3. Apply the refined normal to the sparse geometry branch as an extra local teacher / constraint, instead of only visualizing normals.
4. If needed for report consistency, add `13v` and `7v` ROI export/eval using the same checkpoint path.
5. Continue using Open3D head/face ROI renders as the geometry acceptance gate.
