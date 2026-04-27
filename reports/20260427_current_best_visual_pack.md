# 2026-04-27 当前最佳可展示图包（truthful）

最终状态：**NOT_MENTOR_FINAL_PASS**。本页只整理当前可展示成果图与负结果证据入口，不把未达标结果包装成达标。

核心口径：**当前 sparse-view 6v head/face 点云质量仍未达到导师最终 bar**。正式科研报告见 `D:\vggt\vggt-main\reports\20260427_research_style_progress_report.md`；图注与 storyboard 见 `D:\vggt\vggt-main\output\mentor_progress_figures_20260427\figure_captions_research_style.md`。

## 图包入口

| 用途 | 汇报图 | 结论 |
|---|---|---|
| 总览索引 | `D:\vggt\vggt-main\output\mentor_progress_figures_20260427\00_current_best_visual_pack_index.png` | 汇总 coarse prior、signfix ckpt4、humancrop6v +17、负结果证据。 |
| coarse prior normal advisor pack 最佳入口 | `D:\vggt\vggt-main\output\mentor_progress_figures_20260427\01_coarse_prior_normal_storyboard_60v.png` | coarse prior normal 链路已可对导师展示，但只是 coarse-prior 证据，不是最终 sparse-view 几何 pass。 |
| 当前 sparse-view best：signfix ckpt4 | `D:\vggt\vggt-main\output\mentor_progress_figures_20260427\02_signfix_ckpt4_open3d_face_head.png` | 同协议参考为 face ROI `16825`、head ROI `40527`、p40 conf threshold `38.507`；视觉仍非导师最终质量。 |
| humancrop6v +17 对比 | `D:\vggt\vggt-main\output\mentor_progress_figures_20260427\03_humancrop6v_plus17_visual_same_not_breakthrough.png` | face ROI `16842` 仅比 `16825` 多 `+17`，Open3D 形态同型，**不构成导师级突破**。 |
| 失败/负结果缩略图入口 | `D:\vggt\vggt-main\output\mentor_progress_figures_20260427\04_negative_evidence_thumbnails_depthpro_pshuman_geoonly.png` | DepthPro、PSHuman、geoonly 至少各 1 个负证据入口；均不能作为 pass。 |
| 图包 manifest | `D:\vggt\vggt-main\output\mentor_progress_figures_20260427\visual_pack_manifest.json` | `missing_required_images = []`。 |

图包解释规则：每张图都必须同时讲清“说明了什么”和“不能说明什么”。尤其 Figure 2 只能证明 coarse prior normal 链已成立；它当前只是 SMPL-X view-aligned coarse prior，下一步是 image-aligned residual/detail refinement。

![current best visual pack index](../output/mentor_progress_figures_20260427/00_current_best_visual_pack_index.png)

## 1. Coarse Prior Normal Advisor Pack

![coarse prior normal storyboard](../output/mentor_progress_figures_20260427/01_coarse_prior_normal_storyboard_60v.png)

- 原始入口：`D:\vggt\vggt-main\output\normal_advisor_pack_20260421_coarseprior\final_coarse_prior_normal_pass_pack\figures\00_coarse_prior_normal_storyboard_60v.png`。
- 已闭环：60v coarse SMPL-X prior normal 可以稳定对齐并形成 advisor-facing pack；legacy mirror 也已同步。
- 未达标：这不是 VGGT sparse-view 最终点云质量证明；不能说 6-view 已经有 PSHuman/HumanRAM 级 face/head 细节。
- 对外措辞：可以说“coarse prior normal chain is established”，不能说“VGGT 已输出高质量最终 normal/geometry”。

## 2. 当前 Sparse-View Best：Signfix Ckpt4

![signfix ckpt4 open3d face head](../output/mentor_progress_figures_20260427/02_signfix_ckpt4_open3d_face_head.png)

- 参考预测：`D:\vggt\vggt-main\output\modal_results\20260424_signfix_ckpt4_on6v_headshoulder\predictions.npz`。
- 同协议 scene：`D:\vggt\vggt-main\output\4k4d_preprocessed_scene_variants\0012_11_frame0000_6views_sparseproto_headshoulder_crop`。
- 量化参考：face ROI `16825`，head ROI `40527`，p40 confidence threshold 约 `38.507`。
- 已闭环：Open3D face/head close-up 渲染链路可作为统一可视化门槛。
- 未达标：脸部仍缺少可靠眼、鼻、口、发际线和连续面部表面；这是当前 reference best，不是 mentor-final pass。

## 3. Humancrop6v +17：不能过度包装

![humancrop6v plus17 visual same](../output/mentor_progress_figures_20260427/03_humancrop6v_plus17_visual_same_not_breakthrough.png)

- 候选结果：`20260424_humancrop6v_ckpt0_on6v_headshoulder`，face ROI `16842`。
- 对比基线：`signfix_ckpt4_ref`，face ROI `16825`。
- 真实解读：`+17` 属于微小数值抖动，Open3D face/head 视觉形态基本同型。
- 结论：这张图必须标注为“视觉同型，不构成导师级突破”；不能作为 sparse-view 质量达标证据。

## 4. 失败/负结果证据入口

![negative evidence thumbnails](../output/mentor_progress_figures_20260427/04_negative_evidence_thumbnails_depthpro_pshuman_geoonly.png)

| 分支 | 证据源 | 负结果原因 |
|---|---|---|
| DepthPro | `D:\vggt\vggt-main\output\detail_normal_refiner_20260427\original6v_headshoulder_depthpro_gate\original6v_depthpro_direct_3droi_face_comparison_sheet.png` | conservative 只有 `+3/+15` 抖动，direct/full fusion 产生面部/头部扭曲。 |
| PSHuman true1024 | `D:\vggt\vggt-main\output\detail_normal_refiner_20260427\pshuman_true1024_human_face_close_compare_sheet.png` | mesh 生成链路真实可跑，但 visible-surface gate 失败；face coverage 碎片化。 |
| geoonly continuation | `D:\vggt\vggt-main\output\detail_normal_refiner_20260427\fixedconf385_open3d_20260427_geoonly_continue_ckpt3_on6v_headshoulder_face\camera_view_03_crop.png` | p40 点数可增加，但 confidence/表面质量恶化，Open3D face 呈 torn/sheet-like。 |

## 已闭环

- `coarse prior normal advisor pack`：可展示的 coarse-prior normal 证据包已成型。
- `external bundle / prior bridge`：外部 SMPL-X bundle 到 scene prior 再到 prior-enabled inference 的工程链路已 smoke-tested。
- `detail_normal_refiner ROI chain`：ROI normal 层面可训练、可评估，但尚未转化为最终 6-view 点云质量。
- `Open3D ROI visualization`：face/head close-up、固定阈值对比、负结果证据入口已可复用。
- `DepthPro / PSHuman / geoonly`：这些分支已有负结果证据，不能继续当作已成功方向宣传。

## 仍未达标

- 6-view sparse-view human geometry 的导师最终门槛仍未达到。
- 当前 reference best `signfix ckpt4` 只是“最好的同协议参考”，不是最终可发表质量。
- `humancrop6v +17`、DepthPro `+3/+15` 这类微小 ROI 增量没有视觉增益，不能算突破。
- PSHuman/PIFuHD/LHM/DepthPro 等外部 teacher 当前都没有提供连续、对齐、depth-compatible 的 face/head surface。
- geoonly 继续训练出现点数/置信度表面假象，视觉更差，不应原样继续。

## 下一步为什么转向 Kinect 真深度教师

- 现有失败共同点：外部 monocular depth/mesh teacher 在目标 sparse-view 相机中不够连续或不够对齐，容易变成碎片、扭曲、sheet-like surface 或数值抖动。
- Kinect 是同一 4K4D capture 的传感器真深度，具备 metric depth、mask、K/RT 标定，理论上比单目伪深度或单视角 mesh 更适合作为可见表面教师。
- 当前 Kinect smoke 已确认数据可读并导出了点云：`D:\vggt\vggt-main\output\detail_normal_refiner_20260427\kinect_depth_smoke\summary.json`，mask/range 后总点数 `90264`。
- 但 Kinect 也不能直接宣布成功：下一步必须先做 calibration/coordinate sanity、可见表面 gate、Open3D face/head 直观检查，再决定是否进入 ROI refiner 或小规模训练。

## 缺图检查

- 必需图：无缺失。
- manifest：`D:\vggt\vggt-main\output\mentor_progress_figures_20260427\visual_pack_manifest.json`。
- 说明：本支线只复用已有图片并用 PIL 拼 contact sheet；未重跑训练、未改模型/训练/工具代码。
