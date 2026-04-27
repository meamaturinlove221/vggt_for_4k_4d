# 2026-04-27 Sparse-View 人体高质量几何恢复阶段进展报告（科研流程版）

> 执行范围：本支线只在 Codex 内整理 `reports/` 与 `output/mentor_progress_figures_20260427/` 下的报告、图包索引、caption 与 manifest；未开启外部窗口，未重跑训练，未修改训练/推理代码。  
> 口径声明：本报告基于当前仓库产物与已有 truthful 状态重写，参考旧版 Yuque 风格的“问题—原理—流程—证据—下一步”叙事，但不引用外部不可验证内容。  
> 当前结论必须保持：**当前 sparse-view 6v head/face 点云质量仍未达到导师最终 bar**。工程闭环成立，但 face/head Open3D 点云仍不能包装为最终达标。

## 摘要

本阶段工作从“将 SMPL-X 接入 VGGT”推进到“sparse-view 人体高质量几何恢复的工程闭环”。已完成的闭环包括：SMPL/SMPL-X pose-aligned prior、scene-level dense `prior_maps`、input-side / layer-wise prior fusion、output-side depth / point / normal 监督、coarse prior normal advisor pack、ROI-first `detail_normal_refiner`、external SMPL-X real-data bridge、Open3D ROI 同协议评估，以及 DepthPro / PSHuman / geoonly 等外部 teacher 路线排查。

需要强调的是，这些成果解决了“人体先验如何接入 VGGT、如何监督、如何可视化、如何评估”的工程和科研流程问题；但最终目标“6-view sparse-view 下 head/face 点云清晰到导师认可”仍未达到。当前同协议参考仍是 `signfix ckpt4`，face ROI `16825`、head ROI `40527`、p40 confidence threshold 约 `38.507`；其 Open3D 视觉仍缺少可靠眼、鼻、口、发际线和连续脸部曲面。

![当前最佳图包总览](../output/mentor_progress_figures_20260427/00_current_best_visual_pack_index.png)

## 0. 导师 Checklist 与 Truthful 口径

本报告按“科研正规流程”组织：先给任务背景和导师验收标准，再说明原版 VGGT 局限、SMPL/SMPL-X 原理、架构路线、训练/评估闭环，最后用当前最好结果和负结果证据给出下一步计划。

必须对齐的导师口径如下：

1. **核心验收 bar**：6-view sparse-view 下，face/head 的 Open3D 点云必须能看清脸部细节；若看不清眼、鼻、口、脸部连续曲面和头部边界，就没有竞争力。
2. **当前 truthful state**：当前 sparse-view 6v head/face 点云质量仍未达到导师最终 bar；`signfix ckpt4` 只是当前同协议 reference best，不是 final pass。
3. **可说的正结果**：coarse prior normal 链已成立，但当前只是 SMPL-X view-aligned coarse prior；下一步是 image-aligned residual/detail refinement。
4. **外部方法定位**：PIFuHD / PSHuman / HumanRAM 只作为方向论据：coarse-to-fine、高分辨人体细节、pose-conditioned quality；不能说当前 VGGT 支线已经达到这些方法的 head/face 质量。
5. **禁止包装**：不得把 60-view coarse-prior 图讲成最终达标，不得把 4-view probe 讲成接近成功，不得把 coarse prior normal 讲成 VGGT 的高质量最终预测；也不能把 `humancrop6v +17`、DepthPro `+3/+15` 这类微小数值抖动讲成突破。

## 1. 研究背景与问题定义

### 1.1 背景

VGGT 类多视角几何模型擅长从多张图像恢复相机、深度、点云和三维结构。但在人体场景中，尤其是 sparse-view（例如 6-view）设置下，仅依赖 RGB 多视角匹配和通用几何先验，容易在以下部位出现不稳定：

- 人体头部与脸部：眼、鼻、口、耳、发际线等局部高频结构不足。
- 头发与衣物边界：非刚性、遮挡、纹理弱或视角覆盖不足时，点云容易破碎或漂浮。
- sparse-view 几何约束不足：6-view 相比 60-view 缺少足够视差和可见面覆盖。

导师提出的核心要求不是“只把 SMPL-X 接进去”，也不是“只展示 normal map”，而是要让 sparse-view 下的人体区域点云具备竞争力；尤其是 6-view head/face 需要展示可用的三维几何细节。如果 6-view 只能得到完整轮廓而没有清晰脸部结构，则方法仍不足以说明高质量 sparse-view 人体恢复成立。

### 1.2 问题定义

本阶段问题可以定义为：

> 在 VGGT 多视角几何框架内，引入 SMPL/SMPL-X pose-aligned human prior 和局部 detail normal / surface teacher，使 sparse-view 输入下的人体点云，尤其 head/face ROI，在同协议评估中同时获得量化和 Open3D 可视化提升。

同协议评估必须同时满足：

1. 在 `6views_sparseproto_headshoulder_crop` 场景上，face ROI 和 head ROI 明显优于当前 `signfix ckpt4` 参考，而不是 `+3/+15/+17` 级别的统计抖动。
2. Open3D face/head close-up 中能看出更连续、更清晰的脸部和头部几何，而不是更多噪声点、折叠面、碎片或 confidence collapse。

### 1.3 科研正规流程拆解

为了避免只展示工程截图，本阶段按以下科研流程组织证据：

1. **问题提出**：原版 VGGT 在 sparse-view 人体场景中缺少人体拓扑与局部高频先验，face/head 是最容易暴露问题的 ROI。
2. **假设建立**：如果把 SMPL/SMPL-X pose-aligned prior 转成 view-aligned dense surface maps，再加入局部 detail normal / visible-surface teacher，理论上可缓解 sparse-view 可见面不足。
3. **先验构建**：用 SMPL/SMPL-X 参数生成 mesh，再按每个相机 rasterize / raycast 出 prior depth、prior point、prior normal、prior mask。
4. **模型接入**：在 VGGT input-side / layer-wise 融合 prior maps 与 summary tokens，并在 output-side 增加 depth、point、normal、confidence-aware 监督。
5. **同协议评估**：固定 `6views_sparseproto_headshoulder_crop`、固定 confidence 协议和 Open3D face/head ROI close-up，防止不同 crop、不同阈值造成误判。
6. **负结果归档**：将 DepthPro、PSHuman、geoonly、humancrop6v 等分支按同一标准归档，明确哪些只证明“可跑通”，哪些不能证明“质量提升”。
7. **下一步收敛**：将主线收敛到 image-aligned residual/detail refinement，并用 Kinect 真深度或 60v RGB-MVS teacher 先过 visible-surface gate，再进入训练。

## 2. 原版 VGGT 定位与局限

原始 VGGT 路线主要从图像序列中学习相机、深度和点云。对于常规场景，这种通用几何范式具有较强泛化性；但在人类 sparse-view 重建中存在几个局限：

- **人体拓扑先验不足**：通用点云预测不知道人体头、脸、四肢、躯干之间的可变形但受约束结构。
- **稀疏视角下可见面不足**：6-view 中 face/head 可见区域有限，多视角一致性不足以恢复稳定高频曲面。
- **深度监督不足以表达表面方向**：单纯深度/点云 loss 可以约束位置，但对局部表面法向、曲率和边界细节约束较弱。
- **ROI 质量被全图平均稀释**：全图 loss 可能改善整体人体轮廓，却不一定改善导师最关心的 face/head 局部质量。
- **可视化误判风险**：点数增加、2D 投影更密或 confidence threshold 降低，均可能掩盖真实三维几何质量没有提升。

因此，本阶段选择的研究路线不是替换 VGGT，而是在 VGGT 主体上叠加人体 pose-aligned prior、dense surface maps、output-side 几何监督和 ROI-first detail refinement。

## 3. SMPL / SMPL-X 原理与定位

### 3.1 SMPL / SMPL-X 基本原理

SMPL / SMPL-X 是参数化人体模型。其核心思想是用低维参数控制一个具有固定拓扑的人体 mesh：

- **shape / betas**：控制个体体型差异，例如高矮胖瘦。
- **pose / fullpose**：控制各关节旋转，使 mesh 与当前动作对齐。
- **global translation / scale**：控制人体整体在场景坐标中的位置和尺度。
- **SMPL-X 扩展**：相较 SMPL，SMPL-X 通常包含更丰富的手、脸、表情等参数，更适合精细人体建模。

在当前任务中，SMPL/SMPL-X 的角色是“pose-aligned geometry prior”：它提供人体在每个相机视角下的大体位置、深度、表面方向和人体区域约束。

### 3.2 为什么不是直接回归 SMPL-X 参数

本仓库当前目标不是把 VGGT 改成一个 raw image -> SMPL-X 参数回归器，原因如下：

- **VGGT 的主任务仍是多视角几何恢复**：它需要处理背景、衣物、头发、非 SMPL 表面和相机几何，而不仅是裸体模板人体。
- **SMPL-X 本身不能表达全部真实表面**：衣物褶皱、头发体积、脸部真实几何和传感器可见表面都超出单一参数 mesh 的表达上限。
- **真实数据更适合外部估计器/拟合器接入**：后续 real-data 路线中，SMPL-X 可以来自外部 estimator/fitter，再由仓库导入、对齐、生成 prior，而不是要求仓库内部从零训练 SMPL-X 回归。
- **科研路径更清晰**：当前要验证的是“pose-aligned prior 是否能增强 sparse-view VGGT 几何”，而不是同时解决一个完整 SMPL-X regressor 的训练问题。

因此，当前路线是：

```text
外部或标注 SMPL/SMPL-X 参数
        ↓
相机对齐 / pose-aligned mesh
        ↓
dense prior maps / prior summary tokens
        ↓
VGGT input-side / layer-wise conditioning
        ↓
depth / point / normal output-side supervision
        ↓
Open3D ROI 同协议评估
```

## 4. Pose-Aligned Prior 总体框架

### 4.1 总体流程

本阶段形成的 pose-aligned prior 框架可分为五步：

1. **人体参数准备**：读取 4K4D annotation 或外部 SMPL-X bundle，获得 `betas`、`fullpose`、`transl`、`scale`、相机参数等。
2. **mesh 生成与场景对齐**：将 SMPL/SMPL-X mesh 放入与 sparse-view 图像一致的世界/相机坐标系。
3. **视角 rasterize / raycast**：对每个输入 view 生成 dense surface maps，包括 prior depth、prior point、prior normal、prior mask。
4. **VGGT 侧融合**：将 prior maps 作为空间条件，将 summary tokens 作为全局条件，参与 image tokens 的多层融合。
5. **输出侧监督与评估**：对 depth、camera/world points、normal 和 confidence 进行监督，并用 Open3D 统一评估 head/face ROI。

### 4.2 工程闭环意义

这个框架使“人体先验”不再停留在一张 mesh 或一张 normal 可视化，而是进入了完整科研流程：

- 数据层：scene-level `prior_maps.npz` 可被训练/推理 case 读取。
- 模型层：input-side / layer-wise prior fusion 可以给 VGGT 提供人体结构条件。
- 监督层：output-side loss 能把 prior depth / points / normals 写入训练目标。
- 评估层：Open3D ROI 渲染能检查三维点云是否真的变好。

## 5. Dense Surface Maps 细节

Dense surface maps 是把参数化人体 mesh 转为每个 view 上与图像像素对齐的 dense prior。当前报告中所说的 `prior_maps` 主要承载以下信息：

- **prior depth**：人体 mesh 在该视角下的深度提示。
- **prior camera/world points**：人体表面点在相机或世界坐标中的位置提示。
- **prior normals**：人体表面法向，用于表达局部表面方向。
- **prior mask / human mask**：标记人体可见区域，避免背景对人体监督产生干扰。
- **summary tokens**：从 dense prior 中汇总的人体结构 token，用于给模型提供更全局的 pose / shape 上下文。

在训练 case 中，这些内容会进入 `inputs.npz` 和 `targets.npz` 一类结构：输入侧包含图像、mask、prior maps、prior summary tokens；目标侧包含 depths、cam/world points、depth/world confidence、prior depths、prior points、prior normals 等。

这一步的关键是“view-aligned”：prior 不是一个孤立 mesh，而是被投到同一批 sparse cameras 中，和图像像素、mask、VGGT 输出逐像素对应。

## 6. Input-Side / Layer-Wise Fusion 细节

### 6.1 Input-side prior conditioning

Input-side fusion 的目标是在模型看到 RGB 图像的同时，也看到人体结构先验。其基本原则是：

- RGB 提供真实外观、衣物、头发和背景信息。
- prior maps 提供 pose-aligned 人体几何位置和粗表面方向。
- mask 限制人体 prior 的作用范围，避免错误影响背景。

这种设计让模型不是从 sparse-view RGB 中“盲猜人体结构”，而是在图像观测和人体先验之间做几何融合。

### 6.2 Layer-wise fusion

Layer-wise fusion 的意义是：人体 prior 不只在输入时拼接一次，而是在模型多层特征演化过程中持续提供约束。对于 sparse-view 人体，这比单次输入更合理：

- 浅层可以利用 prior 提供的人体区域和姿态位置。
- 中层可以将多视角特征和人体 surface prior 对齐。
- 深层可以把 prior 信息转化为最终 depth / point / normal 输出。

当前工作证明了该类 prior/fusion 训练和推理链路可以跑通；但最终是否能显著改善 head/face 点云，仍取决于 teacher 质量、ROI 监督设计和可见表面一致性。

## 7. Output-Side 监督与 Loss 细节

Output-side 监督的目的，是把 pose-aligned prior 不仅作为输入条件，还作为训练目标或辅助约束。当前阶段涉及的监督包括：

- **depth loss**：约束预测深度接近目标深度或 prior depth。
- **camera/world point loss**：约束相机/世界坐标点位置，提高三维点云稳定性。
- **normal / cosine normal loss**：约束局部表面方向，补足单纯深度监督对曲面方向表达不足的问题。
- **point-normal consistency**：让预测点云局部结构与法向约束一致，避免点的位置和表面方向互相矛盾。
- **confidence-aware filtering / evaluation**：通过 p40 或固定 threshold 避免把低置信噪声点当作质量提升。
- **ROI-restricted / boundary-upweighted loss**：在 head、face、hairline、ear、neck 等导师关心区域提高监督密度。

这一路线已形成可运行的工程闭环，但已验证：仅增加或微调这些 loss，如果 teacher 本身不够连续、对齐、可见面不完整，仍可能产生“点数增加但 Open3D 更差”的伪阳性。

## 8. Coarse Prior Normal 链路及边界

![coarse prior normal storyboard](../output/mentor_progress_figures_20260427/01_coarse_prior_normal_storyboard_60v.png)

Coarse prior normal 是当前可对导师展示的稳定成果之一。其作用是把 SMPL-X view-aligned prior 转为可读的 normal 图，证明：

- pose-aligned prior 能在 view 空间稳定显示。
- full-body、head ROI、face ROI 都有可解释的 coarse surface direction。
- 60v / 13v / 7v ladder 中，coarse prior normal 大体保持稳定。

本阶段的准确表述是：**coarse prior normal 链已成立，但当前只是 SMPL-X view-aligned coarse prior；下一步是 image-aligned residual/detail refinement。** 这句话是汇报时最安全的正向结论。

但它的边界同样明确：

- 这是 **coarse prior normal**，不是 VGGT 已经预测出高质量 normal。
- 它不能证明 6-view sparse-view 点云已经达到 PSHuman / HumanRAM-like 质量。
- 4v predicted normal probe 已归档为 silhouette-only collapse，不是正结果。
- source view count 增加只带来有限可见变化，说明瓶颈不是简单增加视角，而是 prior 表达和局部 detail teacher。

## 9. Detail Normal Refiner 定位

导师要求“借鉴 PIFuHD coarse-to-fine 思路”时，当前正确理解不是把 VGGT 替换为 PIFuHD，而是在 VGGT 框架内加入 ROI-first detail refinement：

```text
RGB crop + coarse prior normal crop + human mask
        ↓
detail_normal_refiner
        ↓
refined normal / normal residual
        ↓
patch back to prior maps 或作为 ROI teacher
        ↓
再进入 sparse-view geometry 评估
```

`detail_normal_refiner` 的定位是：

- 不替代 VGGT。
- 不替代 coarse SMPL-X prior normal。
- 在 head / neck / shoulder / face ROI 上做 image-aligned residual/detail refinement。
- 先证明 60-view / teacher-rich 设置下 ROI normal refinement 成立，再下放到 13v、7v、6v。

当前状态是：ROI normal refinement 链路已能训练、应用、评估，并可 patch 回 dense prior normal channels；但“refined normal -> 最终 sparse-view 点云显著提升”尚未闭环。已有结果表明，仅修改 dense normal channels 或微调局部 teacher，还不足以让最终 face/head 点云达到导师标准。下一步必须让 refiner 真正以 RGB crop、coarse prior normal、mask 和可见表面 teacher 为条件，学习 image-aligned residual/detail，而不是只复用 coarse normal。

## 10. Real-Data External SMPL-X Bridge

真实数据路线不能依赖 4K4D 内部 annotation。为此，本阶段已构建 external SMPL-X bridge：

1. 接收外部 SMPL-X 参数和相机参数：`.npz` 或 `.json`。
2. 归一化常见字段别名，例如 `betas / shape`、`fullpose / body_pose`、`transl / translation`、`scale`、`gender`。
3. 生成 normalized external prior bundle。
4. 将 bundle 转为 scene-level `prior_maps.npz`、`inputs.npz`、`targets.npz`。
5. 进入 prior-enabled VGGT inference / training smoke。

真实口径必须保持：

- 仓库已具备“外部 estimator/fitter 结果 -> repo prior bridge -> VGGT prior-enabled inference”的工程路径。
- 仓库目前不能宣称“已完成 raw real images -> in-repo SMPL-X regressor -> final reconstruction”的全闭环。

## 11. 同协议评估标准

![当前 sparse-view best](../output/mentor_progress_figures_20260427/02_signfix_ckpt4_open3d_face_head.png)

当前统一评估协议：

- 场景：`output\4k4d_preprocessed_scene_variants\0012_11_frame0000_6views_sparseproto_headshoulder_crop`
- 参考预测：`output\modal_results\20260424_signfix_ckpt4_on6v_headshoulder\predictions.npz`
- 当前 reference：face ROI `16825`，head ROI `40527`，p40 confidence threshold 约 `38.507`

pass 判定必须同时满足：

| 维度 | 要求 | 当前状态 |
|---|---|---|
| face ROI | 明显超过 `16825`，不能只是 `+3/+15/+17` | 未满足 |
| head ROI | 不回退，最好同步改善 | 未证明明显改善 |
| confidence | 不能靠 threshold collapse 或 synthetic floor | 多个分支已出现伪阳性 |
| Open3D face | 中央脸部更连续，眼鼻口结构更清楚 | 未满足 |
| Open3D head | 发际线、头边界、后脑更稳定 | 未满足 |
| 协议一致性 | 必须在 original 6v headshoulder 同协议上成立 | targetcam30-only 不算 final |

## 12. 已排除路线和负结果

![负结果证据入口](../output/mentor_progress_figures_20260427/04_negative_evidence_thumbnails_depthpro_pshuman_geoonly.png)

### 12.1 Humancrop6v +17：数值抖动而非突破

![humancrop6v plus17](../output/mentor_progress_figures_20260427/03_humancrop6v_plus17_visual_same_not_breakthrough.png)

`humancrop6v ckpt0` 在早期各自 p40 阈值下 face ROI 达到 `16842`，相对 baseline `16825` 仅 `+17`。该图使用的是 3D percentile `face_close/head_close`，不能单独作为 semantic face 判据。后续 fixed-conf + camera-aligned re-audit 显示：固定 `38.5067` 后 3D face ROI 为 `16809`，camera03 2D face ROI 虽略增到 `45489`，但眼、鼻、口和连续脸部曲面仍不成型。因此只能作为诊断分支归档，不能包装为突破。

### 12.2 DepthPro

DepthPro 在 original same-protocol 中已验证为负：

- conservative variant 保持类似 confidence，但 face ROI 只出现 `+3/+15` 级别 jitter。
- direct/full fusion 造成 face/head 扭曲和侧向 artifact。
- 结论：DepthPro 当前不应进入大规模 sparse-view 训练，除非未来先通过连续、深度兼容的 visible-surface gate。

### 12.3 PSHuman

PSHuman official / true1024 路线证明了外部 human mesh teacher bridge 可以跑通，但仍未通过可见表面 gate：

- true1024 mesh 生成是真实的，不是早期 518 resize 误路径。
- 但在 sparse-view cameras 中 face-core coverage 碎片化，hole ratio / component / depth-compatible hits 不达标。
- single-view 或 multi-view consensus 都没有恢复清晰眼鼻口结构。

### 12.4 Geoonly continuation

Geoonly continuation 出现了典型伪阳性：

- p40 point count 可上升。
- 但 confidence collapse 或表面 torn/sheet-like，Open3D face/head 更差。
- 结论：不能继续沿这个分支盲训，除非 teacher mask / visible surface gate 先显著改善。

### 12.5 其他已降级路线

- projected targetpatch / summary-token patch：已判负，不应回到主线。
- confidence floor / boost：会改变阈值，不可作为真实模型质量提升。
- 2D-only ROI projection：可用于诊断和对齐，不可作为三维质量提升证据。
- Sapiens / NormalBae / PIFuHD / LHM / 60v surfacepose teacher：工程上有价值，但目前未提供足够连续、对齐、depth-compatible 的 face/head teacher。
- PIFuHD / PSHuman / HumanRAM：本报告只把它们作为方向论据，分别对应 coarse-to-fine、高分辨人体细节、pose-conditioned quality；不能把当前结果描述为达到或接近这些方法的 head/face 质量。

## 13. 当前 Truthful 结论

截至 2026-04-27，当前可以正式汇报为：

1. **工程闭环成立**：SMPL-X pose-aligned prior、dense prior maps、layer-wise fusion、output-side supervision、coarse prior normal、detail normal refiner、real-data bridge 和 Open3D ROI 评估链路均已建立。
2. **科研流程更完整**：从问题定义、先验构建、模型融合、监督设计、同协议评估、负结果排查到下一步 teacher 选择，已经形成可复现流程。
3. **质量 final bar 未达标**：当前 sparse-view 6v head/face 点云质量仍未达到导师最终 bar；不能宣传为 HumanRAM / PSHuman-like face detail。
4. **最可信 reference 仍是 signfix ckpt4**：它是当前同协议 best baseline，但不是最终 pass。
5. **当前瓶颈不再是脚本缺失**：瓶颈是高质量、连续、对齐、可见面兼容的 head/face teacher 或直接 sparse-view 几何提升信号。

## 14. 下一步：Image-Aligned Detail Refinement 与 Kinect/60v Teacher

### 14.1 主线目标：image-aligned residual/detail refinement

下一阶段不应再把 coarse prior normal 当成最终 normal，而应把它作为低频几何条件，叠加 RGB 对齐的细节残差：

```text
RGB / mask / SMPL-X view-aligned coarse prior normal
        ↓
image-aligned residual/detail normal refiner
        ↓
visible-surface gated normal / depth / point teacher
        ↓
VGGT sparse-view head/face ROI training
        ↓
Open3D same-protocol face/head pass check
```

该路线的成功标准不是 normal 图“看起来更细”，而是同协议 6v Open3D face/head 点云显著更清楚，并且不依赖 confidence threshold collapse。

### 14.2 为什么转向 Kinect 真深度

现有外部 monocular teacher 的共同失败点是：在目标 sparse-view 相机中不够连续、不够对齐，容易产生碎片、扭曲、sheet-like surface 或微小数值抖动。Kinect 路线的优势是：

- 来自同一 capture 的真实 metric depth。
- 具备 mask、intrinsics、extrinsics，可做严格坐标系检查。
- 能作为 visible surface teacher，避免单目 mesh 的不可见面幻觉。
- 如果通过 gate，可用于 head/face ROI 的 depth / point / normal teacher。

当前 Kinect smoke 已确认数据可读并导出点云：

- `output\detail_normal_refiner_20260427\kinect_depth_smoke\summary.json`
- mask/range 后总点数：`90264`

但 Kinect 仍不能直接宣布成功。下一步必须先完成：

1. calibration / coordinate sanity check；
2. Kinect depth 与 VGGT / RGB camera 的跨相机对齐；
3. visible-surface gate；
4. Open3D face/head ROI 可视化；
5. 通过后才进入 ROI refiner 或小规模训练。

### 14.3 为什么保留 60v RGB-MVS teacher 方向

60-view RGB-MVS / fused geometry 仍有科研价值，因为它能提供比 6-view 更充分的多视角可见约束。合理方向是：

- 用 60v 产生更稳定的 target-view surface / normal teacher。
- 先在 head/face ROI 做局部 teacher 质量验证。
- 再从 60v teacher 下放到 13v、7v、6v。
- 避免直接大规模盲训，避免 teacher 未通过 gate 就进入 end-to-end sparse-view training。

## 15. 报告图包与图注

配图清单和图注见：

- `output\mentor_progress_figures_20260427\figure_captions_research_style.md`
- `output\mentor_progress_figures_20260427\visual_pack_manifest.json`

必需图当前无缺失。所有图均复用已有产物或已有图包 contact sheet，未重跑训练，未修改模型/训练/工具代码。
