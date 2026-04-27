# 2026-04-27 Next-Window Handoff: Sparse-View Human Geometry / Mentor Checklist

This document is the migration packet for the next Codex conversation window.
It is intentionally truthful: **mentor-final sparse-view human geometry is still not reached**.
Do not repackage diagnostic or partially closed results as final success.

## 0. Hard Constraints For The Next Window

- **Model constraint**: all main-agent and sub-agent work must use `gpt-5.5` with `xhigh` reasoning.
- **Workspace**: `D:\vggt\vggt-main`.
- **Shell**: PowerShell inside Codex only; do not open separate Windows windows.
- **Command default**: use `login:false` for `shell_command`.
- **Python/Modal prefix**:
  ```powershell
  $env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'
  ```
- **Open3D local Python**:
  ```powershell
  D:\anaconda\envs\g3splat\python.exe
  ```
- **Patch rule**: edit files with the Codex `apply_patch` tool, not shell-based patching.
- **Truth rule**: only say “pass / 达标” when both are true:
  - same-protocol quantitative metrics clearly beat the current reference, not by tiny jitter;
  - Open3D head/face close-ups visibly show cleaner facial/head geometry, not just more points.
- **Forbidden mainline**: do not return to `projected targetpatch` / summary-token patch. It is already rejected.
- **Do not kill processes blindly**: only clean known rogue Modal/Python/Open3D processes. Do not kill Codex or unrelated WPS/desktop Python processes without clear evidence.

## 1. Mentor Guidance: Full Technical Intent

The mentor’s core request is not just “add SMPL-X” or “show normal maps”.
The actual target is:

> With SMPL/SMPL-X prior available, make sparse-view human reconstruction competitive: at 6 views, the human point cloud, especially head/face, must show clear face geometry details. If 6-view cannot show useful eyes/nose/mouth/head detail, the method is not competitive because 60-view, GS, HumanRAM-like methods can already produce high-quality human appearance/geometry.

### 1.1 SMPL / SMPL-X Requirement

- The SMPL pose in the current 4K4D case can come from input/annotation, but the repository must also contain a real-data route for later:
  - external SMPL-X estimator/fitter output import;
  - conversion to repo bundle;
  - scene-level `prior_maps.npz`;
  - prior-enabled VGGT inference.
- It is acceptable that the repo does **not** directly regress SMPL-X from raw real images yet, but this must be stated truthfully:
  - current route is **external estimator/fitter + repo import/alignment/scene-prior bridge**;
  - not a fully in-repo raw-image-to-SMPL-X regressor.
- The VGGT model should not be replaced by a pure SMPL-X regressor:
  - VGGT remains the general multi-view geometry backbone;
  - SMPL-X is a human geometry prior/condition/supervision path;
  - background, clothing, hair, and non-SMPL geometry must still be handled by VGGT/RGB/multi-view geometry.

### 1.2 Sparse-View Human Geometry Requirement

- Main target is sparse-view geometry, especially:
  - `6views_sparseproto_headshoulder_crop`;
  - head ROI;
  - face ROI;
  - Open3D close-ups.
- Final quality must be judged on point-cloud geometry, not only normal maps or 2D projected visuals.
- Required visual standard:
  - clearer face/head point surface;
  - less central face hole;
  - less ghost/double-head artifact;
  - better hairline/head boundary;
  - evidence of eyes/nose/mouth-level surface detail if possible.
- High point count alone is not enough, especially when:
  - confidence threshold collapses to `1.0`;
  - points are noisy, folded, twisted, or floating;
  - face/head Open3D close-up is worse.

### 1.3 Open3D Visualization Requirement

- Use Open3D to visualize human point cloud, not only MeshLab screenshots.
- Always include:
  - face ROI close-up;
  - head ROI close-up;
  - camera-aligned render/crop when useful;
  - baseline vs candidate comparison.
- Use the repo renderer:
  - `tools/render_open3d_pointcloud.py`
  - `scripts/render_pointcloud_open3d.ps1`
- Current renderer supports:
  - `--roi face|head|full`;
  - `--roi-source 3d|2d`;
  - `--point-source world_points|depth_unprojection`;
  - camera-aligned render indices.
- 2D ROI renders are clearer for camera-aligned evidence, but are a visualization/ROI correction only. They are not model-quality improvement by themselves.

### 1.4 Normal / PIFuHD-Style Requirement

The mentor asked:

- “让 VGGT 输出人体法向图”
- “增强几何约束，深度估计不够，看看 PIFuHD”
- “点云质量参考 PSHuman”

The correct technical interpretation is:

- Current **coarse prior normal** chain is useful but not enough.
- Do not claim “VGGT already outputs high-quality predicted normal” unless it is truly end-to-end and improves geometry.
- The next normal branch should be:
  - branch name: `detail_normal_refiner` or `pifuhd_style_normal_refine`;
  - role: image-aligned residual refinement of coarse prior normal;
  - not a replacement for VGGT;
  - not a replacement for SMPL-X coarse prior.
- Inputs:
  - RGB crop;
  - coarse prior normal crop;
  - human mask.
- Output:
  - refined normal or normal residual.
- First ROI:
  - head / neck;
  - shoulder line;
  - do not start with full-body high-resolution clothing folds.
- First goal:
  - clearer head boundary and hairline;
  - not “solve all detail at once”.

### 1.5 Teacher / Supervision Requirement

- The coarse prior normal itself cannot be the detail teacher.
- Teacher priority:
  1. multi-view fused geometry normal from reliable 60v/multi-view surface;
  2. high-quality external human normal/mesh/depth estimator;
  3. local mesh/surface fitting pseudo-GT;
  4. only visible-region refinement if teacher quality is limited.
- Loss should include at least:
  - cosine normal loss;
  - edge-aware loss;
  - mask-restricted loss;
  - ROI boundary upweighting.
- Metrics must separately track:
  - head ROI;
  - face ROI;
  - hairline / ear / boundary if available.
- Do not rely only on full-image average loss.
- Every branch must output fixed visualization:
  - RGB;
  - coarse prior normal;
  - refined normal;
  - coarse-vs-refined diff;
  - head ROI;
  - face ROI;
  - failure cases.

### 1.6 Experiment Order Required By Mentor

- First make 60v / high-quality teacher/refiner work.
- Then 13v.
- Then 7v.
- Then only after stable teacher/refiner, move back to 6v end-to-end sparse-view training.
- Before any large cloud training:
  - run small ROI;
  - run small batch;
  - overfit one frame;
  - verify head/face detail visually.
- Do not launch another large sparse-view training run from a teacher that cannot pass direct visible-surface/Open3D gate.

### 1.7 Mentor-Facing Wording

Allowed:

- “Current coarse prior normal chain is established.”
- “Next step borrows PIFuHD-style coarse-to-fine refinement to improve high-resolution local human detail.”
- “4v probe collapsed to silhouette-only and is downgraded to internal diagnostic.”
- “60v proves coarse prior alignment/stability/showability, but fine detail quality still needs improvement.”
- “HumanRAM supports the argument that aligned human pose conditioning can help quality.”
- “PIFuHD supports the argument that high-resolution human detail needs coarse-to-fine/image-aligned refinement.”

Forbidden:

- “60v is fully mentor-final.”
- “VGGT already outputs high-quality predicted normal.”
- “4v probe is almost good.”
- “The method is complete because point count increased.”
- “No unmet items remain” unless the Open3D and quantitative gates both pass.

## 2. Old Yuque / Previous Report Narrative To Preserve

The previous long-form narrative was broadly correct as architecture motivation:

- Original VGGT is general-purpose multi-view geometry:
  - cameras;
  - depth maps;
  - world point maps;
  - tracks.
- Original VGGT is not human-specialized and does not explicitly use:
  - stable body topology;
  - limbs/head/torso structure;
  - parametric body surface;
  - dedicated hand/face geometry rules.
- SMPL/SMPL-X supplies a structured body-surface space:
  - fixed topology template;
  - shape parameters;
  - pose parameters;
  - skeleton/skinning;
  - pose corrective blend shapes;
  - continuous posed mesh surface.
- The project’s intended architecture is:
  - preserve VGGT backbone;
  - add pose-aligned SMPL-X driver;
  - project vertex/surface features into dense per-view prior maps;
  - fuse human prior tokens input-side and layer-wise;
  - keep output-side camera-aligned human geometry supervision.

But the next window must revise the old “results” wording:

- The 60v result is **not** mentor-final.
- 60v / coarse prior normal is evidence of alignment and stability, not proof of PSHuman-level detail.
- Any “large improvement” claim must be backed by current Open3D evidence.
- The correct present conclusion is:
  - architecture/infrastructure is real;
  - coarse prior normal/advisor pack is closed;
  - sparse-view head/face point-cloud quality is still below mentor bar.

## 3. Current Truthful State

### 3.1 Overall Verdict

- Mentor-final sparse-view human geometry bar is **not reached**.
- The current acceptable mainline reference is still around:
  - `signfix ckpt4`;
  - same protocol `6views_sparseproto_headshoulder_crop`;
  - face ROI `16825`;
  - conf p40 `38.5067`;
  - visual still not mentor-final.
- Recent improvements are either:
  - tiny numerical jitter;
  - protocol-specific targetcam30 gains that do not transfer;
  - confidence-collapse pseudo-positives;
  - visual negatives.

### 3.2 Current Reference Best

Reference prediction:

```text
D:\vggt\vggt-main\output\modal_results\20260424_signfix_ckpt4_on6v_headshoulder\predictions.npz
```

Reference scene:

```text
D:\vggt\vggt-main\output\4k4d_preprocessed_scene_variants\0012_11_frame0000_6views_sparseproto_headshoulder_crop
```

Reference Open3D / ROI evidence:

```text
D:\vggt\vggt-main\output\detail_normal_refiner_20260427\original6v_headshoulder_depthpro_gate\open3d_baseline_signfix_ckpt4_3droi_face\open3d_summary.json
D:\vggt\vggt-main\output\detail_normal_refiner_20260427\original6v_headshoulder_depthpro_gate\open3d_baseline_signfix_ckpt4_3droi_head\open3d_summary.json
```

Key numbers:

| Protocol | Run | Conf pctl | Face ROI | Head ROI | Conf threshold | Truth |
|---|---:|---:|---:|---:|---:|---|
| original `6views_sparseproto_headshoulder_crop` | `signfix_ckpt4_ref` | 40 | 16825 | 40527 | 38.507 | current reference, still not mentor-final |

### 3.3 “Small Positive” That Must Not Be Overclaimed

Humancrop6v latest ROI:

- `20260424_humancrop6v_ckpt0_on6v_headshoulder`: face ROI `16842`.
- `20260424_humancrop6v_ckpt1_on6v_headshoulder`: face ROI `16785`.
- `20260424_humancrop6v_inference_on6v_headshoulder`: face ROI `16785`.

Interpretation:

- `16842` is only `+17` over `16825`.
- Open3D face close-up is essentially same morphology as signfix ckpt4.
- This is measurement jitter, not mentor-level breakthrough.

### 3.4 Best Diagnostic Numeric-Positive That Still Is Not Pass

The strongest non-collapsed diagnostic numeric-positive mentioned by side audit is:

```text
D:\vggt\vggt-main\output\modal_results\20260426_signfix_pifuhd512_multi3_mean_v1_xz050_y000_d004_noboost\local_roi_summary.json
```

Reported status:

- face ROI around `17315`;
- confidence did not collapse like the `conf_threshold=1.0` pseudo-positive branches;
- still has Open3D seam/noise and no reliable eyes/nose/mouth or continuous face surface.

Interpretation:

- Keep as a diagnostic “closest numeric-positive” reference.
- Do **not** promote it to mentor-final or main pass.
- Only revisit if a new visual gate or teacher refinement specifically fixes the visible face/head defects.

## 4. Completed Engineering / Checklist Items

These are real, but not final sparse-view quality passes.

### 4.1 SMPL-X / Human Prior Architecture

Core files:

```text
D:\vggt\vggt-main\vggt\models\vggt.py
D:\vggt\vggt-main\vggt\models\human_prior.py
D:\vggt\vggt-main\vggt\models\aggregator.py
D:\vggt\vggt-main\training\loss.py
D:\vggt\vggt-main\training\data\datasets\dna4k4d_pseudo.py
D:\vggt\vggt-main\tools\prepare_4k4d_prior_training_case.py
```

Status:

- Pose-aligned SMPL-X coarse prior maps exist.
- Prior maps include dense channels such as posed camera positions, camera normals, visibility, canonical coordinates, vertex/body-part features.
- Layer-wise / input-side human prior fusion exists in the model path.
- Output-side depth/points/normal/point-normal losses exist and support ROI/boundary weighting.

Truth:

- Architecture and training hooks are real.
- They have not yet produced mentor-final 6-view face/head point quality.

### 4.2 Coarse Prior Normal Advisor Pack

Canonical pack:

```text
D:\vggt\vggt-main\output\normal_advisor_pack_20260421_coarseprior
```

Legacy entry synchronized:

```text
D:\vggt\vggt-main\output\normal_advisor_pack_20260421
```

Key points:

- 4v probe predicted normal was removed from main conclusion.
- `prior normal` wording was changed to `coarse prior normal`.
- Failed 4v probe is isolated under `failed_predicted_normal_probe`.
- Must-send figures focus on 60v full/head/face prior normal and overview.
- 7v/13v only supplement “coarse prior remains stable”, not main proof.

Truth:

- Coarse prior normal pack is a closed presentation artifact.
- It is not sparse-view geometry pass.

### 4.3 Detail Normal Refiner

Core files:

```text
D:\vggt\vggt-main\vggt\models\detail_normal_refiner.py
D:\vggt\vggt-main\training\detail_normal_refiner_loss.py
D:\vggt\vggt-main\tools\train_detail_normal_refiner.py
D:\vggt\vggt-main\tools\apply_detail_normal_refiner.py
D:\vggt\vggt-main\tools\export_detail_normal_refiner_dataset.py
D:\vggt\vggt-main\tools\export_external_normal_refiner_dataset.py
```

Status:

- Branch exists.
- Inputs match mentor guidance:
  - RGB crop;
  - coarse prior normal crop;
  - human mask.
- Output is refined normal/residual.
- Loss/metrics include:
  - cosine;
  - edge;
  - mask restriction;
  - boundary/hairline/ear-related ROI metrics.
- ROI normal-map overfit/evaluation exists.

Truth:

- Normal-map ROI/refiner branch is operational.
- It has not yet produced final sparse-view point-cloud quality.
- `apply_detail_normal_refiner.py` outputs visuals/metrics; it does not by itself solve `world_points`.

### 4.4 Real-Data SMPL-X Bridge

Core files:

```text
D:\vggt\vggt-main\tools\run_realdata_smplx_driver.py
D:\vggt\vggt-main\tools\build_scene_prior_from_external_bundle.py
D:\vggt\vggt-main\docs\realdata_smplx_driver.md
```

Smoke-tested scene:

```text
D:\vggt\vggt-main\output\smoke_external_bundle_case\scene_with_external_prior_bridge
```

Modal inference smoke:

```text
D:\vggt\vggt-main\output\modal_results\20260424_smoke_external_prior_scene_bridge_ckpt4
```

Smoke summary confirmed:

- `num_images=2`;
- input tensor `[2,3,518,518]`;
- prior tensor `[2,30,518,518]`;
- prior summary tensor `[2,16,27]`.

Truth:

- External bundle -> scene `prior_maps.npz` -> Modal prior-enabled inference is closed.
- Raw real image -> in-repo SMPL-X regressor/fitter is not closed.

### 4.5 Open3D Visualization Chain

Core files:

```text
D:\vggt\vggt-main\tools\render_open3d_pointcloud.py
D:\vggt\vggt-main\scripts\render_pointcloud_open3d.ps1
D:\vggt\vggt-main\scripts\open_pointcloud_open3d.ps1
```

Status:

- Supports full/head/face ROI.
- Supports 3D ROI and 2D ROI.
- Supports `world_points` and `depth_unprojection`.
- Supports camera-aligned render/crop.

Truth:

- Visualization infrastructure is sufficient.
- The model output quality is still insufficient.

## 5. Negative / Rejected Branches

Do not repeat these unless a genuinely new teacher or protocol changes the premise.

### 5.1 Hard Forbidden

- `projected targetpatch`;
- summary-token patch;
- targetpatch-style headrefine projection hacks.

Reason:

- Already判负.
- It is no longer the mainline.

Report:

```text
D:\vggt\vggt-main\reports\20260422_projected_targetpatch_eval.md
```

### 5.2 Same-Checkpoint Micro-Tuning From ckpt4

Rejected or non-pass:

- direct `headshoulder6v pointnormal` from ckpt4 regressed;
- direct `humancrop6v pointnormal` from ckpt4 gave at best `+17`, visual same;
- r3 faceboost / teachergeom / roi combo runs did not beat `16825` with convincing visual quality.

### 5.3 TeacherGeom / ROI Combo From Base ckpt0

Same-protocol original 6v headshoulder failures:

| Run | Face ROI |
|---|---:|
| `20260424_sparseproto_headshoulder_teachergeom_teachernormal_r1_on6v` | 16617 |
| `20260424_sparseproto_headshoulder_teachergeom_teachernormal_roi_combo_r1_on6v` | 16494 |
| `20260424_6view_focus_headshoulder_teachergeom_teachernormal_r1` | 16636 |
| `20260424_6view_focus_headshoulder_teachergeom_teachernormal_roi_combo_r1` | 16559 |

Interpretation:

- Below signfix reference.
- Do not rerun unchanged.

### 5.4 Confidence-Collapse Pseudo-Positives

Rejected pattern:

- face ROI count jumps very high;
- `conf_threshold=1.0`;
- Open3D visual becomes worse/noisy/smeared.

Examples:

- `20260424_sparseproto_humancrop_pointnormal_r1_open3d_face`: high point count but false positive.
- Sapiens normal teacher e2/conservative.
- normal-guided depth R2.
- mesh/surface teacher confidence-supervised variants.

Rule:

- Any result with collapsed confidence and worse Open3D is negative, not “high ROI”.

### 5.5 Surface Teacher / 60v Teacher Attempts

Reports:

```text
D:\vggt\vggt-main\reports\20260424_truthful_sparse_view_headface_status_update.md
D:\vggt\vggt-main\output\detail_normal_refiner_20260427\side_60v_teacher_upperbound\audit_summary.md
```

Truth:

- 60v target-view upper-bound has dense ROI pixels but not clearly stronger continuous face/head teacher.
- Multi-view surface teacher / Poisson mesh teacher did not improve original 6v.
- Removing stale summary tokens (`denseonly`) did not fix it.

### 5.6 Sapiens / NormalBae / DepthAnything / DepthPro

Status:

- NormalBae: smooth/image-aligned normal; did not improve geometry.
- Sapiens normal: better 2D normal; normal-only prior/supervision did not anchor geometry and caused pseudo-positives.
- Sapiens depth: low correlation after affine alignment; not reliable.
- DepthAnything prior: same 2D face ROI count as baseline and smoother/less detailed; not pass.
- DepthPro:
  - targetcam30 pre-gate passed, but direct/prior/original6v tests did not pass visual geometry;
  - same-protocol conservative fusions had tiny point jitter only.

DepthPro same-protocol result:

| Run | Face ROI | Head ROI | Conf | Truth |
|---|---:|---:|---:|---|
| baseline signfix ckpt4 | 16825 | 40527 | 38.507 | reference |
| `depthpro_xz020_y000_d020` | 16828 | 40527 | 38.507 | +3 jitter |
| `depthpro_xz035_y000_d020` | 16840 | 40527 | 38.507 | +15 jitter |
| `depthpro_xz050_y000_d035` | 16808 | 40527 | 38.507 | below baseline |
| direct full | n/a | n/a | 38.507 | visibly twisted/distorted |

Evidence:

```text
D:\vggt\vggt-main\output\detail_normal_refiner_20260427\original6v_headshoulder_depthpro_gate\original6v_depthpro_conservative_3droi_face_comparison_sheet.png
D:\vggt\vggt-main\output\detail_normal_refiner_20260427\original6v_headshoulder_depthpro_gate\original6v_depthpro_direct_3droi_face_comparison_sheet.png
```

### 5.7 PSHuman / PIFuHD External Mesh Attempts

Status:

- Official PSHuman bridge is real.
- HQ1024 / true1024 generation is real.
- PIFuHD512 and PSHuman mesh adapters are real.
- But current external meshes do not provide continuous, aligned target-view face surfaces.
- PIFuHD512 no-boost residual can be kept as a diagnostic numeric-positive (`~17315` face ROI, confidence preserved), but it is still a visual non-pass because Open3D face/head remains seamed/noisy.

Representative failures:

```text
D:\vggt\vggt-main\output\detail_normal_refiner_20260427\visible_surface_audit_batch_summary.json
D:\vggt\vggt-main\output\detail_normal_refiner_20260427\side_teacher_gate_scout\side_teacher_gate_scout_summary.md
```

Gate failures:

- `pshuman_src34_targetcam30`: hole `0.984`, median residual `0.0472m`.
- `pshuman_src00_targetcam30`: hole `0.971`, median residual `0.0440m`.
- `pshuman_hq1024_targetcam30`: hole `0.989`, median residual `0.0376m`.
- `pifuhd512_targetcam30`: hole `0.779`, median residual `0.0136m`.
- `pifuhd512_original`: hole `0.389`, median residual `0.0343m`.

True1024 PSHuman:

```text
D:\vggt\vggt-main\output\detail_normal_refiner_20260427\pshuman_true1024_human_face_close_compare_sheet.png
D:\vggt\vggt-main\output\detail_normal_refiner_20260427\pshuman_true1024_human_head_close_compare_sheet.png
```

Truth:

- Tiny face ROI gain (`+298` in one targetcam30 local gate) adds fragments and does not fix holes.
- Not a pass.

## 6. Key Reports And Artifacts To Read First

Read in this order:

1. Truthful global status:
   ```text
   D:\vggt\vggt-main\reports\20260424_truthful_sparse_view_headface_status_update.md
   ```
2. Earlier concise status:
   ```text
   D:\vggt\vggt-main\reports\20260424_truthful_sparse_view_headface_status.md
   ```
3. Real-data readiness:
   ```text
   D:\vggt\vggt-main\reports\20260422_external_smplx_realdata_readiness.md
   D:\vggt\vggt-main\docs\realdata_smplx_driver.md
   ```
4. Coarse prior normal pack:
   ```text
   D:\vggt\vggt-main\output\normal_advisor_pack_20260421_coarseprior
   D:\vggt\vggt-main\output\normal_advisor_pack_20260421
   ```
5. Current 2026-04-27 teacher / Open3D evidence:
   ```text
   D:\vggt\vggt-main\output\detail_normal_refiner_20260427
   ```
6. Side teacher-gate scout:
   ```text
   D:\vggt\vggt-main\output\detail_normal_refiner_20260427\side_teacher_gate_scout\side_teacher_gate_scout_summary.md
   D:\vggt\vggt-main\output\detail_normal_refiner_20260427\side_external_teacher_scout\scout_report.md
   ```

## 7. Mandatory Pass Criteria For Next Work

### 7.1 Quantitative Gate

For the original same protocol:

```text
scene = output\4k4d_preprocessed_scene_variants\0012_11_frame0000_6views_sparseproto_headshoulder_crop
baseline = output\modal_results\20260424_signfix_ckpt4_on6v_headshoulder\predictions.npz
```

Candidate must:

- beat face ROI `16825` by a meaningful margin;
- preserve or improve head ROI `40527`;
- avoid confidence collapse;
- keep p40 confidence in the same broad range unless there is a justified better calibration.

Minimum practical bar:

- `+17` or `+15` is not meaningful.
- A candidate should show a clear numerical gain and visual gain.
- If `conf_threshold=1.0`, treat as failed unless proven otherwise with exceptional visual quality, which has not happened so far.

### 7.2 Visual Gate

Candidate must show:

- cleaner face surface in Open3D;
- fewer holes in central face;
- no side-view twisting;
- no double-head / ghost face;
- less floating hair/face fragment noise;
- head/face close-up better than signfix ckpt4.

Required evidence:

- face ROI Open3D close-up;
- head ROI Open3D close-up;
- baseline vs candidate comparison sheet;
- same-protocol ROI JSON / summary.

### 7.3 Teacher Pre-Gate Before Training

Before launching cloud training from a new teacher:

- face-core hit pixels `>= 11000`;
- largest connected component `>= 0.80`;
- hole ratio `<= 0.15`;
- median depth residual `<= 0.012m`;
- visual overlay must not show fragmented or off-face hits.

Tool:

```text
D:\vggt\vggt-main\tools\audit_visible_surface_teacher.py
```

If the teacher fails this gate, do not train VGGT on it.

## 8. Recommended Next Mainline: Avoid The Wall

The next credible path is not another blind training run.
It should be a **teacher-quality-first** loop.

### 8.1 Mainline Candidate A: New Non-DepthPro External Mesh Teacher Gate

Use only if a genuinely new external mesh is available.
Do not reuse the already-failed PSHuman/PIFuHD meshes unchanged.

Commands:

```powershell
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'; $env:KMP_DUPLICATE_LIB_OK='TRUE'
python D:\vggt\vggt-main\tools\build_external_mesh_raycast_training_case.py `
  --source-case-dir D:\vggt\vggt-main\output\training_cases\0012_11_frame0000_6views_sparseproto_headshoulder_teachergeom_r1_raw `
  --external-mesh-path <NEW_NON_DEPTHPRO_EXTERNAL_MESH.obj_or.ply> `
  --target-scene-dir D:\vggt\vggt-main\output\4k4d_preprocessed_scene_variants\0012_11_frame0000_6views_sparseproto_headshoulder_crop `
  --anchor-predictions-npz D:\vggt\vggt-main\output\modal_results\20260424_signfix_ckpt4_on6v_headshoulder\predictions.npz `
  --output-case-dir D:\vggt\vggt-main\output\detail_normal_refiner_20260427\next_non_depthpro_mesh_gate\case `
  --output-diagnostics-dir D:\vggt\vggt-main\output\detail_normal_refiner_20260427\next_non_depthpro_mesh_gate\diagnostics `
  --roi-kind face_core --align-mode umeyama_icp --depth-tolerance 0.012 --conf-boost 0.0 --overwrite
```

Then:

```powershell
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'; $env:KMP_DUPLICATE_LIB_OK='TRUE'
python D:\vggt\vggt-main\tools\audit_visible_surface_teacher.py `
  --mesh-path D:\vggt\vggt-main\output\detail_normal_refiner_20260427\next_non_depthpro_mesh_gate\diagnostics\external_mesh_transformed.ply `
  --predictions-npz D:\vggt\vggt-main\output\modal_results\20260424_signfix_ckpt4_on6v_headshoulder\predictions.npz `
  --scene-dir D:\vggt\vggt-main\output\4k4d_preprocessed_scene_variants\0012_11_frame0000_6views_sparseproto_headshoulder_crop `
  --output-dir D:\vggt\vggt-main\output\detail_normal_refiner_20260427\next_non_depthpro_mesh_gate\visible_surface_audit `
  --view-index 0 --roi-kind face_core
```

Only if gate passes, run conservative no-boost fusion:

```powershell
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'; $env:KMP_DUPLICATE_LIB_OK='TRUE'
python D:\vggt\vggt-main\tools\fuse_external_teacher_into_predictions.py `
  --base-predictions D:\vggt\vggt-main\output\modal_results\20260424_signfix_ckpt4_on6v_headshoulder\predictions.npz `
  --teacher-targets D:\vggt\vggt-main\output\detail_normal_refiner_20260427\next_non_depthpro_mesh_gate\case\targets.npz `
  --output-dir D:\vggt\vggt-main\output\detail_normal_refiner_20260427\next_non_depthpro_mesh_gate\fused_noboost `
  --alpha-x 0.5 --alpha-y 0.0 --alpha-z 0.5 --max-distance 0.012 --confidence-boost 0.0
```

Then render:

```powershell
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'; $env:KMP_DUPLICATE_LIB_OK='TRUE'
D:\anaconda\envs\g3splat\python.exe D:\vggt\vggt-main\tools\render_open3d_pointcloud.py `
  --predictions-npz D:\vggt\vggt-main\output\detail_normal_refiner_20260427\next_non_depthpro_mesh_gate\fused_noboost\predictions.npz `
  --scene-dir D:\vggt\vggt-main\output\4k4d_preprocessed_scene_variants\0012_11_frame0000_6views_sparseproto_headshoulder_crop `
  --output-dir D:\vggt\vggt-main\output\detail_normal_refiner_20260427\next_non_depthpro_mesh_gate\open3d_face `
  --roi face --roi-source 3d --conf-percentile 40 --human-only --point-source world_points
```

### 8.2 Mainline Candidate B: Photometric / Multi-View Local Surface Fit

This is now implemented as a reusable diagnostic tool, but the first same-protocol gate is negative.
It remains useful as a rejection gate, not as a pass result.
It is the most plausible “avoid wall” route if no better external mesh appears.

Goal:

- use multi-view RGB consistency around the target head/face ROI;
- start from signfix ckpt4 depth/points;
- perform local depth-sweep or patch-match only in face/head ROI;
- keep confidence unchanged initially;
- render Open3D before any training.

Why:

- monocular normal/depth teachers failed;
- external meshes are misaligned/fragmented;
- VGGT’s own 6v prediction has partial surface but holes;
- local multi-view RGB may provide the missing continuous target-view surface check.

First implementation should be diagnostic only:

- input:
  - `signfix_ckpt4` prediction;
  - original 6v headshoulder scene;
  - masks;
  - target view 0;
  - source views 1..5.
- output:
  - patched `predictions.npz`;
  - depth/refinement map;
  - photometric cost map;
  - Open3D face/head renders;
  - report saying diagnostic, not raw VGGT.

Implemented tool:

```text
D:\vggt\vggt-main\tools\patch_predictions_photometric_depth_refine.py
```

Latest same-protocol outputs:

```text
D:\vggt\vggt-main\output\detail_normal_refiner_20260427\photometric_depth_refine_original6v_facecore_r1
D:\vggt\vggt-main\output\detail_normal_refiner_20260427\photometric_depth_refine_original6v_face_loose_v2
```

Result:

- strict `face_core_r1`: accepted coverage `0.160`, face ROI `16728`, head ROI `40527`;
- loose corrected `face_loose_v2`: accepted coverage `0.341`, face ROI `16825`, head ROI `40527`, conf threshold `38.507`;
- visual Open3D is effectively same as baseline and not mentor-quality.

Do not train from this photometric teacher unless a later version first passes accepted coverage and Open3D visual gates.

### 8.3 Mainline Candidate C: Better External Human Reconstruction Source

If cloud or external tool can produce a genuinely better full-human/head mesh:

- prefer multi-view or calibrated reconstruction, not single-view monocular mesh;
- must align to 4K4D cameras;
- must pass `audit_visible_surface_teacher.py`;
- only then build training case and run small ROI overfit.

Potential sources to consider:

- LHM / LHM++ official human reconstruction as a new mesh teacher, because it is not one of the already-failed PSHuman/PIFuHD/Sapiens/DepthPro branches;
- stronger PSHuman settings if there is a real quality knob not already tested;
- ECON/ICON/PIFu-style mesh with better alignment;
- high-quality local surface fitting from 60v but not the already-failed Poisson/surface teacher.

Do not reuse failed artifacts unchanged.

Current LHM attempt:

```text
D:\vggt\vggt-main\modal_lhm_mesh_teacher.py
D:\vggt\vggt-main\output\logs\20260427_lhm_mini_mesh_teacher_stdout.log
D:\vggt\vggt-main\output\logs\20260427_lhm_mini_mesh_teacher_stderr.log
```

If it completes and downloads a `.ply`, do **not** call it a result yet. First run:

1. `tools\build_external_mesh_raycast_training_case.py` with `--align-mode umeyama_icp`;
2. `tools\audit_visible_surface_teacher.py`;
3. no-boost fusion;
4. same-protocol Open3D face/head render.

## 9. Commands For Routine Verification

### 9.1 Check Processes

```powershell
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'
Get-CimInstance Win32_Process |
  Where-Object { $_.Name -match 'python|modal|powershell|node|open3d' } |
  Select-Object ProcessId,Name,CommandLine | Format-List
```

Only kill clearly stale project processes.

### 9.2 Render Baseline Same-Protocol Open3D

```powershell
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'; $env:KMP_DUPLICATE_LIB_OK='TRUE'
D:\anaconda\envs\g3splat\python.exe D:\vggt\vggt-main\tools\render_open3d_pointcloud.py `
  --predictions-npz D:\vggt\vggt-main\output\modal_results\20260424_signfix_ckpt4_on6v_headshoulder\predictions.npz `
  --scene-dir D:\vggt\vggt-main\output\4k4d_preprocessed_scene_variants\0012_11_frame0000_6views_sparseproto_headshoulder_crop `
  --output-dir D:\vggt\vggt-main\output\detail_normal_refiner_20260427\handoff_baseline_signfix_ckpt4_face `
  --roi face --roi-source 3d --conf-percentile 40 --human-only --point-source world_points --camera-view-indices 0
```

### 9.3 Summarize Same-Protocol ROI Remotely

Use the actual remote output subdir of a candidate:

```powershell
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'
modal run modal_4k4d_vggt_infer.py::summarize_prediction_roi `
  --remote-output-subdir <evals/...> `
  --scene-subdir scenes/0012_11_frame0000_6views_sparseproto_headshoulder_crop `
  --conf-percentile 40
```

### 9.4 Run Prior-Enabled External Bridge Smoke If Needed

This was already completed, but this is the correct command shape:

```powershell
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'
modal run modal_4k4d_vggt_infer.py::run_scene_from_local `
  --local-scene-dir output\smoke_external_bundle_case\scene_with_external_prior_bridge `
  --remote-scene-subdir smoke_external_scene_bridge\scene_with_external_prior_bridge `
  --output-subdir evals/20260424_smoke_external_prior_scene_bridge_ckpt4 `
  --checkpoint-relpath vggt_4k4d_train/20260423_sparseproto_humancrop_pointnormal_r2_signfix/logs/ckpts/checkpoint_4.pt `
  --download-local-dir output\modal_results\20260424_smoke_external_prior_scene_bridge_ckpt4
```

## 10. Files Added / Important Recent Code

Recent tools and configs that matter:

```text
D:\vggt\vggt-main\training\config\4k4d_prior_case_sparseproto_humancrop_pointnormal_r3_faceboost.yaml
D:\vggt\vggt-main\tools\build_scene_prior_from_external_bundle.py
D:\vggt\vggt-main\tools\render_open3d_pointcloud.py
D:\vggt\vggt-main\tools\audit_visible_surface_teacher.py
D:\vggt\vggt-main\tools\build_external_mesh_raycast_training_case.py
D:\vggt\vggt-main\tools\build_external_depth_training_case.py
D:\vggt\vggt-main\tools\fuse_external_teacher_into_predictions.py
D:\vggt\vggt-main\tools\fuse_multi_external_teachers_into_predictions.py
D:\vggt\vggt-main\tools\patch_predictions_surface_completion.py
D:\vggt\vggt-main\tools\patch_scene_prior_with_refined_normals.py
D:\vggt\vggt-main\tools\patch_training_case_with_refined_normals.py
D:\vggt\vggt-main\tools\build_highres_headface_scene.py
D:\vggt\vggt-main\tools\patch_predictions_photometric_depth_refine.py
D:\vggt\vggt-main\modal_pshuman_official_teacher.py
D:\vggt\vggt-main\modal_pifuhd_mesh_teacher.py
D:\vggt\vggt-main\modal_lhm_mesh_teacher.py
D:\vggt\vggt-main\modal_sapiens_normal_teacher.py
D:\vggt\vggt-main\modal_sapiens_depth_teacher.py
D:\vggt\vggt-main\modal_external_depth_teacher.py
D:\vggt\vggt-main\modal_external_normal_teacher.py
```

Potentially large/untracked repo state exists.
Do not assume a clean Git tree.
Run:

```powershell
git status --short
```

before editing.

## 11. What The Next Window Should Do First

1. Read this handoff file and preserve its truth rule.
2. Read:
   ```text
   D:\vggt\vggt-main\reports\20260424_truthful_sparse_view_headface_status_update.md
   ```
3. Check live processes and clean only obvious stale project jobs.
4. Do **not** launch large training immediately.
5. Decide whether a genuinely new teacher is available:
   - if yes, run teacher visible-surface gate first;
   - if no, do not repeat the current photometric sweep settings; they already tied/regressed the baseline.
6. If the LHM Modal job is still running, poll:
   ```text
   D:\vggt\vggt-main\output\logs\20260427_lhm_mini_mesh_teacher_stdout.log
   D:\vggt\vggt-main\output\logs\20260427_lhm_mini_mesh_teacher_stderr.log
   ```
7. Only after direct diagnostic Open3D is visibly better:
   - build a small ROI training case;
   - overfit one frame;
   - evaluate same-protocol 6v headshoulder;
   - render head/face Open3D.

## 11.1 Paste-Ready Prompt For The Next Window

Use this if the next conversation needs a compact start prompt:

```text
继续 D:\vggt\vggt-main 的 sparse-view human geometry 主线。必须使用 gpt-5.5 xhigh；PowerShell 在 Codex 内运行，shell_command 用 login:false；Python/Modal 前设置 $env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'。先读 reports\20260427_next_window_handoff_checklist.md 和 reports\20260424_truthful_sparse_view_headface_status_update.md。

真实状态：导师最终 6-view head/face 点云质量还未达标。当前同协议参考仍是 output\modal_results\20260424_signfix_ckpt4_on6v_headshoulder\predictions.npz，在 scene output\4k4d_preprocessed_scene_variants\0012_11_frame0000_6views_sparseproto_headshoulder_crop 上 face ROI 16825/head ROI 40527，但视觉仍非 mentor-final。PIFuHD512 no-boost residual 有约 17315 face ROI 且 confidence 未塌，但 Open3D 仍有 seam/noise，不是 pass。

禁止回到 projected targetpatch / summary-token patch。禁止把 targetcam30-only、高 ROI 但 conf_threshold=1.0、2D ROI 可视化、surface completion/post-process、工程 smoke 当成达标。必须同时满足同协议量化明显超过参考并且 Open3D face/head close-up 真的更清楚，才能说 pass。

已闭环但不是质量 pass：SMPL-X coarse prior / layer-wise human prior、coarse prior normal advisor pack、detail_normal_refiner、external SMPL-X bundle -> prior_maps.npz -> Modal inference bridge、Open3D ROI renderer、PSHuman/PIFuHD/Sapiens/NormalBae/Depth/mesh adapters。

下一步不要盲训。先找/造更强 continuous target-view face/head teacher，并先过 visible-surface gate：face-core hit pixels >=11000、largest component >=0.80、hole ratio <=0.15、median residual <=0.012m。工具：tools\audit_visible_surface_teacher.py。teacher 没过 gate 不准开大规模 sparse-view end-to-end training。若没有新外部 mesh，优先实现/尝试局部多视角 photometric surface fit 诊断，只改 head/face ROI，先 Open3D 直观过门再训练。
```

## 12. Final Truth To Preserve

- The mentor’s final bar is **not yet met**.
- The work has substantial real infrastructure:
  - SMPL-X prior;
  - layer-wise human prior fusion;
  - coarse prior normal pack;
  - detail normal refiner;
  - external real-data bridge;
  - Open3D ROI visualization;
  - multiple external teacher adapters.
- The blocker is no longer “missing scripts”.
- The blocker is **high-quality, continuous, aligned head/face geometry teacher or direct sparse-view geometry improvement**.
- Do not let the next window confuse:
  - 2D normal visual quality;
  - targetcam30-only gains;
  - confidence-collapse point inflation;
  - visualization ROI corrections;
  - engineering smoke tests;
  with final sparse-view human point-cloud quality.
