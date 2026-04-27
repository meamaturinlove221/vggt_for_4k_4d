# 2026-04-27 Next-Window Guidance and Truthful Checklist

## 0. Non-Negotiable Truth State
- Mentor-final sparse-view human geometry is **not reached yet**.
- Do **not** claim "达标" unless both conditions hold:
  1. Same-protocol `6views_sparseproto_headshoulder_crop` ROI metrics clearly beat `signfix ckpt4`.
  2. Open3D head/face close-ups visibly improve face/head structure, especially eyes, nose, mouth, hairline, and face boundary.
- The current valid reference is:
  - `D:/vggt/vggt-main/output/modal_results/20260424_signfix_ckpt4_on6v_headshoulder`
  - official same-protocol face ROI `16825`, head ROI `40527`, p40 confidence threshold about `38.5067`.
- High point count alone is not a pass. Reject results caused by `conf_threshold≈1`, synthetic confidence flooring, 2D-only projection, or visibly noisier/fragmented surfaces.

## 1. Mentor's Full Technical Guidance
### 1.1 SMPL / SMPL-X Positioning
- The SMPL pose/shape for current 4K4D experiments is available from input annotations; it is used as a pose-aligned geometric prior.
- The repo still needs a real-data route that can consume external SMPL-X estimator/fitter outputs.
- The current repo route is **not** an in-repo image-to-SMPL-X regressor. It is:
  - external estimator/fitter results
  - repo import/alignment bundle
  - scene-level `prior_maps.npz`
  - prior-enabled VGGT inference/training.
- Do not describe this as "repo already regresses SMPL-X from real images".

### 1.2 Point Cloud Quality Target
- Mentor wants sparse-view reconstruction, especially 6-view, to show competitive human-region point quality.
- The target is not just complete body silhouette; head/face must show useful geometry detail.
- Compared references:
  - 60-view / GS / HumanRAM can show eyes and high-frequency facial detail.
  - PSHuman-style quality is the normal-map / high-quality human surface reference.
- If 6-view cannot clearly show face detail, the method has weak competitiveness even if 60-view works.

### 1.3 Normal Branch / PIFuHD-Style Direction
- VGGT should output a dense normal map / human normal branch in addition to depth/points.
- Depth supervision alone is not enough; use normal constraints to improve surface geometry.
- PIFuHD is guidance for **coarse-to-fine, image-aligned local detail refinement**, not a command to abandon VGGT.
- The correct next branch wording:
  - `detail_normal_refiner`
  - or `pifuhd_style_normal_refine`
- Its role:
  - not replacing VGGT
  - not replacing coarse SMPL-X prior normal
  - refining coarse prior normal with image-aligned residual detail.

### 1.4 Required Refiner Design
- Inputs:
  - RGB crop
  - coarse prior normal crop
  - human mask
- Output:
  - refined normal
  - or normal residual
- First ROI:
  - head / neck
  - shoulder line
- First objective:
  - make head boundary and hairline clearer
  - do not try full-body high-resolution wrinkles first.
- Supervision must not use coarse prior normal itself as detail teacher.
- Priority teacher sources:
  1. 60-view multiview fused geometry surface normals
  2. high-quality external normal estimator
  3. local mesh/surface fitting pseudo GT
- If teacher quality is insufficient, train/refine only visible regions.
- Losses should include:
  - cosine normal loss
  - edge-aware loss
  - mask-restricted loss
  - ROI boundary upweighting
- Metrics must separately track:
  - head ROI
  - face ROI
  - hairline / ear / back-of-head boundary
  - not only full-image average loss.

### 1.5 Experiment Order
- First make 60-view work.
- Then downshift to 13-view.
- Then 7-view.
- Only then push 6-view end-to-end sparse-view training.
- Before stable detail branch, avoid large blind sparse-view training.
- Start with:
  - single frame
  - small ROI
  - small batch
  - overfit check
  - then cross-frame generalization
  - then multi-case.

### 1.6 Visualization Requirements
- Every serious candidate must save:
  - RGB
  - coarse prior normal
  - refined normal
  - coarse-vs-refined diff
  - head ROI
  - face ROI
  - Open3D point cloud close-ups
  - fixed camera-aligned views
  - failure cases.
- Full-body only is insufficient.
- Open3D must be used for point cloud visualization; Meshlab screenshots alone are not enough.

### 1.7 Advisor Communication Wording
- Say:
  - "current coarse prior normal chain is established"
  - "next step borrows PIFuHD coarse-to-fine idea for high-resolution local detail refinement"
  - "60-view proves coarse prior is aligned/stable/displayable, but visible detail still needs improvement"
  - "4v probe collapsed to silhouette-only and is downgraded to an internal failed branch"
- Do not say:
  - "VGGT already outputs high-quality predicted normal"
  - "60v already fully meets final quality"
  - "we are switching to PIFuHD as the main method"
  - "4v probe is only slightly worse".

## 2. Current Repo State
### 2.1 Main Code Paths
- Model: `D:/vggt/vggt-main/vggt/models/vggt.py`
- Loss: `D:/vggt/vggt-main/training/loss.py`
- Dataset: `D:/vggt/vggt-main/training/data/datasets/dna4k4d_pseudo.py`
- Train: `D:/vggt/vggt-main/modal_4k4d_vggt_train.py`
- Infer: `D:/vggt/vggt-main/modal_4k4d_vggt_infer.py`
- Open3D render: `D:/vggt/vggt-main/tools/render_open3d_pointcloud.py`
- External bundle bridge: `D:/vggt/vggt-main/tools/build_scene_prior_from_external_bundle.py`
- External mesh raycast case: `D:/vggt/vggt-main/tools/build_external_mesh_raycast_training_case.py`
- Visible teacher audit: `D:/vggt/vggt-main/tools/audit_visible_surface_teacher.py`
- Truth report: `D:/vggt/vggt-main/reports/20260424_truthful_sparse_view_headface_status_update.md`

### 2.2 Important New/Modified Tools
- `D:/vggt/vggt-main/tools/render_open3d_pointcloud.py`
  - now supports `--conf-threshold` for absolute confidence comparisons.
- `D:/vggt/vggt-main/tools/refine_mesh_translation_for_visible_surface.py`
  - local PSHuman/LHM mesh translation gate helper.
- `D:/vggt/vggt-main/tools/refine_mesh_similarity_for_visible_surface.py`
  - side worker may have added this for translation+scale+rotation gate search.
- `D:/vggt/vggt-main/modal_lhm_mesh_teacher.py`
  - supports LHM Modal runs and PLY/OBJ collection.
- `D:/vggt/vggt-main/modal_econ_mesh_teacher.py`
  - backup ECON wrapper, not mainline-proven.

### 2.3 Coarse Prior Normal Pack
- Canonical:
  `D:/vggt/vggt-main/output/normal_advisor_pack_20260421_coarseprior`
- Legacy mirror:
  `D:/vggt/vggt-main/output/normal_advisor_pack_20260421`
- These packs are advisor-facing coarse-prior normal evidence only.
- They are not final sparse-view geometry pass evidence.

## 3. What Has Been Ruled Out
### 3.1 Do Not Reopen
- `projected targetpatch` / summary-token patch: rejected.
- `conf_threshold=1.0` high-point-count results: false positive.
- Synthetic confidence floor / boost: cannot be pass evidence.
- 2D-only ROI projections: diagnostic only.
- Open3D outlier filtering/postprocess: can clean display but not improve geometry.

### 3.2 Negative Teacher / Fusion Lines
- DepthPro direct/fusion/prior oracle:
  - not pass; direct variants twist/side-artifact.
- LHM-MINI / LHM-500M-HF / LHM-1B-HF:
  - mesh export works, teacher gate fails coverage.
- PSHuman true1024 cam30:
  - mesh is stronger than LHM but still fails visible-surface gate.
- PSHuman translation refinement:
  - improved depth-compatible hits to `2706/8058`, but still hole ratio `0.664`.
- Photometric depth sweep:
  - accepted coverage too low; no visual Open3D improvement.
- Headface crop / depth-unprojection display:
  - not a pass; mostly fragmented/noisy.

### 3.3 Negative Training Lines
- `pointnormal_r3_mixed` from ckpt4:
  - all official face ROI below `16825`.
- `pointnormal_r3_6vonly` from ckpt4:
  - all official face ROI below `16825`.
- `teachergeom_roi_combo` from ckpt4:
  - all official face ROI below `16825`.
- Local fixed-threshold checks can show more points, but Open3D face view is noisier/fragmented, not clearer.

## 4. Latest Long-Running Work Results
### 4.1 PSHuman Similarity Gate
- Result: negative.
- Best final similarity search:
  - depth-compatible hits `6129/8058`
  - depth hole ratio `0.239`
  - largest component `1.0`
  - median residual `0.0064m`
- It fails because hole ratio is above `0.15`.
- Diagnostic no-boost fusion did not improve Open3D face/head evidence.
- Evidence:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/pshuman_original6v_true1024_cam30_similarity_refine_v01/mesh_similarity_refine_summary.json`.

### 4.2 PSHuman Alternate Views
- Result: negative.
- Best alternate source was `cam45 align=all`, only `1597` depth-compatible hits with hole ratio `0.795`.
- `cam15`, `cam59`, and `head_face` alignments are worse.
- Evidence:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/pshuman_original6v_true1024_altviews_gate_summary.json`.

### 4.3 Geoonly Continuation Training
- Result: negative.
- Output:
  `vggt_4k4d_train/20260427_original6v_external60v_surfacepose_facecore_geoonly_continue_lr2e8_e4`.
- P40 point counts increase, but p40 confidence collapses near `1.1-1.3` and the Open3D face becomes torn/sheet-like.
- Evidence:
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/geoonly_continue_p40_roi_eval_20260427.json`,
  `D:/vggt/vggt-main/output/detail_normal_refiner_20260427/fixedconf385_open3d_20260427_geoonly_continue_ckpt3_on6v_headshoulder_face/camera_view_03.png`.
- Do not continue this exact branch without a better teacher mask/surface gate.

## 5. Next Execution Checklist
### 5.1 Always Use These Command Rules
- Workdir:
  `D:/vggt/vggt-main`
- PowerShell only inside Codex; no external Windows windows.
- `shell_command` should use `login:false`.
- Python/Modal prefix:
  `$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'`
- Local Open3D Python:
  `D:/anaconda/envs/g3splat/python.exe`
- Use `apply_patch` tool for file edits; do not patch via shell.

### 5.2 Poll Active Agents
- Poll:
  - `019dcdd0-69d5-7d80-8421-de3f1125fc71`
  - `019dcdd0-a682-7fc0-b4bf-2c1d499df34f`
  - `019dcde0-caed-7f90-b9fc-d48193a2c82c`
- Close completed agents to free slots.
- Periodically inspect processes and kill only obvious stale Modal/Python children; never kill Codex/WPS blindly.

### 5.3 Evaluate Any New Training Result
- Run official same-protocol ROI:
  - scene: `scenes/0012_11_frame0000_6views_sparseproto_headshoulder_crop`
  - conf percentile: `40`
  - compare against `face=16825`, `head=40527`, threshold `38.5067`.
- Also render fixed absolute threshold:
  - `tools/render_open3d_pointcloud.py`
  - `--conf-threshold 38.5067`
  - `--roi face` and `--roi head`
  - `--camera-view-indices 3`
- Required visuals:
  - `face_close.png`
  - `head_close.png`
  - `camera_view_03.png` or crop if generated.

### 5.4 Evaluate Any New Teacher
- Teacher must pass visible-surface gate before fusion/training.
- Required gate thresholds:
  - depth-compatible hit pixels `>=5000`
  - depth hole ratio `<=0.15`
  - largest component ratio `>=0.80`
  - median depth residual `<=0.012m`
- If gate fails, update truth report and stop that teacher branch.

### 5.5 If Active Branches Fail
- Do not go back to targetpatch.
- Do not launch large blind training.
- Most credible next routes:
  1. improve high-quality teacher alignment/surface quality until visible-surface gate passes
  2. build a true `detail_normal_refiner` ROI overfit on a teacher that passes gate
  3. move from 60v -> 13v -> 7v -> 6v after teacher/refiner quality is verified.

## 6. Pass/Fail Checklist
- [ ] Same-protocol official face ROI clearly exceeds `16825`.
- [ ] Same-protocol head ROI does not regress.
- [ ] Confidence comparison is fair and not created by `conf=1` collapse.
- [ ] Open3D face close-up is visibly clearer, not just denser/noisier.
- [ ] Open3D head close-up improves hairline/boundary without severe artifacts.
- [ ] Camera-aligned view shows more believable face structure.
- [ ] Any teacher used for fusion/training passes visible-surface gate first.
- [ ] Report states remaining failures truthfully.
- [ ] Advisor wording never claims final pass before visual and quantitative evidence agree.

## 7. Current Bottom Line
- Coarse SMPL-X prior normal chain: closed as a coarse-prior evidence pack.
- Real-data bridge: external bundle -> scene prior bridge exists and smoke-tested.
- Sparse-view 6-view mentor-final geometry: **still open**.
- The immediate bottleneck is not code plumbing; it is finding/learning a high-quality, depth-compatible head/face teacher/refinement signal that actually improves Open3D sparse-view face geometry.
