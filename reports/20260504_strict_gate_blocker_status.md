# Teacher/Candidate Strict Gate Blocker Status

Date: 2026-05-06

## Current Truth

No local candidate or teacher currently passes the strict mentor gate. Cloud upload/run remains blocked. Numeric point counts, normal consistency, or depth-compatible teacher coverage are not accepted without explicit Open3D visual pass and full-body/hand bottom-line pass.

## Counts

- Generated at: `2026-05-05T23:01:08.280588+00:00`
- Schema version: `20260504_visual_fullbody_hands_v2`
- Roots scanned: `13`
- Candidate gate summaries scanned: `26`
- Teacher gate summaries scanned: `81`
- Strict full mentor candidate passes: `0`
- Strict teacher passes: `0`
- Kinect coordinate audits scanned: `18`
- Kinect coordinate audit passes: `2`
- Raw Kinect sensor full-body/hand audits scanned: `0`
- Raw Kinect sensor full-body/hand passes: `0`
- SMPL-X weak-anchor audits scanned: `7`
- SMPL-X weak-anchor passes: `1`
- Legacy/diagnostic apparent green packages: `0`
- Full mentor packages with numeric pass but visual fail: `1`
- Legacy visual passes failing current full-body/hand visual schema: `0`
- Teacher packages with numeric pass but visual fail: `0`
- Other strict teacher non-passes: `81`
- Orphan visible-surface teacher audits scanned: `0`
- Orphan visible-surface teacher audit passes: `0`

## Strict Passes

None.

## Numeric Positive But Visual Negative Candidates

- `r38_r37_depthauth_worldsync`: failed=`fullbody, normal, shape, visual`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260503\candidate_gate_r38_r37_depthauth_worldsync`

## Legacy Visual Passes Failing Current Schema

These packages contain an older visual review payload that may mark visual review as passed, but it does not satisfy the current mentor visual schema with full-body side/back/iso and attached-hand checks. They are blocked.


## Numeric Positive But Visual Negative Teachers


## Other Strict Teacher Non-Passes

These teacher audits fail numeric and/or explicit visual gates. They are listed because numeric-only green subsets can otherwise hide a failed overall teacher.

- `teacher_gate_existing60v_denseonly_face_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_existing60v_denseonly_face_allviews`
- `teacher_gate_existing60v_denseonly_head_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_existing60v_denseonly_head_allviews`
- `teacher_gate_external_60v_surfacepose_facecore_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_external_60v_surfacepose_facecore_allviews`
- `teacher_gate_external_lhm500m_cam30_alignheadface_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_external_lhm500m_cam30_alignheadface_allviews`
- `teacher_gate_external_lhm_mini_cam45_sweep01_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_external_lhm_mini_cam45_sweep01_allviews`
- `teacher_gate_external_lhm_stronger_cam30_alignheadface_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_external_lhm_stronger_cam30_alignheadface_allviews`
- `teacher_gate_external_pifuhd512_cam00_facecore_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_external_pifuhd512_cam00_facecore_allviews`
- `teacher_gate_external_pshuman_altviews_cam15_alignall_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_external_pshuman_altviews_cam15_alignall_allviews`
- `teacher_gate_external_pshuman_altviews_cam45_alignall_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_external_pshuman_altviews_cam45_alignall_allviews`
- `teacher_gate_external_pshuman_official_hq1024_headface_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_external_pshuman_official_hq1024_headface_allviews`
- `teacher_gate_external_pshuman_orig6_true1024_cam30_alignheadface_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_external_pshuman_orig6_true1024_cam30_alignheadface_allviews`
- `teacher_gate_external_pshuman_orig6_true1024_cam30_similarity_refined_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_external_pshuman_orig6_true1024_cam30_similarity_refined_allviews`
- `teacher_gate_facelandmark_internal60v_tsdf_r2_v22_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_facelandmark_internal60v_tsdf_r2_v22_allviews`
- `teacher_gate_facelandmark_internal60v_tsdf_r2_v22_clean_min100_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_facelandmark_internal60v_tsdf_r2_v22_clean_min100_allviews`
- `teacher_gate_facelandmark_internal60v_tsdf_r2_v22_facepatch_only_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_facelandmark_internal60v_tsdf_r2_v22_facepatch_only_allviews`
- `teacher_gate_facelandmark_realcam60v_smplxfull_v09_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_facelandmark_realcam60v_smplxfull_v09_allviews`
- `teacher_gate_facelandmark_smplxfull_kinecthair_v10_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_facelandmark_smplxfull_kinecthair_v10_allviews`
- `teacher_gate_facelandmark_smplxfull_kinecthairline_v11_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_facelandmark_smplxfull_kinecthairline_v11_allviews`
- `teacher_gate_internal60v_tsdf_r2_depthnormal_v01_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_internal60v_tsdf_r2_depthnormal_v01_allviews`
- `teacher_gate_internal60v_tsdf_r2_depthnormal_v02_p20_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_internal60v_tsdf_r2_depthnormal_v02_p20_allviews`
- `teacher_gate_internal60v_tsdf_r2_depthnormal_v03_p0_thick_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_internal60v_tsdf_r2_depthnormal_v03_p0_thick_allviews`
- `teacher_gate_internal60v_tsdf_r2_v03_clean_largest_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_internal60v_tsdf_r2_v03_clean_largest_allviews`
- `teacher_gate_kinect60v_all_fused_bridged_on_original6v_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_kinect60v_all_fused_bridged_on_original6v_allviews`
- `teacher_gate_kinect60v_all_fused_sourceworld_on_original6v_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_kinect60v_all_fused_sourceworld_on_original6v_allviews`
- `teacher_gate_kinect60v_all_poisson_bridged_on_original6v_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_kinect60v_all_poisson_bridged_on_original6v_allviews`
- `teacher_gate_kinect60v_all_poisson_sourceworld_on_original6v_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_kinect60v_all_poisson_sourceworld_on_original6v_allviews`
- `teacher_gate_kinect60v_all_s0005_original6v_subset_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_kinect60v_all_s0005_original6v_subset_allviews`
- `teacher_gate_kinect60v_all_s0005_to_original6v_bridge_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_kinect60v_all_s0005_to_original6v_bridge_allviews`
- `teacher_gate_kinect60v_axes_s0005_original6v_subset_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_kinect60v_axes_s0005_original6v_subset_allviews`
- `teacher_gate_kinect60v_axes_s0005_to_original6v_bridge_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_kinect60v_axes_s0005_to_original6v_bridge_allviews`
- `teacher_gate_kinect60v_fused_bridged_on_original6v_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_kinect60v_fused_bridged_on_original6v_allviews`
- `teacher_gate_kinect60v_fused_sourceworld_on_original6v_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_kinect60v_fused_sourceworld_on_original6v_allviews`
- `teacher_gate_kinect_original6v_all_pointalign_allviews_diagnostic`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_kinect_original6v_all_pointalign_allviews_diagnostic`
- `teacher_gate_kinect_original6v_all_s0005_diagnostic_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_kinect_original6v_all_s0005_diagnostic_allviews`
- `teacher_gate_kinect_tsdf_v21_original6v_camaxes_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_kinect_tsdf_v21_original6v_camaxes_allviews`
- `teacher_gate_shared_mesh_colmap_kinect_smplx_v01_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_shared_mesh_colmap_kinect_smplx_v01_allviews`
- `teacher_gate_shared_mesh_frustum_colmap_smplx_kinect_v02_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_shared_mesh_frustum_colmap_smplx_kinect_v02_allviews`
- `teacher_gate_shared_mesh_kinect60v_hairline_v11_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_shared_mesh_kinect60v_hairline_v11_allviews`
- `teacher_gate_shared_mesh_kinect60v_headface_v04_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_shared_mesh_kinect60v_headface_v04_allviews`
- `teacher_gate_shared_mesh_kinect_original6v_subset_v13_allviews`: failed=`numeric, visual, overall`; output=`D:\vggt\vggt-main\output\normal_line_multiview_20260502\teacher_gate_shared_mesh_kinect_original6v_subset_v13_allviews`
- ... 41 more omitted from markdown; see JSON registry.

## Orphan Visible-Surface Teacher Audits

These are legacy visibility/coverage audits whose directory names can look like teacher gates, but they do not contain a current `teacher_gate_summary.json`. They are tracked explicitly so they cannot be mistaken for passing strict teachers.


## Kinect Coordinate Audits

These are diagnostic coordinate-chain audits, not candidate passes. A Kinect route must still become a strict teacher-gate pass before any training.

- `kinect_teacher_60v_all_camera_axes_s0005_diagnostic_targets`: pass=`False`, teacher_targets_written=`True`, roi=`all`, align_p50=`0.06966881175102355`, dist_to_base_p50=`0.09091923385858536`, views=`58/60`, failed=`distance_to_base_p50`
- `kinect_teacher_60v_all_camera_axes_s0005_gate`: pass=`False`, teacher_targets_written=`False`, roi=`all`, align_p50=`0.06966881175102355`, dist_to_base_p50=`0.09091923385858536`, views=`58/60`, failed=`distance_to_base_p50`
- `kinect_teacher_original6v_all_axisaffine_camera_axes_s0005_gate`: pass=`False`, teacher_targets_written=`False`, roi=`all`, align_p50=`0.05450006317508181`, dist_to_base_p50=`0.14861463010311127`, views=`5/6`, failed=`alignment_residual_p95, distance_to_base_p50, distance_to_base_p95`
- `kinect_teacher_original6v_all_camera_axes_s0005_diagnostic_targets`: pass=`False`, teacher_targets_written=`True`, roi=`all`, align_p50=`0.03156999227160744`, dist_to_base_p50=`0.10024239122867584`, views=`5/6`, failed=`distance_to_base_p50`
- `kinect_teacher_original6v_all_camera_axes_s0005_gate`: pass=`False`, teacher_targets_written=`False`, roi=`all`, align_p50=`0.03156999227160744`, dist_to_base_p50=`0.10024239122867584`, views=`5/6`, failed=`distance_to_base_p50`
- `kinect_teacher_original6v_all_camera_axes_s0p001_gate`: pass=`False`, teacher_targets_written=`False`, roi=`all`, align_p50=`0.029741198818909256`, dist_to_base_p50=`0.09992315992712975`, views=`5/6`, failed=`distance_to_base_p50`
- `kinect_teacher_original6v_all_camera_axes_s0p002_gate`: pass=`False`, teacher_targets_written=`False`, roi=`all`, align_p50=`0.030152260014624448`, dist_to_base_p50=`0.10001049190759659`, views=`5/6`, failed=`distance_to_base_p50`
- `kinect_teacher_original6v_all_camera_axes_s0p003_gate`: pass=`False`, teacher_targets_written=`False`, roi=`all`, align_p50=`0.030594950750398427`, dist_to_base_p50=`0.10007932037115097`, views=`5/6`, failed=`distance_to_base_p50`
- `kinect_teacher_original6v_all_camera_axes_s0p004_gate`: pass=`False`, teacher_targets_written=`False`, roi=`all`, align_p50=`0.03106804065543843`, dist_to_base_p50=`0.10015604645013809`, views=`5/6`, failed=`distance_to_base_p50`
- `kinect_teacher_original6v_all_camera_axes_s0p006_gate`: pass=`False`, teacher_targets_written=`False`, roi=`all`, align_p50=`0.032099502522875925`, dist_to_base_p50=`0.10032584890723228`, views=`5/6`, failed=`distance_to_base_p50`
- `kinect_teacher_original6v_all_camera_axes_s0p008_gate`: pass=`False`, teacher_targets_written=`False`, roi=`all`, align_p50=`0.033120865596582325`, dist_to_base_p50=`0.10048520565032959`, views=`5/6`, failed=`distance_to_base_p50`
- `kinect_teacher_original6v_all_camera_axes_s0p010_gate`: pass=`False`, teacher_targets_written=`False`, roi=`all`, align_p50=`0.03392090358236782`, dist_to_base_p50=`0.10066233202815056`, views=`5/6`, failed=`distance_to_base_p50`
- `kinect_teacher_original6v_all_camera_axes_s0p015_gate`: pass=`False`, teacher_targets_written=`False`, roi=`all`, align_p50=`0.03777411044791482`, dist_to_base_p50=`0.1011340282857418`, views=`5/6`, failed=`distance_to_base_p50`
- `kinect_teacher_original6v_all_camera_axes_s0p020_gate`: pass=`False`, teacher_targets_written=`False`, roi=`all`, align_p50=`0.044383048093371916`, dist_to_base_p50=`0.10165897384285927`, views=`5/6`, failed=`distance_to_base_p50`
- `kinect_teacher_original6v_all_camera_axes_s0p050_gate`: pass=`False`, teacher_targets_written=`False`, roi=`all`, align_p50=`0.07179964253297587`, dist_to_base_p50=`0.1075281873345375`, views=`5/6`, failed=`distance_to_base_p50`
- `kinect_teacher_original6v_all_pointalign_allviews_diagnostic`: pass=`False`, teacher_targets_written=`True`, roi=`all`, align_p50=`0.027388959854692252`, dist_to_base_p50=`0.027482631616294384`, views=`5/6`, failed=`non_circular_alignment`
- `kinect_teacher_60v_head_camera_axes_s0005_gate`: pass=`True`, teacher_targets_written=`True`, roi=`head`, align_p50=`0.06966881175102355`, dist_to_base_p50=`0.07487250864505768`, views=`49/60`, failed=``
- `kinect_teacher_60v_headface_camera_axes_s0005_gate`: pass=`True`, teacher_targets_written=`True`, roi=`head_face`, align_p50=`0.06966881175102355`, dist_to_base_p50=`0.07487250864505768`, views=`49/60`, failed=``

## Raw Kinect Sensor Full-Body/Hand Audits

These audits render raw Kinect sensor depth in real calibrated space. They can disqualify a sensor route for full-body/hands, but they are not candidate passes and do not authorize training without a strict teacher pass.


## SMPL-X Weak Full-Body/Hand Anchor Audits

These audits check SMPL-X only as a weak body/hand topology anchor. They are not face teachers, not candidate passes, and do not override the final Open3D full/head/face/hands gate.

- `raw_smplx_mesh_hand_anchor_preflight_softmatte_fullbody`: pass=`False`, body=`True` (6 views), hand=`False` (1/2 eligible), compact_hand_views=`1`, implausible_hand_boxes=`1`, failed=`hand_gate`
- `raw_smplx_mesh_hand_anchor_preflight_softmatte_fullbody_both_all`: pass=`False`, body=`True` (6 views), hand=`False` (1/2 eligible), compact_hand_views=`1`, implausible_hand_boxes=`1`, failed=`hand_gate`
- `raw_smplx_mesh_hand_anchor_preflight_softmatte_fullbody_left_all`: pass=`False`, body=`True` (6 views), hand=`False` (1/2 eligible), compact_hand_views=`2`, implausible_hand_boxes=`0`, failed=`hand_gate`
- `raw_smplx_mesh_hand_anchor_preflight_softmatte_fullbody_left_any`: pass=`False`, body=`True` (6 views), hand=`False` (1/2 eligible), compact_hand_views=`2`, implausible_hand_boxes=`0`, failed=`hand_gate`
- `raw_smplx_mesh_hand_anchor_preflight_softmatte_fullbody_right_any`: pass=`False`, body=`True` (6 views), hand=`False` (1/2 eligible), compact_hand_views=`1`, implausible_hand_boxes=`1`, failed=`hand_gate`
- `smplx_weak_anchor_preflight_softmatte_fullbody`: pass=`False`, body=`True` (5 views), hand=`False` (0/2 eligible), compact_hand_views=`0`, implausible_hand_boxes=`2`, failed=`hand_gate`
- `raw_smplx_mesh_hand_anchor_preflight_softmatte_fullbody_perbox_best`: pass=`True`, body=`True` (6 views), hand=`True` (2/2 eligible), compact_hand_views=`2`, implausible_hand_boxes=`0`, failed=``

## Legacy Or Diagnostic Apparent Green

These entries have old or diagnostic status fields that can look green, but they are not strict full mentor gates. They must not be used for a pass claim without re-packaging under the current full protocol.


## Frozen / Negative Routes

- HART-style PnP camera replacement: local ablation did not improve head/face Open3D or beat the VGGT camera-head chain.
- r16/r18/r19 more epoch or same-config retry: consistency gains did not translate to face/head point cloud quality.
- r57/r58/r59/r60 and r61-r68: blocked by strict same-protocol, normal/shape, full-body/hand, or visual gates.
- TSDF from signfix depth and 60-view direct/fused surfaces: numeric/depth-compatible positives are shell-like or coordinate/depth incompatible.
- Kinect/MVS/COLMAP/external pointcloud projection patches: not a passing teacher under same-protocol depth/visual gates.
- Visual hull/keypoint/MediaPipe relief/SMPL-X face scaffold: numeric gains do not produce modeled personal face/head/hairline geometry.

## Active Blocker

The local repo still lacks a continuous, aligned, visually valid head/face/hairline surface teacher that can be projected back to `0012_11_frame0000_6views_sparseproto_headshoulder_crop` and pass both numeric and explicit Open3D visual teacher gates.

## Allowed Next Actions Before Any Training

- Kinect coordinate convention or raw-sensor teacher audit only: no projection patch, no training, no cloud unless the resulting teacher passes strict numeric plus explicit Open3D visual gate.
- SMPL-X weak full-body/hand anchor audit only: not a face teacher and not a pass claim.
- Multi-view consistent face surface teacher design: must produce one shared 3D surface and pass teacher-gate before one-frame overfit.

## Cloud Policy

No cloud upload or cloud run until a local candidate passes the full strict mentor gate, including full-body/hands and explicit Open3D visual review. If the route uses teacher-supervised training, that teacher must also pass the strict teacher gate first. Teacherless/self-supervised routes are not required to produce a teacher pass, but they still need a strict candidate pass.
