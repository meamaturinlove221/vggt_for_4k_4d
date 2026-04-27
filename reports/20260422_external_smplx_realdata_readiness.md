# External SMPL-X Real-Data Readiness Note

Scope: `tools/import_external_smplx_params.py` and `tools/prepare_4k4d_prior_training_case.py`

## Bottom line

The current external SMPL-X path is usable for a real-data operator flow if you already have:

- a scene directory with `scene_manifest.json`, exported RGBs, and masks
- a `predictions.npz` whose view order matches `scene_manifest.json`
- an external SMPL-X payload with at least `betas` and `fullpose`
- ideally an external camera payload aligned to the same scene views

It is not fully "safe by construction" for arbitrary real bundles yet. The code enforces order and basic tensor shapes in several places, but it still accepts some semantically weak inputs and does not verify frame/gender/calibration consistency end-to-end.

## Recommended operator flow

### Shortest flow when `scene_dir` and `predictions.npz` already exist

1. Import the external SMPL-X bundle into the repo's normalized intermediate format.

```powershell
python tools/import_external_smplx_params.py `
  --smplx-input <external_smplx.npz_or_json> `
  --camera-input <external_cameras.npz_or_json> `
  --scene-dir <scene_dir> `
  --output-dir <bundle_dir> `
  --frame-idx <frame_idx> `
  --strict `
  --overwrite
```

2. Build the self-contained training case from the normalized bundle.

```powershell
python tools/prepare_4k4d_prior_training_case.py `
  --scene-dir <scene_dir> `
  --predictions-npz <predictions.npz> `
  --external-prior-bundle <bundle_dir> `
  --output-dir <case_dir> `
  --smplx-model-dir <smplx_model_dir> `
  --smplx-gender <neutral|female|male> `
  --geometry-prior-source smplx_mesh `
  --overwrite
```

### Full repo-local flow from a raw 4K4D scene

1. Export the scene and scene manifest.

```powershell
python tools/export_4k4d_scene.py `
  --dataset-root <dataset_root> `
  --seq <seq_id> `
  --frame <frame_id> `
  --target-camera <camera_id> `
  --auto-sources <N> `
  --output-dir <scene_dir> `
  --smplx-model-dir <smplx_model_dir> `
  --overwrite
```

2. Run VGGT inference to produce `predictions.npz`.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_modal_4k4d_vggt_infer.ps1 `
  -LocalSceneDir <scene_dir> `
  -OutputSubdir <modal_output_subdir>
```

3. Import the external SMPL-X/camera bundle.

4. Prepare the training case with `--external-prior-bundle`.

## What the importer currently guarantees

`tools/import_external_smplx_params.py`:

- accepts `.npz` and `.json` SMPL-X and camera payloads
- normalizes common aliases:
  - SMPL-X: `betas`, `shape`, `shape_params`; `fullpose`, `body_pose`, `pose`, `poses`; `transl`, `translation`, `global_trans`; `scale`, `global_scale`; `expression`, `expr`; `gender`, `smplx_gender`
  - cameras: `camera_ids`, `camera_id`, `ids`, `names`; `intrinsic`, `intrinsics`, `K`; `cam_to_world`, `c2w`, `RT`, `camera_to_world`; `world_to_cam`, `w2c`, `extrinsic`, `extrinsics`
- slices time-major arrays by `--frame-idx`
- converts camera matrices to `4x4`
- derives missing `cam_to_world` from `world_to_cam` and vice versa
- when a scene manifest is supplied, reorders cameras to `scene_manifest.json -> exported_views -> camera_id`
- writes:
  - `normalized_smplx_params.npz`
  - `normalized_camera_params.npz`
  - `external_prior_bundle_manifest.json`
- in `--strict` mode, fails for:
  - missing required SMPL-X keys after normalization (`betas`, `fullpose`)
  - missing manifest
  - unparsable camera payloads
  - missing cameras needed by the scene view order
  - no aligned cameras after parsing

## What case preparation currently guarantees

`tools/prepare_4k4d_prior_training_case.py`:

- resolves a bundle directory to `external_prior_bundle_manifest.json`
- requires the referenced normalized SMPL-X payload to exist
- if a camera payload is present, requires:
  - keys `camera_ids`, `intrinsics`, `cam_to_world`, `world_to_cam`
  - `intrinsics.shape == (V, 3, 3)`
  - `cam_to_world.shape == (V, 4, 4)`
  - `world_to_cam.shape == (V, 4, 4)`
  - `camera_ids` exactly equal the scene manifest view order
- requires `predictions["depth"]` to be square `[V, H, W, 1]`
- requires the prediction view count to match the scene manifest view count
- resizes scene RGBs and masks to the prediction target size
- if SMPL-X model files, `betas`, `fullpose`, and external cameras are all available, builds dense external SMPL-X surface priors and summary tokens
- writes a self-contained case:
  - `inputs.npz`
  - `targets.npz`
  - `case_manifest.json`
  - `smplx_frame_<frame>.npz`

On the current smoke fixture, the prepared case contains:

- `inputs.npz`: `images`, `point_masks`, `prior_maps`, `prior_mask`, `camera_ids`, `view_roles`, `prior_summary_tokens`
- `targets.npz`: `depths`, `extrinsics`, `intrinsics`, `cam_points`, `world_points`, `depth_conf`, `world_points_conf`, `prior_depths`, `prior_points`, `prior_normals`

## Current behavior that matters for real data

- External prior maps always start from the scene masks.
- External 2D keypoints are not imported. The external path writes `keypoint_heatmap_source = zeros_no_external_2d_keypoints`.
- If external cameras are present, geometry priors are rasterized with those cameras.
- If external cameras are absent, geometry prior generation falls back to scene/dataset cameras when available.
- If `scene_manifest` has a `preprocess_variant`, geometry regeneration is skipped entirely.

## Remaining gaps

These are the main reasons I would call the path "usable but not yet real-data hardened":

- `body_pose` is accepted as an alias for `fullpose`. That is convenient, but semantically risky because many external bundles store only body joints there, while the mesh path later requires the full SMPL-X joint count.
- Import-time SMPL-X validation is shallow. The importer checks presence of `betas` and `fullpose`, but not whether `fullpose` has the exact joint count expected by the SMPL-X model.
- Non-strict import mode is too permissive for real data:
  - missing camera intrinsics become identity matrices
  - missing camera extrinsics can become identity transforms
  - missing `camera_ids` can be synthesized as `0..N-1`
  - missing required SMPL-X keys only emit a warning and still write a bundle
- The prepare step does not verify that external cameras match `predictions.npz` intrinsics/extrinsics; it only checks view order and array shapes.
- The prepare step does not verify that `bundle_manifest.frame_idx` matches `scene_manifest["frame_id"]` or the frame used to generate `predictions.npz`.
- Imported `gender` is preserved in the normalized SMPL-X payload but is not consumed automatically by case prep. The mesh path uses `--smplx-gender`, defaulting to `neutral`.
- There is no external import path for 3D keypoints, so `--geometry-prior-source keypoints3d` is not useful on the external bundle branch today.
- Prediction payload validation is partial. Missing keys like `depth_conf`, `world_points_conf`, or `world_points` fail later with raw key errors instead of a targeted preflight message.
- The importer accepts `.npz` with `allow_pickle=True`, which is flexible but weakens the input contract.

## Verified smoke results

I re-ran the current path on the repo fixture under `output/smoke_external_bundle_case`.

### Verified success

```powershell
python tools/import_external_smplx_params.py `
  --smplx-input output\smoke_external_bundle_case\bundle\normalized_smplx_params.npz `
  --camera-input output\smoke_external_bundle_case\bundle\normalized_camera_params.npz `
  --scene-dir output\smoke_external_bundle_case\scene `
  --output-dir output\smoke_external_bundle_case\bundle_reimport_strict `
  --overwrite `
  --strict

python tools/prepare_4k4d_prior_training_case.py `
  --scene-dir output\smoke_external_bundle_case\scene `
  --predictions-npz output\smoke_external_bundle_case\predictions.npz `
  --external-prior-bundle output\smoke_external_bundle_case\bundle `
  --output-dir output\smoke_external_bundle_case\case_verify `
  --overwrite
```

Observed result:

- import succeeded and produced an aligned bundle manifest with `camera_ids_ordered = ["00", "01"]`
- case prep succeeded
- resulting `case_manifest.json` reports:
  - `prior_geometry_source = "smplx_mesh_external_camera_rasterize_knnfill"`
  - `smplx_vertex_feature_meta.source = "external_smplx_bundle_pose_aligned_surface_prior"`
  - `keypoint_heatmap_source = "zeros_no_external_2d_keypoints"`
  - `smplx_output = "smplx_frame_0000.npz"`
  - `keypoints3d_output = null`

### Verified failure

```powershell
python tools/prepare_4k4d_prior_training_case.py `
  --scene-dir output\smoke_external_bundle_case\scene `
  --predictions-npz output\smoke_external_bundle_case\predictions.npz `
  --external-prior-bundle output\smoke_external_bundle_case\bundle_bad_order `
  --output-dir output\smoke_external_bundle_case\case_bad_order_verify `
  --overwrite
```

Observed result:

- hard failure with:

```text
ValueError: External prior camera ids do not match the scene manifest view order. expected=['00', '01'], got=['01', '00']
```

## Operator recommendation

For real data today, I would treat this path as:

- ready for controlled ingestion where you can curate the external payload format
- not yet ready for unattended ingestion of heterogeneous third-party SMPL-X bundles

Minimum safe operator settings:

- always pass `--strict` to `tools/import_external_smplx_params.py`
- always pass `--scene-dir` or `--scene-manifest` at import time
- always pass `--smplx-model-dir`
- always pass `--smplx-gender` explicitly
- prefer `--geometry-prior-source smplx_mesh`
- treat missing external cameras as a red flag unless you explicitly want fallback to dataset cameras

## 2026-04-26 bridge-to-inference smoke update

The repo-side real-data bridge has now been checked beyond bundle/case prep:

```powershell
python tools/build_scene_prior_from_external_bundle.py `
  --scene-dir output\smoke_external_bundle_case\scene `
  --external-prior-bundle output\smoke_external_bundle_case\bundle `
  --output-scene-dir output\smoke_external_bundle_case\scene_with_external_prior_bridge `
  --overwrite

modal run modal_4k4d_vggt_infer.py::run_scene_from_local `
  --local-scene-dir output\smoke_external_bundle_case\scene_with_external_prior_bridge `
  --remote-scene-subdir smoke_external_scene_bridge\scene_with_external_prior_bridge `
  --output-subdir evals/20260424_smoke_external_prior_scene_bridge_ckpt4 `
  --checkpoint-relpath vggt_4k4d_train/20260423_sparseproto_humancrop_pointnormal_r2_signfix/logs/ckpts/checkpoint_4.pt `
  --download-local-dir output\modal_results\20260424_smoke_external_prior_scene_bridge_ckpt4
```

Observed result:

- `tools/build_scene_prior_from_external_bundle.py` produced scene-level `prior_maps.npz`.
- Modal prior-enabled inference loaded the bridged scene successfully.
- `summary.json` reports `prior_tensor_shape = [2, 30, 518, 518]`.
- `summary.json` reports `prior_summary_tensor_shape = [2, 16, 27]`.
- Downloaded `predictions.npz` passed ZIP CRC validation.
- Output directory:
  `D:/vggt/vggt-main/output/modal_results/20260424_smoke_external_prior_scene_bridge_ckpt4`.

Truthful interpretation:

- The path `external bundle -> scene prior_maps.npz -> prior-enabled Modal inference -> local predictions.npz` is now end-to-end verified on the smoke fixture.
- This still does not mean the repo contains a direct image-to-SMPL-X regressor. The production path remains external estimator/fitter plus repo-side import, validation, prior generation, and inference.
