# Real-Data SMPL-X Driver

This document answers the mentor question: when real captures do not ship with
SMPL-X labels, where does the pose come from?

The truthful answer is: pose comes from an external SMPL-X regressor/fitter, or
from an estimator command launched through this repository. The repository does
not bundle large third-party regressor weights, but it now contains the code
path that runs an estimator command, imports the result, validates it against a
scene, and turns it into the same prior bundle used by VGGT inference/training.

## Supported Modes

### 1. Import an external regressor output

Use this when another tool has already written SMPL-X/camera files:

```powershell
python tools/run_realdata_smplx_driver.py `
  --mode external-regressor-json-npz `
  --input-path <external_output_dir> `
  --scene-dir <scene_dir> `
  --output-dir <bundle_dir> `
  --strict `
  --overwrite
```

Expected outputs are:

- `normalized_smplx_params.npz`
- `normalized_camera_params.npz`
- `external_prior_bundle_manifest.json`

### 2. Import an external fitting result

Use this when a fitter writes one combined JSON/NPZ result:

```powershell
python tools/run_realdata_smplx_driver.py `
  --mode fitting-result `
  --input-path <fitting_result_json_or_dir> `
  --scene-dir <scene_dir> `
  --output-dir <bundle_dir> `
  --strict `
  --overwrite
```

The driver searches for nested `smplx` / `smplx_params` and `camera` /
`camera_params` payloads, then normalizes them into the canonical bundle.

### 3. Run an estimator command, then import the result

Use this when the estimator/fitter is installed elsewhere but should be launched
from the repo workflow:

```powershell
python tools/run_realdata_smplx_driver.py `
  --mode estimator-command `
  --input-path <estimator_output_dir> `
  --scene-dir <scene_dir> `
  --output-dir <bundle_dir> `
  --estimator-command "python <estimator_entry.py> --scene {scene_dir} --out {estimator_output_dir}" `
  --estimator-output-dir <estimator_output_dir> `
  --strict `
  --overwrite
```

Available command placeholders:

- `{scene_dir}`
- `{scene_manifest}`
- `{input_path}`
- `{estimator_output_dir}`

The command log is written to `<bundle_dir>/estimator_command_summary.json` and
is also referenced in `external_prior_bundle_manifest.json`.

For smoke testing an already-generated estimator output without running the
estimator again:

```powershell
python tools/run_realdata_smplx_driver.py `
  --mode estimator-command `
  --input-path output\smoke_external_bundle_case\bundle `
  --scene-dir output\smoke_external_bundle_case\scene `
  --output-dir output\smoke_external_bundle_case\bundle_via_estimator_command_skip `
  --skip-estimator-run `
  --strict `
  --overwrite
```

## Bridge into Prior-Enabled Scene Inference

After a bundle is produced, convert it into scene-level `prior_maps.npz`:

```powershell
python tools/build_scene_prior_from_external_bundle.py `
  --scene-dir <scene_dir> `
  --external-prior-bundle <bundle_dir> `
  --output-scene-dir <prior_enabled_scene_dir> `
  --overwrite
```

Then run prior-enabled VGGT inference:

```powershell
modal run modal_4k4d_vggt_infer.py::run_scene_from_local `
  --local-scene-dir <prior_enabled_scene_dir> `
  --remote-scene-subdir <remote_scene_subdir> `
  --output-subdir <remote_output_subdir> `
  --checkpoint-relpath <checkpoint_relpath> `
  --download-local-dir <local_output_dir>
```

## Current Smoke Status

The smoke case `output\smoke_external_bundle_case` has been verified end to end:

- `build_scene_prior_from_external_bundle.py` writes `prior_maps.npz`.
- Modal prior-enabled inference reads `prior_tensor_shape=[2, 30, 518, 518]`.
- Modal prior-enabled inference also reads
  `prior_summary_tensor_shape=[2, 16, 27]` and produces normal/depth/point
  predictions, confirming the scene can enter the prior-enabled VGGT path.
- The downloaded output is in
  `output\modal_results\20260424_smoke_external_prior_scene_bridge_ckpt4`.
- The smoke summary is
  `output\modal_results\20260424_smoke_external_prior_scene_bridge_ckpt4\summary.json`.

## Important Limit

This is a repo-side estimator launch/import/bridge path. It is not a bundled
SMPL-X regressor model and does not claim that the repository trains or ships a
new SMPL-X estimator. A production real-data run still needs a chosen external
estimator/fitter and its weights installed in the execution environment.
