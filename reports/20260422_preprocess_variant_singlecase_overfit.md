# 2026-04-22 Preprocess-Variant Single-Case Overfit Notes

## Existing flow inspected

- Remote baseline run `vggt_4k4d_train/20260421_6view_singlecase_overfit_b40` is still present in Modal output volume.
- Remote `run_summary.json` confirms:
  - `config_name = 4k4d_prior_case_6view_focus_strongfusion`
  - `case_subdirs = ["training_cases/0012_11_frame0000_6views_sparseproto_smplxsurfacepose_v2"]`
  - `max_epochs = 1`
  - `limit_train_batches = 40`
  - `limit_val_batches = 5`
- Remote `b200` run kept the same single-case flow and config family, but changed `limit_train_batches = 200`.
- Remote `logs/log.txt` for `b40` shows the train and val datasets both loaded exactly one 6-view case and resumed from `facebook/VGGT-1B`.

## Added files

- `training/config/4k4d_prior_case_6view_singlecase_overfit_b40.yaml`
- `training/config/4k4d_prior_case_6view_singlecase_human_crop_overfit_b40.yaml`
- `training/config/4k4d_prior_case_6view_singlecase_human_crop_softmatte_overfit_b40.yaml`
- `scripts/run_modal_4k4d_preprocess_singlecase_overfit.ps1`

## Exact commands

Local training from repo root:

```powershell
python D:\vggt\vggt-main\training\launch.py --config 4k4d_prior_case_6view_singlecase_human_crop_overfit_b40
python D:\vggt\vggt-main\training\launch.py --config 4k4d_prior_case_6view_singlecase_human_crop_softmatte_overfit_b40
```

Cloud training with current Modal tooling:

```powershell
powershell -ExecutionPolicy Bypass -File D:\vggt\vggt-main\scripts\run_modal_4k4d_preprocess_singlecase_overfit.ps1 -Variant human_crop
powershell -ExecutionPolicy Bypass -File D:\vggt\vggt-main\scripts\run_modal_4k4d_preprocess_singlecase_overfit.ps1 -Variant human_crop_softmatte
```

Cloud train + eval on the matching preprocess scene:

```powershell
powershell -ExecutionPolicy Bypass -File D:\vggt\vggt-main\scripts\run_modal_4k4d_preprocess_singlecase_overfit.ps1 -Variant human_crop -RunEval
powershell -ExecutionPolicy Bypass -File D:\vggt\vggt-main\scripts\run_modal_4k4d_preprocess_singlecase_overfit.ps1 -Variant human_crop_softmatte -RunEval
```

The helper keeps the inspected `b40` shape explicit:

- `max_epochs = 1`
- `limit_train_batches = 40`
- `limit_val_batches = 5`
- `max_img_per_gpu = 6`
- `img_nums = [6, 6]`

## Validation

Executed locally:

- Hydra compose check for all three new configs.
- Dry-run of `run_modal_4k4d_preprocess_singlecase_overfit.ps1` for:
  - `human_crop`
  - `human_crop_softmatte`

Validation intent:

- configs resolve to the expected single-case roots
- Modal helper emits the expected train command
- optional eval path points at the matching preprocess scene and derived checkpoint relpath
