# 2026-04-22 HumanCrop Mainline Bootstrap

## Bottom line

The new `human_crop` mainline is now materially more complete than before:

- crop scene variants now exist for `6 / 8 / 12 / 20 / 60`
- resumed-strongfusion crop inference outputs now exist for `6 / 8 / 12 / 20`
- crop training cases now exist for `6 / 8 / 12 / 20`
- dedicated crop-mainline configs and a Modal helper script now exist

This is still **not** the mentor-final endpoint yet. What is now closed is the
`human_crop as default sparse-view base` engineering path. The remaining gap is
still final face/head quality, not whether the crop branch can be generated,
trained, or evaluated.

## New crop-trained results on April 22

Two new cloud runs have now completed on top of the crop mainline:

- `focus6` crop-only training:
  - remote checkpoint: `vggt_4k4d_train/20260422_6view_focus_humancrop_resume_r1/inference_model.pt`
  - local eval: `output/modal_results/20260422_6views_humancrop_eval_from_humancrop_resume_r1`
- `family` crop training (`6 / 8 / 12 / 20`, conservative sparseproto mix):
  - remote checkpoint: `vggt_4k4d_train/20260422_sparseproto_humancrop_resume_r1/inference_model.pt`
  - local eval: `output/modal_results/20260422_sparseproto_humancrop_resume_r1_eval6`
  - remote ROI summaries:
    - `vggt_4k4d_infer/20260422_sparseproto_humancrop_resume_r1_eval6`
    - `vggt_4k4d_infer/20260422_sparseproto_humancrop_resume_r1_eval8`
    - `vggt_4k4d_infer/20260422_sparseproto_humancrop_resume_r1_eval12`
    - `vggt_4k4d_infer/20260422_sparseproto_humancrop_resume_r1_eval20`

Both training runs completed on cloud `A100 80GB`. The current inference path
still defaults to `A100 40GB`, which is fine for evaluation because these crop
scenes fit comfortably at inference time.

## What was generated

### Crop scene variants

- `output/4k4d_preprocessed_scene_variants/0012_11_frame0000_6views_sparseproto_human_crop`
- `output/4k4d_preprocessed_scene_variants/0012_11_frame0000_8views_sparseproto_human_crop`
- `output/4k4d_preprocessed_scene_variants/0012_11_frame0000_12views_sparseproto_human_crop`
- `output/4k4d_preprocessed_scene_variants/0012_11_frame0000_20views_sparseproto_human_crop`
- `output/4k4d_preprocessed_scene_variants/0012_11_frame0000_60views_human_crop`

### Resume-strongfusion crop inference outputs

- `output/modal_results/20260422_6views_humancrop_from_resume_strongfusion_r1`
- `output/modal_results/20260422_8views_humancrop_from_resume_strongfusion_r1`
- `output/modal_results/20260422_12views_humancrop_from_resume_strongfusion_r1`
- `output/modal_results/20260422_20views_humancrop_from_resume_strongfusion_r1`

Note:

- the direct Modal download path corrupted the local `6v` and `12v` `predictions.npz`
- these two runs were repaired with `modal_4k4d_vggt_infer.py::download_prediction_chunks_rpc`
- all four local `predictions.npz` files are now readable and valid

### Crop training cases

- `output/training_cases/0012_11_frame0000_6views_sparseproto_humancrop_resume_r1`
- `output/training_cases/0012_11_frame0000_8views_sparseproto_humancrop_resume_r1`
- `output/training_cases/0012_11_frame0000_12views_sparseproto_humancrop_resume_r1`
- `output/training_cases/0012_11_frame0000_20views_sparseproto_humancrop_resume_r1`

### New configs / launcher

- `training/config/4k4d_prior_case_sparseproto_humancrop_resume_r1.yaml`
- `training/config/4k4d_prior_case_6view_focus_humancrop_resume_r1.yaml`
- `scripts/run_modal_4k4d_humancrop_sparseproto_strongfusion.ps1`

## ROI summary from the resumed strongfusion checkpoint

ROI counts below come from `modal_4k4d_vggt_infer.py::summarize_prediction_roi`
with `conf_percentile = 40.0`.

| Variant | Full-body points | Head ROI points | Face ROI points |
| --- | ---: | ---: | ---: |
| `6v full` | `40,880` | `8,994` | `3,673` |
| `6v human_crop` | `111,082` | `24,438` | `9,634` |
| `8v human_crop` | `131,723` | `28,979` | `11,144` |
| `12v human_crop` | `210,913` | `46,401` | `17,395` |
| `20v human_crop` | `331,924` | `73,024` | `26,425` |

### New crop-trained ROI summary

| Variant | Full-body points | Head ROI points | Face ROI points |
| --- | ---: | ---: | ---: |
| `6v human_crop baseline` | `111,082` | `24,438` | `9,634` |
| `6v human_crop focus6 train` | `111,078` | `24,437` | `11,774` |
| `6v human_crop family train` | `111,078` | `24,437` | `11,913` |
| `8v human_crop family train` | `131,704` | `28,975` | `13,859` |
| `12v human_crop family train` | `210,812` | `46,379` | `21,620` |
| `20v human_crop family train` | `331,823` | `73,001` | `30,749` |

## Immediate reading

For the current resumed-strongfusion checkpoint:

- switching from `6v full` to `6v human_crop` lifts
  - full retained points from `40,880 -> 111,082`
  - head ROI from `8,994 -> 24,438`
  - face ROI from `3,673 -> 9,634`
- that means the crop base is already giving a large occupancy win before any new crop-specific training
- the new bottleneck is no longer "do we have a crop mainline?" but "does crop-trained geometry become visibly sharper at face/head ROI?"

For the new crop-trained checkpoints:

- `focus6` pushes `6v face ROI` from `9,634 -> 11,774`
- `family` pushes `6v face ROI` slightly further to `11,913`
- compared to the original `6v full` baseline, the current best `6v face ROI`
  count is now `3.24x` higher (`3,673 -> 11,913`)
- `family` also improves the higher-view crop ladder over the resumed baseline:
  - `8v face ROI`: `11,144 -> 13,859`
  - `12v face ROI`: `17,395 -> 21,620`
  - `20v face ROI`: `26,425 -> 30,749`

This is meaningful forward motion. It is still not enough, by itself, to claim
mentor-final sparse-view face quality. We now have a stronger crop mainline,
but the visual sharpness bar still needs direct ROI figure review and likely a
follow-up detail branch or stronger image-aligned refinement.

## What is still not done

- no mentor-final claim should be made from the ROI counts alone
- `projected targetpatch` remains a negative-result branch
- `single-case preprocess overfit` remains a negative-result branch

## Recommended next action

The truthful current project line is now:

> `human_crop` has been promoted from a 6-view ablation into a real sparse-view mainline base, and crop-trained checkpoints now improve face ROI over the resumed baseline; however, the mentor-final quality gate still depends on whether those gains translate into clearly better head/face geometry in the final visual evidence.
