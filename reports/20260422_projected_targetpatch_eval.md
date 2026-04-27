# 2026-04-22 Projected TargetPatch Eval

## Bottom line

The stronger `projected_token_sample` integration path is still **not enough** to lift the final 6-view face ROI above the existing smoke baseline.

This follow-up is materially stronger than the earlier dense-only stitch-back test because it now patches:

- `inputs.prior_maps` dense normal channels
- `targets.prior_normals`
- `prior_summary_tokens` normal channels at token locations projected into the ROI

Even with that stronger integration:

- the `teacher` upper-bound run regresses `face ROI`
- the deployable `refined` run is better than `teacher`, but still below the smoke baseline

## What changed in this round

Updated tools:

- [patch_training_case_with_refined_normals.py](D:/vggt/vggt-main/tools/patch_training_case_with_refined_normals.py)
- [patch_scene_prior_with_refined_normals.py](D:/vggt/vggt-main/tools/patch_scene_prior_with_refined_normals.py)

New integration mode:

- `summary_update=projected_token_sample`

This mode projects each summary token's `smplx_summary_posed_cam_{x,y,z}` back into the image using aligned intrinsics, then samples the ROI refined normal at that token's projected pixel and writes it back into:

- `smplx_summary_cam_nx`
- `smplx_summary_cam_ny`
- `smplx_summary_cam_nz`

On the 6-view case, this updated `2` to `8` summary tokens per view instead of only applying a view-global mean delta.

## Patched assets

Training cases:

- refined:
  - [0012_11_frame0000_6views_sparseproto_smplxsurfacepose_v2_headrefineproj_targetpatch](D:/vggt/vggt-main/output/training_cases/0012_11_frame0000_6views_sparseproto_smplxsurfacepose_v2_headrefineproj_targetpatch)
- teacher:
  - [0012_11_frame0000_6views_sparseproto_smplxsurfacepose_v2_headteacherproj_targetpatch](D:/vggt/vggt-main/output/training_cases/0012_11_frame0000_6views_sparseproto_smplxsurfacepose_v2_headteacherproj_targetpatch)

Patched scenes for inference:

- refined:
  - [0012_11_frame0000_6views_sparseproto_headrefineproj](D:/vggt/vggt-main/output/detail_normal_refiner_20260422/0012_11_frame0000_6views_sparseproto_headrefineproj)
- teacher:
  - [0012_11_frame0000_6views_sparseproto_headteacherproj](D:/vggt/vggt-main/output/detail_normal_refiner_20260422/0012_11_frame0000_6views_sparseproto_headteacherproj)

Remote training checkpoints:

- refined:
  - `20260422_6view_headrefineproj_targetpatch_r1/inference_model.pt`
- teacher:
  - `20260422_6view_headteacherproj_targetpatch_r1/inference_model.pt`

Remote inference outputs:

- refined:
  - `vggt_4k4d_infer/20260422_6views_headrefineproj_targetpatch_infer_r1`
- teacher:
  - `vggt_4k4d_infer/20260422_6views_headteacherproj_targetpatch_infer_r1`

## ROI summary

Reference points:

- smoke baseline:
  - full `40,878`
  - head `8,993`
  - face `4,018`
- earlier refined targetpatch without projected token sampling:
  - full `40,885`
  - head `8,995`
  - face `3,637`

Current projected targetpatch results:

| Variant | Full-body points | Head ROI points | Face ROI points |
| --- | ---: | ---: | ---: |
| smoke baseline | `40,878` | `8,993` | `4,018` |
| teacher projected targetpatch | `40,883` | `8,995` | `3,546` |
| refined projected targetpatch | `40,892` | `8,997` | `3,852` |

## Interpretation

What this proves:

- the stronger integration path is technically closed
- `prior_maps`, `prior_normals`, and projected summary-token normal channels can all be patched consistently
- 6-view fine-tuning from the resumed strongfusion checkpoint is stable on this patched case

What it still does **not** prove:

- that stronger normal integration improves the final sparse-view face geometry

The most important read is:

- `teacher projected targetpatch` is already below the smoke baseline, so the current integration path does not show a useful upper bound
- `refined projected targetpatch` improves over the earlier `3,637` face-ROI result, but still does not recover to the `4,018` smoke baseline

So the current truthful conclusion is:

> even after upgrading from dense-only stitch-back to projected summary-token targetpatch, this detail-normal integration path remains a negative result for final 6-view face ROI quality.

## Side note on the multi-case family attempt

A larger `6/8/12/20` patched-family strongfusion resume was launched first, but the initial `A100-40GB` run with `max_img_per_gpu=13` hit CUDA OOM before the first epoch completed.

That means the multi-case projected-targetpatch family path is not ruled out forever, but it was not validated in this round. The only completed result here is the 6-view single-case projected-targetpatch eval above.
