# 2026-04-22 Internal Negative-Result Supplement: Overfit Eval Pack

- Scope: internal archive/supplement built from existing outputs only; no new model runs or renders.
- Changed files: `tools/build_mentor_overfit_eval_pack.py`, `reports/20260422_mentor_overfit_eval_pack.md`
- Main pack directory: `reports/mentor_overfit_eval_pack_20260422`

## Truthful Mentor Read

- The single-case overfit checkpoints do not meet mentor-final quality.
- Both trained branches still visually collapse the face ROI toward planar or silhouette-like geometry.
- Confidence values rise sharply, but retained face ROI points and exported z-span do not improve.

## Generated Outputs

- `reports/mentor_overfit_eval_pack_20260422/README.md`
- `reports/mentor_overfit_eval_pack_20260422/mentor_overfit_eval_report.md`
- `reports/mentor_overfit_eval_pack_20260422/pack_manifest.json`
- `reports/mentor_overfit_eval_pack_20260422/assets/preprocess/compare_face_roi_variants.png`
- `reports/mentor_overfit_eval_pack_20260422/assets/crop_open3d/face_close.png`
- `reports/mentor_overfit_eval_pack_20260422/assets/crop_open3d/side.png`
- `reports/mentor_overfit_eval_pack_20260422/assets/softmatte_open3d/face_close.png`
- `reports/mentor_overfit_eval_pack_20260422/assets/softmatte_open3d/side.png`
