# 2026-04-22 Mentor Benchmark Pack

- Scope: consolidated from existing outputs only; no new model runs or renders.
- Changed files: `tools/build_mentor_benchmark_pack.py`, `reports/20260422_mentor_benchmark_pack.md`
- Main pack directory: `output/mentor_benchmark_pack_20260422`

## Generated Outputs

- `output/mentor_benchmark_pack_20260422/README.md`
- `output/mentor_benchmark_pack_20260422/mentor_benchmark_report.md`
- `output/mentor_benchmark_pack_20260422/pack_manifest.json`
- `output/mentor_benchmark_pack_20260422/assets/preprocess/compare_head_roi_variants.png`
- `output/mentor_benchmark_pack_20260422/assets/preprocess/compare_face_roi_variants.png`
- `output/mentor_benchmark_pack_20260422/assets/preprocess/compare_fullbody_faceclose_variants.png`
- `output/mentor_benchmark_pack_20260422/assets/open3d/current_sparseproto_head_close.png`

## Notes

- The pack ties together the preprocess-variant benchmark table, ROI Open3D evidence, and detail-normal status in one report.
- The report keeps the current truthful read: `human_crop` is the stable default, `human_crop_softmatte` is the densest ROI branch, and ROI detail refinement is promising but not yet mentor-final.