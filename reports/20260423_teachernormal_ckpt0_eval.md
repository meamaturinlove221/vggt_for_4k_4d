# 2026-04-23 Teachernormal Family Checkpoint-0 Eval

## Status

This is an intermediate result, not a mentor-final claim.

- `family teachernormal` is still running:
  - remote train run: `vggt_4k4d_train/20260423_sparseproto_humancrop_teachernormal_r1`
- `focus6 teachernormal from checkpoint_0` is also running:
  - remote train run: `vggt_4k4d_train/20260423_6view_focus_humancrop_teachernormal_from_family_ckpt0_r1`

## What Was Evaluated

- checkpoint source:
  - `vggt_4k4d_train/20260423_sparseproto_humancrop_teachernormal_r1/logs/ckpts/checkpoint_0.pt`
- eval scene:
  - `scenes/0012_11_frame0000_6views_sparseproto_human_crop`
- remote eval output:
  - `vggt_4k4d_infer/20260423_sparseproto_humancrop_teachernormal_r1_ckpt0_eval6`
- local eval output:
  - [20260423_sparseproto_humancrop_teachernormal_r1_ckpt0_eval6](D:/vggt/vggt-main/output/modal_results/20260423_sparseproto_humancrop_teachernormal_r1_ckpt0_eval6)

## ROI Metrics

Current best pre-teachernormal reference on the same `6-view human_crop` scene:

| Variant | Full-body points | Head ROI points | Face ROI points |
| --- | ---: | ---: | ---: |
| `crop family train` | `111,078` | `24,437` | `11,913` |

Checkpoint-0 teachernormal result:

| Variant | Full-body points | Head ROI points | Face ROI points |
| --- | ---: | ---: | ---: |
| `teachernormal family checkpoint_0` | `111,078` | `24,437` | `12,017` |

Delta vs `crop family train`:

- full-body ROI: `0`
- head ROI: `0`
- face ROI: `+104`

This is the first teachernormal result that exceeds the current best known `6-view human_crop` face ROI count.

## Family Checkpoint-2 Follow-Up

Second intermediate family checkpoint:

- checkpoint source:
  - `vggt_4k4d_train/20260423_sparseproto_humancrop_teachernormal_r1/logs/ckpts/checkpoint_2.pt`
- eval output:
  - [20260423_sparseproto_humancrop_teachernormal_r1_ckpt2_eval6](D:/vggt/vggt-main/output/modal_results/20260423_sparseproto_humancrop_teachernormal_r1_ckpt2_eval6)

Checkpoint-2 ROI metrics:

| Variant | Full-body points | Head ROI points | Face ROI points |
| --- | ---: | ---: | ---: |
| `teachernormal family checkpoint_2` | `111,078` | `24,437` | `12,030` |

Delta vs `crop family train`:

- full-body ROI: `0`
- head ROI: `0`
- face ROI: `+117`

Delta vs `teachernormal family checkpoint_0`:

- full-body ROI: `0`
- head ROI: `0`
- face ROI: `+13`

## Visualization Outputs

Per-ROI Open3D renders:

- [full](D:/vggt/vggt-main/output/comparisons/20260423_humancrop_teachernormal_ckpt0_compare/teachernormal_family_ckpt0/full)
- [head](D:/vggt/vggt-main/output/comparisons/20260423_humancrop_teachernormal_ckpt0_compare/teachernormal_family_ckpt0/head)
- [face](D:/vggt/vggt-main/output/comparisons/20260423_humancrop_teachernormal_ckpt0_compare/teachernormal_family_ckpt0/face)

4-way comparison sheets:

- [compare_full_iso_4way.png](D:/vggt/vggt-main/output/comparisons/20260423_humancrop_teachernormal_ckpt0_compare/compare_full_iso_4way.png)
- [compare_head_close_4way.png](D:/vggt/vggt-main/output/comparisons/20260423_humancrop_teachernormal_ckpt0_compare/compare_head_close_4way.png)
- [compare_face_close_4way.png](D:/vggt/vggt-main/output/comparisons/20260423_humancrop_teachernormal_ckpt0_compare/compare_face_close_4way.png)

Checkpoint-2 visualization outputs:

- [full](D:/vggt/vggt-main/output/comparisons/20260423_humancrop_teachernormal_ckpt2_compare/teachernormal_family_ckpt2/full)
- [head](D:/vggt/vggt-main/output/comparisons/20260423_humancrop_teachernormal_ckpt2_compare/teachernormal_family_ckpt2/head)
- [face](D:/vggt/vggt-main/output/comparisons/20260423_humancrop_teachernormal_ckpt2_compare/teachernormal_family_ckpt2/face)
- [compare_full_iso_4way.png](D:/vggt/vggt-main/output/comparisons/20260423_humancrop_teachernormal_ckpt2_compare/compare_full_iso_4way.png)
- [compare_head_close_4way.png](D:/vggt/vggt-main/output/comparisons/20260423_humancrop_teachernormal_ckpt2_compare/compare_head_close_4way.png)
- [compare_face_close_4way.png](D:/vggt/vggt-main/output/comparisons/20260423_humancrop_teachernormal_ckpt2_compare/compare_face_close_4way.png)

## Current Read

- The `teachernormal` direction is now showing a positive signal on the actual `6-view human_crop` metric panel.
- The gain is still small, so this is not yet enough to declare the mentor bar closed.
- `family checkpoint_2` continues the same trend and is slightly better than `checkpoint_0`.
- The next important question is whether either:
  - the full `family teachernormal` run, or
  - the `focus6 teachernormal from checkpoint_0` run
  can widen the face/head advantage beyond this `+104` face ROI gain and show clearer qualitative improvement in the ROI close-ups.
