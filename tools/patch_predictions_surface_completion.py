#!/usr/bin/env python
"""Patch VGGT predictions with a small image-aligned ROI surface completion.

This is a diagnostic post-process, not an end-to-end training result.  It fills
an ROI depth surface from high-confidence VGGT points in the same view, then
reconstructs camera-ray-aligned world points for low-confidence or all ROI
pixels.  The script writes a patched predictions.npz plus an explicit summary
so later reports cannot confuse the result with raw model output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.render_open3d_pointcloud import load_2d_roi_mask_stack  # noqa: E402
from vggt.utils.normal_refiner import normal_to_rgb, point_map_to_normal_numpy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-npz", required=True)
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--roi", choices=("head", "face"), default="face")
    parser.add_argument("--anchor-conf-percentile", type=float, default=55.0)
    parser.add_argument("--patch-mode", choices=("low_conf", "all_roi"), default="low_conf")
    parser.add_argument("--confidence-mode", choices=("keep", "floor_anchor"), default="keep")
    parser.add_argument("--completion-source", choices=("vggt_inpaint", "smpl_prior"), default="vggt_inpaint")
    parser.add_argument("--prior-maps-npz", help="Scene prior_maps.npz. Defaults to <scene-dir>/prior_maps.npz for smpl_prior.")
    parser.add_argument("--align-smpl-depth", choices=("none", "affine"), default="affine")
    parser.add_argument("--low-conf-margin", type=float, default=0.0)
    parser.add_argument("--inpaint-radius", type=float, default=5.0)
    parser.add_argument("--smooth-sigma", type=float, default=1.2)
    parser.add_argument("--blend-alpha", type=float, default=1.0)
    parser.add_argument("--max-depth-delta", type=float, default=0.06)
    parser.add_argument("--write-previews", action="store_true")
    return parser.parse_args()


def _world_to_camera(points_world: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    rotation = np.asarray(extrinsic[:, :3], dtype=np.float32)
    translation = np.asarray(extrinsic[:, 3], dtype=np.float32)
    return points_world @ rotation.T + translation[None, None, :]


def _camera_to_world(points_cam: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    rotation = np.asarray(extrinsic[:, :3], dtype=np.float32)
    translation = np.asarray(extrinsic[:, 3], dtype=np.float32)
    return (points_cam - translation[None, None, :]) @ rotation


def _ray_points_from_depth(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    z = depth.astype(np.float32)
    x = (xx - cx) * z / max(fx, 1e-6)
    y = (yy - cy) * z / max(fy, 1e-6)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def _inpaint_depth(depth: np.ndarray, known_mask: np.ndarray, roi_mask: np.ndarray, radius: float, sigma: float) -> np.ndarray:
    finite = np.isfinite(depth) & (depth > 0)
    known = known_mask & finite
    if int(known.sum()) < 16:
        raise RuntimeError("Not enough high-confidence anchor pixels to inpaint ROI depth.")

    work = np.asarray(depth, dtype=np.float32).copy()
    fill_value = float(np.median(work[known]))
    work[~finite] = fill_value
    missing = (roi_mask & ~known).astype(np.uint8) * 255
    inpainted = cv2.inpaint(work, missing, float(radius), cv2.INPAINT_TELEA).astype(np.float32)
    if sigma > 0:
        smooth = cv2.GaussianBlur(inpainted, ksize=(0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
        inpainted = np.where(roi_mask, smooth, inpainted).astype(np.float32)
    return inpainted


def _normal_preview(cam_points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    normal, valid = point_map_to_normal_numpy(cam_points.astype(np.float32), mask.astype(bool))
    return normal_to_rgb(normal, valid)


def _load_smpl_prior(prior_maps_npz: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(prior_maps_npz, allow_pickle=False) as payload:
        prior_maps = np.asarray(payload["prior_maps"], dtype=np.float32)
        channels = [str(item) for item in payload["prior_channels"]]
    lookup = {name: idx for idx, name in enumerate(channels)}
    required = [
        "smplx_posed_cam_x",
        "smplx_posed_cam_y",
        "smplx_posed_cam_z",
        "smplx_cam_nx",
        "smplx_cam_ny",
        "smplx_cam_nz",
        "smplx_visible_mask",
    ]
    missing = [name for name in required if name not in lookup]
    if missing:
        raise KeyError(f"prior maps missing required SMPL channels: {missing}")
    posed = np.stack(
        [
            prior_maps[:, lookup["smplx_posed_cam_x"]],
            prior_maps[:, lookup["smplx_posed_cam_y"]],
            prior_maps[:, lookup["smplx_posed_cam_z"]],
        ],
        axis=-1,
    ).astype(np.float32)
    normals = np.stack(
        [
            prior_maps[:, lookup["smplx_cam_nx"]],
            prior_maps[:, lookup["smplx_cam_ny"]],
            prior_maps[:, lookup["smplx_cam_nz"]],
        ],
        axis=-1,
    ).astype(np.float32)
    visible = prior_maps[:, lookup["smplx_visible_mask"]] > 0.5
    return posed, normals, visible


def _fit_affine(source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    valid = mask & np.isfinite(source) & np.isfinite(target)
    if int(valid.sum()) < 16:
        return 1.0, 0.0
    x = source[valid].astype(np.float64)
    y = target[valid].astype(np.float64)
    a = np.vstack([x, np.ones_like(x)]).T
    scale, bias = np.linalg.lstsq(a, y, rcond=None)[0]
    return float(scale), float(bias)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"
    if args.write_previews:
        preview_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = Path(args.predictions_npz).expanduser().resolve()
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    with np.load(predictions_path, allow_pickle=False) as payload:
        predictions = {key: payload[key] for key in payload.files}

    world_points = np.asarray(predictions["world_points"], dtype=np.float32).copy()
    world_conf = np.asarray(predictions.get("world_points_conf", predictions["depth_conf"]), dtype=np.float32).copy()
    depth = np.asarray(predictions["depth"], dtype=np.float32).copy()
    normal = np.asarray(predictions.get("normal"), dtype=np.float32).copy() if "normal" in predictions else None
    normal_conf = np.asarray(predictions.get("normal_conf"), dtype=np.float32).copy() if "normal_conf" in predictions else None
    extrinsic = np.asarray(predictions["extrinsic"], dtype=np.float32)
    intrinsic = np.asarray(predictions["intrinsic"], dtype=np.float32)

    target_size = int(world_points.shape[1])
    roi_masks = load_2d_roi_mask_stack(scene_dir / "masks", target_size=target_size, roi=str(args.roi)).astype(bool)
    smpl_cam = smpl_normals = smpl_visible = None
    if args.completion_source == "smpl_prior":
        prior_path = Path(args.prior_maps_npz).expanduser().resolve() if args.prior_maps_npz else scene_dir / "prior_maps.npz"
        smpl_cam, smpl_normals, smpl_visible = _load_smpl_prior(prior_path)
        if smpl_cam.shape[:3] != world_points.shape[:3]:
            raise ValueError(f"SMPL prior shape {smpl_cam.shape[:3]} does not match predictions {world_points.shape[:3]}")
    records: list[dict[str, float | int | str]] = []

    for view_idx in range(world_points.shape[0]):
        roi = roi_masks[view_idx]
        if not roi.any():
            records.append({"view_index": view_idx, "roi_pixels": 0, "patched_pixels": 0, "status": "empty_roi"})
            continue

        cam_points = _world_to_camera(world_points[view_idx], extrinsic[view_idx])
        z = cam_points[..., 2].astype(np.float32)
        conf = world_conf[view_idx]
        finite = roi & np.isfinite(z) & (z > 0) & np.isfinite(conf) & (conf > 0)
        if not finite.any():
            records.append({"view_index": view_idx, "roi_pixels": int(roi.sum()), "patched_pixels": 0, "status": "empty_finite"})
            continue

        anchor_threshold = float(np.percentile(conf[finite], float(args.anchor_conf_percentile)))
        anchor = finite & (conf >= anchor_threshold)
        if int(anchor.sum()) < 16:
            records.append(
                {
                    "view_index": view_idx,
                    "roi_pixels": int(roi.sum()),
                    "patched_pixels": 0,
                    "anchor_pixels": int(anchor.sum()),
                    "anchor_conf_threshold": anchor_threshold,
                    "status": "too_few_anchor_pixels",
                }
            )
            continue

        low_conf_threshold = anchor_threshold + float(args.low_conf_margin)
        if args.patch_mode == "all_roi":
            patch_mask = roi.copy()
        else:
            patch_mask = roi & (~finite | (conf < low_conf_threshold))

        if args.completion_source == "smpl_prior":
            assert smpl_cam is not None and smpl_normals is not None and smpl_visible is not None
            source_cam = smpl_cam[view_idx]
            source_valid = smpl_visible[view_idx] & np.isfinite(source_cam).all(axis=-1) & (source_cam[..., 2] > 0)
            patch_mask = patch_mask & source_valid
            if not patch_mask.any():
                records.append(
                    {
                        "view_index": view_idx,
                        "roi_pixels": int(roi.sum()),
                        "patched_pixels": 0,
                        "anchor_pixels": int(anchor.sum()),
                        "anchor_conf_threshold": anchor_threshold,
                        "status": "empty_smpl_prior_patch",
                    }
                )
                continue
            source_z = source_cam[..., 2].astype(np.float32)
            scale, bias = _fit_affine(source_z, z, anchor & source_valid) if args.align_smpl_depth == "affine" else (1.0, 0.0)
            completed_z = (scale * source_z + bias).astype(np.float32)
            patched_cam = _ray_points_from_depth(completed_z, intrinsic[view_idx])
            depth_delta = completed_z - z
            patch_normals = smpl_normals[view_idx].astype(np.float32)
        else:
            completed_z = _inpaint_depth(
                z,
                known_mask=anchor,
                roi_mask=roi,
                radius=float(args.inpaint_radius),
                sigma=float(args.smooth_sigma),
            )
            delta = np.clip(completed_z - z, -float(args.max_depth_delta), float(args.max_depth_delta))
            completed_z = z + delta
            if float(args.blend_alpha) < 1.0:
                completed_z = (1.0 - float(args.blend_alpha)) * z + float(args.blend_alpha) * completed_z
            patch_mask = patch_mask & np.isfinite(completed_z) & (completed_z > 0)
            patched_cam = _ray_points_from_depth(completed_z, intrinsic[view_idx])
            depth_delta = completed_z - z
            patch_normals, valid_patch_normals = point_map_to_normal_numpy(patched_cam.astype(np.float32), roi)
            patch_normals[~valid_patch_normals] = 0.0

        patched_world = _camera_to_world(patched_cam, extrinsic[view_idx])
        world_points[view_idx][patch_mask] = patched_world[patch_mask].astype(np.float32)
        depth[view_idx, ..., 0][patch_mask] = completed_z[patch_mask].astype(np.float32)

        if args.confidence_mode == "floor_anchor" and patch_mask.any():
            world_conf[view_idx][patch_mask] = np.maximum(world_conf[view_idx][patch_mask], anchor_threshold).astype(np.float32)
            if "depth_conf" in predictions:
                predictions["depth_conf"] = np.asarray(predictions["depth_conf"], dtype=np.float32)
                predictions["depth_conf"][view_idx][patch_mask] = np.maximum(
                    predictions["depth_conf"][view_idx][patch_mask], anchor_threshold
                ).astype(np.float32)

        if normal is not None:
            valid_normal = patch_mask & np.isfinite(patch_normals).all(axis=-1) & (np.linalg.norm(patch_normals, axis=-1) > 0.1)
            normal[view_idx][valid_normal] = patch_normals[valid_normal].astype(np.float32)
            if normal_conf is not None and args.confidence_mode == "floor_anchor":
                normal_conf[view_idx][valid_normal] = np.maximum(normal_conf[view_idx][valid_normal], 0.25)

        record = {
            "view_index": int(view_idx),
            "roi": str(args.roi),
            "roi_pixels": int(roi.sum()),
            "finite_pixels": int(finite.sum()),
            "anchor_pixels": int(anchor.sum()),
            "patched_pixels": int(patch_mask.sum()),
            "anchor_conf_threshold": anchor_threshold,
            "patch_mode": str(args.patch_mode),
            "confidence_mode": str(args.confidence_mode),
            "completion_source": str(args.completion_source),
            "smpl_depth_align_scale": float(scale) if args.completion_source == "smpl_prior" else 1.0,
            "smpl_depth_align_bias": float(bias) if args.completion_source == "smpl_prior" else 0.0,
            "max_abs_depth_delta": float(np.max(np.abs(depth_delta[patch_mask]))) if patch_mask.any() else 0.0,
            "median_abs_depth_delta": float(np.median(np.abs(depth_delta[patch_mask]))) if patch_mask.any() else 0.0,
            "status": "ok",
        }
        records.append(record)

        if args.write_previews:
            z_vis = completed_z.copy()
            values = z_vis[roi & np.isfinite(z_vis)]
            lo, hi = np.percentile(values, [2, 98]) if values.size else (0.0, 1.0)
            z_rgb = np.clip((z_vis - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
            patch_rgb = np.zeros((*roi.shape, 3), dtype=np.uint8)
            patch_rgb[..., 0] = (patch_mask.astype(np.uint8) * 255)
            patch_rgb[..., 1] = (anchor.astype(np.uint8) * 255)
            Image.fromarray((z_rgb * 255.0).astype(np.uint8)).save(preview_dir / f"{view_idx:02d}_{args.roi}_completed_depth.png")
            Image.fromarray(patch_rgb).save(preview_dir / f"{view_idx:02d}_{args.roi}_patch_mask_red_anchor_green.png")
            Image.fromarray(_normal_preview(patched_cam, roi)).save(preview_dir / f"{view_idx:02d}_{args.roi}_completed_normal.png")

    predictions["world_points"] = world_points.astype(np.float32)
    predictions["world_points_conf"] = world_conf.astype(np.float32)
    predictions["depth"] = depth.astype(np.float32)
    if normal is not None:
        predictions["normal"] = normal.astype(np.float32)
    if normal_conf is not None:
        predictions["normal_conf"] = normal_conf.astype(np.float32)

    out_npz = output_dir / "predictions.npz"
    np.savez_compressed(out_npz, **predictions)
    summary = {
        "message": "surface completion predictions patch written",
        "predictions_npz": str(predictions_path),
        "scene_dir": str(scene_dir),
        "output_predictions_npz": str(out_npz),
        "roi": str(args.roi),
        "anchor_conf_percentile": float(args.anchor_conf_percentile),
        "patch_mode": str(args.patch_mode),
        "confidence_mode": str(args.confidence_mode),
        "completion_source": str(args.completion_source),
        "inpaint_radius": float(args.inpaint_radius),
        "smooth_sigma": float(args.smooth_sigma),
        "blend_alpha": float(args.blend_alpha),
        "max_depth_delta": float(args.max_depth_delta),
        "records": records,
        "truthful_note": "Diagnostic post-process only; not raw VGGT output and not a mentor pass without Open3D visual gate.",
    }
    (output_dir / "surface_completion_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
