from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_kinect_depth_teacher_targets import (  # noqa: E402
    apply_axis_affine,
    apply_similarity,
    json_ready,
    load_mask,
    make_overlay,
    robust_transform,
    roi_mask_from_human,
    write_ply,
)
from tools.dna_4k4d import normalize_camera_id  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert known-camera COLMAP PatchMatch depth maps into per-pixel teacher targets for a "
            "sparse 4K4D scene. This is a teacher gate, not a final sparse-view pass."
        )
    )
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--base-predictions", required=True, type=Path)
    parser.add_argument("--colmap-export-summary", required=True, type=Path)
    parser.add_argument("--colmap-depth-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--roi-kind", choices=("all", "head", "face", "face_core", "head_face"), default="head_face")
    parser.add_argument("--transform-mode", choices=("similarity", "axis_affine"), default="axis_affine")
    parser.add_argument("--conf-percentile", type=float, default=40.0)
    parser.add_argument("--min-depth-m", type=float, default=0.5)
    parser.add_argument("--max-depth-m", type=float, default=6.0)
    parser.add_argument("--max-correspondences", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_colmap_depth(path: Path) -> np.ndarray:
    data = path.read_bytes()
    amp_count = 0
    header_end = -1
    for index, value in enumerate(data):
        if value == ord("&"):
            amp_count += 1
            if amp_count == 3:
                header_end = index + 1
                break
    if header_end < 0:
        raise ValueError(f"Invalid COLMAP depth header: {path}")
    width, height, channels = [int(item) for item in data[:header_end].decode("ascii")[:-1].split("&")]
    if channels != 1:
        raise ValueError(f"Expected one-channel depth map, got {channels}: {path}")
    depth = np.frombuffer(data[header_end:], dtype=np.float32)
    if depth.size != width * height:
        raise ValueError(f"Depth payload size mismatch: {path}, expected {width * height}, got {depth.size}")
    return depth.reshape(height, width).astype(np.float32)


def ray_points_from_depth(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    z = np.asarray(depth, dtype=np.float32)
    x = (xx - cx) * z / max(fx, 1e-8)
    y = (yy - cy) * z / max(fy, 1e-8)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def cam_to_world(cam_points: np.ndarray, world_to_cam: np.ndarray) -> np.ndarray:
    cam_to_world_matrix = np.linalg.inv(np.asarray(world_to_cam, dtype=np.float64))
    flat = cam_points.reshape(-1, 3).astype(np.float64)
    world = flat @ cam_to_world_matrix[:3, :3].T + cam_to_world_matrix[:3, 3]
    return world.reshape(cam_points.shape).astype(np.float32)


def world_to_target_cam(world_points: np.ndarray, target_world_to_cam: np.ndarray) -> np.ndarray:
    flat = world_points.reshape(-1, 3).astype(np.float64)
    target = flat @ target_world_to_cam[:3, :3].T + target_world_to_cam[:3, 3]
    return target.reshape(world_points.shape).astype(np.float32)


def main() -> int:
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty. Re-run with --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_manifest = read_json(scene_dir / "scene_manifest.json")
    colmap_summary = read_json(args.colmap_export_summary)
    colmap_by_camera = {
        normalize_camera_id(view["camera_id"]): view for view in colmap_summary["exported_views"]
    }

    with np.load(args.base_predictions, allow_pickle=False) as payload:
        base_world = np.asarray(payload["world_points"], dtype=np.float32)
        base_conf = np.asarray(payload["world_points_conf"], dtype=np.float32)
    height, width = base_world.shape[1:3]

    masks = []
    roi_masks = []
    for view in scene_manifest["exported_views"]:
        mask = load_mask(Path(view["mask_path"]), height)
        masks.append(mask)
        roi_masks.append(roi_mask_from_human(mask, args.roi_kind))
    masks_arr = np.stack(masks, axis=0)
    roi_arr = np.stack(roi_masks, axis=0)

    first_camera = normalize_camera_id(scene_manifest["exported_views"][0]["camera_id"])
    if first_camera not in colmap_by_camera:
        raise KeyError(f"First scene camera {first_camera} missing from COLMAP export.")
    target_world_to_cam = np.asarray(colmap_by_camera[first_camera]["world_to_cam"], dtype=np.float64)

    real_world_maps = np.zeros_like(base_world, dtype=np.float32)
    real_target_cam_maps = np.zeros_like(base_world, dtype=np.float32)
    hit_masks = np.zeros(base_world.shape[:3], dtype=bool)
    per_view_summary = []

    for view_idx, view in enumerate(scene_manifest["exported_views"]):
        camera_id = normalize_camera_id(view["camera_id"])
        if camera_id not in colmap_by_camera:
            raise KeyError(f"Scene camera {camera_id} missing from COLMAP export.")
        colmap_view = colmap_by_camera[camera_id]
        depth_path = args.colmap_depth_dir / f"{colmap_view['image_name']}.photometric.bin"
        if not depth_path.is_file():
            raise FileNotFoundError(f"Missing COLMAP depth map: {depth_path}")
        depth = read_colmap_depth(depth_path)
        if depth.shape != (height, width):
            depth = np.asarray(Image.fromarray(depth).resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)
        intrinsic = np.asarray(colmap_view["intrinsic"], dtype=np.float32)
        world_to_cam = np.asarray(colmap_view["world_to_cam"], dtype=np.float64)
        valid = (
            roi_arr[view_idx]
            & np.isfinite(depth)
            & (depth >= float(args.min_depth_m))
            & (depth <= float(args.max_depth_m))
        )
        cam_points = ray_points_from_depth(depth, intrinsic)
        world_points = cam_to_world(cam_points, world_to_cam)
        target_cam_points = world_to_target_cam(world_points, target_world_to_cam)
        world_points[~valid] = 0.0
        target_cam_points[~valid] = 0.0
        real_world_maps[view_idx] = world_points
        real_target_cam_maps[view_idx] = target_cam_points
        hit_masks[view_idx] = valid
        per_view_summary.append(
            {
                "view_index": int(view_idx),
                "camera_id": camera_id,
                "colmap_image_name": str(colmap_view["image_name"]),
                "roi_pixels": int(roi_arr[view_idx].sum()),
                "hit_pixels": int(valid.sum()),
                "hit_ratio_in_roi": float(valid.sum() / max(1, int(roi_arr[view_idx].sum()))),
                "depth_percentiles_on_hits": [float(v) for v in np.percentile(depth[valid], [1, 25, 50, 75, 99])]
                if valid.any()
                else [],
            }
        )

    target_hits = hit_masks[0]
    target_valid_base = (
        target_hits
        & roi_arr[0]
        & np.isfinite(base_world[0]).all(axis=-1)
        & (base_conf[0] >= np.percentile(base_conf[0][masks_arr[0]], float(args.conf_percentile)))
    )
    source_corr = real_target_cam_maps[0][target_valid_base]
    target_corr = base_world[0][target_valid_base]
    transform_summary, transform_matrix = robust_transform(
        source_corr,
        target_corr,
        mode=args.transform_mode,
        max_correspondences=int(args.max_correspondences),
        seed=int(args.seed),
    )
    if args.transform_mode == "similarity":
        teacher_world = apply_similarity(
            real_target_cam_maps,
            float(transform_summary["scale"]),
            np.asarray(transform_summary["rotation"], dtype=np.float64),
            np.asarray(transform_summary["translation"], dtype=np.float64),
        )
    else:
        teacher_world = apply_axis_affine(
            real_target_cam_maps,
            np.asarray(transform_summary["scale_xyz"], dtype=np.float64),
            np.asarray(transform_summary["translation_xyz"], dtype=np.float64),
        )
    teacher_world[~hit_masks] = 0.0
    distance_to_base = np.linalg.norm(teacher_world - base_world, axis=-1)
    distance_valid = hit_masks & np.isfinite(distance_to_base)

    teacher_targets_path = output_dir / "teacher_targets.npz"
    np.savez_compressed(
        teacher_targets_path,
        world_points=teacher_world.astype(np.float32),
        teacher_mask=hit_masks.astype(bool),
        real_world_points=real_world_maps.astype(np.float32),
        real_target_cam_points=real_target_cam_maps.astype(np.float32),
        roi_mask=roi_arr.astype(bool),
        transform_matrix_real_targetcam_to_vggt_world=transform_matrix.astype(np.float32),
    )

    overlay_dir = output_dir / "overlays"
    for view_idx, view in enumerate(scene_manifest["exported_views"]):
        make_overlay(
            Path(view["image_path"]),
            roi_arr[view_idx],
            hit_masks[view_idx],
            overlay_dir / f"{view_idx:02d}_cam{normalize_camera_id(view['camera_id'])}_{args.roi_kind}_colmap_depth_hits.png",
        )

    hit_points = teacher_world[hit_masks]
    if hit_points.shape[0] > 0:
        write_ply(output_dir / "colmap_depth_teacher_vggt_world_hits.ply", hit_points, np.full((hit_points.shape[0], 3), [80, 170, 255], dtype=np.uint8))

    summary = {
        "task": "colmap_depth_teacher_targets",
        "truthful_status": "gate_candidate_not_final_pass",
        "scene_dir": str(scene_dir),
        "base_predictions": str(args.base_predictions.resolve()),
        "colmap_export_summary": str(args.colmap_export_summary.resolve()),
        "colmap_depth_dir": str(args.colmap_depth_dir.resolve()),
        "output_dir": str(output_dir),
        "teacher_targets": str(teacher_targets_path),
        "roi_kind": str(args.roi_kind),
        "transform_mode": str(args.transform_mode),
        "per_view": per_view_summary,
        "target_alignment": transform_summary,
        "distance_to_base_on_teacher_mask_percentiles": [float(v) for v in np.percentile(distance_to_base[distance_valid], [0, 25, 50, 75, 90, 95, 99])]
        if distance_valid.any()
        else [],
        "notes": [
            "COLMAP depth comes from a 60-view known-camera teacher workspace, not from sparse 6-view inference.",
            "This target only becomes useful if direct fusion or training improves same-protocol Open3D face/head close-ups.",
            "Do not claim mentor-final quality from teacher coverage alone.",
        ],
    }
    (output_dir / "colmap_depth_teacher_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(json_ready(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
