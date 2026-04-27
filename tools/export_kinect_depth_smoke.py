from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_SMC = Path("G:/数据集/datasets/data_used_in_4K4D/kinect/0012_11_kinect.smc")
DEFAULT_OUTPUT_DIR = Path("output/detail_normal_refiner_20260427/kinect_depth_smoke")

CAMERA_PALETTE = np.asarray(
    [
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
    ],
    dtype=np.uint8,
)

CANDIDATES = {
    "rt_as_cam_to_world": "Apply Calibration/Kinect/*/RT directly to camera-space points.",
    "rt_as_world_to_camera_inverse": "Treat RT as world-to-camera and apply inv(RT) to camera-space points.",
}


@dataclass
class CameraPayload:
    camera_id: str
    calibration_id: str
    points_camera: np.ndarray
    colors: np.ndarray
    intrinsics: np.ndarray
    rt_matrix: np.ndarray
    stats: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal Kinect depth teacher smoke: read frame depth/mask, unproject with K, "
            "fuse all selected Kinect views with both plausible RT conventions, and export PLY summaries."
        )
    )
    parser.add_argument("--smc", default=str(DEFAULT_SMC), help="Kinect SMC path.")
    parser.add_argument("--frame", type=int, default=0, help="Frame id to read. Default: 0.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument(
        "--cameras",
        nargs="*",
        default=["all"],
        help="Kinect camera ids to fuse, or 'all'. Default: all.",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="Raw depth units per meter. Kinect uint16 depth is expected to be millimeters.",
    )
    parser.add_argument("--min-depth-m", type=float, default=0.1, help="Drop depths below this value in meters.")
    parser.add_argument("--max-depth-m", type=float, default=8.0, help="Drop depths above this value in meters.")
    parser.add_argument("--stride", type=int, default=1, help="Pixel stride before optional random capping.")
    parser.add_argument(
        "--max-points-per-camera",
        type=int,
        default=0,
        help="Optional random cap per camera after stride. 0 keeps all selected points.",
    )
    parser.add_argument("--seed", type=int, default=20260427, help="Random seed for point capping.")
    parser.add_argument(
        "--skip-open3d-render",
        action="store_true",
        help="Skip the optional Open3D overview render even when Open3D is installed.",
    )
    return parser.parse_args()


def numeric_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def resolve_smc_path(requested_path: Path) -> tuple[Path, str]:
    if requested_path.is_file():
        return requested_path.resolve(), "requested"

    fallback_root = Path("G:/")
    if fallback_root.exists():
        matches = sorted(
            fallback_root.glob(f"*/datasets/data_used_in_4K4D/kinect/{requested_path.name}"),
            key=lambda item: str(item).lower(),
        )
        for match in matches:
            if match.is_file():
                return match.resolve(), "glob_fallback"

    raise FileNotFoundError(f"Could not find Kinect SMC: {requested_path}")


def select_cameras(handle: h5py.File, requested_cameras: list[str]) -> list[str]:
    available = sorted((str(key) for key in handle["Kinect"].keys()), key=numeric_key)
    requested_normalized = [str(item).lower() for item in requested_cameras]
    if not requested_normalized or "all" in requested_normalized:
        return available

    selected = [str(int(item)) for item in requested_cameras]
    missing = [camera_id for camera_id in selected if camera_id not in available]
    if missing:
        raise KeyError(f"Missing Kinect camera ids {missing}; available={available}")
    return selected


def percentile_list(values: np.ndarray, percentiles: list[float]) -> list[float] | None:
    if values.size == 0:
        return None
    return [float(item) for item in np.percentile(values, percentiles)]


def unproject_depth(
    depth_raw: np.ndarray,
    mask_raw: np.ndarray,
    intrinsics: np.ndarray,
    depth_scale: float,
    min_depth_m: float,
    max_depth_m: float,
    stride: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    depth_m = depth_raw.astype(np.float32) / float(depth_scale)
    mask = np.asarray(mask_raw) > 0
    depth_nonzero = depth_raw > 0
    valid_before_range = mask & depth_nonzero & np.isfinite(depth_m)
    valid = valid_before_range & (depth_m >= float(min_depth_m)) & (depth_m <= float(max_depth_m))

    if stride > 1:
        stride_mask = np.zeros_like(valid, dtype=bool)
        stride_mask[::stride, ::stride] = True
        valid &= stride_mask

    pixels_y, pixels_x = np.nonzero(valid)
    selected_depth_m = depth_m[pixels_y, pixels_x].astype(np.float32)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    points_camera = np.column_stack(
        (
            (pixels_x.astype(np.float32) - cx) * selected_depth_m / fx,
            (pixels_y.astype(np.float32) - cy) * selected_depth_m / fy,
            selected_depth_m,
        )
    ).astype(np.float32)

    depth_values_before_range = depth_m[valid_before_range]
    depth_values_after_range = depth_m[valid]
    stats = {
        "image_shape_hw": [int(depth_raw.shape[0]), int(depth_raw.shape[1])],
        "depth_dtype": str(depth_raw.dtype),
        "mask_dtype": str(mask_raw.dtype),
        "depth_raw_min": int(depth_raw.min()),
        "depth_raw_max": int(depth_raw.max()),
        "mask_nonzero_pixels": int(mask.sum()),
        "depth_nonzero_pixels": int(depth_nonzero.sum()),
        "valid_pixels_before_range": int(valid_before_range.sum()),
        "valid_pixels_after_range_and_stride": int(valid.sum()),
        "depth_m_percentiles_before_range": percentile_list(depth_values_before_range, [0, 1, 5, 50, 95, 99, 100]),
        "depth_m_percentiles_after_range_and_stride": percentile_list(
            depth_values_after_range, [0, 1, 5, 50, 95, 99, 100]
        ),
    }
    return points_camera, stats


def maybe_cap_points(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, colors, {"capped": False, "kept_points": int(points.shape[0])}

    rng = np.random.default_rng(seed)
    keep_indices = rng.choice(points.shape[0], size=max_points, replace=False)
    keep_indices.sort()
    return (
        points[keep_indices],
        colors[keep_indices],
        {"capped": True, "kept_points": int(max_points), "original_points": int(points.shape[0])},
    )


def load_camera_payloads(
    handle: h5py.File,
    camera_ids: list[str],
    frame: int,
    depth_scale: float,
    min_depth_m: float,
    max_depth_m: float,
    stride: int,
    max_points_per_camera: int,
    seed: int,
) -> list[CameraPayload]:
    payloads: list[CameraPayload] = []
    frame_key = str(int(frame))
    for camera_id in camera_ids:
        camera_key = str(int(camera_id))
        calibration_id = f"{int(camera_key):02d}"
        depth_path = f"Kinect/{camera_key}/depth/{frame_key}"
        mask_path = f"Kinect/{camera_key}/mask/{frame_key}"
        intrinsics_path = f"Calibration/Kinect/{calibration_id}/K"
        rt_path = f"Calibration/Kinect/{calibration_id}/RT"

        depth_raw = handle[depth_path][()]
        mask_raw = handle[mask_path][()]
        intrinsics = handle[intrinsics_path][()].astype(np.float64)
        rt_matrix = handle[rt_path][()].astype(np.float64)

        points_camera, stats = unproject_depth(
            depth_raw=depth_raw,
            mask_raw=mask_raw,
            intrinsics=intrinsics,
            depth_scale=depth_scale,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
            stride=stride,
        )
        palette_color = CAMERA_PALETTE[int(camera_key) % len(CAMERA_PALETTE)]
        colors = np.repeat(palette_color[None, :], points_camera.shape[0], axis=0)
        points_camera, colors, cap_stats = maybe_cap_points(
            points_camera,
            colors,
            max_points=max_points_per_camera,
            seed=seed + int(camera_key),
        )

        rotation = rt_matrix[:3, :3]
        translation = rt_matrix[:3, 3]
        camera_center_if_world_to_camera = -rotation.T @ translation
        stats.update(
            {
                "camera_id": camera_key,
                "calibration_id": calibration_id,
                "depth_path": depth_path,
                "mask_path": mask_path,
                "intrinsics_path": intrinsics_path,
                "rt_path": rt_path,
                "points_used": int(points_camera.shape[0]),
                "cap": cap_stats,
                "palette_rgb": palette_color.tolist(),
                "K": intrinsics,
                "RT": rt_matrix,
                "camera_center_if_rt_cam_to_world": translation,
                "camera_center_if_rt_world_to_camera": camera_center_if_world_to_camera,
            }
        )

        payloads.append(
            CameraPayload(
                camera_id=camera_key,
                calibration_id=calibration_id,
                points_camera=points_camera,
                colors=colors.astype(np.uint8),
                intrinsics=intrinsics,
                rt_matrix=rt_matrix,
                stats=stats,
            )
        )
    return payloads


def transform_points(points_camera: np.ndarray, rt_matrix: np.ndarray, candidate_name: str) -> np.ndarray:
    if points_camera.size == 0:
        return points_camera.copy()

    if candidate_name == "rt_as_cam_to_world":
        transform = rt_matrix
    elif candidate_name == "rt_as_world_to_camera_inverse":
        transform = np.linalg.inv(rt_matrix)
    else:
        raise KeyError(f"Unknown RT candidate: {candidate_name}")

    homogeneous = np.concatenate(
        [points_camera.astype(np.float64), np.ones((points_camera.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    return (homogeneous @ transform.T)[:, :3].astype(np.float32)


def point_summary(points: np.ndarray) -> dict[str, Any]:
    if points.size == 0:
        return {
            "point_count": 0,
            "bbox_min": None,
            "bbox_max": None,
            "bbox_extent": None,
            "bbox_diagonal": None,
            "bbox_volume": None,
            "robust_bbox_min_p01": None,
            "robust_bbox_max_p99": None,
            "robust_bbox_extent_p01_p99": None,
            "robust_bbox_diagonal_p01_p99": None,
            "centroid": None,
            "median": None,
        }

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_extent = bbox_max - bbox_min
    robust_bbox_min = np.percentile(points, 1, axis=0)
    robust_bbox_max = np.percentile(points, 99, axis=0)
    robust_bbox_extent = robust_bbox_max - robust_bbox_min
    return {
        "point_count": int(points.shape[0]),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "bbox_extent": bbox_extent,
        "bbox_diagonal": float(np.linalg.norm(bbox_extent)),
        "bbox_volume": float(np.prod(np.maximum(bbox_extent, 0.0))),
        "robust_bbox_min_p01": robust_bbox_min,
        "robust_bbox_max_p99": robust_bbox_max,
        "robust_bbox_extent_p01_p99": robust_bbox_extent,
        "robust_bbox_diagonal_p01_p99": float(np.linalg.norm(robust_bbox_extent)),
        "centroid": points.mean(axis=0),
        "median": np.median(points, axis=0),
        "xyz_percentiles_1_5_50_95_99": np.percentile(points, [1, 5, 50, 95, 99], axis=0),
    }


def build_candidate_cloud(
    payloads: list[CameraPayload],
    candidate_name: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    transformed_per_camera = []
    colors_per_camera = []
    per_camera_summary: dict[str, Any] = {}
    for payload in payloads:
        points_world = transform_points(payload.points_camera, payload.rt_matrix, candidate_name)
        transformed_per_camera.append(points_world)
        colors_per_camera.append(payload.colors)
        per_camera_summary[payload.camera_id] = point_summary(points_world)

    if transformed_per_camera:
        points = np.concatenate(transformed_per_camera, axis=0)
        colors = np.concatenate(colors_per_camera, axis=0)
    else:
        points = np.empty((0, 3), dtype=np.float32)
        colors = np.empty((0, 3), dtype=np.uint8)

    summary = point_summary(points)
    summary["description"] = CANDIDATES[candidate_name]
    summary["per_camera"] = per_camera_summary
    return points, colors, summary


def write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertex_count = int(points.shape[0])
    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertices = np.empty(vertex_count, dtype=vertex_dtype)
    if vertex_count:
        vertices["x"] = points[:, 0].astype(np.float32)
        vertices["y"] = points[:, 1].astype(np.float32)
        vertices["z"] = points[:, 2].astype(np.float32)
        vertices["red"] = colors[:, 0].astype(np.uint8)
        vertices["green"] = colors[:, 1].astype(np.uint8)
        vertices["blue"] = colors[:, 2].astype(np.uint8)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {vertex_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        vertices.tofile(handle)


def try_render_open3d(points: np.ndarray, colors: np.ndarray, output_path: Path, skip_render: bool) -> dict[str, Any]:
    if skip_render:
        return {"available": None, "attempted": False, "path": None, "error": "Skipped by --skip-open3d-render."}

    try:
        import open3d as o3d  # type: ignore
        from open3d.visualization import rendering  # type: ignore
    except Exception as exc:
        return {"available": False, "attempted": False, "path": None, "error": repr(exc)}

    try:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        point_cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

        renderer = rendering.OffscreenRenderer(1400, 1000)
        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = 2.0
        renderer.scene.add_geometry("kinect_depth", point_cloud, material)

        bbox = point_cloud.get_axis_aligned_bounding_box()
        center = np.asarray(bbox.get_center(), dtype=np.float64)
        extent = float(max(np.linalg.norm(np.asarray(bbox.get_extent(), dtype=np.float64)), 1.0))
        eye = center + np.asarray([0.0, -1.35 * extent, 0.35 * extent], dtype=np.float64)
        renderer.setup_camera(55.0, eye, center, [0.0, 0.0, 1.0])

        image = renderer.render_to_image()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        success = bool(o3d.io.write_image(str(output_path), image))
        return {
            "available": True,
            "attempted": True,
            "path": str(output_path) if success else None,
            "error": None if success else "open3d.io.write_image returned False",
        }
    except Exception as exc:
        return {"available": True, "attempted": True, "path": None, "error": repr(exc)}


def assess_rt_convention(candidate_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scored = []
    for candidate_name, summary in candidate_summaries.items():
        diagonal = summary.get("robust_bbox_diagonal_p01_p99") or summary.get("bbox_diagonal")
        if diagonal is not None:
            scored.append((float(diagonal), candidate_name))
    if not scored:
        return {"likely": None, "reason": "No points available."}

    scored.sort()
    likely_name = scored[0][1]
    ratio = None
    if len(scored) > 1 and scored[0][0] > 0:
        ratio = float(scored[1][0] / scored[0][0])
    return {
        "likely": likely_name,
        "score": "smallest_fused_robust_bbox_diagonal_p01_p99",
        "robust_bbox_diagonal_ranking": [
            {"candidate": name, "robust_bbox_diagonal_p01_p99": diagonal} for diagonal, name in scored
        ],
        "best_to_second_ratio": ratio,
        "note": (
            "Smoke heuristic only. For this capture, a compact human-scale fused cloud suggests the direct RT "
            "candidate is the usable teacher convention when it wins by a large ratio."
        ),
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    smc_path, smc_resolution = resolve_smc_path(Path(args.smc))
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(smc_path, "r") as handle:
        selected_cameras = select_cameras(handle, list(args.cameras))
        payloads = load_camera_payloads(
            handle=handle,
            camera_ids=selected_cameras,
            frame=args.frame,
            depth_scale=args.depth_scale,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
            stride=max(1, int(args.stride)),
            max_points_per_camera=max(0, int(args.max_points_per_camera)),
            seed=int(args.seed),
        )

    candidate_summaries: dict[str, dict[str, Any]] = {}
    candidate_paths: dict[str, str] = {}
    candidate_clouds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for candidate_name in CANDIDATES:
        points, colors, candidate_summary = build_candidate_cloud(payloads, candidate_name)
        ply_path = output_dir / f"kinect_depth_frame{int(args.frame):06d}_{candidate_name}.ply"
        write_binary_ply(ply_path, points, colors)
        candidate_summary["ply_path"] = str(ply_path)
        candidate_summaries[candidate_name] = candidate_summary
        candidate_paths[candidate_name] = str(ply_path)
        candidate_clouds[candidate_name] = (points, colors)

    rt_assessment = assess_rt_convention(candidate_summaries)
    likely_candidate = rt_assessment.get("likely") or next(iter(CANDIDATES))
    overview_path = output_dir / f"kinect_depth_frame{int(args.frame):06d}_{likely_candidate}_overview.png"
    overview = try_render_open3d(
        points=candidate_clouds[likely_candidate][0],
        colors=candidate_clouds[likely_candidate][1],
        output_path=overview_path,
        skip_render=bool(args.skip_open3d_render),
    )

    total_points = int(sum(payload.points_camera.shape[0] for payload in payloads))
    best_summary = candidate_summaries[likely_candidate]
    teacher_hint = "likely_usable_visible_surface_teacher"
    if total_points == 0:
        teacher_hint = "not_usable_no_points"
    elif best_summary.get("robust_bbox_diagonal_p01_p99") is None:
        teacher_hint = "uncertain_no_robust_geometry"
    elif float(best_summary["robust_bbox_diagonal_p01_p99"]) > 3.0:
        teacher_hint = "uncertain_geometry_too_spread"

    summary = {
        "task": "minimal_kinect_depth_teacher_smoke",
        "smc_path": str(smc_path),
        "smc_resolution": smc_resolution,
        "frame": int(args.frame),
        "selected_cameras": selected_cameras,
        "output_dir": str(output_dir),
        "depth_scale_raw_units_per_meter": float(args.depth_scale),
        "filters": {
            "min_depth_m": float(args.min_depth_m),
            "max_depth_m": float(args.max_depth_m),
            "stride": max(1, int(args.stride)),
            "max_points_per_camera": max(0, int(args.max_points_per_camera)),
        },
        "total_points_after_mask_filter_and_sampling": total_points,
        "camera_stats": [payload.stats for payload in payloads],
        "candidate_plys": candidate_paths,
        "candidate_summaries": candidate_summaries,
        "rt_convention_assessment": rt_assessment,
        "open3d_overview": overview,
        "teacher_hint": teacher_hint,
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(summary), handle, indent=2, ensure_ascii=False)

    print(json.dumps(json_ready({
        "summary": str(summary_path),
        "candidate_plys": candidate_paths,
        "likely_rt_candidate": likely_candidate,
        "teacher_hint": teacher_hint,
        "open3d_overview": overview,
    }), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
