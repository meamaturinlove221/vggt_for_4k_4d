from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.render_open3d_pointcloud import (  # noqa: E402
    apply_roi_filter,
    load_2d_roi_mask_stack,
    load_mask_stack,
    load_rgb_stack,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnostic multi-view-consistent ROI point cloud from VGGT predictions. "
            "This is a geometry post-process / teacher gate, not raw model output."
        )
    )
    parser.add_argument("--predictions-npz", required=True, type=Path)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--roi", choices=("head", "face"), default="face")
    parser.add_argument("--roi-source", choices=("2d", "3d"), default="2d")
    parser.add_argument("--conf-threshold", type=float, default=38.5067)
    parser.add_argument("--max-input-points", type=int, default=240000)
    parser.add_argument("--depth-tolerance", type=float, default=0.035)
    parser.add_argument("--min-consistent-views", type=int, default=2)
    parser.add_argument("--project-mask", choices=("human", "roi", "none"), default="human")
    parser.add_argument("--voxel-size", type=float, default=0.0)
    parser.add_argument("--statistical-outlier-neighbors", type=int, default=20)
    parser.add_argument("--statistical-outlier-std-ratio", type=float, default=1.8)
    parser.add_argument("--poisson-depth", type=int, default=0)
    parser.add_argument("--poisson-density-quantile", type=float, default=0.04)
    parser.add_argument("--sample-surface-points", type=int, default=0)
    parser.add_argument("--render-width", type=int, default=1200)
    parser.add_argument("--render-height", type=int, default=900)
    parser.add_argument("--point-size", type=float, default=2.2)
    parser.add_argument("--seed", type=int, default=20260427)
    return parser.parse_args()


def closed_form_inverse_se3(extrinsic: np.ndarray) -> np.ndarray:
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3:]
    rotation_t = np.transpose(rotation, (0, 2, 1))
    inverted = np.tile(np.eye(4, dtype=extrinsic.dtype), (extrinsic.shape[0], 1, 1))
    inverted[:, :3, :3] = rotation_t
    inverted[:, :3, 3:] = -rotation_t @ translation
    return inverted


def world_to_camera(points: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    rotation = extrinsic[:, :3]
    translation = extrinsic[:, 3]
    return points @ rotation.T + translation[None, :]


def build_initial_points(
    *,
    predictions: dict[str, np.ndarray],
    scene_dir: Path,
    roi: str,
    roi_source: str,
    conf_threshold: float,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    world_points = np.asarray(predictions["world_points"], dtype=np.float32)
    conf = np.asarray(predictions["world_points_conf"], dtype=np.float32)
    target_size = int(world_points.shape[1])
    colors = load_rgb_stack(scene_dir / "images", target_size=target_size)

    if roi_source == "2d":
        masks = load_2d_roi_mask_stack(scene_dir / "masks", target_size=target_size, roi=roi).astype(bool)
        selected = masks & np.isfinite(world_points).all(axis=-1) & np.isfinite(conf) & (conf >= float(conf_threshold))
        roi_summary = {
            "roi": roi,
            "roi_source": "2d_mask",
            "roi_mask_pixels": int(masks.sum()),
            "selected_pixels_before_sampling": int(selected.sum()),
        }
        points = world_points[selected]
        rgb = colors[selected]
    else:
        human_masks = load_mask_stack(scene_dir / "masks", target_size=target_size).astype(bool)
        valid = human_masks & np.isfinite(world_points).all(axis=-1) & np.isfinite(conf) & (conf >= float(conf_threshold))
        points_all = world_points[valid]
        rgb_all = colors[valid]
        points, rgb, roi_summary = apply_roi_filter(points_all, rgb_all, roi=roi)
        roi_summary["selected_pixels_before_sampling"] = int(points.shape[0])

    if points.shape[0] > int(max_points) > 0:
        rng = np.random.default_rng(int(seed))
        indices = rng.choice(points.shape[0], size=int(max_points), replace=False)
        points = points[indices]
        rgb = rgb[indices]
        roi_summary["sampled_to"] = int(max_points)
    return points.astype(np.float32), rgb.astype(np.uint8), roi_summary


def consistency_votes(
    *,
    points: np.ndarray,
    predictions: dict[str, np.ndarray],
    scene_dir: Path,
    roi: str,
    project_mask: str,
    depth_tolerance: float,
) -> tuple[np.ndarray, dict[str, object]]:
    depth = np.asarray(predictions["depth"], dtype=np.float32)[..., 0]
    extrinsic = np.asarray(predictions["extrinsic"], dtype=np.float32)
    intrinsic = np.asarray(predictions["intrinsic"], dtype=np.float32)
    target_size = int(depth.shape[1])
    if project_mask == "roi":
        masks = load_2d_roi_mask_stack(scene_dir / "masks", target_size=target_size, roi=roi).astype(bool)
    elif project_mask == "human":
        masks = load_mask_stack(scene_dir / "masks", target_size=target_size).astype(bool)
    else:
        masks = np.ones(depth.shape, dtype=bool)

    votes = np.zeros(points.shape[0], dtype=np.uint8)
    min_abs_residual = np.full(points.shape[0], np.inf, dtype=np.float32)
    median_residuals = []
    for view_idx in range(depth.shape[0]):
        cam = world_to_camera(points, extrinsic[view_idx])
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.rint(intrinsic[view_idx, 0, 0] * cam[:, 0] / z + intrinsic[view_idx, 0, 2]).astype(np.int32)
            v = np.rint(intrinsic[view_idx, 1, 1] * cam[:, 1] / z + intrinsic[view_idx, 1, 2]).astype(np.int32)
        inside = (
            np.isfinite(z)
            & (z > 0)
            & (u >= 1)
            & (u < target_size - 1)
            & (v >= 1)
            & (v < target_size - 1)
        )
        valid_indices = np.flatnonzero(inside)
        if valid_indices.size == 0:
            continue
        uu = u[valid_indices]
        vv = v[valid_indices]
        mask_ok = masks[view_idx, vv, uu]
        valid_indices = valid_indices[mask_ok]
        uu = u[valid_indices]
        vv = v[valid_indices]
        if valid_indices.size == 0:
            continue
        z_pred = depth[view_idx, vv, uu]
        residual = np.abs(z[valid_indices] - z_pred)
        ok = np.isfinite(z_pred) & (z_pred > 0) & np.isfinite(residual) & (residual <= float(depth_tolerance))
        if ok.any():
            ok_indices = valid_indices[ok]
            votes[ok_indices] += 1
            min_abs_residual[ok_indices] = np.minimum(min_abs_residual[ok_indices], residual[ok].astype(np.float32))
            median_residuals.append(residual[ok].astype(np.float32))

    finite_residual = min_abs_residual[np.isfinite(min_abs_residual)]
    vote_hist = {str(int(vote)): int((votes == vote).sum()) for vote in np.unique(votes)}
    summary = {
        "project_mask": project_mask,
        "depth_tolerance": float(depth_tolerance),
        "vote_histogram": vote_hist,
        "min_residual_percentiles": [float(v) for v in np.percentile(finite_residual, [0, 25, 50, 75, 90, 95, 99])]
        if finite_residual.size
        else [],
        "all_view_ok_residual_count": int(sum(arr.size for arr in median_residuals)),
    }
    return votes, summary


def load_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Open3D is required; run with the g3splat Python environment.") from exc
    return o3d


def make_cloud(points: np.ndarray, colors: np.ndarray):
    o3d = load_open3d()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector((colors.astype(np.float32) / 255.0).clip(0, 1).astype(np.float64))
    return cloud


def postprocess_cloud(cloud, args: argparse.Namespace):
    summary: dict[str, object] = {
        "points_before_postprocess": int(len(cloud.points)),
        "voxel_size": float(args.voxel_size),
        "statistical_outlier_neighbors": int(args.statistical_outlier_neighbors),
        "statistical_outlier_std_ratio": float(args.statistical_outlier_std_ratio),
    }
    if float(args.voxel_size) > 0.0 and len(cloud.points) > 0:
        cloud = cloud.voxel_down_sample(voxel_size=float(args.voxel_size))
        summary["points_after_voxel"] = int(len(cloud.points))
    if int(args.statistical_outlier_neighbors) > 0 and len(cloud.points) > int(args.statistical_outlier_neighbors):
        cloud, inliers = cloud.remove_statistical_outlier(
            nb_neighbors=int(args.statistical_outlier_neighbors),
            std_ratio=float(args.statistical_outlier_std_ratio),
        )
        summary["points_after_statistical_outlier"] = int(len(cloud.points))
        summary["statistical_inliers"] = int(len(inliers))
    return cloud, summary


def poisson_resample(cloud, args: argparse.Namespace):
    if int(args.poisson_depth) <= 0 or int(args.sample_surface_points) <= 0 or len(cloud.points) < 128:
        return cloud, {"poisson_enabled": False}
    o3d = load_open3d()
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=36))
    cloud.orient_normals_consistent_tangent_plane(36)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud,
        depth=int(args.poisson_depth),
        n_threads=0,
    )
    densities = np.asarray(densities)
    if densities.size:
        threshold = float(np.quantile(densities, float(args.poisson_density_quantile)))
        mesh.remove_vertices_by_mask(densities < threshold)
    sampled = mesh.sample_points_poisson_disk(number_of_points=int(args.sample_surface_points), init_factor=3)
    source_points = np.asarray(cloud.points)
    source_colors = np.asarray(cloud.colors)
    sampled_points = np.asarray(sampled.points)
    if source_points.size and sampled_points.size:
        tree = o3d.geometry.KDTreeFlann(cloud)
        colors = []
        for point in sampled_points:
            _, indices, _ = tree.search_knn_vector_3d(point, 1)
            colors.append(source_colors[int(indices[0])] if indices else np.array([0.7, 0.7, 0.7]))
        sampled.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return sampled, {
        "poisson_enabled": True,
        "poisson_depth": int(args.poisson_depth),
        "poisson_density_quantile": float(args.poisson_density_quantile),
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_triangles": int(len(mesh.triangles)),
        "sampled_surface_points": int(len(sampled.points)),
    }


def save_renders(cloud, output_dir: Path, width: int, height: int, point_size: float) -> list[str]:
    o3d = load_open3d()
    vis = o3d.visualization.Visualizer()
    vis.create_window("Multi-view Consistent ROI Cloud", width=int(width), height=int(height), visible=False)
    vis.add_geometry(cloud)
    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    render_option.point_size = float(point_size)
    render_option.light_on = True
    bounds = cloud.get_axis_aligned_bounding_box()
    center = np.asarray(bounds.get_center(), dtype=np.float64)
    extent = np.asarray(bounds.get_extent(), dtype=np.float64)
    radius = float(np.linalg.norm(extent) + 1e-6)
    presets = [
        ("front", [0.0, 0.0, -1.0], center, [0.0, -1.0, 0.0], 0.72),
        ("side", [1.0, 0.0, 0.0], center, [0.0, -1.0, 0.0], 0.72),
        ("top", [0.0, -1.0, 0.0], center, [0.0, 0.0, -1.0], 0.70),
        ("face_close", [0.15, -0.05, -0.99], center + np.array([0.0, -0.08 * radius, 0.0]), [0.0, -1.0, 0.0], 1.35),
    ]
    saved: list[str] = []
    control = vis.get_view_control()
    for name, front, lookat, up, zoom in presets:
        control.set_front(front)
        control.set_lookat(lookat)
        control.set_up(up)
        control.set_zoom(float(zoom))
        vis.poll_events()
        vis.update_renderer()
        path = output_dir / f"{name}.png"
        vis.capture_screen_image(str(path), do_render=True)
        saved.append(str(path))
    vis.destroy_window()
    return saved


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.predictions_npz, allow_pickle=False) as payload:
        predictions = {key: payload[key] for key in payload.files}

    points, colors, roi_summary = build_initial_points(
        predictions=predictions,
        scene_dir=args.scene_dir,
        roi=args.roi,
        roi_source=args.roi_source,
        conf_threshold=float(args.conf_threshold),
        max_points=int(args.max_input_points),
        seed=int(args.seed),
    )
    votes, consistency_summary = consistency_votes(
        points=points,
        predictions=predictions,
        scene_dir=args.scene_dir,
        roi=args.roi,
        project_mask=str(args.project_mask),
        depth_tolerance=float(args.depth_tolerance),
    )
    keep = votes >= int(args.min_consistent_views)
    filtered_points = points[keep]
    filtered_colors = colors[keep]
    if filtered_points.shape[0] == 0:
        raise RuntimeError("No points left after multi-view consistency filtering.")

    cloud = make_cloud(filtered_points, filtered_colors)
    cloud, post_summary = postprocess_cloud(cloud, args)
    cloud, poisson_summary = poisson_resample(cloud, args)
    output_ply = output_dir / "multiview_consistent_roi_cloud.ply"
    load_open3d().io.write_point_cloud(str(output_ply), cloud, write_ascii=False, compressed=False)
    screenshots = save_renders(cloud, output_dir, int(args.render_width), int(args.render_height), float(args.point_size))

    summary = {
        "task": "multiview_consistent_roi_pointcloud",
        "truthful_status": "diagnostic_postprocess_not_raw_model_output",
        "predictions_npz": str(args.predictions_npz.resolve()),
        "scene_dir": str(args.scene_dir.resolve()),
        "output_dir": str(output_dir),
        "roi_summary": roi_summary,
        "consistency": consistency_summary,
        "min_consistent_views": int(args.min_consistent_views),
        "points_after_consistency": int(filtered_points.shape[0]),
        "postprocess": post_summary,
        "poisson": poisson_summary,
        "output_ply": str(output_ply),
        "screenshots": screenshots,
        "notes": [
            "This artifact tests whether geometry-only multi-view filtering can clean head/face point clouds.",
            "It is not a mentor-final pass unless the visual close-up is clearly better and the method is reported as post-processing.",
        ],
    }
    (output_dir / "multiview_consistent_roi_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
