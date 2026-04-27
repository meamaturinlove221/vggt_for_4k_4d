from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_mesh_raycast_training_case import (  # noqa: E402
    _cam_to_world,
    _copy_case,
    _load_masks,
    _make_preview,
    _raycast_mesh,
    _roi_mask,
)
from vggt.utils.normal_refiner import point_map_to_normal_numpy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import an external human mesh/OBJ/PLY, optionally Sim(3)-align it to an anchor VGGT "
            "point cloud, raycast it through the sparse 4K4D cameras, and patch a training case."
        )
    )
    parser.add_argument("--source-case-dir", required=True)
    parser.add_argument("--external-mesh-path", required=True)
    parser.add_argument("--target-scene-dir", required=True)
    parser.add_argument("--anchor-predictions-npz", required=True)
    parser.add_argument("--output-case-dir", required=True)
    parser.add_argument("--output-diagnostics-dir", required=True)
    parser.add_argument(
        "--roi-kind",
        choices=("head", "face", "face_core", "head_face", "shoulder", "all"),
        default="head_face",
    )
    parser.add_argument(
        "--align-roi-kind",
        choices=("head", "face", "face_core", "head_face", "shoulder", "all"),
        default=None,
        help="ROI used to align the external mesh. Defaults to --roi-kind for backward compatibility.",
    )
    parser.add_argument("--align-mode", choices=("none", "umeyama_icp"), default="none")
    parser.add_argument("--anchor-conf-percentile", type=float, default=40.0)
    parser.add_argument("--max-anchor-points", type=int, default=120000)
    parser.add_argument("--mesh-sample-points", type=int, default=60000)
    parser.add_argument("--max-correspondence-distance", type=float, default=0.06)
    parser.add_argument("--align-iterations", type=int, default=12)
    parser.add_argument("--pointcloud-poisson-depth", type=int, default=8)
    parser.add_argument("--depth-tolerance", type=float, default=0.08)
    parser.add_argument("--conf-boost", type=float, default=96.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "Open3D is required. Run with the g3splat Python used by scripts/render_pointcloud_open3d.ps1."
        ) from exc
    return o3d


def _load_mesh(mesh_path: Path, poisson_depth: int):
    o3d = _load_open3d()
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(np.asarray(mesh.triangles)) > 0:
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.compute_vertex_normals()
        return mesh, {"source_type": "triangle_mesh"}

    point_cloud = o3d.io.read_point_cloud(str(mesh_path))
    if len(np.asarray(point_cloud.points)) == 0:
        raise RuntimeError(f"External mesh has no triangles or points: {mesh_path}")
    point_cloud.remove_non_finite_points()
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=48))
    point_cloud.orient_normals_consistent_tangent_plane(48)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud, depth=int(poisson_depth), n_threads=0
    )
    densities = np.asarray(densities)
    if densities.size:
        keep_mask = densities >= np.quantile(densities, 0.04)
        mesh.remove_vertices_by_mask(~keep_mask)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh, {"source_type": "point_cloud_poisson", "source_points": int(len(point_cloud.points))}


def _sample_mesh_points(mesh, sample_count: int) -> np.ndarray:
    if sample_count <= 0:
        return np.asarray(mesh.vertices, dtype=np.float32)
    sampled = mesh.sample_points_uniformly(number_of_points=int(sample_count), use_triangle_normal=False)
    return np.asarray(sampled.points, dtype=np.float32)


def _transform_points(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (float(scale) * points @ rotation.T + translation[None]).astype(np.float32)


def _estimate_similarity(source_points: np.ndarray, target_points: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if len(source) < 4 or len(target) < 4 or len(source) != len(target):
        raise ValueError("Need matching source/target correspondences for similarity estimation.")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean[None]
    target_centered = target - target_mean[None]
    covariance = target_centered.T @ source_centered / float(len(source))
    left_singular, singular_values, right_singular_t = np.linalg.svd(covariance)
    handedness = np.ones(3, dtype=np.float64)
    if np.linalg.det(left_singular @ right_singular_t) < 0:
        handedness[-1] = -1.0
    rotation = left_singular @ np.diag(handedness) @ right_singular_t
    source_variance = np.mean(np.sum(source_centered * source_centered, axis=1))
    scale = float(np.sum(singular_values * handedness) / max(source_variance, 1e-12))
    translation = target_mean - scale * (source_mean @ rotation.T)
    return scale, rotation.astype(np.float64), translation.astype(np.float64)


def _load_anchor_points(
    *,
    anchor_predictions: dict[str, np.ndarray],
    target_scene_dir: Path,
    roi_kind: str,
    conf_percentile: float,
    max_points: int,
) -> np.ndarray:
    points = np.asarray(anchor_predictions["world_points"], dtype=np.float32)
    conf = np.asarray(anchor_predictions.get("world_points_conf", anchor_predictions.get("depth_conf")), dtype=np.float32)
    masks = _load_masks(target_scene_dir, int(points.shape[1]))
    selected: list[np.ndarray] = []
    for view_idx in range(points.shape[0]):
        roi = _roi_mask(masks[view_idx], roi_kind)
        valid = roi & np.isfinite(points[view_idx]).all(axis=-1) & np.isfinite(conf[view_idx])
        if not valid.any():
            continue
        threshold = float(np.percentile(conf[view_idx][valid], float(conf_percentile)))
        valid &= conf[view_idx] >= threshold
        if valid.any():
            selected.append(points[view_idx][valid])
    if not selected:
        raise RuntimeError("No anchor points selected for external mesh alignment.")
    anchor_points = np.concatenate(selected, axis=0).astype(np.float32)
    if len(anchor_points) > int(max_points):
        rng = np.random.default_rng(20260425)
        keep_indices = rng.choice(len(anchor_points), size=int(max_points), replace=False)
        anchor_points = anchor_points[keep_indices]
    return anchor_points


def _nearest_correspondences(source_points: np.ndarray, target_points: np.ndarray, max_distance: float):
    o3d = _load_open3d()
    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(target_points.astype(np.float64))
    tree = o3d.geometry.KDTreeFlann(target_cloud)
    max_distance_sq = float(max_distance) * float(max_distance)
    source_corr = []
    target_corr = []
    for point in source_points.astype(np.float64):
        _, indices, distances = tree.search_knn_vector_3d(point, 1)
        if indices and distances[0] <= max_distance_sq:
            source_corr.append(point)
            target_corr.append(target_points[int(indices[0])])
    if not source_corr:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    return np.asarray(source_corr, dtype=np.float32), np.asarray(target_corr, dtype=np.float32)


def _align_mesh_to_anchor(mesh, anchor_points: np.ndarray, args: argparse.Namespace) -> tuple[object, dict[str, object]]:
    o3d = _load_open3d()
    mesh_points = _sample_mesh_points(mesh, int(args.mesh_sample_points))
    rng = np.random.default_rng(20260425)
    if len(mesh_points) > 25000:
        mesh_points = mesh_points[rng.choice(len(mesh_points), size=25000, replace=False)]
    if len(anchor_points) > 25000:
        anchor_points = anchor_points[rng.choice(len(anchor_points), size=25000, replace=False)]

    source_center = mesh_points.mean(axis=0)
    target_center = anchor_points.mean(axis=0)
    source_radius = float(np.sqrt(np.mean(np.sum((mesh_points - source_center[None]) ** 2, axis=1))))
    target_radius = float(np.sqrt(np.mean(np.sum((anchor_points - target_center[None]) ** 2, axis=1))))
    scale = target_radius / max(source_radius, 1e-8)
    rotation = np.eye(3, dtype=np.float64)
    translation = target_center.astype(np.float64) - scale * source_center.astype(np.float64)

    history = []
    for iteration in range(int(args.align_iterations)):
        transformed = _transform_points(mesh_points, scale, rotation, translation)
        source_corr, target_corr = _nearest_correspondences(
            transformed,
            anchor_points,
            max_distance=float(args.max_correspondence_distance),
        )
        if len(source_corr) < 64:
            history.append({"iteration": iteration, "correspondences": int(len(source_corr)), "stopped": "too_few"})
            break
        delta_scale, delta_rotation, delta_translation = _estimate_similarity(source_corr, target_corr)
        rotation = delta_rotation @ rotation
        translation = delta_scale * (translation @ delta_rotation.T) + delta_translation
        scale = float(delta_scale * scale)
        residual = np.linalg.norm(_transform_points(source_corr, delta_scale, delta_rotation, delta_translation) - target_corr, axis=1)
        history.append(
            {
                "iteration": iteration,
                "correspondences": int(len(source_corr)),
                "median_residual": float(np.median(residual)),
                "mean_residual": float(np.mean(residual)),
                "scale": float(scale),
            }
        )

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    mesh.vertices = o3d.utility.Vector3dVector(_transform_points(vertices.astype(np.float32), scale, rotation, translation).astype(np.float64))
    mesh.compute_vertex_normals()
    return mesh, {
        "align_mode": "umeyama_icp",
        "scale": float(scale),
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "history": history,
    }


def main() -> int:
    args = parse_args()
    source_case_dir = Path(args.source_case_dir)
    output_case_dir = Path(args.output_case_dir)
    diagnostics_dir = Path(args.output_diagnostics_dir)
    target_scene_dir = Path(args.target_scene_dir)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    _copy_case(source_case_dir, output_case_dir, overwrite=bool(args.overwrite))
    with np.load(output_case_dir / "inputs.npz", allow_pickle=False) as payload:
        inputs = {key: np.array(payload[key]) for key in payload.files}
    with np.load(output_case_dir / "targets.npz", allow_pickle=False) as payload:
        targets = {key: np.array(payload[key]) for key in payload.files}
    with np.load(args.anchor_predictions_npz, allow_pickle=False) as payload:
        anchor = {key: np.array(payload[key]) for key in payload.files}

    mesh, mesh_meta = _load_mesh(Path(args.external_mesh_path), poisson_depth=int(args.pointcloud_poisson_depth))
    raw_vertices = np.asarray(mesh.vertices)
    mesh_meta.update(
        {
            "external_mesh_path": str(Path(args.external_mesh_path).resolve()),
            "raw_vertices": int(raw_vertices.shape[0]),
            "raw_triangles": int(np.asarray(mesh.triangles).shape[0]),
        }
    )

    alignment_summary: dict[str, object] = {"align_mode": str(args.align_mode)}
    if args.align_mode == "umeyama_icp":
        align_roi_kind = str(args.align_roi_kind or args.roi_kind)
        anchor_points = _load_anchor_points(
            anchor_predictions=anchor,
            target_scene_dir=target_scene_dir,
            roi_kind=align_roi_kind,
            conf_percentile=float(args.anchor_conf_percentile),
            max_points=int(args.max_anchor_points),
        )
        mesh, alignment_summary = _align_mesh_to_anchor(mesh, anchor_points, args)
        alignment_summary["align_roi_kind"] = align_roi_kind
        mesh_meta["anchor_points_for_alignment"] = int(len(anchor_points))

    o3d = _load_open3d()
    o3d.io.write_triangle_mesh(str(diagnostics_dir / "external_mesh_transformed.ply"), mesh)

    target_masks = np.asarray(inputs.get("point_masks"), dtype=bool)
    hit_cam, hit_depth, hit_mask = _raycast_mesh(mesh, target_masks, targets, anchor, args)
    hit_world = _cam_to_world(hit_cam, np.asarray(targets["extrinsics"], dtype=np.float32))
    teacher_normals = np.zeros_like(hit_cam, dtype=np.float32)
    for view_idx in range(hit_cam.shape[0]):
        normal_map, normal_valid = point_map_to_normal_numpy(hit_cam[view_idx], hit_mask[view_idx])
        teacher_normals[view_idx] = normal_map
        hit_mask[view_idx] &= normal_valid
        _make_preview(
            np.asarray(inputs["images"][view_idx], dtype=np.uint8),
            np.asarray(anchor["depth"], dtype=np.float32)[view_idx, ..., 0],
            hit_depth[view_idx],
            hit_mask[view_idx],
            diagnostics_dir / f"{view_idx:02d}_external_mesh_raycast_preview.png",
        )

    targets["depths"] = np.asarray(targets["depths"], dtype=np.float32)
    targets["cam_points"] = np.asarray(targets["cam_points"], dtype=np.float32)
    targets["world_points"] = np.asarray(targets["world_points"], dtype=np.float32)
    targets["depth_conf"] = np.asarray(targets["depth_conf"], dtype=np.float32)
    targets["world_points_conf"] = np.asarray(targets["world_points_conf"], dtype=np.float32)
    targets["depths"][hit_mask] = hit_depth[hit_mask]
    targets["cam_points"][hit_mask] = hit_cam[hit_mask]
    targets["world_points"][hit_mask] = hit_world[hit_mask]
    targets["depth_conf"][hit_mask] = np.maximum(targets["depth_conf"][hit_mask], float(args.conf_boost))
    targets["world_points_conf"][hit_mask] = np.maximum(targets["world_points_conf"][hit_mask], float(args.conf_boost))
    targets["teacher_normals"] = teacher_normals.astype(np.float32)
    targets["teacher_mask"] = hit_mask.astype(bool)
    targets["prior_normals"] = np.asarray(targets.get("prior_normals", teacher_normals), dtype=np.float32)
    targets["prior_normals"][hit_mask] = teacher_normals[hit_mask]
    for key in ("head_roi_mask", "face_roi_mask", "hairline_mask", "ear_band_mask"):
        if key in targets:
            targets[key] = (np.asarray(targets[key], dtype=bool) | hit_mask).astype(bool)
    np.savez_compressed(output_case_dir / "targets.npz", **targets)

    summary = {
        "source_case_dir": str(source_case_dir.resolve()),
        "external_mesh_path": str(Path(args.external_mesh_path).resolve()),
        "target_scene_dir": str(target_scene_dir.resolve()),
        "anchor_predictions_npz": str(Path(args.anchor_predictions_npz).resolve()),
        "output_case_dir": str(output_case_dir.resolve()),
        "roi_kind": str(args.roi_kind),
        "depth_tolerance": float(args.depth_tolerance),
        "conf_boost": float(args.conf_boost),
        "mesh_meta": mesh_meta,
        "alignment": alignment_summary,
        "hit_pixels_total": int(hit_mask.sum()),
        "hit_pixels_per_view": [int(value) for value in hit_mask.reshape(hit_mask.shape[0], -1).sum(axis=1)],
    }
    (diagnostics_dir / "external_mesh_raycast_teacher_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest_path = output_case_dir / "case_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    manifest["external_mesh_raycast_training_patch"] = summary
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
