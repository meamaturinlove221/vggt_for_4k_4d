from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.audit_visible_surface_teacher import (  # noqa: E402
    connected_component_stats,
    load_scene_image,
    load_scene_mask,
    overlay_mask,
)
from tools.build_mesh_raycast_training_case import _rays_for_pixels, _roi_mask, _world_to_cam  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search a small camera-coordinate translation plus uniform scale and small "
            "camera-axis yaw/pitch/roll rotation for an already-aligned mesh so a target-view "
            "ROI has better visible-surface gate metrics. This is a gate helper only; it "
            "does not patch predictions or train."
        )
    )
    parser.add_argument("--mesh-path", required=True)
    parser.add_argument("--predictions-npz", required=True)
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--view-index", type=int, default=0)
    parser.add_argument(
        "--roi-kind",
        choices=("head", "face", "face_core", "head_face", "shoulder", "all"),
        default="face_core",
    )
    parser.add_argument("--depth-tolerance", type=float, default=0.012)
    parser.add_argument("--dx", default="-0.04,0.04,0.01", help="Camera-x translation min,max,step in meters.")
    parser.add_argument("--dy", default="-0.04,0.04,0.01", help="Camera-y translation min,max,step in meters.")
    parser.add_argument("--dz", default="-0.04,0.04,0.01", help="Camera-z translation min,max,step in meters.")
    parser.add_argument("--scale", default="0.98,1.02,0.01", help="Uniform scale min,max,step.")
    parser.add_argument("--yaw-deg", default="-3,3,1", help="Camera-y yaw min,max,step in degrees.")
    parser.add_argument("--pitch-deg", default="0,0,1", help="Camera-x pitch min,max,step in degrees.")
    parser.add_argument("--roll-deg", default="0,0,1", help="Camera-z roll min,max,step in degrees.")
    parser.add_argument(
        "--pivot",
        choices=("centroid", "bbox", "origin"),
        default="centroid",
        help="World-space pivot for scale and rotation.",
    )
    parser.add_argument(
        "--search-stride",
        type=int,
        default=1,
        help="Evaluate candidates on every Nth ROI pixel during ranking; final metrics are always full ROI.",
    )
    parser.add_argument(
        "--refine-top-k",
        type=int,
        default=30,
        help="Number of ranked candidates to re-evaluate on the full ROI before selecting the final best.",
    )
    parser.add_argument("--min-hit-pixels", type=int, default=5000)
    parser.add_argument("--max-hole-ratio", type=float, default=0.15)
    parser.add_argument("--min-largest-component-ratio", type=float, default=0.80)
    parser.add_argument("--max-median-depth-residual", type=float, default=0.012)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def _parse_range(spec: str) -> np.ndarray:
    parts = [float(part.strip()) for part in str(spec).split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected min,max,step range, got: {spec}")
    start, stop, step = parts
    if step <= 0:
        raise ValueError(f"Step must be positive: {spec}")
    lo = min(start, stop)
    hi = max(start, stop)
    count = int(np.floor((hi - lo) / step + 0.5)) + 1
    values = lo + np.arange(max(count, 1), dtype=np.float64) * step
    values = values[(values >= lo - 1e-9) & (values <= hi + 1e-9)]
    if start > stop:
        values = values[::-1]
    return values.astype(np.float32)


def _percentiles(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return []
    return [float(v) for v in np.percentile(values, [0, 25, 50, 75, 90, 95, 99])]


def _axis_rotation_x(deg: float) -> np.ndarray:
    rad = math.radians(float(deg))
    c = math.cos(rad)
    s = math.sin(rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )


def _axis_rotation_y(deg: float) -> np.ndarray:
    rad = math.radians(float(deg))
    c = math.cos(rad)
    s = math.sin(rad)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def _axis_rotation_z(deg: float) -> np.ndarray:
    rad = math.radians(float(deg))
    c = math.cos(rad)
    s = math.sin(rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _camera_euler_to_world_rotation(
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    extrinsic: np.ndarray,
) -> np.ndarray:
    rotation_world_to_cam = np.asarray(extrinsic[:, :3], dtype=np.float64)
    rotation_cam = (
        _axis_rotation_z(roll_deg)
        @ _axis_rotation_x(pitch_deg)
        @ _axis_rotation_y(yaw_deg)
    )
    return rotation_world_to_cam.T @ rotation_cam @ rotation_world_to_cam


def _candidate_sort_key(row: dict[str, Any]) -> tuple[int, float, int, float, float]:
    return (
        int(row["depth_hit_pixels"]),
        float(row["depth_largest_component_ratio"]),
        int(row["raw_hit_pixels"]),
        -float(row["median_depth_residual"]),
        -abs(float(row["scale"]) - 1.0),
    )


def _metrics_for_similarity(
    scene: Any,
    base_rays: np.ndarray,
    delta_world: np.ndarray,
    rotation_world: np.ndarray,
    scale: float,
    pivot_world: np.ndarray,
    extrinsic: np.ndarray,
    anchor_depth: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    height: int,
    width: int,
    depth_tolerance: float,
) -> dict[str, Any]:
    if scale <= 0:
        raise ValueError(f"Scale must be positive, got {scale}")

    rays = base_rays.copy()
    origins = np.asarray(base_rays[:, :3], dtype=np.float32)
    directions = np.asarray(base_rays[:, 3:], dtype=np.float32)

    rotation_world = np.asarray(rotation_world, dtype=np.float32)
    delta_world = np.asarray(delta_world, dtype=np.float32)
    pivot_world = np.asarray(pivot_world, dtype=np.float32)

    inv_origins = pivot_world[None] + ((origins - pivot_world[None] - delta_world[None]) @ rotation_world) / float(scale)
    inv_directions = directions @ rotation_world
    inv_directions /= np.clip(np.linalg.norm(inv_directions, axis=-1, keepdims=True), 1e-6, None)
    rays[:, :3] = inv_origins.astype(np.float32)
    rays[:, 3:] = inv_directions.astype(np.float32)

    import open3d as o3d

    answers = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
    t_hit = answers["t_hit"].numpy()
    raw_valid = np.isfinite(t_hit)

    original_hit = inv_origins + inv_directions * np.where(raw_valid, t_hit, 0.0)[:, None]
    world_hit = pivot_world[None] + (float(scale) * (original_hit - pivot_world[None])) @ rotation_world.T + delta_world[None]
    cam_hit = _world_to_cam(world_hit.astype(np.float32), extrinsic)
    hit_depth = cam_hit[:, 2]
    residual = np.abs(hit_depth - anchor_depth[ys, xs])
    depth_ok = raw_valid & (hit_depth > 0.05) & (residual <= float(depth_tolerance))

    raw_hit_mask = np.zeros((height, width), dtype=bool)
    depth_ok_mask = np.zeros((height, width), dtype=bool)
    raw_hit_mask[ys[raw_valid], xs[raw_valid]] = True
    depth_ok_mask[ys[depth_ok], xs[depth_ok]] = True
    depth_components = connected_component_stats(depth_ok_mask)
    raw_components = connected_component_stats(raw_hit_mask)
    depth_values = residual[depth_ok]
    raw_values = residual[raw_valid]
    return {
        "raw_hit_pixels": int(raw_hit_mask.sum()),
        "depth_hit_pixels": int(depth_ok_mask.sum()),
        "raw_largest_component_ratio": float(raw_components["largest_component_ratio"]),
        "depth_largest_component_ratio": float(depth_components["largest_component_ratio"]),
        "depth_components": depth_components,
        "raw_components": raw_components,
        "median_depth_residual": float(np.median(depth_values)) if depth_values.size else float("inf"),
        "raw_residual_percentiles": _percentiles(raw_values),
        "depth_residual_percentiles": _percentiles(depth_values),
        "raw_hit_mask": raw_hit_mask,
        "depth_ok_mask": depth_ok_mask,
    }


def _row_from_metrics(
    *,
    dx: float,
    dy: float,
    dz: float,
    scale: float,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    delta_world: np.ndarray,
    metrics: dict[str, Any],
    roi_pixels: int,
) -> dict[str, Any]:
    return {
        "delta_cam": [float(dx), float(dy), float(dz)],
        "delta_world": [float(v) for v in delta_world],
        "scale": float(scale),
        "yaw_deg": float(yaw_deg),
        "pitch_deg": float(pitch_deg),
        "roll_deg": float(roll_deg),
        "raw_hit_pixels": metrics["raw_hit_pixels"],
        "depth_hit_pixels": metrics["depth_hit_pixels"],
        "raw_coverage": float(metrics["raw_hit_pixels"] / max(roi_pixels, 1)),
        "depth_coverage": float(metrics["depth_hit_pixels"] / max(roi_pixels, 1)),
        "raw_hole_ratio": float(1.0 - metrics["raw_hit_pixels"] / max(roi_pixels, 1)),
        "depth_hole_ratio": float(1.0 - metrics["depth_hit_pixels"] / max(roi_pixels, 1)),
        "raw_largest_component_ratio": metrics["raw_largest_component_ratio"],
        "depth_largest_component_ratio": metrics["depth_largest_component_ratio"],
        "median_depth_residual": metrics["median_depth_residual"],
        "raw_residual_percentiles": metrics["raw_residual_percentiles"],
        "depth_residual_percentiles": metrics["depth_residual_percentiles"],
    }


def _select_pivot(mesh: Any, pivot_kind: str) -> np.ndarray:
    if pivot_kind == "origin":
        return np.zeros(3, dtype=np.float64)
    if pivot_kind == "bbox":
        return np.asarray(mesh.get_axis_aligned_bounding_box().get_center(), dtype=np.float64)
    return np.asarray(mesh.get_center(), dtype=np.float64)


def _apply_similarity_to_mesh(
    mesh: Any,
    delta_world: np.ndarray,
    rotation_world: np.ndarray,
    scale: float,
    pivot_world: np.ndarray,
) -> Any:
    import open3d as o3d

    refined_mesh = o3d.geometry.TriangleMesh(mesh)
    vertices = np.asarray(refined_mesh.vertices, dtype=np.float64)
    transformed = pivot_world[None] + (float(scale) * (vertices - pivot_world[None])) @ rotation_world.T + delta_world[None]
    refined_mesh.vertices = o3d.utility.Vector3dVector(transformed)
    refined_mesh.compute_vertex_normals()
    return refined_mesh


def main() -> int:
    args = parse_args()
    mesh_path = Path(args.mesh_path).expanduser().resolve()
    predictions_path = Path(args.predictions_npz).expanduser().resolve()
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)

    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.triangles) == 0:
        raise ValueError(f"Mesh has no triangles: {mesh_path}")
    mesh.compute_vertex_normals()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    pivot_world = _select_pivot(mesh, str(args.pivot))

    with np.load(predictions_path, allow_pickle=False) as payload:
        intrinsics = np.asarray(payload["intrinsic"], dtype=np.float32)
        extrinsics = np.asarray(payload["extrinsic"], dtype=np.float32)
        depth = np.asarray(payload["depth"], dtype=np.float32)[..., 0]

    view_index = int(args.view_index)
    if view_index >= intrinsics.shape[0]:
        raise IndexError(f"view_index={view_index} but predictions only contain {intrinsics.shape[0]} views")
    anchor_depth = depth[view_index]
    height, width = anchor_depth.shape
    target_image = load_scene_image(scene_dir, view_index=view_index)
    target_mask = load_scene_mask(scene_dir, view_index=view_index, target_size=int(height))
    roi = _roi_mask(target_mask, str(args.roi_kind))
    ys, xs = np.nonzero(roi)
    if len(xs) == 0:
        raise RuntimeError(f"Empty ROI: {args.roi_kind}")

    stride = max(1, int(args.search_stride))
    search_ys = ys[::stride]
    search_xs = xs[::stride]
    search_roi_pixels = int(len(search_xs))
    base_rays = _rays_for_pixels(xs, ys, intrinsics[view_index], extrinsics[view_index])
    search_rays = _rays_for_pixels(search_xs, search_ys, intrinsics[view_index], extrinsics[view_index])
    camera_rotation = np.asarray(extrinsics[view_index, :3, :3], dtype=np.float32)
    roi_pixels = int(roi.sum())

    dx_values = _parse_range(args.dx)
    dy_values = _parse_range(args.dy)
    dz_values = _parse_range(args.dz)
    scale_values = _parse_range(args.scale)
    yaw_values = _parse_range(args.yaw_deg)
    pitch_values = _parse_range(args.pitch_deg)
    roll_values = _parse_range(args.roll_deg)
    total_candidates = (
        len(dx_values)
        * len(dy_values)
        * len(dz_values)
        * len(scale_values)
        * len(yaw_values)
        * len(pitch_values)
        * len(roll_values)
    )
    if total_candidates <= 0:
        raise RuntimeError("Empty search grid.")

    ranked_results: list[dict[str, Any]] = []
    for yaw_deg in yaw_values:
        for pitch_deg in pitch_values:
            for roll_deg in roll_values:
                rotation_world = _camera_euler_to_world_rotation(
                    float(yaw_deg),
                    float(pitch_deg),
                    float(roll_deg),
                    extrinsics[view_index],
                )
                for scale in scale_values:
                    for dx in dx_values:
                        for dy in dy_values:
                            for dz in dz_values:
                                delta_cam = np.array([dx, dy, dz], dtype=np.float32)
                                delta_world = camera_rotation.T @ delta_cam
                                metrics = _metrics_for_similarity(
                                    scene=scene,
                                    base_rays=search_rays,
                                    delta_world=delta_world,
                                    rotation_world=rotation_world,
                                    scale=float(scale),
                                    pivot_world=pivot_world,
                                    extrinsic=extrinsics[view_index],
                                    anchor_depth=anchor_depth,
                                    ys=search_ys,
                                    xs=search_xs,
                                    height=height,
                                    width=width,
                                    depth_tolerance=float(args.depth_tolerance),
                                )
                                ranked_results.append(
                                    _row_from_metrics(
                                        dx=float(dx),
                                        dy=float(dy),
                                        dz=float(dz),
                                        scale=float(scale),
                                        yaw_deg=float(yaw_deg),
                                        pitch_deg=float(pitch_deg),
                                        roll_deg=float(roll_deg),
                                        delta_world=delta_world,
                                        metrics=metrics,
                                        roi_pixels=search_roi_pixels,
                                    )
                                )

    ranked_results.sort(key=_candidate_sort_key, reverse=True)
    full_results: list[dict[str, Any]] = []
    full_metrics_by_index: list[dict[str, Any]] = []
    refine_count = min(max(1, int(args.refine_top_k)), len(ranked_results))
    for row in ranked_results[:refine_count]:
        rotation_world = _camera_euler_to_world_rotation(
            row["yaw_deg"],
            row["pitch_deg"],
            row["roll_deg"],
            extrinsics[view_index],
        )
        delta_world = np.asarray(row["delta_world"], dtype=np.float32)
        metrics = _metrics_for_similarity(
            scene=scene,
            base_rays=base_rays,
            delta_world=delta_world,
            rotation_world=rotation_world,
            scale=float(row["scale"]),
            pivot_world=pivot_world,
            extrinsic=extrinsics[view_index],
            anchor_depth=anchor_depth,
            ys=ys,
            xs=xs,
            height=height,
            width=width,
            depth_tolerance=float(args.depth_tolerance),
        )
        full_results.append(
            _row_from_metrics(
                dx=row["delta_cam"][0],
                dy=row["delta_cam"][1],
                dz=row["delta_cam"][2],
                scale=row["scale"],
                yaw_deg=row["yaw_deg"],
                pitch_deg=row["pitch_deg"],
                roll_deg=row["roll_deg"],
                delta_world=delta_world,
                metrics=metrics,
                roi_pixels=roi_pixels,
            )
        )
        full_metrics_by_index.append(metrics)

    best_index = max(range(len(full_results)), key=lambda idx: _candidate_sort_key(full_results[idx]))
    best = full_results[best_index]
    best_metrics = full_metrics_by_index[best_index]
    best_rotation_world = _camera_euler_to_world_rotation(
        best["yaw_deg"],
        best["pitch_deg"],
        best["roll_deg"],
        extrinsics[view_index],
    )
    best_delta_world = np.asarray(best["delta_world"], dtype=np.float64)

    refined_mesh = _apply_similarity_to_mesh(
        mesh,
        delta_world=best_delta_world,
        rotation_world=best_rotation_world,
        scale=float(best["scale"]),
        pivot_world=pivot_world,
    )
    refined_mesh_path = output_dir / "mesh_similarity_refined.ply"
    o3d.io.write_triangle_mesh(str(refined_mesh_path), refined_mesh)

    preview = overlay_mask(
        target_image,
        roi,
        best_metrics["raw_hit_mask"],
        best_metrics["depth_ok_mask"],
    )
    preview_path = output_dir / "mesh_similarity_refined_overlay.png"
    preview.save(preview_path)

    gate = {
        "hit_pixels": int(best["depth_hit_pixels"]) >= int(args.min_hit_pixels),
        "hole_ratio": float(best["depth_hole_ratio"]) <= float(args.max_hole_ratio),
        "largest_component": float(best["depth_largest_component_ratio"]) >= float(args.min_largest_component_ratio),
        "median_depth_residual": float(best["median_depth_residual"]) <= float(args.max_median_depth_residual),
    }
    gate["pass"] = bool(all(gate.values()))
    result_status = "PASS" if gate["pass"] else "FAIL"
    gate_result_path = output_dir / "mesh_similarity_refine_gate_result.txt"
    gate_result_path.write_text(
        "\n".join(
            [
                f"result: {result_status}",
                f"depth_hit_pixels: {int(best['depth_hit_pixels'])}",
                f"depth_hole_ratio: {float(best['depth_hole_ratio']):.9f}",
                f"depth_largest_component_ratio: {float(best['depth_largest_component_ratio']):.9f}",
                f"median_depth_residual: {float(best['median_depth_residual']):.9f}",
                (
                    "negative_result: true - depth-compatible visible-surface gate did not pass"
                    if not gate["pass"]
                    else "negative_result: false"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "mesh_path": str(mesh_path),
        "refined_mesh_path": str(refined_mesh_path),
        "predictions_npz": str(predictions_path),
        "scene_dir": str(scene_dir),
        "view_index": view_index,
        "roi_kind": str(args.roi_kind),
        "roi_pixels": roi_pixels,
        "depth_tolerance": float(args.depth_tolerance),
        "pivot": {
            "kind": str(args.pivot),
            "world": [float(v) for v in pivot_world],
        },
        "search": {
            "dx": [float(v) for v in dx_values],
            "dy": [float(v) for v in dy_values],
            "dz": [float(v) for v in dz_values],
            "scale": [float(v) for v in scale_values],
            "yaw_deg": [float(v) for v in yaw_values],
            "pitch_deg": [float(v) for v in pitch_values],
            "roll_deg": [float(v) for v in roll_values],
            "search_stride": stride,
            "search_roi_pixels": search_roi_pixels,
            "total_candidates": int(total_candidates),
            "full_refine_candidates": int(refine_count),
        },
        "best": best,
        "top_full": sorted(full_results, key=_candidate_sort_key, reverse=True)[: int(args.top_k)],
        "top_ranked_search": ranked_results[: int(args.top_k)],
        "gate_thresholds": {
            "min_hit_pixels": int(args.min_hit_pixels),
            "max_hole_ratio": float(args.max_hole_ratio),
            "min_largest_component_ratio": float(args.min_largest_component_ratio),
            "max_median_depth_residual": float(args.max_median_depth_residual),
        },
        "gate": gate,
        "result_status": result_status,
        "negative_result": None
        if gate["pass"]
        else "FAIL: depth-compatible visible-surface gate did not pass; do not treat this mesh as gate-approved.",
        "gate_result_path": str(gate_result_path),
        "preview_path": str(preview_path),
        "truthful_note": (
            "Similarity refinement is only a teacher gate helper. PASS means an external "
            "teacher is eligible for no-boost fusion/training, not that sparse-view geometry is solved. "
            "If gate.pass is false, this run is a truthful negative result."
        ),
    }
    (output_dir / "mesh_similarity_refine_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if gate["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
