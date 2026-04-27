from __future__ import annotations

import argparse
import json
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
            "Search a small camera-coordinate translation for an already-aligned mesh so "
            "a target-view ROI has better visible-surface gate metrics. This is a gate "
            "helper only; it does not patch predictions or train."
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
    parser.add_argument("--dx", default="-0.04,0.04,0.01", help="Camera-x min,max,step in meters.")
    parser.add_argument("--dy", default="-0.04,0.04,0.01", help="Camera-y min,max,step in meters.")
    parser.add_argument("--dz", default="-0.04,0.04,0.01", help="Camera-z min,max,step in meters.")
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
    count = int(np.floor((stop - start) / step + 0.5)) + 1
    values = start + np.arange(max(count, 1), dtype=np.float32) * step
    return values[(values >= min(start, stop) - 1e-6) & (values <= max(start, stop) + 1e-6)]


def _percentiles(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return []
    return [float(v) for v in np.percentile(values, [0, 25, 50, 75, 90, 95, 99])]


def _metrics_for_delta(
    scene: Any,
    base_rays: np.ndarray,
    delta_world: np.ndarray,
    extrinsic: np.ndarray,
    anchor_depth: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    height: int,
    width: int,
    depth_tolerance: float,
) -> dict[str, Any]:
    rays = base_rays.copy()
    rays[:, :3] -= delta_world.astype(np.float32)[None]
    import open3d as o3d

    answers = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
    t_hit = answers["t_hit"].numpy()
    raw_valid = np.isfinite(t_hit)
    origins = base_rays[:, :3]
    directions = base_rays[:, 3:]
    world_hit = origins + directions * np.where(raw_valid, t_hit, 0.0)[:, None]
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


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(Path(args.mesh_path).expanduser().resolve()))
    if len(mesh.triangles) == 0:
        raise ValueError(f"Mesh has no triangles: {args.mesh_path}")
    mesh.compute_vertex_normals()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    with np.load(args.predictions_npz, allow_pickle=False) as payload:
        intrinsics = np.asarray(payload["intrinsic"], dtype=np.float32)
        extrinsics = np.asarray(payload["extrinsic"], dtype=np.float32)
        depth = np.asarray(payload["depth"], dtype=np.float32)[..., 0]

    view_index = int(args.view_index)
    anchor_depth = depth[view_index]
    height, width = anchor_depth.shape
    target_image = load_scene_image(Path(args.scene_dir), view_index=view_index)
    target_mask = load_scene_mask(Path(args.scene_dir), view_index=view_index, target_size=int(height))
    roi = _roi_mask(target_mask, str(args.roi_kind))
    ys, xs = np.nonzero(roi)
    if len(xs) == 0:
        raise RuntimeError(f"Empty ROI: {args.roi_kind}")

    base_rays = _rays_for_pixels(xs, ys, intrinsics[view_index], extrinsics[view_index])
    rotation = np.asarray(extrinsics[view_index, :3, :3], dtype=np.float32)
    roi_pixels = int(roi.sum())

    results: list[dict[str, Any]] = []
    for dx in _parse_range(args.dx):
        for dy in _parse_range(args.dy):
            for dz in _parse_range(args.dz):
                delta_cam = np.array([dx, dy, dz], dtype=np.float32)
                delta_world = rotation.T @ delta_cam
                metrics = _metrics_for_delta(
                    scene=scene,
                    base_rays=base_rays,
                    delta_world=delta_world,
                    extrinsic=extrinsics[view_index],
                    anchor_depth=anchor_depth,
                    ys=ys,
                    xs=xs,
                    height=height,
                    width=width,
                    depth_tolerance=float(args.depth_tolerance),
                )
                results.append(
                    {
                        "delta_cam": [float(dx), float(dy), float(dz)],
                        "delta_world": [float(v) for v in delta_world],
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
                )

    results.sort(
        key=lambda row: (
            int(row["depth_hit_pixels"]),
            float(row["depth_largest_component_ratio"]),
            int(row["raw_hit_pixels"]),
            -float(row["median_depth_residual"]),
        ),
        reverse=True,
    )
    best = results[0]
    best_delta_world = np.asarray(best["delta_world"], dtype=np.float64)
    best_metrics = _metrics_for_delta(
        scene=scene,
        base_rays=base_rays,
        delta_world=best_delta_world.astype(np.float32),
        extrinsic=extrinsics[view_index],
        anchor_depth=anchor_depth,
        ys=ys,
        xs=xs,
        height=height,
        width=width,
        depth_tolerance=float(args.depth_tolerance),
    )

    refined_mesh = o3d.io.read_triangle_mesh(str(Path(args.mesh_path).expanduser().resolve()))
    vertices = np.asarray(refined_mesh.vertices, dtype=np.float64)
    refined_mesh.vertices = o3d.utility.Vector3dVector(vertices + best_delta_world[None])
    refined_mesh.compute_vertex_normals()
    refined_mesh_path = output_dir / "mesh_translation_refined.ply"
    o3d.io.write_triangle_mesh(str(refined_mesh_path), refined_mesh)

    preview = overlay_mask(
        target_image,
        roi,
        best_metrics["raw_hit_mask"],
        best_metrics["depth_ok_mask"],
    )
    preview_path = output_dir / "mesh_translation_refined_overlay.png"
    preview.save(preview_path)

    gate = {
        "hit_pixels": int(best["depth_hit_pixels"]) >= int(args.min_hit_pixels),
        "hole_ratio": float(best["depth_hole_ratio"]) <= float(args.max_hole_ratio),
        "largest_component": float(best["depth_largest_component_ratio"]) >= float(args.min_largest_component_ratio),
        "median_depth_residual": float(best["median_depth_residual"]) <= float(args.max_median_depth_residual),
    }
    gate["pass"] = bool(all(gate.values()))
    summary = {
        "mesh_path": str(Path(args.mesh_path).expanduser().resolve()),
        "refined_mesh_path": str(refined_mesh_path),
        "predictions_npz": str(Path(args.predictions_npz).expanduser().resolve()),
        "scene_dir": str(Path(args.scene_dir).expanduser().resolve()),
        "view_index": view_index,
        "roi_kind": str(args.roi_kind),
        "roi_pixels": roi_pixels,
        "depth_tolerance": float(args.depth_tolerance),
        "best": best,
        "top": results[: int(args.top_k)],
        "gate_thresholds": {
            "min_hit_pixels": int(args.min_hit_pixels),
            "max_hole_ratio": float(args.max_hole_ratio),
            "min_largest_component_ratio": float(args.min_largest_component_ratio),
            "max_median_depth_residual": float(args.max_median_depth_residual),
        },
        "gate": gate,
        "preview_path": str(preview_path),
        "truthful_note": (
            "Translation refinement is only a teacher gate helper. PASS would mean an external "
            "teacher is eligible for no-boost fusion/training, not that sparse-view geometry is solved."
        ),
    }
    (output_dir / "mesh_translation_refine_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if gate["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
