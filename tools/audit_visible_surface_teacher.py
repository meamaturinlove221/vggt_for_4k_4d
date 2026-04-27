from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_mesh_raycast_training_case import _rays_for_pixels, _roi_mask, _world_to_cam  # noqa: E402
from vggt.utils.normal_refiner import preprocess_mask_image  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether an already-aligned external mesh provides a continuous visible "
            "surface in a target sparse-view ROI. This is a gate only: it does not patch "
            "predictions and does not train."
        )
    )
    parser.add_argument("--mesh-path", required=True, help="Aligned mesh path, e.g. external_mesh_transformed.ply")
    parser.add_argument("--predictions-npz", required=True, help="Anchor VGGT predictions with intrinsics/extrinsics/depth")
    parser.add_argument("--scene-dir", required=True, help="Scene directory containing images/ and masks/")
    parser.add_argument("--output-dir", required=True, help="Directory for summary and preview images")
    parser.add_argument("--view-index", type=int, default=0, help="View index to audit")
    parser.add_argument(
        "--roi-kind",
        choices=("head", "face", "face_core", "head_face", "shoulder", "all"),
        default="face_core",
    )
    parser.add_argument(
        "--depth-tolerance",
        type=float,
        default=0.06,
        help="Depth residual tolerance for anchor-compatible hit statistics.",
    )
    parser.add_argument("--min-hit-pixels", type=int, default=11000)
    parser.add_argument("--max-hole-ratio", type=float, default=0.15)
    parser.add_argument("--min-largest-component-ratio", type=float, default=0.80)
    parser.add_argument("--max-median-depth-residual", type=float, default=0.012)
    return parser.parse_args()


def load_scene_image(scene_dir: Path, view_index: int) -> Image.Image:
    image_paths = sorted((scene_dir / "images").glob("*"))
    if not image_paths:
        raise FileNotFoundError(f"No images found under {scene_dir / 'images'}")
    if view_index >= len(image_paths):
        raise IndexError(f"view_index={view_index} but only {len(image_paths)} images are available")
    return Image.open(image_paths[view_index]).convert("RGB")


def load_scene_mask(scene_dir: Path, view_index: int, target_size: int) -> np.ndarray:
    mask_paths = sorted((scene_dir / "masks").glob("*"))
    if not mask_paths:
        raise FileNotFoundError(f"No masks found under {scene_dir / 'masks'}")
    if view_index >= len(mask_paths):
        raise IndexError(f"view_index={view_index} but only {len(mask_paths)} masks are available")
    return preprocess_mask_image(mask_paths[view_index], target_size=target_size).astype(bool)


def connected_component_stats(mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    total = int(mask.sum())
    if total == 0:
        return {
            "components": 0,
            "largest_component_pixels": 0,
            "largest_component_ratio": 0.0,
        }

    largest = 0
    components = 0
    ys, xs = np.nonzero(mask)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if visited[start_y, start_x]:
            continue
        components += 1
        count = 0
        queue: deque[tuple[int, int]] = deque([(start_y, start_x)])
        visited[start_y, start_x] = True
        while queue:
            y, x = queue.popleft()
            count += 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        largest = max(largest, count)

    return {
        "components": int(components),
        "largest_component_pixels": int(largest),
        "largest_component_ratio": float(largest / max(total, 1)),
    }


def overlay_mask(image: Image.Image, roi: np.ndarray, raw_hits: np.ndarray, depth_ok_hits: np.ndarray) -> Image.Image:
    base = np.asarray(image.resize((roi.shape[1], roi.shape[0]), Image.Resampling.BILINEAR), dtype=np.float32)
    out = base.copy()
    roi_only = roi & ~raw_hits
    raw_only = raw_hits & ~depth_ok_hits
    ok = depth_ok_hits
    out[roi_only] = 0.65 * out[roi_only] + np.array([30, 144, 255], dtype=np.float32) * 0.35
    out[raw_only] = 0.60 * out[raw_only] + np.array([255, 210, 0], dtype=np.float32) * 0.40
    out[ok] = 0.50 * out[ok] + np.array([0, 220, 70], dtype=np.float32) * 0.50
    preview = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(preview)
    draw.text((8, 8), "blue=ROI, yellow=mesh hit, green=depth-compatible hit", fill=(0, 0, 0))
    return preview


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
        raise ValueError(f"Mesh has no triangles and cannot be raycast: {mesh_path}")
    mesh.compute_vertex_normals()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    with np.load(predictions_path, allow_pickle=False) as payload:
        intrinsics = np.asarray(payload["intrinsic"], dtype=np.float32)
        extrinsics = np.asarray(payload["extrinsic"], dtype=np.float32)
        anchor_depth = np.asarray(payload["depth"], dtype=np.float32)[..., 0]

    view_index = int(args.view_index)
    if view_index >= intrinsics.shape[0]:
        raise IndexError(f"view_index={view_index} but predictions only contain {intrinsics.shape[0]} views")
    height, width = anchor_depth.shape[1:]
    target_image = load_scene_image(scene_dir, view_index=view_index)
    target_mask = load_scene_mask(scene_dir, view_index=view_index, target_size=int(height))
    roi = _roi_mask(target_mask, str(args.roi_kind))
    ys, xs = np.nonzero(roi)
    if len(xs) == 0:
        raise RuntimeError(f"ROI {args.roi_kind} is empty for view {view_index}")

    rays = _rays_for_pixels(xs, ys, intrinsics[view_index], extrinsics[view_index])
    answers = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
    t_hit = answers["t_hit"].numpy()
    raw_valid = np.isfinite(t_hit)
    origins = rays[:, :3]
    directions = rays[:, 3:]
    world_hit = origins + directions * np.where(raw_valid, t_hit, 0.0)[:, None]
    cam_hit = _world_to_cam(world_hit.astype(np.float32), extrinsics[view_index])
    hit_depth = cam_hit[:, 2]
    residual = np.abs(hit_depth - anchor_depth[view_index, ys, xs])
    depth_ok = raw_valid & (hit_depth > 0.05) & (residual <= float(args.depth_tolerance))

    raw_hit_mask = np.zeros((height, width), dtype=bool)
    depth_ok_mask = np.zeros((height, width), dtype=bool)
    raw_hit_mask[ys[raw_valid], xs[raw_valid]] = True
    depth_ok_mask[ys[depth_ok], xs[depth_ok]] = True

    raw_components = connected_component_stats(raw_hit_mask)
    depth_ok_components = connected_component_stats(depth_ok_mask)
    roi_pixels = int(roi.sum())
    raw_hit_pixels = int(raw_hit_mask.sum())
    depth_ok_hit_pixels = int(depth_ok_mask.sum())
    raw_residual_values = residual[raw_valid]
    depth_ok_residual_values = residual[depth_ok]

    def percentiles(values: np.ndarray) -> list[float]:
        if values.size == 0:
            return []
        return [float(v) for v in np.percentile(values, [0, 25, 50, 75, 90, 95, 99])]

    gate = {
        "hit_pixels": depth_ok_hit_pixels >= int(args.min_hit_pixels),
        "hole_ratio": (1.0 - depth_ok_hit_pixels / max(roi_pixels, 1)) <= float(args.max_hole_ratio),
        "largest_component": depth_ok_components["largest_component_ratio"] >= float(args.min_largest_component_ratio),
        "median_depth_residual": (
            bool(depth_ok_residual_values.size)
            and float(np.median(depth_ok_residual_values)) <= float(args.max_median_depth_residual)
        ),
    }
    gate["pass"] = bool(all(gate.values()))

    summary: dict[str, Any] = {
        "mesh_path": str(mesh_path),
        "predictions_npz": str(predictions_path),
        "scene_dir": str(scene_dir),
        "view_index": view_index,
        "roi_kind": str(args.roi_kind),
        "depth_tolerance": float(args.depth_tolerance),
        "roi_pixels": roi_pixels,
        "raw_visible": {
            "hit_pixels": raw_hit_pixels,
            "coverage": float(raw_hit_pixels / max(roi_pixels, 1)),
            "hole_ratio": float(1.0 - raw_hit_pixels / max(roi_pixels, 1)),
            "components": raw_components,
            "depth_residual_percentiles": percentiles(raw_residual_values),
        },
        "depth_compatible": {
            "hit_pixels": depth_ok_hit_pixels,
            "coverage": float(depth_ok_hit_pixels / max(roi_pixels, 1)),
            "hole_ratio": float(1.0 - depth_ok_hit_pixels / max(roi_pixels, 1)),
            "components": depth_ok_components,
            "depth_residual_percentiles": percentiles(depth_ok_residual_values),
        },
        "gate_thresholds": {
            "min_hit_pixels": int(args.min_hit_pixels),
            "max_hole_ratio": float(args.max_hole_ratio),
            "min_largest_component_ratio": float(args.min_largest_component_ratio),
            "max_median_depth_residual": float(args.max_median_depth_residual),
        },
        "gate": gate,
        "truthful_note": "Teacher visibility gate only; PASS means the teacher surface is eligible for fusion/training, not that VGGT sparse-view geometry is solved.",
    }

    preview_path = output_dir / "visible_surface_teacher_overlay.png"
    overlay_mask(target_image, roi=roi, raw_hits=raw_hit_mask, depth_ok_hits=depth_ok_mask).save(preview_path)
    summary["preview_path"] = str(preview_path)

    summary_path = output_dir / "visible_surface_teacher_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if gate["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
