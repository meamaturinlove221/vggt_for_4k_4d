from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage import draw, measure

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from prepare_4k4d_prior_training_case import (  # noqa: E402
    align_intrinsics_for_scene_view,
    load_scene_manifest,
    recover_legacy_crop_source_sizes,
    resolve_scene_camera_params,
)
from research_scene_assets import load_camera_params_sidecar, localize_scene_manifest_paths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "A3 visual-hull initialization preflight from raw masks and known cameras. "
            "This is research-only initialization, not a teacher/candidate/pass."
        )
    )
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--template-payload", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--view-indices", default="0,10,20,30,40,50")
    parser.add_argument("--target-size", type=int, default=96)
    parser.add_argument("--grid-resolution", type=int, default=56)
    parser.add_argument("--support-threshold", type=int, default=4)
    parser.add_argument("--bbox-pad-ratio", type=float, default=0.08)
    parser.add_argument("--max-points", type=int, default=300000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_view_indices(spec: str, view_count: int) -> list[int]:
    out: list[int] = []
    for raw in str(spec).split(","):
        item = raw.strip()
        if not item:
            continue
        value = int(item)
        if value < 0:
            value = view_count + value
        if value < 0 or value >= view_count:
            raise IndexError(f"view index {raw} resolved to {value}, outside [0, {view_count})")
        out.append(value)
    if not out:
        out = list(range(min(6, view_count)))
    return sorted(dict.fromkeys(out))


def align_intrinsics_for_loaded_scene_view(intrinsic: np.ndarray, view: dict[str, Any], target_size: int) -> np.ndarray:
    image_size = view.get("image_size") or [target_size, target_size]
    native_size = int(image_size[0]) if len(image_size) >= 1 else int(target_size)
    meta = view.get("preprocess_meta") or {}
    if meta.get("transform") == "crop_pad_to_square" and native_size != int(target_size):
        native = align_intrinsics_for_scene_view(intrinsic, view, target_size=native_size)
        scale = float(target_size) / float(max(1, native_size))
        out = native.astype(np.float32).copy()
        out[0, :] *= scale
        out[1, :] *= scale
        return out
    return align_intrinsics_for_scene_view(intrinsic, view, target_size=target_size)


def load_template_vertices(payload_path: Path) -> np.ndarray:
    with np.load(payload_path, allow_pickle=False) as payload:
        if "hybrid_vertices" in payload.files:
            return np.asarray(payload["hybrid_vertices"], dtype=np.float32)
        return np.asarray(payload["vertices"], dtype=np.float32)


def load_mask(view: dict[str, Any], target_size: int) -> np.ndarray:
    mask = Image.open(Path(str(view["mask_path"]))).convert("L")
    if mask.size != (target_size, target_size):
        mask = mask.resize((target_size, target_size), Image.Resampling.NEAREST)
    return np.asarray(mask, dtype=np.uint8) > 127


def make_grid(vertices: np.ndarray, resolution: int, pad_ratio: float) -> np.ndarray:
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    span = np.maximum(hi - lo, 1e-5)
    lo = lo - span * float(pad_ratio)
    hi = hi + span * float(pad_ratio)
    xs = np.linspace(lo[0], hi[0], resolution, dtype=np.float32)
    ys = np.linspace(lo[1], hi[1], resolution, dtype=np.float32)
    zs = np.linspace(lo[2], hi[2], resolution, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    return grid.astype(np.float32)


def project_support(points: np.ndarray, mask: np.ndarray, intrinsic: np.ndarray, world_to_cam: np.ndarray) -> tuple[np.ndarray, int]:
    height, width = mask.shape
    rotation = np.asarray(world_to_cam[:3, :3], dtype=np.float32)
    translation = np.asarray(world_to_cam[:3, 3], dtype=np.float32)
    cam = points @ rotation.T + translation[None, :]
    z = cam[:, 2]
    uvw = (intrinsic @ cam.T).T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-8, None)
    xi = np.rint(uv[:, 0]).astype(np.int64)
    yi = np.rint(uv[:, 1]).astype(np.int64)
    inside_image = np.isfinite(uv).all(axis=1) & (z > 1e-6) & (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    support = np.zeros((points.shape[0],), dtype=bool)
    idx = np.nonzero(inside_image)[0]
    if idx.size > 0:
        support[idx] = mask[yi[idx], xi[idx]]
    return support, int(inside_image.sum())


def write_ply(path: Path, points: np.ndarray, support: np.ndarray, max_support: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if points.shape[0] == 0:
        colors = np.zeros((0, 3), dtype=np.uint8)
    else:
        t = np.clip(support.astype(np.float32) / max(float(max_support), 1.0), 0.0, 1.0)
        colors = np.stack([255.0 * t, 180.0 * (1.0 - np.abs(t - 0.5) * 2.0), 255.0 * (1.0 - t)], axis=1).astype(np.uint8)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors, strict=False):
            handle.write(
                f"{float(point[0]):.7f} {float(point[1]):.7f} {float(point[2]):.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def write_mesh_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {vertices.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write(f"element face {faces.shape[0]}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for vertex in vertices:
            handle.write(f"{float(vertex[0]):.7f} {float(vertex[1]):.7f} {float(vertex[2]):.7f} 180 210 255\n")
        for face in faces:
            handle.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def render_mesh_silhouette(
    vertices: np.ndarray,
    faces: np.ndarray,
    intrinsic: np.ndarray,
    world_to_cam: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        return np.zeros((height, width), dtype=bool)
    rotation = np.asarray(world_to_cam[:3, :3], dtype=np.float32)
    translation = np.asarray(world_to_cam[:3, 3], dtype=np.float32)
    cam = vertices @ rotation.T + translation[None, :]
    z = cam[:, 2]
    uvw = (intrinsic @ cam.T).T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-8, None)
    rendered = np.zeros((height, width), dtype=bool)
    for face in faces:
        idx = np.asarray(face, dtype=np.int64)
        if np.any(z[idx] <= 1e-6):
            continue
        tri = uv[idx]
        if not np.isfinite(tri).all():
            continue
        x0 = float(np.min(tri[:, 0]))
        x1 = float(np.max(tri[:, 0]))
        y0 = float(np.min(tri[:, 1]))
        y1 = float(np.max(tri[:, 1]))
        if x1 < 0 or y1 < 0 or x0 >= width or y0 >= height:
            continue
        # Ignore pathological triangles that usually indicate a projection issue.
        if (x1 - x0) > width * 2.5 or (y1 - y0) > height * 2.5:
            continue
        rr, cc = draw.polygon(tri[:, 1], tri[:, 0], shape=rendered.shape)
        rendered[rr, cc] = True
    return rendered


def write_overlay(path: Path, target: np.ndarray, rendered: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.uint8)
    image[..., 1] = np.where(target, 190, 25).astype(np.uint8)
    image[..., 0] = np.where(rendered, 220, image[..., 0]).astype(np.uint8)
    image[..., 2] = np.where(target & rendered, 70, 25).astype(np.uint8)
    Image.fromarray(image).save(path)


def mesh_mask_metrics(target: np.ndarray, rendered: np.ndarray) -> dict[str, float]:
    intersection = np.logical_and(target, rendered).sum()
    union = np.logical_or(target, rendered).sum()
    target_sum = max(int(target.sum()), 1)
    rendered_sum = max(int(rendered.sum()), 1)
    return {
        "mesh_mask_iou": float(intersection / max(int(union), 1)),
        "mesh_mask_recall": float(intersection / target_sum),
        "mesh_mask_precision": float(intersection / rendered_sum),
        "mesh_mask_coverage": float(rendered.mean()),
    }


def extract_visual_hull_mesh(
    occupied: np.ndarray,
    grid: np.ndarray,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    volume = occupied.reshape(resolution, resolution, resolution).astype(np.float32)
    if occupied.sum() == 0 or occupied.sum() == occupied.size:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int32),
            {"mesh_status": "not_extractable_degenerate_occupancy"},
        )
    coords = grid.reshape(resolution, resolution, resolution, 3)
    x_values = coords[:, 0, 0, 0]
    y_values = coords[0, :, 0, 1]
    z_values = coords[0, 0, :, 2]
    spacing = (
        float(x_values[1] - x_values[0]) if resolution > 1 else 1.0,
        float(y_values[1] - y_values[0]) if resolution > 1 else 1.0,
        float(z_values[1] - z_values[0]) if resolution > 1 else 1.0,
    )
    try:
        verts, faces, _normals, _values = measure.marching_cubes(volume, level=0.5, spacing=spacing)
    except Exception as exc:  # noqa: BLE001
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int32),
            {"mesh_status": "marching_cubes_failed", "mesh_error": repr(exc)},
        )
    origin = np.asarray([x_values[0], y_values[0], z_values[0]], dtype=np.float32)
    world_vertices = verts.astype(np.float32) + origin[None, :]
    return (
        world_vertices.astype(np.float32),
        faces.astype(np.int32),
        {
            "mesh_status": "extracted",
            "mesh_vertices": int(world_vertices.shape[0]),
            "mesh_faces": int(faces.shape[0]),
            "mesh_spacing": [float(item) for item in spacing],
        },
    )


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# A3 Visual Hull Initialization Preflight",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This is raw-mask known-camera initialization only. It is not a teacher, candidate, or pass.",
        "",
        "```json",
        json.dumps(summary["summary"], indent=2, ensure_ascii=False),
        "```",
        "",
        "Decision:",
        "",
        "```text",
        summary["decision"],
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_dir = args.scene_dir.resolve()
    manifest = localize_scene_manifest_paths(recover_legacy_crop_source_sizes(scene_dir, load_scene_manifest(scene_dir)), scene_dir)
    views = manifest["exported_views"]
    view_indices = parse_view_indices(args.view_indices, len(views))
    dataset_root = args.dataset_root or Path(str(manifest.get("dataset_root", "")))
    camera_override = load_camera_params_sidecar(scene_dir)
    cameras, camera_source = resolve_scene_camera_params(manifest, dataset_root, args.subset_name, camera_override)
    vertices = load_template_vertices(args.template_payload)
    grid = make_grid(vertices, max(8, int(args.grid_resolution)), float(args.bbox_pad_ratio))

    support_counts = np.zeros((grid.shape[0],), dtype=np.int16)
    visible_counts = np.zeros((grid.shape[0],), dtype=np.int16)
    per_view = []
    view_render_inputs: list[tuple[int, str, np.ndarray, np.ndarray, np.ndarray]] = []
    for view_index in view_indices:
        view = views[view_index]
        camera_id = str(view["camera_id"])
        mask = load_mask(view, int(args.target_size))
        intrinsic = align_intrinsics_for_loaded_scene_view(
            np.asarray(cameras[camera_id]["intrinsic"], dtype=np.float32),
            view,
            int(args.target_size),
        )
        world_to_cam = np.asarray(cameras[camera_id]["world_to_cam"], dtype=np.float32)
        view_render_inputs.append((int(view_index), camera_id, mask, intrinsic, world_to_cam))
        support, visible = project_support(grid, mask, intrinsic, world_to_cam)
        support_counts += support.astype(np.int16)
        # Count visibility per point separately to diagnose too-tight frusta.
        visible_support, _ = project_support(grid, np.ones_like(mask, dtype=bool), intrinsic, world_to_cam)
        visible_counts += visible_support.astype(np.int16)
        per_view.append(
            {
                "view_index": int(view_index),
                "camera_id": camera_id,
                "mask_coverage": float(mask.mean()),
                "grid_points_in_image": int(visible),
                "grid_points_in_mask": int(support.sum()),
            }
        )

    threshold = max(1, int(args.support_threshold))
    occupied = support_counts >= threshold
    occupied_points = grid[occupied]
    occupied_support = support_counts[occupied]
    if occupied_points.shape[0] > int(args.max_points) > 0:
        order = np.argsort(-occupied_support.astype(np.int32))[: int(args.max_points)]
        occupied_points = occupied_points[order]
        occupied_support = occupied_support[order]

    all_ply = output_dir / "visual_hull_supported_points.ply"
    write_ply(all_ply, occupied_points, occupied_support, len(view_indices))
    mesh_vertices, mesh_faces, mesh_summary = extract_visual_hull_mesh(occupied, grid, max(8, int(args.grid_resolution)))
    mesh_ply = output_dir / "visual_hull_init_mesh.ply"
    write_mesh_ply(mesh_ply, mesh_vertices, mesh_faces)
    mesh_review_dir = output_dir / "mesh_projection_review"
    mesh_review_dir.mkdir(parents=True, exist_ok=True)
    mesh_projection_metrics = []
    for view_index, camera_id, mask, intrinsic, world_to_cam in view_render_inputs:
        rendered = render_mesh_silhouette(
            mesh_vertices,
            mesh_faces,
            intrinsic,
            world_to_cam,
            height=int(args.target_size),
            width=int(args.target_size),
        )
        metrics = {
            "view_index": int(view_index),
            "camera_id": camera_id,
            "target_mask_coverage": float(mask.mean()),
            **mesh_mask_metrics(mask, rendered),
        }
        mesh_projection_metrics.append(metrics)
        Image.fromarray((rendered.astype(np.uint8) * 255)).save(mesh_review_dir / f"view_{camera_id}_rendered_mask.png")
        write_overlay(mesh_review_dir / f"view_{camera_id}_overlay.png", mask, rendered)
    np.savez_compressed(
        output_dir / "visual_hull_init_points.npz",
        points=occupied_points.astype(np.float32),
        support=occupied_support.astype(np.int16),
        threshold=np.asarray(threshold, dtype=np.int16),
        view_indices=np.asarray(view_indices, dtype=np.int16),
    )
    np.savez_compressed(
        output_dir / "visual_hull_init_mesh.npz",
        vertices=mesh_vertices.astype(np.float32),
        faces=mesh_faces.astype(np.int32),
        threshold=np.asarray(threshold, dtype=np.int16),
        view_indices=np.asarray(view_indices, dtype=np.int16),
        grid_resolution=np.asarray(max(8, int(args.grid_resolution)), dtype=np.int16),
    )
    support_hist = {str(i): int((support_counts == i).sum()) for i in range(len(view_indices) + 1)}
    summary = {
        "status": "visual_hull_init_preflight_complete",
        "summary": {
            "research_only": True,
            "no_teacher_export": True,
            "no_candidate_export": True,
            "strict_candidate_passes": 0,
            "strict_teacher_passes": 0,
            "scene_dir": str(scene_dir),
            "template_payload": str(args.template_payload.resolve()),
            "camera_source": camera_source,
            "selected_views": view_indices,
            "target_size": int(args.target_size),
            "grid_resolution": int(args.grid_resolution),
            "grid_points": int(grid.shape[0]),
            "support_threshold": threshold,
            "occupied_points": int(occupied_points.shape[0]),
            "occupied_fraction": float(occupied.mean()),
            "support_histogram": support_hist,
            "visible_points_any": int((visible_counts > 0).sum()),
            **mesh_summary,
            "mesh_projection_mean_iou": float(np.mean([item["mesh_mask_iou"] for item in mesh_projection_metrics]))
            if mesh_projection_metrics
            else 0.0,
            "mesh_projection_mean_recall": float(np.mean([item["mesh_mask_recall"] for item in mesh_projection_metrics]))
            if mesh_projection_metrics
            else 0.0,
            "mesh_projection_mean_precision": float(np.mean([item["mesh_mask_precision"] for item in mesh_projection_metrics]))
            if mesh_projection_metrics
            else 0.0,
            "mesh_projection_per_view": mesh_projection_metrics,
            "per_view": per_view,
        },
        "decision": (
            "Use this only as an initialization/readiness diagnostic for A-line dense reconstruction. "
            "A visual hull point cloud is not continuous enough to be a teacher and must still be replaced "
            "by a strict-passing dense surface reconstruction before any teacher-supervised training."
        ),
        "outputs": [
            str(all_ply),
            str(output_dir / "visual_hull_init_points.npz"),
            str(mesh_ply),
            str(output_dir / "visual_hull_init_mesh.npz"),
        ],
    }
    (output_dir / "visual_hull_init_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output_dir / "visual_hull_init_summary.md", summary)
    print(json.dumps(summary["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
