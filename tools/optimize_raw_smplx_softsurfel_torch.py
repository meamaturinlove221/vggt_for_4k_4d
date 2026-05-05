from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from normal_line_multiview_eval import load_scene_view  # noqa: E402
from prepare_4k4d_prior_training_case import (  # noqa: E402
    align_intrinsics_for_scene_view,
    load_optional_annotation_payload,
    load_scene_manifest,
    recover_legacy_crop_source_sizes,
    resolve_scene_camera_params,
    resolve_smplx_model_dir,
)
from tools.smplx_numpy import (  # noqa: E402
    build_smplx_vertex_features,
    compute_vertex_normals,
    forward_smplx_mesh,
    rasterize_world_mesh,
    resolve_smplx_model_path,
)


PART_NAMES = {
    0: "torso_limbs",
    1: "hands_wide",
    2: "head_face",
    3: "head_top_hairline_proxy",
    4: "lower_clothing_proxy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Raw-image soft-surface upper-bound v1 smoke. This script uses raw RGB/masks/"
            "calibrated cameras/SMPL-X only. It intentionally does not use VGGT depth, "
            "point maps, normals, confidence, or r-candidate outputs."
        )
    )
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--smplx-model-dir", type=Path)
    parser.add_argument("--smplx-gender", choices=("neutral", "female", "male"), default="neutral")
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument("--max-views", type=int, default=6)
    parser.add_argument("--view-stride", type=int, default=1)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--surfel-samples", type=int, default=1200)
    parser.add_argument("--surface-samples-for-sdf", type=int, default=2500)
    parser.add_argument("--boundary-samples", type=int, default=192)
    parser.add_argument("--render-pixel-chunk", type=int, default=4096)
    parser.add_argument("--gaussian-sigma", type=float, default=1.7)
    parser.add_argument("--mask-weight", type=float, default=1.0)
    parser.add_argument(
        "--recall-weight",
        type=float,
        default=0.45,
        help="Soft target-coverage guard. Prevents the optimizer from improving IoU by shrinking the rendered body.",
    )
    parser.add_argument("--outside-weight", type=float, default=0.20)
    parser.add_argument("--boundary-weight", type=float, default=0.05)
    parser.add_argument("--photo-weight", type=float, default=0.08)
    parser.add_argument(
        "--photo-depth-tolerance",
        type=float,
        default=0.0,
        help=(
            "Optional visibility filter for the photometric loss. When >0, a surfel must be "
            "near the current rendered front depth in that view before its sampled RGB counts."
        ),
    )
    parser.add_argument("--translation-reg", type=float, default=0.05)
    parser.add_argument("--scale-reg", type=float, default=0.05)
    parser.add_argument("--offset-reg", type=float, default=0.35)
    parser.add_argument("--offset-smooth-reg", type=float, default=0.08)
    parser.add_argument("--normal-offset-limit-body", type=float, default=0.015)
    parser.add_argument("--normal-offset-limit-hands", type=float, default=0.030)
    parser.add_argument("--normal-offset-limit-head", type=float, default=0.022)
    parser.add_argument("--normal-offset-limit-hairline", type=float, default=0.040)
    parser.add_argument("--normal-offset-limit-clothing", type=float, default=0.035)
    parser.add_argument(
        "--extra-hairline-surfels",
        type=int,
        default=0,
        help=(
            "Optional diagnostic: build a shared 3D head-top/hairline support set from raw mask pixels "
            "that are missing from the optimized SMPL-X raster. Disabled by default."
        ),
    )
    parser.add_argument("--extra-hairline-max-per-view", type=int, default=256)
    parser.add_argument("--extra-hairline-min-support", type=int, default=2)
    parser.add_argument("--extra-hairline-top-frac", type=float, default=0.40)
    parser.add_argument("--extra-hairline-nearest-radius", type=float, default=10.0)
    parser.add_argument("--extra-hairline-depth-offset", type=float, default=0.0)
    parser.add_argument("--extra-hairline-voxel", type=float, default=0.004)
    parser.add_argument("--extra-hairline-export-radius", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overlay-limit", type=int, default=8)
    parser.add_argument("--export-raster-targets", action="store_true")
    parser.add_argument(
        "--export-scene-dir",
        type=Path,
        help="Optional scene whose protocol should receive hard-rasterized depth/world/normal targets.",
    )
    parser.add_argument("--export-target-size", type=int, default=518)
    parser.add_argument("--export-target-views", default="all", help="Comma-separated view indices or 'all'.")
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value).replace("\\", "/")
    if isinstance(value, str):
        return value.replace("\\", "/")
    return value


def homogeneous(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :4] = matrix
        return out
    raise ValueError(f"Expected 3x4 or 4x4 matrix, got {matrix.shape}")


def align_intrinsics_for_loaded_scene_view(intrinsic: np.ndarray, view: dict[str, Any], target_size: int) -> np.ndarray:
    """Align intrinsics to the already-exported crop PNG then optional smoke resize.

    `crop_pad_to_square` manifests store crop bboxes in the native exported image
    frame (normally 518x518). The shared helper is correct for the native training
    size, but if we pass a smaller CPU-smoke size directly it applies those native
    bbox coordinates to the smaller frame. For this raw-image smoke we first align
    at the manifest image size and then scale the exported square view down.
    """
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


def mask_sdf(mask: np.ndarray) -> np.ndarray:
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    inside = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
    outside = cv2.distanceTransform((1 - mask_u8).astype(np.uint8), cv2.DIST_L2, 3)
    sdf = outside - inside
    return (sdf / float(max(mask.shape))).astype(np.float32)


def boundary_points(mask: np.ndarray, max_points: int) -> np.ndarray:
    mask_u8 = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    grad = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, kernel)
    ys, xs = np.nonzero(grad)
    if xs.size == 0:
        ys, xs = np.nonzero(mask_u8)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    count = min(int(max_points), xs.size)
    indices = np.linspace(0, xs.size - 1, count).round().astype(np.int64)
    return np.stack([xs[indices], ys[indices]], axis=1).astype(np.float32)


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.max() > 1.5:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def project_points(points: torch.Tensor, world_to_cam: torch.Tensor, intrinsic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rotation = world_to_cam[:3, :3]
    translation = world_to_cam[:3, 3]
    cam = points @ rotation.T + translation[None, :]
    z = cam[:, 2]
    uvw = cam @ intrinsic.T
    uv = uvw[:, :2] / uvw[:, 2:3].clamp_min(1e-6)
    return uv, z, cam


def sample_grid_values(image: torch.Tensor, uv: torch.Tensor, height: int, width: int) -> torch.Tensor:
    x = uv[:, 0] / float(max(1, width - 1)) * 2.0 - 1.0
    y = uv[:, 1] / float(max(1, height - 1)) * 2.0 - 1.0
    grid = torch.stack([x, y], dim=-1).view(1, 1, -1, 2)
    sampled = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    # grid_sample returns [1, C, 1, N]; expose samples as [N, C].
    return sampled[0, :, 0, :].transpose(0, 1)


def sample_sdf(sdf: torch.Tensor, uv: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return sample_grid_values(sdf, uv, height, width).reshape(-1)


def classify_vertex_parts(canonical_positions: np.ndarray) -> np.ndarray:
    canonical = np.asarray(canonical_positions, dtype=np.float32)
    x = canonical[:, 0]
    y = canonical[:, 1]
    abs_x = np.abs(x - np.median(x))
    y20, y55, y82, y90, y95 = np.percentile(y, [20, 55, 82, 90, 95])
    abs_x88 = np.percentile(abs_x, 88)
    abs_x94 = np.percentile(abs_x, 94)

    parts = np.zeros((canonical.shape[0],), dtype=np.int64)
    parts[y < y20] = 4
    parts[y > y82] = 2
    parts[y > y95] = 3
    hands = (abs_x > abs_x88) & (y > y20) & (y < y90)
    parts[hands] = 1
    far_hands = (abs_x > abs_x94) & (y > y20) & (y < y95)
    parts[far_hands] = 1
    return parts.astype(np.int64)


def make_part_limits(parts: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    limits = np.full(parts.shape, float(args.normal_offset_limit_body), dtype=np.float32)
    limits[parts == 1] = float(args.normal_offset_limit_hands)
    limits[parts == 2] = float(args.normal_offset_limit_head)
    limits[parts == 3] = float(args.normal_offset_limit_hairline)
    limits[parts == 4] = float(args.normal_offset_limit_clothing)

    reg_weights = np.full(parts.shape, 1.0, dtype=np.float32)
    reg_weights[parts == 1] = 0.55
    reg_weights[parts == 2] = 0.65
    reg_weights[parts == 3] = 0.35
    reg_weights[parts == 4] = 0.45
    return limits, reg_weights


def unique_edges(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    edges = np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0).astype(np.int64)


def sample_surface_plan(
    base_vertices: np.ndarray,
    faces: np.ndarray,
    vertex_parts: np.ndarray,
    sample_count: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    triangles = np.asarray(base_vertices, dtype=np.float32)[np.asarray(faces, dtype=np.int64)]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    probs = areas / np.clip(areas.sum(), 1e-8, None)
    face_indices = rng.choice(len(faces), size=max(1, int(sample_count)), replace=True, p=probs)
    u = rng.random(face_indices.shape[0]).astype(np.float32)
    v = rng.random(face_indices.shape[0]).astype(np.float32)
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    bary = np.stack([1.0 - u - v, u, v], axis=1).astype(np.float32)
    surfel_vertex_ids = np.asarray(faces, dtype=np.int64)[face_indices]
    surfel_vertex_parts = vertex_parts[surfel_vertex_ids]
    surfel_parts = np.asarray(
        [np.bincount(row.astype(np.int64), minlength=len(PART_NAMES)).argmax() for row in surfel_vertex_parts],
        dtype=np.int64,
    )
    return {
        "face_indices": face_indices.astype(np.int64),
        "vertex_ids": surfel_vertex_ids.astype(np.int64),
        "barycentric": bary.astype(np.float32),
        "part_ids": surfel_parts.astype(np.int64),
    }


def compute_surfels(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    face_indices: torch.Tensor,
    barycentric: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sampled_faces = faces.index_select(0, face_indices)
    tri = vertices[sampled_faces]
    surfels = (tri * barycentric[:, :, None]).sum(dim=1)
    normals = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    normals = F.normalize(normals, dim=1, eps=1e-6)
    return surfels, normals


def render_soft_surfel_maps(
    surfels: torch.Tensor,
    normals: torch.Tensor,
    world_to_cam: torch.Tensor,
    intrinsic: torch.Tensor,
    height: int,
    width: int,
    sigma: float,
    pixel_chunk: int,
) -> dict[str, torch.Tensor]:
    uv, z, cam = project_points(surfels, world_to_cam, intrinsic)
    valid = (
        torch.isfinite(uv).all(dim=1)
        & torch.isfinite(z)
        & (z > 1e-5)
        & (uv[:, 0] >= -3.0 * float(sigma))
        & (uv[:, 0] <= float(width - 1) + 3.0 * float(sigma))
        & (uv[:, 1] >= -3.0 * float(sigma))
        & (uv[:, 1] <= float(height - 1) + 3.0 * float(sigma))
    )
    uv_valid = uv[valid]
    z_valid = z[valid]
    normals_valid = normals[valid]
    if uv_valid.shape[0] == 0:
        zeros = torch.zeros((height, width), dtype=surfels.dtype, device=surfels.device)
        return {
            "mask": zeros,
            "depth": zeros,
            "normal": torch.zeros((height, width, 3), dtype=surfels.dtype, device=surfels.device),
            "visibility": zeros,
            "valid_count": torch.zeros((), dtype=surfels.dtype, device=surfels.device),
        }

    ys = torch.arange(height, dtype=surfels.dtype, device=surfels.device) + 0.5
    xs = torch.arange(width, dtype=surfels.dtype, device=surfels.device) + 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    pixels = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

    masks = []
    depths = []
    normal_maps = []
    vis_maps = []
    sigma2 = max(1e-6, float(sigma) ** 2)
    chunk = max(1, int(pixel_chunk))
    for start in range(0, pixels.shape[0], chunk):
        pixel_chunk_xy = pixels[start : start + chunk]
        d2 = (pixel_chunk_xy[:, None, :] - uv_valid[None, :, :]).square().sum(dim=2)
        weights = torch.exp(-0.5 * d2 / sigma2)
        sumw = weights.sum(dim=1).clamp_min(1e-8)
        # Saturating alpha keeps the mask differentiable without pretending to be a z-buffer.
        alpha = 1.0 - torch.exp(-sumw)
        depth = (weights * z_valid[None, :]).sum(dim=1) / sumw
        normal = (weights @ normals_valid) / sumw[:, None]
        normal = F.normalize(normal, dim=1, eps=1e-6)
        masks.append(alpha)
        depths.append(depth)
        normal_maps.append(normal)
        vis_maps.append(sumw)

    mask = torch.cat(masks, dim=0).reshape(height, width).clamp(0.0, 1.0)
    depth = torch.cat(depths, dim=0).reshape(height, width)
    normal = torch.cat(normal_maps, dim=0).reshape(height, width, 3)
    visibility = torch.cat(vis_maps, dim=0).reshape(height, width)
    return {
        "mask": mask,
        "depth": depth,
        "normal": normal,
        "visibility": visibility,
        "valid_count": valid.float().sum(),
    }


def photometric_consistency_loss(
    surfels: torch.Tensor,
    view_payloads: list[dict[str, Any]],
    height: int,
    width: int,
    *,
    visibility_depths: list[torch.Tensor] | None = None,
    depth_tolerance: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    colors = []
    weights = []
    for view_i, payload in enumerate(view_payloads):
        uv, z, _ = project_points(surfels, payload["world_to_cam"], payload["intrinsic"])
        in_image = (
            (z > 1e-5)
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] <= width - 1)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] <= height - 1)
        )
        if visibility_depths is not None and depth_tolerance > 0.0 and view_i < len(visibility_depths):
            sampled_depth = sample_grid_values(visibility_depths[view_i][None, None], uv, height, width).reshape(-1)
            visible = torch.isfinite(sampled_depth) & (sampled_depth > 1e-5)
            visible = visible & ((z - sampled_depth).abs() <= float(depth_tolerance))
            in_image = in_image & visible
        sampled_rgb = sample_grid_values(payload["rgb_t"], uv, height, width)
        sampled_mask = sample_grid_values(payload["mask_t"], uv, height, width).reshape(-1).clamp(0.0, 1.0)
        weight = sampled_mask * in_image.float()
        colors.append(sampled_rgb)
        weights.append(weight)

    color_stack = torch.stack(colors, dim=0)
    weight_stack = torch.stack(weights, dim=0)
    support = (weight_stack > 0.25).float().sum(dim=0)
    valid = support >= 2.0
    if not valid.any():
        zero = surfels.sum() * 0.0
        return zero, {"valid_surfels": 0.0, "mean_support": 0.0}
    local_colors = color_stack[:, valid, :]
    local_weights = weight_stack[:, valid].clamp_min(0.0)
    norm = local_weights.sum(dim=0).clamp_min(1e-6)
    mean = (local_colors * local_weights[:, :, None]).sum(dim=0) / norm[:, None]
    residual = torch.sqrt((local_colors - mean[None, :, :]).square().sum(dim=2) + 1e-6)
    loss = (residual * local_weights).sum() / local_weights.sum().clamp_min(1e-6)
    return loss, {
        "valid_surfels": float(valid.float().sum().detach().cpu()),
        "mean_support": float(support[valid].mean().detach().cpu()),
    }


def compute_mask_metrics(rendered: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    rendered = np.asarray(rendered, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = rendered & target
    union = rendered | target
    render_pixels = int(rendered.sum())
    target_pixels = int(target.sum())
    intersection_pixels = int(intersection.sum())
    union_pixels = int(union.sum())
    return {
        "render_pixels": render_pixels,
        "target_pixels": target_pixels,
        "intersection_pixels": intersection_pixels,
        "union_pixels": union_pixels,
        "iou": float(intersection_pixels / union_pixels) if union_pixels else None,
        "target_recall": float(intersection_pixels / target_pixels) if target_pixels else None,
        "render_precision": float(intersection_pixels / render_pixels) if render_pixels else None,
    }


def summarize(values: list[float | None]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float32)
    if arr.size == 0:
        return {"count": 0, "mean": None, "p10": None, "p50": None, "p90": None, "min": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def overlay_masks(rgb: np.ndarray, target: np.ndarray, rendered: np.ndarray) -> np.ndarray:
    out = np.asarray(rgb, dtype=np.float32)
    if out.max() <= 1.5:
        out *= 255.0
    target = np.asarray(target, dtype=bool)
    rendered = np.asarray(rendered, dtype=bool)
    both = target & rendered
    target_only = target & ~rendered
    rendered_only = rendered & ~target
    out[target_only] = 0.45 * out[target_only] + 0.55 * np.array([0, 220, 0], dtype=np.float32)
    out[rendered_only] = 0.45 * out[rendered_only] + 0.55 * np.array([240, 0, 0], dtype=np.float32)
    out[both] = 0.55 * out[both] + 0.45 * np.array([255, 220, 0], dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def save_contact_sheet(paths: list[Path], output_path: Path, columns: int = 4) -> None:
    if not paths:
        return
    images = [Image.open(path).convert("RGB") for path in paths]
    width, height = images[0].size
    rows = int(np.ceil(len(images) / max(1, columns)))
    sheet = Image.new("RGB", (columns * width, rows * height), (255, 255, 255))
    for idx, image in enumerate(images):
        sheet.paste(image, ((idx % columns) * width, (idx // columns) * height))
    sheet.save(output_path)


def save_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {vertices.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write(f"element face {faces.shape[0]}\n")
        handle.write("property list uchar int vertex_indices\nend_header\n")
        for vertex in vertices:
            handle.write(f"{float(vertex[0])} {float(vertex[1])} {float(vertex[2])}\n")
        for face in faces:
            handle.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def save_point_ply(path: Path, points: np.ndarray, normals: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    normals = None if normals is None else np.asarray(normals, dtype=np.float32)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        has_normals = normals is not None and normals.shape == points.shape
        if has_normals:
            handle.write("property float nx\nproperty float ny\nproperty float nz\n")
        handle.write("end_header\n")
        if has_normals:
            for point, normal in zip(points, normals):
                handle.write(
                    f"{float(point[0])} {float(point[1])} {float(point[2])} "
                    f"{float(normal[0])} {float(normal[1])} {float(normal[2])}\n"
                )
        else:
            for point in points:
                handle.write(f"{float(point[0])} {float(point[1])} {float(point[2])}\n")


def save_depth_image(path: Path, depth: np.ndarray, mask: np.ndarray) -> None:
    depth = np.asarray(depth, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if mask.any():
        vals = depth[mask]
        lo, hi = np.percentile(vals[np.isfinite(vals)], [2, 98])
        denom = max(1e-6, float(hi - lo))
        img = np.clip((depth - lo) / denom, 0.0, 1.0)
    else:
        img = np.zeros_like(depth, dtype=np.float32)
    Image.fromarray((img * 255.0).astype(np.uint8)).save(path)


def save_normal_image(path: Path, normal: np.ndarray, mask: np.ndarray) -> None:
    normal = np.asarray(normal, dtype=np.float32)
    img = np.clip((normal + 1.0) * 0.5, 0.0, 1.0)
    img[~np.asarray(mask, dtype=bool)] = 0.0
    Image.fromarray((img * 255.0).astype(np.uint8)).save(path)


def parse_view_spec(spec: str, count: int) -> list[int]:
    text = str(spec).strip().lower()
    if text == "all":
        return list(range(count))
    out: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0 or idx >= count:
            raise IndexError(f"view index {idx} outside [0,{count})")
        out.append(idx)
    return sorted(set(out))


def backproject_pixels_to_world(
    pixels_xy: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    world_to_cam: np.ndarray,
) -> np.ndarray:
    pixels_xy = np.asarray(pixels_xy, dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32).reshape(-1)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    world_to_cam = homogeneous(world_to_cam)
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    cam = np.stack(
        [
            (pixels_xy[:, 0] + 0.5 - cx) * depth / max(1e-8, fx),
            (pixels_xy[:, 1] + 0.5 - cy) * depth / max(1e-8, fy),
            depth,
        ],
        axis=1,
    ).astype(np.float32)
    cam_to_world = np.linalg.inv(world_to_cam).astype(np.float32)
    return (cam @ cam_to_world[:3, :3].T + cam_to_world[:3, 3][None, :]).astype(np.float32)


def nearest_rendered_depths(
    query_xy: np.ndarray,
    source_xy: np.ndarray,
    source_depth: np.ndarray,
    max_radius: float,
    *,
    chunk: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query_xy = np.asarray(query_xy, dtype=np.float32)
    source_xy = np.asarray(source_xy, dtype=np.float32)
    source_depth = np.asarray(source_depth, dtype=np.float32).reshape(-1)
    if query_xy.size == 0 or source_xy.size == 0:
        return (
            np.zeros((0,), dtype=bool),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    keep = np.zeros((query_xy.shape[0],), dtype=bool)
    depth = np.zeros((query_xy.shape[0],), dtype=np.float32)
    distance = np.full((query_xy.shape[0],), np.inf, dtype=np.float32)
    max_r2 = float(max_radius) ** 2
    for start in range(0, query_xy.shape[0], max(1, int(chunk))):
        q = query_xy[start : start + chunk]
        d2 = ((q[:, None, :] - source_xy[None, :, :]) ** 2).sum(axis=2)
        nearest = d2.argmin(axis=1)
        best = d2[np.arange(q.shape[0]), nearest]
        local_keep = np.isfinite(best) & (best <= max_r2)
        keep[start : start + q.shape[0]] = local_keep
        depth[start : start + q.shape[0]] = source_depth[nearest]
        distance[start : start + q.shape[0]] = np.sqrt(np.maximum(best, 0.0)).astype(np.float32)
    return keep, depth, distance


def mask_support_for_points(
    points_world: np.ndarray,
    view_payloads: list[dict[str, Any]],
    height: int,
    width: int,
) -> np.ndarray:
    points_world = np.asarray(points_world, dtype=np.float32)
    support = np.zeros((points_world.shape[0],), dtype=np.int32)
    if points_world.size == 0:
        return support
    points_t = torch.from_numpy(points_world)
    for payload in view_payloads:
        uv, z, _ = project_points(
            points_t,
            torch.from_numpy(payload["world_to_cam_np"].astype(np.float32)),
            torch.from_numpy(payload["intrinsic_np"].astype(np.float32)),
        )
        uv_np = uv.numpy()
        z_np = z.numpy()
        px = np.rint(uv_np[:, 0]).astype(np.int32)
        py = np.rint(uv_np[:, 1]).astype(np.int32)
        valid = (
            np.isfinite(uv_np).all(axis=1)
            & np.isfinite(z_np)
            & (z_np > 1e-5)
            & (px >= 0)
            & (px < width)
            & (py >= 0)
            & (py < height)
        )
        if valid.any():
            mask = np.asarray(payload["mask"], dtype=bool)
            hit = np.zeros_like(valid)
            hit[valid] = mask[py[valid], px[valid]]
            support += hit.astype(np.int32)
    return support


def voxel_downsample_points(
    points: np.ndarray,
    normals: np.ndarray,
    support: np.ndarray,
    voxel: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    support = np.asarray(support, dtype=np.int32)
    if points.shape[0] == 0:
        return points, normals, support
    if voxel > 0:
        keys = np.round(points / float(voxel)).astype(np.int64)
        _, unique_idx = np.unique(keys, axis=0, return_index=True)
        points = points[unique_idx]
        normals = normals[unique_idx]
        support = support[unique_idx]
    order = np.lexsort((np.arange(points.shape[0]), -support))
    if max_points > 0:
        order = order[: int(max_points)]
    points = points[order].astype(np.float32)
    normals = normals[order].astype(np.float32)
    normals = normals / np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8, None)
    return points, normals, support[order].astype(np.int32)


def build_extra_hairline_surfels(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    view_payloads: list[dict[str, Any]],
    max_points: int,
    max_per_view: int,
    min_support: int,
    top_frac: float,
    nearest_radius: float,
    depth_offset: float,
    voxel: float,
    output_dir: Path,
) -> dict[str, Any]:
    if int(max_points) <= 0:
        return {
            "enabled": False,
            "points": np.zeros((0, 3), dtype=np.float32),
            "normals": np.zeros((0, 3), dtype=np.float32),
            "support": np.zeros((0,), dtype=np.int32),
            "summary": {"candidate_points": 0, "kept_points": 0},
        }

    height, width = view_payloads[0]["mask"].shape
    head_center = np.asarray(vertices, dtype=np.float32)[np.asarray(vertices[:, 1]).argsort()[-max(16, vertices.shape[0] // 12) :]].mean(axis=0)
    candidates: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    per_view_rows: list[dict[str, Any]] = []
    for payload in view_payloads:
        depth_map, _, _, raster_mask, _ = rasterize_world_mesh(
            world_vertices=vertices,
            faces=faces,
            world_to_cam=payload["world_to_cam_np"],
            intrinsic=payload["intrinsic_np"],
            image_hw=payload["mask"].shape,
            silhouette_mask=None,
            fill_knn=0,
            return_raster_mask=True,
        )
        target = np.asarray(payload["mask"], dtype=bool)
        ys_mask, xs_mask = np.nonzero(target)
        if ys_mask.size == 0:
            per_view_rows.append({"view_index": payload["view_index"], "candidate_pixels": 0, "kept_pixels": 0})
            continue
        top = int(ys_mask.min())
        bottom = int(ys_mask.max())
        top_limit = int(round(top + np.clip(float(top_frac), 0.05, 0.80) * max(1, bottom - top + 1)))
        yy = np.arange(height)[:, None]
        head_top_band = yy <= top_limit
        missing = target & ~raster_mask & head_top_band
        source = raster_mask & target & (yy <= min(height - 1, top_limit + int(max(2.0, nearest_radius * 1.5))))
        my, mx = np.nonzero(missing)
        sy, sx = np.nonzero(source)
        if my.size == 0 or sy.size == 0:
            per_view_rows.append(
                {
                    "view_index": payload["view_index"],
                    "top_limit": top_limit,
                    "candidate_pixels": int(my.size),
                    "kept_pixels": 0,
                }
            )
            continue
        if my.size > int(max_per_view):
            pick = np.linspace(0, my.size - 1, int(max_per_view)).round().astype(np.int64)
            my = my[pick]
            mx = mx[pick]
        query_xy = np.stack([mx, my], axis=1).astype(np.float32)
        source_xy = np.stack([sx, sy], axis=1).astype(np.float32)
        source_depth = depth_map[sy, sx].astype(np.float32)
        keep, nearest_depth, nearest_distance = nearest_rendered_depths(
            query_xy,
            source_xy,
            source_depth,
            max_radius=float(nearest_radius),
        )
        valid_depth = np.isfinite(nearest_depth) & ((nearest_depth + float(depth_offset)) > 1e-5)
        keep &= valid_depth
        world = backproject_pixels_to_world(
            query_xy[keep],
            nearest_depth[keep] + float(depth_offset),
            payload["intrinsic_np"],
            payload["world_to_cam_np"],
        )
        if world.shape[0] > 0:
            candidates.append(world)
            distances.append(nearest_distance[keep].astype(np.float32))
        per_view_rows.append(
            {
                "view_index": payload["view_index"],
                "camera_id": payload["camera_id"],
                "top_limit": top_limit,
                "candidate_pixels": int(query_xy.shape[0]),
                "kept_pixels": int(world.shape[0]),
                "nearest_distance_p50": None if not keep.any() else float(np.percentile(nearest_distance[keep], 50)),
                "nearest_distance_p90": None if not keep.any() else float(np.percentile(nearest_distance[keep], 90)),
            }
        )

    if not candidates:
        return {
            "enabled": True,
            "points": np.zeros((0, 3), dtype=np.float32),
            "normals": np.zeros((0, 3), dtype=np.float32),
            "support": np.zeros((0,), dtype=np.int32),
            "summary": {
                "candidate_points": 0,
                "kept_points": 0,
                "min_support": int(min_support),
                "per_view": per_view_rows,
            },
        }
    points = np.concatenate(candidates, axis=0).astype(np.float32)
    support = mask_support_for_points(points, view_payloads, height, width)
    keep = support >= int(max(1, min_support))
    points = points[keep]
    support = support[keep]
    normals = points - head_center[None, :]
    normals = normals / np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8, None)
    points, normals, support = voxel_downsample_points(points, normals, support, float(voxel), int(max_points))
    ply_path = output_dir / "optimized_softsurfel_extra_hairline_points.ply"
    save_point_ply(ply_path, points, normals)
    summary = {
        "enabled": True,
        "candidate_points": int(sum(row.get("kept_pixels", 0) for row in per_view_rows)),
        "kept_points": int(points.shape[0]),
        "min_support": int(min_support),
        "support": summarize(support.astype(np.float32).tolist()),
        "top_frac": float(top_frac),
        "nearest_radius": float(nearest_radius),
        "depth_offset": float(depth_offset),
        "voxel": float(voxel),
        "ply_path": ply_path,
        "per_view": per_view_rows,
        "note": (
            "Extra hairline surfels are shared 3D diagnostic support built from raw mask missing pixels "
            "and SMPL-X depth anchors. They are not an external teacher and do not pass any gate by themselves."
        ),
    }
    return {"enabled": True, "points": points, "normals": normals, "support": support, "summary": summary}


def splat_points_into_maps(
    *,
    depth_map: np.ndarray,
    point_map: np.ndarray,
    mask_map: np.ndarray,
    normal_map: np.ndarray,
    points: np.ndarray,
    normals_world: np.ndarray,
    world_to_cam: np.ndarray,
    intrinsic: np.ndarray,
    radius: int,
) -> dict[str, int]:
    points = np.asarray(points, dtype=np.float32)
    normals_world = np.asarray(normals_world, dtype=np.float32)
    if points.shape[0] == 0:
        return {"candidate_points": 0, "projected_points": 0, "updated_pixels": 0}
    world_to_cam = homogeneous(world_to_cam)
    rotation = world_to_cam[:3, :3].astype(np.float32)
    translation = world_to_cam[:3, 3].astype(np.float32)
    cam = points @ rotation.T + translation[None, :]
    z = cam[:, 2]
    uvw = cam @ np.asarray(intrinsic, dtype=np.float32).T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-8, None)
    height, width = depth_map.shape
    px = np.rint(uv[:, 0]).astype(np.int32)
    py = np.rint(uv[:, 1]).astype(np.int32)
    valid = (
        np.isfinite(uv).all(axis=1)
        & np.isfinite(z)
        & (z > 1e-5)
        & (px >= -int(radius))
        & (px < width + int(radius))
        & (py >= -int(radius))
        & (py < height + int(radius))
    )
    cam_normals = normals_world @ rotation.T
    cam_normals = cam_normals / np.clip(np.linalg.norm(cam_normals, axis=1, keepdims=True), 1e-8, None)
    updated = 0
    projected = int(valid.sum())
    rad = max(0, int(radius))
    for idx in np.nonzero(valid)[0].tolist():
        for dy in range(-rad, rad + 1):
            yy = int(py[idx] + dy)
            if yy < 0 or yy >= height:
                continue
            for dx in range(-rad, rad + 1):
                xx = int(px[idx] + dx)
                if xx < 0 or xx >= width:
                    continue
                if rad > 0 and dx * dx + dy * dy > rad * rad:
                    continue
                if (not mask_map[yy, xx]) or float(z[idx]) < float(depth_map[yy, xx]):
                    depth_map[yy, xx] = float(z[idx])
                    point_map[yy, xx] = points[idx]
                    normal_map[yy, xx] = cam_normals[idx]
                    mask_map[yy, xx] = True
                    updated += 1
    return {"candidate_points": int(points.shape[0]), "projected_points": projected, "updated_pixels": int(updated)}


def export_raster_targets(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_normals: np.ndarray,
    extra_points: np.ndarray | None = None,
    extra_normals: np.ndarray | None = None,
    extra_splat_radius: int = 1,
    scene_dir: Path,
    dataset_root: Path,
    subset_name: str,
    target_size: int,
    view_spec: str,
    output_dir: Path,
) -> dict[str, Any]:
    export_dir = output_dir / "rasterized_surface_targets"
    image_dir = export_dir / "debug_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    manifest = recover_legacy_crop_source_sizes(scene_dir, load_scene_manifest(scene_dir))
    camera_params, camera_source = resolve_scene_camera_params(manifest, dataset_root, subset_name)
    views = list(manifest["exported_views"])
    view_indices = parse_view_spec(view_spec, len(views))

    depths = []
    world_points = []
    normal_maps = []
    masks = []
    intrinsics = []
    extrinsics = []
    camera_ids = []
    rows = []
    extra_points_np = np.zeros((0, 3), dtype=np.float32) if extra_points is None else np.asarray(extra_points, dtype=np.float32)
    extra_normals_np = (
        np.zeros_like(extra_points_np, dtype=np.float32)
        if extra_normals is None
        else np.asarray(extra_normals, dtype=np.float32)
    )
    if extra_normals_np.shape != extra_points_np.shape:
        extra_normals_np = np.zeros_like(extra_points_np, dtype=np.float32)
    extra_rows = []
    for view_idx in view_indices:
        view = views[view_idx]
        camera_id = str(view["camera_id"]).zfill(2)
        intrinsic_np = align_intrinsics_for_loaded_scene_view(
            np.asarray(camera_params[camera_id]["intrinsic"], dtype=np.float32),
            view,
            target_size=int(target_size),
        )
        world_to_cam_np = homogeneous(np.asarray(camera_params[camera_id]["world_to_cam"], dtype=np.float32))
        cam_normals = vertex_normals @ world_to_cam_np[:3, :3].T
        cam_normals = cam_normals / np.clip(np.linalg.norm(cam_normals, axis=1, keepdims=True), 1e-8, None)
        depth_map, point_map, mask_map, normal_map, raster_mask, meta = rasterize_world_mesh(
            world_vertices=vertices,
            faces=faces,
            world_to_cam=world_to_cam_np,
            intrinsic=intrinsic_np,
            image_hw=(int(target_size), int(target_size)),
            silhouette_mask=None,
            fill_knn=0,
            vertex_features=cam_normals.astype(np.float32),
            return_vertex_features=True,
            return_raster_mask=True,
        )
        depth_map = depth_map.astype(np.float32)
        point_map = point_map.astype(np.float32)
        normal_map = normal_map.astype(np.float32)
        mask_map = raster_mask.astype(bool)
        extra_meta = splat_points_into_maps(
            depth_map=depth_map,
            point_map=point_map,
            mask_map=mask_map,
            normal_map=normal_map,
            points=extra_points_np,
            normals_world=extra_normals_np,
            world_to_cam=world_to_cam_np,
            intrinsic=intrinsic_np,
            radius=int(extra_splat_radius),
        )
        depths.append(depth_map)
        world_points.append(point_map)
        normal_maps.append(normal_map)
        masks.append(mask_map)
        intrinsics.append(intrinsic_np.astype(np.float32))
        extrinsics.append(world_to_cam_np.astype(np.float32))
        camera_ids.append(camera_id)
        rows.append(
            {
                "view_index": int(view_idx),
                "camera_id": camera_id,
                "rasterized_pixels": int(mask_map.sum()),
                "extra_hairline_splat": extra_meta,
                "meta": meta,
            }
        )
        extra_rows.append({"view_index": int(view_idx), "camera_id": camera_id, **extra_meta})
        prefix = image_dir / f"view_{view_idx:02d}_cam{camera_id}"
        Image.fromarray(mask_map.astype(np.uint8) * 255).save(prefix.with_name(prefix.name + "_mask.png"))
        save_depth_image(prefix.with_name(prefix.name + "_depth.png"), depth_map, mask_map)
        save_normal_image(prefix.with_name(prefix.name + "_normal.png"), normal_map, mask_map)

    depths_np = np.stack(depths, axis=0).astype(np.float32)
    worlds_np = np.stack(world_points, axis=0).astype(np.float32)
    normals_np = np.stack(normal_maps, axis=0).astype(np.float32)
    masks_np = np.stack(masks, axis=0).astype(bool)
    intrinsics_np = np.stack(intrinsics, axis=0).astype(np.float32)
    extrinsics_np = np.stack(extrinsics, axis=0).astype(np.float32)
    npz_path = export_dir / "rasterized_surface_targets.npz"
    np.savez_compressed(
        npz_path,
        depths=depths_np,
        depth=depths_np[..., None],
        world_points=worlds_np,
        normals=normals_np,
        teacher_mask=masks_np,
        intrinsic=intrinsics_np,
        extrinsic=extrinsics_np,
        camera_ids=np.asarray(camera_ids),
    )
    summary = {
        "npz_path": npz_path,
        "scene_dir": scene_dir,
        "camera_source": camera_source,
        "target_size": int(target_size),
        "view_indices": view_indices,
        "camera_ids": camera_ids,
        "extra_hairline": {
            "points": int(extra_points_np.shape[0]),
            "splat_radius": int(extra_splat_radius),
            "rows": extra_rows,
        },
        "rows": rows,
        "debug_image_dir": image_dir,
        "note": (
            "These are hard-rasterized debug targets from the raw-image optimized mesh. "
            "They are not a strict-passing teacher unless the external teacher gate and "
            "explicit Open3D visual review pass."
        ),
    }
    (export_dir / "rasterized_surface_targets_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def evaluate_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    view_payloads: list[dict[str, Any]],
    output_dir: Path,
    overlay_limit: int,
) -> tuple[list[dict[str, Any]], list[Path]]:
    rows: list[dict[str, Any]] = []
    overlay_paths: list[Path] = []
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for payload in view_payloads:
        rendered = rasterize_world_mesh(
            world_vertices=vertices,
            faces=faces,
            world_to_cam=payload["world_to_cam_np"],
            intrinsic=payload["intrinsic_np"],
            image_hw=payload["mask"].shape,
            silhouette_mask=None,
            fill_knn=0,
            return_raster_mask=True,
        )[3]
        metrics = compute_mask_metrics(rendered, payload["mask"])
        rows.append({"view_index": payload["view_index"], "camera_id": payload["camera_id"], "metrics": metrics})
        if len(overlay_paths) < overlay_limit:
            overlay_path = overlay_dir / f"view_{payload['view_index']:02d}_cam{payload['camera_id']}_overlay.png"
            Image.fromarray(overlay_masks(payload["rgb"], payload["mask"], rendered)).save(overlay_path)
            overlay_paths.append(overlay_path)
    return rows, overlay_paths


def save_soft_render_debug(
    vertices_t: torch.Tensor,
    faces_t: torch.Tensor,
    face_indices_t: torch.Tensor,
    barycentric_t: torch.Tensor,
    view_payloads: list[dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
) -> list[Path]:
    debug_dir = output_dir / "soft_render_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with torch.no_grad():
        surfels, normals = compute_surfels(vertices_t, faces_t, face_indices_t, barycentric_t)
        for payload in view_payloads[: max(1, int(args.overlay_limit))]:
            render = render_soft_surfel_maps(
                surfels=surfels,
                normals=normals,
                world_to_cam=payload["world_to_cam"],
                intrinsic=payload["intrinsic"],
                height=int(args.target_size),
                width=int(args.target_size),
                sigma=float(args.gaussian_sigma),
                pixel_chunk=int(args.render_pixel_chunk),
            )
            mask_np = render["mask"].detach().cpu().numpy()
            hard_mask = mask_np > 0.30
            prefix = debug_dir / f"view_{payload['view_index']:02d}_cam{payload['camera_id']}"
            mask_path = prefix.with_name(prefix.name + "_soft_mask.png")
            depth_path = prefix.with_name(prefix.name + "_depth.png")
            normal_path = prefix.with_name(prefix.name + "_normal.png")
            overlay_path = prefix.with_name(prefix.name + "_soft_overlay.png")
            Image.fromarray((np.clip(mask_np, 0, 1) * 255.0).astype(np.uint8)).save(mask_path)
            save_depth_image(depth_path, render["depth"].detach().cpu().numpy(), hard_mask)
            save_normal_image(normal_path, render["normal"].detach().cpu().numpy(), hard_mask)
            Image.fromarray(overlay_masks(payload["rgb"], payload["mask"], hard_mask)).save(overlay_path)
            paths.append(overlay_path)
    return paths


def main() -> int:
    args = parse_args()
    scene_dir = args.scene_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} already exists and is not empty. Use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = recover_legacy_crop_source_sizes(scene_dir, load_scene_manifest(scene_dir))
    dataset_root = Path(args.dataset_root or manifest["dataset_root"]).expanduser()
    smplx_model_dir = resolve_smplx_model_dir(None if args.smplx_model_dir is None else str(args.smplx_model_dir))
    if smplx_model_dir is None:
        raise FileNotFoundError("Could not resolve SMPL-X model dir; pass --smplx-model-dir.")
    model_path = resolve_smplx_model_path(smplx_model_dir, args.smplx_gender)
    smplx_params, _ = load_optional_annotation_payload(manifest, dataset_root, args.subset_name)
    if not smplx_params:
        raise ValueError("Scene annotations do not provide SMPL-X parameters.")

    mesh = forward_smplx_mesh(
        model_path=model_path,
        betas=smplx_params["betas"],
        expression=smplx_params.get("expression"),
        fullpose=smplx_params["fullpose"],
        transl=smplx_params.get("transl"),
        scale=smplx_params.get("scale", 1.0),
    )
    static_features = build_smplx_vertex_features(
        model_path=model_path,
        betas=smplx_params["betas"],
        expression=smplx_params.get("expression"),
    )

    base_vertices_np = np.asarray(mesh["vertices"], dtype=np.float32)
    faces_np = np.asarray(mesh["faces"], dtype=np.int32)
    normals_np = compute_vertex_normals(base_vertices_np, faces_np).astype(np.float32)
    vertex_parts_np = classify_vertex_parts(np.asarray(static_features["canonical_positions"], dtype=np.float32))
    part_limits_np, part_reg_weights_np = make_part_limits(vertex_parts_np, args)
    edges_np = unique_edges(faces_np)

    surfel_plan = sample_surface_plan(
        base_vertices=base_vertices_np,
        faces=faces_np,
        vertex_parts=vertex_parts_np,
        sample_count=int(args.surfel_samples),
        seed=int(args.seed),
    )
    sdf_sample_count = min(int(args.surface_samples_for_sdf), base_vertices_np.shape[0])
    sdf_indices_np = np.linspace(0, base_vertices_np.shape[0] - 1, sdf_sample_count).round().astype(np.int64)

    camera_params, camera_source = resolve_scene_camera_params(manifest, dataset_root, args.subset_name)
    views = list(manifest["exported_views"])
    selected_indices = list(range(0, len(views), max(1, int(args.view_stride))))[: max(1, int(args.max_views))]

    requested_device = str(args.device).strip().lower()
    if requested_device != "cpu" and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    height = width = int(args.target_size)
    view_payloads: list[dict[str, Any]] = []
    for view_idx in selected_indices:
        view = views[view_idx]
        camera_id = str(view["camera_id"]).zfill(2)
        scene = load_scene_view(scene_dir, view_idx, (height, width))
        rgb_np = normalize_rgb(scene.rgb)
        mask_np = np.asarray(scene.mask, dtype=bool)
        intrinsic_np = align_intrinsics_for_loaded_scene_view(
            np.asarray(camera_params[camera_id]["intrinsic"], dtype=np.float32),
            view,
            target_size=height,
        )
        world_to_cam_np = homogeneous(np.asarray(camera_params[camera_id]["world_to_cam"], dtype=np.float32))
        boundary_np = boundary_points(mask_np, args.boundary_samples)
        view_payloads.append(
            {
                "view_index": int(view_idx),
                "camera_id": camera_id,
                "rgb": rgb_np,
                "mask": mask_np,
                "rgb_t": torch.from_numpy(rgb_np).permute(2, 0, 1)[None].to(device=device),
                "mask_t": torch.from_numpy(mask_np.astype(np.float32))[None, None].to(device=device),
                "sdf": torch.from_numpy(mask_sdf(mask_np))[None, None].to(device=device),
                "boundary": torch.from_numpy(boundary_np).to(device=device),
                "intrinsic": torch.from_numpy(intrinsic_np).to(device=device),
                "world_to_cam": torch.from_numpy(world_to_cam_np).to(device=device),
                "intrinsic_np": intrinsic_np,
                "world_to_cam_np": world_to_cam_np,
            }
        )

    base_vertices = torch.from_numpy(base_vertices_np).to(device=device)
    base_normals = torch.from_numpy(normals_np).to(device=device)
    faces_t = torch.from_numpy(faces_np.astype(np.int64)).to(device=device)
    face_indices_t = torch.from_numpy(surfel_plan["face_indices"]).to(device=device)
    barycentric_t = torch.from_numpy(surfel_plan["barycentric"]).to(device=device)
    sdf_indices_t = torch.from_numpy(sdf_indices_np).to(device=device)
    part_limits_t = torch.from_numpy(part_limits_np).to(device=device)
    part_reg_weights_t = torch.from_numpy(part_reg_weights_np).to(device=device)
    edges_t = torch.from_numpy(edges_np).to(device=device)
    center = torch.from_numpy(base_vertices_np.mean(axis=0, keepdims=True).astype(np.float32)).to(device=device)

    delta_t = torch.zeros(3, device=device, requires_grad=True)
    log_scale = torch.zeros(1, device=device, requires_grad=True)
    normal_offsets = torch.zeros(base_vertices_np.shape[0], device=device, requires_grad=True)
    optimizer = torch.optim.Adam([delta_t, log_scale, normal_offsets], lr=float(args.lr))

    history: list[dict[str, Any]] = []
    for step in range(int(args.steps)):
        optimizer.zero_grad(set_to_none=True)
        bounded_offsets = torch.tanh(normal_offsets) * part_limits_t
        vertices = center + torch.exp(log_scale).clamp(0.85, 1.15) * (base_vertices - center) + delta_t[None, :]
        vertices = vertices + base_normals * bounded_offsets[:, None]
        surfels, surfel_normals = compute_surfels(vertices, faces_t, face_indices_t, barycentric_t)

        mask_losses = []
        recall_losses = []
        outside_losses = []
        boundary_losses = []
        visibility_depths = []
        sampled_vertices = vertices.index_select(0, sdf_indices_t)
        for payload in view_payloads:
            render = render_soft_surfel_maps(
                surfels=surfels,
                normals=surfel_normals,
                world_to_cam=payload["world_to_cam"],
                intrinsic=payload["intrinsic"],
                height=height,
                width=width,
                sigma=float(args.gaussian_sigma),
                pixel_chunk=int(args.render_pixel_chunk),
            )
            visibility_depths.append(render["depth"])
            rendered_mask = render["mask"].clamp(1e-4, 1.0 - 1e-4)
            target_mask = payload["mask_t"].reshape(height, width)
            mask_losses.append(F.binary_cross_entropy(rendered_mask, target_mask))
            target_area = target_mask.sum().clamp_min(1.0)
            recall_losses.append((target_mask * (1.0 - rendered_mask)).sum() / target_area)

            uv, z, _ = project_points(sampled_vertices, payload["world_to_cam"], payload["intrinsic"])
            sdf_values = sample_sdf(payload["sdf"], uv, height, width)
            in_front = z > 1e-5
            outside_losses.append(torch.relu(sdf_values[in_front]).mean() if in_front.any() else sdf_values.mean() * 0.0)

            boundary = payload["boundary"]
            if boundary.numel() > 0:
                uv_surfel, z_surfel, _ = project_points(surfels, payload["world_to_cam"], payload["intrinsic"])
                valid = (
                    (z_surfel > 1e-5)
                    & (uv_surfel[:, 0] >= 0.0)
                    & (uv_surfel[:, 0] <= width - 1)
                    & (uv_surfel[:, 1] >= 0.0)
                    & (uv_surfel[:, 1] <= height - 1)
                )
                uv_valid = uv_surfel[valid]
                if uv_valid.shape[0] > 0:
                    uv_norm = uv_valid / float(max(height, width))
                    boundary_norm = boundary / float(max(height, width))
                    dists = torch.cdist(boundary_norm, uv_norm)
                    boundary_losses.append(dists.min(dim=1).values.mean())

        mask_loss = torch.stack(mask_losses).mean() if mask_losses else torch.zeros((), device=device)
        recall_loss = torch.stack(recall_losses).mean() if recall_losses else torch.zeros((), device=device)
        outside_loss = torch.stack(outside_losses).mean() if outside_losses else torch.zeros((), device=device)
        boundary_loss = torch.stack(boundary_losses).mean() if boundary_losses else torch.zeros((), device=device)
        photo_loss, photo_meta = photometric_consistency_loss(
            surfels,
            view_payloads,
            height,
            width,
            visibility_depths=visibility_depths,
            depth_tolerance=float(args.photo_depth_tolerance),
        )

        global_reg = float(args.translation_reg) * delta_t.square().sum() + float(args.scale_reg) * log_scale.square().sum()
        offset_values = bounded_offsets
        offset_reg = (part_reg_weights_t * offset_values.square()).mean()
        smooth_reg = (offset_values[edges_t[:, 0]] - offset_values[edges_t[:, 1]]).square().mean()
        loss = (
            float(args.mask_weight) * mask_loss
            + float(args.recall_weight) * recall_loss
            + float(args.outside_weight) * outside_loss
            + float(args.boundary_weight) * boundary_loss
            + float(args.photo_weight) * photo_loss
            + global_reg
            + float(args.offset_reg) * offset_reg
            + float(args.offset_smooth_reg) * smooth_reg
        )
        loss.backward()
        optimizer.step()

        if step == 0 or step == int(args.steps) - 1 or (step + 1) % max(1, int(args.steps) // 5) == 0:
            history.append(
                {
                    "step": int(step),
                    "loss": float(loss.detach().cpu()),
                    "mask_loss": float(mask_loss.detach().cpu()),
                    "soft_recall_loss": float(recall_loss.detach().cpu()),
                    "outside_loss": float(outside_loss.detach().cpu()),
                    "boundary_loss": float(boundary_loss.detach().cpu()),
                    "photometric_consistency_loss": float(photo_loss.detach().cpu()),
                    "offset_reg": float(offset_reg.detach().cpu()),
                    "offset_smooth_reg": float(smooth_reg.detach().cpu()),
                    "photo_valid_surfels": photo_meta["valid_surfels"],
                    "photo_mean_support": photo_meta["mean_support"],
                    "translation": [float(v) for v in delta_t.detach().cpu().numpy().reshape(-1)],
                    "scale": float(torch.exp(log_scale.detach()).cpu().item()),
                }
            )

    with torch.no_grad():
        final_offsets = torch.tanh(normal_offsets) * part_limits_t
        optimized = center + torch.exp(log_scale).clamp(0.85, 1.15) * (base_vertices - center) + delta_t[None, :]
        optimized = optimized + base_normals * final_offsets[:, None]
        optimized_np = optimized.detach().cpu().numpy().astype(np.float32)
        final_offsets_np = final_offsets.detach().cpu().numpy().astype(np.float32)

    initial_rows, initial_overlays = evaluate_mesh(base_vertices_np, faces_np, view_payloads, output_dir / "initial", args.overlay_limit)
    optimized_rows, optimized_overlays = evaluate_mesh(optimized_np, faces_np, view_payloads, output_dir / "optimized", args.overlay_limit)
    save_contact_sheet(initial_overlays, output_dir / "initial_overlay_contact_sheet.png")
    save_contact_sheet(optimized_overlays, output_dir / "optimized_overlay_contact_sheet.png")
    save_ply(output_dir / "optimized_softsurfel_surface_mesh.ply", optimized_np, faces_np)

    optimized_t = torch.from_numpy(optimized_np).to(device=device)
    soft_overlay_paths = save_soft_render_debug(
        vertices_t=optimized_t,
        faces_t=faces_t,
        face_indices_t=face_indices_t,
        barycentric_t=barycentric_t,
        view_payloads=view_payloads,
        args=args,
        output_dir=output_dir,
    )
    save_contact_sheet(soft_overlay_paths, output_dir / "soft_render_overlay_contact_sheet.png")

    extra_hairline = build_extra_hairline_surfels(
        vertices=optimized_np,
        faces=faces_np,
        view_payloads=view_payloads,
        max_points=int(args.extra_hairline_surfels),
        max_per_view=int(args.extra_hairline_max_per_view),
        min_support=int(args.extra_hairline_min_support),
        top_frac=float(args.extra_hairline_top_frac),
        nearest_radius=float(args.extra_hairline_nearest_radius),
        depth_offset=float(args.extra_hairline_depth_offset),
        voxel=float(args.extra_hairline_voxel),
        output_dir=output_dir,
    )
    extra_hairline_summary = extra_hairline["summary"]

    initial_iou = summarize([row["metrics"]["iou"] for row in initial_rows])
    optimized_iou = summarize([row["metrics"]["iou"] for row in optimized_rows])
    initial_recall = summarize([row["metrics"]["target_recall"] for row in initial_rows])
    optimized_recall = summarize([row["metrics"]["target_recall"] for row in optimized_rows])
    iou_delta = (
        float(optimized_iou["mean"] - initial_iou["mean"])
        if optimized_iou["mean"] is not None and initial_iou["mean"] is not None
        else None
    )
    recall_delta = (
        float(optimized_recall["mean"] - initial_recall["mean"])
        if optimized_recall["mean"] is not None and initial_recall["mean"] is not None
        else None
    )

    part_stats = {}
    for part_id, part_name in PART_NAMES.items():
        mask = vertex_parts_np == part_id
        values = final_offsets_np[mask]
        part_stats[part_name] = {
            "vertices": int(mask.sum()),
            "mean_abs_offset": float(np.mean(np.abs(values))) if values.size else 0.0,
            "p90_abs_offset": float(np.percentile(np.abs(values), 90)) if values.size else 0.0,
            "limit": float(np.max(part_limits_np[mask])) if mask.any() else 0.0,
        }

    truthful_status = "raw_softsurfel_surface_smoke_complete_not_teacher_or_candidate"
    export_summary = None
    if args.export_raster_targets:
        export_scene = args.export_scene_dir.expanduser().resolve() if args.export_scene_dir else scene_dir
        export_summary = export_raster_targets(
            vertices=optimized_np,
            faces=faces_np,
            vertex_normals=compute_vertex_normals(optimized_np, faces_np).astype(np.float32),
            extra_points=extra_hairline["points"],
            extra_normals=extra_hairline["normals"],
            extra_splat_radius=int(args.extra_hairline_export_radius),
            scene_dir=export_scene,
            dataset_root=dataset_root,
            subset_name=str(args.subset_name),
            target_size=int(args.export_target_size),
            view_spec=str(args.export_target_views),
            output_dir=output_dir,
        )

    summary = {
        "task": "raw_image_softsurfel_surface_upperbound_v1_smoke",
        "truthful_status": truthful_status,
        "scene_dir": scene_dir,
        "output_dir": output_dir,
        "uses_vggt_depth_point_normal": False,
        "creates_candidate_predictions": False,
        "creates_teacher_targets": bool(args.export_raster_targets),
        "teacher_targets_strict_pass": False,
        "allows_cloud": False,
        "scene": {
            "selected_view_count": len(selected_indices),
            "selected_indices": selected_indices,
            "camera_source": camera_source,
            "target_size": int(args.target_size),
        },
        "config": vars(args),
        "part_names": PART_NAMES,
        "part_stats": part_stats,
        "extra_hairline": extra_hairline_summary,
        "optimization_history": history,
        "metrics": {
            "initial_iou": initial_iou,
            "optimized_iou": optimized_iou,
            "iou_delta": iou_delta,
            "initial_target_recall": initial_recall,
            "optimized_target_recall": optimized_recall,
            "target_recall_delta": recall_delta,
        },
        "outputs": {
            "optimized_mesh": output_dir / "optimized_softsurfel_surface_mesh.ply",
            "initial_contact_sheet": output_dir / "initial_overlay_contact_sheet.png",
            "optimized_contact_sheet": output_dir / "optimized_overlay_contact_sheet.png",
            "soft_render_contact_sheet": output_dir / "soft_render_overlay_contact_sheet.png",
            "extra_hairline_points": output_dir / "optimized_softsurfel_extra_hairline_points.ply",
            "summary_json": output_dir / "raw_softsurfel_surface_summary.json",
            "report_md": output_dir / "report.md",
            "rasterized_surface_targets": None if export_summary is None else export_summary.get("npz_path"),
        },
        "rasterized_surface_targets": export_summary,
        "current_blocker": (
            "This is a CPU small-resolution soft surfel surface smoke. It adds differentiable "
            "soft mask rendering, multi-view RGB consistency, and part-aware residual limits, "
            "but it is not a full soft triangle renderer, not a strict-passing teacher, and "
            "not a mentor candidate. It must not unblock cloud."
        ),
        "next_required_action": (
            "Scale the renderer carefully, add true visibility/depth ordering and surface-to-view "
            "depth/world/normal target export, then run strict teacher/candidate visual gates. "
            "Do not return to r-candidate threshold/confidence loops."
        ),
    }
    (output_dir / "raw_softsurfel_surface_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report = [
        "# Raw-Image Soft Surfel Surface Upper-Bound v1 Smoke",
        "",
        f"Status: `{truthful_status}`",
        "",
        f"- selected views: `{len(selected_indices)}`",
        f"- target size: `{int(args.target_size)}`",
        f"- surfel samples: `{int(args.surfel_samples)}`",
        f"- uses VGGT depth/point/normal: `False`",
        f"- creates teacher targets: `{bool(args.export_raster_targets)}`",
        f"- creates candidate predictions: `False`",
        f"- extra hairline surfels: `{extra_hairline_summary.get('kept_points', 0)}`",
        f"- initial mean IoU: `{initial_iou['mean']}`",
        f"- optimized mean IoU: `{optimized_iou['mean']}`",
        f"- IoU delta: `{iou_delta}`",
        f"- initial target recall: `{initial_recall['mean']}`",
        f"- optimized target recall: `{optimized_recall['mean']}`",
        f"- target recall delta: `{recall_delta}`",
        "",
        "This is the first raw-image v1 soft-surface smoke: it adds a pure-Torch CPU soft",
        "surfel renderer, multi-view photometric consistency, and part-aware residual limits.",
        "It is still not a full human surface backend, not a strict teacher, and not a cloud",
        "unblocker.",
        "",
        "Current blocker:",
        "",
        str(summary["current_blocker"]),
        "",
        "Next required action:",
        "",
        str(summary["next_required_action"]),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(json_ready({k: summary[k] for k in ("truthful_status", "metrics", "outputs")}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
