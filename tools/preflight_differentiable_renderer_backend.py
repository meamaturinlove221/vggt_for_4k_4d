from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

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
from tools.smplx_numpy import compute_vertex_normals, rasterize_world_mesh  # noqa: E402


PART_COLORS = np.asarray(
    [
        [0.72, 0.72, 0.72],
        [0.20, 0.60, 1.00],
        [1.00, 0.45, 0.25],
        [0.95, 0.75, 0.25],
        [0.70, 0.35, 0.95],
        [0.25, 0.85, 0.45],
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight a real differentiable renderer backend for the VGGT human-surface route. "
            "This is backend environment validation only: it does not create a candidate, a teacher, "
            "or a cloud-unblock signal."
        )
    )
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--template-payload", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--target-size", type=int, default=96)
    parser.add_argument("--view-indices", default="0,10,20,30,40,50")
    parser.add_argument("--backend", choices=("auto", "nvdiffrast"), default="auto")
    parser.add_argument("--max-hard-raster-faces", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def align_intrinsics_for_loaded_scene_view(intrinsic: np.ndarray, view: dict[str, Any], target_size: int) -> np.ndarray:
    """Align crop-scene intrinsics without importing the old soft-surface optimizer.

    Importing that optimizer currently mutates CUDA device discovery on this
    Windows workstation, so the backend preflight keeps this tiny compatibility
    helper local.
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


def import_nvdiffrast() -> tuple[Any | None, str | None]:
    try:
        import nvdiffrast.torch as dr  # type: ignore

        return dr, None
    except Exception as exc:  # pragma: no cover - environment report path
        return None, repr(exc)


def load_connected_mesh(payload_path: Path) -> dict[str, np.ndarray]:
    payload_path = payload_path.expanduser().resolve()
    if not payload_path.exists():
        raise FileNotFoundError(payload_path)
    with np.load(payload_path, allow_pickle=False) as payload:
        vertices = np.asarray(payload["hybrid_vertices"] if "hybrid_vertices" in payload.files else payload["vertices"], dtype=np.float32)
        faces = np.asarray(payload["hybrid_faces"] if "hybrid_faces" in payload.files else payload["faces"], dtype=np.int32)
        if "part_ids" in payload.files and payload["part_ids"].shape[0] == vertices.shape[0]:
            part_ids = np.asarray(payload["part_ids"], dtype=np.int64)
        else:
            part_ids = np.zeros((vertices.shape[0],), dtype=np.int64)
    normals = compute_vertex_normals(vertices, faces).astype(np.float32)
    colors = PART_COLORS[np.clip(part_ids, 0, PART_COLORS.shape[0] - 1)]
    return {"vertices": vertices, "faces": faces, "normals": normals, "part_ids": part_ids, "part_colors": colors}


def normalize_depth(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros(depth.shape, dtype=np.float32)
    valid = mask & np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return out
    lo, hi = np.percentile(depth[valid], [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = float(depth[valid].max())
        lo = float(depth[valid].min())
    if hi <= lo:
        out[valid] = 1.0
    else:
        out[valid] = np.clip((depth[valid] - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    return out


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
        Image.fromarray(array, mode="L").save(path)
    else:
        array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
        Image.fromarray(array, mode="RGB").save(path)


def load_view_rgb_mask(view: dict[str, Any], target_size: int) -> tuple[np.ndarray, np.ndarray]:
    image_path = Path(str(view["image_path"]))
    mask_path = Path(str(view["mask_path"]))
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if image.size != (target_size, target_size):
        image = image.resize((target_size, target_size), Image.Resampling.BICUBIC)
    if mask.size != (target_size, target_size):
        mask = mask.resize((target_size, target_size), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8), np.asarray(mask, dtype=np.uint8) > 127


def sample_vertex_rgb(vertices: np.ndarray, rgb: np.ndarray, world_to_cam: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    rotation = np.asarray(world_to_cam[:3, :3], dtype=np.float32)
    translation = np.asarray(world_to_cam[:3, 3], dtype=np.float32)
    cam = vertices @ rotation.T + translation[None, :]
    z = cam[:, 2]
    uvw = (intrinsic @ cam.T).T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-8, None)
    xi = np.rint(uv[:, 0]).astype(np.int64)
    yi = np.rint(uv[:, 1]).astype(np.int64)
    inside = np.isfinite(uv).all(axis=1) & (z > 1e-6) & (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    out = np.zeros((vertices.shape[0], 3), dtype=np.float32)
    if np.any(inside):
        out[inside] = rgb[yi[inside], xi[inside]].astype(np.float32) / 255.0
    return out


def make_clip_positions(
    vertices: torch.Tensor,
    world_to_cam: torch.Tensor,
    intrinsic: torch.Tensor,
    height: int,
    width: int,
    *,
    z_sign: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ones = torch.ones((vertices.shape[0], 1), dtype=vertices.dtype, device=vertices.device)
    hom = torch.cat([vertices, ones], dim=1)
    cam = (world_to_cam @ hom.T).T[:, :3]
    z = cam[:, 2].clamp_min(1e-6)
    uvw = (intrinsic @ cam.T).T
    uv = uvw[:, :2] / uvw[:, 2:3].clamp_min(1e-8)
    x_ndc = ((uv[:, 0] + 0.5) / max(width, 1)) * 2.0 - 1.0
    # nvdiffrast's tensor output origin matches the top-left image rows in this
    # path, so keep the camera-space pixel y direction instead of applying an
    # OpenGL-style vertical flip.
    y_ndc = ((uv[:, 1] + 0.5) / max(height, 1)) * 2.0 - 1.0
    z_valid = z[torch.isfinite(z)]
    if z_valid.numel() > 0:
        z_near = z_valid.min().detach()
        z_far = z_valid.max().detach()
        z_span = (z_far - z_near).clamp_min(1e-6)
        z_ndc = ((z - z_near) / z_span) * 2.0 - 1.0
    else:
        z_ndc = torch.zeros_like(z)
    clip_z = z_ndc * float(z_sign)
    clip = torch.stack([x_ndc, y_ndc, clip_z, torch.ones_like(z)], dim=1)
    return clip.unsqueeze(0).contiguous(), cam


def render_nvdiffrast_view(
    dr: Any,
    ctx: Any,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    normals_world: torch.Tensor,
    colors: torch.Tensor,
    world_to_cam: torch.Tensor,
    intrinsic: torch.Tensor,
    height: int,
    width: int,
    *,
    z_sign: float,
) -> dict[str, torch.Tensor]:
    clip, cam = make_clip_positions(vertices, world_to_cam, intrinsic, height, width, z_sign=z_sign)
    rast, _ = dr.rasterize(ctx, clip, faces, resolution=[height, width])
    depth_attr = cam[:, 2:3].unsqueeze(0).contiguous()
    world_attr = vertices.unsqueeze(0).contiguous()
    normal_attr = normals_world.unsqueeze(0).contiguous()
    color_attr = colors.unsqueeze(0).contiguous()
    depth, _ = dr.interpolate(depth_attr, rast, faces)
    world, _ = dr.interpolate(world_attr, rast, faces)
    normal, _ = dr.interpolate(normal_attr, rast, faces)
    color, _ = dr.interpolate(color_attr, rast, faces)
    mask = (rast[..., 3] > 0).to(vertices.dtype)
    normal = torch.nn.functional.normalize(normal, dim=-1, eps=1e-6)
    return {
        "mask": mask[0],
        "depth": depth[0, ..., 0],
        "world": world[0],
        "normal": normal[0],
        "color": color[0].clamp(0.0, 1.0),
        "visibility": rast[0, ..., 3],
    }


def hard_reference(
    vertices: np.ndarray,
    faces: np.ndarray,
    world_to_cam: np.ndarray,
    intrinsic: np.ndarray,
    height: int,
    width: int,
    max_faces: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    ref_faces = faces
    if max_faces > 0 and max_faces < faces.shape[0]:
        idx = np.linspace(0, faces.shape[0] - 1, int(max_faces)).round().astype(np.int64)
        ref_faces = faces[idx]
    depth, _points, _completed, raster_mask, meta = rasterize_world_mesh(
        vertices,
        ref_faces,
        world_to_cam,
        intrinsic,
        (height, width),
        silhouette_mask=None,
        fill_knn=0,
        return_raster_mask=True,
    )
    return depth.astype(np.float32), raster_mask.astype(bool), meta


def compare_masks_and_depth(
    backend_mask: np.ndarray,
    backend_depth: np.ndarray,
    hard_mask: np.ndarray,
    hard_depth: np.ndarray,
) -> dict[str, Any]:
    backend = np.asarray(backend_mask, dtype=bool)
    hard = np.asarray(hard_mask, dtype=bool)
    inter = int((backend & hard).sum())
    union = int((backend | hard).sum())
    common = backend & hard & np.isfinite(backend_depth) & np.isfinite(hard_depth) & (backend_depth > 0) & (hard_depth > 0)
    residual = np.abs(backend_depth[common] - hard_depth[common]) if np.any(common) else np.zeros((0,), dtype=np.float32)
    return {
        "backend_pixels": int(backend.sum()),
        "hard_pixels": int(hard.sum()),
        "mask_iou": float(inter / union) if union else 0.0,
        "common_pixels": int(common.sum()),
        "median_abs_depth_residual": float(np.median(residual)) if residual.size else None,
        "p90_abs_depth_residual": float(np.percentile(residual, 90)) if residual.size else None,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Differentiable Renderer Backend Preflight",
        "",
        "Status: `{}`".format(summary["status"]),
        "",
        "This report validates renderer backend capability only. It is not a mentor pass, not a teacher, not a candidate, and not a cloud unblocker.",
        "",
        "## Gate Truth",
        "",
        "```text",
        "strict_candidate_passes = 0",
        "strict_teacher_passes = 0",
        "cloud = blocked",
        "```",
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(summary["environment"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Backend Result",
        "",
        "```json",
        json.dumps(summary["backend"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## View Metrics",
        "",
    ]
    for row in summary.get("views", []):
        lines.extend(
            [
                f"### View `{row['view_index']}` camera `{row['camera_id']}`",
                "",
                "```json",
                json.dumps(row, indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            "",
            "```text",
            summary["decision"],
            "```",
            "",
            "## Outputs",
            "",
        ]
    )
    for item in summary.get("outputs", []):
        lines.append(f"- `{item}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def describe_cuda_device() -> dict[str, Any]:
    available = bool(torch.cuda.is_available())
    count = 0
    name = None
    error = None
    try:
        count = int(torch.cuda.device_count())
        if available and count > 0:
            name = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - environment report path
        error = repr(exc)
    return {"available": available, "device_count": count, "device_0": name, "error": error}


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace/report into it")
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_dir = args.scene_dir.resolve()
    manifest = recover_legacy_crop_source_sizes(scene_dir, load_scene_manifest(scene_dir))
    views = manifest.get("exported_views", [])
    view_indices = parse_view_indices(args.view_indices, len(views))
    dataset_root = args.dataset_root or Path(str(manifest.get("dataset_root", "")))
    cameras, camera_source = resolve_scene_camera_params(manifest, dataset_root, args.subset_name)

    mesh = load_connected_mesh(args.template_payload)
    vertices_np = mesh["vertices"]
    faces_np = mesh["faces"]
    normals_np = mesh["normals"]
    part_colors_np = mesh["part_colors"]
    height = width = int(args.target_size)

    cuda_info = describe_cuda_device()
    environment: dict[str, Any] = {
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda": cuda_info,
        "camera_source": camera_source,
        "scene_dir": str(scene_dir),
        "template_payload": str(args.template_payload.resolve()),
        "vertices": int(vertices_np.shape[0]),
        "faces": int(faces_np.shape[0]),
        "target_size": int(args.target_size),
        "view_indices": view_indices,
    }

    dr, import_error = import_nvdiffrast()
    environment["nvdiffrast_import_error"] = import_error
    if dr is None:
        summary = {
            "status": "blocked_no_backend",
            "environment": environment,
            "backend": {"name": "nvdiffrast", "available": False, "error": import_error},
            "views": [],
            "decision": "No production differentiable mesh backend is importable. Use Linux/WSL2/Docker/lab Linux or install nvdiffrast/PyTorch3D/Kaolin before surface optimization.",
            "outputs": [],
        }
        (output_dir / "renderer_backend_preflight_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        write_report(output_dir / "renderer_backend_preflight_summary.md", summary)
        return 2

    if not cuda_info["available"] or int(cuda_info["device_count"]) <= 0:
        raise RuntimeError("nvdiffrast CUDA backend is available but torch.cuda.is_available() is false")

    device = torch.device("cuda")
    ctx = dr.RasterizeCudaContext(device=device)
    faces_t = torch.as_tensor(faces_np.astype(np.int32), dtype=torch.int32, device=device).contiguous()
    normals_t = torch.as_tensor(normals_np, dtype=torch.float32, device=device).contiguous()
    part_colors_t = torch.as_tensor(part_colors_np, dtype=torch.float32, device=device).contiguous()

    view_rows: list[dict[str, Any]] = []
    output_paths: list[str] = []
    total_loss: torch.Tensor | None = None
    vertices_grad_t = torch.as_tensor(vertices_np, dtype=torch.float32, device=device).contiguous().requires_grad_(True)

    start_time = time.perf_counter()
    for view_index in view_indices:
        view = views[view_index]
        camera_id = str(view["camera_id"])
        params = cameras[camera_id]
        intrinsic_np = align_intrinsics_for_loaded_scene_view(np.asarray(params["intrinsic"], dtype=np.float32), view, height)
        world_to_cam_np = np.asarray(params["world_to_cam"], dtype=np.float32)
        raw_rgb, _raw_mask = load_view_rgb_mask(view, height)
        sampled_rgb = sample_vertex_rgb(vertices_np, raw_rgb, world_to_cam_np, intrinsic_np)
        vertex_rgb_np = np.where(sampled_rgb.sum(axis=1, keepdims=True) > 0, sampled_rgb, part_colors_np)

        world_to_cam_t = torch.as_tensor(world_to_cam_np, dtype=torch.float32, device=device).contiguous()
        intrinsic_t = torch.as_tensor(intrinsic_np, dtype=torch.float32, device=device).contiguous()
        vertex_rgb_t = torch.as_tensor(vertex_rgb_np, dtype=torch.float32, device=device).contiguous()

        hard_depth, hard_mask, hard_meta = hard_reference(
            vertices_np,
            faces_np,
            world_to_cam_np,
            intrinsic_np,
            height,
            width,
            int(args.max_hard_raster_faces),
        )

        candidate_renders: list[tuple[float, dict[str, torch.Tensor], dict[str, Any]]] = []
        for z_sign in (1.0, -1.0):
            render = render_nvdiffrast_view(
                dr,
                ctx,
                vertices_grad_t,
                faces_t,
                normals_t,
                vertex_rgb_t,
                world_to_cam_t,
                intrinsic_t,
                height,
                width,
                z_sign=z_sign,
            )
            mask_np = (render["mask"].detach().cpu().numpy() > 0.5)
            depth_np = render["depth"].detach().cpu().numpy().astype(np.float32)
            metrics = compare_masks_and_depth(mask_np, depth_np, hard_mask, hard_depth)
            candidate_renders.append((z_sign, render, metrics))
        best_z_sign, best_render, best_metrics = max(
            candidate_renders,
            key=lambda item: (
                item[2]["mask_iou"],
                -float(item[2]["median_abs_depth_residual"] if item[2]["median_abs_depth_residual"] is not None else 1e9),
            ),
        )

        visible = best_render["mask"] > 0.5
        if visible.any():
            loss = best_render["depth"][visible].mean() * 1e-4 + best_render["color"][visible].mean() * 1e-5
            total_loss = loss if total_loss is None else total_loss + loss

        prefix = output_dir / f"view_{view_index:02d}_cam{camera_id}"
        mask_np = best_render["mask"].detach().cpu().numpy().astype(np.float32)
        depth_np = best_render["depth"].detach().cpu().numpy().astype(np.float32)
        normal_np = best_render["normal"].detach().cpu().numpy().astype(np.float32)
        color_np = best_render["color"].detach().cpu().numpy().astype(np.float32)
        visibility_np = (best_render["visibility"].detach().cpu().numpy() > 0).astype(np.float32)
        save_image(prefix.with_name(prefix.name + "_mask.png"), mask_np)
        save_image(prefix.with_name(prefix.name + "_hard_mask.png"), hard_mask.astype(np.float32))
        save_image(prefix.with_name(prefix.name + "_depth.png"), normalize_depth(depth_np, mask_np > 0.5))
        save_image(prefix.with_name(prefix.name + "_hard_depth.png"), normalize_depth(hard_depth, hard_mask))
        save_image(prefix.with_name(prefix.name + "_normal.png"), (normal_np + 1.0) * 0.5)
        save_image(prefix.with_name(prefix.name + "_rgb.png"), color_np)
        save_image(prefix.with_name(prefix.name + "_visibility.png"), visibility_np)
        for suffix in ("mask", "hard_mask", "depth", "hard_depth", "normal", "rgb", "visibility"):
            output_paths.append(str(prefix.with_name(prefix.name + f"_{suffix}.png")))

        view_rows.append(
            {
                "view_index": int(view_index),
                "camera_id": camera_id,
                "z_sign": float(best_z_sign),
                "hard_raster_meta": hard_meta,
                **best_metrics,
            }
        )

    backward: dict[str, Any] = {"ran": False}
    if total_loss is not None:
        total_loss.backward()
        grad = vertices_grad_t.grad
        backward = {
            "ran": True,
            "loss": float(total_loss.detach().cpu()),
            "grad_finite": bool(torch.isfinite(grad).all().item()) if grad is not None else False,
            "grad_nonzero_vertices": int((grad.norm(dim=1) > 0).sum().item()) if grad is not None else 0,
            "grad_max_norm": float(grad.norm(dim=1).max().detach().cpu()) if grad is not None else 0.0,
        }

    elapsed = float(time.perf_counter() - start_time)
    min_iou = min((float(row["mask_iou"]) for row in view_rows), default=0.0)
    max_median_depth = max(
        (
            float(row["median_abs_depth_residual"])
            for row in view_rows
            if row.get("median_abs_depth_residual") is not None
        ),
        default=float("inf"),
    )
    max_p90_depth = max(
        (
            float(row["p90_abs_depth_residual"])
            for row in view_rows
            if row.get("p90_abs_depth_residual") is not None
        ),
        default=float("inf"),
    )
    all_common = all(int(row["common_pixels"]) > 0 for row in view_rows)
    backward_ok = bool(backward.get("ran") and backward.get("grad_finite") and backward.get("grad_nonzero_vertices", 0) > 0)
    thresholds = {
        "min_mask_iou_vs_hard_zbuffer": 0.88,
        "max_median_abs_depth_residual": 0.02,
        "max_p90_abs_depth_residual": 0.05,
        "min_views": 6,
    }
    status = (
        "backend_preflight_pass"
        if min_iou >= thresholds["min_mask_iou_vs_hard_zbuffer"]
        and max_median_depth <= thresholds["max_median_abs_depth_residual"]
        and max_p90_depth <= thresholds["max_p90_abs_depth_residual"]
        and all_common
        and backward_ok
        and len(view_rows) >= thresholds["min_views"]
        else "backend_preflight_fail"
    )
    decision = (
        "nvdiffrast renders the full connected mesh, aligns with the existing NumPy z-buffer, and supports a stable backward pass. "
        "This only unblocks renderer-backend development; strict mentor gates remain 0/0."
        if status == "backend_preflight_pass"
        else "Renderer backend is installed but did not satisfy the full-mesh alignment/backward preflight. Do not resume surface optimization until this is fixed or moved to Linux/WSL2/Docker/lab Linux."
    )

    summary = {
        "status": status,
        "environment": environment,
        "backend": {
            "name": "nvdiffrast",
            "available": True,
            "elapsed_seconds": elapsed,
            "backward": backward,
            "min_mask_iou_vs_hard_zbuffer": min_iou,
            "max_median_abs_depth_residual": max_median_depth,
            "max_p90_abs_depth_residual": max_p90_depth,
            "thresholds": thresholds,
            "views_tested": len(view_rows),
        },
        "views": view_rows,
        "decision": decision,
        "outputs": output_paths,
    }
    (output_dir / "renderer_backend_preflight_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output_dir / "renderer_backend_preflight_summary.md", summary)
    return 0 if status == "backend_preflight_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
