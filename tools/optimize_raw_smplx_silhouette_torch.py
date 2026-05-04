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
    compute_vertex_normals,
    forward_smplx_mesh,
    rasterize_world_mesh,
    resolve_smplx_model_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal torch differentiable raw-image SMPL-X silhouette fitting smoke. "
            "This uses raw masks/cameras and projected SMPL-X vertices, not VGGT depth/point/normal."
        )
    )
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--smplx-model-dir", type=Path)
    parser.add_argument("--smplx-gender", choices=("neutral", "female", "male"), default="neutral")
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument("--max-views", type=int, default=12)
    parser.add_argument("--view-stride", type=int, default=1)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--boundary-samples", type=int, default=256)
    parser.add_argument("--surface-samples", type=int, default=2500)
    parser.add_argument("--outside-weight", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, default=0.15)
    parser.add_argument("--translation-reg", type=float, default=0.05)
    parser.add_argument("--scale-reg", type=float, default=0.05)
    parser.add_argument("--normal-offset-reg", type=float, default=0.5)
    parser.add_argument("--optimize-normal-offsets", action="store_true")
    parser.add_argument("--normal-offset-limit", type=float, default=0.035)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overlay-limit", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
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


def project_vertices(vertices: torch.Tensor, world_to_cam: torch.Tensor, intrinsic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rotation = world_to_cam[:3, :3]
    translation = world_to_cam[:3, 3]
    cam = vertices @ rotation.T + translation[None, :]
    z = cam[:, 2].clamp_min(1e-6)
    uvw = cam @ intrinsic.T
    uv = uvw[:, :2] / uvw[:, 2:3].clamp_min(1e-6)
    return uv, z


def sample_sdf(sdf: torch.Tensor, uv: torch.Tensor, height: int, width: int) -> torch.Tensor:
    x = uv[:, 0] / float(max(1, width - 1)) * 2.0 - 1.0
    y = uv[:, 1] / float(max(1, height - 1)) * 2.0 - 1.0
    grid = torch.stack([x, y], dim=-1).view(1, 1, -1, 2)
    sampled = F.grid_sample(sdf, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled.view(-1)


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
        row = {
            "view_index": payload["view_index"],
            "camera_id": payload["camera_id"],
            "metrics": metrics,
        }
        rows.append(row)
        if len(overlay_paths) < overlay_limit:
            overlay_path = overlay_dir / f"view_{payload['view_index']:02d}_cam{payload['camera_id']}_optimized_overlay.png"
            Image.fromarray(overlay_masks(payload["rgb"], payload["mask"], rendered)).save(overlay_path)
            overlay_paths.append(overlay_path)
    return rows, overlay_paths


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
    base_vertices_np = np.asarray(mesh["vertices"], dtype=np.float32)
    faces_np = np.asarray(mesh["faces"], dtype=np.int32)
    normals_np = compute_vertex_normals(base_vertices_np, faces_np).astype(np.float32)
    center_np = base_vertices_np.mean(axis=0, keepdims=True).astype(np.float32)

    camera_params, camera_source = resolve_scene_camera_params(manifest, dataset_root, args.subset_name)
    views = list(manifest["exported_views"])
    selected_indices = list(range(0, len(views), max(1, int(args.view_stride))))[: max(1, int(args.max_views))]

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    view_payloads: list[dict[str, Any]] = []
    height = width = int(args.target_size)
    for view_idx in selected_indices:
        view = views[view_idx]
        camera_id = str(view["camera_id"]).zfill(2)
        scene = load_scene_view(scene_dir, view_idx, (height, width))
        mask = np.asarray(scene.mask, dtype=bool)
        intrinsic_np = align_intrinsics_for_scene_view(
            np.asarray(camera_params[camera_id]["intrinsic"], dtype=np.float32),
            view,
            target_size=height,
        )
        world_to_cam_np = homogeneous(np.asarray(camera_params[camera_id]["world_to_cam"], dtype=np.float32))
        boundary_np = boundary_points(mask, args.boundary_samples)
        view_payloads.append(
            {
                "view_index": int(view_idx),
                "camera_id": camera_id,
                "rgb": scene.rgb,
                "mask": mask,
                "sdf": torch.from_numpy(mask_sdf(mask))[None, None].to(device=device),
                "boundary": torch.from_numpy(boundary_np).to(device=device),
                "intrinsic": torch.from_numpy(intrinsic_np).to(device=device),
                "world_to_cam": torch.from_numpy(world_to_cam_np).to(device=device),
                "intrinsic_np": intrinsic_np,
                "world_to_cam_np": world_to_cam_np,
            }
        )

    vertex_count = base_vertices_np.shape[0]
    sample_count = min(int(args.surface_samples), vertex_count)
    sample_indices_np = np.linspace(0, vertex_count - 1, sample_count).round().astype(np.int64)
    sample_indices = torch.from_numpy(sample_indices_np).to(device=device)
    base_vertices = torch.from_numpy(base_vertices_np).to(device=device)
    normals = torch.from_numpy(normals_np).to(device=device)
    center = torch.from_numpy(center_np).to(device=device)
    delta_t = torch.zeros(3, device=device, requires_grad=True)
    log_scale = torch.zeros(1, device=device, requires_grad=True)
    params: list[torch.Tensor] = [delta_t, log_scale]
    normal_offsets = None
    if args.optimize_normal_offsets:
        normal_offsets = torch.zeros(vertex_count, device=device, requires_grad=True)
        params.append(normal_offsets)

    optimizer = torch.optim.Adam(params, lr=float(args.lr))
    history: list[dict[str, Any]] = []
    for step in range(int(args.steps)):
        optimizer.zero_grad(set_to_none=True)
        vertices = center + torch.exp(log_scale).clamp(0.85, 1.15) * (base_vertices - center) + delta_t[None, :]
        if normal_offsets is not None:
            bounded = torch.tanh(normal_offsets) * float(args.normal_offset_limit)
            vertices = vertices + normals * bounded[:, None]
        sampled_vertices = vertices.index_select(0, sample_indices)

        outside_losses = []
        boundary_losses = []
        for payload in view_payloads:
            uv, z = project_vertices(sampled_vertices, payload["world_to_cam"], payload["intrinsic"])
            sdf_values = sample_sdf(payload["sdf"], uv, height, width)
            in_front = z > 1e-5
            outside_losses.append(torch.relu(sdf_values[in_front]).mean() if in_front.any() else sdf_values.mean() * 0.0)

            boundary = payload["boundary"]
            if boundary.numel() > 0 and in_front.any():
                uv_valid = uv[in_front]
                in_image = (
                    (uv_valid[:, 0] >= 0)
                    & (uv_valid[:, 0] <= width - 1)
                    & (uv_valid[:, 1] >= 0)
                    & (uv_valid[:, 1] <= height - 1)
                )
                uv_valid = uv_valid[in_image]
                if uv_valid.shape[0] > 0:
                    uv_norm = uv_valid / float(max(height, width))
                    boundary_norm = boundary / float(max(height, width))
                    dists = torch.cdist(boundary_norm, uv_norm)
                    boundary_losses.append(dists.min(dim=1).values.mean())
        outside_loss = torch.stack(outside_losses).mean() if outside_losses else torch.zeros((), device=device)
        boundary_loss = torch.stack(boundary_losses).mean() if boundary_losses else torch.zeros((), device=device)
        reg = float(args.translation_reg) * delta_t.square().sum() + float(args.scale_reg) * log_scale.square().sum()
        if normal_offsets is not None:
            reg = reg + float(args.normal_offset_reg) * (torch.tanh(normal_offsets) * float(args.normal_offset_limit)).square().mean()
        loss = float(args.outside_weight) * outside_loss + float(args.boundary_weight) * boundary_loss + reg
        loss.backward()
        optimizer.step()
        if step == 0 or step == int(args.steps) - 1 or (step + 1) % max(1, int(args.steps) // 5) == 0:
            history.append(
                {
                    "step": int(step),
                    "loss": float(loss.detach().cpu()),
                    "outside_loss": float(outside_loss.detach().cpu()),
                    "boundary_loss": float(boundary_loss.detach().cpu()),
                    "translation": [float(v) for v in delta_t.detach().cpu().numpy().reshape(-1)],
                    "scale": float(torch.exp(log_scale.detach()).cpu().item()),
                }
            )

    with torch.no_grad():
        optimized = center + torch.exp(log_scale).clamp(0.85, 1.15) * (base_vertices - center) + delta_t[None, :]
        if normal_offsets is not None:
            optimized = optimized + normals * (torch.tanh(normal_offsets) * float(args.normal_offset_limit))[:, None]
        optimized_np = optimized.detach().cpu().numpy().astype(np.float32)

    initial_rows, initial_overlays = evaluate_mesh(base_vertices_np, faces_np, view_payloads, output_dir / "initial", args.overlay_limit)
    optimized_rows, optimized_overlays = evaluate_mesh(optimized_np, faces_np, view_payloads, output_dir / "optimized", args.overlay_limit)
    save_contact_sheet(initial_overlays, output_dir / "initial_overlay_contact_sheet.png")
    save_contact_sheet(optimized_overlays, output_dir / "optimized_overlay_contact_sheet.png")
    save_ply(output_dir / "optimized_smplx_silhouette_mesh.ply", optimized_np, faces_np)

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
    truthful_status = "raw_silhouette_differentiable_smoke_complete_not_surface_backend"
    summary = {
        "task": "raw_smplx_silhouette_torch_optimization_smoke",
        "truthful_status": truthful_status,
        "scene_dir": scene_dir,
        "output_dir": output_dir,
        "uses_vggt_depth_point_normal": False,
        "creates_candidate_predictions": False,
        "allows_cloud": False,
        "scene": {
            "selected_view_count": len(selected_indices),
            "selected_indices": selected_indices,
            "camera_source": camera_source,
        },
        "config": vars(args),
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
            "optimized_mesh": output_dir / "optimized_smplx_silhouette_mesh.ply",
            "initial_contact_sheet": output_dir / "initial_overlay_contact_sheet.png",
            "optimized_contact_sheet": output_dir / "optimized_overlay_contact_sheet.png",
            "summary_json": output_dir / "raw_smplx_silhouette_optimization_summary.json",
            "report_md": output_dir / "report.md",
        },
        "next_required_action": (
            "This proves a raw-mask differentiable optimization loop can run without "
            "VGGT observation recycling. It is still only silhouette fitting, not a "
            "learned surface backend or mentor candidate. Next add differentiable "
            "triangle/soft rasterization, photometric loss, normal/displacement residuals, "
            "and full strict teacher/candidate gates."
        ),
    }
    (output_dir / "raw_smplx_silhouette_optimization_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report = [
        "# Raw SMPL-X Silhouette Torch Optimization Smoke",
        "",
        f"Status: `{truthful_status}`",
        "",
        f"- selected views: `{len(selected_indices)}`",
        f"- uses VGGT depth/point/normal: `False`",
        f"- creates candidate predictions: `False`",
        f"- initial mean IoU: `{initial_iou['mean']}`",
        f"- optimized mean IoU: `{optimized_iou['mean']}`",
        f"- IoU delta: `{iou_delta}`",
        f"- initial target recall: `{initial_recall['mean']}`",
        f"- optimized target recall: `{optimized_recall['mean']}`",
        f"- target recall delta: `{recall_delta}`",
        "",
        "This is a non-wall Stage A smoke. It checks whether raw masks/cameras/SMPL-X can",
        "enter a differentiable optimization loop without recycling VGGT shell observations.",
        "It is not a mentor candidate and does not unblock cloud.",
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
