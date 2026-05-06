from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from preflight_differentiable_renderer_backend import (  # noqa: E402
    import_nvdiffrast,
    load_connected_mesh,
    load_view_rgb_mask,
    make_clip_positions,
    parse_view_indices,
    save_image,
)
from prepare_4k4d_prior_training_case import (  # noqa: E402
    load_scene_manifest,
    recover_legacy_crop_source_sizes,
    resolve_scene_camera_params,
)
from preflight_differentiable_renderer_backend import align_intrinsics_for_loaded_scene_view  # noqa: E402
from tools.smplx_numpy import compute_vertex_normals  # noqa: E402


PART_LIMITS = {
    0: 0.002,
    1: 0.006,
    2: 0.006,
    3: 0.004,
    4: 0.012,
    5: 0.010,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Raw-image connected-surface optimization v3 smoke using nvdiffrast full-mesh rendering. "
            "This is not a teacher, not a candidate, and not a cloud-unblock signal."
        )
    )
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--template-payload", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--target-size", type=int, default=96)
    parser.add_argument("--view-indices", default="0,10,20,30,40,50")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--offset-mode",
        choices=("vertex", "mlp"),
        default="vertex",
        help=(
            "vertex optimizes bounded per-vertex residuals. mlp optimizes a tiny shared residual decoder "
            "over canonical mesh features, a first learned-surface-backend smoke."
        ),
    )
    parser.add_argument("--mlp-hidden", type=int, default=64)
    parser.add_argument("--mask-bce-weight", type=float, default=1.0)
    parser.add_argument("--target-recall-weight", type=float, default=0.5)
    parser.add_argument("--overfill-weight", type=float, default=0.35)
    parser.add_argument("--offset-reg-weight", type=float, default=0.10)
    parser.add_argument("--edge-reg-weight", type=float, default=0.05)
    parser.add_argument(
        "--photometric-variance-weight",
        type=float,
        default=0.0,
        help=(
            "Optional raw-image multi-view color consistency on the optimized vertices. "
            "This is the first non-mask raw-image signal in the nvdiffrast route."
        ),
    )
    parser.add_argument("--photometric-mask-threshold", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_colored_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    colors_u8 = np.clip(np.asarray(colors, dtype=np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {vertices.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {faces.shape[0]}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for vertex, color in zip(vertices, colors_u8, strict=False):
            f.write(
                f"{float(vertex[0]):.7f} {float(vertex[1]):.7f} {float(vertex[2]):.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for face in faces:
            f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def unique_edges(faces: np.ndarray) -> np.ndarray:
    faces_np = np.asarray(faces, dtype=np.int64)
    edges = np.concatenate([faces_np[:, [0, 1]], faces_np[:, [1, 2]], faces_np[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0).astype(np.int64)


def part_offset_limits(part_ids: np.ndarray) -> np.ndarray:
    limits = np.zeros((part_ids.shape[0],), dtype=np.float32)
    for part_id, value in PART_LIMITS.items():
        limits[np.asarray(part_ids) == int(part_id)] = float(value)
    limits[limits <= 0] = 0.002
    return limits


class SurfaceOffsetMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(features)) * limits


def build_vertex_features(vertices: np.ndarray, normals: np.ndarray, part_ids: np.ndarray) -> np.ndarray:
    vertices_np = np.asarray(vertices, dtype=np.float32)
    normals_np = np.asarray(normals, dtype=np.float32)
    centered = vertices_np - vertices_np.mean(axis=0, keepdims=True)
    scale = float(np.percentile(np.linalg.norm(centered, axis=1), 95))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    geom = centered / scale
    one_hot = np.zeros((vertices_np.shape[0], len(PART_LIMITS)), dtype=np.float32)
    clipped_parts = np.clip(np.asarray(part_ids, dtype=np.int64), 0, len(PART_LIMITS) - 1)
    one_hot[np.arange(vertices_np.shape[0]), clipped_parts] = 1.0
    return np.concatenate([geom, normals_np, one_hot], axis=1).astype(np.float32)


def render_mask_depth(
    dr: Any,
    ctx: Any,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    world_to_cam: torch.Tensor,
    intrinsic: torch.Tensor,
    height: int,
    width: int,
) -> dict[str, torch.Tensor]:
    clip, cam = make_clip_positions(vertices, world_to_cam, intrinsic, height, width, z_sign=1.0)
    rast, _ = dr.rasterize(ctx, clip, faces, resolution=[height, width])
    ones = torch.ones((1, vertices.shape[0], 1), dtype=vertices.dtype, device=vertices.device)
    mask_interp, _ = dr.interpolate(ones, rast, faces)
    mask_aa = dr.antialias(mask_interp, rast, clip, faces).clamp(0.0, 1.0)[0, ..., 0]
    depth_attr = cam[:, 2:3].unsqueeze(0).contiguous()
    depth, _ = dr.interpolate(depth_attr, rast, faces)
    visible = (rast[0, ..., 3] > 0).to(vertices.dtype)
    return {"mask": mask_aa, "hard_visible": visible, "depth": depth[0, ..., 0], "rast": rast}


def project_vertices_to_grid(
    vertices: torch.Tensor,
    world_to_cam: torch.Tensor,
    intrinsic: torch.Tensor,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ones = torch.ones((vertices.shape[0], 1), dtype=vertices.dtype, device=vertices.device)
    hom = torch.cat([vertices, ones], dim=1)
    cam = (world_to_cam @ hom.T).T[:, :3]
    z = cam[:, 2]
    uvw = (intrinsic @ cam.T).T
    uv = uvw[:, :2] / uvw[:, 2:3].clamp_min(1e-8)
    x = (uv[:, 0] / max(width - 1, 1)) * 2.0 - 1.0
    y = (uv[:, 1] / max(height - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([x, y], dim=1)
    inside = (z > 1e-6) & (x >= -1.0) & (x <= 1.0) & (y >= -1.0) & (y <= 1.0)
    return grid, inside


def multiview_photometric_variance_loss(
    vertices: torch.Tensor,
    view_payloads: list[dict[str, Any]],
    height: int,
    width: int,
    mask_threshold: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    colors: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for payload in view_payloads:
        grid, inside = project_vertices_to_grid(
            vertices,
            payload["world_to_cam"],
            payload["intrinsic"],
            height,
            width,
        )
        sample_grid = grid.view(1, -1, 1, 2)
        rgb = F.grid_sample(
            payload["rgb_tensor"],
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0, :, :, 0].T
        mask = F.grid_sample(
            payload["mask_tensor"],
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0, 0, :, 0]
        weight = inside.to(vertices.dtype) * (mask > float(mask_threshold)).to(vertices.dtype)
        colors.append(rgb)
        weights.append(weight)
    color_stack = torch.stack(colors, dim=0)
    weight_stack = torch.stack(weights, dim=0)
    support = weight_stack.sum(dim=0)
    valid = support >= 2.0
    if not valid.any():
        zero = vertices.sum() * 0.0
        return zero, {"valid_vertices": 0, "mean_support": 0.0}
    weighted = weight_stack[:, :, None]
    mean = (color_stack * weighted).sum(dim=0) / support.clamp_min(1.0)[:, None]
    variance = (((color_stack - mean[None, :, :]) ** 2) * weighted).sum(dim=(0, 2)) / support.clamp_min(1.0)
    loss = variance[valid].mean()
    return loss, {
        "valid_vertices": int(valid.detach().sum().cpu()),
        "mean_support": float(support[valid].detach().mean().cpu()),
    }


def mask_metrics(mask: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    pred = np.asarray(mask) >= float(threshold)
    tgt = np.asarray(target).astype(bool)
    inter = int((pred & tgt).sum())
    union = int((pred | tgt).sum())
    return {
        "pred_pixels": int(pred.sum()),
        "target_pixels": int(tgt.sum()),
        "iou": float(inter / union) if union else 0.0,
        "target_recall": float(inter / max(1, int(tgt.sum()))),
        "overfill_ratio": float(((pred & ~tgt).sum()) / max(1, int(pred.sum()))),
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Raw Surface nvdiffrast v3 Smoke",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This is a local backend-integration smoke. It is not a teacher, not a candidate, and not a cloud unblocker.",
        "",
        "## Gate Truth",
        "",
        "```text",
        "strict_candidate_passes = 0",
        "strict_teacher_passes = 0",
        "cloud = blocked",
        "```",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary["summary"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Loss Curve",
        "",
        "```json",
        json.dumps(summary["loss_curve"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Decision",
        "",
        "```text",
        summary["decision"],
        "```",
        "",
        "## Outputs",
        "",
    ]
    for output in summary["outputs"]:
        lines.append(f"- `{output}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    dr, import_error = import_nvdiffrast()
    if dr is None:
        raise RuntimeError(f"nvdiffrast unavailable: {import_error}")
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        raise RuntimeError("CUDA is unavailable for nvdiffrast optimization")
    device = torch.device("cuda")
    ctx = dr.RasterizeCudaContext(device=device)

    scene_dir = args.scene_dir.resolve()
    manifest = recover_legacy_crop_source_sizes(scene_dir, load_scene_manifest(scene_dir))
    views = manifest["exported_views"]
    view_indices = parse_view_indices(args.view_indices, len(views))
    dataset_root = args.dataset_root or Path(str(manifest.get("dataset_root", "")))
    cameras, camera_source = resolve_scene_camera_params(manifest, dataset_root, args.subset_name)

    mesh = load_connected_mesh(args.template_payload)
    base_vertices_np = mesh["vertices"].astype(np.float32)
    faces_np = mesh["faces"].astype(np.int32)
    part_ids = mesh["part_ids"].astype(np.int64)
    colors_np = mesh["part_colors"].astype(np.float32)
    limits_np = part_offset_limits(part_ids)
    edges_np = unique_edges(faces_np)

    faces_t = torch.as_tensor(faces_np, dtype=torch.int32, device=device).contiguous()
    base_vertices = torch.as_tensor(base_vertices_np, dtype=torch.float32, device=device).contiguous()
    limits = torch.as_tensor(limits_np, dtype=torch.float32, device=device).view(-1, 1)
    edges = torch.as_tensor(edges_np, dtype=torch.long, device=device).contiguous()
    base_normals_np = compute_vertex_normals(base_vertices_np, faces_np)
    vertex_features_np = build_vertex_features(base_vertices_np, base_normals_np, part_ids)
    vertex_features = torch.as_tensor(vertex_features_np, dtype=torch.float32, device=device).contiguous()
    if args.offset_mode == "mlp":
        offset_model = SurfaceOffsetMLP(vertex_features.shape[1], int(args.mlp_hidden)).to(device)
        raw_delta = None
        optimizer = torch.optim.Adam(offset_model.parameters(), lr=float(args.lr))
    else:
        offset_model = None
        raw_delta = torch.zeros_like(base_vertices, requires_grad=True)
        optimizer = torch.optim.Adam([raw_delta], lr=float(args.lr))

    view_payloads: list[dict[str, Any]] = []
    height = width = int(args.target_size)
    for view_index in view_indices:
        view = views[view_index]
        camera_id = str(view["camera_id"])
        params = cameras[camera_id]
        intrinsic_np = align_intrinsics_for_loaded_scene_view(np.asarray(params["intrinsic"], dtype=np.float32), view, height)
        world_to_cam_np = np.asarray(params["world_to_cam"], dtype=np.float32)
        rgb_np, target_mask_np = load_view_rgb_mask(view, height)
        rgb_tensor = (
            torch.as_tensor(rgb_np.astype(np.float32) / 255.0, dtype=torch.float32, device=device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .contiguous()
        )
        mask_tensor = (
            torch.as_tensor(target_mask_np.astype(np.float32), dtype=torch.float32, device=device)
            .view(1, 1, height, width)
            .contiguous()
        )
        view_payloads.append(
            {
                "view_index": int(view_index),
                "camera_id": camera_id,
                "target_mask": torch.as_tensor(target_mask_np.astype(np.float32), dtype=torch.float32, device=device),
                "target_mask_np": target_mask_np,
                "rgb_tensor": rgb_tensor,
                "mask_tensor": mask_tensor,
                "world_to_cam": torch.as_tensor(world_to_cam_np, dtype=torch.float32, device=device).contiguous(),
                "intrinsic": torch.as_tensor(intrinsic_np, dtype=torch.float32, device=device).contiguous(),
            }
        )

    loss_curve: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    for step in range(int(args.steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        if offset_model is not None:
            delta = offset_model(vertex_features, limits)
        else:
            assert raw_delta is not None
            delta = torch.tanh(raw_delta) * limits
        vertices = base_vertices + delta
        mask_bce_total = torch.zeros((), dtype=torch.float32, device=device)
        recall_total = torch.zeros((), dtype=torch.float32, device=device)
        overfill_total = torch.zeros((), dtype=torch.float32, device=device)
        metrics_rows: list[dict[str, Any]] = []
        for payload in view_payloads:
            render = render_mask_depth(
                dr,
                ctx,
                vertices,
                faces_t,
                payload["world_to_cam"],
                payload["intrinsic"],
                height,
                width,
            )
            mask = render["mask"].clamp(1e-4, 1.0 - 1e-4)
            target = payload["target_mask"]
            mask_bce_total = mask_bce_total + F.binary_cross_entropy(mask, target)
            recall_total = recall_total + (target * (1.0 - mask)).sum() / target.sum().clamp_min(1.0)
            overfill_total = overfill_total + ((1.0 - target) * mask).sum() / mask.sum().clamp_min(1.0)
            if step == 0 or step == int(args.steps):
                metrics_rows.append(
                    {
                        "view_index": payload["view_index"],
                        "camera_id": payload["camera_id"],
                        **mask_metrics(mask.detach().cpu().numpy(), payload["target_mask_np"]),
                    }
                )
        mask_bce = mask_bce_total / max(1, len(view_payloads))
        recall_loss = recall_total / max(1, len(view_payloads))
        overfill_loss = overfill_total / max(1, len(view_payloads))
        offset_reg = (delta / limits.clamp_min(1e-6)).pow(2).mean()
        edge_reg = (delta[edges[:, 0]] - delta[edges[:, 1]]).pow(2).mean() / (limits.mean().clamp_min(1e-6) ** 2)
        if float(args.photometric_variance_weight) > 0.0:
            photo_loss, photo_meta = multiview_photometric_variance_loss(
                vertices,
                view_payloads,
                height,
                width,
                float(args.photometric_mask_threshold),
            )
        else:
            photo_loss = torch.zeros((), dtype=torch.float32, device=device)
            photo_meta = {"valid_vertices": 0, "mean_support": 0.0}
        loss = (
            float(args.mask_bce_weight) * mask_bce
            + float(args.target_recall_weight) * recall_loss
            + float(args.overfill_weight) * overfill_loss
            + float(args.offset_reg_weight) * offset_reg
            + float(args.edge_reg_weight) * edge_reg
            + float(args.photometric_variance_weight) * photo_loss
        )
        if step < int(args.steps):
            loss.backward()
            optimizer.step()
        loss_curve.append(
            {
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "mask_bce": float(mask_bce.detach().cpu()),
                "target_recall_loss": float(recall_loss.detach().cpu()),
                "overfill_loss": float(overfill_loss.detach().cpu()),
                "offset_reg": float(offset_reg.detach().cpu()),
                "edge_reg": float(edge_reg.detach().cpu()),
                "photometric_variance": float(photo_loss.detach().cpu()),
                "photometric_meta": photo_meta,
                "metrics": metrics_rows,
            }
        )

    elapsed = float(time.perf_counter() - start_time)
    with torch.no_grad():
        if offset_model is not None:
            delta = offset_model(vertex_features, limits)
        else:
            assert raw_delta is not None
            delta = torch.tanh(raw_delta) * limits
        optimized_vertices_np = (base_vertices + delta).detach().cpu().numpy().astype(np.float32)
        delta_np = delta.detach().cpu().numpy().astype(np.float32)
    normals_np = compute_vertex_normals(optimized_vertices_np, faces_np)
    normal_colors = (normals_np + 1.0) * 0.5
    mesh_path = output_dir / "optimized_nvdiffrast_surface_mesh.ply"
    carrier_path = output_dir / "carrier_initial_mesh.ply"
    write_colored_ply(mesh_path, optimized_vertices_np, faces_np, colors_np)
    write_colored_ply(carrier_path, base_vertices_np, faces_np, colors_np)
    write_colored_ply(output_dir / "optimized_nvdiffrast_surface_normals.ply", optimized_vertices_np, faces_np, normal_colors)

    final_metrics = loss_curve[-1]["metrics"]
    initial_metrics = loss_curve[0]["metrics"]
    avg_initial_iou = float(np.mean([row["iou"] for row in initial_metrics])) if initial_metrics else 0.0
    avg_final_iou = float(np.mean([row["iou"] for row in final_metrics])) if final_metrics else 0.0
    max_delta = float(np.linalg.norm(delta_np, axis=1).max()) if delta_np.size else 0.0
    mean_delta = float(np.linalg.norm(delta_np, axis=1).mean()) if delta_np.size else 0.0

    for payload in view_payloads:
        with torch.no_grad():
            render = render_mask_depth(
                dr,
                ctx,
                torch.as_tensor(optimized_vertices_np, dtype=torch.float32, device=device),
                faces_t,
                payload["world_to_cam"],
                payload["intrinsic"],
                height,
                width,
            )
        prefix = output_dir / f"view_{payload['view_index']:02d}_cam{payload['camera_id']}"
        save_image(prefix.with_name(prefix.name + "_target_mask.png"), payload["target_mask_np"].astype(np.float32))
        save_image(prefix.with_name(prefix.name + "_render_mask.png"), render["mask"].detach().cpu().numpy())
        save_image(prefix.with_name(prefix.name + "_depth.png"), render["depth"].detach().cpu().numpy() * (render["hard_visible"].detach().cpu().numpy() > 0))

    status = "local_smoke_complete"
    decision = (
        "nvdiffrast full-mesh optimization loop runs locally. This is still only a backend integration smoke; "
        "Open3D strict visual review is required before any teacher export or training."
    )
    summary = {
        "status": status,
        "summary": {
            "scene_dir": str(scene_dir),
            "template_payload": str(args.template_payload.resolve()),
            "camera_source": camera_source,
            "views": view_indices,
            "target_size": int(args.target_size),
            "steps": int(args.steps),
            "offset_mode": str(args.offset_mode),
            "mlp_hidden": int(args.mlp_hidden) if args.offset_mode == "mlp" else None,
            "photometric_variance_weight": float(args.photometric_variance_weight),
            "elapsed_seconds": elapsed,
            "avg_initial_iou": avg_initial_iou,
            "avg_final_iou": avg_final_iou,
            "avg_iou_delta": avg_final_iou - avg_initial_iou,
            "max_vertex_delta": max_delta,
            "mean_vertex_delta": mean_delta,
            "strict_candidate_passes": 0,
            "strict_teacher_passes": 0,
            "cloud": "blocked",
        },
        "loss_curve": loss_curve,
        "decision": decision,
        "outputs": [
            str(mesh_path),
            str(carrier_path),
            str(output_dir / "optimized_nvdiffrast_surface_normals.ply"),
            str(output_dir / "raw_surface_nvdiffrast_summary.json"),
            str(output_dir / "raw_surface_nvdiffrast_summary.md"),
        ],
    }
    (output_dir / "raw_surface_nvdiffrast_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output_dir / "raw_surface_nvdiffrast_summary.md", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
