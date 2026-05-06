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

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from optimize_raw_surface_nvdiffrast import (  # noqa: E402
    build_vertex_features,
    compute_image_condition_features,
    mask_metrics,
    multiview_photometric_variance_loss,
    part_offset_limits,
    render_mask_depth,
    unique_edges,
    write_colored_ply,
)
from preflight_differentiable_renderer_backend import (  # noqa: E402
    align_intrinsics_for_loaded_scene_view,
    import_nvdiffrast,
    load_connected_mesh,
    load_view_rgb_mask,
    parse_view_indices,
    save_image,
)
from prepare_4k4d_prior_training_case import (  # noqa: E402
    load_scene_manifest,
    recover_legacy_crop_source_sizes,
    resolve_scene_camera_params,
)
from tools.smplx_numpy import compute_vertex_normals  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "B0 learned local surface-token backend smoke. This is research-only: "
            "it is not a teacher, not a candidate, and not a cloud-unblock signal."
        )
    )
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--template-payload", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--target-size", type=int, default=96)
    parser.add_argument("--view-indices", default="0,10,20,30,40,50")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--token-grid", type=int, default=5)
    parser.add_argument("--token-hidden", type=int, default=64)
    parser.add_argument("--mask-bce-weight", type=float, default=1.0)
    parser.add_argument("--target-recall-weight", type=float, default=0.5)
    parser.add_argument("--overfill-weight", type=float, default=0.35)
    parser.add_argument("--token-offset-reg-weight", type=float, default=0.08)
    parser.add_argument("--edge-reg-weight", type=float, default=0.05)
    parser.add_argument("--photometric-variance-weight", type=float, default=0.05)
    parser.add_argument("--normal-depth-proxy-weight", type=float, default=0.02)
    parser.add_argument("--photometric-mask-threshold", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def quantized_surface_tokens(vertices: np.ndarray, part_ids: np.ndarray, token_grid: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Build part-aware occupied spatial token ids for a connected surface.

    This is intentionally not a per-vertex MLP: vertices share a local token
    residual according to body part and coarse canonical position.
    """

    vertices = np.asarray(vertices, dtype=np.float32)
    part_ids = np.asarray(part_ids, dtype=np.int64)
    token_grid = max(2, int(token_grid))
    token_keys: list[tuple[int, int, int, int]] = []
    per_vertex_keys: list[tuple[int, int, int, int]] = []
    for part in sorted(int(v) for v in np.unique(part_ids)):
        idx = np.nonzero(part_ids == part)[0]
        pts = vertices[idx]
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        span = np.maximum(hi - lo, 1e-6)
        q = np.floor(((pts - lo[None, :]) / span[None, :]) * float(token_grid)).astype(np.int64)
        q = np.clip(q, 0, token_grid - 1)
        for row in q:
            key = (part, int(row[0]), int(row[1]), int(row[2]))
            token_keys.append(key)
            per_vertex_keys.append(key)
    unique_keys = sorted(set(token_keys))
    key_to_id = {key: i for i, key in enumerate(unique_keys)}
    token_ids = np.asarray([key_to_id[key] for key in per_vertex_keys], dtype=np.int64)
    # per_vertex_keys were generated part-by-part, so restore original order.
    restored = np.zeros_like(token_ids)
    cursor = 0
    for part in sorted(int(v) for v in np.unique(part_ids)):
        idx = np.nonzero(part_ids == part)[0]
        restored[idx] = token_ids[cursor : cursor + len(idx)]
        cursor += len(idx)
    token_part_ids = np.asarray([key[0] for key in unique_keys], dtype=np.int64)
    meta = {
        "token_grid": token_grid,
        "token_count": int(len(unique_keys)),
        "token_part_histogram": {str(int(p)): int((token_part_ids == p).sum()) for p in np.unique(token_part_ids)},
    }
    return restored, {"token_part_ids": token_part_ids, **meta}


def aggregate_token_features(vertex_features: np.ndarray, token_ids: np.ndarray, token_count: int) -> np.ndarray:
    sums = np.zeros((token_count, vertex_features.shape[1]), dtype=np.float32)
    counts = np.zeros((token_count, 1), dtype=np.float32)
    np.add.at(sums, token_ids, vertex_features.astype(np.float32))
    np.add.at(counts, token_ids, 1.0)
    return sums / np.maximum(counts, 1.0)


class PartSurfaceTokenDecoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, part_count: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 3),
                )
                for _ in range(part_count)
            ]
        )
        for head in self.heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(
        self,
        token_features: torch.Tensor,
        token_part_ids: torch.Tensor,
        vertex_token_ids: torch.Tensor,
        vertex_limits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_delta = torch.zeros((token_features.shape[0], 3), dtype=token_features.dtype, device=token_features.device)
        for part_id, head in enumerate(self.heads):
            mask = token_part_ids == int(part_id)
            if bool(mask.any()):
                token_delta[mask] = torch.tanh(head(token_features[mask]))
        vertex_delta = token_delta[vertex_token_ids] * vertex_limits
        return vertex_delta, token_delta


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Surface Token Backend B0 Smoke",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This is a research-only local smoke. It is not a teacher, not a candidate, and not a cloud unblocker.",
        "",
        "## Gate Truth",
        "",
        "```text",
        "strict_candidate_passes = 0",
        "strict_teacher_passes = 0",
        "cloud formal train/infer/export = blocked",
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
        raise RuntimeError("CUDA is unavailable for B0 surface-token smoke")
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

    height = width = int(args.target_size)
    view_payloads: list[dict[str, Any]] = []
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
        mask_tensor = torch.as_tensor(target_mask_np.astype(np.float32), dtype=torch.float32, device=device).view(1, 1, height, width)
        view_payloads.append(
            {
                "view_index": int(view_index),
                "camera_id": camera_id,
                "target_mask": torch.as_tensor(target_mask_np.astype(np.float32), dtype=torch.float32, device=device),
                "target_mask_np": target_mask_np,
                "rgb_tensor": rgb_tensor,
                "mask_tensor": mask_tensor.contiguous(),
                "world_to_cam": torch.as_tensor(world_to_cam_np, dtype=torch.float32, device=device).contiguous(),
                "intrinsic": torch.as_tensor(intrinsic_np, dtype=torch.float32, device=device).contiguous(),
            }
        )

    base_normals_np = compute_vertex_normals(base_vertices_np, faces_np)
    geom_features_np = build_vertex_features(base_vertices_np, base_normals_np, part_ids)
    image_features, image_meta = compute_image_condition_features(
        base_vertices,
        view_payloads,
        height,
        width,
        float(args.photometric_mask_threshold),
    )
    vertex_features_np = np.concatenate([geom_features_np, image_features.detach().cpu().numpy().astype(np.float32)], axis=1)
    token_ids_np, token_meta = quantized_surface_tokens(base_vertices_np, part_ids, int(args.token_grid))
    token_features_np = aggregate_token_features(vertex_features_np, token_ids_np, int(token_meta["token_count"]))
    token_part_ids_np = np.asarray(token_meta["token_part_ids"], dtype=np.int64)

    token_features = torch.as_tensor(token_features_np, dtype=torch.float32, device=device).contiguous()
    token_part_ids = torch.as_tensor(token_part_ids_np, dtype=torch.long, device=device).contiguous()
    vertex_token_ids = torch.as_tensor(token_ids_np, dtype=torch.long, device=device).contiguous()
    part_count = max(6, int(part_ids.max()) + 1)
    decoder = PartSurfaceTokenDecoder(token_features.shape[1], int(args.token_hidden), part_count).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=float(args.lr))

    loss_curve: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    for step in range(int(args.steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        delta, token_delta = decoder(token_features, token_part_ids, vertex_token_ids, limits)
        vertices = base_vertices + delta
        mask_bce_total = torch.zeros((), dtype=torch.float32, device=device)
        recall_total = torch.zeros((), dtype=torch.float32, device=device)
        overfill_total = torch.zeros((), dtype=torch.float32, device=device)
        depth_tv_total = torch.zeros((), dtype=torch.float32, device=device)
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
            depth = render["depth"]
            visible = render["hard_visible"]
            dx = (depth[:, 1:] - depth[:, :-1]).abs() * visible[:, 1:] * visible[:, :-1]
            dy = (depth[1:, :] - depth[:-1, :]).abs() * visible[1:, :] * visible[:-1, :]
            depth_tv_total = depth_tv_total + dx.mean() + dy.mean()
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
        offset_reg = (token_delta.pow(2).sum(dim=1)).mean()
        edge_reg = (delta[edges[:, 0]] - delta[edges[:, 1]]).pow(2).mean() / (limits.mean().clamp_min(1e-6) ** 2)
        photo_loss, photo_meta = multiview_photometric_variance_loss(
            vertices,
            view_payloads,
            height,
            width,
            float(args.photometric_mask_threshold),
        )
        depth_tv = depth_tv_total / max(1, len(view_payloads))
        loss = (
            float(args.mask_bce_weight) * mask_bce
            + float(args.target_recall_weight) * recall_loss
            + float(args.overfill_weight) * overfill_loss
            + float(args.token_offset_reg_weight) * offset_reg
            + float(args.edge_reg_weight) * edge_reg
            + float(args.photometric_variance_weight) * photo_loss
            + float(args.normal_depth_proxy_weight) * depth_tv
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
                "token_offset_reg": float(offset_reg.detach().cpu()),
                "edge_reg": float(edge_reg.detach().cpu()),
                "photometric_variance": float(photo_loss.detach().cpu()),
                "depth_tv_proxy": float(depth_tv.detach().cpu()),
                "photometric_meta": photo_meta,
                "metrics": metrics_rows,
            }
        )

    elapsed = float(time.perf_counter() - start_time)
    with torch.no_grad():
        delta, token_delta = decoder(token_features, token_part_ids, vertex_token_ids, limits)
        optimized_vertices_np = (base_vertices + delta).detach().cpu().numpy().astype(np.float32)
        delta_np = delta.detach().cpu().numpy().astype(np.float32)
        token_delta_np = token_delta.detach().cpu().numpy().astype(np.float32)

    normals_np = compute_vertex_normals(optimized_vertices_np, faces_np)
    mesh_path = output_dir / "surface_token_b0_optimized_mesh.ply"
    carrier_path = output_dir / "surface_token_b0_carrier_mesh.ply"
    write_colored_ply(mesh_path, optimized_vertices_np, faces_np, colors_np)
    write_colored_ply(carrier_path, base_vertices_np, faces_np, colors_np)
    write_colored_ply(output_dir / "surface_token_b0_normals.ply", optimized_vertices_np, faces_np, (normals_np + 1.0) * 0.5)

    final_metrics = loss_curve[-1]["metrics"]
    initial_metrics = loss_curve[0]["metrics"]
    avg_initial_iou = float(np.mean([row["iou"] for row in initial_metrics])) if initial_metrics else 0.0
    avg_final_iou = float(np.mean([row["iou"] for row in final_metrics])) if final_metrics else 0.0
    max_delta = float(np.linalg.norm(delta_np, axis=1).max()) if delta_np.size else 0.0
    mean_delta = float(np.linalg.norm(delta_np, axis=1).mean()) if delta_np.size else 0.0
    max_token_delta = float(np.linalg.norm(token_delta_np, axis=1).max()) if token_delta_np.size else 0.0

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

    summary = {
        "status": "surface_token_b0_smoke_complete",
        "summary": {
            "research_only": True,
            "no_teacher_export": True,
            "no_candidate_export": True,
            "strict_candidate_passes": 0,
            "strict_teacher_passes": 0,
            "formal_cloud": "blocked",
            "scene_dir": str(scene_dir),
            "template_payload": str(args.template_payload.resolve()),
            "camera_source": camera_source,
            "views": view_indices,
            "target_size": int(args.target_size),
            "steps": int(args.steps),
            "token_grid": int(args.token_grid),
            "token_hidden": int(args.token_hidden),
            "token_meta": {k: v for k, v in token_meta.items() if k != "token_part_ids"},
            "image_condition_meta": image_meta,
            "elapsed_seconds": elapsed,
            "avg_initial_iou": avg_initial_iou,
            "avg_final_iou": avg_final_iou,
            "avg_iou_delta": avg_final_iou - avg_initial_iou,
            "max_vertex_delta": max_delta,
            "mean_vertex_delta": mean_delta,
            "max_token_delta_unit": max_token_delta,
            "gpu_name": torch.cuda.get_device_name(0),
        },
        "loss_curve": loss_curve,
        "decision": (
            "B0 tests a real surface-token representation with visibility-aware multi-view features and "
            "part-specific heads. It still requires Open3D visual precheck; numeric deltas cannot create a teacher."
        ),
        "outputs": [
            str(mesh_path),
            str(carrier_path),
            str(output_dir / "surface_token_b0_normals.ply"),
            str(output_dir / "surface_token_b0_summary.json"),
            str(output_dir / "surface_token_b0_summary.md"),
        ],
    }
    (output_dir / "surface_token_b0_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output_dir / "surface_token_b0_summary.md", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
