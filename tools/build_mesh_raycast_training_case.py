from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt.utils.normal_refiner import (  # noqa: E402
    face_box_from_mask,
    head_box_from_mask,
    point_map_to_normal_numpy,
    preprocess_mask_image,
    shoulder_box_from_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a geometry-consistent sparse-view training case by raycasting a 60v Open3D mesh teacher."
    )
    parser.add_argument("--source-case-dir", required=True)
    parser.add_argument("--teacher-predictions-npz", required=True)
    parser.add_argument("--teacher-scene-dir", required=True)
    parser.add_argument("--target-scene-dir", required=True)
    parser.add_argument("--anchor-predictions-npz", required=True)
    parser.add_argument("--output-case-dir", required=True)
    parser.add_argument("--output-diagnostics-dir", required=True)
    parser.add_argument("--roi-kind", choices=("head", "face", "face_core", "head_face", "shoulder", "all"), default="head_face")
    parser.add_argument("--teacher-conf-percentile", type=float, default=70.0)
    parser.add_argument("--max-source-points", type=int, default=280000)
    parser.add_argument("--voxel-size", type=float, default=0.004)
    parser.add_argument("--poisson-depth", type=int, default=8)
    parser.add_argument("--density-quantile", type=float, default=0.04)
    parser.add_argument("--depth-tolerance", type=float, default=0.08)
    parser.add_argument("--conf-boost", type=float, default=96.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_masks(scene_dir: Path, size: int) -> np.ndarray:
    mask_dir = scene_dir / "masks"
    paths = sorted(path for path in mask_dir.iterdir() if path.is_file())
    return np.stack([preprocess_mask_image(path, size) for path in paths], axis=0).astype(bool)


def _roi_mask(mask: np.ndarray, roi_kind: str) -> np.ndarray:
    if roi_kind == "all":
        return np.asarray(mask, dtype=bool)
    boxes = []
    if roi_kind in {"head", "head_face"}:
        boxes.append(head_box_from_mask(mask))
    if roi_kind in {"face", "head_face"}:
        boxes.append(face_box_from_mask(mask))
    if roi_kind == "face_core":
        face_box = face_box_from_mask(mask)
        if face_box is not None:
            x0, y0, x1, y1 = face_box
            width = x1 - x0
            height = y1 - y0
            core_w = max(16, int(round(width * 0.72)))
            core_h = max(16, int(round(height * 0.70)))
            cx = int(round((x0 + x1) * 0.5))
            cy = y0 + int(round(height * 0.46))
            boxes.append(
                (
                    cx - core_w // 2,
                    cy - core_h // 2,
                    cx + core_w // 2,
                    cy + core_h // 2,
                )
            )
    if roi_kind == "shoulder":
        boxes.append(shoulder_box_from_mask(mask))
    out = np.zeros(mask.shape, dtype=bool)
    for box in boxes:
        if box is None:
            continue
        x0, y0, x1, y1 = box
        x0 = max(0, min(mask.shape[1], x0))
        x1 = max(0, min(mask.shape[1], x1))
        y0 = max(0, min(mask.shape[0], y0))
        y1 = max(0, min(mask.shape[0], y1))
        if x1 <= x0 or y1 <= y0:
            continue
        out[y0:y1, x0:x1] |= mask[y0:y1, x0:x1]
    return out


def _copy_case(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(dst)
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _sample_teacher_points(predictions: dict[str, np.ndarray], masks: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    points = np.asarray(predictions["world_points"], dtype=np.float32)
    conf = np.asarray(predictions.get("world_points_conf", predictions.get("depth_conf")), dtype=np.float32)
    selected = []
    for view_idx in range(points.shape[0]):
        roi = _roi_mask(masks[view_idx], args.roi_kind)
        valid = roi & np.isfinite(points[view_idx]).all(axis=-1)
        if not valid.any():
            continue
        threshold = float(np.percentile(conf[view_idx][valid], float(args.teacher_conf_percentile)))
        valid &= conf[view_idx] >= threshold
        if valid.any():
            selected.append(points[view_idx][valid])
    if not selected:
        raise RuntimeError("No teacher points selected.")
    all_points = np.concatenate(selected, axis=0).astype(np.float32)
    if len(all_points) > int(args.max_source_points):
        rng = np.random.default_rng(20260425)
        keep = rng.choice(len(all_points), size=int(args.max_source_points), replace=False)
        all_points = all_points[keep]
    return all_points


def _build_mesh(points: np.ndarray, args: argparse.Namespace):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if float(args.voxel_size) > 0:
        pcd = pcd.voxel_down_sample(float(args.voxel_size))
    pcd.remove_non_finite_points()
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=48))
    pcd.orient_normals_consistent_tangent_plane(48)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=int(args.poisson_depth), n_threads=0
    )
    densities = np.asarray(densities)
    if densities.size:
        keep = densities >= np.quantile(densities, float(args.density_quantile))
        mesh.remove_vertices_by_mask(~keep)
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox = bbox.scale(1.08, bbox.get_center())
    mesh = mesh.crop(bbox)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return pcd, mesh


def _rays_for_pixels(xs: np.ndarray, ys: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    dirs_cam = np.stack([(xs.astype(np.float32) - cx) / max(fx, 1e-6), (ys.astype(np.float32) - cy) / max(fy, 1e-6), np.ones_like(xs, dtype=np.float32)], axis=-1)
    rotation = np.asarray(extrinsic[:, :3], dtype=np.float32)
    translation = np.asarray(extrinsic[:, 3], dtype=np.float32)
    origin = -rotation.T @ translation
    dirs_world = dirs_cam @ rotation
    dirs_world /= np.clip(np.linalg.norm(dirs_world, axis=-1, keepdims=True), 1e-6, None)
    origins = np.broadcast_to(origin[None, :], dirs_world.shape)
    return np.concatenate([origins, dirs_world], axis=-1).astype(np.float32)


def _world_to_cam(points: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    rotation = np.asarray(extrinsic[:, :3], dtype=np.float32)
    translation = np.asarray(extrinsic[:, 3], dtype=np.float32)
    return points @ rotation.T + translation[None]


def _cam_to_world(cam_points: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    out = np.zeros_like(cam_points, dtype=np.float32)
    for view_idx in range(cam_points.shape[0]):
        rotation = extrinsics[view_idx, :, :3].astype(np.float32)
        translation = extrinsics[view_idx, :, 3].astype(np.float32)
        flat = cam_points[view_idx].reshape(-1, 3)
        out[view_idx] = ((flat - translation[None]) @ rotation).reshape(cam_points.shape[1:])
    return out.astype(np.float32)


def _raycast_mesh(mesh, target_masks: np.ndarray, targets: dict[str, np.ndarray], anchor: dict[str, np.ndarray], args: argparse.Namespace):
    import open3d as o3d

    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)
    intrinsics = np.asarray(targets["intrinsics"], dtype=np.float32)
    extrinsics = np.asarray(targets["extrinsics"], dtype=np.float32)
    anchor_depth = np.asarray(anchor["depth"], dtype=np.float32)[..., 0]
    views, height, width = target_masks.shape
    hit_cam = np.zeros((views, height, width, 3), dtype=np.float32)
    hit_mask = np.zeros((views, height, width), dtype=bool)
    hit_depth = np.zeros((views, height, width), dtype=np.float32)
    for view_idx in range(views):
        roi = _roi_mask(target_masks[view_idx], args.roi_kind)
        ys, xs = np.nonzero(roi)
        if len(xs) == 0:
            continue
        rays = _rays_for_pixels(xs, ys, intrinsics[view_idx], extrinsics[view_idx])
        ans = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
        t_hit = ans["t_hit"].numpy()
        valid = np.isfinite(t_hit)
        if not valid.any():
            continue
        origins = rays[:, :3]
        dirs = rays[:, 3:]
        world_hit = origins + dirs * t_hit[:, None]
        cam_hit = _world_to_cam(world_hit.astype(np.float32), extrinsics[view_idx])
        depth = cam_hit[:, 2]
        depth_ok = valid & (depth > 0.05) & (np.abs(depth - anchor_depth[view_idx, ys, xs]) <= float(args.depth_tolerance))
        out_y = ys[depth_ok]
        out_x = xs[depth_ok]
        hit_cam[view_idx, out_y, out_x] = cam_hit[depth_ok]
        hit_depth[view_idx, out_y, out_x] = depth[depth_ok]
        hit_mask[view_idx, out_y, out_x] = True
    return hit_cam, hit_depth, hit_mask


def _make_preview(image: np.ndarray, anchor_depth: np.ndarray, hit_depth: np.ndarray, hit_mask: np.ndarray, out_path: Path) -> None:
    def depth_rgb(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
        values = depth[mask]
        if values.size == 0:
            values = depth[np.isfinite(depth)]
        lo, hi = np.percentile(values, [2, 98]) if values.size else (0, 1)
        gray = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
        rgb = np.stack([gray, gray, gray], axis=-1)
        rgb[~mask] = 1
        return (rgb * 255).astype(np.uint8)

    overlay = image.copy()
    overlay[hit_mask] = (0.55 * overlay[hit_mask].astype(np.float32) + np.array([255, 0, 0], dtype=np.float32) * 0.45).astype(np.uint8)
    tiles = [
        image.astype(np.uint8),
        depth_rgb(anchor_depth, np.isfinite(anchor_depth)),
        depth_rgb(hit_depth, hit_mask),
        overlay,
    ]
    labels = ["RGB", "anchor depth", "mesh ray depth", "accepted hits"]
    canvas = Image.new("RGB", (image.shape[1] * 4, image.shape[0] + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (tile, label) in enumerate(zip(tiles, labels)):
        canvas.paste(Image.fromarray(tile), (idx * image.shape[1], 24))
        draw.text((idx * image.shape[1] + 4, 4), label, fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> int:
    args = parse_args()
    source_case = Path(args.source_case_dir)
    output_case = Path(args.output_case_dir)
    diagnostics_dir = Path(args.output_diagnostics_dir)
    _copy_case(source_case, output_case, overwrite=bool(args.overwrite))
    with np.load(output_case / "inputs.npz", allow_pickle=False) as payload:
        inputs = {key: np.array(payload[key]) for key in payload.files}
    with np.load(output_case / "targets.npz", allow_pickle=False) as payload:
        targets = {key: np.array(payload[key]) for key in payload.files}
    teacher_predictions = dict(np.load(args.teacher_predictions_npz, allow_pickle=False))
    anchor = dict(np.load(args.anchor_predictions_npz, allow_pickle=False))
    size = int(teacher_predictions["world_points"].shape[1])
    teacher_masks = _load_masks(Path(args.teacher_scene_dir), size)
    target_masks = np.asarray(inputs.get("point_masks"), dtype=bool)
    points = _sample_teacher_points(teacher_predictions, teacher_masks, args)
    pcd, mesh = _build_mesh(points, args)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    import open3d as o3d

    o3d.io.write_point_cloud(str(diagnostics_dir / "teacher_points_downsampled.ply"), pcd)
    o3d.io.write_triangle_mesh(str(diagnostics_dir / "teacher_mesh_poisson.ply"), mesh)
    hit_cam, hit_depth, hit_mask = _raycast_mesh(mesh, target_masks, targets, anchor, args)
    hit_world = _cam_to_world(hit_cam, np.asarray(targets["extrinsics"], dtype=np.float32))
    teacher_normals = np.zeros_like(hit_cam, dtype=np.float32)
    for view_idx in range(hit_cam.shape[0]):
        normal, valid = point_map_to_normal_numpy(hit_cam[view_idx], hit_mask[view_idx])
        teacher_normals[view_idx] = normal
        hit_mask[view_idx] &= valid
        _make_preview(
            np.asarray(inputs["images"][view_idx], dtype=np.uint8),
            np.asarray(anchor["depth"], dtype=np.float32)[view_idx, ..., 0],
            hit_depth[view_idx],
            hit_mask[view_idx],
            diagnostics_dir / f"{view_idx:02d}_mesh_raycast_teacher_preview.png",
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
    np.savez_compressed(output_case / "targets.npz", **targets)
    summary = {
        "source_case_dir": str(source_case.resolve()),
        "teacher_predictions_npz": str(Path(args.teacher_predictions_npz).resolve()),
        "anchor_predictions_npz": str(Path(args.anchor_predictions_npz).resolve()),
        "output_case_dir": str(output_case.resolve()),
        "roi_kind": args.roi_kind,
        "source_points_selected": int(len(points)),
        "mesh_vertices": int(np.asarray(mesh.vertices).shape[0]),
        "mesh_triangles": int(np.asarray(mesh.triangles).shape[0]),
        "hit_pixels_total": int(hit_mask.sum()),
        "hit_pixels_per_view": [int(v) for v in hit_mask.reshape(hit_mask.shape[0], -1).sum(axis=1)],
        "depth_tolerance": float(args.depth_tolerance),
    }
    (diagnostics_dir / "mesh_raycast_teacher_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest_path = output_case / "case_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    manifest["mesh_raycast_training_patch"] = summary
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
