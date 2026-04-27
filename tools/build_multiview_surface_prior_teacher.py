from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt.utils.normal_refiner import (  # noqa: E402
    face_box_from_mask,
    head_box_from_mask,
    normal_to_rgb,
    point_map_to_normal_numpy,
    points_world_to_camera,
    preprocess_mask_image,
    preprocess_rgb_image,
    shoulder_box_from_mask,
)


POSITION_CHANNELS = ("smplx_posed_cam_x", "smplx_posed_cam_y", "smplx_posed_cam_z")
NORMAL_CHANNELS = ("smplx_cam_nx", "smplx_cam_ny", "smplx_cam_nz")
VISIBLE_CHANNEL = "smplx_visible_mask"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse high-confidence 60-view VGGT points into a local surface teacher, then patch "
            "a sparse-view scene's head/face/shoulder prior positions and normals."
        )
    )
    parser.add_argument("--teacher-predictions-npz", required=True)
    parser.add_argument("--teacher-scene-dir", required=True)
    parser.add_argument("--target-scene-dir", required=True)
    parser.add_argument("--output-scene-dir", required=True)
    parser.add_argument("--output-diagnostics-dir", required=True)
    parser.add_argument("--roi-kind", choices=("head", "face", "shoulder", "head_face", "all"), default="head_face")
    parser.add_argument("--teacher-conf-percentile", type=float, default=75.0)
    parser.add_argument("--target-conf-percentile", type=float, default=20.0)
    parser.add_argument("--max-source-points", type=int, default=350000)
    parser.add_argument("--voxel-size", type=float, default=0.006)
    parser.add_argument("--z-tolerance", type=float, default=0.035)
    parser.add_argument("--knn", type=int, default=10)
    parser.add_argument("--max-query-distance", type=float, default=0.045)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_manifest(scene_dir: Path) -> dict[str, object]:
    path = scene_dir / "scene_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _view_paths(scene_dir: Path) -> tuple[list[Path], list[Path], list[str]]:
    manifest = _load_manifest(scene_dir)
    exported = manifest.get("exported_views") or []
    if exported:
        image_paths = [Path(item["image_path"]) for item in exported]
        mask_paths = [Path(item["mask_path"]) for item in exported]
        names = [Path(item["image_path"]).stem for item in exported]
    else:
        image_paths = sorted((scene_dir / "images").iterdir())
        mask_paths = sorted((scene_dir / "masks").iterdir())
        names = [path.stem for path in image_paths]
    return image_paths, mask_paths, names


def _load_masks(scene_dir: Path, size: int) -> np.ndarray:
    _, mask_paths, _ = _view_paths(scene_dir)
    return np.stack([preprocess_mask_image(path, size) for path in mask_paths], axis=0).astype(bool)


def _load_images(scene_dir: Path, size: int) -> np.ndarray:
    image_paths, _, _ = _view_paths(scene_dir)
    return np.stack([preprocess_rgb_image(path, size) for path in image_paths], axis=0)


def _roi_mask(mask: np.ndarray, roi_kind: str) -> np.ndarray:
    boxes: list[tuple[int, int, int, int] | None] = []
    if roi_kind in {"head", "head_face", "all"}:
        boxes.append(head_box_from_mask(mask))
    if roi_kind in {"face", "head_face", "all"}:
        boxes.append(face_box_from_mask(mask))
    if roi_kind in {"shoulder", "all"}:
        boxes.append(shoulder_box_from_mask(mask))
    out = np.zeros(mask.shape, dtype=bool)
    for box in boxes:
        if box is None:
            continue
        x0, y0, x1, y1 = box
        out[y0:y1, x0:x1] |= mask[y0:y1, x0:x1]
    return out


def _lookup_indices(channels: np.ndarray, names: tuple[str, ...]) -> list[int]:
    channel_names = [str(name) for name in np.asarray(channels).tolist()]
    lookup = {name: idx for idx, name in enumerate(channel_names)}
    return [lookup[name] for name in names]


def _camera_ids(scene_dir: Path) -> list[str]:
    manifest = _load_manifest(scene_dir)
    exported = manifest.get("exported_views") or []
    return [str(item.get("camera_id")) for item in exported]


def _target_to_teacher_indices(teacher_scene_dir: Path, target_scene_dir: Path) -> list[int]:
    teacher_ids = _camera_ids(teacher_scene_dir)
    target_ids = _camera_ids(target_scene_dir)
    if not teacher_ids or not target_ids:
        raise ValueError("Both teacher and target scenes must expose exported_views camera_id entries.")
    lookup = {camera_id: idx for idx, camera_id in enumerate(teacher_ids)}
    missing = [camera_id for camera_id in target_ids if camera_id not in lookup]
    if missing:
        raise ValueError(f"Target cameras not present in teacher scene: {missing}")
    return [lookup[camera_id] for camera_id in target_ids]


def _copy_scene(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(dst)
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _sample_teacher_points(
    predictions: dict[str, np.ndarray],
    masks: np.ndarray,
    roi_kind: str,
    conf_percentile: float,
    max_points: int,
) -> np.ndarray:
    world_points = np.asarray(predictions["world_points"], dtype=np.float32)
    conf = np.asarray(predictions.get("world_points_conf", predictions.get("depth_conf")), dtype=np.float32)
    selected: list[np.ndarray] = []
    for view_idx in range(world_points.shape[0]):
        roi = _roi_mask(masks[view_idx], roi_kind)
        valid = roi & np.isfinite(world_points[view_idx]).all(axis=-1)
        if not valid.any():
            continue
        threshold = float(np.percentile(conf[view_idx][valid], conf_percentile))
        valid &= conf[view_idx] >= threshold
        pts = world_points[view_idx][valid]
        if pts.size:
            selected.append(pts)
    if not selected:
        raise RuntimeError("No teacher points selected; lower --teacher-conf-percentile or check masks.")
    points = np.concatenate(selected, axis=0).astype(np.float32)
    if len(points) > max_points:
        rng = np.random.default_rng(20260425)
        keep = rng.choice(len(points), size=max_points, replace=False)
        points = points[keep]
    return points


def _build_o3d_surface(points: np.ndarray, voxel_size: float, knn: int):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))
    pcd.remove_non_finite_points()
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=max(8, int(knn * 3))))
    pcd.orient_normals_consistent_tangent_plane(max(10, int(knn * 3)))
    tree = o3d.geometry.KDTreeFlann(pcd)
    return pcd, tree


def _project_world_to_camera(points_world: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    rotation = np.asarray(extrinsic[:, :3], dtype=np.float32)
    translation = np.asarray(extrinsic[:, 3], dtype=np.float32)
    return points_world @ rotation.T + translation[None]


def _query_surface_for_target(
    *,
    pcd,
    tree,
    predictions: dict[str, np.ndarray],
    teacher_view_indices: list[int],
    target_masks: np.ndarray,
    roi_kind: str,
    target_conf_percentile: float,
    z_tolerance: float,
    max_query_distance: float,
    knn: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    surface_points = np.asarray(pcd.points, dtype=np.float32)
    surface_normals = np.asarray(pcd.normals, dtype=np.float32)
    target_world = np.asarray(predictions["world_points"], dtype=np.float32)[teacher_view_indices]
    target_conf = np.asarray(predictions.get("world_points_conf", predictions.get("depth_conf")), dtype=np.float32)[
        teacher_view_indices
    ]
    extrinsic = np.asarray(predictions["extrinsic"], dtype=np.float32)[teacher_view_indices]
    if target_world.shape[0] != target_masks.shape[0]:
        raise ValueError(
            f"Mapped teacher views ({target_world.shape[0]}) do not match target masks ({target_masks.shape[0]})."
        )

    patched_points = np.zeros_like(target_world, dtype=np.float32)
    patched_normals_cam = np.zeros_like(target_world, dtype=np.float32)
    patch_masks = np.zeros(target_masks.shape, dtype=bool)
    for view_idx in range(target_world.shape[0]):
        roi = _roi_mask(target_masks[view_idx], roi_kind)
        valid = roi & np.isfinite(target_world[view_idx]).all(axis=-1)
        if valid.any():
            threshold = float(np.percentile(target_conf[view_idx][valid], target_conf_percentile))
            valid &= target_conf[view_idx] >= threshold
        ys, xs = np.nonzero(valid)
        if len(xs) == 0:
            continue
        query_points = target_world[view_idx, ys, xs]
        query_cam = _project_world_to_camera(query_points, extrinsic[view_idx])
        view_points = []
        view_normals_world = []
        keep_indices = []
        for local_idx, query in enumerate(query_points):
            count, indices, dists2 = tree.search_knn_vector_3d(query.astype(np.float64), int(knn))
            if count <= 0:
                continue
            idx = np.asarray(indices, dtype=np.int64)
            dists = np.sqrt(np.asarray(dists2, dtype=np.float32))
            if float(dists[0]) > max_query_distance:
                continue
            neighbors = surface_points[idx]
            neighbors_cam = _project_world_to_camera(neighbors, extrinsic[view_idx])
            dz = np.abs(neighbors_cam[:, 2] - query_cam[local_idx, 2])
            z_ok = dz <= z_tolerance
            if not z_ok.any():
                continue
            weights = 1.0 / np.clip(dists[z_ok], 1e-4, None)
            pts = neighbors[z_ok]
            nrms = surface_normals[idx[z_ok]]
            point = (pts * weights[:, None]).sum(axis=0) / weights.sum()
            normal = (nrms * weights[:, None]).sum(axis=0) / weights.sum()
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm < 1e-6:
                continue
            normal = normal / normal_norm
            cam_normal = normal @ extrinsic[view_idx, :, :3].T
            if cam_normal[2] > 0:
                normal = -normal
                cam_normal = -cam_normal
            view_points.append(point.astype(np.float32))
            view_normals_world.append(normal.astype(np.float32))
            keep_indices.append(local_idx)
        if not keep_indices:
            continue
        keep = np.asarray(keep_indices, dtype=np.int64)
        out_y = ys[keep]
        out_x = xs[keep]
        points_arr = np.stack(view_points, axis=0).astype(np.float32)
        normals_world = np.stack(view_normals_world, axis=0).astype(np.float32)
        normals_cam = normals_world @ extrinsic[view_idx, :, :3].T
        normals_cam /= np.clip(np.linalg.norm(normals_cam, axis=-1, keepdims=True), 1e-6, None)
        patched_points[view_idx, out_y, out_x] = _project_world_to_camera(points_arr, extrinsic[view_idx])
        patched_normals_cam[view_idx, out_y, out_x] = normals_cam
        patch_masks[view_idx, out_y, out_x] = True
    return patched_points, patched_normals_cam, patch_masks


def _patch_scene_prior(
    target_scene_dir: Path,
    output_scene_dir: Path,
    positions_cam: np.ndarray,
    normals_cam: np.ndarray,
    patch_masks: np.ndarray,
    overwrite: bool,
) -> dict[str, object]:
    _copy_scene(target_scene_dir, output_scene_dir, overwrite=overwrite)
    prior_path = output_scene_dir / "prior_maps.npz"
    payload = np.load(prior_path, allow_pickle=False)
    prior_maps = np.asarray(payload["prior_maps"], dtype=np.float32).copy()
    prior_mask = np.asarray(payload["prior_mask"], dtype=bool).copy()
    prior_channels = np.asarray(payload["prior_channels"])
    position_idx = _lookup_indices(prior_channels, POSITION_CHANNELS)
    normal_idx = _lookup_indices(prior_channels, NORMAL_CHANNELS)
    visible_idx = _lookup_indices(prior_channels, (VISIBLE_CHANNEL,))[0]

    for view_idx in range(prior_maps.shape[0]):
        mask = patch_masks[view_idx]
        if not mask.any():
            continue
        for channel_offset, channel_idx in enumerate(position_idx):
            prior_maps[view_idx, channel_idx][mask] = positions_cam[view_idx, ..., channel_offset][mask]
        for channel_offset, channel_idx in enumerate(normal_idx):
            prior_maps[view_idx, channel_idx][mask] = normals_cam[view_idx, ..., channel_offset][mask]
        prior_maps[view_idx, visible_idx][mask] = 1.0
        prior_mask[view_idx][mask] = True

    save_kwargs = {key: payload[key] for key in payload.files if key not in {"prior_maps", "prior_mask"}}
    np.savez_compressed(
        prior_path,
        prior_maps=prior_maps.astype(np.float16),
        prior_mask=prior_mask,
        **save_kwargs,
    )
    manifest_path = output_scene_dir / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["multiview_surface_prior_teacher"] = {
        "patched_pixels_total": int(patch_masks.sum()),
        "patched_pixels_per_view": [int(value) for value in patch_masks.reshape(patch_masks.shape[0], -1).sum(axis=1)],
        "source": "60v high-confidence VGGT point surface, local PCA/Open3D normal teacher",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest["multiview_surface_prior_teacher"]


def _write_diagnostics(
    output_dir: Path,
    target_scene_dir: Path,
    target_masks: np.ndarray,
    positions_cam: np.ndarray,
    normals_cam: np.ndarray,
    patch_masks: np.ndarray,
    summary: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    images = _load_images(target_scene_dir, int(target_masks.shape[-1]))
    _, _, view_names = _view_paths(target_scene_dir)
    for view_idx, view_name in enumerate(view_names):
        normal_rgb = normal_to_rgb(normals_cam[view_idx], patch_masks[view_idx])
        Image.fromarray(normal_rgb).save(output_dir / f"{view_idx:02d}_{view_name}_surface_teacher_normal.png")
        mask_rgb = np.zeros((*patch_masks.shape[1:], 3), dtype=np.uint8)
        mask_rgb[..., 0] = patch_masks[view_idx].astype(np.uint8) * 255
        overlay = (0.65 * images[view_idx].astype(np.float32) + 0.35 * mask_rgb.astype(np.float32)).clip(0, 255).astype(np.uint8)
        Image.fromarray(overlay).save(output_dir / f"{view_idx:02d}_{view_name}_surface_teacher_mask_overlay.png")
        teacher_normal, valid = point_map_to_normal_numpy(positions_cam[view_idx], patch_masks[view_idx])
        Image.fromarray(normal_to_rgb(teacher_normal, valid)).save(output_dir / f"{view_idx:02d}_{view_name}_position_gradient_normal_check.png")
    (output_dir / "surface_teacher_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    teacher_scene_dir = Path(args.teacher_scene_dir)
    target_scene_dir = Path(args.target_scene_dir)
    output_scene_dir = Path(args.output_scene_dir)
    output_diag_dir = Path(args.output_diagnostics_dir)
    predictions = dict(np.load(args.teacher_predictions_npz, allow_pickle=False))
    size = int(predictions["world_points"].shape[1])
    teacher_masks = _load_masks(teacher_scene_dir, size)
    target_masks = _load_masks(target_scene_dir, size)
    teacher_view_indices = _target_to_teacher_indices(teacher_scene_dir, target_scene_dir)
    source_points = _sample_teacher_points(
        predictions=predictions,
        masks=teacher_masks,
        roi_kind=args.roi_kind,
        conf_percentile=args.teacher_conf_percentile,
        max_points=args.max_source_points,
    )
    pcd, tree = _build_o3d_surface(source_points, voxel_size=args.voxel_size, knn=args.knn)
    positions_cam, normals_cam, patch_masks = _query_surface_for_target(
        pcd=pcd,
        tree=tree,
        predictions=predictions,
        teacher_view_indices=teacher_view_indices,
        target_masks=target_masks,
        roi_kind=args.roi_kind,
        target_conf_percentile=args.target_conf_percentile,
        z_tolerance=args.z_tolerance,
        max_query_distance=args.max_query_distance,
        knn=args.knn,
    )
    patch_summary = _patch_scene_prior(
        target_scene_dir=target_scene_dir,
        output_scene_dir=output_scene_dir,
        positions_cam=positions_cam,
        normals_cam=normals_cam,
        patch_masks=patch_masks,
        overwrite=bool(args.overwrite),
    )
    summary = {
        "teacher_predictions_npz": str(Path(args.teacher_predictions_npz).resolve()),
        "teacher_scene_dir": str(teacher_scene_dir.resolve()),
        "target_scene_dir": str(target_scene_dir.resolve()),
        "output_scene_dir": str(output_scene_dir.resolve()),
        "target_to_teacher_indices": [int(idx) for idx in teacher_view_indices],
        "roi_kind": args.roi_kind,
        "source_points_before_surface": int(len(source_points)),
        "surface_points_after_voxel": int(np.asarray(pcd.points).shape[0]),
        "teacher_conf_percentile": float(args.teacher_conf_percentile),
        "target_conf_percentile": float(args.target_conf_percentile),
        "voxel_size": float(args.voxel_size),
        "z_tolerance": float(args.z_tolerance),
        "max_query_distance": float(args.max_query_distance),
        **patch_summary,
    }
    _write_diagnostics(output_diag_dir, target_scene_dir, target_masks, positions_cam, normals_cam, patch_masks, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
