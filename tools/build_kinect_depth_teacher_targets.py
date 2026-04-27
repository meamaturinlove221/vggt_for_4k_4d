from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.dna_4k4d import SUBSET_NAME, build_context, materialize_rgb_cams_smc, normalize_camera_id  # noqa: E402
from tools.prepare_4k4d_prior_training_case import align_intrinsics_for_scene_view  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project real 4K4D Kinect depth into an exported 6-view scene and build VGGT-coordinate "
            "teacher targets after a robust target-camera-to-VGGT alignment gate."
        )
    )
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--base-predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--kinect-smc", type=Path, default=None)
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--roi-kind", choices=("all", "head", "face", "face_core", "head_face"), default="head_face")
    parser.add_argument("--transform-mode", choices=("similarity", "axis_affine"), default="similarity")
    parser.add_argument("--conf-percentile", type=float, default=40.0)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--min-depth-m", type=float, default=0.4)
    parser.add_argument("--max-depth-m", type=float, default=6.0)
    parser.add_argument("--max-correspondences", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_mask(path: Path, target_size: int) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != (target_size, target_size):
        image = image.resize((target_size, target_size), Image.Resampling.NEAREST)
    return np.asarray(image) > 127


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def clamp_box(box: tuple[int, int, int, int], image_hw: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = int(image_hw[0]), int(image_hw[1])
    x0, y0, x1, y1 = box
    x0 = max(0, min(width, int(x0)))
    y0 = max(0, min(height, int(y0)))
    x1 = max(x0 + 1, min(width, int(x1)))
    y1 = max(y0 + 1, min(height, int(y1)))
    return x0, y0, x1, y1


def expand_box(
    box: tuple[int, int, int, int],
    image_hw: tuple[int, int],
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    pad_x: int = 0,
    pad_y: int = 0,
    min_size: int = 16,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    half_w = max(min_size / 2.0, 0.5 * (x1 - x0) * scale_x + pad_x)
    half_h = max(min_size / 2.0, 0.5 * (y1 - y0) * scale_y + pad_y)
    return clamp_box(
        (
            int(round(cx - half_w)),
            int(round(cy - half_h)),
            int(round(cx + half_w)),
            int(round(cy + half_h)),
        ),
        image_hw,
    )


def head_box_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    bbox = mask_bbox(mask)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    body_h = y1 - y0
    head_h = max(24, int(round(body_h * 0.45)))
    raw = (x0, y0, x1, min(y1, y0 + head_h))
    return expand_box(
        raw,
        mask.shape,
        scale_x=1.15,
        scale_y=1.08,
        pad_x=max(4, int(round((x1 - x0) * 0.03))),
        pad_y=max(4, int(round(body_h * 0.02))),
    )


def face_box_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    head_box = head_box_from_mask(mask)
    if head_box is None:
        return None
    x0, y0, x1, y1 = head_box
    width = x1 - x0
    height = y1 - y0
    face_w = max(24, int(round(width * 0.62)))
    face_h = max(24, int(round(height * 0.62)))
    cx = int(round(0.5 * (x0 + x1)))
    cy = y0 + int(round(height * 0.42))
    return clamp_box((cx - face_w // 2, cy - face_h // 2, cx + face_w // 2, cy + face_h // 2), mask.shape)


def roi_mask_from_human(mask: np.ndarray, roi_kind: str) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if roi_kind == "all":
        return mask.copy()

    boxes: list[tuple[int, int, int, int] | None] = []
    if roi_kind in {"head", "head_face"}:
        boxes.append(head_box_from_mask(mask))
    if roi_kind in {"face", "head_face"}:
        boxes.append(face_box_from_mask(mask))
    if roi_kind == "face_core":
        face = face_box_from_mask(mask)
        if face is not None:
            x0, y0, x1, y1 = face
            width = x1 - x0
            height = y1 - y0
            boxes.append(
                clamp_box(
                    (
                        x0 + int(round(width * 0.18)),
                        y0 + int(round(height * 0.12)),
                        x1 - int(round(width * 0.18)),
                        y1 - int(round(height * 0.18)),
                    ),
                    mask.shape,
                )
            )

    out = np.zeros_like(mask, dtype=bool)
    for box in boxes:
        if box is None:
            continue
        x0, y0, x1, y1 = box
        out[y0:y1, x0:x1] |= mask[y0:y1, x0:x1]
    return out


def resolve_kinect_smc(scene_manifest: dict[str, Any], requested: Path | None) -> Path:
    if requested is not None and requested.is_file():
        return requested.resolve()

    dataset_root = Path(scene_manifest["dataset_root"]).expanduser()
    seq_id = scene_manifest["seq_id"]
    candidate = dataset_root / "kinect" / f"{seq_id}_kinect.smc"
    if candidate.is_file():
        return candidate.resolve()

    root = Path("G:/")
    if root.exists():
        matches = sorted(root.glob(f"*/datasets/data_used_in_4K4D/kinect/{seq_id}_kinect.smc"))
        for match in matches:
            if match.is_file():
                return match.resolve()
    raise FileNotFoundError(f"Could not resolve Kinect SMC for sequence {seq_id}.")


def load_rgb_camera_params(scene_manifest: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    dataset_root = Path(scene_manifest["dataset_root"]).expanduser()
    context = build_context(dataset_root, SUBSET_NAME)
    with tempfile.TemporaryDirectory(prefix="kinect_teacher_rgbcams_") as temp_name:
        rgb_cams_path, source = materialize_rgb_cams_smc(context, scene_manifest["seq_id"], Path(temp_name))
        if rgb_cams_path is None:
            raise FileNotFoundError(f"Could not resolve rgb_cams.smc for {scene_manifest['seq_id']}")
        params: dict[str, dict[str, np.ndarray]] = {"_source": {"rgb_cams_smc": np.asarray(str(source))}}
        with h5py.File(rgb_cams_path, "r") as handle:
            for view in scene_manifest["exported_views"]:
                camera_id = normalize_camera_id(view["camera_id"])
                group = handle["Camera_Parameter"][camera_id]
                cam_to_world = group["RT"][()].astype(np.float32)
                params[camera_id] = {
                    "intrinsic": group["K"][()].astype(np.float32),
                    "aligned_intrinsic": align_intrinsics_for_scene_view(group["K"][()].astype(np.float32), view, 518),
                    "cam_to_world": cam_to_world,
                    "world_to_cam": np.linalg.inv(cam_to_world).astype(np.float32),
                }
        return params


def unproject_kinect_cloud(
    kinect_smc: Path,
    frame: int,
    *,
    depth_scale: float,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    points_world: list[np.ndarray] = []
    camera_ids: list[int] = []
    stats: dict[str, Any] = {}
    with h5py.File(kinect_smc, "r") as handle:
        available = sorted([key for key in handle["Kinect"].keys() if str(key).isdigit()], key=lambda x: int(x))
        for camera_id in available:
            calib_id = f"{int(camera_id):02d}"
            depth = handle[f"Kinect/{int(camera_id)}/depth/{int(frame)}"][()]
            mask = handle[f"Kinect/{int(camera_id)}/mask/{int(frame)}"][()] > 0
            intrinsic = handle[f"Calibration/Kinect/{calib_id}/K"][()].astype(np.float64)
            cam_to_world = handle[f"Calibration/Kinect/{calib_id}/RT"][()].astype(np.float64)
            depth_m = depth.astype(np.float64) / float(depth_scale)
            valid = mask & (depth > 0) & np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
            ys, xs = np.nonzero(valid)
            z = depth_m[ys, xs]
            x = (xs.astype(np.float64) - intrinsic[0, 2]) * z / intrinsic[0, 0]
            y = (ys.astype(np.float64) - intrinsic[1, 2]) * z / intrinsic[1, 1]
            points_camera = np.column_stack((x, y, z))
            world = points_camera @ cam_to_world[:3, :3].T + cam_to_world[:3, 3]
            points_world.append(world.astype(np.float32))
            camera_ids.extend([int(camera_id)] * int(world.shape[0]))
            stats[str(camera_id)] = {
                "valid_points": int(world.shape[0]),
                "depth_m_percentiles": [float(v) for v in np.percentile(z, [0, 5, 50, 95, 100])] if z.size else [],
            }
    if not points_world:
        raise RuntimeError("No Kinect points selected.")
    return np.concatenate(points_world, axis=0), np.asarray(camera_ids, dtype=np.int16), stats


def project_zbuffer(
    points_world: np.ndarray,
    rgb_params: dict[str, np.ndarray],
    roi_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = roi_mask.shape
    world_to_cam = np.asarray(rgb_params["world_to_cam"], dtype=np.float64)
    intrinsic = np.asarray(rgb_params["aligned_intrinsic"], dtype=np.float64)
    cam_points = points_world.astype(np.float64) @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
    depth = cam_points[:, 2]
    positive = np.isfinite(cam_points).all(axis=1) & (depth > 1e-6)
    uvw = cam_points @ intrinsic.T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-8, None)
    xi = np.rint(uv[:, 0]).astype(np.int32)
    yi = np.rint(uv[:, 1]).astype(np.int32)
    inside = positive & (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    inside_indices = np.nonzero(inside)[0]
    if inside_indices.size == 0:
        return (
            np.zeros((height, width, 3), dtype=np.float32),
            np.zeros((height, width, 3), dtype=np.float32),
            np.zeros((height, width), dtype=bool),
        )

    xi_inside = xi[inside_indices]
    yi_inside = yi[inside_indices]
    roi_ok = roi_mask[yi_inside, xi_inside]
    inside_indices = inside_indices[roi_ok]
    xi_inside = xi_inside[roi_ok]
    yi_inside = yi_inside[roi_ok]
    if inside_indices.size == 0:
        return (
            np.zeros((height, width, 3), dtype=np.float32),
            np.zeros((height, width, 3), dtype=np.float32),
            np.zeros((height, width), dtype=bool),
        )

    pixel_index = yi_inside.astype(np.int64) * width + xi_inside.astype(np.int64)
    depth_inside = depth[inside_indices]
    order = np.lexsort((depth_inside, pixel_index))
    sorted_pixels = pixel_index[order]
    keep_sorted = np.r_[True, sorted_pixels[1:] != sorted_pixels[:-1]]
    keep_indices = inside_indices[order][keep_sorted]
    keep_pixels = sorted_pixels[keep_sorted]
    out_y = (keep_pixels // width).astype(np.int64)
    out_x = (keep_pixels % width).astype(np.int64)

    real_world_map = np.zeros((height, width, 3), dtype=np.float32)
    real_cam_map = np.zeros((height, width, 3), dtype=np.float32)
    hit_mask = np.zeros((height, width), dtype=bool)
    real_world_map[out_y, out_x] = points_world[keep_indices].astype(np.float32)
    real_cam_map[out_y, out_x] = cam_points[keep_indices].astype(np.float32)
    hit_mask[out_y, out_x] = True
    return real_world_map, real_cam_map, hit_mask


def estimate_umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape[0] < 16:
        raise RuntimeError(f"Need at least 16 correspondences for similarity; got {source.shape[0]}")
    mu_source = source.mean(axis=0)
    mu_target = target.mean(axis=0)
    src_centered = source - mu_source
    tgt_centered = target - mu_target
    covariance = (tgt_centered.T @ src_centered) / source.shape[0]
    u_mat, singular_values, vt_mat = np.linalg.svd(covariance)
    rotation = u_mat @ vt_mat
    if np.linalg.det(rotation) < 0:
        u_mat[:, -1] *= -1.0
        rotation = u_mat @ vt_mat
    variance = float((src_centered**2).sum() / source.shape[0])
    scale = float(singular_values.sum() / max(variance, 1e-12))
    translation = mu_target - scale * (rotation @ mu_source)
    return scale, rotation.astype(np.float64), translation.astype(np.float64)


def apply_similarity(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    original_shape = points.shape
    flat = points.reshape(-1, 3).astype(np.float64)
    transformed = scale * (flat @ rotation.T) + translation[None, :]
    return transformed.reshape(original_shape).astype(np.float32)


def estimate_axis_affine(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    src_iqr = np.percentile(source, 75, axis=0) - np.percentile(source, 25, axis=0)
    tgt_iqr = np.percentile(target, 75, axis=0) - np.percentile(target, 25, axis=0)
    scale = tgt_iqr / np.clip(src_iqr, 1e-8, None)
    translation = np.median(target, axis=0) - scale * np.median(source, axis=0)
    return scale.astype(np.float64), translation.astype(np.float64)


def apply_axis_affine(points: np.ndarray, scale: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (points.astype(np.float64) * scale.reshape(1, 1, 1, 3) + translation.reshape(1, 1, 1, 3)).astype(np.float32)


def robust_transform(
    source: np.ndarray,
    target: np.ndarray,
    *,
    mode: str,
    max_correspondences: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source = source[finite]
    target = target[finite]
    if source.shape[0] > max_correspondences > 0:
        rng = np.random.default_rng(seed)
        indices = rng.choice(source.shape[0], size=max_correspondences, replace=False)
        source = source[indices]
        target = target[indices]

    if mode == "similarity":
        scale, rotation, translation = estimate_umeyama(source, target)
        predicted = apply_similarity(source.reshape(1, -1, 3), scale, rotation, translation).reshape(-1, 3)
        residual = np.linalg.norm(predicted - target, axis=1)
        threshold = float(np.percentile(residual, 80.0))
        keep = residual <= threshold
        scale, rotation, translation = estimate_umeyama(source[keep], target[keep])
        predicted = apply_similarity(source.reshape(1, -1, 3), scale, rotation, translation).reshape(-1, 3)
        residual = np.linalg.norm(predicted - target, axis=1)
        summary = {
            "mode": mode,
            "input_correspondences": int(finite.sum()),
            "used_correspondences": int(source.shape[0]),
            "refit_correspondences": int(keep.sum()),
            "scale": float(scale),
            "rotation": rotation,
            "translation": translation,
            "residual_percentiles": [float(v) for v in np.percentile(residual, [0, 25, 50, 75, 90, 95, 99])],
        }
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = scale * rotation
        matrix[:3, 3] = translation
        return summary, matrix

    scale_vec, translation_vec = estimate_axis_affine(source, target)
    predicted = source * scale_vec.reshape(1, 3) + translation_vec.reshape(1, 3)
    residual = np.linalg.norm(predicted - target, axis=1)
    threshold = float(np.percentile(residual, 80.0))
    keep = residual <= threshold
    scale_vec, translation_vec = estimate_axis_affine(source[keep], target[keep])
    predicted = source * scale_vec.reshape(1, 3) + translation_vec.reshape(1, 3)
    residual = np.linalg.norm(predicted - target, axis=1)
    summary = {
        "mode": mode,
        "input_correspondences": int(finite.sum()),
        "used_correspondences": int(source.shape[0]),
        "refit_correspondences": int(keep.sum()),
        "scale_xyz": scale_vec,
        "translation_xyz": translation_vec,
        "residual_percentiles": [float(v) for v in np.percentile(residual, [0, 25, 50, 75, 90, 95, 99])],
    }
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0] = scale_vec[0]
    matrix[1, 1] = scale_vec[1]
    matrix[2, 2] = scale_vec[2]
    matrix[:3, 3] = translation_vec
    return summary, matrix


def make_overlay(image_path: Path, roi: np.ndarray, hits: np.ndarray, output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    if image.size != (roi.shape[1], roi.shape[0]):
        image = image.resize((roi.shape[1], roi.shape[0]), Image.Resampling.BILINEAR)
    arr = np.asarray(image).astype(np.float32)
    roi_only = roi & ~hits
    arr[roi_only] = arr[roi_only] * 0.65 + np.asarray([255, 210, 0], dtype=np.float32) * 0.35
    arr[hits] = arr[hits] * 0.45 + np.asarray([0, 220, 80], dtype=np.float32) * 0.55
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(output_path)


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if colors is None:
        colors = np.full((points.shape[0], 3), 210, dtype=np.uint8)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors, strict=False):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def main() -> int:
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty. Re-run with --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_manifest = load_json(scene_dir / "scene_manifest.json")
    predictions = np.load(args.base_predictions, allow_pickle=False)
    base_world = np.asarray(predictions["world_points"], dtype=np.float32)
    base_conf = np.asarray(predictions["world_points_conf"], dtype=np.float32)
    height, width = base_world.shape[1:3]
    if height != width:
        raise ValueError(f"Expected square predictions, got {base_world.shape}")
    if height != 518:
        raise ValueError(f"This smoke tool currently expects 518x518 predictions, got {height}.")

    frame = int(args.frame if args.frame is not None else scene_manifest.get("frame_id", 0))
    kinect_smc = resolve_kinect_smc(scene_manifest, args.kinect_smc)
    rgb_params = load_rgb_camera_params(scene_manifest)
    kinect_world, kinect_camera_ids, kinect_stats = unproject_kinect_cloud(
        kinect_smc,
        frame,
        depth_scale=float(args.depth_scale),
        min_depth_m=float(args.min_depth_m),
        max_depth_m=float(args.max_depth_m),
    )

    masks = []
    roi_masks = []
    for view in scene_manifest["exported_views"]:
        mask = load_mask(Path(view["mask_path"]), height)
        masks.append(mask)
        roi_masks.append(roi_mask_from_human(mask, args.roi_kind))
    masks_arr = np.stack(masks, axis=0)
    roi_arr = np.stack(roi_masks, axis=0)

    view_real_world = np.zeros_like(base_world, dtype=np.float32)
    view_real_target_cam = np.zeros_like(base_world, dtype=np.float32)
    hit_masks = np.zeros(base_world.shape[:3], dtype=bool)
    real_cam_maps: list[np.ndarray] = []

    target_camera_id = normalize_camera_id(scene_manifest["exported_views"][0]["camera_id"])
    target_w2c_real = np.asarray(rgb_params[target_camera_id]["world_to_cam"], dtype=np.float64)

    per_view_summary = []
    for view_idx, view in enumerate(scene_manifest["exported_views"]):
        camera_id = normalize_camera_id(view["camera_id"])
        world_map, cam_map, hit_mask = project_zbuffer(kinect_world, rgb_params[camera_id], roi_arr[view_idx])
        target_cam_flat = world_map.reshape(-1, 3).astype(np.float64)
        target_cam_flat = target_cam_flat @ target_w2c_real[:3, :3].T + target_w2c_real[:3, 3]
        target_cam_map = target_cam_flat.reshape(height, width, 3).astype(np.float32)
        view_real_world[view_idx] = world_map
        view_real_target_cam[view_idx] = target_cam_map
        hit_masks[view_idx] = hit_mask
        real_cam_maps.append(cam_map)
        per_view_summary.append(
            {
                "view_index": int(view_idx),
                "camera_id": camera_id,
                "roi_pixels": int(roi_arr[view_idx].sum()),
                "hit_pixels": int(hit_mask.sum()),
                "hit_ratio_in_roi": float(hit_mask.sum() / max(1, int(roi_arr[view_idx].sum()))),
            }
        )

    target_hits = hit_masks[0]
    target_roi = roi_arr[0]
    target_valid_base = (
        target_hits
        & target_roi
        & np.isfinite(base_world[0]).all(axis=-1)
        & (base_conf[0] >= np.percentile(base_conf[0][masks_arr[0]], float(args.conf_percentile)))
    )
    source_corr = view_real_target_cam[0][target_valid_base]
    target_corr = base_world[0][target_valid_base]
    transform_summary, transform_matrix = robust_transform(
        source_corr,
        target_corr,
        mode=args.transform_mode,
        max_correspondences=int(args.max_correspondences),
        seed=int(args.seed),
    )
    if args.transform_mode == "similarity":
        teacher_world = apply_similarity(
            view_real_target_cam,
            float(transform_summary["scale"]),
            np.asarray(transform_summary["rotation"], dtype=np.float64),
            np.asarray(transform_summary["translation"], dtype=np.float64),
        )
    else:
        teacher_world = apply_axis_affine(
            view_real_target_cam,
            np.asarray(transform_summary["scale_xyz"], dtype=np.float64),
            np.asarray(transform_summary["translation_xyz"], dtype=np.float64),
        )
    teacher_world[~hit_masks] = 0.0

    distance_to_base = np.linalg.norm(teacher_world - base_world, axis=-1)
    distance_valid = hit_masks & np.isfinite(distance_to_base)
    distance_percentiles = []
    if distance_valid.any():
        distance_percentiles = [float(v) for v in np.percentile(distance_to_base[distance_valid], [0, 25, 50, 75, 90, 95, 99])]

    teacher_targets_path = output_dir / "teacher_targets.npz"
    np.savez_compressed(
        teacher_targets_path,
        world_points=teacher_world.astype(np.float32),
        teacher_mask=hit_masks.astype(bool),
        real_world_points=view_real_world.astype(np.float32),
        real_target_cam_points=view_real_target_cam.astype(np.float32),
        roi_mask=roi_arr.astype(bool),
        kinect_camera_ids=kinect_camera_ids,
        transform_matrix_real_targetcam_to_vggt_world=transform_matrix.astype(np.float32),
    )

    overlay_dir = output_dir / "overlays"
    for view_idx, view in enumerate(scene_manifest["exported_views"]):
        make_overlay(
            Path(view["image_path"]),
            roi_arr[view_idx],
            hit_masks[view_idx],
            overlay_dir / f"{view_idx:02d}_cam{normalize_camera_id(view['camera_id'])}_{args.roi_kind}_kinect_hits.png",
        )

    hit_points = teacher_world[hit_masks]
    hit_colors = np.full((hit_points.shape[0], 3), [40, 220, 90], dtype=np.uint8)
    if hit_points.shape[0] > 0:
        write_ply(output_dir / "kinect_teacher_vggt_world_hits.ply", hit_points, hit_colors)

    summary = {
        "task": "kinect_depth_teacher_targets",
        "truthful_status": "gate_candidate_not_final_pass",
        "scene_dir": str(scene_dir),
        "base_predictions": str(args.base_predictions.resolve()),
        "output_dir": str(output_dir),
        "teacher_targets": str(teacher_targets_path),
        "kinect_smc": str(kinect_smc),
        "frame": int(frame),
        "roi_kind": str(args.roi_kind),
        "transform_mode": str(args.transform_mode),
        "kinect_stats": kinect_stats,
        "per_view": per_view_summary,
        "target_alignment": transform_summary,
        "distance_to_base_on_teacher_mask_percentiles": distance_percentiles,
        "notes": [
            "Kinect depth is real metric depth but low resolution; this gate only proves projected coverage/alignment.",
            "Teacher points are transformed into VGGT prediction world coordinates using target-view correspondences.",
            "Do not claim mentor-final quality before same-protocol ROI and Open3D close-ups improve visibly.",
        ],
    }
    (output_dir / "kinect_teacher_summary.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
