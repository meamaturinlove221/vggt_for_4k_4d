from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt.utils.normal_refiner import face_box_from_mask, head_box_from_mask


POINT_SOURCES = ("world_points", "depth_unprojection")
ROI_MODES = ("full", "head", "face")
ROI_SOURCES = ("3d", "2d")


def preprocess_rgb(path: Path, target_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    width, height = img.size
    if width >= height:
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
    else:
        new_height = target_size
        new_width = round(width * (new_height / height) / 14) * 14

    img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)

    canvas = np.full((target_size, target_size, 3), 255, dtype=np.uint8)
    top = (target_size - new_height) // 2
    left = (target_size - new_width) // 2
    canvas[top : top + new_height, left : left + new_width] = arr
    return canvas


def load_rgb_stack(image_dir: Path, target_size: int) -> np.ndarray:
    image_paths = sorted(path for path in image_dir.iterdir() if path.is_file())
    images = [preprocess_rgb(path, target_size=target_size) for path in image_paths]
    return np.stack(images, axis=0)


def preprocess_mask(mask_path: Path, target_size: int) -> np.ndarray:
    img = Image.open(mask_path).convert("L")
    width, height = img.size
    if width >= height:
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
    else:
        new_height = target_size
        new_width = round(width * (new_height / height) / 14) * 14

    img = img.resize((new_width, new_height), Image.Resampling.NEAREST)
    arr = np.asarray(img, dtype=np.uint8)

    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    top = (target_size - new_height) // 2
    left = (target_size - new_width) // 2
    canvas[top : top + new_height, left : left + new_width] = arr
    return canvas


def load_mask_stack(mask_dir: Path, target_size: int) -> np.ndarray:
    mask_paths = sorted(path for path in mask_dir.iterdir() if path.is_file())
    masks = [preprocess_mask(path, target_size=target_size) for path in mask_paths]
    return np.stack(masks, axis=0)


def mask_to_2d_roi(mask: np.ndarray, roi: str) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if roi == "full":
        return mask
    box = head_box_from_mask(mask) if roi == "head" else face_box_from_mask(mask)
    out = np.zeros_like(mask, dtype=bool)
    if box is None:
        return out
    x0, y0, x1, y1 = box
    x0 = max(0, min(mask.shape[1], int(x0)))
    x1 = max(0, min(mask.shape[1], int(x1)))
    y0 = max(0, min(mask.shape[0], int(y0)))
    y1 = max(0, min(mask.shape[0], int(y1)))
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return out


def load_2d_roi_mask_stack(mask_dir: Path, target_size: int, roi: str) -> np.ndarray:
    masks = load_mask_stack(mask_dir, target_size=target_size)
    return np.stack([mask_to_2d_roi(mask, roi=roi) for mask in masks], axis=0)


def closed_form_inverse_se3_numpy(se3: np.ndarray) -> np.ndarray:
    rotation = se3[:, :3, :3]
    translation = se3[:, :3, 3:]
    rotation_t = np.transpose(rotation, (0, 2, 1))
    top_right = -np.matmul(rotation_t, translation)
    inverted = np.tile(np.eye(4, dtype=se3.dtype), (len(rotation), 1, 1))
    inverted[:, :3, :3] = rotation_t
    inverted[:, :3, 3:] = top_right
    return inverted


def unproject_depth_map_to_point_map_numpy(
    depth_map: np.ndarray,
    extrinsics_cam: np.ndarray,
    intrinsics_cam: np.ndarray,
) -> np.ndarray:
    world_points = []
    cam_to_world = closed_form_inverse_se3_numpy(extrinsics_cam)
    for frame_idx in range(depth_map.shape[0]):
        depth = depth_map[frame_idx].squeeze(-1).astype(np.float32)
        intrinsic = intrinsics_cam[frame_idx].astype(np.float32)

        height, width = depth.shape
        u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))

        fu, fv = intrinsic[0, 0], intrinsic[1, 1]
        cu, cv = intrinsic[0, 2], intrinsic[1, 2]

        x_cam = (u - cu) * depth / fu
        y_cam = (v - cv) * depth / fv
        z_cam = depth
        cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1)

        rotation = cam_to_world[frame_idx, :3, :3]
        translation = cam_to_world[frame_idx, :3, 3]
        world = np.dot(cam_coords, rotation.T) + translation
        world_points.append(world.astype(np.float32))

    return np.stack(world_points, axis=0)


def resolve_point_source(data: np.lib.npyio.NpzFile, point_source: str) -> tuple[np.ndarray, np.ndarray]:
    if point_source == "world_points":
        return data["world_points"], data["world_points_conf"]
    if point_source == "depth_unprojection":
        world_points = unproject_depth_map_to_point_map_numpy(data["depth"], data["extrinsic"], data["intrinsic"])
        return world_points, data["depth_conf"]
    raise ValueError(f"Unsupported point source: {point_source}")


def build_filtered_cloud(
    world_points: np.ndarray,
    world_points_conf: np.ndarray,
    colors: np.ndarray,
    masks: np.ndarray | None,
    max_points: int,
    conf_percentile: float,
    conf_threshold_override: float | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    points = world_points.reshape(-1, 3)
    conf = world_points_conf.reshape(-1)
    rgb = colors.reshape(-1, 3)

    valid = np.isfinite(points).all(axis=1) & np.isfinite(conf) & (conf > 0)
    if masks is not None:
        valid &= masks.reshape(-1) > 0

    if not np.any(valid):
        raise RuntimeError("No valid points after filtering.")

    conf_valid = conf[valid]
    conf_threshold = (
        float(conf_threshold_override)
        if conf_threshold_override is not None
        else float(np.percentile(conf_valid, conf_percentile))
    )
    keep = valid & (conf >= conf_threshold)
    if not np.any(keep):
        keep = valid

    kept_indices = np.flatnonzero(keep)
    if len(kept_indices) > max_points:
        kept_indices = rng.choice(kept_indices, size=max_points, replace=False)

    kept_points = points[kept_indices]
    kept_rgb = rgb[kept_indices]
    summary = {
        "valid_points_before_conf": int(valid.sum()),
        "conf_threshold": conf_threshold,
        "conf_threshold_source": "absolute" if conf_threshold_override is not None else "percentile",
        "conf_percentile": float(conf_percentile),
        "points_after_conf": int(keep.sum()),
        "points_written": int(len(kept_indices)),
    }
    return kept_points, kept_rgb, summary


def apply_roi_filter(
    points: np.ndarray,
    colors: np.ndarray,
    roi: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
    if roi == "full":
        return points, colors, {"roi": roi, "points_after_roi": int(len(points))}

    if len(points) < 32:
        return points, colors, {"roi": roi, "fallback": "too_few_points", "points_after_roi": int(len(points))}

    # Open3D renders in this project consistently use up=(0, -1, 0), so smaller
    # world-space y appears higher on screen. Convert to a height-like scalar
    # where larger values mean "higher on the body" before selecting head/face.
    height_like = -points[:, 1]
    head_percentile = 78.0 if roi == "head" else 74.0
    head_cut = float(np.percentile(height_like, head_percentile))
    head_mask = height_like >= head_cut
    if int(head_mask.sum()) < 512:
        relaxed_cut = float(np.percentile(height_like, 68.0))
        head_mask = height_like >= relaxed_cut
        head_cut = relaxed_cut

    roi_mask = head_mask
    summary: dict[str, float | int | str] = {
        "roi": roi,
        "vertical_axis": "-y_is_up",
        "head_cut_height_like": head_cut,
        "points_after_head_cut": int(head_mask.sum()),
    }

    if roi == "face":
        head_points = points[head_mask]
        if len(head_points) >= 256:
            x_lo, x_hi = np.percentile(head_points[:, 0], [20.0, 80.0])
            z_lo, z_hi = np.percentile(head_points[:, 2], [15.0, 85.0])
            head_height_like = -head_points[:, 1]
            height_lo = float(np.percentile(head_height_like, 25.0))
            face_mask = (
                head_mask
                & (points[:, 0] >= float(x_lo))
                & (points[:, 0] <= float(x_hi))
                & (points[:, 2] >= float(z_lo))
                & (points[:, 2] <= float(z_hi))
                & (height_like >= height_lo)
            )
            if int(face_mask.sum()) >= 128:
                roi_mask = face_mask
                summary.update(
                    {
                        "x_lo": float(x_lo),
                        "x_hi": float(x_hi),
                        "z_lo": float(z_lo),
                        "z_hi": float(z_hi),
                        "face_height_like_lo": height_lo,
                    }
                )
            else:
                summary["fallback"] = "face_mask_too_small"
        else:
            summary["fallback"] = "head_mask_too_small"

    filtered_points = points[roi_mask]
    filtered_colors = colors[roi_mask]
    summary["points_after_roi"] = int(len(filtered_points))
    return filtered_points, filtered_colors, summary


def _load_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Open3D is required for render_open3d_pointcloud.py. "
            "Install it in the active environment, or run the project wrapper "
            "`scripts/render_pointcloud_open3d.ps1` which defaults to the "
            "`g3splat` conda env that already has Open3D."
        ) from exc
    return o3d


def _camera_presets(center: np.ndarray, radius: float, roi: str) -> list[tuple[str, dict[str, object]]]:
    front_zoom = {"full": 0.55, "head": 0.72, "face": 0.82}[roi]
    side_zoom = {"full": 0.55, "head": 0.72, "face": 0.82}[roi]
    top_zoom = {"full": 0.55, "head": 0.70, "face": 0.80}[roi]
    iso_zoom = {"full": 0.52, "head": 0.70, "face": 0.80}[roi]
    head_close_zoom = {"full": 0.78, "head": 0.88, "face": 0.92}[roi]
    face_close_zoom = {"full": 0.95, "head": 0.92, "face": 0.97}[roi]
    full_head_lookat = center + np.array([0.0, -0.10 * radius, 0.10 * radius], dtype=np.float32)
    full_face_lookat = center + np.array([0.0, -0.18 * radius, 0.18 * radius], dtype=np.float32)
    head_lookat = full_head_lookat if roi == "full" else center
    face_lookat = full_face_lookat if roi == "full" else center
    return [
        (
            "front",
            {
                "front": [0.0, 0.0, -1.0],
                "lookat": center.tolist(),
                "up": [0.0, -1.0, 0.0],
                "zoom": front_zoom,
            },
        ),
        (
            "side",
            {
                "front": [1.0, 0.0, 0.0],
                "lookat": center.tolist(),
                "up": [0.0, -1.0, 0.0],
                "zoom": side_zoom,
            },
        ),
        (
            "top",
            {
                "front": [0.0, -1.0, 0.0],
                "lookat": center.tolist(),
                "up": [0.0, 0.0, -1.0],
                "zoom": top_zoom,
            },
        ),
        (
            "iso",
            {
                "front": [0.65, -0.25, -0.72],
                "lookat": center.tolist(),
                "up": [0.0, -1.0, 0.0],
                "zoom": iso_zoom,
            },
        ),
        (
            "head_close",
            {
                "front": [0.08, -0.02, -0.996],
                "lookat": head_lookat.tolist(),
                "up": [0.0, -1.0, 0.0],
                "zoom": head_close_zoom,
            },
        ),
        (
            "face_close",
            {
                "front": [0.15, -0.05, -0.99],
                "lookat": face_lookat.tolist(),
                "up": [0.0, -1.0, 0.0],
                "zoom": face_close_zoom,
            },
        ),
    ]


def _save_open3d_renders(
    points: np.ndarray,
    colors: np.ndarray,
    output_dir: Path,
    roi: str,
    width: int,
    height: int,
    point_size: float,
    interactive: bool,
) -> list[str]:
    o3d = _load_open3d()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector((colors.astype(np.float32) / 255.0).clip(0.0, 1.0).astype(np.float64))

    bounds = pcd.get_axis_aligned_bounding_box()
    center = np.asarray(bounds.get_center(), dtype=np.float32)
    extent = np.asarray(bounds.get_extent(), dtype=np.float32)
    radius = float(np.linalg.norm(extent) + 1e-6)

    if interactive:  # pragma: no cover - requires GUI
        o3d.visualization.draw_geometries([pcd], window_name="VGGT Human Point Cloud", width=width, height=height)
        return []

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="VGGT Open3D Render", width=width, height=height, visible=False)
    vis.add_geometry(pcd)

    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    render_option.point_size = float(point_size)
    render_option.light_on = True

    ctr = vis.get_view_control()
    saved = []
    for name, preset in _camera_presets(center=center, radius=radius, roi=roi):
        ctr.set_front(preset["front"])
        ctr.set_lookat(preset["lookat"])
        ctr.set_up(preset["up"])
        ctr.set_zoom(float(preset["zoom"]))
        vis.poll_events()
        vis.update_renderer()
        output_path = output_dir / f"{name}.png"
        vis.capture_screen_image(str(output_path), do_render=True)
        saved.append(str(output_path))

    vis.destroy_window()
    return saved


def _parse_camera_view_indices(value: str, num_views: int) -> list[int]:
    if not value:
        return []
    indices = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        index = int(item)
        if index < 0 or index >= num_views:
            raise ValueError(f"camera view index out of range: {index}; num_views={num_views}")
        indices.append(index)
    return indices


def _save_open3d_camera_renders(
    points: np.ndarray,
    colors: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    output_dir: Path,
    camera_indices: list[int],
    point_size: float,
    render_size: int,
) -> list[str]:
    if not camera_indices:
        return []

    o3d = _load_open3d()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector((colors.astype(np.float32) / 255.0).clip(0.0, 1.0).astype(np.float64))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="VGGT Camera-Aligned Open3D Render", width=render_size, height=render_size, visible=False)
    vis.add_geometry(pcd)

    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    render_option.point_size = float(point_size)
    render_option.light_on = True

    saved = []
    ctr = vis.get_view_control()
    for index in camera_indices:
        params = o3d.camera.PinholeCameraParameters()
        params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            render_size,
            render_size,
            float(intrinsic[index, 0, 0]),
            float(intrinsic[index, 1, 1]),
            float(intrinsic[index, 0, 2]),
            float(intrinsic[index, 1, 2]),
        )
        camera_extrinsic = np.eye(4, dtype=np.float64)
        camera_extrinsic[:3, :4] = extrinsic[index].astype(np.float64)
        params.extrinsic = camera_extrinsic
        try:
            ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        except TypeError:
            ctr.convert_from_pinhole_camera_parameters(params)
        vis.poll_events()
        vis.update_renderer()
        output_path = output_dir / f"camera_view_{index:02d}.png"
        vis.capture_screen_image(str(output_path), do_render=True)
        saved.append(str(output_path))
        cropped_path = output_dir / f"camera_view_{index:02d}_crop.png"
        if _save_nonwhite_crop(output_path, cropped_path, render_size):
            saved.append(str(cropped_path))

    vis.destroy_window()
    return saved


def _save_nonwhite_crop(input_path: Path, output_path: Path, output_size: int) -> bool:
    image = Image.open(input_path).convert("RGB")
    arr = np.asarray(image, dtype=np.uint8)
    nonwhite = np.any(arr < 248, axis=-1)
    if int(nonwhite.sum()) < 16:
        return False
    ys, xs = np.nonzero(nonwhite)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pad = max(8, int(0.12 * max(x1 - x0, y1 - y0)))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(arr.shape[1], x1 + pad)
    y1 = min(arr.shape[0], y1 + pad)
    crop = image.crop((x0, y0, x1, y1))
    scale = min(output_size / max(1, crop.size[0]), output_size / max(1, crop.size[1]))
    resized_size = (
        max(1, int(round(crop.size[0] * scale))),
        max(1, int(round(crop.size[1] * scale))),
    )
    crop = crop.resize(resized_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (output_size, output_size), "white")
    left = (output_size - crop.size[0]) // 2
    top = (output_size - crop.size[1]) // 2
    canvas.paste(crop, (left, top))
    canvas.save(output_path)
    return True


def _save_projection_fallback(
    points: np.ndarray,
    colors: np.ndarray,
    output_dir: Path,
    roi: str,
    width: int,
    height: int,
) -> list[str]:
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if len(points) == 0:
        return []

    center = np.median(points, axis=0)
    centered = points - center[None, :]
    views = {
        "front_fallback": (centered[:, 0], -centered[:, 1], centered[:, 2]),
        "side_fallback": (centered[:, 2], -centered[:, 1], centered[:, 0]),
        "top_fallback": (centered[:, 0], centered[:, 2], -centered[:, 1]),
    }
    saved = []
    for name, (axis_x, axis_y, depth) in views.items():
        order = np.argsort(depth)
        x = axis_x[order]
        y = axis_y[order]
        rgb = colors[order]
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        rgb = rgb[finite]
        if len(x) == 0:
            continue
        lo_x, hi_x = np.percentile(x, [1.0, 99.0])
        lo_y, hi_y = np.percentile(y, [1.0, 99.0])
        pad_x = max(1e-6, float(hi_x - lo_x) * 0.08)
        pad_y = max(1e-6, float(hi_y - lo_y) * 0.08)
        lo_x -= pad_x
        hi_x += pad_x
        lo_y -= pad_y
        hi_y += pad_y
        px = np.clip(((x - lo_x) / max(1e-6, hi_x - lo_x) * (width - 1)).round().astype(np.int32), 0, width - 1)
        py = np.clip(((1.0 - (y - lo_y) / max(1e-6, hi_y - lo_y)) * (height - 1)).round().astype(np.int32), 0, height - 1)
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        canvas[py, px] = rgb
        output_path = output_dir / f"{name}.png"
        Image.fromarray(canvas).save(output_path)
        saved.append(str(output_path))

    alias = {
        "front_fallback": "front.png",
        "side_fallback": "side.png",
        "top_fallback": "top.png",
    }
    for src_name, dst_name in alias.items():
        src = output_dir / f"{src_name}.png"
        if src.exists() and not (output_dir / dst_name).exists():
            Image.open(src).save(output_dir / dst_name)
    front = output_dir / "front_fallback.png"
    if front.exists():
        for close_name in ("head_close.png", "face_close.png"):
            dst = output_dir / close_name
            if not dst.exists():
                Image.open(front).save(dst)
                saved.append(str(dst))
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render VGGT point clouds with Open3D for clearer human-region visualization.")
    parser.add_argument("--predictions-npz", required=True, help="Path to predictions.npz")
    parser.add_argument("--scene-dir", required=True, help="Scene directory containing images/ and optionally masks/")
    parser.add_argument("--output-dir", required=True, help="Output directory for PLY and Open3D screenshots")
    parser.add_argument(
        "--point-source",
        choices=POINT_SOURCES,
        default="world_points",
        help="3D point source: precomputed world_points or depth+camera unprojection.",
    )
    parser.add_argument("--max-points", type=int, default=300000, help="Maximum points to keep after filtering")
    parser.add_argument("--conf-percentile", type=float, default=40.0, help="Keep points at or above this confidence percentile")
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=None,
        help="Optional absolute confidence threshold; when set, overrides --conf-percentile for reproducible same-threshold comparisons.",
    )
    parser.add_argument("--human-only", action="store_true", help="Filter the point cloud using scene masks if available")
    parser.add_argument(
        "--roi",
        choices=ROI_MODES,
        default="full",
        help="Optional 3D ROI filter before rendering: full body, head crop, or tighter face crop.",
    )
    parser.add_argument(
        "--roi-source",
        choices=ROI_SOURCES,
        default="3d",
        help=(
            "Use the legacy 3D percentile ROI filter, or apply a per-view 2D mask ROI before "
            "flattening points. 2d is useful for camera-aligned human face/head evidence."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for subsampling")
    parser.add_argument("--width", type=int, default=1600, help="Render width")
    parser.add_argument("--height", type=int, default=1200, help="Render height")
    parser.add_argument("--point-size", type=float, default=2.0, help="Open3D point size")
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.0,
        help="Optional Open3D voxel downsample size after ROI filtering; 0 disables it.",
    )
    parser.add_argument(
        "--statistical-outlier-neighbors",
        type=int,
        default=0,
        help="Optional Open3D statistical outlier removal neighbor count; 0 disables it.",
    )
    parser.add_argument(
        "--statistical-outlier-std-ratio",
        type=float,
        default=1.5,
        help="Std-ratio for statistical outlier removal when enabled.",
    )
    parser.add_argument(
        "--radius-outlier-nb-points",
        type=int,
        default=0,
        help="Optional radius outlier removal minimum neighbors; 0 disables it.",
    )
    parser.add_argument(
        "--radius-outlier-radius",
        type=float,
        default=0.01,
        help="Radius for radius outlier removal when enabled.",
    )
    parser.add_argument(
        "--camera-view-indices",
        default="",
        help="Optional comma-separated input view indices for camera-aligned Open3D renders.",
    )
    parser.add_argument("--interactive", action="store_true", help="Open an interactive Open3D viewer instead of offscreen screenshots")
    parser.add_argument("--projection-only", action="store_true", help="Skip Open3D screen capture and write crash-safe 2D point projections")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = np.load(args.predictions_npz, allow_pickle=False)
    world_points, world_points_conf = resolve_point_source(predictions, args.point_source)
    camera_view_indices = _parse_camera_view_indices(str(args.camera_view_indices), num_views=int(world_points.shape[0]))

    scene_dir = Path(args.scene_dir).resolve()
    target_size = int(world_points.shape[1])
    colors = load_rgb_stack(scene_dir / "images", target_size=target_size)
    masks = None
    roi_mask_pixels = None
    if args.human_only and (scene_dir / "masks").is_dir():
        if args.roi_source == "2d":
            masks = load_2d_roi_mask_stack(scene_dir / "masks", target_size=target_size, roi=str(args.roi))
            roi_mask_pixels = int(masks.sum())
        else:
            masks = load_mask_stack(scene_dir / "masks", target_size=target_size)
    elif args.roi_source == "2d" and (scene_dir / "masks").is_dir():
        masks = load_2d_roi_mask_stack(scene_dir / "masks", target_size=target_size, roi=str(args.roi))
        roi_mask_pixels = int(masks.sum())

    rng = np.random.default_rng(args.seed)
    points, rgb, summary = build_filtered_cloud(
        world_points=world_points,
        world_points_conf=world_points_conf,
        colors=colors,
        masks=masks,
        max_points=int(args.max_points),
        conf_percentile=float(args.conf_percentile),
        conf_threshold_override=None if args.conf_threshold is None else float(args.conf_threshold),
        rng=rng,
    )
    if args.roi_source == "2d":
        roi_summary = {
            "roi": str(args.roi),
            "roi_source": "2d_mask",
            "roi_mask_pixels": int(roi_mask_pixels or 0),
            "points_after_roi": int(len(points)),
        }
    else:
        points, rgb, roi_summary = apply_roi_filter(points=points, colors=rgb, roi=str(args.roi))
    if len(points) == 0:
        raise RuntimeError(f"No points left after applying roi={args.roi}")

    o3d = _load_open3d()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector((rgb.astype(np.float32) / 255.0).astype(np.float64))
    postprocess_summary = {
        "points_before_postprocess": int(len(points)),
        "voxel_size": float(args.voxel_size),
        "statistical_outlier_neighbors": int(args.statistical_outlier_neighbors),
        "statistical_outlier_std_ratio": float(args.statistical_outlier_std_ratio),
        "radius_outlier_nb_points": int(args.radius_outlier_nb_points),
        "radius_outlier_radius": float(args.radius_outlier_radius),
    }
    if float(args.voxel_size) > 0.0 and len(pcd.points) > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(args.voxel_size))
        postprocess_summary["points_after_voxel_downsample"] = int(len(pcd.points))
    if int(args.statistical_outlier_neighbors) > 0 and len(pcd.points) > int(args.statistical_outlier_neighbors):
        pcd, inlier_indices = pcd.remove_statistical_outlier(
            nb_neighbors=int(args.statistical_outlier_neighbors),
            std_ratio=float(args.statistical_outlier_std_ratio),
        )
        postprocess_summary["points_after_statistical_outlier"] = int(len(pcd.points))
        postprocess_summary["statistical_inlier_count"] = int(len(inlier_indices))
    if int(args.radius_outlier_nb_points) > 0 and len(pcd.points) > int(args.radius_outlier_nb_points):
        pcd, inlier_indices = pcd.remove_radius_outlier(
            nb_points=int(args.radius_outlier_nb_points),
            radius=float(args.radius_outlier_radius),
        )
        postprocess_summary["points_after_radius_outlier"] = int(len(pcd.points))
        postprocess_summary["radius_inlier_count"] = int(len(inlier_indices))
    if len(pcd.points) == 0:
        raise RuntimeError("No points left after Open3D post-processing filters.")
    points = np.asarray(pcd.points, dtype=np.float32)
    rgb = np.clip(np.asarray(pcd.colors, dtype=np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)
    postprocess_summary["points_after_postprocess"] = int(len(points))
    ply_path = output_dir / "pointcloud_open3d.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd, write_ascii=False, compressed=False)

    render_backend = "projection_only" if args.projection_only else "open3d_visualizer"
    if args.projection_only:
        screenshots = _save_projection_fallback(
            points=points,
            colors=rgb,
            output_dir=output_dir,
            roi=str(args.roi),
            width=int(args.width),
            height=int(args.height),
        )
    else:
        try:
            screenshots = _save_open3d_renders(
                points=points,
                colors=rgb,
                output_dir=output_dir,
                roi=str(args.roi),
                width=int(args.width),
                height=int(args.height),
                point_size=float(args.point_size),
                interactive=bool(args.interactive),
            )
            if camera_view_indices:
                screenshots.extend(
                    _save_open3d_camera_renders(
                        points=points,
                        colors=rgb,
                        extrinsic=np.asarray(predictions["extrinsic"], dtype=np.float32),
                        intrinsic=np.asarray(predictions["intrinsic"], dtype=np.float32),
                        output_dir=output_dir,
                        camera_indices=camera_view_indices,
                        point_size=float(args.point_size),
                        render_size=int(world_points.shape[1]),
                    )
                )
        except Exception as exc:
            render_backend = f"projection_fallback_after_open3d_error:{type(exc).__name__}:{exc}"
            screenshots = _save_projection_fallback(
                points=points,
                colors=rgb,
                output_dir=output_dir,
                roi=str(args.roi),
                width=int(args.width),
                height=int(args.height),
            )
        if not screenshots or not all(Path(path).is_file() for path in screenshots):
            render_backend = "projection_fallback_after_missing_open3d_screenshots"
            screenshots = _save_projection_fallback(
                points=points,
                colors=rgb,
                output_dir=output_dir,
                roi=str(args.roi),
                width=int(args.width),
                height=int(args.height),
            )

    payload = {
        "predictions_npz": str(Path(args.predictions_npz).resolve()),
        "scene_dir": str(scene_dir),
        "output_dir": str(output_dir),
        "point_source": args.point_source,
        "human_only": bool(args.human_only),
        "roi": args.roi,
        "roi_source": args.roi_source,
        "summary": summary,
        "roi_summary": roi_summary,
        "postprocess_summary": postprocess_summary,
        "ply_path": str(ply_path),
        "screenshots": screenshots,
        "render_backend": render_backend,
    }
    (output_dir / "open3d_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
