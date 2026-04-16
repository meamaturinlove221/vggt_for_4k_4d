from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt.utils.load_fn import load_and_preprocess_images


TARGET_SIZE = 518
POINT_SOURCES = ("world_points", "depth_unprojection")


def load_rgb_stack(image_dir: Path) -> np.ndarray:
    image_paths = sorted(path for path in image_dir.iterdir() if path.is_file())
    images = load_and_preprocess_images([str(path) for path in image_paths], mode="pad")
    images = (images.permute(0, 2, 3, 1).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return images


def preprocess_mask(mask_path: Path, target_size: int = TARGET_SIZE) -> np.ndarray:
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


def load_mask_stack(mask_dir: Path) -> np.ndarray:
    mask_paths = sorted(path for path in mask_dir.iterdir() if path.is_file())
    masks = [preprocess_mask(path) for path in mask_paths]
    return np.stack(masks, axis=0)


def write_ascii_ply(points: np.ndarray, colors: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def camera_centers(extrinsic: np.ndarray) -> np.ndarray:
    centers = []
    for camera in extrinsic:
        rotation = camera[:, :3]
        translation = camera[:, 3]
        center = -(rotation.T @ translation)
        centers.append(center)
    return np.asarray(centers, dtype=np.float32)


def render_views(
    points: np.ndarray,
    colors: np.ndarray,
    camera_xyz: np.ndarray,
    output_path: Path,
    title: str,
    figure_width: float,
    figure_height: float,
    figure_dpi: int,
    point_size: float,
    point_alpha: float,
    camera_size: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(figure_width, figure_height), dpi=figure_dpi)
    views = [
        ("Front (X/Y)", 0, 1),
        ("Side (Z/Y)", 2, 1),
        ("Top (X/Z)", 0, 2),
    ]
    norm_colors = colors.astype(np.float32) / 255.0
    for ax, (label, ax_x, ax_y) in zip(axes, views):
        ax.scatter(points[:, ax_x], points[:, ax_y], s=point_size, c=norm_colors, linewidths=0, alpha=point_alpha)
        ax.scatter(
            camera_xyz[:, ax_x],
            camera_xyz[:, ax_y],
            s=camera_size,
            c="red",
            marker="^",
            edgecolors="white",
            linewidths=0.4,
            alpha=0.9,
        )
        ax.set_title(label)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.15)
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_filtered_cloud(
    world_points: np.ndarray,
    world_points_conf: np.ndarray,
    colors: np.ndarray,
    masks: np.ndarray | None,
    max_points: int,
    conf_percentile: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    points = world_points.reshape(-1, 3)
    conf = world_points_conf.reshape(-1)
    rgb = colors.reshape(-1, 3)

    valid = np.isfinite(points).all(axis=1) & np.isfinite(conf) & (conf > 0)
    if masks is not None:
        mask_flat = (masks.reshape(-1) > 0)
        valid &= mask_flat

    if not np.any(valid):
        raise RuntimeError("No valid points after filtering.")

    conf_valid = conf[valid]
    conf_threshold = float(np.percentile(conf_valid, conf_percentile))
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
        "points_after_conf": int(keep.sum()),
        "points_written": int(len(kept_indices)),
    }
    return kept_points, kept_rgb, summary


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render fused point cloud outputs from VGGT predictions.")
    parser.add_argument("--predictions-npz", required=True, help="Path to predictions.npz")
    parser.add_argument("--scene-dir", required=True, help="Scene directory containing images/ and optionally masks/")
    parser.add_argument("--output-dir", required=True, help="Output directory for point cloud artifacts")
    parser.add_argument(
        "--point-source",
        choices=POINT_SOURCES,
        default="world_points",
        help="3D point source: precomputed world_points or depth+camera unprojection.",
    )
    parser.add_argument("--max-points", type=int, default=180000, help="Maximum points to write per cloud")
    parser.add_argument(
        "--conf-percentile",
        type=float,
        default=70.0,
        help="Keep points at or above this confidence percentile among valid points.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for subsampling")
    parser.add_argument("--figure-width", type=float, default=18.0, help="Figure width in inches.")
    parser.add_argument("--figure-height", type=float, default=6.0, help="Figure height in inches.")
    parser.add_argument("--figure-dpi", type=int, default=180, help="Figure DPI for static renders.")
    parser.add_argument("--point-size", type=float, default=0.15, help="Scatter point size for the cloud render.")
    parser.add_argument("--point-alpha", type=float, default=0.65, help="Scatter alpha for the cloud render.")
    parser.add_argument("--camera-size", type=float, default=24.0, help="Camera marker size in the render.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.predictions_npz)
    extrinsic = data["extrinsic"]
    world_points, world_points_conf = resolve_point_source(data, args.point_source)

    scene_dir = Path(args.scene_dir).resolve()
    colors = load_rgb_stack(scene_dir / "images")
    masks = load_mask_stack(scene_dir / "masks") if (scene_dir / "masks").is_dir() else None
    camera_xyz = camera_centers(extrinsic)
    rng = np.random.default_rng(args.seed)

    raw_points, raw_colors, raw_summary = build_filtered_cloud(
        world_points,
        world_points_conf,
        colors,
        masks=None,
        max_points=args.max_points,
        conf_percentile=args.conf_percentile,
        rng=rng,
    )
    write_ascii_ply(raw_points, raw_colors, output_dir / "fused_pointcloud_raw.ply")
    render_views(
        raw_points,
        raw_colors,
        camera_xyz,
        output_dir / "fused_pointcloud_raw_views.png",
        f"VGGT Fused Point Cloud ({args.point_source}, raw)",
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        figure_dpi=args.figure_dpi,
        point_size=args.point_size,
        point_alpha=args.point_alpha,
        camera_size=args.camera_size,
    )

    summary = {
        "point_source": args.point_source,
        "num_views": int(world_points.shape[0]),
        "input_image_shape": list(colors.shape),
        "conf_percentile": float(args.conf_percentile),
        "max_points": int(args.max_points),
        "raw": raw_summary,
    }

    if masks is not None:
        masked_points, masked_colors, masked_summary = build_filtered_cloud(
            world_points,
            world_points_conf,
            colors,
            masks=masks,
            max_points=args.max_points,
            conf_percentile=args.conf_percentile,
            rng=rng,
        )
        write_ascii_ply(masked_points, masked_colors, output_dir / "fused_pointcloud_masked.ply")
        render_views(
            masked_points,
            masked_colors,
            camera_xyz,
            output_dir / "fused_pointcloud_masked_views.png",
            f"VGGT Fused Point Cloud ({args.point_source}, mask filtered)",
            figure_width=args.figure_width,
            figure_height=args.figure_height,
            figure_dpi=args.figure_dpi,
            point_size=args.point_size,
            point_alpha=args.point_alpha,
            camera_size=args.camera_size,
        )
        summary["masked"] = masked_summary

    with (output_dir / "pointcloud_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Wrote point cloud artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
