from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


POINT_CLOUD_EXTENSIONS = {".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".pts"}
PREFERRED_FILENAMES = (
    "fused_pointcloud_masked.ply",
    "fused_pointcloud_raw.ply",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open a point cloud in an Open3D desktop window.")
    parser.add_argument("input_path", nargs="?", help="Point cloud file or a folder containing one.")
    parser.add_argument("--point-size", type=float, default=2.0, help="Rendered point size.")
    parser.add_argument("--width", type=int, default=1600, help="Window width.")
    parser.add_argument("--height", type=int, default=900, help="Window height.")
    parser.add_argument(
        "--background",
        nargs=3,
        type=float,
        metavar=("R", "G", "B"),
        default=(0.04, 0.04, 0.04),
        help="Background RGB color in [0, 1].",
    )
    parser.add_argument("--no-axis", action="store_true", help="Hide the coordinate frame.")
    return parser.parse_args()


def resolve_point_cloud_path(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    for filename in PREFERRED_FILENAMES:
        candidate = input_path / filename
        if candidate.is_file():
            return candidate

    candidates = sorted(
        path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() in POINT_CLOUD_EXTENSIONS
    )
    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"No point cloud file found under: {input_path}")


def format_vec3(vector: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in vector.tolist()) + "]"


def main() -> int:
    args = parse_args()
    input_arg = args.input_path or str(
        Path(
            r"D:\vggt\vggt-main\output\modal_results\0012_11_frame0000_60views_smplxsurfacepose_a10080_e2_r2\pointcloud_dense_p40_hires"
        )
    )
    cloud_path = resolve_point_cloud_path(Path(input_arg).expanduser().resolve())

    point_cloud = o3d.io.read_point_cloud(str(cloud_path))
    if point_cloud.is_empty():
        raise RuntimeError(f"Loaded point cloud is empty: {cloud_path}")

    points = np.asarray(point_cloud.points)
    min_bound = point_cloud.get_min_bound()
    max_bound = point_cloud.get_max_bound()
    extent = max_bound - min_bound
    diagonal = float(np.linalg.norm(extent))

    print(f"Open3D loading: {cloud_path}")
    print(f"points: {len(points)}")
    print(f"has_colors: {point_cloud.has_colors()}")
    print(f"min_bound: {format_vec3(min_bound)}")
    print(f"max_bound: {format_vec3(max_bound)}")

    geometries: list[o3d.geometry.Geometry] = [point_cloud]
    if not args.no_axis:
        axis_size = diagonal * 0.15 if diagonal > 0 else 0.1
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Open3D - {cloud_path.name}", width=args.width, height=args.height)
    for geometry in geometries:
        vis.add_geometry(geometry)

    render_option = vis.get_render_option()
    render_option.point_size = args.point_size
    render_option.background_color = np.asarray(args.background, dtype=np.float64)

    vis.poll_events()
    vis.update_renderer()
    view_control = vis.get_view_control()
    view_control.set_zoom(0.7)

    print("Viewer controls: drag to orbit, mouse wheel to zoom, close the window to exit.")
    vis.run()
    vis.destroy_window()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
