from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_SCENE = Path("output/4k4d_preprocessed_scene_variants/0012_11_frame0000_6views_sparseproto_headshoulder_crop")
DEFAULT_BBOX_TEACHER = Path(
    "output/detail_normal_refiner_20260427/kinect_teacher_original6v_headface_similarity_gate/teacher_targets.npz"
)
DEFAULT_OUTPUT = Path("output/detail_normal_refiner_20260427/visual_hull_smoke")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lightweight 4K4D mask visual-hull smoke for head/face teacher feasibility. "
            "This is a diagnostic tool, not a final sparse-view pass claim."
        )
    )
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--annotations-smc", type=Path, default=None)
    parser.add_argument("--bbox-teacher-targets", type=Path, default=DEFAULT_BBOX_TEACHER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--grid-resolution", type=int, default=128)
    parser.add_argument("--mask-long-side", type=int, default=768)
    parser.add_argument("--min-view-ratio", type=float, default=0.86)
    parser.add_argument("--bbox-percentiles", nargs=2, type=float, default=[1.0, 99.0])
    parser.add_argument("--bbox-pad", type=float, default=0.05)
    parser.add_argument("--chunk-size", type=int, default=250000)
    parser.add_argument("--max-ply-points", type=int, default=300000)
    parser.add_argument("--target-cameras", nargs="*", default=["00", "30", "45"])
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


def normalize_camera_id(camera_id: str | int) -> str:
    return f"{int(camera_id):02d}"


def load_manifest(scene_dir: Path) -> dict[str, Any]:
    path = scene_dir / "scene_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"scene_manifest.json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_annotations_smc(scene_manifest: dict[str, Any], requested: Path | None) -> Path:
    if requested is not None and requested.is_file():
        return requested.resolve()
    manifest_path = Path(scene_manifest.get("annotations_smc", ""))
    if manifest_path.is_file():
        return manifest_path.resolve()
    dataset_root = Path(scene_manifest["dataset_root"])
    candidate = dataset_root / "annotations" / f"{scene_manifest['seq_id']}_annots.smc"
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve annotations SMC for {scene_manifest['seq_id']}")


def decode_encoded_image(buffer: np.ndarray) -> np.ndarray:
    encoded = np.asarray(buffer, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise RuntimeError("Failed to decode encoded mask/image bytes.")
    if decoded.ndim == 3:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        decoded = np.max(decoded, axis=2)
    return decoded.astype(np.uint8)


def load_resized_mask(
    handle: h5py.File,
    camera_id: str,
    frame: int,
    long_side: int,
) -> tuple[np.ndarray, tuple[int, int], tuple[float, float]]:
    cam_key = str(int(camera_id))
    frame_key = str(int(frame))
    raw = decode_encoded_image(handle["Mask"][cam_key]["mask"][frame_key][()])
    original_h, original_w = raw.shape[:2]
    if long_side <= 0:
        resized = raw
    else:
        scale = float(long_side) / float(max(original_h, original_w))
        new_w = max(1, int(round(original_w * scale)))
        new_h = max(1, int(round(original_h * scale)))
        resized = cv2.resize(raw, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    scale_x = resized.shape[1] / float(original_w)
    scale_y = resized.shape[0] / float(original_h)
    return resized > 0, (original_h, original_w), (scale_x, scale_y)


def scaled_intrinsic(intrinsic: np.ndarray, scale_xy: tuple[float, float]) -> np.ndarray:
    out = np.asarray(intrinsic, dtype=np.float64).copy()
    out[0, :] *= float(scale_xy[0])
    out[1, :] *= float(scale_xy[1])
    return out


def bbox_from_teacher(path: Path, percentiles: list[float], pad: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Teacher targets for bbox not found: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if "real_world_points" not in payload or "teacher_mask" not in payload:
            raise KeyError("bbox teacher must contain real_world_points and teacher_mask")
        points = np.asarray(payload["real_world_points"], dtype=np.float32)[np.asarray(payload["teacher_mask"], dtype=bool)]
    if points.shape[0] == 0:
        raise RuntimeError("No teacher points available for visual-hull bbox.")
    lo_p, hi_p = float(percentiles[0]), float(percentiles[1])
    bbox_min = np.percentile(points, lo_p, axis=0)
    bbox_max = np.percentile(points, hi_p, axis=0)
    extent = np.maximum(bbox_max - bbox_min, 1e-4)
    bbox_min = bbox_min - extent * float(pad)
    bbox_max = bbox_max + extent * float(pad)
    return bbox_min.astype(np.float32), bbox_max.astype(np.float32), {
        "source": str(path),
        "source_points": int(points.shape[0]),
        "percentiles": [lo_p, hi_p],
        "pad_fraction": float(pad),
    }


def make_grid(bbox_min: np.ndarray, bbox_max: np.ndarray, resolution: int) -> tuple[np.ndarray, tuple[int, int, int]]:
    resolution = int(resolution)
    xs = np.linspace(float(bbox_min[0]), float(bbox_max[0]), resolution, dtype=np.float32)
    ys = np.linspace(float(bbox_min[1]), float(bbox_max[1]), resolution, dtype=np.float32)
    zs = np.linspace(float(bbox_min[2]), float(bbox_max[2]), resolution, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
    return grid.reshape(-1, 3), (resolution, resolution, resolution)


def project_hits(points: np.ndarray, mask: np.ndarray, intrinsic: np.ndarray, world_to_cam: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    cam = points.astype(np.float64) @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
    z = cam[:, 2]
    positive = np.isfinite(cam).all(axis=1) & (z > 1e-6)
    uvw = cam @ intrinsic.T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-8, None)
    xi = np.rint(uv[:, 0]).astype(np.int64)
    yi = np.rint(uv[:, 1]).astype(np.int64)
    inside = positive & (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    hits = np.zeros(points.shape[0], dtype=bool)
    valid_indices = np.nonzero(inside)[0]
    if valid_indices.size:
        hits[valid_indices] = mask[yi[valid_indices], xi[valid_indices]]
    return hits


def surface_from_occupancy(occupied: np.ndarray) -> np.ndarray:
    padded = np.pad(occupied, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1, 1:-1]
    neighbors = (
        padded[:-2, 1:-1, 1:-1]
        & padded[2:, 1:-1, 1:-1]
        & padded[1:-1, :-2, 1:-1]
        & padded[1:-1, 2:, 1:-1]
        & padded[1:-1, 1:-1, :-2]
        & padded[1:-1, 1:-1, 2:]
    )
    return center & ~neighbors


def write_ply(path: Path, points: np.ndarray, max_points: int, seed: int = 20260427) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    original_count = int(points.shape[0])
    if max_points > 0 and points.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(points.shape[0], size=int(max_points), replace=False)
        indices.sort()
        points = points[indices]
    if points.shape[0] == 0:
        colors = np.zeros((0, 3), dtype=np.uint8)
    else:
        y = points[:, 1]
        t = (y - y.min()) / max(float(y.max() - y.min()), 1e-8)
        colors = np.stack([255 * (1.0 - t), 80 + 80 * t, 255 * t], axis=1).astype(np.uint8)
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
    return {"path": str(path), "written_points": int(points.shape[0]), "original_points": original_count}


def overlay_projection(
    points: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    world_to_cam: np.ndarray,
    output_path: Path,
    title: str,
) -> dict[str, Any]:
    height, width = mask.shape
    cam = points.astype(np.float64) @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
    z = cam[:, 2]
    positive = np.isfinite(cam).all(axis=1) & (z > 1e-6)
    uvw = cam @ intrinsic.T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-8, None)
    xi = np.rint(uv[:, 0]).astype(np.int64)
    yi = np.rint(uv[:, 1]).astype(np.int64)
    inside = positive & (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    hit_mask = np.zeros_like(mask, dtype=bool)
    if inside.any():
        hit_mask[yi[inside], xi[inside]] = True
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    image[mask] = np.asarray([235, 235, 235], dtype=np.uint8)
    image[hit_mask & ~mask] = np.asarray([255, 80, 80], dtype=np.uint8)
    image[hit_mask & mask] = np.asarray([20, 210, 80], dtype=np.uint8)
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(0, 0, 0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(output_path)
    mask_pixels = int(mask.sum())
    hit_pixels = int(hit_mask.sum())
    in_mask_hits = int((hit_mask & mask).sum())
    return {
        "path": str(output_path),
        "mask_pixels": mask_pixels,
        "projected_hit_pixels": hit_pixels,
        "hits_inside_mask": in_mask_hits,
        "hit_to_mask_ratio": float(in_mask_hits / max(1, mask_pixels)),
    }


def main() -> int:
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty. Re-run with --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(scene_dir)
    frame = int(args.frame if args.frame is not None else manifest.get("frame_id", 0))
    annotations_smc = resolve_annotations_smc(manifest, args.annotations_smc)
    bbox_min, bbox_max, bbox_meta = bbox_from_teacher(args.bbox_teacher_targets, args.bbox_percentiles, args.bbox_pad)
    grid_points, grid_shape = make_grid(bbox_min, bbox_max, int(args.grid_resolution))

    with h5py.File(annotations_smc, "r") as handle:
        camera_ids = sorted([key for key in handle["Camera_Parameter"].keys() if str(key).isdigit()], key=lambda x: int(x))
        camera_payloads: list[dict[str, Any]] = []
        for camera_id in camera_ids:
            mask, original_hw, scale_xy = load_resized_mask(handle, camera_id, frame, int(args.mask_long_side))
            camera_group = handle["Camera_Parameter"][normalize_camera_id(camera_id)]
            cam_to_world = camera_group["RT"][()].astype(np.float64)
            camera_payloads.append(
                {
                    "camera_id": normalize_camera_id(camera_id),
                    "mask": mask,
                    "original_hw": original_hw,
                    "resized_hw": mask.shape,
                    "scale_xy": scale_xy,
                    "intrinsic": scaled_intrinsic(camera_group["K"][()].astype(np.float64), scale_xy),
                    "world_to_cam": np.linalg.inv(cam_to_world).astype(np.float64),
                }
            )

    min_views = max(1, int(np.ceil(len(camera_payloads) * float(args.min_view_ratio))))
    hit_counts = np.zeros(grid_points.shape[0], dtype=np.uint16)
    chunk_size = max(1000, int(args.chunk_size))
    for start in range(0, grid_points.shape[0], chunk_size):
        end = min(start + chunk_size, grid_points.shape[0])
        chunk = grid_points[start:end]
        counts = np.zeros(chunk.shape[0], dtype=np.uint16)
        for payload in camera_payloads:
            counts += project_hits(chunk, payload["mask"], payload["intrinsic"], payload["world_to_cam"]).astype(np.uint16)
        hit_counts[start:end] = counts

    occupied = (hit_counts >= min_views).reshape(grid_shape)
    surface = surface_from_occupancy(occupied)
    surface_points = grid_points[surface.reshape(-1)]
    occupied_points = grid_points[occupied.reshape(-1)]

    ply_surface = write_ply(output_dir / "visual_hull_surface_points.ply", surface_points, int(args.max_ply_points))
    ply_occupied = write_ply(output_dir / "visual_hull_occupied_points_sample.ply", occupied_points, int(args.max_ply_points))

    overlays = []
    target_set = {normalize_camera_id(camera_id) for camera_id in args.target_cameras}
    for payload in camera_payloads:
        if payload["camera_id"] not in target_set:
            continue
        overlays.append(
            overlay_projection(
                surface_points,
                payload["mask"],
                payload["intrinsic"],
                payload["world_to_cam"],
                output_dir / "overlays" / f"cam{payload['camera_id']}_visual_hull_surface_overlay.png",
                f"cam{payload['camera_id']} visual hull surface",
            )
        )

    summary = {
        "task": "visual_hull_teacher_smoke",
        "truthful_status": "diagnostic_only_not_final_pass",
        "scene_dir": str(scene_dir),
        "annotations_smc": str(annotations_smc),
        "output_dir": str(output_dir),
        "frame": int(frame),
        "camera_count": int(len(camera_payloads)),
        "grid_shape": list(grid_shape),
        "grid_points": int(grid_points.shape[0]),
        "mask_long_side": int(args.mask_long_side),
        "min_view_ratio": float(args.min_view_ratio),
        "min_views_required": int(min_views),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "bbox_meta": bbox_meta,
        "occupied_voxels": int(occupied.sum()),
        "surface_voxels": int(surface.sum()),
        "occupied_ratio": float(occupied.sum() / max(1, occupied.size)),
        "surface_ply": ply_surface,
        "occupied_ply": ply_occupied,
        "overlays": overlays,
        "interpretation": [
            "Mask visual hull can test multi-view silhouette consistency and produce a continuous outer shell.",
            "It cannot by itself create eyes/nose/mouth detail because silhouettes do not constrain interior facial relief.",
            "Use this only as a coarse continuity/visibility candidate, and gate any downstream use with Kinect/depth and Open3D close-ups.",
        ],
    }
    (output_dir / "visual_hull_summary.json").write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")
    print(json.dumps(json_ready(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
