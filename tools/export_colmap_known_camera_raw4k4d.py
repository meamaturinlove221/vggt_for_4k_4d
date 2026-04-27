from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.dna_4k4d import normalize_camera_id, sort_numeric  # noqa: E402
from tools.export_colmap_known_camera_scene import json_ready, rotation_matrix_to_qvec  # noqa: E402


DEFAULT_SCENE = Path("output/4k4d_preprocessed_scene_variants/0012_11_frame0000_60views_headshoulder_crop")
DEFAULT_OUTPUT = Path("output/detail_normal_refiner_20260427/colmap_raw4k4d_known_camera_60v")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export raw-resolution 4K4D RGB/mask frames and known cameras to a COLMAP text model. "
            "This is a teacher-gate input builder, not a reconstruction pass claim."
        )
    )
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--camera-ids", default="", help="Comma-separated camera ids. Empty uses manifest cameras.")
    parser.add_argument("--max-views", type=int, default=60)
    parser.add_argument("--mask-background", action="store_true", default=True)
    parser.add_argument("--no-mask-background", action="store_false", dest="mask_background")
    parser.add_argument("--background-rgb", nargs=3, type=int, default=[255, 255, 255])
    parser.add_argument("--copy-masks", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_manifest(scene_dir: Path) -> dict[str, Any]:
    manifest_path = scene_dir / "scene_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"scene_manifest.json not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def selected_camera_ids(manifest: dict[str, Any], camera_ids_arg: str, max_views: int) -> list[str]:
    if camera_ids_arg.strip():
        camera_ids = [normalize_camera_id(item.strip()) for item in camera_ids_arg.split(",") if item.strip()]
    else:
        camera_ids = [normalize_camera_id(view["camera_id"]) for view in manifest["exported_views"]]
    if max_views > 0:
        camera_ids = camera_ids[: int(max_views)]
    return camera_ids


def camera_group_for_id(main_handle: h5py.File, camera_id: str) -> str:
    raw_key = str(int(camera_id))
    if "Camera_5mp" in main_handle and raw_key in main_handle["Camera_5mp"]:
        return "Camera_5mp"
    if "Camera_12mp" in main_handle and raw_key in main_handle["Camera_12mp"]:
        return "Camera_12mp"
    available = []
    for group_name in ("Camera_5mp", "Camera_12mp"):
        if group_name in main_handle:
            available.extend([normalize_camera_id(item) for item in sort_numeric(main_handle[group_name].keys())])
    raise KeyError(f"Camera {camera_id} not found in main SMC. Available sample: {available[:12]}")


def decode_image_bytes(data: np.ndarray, mode: str) -> Image.Image:
    return Image.open(io.BytesIO(np.asarray(data, dtype=np.uint8).tobytes())).convert(mode)


def load_raw_rgb(main_handle: h5py.File, camera_id: str, frame: int) -> Image.Image:
    raw_key = str(int(camera_id))
    group_name = camera_group_for_id(main_handle, camera_id)
    data = main_handle[f"{group_name}/{raw_key}/color/{int(frame)}"][()]
    return decode_image_bytes(data, "RGB")


def load_raw_mask(annot_handle: h5py.File, camera_id: str, frame: int) -> Image.Image:
    raw_key = str(int(camera_id))
    data = annot_handle[f"Mask/{raw_key}/mask/{int(frame)}"][()]
    return decode_image_bytes(data, "L")


def write_raw_image(
    image: Image.Image,
    mask: Image.Image,
    output_image_path: Path,
    output_mask_path: Path,
    *,
    mask_background: bool,
    background_rgb: tuple[int, int, int],
    copy_masks: bool,
) -> dict[str, Any]:
    image = image.convert("RGB")
    mask = mask.convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)
    mask_arr = np.asarray(mask) > 127
    if mask_background:
        arr = np.asarray(image).copy()
        arr[~mask_arr] = np.asarray(background_rgb, dtype=np.uint8)
        Image.fromarray(arr).save(output_image_path)
    else:
        image.save(output_image_path)
    if copy_masks:
        mask.save(output_mask_path)
    return {
        "image_size": [int(image.width), int(image.height)],
        "mask_coverage": float(mask_arr.mean()),
        "masked_background": bool(mask_background),
    }


def main() -> int:
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists and is not empty. Re-run with --overwrite.")
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    sparse_dir = output_dir / "sparse_text"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(scene_dir)
    camera_ids = selected_camera_ids(manifest, str(args.camera_ids), int(args.max_views))
    frame = int(manifest.get("frame_id", 0))
    background_rgb = tuple(int(max(0, min(255, value))) for value in args.background_rgb)

    camera_lines = ["# Camera list with one line of data per camera:", "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    image_lines = [
        "# Image list with two lines of data per image:",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    exported_views: list[dict[str, Any]] = []
    with h5py.File(manifest["main_smc"], "r") as main_handle, h5py.File(manifest["annotations_smc"], "r") as annot_handle:
        for image_id, camera_id in enumerate(camera_ids, start=1):
            group = annot_handle["Camera_Parameter"][camera_id]
            intrinsic = group["K"][()].astype(np.float64)
            cam_to_world = group["RT"][()].astype(np.float64)
            world_to_cam = np.linalg.inv(cam_to_world).astype(np.float64)
            image = load_raw_rgb(main_handle, camera_id, frame)
            mask = load_raw_mask(annot_handle, camera_id, frame)
            width, height = image.size
            fx = float(intrinsic[0, 0])
            fy = float(intrinsic[1, 1])
            cx = float(intrinsic[0, 2])
            cy = float(intrinsic[1, 2])
            camera_lines.append(f"{image_id} PINHOLE {width} {height} {fx:.12f} {fy:.12f} {cx:.12f} {cy:.12f}")

            qvec = rotation_matrix_to_qvec(world_to_cam[:3, :3])
            tvec = world_to_cam[:3, 3]
            image_name = f"{image_id:03d}_cam{camera_id}.png"
            mask_name = f"{image_id:03d}_cam{camera_id}.png"
            image_stats = write_raw_image(
                image,
                mask,
                images_dir / image_name,
                masks_dir / mask_name,
                mask_background=bool(args.mask_background),
                background_rgb=background_rgb,
                copy_masks=bool(args.copy_masks),
            )
            image_lines.append(
                (
                    f"{image_id} {qvec[0]:.12f} {qvec[1]:.12f} {qvec[2]:.12f} {qvec[3]:.12f} "
                    f"{tvec[0]:.12f} {tvec[1]:.12f} {tvec[2]:.12f} {image_id} {image_name}"
                )
            )
            image_lines.append("")
            exported_views.append(
                {
                    "image_id": int(image_id),
                    "camera_id": camera_id,
                    "image_name": image_name,
                    "mask_name": mask_name,
                    "intrinsic": intrinsic,
                    "world_to_cam": world_to_cam,
                    **image_stats,
                }
            )

    (sparse_dir / "cameras.txt").write_text("\n".join(camera_lines) + "\n", encoding="utf-8")
    (sparse_dir / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    (sparse_dir / "points3D.txt").write_text("# 3D point list with one line of data per point:\n", encoding="utf-8")

    summary = {
        "task": "raw4k4d_known_camera_colmap_export",
        "truthful_status": "mvs_input_only_not_reconstruction_pass",
        "scene_dir": str(scene_dir),
        "output_dir": str(output_dir),
        "images_dir": str(images_dir),
        "masks_dir": str(masks_dir),
        "sparse_text_dir": str(sparse_dir),
        "frame": int(frame),
        "view_count": int(len(exported_views)),
        "mask_background": bool(args.mask_background),
        "background_rgb": list(background_rgb),
        "exported_views": exported_views,
        "suggested_next_commands": [
            "colmap model_converter --input_path sparse_text --output_path sparse_bin --output_type BIN",
            "colmap image_undistorter --image_path images --input_path sparse_bin --output_path dense --output_type COLMAP --max_image_size 1200",
            "colmap patch_match_stereo --workspace_path dense --workspace_format COLMAP --PatchMatchStereo.geom_consistency true",
            "colmap stereo_fusion --workspace_path dense --workspace_format COLMAP --input_type geometric --output_path fused.ply",
        ],
    }
    (output_dir / "export_summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
