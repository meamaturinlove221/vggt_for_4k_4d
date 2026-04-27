from __future__ import annotations

import argparse
import json
import shutil
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


DEFAULT_SCENE = Path("output/4k4d_preprocessed_scene_variants/0012_11_frame0000_60views_headshoulder_crop")
DEFAULT_OUTPUT = Path("output/detail_normal_refiner_20260427/colmap_known_camera_smoke")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export an existing 4K4D scene as a known-camera COLMAP text model. "
            "The output is a diagnostic MVS input, not a final reconstruction claim."
        )
    )
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-views", type=int, default=60)
    parser.add_argument("--mask-background", action="store_true", default=True)
    parser.add_argument("--no-mask-background", action="store_false", dest="mask_background")
    parser.add_argument("--background-rgb", nargs=3, type=int, default=[255, 255, 255])
    parser.add_argument("--camera-model", choices=("PINHOLE",), default="PINHOLE")
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


def load_manifest(scene_dir: Path) -> dict[str, Any]:
    manifest_path = scene_dir / "scene_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"scene_manifest.json not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_rgb_camera_params(scene_manifest: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    dataset_root = Path(scene_manifest["dataset_root"]).expanduser()
    context = build_context(dataset_root, SUBSET_NAME)
    with tempfile.TemporaryDirectory(prefix="colmap_known_rgbcams_") as temp_name:
        rgb_cams_path, source = materialize_rgb_cams_smc(context, scene_manifest["seq_id"], Path(temp_name))
        if rgb_cams_path is None:
            raise FileNotFoundError(f"Could not resolve rgb_cams.smc for {scene_manifest['seq_id']}")
        out: dict[str, dict[str, np.ndarray]] = {"_source": {"rgb_cams_smc": np.asarray(str(source))}}
        with h5py.File(rgb_cams_path, "r") as handle:
            for view in scene_manifest["exported_views"]:
                camera_id = normalize_camera_id(view["camera_id"])
                group = handle["Camera_Parameter"][camera_id]
                cam_to_world = group["RT"][()].astype(np.float64)
                intrinsic = align_intrinsics_for_scene_view(group["K"][()].astype(np.float32), view, int(view["image_size"][0]))
                out[camera_id] = {
                    "intrinsic": intrinsic.astype(np.float64),
                    "cam_to_world": cam_to_world,
                    "world_to_cam": np.linalg.inv(cam_to_world).astype(np.float64),
                }
        return out


def rotation_matrix_to_qvec(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(matrix)
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        if diagonal[0] > diagonal[1] and diagonal[0] > diagonal[2]:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif diagonal[1] > diagonal[2]:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale
    qvec = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    if qvec[0] < 0:
        qvec = -qvec
    qvec /= np.clip(np.linalg.norm(qvec), 1e-12, None)
    return qvec


def copy_or_mask_image(
    image_path: Path,
    mask_path: Path,
    output_path: Path,
    *,
    mask_background: bool,
    background_rgb: tuple[int, int, int],
) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)
    arr = np.asarray(image).copy()
    mask_arr = np.asarray(mask) > 127
    if mask_background:
        arr[~mask_arr] = np.asarray(background_rgb, dtype=np.uint8)
        Image.fromarray(arr).save(output_path)
    else:
        shutil.copy2(image_path, output_path)
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
    images_dir = output_dir / "images"
    sparse_dir = output_dir / "sparse_text"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(scene_dir)
    camera_params = load_rgb_camera_params(manifest)
    views = list(manifest["exported_views"])
    if args.max_views > 0:
        views = views[: int(args.max_views)]
    background_rgb = tuple(int(max(0, min(255, v))) for v in args.background_rgb)

    camera_lines = ["# Camera list with one line of data per camera:", "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    image_lines = [
        "# Image list with two lines of data per image:",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    exported_views = []
    for image_id, view in enumerate(views, start=1):
        camera_id = normalize_camera_id(view["camera_id"])
        params = camera_params[camera_id]
        intrinsic = np.asarray(params["intrinsic"], dtype=np.float64)
        width, height = [int(v) for v in view["image_size"]]
        fx = float(intrinsic[0, 0])
        fy = float(intrinsic[1, 1])
        cx = float(intrinsic[0, 2])
        cy = float(intrinsic[1, 2])
        camera_lines.append(f"{image_id} {args.camera_model} {width} {height} {fx:.12f} {fy:.12f} {cx:.12f} {cy:.12f}")

        world_to_cam = np.asarray(params["world_to_cam"], dtype=np.float64)
        qvec = rotation_matrix_to_qvec(world_to_cam[:3, :3])
        tvec = world_to_cam[:3, 3]
        image_name = f"{image_id:03d}_cam{camera_id}.png"
        image_stats = copy_or_mask_image(
            Path(view["image_path"]),
            Path(view["mask_path"]),
            images_dir / image_name,
            mask_background=bool(args.mask_background),
            background_rgb=background_rgb,
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
                "intrinsic": intrinsic,
                "world_to_cam": world_to_cam,
                **image_stats,
            }
        )

    (sparse_dir / "cameras.txt").write_text("\n".join(camera_lines) + "\n", encoding="utf-8")
    (sparse_dir / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    (sparse_dir / "points3D.txt").write_text("# 3D point list with one line of data per point:\n", encoding="utf-8")

    summary = {
        "task": "known_camera_colmap_scene_export",
        "truthful_status": "mvs_input_only_not_reconstruction_pass",
        "scene_dir": str(scene_dir),
        "output_dir": str(output_dir),
        "images_dir": str(images_dir),
        "sparse_text_dir": str(sparse_dir),
        "view_count": int(len(exported_views)),
        "camera_model": str(args.camera_model),
        "mask_background": bool(args.mask_background),
        "background_rgb": list(background_rgb),
        "exported_views": exported_views,
        "suggested_next_commands": [
            "colmap model_converter --input_path sparse_text --output_path sparse_bin --output_type BIN",
            "colmap image_undistorter --image_path images --input_path sparse_bin --output_path dense --output_type COLMAP --max_image_size 518",
            "colmap patch_match_stereo --workspace_path dense --workspace_format COLMAP --PatchMatchStereo.geom_consistency true",
            "colmap stereo_fusion --workspace_path dense --workspace_format COLMAP --input_type geometric --output_path fused.ply",
        ],
    }
    (output_dir / "export_summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
