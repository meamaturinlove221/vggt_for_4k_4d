from __future__ import annotations

import argparse
import io
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.dna_4k4d import (  # noqa: E402
    SUBSET_NAME,
    build_context,
    describe_expected_file,
    materialize_archived_member,
    normalize_camera_id,
    require_h5py,
    sort_numeric,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export 4K4D human-prior files for VGGT training.")
    parser.add_argument("--dataset-path", required=True, help="Folder containing extracted 4K4D data or raw zip parts.")
    parser.add_argument("--seq", required=True, help="Sequence id such as 0012_11.")
    parser.add_argument("--frame", type=int, required=True, help="Frame index to export.")
    parser.add_argument("--output-dir", required=True, help="Output directory for the exported priors.")
    parser.add_argument("--subset-name", default=SUBSET_NAME, help=f"Subset name. Default: {SUBSET_NAME}")
    parser.add_argument("--camera-ids", nargs="*", default=[], help="Optional camera ids. Default exports all available cameras.")
    parser.add_argument("--materialize-archived", action="store_true", help="Temporarily materialize archived annotation SMCs if needed.")
    parser.add_argument("--mask-only", action="store_true", help="Export silhouette-only prior maps.")
    parser.add_argument("--sigma", type=float, default=6.0, help="Gaussian sigma for the aggregated 2D keypoint heatmap.")
    parser.add_argument("--min-conf", type=float, default=0.05, help="Minimum keypoint confidence used for the heatmap.")
    parser.add_argument("--target-size", type=int, help="VGGT-style pad target size. Preserves aspect ratio and pads to square.")
    parser.add_argument("--resize-height", type=int, help="Optional output height for exported prior maps.")
    parser.add_argument("--resize-width", type=int, help="Optional output width for exported prior maps.")
    return parser.parse_args()


def resolve_annotations_smc(context, subset_name: str, seq: str, materialize_archived: bool, temp_dir: Path) -> tuple[Path, dict]:
    canonical = f"{subset_name}/annotations/{seq}_annots.smc"
    file_info = describe_expected_file(context, canonical)

    if file_info["status"] == "extracted":
        return Path(file_info["path"]), file_info

    if file_info["status"] == "archived" and materialize_archived:
        materialized = materialize_archived_member(context, canonical, temp_dir)
        if materialized is not None:
            return materialized, {
                "status": "materialized",
                "path": str(materialized),
                "archives": file_info["archives"],
            }

    raise FileNotFoundError(
        f"Could not access {canonical}. "
        "If the file only exists inside zip parts, re-run with --materialize-archived."
    )


def decode_png_bytes(encoded: np.ndarray) -> np.ndarray:
    image = Image.open(io.BytesIO(encoded.tobytes())).convert("L")
    return np.asarray(image, dtype=np.uint8)


def load_mask(annotation_handle, camera_id: str, frame_idx: int) -> np.ndarray | None:
    if "Mask" not in annotation_handle:
        return None

    camera_key = str(int(camera_id))
    if camera_key not in annotation_handle["Mask"]:
        return None

    mask_group = annotation_handle["Mask"][camera_key]["mask"]
    frame_key = str(int(frame_idx))
    if frame_key not in mask_group:
        return None

    return decode_png_bytes(mask_group[frame_key][()])


def load_keypoints_2d(annotation_handle, camera_id: str, frame_idx: int) -> np.ndarray | None:
    if "Keypoints_2D" not in annotation_handle or camera_id not in annotation_handle["Keypoints_2D"]:
        return None

    dataset = annotation_handle["Keypoints_2D"][camera_id]
    if frame_idx >= dataset.shape[0]:
        raise IndexError(f"Frame index {frame_idx} exceeds Keypoints_2D length {dataset.shape[0]}.")
    return dataset[frame_idx].astype(np.float32)


def load_keypoints_3d(annotation_handle, frame_idx: int) -> np.ndarray | None:
    if "Keypoints_3D" not in annotation_handle or "keypoints3d" not in annotation_handle["Keypoints_3D"]:
        return None
    dataset = annotation_handle["Keypoints_3D"]["keypoints3d"]
    if frame_idx >= dataset.shape[0]:
        raise IndexError(f"Frame index {frame_idx} exceeds Keypoints_3D length {dataset.shape[0]}.")
    return dataset[frame_idx].astype(np.float32)


def load_smplx_params(annotation_handle, frame_idx: int) -> dict[str, np.ndarray]:
    if "SMPLx" not in annotation_handle:
        return {}

    smplx_params = {}
    for key in sort_numeric(annotation_handle["SMPLx"].keys()):
        dataset = annotation_handle["SMPLx"][key]
        values = dataset[()]
        if np.ndim(values) == 0:
            smplx_params[key] = np.asarray(values)
        else:
            if frame_idx >= values.shape[0]:
                raise IndexError(f"Frame index {frame_idx} exceeds SMPLx/{key} length {values.shape[0]}.")
            smplx_params[key] = np.asarray(values[frame_idx])
    return smplx_params


def infer_image_hw(mask: np.ndarray | None, keypoints_2d: np.ndarray | None) -> tuple[int, int]:
    if mask is not None:
        return int(mask.shape[0]), int(mask.shape[1])

    if keypoints_2d is None or len(keypoints_2d) == 0:
        raise ValueError("Unable to infer image size because both mask and keypoints are missing.")

    valid = keypoints_2d[np.isfinite(keypoints_2d).all(axis=1) & (keypoints_2d[:, 2] > 0)]
    if len(valid) == 0:
        raise ValueError("Unable to infer image size because all keypoints are invalid.")

    width = int(math.ceil(valid[:, 0].max())) + 1
    height = int(math.ceil(valid[:, 1].max())) + 1
    return max(height, 1), max(width, 1)


def resize_map(array: np.ndarray, target_hw: tuple[int, int], mode: str) -> np.ndarray:
    if array.shape == target_hw:
        return array.astype(np.float32, copy=False)

    if mode == "nearest":
        resample = Image.Resampling.NEAREST
    elif mode == "bilinear":
        resample = Image.Resampling.BILINEAR
    elif mode == "bicubic":
        resample = Image.Resampling.BICUBIC
    else:
        raise ValueError(f"Unsupported resize mode: {mode}")

    image = Image.fromarray(array.astype(np.float32, copy=False), mode="F")
    resized = image.resize((int(target_hw[1]), int(target_hw[0])), resample=resample)
    return np.asarray(resized, dtype=np.float32)


def pad_resize_map(array: np.ndarray, target_size: int, mode: str) -> np.ndarray:
    height, width = array.shape
    if width >= height:
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
    else:
        new_height = target_size
        new_width = round(width * (new_height / height) / 14) * 14

    resized = resize_map(array, (new_height, new_width), mode=mode)
    canvas = np.zeros((target_size, target_size), dtype=np.float32)
    top = (target_size - new_height) // 2
    left = (target_size - new_width) // 2
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas


def render_keypoint_heatmap(
    keypoints_2d: np.ndarray | None,
    image_hw: tuple[int, int],
    sigma: float,
    min_conf: float,
) -> np.ndarray:
    height, width = image_hw
    heatmap = np.zeros((height, width), dtype=np.float32)
    if keypoints_2d is None:
        return heatmap

    radius = max(1, int(math.ceil(3.0 * sigma)))
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-((xx ** 2 + yy ** 2) / (2.0 * sigma * sigma))).astype(np.float32)

    for x, y, conf in keypoints_2d:
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(conf)):
            continue
        if conf < min_conf:
            continue

        center_x = int(round(float(x)))
        center_y = int(round(float(y)))
        if center_x < 0 or center_x >= width or center_y < 0 or center_y >= height:
            continue

        x0 = max(0, center_x - radius)
        x1 = min(width, center_x + radius + 1)
        y0 = max(0, center_y - radius)
        y1 = min(height, center_y + radius + 1)

        kernel_x0 = x0 - (center_x - radius)
        kernel_x1 = kernel_x0 + (x1 - x0)
        kernel_y0 = y0 - (center_y - radius)
        kernel_y1 = kernel_y0 + (y1 - y0)

        patch = kernel[kernel_y0:kernel_y1, kernel_x0:kernel_x1] * float(conf)
        np.maximum(heatmap[y0:y1, x0:x1], patch, out=heatmap[y0:y1, x0:x1])

    max_value = float(heatmap.max())
    if max_value > 0:
        heatmap /= max_value
    return heatmap


def save_grayscale_png(array: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8))
    image.save(output_path)


def save_preview(mask: np.ndarray, keypoint_heatmap: np.ndarray | None, output_path: Path) -> None:
    tiles = [np.clip(mask * 255.0, 0, 255).astype(np.uint8)]
    if keypoint_heatmap is not None:
        tiles.append(np.clip(keypoint_heatmap * 255.0, 0, 255).astype(np.uint8))
    preview = np.concatenate(tiles, axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview).save(output_path)


def main() -> int:
    args = parse_args()
    if args.target_size is not None and (args.resize_height is not None or args.resize_width is not None):
        raise SystemExit("Use either --target-size or --resize-height/--resize-width, not both.")
    if (args.resize_height is None) != (args.resize_width is None):
        raise SystemExit("Please provide both --resize-height and --resize-width, or neither.")

    context = build_context(Path(args.dataset_path), args.subset_name)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resize_hw = (args.resize_height, args.resize_width) if args.resize_height and args.resize_width else None

    temp_handle = tempfile.TemporaryDirectory(prefix="vggt_4k4d_prior_")
    temp_dir = Path(temp_handle.name)

    try:
        annotation_path, annotation_file_info = resolve_annotations_smc(
            context=context,
            subset_name=args.subset_name,
            seq=args.seq,
            materialize_archived=args.materialize_archived,
            temp_dir=temp_dir,
        )

        h5py = require_h5py()
        with h5py.File(annotation_path, "r") as annotation_handle:
            mask_cameras = (
                [normalize_camera_id(camera_id) for camera_id in sort_numeric(annotation_handle["Mask"].keys())]
                if "Mask" in annotation_handle
                else []
            )
            keypoint_cameras = (
                [normalize_camera_id(camera_id) for camera_id in sort_numeric(annotation_handle["Keypoints_2D"].keys())]
                if "Keypoints_2D" in annotation_handle
                else []
            )
            available_cameras = sorted(set(mask_cameras) | set(keypoint_cameras))
            camera_ids = [normalize_camera_id(camera_id) for camera_id in args.camera_ids] if args.camera_ids else available_cameras

            exported_cameras = []
            skipped_cameras = []

            smplx_params = load_smplx_params(annotation_handle, args.frame)
            if smplx_params:
                np.savez_compressed(output_dir / f"smplx_frame_{args.frame:04d}.npz", **smplx_params)

            keypoints_3d = load_keypoints_3d(annotation_handle, args.frame)
            if keypoints_3d is not None:
                np.save(output_dir / f"keypoints3d_frame_{args.frame:04d}.npy", keypoints_3d)

            for camera_id in camera_ids:
                mask = load_mask(annotation_handle, camera_id, args.frame)
                keypoints_2d = load_keypoints_2d(annotation_handle, camera_id, args.frame)

                if mask is None and keypoints_2d is None:
                    skipped_cameras.append({"camera_id": camera_id, "reason": "missing_mask_and_keypoints"})
                    continue

                image_hw = infer_image_hw(mask, keypoints_2d)
                if mask is None:
                    mask_float = np.zeros(image_hw, dtype=np.float32)
                else:
                    mask_float = (mask > 0).astype(np.float32)
                    if args.target_size is not None:
                        mask_float = pad_resize_map(mask_float, args.target_size, mode="nearest")
                        mask_float = (mask_float > 0.5).astype(np.float32)
                    elif resize_hw is not None:
                        mask_float = resize_map(mask_float, resize_hw, mode="nearest")
                        mask_float = (mask_float > 0.5).astype(np.float32)

                channel_names = ["silhouette"]
                prior_channels = [mask_float.astype(np.float32)]
                keypoint_heatmap = None

                if not args.mask_only:
                    keypoint_heatmap = render_keypoint_heatmap(
                        keypoints_2d=keypoints_2d,
                        image_hw=image_hw,
                        sigma=args.sigma,
                        min_conf=args.min_conf,
                    )
                    if args.target_size is not None:
                        keypoint_heatmap = pad_resize_map(keypoint_heatmap, args.target_size, mode="bilinear")
                    elif resize_hw is not None:
                        keypoint_heatmap = resize_map(keypoint_heatmap, resize_hw, mode="bilinear")
                    channel_names.append("keypoint_heatmap")
                    prior_channels.append(keypoint_heatmap.astype(np.float32))

                prior_maps = np.stack(prior_channels, axis=0).astype(np.float32)
                prior_mask = (mask_float > 0.5)

                camera_dir = output_dir / f"camera_{camera_id}"
                camera_dir.mkdir(parents=True, exist_ok=True)

                np.savez_compressed(
                    camera_dir / "prior_maps.npz",
                    prior_maps=prior_maps,
                    prior_mask=prior_mask,
                    keypoints_2d=np.zeros((0, 3), dtype=np.float32) if keypoints_2d is None else keypoints_2d.astype(np.float32),
                    frame_index=np.int32(args.frame),
                    camera_id=np.array(camera_id),
                )
                save_grayscale_png(mask_float, camera_dir / "silhouette.png")
                if keypoint_heatmap is not None:
                    save_grayscale_png(keypoint_heatmap, camera_dir / "keypoint_heatmap.png")
                save_preview(mask_float, keypoint_heatmap, camera_dir / "prior_preview.png")

                exported_cameras.append(
                    {
                        "camera_id": camera_id,
                        "image_hw": list(mask_float.shape),
                        "prior_channels": channel_names,
                        "prior_npz": str((camera_dir / "prior_maps.npz").relative_to(output_dir)),
                    }
                )

        summary = {
            "dataset_path": str(Path(args.dataset_path).resolve()),
            "seq_id": args.seq,
            "frame_index": int(args.frame),
            "annotation_smc": annotation_file_info,
            "target_size": int(args.target_size) if args.target_size is not None else None,
            "resize_hw": list(resize_hw) if resize_hw is not None else None,
            "mask_only": bool(args.mask_only),
            "sigma": float(args.sigma),
            "min_conf": float(args.min_conf),
            "exported_camera_count": len(exported_cameras),
            "exported_cameras": exported_cameras,
            "skipped_cameras": skipped_cameras,
            "smplx_output": str((output_dir / f"smplx_frame_{args.frame:04d}.npz").relative_to(output_dir)) if (output_dir / f"smplx_frame_{args.frame:04d}.npz").exists() else None,
            "keypoints3d_output": str((output_dir / f"keypoints3d_frame_{args.frame:04d}.npy").relative_to(output_dir)) if (output_dir / f"keypoints3d_frame_{args.frame:04d}.npy").exists() else None,
        }
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

        print(f"Exported 4K4D human priors to {output_dir}")
        return 0
    finally:
        temp_handle.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
