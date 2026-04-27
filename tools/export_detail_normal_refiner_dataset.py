from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt.utils.normal_refiner import (  # noqa: E402
    extract_coarse_prior_normal,
    face_box_from_mask,
    head_box_from_mask,
    normal_to_rgb,
    point_map_to_normal_numpy,
    points_world_to_camera,
    preprocess_mask_image,
    preprocess_rgb_image,
    shoulder_box_from_mask,
    crop_array,
    resize_array,
    resize_float_array,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ROI training samples for detail normal refinement.")
    parser.add_argument("--scene-dir", required=True, help="Scene directory with images/ and masks/")
    parser.add_argument("--prior-maps-npz", required=True, help="prior_maps.npz from the aligned scene")
    parser.add_argument("--predictions-npz", required=True, help="60v predictions.npz used as pseudo teacher source")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--roi-kind", choices=("head", "face", "shoulder", "both", "all"), default="both")
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument("--teacher-conf-percentile", type=float, default=15.0)
    return parser.parse_args()


def _load_scene_rgb_and_mask(scene_dir: Path, target_size: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    manifest_path = scene_dir / "scene_manifest.json"
    image_paths: list[Path]
    mask_paths: list[Path]
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        exported_views = manifest.get("exported_views", [])
        if exported_views:
            image_paths = [Path(item["image_path"]).expanduser().resolve() for item in exported_views]
            mask_paths = [Path(item["mask_path"]).expanduser().resolve() for item in exported_views]
        else:
            image_paths = sorted(path for path in (scene_dir / "images").iterdir() if path.is_file())
            mask_paths = sorted(path for path in (scene_dir / "masks").iterdir() if path.is_file())
    else:
        image_paths = sorted(path for path in (scene_dir / "images").iterdir() if path.is_file())
        mask_paths = sorted(path for path in (scene_dir / "masks").iterdir() if path.is_file())

    if len(image_paths) != len(mask_paths):
        raise ValueError(f"Image/mask count mismatch under {scene_dir}: {len(image_paths)} vs {len(mask_paths)}")
    images = np.stack([preprocess_rgb_image(path, target_size) for path in image_paths], axis=0)
    masks = np.stack([preprocess_mask_image(path, target_size) for path in mask_paths], axis=0)
    view_names = [path.stem for path in image_paths]
    return images, masks, view_names


def _make_region_masks(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bbox = head_box_from_mask(mask)
    if bbox is None:
        shape = mask.shape
        return np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
    x0, y0, x1, y1 = bbox
    roi_mask = np.zeros(mask.shape, dtype=bool)
    roi_mask[y0:y1, x0:x1] = True
    hairline_mask = roi_mask.copy()
    local_h = max(1, y1 - y0)
    top_limit = y0 + max(4, int(round(local_h * 0.22)))
    hairline_mask[top_limit:, :] = False

    ear_band_mask = roi_mask.copy()
    local_w = max(1, x1 - x0)
    left_limit = x0 + max(4, int(round(local_w * 0.18)))
    right_limit = x1 - max(4, int(round(local_w * 0.18)))
    ear_band_mask[:, left_limit:right_limit] = False
    return hairline_mask & mask, ear_band_mask & mask


def _compute_teacher_normals(predictions: dict[str, np.ndarray], human_mask: np.ndarray, conf_percentile: float) -> tuple[np.ndarray, np.ndarray]:
    world_points = predictions["world_points"].astype(np.float32)
    extrinsic = predictions["extrinsic"].astype(np.float32)
    point_conf = predictions.get("world_points_conf")
    teacher_normals = []
    teacher_masks = []
    for view_idx in range(world_points.shape[0]):
        conf_mask = np.ones(human_mask[view_idx].shape, dtype=bool)
        if point_conf is not None:
            conf_map = point_conf[view_idx].astype(np.float32)
            threshold = np.percentile(conf_map[human_mask[view_idx]], conf_percentile) if human_mask[view_idx].any() else 0.0
            conf_mask = conf_map >= threshold
        cam_points = points_world_to_camera(world_points[view_idx], extrinsic[view_idx])
        teacher_mask = human_mask[view_idx] & conf_mask & np.isfinite(cam_points).all(axis=-1)
        teacher_normal, valid = point_map_to_normal_numpy(cam_points, teacher_mask)
        teacher_normals.append(teacher_normal)
        teacher_masks.append(valid & teacher_mask)
    return np.stack(teacher_normals, axis=0), np.stack(teacher_masks, axis=0)


def _export_roi_pack(
    *,
    roi_name: str,
    box_fn,
    images: np.ndarray,
    human_mask: np.ndarray,
    coarse_normal: np.ndarray,
    coarse_valid_mask: np.ndarray,
    teacher_normal: np.ndarray,
    teacher_mask: np.ndarray,
    view_names: list[str],
    output_dir: Path,
    target_size: int,
) -> None:
    records = {
        "rgb": [],
        "human_mask": [],
        "coarse_prior_normal": [],
        "coarse_prior_valid_mask": [],
        "teacher_normal": [],
        "teacher_mask": [],
        "hairline_mask": [],
        "ear_band_mask": [],
        "roi_box_xyxy": [],
        "view_index": [],
        "view_name": [],
    }
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    for view_idx in range(images.shape[0]):
        box = box_fn(human_mask[view_idx])
        if box is None:
            continue
        rgb_crop = resize_array(crop_array(images[view_idx], box), target_size)
        human_crop = resize_array(crop_array(human_mask[view_idx], box), target_size, is_mask=True).astype(bool)
        coarse_crop = resize_array(
            normal_to_rgb(crop_array(coarse_normal[view_idx], box), crop_array(coarse_valid_mask[view_idx], box)),
            target_size,
        )
        coarse_float = resize_float_array(crop_array(coarse_normal[view_idx], box), target_size).astype(np.float32)
        coarse_valid = resize_array(crop_array(coarse_valid_mask[view_idx], box), target_size, is_mask=True).astype(bool)
        teacher_crop = resize_float_array(crop_array(teacher_normal[view_idx], box), target_size).astype(np.float32)
        teacher_valid = resize_array(crop_array(teacher_mask[view_idx], box), target_size, is_mask=True).astype(bool)
        coarse_norm = np.linalg.norm(coarse_float, axis=-1, keepdims=True)
        coarse_float = coarse_float / np.clip(coarse_norm, 1e-6, None)
        coarse_float[~coarse_valid] = 0.0
        teacher_norm = np.linalg.norm(teacher_crop, axis=-1, keepdims=True)
        teacher_crop = teacher_crop / np.clip(teacher_norm, 1e-6, None)
        teacher_crop[~teacher_valid] = 0.0
        hairline_mask, ear_band_mask = _make_region_masks(crop_array(human_mask[view_idx], box))
        hairline_mask = resize_array(hairline_mask, target_size, is_mask=True).astype(bool)
        ear_band_mask = resize_array(ear_band_mask, target_size, is_mask=True).astype(bool)

        preview = np.concatenate(
            [
                rgb_crop,
                coarse_crop,
                normal_to_rgb(teacher_crop, teacher_valid),
            ],
            axis=1,
        )
        Image.fromarray(preview).save(preview_dir / f"{view_idx:02d}_{view_names[view_idx]}_{roi_name}_rgb_coarse_teacher.png")

        records["rgb"].append(rgb_crop.astype(np.uint8))
        records["human_mask"].append(human_crop)
        records["coarse_prior_normal"].append(coarse_float.astype(np.float32))
        records["coarse_prior_valid_mask"].append(coarse_valid)
        records["teacher_normal"].append(teacher_crop.astype(np.float32))
        records["teacher_mask"].append(teacher_valid)
        records["hairline_mask"].append(hairline_mask)
        records["ear_band_mask"].append(ear_band_mask)
        records["roi_box_xyxy"].append(np.asarray(box, dtype=np.int32))
        records["view_index"].append(np.int32(view_idx))
        records["view_name"].append(view_names[view_idx])

    if not records["rgb"]:
        raise RuntimeError(f"No {roi_name} ROI samples were exported.")

    np.savez_compressed(
        output_dir / f"{roi_name}_samples.npz",
        rgb=np.stack(records["rgb"], axis=0),
        human_mask=np.stack(records["human_mask"], axis=0),
        coarse_prior_normal=np.stack(records["coarse_prior_normal"], axis=0),
        coarse_prior_valid_mask=np.stack(records["coarse_prior_valid_mask"], axis=0),
        teacher_normal=np.stack(records["teacher_normal"], axis=0),
        teacher_mask=np.stack(records["teacher_mask"], axis=0),
        hairline_mask=np.stack(records["hairline_mask"], axis=0),
        ear_band_mask=np.stack(records["ear_band_mask"], axis=0),
        roi_box_xyxy=np.stack(records["roi_box_xyxy"], axis=0),
        view_index=np.asarray(records["view_index"], dtype=np.int32),
        view_name=np.asarray(records["view_name"]),
    )


def main() -> int:
    args = parse_args()
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prior_payload = np.load(args.prior_maps_npz, allow_pickle=False)
    coarse_normal, coarse_valid_mask = extract_coarse_prior_normal(
        prior_payload["prior_maps"],
        prior_payload["prior_channels"],
    )
    human_mask = np.asarray(prior_payload["prior_mask"], dtype=bool)
    images, scene_mask, view_names = _load_scene_rgb_and_mask(scene_dir, target_size=int(coarse_normal.shape[1]))
    human_mask = human_mask & scene_mask

    predictions = np.load(args.predictions_npz, allow_pickle=False)
    teacher_normal, teacher_mask = _compute_teacher_normals(predictions, human_mask, args.teacher_conf_percentile)

    exported = []
    if args.roi_kind in {"head", "both", "all"}:
        roi_dir = output_dir / "head_roi"
        roi_dir.mkdir(parents=True, exist_ok=True)
        _export_roi_pack(
            roi_name="head",
            box_fn=head_box_from_mask,
            images=images,
            human_mask=human_mask,
            coarse_normal=coarse_normal,
            coarse_valid_mask=coarse_valid_mask,
            teacher_normal=teacher_normal,
            teacher_mask=teacher_mask,
            view_names=view_names,
            output_dir=roi_dir,
            target_size=args.target_size,
        )
        exported.append("head_roi/head_samples.npz")

    if args.roi_kind in {"face", "all"}:
        roi_dir = output_dir / "face_roi"
        roi_dir.mkdir(parents=True, exist_ok=True)
        _export_roi_pack(
            roi_name="face",
            box_fn=face_box_from_mask,
            images=images,
            human_mask=human_mask,
            coarse_normal=coarse_normal,
            coarse_valid_mask=coarse_valid_mask,
            teacher_normal=teacher_normal,
            teacher_mask=teacher_mask,
            view_names=view_names,
            output_dir=roi_dir,
            target_size=args.target_size,
        )
        exported.append("face_roi/face_samples.npz")

    if args.roi_kind in {"shoulder", "both", "all"}:
        roi_dir = output_dir / "shoulder_roi"
        roi_dir.mkdir(parents=True, exist_ok=True)
        _export_roi_pack(
            roi_name="shoulder",
            box_fn=shoulder_box_from_mask,
            images=images,
            human_mask=human_mask,
            coarse_normal=coarse_normal,
            coarse_valid_mask=coarse_valid_mask,
            teacher_normal=teacher_normal,
            teacher_mask=teacher_mask,
            view_names=view_names,
            output_dir=roi_dir,
            target_size=args.target_size,
        )
        exported.append("shoulder_roi/shoulder_samples.npz")

    summary = {
        "scene_dir": str(scene_dir),
        "prior_maps_npz": str(Path(args.prior_maps_npz).expanduser().resolve()),
        "predictions_npz": str(Path(args.predictions_npz).expanduser().resolve()),
        "target_size": args.target_size,
        "teacher_conf_percentile": args.teacher_conf_percentile,
        "exported": exported,
        "notes": [
            "coarse prior normal comes from SMPL-X view-aligned channels inside prior_maps.npz",
            "teacher normal comes from 60v world_points converted to camera-space pseudo normals",
            "this export is intended for ROI-first detail_normal_refiner smoke training",
        ],
    }
    (output_dir / "export_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
