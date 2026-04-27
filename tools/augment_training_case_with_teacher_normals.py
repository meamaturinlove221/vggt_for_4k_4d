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
    extract_coarse_prior_normal,
    face_box_from_mask,
    head_box_from_mask,
    normal_to_rgb,
    point_map_to_normal_numpy,
    points_world_to_camera,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a self-contained training case and augment targets.npz with teacher normal "
            "supervision derived from a stronger predictions.npz payload."
        )
    )
    parser.add_argument("--case-dir", required=True, help="Source training case directory")
    parser.add_argument("--predictions-npz", required=True, help="Predictions aligned to the case view order")
    parser.add_argument("--output-case-dir", required=True, help="Augmented copied case directory")
    parser.add_argument("--teacher-conf-percentile", type=float, default=15.0)
    parser.add_argument("--preview-count", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _copy_case_tree(case_dir: Path, output_case_dir: Path, overwrite: bool) -> None:
    if output_case_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output case dir already exists: {output_case_dir}")
        shutil.rmtree(output_case_dir)
    shutil.copytree(case_dir, output_case_dir)


def _load_case_payloads(case_dir: Path) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
    case_manifest = json.loads((case_dir / "case_manifest.json").read_text(encoding="utf-8"))
    with np.load(case_dir / "inputs.npz", allow_pickle=False) as inputs_payload:
        inputs = {key: np.array(inputs_payload[key]) for key in inputs_payload.files}
    with np.load(case_dir / "targets.npz", allow_pickle=False) as targets_payload:
        targets = {key: np.array(targets_payload[key]) for key in targets_payload.files}
    return case_manifest, inputs, targets


def _load_scene_masks(scene_dir: Path, expected_views: int) -> np.ndarray | None:
    scene_manifest_path = scene_dir / "scene_manifest.json"
    mask_paths: list[Path] = []
    if scene_manifest_path.is_file():
        scene_manifest = json.loads(scene_manifest_path.read_text(encoding="utf-8"))
        exported_views = scene_manifest.get("exported_views", [])
        if exported_views:
            mask_paths = [Path(item["mask_path"]).expanduser().resolve() for item in exported_views]
    if not mask_paths and (scene_dir / "masks").is_dir():
        mask_paths = sorted(path for path in (scene_dir / "masks").iterdir() if path.is_file())
    if len(mask_paths) != int(expected_views):
        return None
    masks = []
    for path in mask_paths:
        mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127
        masks.append(mask)
    return np.stack(masks, axis=0).astype(bool)


def _make_box_mask(box: tuple[int, int, int, int] | None, shape: tuple[int, int], support_mask: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if box is None:
        return mask
    x0, y0, x1, y1 = [int(v) for v in box]
    mask[y0:y1, x0:x1] = True
    return mask & np.asarray(support_mask, dtype=bool)


def _make_head_region_masks(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    shape = mask.shape
    head_box = head_box_from_mask(mask)
    face_box = face_box_from_mask(mask)
    head_mask = _make_box_mask(head_box, shape, mask)
    face_mask = _make_box_mask(face_box, shape, mask)
    hairline_mask = np.zeros(shape, dtype=bool)
    ear_band_mask = np.zeros(shape, dtype=bool)
    if head_box is not None:
        x0, y0, x1, y1 = [int(v) for v in head_box]
        local_h = max(1, y1 - y0)
        local_w = max(1, x1 - x0)
        top_limit = y0 + max(4, int(round(local_h * 0.22)))
        left_limit = x0 + max(4, int(round(local_w * 0.18)))
        right_limit = x1 - max(4, int(round(local_w * 0.18)))
        hairline_mask[y0:top_limit, x0:x1] = True
        ear_band_mask[y0:y1, x0:left_limit] = True
        ear_band_mask[y0:y1, right_limit:x1] = True
    return head_mask, face_mask, hairline_mask & mask, ear_band_mask & mask


def _compute_teacher_normals(
    predictions: dict[str, np.ndarray],
    human_mask: np.ndarray,
    conf_percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    world_points = np.asarray(predictions["world_points"], dtype=np.float32)
    extrinsic = np.asarray(predictions["extrinsic"], dtype=np.float32)
    point_conf = np.asarray(predictions["world_points_conf"], dtype=np.float32) if "world_points_conf" in predictions else None
    teacher_normals: list[np.ndarray] = []
    teacher_masks: list[np.ndarray] = []
    for view_idx in range(world_points.shape[0]):
        mask_view = np.asarray(human_mask[view_idx], dtype=bool)
        conf_mask = np.ones(mask_view.shape, dtype=bool)
        if point_conf is not None and np.any(mask_view):
            conf_view = point_conf[view_idx]
            threshold = float(np.percentile(conf_view[mask_view], conf_percentile))
            conf_mask = conf_view >= threshold
        cam_points = points_world_to_camera(world_points[view_idx], extrinsic[view_idx])
        teacher_mask = mask_view & conf_mask & np.isfinite(cam_points).all(axis=-1)
        teacher_normal, valid = point_map_to_normal_numpy(cam_points, teacher_mask)
        teacher_normals.append(teacher_normal.astype(np.float32))
        teacher_masks.append((valid & teacher_mask).astype(bool))
    return np.stack(teacher_normals, axis=0), np.stack(teacher_masks, axis=0)


def _align_teacher_normal_sign(
    *,
    teacher_normals: np.ndarray,
    teacher_masks: np.ndarray,
    reference_normals: np.ndarray | None,
    reference_mask: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, float | bool | int]]:
    if reference_normals is None or reference_mask is None:
        return teacher_normals, {
            "applied": False,
            "overlap_pixels": 0,
            "mean_dot_before": 0.0,
            "neg_frac_before": 0.0,
        }

    ref_normals = np.asarray(reference_normals, dtype=np.float32)
    ref_mask = np.asarray(reference_mask, dtype=bool)
    teacher_normals = np.asarray(teacher_normals, dtype=np.float32)
    teacher_masks = np.asarray(teacher_masks, dtype=bool)
    overlap = (
        teacher_masks
        & ref_mask
        & np.isfinite(teacher_normals).all(axis=-1)
        & np.isfinite(ref_normals).all(axis=-1)
        & (np.linalg.norm(teacher_normals, axis=-1) > 0.5)
        & (np.linalg.norm(ref_normals, axis=-1) > 0.5)
    )
    if int(overlap.sum()) < 100:
        return teacher_normals, {
            "applied": False,
            "overlap_pixels": int(overlap.sum()),
            "mean_dot_before": 0.0,
            "neg_frac_before": 0.0,
        }

    dot = np.sum(teacher_normals * ref_normals, axis=-1)
    mean_dot = float(dot[overlap].mean())
    neg_frac = float((dot[overlap] < 0.0).mean())
    should_flip = mean_dot < 0.0 and neg_frac > 0.5
    if should_flip:
        teacher_normals = (-teacher_normals).astype(np.float32)
    return teacher_normals, {
        "applied": bool(should_flip),
        "overlap_pixels": int(overlap.sum()),
        "mean_dot_before": mean_dot,
        "neg_frac_before": neg_frac,
    }


def _load_reference_normals(
    *,
    case_manifest: dict,
    inputs: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    if "prior_normals" in targets and "prior_mask" in inputs:
        return (
            np.asarray(targets["prior_normals"], dtype=np.float32),
            np.asarray(inputs["prior_mask"], dtype=bool),
            "targets.prior_normals",
        )

    prior_maps = inputs.get("prior_maps")
    channel_names = case_manifest.get("prior_input_meta", {}).get("channel_names", [])
    if prior_maps is None or not channel_names:
        return None, None, "none"

    try:
        coarse_normals, coarse_visible = extract_coarse_prior_normal(
            np.asarray(prior_maps, dtype=np.float32),
            np.asarray(channel_names),
        )
    except Exception:
        return None, None, "none"

    if "prior_mask" in inputs:
        coarse_visible = coarse_visible & np.asarray(inputs["prior_mask"], dtype=bool)
    return coarse_normals.astype(np.float32), coarse_visible.astype(bool), "inputs.prior_maps"


def _to_preview_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    if image.max() <= 1.5:
        return np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def _write_previews(
    *,
    output_dir: Path,
    images: np.ndarray,
    teacher_normals: np.ndarray,
    teacher_masks: np.ndarray,
    head_masks: np.ndarray,
    face_masks: np.ndarray,
    preview_count: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    limit = min(int(preview_count), int(images.shape[0]))
    for view_idx in range(limit):
        rgb = _to_preview_rgb(images[view_idx])
        teacher_rgb = normal_to_rgb(teacher_normals[view_idx], teacher_masks[view_idx])
        overlay = rgb.copy()
        overlay[head_masks[view_idx]] = np.array([255, 220, 0], dtype=np.uint8)
        overlay[face_masks[view_idx]] = np.array([255, 80, 80], dtype=np.uint8)
        strip = np.concatenate([rgb, teacher_rgb, overlay], axis=1)
        Image.fromarray(strip).save(output_dir / f"{view_idx:02d}_rgb_teacher_roi.png")


def main() -> int:
    args = parse_args()
    case_dir = Path(args.case_dir).expanduser().resolve()
    predictions_npz = Path(args.predictions_npz).expanduser().resolve()
    output_case_dir = Path(args.output_case_dir).expanduser().resolve()

    if not case_dir.is_dir():
        raise NotADirectoryError(f"Case directory not found: {case_dir}")
    if not predictions_npz.is_file():
        raise FileNotFoundError(f"Predictions NPZ not found: {predictions_npz}")

    _copy_case_tree(case_dir, output_case_dir, overwrite=bool(args.overwrite))
    case_manifest, inputs, targets = _load_case_payloads(output_case_dir)
    with np.load(predictions_npz, allow_pickle=False) as predictions_payload:
        predictions = {key: np.array(predictions_payload[key]) for key in predictions_payload.files}

    if "prior_mask" not in inputs:
        raise KeyError(f"inputs.npz under {output_case_dir} does not contain prior_mask")
    human_mask = np.asarray(inputs["prior_mask"], dtype=bool)
    scene_dir_value = case_manifest.get("scene_dir", "")
    scene_masks = None
    if scene_dir_value:
        scene_dir = Path(scene_dir_value).expanduser()
        if scene_dir.is_dir():
            scene_masks = _load_scene_masks(scene_dir.resolve(), expected_views=human_mask.shape[0])
    if scene_masks is not None and scene_masks.shape == human_mask.shape:
        human_mask = scene_masks
    teacher_normals, teacher_masks = _compute_teacher_normals(
        predictions,
        human_mask,
        float(args.teacher_conf_percentile),
    )
    reference_normals, reference_mask, reference_source = _load_reference_normals(
        case_manifest=case_manifest,
        inputs=inputs,
        targets=targets,
    )
    teacher_normals, sign_meta = _align_teacher_normal_sign(
        teacher_normals=teacher_normals,
        teacher_masks=teacher_masks,
        reference_normals=reference_normals,
        reference_mask=reference_mask,
    )

    head_masks: list[np.ndarray] = []
    face_masks: list[np.ndarray] = []
    hairline_masks: list[np.ndarray] = []
    ear_band_masks: list[np.ndarray] = []
    for view_idx in range(human_mask.shape[0]):
        head_mask, face_mask, hairline_mask, ear_band = _make_head_region_masks(human_mask[view_idx])
        head_masks.append(head_mask)
        face_masks.append(face_mask)
        hairline_masks.append(hairline_mask)
        ear_band_masks.append(ear_band)

    targets["teacher_normals"] = teacher_normals.astype(np.float32)
    targets["teacher_mask"] = teacher_masks.astype(bool)
    targets["head_roi_mask"] = np.stack(head_masks, axis=0).astype(bool)
    targets["face_roi_mask"] = np.stack(face_masks, axis=0).astype(bool)
    targets["hairline_mask"] = np.stack(hairline_masks, axis=0).astype(bool)
    targets["ear_band_mask"] = np.stack(ear_band_masks, axis=0).astype(bool)
    np.savez_compressed(output_case_dir / "targets.npz", **targets)

    preview_dir = output_case_dir / "teacher_normal_previews"
    _write_previews(
        output_dir=preview_dir,
        images=np.asarray(inputs["images"]),
        teacher_normals=teacher_normals,
        teacher_masks=teacher_masks,
        head_masks=np.asarray(targets["head_roi_mask"]),
        face_masks=np.asarray(targets["face_roi_mask"]),
        preview_count=int(args.preview_count),
    )

    case_manifest["teacher_normal_meta"] = {
        "predictions_npz": str(predictions_npz),
        "teacher_conf_percentile": float(args.teacher_conf_percentile),
        "notes": [
            "teacher normals are computed from stronger world_points predictions in camera space",
            "teacher normals supervise VGGT output normals without replacing coarse prior input maps",
            "ROI masks are exported for head/face/hairline/ear-focused weighting",
        ],
        "sign_alignment": sign_meta,
        "sign_reference_source": reference_source,
    }
    (output_case_dir / "case_manifest.json").write_text(
        json.dumps(case_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {
        "case_dir": str(case_dir),
        "output_case_dir": str(output_case_dir),
        "predictions_npz": str(predictions_npz),
        "teacher_conf_percentile": float(args.teacher_conf_percentile),
        "num_views": int(teacher_normals.shape[0]),
        "human_mask_source": "scene_masks" if scene_masks is not None and scene_masks.shape == human_mask.shape else "prior_mask",
        "sign_reference_source": reference_source,
        "sign_alignment": sign_meta,
        "head_mask_pixels": [int(mask.sum()) for mask in targets["head_roi_mask"]],
        "face_mask_pixels": [int(mask.sum()) for mask in targets["face_roi_mask"]],
    }
    (output_case_dir / "teacher_normal_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
