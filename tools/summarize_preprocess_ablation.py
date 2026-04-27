from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize sparse-view preprocessing ablations from modal output directories "
            "and generate comparison-ready CSV/JSON tables plus aligned preview sheets."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run spec in the form 'label=/path/to/output_dir'. Repeat for each variant.",
    )
    parser.add_argument("--baseline", default=None, help="Baseline label used for aligned dense comparisons.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated comparison artifacts.")
    parser.add_argument("--preview-tile-width", type=int, default=220, help="Preview tile width in pixels.")
    return parser.parse_args()


@dataclass
class RunRecord:
    label: str
    run_dir: Path
    summary: dict[str, Any]
    scene_manifest: dict[str, Any]
    views: list[dict[str, Any]]
    predictions: dict[str, np.ndarray]
    prediction_source: str
    prediction_warning: str | None
    masks: np.ndarray


def parse_run_specs(raw_specs: list[str]) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    for raw_spec in raw_specs:
        if "=" not in raw_spec:
            raise ValueError(f"Invalid --run value: {raw_spec!r}")
        label, raw_path = raw_spec.split("=", 1)
        label = label.strip()
        run_dir = Path(raw_path.strip()).expanduser().resolve()
        if not label:
            raise ValueError(f"Missing label in --run value: {raw_spec!r}")
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        specs.append((label, run_dir))
    if not specs:
        raise ValueError("Please provide at least one --run label=path.")
    return specs


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_predictions_from_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.array(payload[key]) for key in payload.files}


def load_predictions_from_chunks(chunk_dir: Path) -> dict[str, np.ndarray]:
    manifest = load_json(chunk_dir / "manifest.json")
    chunks = sorted(manifest["chunks"], key=lambda item: int(item["start"]))
    per_view_keys = [str(key) for key in manifest.get("per_view_keys", [])]
    static_keys = [str(key) for key in manifest.get("static_keys", [])]

    collected: dict[str, list[np.ndarray]] = {key: [] for key in per_view_keys}
    static_values: dict[str, np.ndarray] = {}
    expected_start = 0

    for chunk in chunks:
        start = int(chunk["start"])
        end = int(chunk["end"])
        if start != expected_start:
            raise ValueError(f"Chunk sequence is not contiguous at {chunk['file']}: expected {expected_start}, got {start}")
        chunk_path = chunk_dir / str(chunk["file"])
        with np.load(chunk_path, allow_pickle=False) as payload:
            for key in per_view_keys:
                collected[key].append(np.array(payload[key]))
            for key in static_keys:
                if key not in static_values:
                    static_values[key] = np.array(payload[key])
        expected_start = end

    predictions = {key: np.concatenate(parts, axis=0) for key, parts in collected.items()}
    predictions.update(static_values)

    num_views = int(manifest["num_views"])
    for key, value in predictions.items():
        if key in per_view_keys and value.shape[0] != num_views:
            raise ValueError(f"Chunked key {key!r} has {value.shape[0]} views, expected {num_views}")
    return predictions


def load_predictions(run_dir: Path) -> tuple[dict[str, np.ndarray], str, str | None]:
    prediction_path = run_dir / "predictions.npz"
    chunk_dir = run_dir / "predictions_chunks_v2"
    warning: str | None = None

    if prediction_path.is_file():
        try:
            return load_predictions_from_npz(prediction_path), "predictions.npz", None
        except Exception as exc:  # noqa: BLE001
            warning = f"{type(exc).__name__}: {exc}"

    if chunk_dir.is_dir() and (chunk_dir / "manifest.json").is_file():
        predictions = load_predictions_from_chunks(chunk_dir)
        source = "predictions_chunks_v2"
        if warning:
            source = f"{source} (fallback)"
        return predictions, source, warning

    raise FileNotFoundError(f"No readable predictions were found under {run_dir}")


def load_run_record(label: str, run_dir: Path) -> RunRecord:
    summary_path = run_dir / "summary.json"
    summary = load_json(summary_path)
    scene_manifest = summary["scene_manifest"]
    views = list(scene_manifest["exported_views"])
    predictions, prediction_source, prediction_warning = load_predictions(run_dir)
    masks = load_view_masks(views)
    return RunRecord(
        label=label,
        run_dir=run_dir,
        summary=summary,
        scene_manifest=scene_manifest,
        views=views,
        predictions=predictions,
        prediction_source=prediction_source,
        prediction_warning=prediction_warning,
        masks=masks,
    )


def load_view_masks(views: list[dict[str, Any]]) -> np.ndarray:
    mask_arrays = []
    for view in views:
        mask_path = Path(view["mask_path"])
        mask = Image.open(mask_path).convert("L")
        mask_arrays.append(np.asarray(mask, dtype=np.uint8) > 0)
    return np.stack(mask_arrays, axis=0)


def load_font(size: int) -> ImageFont.ImageFont:
    for font_name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def get_preprocess_meta(view: dict[str, Any]) -> dict[str, Any]:
    return dict(view.get("preprocess_meta") or {})


def get_crop_bbox(view: dict[str, Any], canvas_hw: tuple[int, int]) -> tuple[int, int, int, int]:
    meta = get_preprocess_meta(view)
    if "crop_bbox_xyxy" in meta:
        x0, y0, x1, y1 = [int(value) for value in meta["crop_bbox_xyxy"]]
        return x0, y0, x1, y1
    height, width = canvas_hw
    return 0, 0, width, height


def compute_forward_layout(bbox_xyxy: tuple[int, int, int, int], target_size: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox_xyxy
    crop_width = max(1, int(x1) - int(x0))
    crop_height = max(1, int(y1) - int(y0))
    if crop_width >= crop_height:
        new_width = target_size
        new_height = round(crop_height * (new_width / crop_width) / 14) * 14
    else:
        new_height = target_size
        new_width = round(crop_width * (new_height / crop_height) / 14) * 14
    new_width = max(14, min(target_size, int(new_width)))
    new_height = max(14, min(target_size, int(new_height)))
    left = (target_size - new_width) // 2
    top = (target_size - new_height) // 2
    return left, top, new_width, new_height


def resize_float_array(array: np.ndarray, out_h: int, out_w: int, *, resample: Image.Resampling) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        image = Image.fromarray(array, mode="F")
        resized = image.resize((int(out_w), int(out_h)), resample=resample)
        return np.asarray(resized, dtype=np.float32)
    channels = [
        resize_float_array(array[..., channel_idx], out_h, out_w, resample=resample)
        for channel_idx in range(array.shape[-1])
    ]
    return np.stack(channels, axis=-1)


def restore_float_field_to_canvas(
    field: np.ndarray,
    view: dict[str, Any],
    canvas_hw: tuple[int, int],
    *,
    fill_value: float = np.nan,
) -> tuple[np.ndarray, np.ndarray]:
    field = np.asarray(field, dtype=np.float32)
    canvas_h, canvas_w = [int(value) for value in canvas_hw]
    meta = get_preprocess_meta(view)
    transform = str(meta.get("transform") or "")
    if "crop_bbox_xyxy" not in meta or transform != "crop_pad_to_square":
        valid = np.ones((canvas_h, canvas_w), dtype=bool)
        return field.copy(), valid

    x0, y0, x1, y1 = get_crop_bbox(view, canvas_hw)
    left, top, new_width, new_height = compute_forward_layout((x0, y0, x1, y1), target_size=canvas_w)
    cropped = field[top : top + new_height, left : left + new_width]
    restored = resize_float_array(cropped, y1 - y0, x1 - x0, resample=Image.Resampling.BILINEAR)

    if field.ndim == 2:
        canvas = np.full((canvas_h, canvas_w), fill_value, dtype=np.float32)
        canvas[y0:y1, x0:x1] = restored
    else:
        channels = field.shape[-1]
        canvas = np.full((canvas_h, canvas_w, channels), fill_value, dtype=np.float32)
        canvas[y0:y1, x0:x1, :] = restored

    valid = np.zeros((canvas_h, canvas_w), dtype=bool)
    valid[y0:y1, x0:x1] = True
    return canvas, valid


def restore_preview_image_to_canvas(image_array: np.ndarray, view: dict[str, Any], canvas_hw: tuple[int, int]) -> np.ndarray:
    image_array = np.asarray(image_array, dtype=np.uint8)
    canvas_h, canvas_w = [int(value) for value in canvas_hw]
    meta = get_preprocess_meta(view)
    transform = str(meta.get("transform") or "")
    if "crop_bbox_xyxy" not in meta or transform != "crop_pad_to_square":
        return image_array.copy()

    x0, y0, x1, y1 = get_crop_bbox(view, canvas_hw)
    left, top, new_width, new_height = compute_forward_layout((x0, y0, x1, y1), target_size=canvas_w)
    cropped = image_array[top : top + new_height, left : left + new_width]
    image = Image.fromarray(cropped, mode="RGB")
    restored = image.resize((x1 - x0, y1 - y0), resample=Image.Resampling.BILINEAR)
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    canvas[y0:y1, x0:x1, :] = np.asarray(restored, dtype=np.uint8)
    return canvas


def scalar_stats(values: np.ndarray) -> dict[str, float | int | None]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    percentiles = np.percentile(flat, [5, 50, 95])
    return {
        "count": int(flat.size),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "p05": float(percentiles[0]),
        "p50": float(percentiles[1]),
        "p95": float(percentiles[2]),
        "max": float(flat.max()),
    }


def apply_stats(row: dict[str, Any], prefix: str, stats: dict[str, float | int | None]) -> None:
    for key, value in stats.items():
        row[f"{prefix}_{key}"] = value


def summarize_scalar_field(field: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float | int | None]:
    values = np.asarray(field, dtype=np.float32)
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool)]
    return scalar_stats(values)


def summarize_vector_norm_field(field: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float | int | None]:
    norms = np.linalg.norm(np.asarray(field, dtype=np.float32), axis=-1)
    if mask is not None:
        norms = norms[np.asarray(mask, dtype=bool)]
    return scalar_stats(norms)


def build_variant_summary_row(run: RunRecord) -> dict[str, Any]:
    views = run.views
    depth = np.asarray(run.predictions["depth"], dtype=np.float32)[..., 0]
    depth_conf = np.asarray(run.predictions["depth_conf"], dtype=np.float32)
    world_points = np.asarray(run.predictions["world_points"], dtype=np.float32)
    world_points_conf = np.asarray(run.predictions["world_points_conf"], dtype=np.float32)
    normal = np.asarray(run.predictions["normal"], dtype=np.float32)
    normal_conf = np.asarray(run.predictions["normal_conf"], dtype=np.float32)
    intrinsic = np.asarray(run.predictions["intrinsic"], dtype=np.float32)
    extrinsic = np.asarray(run.predictions["extrinsic"], dtype=np.float32)
    translations = extrinsic[:, :, 3]

    canvas_hw = depth.shape[1:3]
    crop_area_ratios = []
    mask_coverages = []
    for view_idx, view in enumerate(views):
        x0, y0, x1, y1 = get_crop_bbox(view, canvas_hw)
        crop_area_ratios.append(((x1 - x0) * (y1 - y0)) / float(canvas_hw[0] * canvas_hw[1]))
        mask_coverages.append(float(run.masks[view_idx].mean()))

    row: dict[str, Any] = {
        "label": run.label,
        "run_dir": str(run.run_dir),
        "preprocess_variant": run.scene_manifest.get("preprocess_variant"),
        "scene_subdir": run.summary.get("scene_subdir"),
        "image_mode": run.summary.get("image_mode"),
        "num_images": run.summary.get("num_images"),
        "elapsed_seconds": run.summary.get("elapsed_seconds"),
        "gpu_name": run.summary.get("gpu_name"),
        "prediction_source": run.prediction_source,
        "prediction_warning": run.prediction_warning,
        "target_camera": run.scene_manifest.get("target_camera"),
        "source_cameras": ",".join(run.scene_manifest.get("source_cameras") or []),
        "mask_coverage_mean": float(np.mean(mask_coverages)),
        "mask_coverage_min": float(np.min(mask_coverages)),
        "mask_coverage_max": float(np.max(mask_coverages)),
        "crop_area_ratio_mean": float(np.mean(crop_area_ratios)),
        "crop_area_ratio_min": float(np.min(crop_area_ratios)),
        "crop_area_ratio_max": float(np.max(crop_area_ratios)),
        "fx_mean": float(intrinsic[:, 0, 0].mean()),
        "fy_mean": float(intrinsic[:, 1, 1].mean()),
        "cx_mean": float(intrinsic[:, 0, 2].mean()),
        "cy_mean": float(intrinsic[:, 1, 2].mean()),
        "translation_norm_mean": float(np.linalg.norm(translations, axis=1).mean()),
        "translation_norm_std": float(np.linalg.norm(translations, axis=1).std()),
    }

    apply_stats(row, "depth_all", summarize_scalar_field(depth))
    apply_stats(row, "depth_fg", summarize_scalar_field(depth, run.masks))
    apply_stats(row, "depth_conf_all", summarize_scalar_field(depth_conf))
    apply_stats(row, "depth_conf_fg", summarize_scalar_field(depth_conf, run.masks))
    apply_stats(row, "world_points_conf_all", summarize_scalar_field(world_points_conf))
    apply_stats(row, "world_points_conf_fg", summarize_scalar_field(world_points_conf, run.masks))
    apply_stats(row, "world_points_radius_all", summarize_vector_norm_field(world_points))
    apply_stats(row, "world_points_radius_fg", summarize_vector_norm_field(world_points, run.masks))
    apply_stats(row, "normal_conf_all", summarize_scalar_field(normal_conf))
    apply_stats(row, "normal_conf_fg", summarize_scalar_field(normal_conf, run.masks))
    apply_stats(row, "normal_norm_all", summarize_vector_norm_field(normal))
    apply_stats(row, "normal_norm_fg", summarize_vector_norm_field(normal, run.masks))
    return row


def build_per_view_rows(run: RunRecord) -> list[dict[str, Any]]:
    depth = np.asarray(run.predictions["depth"], dtype=np.float32)[..., 0]
    depth_conf = np.asarray(run.predictions["depth_conf"], dtype=np.float32)
    world_points = np.asarray(run.predictions["world_points"], dtype=np.float32)
    world_points_conf = np.asarray(run.predictions["world_points_conf"], dtype=np.float32)
    normal = np.asarray(run.predictions["normal"], dtype=np.float32)
    normal_conf = np.asarray(run.predictions["normal_conf"], dtype=np.float32)
    intrinsic = np.asarray(run.predictions["intrinsic"], dtype=np.float32)
    extrinsic = np.asarray(run.predictions["extrinsic"], dtype=np.float32)
    canvas_hw = depth.shape[1:3]

    rows = []
    for view_idx, view in enumerate(run.views):
        x0, y0, x1, y1 = get_crop_bbox(view, canvas_hw)
        translation = extrinsic[view_idx, :, 3]
        row: dict[str, Any] = {
            "label": run.label,
            "run_dir": str(run.run_dir),
            "view_index": view_idx,
            "image_name": run.summary["image_names"][view_idx],
            "view_stem": Path(run.summary["image_names"][view_idx]).stem,
            "camera_id": view.get("camera_id"),
            "role": view.get("role"),
            "mask_path": view.get("mask_path"),
            "mask_coverage": float(run.masks[view_idx].mean()),
            "crop_x0": x0,
            "crop_y0": y0,
            "crop_x1": x1,
            "crop_y1": y1,
            "crop_area_ratio": ((x1 - x0) * (y1 - y0)) / float(canvas_hw[0] * canvas_hw[1]),
            "fx": float(intrinsic[view_idx, 0, 0]),
            "fy": float(intrinsic[view_idx, 1, 1]),
            "cx": float(intrinsic[view_idx, 0, 2]),
            "cy": float(intrinsic[view_idx, 1, 2]),
            "translation_x": float(translation[0]),
            "translation_y": float(translation[1]),
            "translation_z": float(translation[2]),
            "translation_norm": float(np.linalg.norm(translation)),
        }

        apply_stats(row, "depth_all", summarize_scalar_field(depth[view_idx]))
        apply_stats(row, "depth_fg", summarize_scalar_field(depth[view_idx], run.masks[view_idx]))
        apply_stats(row, "depth_conf_all", summarize_scalar_field(depth_conf[view_idx]))
        apply_stats(row, "depth_conf_fg", summarize_scalar_field(depth_conf[view_idx], run.masks[view_idx]))
        apply_stats(row, "world_points_conf_all", summarize_scalar_field(world_points_conf[view_idx]))
        apply_stats(row, "world_points_conf_fg", summarize_scalar_field(world_points_conf[view_idx], run.masks[view_idx]))
        apply_stats(row, "world_points_radius_all", summarize_vector_norm_field(world_points[view_idx]))
        apply_stats(row, "world_points_radius_fg", summarize_vector_norm_field(world_points[view_idx], run.masks[view_idx]))
        apply_stats(row, "normal_conf_all", summarize_scalar_field(normal_conf[view_idx]))
        apply_stats(row, "normal_conf_fg", summarize_scalar_field(normal_conf[view_idx], run.masks[view_idx]))
        apply_stats(row, "normal_norm_all", summarize_vector_norm_field(normal[view_idx]))
        apply_stats(row, "normal_norm_fg", summarize_vector_norm_field(normal[view_idx], run.masks[view_idx]))
        rows.append(row)
    return rows


def angle_degrees(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a_norm = np.linalg.norm(a, axis=-1)
    b_norm = np.linalg.norm(b, axis=-1)
    valid = (a_norm > 1e-8) & (b_norm > 1e-8)
    dots = np.zeros_like(a_norm, dtype=np.float32)
    dots[valid] = np.sum(a[valid] * b[valid], axis=-1) / (a_norm[valid] * b_norm[valid])
    dots = np.clip(dots, -1.0, 1.0)
    return np.degrees(np.arccos(dots[valid]))


def build_aligned_diff_rows(baseline: RunRecord, other: RunRecord) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base_depth = np.asarray(baseline.predictions["depth"], dtype=np.float32)[..., 0]
    base_depth_conf = np.asarray(baseline.predictions["depth_conf"], dtype=np.float32)
    base_world_points = np.asarray(baseline.predictions["world_points"], dtype=np.float32)
    base_world_points_conf = np.asarray(baseline.predictions["world_points_conf"], dtype=np.float32)
    base_normal = np.asarray(baseline.predictions["normal"], dtype=np.float32)
    base_normal_conf = np.asarray(baseline.predictions["normal_conf"], dtype=np.float32)
    base_intrinsic = np.asarray(baseline.predictions["intrinsic"], dtype=np.float32)
    base_extrinsic = np.asarray(baseline.predictions["extrinsic"], dtype=np.float32)

    other_depth = np.asarray(other.predictions["depth"], dtype=np.float32)[..., 0]
    other_depth_conf = np.asarray(other.predictions["depth_conf"], dtype=np.float32)
    other_world_points = np.asarray(other.predictions["world_points"], dtype=np.float32)
    other_world_points_conf = np.asarray(other.predictions["world_points_conf"], dtype=np.float32)
    other_normal = np.asarray(other.predictions["normal"], dtype=np.float32)
    other_normal_conf = np.asarray(other.predictions["normal_conf"], dtype=np.float32)
    other_intrinsic = np.asarray(other.predictions["intrinsic"], dtype=np.float32)
    other_extrinsic = np.asarray(other.predictions["extrinsic"], dtype=np.float32)

    canvas_hw = base_depth.shape[1:3]
    depth_deltas = []
    depth_abs = []
    depth_sq = []
    depth_conf_abs = []
    world_point_l2 = []
    world_points_conf_abs = []
    normal_abs = []
    normal_angles = []
    normal_conf_abs = []
    per_view_rows = []

    for view_idx, baseline_view in enumerate(baseline.views):
        aligned_depth, valid_mask = restore_float_field_to_canvas(other_depth[view_idx], other.views[view_idx], canvas_hw)
        aligned_depth_conf, _ = restore_float_field_to_canvas(other_depth_conf[view_idx], other.views[view_idx], canvas_hw)
        aligned_world_points, _ = restore_float_field_to_canvas(other_world_points[view_idx], other.views[view_idx], canvas_hw)
        aligned_world_points_conf, _ = restore_float_field_to_canvas(
            other_world_points_conf[view_idx], other.views[view_idx], canvas_hw
        )
        aligned_normal, _ = restore_float_field_to_canvas(other_normal[view_idx], other.views[view_idx], canvas_hw)
        aligned_normal_conf, _ = restore_float_field_to_canvas(other_normal_conf[view_idx], other.views[view_idx], canvas_hw)

        compare_mask = baseline.masks[view_idx] & valid_mask
        compare_pixels = int(compare_mask.sum())
        if compare_pixels == 0:
            compare_mask = valid_mask
            compare_pixels = int(compare_mask.sum())

        depth_delta = aligned_depth[compare_mask] - base_depth[view_idx][compare_mask]
        depth_delta_abs = np.abs(depth_delta)
        depth_delta_sq = depth_delta ** 2
        depth_conf_delta_abs = np.abs(aligned_depth_conf[compare_mask] - base_depth_conf[view_idx][compare_mask])
        world_point_delta_l2 = np.linalg.norm(aligned_world_points[compare_mask] - base_world_points[view_idx][compare_mask], axis=-1)
        world_points_conf_delta_abs = np.abs(
            aligned_world_points_conf[compare_mask] - base_world_points_conf[view_idx][compare_mask]
        )
        normal_delta_abs = np.abs(aligned_normal[compare_mask] - base_normal[view_idx][compare_mask]).reshape(-1)
        normal_delta_angles = angle_degrees(aligned_normal[compare_mask], base_normal[view_idx][compare_mask])
        normal_conf_delta_abs = np.abs(aligned_normal_conf[compare_mask] - base_normal_conf[view_idx][compare_mask])

        depth_deltas.append(depth_delta)
        depth_abs.append(depth_delta_abs)
        depth_sq.append(depth_delta_sq)
        depth_conf_abs.append(depth_conf_delta_abs)
        world_point_l2.append(world_point_delta_l2)
        world_points_conf_abs.append(world_points_conf_delta_abs)
        normal_abs.append(normal_delta_abs)
        normal_angles.append(normal_delta_angles)
        normal_conf_abs.append(normal_conf_delta_abs)

        base_translation = base_extrinsic[view_idx, :, 3]
        other_translation = other_extrinsic[view_idx, :, 3]
        per_view_rows.append(
            {
                "label": other.label,
                "baseline": baseline.label,
                "view_index": view_idx,
                "image_name": baseline.summary["image_names"][view_idx],
                "camera_id": baseline_view.get("camera_id"),
                "role": baseline_view.get("role"),
                "compare_pixel_count": compare_pixels,
                "depth_mean_delta": float(depth_delta.mean()),
                "depth_mae": float(depth_delta_abs.mean()),
                "depth_rmse": float(np.sqrt(depth_delta_sq.mean())),
                "depth_conf_mae": float(depth_conf_delta_abs.mean()),
                "world_points_l2_mean": float(world_point_delta_l2.mean()),
                "world_points_l2_p95": float(np.percentile(world_point_delta_l2, 95)),
                "world_points_conf_mae": float(world_points_conf_delta_abs.mean()),
                "normal_mae": float(normal_delta_abs.mean()),
                "normal_angle_mean_deg": float(normal_delta_angles.mean()) if normal_delta_angles.size else None,
                "normal_angle_p95_deg": float(np.percentile(normal_delta_angles, 95)) if normal_delta_angles.size else None,
                "normal_conf_mae": float(normal_conf_delta_abs.mean()),
                "translation_l2": float(np.linalg.norm(other_translation - base_translation)),
                "fx_delta": float(other_intrinsic[view_idx, 0, 0] - base_intrinsic[view_idx, 0, 0]),
                "fy_delta": float(other_intrinsic[view_idx, 1, 1] - base_intrinsic[view_idx, 1, 1]),
                "cx_delta": float(other_intrinsic[view_idx, 0, 2] - base_intrinsic[view_idx, 0, 2]),
                "cy_delta": float(other_intrinsic[view_idx, 1, 2] - base_intrinsic[view_idx, 1, 2]),
            }
        )

    depth_deltas_flat = np.concatenate(depth_deltas) if depth_deltas else np.asarray([], dtype=np.float32)
    depth_abs_flat = np.concatenate(depth_abs) if depth_abs else np.asarray([], dtype=np.float32)
    depth_sq_flat = np.concatenate(depth_sq) if depth_sq else np.asarray([], dtype=np.float32)
    depth_conf_abs_flat = np.concatenate(depth_conf_abs) if depth_conf_abs else np.asarray([], dtype=np.float32)
    world_point_l2_flat = np.concatenate(world_point_l2) if world_point_l2 else np.asarray([], dtype=np.float32)
    world_points_conf_abs_flat = (
        np.concatenate(world_points_conf_abs) if world_points_conf_abs else np.asarray([], dtype=np.float32)
    )
    normal_abs_flat = np.concatenate(normal_abs) if normal_abs else np.asarray([], dtype=np.float32)
    normal_angles_flat = np.concatenate(normal_angles) if normal_angles else np.asarray([], dtype=np.float32)
    normal_conf_abs_flat = np.concatenate(normal_conf_abs) if normal_conf_abs else np.asarray([], dtype=np.float32)

    translation_delta = np.linalg.norm(other_extrinsic[:, :, 3] - base_extrinsic[:, :, 3], axis=1)
    diff_row: dict[str, Any] = {
        "label": other.label,
        "baseline": baseline.label,
        "prediction_source": other.prediction_source,
        "prediction_warning": other.prediction_warning,
        "compare_pixel_count": int(depth_deltas_flat.size),
        "depth_mean_delta": float(depth_deltas_flat.mean()) if depth_deltas_flat.size else None,
        "depth_mae": float(depth_abs_flat.mean()) if depth_abs_flat.size else None,
        "depth_rmse": float(np.sqrt(depth_sq_flat.mean())) if depth_sq_flat.size else None,
        "depth_max_abs": float(depth_abs_flat.max()) if depth_abs_flat.size else None,
        "depth_conf_mae": float(depth_conf_abs_flat.mean()) if depth_conf_abs_flat.size else None,
        "world_points_l2_mean": float(world_point_l2_flat.mean()) if world_point_l2_flat.size else None,
        "world_points_l2_p95": float(np.percentile(world_point_l2_flat, 95)) if world_point_l2_flat.size else None,
        "world_points_conf_mae": float(world_points_conf_abs_flat.mean()) if world_points_conf_abs_flat.size else None,
        "normal_mae": float(normal_abs_flat.mean()) if normal_abs_flat.size else None,
        "normal_angle_mean_deg": float(normal_angles_flat.mean()) if normal_angles_flat.size else None,
        "normal_angle_p95_deg": float(np.percentile(normal_angles_flat, 95)) if normal_angles_flat.size else None,
        "normal_conf_mae": float(normal_conf_abs_flat.mean()) if normal_conf_abs_flat.size else None,
        "pose_enc_mae": float(np.abs(other.predictions["pose_enc"] - baseline.predictions["pose_enc"]).mean()),
        "extrinsic_mae": float(np.abs(other.predictions["extrinsic"] - baseline.predictions["extrinsic"]).mean()),
        "intrinsic_mae": float(np.abs(other.predictions["intrinsic"] - baseline.predictions["intrinsic"]).mean()),
        "translation_l2_mean": float(translation_delta.mean()),
        "translation_l2_max": float(translation_delta.max()),
        "fx_mean_delta": float(other_intrinsic[:, 0, 0].mean() - base_intrinsic[:, 0, 0].mean()),
        "fy_mean_delta": float(other_intrinsic[:, 1, 1].mean() - base_intrinsic[:, 1, 1].mean()),
        "cx_mean_delta": float(other_intrinsic[:, 0, 2].mean() - base_intrinsic[:, 0, 2].mean()),
        "cy_mean_delta": float(other_intrinsic[:, 1, 2].mean() - base_intrinsic[:, 1, 2].mean()),
    }
    return diff_row, per_view_rows


def build_placeholder_tile(size: tuple[int, int], message: str) -> Image.Image:
    image = Image.new("RGB", size, (245, 245, 245))
    draw = ImageDraw.Draw(image)
    font = load_font(20)
    bbox = draw.textbbox((0, 0), message, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline=(200, 200, 200), width=2)
    draw.text(((size[0] - text_w) // 2, (size[1] - text_h) // 2), message, fill=(96, 96, 96), font=font)
    return image


def collect_preview_metrics(runs: list[RunRecord]) -> tuple[list[str], list[str]]:
    view_stems = [Path(name).stem for name in runs[0].summary["image_names"]]
    shared_metrics: set[str] | None = None
    for run in runs:
        metrics_for_run = set()
        preview_dir = run.run_dir / "previews"
        for view_stem in view_stems:
            for preview_path in preview_dir.glob(f"{view_stem}_*.png"):
                metric = preview_path.stem[len(view_stem) + 1 :]
                metrics_for_run.add(metric)
        shared_metrics = metrics_for_run if shared_metrics is None else (shared_metrics & metrics_for_run)
    return view_stems, sorted(shared_metrics or [])


def render_aligned_preview_sheet(
    *,
    runs: list[RunRecord],
    baseline: RunRecord,
    metric: str,
    output_path: Path,
    tile_width: int,
) -> None:
    view_stems = [Path(name).stem for name in baseline.summary["image_names"]]
    canvas_hw = tuple(np.asarray(baseline.predictions["depth"]).shape[1:3])
    variant_count = len(runs)
    row_labels = [f"{view['camera_id']} ({view['role']})" for view in baseline.views]

    sample_image = None
    for run in runs:
        preview_path = run.run_dir / "previews" / f"{view_stems[0]}_{metric}.png"
        if preview_path.is_file():
            sample_image = Image.open(preview_path).convert("RGB")
            break
    if sample_image is None:
        return

    tile_height = int(round(sample_image.height * (tile_width / sample_image.width)))
    left_label_width = 170
    top_label_height = 60
    padding = 18
    header_font = load_font(24)
    label_font = load_font(20)

    canvas_width = left_label_width + padding + variant_count * tile_width + (variant_count + 1) * padding
    canvas_height = top_label_height + padding + len(view_stems) * tile_height + (len(view_stems) + 1) * padding
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for col_idx, run in enumerate(runs):
        x0 = left_label_width + padding * 2 + col_idx * (tile_width + padding)
        bbox = draw.textbbox((0, 0), run.label, font=header_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.text((x0 + (tile_width - text_w) // 2, padding + (top_label_height - text_h) // 2), run.label, fill=(16, 16, 16), font=header_font)

    title = f"{metric} aligned to {baseline.label}"
    title_font = load_font(28)
    bbox = draw.textbbox((0, 0), title, font=title_font)
    title_w = bbox[2] - bbox[0]
    draw.text(((canvas_width - title_w) // 2, 6), title, fill=(16, 16, 16), font=title_font)

    placeholder = build_placeholder_tile((tile_width, tile_height), "missing")
    for row_idx, view_stem in enumerate(view_stems):
        y0 = top_label_height + padding * 2 + row_idx * (tile_height + padding)
        label = row_labels[row_idx]
        bbox = draw.textbbox((0, 0), label, font=label_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.text((padding + max(0, left_label_width - text_w - 12), y0 + (tile_height - text_h) // 2), label, fill=(40, 40, 40), font=label_font)

        for col_idx, run in enumerate(runs):
            x0 = left_label_width + padding * 2 + col_idx * (tile_width + padding)
            preview_path = run.run_dir / "previews" / f"{view_stem}_{metric}.png"
            if preview_path.is_file():
                preview = np.asarray(Image.open(preview_path).convert("RGB"), dtype=np.uint8)
                aligned = restore_preview_image_to_canvas(preview, run.views[row_idx], canvas_hw)
                tile = Image.fromarray(aligned, mode="RGB")
                tile = tile.resize((tile_width, tile_height), resample=Image.Resampling.BILINEAR)
            else:
                tile = placeholder.copy()
            canvas.paste(tile, (x0, y0))
            draw.rectangle((x0, y0, x0 + tile_width - 1, y0 + tile_height - 1), outline=(205, 205, 205), width=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_csv(path: Path, rows: list[dict[str, Any]], preferred_fields: list[str] | None = None) -> None:
    preferred_fields = preferred_fields or []
    field_names: list[str] = []
    seen = set()
    for field in preferred_fields:
        if field not in seen:
            field_names.append(field)
            seen.add(field)
    for row in rows:
        for field in row.keys():
            if field not in seen:
                field_names.append(field)
                seen.add(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def ensure_consistent_image_order(runs: list[RunRecord]) -> None:
    baseline_names = list(runs[0].summary["image_names"])
    for run in runs[1:]:
        if list(run.summary["image_names"]) != baseline_names:
            raise ValueError(
                f"Image/view order mismatch between {runs[0].label!r} and {run.label!r}; "
                "the comparison utility expects the same ordered views."
            )


def build_outputs(runs: list[RunRecord], baseline_label: str, output_dir: Path, tile_width: int) -> dict[str, Any]:
    ensure_consistent_image_order(runs)
    run_map = {run.label: run for run in runs}
    if baseline_label not in run_map:
        raise KeyError(f"Baseline label {baseline_label!r} is not among the provided runs.")
    baseline = run_map[baseline_label]

    variant_summary_rows = [build_variant_summary_row(run) for run in runs]
    per_view_rows = [row for run in runs for row in build_per_view_rows(run)]

    diff_summary_rows = []
    diff_per_view_rows = []
    for run in runs:
        if run.label == baseline.label:
            continue
        diff_row, per_view_diff_rows = build_aligned_diff_rows(baseline, run)
        diff_summary_rows.append(diff_row)
        diff_per_view_rows.extend(per_view_diff_rows)

    view_stems, preview_metrics = collect_preview_metrics(runs)
    preview_outputs = []
    for metric in preview_metrics:
        output_path = output_dir / "preview_sheets" / f"{metric}_aligned_to_{baseline.label}.png"
        render_aligned_preview_sheet(runs=runs, baseline=baseline, metric=metric, output_path=output_path, tile_width=tile_width)
        preview_outputs.append(
            {
                "metric": metric,
                "aligned_to": baseline.label,
                "path": str(output_path),
            }
        )

    summary_payload = {
        "baseline": baseline.label,
        "runs": [
            {
                "label": run.label,
                "run_dir": str(run.run_dir),
                "preprocess_variant": run.scene_manifest.get("preprocess_variant"),
                "prediction_source": run.prediction_source,
                "prediction_warning": run.prediction_warning,
            }
            for run in runs
        ],
        "view_stems": view_stems,
        "preview_metrics": preview_metrics,
        "preview_outputs": preview_outputs,
        "variant_summary_rows": variant_summary_rows,
        "per_view_rows": per_view_rows,
        "aligned_diff_summary_rows": diff_summary_rows,
        "aligned_diff_per_view_rows": diff_per_view_rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "variant_summary.csv",
        variant_summary_rows,
        preferred_fields=[
            "label",
            "preprocess_variant",
            "prediction_source",
            "prediction_warning",
            "elapsed_seconds",
            "mask_coverage_mean",
            "crop_area_ratio_mean",
            "depth_fg_mean",
            "depth_conf_fg_mean",
            "world_points_conf_fg_mean",
            "normal_conf_fg_mean",
            "normal_norm_fg_mean",
            "fx_mean",
            "fy_mean",
            "translation_norm_mean",
            "run_dir",
        ],
    )
    write_csv(
        output_dir / "per_view_metrics.csv",
        per_view_rows,
        preferred_fields=[
            "label",
            "view_index",
            "image_name",
            "camera_id",
            "role",
            "mask_coverage",
            "crop_area_ratio",
            "depth_fg_mean",
            "depth_conf_fg_mean",
            "world_points_conf_fg_mean",
            "normal_conf_fg_mean",
            "normal_norm_fg_mean",
            "fx",
            "fy",
            "translation_norm",
        ],
    )
    write_csv(
        output_dir / "aligned_diff_summary_vs_baseline.csv",
        diff_summary_rows,
        preferred_fields=[
            "label",
            "baseline",
            "prediction_source",
            "prediction_warning",
            "compare_pixel_count",
            "depth_mae",
            "depth_rmse",
            "world_points_l2_mean",
            "normal_angle_mean_deg",
            "normal_conf_mae",
            "translation_l2_mean",
            "intrinsic_mae",
            "extrinsic_mae",
        ],
    )
    write_csv(
        output_dir / "aligned_diff_per_view_vs_baseline.csv",
        diff_per_view_rows,
        preferred_fields=[
            "label",
            "baseline",
            "view_index",
            "image_name",
            "camera_id",
            "role",
            "compare_pixel_count",
            "depth_mae",
            "world_points_l2_mean",
            "normal_angle_mean_deg",
            "normal_conf_mae",
            "translation_l2",
            "fx_delta",
            "fy_delta",
        ],
    )
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary_payload


def main() -> int:
    args = parse_args()
    run_specs = parse_run_specs(args.run)
    runs = [load_run_record(label, run_dir) for label, run_dir in run_specs]
    baseline_label = args.baseline or run_specs[0][0]
    output_dir = Path(args.output_dir).expanduser().resolve()
    payload = build_outputs(runs, baseline_label, output_dir, tile_width=int(args.preview_tile_width))
    print(f"Wrote preprocess ablation comparison to {output_dir}")
    print(f"Baseline: {payload['baseline']}")
    print(f"Preview sheets: {len(payload['preview_outputs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
