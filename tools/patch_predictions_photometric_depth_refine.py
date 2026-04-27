from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt.utils.normal_refiner import face_box_from_mask, head_box_from_mask, preprocess_mask_image, shoulder_box_from_mask  # noqa: E402


IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch VGGT prediction depth in a small human ROI with multi-view photometric depth sweep diagnostics."
    )
    parser.add_argument("--predictions-npz", required=True)
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--view-indices", default="0", help="Comma-separated view indices or 'all'.")
    parser.add_argument("--source-views", default="1,2,4,5", help="Comma-separated source indices or 'all'.")
    parser.add_argument("--roi-kind", choices=("head", "face", "face_core", "head_face", "shoulder", "all"), default="face_core")
    parser.add_argument("--max-depth-delta", type=float, default=0.05)
    parser.add_argument("--num-depth-samples", type=int, default=21)
    parser.add_argument("--patch-radius", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--min-source-views", type=int, default=2)
    parser.add_argument("--source-depth-tolerance", type=float, default=0.05)
    parser.add_argument("--accept-margin", type=float, default=0.01)
    parser.add_argument("--prior-weight", type=float, default=0.07)
    parser.add_argument("--gradient-weight", type=float, default=0.30)
    parser.add_argument("--single-source-clip", type=float, default=0.16)
    parser.add_argument("--max-source-anchor-p75", type=float, default=0.12)
    parser.add_argument("--min-anchor-valid-ratio", type=float, default=0.20)
    parser.add_argument("--mask-erode-pixels", type=int, default=1)
    parser.add_argument("--reject-boundary-best", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--copy-sidecar-files", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _list_images(directory: Path) -> list[Path]:
    files = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS]
    return sorted(files)


def _parse_indices(text: str, count: int, *, exclude: set[int] | None = None) -> list[int]:
    exclude = exclude or set()
    text = text.strip().lower()
    if text == "all":
        return [idx for idx in range(count) if idx not in exclude]
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0 or idx >= count:
            raise ValueError(f"view index {idx} outside [0, {count})")
        if idx not in exclude and idx not in out:
            out.append(idx)
    if not out:
        raise ValueError(f"empty index list from {text!r}")
    return out


def _load_manifest_order(scene_dir: Path) -> list[str] | None:
    manifest_path = scene_dir / "scene_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exported = manifest.get("exported_views")
    if not isinstance(exported, list):
        return None
    names = []
    for item in exported:
        image_path = item.get("image_path") if isinstance(item, dict) else None
        if image_path:
            names.append(Path(image_path).name)
    return names or None


def _load_scene_images_and_masks(scene_dir: Path, height: int, width: int) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    image_files = _list_images(scene_dir / "images")
    mask_files = _list_images(scene_dir / "masks")
    if not image_files:
        raise FileNotFoundError(scene_dir / "images")
    if len(image_files) != len(mask_files):
        raise ValueError(f"image/mask count mismatch: {len(image_files)} vs {len(mask_files)}")
    images = []
    masks = []
    for image_path, mask_path in zip(image_files, mask_files):
        image = Image.open(image_path).convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
        images.append(np.asarray(image, dtype=np.float32) / 255.0)
        masks.append(preprocess_mask_image(mask_path, target_size=height))
    sorted_names = [path.name for path in image_files]
    manifest_names = _load_manifest_order(scene_dir)
    order_check = {
        "sorted_image_basenames": sorted_names,
        "manifest_image_basenames": manifest_names,
        "sorted_matches_manifest": manifest_names == sorted_names if manifest_names is not None else None,
    }
    return np.stack(images, axis=0), np.stack(masks, axis=0).astype(bool), sorted_names, order_check


def _erode_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(pixels))):
        padded = np.pad(out, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        acc = np.ones_like(out, dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                acc &= padded[1 + dy : 1 + dy + out.shape[0], 1 + dx : 1 + dx + out.shape[1]]
        out = acc
    return out


def _roi_mask(mask: np.ndarray, roi_kind: str) -> np.ndarray:
    if roi_kind == "all":
        return np.asarray(mask, dtype=bool)
    boxes = []
    if roi_kind in {"head", "head_face"}:
        boxes.append(head_box_from_mask(mask))
    if roi_kind in {"face", "head_face"}:
        boxes.append(face_box_from_mask(mask))
    if roi_kind == "face_core":
        face_box = face_box_from_mask(mask)
        if face_box is not None:
            x0, y0, x1, y1 = face_box
            width = x1 - x0
            height = y1 - y0
            core_w = max(16, int(round(width * 0.72)))
            core_h = max(16, int(round(height * 0.70)))
            cx = int(round((x0 + x1) * 0.5))
            cy = y0 + int(round(height * 0.46))
            boxes.append((cx - core_w // 2, cy - core_h // 2, cx + core_w // 2, cy + core_h // 2))
    if roi_kind == "shoulder":
        boxes.append(shoulder_box_from_mask(mask))
    out = np.zeros(mask.shape, dtype=bool)
    for box in boxes:
        if box is None:
            continue
        x0, y0, x1, y1 = box
        x0 = max(0, min(mask.shape[1], x0))
        x1 = max(0, min(mask.shape[1], x1))
        y0 = max(0, min(mask.shape[0], y0))
        y1 = max(0, min(mask.shape[0], y1))
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] |= mask[y0:y1, x0:x1]
    return out


def _gray_gradient(image: np.ndarray) -> np.ndarray:
    gray = (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]).astype(np.float32)
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx * gx + gy * gy)
    return np.clip(grad / max(float(np.percentile(grad, 99)), 1e-6), 0.0, 1.0).astype(np.float32)


def _bilinear_sample(array: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    height, width = array.shape[:2]
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    x0 = np.clip(x0, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    wa = (x1.astype(np.float32) - u) * (y1.astype(np.float32) - v)
    wb = (x1.astype(np.float32) - u) * (v - y0.astype(np.float32))
    wc = (u - x0.astype(np.float32)) * (y1.astype(np.float32) - v)
    wd = (u - x0.astype(np.float32)) * (v - y0.astype(np.float32))
    if array.ndim == 2:
        return array[y0, x0] * wa + array[y1, x0] * wb + array[y0, x1] * wc + array[y1, x1] * wd
    return (
        array[y0, x0] * wa[..., None]
        + array[y1, x0] * wb[..., None]
        + array[y0, x1] * wc[..., None]
        + array[y1, x1] * wd[..., None]
    )


def _nearest_sample_bool(mask: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    x = np.rint(u).astype(np.int64)
    y = np.rint(v).astype(np.int64)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    out = np.zeros(u.shape, dtype=bool)
    if np.any(valid):
        out[valid] = mask[y[valid], x[valid]]
    return out


def _depth_to_world(depth: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    cam = np.stack(
        [
            (xs[:, None] - cx) * depth / max(fx, 1e-6),
            (ys[:, None] - cy) * depth / max(fy, 1e-6),
            depth,
        ],
        axis=-1,
    ).astype(np.float32)
    rotation = extrinsic[:, :3].astype(np.float32)
    translation = extrinsic[:, 3].astype(np.float32)
    return (cam - translation[None, None, :]) @ rotation


def _world_to_source_uv(points_world: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotation = extrinsic[:, :3].astype(np.float32)
    translation = extrinsic[:, 3].astype(np.float32)
    cam = points_world @ rotation.T + translation[None, None, :]
    z = cam[..., 2]
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    u = fx * cam[..., 0] / np.clip(z, 1e-6, None) + cx
    v = fy * cam[..., 1] / np.clip(z, 1e-6, None) + cy
    return u.astype(np.float32), v.astype(np.float32), z.astype(np.float32)


def _world_from_full_depth(depth: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    views, height, width = depth.shape
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    out = np.zeros((views, height, width, 3), dtype=np.float32)
    for view_idx in range(views):
        flat_world = _depth_to_world(
            depth[view_idx].reshape(-1, 1),
            intrinsics[view_idx],
            extrinsics[view_idx],
            xx.reshape(-1),
            yy.reshape(-1),
        )
        out[view_idx] = flat_world[:, 0, :].reshape(height, width, 3)
    return out.astype(np.float32)


def _select_sources_by_anchor(
    target_view: int,
    source_views: list[int],
    images: np.ndarray,
    masks: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    roi: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[int], list[dict]]:
    ys, xs = np.nonzero(roi)
    if xs.size == 0:
        return [], []
    sample_count = min(4096, xs.size)
    if xs.size > sample_count:
        pick = np.linspace(0, xs.size - 1, sample_count).astype(np.int64)
        xs_eval = xs[pick].astype(np.float32)
        ys_eval = ys[pick].astype(np.float32)
    else:
        xs_eval = xs.astype(np.float32)
        ys_eval = ys.astype(np.float32)
    target_rgb = images[target_view, ys_eval.astype(np.int64), xs_eval.astype(np.int64)]
    points = _depth_to_world(
        depth[target_view, ys_eval.astype(np.int64), xs_eval.astype(np.int64)][:, None],
        intrinsics[target_view],
        extrinsics[target_view],
        xs_eval,
        ys_eval,
    )
    records = []
    selected = []
    for src in source_views:
        u, v, z = _world_to_source_uv(points, intrinsics[src], extrinsics[src])
        u = u[:, 0]
        v = v[:, 0]
        z = z[:, 0]
        valid = (z > 0.05) & (u >= 1) & (u < images.shape[2] - 2) & (v >= 1) & (v < images.shape[1] - 2)
        valid &= _nearest_sample_bool(masks[src], u, v)
        if np.any(valid):
            sampled = _bilinear_sample(images[src], u[valid], v[valid])
            err = np.mean(np.abs(sampled - target_rgb[valid]), axis=-1)
            p75 = float(np.percentile(err, 75))
            median = float(np.median(err))
            valid_ratio = float(valid.mean())
        else:
            p75 = math.inf
            median = math.inf
            valid_ratio = 0.0
        keep = p75 <= float(args.max_source_anchor_p75) and valid_ratio >= float(args.min_anchor_valid_ratio)
        records.append({"source_view": int(src), "anchor_rgb_l1_p75": p75, "anchor_rgb_l1_median": median, "valid_ratio": valid_ratio, "selected": bool(keep)})
        if keep:
            selected.append(src)
    if len(selected) < int(args.min_source_views):
        selected = list(source_views)
        for record in records:
            record["selected"] = int(record["source_view"]) in selected
            record["fallback_selected_all_sources"] = True
    return selected, records


def _photometric_sweep_view(
    target_view: int,
    source_views: list[int],
    images: np.ndarray,
    masks: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    roi: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict, dict[str, np.ndarray]]:
    height, width = roi.shape
    deltas = np.linspace(-float(args.max_depth_delta), float(args.max_depth_delta), int(args.num_depth_samples), dtype=np.float32)
    if not np.any(np.isclose(deltas, 0.0)):
        deltas = np.sort(np.unique(np.concatenate([deltas, np.array([0.0], dtype=np.float32)]))).astype(np.float32)
    zero_idx = int(np.argmin(np.abs(deltas)))
    patched_depth = depth[target_view].copy()
    best_delta_map = np.zeros((height, width), dtype=np.float32)
    best_cost_map = np.full((height, width), np.nan, dtype=np.float32)
    anchor_cost_map = np.full((height, width), np.nan, dtype=np.float32)
    valid_count_map = np.zeros((height, width), dtype=np.uint8)
    accepted = np.zeros((height, width), dtype=bool)
    finite_best = np.zeros((height, width), dtype=bool)
    boundary_best_map = np.zeros((height, width), dtype=bool)
    gradients = np.stack([_gray_gradient(image) for image in images], axis=0)
    offsets = [(0, 0)]
    radius = max(0, int(args.patch_radius))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if (dy, dx) != (0, 0):
                offsets.append((dy, dx))
    ys_all, xs_all = np.nonzero(roi & np.isfinite(depth[target_view]) & (depth[target_view] > 0.05))
    cost_curve_sum = np.zeros(len(deltas), dtype=np.float64)
    cost_curve_count = np.zeros(len(deltas), dtype=np.int64)
    chunk_size = max(256, int(args.chunk_size))
    for start in range(0, xs_all.size, chunk_size):
        end = min(xs_all.size, start + chunk_size)
        xs = xs_all[start:end].astype(np.float32)
        ys = ys_all[start:end].astype(np.float32)
        anchor_depth = depth[target_view, ys.astype(np.int64), xs.astype(np.int64)]
        candidate_depths = np.clip(anchor_depth[:, None] + deltas[None, :], 0.05, None)
        points = _depth_to_world(candidate_depths, intrinsics[target_view], extrinsics[target_view], xs, ys)
        source_costs = []
        source_valids = []
        for src in source_views:
            u, v, z = _world_to_source_uv(points, intrinsics[src], extrinsics[src])
            valid = (z > 0.05) & (u >= radius + 1) & (u < width - radius - 2) & (v >= radius + 1) & (v < height - radius - 2)
            valid &= _nearest_sample_bool(masks[src], u.reshape(-1), v.reshape(-1)).reshape(u.shape)
            if float(args.source_depth_tolerance) > 0:
                sampled_depth = _bilinear_sample(depth[src], u.reshape(-1), v.reshape(-1)).reshape(u.shape)
                valid &= np.abs(sampled_depth - z) <= float(args.source_depth_tolerance)
            rgb_cost_sum = np.zeros_like(candidate_depths, dtype=np.float32)
            grad_cost_sum = np.zeros_like(candidate_depths, dtype=np.float32)
            sample_count = np.zeros_like(candidate_depths, dtype=np.float32)
            for dy, dx in offsets:
                target_x = xs.astype(np.int64) + dx
                target_y = ys.astype(np.int64) + dy
                target_ok = (target_x >= 0) & (target_x < width) & (target_y >= 0) & (target_y < height)
                if not np.any(target_ok):
                    continue
                src_u = u + dx
                src_v = v + dy
                src_ok = (src_u >= 0) & (src_u < width - 1) & (src_v >= 0) & (src_v < height - 1)
                both = src_ok & target_ok[:, None]
                if not np.any(both):
                    continue
                sampled_rgb = _bilinear_sample(images[src], src_u.reshape(-1), src_v.reshape(-1)).reshape((*src_u.shape, 3))
                sampled_grad = _bilinear_sample(gradients[src], src_u.reshape(-1), src_v.reshape(-1)).reshape(src_u.shape)
                target_rgb = images[target_view, np.clip(target_y, 0, height - 1), np.clip(target_x, 0, width - 1)]
                target_grad = gradients[target_view, np.clip(target_y, 0, height - 1), np.clip(target_x, 0, width - 1)]
                rgb_l1 = np.mean(np.abs(sampled_rgb - target_rgb[:, None, :]), axis=-1)
                grad_l1 = np.abs(sampled_grad - target_grad[:, None])
                rgb_cost_sum += np.where(both, rgb_l1, 0.0).astype(np.float32)
                grad_cost_sum += np.where(both, grad_l1, 0.0).astype(np.float32)
                sample_count += both.astype(np.float32)
            sample_valid = sample_count > 0
            rgb_cost = rgb_cost_sum / np.clip(sample_count, 1.0, None)
            grad_cost = grad_cost_sum / np.clip(sample_count, 1.0, None)
            cost = (1.0 - float(args.gradient_weight)) * rgb_cost + float(args.gradient_weight) * grad_cost
            cost = np.minimum(cost, float(args.single_source_clip))
            valid &= sample_valid
            source_costs.append(np.where(valid, cost, np.nan).astype(np.float32))
            source_valids.append(valid)
        cost_stack = np.stack(source_costs, axis=-1)
        valid_stack = np.stack(source_valids, axis=-1)
        valid_counts = np.sum(valid_stack, axis=-1)
        with np.errstate(all="ignore"):
            photo_cost = np.nanmedian(cost_stack, axis=-1).astype(np.float32)
        enough = valid_counts >= int(args.min_source_views)
        total_cost = photo_cost + float(args.prior_weight) * (np.abs(deltas)[None, :] / max(float(args.max_depth_delta), 1e-6))
        total_cost[~enough] = np.inf
        best_idx = np.argmin(total_cost, axis=1)
        best_cost = total_cost[np.arange(total_cost.shape[0]), best_idx]
        anchor_cost = total_cost[:, zero_idx]
        best_valid = valid_counts[np.arange(valid_counts.shape[0]), best_idx]
        best_delta = deltas[best_idx]
        finite = np.isfinite(best_cost)
        boundary = (best_idx == 0) | (best_idx == len(deltas) - 1)
        accept = finite & np.isfinite(anchor_cost) & ((anchor_cost - best_cost) >= float(args.accept_margin))
        if bool(args.reject_boundary_best):
            accept &= ~boundary
        patched_depth[ys.astype(np.int64)[accept], xs.astype(np.int64)[accept]] = candidate_depths[np.arange(candidate_depths.shape[0])[accept], best_idx[accept]]
        best_delta_map[ys.astype(np.int64), xs.astype(np.int64)] = best_delta
        best_cost_map[ys.astype(np.int64), xs.astype(np.int64)] = best_cost
        anchor_cost_map[ys.astype(np.int64), xs.astype(np.int64)] = anchor_cost
        valid_count_map[ys.astype(np.int64), xs.astype(np.int64)] = np.clip(best_valid, 0, 255).astype(np.uint8)
        accepted[ys.astype(np.int64)[accept], xs.astype(np.int64)[accept]] = True
        finite_best[ys.astype(np.int64)[finite], xs.astype(np.int64)[finite]] = True
        boundary_best_map[ys.astype(np.int64)[boundary], xs.astype(np.int64)[boundary]] = True
        finite_costs = np.where(np.isfinite(photo_cost), photo_cost, np.nan)
        for plane_idx in range(len(deltas)):
            values = finite_costs[:, plane_idx]
            good = np.isfinite(values)
            if np.any(good):
                cost_curve_sum[plane_idx] += float(np.nansum(values[good]))
                cost_curve_count[plane_idx] += int(good.sum())
    roi_pixels = int(roi.sum())
    finite_roi = finite_best & roi
    accepted_roi = accepted & roi
    boundary_roi = boundary_best_map & finite_roi
    delta_accept = best_delta_map[accepted_roi]
    best_cost_roi = best_cost_map[finite_roi]
    valid_roi = valid_count_map[finite_roi]
    summary = {
        "target_view": int(target_view),
        "source_views": [int(src) for src in source_views],
        "roi_pixels": roi_pixels,
        "finite_best_pixels": int(finite_roi.sum()),
        "accepted_pixels": int(accepted_roi.sum()),
        "accepted_coverage": float(accepted_roi.sum() / max(roi_pixels, 1)),
        "finite_best_coverage": float(finite_roi.sum() / max(roi_pixels, 1)),
        "boundary_best_ratio": float(boundary_roi.sum() / max(finite_roi.sum(), 1)),
        "median_valid_sources": float(np.median(valid_roi)) if valid_roi.size else 0.0,
        "best_cost_p50": float(np.nanpercentile(best_cost_roi, 50)) if best_cost_roi.size else math.inf,
        "best_cost_p75": float(np.nanpercentile(best_cost_roi, 75)) if best_cost_roi.size else math.inf,
        "accepted_abs_delta_median": float(np.median(np.abs(delta_accept))) if delta_accept.size else 0.0,
        "accepted_abs_delta_p95": float(np.percentile(np.abs(delta_accept), 95)) if delta_accept.size else 0.0,
        "accepted_delta_min": float(np.min(delta_accept)) if delta_accept.size else 0.0,
        "accepted_delta_max": float(np.max(delta_accept)) if delta_accept.size else 0.0,
        "gate_thresholds": {
            "face_core_accepted_coverage": 0.75,
            "head_face_accepted_coverage": 0.60,
            "median_valid_sources": 3.0,
            "boundary_best_ratio_max": 0.15,
            "accepted_abs_delta_median_max": 0.03,
            "accepted_abs_delta_p95_max": 0.05,
            "best_cost_p50_max": 0.06,
            "best_cost_p75_max": 0.12,
        },
    }
    cost_curve = {
        "deltas": deltas.astype(float).tolist(),
        "mean_photo_cost": [
            float(cost_curve_sum[idx] / cost_curve_count[idx]) if cost_curve_count[idx] > 0 else math.inf
            for idx in range(len(deltas))
        ],
        "counts": cost_curve_count.astype(int).tolist(),
    }
    diagnostics = {
        "roi": roi.astype(bool),
        "accepted": accepted.astype(bool),
        "finite_best": finite_best.astype(bool),
        "boundary_best": boundary_best_map.astype(bool),
        "best_delta": best_delta_map.astype(np.float32),
        "best_cost": best_cost_map.astype(np.float32),
        "anchor_cost": anchor_cost_map.astype(np.float32),
        "valid_count": valid_count_map.astype(np.uint8),
        "patched_depth": patched_depth.astype(np.float32),
        "cost_curve_deltas": np.asarray(cost_curve["deltas"], dtype=np.float32),
        "cost_curve_mean_photo_cost": np.asarray(cost_curve["mean_photo_cost"], dtype=np.float32),
    }
    return patched_depth.astype(np.float32), accepted, summary | {"cost_curve": cost_curve}, diagnostics


def _depth_rgb(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = depth[mask & np.isfinite(depth)]
    if values.size < 16:
        values = depth[np.isfinite(depth)]
    lo, hi = np.percentile(values, [2, 98]) if values.size else (0.0, 1.0)
    gray = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    gray[~np.isfinite(gray)] = 1.0
    return np.stack([gray, gray, gray], axis=-1)


def _delta_rgb(delta: np.ndarray, mask: np.ndarray, max_abs: float) -> np.ndarray:
    out = np.ones((*delta.shape, 3), dtype=np.float32)
    norm = np.clip(delta / max(max_abs, 1e-6), -1.0, 1.0)
    out[..., 0] = np.where(norm > 0, 1.0, 1.0 + norm)
    out[..., 1] = 1.0 - np.abs(norm)
    out[..., 2] = np.where(norm < 0, 1.0, 1.0 - norm)
    out[~mask] = 1.0
    return np.clip(out, 0.0, 1.0)


def _cost_rgb(cost: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = cost[mask & np.isfinite(cost)]
    hi = np.percentile(values, 95) if values.size else 1.0
    norm = np.clip(cost / max(float(hi), 1e-6), 0.0, 1.0)
    out = np.stack([norm, 1.0 - norm, np.full_like(norm, 0.2)], axis=-1)
    out[~mask | ~np.isfinite(cost)] = 1.0
    return np.clip(out, 0.0, 1.0)


def _mask_overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    out = rgb.copy()
    out[mask] = 0.55 * out[mask] + 0.45 * np.asarray(color, dtype=np.float32)
    return np.clip(out, 0.0, 1.0)


def _save_image(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)).save(path)


def _save_cost_curve(cost_curve: dict, out_path: Path) -> None:
    deltas = np.asarray(cost_curve["deltas"], dtype=np.float32)
    costs = np.asarray(cost_curve["mean_photo_cost"], dtype=np.float32)
    finite = np.isfinite(costs)
    canvas = Image.new("RGB", (640, 360), "white")
    draw = ImageDraw.Draw(canvas)
    margin = 48
    draw.rectangle((margin, margin, 620, 310), outline=(0, 0, 0))
    if np.any(finite):
        x0, x1 = float(deltas.min()), float(deltas.max())
        y0, y1 = float(np.nanmin(costs[finite])), float(np.nanmax(costs[finite]))
        if abs(y1 - y0) < 1e-6:
            y1 = y0 + 1e-3
        points = []
        for delta, cost in zip(deltas, costs):
            if not np.isfinite(cost):
                continue
            x = margin + (float(delta) - x0) / max(x1 - x0, 1e-6) * (620 - margin)
            y = 310 - (float(cost) - y0) / max(y1 - y0, 1e-6) * (310 - margin)
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=(220, 40, 40), width=2)
        for point in points:
            draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill=(20, 20, 20))
        draw.text((margin, 16), f"mean photo cost vs depth delta; min={float(np.nanmin(costs[finite])):.4f}", fill=(0, 0, 0))
        draw.text((margin, 318), f"delta range {x0:.3f}m .. {x1:.3f}m", fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _save_view_diagnostics(
    out_dir: Path,
    prefix: str,
    rgb: np.ndarray,
    anchor_depth: np.ndarray,
    diagnostics: dict[str, np.ndarray],
    summary: dict,
    args: argparse.Namespace,
) -> None:
    roi = diagnostics["roi"].astype(bool)
    accepted = diagnostics["accepted"].astype(bool)
    patched_depth = diagnostics["patched_depth"]
    best_delta = diagnostics["best_delta"]
    best_cost = diagnostics["best_cost"]
    valid_count = diagnostics["valid_count"].astype(np.float32)
    _save_image(_mask_overlay(rgb, roi, (1.0, 0.0, 0.0)), out_dir / f"{prefix}_roi_overlay.png")
    _save_image(_depth_rgb(anchor_depth, roi), out_dir / f"{prefix}_anchor_depth.png")
    _save_image(_depth_rgb(patched_depth, roi), out_dir / f"{prefix}_best_depth.png")
    _save_image(_delta_rgb(best_delta, roi, float(args.max_depth_delta)), out_dir / f"{prefix}_delta_depth.png")
    _save_image(_cost_rgb(best_cost, roi), out_dir / f"{prefix}_best_cost.png")
    valid_norm = np.clip(valid_count / max(float(len(summary["source_views"])), 1.0), 0.0, 1.0)
    valid_rgb = np.stack([1.0 - valid_norm, valid_norm, np.zeros_like(valid_norm)], axis=-1)
    valid_rgb[~roi] = 1.0
    valid_rgb[accepted] = np.array([0.1, 0.3, 1.0])
    _save_image(valid_rgb, out_dir / f"{prefix}_accepted_mask_valid_sources.png")
    _save_cost_curve(summary["cost_curve"], out_dir / f"{prefix}_cost_curve_median.png")
    tiles = [
        ("RGB", rgb),
        ("ROI", _mask_overlay(rgb, roi, (1.0, 0.0, 0.0))),
        ("accepted", _mask_overlay(rgb, accepted, (0.0, 0.2, 1.0))),
        ("delta", _delta_rgb(best_delta, roi, float(args.max_depth_delta))),
        ("cost", _cost_rgb(best_cost, roi)),
        ("best depth", _depth_rgb(patched_depth, roi)),
    ]
    tile_h, tile_w = rgb.shape[:2]
    canvas = Image.new("RGB", (tile_w * 3, tile_h * 2 + 48), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, tile) in enumerate(tiles):
        x = (idx % 3) * tile_w
        y = 24 + (idx // 3) * tile_h
        canvas.paste(Image.fromarray((np.clip(tile, 0.0, 1.0) * 255).astype(np.uint8)), (x, y))
        draw.text((x + 4, y - 18), label, fill=(0, 0, 0))
    draw.text(
        (4, 4),
        f"accepted={summary['accepted_pixels']}/{summary['roi_pixels']} cov={summary['accepted_coverage']:.3f} "
        f"boundary={summary['boundary_best_ratio']:.3f} p50cost={summary['best_cost_p50']:.4f}",
        fill=(0, 0, 0),
    )
    canvas.save(out_dir / f"{prefix}_summary_sheet.png")


def _copy_sidecars(src_dir: Path, dst_dir: Path) -> None:
    for child in src_dir.iterdir():
        if child.is_file() and child.name != "predictions.npz":
            target = dst_dir / child.name
            if target.exists():
                continue
            shutil.copy2(child, target)


def main() -> int:
    args = parse_args()
    predictions_path = Path(args.predictions_npz)
    scene_dir = Path(args.scene_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(predictions_path, allow_pickle=False)
    payload = {key: np.array(data[key]) for key in data.files}
    depth = np.asarray(payload["depth"], dtype=np.float32)[..., 0]
    intrinsics = np.asarray(payload["intrinsic"], dtype=np.float32)
    extrinsics = np.asarray(payload["extrinsic"], dtype=np.float32)
    original_world_points = np.asarray(payload["world_points"], dtype=np.float32)
    views, height, width = depth.shape
    images, masks, sorted_names, order_check = _load_scene_images_and_masks(scene_dir, height, width)
    if images.shape[0] != views:
        raise ValueError(f"scene view count {images.shape[0]} != predictions view count {views}")
    target_views = _parse_indices(args.view_indices, views)
    base_source_views = _parse_indices(args.source_views, views)
    patched_depth = depth.copy()
    patched_world = original_world_points.copy()
    all_records = []
    all_diag_payload = {}
    for target_view in target_views:
        available_sources = [idx for idx in base_source_views if idx != target_view]
        target_mask = _erode_mask(masks[target_view], int(args.mask_erode_pixels))
        roi = _roi_mask(target_mask, args.roi_kind)
        selected_sources, source_records = _select_sources_by_anchor(
            target_view,
            available_sources,
            images,
            masks,
            depth,
            intrinsics,
            extrinsics,
            roi,
            args,
        )
        if len(selected_sources) < int(args.min_source_views):
            raise RuntimeError(f"target view {target_view} has too few selected sources: {selected_sources}")
        view_patched_depth, accepted, summary, diagnostics = _photometric_sweep_view(
            target_view,
            selected_sources,
            images,
            masks,
            depth,
            intrinsics,
            extrinsics,
            roi,
            args,
        )
        patched_depth[target_view][accepted] = view_patched_depth[accepted]
        accepted_ys, accepted_xs = np.nonzero(accepted)
        if accepted_xs.size:
            accepted_depth = patched_depth[target_view, accepted_ys, accepted_xs][:, None]
            accepted_world = _depth_to_world(
                accepted_depth,
                intrinsics[target_view],
                extrinsics[target_view],
                accepted_xs.astype(np.float32),
                accepted_ys.astype(np.float32),
            )
            patched_world[target_view, accepted_ys, accepted_xs] = accepted_world[:, 0, :].astype(np.float32)
        prefix = f"view{target_view:02d}"
        summary["source_selection_records"] = source_records
        summary["roi_kind"] = args.roi_kind
        summary["selected_source_basenames"] = [sorted_names[idx] for idx in selected_sources]
        all_records.append(summary)
        for key, value in diagnostics.items():
            all_diag_payload[f"{prefix}_{key}"] = value
        _save_view_diagnostics(output_dir, prefix, images[target_view], depth[target_view], diagnostics, summary, args)
    payload["depth"] = patched_depth[..., None].astype(np.float32)
    payload["world_points"] = patched_world.astype(np.float32)
    np.savez_compressed(output_dir / "predictions.npz", **payload)
    np.savez_compressed(output_dir / "photometric_depth_refine_diagnostics.npz", **all_diag_payload)
    if args.copy_sidecar_files:
        _copy_sidecars(predictions_path.parent, output_dir)
    summary = {
        "predictions_npz": str(predictions_path.resolve()),
        "scene_dir": str(scene_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "view_indices": [int(idx) for idx in target_views],
        "source_views_requested": [int(idx) for idx in base_source_views],
        "roi_kind": args.roi_kind,
        "max_depth_delta": float(args.max_depth_delta),
        "num_depth_samples": int(args.num_depth_samples),
        "patch_radius": int(args.patch_radius),
        "min_source_views": int(args.min_source_views),
        "source_depth_tolerance": float(args.source_depth_tolerance),
        "accept_margin": float(args.accept_margin),
        "prior_weight": float(args.prior_weight),
        "gradient_weight": float(args.gradient_weight),
        "confidence_policy": "unchanged",
        "world_points_policy": "only_accepted_target_roi_pixels_recomputed; all other world_points preserved",
        "order_check": order_check,
        "records": all_records,
    }
    (output_dir / "photometric_depth_refine_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
