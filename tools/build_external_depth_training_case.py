from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt.utils.normal_refiner import face_box_from_mask, head_box_from_mask, shoulder_box_from_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 4K4D training case from external relative depth aligned to a trusted VGGT anchor."
    )
    parser.add_argument("--source-case-dir", required=True)
    parser.add_argument("--external-depth-npz", required=True)
    parser.add_argument("--anchor-predictions-npz", required=True)
    parser.add_argument("--output-case-dir", required=True)
    parser.add_argument("--output-diagnostics-dir", required=True)
    parser.add_argument("--align-roi-kind", choices=("head", "face", "face_core", "head_face", "shoulder", "all"), default="head")
    parser.add_argument("--apply-roi-kind", choices=("head", "face", "face_core", "head_face", "shoulder", "all"), default="face_core")
    parser.add_argument("--max-depth-delta", type=float, default=0.025)
    parser.add_argument("--conf-boost", type=float, default=8.0)
    parser.add_argument("--update-prior", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _copy_case(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(dst)
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


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
        if x1 <= x0 or y1 <= y0:
            continue
        out[y0:y1, x0:x1] |= mask[y0:y1, x0:x1]
    return out


def _fit_affine(external: np.ndarray, anchor: np.ndarray, mask: np.ndarray) -> tuple[float, float, dict]:
    valid = mask & np.isfinite(external) & np.isfinite(anchor)
    if valid.sum() < 64:
        return 0.0, float(np.nanmedian(anchor[np.isfinite(anchor)])), {"valid": int(valid.sum()), "fallback": "too_few_points"}
    x = external[valid].astype(np.float64)
    y = anchor[valid].astype(np.float64)
    design = np.stack([x, np.ones_like(x)], axis=1)
    scale, bias = np.linalg.lstsq(design, y, rcond=None)[0]
    aligned = scale * x + bias
    err = aligned - y
    return float(scale), float(bias), {
        "valid": int(valid.sum()),
        "scale": float(scale),
        "bias": float(bias),
        "mae": float(np.mean(np.abs(err))),
        "p95_abs": float(np.percentile(np.abs(err), 95)),
        "corr": float(np.corrcoef(x, y)[0, 1]) if x.size > 1 and np.std(x) > 1e-8 and np.std(y) > 1e-8 else 0.0,
    }


def _depth_to_cam_points(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    views, height, width = depth.shape
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    out = np.zeros((views, height, width, 3), dtype=np.float32)
    for view_idx in range(views):
        fx = float(intrinsics[view_idx, 0, 0])
        fy = float(intrinsics[view_idx, 1, 1])
        cx = float(intrinsics[view_idx, 0, 2])
        cy = float(intrinsics[view_idx, 1, 2])
        z = depth[view_idx]
        out[view_idx, ..., 0] = (xx - cx) / max(fx, 1e-6) * z
        out[view_idx, ..., 1] = (yy - cy) / max(fy, 1e-6) * z
        out[view_idx, ..., 2] = z
    return out.astype(np.float32)


def _cam_to_world(cam_points: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    out = np.zeros_like(cam_points, dtype=np.float32)
    for view_idx in range(cam_points.shape[0]):
        rotation = extrinsics[view_idx, :, :3].astype(np.float32)
        translation = extrinsics[view_idx, :, 3].astype(np.float32)
        flat = cam_points[view_idx].reshape(-1, 3)
        out[view_idx] = ((flat - translation[None]) @ rotation).reshape(cam_points.shape[1:])
    return out.astype(np.float32)


def _depth_rgb(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = depth[mask & np.isfinite(depth)]
    if values.size < 16:
        values = depth[np.isfinite(depth)]
    lo, hi = np.percentile(values, [2, 98]) if values.size else (0.0, 1.0)
    gray = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    gray[~np.isfinite(gray)] = 1.0
    return np.stack([gray, gray, gray], axis=-1)


def _make_preview(
    rgb: np.ndarray,
    anchor: np.ndarray,
    aligned: np.ndarray,
    patched: np.ndarray,
    mask: np.ndarray,
    out_path: Path,
) -> None:
    diff = np.zeros((*anchor.shape, 3), dtype=np.float32)
    delta = np.clip((patched - anchor) / 0.05, -1.0, 1.0)
    diff[..., 0] = np.clip(delta, 0.0, 1.0)
    diff[..., 2] = np.clip(-delta, 0.0, 1.0)
    diff[~mask] = 1.0
    overlay = rgb.astype(np.float32).copy()
    overlay[mask] = 0.55 * overlay[mask] + np.array([255.0, 0.0, 0.0]) * 0.45
    tiles = [
        rgb.astype(np.uint8),
        (_depth_rgb(anchor, mask) * 255).astype(np.uint8),
        (_depth_rgb(aligned, mask) * 255).astype(np.uint8),
        (diff * 255).astype(np.uint8),
        overlay.astype(np.uint8),
    ]
    labels = ["RGB", "anchor depth", "aligned external depth", "patched-anchor delta", "patch mask"]
    canvas = Image.new("RGB", (rgb.shape[1] * len(tiles), rgb.shape[0] + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (tile, label) in enumerate(zip(tiles, labels)):
        canvas.paste(Image.fromarray(tile), (idx * rgb.shape[1], 24))
        draw.text((idx * rgb.shape[1] + 4, 4), label, fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> int:
    args = parse_args()
    source_case = Path(args.source_case_dir)
    output_case = Path(args.output_case_dir)
    diagnostics_dir = Path(args.output_diagnostics_dir)
    _copy_case(source_case, output_case, overwrite=bool(args.overwrite))

    with np.load(output_case / "inputs.npz", allow_pickle=False) as payload:
        inputs = {key: np.array(payload[key]) for key in payload.files}
    with np.load(output_case / "targets.npz", allow_pickle=False) as payload:
        targets = {key: np.array(payload[key]) for key in payload.files}
    external = np.load(args.external_depth_npz, allow_pickle=False)
    anchor = np.load(args.anchor_predictions_npz, allow_pickle=False)

    external_depth = np.asarray(external["depth"], dtype=np.float32)
    external_mask = np.asarray(external["mask"], dtype=bool) if "mask" in external.files else external_depth > 0
    point_mask = np.asarray(inputs["point_masks"], dtype=bool)
    anchor_depth = np.asarray(anchor["depth"], dtype=np.float32)[..., 0]
    if external_depth.shape != anchor_depth.shape:
        raise ValueError(f"shape mismatch: external {external_depth.shape} vs anchor {anchor_depth.shape}")

    align_masks = np.stack([_roi_mask(point_mask[idx], args.align_roi_kind) for idx in range(point_mask.shape[0])], axis=0)
    apply_masks = np.stack([_roi_mask(point_mask[idx], args.apply_roi_kind) for idx in range(point_mask.shape[0])], axis=0)
    align_masks &= point_mask & external_mask
    apply_masks &= point_mask & external_mask

    patched_depth = anchor_depth.copy()
    aligned_depth = np.zeros_like(anchor_depth, dtype=np.float32)
    records = []
    for view_idx in range(anchor_depth.shape[0]):
        scale, bias, fit = _fit_affine(external_depth[view_idx], anchor_depth[view_idx], align_masks[view_idx])
        aligned = scale * external_depth[view_idx] + bias
        aligned_depth[view_idx] = aligned.astype(np.float32)
        delta = np.clip(aligned - anchor_depth[view_idx], -float(args.max_depth_delta), float(args.max_depth_delta))
        patched_depth[view_idx][apply_masks[view_idx]] = anchor_depth[view_idx][apply_masks[view_idx]] + delta[apply_masks[view_idx]]
        _make_preview(
            np.asarray(inputs["images"][view_idx], dtype=np.uint8),
            anchor_depth[view_idx],
            aligned_depth[view_idx],
            patched_depth[view_idx],
            apply_masks[view_idx],
            diagnostics_dir / f"{view_idx:02d}_external_depth_alignment_preview.png",
        )
        records.append(
            {
                "view": int(view_idx),
                "fit": fit,
                "align_pixels": int(align_masks[view_idx].sum()),
                "apply_pixels": int(apply_masks[view_idx].sum()),
                "patched_delta_mean_abs": float(np.mean(np.abs((patched_depth[view_idx] - anchor_depth[view_idx])[apply_masks[view_idx]]))) if apply_masks[view_idx].any() else 0.0,
                "patched_delta_max_abs": float(np.max(np.abs((patched_depth[view_idx] - anchor_depth[view_idx])[apply_masks[view_idx]]))) if apply_masks[view_idx].any() else 0.0,
            }
        )

    intrinsics = np.asarray(targets["intrinsics"], dtype=np.float32)
    extrinsics = np.asarray(targets["extrinsics"], dtype=np.float32)
    cam_points = _depth_to_cam_points(patched_depth, intrinsics)
    world_points = _cam_to_world(cam_points, extrinsics)
    targets["depths"] = np.asarray(targets["depths"], dtype=np.float32)
    targets["cam_points"] = np.asarray(targets["cam_points"], dtype=np.float32)
    targets["world_points"] = np.asarray(targets["world_points"], dtype=np.float32)
    targets["depth_conf"] = np.asarray(targets["depth_conf"], dtype=np.float32)
    targets["world_points_conf"] = np.asarray(targets["world_points_conf"], dtype=np.float32)
    targets["depths"][apply_masks] = patched_depth[apply_masks]
    targets["cam_points"][apply_masks] = cam_points[apply_masks]
    targets["world_points"][apply_masks] = world_points[apply_masks]
    targets["depth_conf"][apply_masks] = np.maximum(targets["depth_conf"][apply_masks], float(args.conf_boost))
    targets["world_points_conf"][apply_masks] = np.maximum(targets["world_points_conf"][apply_masks], float(args.conf_boost))
    targets["teacher_mask"] = apply_masks.astype(bool)
    if bool(args.update_prior):
        targets["prior_depths"] = np.asarray(targets.get("prior_depths", anchor_depth), dtype=np.float32)
        targets["prior_points"] = np.asarray(targets.get("prior_points", world_points), dtype=np.float32)
        targets["prior_depths"][apply_masks] = patched_depth[apply_masks]
        targets["prior_points"][apply_masks] = world_points[apply_masks]
    for key in ("head_roi_mask", "face_roi_mask", "hairline_mask", "ear_band_mask"):
        if key in targets:
            targets[key] = (np.asarray(targets[key], dtype=bool) | apply_masks).astype(bool)

    np.savez_compressed(output_case / "targets.npz", **targets)
    summary = {
        "source_case_dir": str(source_case.resolve()),
        "external_depth_npz": str(Path(args.external_depth_npz).resolve()),
        "anchor_predictions_npz": str(Path(args.anchor_predictions_npz).resolve()),
        "output_case_dir": str(output_case.resolve()),
        "align_roi_kind": args.align_roi_kind,
        "apply_roi_kind": args.apply_roi_kind,
        "max_depth_delta": float(args.max_depth_delta),
        "conf_boost": float(args.conf_boost),
        "update_prior": bool(args.update_prior),
        "records": records,
    }
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    (diagnostics_dir / "external_depth_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest_path = output_case / "case_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    manifest["external_depth_training_patch"] = summary
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
