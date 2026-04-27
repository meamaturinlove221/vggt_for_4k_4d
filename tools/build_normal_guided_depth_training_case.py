from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt.utils.normal_refiner import face_box_from_mask, head_box_from_mask, shoulder_box_from_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 4K4D training case with normal-guided local depth/point pseudo targets."
    )
    parser.add_argument("--source-case-dir", required=True)
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--anchor-predictions-npz", required=True, help="Trusted baseline predictions used as depth anchor")
    parser.add_argument("--external-normal-npz", required=True, help="Sapiens/external normal NPZ in scene view order")
    parser.add_argument("--output-case-dir", required=True)
    parser.add_argument("--output-diagnostics-dir", required=True)
    parser.add_argument("--transform", default="flip-yz", choices=("identity", "neg", "negxy", "negxyz", "flip-yz", "sapiens_cam", "negz"))
    parser.add_argument("--iterations", type=int, default=220)
    parser.add_argument("--step", type=float, default=0.18)
    parser.add_argument("--anchor-weight", type=float, default=0.08)
    parser.add_argument("--smooth-weight", type=float, default=0.04)
    parser.add_argument("--gradient-scale", type=float, default=0.45)
    parser.add_argument("--max-depth-delta", type=float, default=0.035)
    parser.add_argument("--roi-kind", choices=("head", "face", "head_face", "shoulder", "all"), default="head_face")
    parser.add_argument("--conf-boost", type=float, default=64.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _copy_case(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(dst)
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _transform(normals: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(normals, dtype=np.float32).copy()
    if name == "identity":
        pass
    elif name == "neg":
        out *= -1.0
    elif name == "negxy":
        out[..., 0:2] *= -1.0
    elif name == "negxyz":
        out *= -1.0
    elif name == "flip-yz":
        out[..., 1:3] *= -1.0
    elif name == "sapiens_cam":
        out[..., 1] *= -1.0
    elif name == "negz":
        out[..., 2] *= -1.0
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    out = out / np.clip(norm, 1e-6, None)
    out[norm[..., 0] < 1e-6] = 0.0
    return out.astype(np.float32)


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
    return out


def _cam_to_world(cam_points: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    out = np.zeros_like(cam_points, dtype=np.float32)
    for view_idx in range(cam_points.shape[0]):
        rot = extrinsics[view_idx, :, :3].astype(np.float32)
        trans = extrinsics[view_idx, :, 3].astype(np.float32)
        flat = cam_points[view_idx].reshape(-1, 3)
        out[view_idx] = ((flat - trans[None]) @ rot).reshape(cam_points.shape[1:])
    return out.astype(np.float32)


def _integrate_depth_single(depth: np.ndarray, normal: np.ndarray, mask: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    z = np.asarray(depth, dtype=np.float32).copy()
    anchor = z.copy()
    valid = mask & np.isfinite(z) & np.isfinite(normal).all(axis=-1) & (np.abs(normal[..., 2]) > 0.25)
    target_dx = -normal[..., 0] / np.clip(normal[..., 2], 0.25, None) * float(args.gradient_scale)
    target_dy = -normal[..., 1] / np.clip(normal[..., 2], 0.25, None) * float(args.gradient_scale)
    target_dx = np.clip(target_dx, -0.02, 0.02)
    target_dy = np.clip(target_dy, -0.02, 0.02)
    for _ in range(int(args.iterations)):
        dx = np.zeros_like(z)
        dy = np.zeros_like(z)
        dx[:, :-1] = z[:, 1:] - z[:, :-1]
        dy[:-1, :] = z[1:, :] - z[:-1, :]
        err_x = (dx - target_dx) * valid
        err_y = (dy - target_dy) * valid
        grad = np.zeros_like(z)
        grad[:, :-1] -= err_x[:, :-1]
        grad[:, 1:] += err_x[:, :-1]
        grad[:-1, :] -= err_y[:-1, :]
        grad[1:, :] += err_y[:-1, :]
        lap = np.zeros_like(z)
        lap[1:-1, 1:-1] = (
            4.0 * z[1:-1, 1:-1]
            - z[:-2, 1:-1]
            - z[2:, 1:-1]
            - z[1:-1, :-2]
            - z[1:-1, 2:]
        )
        grad += float(args.anchor_weight) * (z - anchor)
        grad += float(args.smooth_weight) * lap
        z[valid] -= float(args.step) * grad[valid]
        z[~valid] = anchor[~valid]
    max_delta = float(args.max_depth_delta)
    delta = np.clip(z - anchor, -max_delta, max_delta)
    out = anchor.copy()
    out[valid] = anchor[valid] + delta[valid]
    return out.astype(np.float32)


def _make_preview(rgb: np.ndarray, anchor: np.ndarray, refined: np.ndarray, normal: np.ndarray, mask: np.ndarray, out: Path) -> None:
    def norm_depth(d):
        vals = d[mask]
        if vals.size == 0:
            vals = d.reshape(-1)
        lo, hi = np.percentile(vals, [2, 98])
        return ((np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8))
    nrm = np.clip((normal + 1.0) * 0.5, 0, 1)
    nrm[~mask] = 1.0
    tiles = [rgb, np.stack([norm_depth(anchor)] * 3, -1), np.stack([norm_depth(refined)] * 3, -1), (nrm * 255).astype(np.uint8)]
    labels = ["RGB", "anchor depth", "normal-guided depth", "teacher normal"]
    canvas = Image.new("RGB", (rgb.shape[1] * 4, rgb.shape[0] + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (tile, label) in enumerate(zip(tiles, labels)):
        canvas.paste(Image.fromarray(tile.astype(np.uint8)), (idx * rgb.shape[1], 24))
        draw.text((idx * rgb.shape[1] + 4, 4), label, fill=(0, 0, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def _roi_mask(mask: np.ndarray, roi_kind: str) -> np.ndarray:
    if roi_kind == "all":
        return np.asarray(mask, dtype=bool)
    boxes = []
    if roi_kind in {"head", "head_face"}:
        boxes.append(head_box_from_mask(mask))
    if roi_kind in {"face", "head_face"}:
        boxes.append(face_box_from_mask(mask))
    if roi_kind == "shoulder":
        boxes.append(shoulder_box_from_mask(mask))
    out = np.zeros(mask.shape, dtype=bool)
    for box in boxes:
        if box is None:
            continue
        x0, y0, x1, y1 = box
        out[y0:y1, x0:x1] |= mask[y0:y1, x0:x1]
    return out


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
    anchor = np.load(args.anchor_predictions_npz, allow_pickle=False)
    external = np.load(args.external_normal_npz, allow_pickle=False)
    normals = external["normal"] if "normal" in external.files else external["normal_rgb"]
    if normals.max() > 2.0:
        normals = normals / 127.5 - 1.0
    normals = _transform(normals, args.transform)
    ext_mask = external["mask"].astype(bool) if "mask" in external.files else np.ones(normals.shape[:3], dtype=bool)
    point_mask = np.asarray(inputs.get("point_masks"), dtype=bool)
    roi_masks = np.stack([_roi_mask(point_mask[view_idx], args.roi_kind) for view_idx in range(point_mask.shape[0])], axis=0)
    teacher_mask = ext_mask & point_mask & roi_masks & np.isfinite(normals).all(axis=-1)
    depth_anchor = np.asarray(anchor["depth"], dtype=np.float32)[..., 0]
    intrinsics = np.asarray(targets["intrinsics"], dtype=np.float32)
    extrinsics = np.asarray(targets["extrinsics"], dtype=np.float32)
    refined_depth = np.zeros_like(depth_anchor, dtype=np.float32)
    for view_idx in range(depth_anchor.shape[0]):
        refined_depth[view_idx] = _integrate_depth_single(depth_anchor[view_idx], normals[view_idx], teacher_mask[view_idx], args)
        _make_preview(
            np.asarray(inputs["images"][view_idx], dtype=np.uint8),
            depth_anchor[view_idx],
            refined_depth[view_idx],
            normals[view_idx],
            teacher_mask[view_idx],
            diagnostics_dir / f"{view_idx:02d}_normal_guided_depth_preview.png",
        )
    cam_points = _depth_to_cam_points(refined_depth, intrinsics)
    world_points = _cam_to_world(cam_points, extrinsics)
    targets["depths"] = np.asarray(targets["depths"], dtype=np.float32)
    targets["cam_points"] = np.asarray(targets["cam_points"], dtype=np.float32)
    targets["world_points"] = np.asarray(targets["world_points"], dtype=np.float32)
    targets["depth_conf"] = np.asarray(targets["depth_conf"], dtype=np.float32)
    targets["world_points_conf"] = np.asarray(targets["world_points_conf"], dtype=np.float32)
    targets["depths"][teacher_mask] = refined_depth[teacher_mask]
    targets["cam_points"][teacher_mask] = cam_points[teacher_mask]
    targets["world_points"][teacher_mask] = world_points[teacher_mask]
    targets["depth_conf"][teacher_mask] = np.maximum(targets["depth_conf"][teacher_mask], float(args.conf_boost))
    targets["world_points_conf"][teacher_mask] = np.maximum(targets["world_points_conf"][teacher_mask], float(args.conf_boost))
    targets["teacher_normals"] = normals.astype(np.float32)
    targets["teacher_mask"] = teacher_mask.astype(bool)
    targets["prior_normals"] = np.asarray(targets.get("prior_normals", normals), dtype=np.float32)
    targets["prior_normals"][teacher_mask] = normals[teacher_mask]
    for key in ["head_roi_mask", "face_roi_mask", "hairline_mask", "ear_band_mask"]:
        if key in targets:
            targets[key] = (np.asarray(targets[key], dtype=bool) | teacher_mask).astype(bool)
    np.savez_compressed(output_case / "targets.npz", **targets)
    delta = refined_depth - depth_anchor
    summary = {
        "source_case_dir": str(source_case.resolve()),
        "anchor_predictions_npz": str(Path(args.anchor_predictions_npz).resolve()),
        "external_normal_npz": str(Path(args.external_normal_npz).resolve()),
        "output_case_dir": str(output_case.resolve()),
        "transform": args.transform,
        "teacher_mask_pixels": [int(v) for v in teacher_mask.reshape(teacher_mask.shape[0], -1).sum(axis=1)],
        "roi_kind": args.roi_kind,
        "depth_delta_mean_abs": float(np.mean(np.abs(delta[teacher_mask]))) if teacher_mask.any() else 0.0,
        "depth_delta_max_abs": float(np.max(np.abs(delta[teacher_mask]))) if teacher_mask.any() else 0.0,
        "iterations": int(args.iterations),
        "max_depth_delta": float(args.max_depth_delta),
        "notes": "Normal-guided depth is anchored to signfix ckpt4 and clipped to avoid pseudo-depth collapse.",
    }
    (diagnostics_dir / "normal_guided_depth_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest_path = output_case / "case_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    manifest["normal_guided_depth_training_patch"] = summary
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
