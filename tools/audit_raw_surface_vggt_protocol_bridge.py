from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from audit_headface_teacher_surface import _roi_mask, load_scene_mask, parse_indices, parse_rois  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic bridge from raw-camera rasterized surface targets to a VGGT/reference "
            "prediction protocol. This estimates a global similarity transform only; it does "
            "not create a strict-passing teacher or candidate by itself."
        )
    )
    parser.add_argument("--teacher-npz", required=True, type=Path)
    parser.add_argument("--predictions-npz", required=True, type=Path)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-views", default="all")
    parser.add_argument("--fit-roi", default="head_face")
    parser.add_argument("--eval-rois", default="face_core,head_face,hairline,head")
    parser.add_argument("--max-fit-points", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--depth-tolerance", type=float, default=0.06)
    parser.add_argument("--export-transformed-npz", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value).replace("\\", "/")
    if isinstance(value, str):
        return value.replace("\\", "/")
    return value


def load_depth(payload: np.lib.npyio.NpzFile) -> np.ndarray:
    if "depth" in payload.files:
        depth = np.asarray(payload["depth"], dtype=np.float32)
    elif "depths" in payload.files:
        depth = np.asarray(payload["depths"], dtype=np.float32)
    else:
        raise KeyError("NPZ has neither depth nor depths")
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    return depth.astype(np.float32)


def load_mask(payload: np.lib.npyio.NpzFile, world_points: np.ndarray, depth: np.ndarray) -> np.ndarray:
    for key in ("teacher_mask", "mask", "roi_mask"):
        if key in payload.files:
            mask = np.asarray(payload[key], dtype=bool)
            if mask.ndim == 4 and mask.shape[-1] == 1:
                mask = mask[..., 0]
            return mask
    return np.isfinite(world_points).all(axis=-1) & np.isfinite(depth) & (depth > 0.05)


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> dict[str, np.ndarray | float]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.ndim != 2 or dst.ndim != 2 or src.shape != dst.shape or src.shape[1] != 3:
        raise ValueError(f"Expected matched [N,3] points, got {src.shape} and {dst.shape}")
    if src.shape[0] < 8:
        raise ValueError("Need at least 8 points to estimate similarity")
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean[None, :]
    dst_c = dst - dst_mean[None, :]
    covariance = (dst_c.T @ src_c) / float(src.shape[0])
    u, singular_values, vt = np.linalg.svd(covariance)
    d = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        d[-1] = -1.0
    rotation = u @ np.diag(d) @ vt
    var_src = np.mean(np.sum(src_c * src_c, axis=1))
    scale = float(np.sum(singular_values * d) / max(var_src, 1e-12))
    translation = dst_mean - scale * (rotation @ src_mean)
    return {
        "scale": scale,
        "rotation": rotation.astype(np.float32),
        "translation": translation.astype(np.float32),
        "singular_values": singular_values.astype(np.float32),
        "det_rotation": float(np.linalg.det(rotation)),
    }


def apply_similarity(points: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    rotation = np.asarray(transform["rotation"], dtype=np.float32)
    translation = np.asarray(transform["translation"], dtype=np.float32)
    scale = float(transform["scale"])
    flat = points.reshape(-1, 3).astype(np.float32)
    out = scale * (flat @ rotation.T) + translation[None, :]
    return out.reshape(points.shape).astype(np.float32)


def world_to_depth(points: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    extrinsic = np.asarray(extrinsic, dtype=np.float32)
    if extrinsic.shape == (3, 4):
        rotation = extrinsic[:3, :3]
        translation = extrinsic[:3, 3]
    elif extrinsic.shape == (4, 4):
        rotation = extrinsic[:3, :3]
        translation = extrinsic[:3, 3]
    else:
        raise ValueError(f"Unexpected extrinsic shape {extrinsic.shape}")
    flat = points.reshape(-1, 3)
    cam = flat @ rotation.T + translation[None, :]
    return cam[:, 2].reshape(points.shape[:2]).astype(np.float32)


def percentiles(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"p50": None, "p90": None, "p95": None, "p99": None}
    p50, p90, p95, p99 = np.percentile(values, [50, 90, 95, 99])
    return {"p50": float(p50), "p90": float(p90), "p95": float(p95), "p99": float(p99)}


def main() -> int:
    args = parse_args()
    teacher_path = args.teacher_npz.resolve()
    predictions_path = args.predictions_npz.resolve()
    scene_dir = args.scene_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(teacher_path, allow_pickle=False) as teacher_payload:
        teacher_world = np.asarray(teacher_payload["world_points"], dtype=np.float32)
        teacher_depth = load_depth(teacher_payload)
        teacher_mask = load_mask(teacher_payload, teacher_world, teacher_depth)
        teacher_intrinsic = np.asarray(teacher_payload["intrinsic"], dtype=np.float32) if "intrinsic" in teacher_payload.files else None
        teacher_camera_ids = np.asarray(teacher_payload["camera_ids"]) if "camera_ids" in teacher_payload.files else None
    with np.load(predictions_path, allow_pickle=False) as pred_payload:
        pred_world = np.asarray(pred_payload["world_points"], dtype=np.float32)
        pred_depth = load_depth(pred_payload)
        pred_intrinsic = np.asarray(pred_payload["intrinsic"], dtype=np.float32)
        pred_extrinsic = np.asarray(pred_payload["extrinsic"], dtype=np.float32)

    view_count = min(teacher_world.shape[0], pred_world.shape[0], pred_depth.shape[0])
    views = parse_indices(str(args.target_views), view_count)
    eval_rois = parse_rois(str(args.eval_rois))
    fit_roi = parse_rois(str(args.fit_roi))[0]
    height = int(pred_depth.shape[1])
    rng = np.random.default_rng(int(args.seed))

    src_points = []
    dst_points = []
    for view_idx in views:
        scene_mask = load_scene_mask(scene_dir, view_idx, target_size=height)
        roi = _roi_mask(scene_mask, fit_roi)
        valid = (
            roi
            & teacher_mask[view_idx]
            & np.isfinite(teacher_world[view_idx]).all(axis=-1)
            & np.isfinite(pred_world[view_idx]).all(axis=-1)
            & np.isfinite(pred_depth[view_idx])
            & (teacher_depth[view_idx] > float(args.min_depth))
            & (pred_depth[view_idx] > float(args.min_depth))
        )
        if valid.any():
            src_points.append(teacher_world[view_idx][valid])
            dst_points.append(pred_world[view_idx][valid])
    if not src_points:
        raise RuntimeError("No overlapping valid points for bridge fit")
    src = np.concatenate(src_points, axis=0)
    dst = np.concatenate(dst_points, axis=0)
    if src.shape[0] > int(args.max_fit_points):
        idx = rng.choice(src.shape[0], size=int(args.max_fit_points), replace=False)
        src = src[idx]
        dst = dst[idx]
    transform = umeyama_similarity(src, dst)
    transformed_world = apply_similarity(teacher_world, transform)
    transformed_depth = np.stack(
        [world_to_depth(transformed_world[view_idx], pred_extrinsic[view_idx]) for view_idx in range(view_count)],
        axis=0,
    ).astype(np.float32)

    entries = []
    for view_idx in views:
        scene_mask = load_scene_mask(scene_dir, view_idx, target_size=height)
        for roi_kind in eval_rois:
            roi = _roi_mask(scene_mask, roi_kind)
            valid = (
                roi
                & teacher_mask[view_idx]
                & np.isfinite(transformed_depth[view_idx])
                & np.isfinite(pred_depth[view_idx])
                & (transformed_depth[view_idx] > float(args.min_depth))
                & (pred_depth[view_idx] > float(args.min_depth))
            )
            residual = np.abs(transformed_depth[view_idx] - pred_depth[view_idx])
            compatible = valid & (residual <= float(args.depth_tolerance))
            roi_pixels = int(roi.sum())
            valid_pixels = int(valid.sum())
            compat_pixels = int(compatible.sum())
            entries.append(
                {
                    "view_index": int(view_idx),
                    "roi_kind": roi_kind,
                    "roi_pixels": roi_pixels,
                    "valid_pixels": valid_pixels,
                    "compatible_pixels": compat_pixels,
                    "valid_coverage": float(valid_pixels / max(roi_pixels, 1)),
                    "compatible_coverage": float(compat_pixels / max(roi_pixels, 1)),
                    "depth_residual_all": percentiles(residual[valid]),
                    "depth_residual_compatible": percentiles(residual[compatible]),
                }
            )

    roi_summary = {}
    for roi_kind in eval_rois:
        local = [row for row in entries if row["roi_kind"] == roi_kind]
        roi_summary[roi_kind] = {
            "valid_coverage_mean": float(np.mean([row["valid_coverage"] for row in local])) if local else None,
            "compatible_coverage_mean": float(np.mean([row["compatible_coverage"] for row in local])) if local else None,
            "residual_p50_mean": float(np.mean([row["depth_residual_all"]["p50"] for row in local if row["depth_residual_all"]["p50"] is not None])) if local else None,
            "residual_p90_mean": float(np.mean([row["depth_residual_all"]["p90"] for row in local if row["depth_residual_all"]["p90"] is not None])) if local else None,
        }

    transformed_npz = None
    if args.export_transformed_npz:
        transformed_npz = output_dir / "raw_surface_transformed_to_vggt_protocol.npz"
        np.savez_compressed(
            transformed_npz,
            world_points=transformed_world.astype(np.float32),
            depths=transformed_depth.astype(np.float32),
            depth=transformed_depth[..., None].astype(np.float32),
            teacher_mask=teacher_mask[:view_count].astype(bool),
            intrinsic=pred_intrinsic[:view_count].astype(np.float32),
            extrinsic=pred_extrinsic[:view_count].astype(np.float32),
            source_teacher_intrinsic=np.zeros((0,), dtype=np.float32) if teacher_intrinsic is None else teacher_intrinsic,
            source_camera_ids=np.asarray([]) if teacher_camera_ids is None else teacher_camera_ids,
        )

    summary = {
        "task": "raw_surface_to_vggt_protocol_bridge_diagnostic",
        "truthful_status": "bridge_diagnostic_complete_not_teacher_or_candidate",
        "teacher_npz": teacher_path,
        "predictions_npz": predictions_path,
        "scene_dir": scene_dir,
        "output_dir": output_dir,
        "fit_roi": fit_roi,
        "target_views": views,
        "fit_points": int(src.shape[0]),
        "transform": transform,
        "roi_summary": roi_summary,
        "entries": entries,
        "outputs": {
            "summary_json": output_dir / "bridge_summary.json",
            "report_md": output_dir / "report.md",
            "transformed_npz": transformed_npz,
        },
        "uses_vggt_depth_point_normal_as_teacher": False,
        "creates_candidate_predictions": False,
        "allows_cloud": False,
        "interpretation": (
            "This only tests whether a global similarity bridge can align the raw-camera "
            "surface target to the chosen VGGT/reference prediction protocol. It does not "
            "validate visual quality or full-body/hands, and it must not unblock cloud."
        ),
    }
    (output_dir / "bridge_summary.json").write_text(json.dumps(json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    report = [
        "# Raw Surface to VGGT Protocol Bridge Diagnostic",
        "",
        f"Status: `{summary['truthful_status']}`",
        "",
        f"- fit ROI: `{fit_roi}`",
        f"- fit points: `{int(src.shape[0])}`",
        f"- scale: `{float(transform['scale'])}`",
        f"- det(rotation): `{float(transform['det_rotation'])}`",
        f"- creates candidate predictions: `False`",
        f"- allows cloud: `False`",
        "",
        "| ROI | Valid Coverage | Compatible Coverage | Residual p50 | Residual p90 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for roi_kind, row in roi_summary.items():
        report.append(
            f"| {roi_kind} | {row['valid_coverage_mean']} | {row['compatible_coverage_mean']} | "
            f"{row['residual_p50_mean']} | {row['residual_p90_mean']} |"
        )
    report.extend(
        [
            "",
            "Interpretation:",
            "",
            str(summary["interpretation"]),
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(json_ready({"truthful_status": summary["truthful_status"], "transform": transform, "roi_summary": roi_summary, "outputs": summary["outputs"]}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
