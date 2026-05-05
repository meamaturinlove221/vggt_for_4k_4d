#!/usr/bin/env python3
"""Bridge dense teacher targets between two VGGT prediction world frames.

This is a diagnostic/export helper. It uses camera-center and optional
camera-axis correspondences for shared camera IDs to estimate a similarity
transform from a source VGGT world to a target VGGT world, then writes the
target scene's view order from the source teacher NPZ.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-teacher-npz", required=True)
    parser.add_argument("--source-predictions-npz", required=True)
    parser.add_argument("--source-scene-dir", required=True)
    parser.add_argument("--target-predictions-npz", required=True)
    parser.add_argument("--target-scene-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--axis-scale", type=float, default=0.01)
    parser.add_argument("--no-axes", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_manifest(scene_dir: Path) -> dict[str, Any]:
    path = scene_dir / "scene_manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def camera_ids(manifest: dict[str, Any]) -> list[str]:
    exported = manifest.get("exported_views") or []
    ids = [str(view["camera_id"]).zfill(2) for view in exported]
    if ids:
        return ids
    summary_ids = manifest.get("camera_summary", {}).get("camera_ids") or []
    return [str(item).zfill(2) for item in summary_ids]


def camera_center_and_rotation(extrinsic_3x4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    extrinsic_3x4 = np.asarray(extrinsic_3x4, dtype=np.float64)
    rotation_w2c = extrinsic_3x4[:3, :3]
    translation_w2c = extrinsic_3x4[:3, 3]
    rotation_c2w = rotation_w2c.T
    center_world = -rotation_c2w @ translation_w2c
    return center_world.astype(np.float64), rotation_c2w.astype(np.float64)


def median_baseline(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 1.0
    dists = []
    for idx in range(points.shape[0]):
        diff = points[idx + 1 :] - points[idx]
        if diff.size:
            dists.extend(np.linalg.norm(diff, axis=1).tolist())
    arr = np.asarray([item for item in dists if item > 1e-8], dtype=np.float64)
    return float(np.median(arr)) if arr.size else 1.0


def estimate_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"Bad correspondence shapes: {source.shape} vs {target.shape}")
    if source.shape[0] < 3:
        raise ValueError("At least three correspondences are required.")
    mu_source = source.mean(axis=0)
    mu_target = target.mean(axis=0)
    src_centered = source - mu_source
    tgt_centered = target - mu_target
    covariance = (tgt_centered.T @ src_centered) / source.shape[0]
    u_mat, singular_values, vt_mat = np.linalg.svd(covariance)
    rotation = u_mat @ vt_mat
    if np.linalg.det(rotation) < 0:
        u_mat[:, -1] *= -1.0
        rotation = u_mat @ vt_mat
    variance = float((src_centered**2).sum() / source.shape[0])
    scale = float(singular_values.sum() / max(variance, 1e-12))
    translation = mu_target - scale * (rotation @ mu_source)
    return scale, rotation.astype(np.float64), translation.astype(np.float64)


def transform_points(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    original_shape = points.shape
    flat = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(flat).all(axis=1)
    out = flat.copy()
    if finite.any():
        transformed = scale * (flat[finite].astype(np.float64) @ rotation.T) + translation[None, :]
        out[finite] = transformed.astype(np.float32)
    return out.reshape(original_shape)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output dir exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = load_manifest(Path(args.source_scene_dir))
    target_manifest = load_manifest(Path(args.target_scene_dir))
    source_ids = camera_ids(source_manifest)
    target_ids = camera_ids(target_manifest)
    source_index = {cam_id: idx for idx, cam_id in enumerate(source_ids)}

    with np.load(args.source_predictions_npz, allow_pickle=False) as src_pred:
        source_extrinsic = np.asarray(src_pred["extrinsic"], dtype=np.float64)
    with np.load(args.target_predictions_npz, allow_pickle=False) as tgt_pred:
        target_extrinsic = np.asarray(tgt_pred["extrinsic"], dtype=np.float64)
        target_intrinsic = np.asarray(tgt_pred["intrinsic"], dtype=np.float32)

    source_corr: list[np.ndarray] = []
    target_corr: list[np.ndarray] = []
    per_camera = []
    shared_ids = [cam_id for cam_id in target_ids if cam_id in source_index]
    for target_view_idx, cam_id in enumerate(target_ids):
        if cam_id not in source_index:
            continue
        source_view_idx = source_index[cam_id]
        src_center, src_rot = camera_center_and_rotation(source_extrinsic[source_view_idx])
        tgt_center, tgt_rot = camera_center_and_rotation(target_extrinsic[target_view_idx])
        source_corr.append(src_center)
        target_corr.append(tgt_center)
        per_camera.append(
            {
                "camera_id": cam_id,
                "source_view_index": int(source_view_idx),
                "target_view_index": int(target_view_idx),
                "source_center": src_center.tolist(),
                "target_center": tgt_center.tolist(),
            }
        )
        if not args.no_axes:
            # Use each world's own camera-baseline units so the axes constrain
            # orientation without imposing a false scale.
            pass

    if not source_corr:
        raise SystemExit("No shared camera IDs found between source and target scenes.")

    source_centers = np.stack(source_corr, axis=0)
    target_centers = np.stack(target_corr, axis=0)
    source_axis_len = float(args.axis_scale) * median_baseline(source_centers)
    target_axis_len = float(args.axis_scale) * median_baseline(target_centers)

    if not args.no_axes:
        for item in per_camera:
            src_idx = item["source_view_index"]
            tgt_idx = item["target_view_index"]
            src_center, src_rot = camera_center_and_rotation(source_extrinsic[src_idx])
            tgt_center, tgt_rot = camera_center_and_rotation(target_extrinsic[tgt_idx])
            for axis_idx in range(3):
                source_corr.append(src_center + source_axis_len * src_rot[:, axis_idx])
                target_corr.append(tgt_center + target_axis_len * tgt_rot[:, axis_idx])

    source_arr = np.stack(source_corr, axis=0)
    target_arr = np.stack(target_corr, axis=0)
    scale, rotation, translation = estimate_similarity(source_arr, target_arr)
    mapped = scale * (source_arr @ rotation.T) + translation[None, :]
    residual = np.linalg.norm(mapped - target_arr, axis=1)

    selected_source_indices = [source_index[cam_id] for cam_id in target_ids]
    with np.load(args.source_teacher_npz, allow_pickle=False) as teacher:
        payload: dict[str, np.ndarray] = {}
        for key in teacher.files:
            arr = teacher[key]
            if arr.ndim >= 3 and arr.shape[0] == len(source_ids):
                subset = arr[selected_source_indices]
                if key in {"world_points", "real_world_points", "real_target_cam_points"} and subset.shape[-1] == 3:
                    payload[key] = transform_points(subset, scale, rotation, translation)
                elif key in {"normals", "normal", "teacher_normal"} and subset.shape[-1] == 3:
                    normals = subset.reshape(-1, 3).astype(np.float64)
                    finite = np.isfinite(normals).all(axis=1)
                    out = normals.copy()
                    if finite.any():
                        rotated = normals[finite] @ rotation.T
                        rotated /= np.clip(np.linalg.norm(rotated, axis=1, keepdims=True), 1e-8, None)
                        out[finite] = rotated
                    payload[key] = out.reshape(subset.shape).astype(np.float32)
                elif key in {"depth", "depths", "intrinsic", "extrinsic", "camera_ids"}:
                    # Recomputed/overwritten below in the target prediction protocol.
                    continue
                else:
                    payload[key] = subset
            else:
                payload[key] = arr
        if "world_points" in payload:
            world_points = np.asarray(payload["world_points"], dtype=np.float32)
            target_extrinsic_f = target_extrinsic.astype(np.float32)
            depths = []
            for view_idx in range(world_points.shape[0]):
                rotation_w2c = target_extrinsic_f[view_idx, :3, :3]
                translation_w2c = target_extrinsic_f[view_idx, :3, 3]
                cam = world_points[view_idx].reshape(-1, 3) @ rotation_w2c.T + translation_w2c[None, :]
                depth = cam[:, 2].reshape(world_points.shape[1:3]).astype(np.float32)
                mask = np.isfinite(world_points[view_idx]).all(axis=-1) & np.isfinite(depth) & (depth > 0.0)
                depth = np.where(mask, depth, 0.0).astype(np.float32)
                depths.append(depth)
            depth_stack = np.stack(depths, axis=0).astype(np.float32)
            payload["depths"] = depth_stack
            payload["depth"] = depth_stack[..., None]
        payload["intrinsic"] = target_intrinsic.astype(np.float32)
        payload["extrinsic"] = target_extrinsic.astype(np.float32)
        payload["camera_ids"] = np.asarray(target_ids)
        payload["bridge_transform_source_to_target"] = np.asarray(
            [
                [scale * rotation[0, 0], scale * rotation[0, 1], scale * rotation[0, 2], translation[0]],
                [scale * rotation[1, 0], scale * rotation[1, 1], scale * rotation[1, 2], translation[1]],
                [scale * rotation[2, 0], scale * rotation[2, 1], scale * rotation[2, 2], translation[2]],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    np.savez_compressed(output_dir / "teacher_targets.npz", **payload)

    summary = {
        "task": "bridge_teacher_targets_between_vggt_worlds",
        "source_teacher_npz": str(Path(args.source_teacher_npz).resolve()),
        "source_predictions_npz": str(Path(args.source_predictions_npz).resolve()),
        "target_predictions_npz": str(Path(args.target_predictions_npz).resolve()),
        "source_scene_dir": str(Path(args.source_scene_dir).resolve()),
        "target_scene_dir": str(Path(args.target_scene_dir).resolve()),
        "output_teacher_npz": str((output_dir / "teacher_targets.npz").resolve()),
        "target_camera_ids": target_ids,
        "selected_source_indices": selected_source_indices,
        "shared_camera_ids": shared_ids,
        "used_axes": not args.no_axes,
        "axis_scale": float(args.axis_scale),
        "source_axis_len": source_axis_len,
        "target_axis_len": target_axis_len,
        "similarity": {
            "scale": scale,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "residual_percentiles": np.percentile(residual, [0, 25, 50, 75, 90, 95, 100]).tolist(),
            "correspondence_count": int(source_arr.shape[0]),
        },
        "per_camera": per_camera,
        "truthful_status": "strict_teacher_gate_required_before_training",
        "note": "This only bridges VGGT coordinate systems; it is not a teacher pass.",
    }
    (output_dir / "bridge_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
