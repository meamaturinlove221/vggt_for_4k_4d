from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _finite_points(points: np.ndarray) -> np.ndarray:
    return np.isfinite(points).all(axis=-1)


def fuse_predictions(
    base_predictions: Path,
    teacher_targets: Path,
    output_dir: Path,
    *,
    alpha_xyz: tuple[float, float, float],
    max_distance: float,
    confidence_boost: float,
) -> dict:
    base = _load_npz(base_predictions)
    teacher = _load_npz(teacher_targets)
    if "world_points" not in base or "world_points_conf" not in base:
        raise KeyError("base predictions must contain world_points and world_points_conf")
    if "world_points" not in teacher or "teacher_mask" not in teacher:
        raise KeyError("teacher targets must contain world_points and teacher_mask")

    base_points = np.asarray(base["world_points"], dtype=np.float32)
    base_conf = np.asarray(base["world_points_conf"], dtype=np.float32)
    teacher_points = np.asarray(teacher["world_points"], dtype=np.float32)
    teacher_mask = np.asarray(teacher["teacher_mask"], dtype=bool)
    if base_points.shape != teacher_points.shape:
        raise ValueError(f"point shape mismatch: base={base_points.shape} teacher={teacher_points.shape}")
    if teacher_mask.shape != base_points.shape[:3]:
        raise ValueError(f"teacher_mask shape mismatch: {teacher_mask.shape} vs {base_points.shape[:3]}")

    delta = teacher_points - base_points
    distance = np.linalg.norm(delta, axis=-1)
    valid = teacher_mask & _finite_points(base_points) & _finite_points(teacher_points) & np.isfinite(distance)
    if max_distance > 0:
        valid &= distance <= float(max_distance)

    alpha = np.asarray(alpha_xyz, dtype=np.float32).reshape(1, 1, 1, 3)
    fused_points = base_points.copy()
    fused_points[valid] = base_points[valid] + delta[valid] * alpha.reshape(3)

    fused_conf = base_conf.copy()
    if confidence_boost > 0:
        fused_conf[valid] = np.maximum(fused_conf[valid], float(confidence_boost))

    out = dict(base)
    out["world_points"] = fused_points.astype(base["world_points"].dtype, copy=False)
    out["world_points_conf"] = fused_conf.astype(base["world_points_conf"].dtype, copy=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions.npz"
    np.savez_compressed(output_path, **out)

    summary = {
        "base_predictions": str(base_predictions),
        "teacher_targets": str(teacher_targets),
        "output_predictions": str(output_path),
        "alpha_xyz": [float(value) for value in alpha_xyz],
        "max_distance": float(max_distance),
        "confidence_boost": float(confidence_boost),
        "teacher_mask_pixels": int(teacher_mask.sum()),
        "valid_fused_pixels": int(valid.sum()),
        "distance_percentiles_on_teacher_mask": [
            float(value) for value in np.percentile(distance[teacher_mask & np.isfinite(distance)], [0, 25, 50, 75, 90, 95, 99])
        ],
    }
    (output_dir / "fusion_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-predictions", required=True, type=Path)
    parser.add_argument("--teacher-targets", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--alpha-x", type=float, default=0.25)
    parser.add_argument("--alpha-y", type=float, default=0.25)
    parser.add_argument("--alpha-z", type=float, default=0.25)
    parser.add_argument("--max-distance", type=float, default=0.08)
    parser.add_argument("--confidence-boost", type=float, default=96.0)
    args = parser.parse_args()

    summary = fuse_predictions(
        base_predictions=args.base_predictions,
        teacher_targets=args.teacher_targets,
        output_dir=args.output_dir,
        alpha_xyz=(args.alpha_x, args.alpha_y, args.alpha_z),
        max_distance=args.max_distance,
        confidence_boost=args.confidence_boost,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
