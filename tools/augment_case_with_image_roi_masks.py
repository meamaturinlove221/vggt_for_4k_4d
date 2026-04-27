from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add image-space head/face/hairline/ear ROI masks to a 4K4D training case targets.npz.")
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--head-frac", type=float, default=0.42)
    parser.add_argument("--face-x0", type=float, default=0.18)
    parser.add_argument("--face-x1", type=float, default=0.82)
    parser.add_argument("--face-y0", type=float, default=0.18)
    parser.add_argument("--face-y1", type=float, default=0.88)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _make_masks(point_mask: np.ndarray, *, head_frac: float, face_box: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    point_mask = np.asarray(point_mask, dtype=bool)
    num_views, height, width = point_mask.shape
    head = np.zeros_like(point_mask, dtype=bool)
    face = np.zeros_like(point_mask, dtype=bool)
    hairline = np.zeros_like(point_mask, dtype=bool)
    ear = np.zeros_like(point_mask, dtype=bool)
    fx0_frac, fx1_frac, fy0_frac, fy1_frac = face_box

    for view_idx in range(num_views):
        mask = point_mask[view_idx]
        body_bbox = _bbox(mask)
        if body_bbox is None:
            continue
        x0, y0, x1, y1 = body_bbox
        body_h = max(1, y1 - y0)
        upper_limit = min(height, int(round(y0 + body_h * float(head_frac))))
        head_mask = np.zeros((height, width), dtype=bool)
        head_mask[:upper_limit] = mask[:upper_limit]
        head_bbox = _bbox(head_mask) or body_bbox
        hx0, hy0, hx1, hy1 = head_bbox
        head[view_idx] = head_mask

        head_w = max(1, hx1 - hx0)
        head_h = max(1, hy1 - hy0)
        fx0 = int(round(hx0 + head_w * fx0_frac))
        fx1 = int(round(hx0 + head_w * fx1_frac))
        fy0 = int(round(hy0 + head_h * fy0_frac))
        fy1 = int(round(hy0 + head_h * fy1_frac))
        fx0, fx1 = max(0, fx0), min(width, max(fx1, fx0 + 1))
        fy0, fy1 = max(0, fy0), min(height, max(fy1, fy0 + 1))
        face_region = np.zeros((height, width), dtype=bool)
        face_region[fy0:fy1, fx0:fx1] = mask[fy0:fy1, fx0:fx1]
        face[view_idx] = face_region

        hair_h = max(1, int(round(head_h * 0.22)))
        hair_region = np.zeros((height, width), dtype=bool)
        hair_region[hy0 : min(height, hy0 + hair_h), hx0:hx1] = mask[hy0 : min(height, hy0 + hair_h), hx0:hx1]
        hairline[view_idx] = hair_region

        band_w = max(1, int(round(head_w * 0.18)))
        ear_region = np.zeros((height, width), dtype=bool)
        ear_region[hy0:hy1, hx0 : min(width, hx0 + band_w)] |= mask[hy0:hy1, hx0 : min(width, hx0 + band_w)]
        ear_region[hy0:hy1, max(0, hx1 - band_w) : hx1] |= mask[hy0:hy1, max(0, hx1 - band_w) : hx1]
        ear[view_idx] = ear_region

    return head, face, hairline, ear


def main() -> int:
    args = parse_args()
    case_dir = Path(args.case_dir).expanduser().resolve()
    inputs_path = case_dir / "inputs.npz"
    targets_path = case_dir / "targets.npz"
    summary_path = case_dir / "roi_mask_augmentation_summary.json"
    if not inputs_path.is_file() or not targets_path.is_file():
        raise FileNotFoundError(f"Expected inputs.npz and targets.npz under {case_dir}")
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"ROI augmentation already exists; pass --overwrite: {summary_path}")

    with np.load(inputs_path, allow_pickle=False) as inputs:
        point_masks = np.asarray(inputs["point_masks"], dtype=bool)
    head, face, hairline, ear = _make_masks(
        point_masks,
        head_frac=float(args.head_frac),
        face_box=(float(args.face_x0), float(args.face_x1), float(args.face_y0), float(args.face_y1)),
    )
    with np.load(targets_path, allow_pickle=False) as targets:
        payload = {key: np.asarray(targets[key]) for key in targets.files}
    payload.update(
        {
            "head_roi_mask": head,
            "face_roi_mask": face,
            "hairline_mask": hairline,
            "ear_band_mask": ear,
        }
    )
    np.savez_compressed(targets_path, **payload)
    summary = {
        "case_dir": str(case_dir),
        "head_pixels": [int(v) for v in head.sum(axis=(1, 2)).tolist()],
        "face_pixels": [int(v) for v in face.sum(axis=(1, 2)).tolist()],
        "hairline_pixels": [int(v) for v in hairline.sum(axis=(1, 2)).tolist()],
        "ear_band_pixels": [int(v) for v in ear.sum(axis=(1, 2)).tolist()],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
