from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.export_detail_normal_refiner_dataset import (  # noqa: E402
    _export_roi_pack,
    _load_scene_rgb_and_mask,
)
from vggt.utils.normal_refiner import (  # noqa: E402
    extract_coarse_prior_normal,
    face_box_from_mask,
    head_box_from_mask,
    shoulder_box_from_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export detail normal refiner ROI samples from external normal maps.")
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--prior-maps-npz", required=True)
    parser.add_argument("--external-normal-npz", required=True, help="NPZ with normal [S,H,W,3] and optional mask")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--roi-kind", choices=("head", "face", "shoulder", "both", "all"), default="all")
    parser.add_argument("--target-size", type=int, default=256)
    parser.add_argument(
        "--transform",
        choices=("identity", "neg", "negxy", "negxyz", "flip-yz", "sapiens_cam"),
        default="identity",
    )
    return parser.parse_args()


def _transform_normals(normals: np.ndarray, transform: str) -> np.ndarray:
    out = np.asarray(normals, dtype=np.float32).copy()
    if transform == "identity":
        pass
    elif transform == "neg":
        out = -out
    elif transform == "negxy":
        out[..., 0:2] *= -1.0
    elif transform == "negxyz":
        out *= -1.0
    elif transform == "flip-yz":
        out[..., 1:3] *= -1.0
    elif transform == "sapiens_cam":
        out[..., 1] *= -1.0
    else:
        raise ValueError(transform)
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    out = out / np.clip(norm, 1e-6, None)
    out[norm[..., 0] < 1e-6] = 0.0
    return out.astype(np.float32)


def main() -> int:
    args = parse_args()
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prior_payload = np.load(args.prior_maps_npz, allow_pickle=False)
    coarse_normal, coarse_valid_mask = extract_coarse_prior_normal(
        prior_payload["prior_maps"], prior_payload["prior_channels"]
    )
    human_mask = np.asarray(prior_payload["prior_mask"], dtype=bool)
    images, scene_mask, view_names = _load_scene_rgb_and_mask(scene_dir, target_size=int(coarse_normal.shape[1]))
    human_mask = human_mask & scene_mask

    external = np.load(args.external_normal_npz, allow_pickle=False)
    key = "normal" if "normal" in external.files else "normal_rgb"
    teacher = np.asarray(external[key], dtype=np.float32)
    if key == "normal_rgb" or teacher.max() > 2.0:
        teacher = teacher / 127.5 - 1.0
    teacher = _transform_normals(teacher, args.transform)
    teacher_mask = np.asarray(external["mask"], dtype=bool) if "mask" in external.files else human_mask.copy()
    teacher_mask = teacher_mask & human_mask & np.isfinite(teacher).all(axis=-1) & (np.linalg.norm(teacher, axis=-1) > 0.5)
    teacher[~teacher_mask] = 0.0

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
            teacher_normal=teacher,
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
            teacher_normal=teacher,
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
            teacher_normal=teacher,
            teacher_mask=teacher_mask,
            view_names=view_names,
            output_dir=roi_dir,
            target_size=args.target_size,
        )
        exported.append("shoulder_roi/shoulder_samples.npz")

    summary = {
        "scene_dir": str(scene_dir),
        "prior_maps_npz": str(Path(args.prior_maps_npz).expanduser().resolve()),
        "external_normal_npz": str(Path(args.external_normal_npz).expanduser().resolve()),
        "transform": args.transform,
        "teacher_mask_pixels": [int(v) for v in teacher_mask.reshape(teacher_mask.shape[0], -1).sum(axis=1)],
        "target_size": int(args.target_size),
        "exported": exported,
    }
    (output_dir / "external_normal_refiner_dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
