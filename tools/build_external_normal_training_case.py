from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy a 4K4D training case and replace teacher_normals from an external normal NPZ.")
    parser.add_argument("--source-case-dir", required=True)
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--prior-maps-npz", required=True)
    parser.add_argument("--external-normal-npz", required=True)
    parser.add_argument("--output-case-dir", required=True)
    parser.add_argument("--transform", choices=("identity", "neg", "negxy", "negxyz", "flip-yz", "sapiens_cam", "negz"), default="flip-yz")
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
    else:
        raise ValueError(name)
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    out = out / np.clip(norm, 1e-6, None)
    out[norm[..., 0] < 1e-6] = 0.0
    return out.astype(np.float32)


def _roi_masks_from_existing(targets: dict[str, np.ndarray], teacher_mask: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    out["head_roi_mask"] = (np.asarray(targets.get("head_roi_mask", teacher_mask), dtype=bool) | teacher_mask).astype(bool)
    out["face_roi_mask"] = (np.asarray(targets.get("face_roi_mask", teacher_mask), dtype=bool) | teacher_mask).astype(bool)
    if "hairline_mask" in targets:
        out["hairline_mask"] = np.asarray(targets["hairline_mask"], dtype=bool)
    else:
        out["hairline_mask"] = teacher_mask.copy()
    if "ear_band_mask" in targets:
        out["ear_band_mask"] = np.asarray(targets["ear_band_mask"], dtype=bool)
    else:
        out["ear_band_mask"] = teacher_mask.copy()
    return out


def main() -> int:
    args = parse_args()
    source_case = Path(args.source_case_dir)
    output_case = Path(args.output_case_dir)
    _copy_case(source_case, output_case, overwrite=bool(args.overwrite))

    with np.load(output_case / "inputs.npz", allow_pickle=False) as payload:
        inputs = {key: np.array(payload[key]) for key in payload.files}
    with np.load(output_case / "targets.npz", allow_pickle=False) as payload:
        targets = {key: np.array(payload[key]) for key in payload.files}
    prior = np.load(args.prior_maps_npz, allow_pickle=False)
    external = np.load(args.external_normal_npz, allow_pickle=False)
    teacher = np.asarray(external["normal"] if "normal" in external.files else external["normal_rgb"], dtype=np.float32)
    if teacher.max() > 2.0:
        teacher = teacher / 127.5 - 1.0
    teacher = _transform(teacher, args.transform)
    external_mask = np.asarray(external["mask"], dtype=bool) if "mask" in external.files else np.ones(teacher.shape[:3], dtype=bool)
    prior_mask = np.asarray(prior["prior_mask"], dtype=bool)
    input_mask = np.asarray(inputs.get("point_masks", prior_mask), dtype=bool)
    teacher_mask = external_mask & prior_mask & input_mask & np.isfinite(teacher).all(axis=-1)
    teacher_mask &= np.linalg.norm(teacher, axis=-1) > 0.5
    teacher[~teacher_mask] = 0.0

    targets["teacher_normals"] = teacher.astype(np.float32)
    targets["teacher_mask"] = teacher_mask.astype(bool)
    targets["prior_normals"] = np.asarray(targets.get("prior_normals", teacher), dtype=np.float32)
    targets["prior_normals"][teacher_mask] = teacher[teacher_mask]
    targets.update(_roi_masks_from_existing(targets, teacher_mask))
    np.savez_compressed(output_case / "targets.npz", **targets)

    manifest_path = output_case / "case_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    manifest["external_normal_training_patch"] = {
        "source_case_dir": str(source_case.resolve()),
        "scene_dir": str(Path(args.scene_dir).resolve()),
        "prior_maps_npz": str(Path(args.prior_maps_npz).resolve()),
        "external_normal_npz": str(Path(args.external_normal_npz).resolve()),
        "transform": args.transform,
        "teacher_mask_pixels": [int(v) for v in teacher_mask.reshape(teacher_mask.shape[0], -1).sum(axis=1)],
        "fields": ["teacher_normals", "teacher_mask", "prior_normals", "head_roi_mask", "face_roi_mask"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["external_normal_training_patch"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
