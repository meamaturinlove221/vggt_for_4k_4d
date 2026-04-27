from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


POSITION_CHANNELS = ("smplx_posed_cam_x", "smplx_posed_cam_y", "smplx_posed_cam_z")
NORMAL_CHANNELS = ("smplx_cam_nx", "smplx_cam_ny", "smplx_cam_nz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a training case whose head/face target geometry is replaced by the "
            "multi-view surface teacher encoded in a patched scene prior_maps.npz."
        )
    )
    parser.add_argument("--source-case-dir", required=True)
    parser.add_argument("--base-scene-dir", required=True)
    parser.add_argument("--surface-scene-dir", required=True)
    parser.add_argument("--output-case-dir", required=True)
    parser.add_argument("--conf-boost", type=float, default=96.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _lookup_indices(channels: np.ndarray, names: tuple[str, ...]) -> list[int]:
    channel_names = [str(name) for name in np.asarray(channels).tolist()]
    lookup = {name: idx for idx, name in enumerate(channel_names)}
    return [lookup[name] for name in names]


def _load_prior(path: Path) -> dict[str, np.ndarray]:
    with np.load(path / "prior_maps.npz", allow_pickle=False) as payload:
        return {key: np.array(payload[key]) for key in payload.files}


def _copy_case(source: Path, output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    shutil.copytree(source, output)


def _camera_to_world(cam_points: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    out = np.zeros_like(cam_points, dtype=np.float32)
    for view_idx in range(cam_points.shape[0]):
        rotation = extrinsics[view_idx, :, :3].astype(np.float32)
        translation = extrinsics[view_idx, :, 3].astype(np.float32)
        flat = cam_points[view_idx].reshape(-1, 3)
        world = (flat - translation[None]) @ rotation
        out[view_idx] = world.reshape(cam_points.shape[1:])
    return out.astype(np.float32)


def _make_hairline_and_ear_masks(head: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hair = np.zeros_like(head, dtype=bool)
    ear = np.zeros_like(head, dtype=bool)
    for view_idx in range(head.shape[0]):
        ys, xs = np.nonzero(head[view_idx])
        if len(xs) == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        h = max(1, y1 - y0)
        w = max(1, x1 - x0)
        hair_limit = y0 + max(4, int(round(h * 0.22)))
        hair[view_idx, y0:hair_limit, x0:x1] = head[view_idx, y0:hair_limit, x0:x1]
        left_limit = x0 + max(4, int(round(w * 0.18)))
        right_limit = x1 - max(4, int(round(w * 0.18)))
        ear[view_idx, y0:y1, x0:left_limit] = head[view_idx, y0:y1, x0:left_limit]
        ear[view_idx, y0:y1, right_limit:x1] = head[view_idx, y0:y1, right_limit:x1]
    return hair, ear


def main() -> None:
    args = parse_args()
    source_case = Path(args.source_case_dir)
    output_case = Path(args.output_case_dir)
    base_prior = _load_prior(Path(args.base_scene_dir))
    surface_prior = _load_prior(Path(args.surface_scene_dir))

    _copy_case(source_case, output_case, overwrite=bool(args.overwrite))

    with np.load(output_case / "targets.npz", allow_pickle=False) as payload:
        targets = {key: np.array(payload[key]) for key in payload.files}

    channels = np.asarray(surface_prior["prior_channels"])
    position_idx = _lookup_indices(channels, POSITION_CHANNELS)
    normal_idx = _lookup_indices(channels, NORMAL_CHANNELS)
    base_maps = np.asarray(base_prior["prior_maps"], dtype=np.float32)
    surface_maps = np.asarray(surface_prior["prior_maps"], dtype=np.float32)
    position_delta = np.linalg.norm(surface_maps[:, position_idx] - base_maps[:, position_idx], axis=1)
    normal_delta = np.linalg.norm(surface_maps[:, normal_idx] - base_maps[:, normal_idx], axis=1)
    patch_mask = (position_delta > 1e-4) | (normal_delta > 1e-4)

    surface_cam_points = surface_maps[:, position_idx].transpose(0, 2, 3, 1).astype(np.float32)
    surface_normals = surface_maps[:, normal_idx].transpose(0, 2, 3, 1).astype(np.float32)
    normal_norm = np.linalg.norm(surface_normals, axis=-1, keepdims=True)
    surface_normals = surface_normals / np.clip(normal_norm, 1e-6, None)
    surface_normals[~patch_mask] = 0.0

    if surface_cam_points.shape != targets["cam_points"].shape:
        raise ValueError(f"Shape mismatch: {surface_cam_points.shape} vs {targets['cam_points'].shape}")

    targets["cam_points"] = np.asarray(targets["cam_points"], dtype=np.float32)
    targets["cam_points"][patch_mask] = surface_cam_points[patch_mask]
    targets["depths"] = np.asarray(targets["depths"], dtype=np.float32)
    targets["depths"][patch_mask] = surface_cam_points[..., 2][patch_mask]
    targets["world_points"] = _camera_to_world(targets["cam_points"], np.asarray(targets["extrinsics"], dtype=np.float32))
    targets["teacher_normals"] = surface_normals.astype(np.float32)
    targets["teacher_mask"] = patch_mask.astype(bool)
    targets["prior_normals"] = np.asarray(targets.get("prior_normals", surface_normals), dtype=np.float32)
    targets["prior_normals"][patch_mask] = surface_normals[patch_mask]

    head = patch_mask.astype(bool)
    face = patch_mask.astype(bool)
    if "head_roi_mask" in targets:
        head = np.asarray(targets["head_roi_mask"], dtype=bool) | patch_mask
    if "face_roi_mask" in targets:
        face = np.asarray(targets["face_roi_mask"], dtype=bool) | patch_mask
    targets["head_roi_mask"] = head.astype(bool)
    targets["face_roi_mask"] = face.astype(bool)
    hair, ear = _make_hairline_and_ear_masks(head)
    targets["hairline_mask"] = (np.asarray(targets.get("hairline_mask", hair), dtype=bool) | hair).astype(bool)
    targets["ear_band_mask"] = (np.asarray(targets.get("ear_band_mask", ear), dtype=bool) | ear).astype(bool)
    targets["depth_conf"] = np.asarray(targets["depth_conf"], dtype=np.float32)
    targets["world_points_conf"] = np.asarray(targets["world_points_conf"], dtype=np.float32)
    targets["depth_conf"][patch_mask] = np.maximum(targets["depth_conf"][patch_mask], float(args.conf_boost))
    targets["world_points_conf"][patch_mask] = np.maximum(
        targets["world_points_conf"][patch_mask], float(args.conf_boost)
    )

    np.savez_compressed(output_case / "targets.npz", **targets)

    manifest_path = output_case / "case_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    manifest["surface_teacher_training_patch"] = {
        "base_scene_dir": str(Path(args.base_scene_dir).resolve()),
        "surface_scene_dir": str(Path(args.surface_scene_dir).resolve()),
        "patched_pixels_total": int(patch_mask.sum()),
        "patched_pixels_per_view": [int(v) for v in patch_mask.reshape(patch_mask.shape[0], -1).sum(axis=1)],
        "conf_boost": float(args.conf_boost),
        "target_fields": [
            "depths",
            "cam_points",
            "world_points",
            "teacher_normals",
            "teacher_mask",
            "prior_normals",
            "head_roi_mask",
            "face_roi_mask",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["surface_teacher_training_patch"], indent=2))


if __name__ == "__main__":
    main()
