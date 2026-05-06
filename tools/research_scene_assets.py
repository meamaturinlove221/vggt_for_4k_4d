from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


CAMERA_SIDECAR_NAME = "camera_params_sidecar.npz"


def portable_manifest_path(raw_path: str) -> str:
    return Path(str(raw_path).replace("\\", "/")).name


def localize_scene_manifest_paths(scene_manifest: dict[str, Any], scene_dir: Path) -> dict[str, Any]:
    """Point manifest image/mask paths at files inside a copied scene directory.

    Preprocessed scene manifests were produced on Windows and may contain absolute
    local paths. Research-preflight jobs run against an uploaded scene copy, so
    those absolute paths are not valid remotely.
    """

    scene_dir = Path(scene_dir)
    for view in scene_manifest.get("exported_views", []):
        for key, subdir in (("image_path", "images"), ("mask_path", "masks")):
            raw = str(view.get(key, ""))
            if not raw:
                continue
            candidate = scene_dir / subdir / portable_manifest_path(raw)
            if candidate.is_file():
                view[key] = str(candidate)
    return scene_manifest


def load_camera_params_sidecar(scene_dir: Path) -> dict[str, dict[str, np.ndarray]] | None:
    path = Path(scene_dir) / CAMERA_SIDECAR_NAME
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        camera_ids = [str(item) for item in data["camera_ids"].tolist()]
        intrinsics = np.asarray(data["intrinsics"], dtype=np.float32)
        cam_to_world = np.asarray(data["cam_to_world"], dtype=np.float32)
        world_to_cam = np.asarray(data["world_to_cam"], dtype=np.float32)
    if not (len(camera_ids) == intrinsics.shape[0] == cam_to_world.shape[0] == world_to_cam.shape[0]):
        raise ValueError(f"Camera sidecar view-count mismatch: {path}")
    override: dict[str, dict[str, np.ndarray]] = {"_source_tag": "camera_params_sidecar"}
    for idx, camera_id in enumerate(camera_ids):
        override[camera_id] = {
            "intrinsic": intrinsics[idx].astype(np.float32),
            "cam_to_world": cam_to_world[idx].astype(np.float32),
            "world_to_cam": world_to_cam[idx].astype(np.float32),
        }
    return override
