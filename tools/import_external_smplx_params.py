from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


SMPLX_KEY_ALIASES = {
    "betas": ("betas", "shape", "shape_params"),
    "fullpose": ("fullpose", "body_pose", "pose", "poses"),
    "transl": ("transl", "translation", "global_trans"),
    "scale": ("scale", "global_scale"),
    "expression": ("expression", "expr"),
    "gender": ("gender", "smplx_gender"),
}

CAMERA_KEY_ALIASES = {
    "camera_ids": ("camera_ids", "camera_id", "ids", "names"),
    "intrinsic": ("intrinsic", "intrinsics", "K"),
    "cam_to_world": ("cam_to_world", "c2w", "RT", "camera_to_world"),
    "world_to_cam": ("world_to_cam", "w2c", "extrinsic", "extrinsics"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import external SMPL-X/camera params (npz/json), normalize to canonical keys, "
            "and export an intermediate bundle aligned to scene_manifest view order."
        )
    )
    parser.add_argument("--smplx-input", required=True, help="External SMPL-X file (.npz or .json).")
    parser.add_argument("--camera-input", help="External camera file (.npz or .json). Optional.")
    parser.add_argument("--scene-dir", help="Scene directory containing scene_manifest.json. Optional.")
    parser.add_argument(
        "--scene-manifest",
        help="Path to scene_manifest.json. If omitted and --scene-dir is set, use <scene-dir>/scene_manifest.json.",
    )
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame index when external arrays are time-major.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for normalized intermediate bundle.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on weakly inferred fields (default is tolerant fallback).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output files if already present.")
    return parser.parse_args()


def _load_npz_or_json(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as payload:
            return {key: payload[key] for key in payload.files}
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported input file extension: {path.suffix}. Only .npz and .json are supported.")


def _as_float_array(value: Any, dtype=np.float32) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _pick_with_aliases(payload: dict[str, Any], canonical_name: str, aliases: tuple[str, ...]) -> Any:
    if canonical_name in payload:
        return payload[canonical_name]
    for alias in aliases:
        if alias in payload:
            return payload[alias]
    return None


def _normalize_smplx_params(payload: dict[str, Any], frame_idx: int, strict: bool) -> dict[str, np.ndarray]:
    normalized: dict[str, np.ndarray] = {}
    for canonical, aliases in SMPLX_KEY_ALIASES.items():
        raw = _pick_with_aliases(payload, canonical, aliases)
        if raw is None:
            continue
        if canonical == "gender":
            normalized[canonical] = np.asarray(str(raw))
            continue
        if canonical == "scale":
            scale_arr = np.asarray(raw, dtype=np.float32)
            if scale_arr.ndim == 0:
                normalized[canonical] = scale_arr.reshape(())
            else:
                if frame_idx < 0 or frame_idx >= scale_arr.shape[0]:
                    raise IndexError(f"frame_idx={frame_idx} out of range for scale with first dim {scale_arr.shape[0]}")
                normalized[canonical] = np.asarray(scale_arr[frame_idx], dtype=np.float32).reshape(())
            continue
        arr = np.asarray(raw)
        if canonical == "betas":
            if arr.ndim >= 2:
                if frame_idx < 0 or frame_idx >= arr.shape[0]:
                    raise IndexError(f"frame_idx={frame_idx} out of range for betas with first dim {arr.shape[0]}")
                arr = arr[frame_idx]
        elif canonical == "fullpose":
            if arr.ndim >= 3:
                if frame_idx < 0 or frame_idx >= arr.shape[0]:
                    raise IndexError(f"frame_idx={frame_idx} out of range for fullpose with first dim {arr.shape[0]}")
                arr = arr[frame_idx]
            elif arr.ndim == 2 and arr.shape[-1] != 3:
                if frame_idx < 0 or frame_idx >= arr.shape[0]:
                    raise IndexError(f"frame_idx={frame_idx} out of range for fullpose with first dim {arr.shape[0]}")
                arr = arr[frame_idx]
        elif canonical in {"transl", "expression"}:
            if arr.ndim >= 2:
                if frame_idx < 0 or frame_idx >= arr.shape[0]:
                    raise IndexError(f"frame_idx={frame_idx} out of range for {canonical} with first dim {arr.shape[0]}")
                arr = arr[frame_idx]
        normalized[canonical] = _as_float_array(arr)

    required = {"betas", "fullpose"}
    missing = sorted(required - set(normalized.keys()))
    if missing:
        message = (
            "Missing required SMPL-X keys after normalization: "
            f"{missing}. At least betas/fullpose are required by current prior flow."
        )
        if strict:
            raise ValueError(message)
        print(f"[import-external-smplx] warning: {message}")

    return normalized


def _normalize_camera_id(camera_id: Any) -> str:
    try:
        return f"{int(camera_id):02d}"
    except (ValueError, TypeError):
        return str(camera_id)


def _to_4x4(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :4] = matrix
        return out
    if matrix.shape == (3, 3):
        out = np.eye(4, dtype=np.float32)
        out[:3, :3] = matrix
        return out
    raise ValueError(f"Unsupported camera matrix shape: {matrix.shape}, expected (4,4), (3,4), or (3,3)")


def _parse_camera_mapping(mapping: dict[str, Any], strict: bool) -> dict[str, dict[str, np.ndarray]]:
    normalized: dict[str, dict[str, np.ndarray]] = {}
    for camera_id, camera_data in mapping.items():
        cid = _normalize_camera_id(camera_id)
        if not isinstance(camera_data, dict):
            if strict:
                raise ValueError(f"Camera payload for id={camera_id} is not a mapping.")
            continue
        intrinsic = _pick_with_aliases(camera_data, "intrinsic", CAMERA_KEY_ALIASES["intrinsic"])
        c2w = _pick_with_aliases(camera_data, "cam_to_world", CAMERA_KEY_ALIASES["cam_to_world"])
        w2c = _pick_with_aliases(camera_data, "world_to_cam", CAMERA_KEY_ALIASES["world_to_cam"])
        if intrinsic is None and strict:
            raise ValueError(f"Camera {cid} missing intrinsic.")

        intrinsic_np = np.asarray(intrinsic, dtype=np.float32) if intrinsic is not None else np.eye(3, dtype=np.float32)
        c2w_np = _to_4x4(np.asarray(c2w, dtype=np.float32)) if c2w is not None else None
        w2c_np = _to_4x4(np.asarray(w2c, dtype=np.float32)) if w2c is not None else None
        if c2w_np is None and w2c_np is None:
            if strict:
                raise ValueError(f"Camera {cid} missing both cam_to_world and world_to_cam.")
            c2w_np = np.eye(4, dtype=np.float32)
            w2c_np = np.eye(4, dtype=np.float32)
        elif c2w_np is None:
            c2w_np = np.linalg.inv(w2c_np).astype(np.float32)
        elif w2c_np is None:
            w2c_np = np.linalg.inv(c2w_np).astype(np.float32)

        normalized[cid] = {
            "intrinsic": intrinsic_np.astype(np.float32),
            "cam_to_world": c2w_np.astype(np.float32),
            "world_to_cam": w2c_np.astype(np.float32),
        }
    return normalized


def _parse_camera_payload(payload: dict[str, Any], strict: bool) -> dict[str, dict[str, np.ndarray]]:
    if "cameras" in payload and isinstance(payload["cameras"], dict):
        return _parse_camera_mapping(payload["cameras"], strict=strict)

    keys = payload.keys()
    if any(key in keys for key in CAMERA_KEY_ALIASES["intrinsic"]):
        camera_ids_raw = _pick_with_aliases(payload, "camera_ids", CAMERA_KEY_ALIASES["camera_ids"])
        intrinsic_raw = _pick_with_aliases(payload, "intrinsic", CAMERA_KEY_ALIASES["intrinsic"])
        c2w_raw = _pick_with_aliases(payload, "cam_to_world", CAMERA_KEY_ALIASES["cam_to_world"])
        w2c_raw = _pick_with_aliases(payload, "world_to_cam", CAMERA_KEY_ALIASES["world_to_cam"])

        if camera_ids_raw is None:
            if strict:
                raise ValueError("Camera payload missing camera_ids.")
            count = np.asarray(intrinsic_raw).shape[0]
            camera_ids_raw = [str(idx) for idx in range(count)]

        camera_ids = [_normalize_camera_id(x) for x in list(camera_ids_raw)]
        intrinsic_arr = np.asarray(intrinsic_raw, dtype=np.float32)
        c2w_arr = np.asarray(c2w_raw, dtype=np.float32) if c2w_raw is not None else None
        w2c_arr = np.asarray(w2c_raw, dtype=np.float32) if w2c_raw is not None else None

        normalized: dict[str, dict[str, np.ndarray]] = {}
        for idx, camera_id in enumerate(camera_ids):
            intrinsic = intrinsic_arr[idx]
            c2w = _to_4x4(c2w_arr[idx]) if c2w_arr is not None else None
            w2c = _to_4x4(w2c_arr[idx]) if w2c_arr is not None else None
            if c2w is None and w2c is None:
                if strict:
                    raise ValueError(f"Camera {camera_id} missing both c2w/w2c.")
                c2w = np.eye(4, dtype=np.float32)
                w2c = np.eye(4, dtype=np.float32)
            elif c2w is None:
                c2w = np.linalg.inv(w2c).astype(np.float32)
            elif w2c is None:
                w2c = np.linalg.inv(c2w).astype(np.float32)
            normalized[camera_id] = {
                "intrinsic": intrinsic.astype(np.float32),
                "cam_to_world": c2w.astype(np.float32),
                "world_to_cam": w2c.astype(np.float32),
            }
        return normalized

    if strict:
        raise ValueError("Camera payload cannot be parsed. Expected `cameras` mapping or array fields.")
    return {}


def _load_scene_manifest(scene_dir: Path | None, scene_manifest: Path | None) -> dict[str, Any] | None:
    if scene_manifest is None and scene_dir is not None:
        scene_manifest = scene_dir / "scene_manifest.json"
    if scene_manifest is None:
        return None
    if not scene_manifest.is_file():
        raise FileNotFoundError(f"scene_manifest.json not found at {scene_manifest}")
    return json.loads(scene_manifest.read_text(encoding="utf-8"))


def _align_camera_order(
    camera_params: dict[str, dict[str, np.ndarray]],
    scene_manifest: dict[str, Any] | None,
    strict: bool,
) -> tuple[list[str], list[str]]:
    if scene_manifest is None:
        ids = sorted(camera_params.keys())
        return ids, []

    scene_camera_ids = [_normalize_camera_id(view["camera_id"]) for view in scene_manifest.get("exported_views", [])]
    if not scene_camera_ids:
        if strict:
            raise ValueError("scene_manifest exported_views is empty; cannot align camera order.")
        ids = sorted(camera_params.keys())
        return ids, []

    missing = [cid for cid in scene_camera_ids if cid not in camera_params]
    if missing and strict:
        raise KeyError(f"Missing cameras from camera payload for scene order: {missing}")

    aligned = [cid for cid in scene_camera_ids if cid in camera_params]
    return aligned, missing


def _save_camera_npz(output_dir: Path, camera_ids: list[str], camera_params: dict[str, dict[str, np.ndarray]]) -> Path:
    intrinsics = np.stack([camera_params[cid]["intrinsic"] for cid in camera_ids], axis=0).astype(np.float32)
    cam_to_world = np.stack([camera_params[cid]["cam_to_world"] for cid in camera_ids], axis=0).astype(np.float32)
    world_to_cam = np.stack([camera_params[cid]["world_to_cam"] for cid in camera_ids], axis=0).astype(np.float32)
    path = output_dir / "normalized_camera_params.npz"
    np.savez_compressed(
        path,
        camera_ids=np.asarray(camera_ids),
        intrinsics=intrinsics,
        cam_to_world=cam_to_world,
        world_to_cam=world_to_cam,
    )
    return path


def _save_smplx_npz(output_dir: Path, smplx_params: dict[str, np.ndarray]) -> Path:
    path = output_dir / "normalized_smplx_params.npz"
    np.savez_compressed(path, **smplx_params)
    return path


def main() -> int:
    args = parse_args()

    smplx_path = Path(args.smplx_input).expanduser().resolve()
    if not smplx_path.is_file():
        raise FileNotFoundError(f"SMPL-X input not found: {smplx_path}")

    camera_path = Path(args.camera_input).expanduser().resolve() if args.camera_input else None
    if camera_path is not None and not camera_path.is_file():
        raise FileNotFoundError(f"Camera input not found: {camera_path}")

    scene_dir = Path(args.scene_dir).expanduser().resolve() if args.scene_dir else None
    scene_manifest = Path(args.scene_manifest).expanduser().resolve() if args.scene_manifest else None
    manifest = _load_scene_manifest(scene_dir=scene_dir, scene_manifest=scene_manifest)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_output = output_dir / "external_prior_bundle_manifest.json"
    smplx_output = output_dir / "normalized_smplx_params.npz"
    camera_output = output_dir / "normalized_camera_params.npz"
    planned_outputs = [manifest_output, smplx_output]
    if camera_path is not None:
        planned_outputs.append(camera_output)
    if not args.overwrite:
        for path in planned_outputs:
            if path.exists():
                raise FileExistsError(f"{path} already exists. Re-run with --overwrite.")

    smplx_payload = _load_npz_or_json(smplx_path)
    smplx_params = _normalize_smplx_params(smplx_payload, frame_idx=args.frame_idx, strict=args.strict)
    _save_smplx_npz(output_dir, smplx_params)

    camera_ids: list[str] = []
    missing_in_camera: list[str] = []
    if camera_path is not None:
        camera_payload = _load_npz_or_json(camera_path)
        camera_params = _parse_camera_payload(camera_payload, strict=args.strict)
        camera_ids, missing_in_camera = _align_camera_order(camera_params, manifest, strict=args.strict)
        if camera_ids:
            _save_camera_npz(output_dir, camera_ids, camera_params)
        elif args.strict:
            raise ValueError("No camera parameters were produced after parsing/alignment.")
    else:
        camera_params = {}

    bundle_manifest = {
        "format_version": 1,
        "frame_idx": int(args.frame_idx),
        "smplx_source": str(smplx_path),
        "camera_source": str(camera_path) if camera_path else None,
        "scene_dir": str(scene_dir) if scene_dir else None,
        "scene_manifest_loaded": manifest is not None,
        "smplx_output": smplx_output.name,
        "camera_output": camera_output.name if camera_ids else None,
        "camera_ids_ordered": camera_ids,
        "missing_scene_cameras_in_camera_file": missing_in_camera,
        "smplx_keys": sorted(smplx_params.keys()),
        "notes": [
            "This bundle is a normalized adapter layer for external SMPL-X/camera payloads.",
            "Current prior flow minimum SMPL-X requirement is betas + fullpose.",
            "When scene_manifest is provided, camera order follows exported_views camera_id order.",
        ],
    }
    manifest_output.write_text(json.dumps(bundle_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[import-external-smplx] wrote: {smplx_output}")
    if camera_ids:
        print(f"[import-external-smplx] wrote: {camera_output} ({len(camera_ids)} cameras)")
    else:
        print("[import-external-smplx] camera output skipped (no camera input or no aligned camera ids).")
    print(f"[import-external-smplx] wrote: {manifest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
