from __future__ import annotations

import json
import os
import sys
import time
from io import BytesIO
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import modal


REPO_ROOT = Path(__file__).resolve().parent
REMOTE_CODE_DIR = PurePosixPath("/workspace/vggt")
REMOTE_DATA_DIR = PurePosixPath("/mnt/data")
REMOTE_OUTPUT_DIR = PurePosixPath("/mnt/out")


def _load_requirements(path: Path) -> list[str]:
    packages: list[str] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line not in seen:
            seen.add(line)
            packages.append(line)
    return packages


DEFAULT_REQUIREMENTS = [
    "torch==2.3.1",
    "torchvision==0.18.1",
    "numpy==1.26.1",
    "Pillow",
    "huggingface_hub",
    "einops",
    "safetensors",
]


def _resolve_requirements() -> list[str]:
    candidate = REPO_ROOT / "requirements.txt"
    if candidate.exists():
        return _load_requirements(candidate)
    return list(DEFAULT_REQUIREMENTS)


APP_NAME = os.environ.get("VGGT_MODAL_APP_NAME", "vggt-4k4d-infer")
DATA_VOLUME_NAME = os.environ.get("VGGT_MODAL_DATA_VOLUME", "vggt-4k4d-data")
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_MODAL_OUTPUT_VOLUME", "vggt-4k4d-output")
GPU_SPEC = os.environ.get("VGGT_MODAL_GPU", "A100-40GB")
CPU_COUNT = float(os.environ.get("VGGT_MODAL_CPU", "8"))
MEMORY_MB = int(os.environ.get("VGGT_MODAL_MEMORY_MB", "65536"))
TIMEOUT_SEC = int(os.environ.get("VGGT_MODAL_TIMEOUT_SEC", str(6 * 60 * 60)))

CODE_SYNC_IGNORE = [
    ".git",
    ".git/**",
    "__pycache__",
    "__pycache__/**",
    ".venv*",
    ".venv*/**",
    "output",
    "output/**",
    "reports",
    "reports/**",
]

INFER_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(*_resolve_requirements())
    .add_local_dir(
        str(REPO_ROOT / "vggt"),
        remote_path=(REMOTE_CODE_DIR / "vggt").as_posix(),
        ignore=CODE_SYNC_IGNORE,
    )
)

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)


@dataclass
class InferenceConfig:
    scene_subdir: str
    output_subdir: str = ""
    image_mode: str = "pad"
    hf_repo: str = "facebook/VGGT-1B"
    checkpoint_relpath: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(blob: str) -> "InferenceConfig":
        return InferenceConfig(**json.loads(blob))


def _normalize_subpath(value: str) -> str:
    cleaned = (value or "").strip().replace("\\", "/").strip("/")
    if not cleaned:
        raise ValueError("Expected a non-empty volume-relative path.")
    return cleaned


def _remote_data_path(subpath: str) -> Path:
    return Path(str(REMOTE_DATA_DIR / _normalize_subpath(subpath)))


def _remote_output_path(subpath: str) -> Path:
    return Path(str(REMOTE_OUTPUT_DIR / _normalize_subpath(subpath)))


def _resolve_output_root(scene_subdir: str, output_subdir: str) -> Path:
    if output_subdir.strip():
        return _remote_output_path(output_subdir)
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    safe_scene = Path(scene_subdir).name.replace(" ", "_")
    return Path(str(REMOTE_OUTPUT_DIR / "vggt_4k4d_infer" / f"{run_tag}_{safe_scene}"))


def _extract_model_state_dict(payload):
    if isinstance(payload, dict):
        if "model" in payload and isinstance(payload["model"], dict):
            return payload["model"]
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")


def _infer_model_kwargs_from_state_dict(state_dict: dict) -> dict:
    camera_token = state_dict.get("aggregator.camera_token")
    embed_dim = int(camera_token.shape[-1]) if camera_token is not None else 1024
    proj0 = state_dict.get("aggregator.human_prior_adapter.proj.0.weight")
    summary_proj0 = state_dict.get("aggregator.human_prior_adapter.summary_proj.0.weight")
    gate = state_dict.get("aggregator.human_prior_adapter.input_fusion.gate")
    scale_factors = state_dict.get("aggregator.human_prior_adapter.scale_factors_tensor")
    if scale_factors is not None:
        scale_factors = [int(value) for value in scale_factors.tolist()]
    else:
        scale_factors = [1]
    return {
        "img_size": 518,
        "patch_size": 14,
        "embed_dim": embed_dim,
        "enable_camera": any(key.startswith("camera_head.") for key in state_dict),
        "enable_point": any(key.startswith("point_head.") for key in state_dict),
        "enable_depth": any(key.startswith("depth_head.") for key in state_dict),
        "enable_normal": any(key.startswith("normal_head.") for key in state_dict),
        "enable_track": any(key.startswith("track_head.") for key in state_dict),
        "human_prior_channels": int(proj0.shape[1]) if proj0 is not None else 0,
        "human_prior_summary_channels": int(summary_proj0.shape[1]) if summary_proj0 is not None else 0,
        "human_prior_hidden_dim": int(proj0.shape[0]) if proj0 is not None else 64,
        "human_prior_gate_init": float(gate.item()) if gate is not None else 0.0,
        "human_prior_multi_scale_factors": scale_factors,
    }


def _upload_dir(local_dir: Path, remote_subdir: str) -> str:
    local_dir = local_dir.expanduser().resolve()
    if not local_dir.is_dir():
        raise NotADirectoryError(f"Scene directory not found: {local_dir}")
    remote_subdir = _normalize_subpath(remote_subdir)
    print(f"[modal-4k4d] upload scene: {local_dir} -> {DATA_VOLUME_NAME}:{remote_subdir}")
    with data_volume.batch_upload(force=True) as batch:
        for path in local_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(local_dir).as_posix()
            batch.put_file(str(path), f"{remote_subdir}/{rel}")
    return remote_subdir


def _download_volume_dir(remote_subdir: str, local_dir: Path, concurrency: int | None = None) -> None:
    remote_subdir = _normalize_subpath(remote_subdir)
    local_dir = local_dir.expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    remote_prefix = Path(remote_subdir)
    files_downloaded = 0

    for entry in output_volume.listdir(remote_subdir, recursive=True):
        rel_path = Path(entry.path)
        try:
            rel_path = rel_path.relative_to(remote_prefix)
        except ValueError:
            pass
        dest_path = local_dir / rel_path
        if entry.type == modal.volume.FileEntryType.DIRECTORY:
            dest_path.mkdir(parents=True, exist_ok=True)
            continue
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        last_error = None
        for attempt in range(1, 6):
            tmp_path = dest_path.with_suffix(dest_path.suffix + f".download{attempt}.tmp")
            try:
                with tmp_path.open("wb") as handle:
                    if concurrency is not None and concurrency > 0:
                        output_volume._read_file_into_fileobj(entry.path, handle, concurrency=concurrency)
                    else:
                        output_volume.read_file_into_fileobj(entry.path, handle)
                tmp_path.replace(dest_path)
                files_downloaded += 1
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001 - Modal volume reads can fail transiently.
                last_error = exc
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                time.sleep(min(2 ** attempt, 20))
        if last_error is not None:
            raise RuntimeError(f"Failed to download {entry.path} after retries") from last_error
    print(f"[modal-4k4d] downloaded {files_downloaded} files from {remote_subdir} to {local_dir}")


def _write_prediction_chunks(output_dir: Path, arrays: dict, chunk_views: int) -> dict:
    import numpy as np
    import shutil

    if chunk_views < 1:
        raise ValueError("chunk_views must be >= 1")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_views = None
    for value in arrays.values():
        if isinstance(value, np.ndarray) and value.ndim > 0:
            num_views = int(value.shape[0])
            break
    if num_views is None:
        raise ValueError("Could not infer num_views from predictions payload.")

    per_view_arrays = {}
    static_arrays = {}
    for name, value in arrays.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and int(value.shape[0]) == num_views:
            per_view_arrays[name] = value
        else:
            static_arrays[name] = value

    manifest = {
        "num_views": num_views,
        "chunk_views": int(chunk_views),
        "per_view_keys": sorted(per_view_arrays.keys()),
        "static_keys": sorted(static_arrays.keys()),
        "chunks": [],
    }

    if static_arrays:
        np.savez_compressed(output_dir / "static_arrays.npz", **static_arrays)

    for start in range(0, num_views, chunk_views):
        end = min(num_views, start + chunk_views)
        chunk_name = f"chunk_{start:03d}_{end - 1:03d}.npz"
        chunk_path = output_dir / chunk_name
        np.savez_compressed(
            chunk_path,
            **{name: value[start:end] for name, value in per_view_arrays.items()},
        )
        manifest["chunks"].append(
            {
                "file": chunk_name,
                "start": start,
                "end": end,
                "num_views": end - start,
                "size_bytes": chunk_path.stat().st_size,
            }
        )

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def _load_prediction_arrays_from_output_subdir(output_subdir: str) -> dict:
    import numpy as np

    output_subdir = _normalize_subpath(output_subdir)
    output_root = _remote_output_path(output_subdir)
    predictions_path = output_root / "predictions.npz"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Remote predictions not found: {predictions_path}")

    with np.load(predictions_path, allow_pickle=False) as loaded:
        return {key: np.array(loaded[key]) for key in loaded.files}


def _reassemble_prediction_chunks(local_chunk_dir: Path, output_path: Path) -> dict:
    import numpy as np

    manifest_path = local_chunk_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Chunk manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    arrays: dict[str, np.ndarray] = {}
    if manifest.get("static_keys"):
        static_path = local_chunk_dir / "static_arrays.npz"
        if not static_path.is_file():
            raise FileNotFoundError(f"Static chunk payload not found: {static_path}")
        with np.load(static_path, allow_pickle=False) as static_data:
            for key in static_data.files:
                arrays[key] = np.array(static_data[key])

    for key in manifest.get("per_view_keys", []):
        parts = []
        for chunk in manifest.get("chunks", []):
            chunk_path = local_chunk_dir / chunk["file"]
            if not chunk_path.is_file():
                raise FileNotFoundError(f"Missing chunk file: {chunk_path}")
            with np.load(chunk_path, allow_pickle=False) as chunk_data:
                parts.append(np.array(chunk_data[key]))
        arrays[key] = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)

    with np.load(output_path, allow_pickle=False) as final_data:
        output_shapes = {key: list(np.array(final_data[key]).shape) for key in final_data.files}

    return {
        "output_path": str(output_path),
        "num_keys": len(output_shapes),
        "output_shapes": output_shapes,
    }


def _preprocess_remote_mask(mask_path: Path, target_size: int):
    import numpy as np
    from PIL import Image

    img = Image.open(mask_path).convert("L")
    width, height = img.size
    if width >= height:
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
    else:
        new_height = target_size
        new_width = round(width * (new_height / height) / 14) * 14

    img = img.resize((new_width, new_height), Image.Resampling.NEAREST)
    arr = np.asarray(img, dtype=np.uint8)
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    top = (target_size - new_height) // 2
    left = (target_size - new_width) // 2
    canvas[top : top + new_height, left : left + new_width] = arr
    return canvas


def _load_remote_mask_stack(scene_subdir: str, target_size: int):
    import numpy as np

    scene_dir = _remote_data_path(scene_subdir)
    mask_dir = scene_dir / "masks"
    if not mask_dir.is_dir():
        return None
    mask_paths = sorted(path for path in mask_dir.iterdir() if path.is_file())
    if not mask_paths:
        return None
    masks = [_preprocess_remote_mask(path, target_size=target_size) for path in mask_paths]
    return np.stack(masks, axis=0)


def _build_filtered_points(
    world_points,
    world_points_conf,
    masks,
    *,
    conf_percentile: float,
):
    import numpy as np

    points = world_points.reshape(-1, 3)
    conf = world_points_conf.reshape(-1)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(conf) & (conf > 0)
    if masks is not None:
        valid &= masks.reshape(-1) > 0

    if not np.any(valid):
        raise RuntimeError("No valid points after filtering.")

    conf_valid = conf[valid]
    conf_threshold = float(np.percentile(conf_valid, conf_percentile))
    keep = valid & (conf >= conf_threshold)
    if not np.any(keep):
        keep = valid
    return points[keep], {
        "valid_points_before_conf": int(valid.sum()),
        "conf_threshold": conf_threshold,
        "points_after_conf": int(keep.sum()),
    }


def _apply_roi_filter(points, roi: str):
    import numpy as np

    if roi == "full":
        return points, {"roi": roi, "points_after_roi": int(len(points))}

    if len(points) < 32:
        return points, {"roi": roi, "fallback": "too_few_points", "points_after_roi": int(len(points))}

    # Keep ROI selection consistent with the Open3D render tooling: smaller
    # world-space y appears higher on screen because the viewer uses up=(0,-1,0).
    height_like = -points[:, 1]
    head_percentile = 78.0 if roi == "head" else 74.0
    head_cut = float(np.percentile(height_like, head_percentile))
    head_mask = height_like >= head_cut
    if int(head_mask.sum()) < 512:
        relaxed_cut = float(np.percentile(height_like, 68.0))
        head_mask = height_like >= relaxed_cut
        head_cut = relaxed_cut

    roi_mask = head_mask
    summary = {
        "roi": roi,
        "vertical_axis": "-y_is_up",
        "head_cut_height_like": head_cut,
        "points_after_head_cut": int(head_mask.sum()),
    }

    if roi == "face":
        head_points = points[head_mask]
        if len(head_points) >= 256:
            x_lo, x_hi = np.percentile(head_points[:, 0], [20.0, 80.0])
            z_lo, z_hi = np.percentile(head_points[:, 2], [15.0, 85.0])
            head_height_like = -head_points[:, 1]
            height_lo = float(np.percentile(head_height_like, 25.0))
            face_mask = (
                head_mask
                & (points[:, 0] >= float(x_lo))
                & (points[:, 0] <= float(x_hi))
                & (points[:, 2] >= float(z_lo))
                & (points[:, 2] <= float(z_hi))
                & (height_like >= height_lo)
            )
            if int(face_mask.sum()) >= 128:
                roi_mask = face_mask
                summary.update(
                    {
                        "x_lo": float(x_lo),
                        "x_hi": float(x_hi),
                        "z_lo": float(z_lo),
                        "z_hi": float(z_hi),
                        "face_height_like_lo": height_lo,
                    }
                )
            else:
                summary["fallback"] = "face_mask_too_small"
        else:
            summary["fallback"] = "head_mask_too_small"

    filtered_points = points[roi_mask]
    summary["points_after_roi"] = int(len(filtered_points))
    return filtered_points, summary


def _to_numpy(tensor):
    return tensor.detach().float().cpu().numpy()


def _write_preview_png(array, output_path: Path) -> None:
    import numpy as np
    from PIL import Image

    arr = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        preview = np.zeros(arr.shape, dtype=np.uint8)
    else:
        lo = float(np.percentile(arr[finite], 2))
        hi = float(np.percentile(arr[finite], 98))
        if hi <= lo:
            hi = lo + 1e-6
        scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        preview = (scaled * 255.0).astype(np.uint8)
    Image.fromarray(preview).save(output_path)


def _write_normal_preview_png(array, output_path: Path) -> None:
    import numpy as np
    from PIL import Image

    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected normal map with shape [H, W, 3], got {arr.shape}")

    finite = np.isfinite(arr).all(axis=-1, keepdims=True)
    arr = np.where(finite, arr, 0.0)
    arr = arr / np.clip(np.linalg.norm(arr, axis=-1, keepdims=True), 1e-6, None)
    preview = np.clip((arr + 1.0) * 0.5, 0.0, 1.0)
    Image.fromarray((preview * 255.0).astype(np.uint8)).save(output_path)


@app.function(
    image=INFER_IMAGE,
    gpu=GPU_SPEC,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SEC,
    volumes={
        REMOTE_DATA_DIR.as_posix(): data_volume,
        REMOTE_OUTPUT_DIR.as_posix(): output_volume,
    },
)
def run_remote_vggt_inference(cfg_json: str) -> dict:
    cfg = InferenceConfig.from_json(cfg_json)
    remote_code_dir = Path(str(REMOTE_CODE_DIR))
    if str(remote_code_dir) not in sys.path:
        sys.path.insert(0, str(remote_code_dir))

    import numpy as np
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    scene_dir = _remote_data_path(cfg.scene_subdir)
    image_dir = scene_dir / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Remote image dir not found: {image_dir}")
    image_paths = sorted([path for path in image_dir.iterdir() if path.is_file()])
    if not image_paths:
        raise FileNotFoundError(f"No images found under {image_dir}")

    output_root = _resolve_output_root(cfg.scene_subdir, cfg.output_subdir)
    output_root.mkdir(parents=True, exist_ok=True)

    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    start_time = time.time()
    images = load_and_preprocess_images([str(path) for path in image_paths], mode=cfg.image_mode).to(device)
    prior_maps = None
    prior_summary_tokens = None
    prior_maps_path = scene_dir / "prior_maps.npz"
    if prior_maps_path.is_file():
        with np.load(prior_maps_path, allow_pickle=False) as prior_payload:
            prior_maps = torch.from_numpy(np.array(prior_payload["prior_maps"])).to(device=device, dtype=torch.float32)
            if "prior_summary_tokens" in prior_payload.files:
                prior_summary_tokens = torch.from_numpy(
                    np.array(prior_payload["prior_summary_tokens"])
                ).to(device=device, dtype=torch.float32)

    checkpoint_path = None
    if cfg.checkpoint_relpath.strip():
        checkpoint_path = _remote_output_path(cfg.checkpoint_relpath)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Remote checkpoint not found: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location="cpu")
        state_dict = _extract_model_state_dict(payload)
        model_kwargs = payload.get("model_kwargs") if isinstance(payload, dict) else None
        if not isinstance(model_kwargs, dict):
            model_kwargs = _infer_model_kwargs_from_state_dict(state_dict)
        model = VGGT(**model_kwargs)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint load mismatch for {checkpoint_path}: missing={missing}, unexpected={unexpected}"
            )
    else:
        model = VGGT.from_pretrained(cfg.hf_repo)

    if prior_maps is not None and getattr(model.aggregator, "human_prior_channels", 0) <= 0:
        raise RuntimeError(
            "Scene includes prior_maps.npz, but the loaded model does not have a human prior adapter. "
            "Use a prior-enabled checkpoint."
        )
    if prior_summary_tokens is not None and getattr(model.aggregator, "human_prior_summary_channels", 0) <= 0:
        raise RuntimeError(
            "Scene includes prior_summary_tokens, but the loaded model does not have a summary-token adapter. "
            "Use a summary-enabled checkpoint."
        )

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(
                images,
                prior_maps=prior_maps,
                prior_summary_tokens=prior_summary_tokens,
            )

    pose_enc = predictions["pose_enc"]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    arrays = {
        "pose_enc": _to_numpy(pose_enc.squeeze(0)),
        "extrinsic": _to_numpy(extrinsic.squeeze(0)),
        "intrinsic": _to_numpy(intrinsic.squeeze(0)),
        "depth": _to_numpy(predictions["depth"].squeeze(0)),
        "depth_conf": _to_numpy(predictions["depth_conf"].squeeze(0)),
        "world_points": _to_numpy(predictions["world_points"].squeeze(0)),
        "world_points_conf": _to_numpy(predictions["world_points_conf"].squeeze(0)),
    }
    if "normal" in predictions:
        arrays["normal"] = _to_numpy(predictions["normal"].squeeze(0))
    if "normal_conf" in predictions:
        arrays["normal_conf"] = _to_numpy(predictions["normal_conf"].squeeze(0))
    np.savez_compressed(output_root / "predictions.npz", **arrays)

    preview_dir = output_root / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for idx, image_path in enumerate(image_paths):
        stem = image_path.stem
        _write_preview_png(arrays["depth"][idx, ..., 0], preview_dir / f"{stem}_depth.png")
        _write_preview_png(arrays["depth_conf"][idx], preview_dir / f"{stem}_depth_conf.png")
        _write_preview_png(arrays["world_points_conf"][idx], preview_dir / f"{stem}_point_conf.png")
        if "normal" in arrays:
            _write_normal_preview_png(arrays["normal"][idx], preview_dir / f"{stem}_normal.png")
        if "normal_conf" in arrays:
            _write_preview_png(arrays["normal_conf"][idx], preview_dir / f"{stem}_normal_conf.png")

    scene_manifest_path = scene_dir / "scene_manifest.json"
    scene_manifest = {}
    if scene_manifest_path.exists():
        scene_manifest = json.loads(scene_manifest_path.read_text(encoding="utf-8"))

    summary = {
        "scene_subdir": cfg.scene_subdir,
        "image_mode": cfg.image_mode,
        "hf_repo": cfg.hf_repo,
        "checkpoint_relpath": cfg.checkpoint_relpath,
        "image_names": [path.name for path in image_paths],
        "num_images": len(image_paths),
        "device": device,
        "dtype": str(dtype),
        "input_tensor_shape": list(images.shape),
        "prior_tensor_shape": list(prior_maps.shape) if prior_maps is not None else None,
        "prior_summary_tensor_shape": list(prior_summary_tokens.shape) if prior_summary_tokens is not None else None,
        "output_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "gpu_name": torch.cuda.get_device_name(0),
        "elapsed_seconds": round(time.time() - start_time, 3),
        "scene_manifest": scene_manifest,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "scene_subdir": cfg.scene_subdir,
        "output_root": output_root.as_posix(),
        "output_subdir": output_root.relative_to(Path(str(REMOTE_OUTPUT_DIR))).as_posix(),
        "image_mode": cfg.image_mode,
        "hf_repo": cfg.hf_repo,
        "checkpoint_relpath": cfg.checkpoint_relpath,
        "image_names": [path.name for path in image_paths],
        "num_images": len(image_paths),
        "device": device,
        "dtype": str(dtype),
        "input_tensor_shape": list(images.shape),
        "prior_tensor_shape": list(prior_maps.shape) if prior_maps is not None else None,
        "prior_summary_tensor_shape": list(prior_summary_tokens.shape) if prior_summary_tokens is not None else None,
        "output_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "gpu_name": torch.cuda.get_device_name(0),
        "elapsed_seconds": round(time.time() - start_time, 3),
        "scene_manifest": scene_manifest,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    output_volume.commit()
    print("[modal-4k4d] output_root =", output_root.as_posix(), flush=True)
    print("[modal-4k4d] committed output volume", flush=True)
    return summary


@app.function(
    image=INFER_IMAGE,
    cpu=2,
    memory=8192,
    timeout=TIMEOUT_SEC,
    volumes={
        REMOTE_OUTPUT_DIR.as_posix(): output_volume,
    },
)
def export_saved_prediction_chunks_remote(output_subdir: str, chunk_views: int = 5) -> dict:
    output_subdir = _normalize_subpath(output_subdir)
    output_root = _remote_output_path(output_subdir)
    arrays = _load_prediction_arrays_from_output_subdir(output_subdir)

    chunk_dir = output_root / f"predictions_chunks_v{int(chunk_views)}"
    manifest = _write_prediction_chunks(chunk_dir, arrays, chunk_views=int(chunk_views))
    chunk_subdir = chunk_dir.relative_to(Path(str(REMOTE_OUTPUT_DIR))).as_posix()
    output_volume.commit()
    return {
        "output_subdir": output_subdir,
        "chunk_subdir": chunk_subdir,
        "chunk_dir_name": chunk_dir.name,
        "manifest": manifest,
    }


@app.function(
    image=INFER_IMAGE,
    cpu=2,
    memory=8192,
    timeout=TIMEOUT_SEC,
    volumes={
        REMOTE_OUTPUT_DIR.as_posix(): output_volume,
    },
)
def export_prediction_chunk_bytes_remote(output_subdir: str, start: int, end: int) -> bytes:
    import numpy as np

    arrays = _load_prediction_arrays_from_output_subdir(output_subdir)
    num_views = None
    for value in arrays.values():
        if isinstance(value, np.ndarray) and value.ndim > 0:
            num_views = int(value.shape[0])
            break
    if num_views is None:
        raise ValueError("Could not infer num_views from predictions payload.")

    start = max(0, int(start))
    end = min(num_views, int(end))
    if end <= start:
        raise ValueError(f"Invalid chunk range: start={start}, end={end}, num_views={num_views}")

    chunk_arrays = {}
    for name, value in arrays.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and int(value.shape[0]) == num_views:
            chunk_arrays[name] = value[start:end]
        else:
            chunk_arrays[name] = value

    buffer = BytesIO()
    np.savez_compressed(buffer, **chunk_arrays)
    return buffer.getvalue()


@app.function(
    image=INFER_IMAGE,
    cpu=1,
    memory=2048,
    timeout=TIMEOUT_SEC,
    volumes={
        REMOTE_OUTPUT_DIR.as_posix(): output_volume,
    },
)
def read_output_summary_remote(output_subdir: str) -> dict:
    output_subdir = _normalize_subpath(output_subdir)
    output_root = _remote_output_path(output_subdir)
    summary_path = output_root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Remote summary not found: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


@app.function(
    image=INFER_IMAGE,
    cpu=2,
    memory=8192,
    timeout=TIMEOUT_SEC,
    volumes={
        REMOTE_DATA_DIR.as_posix(): data_volume,
        REMOTE_OUTPUT_DIR.as_posix(): output_volume,
    },
)
def summarize_prediction_roi_remote(
    output_subdir: str,
    scene_subdir: str,
    conf_percentile: float = 40.0,
) -> dict:
    import numpy as np

    arrays = _load_prediction_arrays_from_output_subdir(output_subdir)
    world_points = np.asarray(arrays["world_points"], dtype=np.float32)
    world_points_conf = np.asarray(arrays["world_points_conf"], dtype=np.float32)
    target_size = int(world_points.shape[1])
    masks = _load_remote_mask_stack(scene_subdir, target_size=target_size)

    points, filter_summary = _build_filtered_points(
        world_points,
        world_points_conf,
        masks,
        conf_percentile=float(conf_percentile),
    )

    _, full_summary = _apply_roi_filter(points, "full")
    _, head_summary = _apply_roi_filter(points, "head")
    _, face_summary = _apply_roi_filter(points, "face")

    return {
        "output_subdir": _normalize_subpath(output_subdir),
        "scene_subdir": _normalize_subpath(scene_subdir),
        "conf_percentile": float(conf_percentile),
        "filter_summary": filter_summary,
        "full_roi": full_summary,
        "head_roi": head_summary,
        "face_roi": face_summary,
    }


@app.local_entrypoint()
def upload_scene(
    local_scene_dir: str,
    remote_scene_subdir: str = "",
) -> None:
    local_dir = Path(local_scene_dir).expanduser().resolve()
    if not remote_scene_subdir.strip():
        remote_scene_subdir = f"scenes/{local_dir.name}"
    remote_subdir = _upload_dir(local_dir, remote_scene_subdir)
    print(f"[modal-4k4d] scene uploaded to {DATA_VOLUME_NAME}:{remote_subdir}")


@app.local_entrypoint()
def run_scene(
    scene_subdir: str,
    output_subdir: str = "",
    image_mode: str = "pad",
    hf_repo: str = "facebook/VGGT-1B",
    checkpoint_relpath: str = "",
    download_local_dir: str = "",
    skip_download: bool = False,
) -> None:
    cfg = InferenceConfig(
        scene_subdir=scene_subdir,
        output_subdir=output_subdir,
        image_mode=image_mode,
        hf_repo=hf_repo,
        checkpoint_relpath=checkpoint_relpath,
    )
    print("[modal-4k4d] launch config:")
    print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
    summary = run_remote_vggt_inference.remote(cfg.to_json())
    print("[modal-4k4d] remote summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if skip_download:
        print("[modal-4k4d] skip_download=True; artifacts remain on the output volume")
    elif download_local_dir.strip():
        local_dir = Path(download_local_dir).expanduser().resolve()
        _download_volume_dir(summary["output_subdir"], local_dir)
        print(f"[modal-4k4d] downloaded artifacts to {local_dir}")


@app.local_entrypoint()
def run_scene_from_local(
    local_scene_dir: str,
    remote_scene_subdir: str = "",
    output_subdir: str = "",
    image_mode: str = "pad",
    hf_repo: str = "facebook/VGGT-1B",
    checkpoint_relpath: str = "",
    download_local_dir: str = "",
    skip_download: bool = False,
) -> None:
    local_dir = Path(local_scene_dir).expanduser().resolve()
    if not remote_scene_subdir.strip():
        remote_scene_subdir = f"scenes/{local_dir.name}"
    remote_subdir = _upload_dir(local_dir, remote_scene_subdir)
    cfg = InferenceConfig(
        scene_subdir=remote_subdir,
        output_subdir=output_subdir,
        image_mode=image_mode,
        hf_repo=hf_repo,
        checkpoint_relpath=checkpoint_relpath,
    )
    print("[modal-4k4d] upload+run config:")
    print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
    summary = run_remote_vggt_inference.remote(cfg.to_json())
    print("[modal-4k4d] remote summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if skip_download:
        print("[modal-4k4d] skip_download=True; artifacts remain on the output volume")
        return
    if download_local_dir.strip():
        local_dir = Path(download_local_dir).expanduser().resolve()
    else:
        local_dir = REPO_ROOT / "output" / "modal_results" / Path(summary["output_subdir"]).name
    _download_volume_dir(summary["output_subdir"], local_dir)
    print(f"[modal-4k4d] downloaded artifacts to {local_dir}")


@app.local_entrypoint()
def download_run(
    remote_output_subdir: str,
    local_output_dir: str,
) -> None:
    local_dir = Path(local_output_dir).expanduser().resolve()
    _download_volume_dir(remote_output_subdir, local_dir)
    print(f"[modal-4k4d] downloaded artifacts to {local_dir}")


@app.local_entrypoint()
def download_prediction_chunks(
    remote_output_subdir: str,
    local_output_dir: str = "",
    chunk_views: int = 5,
    assemble_predictions_npz: bool = True,
    download_concurrency: int = 1,
) -> None:
    manifest = export_saved_prediction_chunks_remote.remote(
        output_subdir=remote_output_subdir,
        chunk_views=chunk_views,
    )
    print("[modal-4k4d] remote chunk manifest:")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

    if local_output_dir.strip():
        local_dir = Path(local_output_dir).expanduser().resolve()
    else:
        local_dir = REPO_ROOT / "output" / "modal_results" / Path(remote_output_subdir).name

    chunk_local_dir = local_dir / manifest["chunk_dir_name"]
    _download_volume_dir(
        manifest["chunk_subdir"],
        chunk_local_dir,
        concurrency=max(1, int(download_concurrency)),
    )
    print(f"[modal-4k4d] downloaded chunked predictions to {chunk_local_dir}")

    if assemble_predictions_npz:
        assembled = _reassemble_prediction_chunks(chunk_local_dir, local_dir / "predictions.npz")
        print("[modal-4k4d] reassembled predictions:")
        print(json.dumps(assembled, indent=2, ensure_ascii=False))


@app.local_entrypoint()
def download_prediction_chunks_rpc(
    remote_output_subdir: str,
    local_output_dir: str = "",
    chunk_views: int = 1,
    overwrite: bool = False,
    resume_existing: bool = True,
    max_chunk_retries: int = 3,
) -> None:
    import numpy as np
    import time
    import zipfile

    remote_output_subdir = _normalize_subpath(remote_output_subdir)
    if local_output_dir.strip():
        local_dir = Path(local_output_dir).expanduser().resolve()
    else:
        local_dir = REPO_ROOT / "output" / "modal_results" / Path(remote_output_subdir).name

    if overwrite and local_dir.exists():
        import shutil

        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    manifest_payload = export_saved_prediction_chunks_remote.remote(
        output_subdir=remote_output_subdir,
        chunk_views=max(1, int(chunk_views)),
    )
    chunk_local_dir = local_dir / manifest_payload["chunk_dir_name"]
    chunk_local_dir.mkdir(parents=True, exist_ok=True)
    (chunk_local_dir / "manifest.json").write_text(
        json.dumps(manifest_payload["manifest"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = read_output_summary_remote.remote(remote_output_subdir)
    (local_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    downloaded_chunks = []
    skipped_chunks = []
    invalid_existing_chunks = []
    per_view_keys = list(manifest_payload["manifest"].get("per_view_keys", []))

    def _existing_chunk_is_valid(path: Path, chunk_info: dict) -> bool:
        if not path.is_file():
            return False
        expected_size = int(chunk_info.get("size_bytes", -1))
        if expected_size > 0 and path.stat().st_size != expected_size:
            return False
        try:
            with zipfile.ZipFile(path) as zip_file:
                if zip_file.testzip() is not None:
                    return False
            with np.load(path, allow_pickle=False) as chunk_data:
                for key in per_view_keys:
                    _ = chunk_data[key].shape
            return True
        except Exception:
            return False

    for chunk in manifest_payload["manifest"]["chunks"]:
        chunk_path = chunk_local_dir / chunk["file"]
        if resume_existing and _existing_chunk_is_valid(chunk_path, chunk):
            skipped_chunks.append(chunk["file"])
            continue
        if chunk_path.exists():
            invalid_existing_chunks.append(chunk["file"])
        last_error = None
        for attempt in range(1, max(1, int(max_chunk_retries)) + 1):
            try:
                chunk_bytes = export_prediction_chunk_bytes_remote.remote(
                    output_subdir=remote_output_subdir,
                    start=int(chunk["start"]),
                    end=int(chunk["end"]),
                )
                break
            except Exception as exc:
                last_error = exc
                print(
                    f"[modal-4k4d] chunk download failed "
                    f"{chunk['file']} attempt {attempt}/{max_chunk_retries}: {exc}"
                )
                if attempt < max(1, int(max_chunk_retries)):
                    time.sleep(min(30, 2 * attempt))
        else:
            raise RuntimeError(f"Failed to download {chunk['file']}") from last_error
        chunk_path.write_bytes(chunk_bytes)
        downloaded_chunks.append(chunk["file"])

    assembled = _reassemble_prediction_chunks(chunk_local_dir, local_dir / "predictions.npz")
    print("[modal-4k4d] rpc chunk manifest:")
    print(json.dumps(manifest_payload, indent=2, ensure_ascii=False))
    print("[modal-4k4d] rpc chunk resume:")
    print(
        json.dumps(
            {
                "resume_existing": bool(resume_existing),
                "max_chunk_retries": int(max_chunk_retries),
                "skipped_chunks": skipped_chunks,
                "invalid_existing_chunks": invalid_existing_chunks,
                "downloaded_chunks": downloaded_chunks,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print("[modal-4k4d] rpc summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("[modal-4k4d] rpc assembled predictions:")
    print(json.dumps(assembled, indent=2, ensure_ascii=False))


@app.local_entrypoint()
def summarize_prediction_roi(
    remote_output_subdir: str,
    scene_subdir: str,
    conf_percentile: float = 40.0,
) -> None:
    summary = summarize_prediction_roi_remote.remote(
        output_subdir=remote_output_subdir,
        scene_subdir=scene_subdir,
        conf_percentile=conf_percentile,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
