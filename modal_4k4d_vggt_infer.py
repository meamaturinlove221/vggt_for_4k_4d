from __future__ import annotations

import json
import os
import sys
import time
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
        with dest_path.open("wb") as handle:
            if concurrency is not None and concurrency > 0:
                output_volume._read_file_into_fileobj(entry.path, handle, concurrency=concurrency)
            else:
                output_volume.read_file_into_fileobj(entry.path, handle)


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
    import numpy as np

    output_subdir = _normalize_subpath(output_subdir)
    output_root = _remote_output_path(output_subdir)
    predictions_path = output_root / "predictions.npz"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Remote predictions not found: {predictions_path}")

    with np.load(predictions_path, allow_pickle=False) as loaded:
        arrays = {key: np.array(loaded[key]) for key in loaded.files}

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
    if download_local_dir.strip():
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
