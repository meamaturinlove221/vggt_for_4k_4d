from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

REMOTE_IMPORT_ROOT = Path("/workspace/vggt")
if REMOTE_IMPORT_ROOT.is_dir() and str(REMOTE_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(REMOTE_IMPORT_ROOT))

import modal
import numpy as np

from tools.prepare_4k4d_prior_training_case import (
    load_scene_manifest,
    recover_legacy_crop_source_sizes,
    resolve_scene_camera_params,
)
from tools.research_scene_assets import CAMERA_SIDECAR_NAME


REPO_ROOT = Path(__file__).resolve().parent
REMOTE_CODE_DIR = PurePosixPath("/workspace/vggt")
REMOTE_DATA_DIR = PurePosixPath("/mnt/data")
REMOTE_OUTPUT_DIR = PurePosixPath("/mnt/out")
DEFAULT_STRICT_GATE_REGISTRY = REPO_ROOT / "reports" / "20260504_strict_gate_registry.json"
REQUIRED_STRICT_GATE_SCHEMA_VERSION = "20260504_visual_fullbody_hands_v2"

APP_NAME = os.environ.get("VGGT_MODAL_RESEARCH_APP_NAME", "vggt-surface-research-preflight")
DATA_VOLUME_NAME = os.environ.get("VGGT_MODAL_DATA_VOLUME", "vggt-4k4d-data")
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_MODAL_OUTPUT_VOLUME", "vggt-4k4d-output")
GPU_SPEC = os.environ.get("VGGT_MODAL_RESEARCH_GPU", os.environ.get("VGGT_MODAL_GPU", "A100-40GB"))
CPU_COUNT = float(os.environ.get("VGGT_MODAL_RESEARCH_CPU", "8"))
MEMORY_MB = int(os.environ.get("VGGT_MODAL_RESEARCH_MEMORY_MB", str(96 * 1024)))
TIMEOUT_SEC = int(os.environ.get("VGGT_MODAL_RESEARCH_TIMEOUT_SEC", str(6 * 60 * 60)))


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


def _resolve_base_requirements() -> list[str]:
    candidate = REPO_ROOT / "requirements.txt"
    if candidate.exists():
        return _load_requirements(candidate)
    return [
        "torch==2.3.1",
        "torchvision==0.18.1",
        "numpy==1.26.1",
        "Pillow",
        "opencv-python-headless",
    ]


def _registry_status(path: Path) -> dict:
    status = {
        "registry": str(path),
        "schema_version": None,
        "strict_candidate_passes": 0,
        "strict_teacher_passes": 0,
        "formal_cloud_allowed": False,
        "reasons": [],
    }
    if not path.is_file():
        status["reasons"].append("strict gate registry missing")
        return status
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        status["reasons"].append(f"strict gate registry unreadable: {exc}")
        return status
    counts = registry.get("counts", {})
    candidate_passes = int(counts.get("strict_candidate_passes", 0) or 0)
    teacher_passes = int(counts.get("strict_teacher_passes", 0) or 0)
    status.update(
        {
            "schema_version": registry.get("schema_version"),
            "generated_at": registry.get("generated_at"),
            "strict_candidate_passes": candidate_passes,
            "strict_teacher_passes": teacher_passes,
        }
    )
    if registry.get("schema_version") != REQUIRED_STRICT_GATE_SCHEMA_VERSION:
        status["reasons"].append("strict gate registry schema is not current")
    if candidate_passes <= 0:
        status["reasons"].append("strict_candidate_passes is 0")
    status["formal_cloud_allowed"] = not status["reasons"]
    return status


def _normalize_subpath(value: str) -> str:
    cleaned = (value or "").strip().replace("\\", "/").strip("/")
    if not cleaned:
        raise ValueError("Expected a non-empty volume-relative path.")
    if ".." in Path(cleaned).parts:
        raise ValueError(f"Parent traversal is not allowed: {value!r}")
    return cleaned


def _remote_data_path(subpath: str) -> Path:
    return Path(str(REMOTE_DATA_DIR / _normalize_subpath(subpath)))


def _remote_output_path(subpath: str) -> Path:
    return Path(str(REMOTE_OUTPUT_DIR / _normalize_subpath(subpath)))


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _upload_dir(local_dir: Path, remote_subdir: str) -> str:
    local_dir = local_dir.expanduser().resolve()
    if not local_dir.is_dir():
        raise NotADirectoryError(f"Research asset directory not found: {local_dir}")
    remote_subdir = _normalize_subpath(remote_subdir)
    if any(word in remote_subdir.lower() for word in ("strict_pass", "teacher_export", "candidate_export")):
        raise ValueError("Research asset remote path must not look like a pass/export path.")
    print(f"[surface-research] upload dir: {local_dir} -> {DATA_VOLUME_NAME}:{remote_subdir}")
    with data_volume.batch_upload(force=True) as batch:
        for path in local_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(local_dir).as_posix()
            batch.put_file(str(path), f"{remote_subdir}/{rel}")
    return remote_subdir


def _upload_file(local_path: Path, remote_subpath: str) -> str:
    local_path = local_path.expanduser().resolve()
    if not local_path.is_file():
        raise FileNotFoundError(f"Research asset file not found: {local_path}")
    remote_subpath = _normalize_subpath(remote_subpath)
    if any(word in remote_subpath.lower() for word in ("strict_pass", "teacher_export", "candidate_export")):
        raise ValueError("Research asset remote path must not look like a pass/export path.")
    print(f"[surface-research] upload file: {local_path} -> {DATA_VOLUME_NAME}:{remote_subpath}")
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(local_path), remote_subpath)
    return remote_subpath


def _write_scene_camera_sidecar(local_scene_dir: Path) -> Path:
    local_scene_dir = local_scene_dir.expanduser().resolve()
    manifest = recover_legacy_crop_source_sizes(local_scene_dir, load_scene_manifest(local_scene_dir))
    dataset_root = Path(str(manifest.get("dataset_root", "")))
    cameras, source = resolve_scene_camera_params(manifest, dataset_root, "data_used_in_4K4D")
    camera_ids = [str(view["camera_id"]) for view in manifest["exported_views"]]
    intrinsics = np.stack([np.asarray(cameras[camera_id]["intrinsic"], dtype=np.float32) for camera_id in camera_ids], axis=0)
    cam_to_world = np.stack([np.asarray(cameras[camera_id]["cam_to_world"], dtype=np.float32) for camera_id in camera_ids], axis=0)
    world_to_cam = np.stack([np.asarray(cameras[camera_id]["world_to_cam"], dtype=np.float32) for camera_id in camera_ids], axis=0)
    sidecar_path = local_scene_dir / CAMERA_SIDECAR_NAME
    np.savez_compressed(
        sidecar_path,
        camera_ids=np.asarray(camera_ids),
        intrinsics=intrinsics,
        cam_to_world=cam_to_world,
        world_to_cam=world_to_cam,
        source=np.asarray(str(source)),
    )
    return sidecar_path


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

RESEARCH_IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11")
    .env({"CUDA_HOME": "/usr/local/cuda", "FORCE_CUDA": "1", "MAX_JOBS": "8", "TORCH_CUDA_ARCH_LIST": "8.0"})
    .apt_install("git", "build-essential", "clang", "ffmpeg", "libglib2.0-0", "libsm6", "libxext6", "libxrender1")
    .pip_install(
        *_resolve_base_requirements(),
        "hydra-core",
        "omegaconf",
        "opencv-python-headless",
        "scipy",
        "ninja",
        "setuptools",
        "wheel",
    )
    .run_commands("python -m pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git")
    .add_local_dir(str(REPO_ROOT / "tools"), remote_path=(REMOTE_CODE_DIR / "tools").as_posix(), ignore=CODE_SYNC_IGNORE)
    .add_local_dir(str(REPO_ROOT / "vggt"), remote_path=(REMOTE_CODE_DIR / "vggt").as_posix(), ignore=CODE_SYNC_IGNORE)
)

CPU_RESEARCH_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libglib2.0-0", "libsm6", "libxext6", "libxrender1")
    .pip_install("numpy==1.26.1", "Pillow", "opencv-python-headless", "scipy", "h5py")
    .add_local_dir(str(REPO_ROOT / "tools"), remote_path=(REMOTE_CODE_DIR / "tools").as_posix(), ignore=CODE_SYNC_IGNORE)
    .add_local_dir(str(REPO_ROOT / "vggt"), remote_path=(REMOTE_CODE_DIR / "vggt").as_posix(), ignore=CODE_SYNC_IGNORE)
)

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)


@dataclass
class ResearchConfig:
    lane: str
    scene_subdir: str
    output_subdir: str
    template_payload_subpath: str = ""
    view_indices: str = "0,10,20,30,40,50"
    target_size: int = 96
    steps: int = 20
    token_grid: int = 5
    token_hidden: int = 64
    methods: str = "A1_neural_sdf,A2_gaussian_surface,A3_visual_hull_init"
    expected_gpu: str = ""
    grid_resolution: int = 56
    support_threshold: int = 4

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(blob: str) -> "ResearchConfig":
        return ResearchConfig(**json.loads(blob))


def _download_volume_dir(remote_subdir: str, local_dir: Path) -> None:
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
            output_volume.read_file_into_fileobj(entry.path, handle)


def _run_surface_research_impl(cfg_json: str, registry_status_json: str) -> dict:
    cfg = ResearchConfig.from_json(cfg_json)
    registry_status = json.loads(registry_status_json)
    if cfg.lane not in {"ping_scene", "A_readiness", "A3_visual_hull_init", "B0_surface_tokens"}:
        raise ValueError(f"Unsupported research lane: {cfg.lane}")
    if any(word in cfg.output_subdir.lower() for word in ("strict_pass", "teacher_export", "candidate_export")):
        raise ValueError("Research preflight output_subdir must not look like a pass/export path.")

    remote_code_dir = Path(str(REMOTE_CODE_DIR))
    scene_dir = _remote_data_path(cfg.scene_subdir)
    output_dir = _remote_output_path(cfg.output_subdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    run_meta = {
        "research_only": True,
        "no_teacher_export": True,
        "no_candidate_export": True,
        "no_strict_pass_write": True,
        "formal_cloud_train_infer_export": "blocked unless local strict gate passes",
        "strict_registry_status_at_launch": registry_status,
        "modal_gpu_spec_at_import": GPU_SPEC,
        "expected_gpu": cfg.expected_gpu,
        "lane": cfg.lane,
        "launched_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "research_preflight_launch_guard.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if cfg.lane == "ping_scene":
        scene_files = sorted(path.relative_to(scene_dir).as_posix() for path in scene_dir.glob("*"))
        image_count = len(list((scene_dir / "images").glob("*.png")))
        mask_count = len(list((scene_dir / "masks").glob("*.png")))
        sidecar = scene_dir / CAMERA_SIDECAR_NAME
        summary = {
            **run_meta,
            "status": "ping_scene_complete",
            "scene_dir": str(scene_dir),
            "scene_exists": scene_dir.is_dir(),
            "top_level_files": scene_files,
            "image_count": image_count,
            "mask_count": mask_count,
            "camera_sidecar_exists": sidecar.is_file(),
            "torch_cuda_available": None,
            "torch_cuda_device_count": None,
            "elapsed_seconds": round(time.time() - started, 3),
            "output_subdir": output_dir.relative_to(Path(str(REMOTE_OUTPUT_DIR))).as_posix(),
        }
        try:
            import torch

            summary["torch_cuda_available"] = bool(torch.cuda.is_available())
            summary["torch_cuda_device_count"] = int(torch.cuda.device_count())
            if torch.cuda.is_available():
                summary["torch_cuda_device_name"] = torch.cuda.get_device_name(0)
        except Exception as exc:  # noqa: BLE001
            summary["torch_error"] = repr(exc)
        (output_dir / "ping_scene_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        (output_dir / "research_preflight_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        output_volume.commit()
        return summary
    elif cfg.lane == "A_readiness":
        cmd = [
            sys.executable,
            str(remote_code_dir / "tools" / "preflight_dense_teacher_reconstruction.py"),
            "--scene-dir",
            str(scene_dir),
            "--output-dir",
            str(output_dir / "A_readiness"),
            "--view-indices",
            cfg.view_indices,
            "--target-size",
            str(int(cfg.target_size)),
            "--methods",
            cfg.methods,
            "--overwrite",
        ]
    elif cfg.lane == "A3_visual_hull_init":
        if not cfg.template_payload_subpath.strip():
            raise ValueError("A3_visual_hull_init requires template_payload_subpath")
        cmd = [
            sys.executable,
            str(remote_code_dir / "tools" / "preflight_visual_hull_init.py"),
            "--scene-dir",
            str(scene_dir),
            "--template-payload",
            str(_remote_data_path(cfg.template_payload_subpath)),
            "--output-dir",
            str(output_dir / "A3_visual_hull_init"),
            "--view-indices",
            cfg.view_indices,
            "--target-size",
            str(int(cfg.target_size)),
            "--grid-resolution",
            str(int(cfg.grid_resolution)),
            "--support-threshold",
            str(int(cfg.support_threshold)),
            "--overwrite",
        ]
    else:
        if not cfg.template_payload_subpath.strip():
            raise ValueError("B0_surface_tokens requires template_payload_subpath")
        cmd = [
            sys.executable,
            str(remote_code_dir / "tools" / "optimize_surface_token_backend_b0.py"),
            "--scene-dir",
            str(scene_dir),
            "--template-payload",
            str(_remote_data_path(cfg.template_payload_subpath)),
            "--output-dir",
            str(output_dir / "B0_surface_tokens"),
            "--view-indices",
            cfg.view_indices,
            "--target-size",
            str(int(cfg.target_size)),
            "--steps",
            str(int(cfg.steps)),
            "--token-grid",
            str(int(cfg.token_grid)),
            "--token-hidden",
            str(int(cfg.token_hidden)),
            "--overwrite",
        ]

    result = subprocess.run(cmd, cwd=str(remote_code_dir), check=False, capture_output=True, text=True)
    summary = {
        **run_meta,
        "status": "completed" if result.returncode == 0 else "failed",
        "returncode": int(result.returncode),
        "cmd": cmd,
        "stdout_tail": result.stdout[-8000:],
        "stderr_tail": result.stderr[-8000:],
        "output_subdir": output_dir.relative_to(Path(str(REMOTE_OUTPUT_DIR))).as_posix(),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (output_dir / "research_preflight_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    output_volume.commit()
    if result.returncode != 0:
        raise RuntimeError(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


@app.function(
    image=CPU_RESEARCH_IMAGE,
    cpu=2.0,
    memory=16 * 1024,
    timeout=30 * 60,
    volumes={
        REMOTE_DATA_DIR.as_posix(): data_volume,
        REMOTE_OUTPUT_DIR.as_posix(): output_volume,
    },
)
def run_remote_surface_research_cpu(cfg_json: str, registry_status_json: str) -> dict:
    return _run_surface_research_impl(cfg_json, registry_status_json)


@app.function(
    image=RESEARCH_IMAGE,
    gpu=GPU_SPEC,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SEC,
    volumes={
        REMOTE_DATA_DIR.as_posix(): data_volume,
        REMOTE_OUTPUT_DIR.as_posix(): output_volume,
    },
)
def run_remote_surface_research_gpu(cfg_json: str, registry_status_json: str) -> dict:
    return _run_surface_research_impl(cfg_json, registry_status_json)


@app.local_entrypoint()
def upload_research_assets(
    local_scene_dir: str,
    remote_scene_subdir: str,
    local_template_payload: str = "",
    remote_template_subpath: str = "",
) -> None:
    registry_status = _registry_status(DEFAULT_STRICT_GATE_REGISTRY)
    print("[surface-research] upload is research-only and does not unblock formal cloud:")
    print(json.dumps(registry_status, indent=2, ensure_ascii=False))
    sidecar_path = _write_scene_camera_sidecar(Path(local_scene_dir))
    print(f"[surface-research] wrote portable camera sidecar: {sidecar_path}")
    scene_remote = _upload_dir(Path(local_scene_dir), remote_scene_subdir)
    payload_remote = ""
    if local_template_payload.strip():
        if remote_template_subpath.strip():
            payload_remote = _upload_file(Path(local_template_payload), remote_template_subpath)
        else:
            payload_remote = _upload_file(
                Path(local_template_payload),
                f"{_normalize_subpath(remote_scene_subdir)}_assets/{Path(local_template_payload).name}",
            )
    print("[surface-research] uploaded research assets:")
    print(
        json.dumps(
            {
                "research_only": True,
                "scene_subdir": scene_remote,
                "template_payload_subpath": payload_remote,
                "strict_candidate_passes": registry_status["strict_candidate_passes"],
                "strict_teacher_passes": registry_status["strict_teacher_passes"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


@app.local_entrypoint()
def run_research(
    lane: str,
    scene_subdir: str,
    output_subdir: str,
    template_payload_subpath: str = "",
    view_indices: str = "0,10,20,30,40,50",
    target_size: int = 96,
    steps: int = 20,
    token_grid: int = 5,
    token_hidden: int = 64,
    methods: str = "A1_neural_sdf,A2_gaussian_surface,A3_visual_hull_init",
    expected_gpu: str = "",
    grid_resolution: int = 56,
    support_threshold: int = 4,
    download_local_dir: str = "",
) -> None:
    registry_status = _registry_status(DEFAULT_STRICT_GATE_REGISTRY)
    if registry_status["formal_cloud_allowed"]:
        print("[surface-research] local strict gate is green, but this entrypoint remains research-only.")
    else:
        print("[surface-research] formal VGGT cloud train/infer/export remains blocked:")
        for reason in registry_status["reasons"]:
            print(f"- {reason}")
    if expected_gpu.strip() and expected_gpu.strip() != GPU_SPEC:
        print(
            "[surface-research] WARNING: expected_gpu differs from import-time GPU spec. "
            f"expected={expected_gpu!r} actual={GPU_SPEC!r}. Set VGGT_MODAL_RESEARCH_GPU before launch.",
            flush=True,
        )
    cfg = ResearchConfig(
        lane=lane,
        scene_subdir=_normalize_subpath(scene_subdir),
        output_subdir=_normalize_subpath(output_subdir),
        template_payload_subpath=_normalize_subpath(template_payload_subpath) if template_payload_subpath.strip() else "",
        view_indices=view_indices,
        target_size=int(target_size),
        steps=int(steps),
        token_grid=int(token_grid),
        token_hidden=int(token_hidden),
        methods=",".join(_split_csv(methods)),
        expected_gpu=expected_gpu or GPU_SPEC,
        grid_resolution=int(grid_resolution),
        support_threshold=int(support_threshold),
    )
    print("[surface-research] launch config:")
    print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
    if cfg.lane in {"ping_scene", "A_readiness", "A3_visual_hull_init"}:
        summary = run_remote_surface_research_cpu.remote(cfg.to_json(), json.dumps(registry_status, ensure_ascii=False))
    else:
        summary = run_remote_surface_research_gpu.remote(cfg.to_json(), json.dumps(registry_status, ensure_ascii=False))
    print("[surface-research] remote summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if download_local_dir.strip():
        local_dir = Path(download_local_dir).expanduser().resolve()
    else:
        local_dir = REPO_ROOT / "output" / "surface_research_preflight" / Path(summary["output_subdir"]).name
    _download_volume_dir(summary["output_subdir"], local_dir)
    print(f"[surface-research] downloaded artifacts to {local_dir}")


@app.local_entrypoint()
def download_run(remote_output_subdir: str, local_output_dir: str) -> None:
    local_dir = Path(local_output_dir).expanduser().resolve()
    _download_volume_dir(remote_output_subdir, local_dir)
    print(f"[surface-research] downloaded artifacts to {local_dir}")
