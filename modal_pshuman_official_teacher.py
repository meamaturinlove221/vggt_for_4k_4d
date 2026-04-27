from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
import zipfile
from collections import deque
from pathlib import Path, PurePosixPath
import re

import modal


REMOTE_DATA_DIR = PurePosixPath("/mnt/data")
REMOTE_OUTPUT_DIR = PurePosixPath("/mnt/out")
REMOTE_CACHE_DIR = PurePosixPath("/mnt/cache")

APP_NAME = os.environ.get("VGGT_MODAL_PSHUMAN_OFFICIAL_APP_NAME", "vggt-pshuman-official-teacher")
DATA_VOLUME_NAME = os.environ.get("VGGT_MODAL_DATA_VOLUME", "vggt-4k4d-data")
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_MODAL_OUTPUT_VOLUME", "vggt-4k4d-output")
CACHE_VOLUME_NAME = os.environ.get("VGGT_MODAL_PSHUMAN_OFFICIAL_CACHE_VOLUME", "vggt-pshuman-official-cache")
GPU_SPEC = os.environ.get("VGGT_MODAL_PSHUMAN_OFFICIAL_GPU", "A100-80GB")
CPU_COUNT = float(os.environ.get("VGGT_MODAL_PSHUMAN_OFFICIAL_CPU", "16"))
MEMORY_MB = int(os.environ.get("VGGT_MODAL_PSHUMAN_OFFICIAL_MEMORY_MB", str(96 * 1024)))
TIMEOUT_SEC = int(os.environ.get("VGGT_MODAL_PSHUMAN_OFFICIAL_TIMEOUT_SEC", str(6 * 60 * 60)))

DEFAULT_SCENE_SUBDIR = (
    "4k4d_preprocessed_scene_variants/"
    "0012_11_frame0000_6views_sparseproto_headshoulder_crop"
)
DEFAULT_LOCAL_SCENE_DIR = f"output/{DEFAULT_SCENE_SUBDIR}"
DEFAULT_REMOTE_SCENE_SUBDIR = f"pshuman_official_teacher_smoke/{DEFAULT_SCENE_SUBDIR}"
DEFAULT_OUTPUT_SUBDIR = "detail_normal_refiner_20260426/pshuman_official_teacher_cam30"
DEFAULT_DOWNLOAD_LOCAL_DIR = f"output/{DEFAULT_OUTPUT_SUBDIR}"
DEFAULT_IMAGE_NAME = "30_src_cam30.png"

PSHUMAN_REPO = "https://github.com/pengHTYX/PSHuman"
PSHUMAN_REF = os.environ.get("VGGT_MODAL_PSHUMAN_OFFICIAL_REF", "main")
PSHUMAN_SOURCE_ZIP_URL = os.environ.get(
    "VGGT_MODAL_PSHUMAN_OFFICIAL_SOURCE_ZIP_URL",
    f"https://codeload.github.com/pengHTYX/PSHuman/zip/refs/heads/{PSHUMAN_REF}",
)
PSHUMAN_REQUIREMENTS_URL = (
    "https://raw.githubusercontent.com/pengHTYX/PSHuman/main/requirements.txt"
)
PSHUMAN_SMPL_ASSET_REPO = os.environ.get(
    "VGGT_MODAL_PSHUMAN_SMPL_ASSET_REPO",
    "fffiloni/PSHuman-SMPL-related",
)
MIN_SOURCE_ZIP_BYTES = 100_000
PSHUMAN_REQUIRED_SOURCE_FILES = [
    "inference.py",
    "mvdiffusion/models_unclip/unet_mv2d_condition.py",
    "mvdiffusion/pipelines/pipeline_mvdiffusion_unclip.py",
]


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "build-essential",
        "ca-certificates",
        "curl",
        "ffmpeg",
        "git",
        "libegl1",
        "libgl1",
        "libglib2.0-0",
        "libgomp1",
        "libsm6",
        "libxext6",
        "libxrender1",
        "ninja-build",
        "wget",
    )
    .pip_install(
        "Pillow==10.2.0",
        "huggingface_hub==0.24.5",
        "numpy==1.26.3",
        "omegaconf==2.3.0",
        "packaging",
        "requests==2.32.3",
        "wheel",
    )
)

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)


def _norm(value: str) -> str:
    value = (value or "").replace("\\", "/").strip("/")
    if not value:
        raise ValueError("empty subpath")
    return value


def _upload_dir(local_dir: Path, remote_subdir: str) -> str:
    local_dir = local_dir.expanduser().resolve()
    if not local_dir.is_dir():
        raise FileNotFoundError(f"Local scene directory not found: {local_dir}")
    remote_subdir = _norm(remote_subdir)
    with data_volume.batch_upload(force=True) as batch:
        for path in local_dir.rglob("*"):
            if path.is_file():
                batch.put_file(str(path), f"{remote_subdir}/{path.relative_to(local_dir).as_posix()}")
    return remote_subdir


def _download_dir(remote_subdir: str, local_dir: Path) -> None:
    remote_subdir = _norm(remote_subdir)
    local_dir = local_dir.expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    prefix = Path(remote_subdir)
    files_downloaded = 0
    for entry in output_volume.listdir(remote_subdir, recursive=True):
        rel_path = Path(entry.path)
        try:
            rel_path = rel_path.relative_to(prefix)
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
                with tmp_path.open("wb") as file_obj:
                    output_volume.read_file_into_fileobj(entry.path, file_obj)
                tmp_path.replace(dest_path)
                files_downloaded += 1
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001 - Modal downloads can fail transiently.
                last_error = exc
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                time.sleep(min(2 ** attempt, 20))
        if last_error is not None:
            raise RuntimeError(f"Failed to download {entry.path} after retries") from last_error
    print(f"[pshuman] downloaded {files_downloaded} files from {remote_subdir} to {local_dir}")


def _remote_data_path(subdir: str) -> Path:
    return Path(str(REMOTE_DATA_DIR / _norm(subdir)))


def _remote_output_path(subdir: str) -> Path:
    return Path(str(REMOTE_OUTPUT_DIR / _norm(subdir)))


def _run_command(
    cmd: list[str],
    cwd: Path | None,
    log_path: Path,
    env_extra: dict[str, str] | None = None,
    tail_lines: int = 240,
) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    if env_extra:
        env.update(env_extra)

    started = time.time()
    tail = deque(maxlen=tail_lines)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        log_file.write("$ " + " ".join(cmd) + "\n")
        log_file.write(f"cwd={cwd.as_posix() if cwd else os.getcwd()}\n\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
            tail.append(line.rstrip("\n"))
        returncode = proc.wait()

    return {
        "cmd": cmd,
        "cwd": cwd.as_posix() if cwd else None,
        "returncode": int(returncode),
        "seconds": round(time.time() - started, 3),
        "log": log_path.name,
        "stdout_tail": "\n".join(tail),
    }


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return value[:120] or "step"


def _download_url(url: str, dest_path: Path, min_bytes: int) -> dict:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_name(dest_path.name + ".part")
    tmp_path.unlink(missing_ok=True)
    started = time.time()
    request = urllib.request.Request(url, headers={"User-Agent": "vggt-pshuman-official-modal-smoke"})
    bytes_written = 0
    with urllib.request.urlopen(request, timeout=180) as response:
        with tmp_path.open("wb") as file_obj:
            while True:
                chunk = response.read(16 * 1024 * 1024)
                if not chunk:
                    break
                file_obj.write(chunk)
                bytes_written += len(chunk)
                if bytes_written and bytes_written % (256 * 1024 * 1024) < len(chunk):
                    print(f"[pshuman] downloaded {bytes_written / (1024 ** 2):.1f} MiB from {url}")
    size = tmp_path.stat().st_size
    if size < min_bytes:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is too small: {url} -> {size} bytes")
    tmp_path.replace(dest_path)
    return {"url": url, "path": dest_path.as_posix(), "bytes": int(size), "seconds": round(time.time() - started, 3)}


def _missing_pshuman_source_files(source_root: Path) -> list[str]:
    return [rel_path for rel_path in PSHUMAN_REQUIRED_SOURCE_FILES if not (source_root / rel_path).is_file()]


def _ensure_pshuman_source(cache_root: Path, source_zip_url: str) -> tuple[Path, dict]:
    source_root = cache_root / f"PSHuman-{PSHUMAN_REF}"
    marker = source_root / "inference.py"
    cache_invalid = None
    if marker.is_file():
        missing = _missing_pshuman_source_files(source_root)
        if not missing:
            return source_root, {
                "cache_hit": True,
                "path": source_root.as_posix(),
                "required_source_files": PSHUMAN_REQUIRED_SOURCE_FILES,
            }
        cache_invalid = {"missing_required_source_files": missing}
        shutil.rmtree(source_root, ignore_errors=True)

    cache_root.mkdir(parents=True, exist_ok=True)
    zip_path = cache_root / f"PSHuman-{PSHUMAN_REF}.zip"
    if cache_invalid is not None and zip_path.is_file():
        zip_path.unlink()
    download_info = None
    if not zip_path.is_file() or zip_path.stat().st_size < MIN_SOURCE_ZIP_BYTES:
        download_info = _download_url(source_zip_url, zip_path, MIN_SOURCE_ZIP_BYTES)

    tmp_dir = Path(tempfile.mkdtemp(prefix="pshuman_src_"))
    try:
        with zipfile.ZipFile(zip_path) as zip_obj:
            zip_obj.extractall(tmp_dir)
        extracted = next((path for path in tmp_dir.iterdir() if (path / "inference.py").is_file()), None)
        if extracted is None:
            raise RuntimeError(f"PSHuman archive did not contain inference.py: {zip_path}")
        if source_root.exists():
            shutil.rmtree(source_root)
        shutil.copytree(extracted, source_root)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    missing_after = _missing_pshuman_source_files(source_root)
    if missing_after:
        raise RuntimeError(
            "PSHuman source archive is missing required files: "
            + ", ".join(missing_after)
        )

    return source_root, {
        "cache_hit": False,
        "path": source_root.as_posix(),
        "download": download_info,
        "cache_invalid": cache_invalid,
        "required_source_files": PSHUMAN_REQUIRED_SOURCE_FILES,
        "repo": PSHUMAN_REPO,
        "ref": PSHUMAN_REF,
    }


def _prepare_patched_pshuman_model(source_dir: Path, cache_root: Path, model_id: str) -> tuple[Path, dict]:
    from huggingface_hub import snapshot_download

    model_root = cache_root / "patched_hf_models" / _safe_name(model_id)
    model_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    snapshot_path = snapshot_download(
        repo_id=model_id,
        local_dir=model_root,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    model_root = Path(snapshot_path)
    model_index_path = model_root / "model_index.json"
    if not model_index_path.is_file():
        raise FileNotFoundError(f"model_index.json not found after snapshot_download: {model_root}")
    model_index = json.loads(model_index_path.read_text(encoding="utf-8"))
    patched_components = []
    missing_sources = []
    for component, spec in model_index.items():
        if component.startswith("_") or not isinstance(spec, list) or not spec:
            continue
        module_name = spec[0]
        if not isinstance(module_name, str) or not module_name.startswith("mvdiffusion."):
            continue
        source_file = source_dir / (module_name.replace(".", "/") + ".py")
        short_module_name = source_file.stem
        target_file = model_root / component / f"{short_module_name}.py"
        if not source_file.is_file():
            missing_sources.append(source_file.as_posix())
            continue
        spec[0] = short_module_name
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target_file)
        support_files = []
        for support_source in sorted(source_file.parent.glob("*.py")):
            support_target = target_file.parent / support_source.name
            shutil.copy2(support_source, support_target)
            support_files.append(
                {
                    "source": support_source.as_posix(),
                    "target": support_target.as_posix(),
                    "bytes": int(support_target.stat().st_size),
                }
            )
        patched_components.append(
            {
                "component": component,
                "original_module": module_name,
                "patched_module": short_module_name,
                "source": source_file.as_posix(),
                "target": target_file.as_posix(),
                "bytes": int(target_file.stat().st_size),
                "support_files": support_files,
            }
        )
    if missing_sources:
        raise FileNotFoundError("Missing PSHuman custom component source files: " + ", ".join(missing_sources))
    model_index_path.write_text(json.dumps(model_index, ensure_ascii=False, indent=2), encoding="utf-8")
    return model_root, {
        "model_id": model_id,
        "path": model_root.as_posix(),
        "seconds": round(time.time() - started, 3),
        "patched_components": patched_components,
    }


def _select_input_paths(scene_root: Path, image_name: str, mask_name: str) -> tuple[Path, Path | None]:
    images_dir = scene_root / "images"
    masks_dir = scene_root / "masks"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Remote images directory not found: {images_dir}")
    image_path = images_dir / image_name
    if not image_path.is_file():
        available = sorted(path.name for path in images_dir.glob("*") if path.is_file())
        raise FileNotFoundError(f"Requested image not found: {image_path}; available={available}")
    resolved_mask_name = mask_name or image_path.name
    mask_path = masks_dir / resolved_mask_name
    if mask_name and not mask_path.is_file():
        raise FileNotFoundError(f"Requested mask not found: {mask_path}")
    return image_path, mask_path if mask_path.is_file() else None


def _prepare_rgba_input(image_path: Path, mask_path: Path | None, out_path: Path, background: int) -> dict:
    from PIL import Image
    import numpy as np

    rgb = Image.open(image_path).convert("RGB")
    width, height = rgb.size
    if mask_path is not None:
        alpha = Image.open(mask_path).convert("L").resize((width, height), Image.Resampling.NEAREST)
        mask_arr = np.asarray(alpha, dtype=np.uint8) > 127
    else:
        alpha = Image.new("L", (width, height), 255)
        mask_arr = np.ones((height, width), dtype=bool)

    rgb_arr = np.asarray(rgb, dtype=np.uint8)
    bg = np.full_like(rgb_arr, int(max(0, min(255, background))))
    composited = np.where(mask_arr[..., None], rgb_arr, bg)
    rgba = Image.merge("RGBA", (*Image.fromarray(composited, mode="RGB").split(), alpha))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(out_path)
    return {
        "source_image": image_path.name,
        "source_mask": mask_path.name if mask_path is not None else None,
        "prepared_rgba": out_path.name,
        "size_wh": [width, height],
        "mask_used": mask_path is not None,
        "mask_pixels": int(mask_arr.sum()),
        "background": int(background),
    }


def _inspect_assets(source_dir: Path) -> dict:
    relative_paths = [
        "configs/inference-768-6view.yaml",
        "mvdiffusion/data/fixed_prompt_embeds_7view",
        "mvdiffusion/data/six_human_pose",
        "smpl_related/HPS/pixie_data/pixie_model.tar",
        "smpl_related/HPS/pixie_data/SMPLX_NEUTRAL_2020.npz",
        "smpl_related/HPS/pixie_data/SMPL_X_template_FLAME_uv.obj",
        "smpl_related/smpl_vert_segmentation.json",
        "data/HPS/pymaf_data/pretrained_model/PyMAF_model_checkpoint.pt",
        "data/smpl_related/models/smpl/SMPL_NEUTRAL.pkl",
        "data/smpl_related/models/smplx/SMPLX_NEUTRAL.npz",
    ]
    records = []
    missing = []
    for rel_path in relative_paths:
        path = source_dir / rel_path
        exists = path.exists()
        record = {"path": rel_path, "exists": bool(exists)}
        if path.is_file():
            record["bytes"] = int(path.stat().st_size)
        elif path.is_dir():
            record["entries"] = len(list(path.iterdir()))
        if not exists:
            missing.append(rel_path)
        records.append(record)
    return {"records": records, "missing": missing}


def _copytree_merge(src: Path, dst: Path) -> dict:
    if not src.exists():
        return {"src": src.as_posix(), "dst": dst.as_posix(), "copied": False, "reason": "missing_src"}
    dst.mkdir(parents=True, exist_ok=True)
    files = 0
    bytes_total = 0
    for path in src.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(src)
        out_path = dst / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out_path)
        files += 1
        bytes_total += int(out_path.stat().st_size)
    return {
        "src": src.as_posix(),
        "dst": dst.as_posix(),
        "copied": True,
        "files": int(files),
        "bytes": int(bytes_total),
    }


def _copy_file_if_present(src: Path, dst: Path) -> dict:
    if not src.is_file():
        return {"src": src.as_posix(), "dst": dst.as_posix(), "copied": False, "reason": "missing_src"}
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "src": src.as_posix(),
        "dst": dst.as_posix(),
        "copied": True,
        "bytes": int(dst.stat().st_size),
    }


def _ensure_pshuman_smpl_assets(source_dir: Path, cache_root: Path, asset_repo: str = PSHUMAN_SMPL_ASSET_REPO) -> dict:
    """
    Materialize the public PSHuman SMPL/HPS asset snapshot into the layout expected by
    the official PSHuman repository. The source repo intentionally omits these large
    assets, but the public HF Space mirror hosts them separately.
    """
    before = _inspect_assets(source_dir)
    if not before["missing"]:
        return {"asset_repo": asset_repo, "cache_hit": True, "before": before, "after": before, "copies": []}

    from huggingface_hub import snapshot_download

    snapshot_dir = Path(
        snapshot_download(
            repo_id=asset_repo,
            repo_type="model",
            local_dir=(cache_root / "pshuman_smpl_related").as_posix(),
            local_dir_use_symlinks=False,
        )
    )
    copies = [
        _copytree_merge(snapshot_dir / "HPS" / "pixie_data", source_dir / "smpl_related" / "HPS" / "pixie_data"),
        _copytree_merge(snapshot_dir / "HPS" / "pymaf_data", source_dir / "data" / "HPS" / "pymaf_data"),
        _copytree_merge(snapshot_dir / "HPS" / "pare_data", source_dir / "data" / "HPS" / "pare_data"),
        _copytree_merge(snapshot_dir / "HPS" / "hybrik_data", source_dir / "data" / "HPS" / "hybrik_data"),
        _copytree_merge(snapshot_dir / "models", source_dir / "smpl_related" / "models"),
        _copytree_merge(snapshot_dir / "models" / "smpl", source_dir / "data" / "smpl_related" / "models" / "smpl"),
        _copytree_merge(snapshot_dir / "models" / "smplx", source_dir / "data" / "smpl_related" / "models" / "smplx"),
        _copytree_merge(snapshot_dir / "smpl_data", source_dir / "smpl_related" / "smpl_data"),
        _copy_file_if_present(snapshot_dir / "smpl_vert_segmentation.json", source_dir / "smpl_related" / "smpl_vert_segmentation.json"),
    ]
    after = _inspect_assets(source_dir)
    return {
        "asset_repo": asset_repo,
        "cache_hit": False,
        "snapshot_dir": snapshot_dir.as_posix(),
        "before": before,
        "after": after,
        "copies": copies,
    }


def _install_runtime_deps(out_root: Path) -> list[dict]:
    logs_dir = out_root / "install_logs"
    common_packages = [
        "accelerate==1.1.1",
        "attrs==25.3.0",
        "diffusers==0.26.0",
        "einops==0.8.0",
        "icecream==2.1.3",
        "imageio==2.36.0",
        "imageio-ffmpeg==0.5.1",
        "kornia==0.7.4",
        "matplotlib==3.9.2",
        "mediapipe==0.10.18",
        "moviepy==1.0.3",
        "onnxruntime-gpu==1.20.0",
        "open3d==0.18.0",
        "opencv-python-headless==4.10.0.84",
        "peft==0.13.2",
        "pymeshlab==2023.12.post2",
        "PyMatting==1.1.13",
        "rembg==2.0.59",
        "safetensors==0.4.5",
        "scikit-image==0.24.0",
        "scikit-learn==1.5.2",
        "scipy==1.14.1",
        "termcolor==2.5.0",
        "tqdm==4.67.0",
        "transformers==4.46.2",
        "trimesh==4.5.2",
        "yacs==0.1.8",
    ]
    commands = [
        (
            "pip_bootstrap",
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "pip",
                "setuptools==80.9.0",
                "wheel",
                "ninja",
                "packaging",
            ],
        ),
        (
            "torch_xformers_cu121",
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--index-url",
                "https://download.pytorch.org/whl/cu121",
                "torch==2.1.0",
                "torchvision==0.16.0",
                "xformers==0.0.22.post7",
            ],
        ),
        ("pshuman_common_runtime", [sys.executable, "-m", "pip", "install", *common_packages]),
        (
            "torch_scatter_cu121",
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch-scatter==2.1.2",
                "-f",
                "https://data.pyg.org/whl/torch-2.1.0+cu121.html",
            ],
        ),
        (
            "nvdiffrast_git",
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "git+https://github.com/NVlabs/nvdiffrast.git@729261dc64c4241ea36efda84fbf532cc8b425b8",
            ],
        ),
        (
            "pytorch3d_prebuilt_cu121_pyt210",
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "pytorch3d",
                "-f",
                "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/download.html",
            ],
        ),
        (
            "kaolin_cu121_pyt210",
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "kaolin==0.17.0",
                "-f",
                "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu121.html",
            ],
        ),
        (
            "attrs_jsonschema_repair_after_kaolin",
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "attrs==25.3.0",
                "jsonschema==4.25.1",
            ],
        ),
    ]
    fallback_commands = {
        "pytorch3d_prebuilt_cu121_pyt210": (
            "pytorch3d_git_fallback",
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47",
            ],
        ),
    }

    results = []
    install_summary_path = out_root / "pshuman_install_summary.json"
    for index, (label, cmd) in enumerate(commands, start=1):
        result = _run_command(
            cmd,
            cwd=None,
            log_path=logs_dir / f"{index:02d}_{_safe_name(label)}.log",
            env_extra={
                "CUDA_HOME": "/usr/local/cuda",
                "FORCE_CUDA": "1",
                "MAX_JOBS": "8",
            },
        )
        result["label"] = label
        results.append(result)
        install_summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        if result["returncode"] != 0:
            fallback = fallback_commands.get(label)
            if fallback is not None:
                fallback_label, fallback_cmd = fallback
                fallback_result = _run_command(
                    fallback_cmd,
                    cwd=None,
                    log_path=logs_dir / f"{index:02d}_{_safe_name(fallback_label)}.log",
                    env_extra={
                        "CUDA_HOME": "/usr/local/cuda",
                        "FORCE_CUDA": "1",
                        "MAX_JOBS": "8",
                    },
                )
                fallback_result["label"] = fallback_label
                fallback_result["fallback_for"] = label
                results.append(fallback_result)
                install_summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                if fallback_result["returncode"] == 0:
                    continue
                raise RuntimeError(
                    "Dependency install failed at step "
                    f"{index} ({label}) and fallback ({fallback_label}); "
                    f"see install_logs/{result['log']} and install_logs/{fallback_result['log']}"
                )
            raise RuntimeError(f"Dependency install failed at step {index} ({label}); see install_logs/{result['log']}")
    return results


def _import_probe(out_root: Path, source_dir: Path) -> dict:
    probe_code = """
import importlib, json
mods = [
    'torch', 'torchvision', 'diffusers', 'transformers', 'accelerate',
    'pytorch3d', 'nvdiffrast.torch', 'kaolin', 'pymeshlab', 'open3d',
    'rembg', 'kornia', 'trimesh', 'mvdiffusion.pipelines.pipeline_mvdiffusion_unclip',
]
result = {}
for mod in mods:
    try:
        obj = importlib.import_module(mod)
        result[mod] = {'ok': True, 'version': getattr(obj, '__version__', None)}
    except Exception as exc:
        result[mod] = {'ok': False, 'error': repr(exc)}
print(json.dumps(result, indent=2, ensure_ascii=False))
bad = [k for k, v in result.items() if not v['ok']]
raise SystemExit(1 if bad else 0)
""".strip()
    probe_path = out_root / "pshuman_import_probe.py"
    probe_path.write_text(probe_code, encoding="utf-8")
    result = _run_command(
        [sys.executable, probe_path.as_posix()],
        cwd=source_dir,
        log_path=out_root / "pshuman_import_probe.log",
        env_extra={"PYTHONPATH": source_dir.as_posix()},
    )
    return result


def _write_diffusion_export_script(script_path: Path) -> None:
    script_code = r"""
import json
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
from einops import rearrange
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf, open_dict
from PIL import Image
from torch.utils.data import DataLoader

from mvdiffusion.data.single_image_dataset import SingleImageDataset
from mvdiffusion.models_unclip.unet_mv2d_condition import UNetMV2DConditionModel
from mvdiffusion.pipelines.pipeline_mvdiffusion_unclip import StableUnCLIPImg2ImgPipeline
from utils.misc import load_config


def tensor_to_image(tensor):
    array = tensor.detach().float().mul(255).add_(0.5).clamp_(0, 255)
    array = array.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return Image.fromarray(array)


def main():
    config_path = Path(sys.argv[1])
    input_root = Path(sys.argv[2])
    output_root = Path(sys.argv[3])
    num_steps = int(sys.argv[4])
    seed = int(sys.argv[5])
    model_name = sys.argv[6]
    output_root.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(config_path), cli_args=[])
    with open_dict(cfg):
        cfg.pretrained_model_name_or_path = model_name
        cfg.validation_dataset.root_dir = str(input_root)
        cfg.validation_dataset.num_validation_samples = 1
        cfg.validation_dataset.bg_color = "white"
        cfg.validation_dataset.crop_size = 740
        cfg.validation_batch_size = 1
        cfg.dataloader_num_workers = 0
        cfg.with_smpl = False
        cfg.save_mode = "rgb"
        cfg.seed = seed
        cfg.num_views = 7
        cfg.pipe_kwargs.num_views = 7
        cfg.unet_from_pretrained_kwargs.num_views = 7
        cfg.pipe_validation_kwargs.num_inference_steps = num_steps

    set_seed(seed)
    dataset = SingleImageDataset(**cfg.validation_dataset)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    unet = UNetMV2DConditionModel.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=torch.float16,
        **cfg.unet_from_pretrained_kwargs,
    )
    snapshot_root = Path(snapshot_download(cfg.pretrained_model_name_or_path))
    patched_model_root = output_root.parent / "patched_model_snapshot"
    if patched_model_root.exists():
        shutil.rmtree(patched_model_root)
    shutil.copytree(snapshot_root, patched_model_root, symlinks=False)
    model_index_path = patched_model_root / "model_index.json"
    model_index = json.loads(model_index_path.read_text(encoding="utf-8"))
    model_index["unet"] = ["diffusers", "UNet2DConditionModel"]
    model_index_path.write_text(json.dumps(model_index, indent=2, ensure_ascii=False), encoding="utf-8")
    pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(
        patched_model_root.as_posix(),
        unet=unet,
        torch_dtype=torch.float16,
    )
    pipeline.unet.enable_xformers_memory_efficient_attention()
    pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=pipeline.unet.device).manual_seed(seed)

    saved = []
    with torch.no_grad():
        batch = next(iter(loader))
        imgs_in = torch.cat([batch["imgs_in"]] * 2, dim=0)
        num_views = imgs_in.shape[1]
        imgs_in = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W").to("cuda")

        normal_prompt_embeddings = batch["normal_prompt_embeddings"]
        color_prompt_embeddings = batch["color_prompt_embeddings"]
        prompt_embeddings = torch.cat([normal_prompt_embeddings, color_prompt_embeddings], dim=0)
        prompt_embeddings = rearrange(prompt_embeddings, "B Nv N C -> (B Nv) N C").to("cuda")

        with torch.autocast("cuda"):
            result = pipeline(
                imgs_in,
                None,
                prompt_embeds=prompt_embeddings,
                dino_feature=None,
                smpl_in=None,
                generator=generator,
                guidance_scale=float(cfg.validation_guidance_scales),
                output_type="pt",
                num_images_per_prompt=1,
                **cfg.pipe_validation_kwargs,
            )

        out = result.images
        bsz = out.shape[0] // 2
        normals_pred = out[:bsz]
        colors_pred = out[bsz:]
        if num_views >= 7:
            back = F.interpolate(normals_pred[6].unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False).squeeze(0)
            normals_pred[0, :, :256, 256:512] = back

        for view_idx in range(num_views):
            normal_path = output_root / f"pshuman_normal_view{view_idx:02d}.png"
            color_path = output_root / f"pshuman_color_view{view_idx:02d}.png"
            input_path = output_root / f"pshuman_input_view{view_idx:02d}.png"
            tensor_to_image(normals_pred[view_idx]).save(normal_path)
            tensor_to_image(colors_pred[view_idx]).save(color_path)
            tensor_to_image(imgs_in[view_idx].detach().cpu()).save(input_path)
            saved.extend([normal_path.name, color_path.name, input_path.name])

    summary = {
        "ok": True,
        "config_path": str(config_path),
        "input_root": str(input_root),
        "model_name": model_name,
        "num_inference_steps": num_steps,
        "seed": seed,
        "num_views": int(num_views),
        "files": saved,
    }
    (output_root / "pshuman_diffusion_export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
""".strip()
    script_path.write_text(script_code, encoding="utf-8")


def _mesh_record(path: Path) -> dict:
    record = {"path": path.name, "bytes": int(path.stat().st_size), "suffix": path.suffix.lower()}
    if path.suffix.lower() == ".obj":
        vertices = 0
        faces = 0
        with path.open("r", encoding="utf-8", errors="ignore") as file_obj:
            for line in file_obj:
                if line.startswith("v "):
                    vertices += 1
                elif line.startswith("f "):
                    faces += 1
        record.update({"vertices": vertices, "faces": faces})
    elif path.suffix.lower() == ".ply":
        with path.open("rb") as file_obj:
            header = file_obj.read(4096).decode("utf-8", errors="ignore")
        for line in header.splitlines():
            if line.startswith("element vertex "):
                record["vertices"] = int(line.split()[-1])
            elif line.startswith("element face "):
                record["faces"] = int(line.split()[-1])
    return record


def _copy_artifacts(stage_root: Path, out_root: Path) -> list[dict]:
    artifact_exts = {".obj", ".ply", ".glb", ".mp4"}
    records = []
    for path in sorted(stage_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in artifact_exts:
            continue
        dest = out_root / path.name
        if dest.exists():
            dest = out_root / f"{path.stem}_{len(records):02d}{path.suffix}"
        shutil.copy2(path, dest)
        records.append(_mesh_record(dest))
    return records


@app.function(
    image=image,
    gpu=GPU_SPEC,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SEC,
    volumes={
        REMOTE_OUTPUT_DIR.as_posix(): output_volume,
        REMOTE_CACHE_DIR.as_posix(): cache_volume,
    },
)
def probe_pshuman_official_remote(
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    source_zip_url: str = PSHUMAN_SOURCE_ZIP_URL,
    install_deps: bool = False,
) -> dict:
    out_root = _remote_output_path(output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "ok": False,
        "mode": "probe",
        "output_subdir": _norm(output_subdir),
        "pshuman_repo": PSHUMAN_REPO,
        "pshuman_ref": PSHUMAN_REF,
        "source_zip_url": source_zip_url,
        "requirements_url": PSHUMAN_REQUIREMENTS_URL,
        "gpu": GPU_SPEC,
        "started_at_unix": time.time(),
    }
    try:
        summary["nvidia_smi"] = _run_command(
            ["nvidia-smi"],
            cwd=None,
            log_path=out_root / "nvidia_smi.log",
        )
        cache_root = Path(str(REMOTE_CACHE_DIR))
        source_dir, source_info = _ensure_pshuman_source(cache_root, source_zip_url)
        summary["pshuman_source"] = source_info
        summary["asset_materialization"] = _ensure_pshuman_smpl_assets(source_dir, cache_root)
        summary["assets"] = _inspect_assets(source_dir)
        if install_deps:
            summary["install"] = _install_runtime_deps(out_root)
            summary["import_probe"] = _import_probe(out_root, source_dir)
            if summary["import_probe"]["returncode"] != 0:
                raise RuntimeError("Import probe failed after dependency install")
        cache_volume.commit()
        summary["ok"] = True
        summary["finished_at_unix"] = time.time()
    except Exception as exc:
        summary["ok"] = False
        summary["error"] = repr(exc)
        summary["traceback"] = traceback.format_exc()
        (out_root / "pshuman_official_blocker.txt").write_text(
            "PSHuman official probe did not complete.\n\n"
            f"Error: {repr(exc)}\n\n"
            f"Traceback:\n{summary['traceback']}\n",
            encoding="utf-8",
        )
    finally:
        (out_root / "pshuman_official_teacher_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        output_volume.commit()
    return summary


@app.function(
    image=image,
    gpu=GPU_SPEC,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SEC,
    volumes={
        REMOTE_DATA_DIR.as_posix(): data_volume,
        REMOTE_OUTPUT_DIR.as_posix(): output_volume,
        REMOTE_CACHE_DIR.as_posix(): cache_volume,
    },
)
def run_pshuman_official_remote(
    scene_subdir: str,
    output_subdir: str,
    image_name: str = DEFAULT_IMAGE_NAME,
    mask_name: str = "",
    config_name: str = "configs/inference-768-6view.yaml",
    source_zip_url: str = PSHUMAN_SOURCE_ZIP_URL,
    install_deps: bool = True,
    run_import_probe: bool = True,
    num_inference_steps: int = 40,
    recon_iters: int = 700,
    color_iters: int = 200,
    recon_resolution: int = 1024,
    background: int = 255,
) -> dict:
    out_root = _remote_output_path(output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "ok": False,
        "mode": "run",
        "scene_subdir": _norm(scene_subdir),
        "output_subdir": _norm(output_subdir),
        "image_name": image_name,
        "mask_name": mask_name,
        "config_name": config_name,
        "pshuman_repo": PSHUMAN_REPO,
        "pshuman_ref": PSHUMAN_REF,
        "source_zip_url": source_zip_url,
        "requirements_url": PSHUMAN_REQUIREMENTS_URL,
        "gpu": GPU_SPEC,
        "official_route_notes": [
            "Uses official pengHTYX/PSHuman source, not the queued HF Space API.",
            "The 768 6-view config is designed for high-memory GPUs; this smoke requests A100-80GB.",
            "Mesh reconstruction imports PIXIE/SMPLX/HPS assets in addition to diffusion/runtime packages.",
        ],
        "started_at_unix": time.time(),
    }
    try:
        summary["nvidia_smi"] = _run_command(
            ["nvidia-smi"],
            cwd=None,
            log_path=out_root / "nvidia_smi.log",
        )

        scene_root = _remote_data_path(scene_subdir)
        image_path, mask_path = _select_input_paths(scene_root, image_name, mask_name)
        stage_root = Path(tempfile.mkdtemp(prefix="pshuman_official_teacher_"))
        input_root = stage_root / "input_images"
        input_path = input_root / f"{Path(image_name).stem}.png"
        summary["input"] = _prepare_rgba_input(image_path, mask_path, input_path, background=background)
        shutil.copy2(input_path, out_root / input_path.name)

        cache_root = Path(str(REMOTE_CACHE_DIR))
        source_dir, source_info = _ensure_pshuman_source(cache_root, source_zip_url)
        summary["pshuman_source"] = source_info
        summary["asset_materialization"] = _ensure_pshuman_smpl_assets(source_dir, cache_root)
        summary["assets"] = _inspect_assets(source_dir)
        local_model_path, local_model_info = _prepare_patched_pshuman_model(
            source_dir,
            cache_root,
            "pengHTYX/PSHuman_Unclip_768_6views",
        )
        summary["patched_model"] = local_model_info
        cache_volume.commit()

        config_path = source_dir / config_name
        if not config_path.is_file():
            candidates = sorted(source_dir.glob("configs/*inference*6view*.yaml"))
            if candidates:
                config_path = candidates[0]
            else:
                raise FileNotFoundError(f"PSHuman config not found: {source_dir / config_name}")
        shutil.copy2(config_path, out_root / f"input_{config_path.name}")

        tmp_link = source_dir / "tmp"
        if not tmp_link.exists():
            tmp_link.symlink_to(Path("/tmp"), target_is_directory=True)
        summary["tmp_path_bridge"] = {
            "path": tmp_link.as_posix(),
            "is_symlink": bool(tmp_link.is_symlink()),
            "target": os.readlink(tmp_link) if tmp_link.is_symlink() else None,
        }

        if install_deps:
            summary["install"] = _install_runtime_deps(out_root)

        if run_import_probe:
            summary["import_probe"] = _import_probe(out_root, source_dir)
            if summary["import_probe"]["returncode"] != 0:
                raise RuntimeError("Import probe failed; see pshuman_import_probe.log")

        mv_results_dir = stage_root / "mv_results"
        recon_dir = stage_root / "recon"
        cmd = [
            sys.executable,
            "inference.py",
            "--config",
            config_path.as_posix(),
            f"validation_dataset.root_dir={input_root.as_posix()}",
            "validation_dataset.num_validation_samples=1",
            "validation_dataset.bg_color=white",
            f"pretrained_model_name_or_path={local_model_path.as_posix()}",
            "validation_dataset.crop_size=740",
            "with_smpl=false",
            "seed=600",
            "num_views=7",
            "save_mode=rgb",
            "validation_batch_size=1",
            "dataloader_num_workers=0",
            f"save_dir={mv_results_dir.as_posix()}",
            f"recon_opt.res_path={recon_dir.as_posix()}",
            f"recon_opt.iters={int(recon_iters)}",
            f"recon_opt.clr_iters={int(color_iters)}",
            f"recon_opt.resolution={int(recon_resolution)}",
            "recon_opt.gpu_id=0",
            f"pipe_validation_kwargs.num_inference_steps={int(num_inference_steps)}",
        ]
        (out_root / "pshuman_official_command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
        command_result = _run_command(
            cmd,
            cwd=source_dir,
            log_path=out_root / "pshuman_official_command.log",
            env_extra={
                "PYTHONPATH": source_dir.as_posix(),
                "HF_HOME": (cache_root / "huggingface").as_posix(),
                "HUGGINGFACE_HUB_CACHE": (cache_root / "huggingface" / "hub").as_posix(),
                "TORCH_HOME": (cache_root / "torch").as_posix(),
                "CUDA_HOME": "/usr/local/cuda",
                "FORCE_CUDA": "1",
            },
        )
        summary["command"] = command_result
        if command_result["returncode"] != 0:
            raise RuntimeError("PSHuman official inference command failed; see pshuman_official_command.log")

        artifacts = _copy_artifacts(stage_root, out_root)
        summary["artifacts"] = artifacts
        mesh_artifacts = [item for item in artifacts if item["suffix"] in {".obj", ".ply", ".glb"}]
        if not mesh_artifacts:
            raise RuntimeError(f"PSHuman completed but produced no mesh artifact under {stage_root}")

        summary["ok"] = True
        summary["finished_at_unix"] = time.time()
        blocker_path = out_root / "pshuman_official_blocker.txt"
        blocker_path.unlink(missing_ok=True)
    except Exception as exc:
        summary["ok"] = False
        summary["error"] = repr(exc)
        summary["traceback"] = traceback.format_exc()
        (out_root / "pshuman_official_blocker.txt").write_text(
            "PSHuman official/self-hosted mesh teacher smoke did not complete.\n\n"
            f"Error: {repr(exc)}\n\n"
            f"Traceback:\n{summary['traceback']}\n",
            encoding="utf-8",
        )
    finally:
        (out_root / "pshuman_official_teacher_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        output_volume.commit()
    return summary


@app.function(
    image=image,
    gpu=GPU_SPEC,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SEC,
    volumes={
        REMOTE_DATA_DIR.as_posix(): data_volume,
        REMOTE_OUTPUT_DIR.as_posix(): output_volume,
        REMOTE_CACHE_DIR.as_posix(): cache_volume,
    },
)
def run_pshuman_diffusion_remote(
    scene_subdir: str,
    output_subdir: str,
    image_name: str = DEFAULT_IMAGE_NAME,
    mask_name: str = "",
    config_name: str = "configs/inference-768-6view.yaml",
    source_zip_url: str = PSHUMAN_SOURCE_ZIP_URL,
    install_deps: bool = True,
    num_inference_steps: int = 8,
    seed: int = 600,
    background: int = 255,
    model_name: str = "pengHTYX/PSHuman_Unclip_768_6views",
) -> dict:
    out_root = _remote_output_path(output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "ok": False,
        "mode": "diffusion",
        "scene_subdir": _norm(scene_subdir),
        "output_subdir": _norm(output_subdir),
        "image_name": image_name,
        "mask_name": mask_name,
        "config_name": config_name,
        "model_name": model_name,
        "pshuman_repo": PSHUMAN_REPO,
        "pshuman_ref": PSHUMAN_REF,
        "source_zip_url": source_zip_url,
        "gpu": GPU_SPEC,
        "notes": [
            "Exports PSHuman color/normal diffusion outputs only.",
            "This intentionally avoids PIXIE/HPS/SMPLX mesh reconstruction assets.",
            "These normal maps are teacher candidates, not final point-cloud evidence.",
        ],
        "started_at_unix": time.time(),
    }
    try:
        summary["nvidia_smi"] = _run_command(["nvidia-smi"], cwd=None, log_path=out_root / "nvidia_smi.log")
        scene_root = _remote_data_path(scene_subdir)
        image_path, mask_path = _select_input_paths(scene_root, image_name, mask_name)
        stage_root = Path(tempfile.mkdtemp(prefix="pshuman_diffusion_teacher_"))
        input_root = stage_root / "input_images"
        input_path = input_root / f"{Path(image_name).stem}.png"
        summary["input"] = _prepare_rgba_input(image_path, mask_path, input_path, background=background)
        shutil.copy2(input_path, out_root / input_path.name)

        cache_root = Path(str(REMOTE_CACHE_DIR))
        source_dir, source_info = _ensure_pshuman_source(cache_root, source_zip_url)
        summary["pshuman_source"] = source_info
        summary["assets"] = _inspect_assets(source_dir)
        cache_volume.commit()

        config_path = source_dir / config_name
        if not config_path.is_file():
            candidates = sorted(source_dir.glob("configs/*inference*6view*.yaml"))
            if candidates:
                config_path = candidates[0]
            else:
                raise FileNotFoundError(f"PSHuman config not found: {source_dir / config_name}")
        shutil.copy2(config_path, out_root / f"input_{config_path.name}")

        if install_deps:
            summary["install"] = _install_runtime_deps(out_root)

        script_path = out_root / "pshuman_diffusion_export.py"
        _write_diffusion_export_script(script_path)
        export_dir = stage_root / "pshuman_diffusion_export"
        command_result = _run_command(
            [
                sys.executable,
                script_path.as_posix(),
                config_path.as_posix(),
                input_root.as_posix(),
                export_dir.as_posix(),
                str(int(num_inference_steps)),
                str(int(seed)),
                model_name,
            ],
            cwd=source_dir,
            log_path=out_root / "pshuman_diffusion_export.log",
            env_extra={
                "PYTHONPATH": source_dir.as_posix(),
                "HF_HOME": (cache_root / "huggingface").as_posix(),
                "HUGGINGFACE_HUB_CACHE": (cache_root / "huggingface" / "hub").as_posix(),
                "TORCH_HOME": (cache_root / "torch").as_posix(),
            },
            tail_lines=320,
        )
        summary["command"] = command_result
        if command_result["returncode"] != 0:
            raise RuntimeError("PSHuman diffusion export command failed; see pshuman_diffusion_export.log")

        copied = []
        for path in sorted(export_dir.rglob("*")):
            if not path.is_file():
                continue
            dest = out_root / path.name
            shutil.copy2(path, dest)
            copied.append({"path": dest.name, "bytes": int(dest.stat().st_size)})
        summary["artifacts"] = copied
        summary["ok"] = True
        summary["finished_at_unix"] = time.time()
        (out_root / "pshuman_diffusion_blocker.txt").unlink(missing_ok=True)
    except Exception as exc:
        summary["ok"] = False
        summary["error"] = repr(exc)
        summary["traceback"] = traceback.format_exc()
        (out_root / "pshuman_diffusion_blocker.txt").write_text(
            "PSHuman diffusion normal teacher smoke did not complete.\n\n"
            f"Error: {repr(exc)}\n\n"
            f"Traceback:\n{summary['traceback']}\n",
            encoding="utf-8",
        )
    finally:
        (out_root / "pshuman_diffusion_teacher_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        output_volume.commit()
    return summary


@app.local_entrypoint()
def main(
    mode: str = "run",
    local_scene_dir: str = DEFAULT_LOCAL_SCENE_DIR,
    remote_scene_subdir: str = DEFAULT_REMOTE_SCENE_SUBDIR,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    download_local_dir: str = DEFAULT_DOWNLOAD_LOCAL_DIR,
    image_name: str = DEFAULT_IMAGE_NAME,
    mask_name: str = "",
    config_name: str = "configs/inference-768-6view.yaml",
    source_zip_url: str = PSHUMAN_SOURCE_ZIP_URL,
    install_deps: bool = True,
    run_import_probe: bool = True,
    num_inference_steps: int = 40,
    recon_iters: int = 700,
    color_iters: int = 200,
    recon_resolution: int = 1024,
    background: int = 255,
):
    if mode == "probe":
        summary = probe_pshuman_official_remote.remote(
            output_subdir=output_subdir,
            source_zip_url=source_zip_url,
            install_deps=install_deps,
        )
    elif mode == "run":
        remote = _upload_dir(Path(local_scene_dir), remote_scene_subdir)
        summary = run_pshuman_official_remote.remote(
            scene_subdir=remote,
            output_subdir=output_subdir,
            image_name=image_name,
            mask_name=mask_name,
            config_name=config_name,
            source_zip_url=source_zip_url,
            install_deps=install_deps,
            run_import_probe=run_import_probe,
            num_inference_steps=num_inference_steps,
            recon_iters=recon_iters,
            color_iters=color_iters,
            recon_resolution=recon_resolution,
            background=background,
        )
    elif mode == "download":
        summary = {
            "mode": "download",
            "output_subdir": _norm(output_subdir),
        }
    elif mode == "diffusion":
        remote = _upload_dir(Path(local_scene_dir), remote_scene_subdir)
        summary = run_pshuman_diffusion_remote.remote(
            scene_subdir=remote,
            output_subdir=output_subdir,
            image_name=image_name,
            mask_name=mask_name,
            config_name=config_name,
            source_zip_url=source_zip_url,
            install_deps=install_deps,
            num_inference_steps=num_inference_steps,
            seed=600,
            background=background,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if download_local_dir:
        _download_dir(summary["output_subdir"], Path(download_local_dir))
