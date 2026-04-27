from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import urllib.request
import zipfile
from collections import deque
from pathlib import Path, PurePosixPath

import modal


REMOTE_DATA_DIR = PurePosixPath("/mnt/data")
REMOTE_OUTPUT_DIR = PurePosixPath("/mnt/out")
REMOTE_CACHE_DIR = PurePosixPath("/mnt/cache")

APP_NAME = os.environ.get("VGGT_MODAL_LHM_APP_NAME", "vggt-lhm-mesh-teacher")
DATA_VOLUME_NAME = os.environ.get("VGGT_MODAL_DATA_VOLUME", "vggt-4k4d-data")
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_MODAL_OUTPUT_VOLUME", "vggt-4k4d-output")
CACHE_VOLUME_NAME = os.environ.get("VGGT_MODAL_LHM_CACHE_VOLUME", "vggt-lhm-cache")
GPU_SPEC = os.environ.get("VGGT_MODAL_LHM_GPU", "A100-80GB")
CPU_COUNT = float(os.environ.get("VGGT_MODAL_LHM_CPU", "16"))
MEMORY_MB = int(os.environ.get("VGGT_MODAL_LHM_MEMORY_MB", str(96 * 1024)))
TIMEOUT_SEC = int(os.environ.get("VGGT_MODAL_LHM_TIMEOUT_SEC", str(8 * 60 * 60)))

DEFAULT_SCENE_SUBDIR = "4k4d_preprocessed_scene_variants/0012_11_frame0000_6views_sparseproto_headshoulder_crop"
DEFAULT_LOCAL_SCENE_DIR = f"output/{DEFAULT_SCENE_SUBDIR}"
DEFAULT_REMOTE_SCENE_SUBDIR = f"lhm_mesh_teacher/{DEFAULT_SCENE_SUBDIR}"
DEFAULT_OUTPUT_SUBDIR = "detail_normal_refiner_20260427/lhm_mini_mesh_teacher_00_tgt_cam00"
DEFAULT_DOWNLOAD_LOCAL_DIR = f"output/{DEFAULT_OUTPUT_SUBDIR}"
DEFAULT_IMAGE_NAME = "00_tgt_cam00.png"
DEFAULT_MODEL_NAME = "LHM-MINI"

LHM_REPO = "https://github.com/aigc3d/LHM"
LHM_REF = os.environ.get("VGGT_MODAL_LHM_REF", "4f88aaeb3629249fbbddb4d0784a06962d9e1338")
LHM_SOURCE_ZIP_URL = os.environ.get(
    "VGGT_MODAL_LHM_SOURCE_ZIP_URL",
    f"https://codeload.github.com/aigc3d/LHM/zip/{LHM_REF}",
)
MIN_SOURCE_ZIP_BYTES = 1_000_000
LHM_PRIOR_MODEL_URL = os.environ.get(
    "VGGT_MODAL_LHM_PRIOR_MODEL_URL",
    "https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LHM/LHM_prior_model.tar",
)
MIN_PRIOR_TAR_BYTES = 1_000_000


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "build-essential",
        "ca-certificates",
        "clang",
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
    .run_commands("python -m pip install --upgrade pip wheel setuptools ninja packaging")
    .run_commands(
        "python -m pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 "
        "--index-url https://download.pytorch.org/whl/cu121"
    )
    .run_commands(
        "python -m pip install -U xformers==0.0.26.post1 "
        "--index-url https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        "accelerate",
        "basicsr==1.4.2",
        "decord==0.6.0",
        "diffusers==0.32.0",
        "einops",
        "gfpgan==1.3.8",
        "gradio==4.43.0",
        "gsplat==1.4.0",
        "huggingface_hub==0.23.2",
        "imageio==2.34.1",
        "imageio-ffmpeg",
        "jaxtyping==0.2.38",
        "kiui==0.2.14",
        "kornia==0.7.2",
        "loguru==0.7.3",
        "lpips==0.1.4",
        "matplotlib==3.5.3",
        "megfile==4.1.0.post2",
        "modelscope",
        "numpy==1.23.0",
        "omegaconf==2.3.0",
        "onnxruntime",
        "opencv-python-headless",
        "open3d==0.19.0",
        "Pillow==10.4.0",
        "plyfile",
        "pygltflib==1.16.2",
        "pyrender==0.1.45",
        "PyYAML==6.0.1",
        "rembg==2.0.63",
        "Requests==2.32.3",
        "roma",
        "scipy",
        "smplx",
        "taming_transformers_rom1504==0.0.6",
        "timm==1.0.15",
        "transformers==4.41.2",
        "trimesh==4.4.9",
        "typeguard==2.13.3",
        "xatlas==0.0.9",
    )
    .run_commands(
        "python -c \"from pathlib import Path; import site; "
        "site_dirs = site.getsitepackages(); "
        "assert site_dirs, 'no site-packages directory found'; "
        "shim = Path(site_dirs[0]) / 'sitecustomize.py'; "
        "snippet = 'import sys\\\\nimport types\\\\ntry:\\\\n"
        "    import torchvision.transforms.functional as _torchvision_functional\\\\n"
        "    _functional_tensor = types.ModuleType(\\\"torchvision.transforms.functional_tensor\\\")\\\\n"
        "    _functional_tensor.rgb_to_grayscale = _torchvision_functional.rgb_to_grayscale\\\\n"
        "    sys.modules.setdefault(\\\"torchvision.transforms.functional_tensor\\\", _functional_tensor)\\\\n"
        "except Exception:\\\\n"
        "    pass\\\\n'; "
        "old = shim.read_text(encoding='utf-8') if shim.exists() else ''; "
        "shim.write_text(old + '\\\\n' + snippet, encoding='utf-8') "
        "if 'torchvision.transforms.functional_tensor' not in old else None; "
        "print(f'patched torchvision functional_tensor shim at {shim}')\""
    )
    .run_commands("python -m pip install chumpy==0.70 --no-build-isolation")
    .run_commands(
        "FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=8 "
        "python -m pip install --no-build-isolation --no-cache-dir "
        "git+https://github.com/facebookresearch/pytorch3d.git"
    )
    .run_commands(
        "TORCH_CUDA_ARCH_LIST=8.0 python -m pip install --no-build-isolation "
        "git+https://github.com/ashawkey/diff-gaussian-rasterization/"
    )
    .run_commands(
        "TORCH_CUDA_ARCH_LIST=8.0 python -m pip install --no-build-isolation "
        "git+https://github.com/camenduru/simple-knn/"
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
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                tmp_path.unlink(missing_ok=True)
                time.sleep(min(2**attempt, 20))
        if last_error is not None:
            raise RuntimeError(f"Failed to download {entry.path}") from last_error
    print(f"[lhm] downloaded {files_downloaded} files from {remote_subdir} to {local_dir}")


def _remote_data_path(subdir: str) -> Path:
    return Path(str(REMOTE_DATA_DIR / _norm(subdir)))


def _remote_output_path(subdir: str) -> Path:
    return Path(str(REMOTE_OUTPUT_DIR / _norm(subdir)))


def _download_url(url: str, dest_path: Path, min_bytes: int) -> dict:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_name(dest_path.name + ".part")
    tmp_path.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "vggt-lhm-modal"})
    started = time.time()
    bytes_written = 0
    with urllib.request.urlopen(request, timeout=180) as response:
        with tmp_path.open("wb") as file_obj:
            while True:
                chunk = response.read(16 * 1024 * 1024)
                if not chunk:
                    break
                file_obj.write(chunk)
                bytes_written += len(chunk)
                print(f"[lhm] downloaded {bytes_written / (1024 ** 2):.1f} MiB")
    if tmp_path.stat().st_size < min_bytes:
        raise RuntimeError(f"Downloaded source is too small: {tmp_path.stat().st_size} bytes")
    tmp_path.replace(dest_path)
    return {"url": url, "path": dest_path.as_posix(), "bytes": int(dest_path.stat().st_size), "seconds": round(time.time() - started, 3)}


def _ensure_lhm_source(cache_root: Path, source_zip_url: str) -> tuple[Path, dict]:
    source_root = cache_root / f"LHM-{LHM_REF[:12]}"
    marker = source_root / "inference_mesh.sh"
    if marker.is_file():
        return source_root, {"cache_hit": True, "path": source_root.as_posix(), "repo": LHM_REPO, "ref": LHM_REF}
    cache_root.mkdir(parents=True, exist_ok=True)
    zip_path = cache_root / f"LHM-{LHM_REF[:12]}.zip"
    download_info = None
    if not zip_path.is_file() or zip_path.stat().st_size < MIN_SOURCE_ZIP_BYTES:
        download_info = _download_url(source_zip_url, zip_path, MIN_SOURCE_ZIP_BYTES)
    tmp_dir = Path(tempfile.mkdtemp(prefix="lhm_src_"))
    try:
        with zipfile.ZipFile(zip_path) as zip_obj:
            zip_obj.extractall(tmp_dir)
        extracted = next((path for path in tmp_dir.iterdir() if (path / "inference_mesh.sh").is_file()), None)
        if extracted is None:
            raise RuntimeError(f"LHM source archive did not contain inference_mesh.sh: {zip_path}")
        if source_root.exists():
            shutil.rmtree(source_root)
        shutil.copytree(extracted, source_root)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return source_root, {"cache_hit": False, "path": source_root.as_posix(), "repo": LHM_REPO, "ref": LHM_REF, "download": download_info}


def _safe_extract_tar(tar_path: Path, dest_root: Path) -> None:
    dest_root = dest_root.resolve()
    with tarfile.open(tar_path) as tar_obj:
        for member in tar_obj.getmembers():
            target_path = (dest_root / member.name).resolve()
            if not target_path.is_relative_to(dest_root):
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        tar_obj.extractall(dest_root)


def _ensure_lhm_prior_assets(source_root: Path, cache_root: Path, prior_model_url: str) -> dict:
    marker = source_root / "pretrained_models" / "gagatracker" / "vgghead" / "vgg_heads_l.trcd"
    if marker.is_file():
        return {"cache_hit": True, "marker": marker.as_posix(), "url": prior_model_url}
    cache_root.mkdir(parents=True, exist_ok=True)
    tar_path = cache_root / "LHM_prior_model.tar"
    download_info = None
    if not tar_path.is_file() or tar_path.stat().st_size < MIN_PRIOR_TAR_BYTES:
        download_info = _download_url(prior_model_url, tar_path, MIN_PRIOR_TAR_BYTES)
    _safe_extract_tar(tar_path, source_root)
    if not marker.is_file():
        raise RuntimeError(f"LHM prior model archive did not create required marker: {marker}")
    return {
        "cache_hit": False,
        "marker": marker.as_posix(),
        "url": prior_model_url,
        "tar_path": tar_path.as_posix(),
        "download": download_info,
    }


def _patch_lhm_source_for_modal(source_root: Path) -> dict:
    human_lrm_path = source_root / "LHM" / "runners" / "infer" / "human_lrm.py"
    if not human_lrm_path.is_file():
        raise FileNotFoundError(f"LHM human_lrm.py not found: {human_lrm_path}")
    text = human_lrm_path.read_text(encoding="utf-8")
    marker = "VGGT_MODAL_PATCH_PARSINGNET_REMBG_FALLBACK"
    if marker in text:
        return {"patched": False, "path": human_lrm_path.as_posix(), "reason": "already_patched"}
    old = (
        "    @torch.no_grad()\n"
        "    def parsing(self, img_path):\n"
        "\n"
        "        parsing_out = self.parsingnet(img_path=img_path, bbox=None)\n"
        "\n"
        "        alpha = (parsing_out.masks * 255).astype(np.uint8)\n"
        "\n"
        "        return alpha\n"
    )
    new = (
        "    @torch.no_grad()\n"
        "    def parsing(self, img_path):\n"
        "        # VGGT_MODAL_PATCH_PARSINGNET_REMBG_FALLBACK\n"
        "        if self.parsingnet is not None:\n"
        "            parsing_out = self.parsingnet(img_path=img_path, bbox=None)\n"
        "            alpha = (parsing_out.masks * 255).astype(np.uint8)\n"
        "            return alpha\n"
        "        img_np = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)\n"
        "        if img_np is None:\n"
        "            raise FileNotFoundError(img_path)\n"
        "        if img_np.ndim == 3 and img_np.shape[-1] == 4:\n"
        "            return img_np[..., 3]\n"
        "        remove_np = remove(img_np)\n"
        "        return remove_np[..., 3]\n"
    )
    if old not in text:
        raise RuntimeError(f"Could not locate LHM parsing() block to patch: {human_lrm_path}")
    human_lrm_path.write_text(text.replace(old, new), encoding="utf-8")
    return {"patched": True, "path": human_lrm_path.as_posix(), "patch": marker}


def _select_input_paths(scene_root: Path, image_name: str, mask_name: str) -> tuple[Path, Path | None]:
    images_dir = scene_root / "images"
    masks_dir = scene_root / "masks"
    if image_name:
        image_path = images_dir / image_name
        if not image_path.is_file():
            available = sorted(path.name for path in images_dir.glob("*") if path.is_file())
            raise FileNotFoundError(f"Requested image not found: {image_path}; available={available}")
    else:
        candidates = sorted(path for path in images_dir.glob("*") if path.is_file())
        if not candidates:
            raise FileNotFoundError(f"No images under {images_dir}")
        image_path = candidates[0]
    resolved_mask_name = mask_name or image_path.name
    mask_path = masks_dir / resolved_mask_name
    return image_path, mask_path if mask_path.is_file() else None


def _prepare_lhm_input(image_path: Path, mask_path: Path | None, dest_path: Path, background: int, portrait_pad: float) -> dict:
    from PIL import Image
    import numpy as np

    image_obj = Image.open(image_path).convert("RGB")
    image_arr = np.asarray(image_obj, dtype=np.uint8)
    height, width = image_arr.shape[:2]
    if mask_path is not None:
        mask_obj = Image.open(mask_path).convert("L").resize((width, height), Image.Resampling.NEAREST)
        mask = np.asarray(mask_obj, dtype=np.uint8) > 127
    else:
        mask = np.ones((height, width), dtype=bool)
    bg = np.full_like(image_arr, int(max(0, min(255, background))))
    image_arr = np.where(mask[..., None], image_arr, bg)
    ys, xs = np.nonzero(mask)
    if xs.size:
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
    else:
        x0, y0, x1, y1 = 0, 0, width, height
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    pad_x = int(round(box_w * max(0.0, portrait_pad)))
    pad_y = int(round(box_h * max(0.0, portrait_pad)))
    x0 = max(0, x0 - pad_x)
    x1 = min(width, x1 + pad_x)
    y0 = max(0, y0 - pad_y)
    y1 = min(height, y1 + pad_y)
    cropped = image_arr[y0:y1, x0:x1]
    crop_h, crop_w = cropped.shape[:2]
    target_ratio = 5.0 / 3.0
    target_h = max(crop_h, int(round(crop_w * target_ratio)))
    target_w = max(crop_w, int(round(target_h / target_ratio)))
    canvas = np.full((target_h, target_w, 3), int(max(0, min(255, background))), dtype=np.uint8)
    off_x = (target_w - crop_w) // 2
    off_y = (target_h - crop_h) // 2
    canvas[off_y : off_y + crop_h, off_x : off_x + crop_w] = cropped
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="RGB").save(dest_path)
    return {
        "source_image": image_path.name,
        "source_mask": mask_path.name if mask_path is not None else "",
        "source_size_wh": [int(width), int(height)],
        "mask_pixels": int(mask.sum()),
        "crop_xyxy": [x0, y0, x1, y1],
        "prepared_image": dest_path.name,
        "prepared_size_wh": [int(target_w), int(target_h)],
        "portrait_pad": float(portrait_pad),
        "background": int(max(0, min(255, background))),
    }


def _run_command(cmd: list[str], cwd: Path, log_path: Path, env_extra: dict[str, str] | None = None, tail_lines: int = 260) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"})
    if env_extra:
        env.update(env_extra)
    started = time.time()
    tail = deque(maxlen=tail_lines)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        log_file.write("$ " + " ".join(cmd) + "\n")
        log_file.write(f"cwd={cwd.as_posix()}\n\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=cwd.as_posix(),
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
        "cwd": cwd.as_posix(),
        "returncode": int(returncode),
        "seconds": round(time.time() - started, 3),
        "log": log_path.name,
        "stdout_tail": "\n".join(tail),
    }


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
def run_lhm_mesh_remote(
    scene_subdir: str,
    output_subdir: str,
    image_name: str = DEFAULT_IMAGE_NAME,
    mask_name: str = "",
    model_name: str = DEFAULT_MODEL_NAME,
    background: int = 255,
    portrait_pad: float = 0.12,
    source_zip_url: str = LHM_SOURCE_ZIP_URL,
    prior_model_url: str = LHM_PRIOR_MODEL_URL,
) -> dict:
    out_root = _remote_output_path(output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "ok": False,
        "scene_subdir": _norm(scene_subdir),
        "output_subdir": _norm(output_subdir),
        "image_name": image_name,
        "mask_name": mask_name,
        "model_name": model_name,
        "lhm_repo": LHM_REPO,
        "lhm_ref": LHM_REF,
        "source_zip_url": source_zip_url,
        "started_at_unix": time.time(),
    }
    try:
        scene_root = _remote_data_path(scene_subdir)
        image_path, mask_path = _select_input_paths(scene_root, image_name, mask_name)
        stage_root = Path(tempfile.mkdtemp(prefix="lhm_mesh_teacher_"))
        input_dir = stage_root / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        prepared_image = input_dir / image_path.name
        summary["input"] = _prepare_lhm_input(image_path, mask_path, prepared_image, background, portrait_pad)
        shutil.copy2(prepared_image, out_root / prepared_image.name)

        lhm_source, source_info = _ensure_lhm_source(Path(str(REMOTE_CACHE_DIR)), source_zip_url)
        summary["lhm_source"] = source_info
        (lhm_source / "pretrained_models" / "dense_sample_points").mkdir(parents=True, exist_ok=True)
        summary["lhm_prior_assets"] = _ensure_lhm_prior_assets(
            lhm_source,
            Path(str(REMOTE_CACHE_DIR)),
            prior_model_url,
        )
        summary["lhm_modal_source_patch"] = _patch_lhm_source_for_modal(lhm_source)
        cache_volume.commit()

        log_path = out_root / "lhm_command.log"
        cmd = [
            sys.executable,
            "-m",
            "LHM.launch",
            "infer.human_lrm",
            "--infer",
            "./configs/infer-gradio.yaml",
            f"model_name={model_name}",
            f"image_input={input_dir.as_posix()}",
            "export_mesh=True",
            "motion_seqs_dir=None",
            "motion_img_dir=None",
            "vis_motion=True",
            "motion_img_need_mask=True",
            "render_fps=25",
            "motion_video_read_fps=6",
        ]
        command = _run_command(
            cmd,
            cwd=lhm_source,
            log_path=log_path,
            env_extra={
                "PYTHONPATH": lhm_source.as_posix(),
                "HF_HOME": (Path(str(REMOTE_CACHE_DIR)) / "hf_home").as_posix(),
                "MODELSCOPE_CACHE": (Path(str(REMOTE_CACHE_DIR)) / "modelscope").as_posix(),
            },
        )
        summary["command"] = command
        mesh_paths = sorted([*lhm_source.rglob("*.ply"), *lhm_source.rglob("*.obj")])
        mesh_paths = [path for path in mesh_paths if "meshs" in path.as_posix() or "meshes" in path.as_posix()]
        if not mesh_paths:
            raise RuntimeError(f"LHM command produced no mesh PLY/OBJ under {lhm_source}")
        records = []
        for mesh_path in mesh_paths:
            rel_name = mesh_path.name
            dest_path = out_root / rel_name
            shutil.copy2(mesh_path, dest_path)
            records.append({"source": mesh_path.as_posix(), "output": dest_path.name, "bytes": int(dest_path.stat().st_size)})
        summary["ok"] = command["returncode"] == 0
        if command["returncode"] != 0:
            raise RuntimeError(f"LHM command failed with returncode {command['returncode']}")
        summary["meshes"] = records
        summary["finished_at_unix"] = time.time()
        blocker = out_root / "lhm_blocker.txt"
        blocker.unlink(missing_ok=True)
    except Exception as exc:
        summary["ok"] = False
        summary["error"] = repr(exc)
        summary["traceback"] = traceback.format_exc()
        (out_root / "lhm_blocker.txt").write_text(
            "LHM mesh teacher did not complete.\n\n"
            f"Error: {repr(exc)}\n\n"
            f"Traceback:\n{summary['traceback']}\n",
            encoding="utf-8",
        )
    finally:
        (out_root / "lhm_mesh_teacher_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        output_volume.commit()
    return summary


@app.local_entrypoint()
def run_from_local(
    local_scene_dir: str = DEFAULT_LOCAL_SCENE_DIR,
    remote_scene_subdir: str = DEFAULT_REMOTE_SCENE_SUBDIR,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    download_local_dir: str = DEFAULT_DOWNLOAD_LOCAL_DIR,
    image_name: str = DEFAULT_IMAGE_NAME,
    mask_name: str = "",
    model_name: str = DEFAULT_MODEL_NAME,
    background: int = 255,
    portrait_pad: float = 0.12,
    source_zip_url: str = LHM_SOURCE_ZIP_URL,
    prior_model_url: str = LHM_PRIOR_MODEL_URL,
):
    remote = _upload_dir(Path(local_scene_dir), remote_scene_subdir)
    summary = run_lhm_mesh_remote.remote(
        remote,
        output_subdir,
        image_name=image_name,
        mask_name=mask_name,
        model_name=model_name,
        background=background,
        portrait_pad=portrait_pad,
        source_zip_url=source_zip_url,
        prior_model_url=prior_model_url,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if download_local_dir:
        _download_dir(summary["output_subdir"], Path(download_local_dir))
