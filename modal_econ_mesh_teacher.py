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
from pathlib import Path, PurePosixPath

import modal


REMOTE_DATA_DIR = PurePosixPath("/mnt/data")
REMOTE_OUTPUT_DIR = PurePosixPath("/mnt/out")
REMOTE_CACHE_DIR = PurePosixPath("/mnt/cache")

APP_NAME = os.environ.get("VGGT_MODAL_ECON_APP_NAME", "vggt-econ-mesh-teacher")
DATA_VOLUME_NAME = os.environ.get("VGGT_MODAL_DATA_VOLUME", "vggt-4k4d-data")
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_MODAL_OUTPUT_VOLUME", "vggt-4k4d-output")
CACHE_VOLUME_NAME = os.environ.get("VGGT_MODAL_ECON_CACHE_VOLUME", "vggt-econ-cache")
GPU_SPEC = os.environ.get("VGGT_MODAL_ECON_GPU", "A100")
CPU_COUNT = float(os.environ.get("VGGT_MODAL_ECON_CPU", "16"))
MEMORY_MB = int(os.environ.get("VGGT_MODAL_ECON_MEMORY_MB", str(96 * 1024)))
TIMEOUT_SEC = int(os.environ.get("VGGT_MODAL_ECON_TIMEOUT_SEC", str(6 * 60 * 60)))

DEFAULT_SCENE_SUBDIR = (
    "4k4d_preprocessed_scene_variants/"
    "0012_11_frame0000_6views_sparseproto_headshoulder_crop"
)
DEFAULT_LOCAL_SCENE_DIR = f"output/{DEFAULT_SCENE_SUBDIR}"
DEFAULT_REMOTE_SCENE_SUBDIR = f"econ_mesh_teacher/{DEFAULT_SCENE_SUBDIR}"
DEFAULT_OUTPUT_SUBDIR = "detail_normal_refiner_20260427/econ_mesh_teacher_smoke_00_tgt_cam00"
DEFAULT_DOWNLOAD_LOCAL_DIR = f"output/{DEFAULT_OUTPUT_SUBDIR}"
DEFAULT_IMAGE_NAME = "00_tgt_cam00.png"
DEFAULT_MASK_NAME = "00_tgt_cam00.png"
DEFAULT_LOOP_SMPL = 5
DEFAULT_PATIENCE = 2
DEFAULT_VOL_RES = 128
DEFAULT_MCUBE_RES = 192

ECON_REPO = "https://github.com/YuliangXiu/ECON"
ICON_REPO = "https://github.com/YuliangXiu/ICON"
ECON_REF = os.environ.get("VGGT_MODAL_ECON_REF", "d8f4e8b7171e30868acd94a1d1f6fcc1238e3e32")
ECON_SOURCE_ZIP_URL = os.environ.get(
    "VGGT_MODAL_ECON_SOURCE_ZIP_URL",
    f"https://codeload.github.com/YuliangXiu/ECON/zip/{ECON_REF}",
)
MIN_SOURCE_ZIP_BYTES = 1_000_000

OFFICIAL_BLOCKER_NOTES = [
    "ECON/ICON are not treated as accepted teacher results here; this wrapper is a smoke/probe only.",
    "Official ECON setup targets Ubuntu 18/20, Python 3.8, CUDA 11.6, PyTorch >=1.13, CuPy >=11.3, and PyTorch3D 0.7.2.",
    "Official fetch_data.sh requires an ICON account and accepted licenses for SMPL, SMPL-X, SMPLIFY, ECON, PIXIE, and PyMAF-X assets.",
    "The wrapper will not bypass license gates. Upload already-licensed ECON data into the Modal data volume and pass asset_subdir.",
]

OFFICIAL_DOCS = {
    "econ_repo": ECON_REPO,
    "econ_install": "https://github.com/YuliangXiu/ECON/blob/master/docs/installation-ubuntu.md",
    "econ_fetch_data": "https://github.com/YuliangXiu/ECON/blob/master/fetch_data.sh",
    "econ_infer": "https://github.com/YuliangXiu/ECON/blob/master/apps/infer.py",
    "icon_repo": ICON_REPO,
    "icon_install": "https://github.com/YuliangXiu/ICON/blob/master/docs/installation.md",
}


image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04",
        add_python="3.8",
    )
    .apt_install(
        "build-essential",
        "ca-certificates",
        "curl",
        "ffmpeg",
        "git",
        "libeigen3-dev",
        "libgl1",
        "libglib2.0-0",
        "libgomp1",
        "libsm6",
        "libxext6",
        "libxrender1",
        "ninja-build",
        "wget",
    )
    .run_commands("python -m pip install --upgrade pip wheel setuptools packaging ninja")
    .run_commands(
        "python -m pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 "
        "--extra-index-url https://download.pytorch.org/whl/cu116"
    )
    .pip_install(
        "boto3",
        "chumpy==0.70",
        "cupy-cuda11x==11.6.0",
        "cython",
        "einops",
        "fast-simplification",
        "fvcore",
        "huggingface_hub",
        "iopath",
        "kornia==0.6.12",
        "matplotlib",
        "mediapipe==0.10.11",
        "numpy==1.23.5",
        "omegaconf",
        "opencv-contrib-python-headless",
        "opencv-python-headless",
        "Pillow==10.4.0",
        "protobuf==3.20.3",
        "pytorch-lightning==1.9.5",
        "PyYAML",
        "rembg==2.0.63",
        "rtree",
        "scikit-image",
        "scikit-learn",
        "scipy",
        "termcolor",
        "tqdm",
        "trimesh",
        "xatlas",
        "yacs",
    )
    .run_commands("python -m pip install open3d==0.18.0")
    .run_commands(
        "python -m pip install --no-build-isolation "
        "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.2"
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
        tmp_path = dest_path.with_name(dest_path.name + ".part")
        with tmp_path.open("wb") as file_obj:
            output_volume.read_file_into_fileobj(entry.path, file_obj)
        tmp_path.replace(dest_path)
        files_downloaded += 1
    print(f"[econ] downloaded {files_downloaded} files to {local_dir}")


def _remote_data_path(subdir: str) -> Path:
    return Path(str(REMOTE_DATA_DIR / _norm(subdir)))


def _remote_output_path(subdir: str) -> Path:
    return Path(str(REMOTE_OUTPUT_DIR / _norm(subdir)))


def _download_url(url: str, dest_path: Path, min_bytes: int) -> dict:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_name(dest_path.name + ".part")
    tmp_path.unlink(missing_ok=True)
    start = time.time()
    bytes_written = 0
    request = urllib.request.Request(url, headers={"User-Agent": "vggt-econ-modal-wrapper"})
    with urllib.request.urlopen(request, timeout=120) as response:
        with tmp_path.open("wb") as file_obj:
            while True:
                chunk = response.read(16 * 1024 * 1024)
                if not chunk:
                    break
                file_obj.write(chunk)
                bytes_written += len(chunk)
                if bytes_written and bytes_written % (256 * 1024 * 1024) < len(chunk):
                    print(f"[econ] downloaded {bytes_written / (1024 ** 2):.1f} MiB from {url}")
    size = tmp_path.stat().st_size
    if size < min_bytes:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded source is too small: {url} -> {size} bytes")
    tmp_path.replace(dest_path)
    return {"url": url, "path": dest_path.as_posix(), "bytes": int(size), "seconds": round(time.time() - start, 3)}


def _ensure_econ_source(cache_root: Path, source_zip_url: str) -> tuple[Path, dict]:
    source_root = cache_root / f"ECON-{ECON_REF}"
    marker = source_root / "apps" / "infer.py"
    if marker.is_file():
        return source_root, {"cache_hit": True, "path": source_root.as_posix(), "repo": ECON_REPO, "ref": ECON_REF}

    cache_root.mkdir(parents=True, exist_ok=True)
    zip_path = cache_root / f"ECON-{ECON_REF}.zip"
    download_info = None
    if not zip_path.is_file() or zip_path.stat().st_size < MIN_SOURCE_ZIP_BYTES:
        download_info = _download_url(source_zip_url, zip_path, MIN_SOURCE_ZIP_BYTES)

    tmp_dir = Path(tempfile.mkdtemp(prefix="econ_src_"))
    try:
        with zipfile.ZipFile(zip_path) as zip_obj:
            zip_obj.extractall(tmp_dir)
        extracted = next((path for path in tmp_dir.iterdir() if (path / "apps" / "infer.py").is_file()), None)
        if extracted is None:
            raise RuntimeError(f"ECON archive did not contain apps/infer.py: {zip_path}")
        if source_root.exists():
            shutil.rmtree(source_root)
        shutil.copytree(extracted, source_root)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return source_root, {
        "cache_hit": False,
        "path": source_root.as_posix(),
        "download": download_info,
        "repo": ECON_REPO,
        "ref": ECON_REF,
    }


def _select_input_paths(scene_root: Path, image_name: str, mask_name: str) -> tuple[Path, Path | None]:
    images_dir = scene_root / "images"
    masks_dir = scene_root / "masks"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Remote images directory not found: {images_dir}")

    if image_name:
        image_path = images_dir / image_name
        if not image_path.is_file():
            available = sorted(path.name for path in images_dir.glob("*") if path.is_file())
            raise FileNotFoundError(f"Requested image not found: {image_path}; available={available}")
    else:
        image_candidates = sorted(path for path in images_dir.glob("*") if path.is_file())
        if not image_candidates:
            raise FileNotFoundError(f"No input images found in {images_dir}")
        image_path = image_candidates[0]

    resolved_mask_name = mask_name or image_path.name
    mask_path = masks_dir / resolved_mask_name
    if mask_name and not mask_path.is_file():
        raise FileNotFoundError(f"Requested mask not found: {mask_path}")
    return image_path, mask_path if mask_path.is_file() else None


def _prepare_econ_input(
    image_path: Path,
    mask_path: Path | None,
    prepared_image_path: Path,
    prepared_mask_path: Path,
    background: int,
) -> dict:
    from PIL import Image
    import numpy as np

    image_obj = Image.open(image_path).convert("RGB")
    image_arr = np.asarray(image_obj, dtype=np.uint8)
    height, width = image_arr.shape[:2]
    mask_pixels = 0
    bbox_xyxy = [0, 0, int(width), int(height)]

    if mask_path is not None:
        mask_obj = Image.open(mask_path).convert("L").resize((width, height), Image.Resampling.NEAREST)
        mask_arr = np.asarray(mask_obj, dtype=np.uint8) > 127
        mask_pixels = int(mask_arr.sum())
        background_arr = np.full_like(image_arr, int(max(0, min(255, background))))
        image_arr = np.where(mask_arr[..., None], image_arr, background_arr)
        Image.fromarray((mask_arr.astype(np.uint8) * 255), mode="L").save(prepared_mask_path)
        if mask_pixels:
            ys, xs = np.where(mask_arr)
            bbox_xyxy = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]

    prepared_image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_arr, mode="RGB").save(prepared_image_path)
    return {
        "source_image": image_path.name,
        "source_mask": mask_path.name if mask_path is not None else "",
        "prepared_image": prepared_image_path.name,
        "prepared_mask": prepared_mask_path.name if mask_path is not None else "",
        "size_wh": [int(width), int(height)],
        "mask_used": bool(mask_path is not None),
        "mask_pixels": mask_pixels,
        "bbox_xyxy": bbox_xyxy,
        "background": int(max(0, min(255, background))),
        "note": "ECON still runs its own human detector/rembg path; the scene mask is used to suppress background before inference.",
    }


def _link_or_copy_child(source: Path, dest: Path) -> dict:
    if not source.exists():
        return {"source": source.as_posix(), "dest": dest.as_posix(), "status": "missing_source"}
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        else:
            shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        dest.symlink_to(source, target_is_directory=source.is_dir())
        return {"source": source.as_posix(), "dest": dest.as_posix(), "status": "symlinked"}
    except OSError:
        if source.is_dir():
            shutil.copytree(source, dest)
            status = "copied_tree"
        else:
            shutil.copy2(source, dest)
            status = "copied_file"
        return {"source": source.as_posix(), "dest": dest.as_posix(), "status": status}


def _materialize_asset_subdir(source_dir: Path, asset_subdir: str) -> dict:
    if not asset_subdir:
        return {"asset_subdir": "", "status": "not_provided", "items": []}

    asset_root = _remote_data_path(asset_subdir)
    if not asset_root.exists():
        return {"asset_subdir": _norm(asset_subdir), "status": "missing", "path": asset_root.as_posix(), "items": []}

    data_root = source_dir / "data"
    if (asset_root / "data").is_dir():
        children_root = asset_root / "data"
    else:
        children_root = asset_root

    items = []
    for child in sorted(children_root.iterdir()):
        items.append(_link_or_copy_child(child, data_root / child.name))
    return {
        "asset_subdir": _norm(asset_subdir),
        "status": "materialized",
        "path": asset_root.as_posix(),
        "source_layout": children_root.as_posix(),
        "items": items,
    }


def _glob_present(source_dir: Path, patterns: list[str]) -> list[str]:
    paths = []
    for pattern in patterns:
        paths.extend(path.relative_to(source_dir).as_posix() for path in source_dir.glob(pattern) if path.exists())
    return sorted(set(paths))


def _inspect_econ_assets(source_dir: Path, asset_subdir: str, use_sapiens_normal: bool) -> dict:
    groups = [
        {
            "name": "econ_normal_checkpoint",
            "required": True,
            "patterns": ["data/ckpt/normal.ckpt"],
            "why": "apps/infer.py always loads cfg.normal_path.",
        },
        {
            "name": "smpl_data_constants",
            "required": True,
            "patterns": [
                "data/smpl_related/smpl_data/smpl_verts.npy",
                "data/smpl_related/smpl_data/smpl_faces.npy",
                "data/smpl_related/smpl_data/smplx_verts.npy",
                "data/smpl_related/smpl_data/smplx_faces.npy",
                "data/smpl_related/smpl_data/smplx_cmap.npy",
                "data/smpl_related/smpl_data/smplx_to_smpl.pkl",
            ],
            "why": "lib/dataset/mesh_util.py initializes SMPLX constants before inference.",
        },
        {
            "name": "licensed_smpl_models",
            "required": True,
            "patterns": [
                "data/smpl_related/models/smpl/SMPL_FEMALE.pkl",
                "data/smpl_related/models/smpl/SMPL_MALE.pkl",
                "data/smpl_related/models/smpl/SMPL_NEUTRAL.pkl",
            ],
            "why": "Official fetch_data.sh pulls SMPL/SMPLIFY models behind license-gated downloads.",
        },
        {
            "name": "licensed_smplx_neutral",
            "required": True,
            "patterns": [
                "data/smpl_related/models/smplx/SMPLX_NEUTRAL.npz",
                "data/smpl_related/models/smplx/SMPLX_NEUTRAL.pkl",
                "data/smpl_related/models/smplx/SMPLX_NEUTRAL_2020.npz",
                "data/HPS/pixie_data/SMPLX_NEUTRAL_2020.npz",
            ],
            "any_pattern": True,
            "why": "ECON default HPS path is PIXIE/SMPL-X.",
        },
        {
            "name": "pixie_hps_assets",
            "required": True,
            "patterns": [
                "data/HPS/pixie_data/pixie_model.tar",
                "data/HPS/pixie_data/SMPL_X_template_FLAME_uv.obj",
                "data/HPS/pixie_data/smplx_tex.obj",
                "data/HPS/pixie_data/smplx_extra_joints.yaml",
            ],
            "why": "configs/econ.yaml sets bni.hps_type=pixie by default.",
        },
        {
            "name": "pymafx_hps_assets",
            "required": False,
            "patterns": ["data/HPS/pymafx_data/PyMAF-X_model_checkpoint.pt"],
            "why": "Only required if a future run switches hps_type to pymafx.",
        },
        {
            "name": "sapiens_assets",
            "required": bool(use_sapiens_normal),
            "patterns": [
                "data/sapiens/assets/checkpoints/*.pt2",
                "data/checkpoints/sapiens_*_torchscript.pt2",
            ],
            "any_pattern": True,
            "why": "Only required when use_sapiens_normal=True; default wrapper disables it to avoid huge implicit downloads.",
        },
    ]

    checks = []
    blockers = []
    for group in groups:
        present = _glob_present(source_dir, group["patterns"])
        required = bool(group["required"])
        complete = bool(present) if group.get("any_pattern") else len(present) == len(group["patterns"])
        check = {**group, "present": present, "complete": complete}
        checks.append(check)
        if required and not complete:
            blockers.append(
                {
                    "kind": "missing_official_asset",
                    "group": group["name"],
                    "why": group["why"],
                    "expected_patterns": group["patterns"],
                    "present": present,
                }
            )

    if blockers and not asset_subdir:
        blockers.insert(
            0,
            {
                "kind": "asset_subdir_not_provided",
                "message": (
                    "Upload an already-licensed ECON data directory to the Modal data volume and pass asset_subdir. "
                    "Accepted layouts: <asset_subdir>/data/ckpt/... or <asset_subdir>/ckpt/..."
                ),
            },
        )

    return {
        "asset_subdir": _norm(asset_subdir) if asset_subdir else "",
        "checks": checks,
        "blockers": blockers,
        "official_blocker_notes": OFFICIAL_BLOCKER_NOTES,
    }


def _write_runtime_config(
    source_dir: Path,
    config_path: Path,
    vol_res: int,
    mcube_res: int,
    use_sapiens_normal: bool,
    use_ifnet: bool,
) -> dict:
    text = f"""name: econ_modal_smoke
ckpt_dir: "./data/ckpt/"
normal_path: "./data/ckpt/normal.ckpt"
ifnet_path: "./data/ckpt/ifnet.ckpt"
results_path: "./results"

net:
  in_nml: ((\"image\",3), (\"T_normal_F\",3), (\"T_normal_B\",3))
  in_geo: ((\"normal_F\",3), (\"normal_B\",3))

test_mode: True
batch_size: 1

dataset:
  prior_type: "SMPL"

vol_res: {int(vol_res)}
mcube_res: {int(mcube_res)}
clean_mesh: True
cloth_overlap_thres: 0.50
body_overlap_thres: 0.00
force_smpl_optim: True

sapiens:
  use: {str(bool(use_sapiens_normal)).lower()}
  seg_model: "fg-bg-1b"
  normal_model: "1b"

bni:
  k: 4
  lambda1: 1e-4
  boundary_consist: 1e-6
  poisson_depth: 8
  use_smpl: ["hand"]
  use_ifnet: {str(bool(use_ifnet)).lower()}
  use_poisson: True
  hand_thres: 8e-2
  face_thres: 6e-2
  thickness: 0.02
  hps_type: "pixie"
  texture_src: "image"
  cut_intersection: True
"""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(text, encoding="utf-8")
    return {
        "path": config_path.as_posix(),
        "vol_res": int(vol_res),
        "mcube_res": int(mcube_res),
        "use_sapiens_normal": bool(use_sapiens_normal),
        "use_ifnet": bool(use_ifnet),
        "note": "apps/infer.py has a hard-coded mcube_res override; wrapper patches that value before launch.",
        "source_config": (source_dir / "configs" / "econ.yaml").as_posix(),
    }


def _restore_original_text(path: Path) -> str:
    backup_path = path.with_name(path.name + ".modal_original")
    if backup_path.is_file():
        original_text = backup_path.read_text(encoding="utf-8")
        path.write_text(original_text, encoding="utf-8")
        return original_text
    original_text = path.read_text(encoding="utf-8")
    backup_path.write_text(original_text, encoding="utf-8")
    return original_text


def _patch_econ_source_for_modal(source_dir: Path, mcube_res: int, disable_sapiens_import: bool) -> dict:
    patches = []
    infer_path = source_dir / "apps" / "infer.py"
    infer_text = _restore_original_text(infer_path)
    patched_text = infer_text.replace('"mcube_res", 512', f'"mcube_res", {int(mcube_res)}')
    if patched_text != infer_text:
        infer_path.write_text(patched_text, encoding="utf-8")
        patches.append({"file": infer_path.relative_to(source_dir).as_posix(), "patch": f"mcube_res=512 -> {int(mcube_res)}"})

    sapiens_path = source_dir / "apps" / "sapiens.py"
    if disable_sapiens_import and sapiens_path.is_file():
        _restore_original_text(sapiens_path)
        sapiens_path.write_text(
            "class ImageProcessor:\n"
            "    def __init__(self, *args, **kwargs):\n"
            "        raise RuntimeError('Sapiens normal disabled by modal_econ_mesh_teacher.py')\n",
            encoding="utf-8",
        )
        patches.append({"file": sapiens_path.relative_to(source_dir).as_posix(), "patch": "stubbed to avoid import-time Sapiens snapshot download"})
    elif sapiens_path.is_file():
        _restore_original_text(sapiens_path)

    return {"patches": patches}


def _run_command(cmd: list[str], cwd: Path, env_extra: dict[str, str] | None = None, check: bool = True) -> dict:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=cwd.as_posix(),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = proc.stdout or ""
    result = {
        "cmd": cmd,
        "cwd": cwd.as_posix(),
        "returncode": int(proc.returncode),
        "seconds": round(time.time() - start, 3),
        "stdout_tail": "\n".join(output.splitlines()[-220:]),
    }
    if check and proc.returncode != 0:
        raise RuntimeError(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _build_econ_extensions(source_dir: Path) -> dict:
    marker = source_dir / ".modal_econ_extensions_built"
    if marker.is_file():
        return {"cache_hit": True, "marker": marker.as_posix(), "commands": []}

    commands = []
    env = {
        "PYTHONPATH": source_dir.as_posix(),
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST", "7.5;8.0;8.6"),
    }
    for rel_dir in ["lib/common/libmesh", "lib/common/libvoxelize"]:
        build_dir = source_dir / rel_dir
        if not (build_dir / "setup.py").is_file():
            raise FileNotFoundError(f"ECON extension setup.py not found: {build_dir}")
        commands.append(_run_command([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=build_dir, env_extra=env))

    marker.write_text(json.dumps({"built_at_unix": time.time(), "commands": commands}, indent=2), encoding="utf-8")
    return {"cache_hit": False, "marker": marker.as_posix(), "commands": commands}


def _obj_to_ascii_ply(obj_path: Path, ply_path: Path) -> dict:
    vertices: list[tuple[float, float, float]] = []
    colors: list[tuple[int, int, int]] = []
    faces: list[tuple[int, int, int]] = []
    saw_color = False

    with obj_path.open("r", encoding="utf-8", errors="replace") as file_obj:
        for raw_line in file_obj:
            if raw_line.startswith("v "):
                parts = raw_line.strip().split()
                if len(parts) < 4:
                    continue
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                if len(parts) >= 7:
                    rgb = [float(parts[4]), float(parts[5]), float(parts[6])]
                    if max(rgb) <= 1.0:
                        rgb = [value * 255.0 for value in rgb]
                    colors.append(tuple(int(max(0, min(255, round(value)))) for value in rgb))
                    saw_color = True
                else:
                    colors.append((255, 255, 255))
            elif raw_line.startswith("f "):
                indices = []
                for part in raw_line.strip().split()[1:]:
                    if not part:
                        continue
                    index = int(part.split("/")[0])
                    indices.append(len(vertices) + index if index < 0 else index - 1)
                if len(indices) >= 3:
                    for face_index in range(1, len(indices) - 1):
                        faces.append((indices[0], indices[face_index], indices[face_index + 1]))

    ply_path.parent.mkdir(parents=True, exist_ok=True)
    with ply_path.open("w", encoding="utf-8", newline="\n") as file_obj:
        file_obj.write("ply\n")
        file_obj.write("format ascii 1.0\n")
        file_obj.write(f"comment source_obj {obj_path.name}\n")
        file_obj.write(f"element vertex {len(vertices)}\n")
        file_obj.write("property float x\nproperty float y\nproperty float z\n")
        file_obj.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        file_obj.write(f"element face {len(faces)}\n")
        file_obj.write("property list uchar int vertex_indices\n")
        file_obj.write("end_header\n")
        for vertex, color in zip(vertices, colors):
            file_obj.write(
                f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f} {color[0]} {color[1]} {color[2]}\n"
            )
        for face in faces:
            file_obj.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    return {
        "obj": obj_path.name,
        "ply": ply_path.name,
        "vertices": len(vertices),
        "faces": len(faces),
        "vertex_colors": saw_color,
    }


def _copy_mesh_outputs(result_dir: Path, out_root: Path) -> list[dict]:
    obj_paths = sorted(result_dir.rglob("*.obj"))
    records = []
    for obj_path in obj_paths:
        rel = obj_path.relative_to(result_dir)
        safe_stem = "__".join(rel.with_suffix("").parts)
        dest_obj = out_root / f"{safe_stem}.obj"
        shutil.copy2(obj_path, dest_obj)
        dest_ply = out_root / f"{safe_stem}.ply"
        record = _obj_to_ascii_ply(dest_obj, dest_ply)
        record.update(
            {
                "source": obj_path.as_posix(),
                "relative_source": rel.as_posix(),
                "obj_path": dest_obj.name,
                "ply_path": dest_ply.name,
                "obj_bytes": int(dest_obj.stat().st_size),
                "ply_bytes": int(dest_ply.stat().st_size),
                "is_final_candidate": any(token in obj_path.name for token in ["_full", "_refine", "_recon"]),
            }
        )
        records.append(record)
    return records


def _write_blocker(out_root: Path, summary: dict, message: str) -> None:
    blocker_path = out_root / "econ_blocker.txt"
    blocker_path.write_text(
        message
        + "\n\n"
        + json.dumps(
            {
                "status": summary.get("status"),
                "blockers": summary.get("blockers", []),
                "official_blocker_notes": OFFICIAL_BLOCKER_NOTES,
                "docs": OFFICIAL_DOCS,
                "error": summary.get("error", ""),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


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
def run_econ_mesh_remote(
    scene_subdir: str,
    output_subdir: str,
    image_name: str = DEFAULT_IMAGE_NAME,
    mask_name: str = DEFAULT_MASK_NAME,
    asset_subdir: str = "",
    loop_smpl: int = DEFAULT_LOOP_SMPL,
    patience: int = DEFAULT_PATIENCE,
    vol_res: int = DEFAULT_VOL_RES,
    mcube_res: int = DEFAULT_MCUBE_RES,
    background: int = 127,
    use_sapiens_normal: bool = False,
    use_ifnet: bool = False,
    compile_extensions: bool = True,
    run_inference: bool = True,
    source_zip_url: str = ECON_SOURCE_ZIP_URL,
) -> dict:
    out_root = _remote_output_path(output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "ok": False,
        "status": "started",
        "result_is_gate_ready": False,
        "gate_note": "ECON/ICON external wrapper smoke only; do not treat output as an accepted mesh teacher result without separate review.",
        "scene_subdir": _norm(scene_subdir),
        "output_subdir": _norm(output_subdir),
        "image_name": image_name,
        "mask_name": mask_name,
        "asset_subdir": _norm(asset_subdir) if asset_subdir else "",
        "loop_smpl": int(loop_smpl),
        "patience": int(patience),
        "vol_res": int(vol_res),
        "mcube_res": int(mcube_res),
        "background": int(max(0, min(255, background))),
        "use_sapiens_normal": bool(use_sapiens_normal),
        "use_ifnet": bool(use_ifnet),
        "compile_extensions": bool(compile_extensions),
        "run_inference": bool(run_inference),
        "econ_repo": ECON_REPO,
        "icon_repo": ICON_REPO,
        "econ_ref": ECON_REF,
        "source_zip_url": source_zip_url,
        "official_docs": OFFICIAL_DOCS,
        "official_blocker_notes": OFFICIAL_BLOCKER_NOTES,
        "blockers": [],
        "started_at_unix": time.time(),
    }

    try:
        scene_root = _remote_data_path(scene_subdir)
        image_path, mask_path = _select_input_paths(scene_root, image_name, mask_name)
        stage_root = Path(tempfile.mkdtemp(prefix="econ_mesh_teacher_"))
        input_dir = stage_root / "input"
        seg_dir = stage_root / "seg"
        result_dir = stage_root / "results"
        input_dir.mkdir(parents=True, exist_ok=True)
        seg_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        prepared_image_path = input_dir / image_path.name
        prepared_mask_path = seg_dir / image_path.name
        summary["input"] = _prepare_econ_input(
            image_path=image_path,
            mask_path=mask_path,
            prepared_image_path=prepared_image_path,
            prepared_mask_path=prepared_mask_path,
            background=background,
        )
        shutil.copy2(prepared_image_path, out_root / prepared_image_path.name)
        if prepared_mask_path.is_file():
            shutil.copy2(prepared_mask_path, out_root / f"{prepared_image_path.stem}_mask.png")

        cache_root = Path(str(REMOTE_CACHE_DIR))
        source_dir, source_info = _ensure_econ_source(cache_root, source_zip_url)
        summary["econ_source"] = source_info
        summary["source_patches"] = _patch_econ_source_for_modal(
            source_dir=source_dir,
            mcube_res=mcube_res,
            disable_sapiens_import=not bool(use_sapiens_normal),
        )
        summary["asset_materialization"] = _materialize_asset_subdir(source_dir, asset_subdir)
        summary["assets"] = _inspect_econ_assets(source_dir, asset_subdir, use_sapiens_normal)
        summary["blockers"] = summary["assets"]["blockers"]
        cache_volume.commit()

        if summary["blockers"]:
            summary["status"] = "blocked"
            summary["finished_at_unix"] = time.time()
            _write_blocker(
                out_root,
                summary,
                "ECON mesh teacher is blocked by official/license-gated dependencies or missing model assets.",
            )
            return summary

        if not run_inference:
            summary["status"] = "probe_only"
            summary["blockers"] = [
                {
                    "kind": "inference_disabled",
                    "message": "run_inference=False, so wrapper only prepared input and validated ECON assets.",
                }
            ]
            summary["finished_at_unix"] = time.time()
            _write_blocker(out_root, summary, "ECON wrapper probe completed without running inference.")
            return summary

        runtime_config_path = stage_root / "econ_modal_smoke.yaml"
        summary["runtime_config"] = _write_runtime_config(
            source_dir=source_dir,
            config_path=runtime_config_path,
            vol_res=vol_res,
            mcube_res=mcube_res,
            use_sapiens_normal=use_sapiens_normal,
            use_ifnet=use_ifnet,
        )
        shutil.copy2(runtime_config_path, out_root / runtime_config_path.name)

        if compile_extensions:
            summary["extensions"] = _build_econ_extensions(source_dir)
        else:
            summary["extensions"] = {
                "skipped": True,
                "blocker_risk": "ECON libmesh/libvoxelize extensions are required by official setup.",
            }

        command = [
            sys.executable,
            "-m",
            "apps.infer",
            "-gpu",
            "0",
            "-loop_smpl",
            str(int(loop_smpl)),
            "-patience",
            str(int(patience)),
            "-in_dir",
            input_dir.as_posix(),
            "-out_dir",
            result_dir.as_posix(),
            "-seg_dir",
            seg_dir.as_posix(),
            "-cfg",
            runtime_config_path.as_posix(),
            "-novis",
        ]
        command_result = _run_command(
            command,
            cwd=source_dir,
            env_extra={
                "PYTHONPATH": source_dir.as_posix(),
                "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1",
                "TORCH_HOME": (Path(str(REMOTE_CACHE_DIR)) / "torch").as_posix(),
                "U2NET_HOME": (Path(str(REMOTE_CACHE_DIR)) / "u2net").as_posix(),
                "HF_HOME": (Path(str(REMOTE_CACHE_DIR)) / "huggingface").as_posix(),
            },
        )
        (out_root / "econ_command.log").write_text(command_result["stdout_tail"], encoding="utf-8")
        summary["command"] = command_result

        meshes = _copy_mesh_outputs(result_dir, out_root)
        if not meshes:
            raise RuntimeError(f"ECON command completed but produced no OBJ meshes under {result_dir}")

        summary["meshes"] = meshes
        summary["ok"] = True
        summary["status"] = "ok_smoke_only"
        summary["finished_at_unix"] = time.time()
        blocker_path = out_root / "econ_blocker.txt"
        blocker_path.unlink(missing_ok=True)
    except Exception as exc:
        summary["ok"] = False
        summary["status"] = "failed"
        summary["error"] = repr(exc)
        summary["traceback"] = traceback.format_exc()
        summary.setdefault("blockers", []).append(
            {
                "kind": "runtime_failure",
                "message": repr(exc),
                "note": "This may still be an ECON dependency/model-license blocker; see traceback and command tail.",
            }
        )
        _write_blocker(out_root, summary, "ECON mesh teacher did not complete.")
    finally:
        (out_root / "econ_mesh_teacher_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        output_volume.commit()

    return summary


@app.local_entrypoint()
def run_from_local(
    local_scene_dir: str = DEFAULT_LOCAL_SCENE_DIR,
    remote_scene_subdir: str = DEFAULT_REMOTE_SCENE_SUBDIR,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    download_local_dir: str = DEFAULT_DOWNLOAD_LOCAL_DIR,
    image_name: str = DEFAULT_IMAGE_NAME,
    mask_name: str = DEFAULT_MASK_NAME,
    asset_subdir: str = "",
    loop_smpl: int = DEFAULT_LOOP_SMPL,
    patience: int = DEFAULT_PATIENCE,
    vol_res: int = DEFAULT_VOL_RES,
    mcube_res: int = DEFAULT_MCUBE_RES,
    background: int = 127,
    use_sapiens_normal: bool = False,
    use_ifnet: bool = False,
    compile_extensions: bool = True,
    run_inference: bool = True,
    source_zip_url: str = ECON_SOURCE_ZIP_URL,
):
    remote = _upload_dir(Path(local_scene_dir), remote_scene_subdir)
    summary = run_econ_mesh_remote.remote(
        remote,
        output_subdir,
        image_name=image_name,
        mask_name=mask_name,
        asset_subdir=asset_subdir,
        loop_smpl=loop_smpl,
        patience=patience,
        vol_res=vol_res,
        mcube_res=mcube_res,
        background=background,
        use_sapiens_normal=use_sapiens_normal,
        use_ifnet=use_ifnet,
        compile_extensions=compile_extensions,
        run_inference=run_inference,
        source_zip_url=source_zip_url,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if download_local_dir:
        _download_dir(summary["output_subdir"], Path(download_local_dir))
