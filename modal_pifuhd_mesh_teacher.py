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

APP_NAME = os.environ.get("VGGT_MODAL_PIFUHD_APP_NAME", "vggt-pifuhd-mesh-teacher")
DATA_VOLUME_NAME = os.environ.get("VGGT_MODAL_DATA_VOLUME", "vggt-4k4d-data")
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_MODAL_OUTPUT_VOLUME", "vggt-4k4d-output")
CACHE_VOLUME_NAME = os.environ.get("VGGT_MODAL_PIFUHD_CACHE_VOLUME", "vggt-pifuhd-cache")
GPU_SPEC = os.environ.get("VGGT_MODAL_PIFUHD_GPU", "A10G")
CPU_COUNT = float(os.environ.get("VGGT_MODAL_PIFUHD_CPU", "8"))
MEMORY_MB = int(os.environ.get("VGGT_MODAL_PIFUHD_MEMORY_MB", str(48 * 1024)))
TIMEOUT_SEC = int(os.environ.get("VGGT_MODAL_PIFUHD_TIMEOUT_SEC", str(4 * 60 * 60)))

DEFAULT_SCENE_SUBDIR = (
    "4k4d_preprocessed_scene_variants/"
    "0012_11_frame0000_6views_sparseproto_headshoulder_crop"
)
DEFAULT_LOCAL_SCENE_DIR = f"output/{DEFAULT_SCENE_SUBDIR}"
DEFAULT_REMOTE_SCENE_SUBDIR = f"pifuhd_mesh_teacher_smoke/{DEFAULT_SCENE_SUBDIR}"
DEFAULT_OUTPUT_SUBDIR = "detail_normal_refiner_20260426/pifuhd_mesh_teacher_smoke"
DEFAULT_DOWNLOAD_LOCAL_DIR = f"output/{DEFAULT_OUTPUT_SUBDIR}"
DEFAULT_IMAGE_NAME = "30_src_cam30.png"
DEFAULT_RESOLUTION = 256

PIFUHD_REPO = "https://github.com/facebookresearch/pifuhd"
PIFUHD_COMMIT = "e47c4d918aaedd5f5608192b130bda150b1fb0ab"
PIFUHD_SOURCE_ZIP_URL = f"https://codeload.github.com/facebookresearch/pifuhd/zip/{PIFUHD_COMMIT}"
PIFUHD_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/pifuhd/checkpoints/pifuhd.pt"
MIN_SOURCE_ZIP_BYTES = 100_000
MIN_CHECKPOINT_BYTES = 1_000_000_000


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ca-certificates", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "numpy==1.23.5",
        "Pillow",
        "opencv-python-headless",
        "scikit-image==0.22.0",
        "tqdm",
        "matplotlib",
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
        with dest_path.open("wb") as file_obj:
            output_volume.read_file_into_fileobj(entry.path, file_obj)


def _remote_data_path(subdir: str) -> Path:
    return Path(str(REMOTE_DATA_DIR / _norm(subdir)))


def _remote_output_path(subdir: str) -> Path:
    return Path(str(REMOTE_OUTPUT_DIR / _norm(subdir)))


def _download_url(url: str, dest_path: Path, min_bytes: int) -> dict:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_name(dest_path.name + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    start = time.time()
    bytes_written = 0
    request = urllib.request.Request(url, headers={"User-Agent": "vggt-pifuhd-modal-smoke"})
    with urllib.request.urlopen(request, timeout=120) as response:
        with tmp_path.open("wb") as file_obj:
            while True:
                chunk = response.read(16 * 1024 * 1024)
                if not chunk:
                    break
                file_obj.write(chunk)
                bytes_written += len(chunk)
                if bytes_written and bytes_written % (256 * 1024 * 1024) < len(chunk):
                    print(f"[pifuhd] downloaded {bytes_written / (1024 ** 2):.1f} MiB from {url}")

    size = tmp_path.stat().st_size
    if size < min_bytes:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is too small: {url} -> {size} bytes")
    tmp_path.replace(dest_path)
    return {"url": url, "path": dest_path.as_posix(), "bytes": int(size), "seconds": round(time.time() - start, 3)}


def _ensure_pifuhd_source(cache_root: Path, source_zip_url: str) -> tuple[Path, dict]:
    source_root = cache_root / f"pifuhd-{PIFUHD_COMMIT}"
    marker = source_root / "apps" / "simple_test.py"
    if marker.is_file():
        return source_root, {"cache_hit": True, "path": source_root.as_posix()}

    cache_root.mkdir(parents=True, exist_ok=True)
    zip_path = cache_root / f"pifuhd-{PIFUHD_COMMIT}.zip"
    download_info = None
    if not zip_path.is_file() or zip_path.stat().st_size < MIN_SOURCE_ZIP_BYTES:
        download_info = _download_url(source_zip_url, zip_path, MIN_SOURCE_ZIP_BYTES)

    tmp_dir = Path(tempfile.mkdtemp(prefix="pifuhd_src_"))
    try:
        with zipfile.ZipFile(zip_path) as zip_obj:
            zip_obj.extractall(tmp_dir)
        extracted = next((path for path in tmp_dir.iterdir() if (path / "apps" / "simple_test.py").is_file()), None)
        if extracted is None:
            raise RuntimeError(f"PIFuHD source archive did not contain apps/simple_test.py: {zip_path}")
        if source_root.exists():
            shutil.rmtree(source_root)
        shutil.copytree(extracted, source_root)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return source_root, {
        "cache_hit": False,
        "path": source_root.as_posix(),
        "download": download_info,
        "repo": PIFUHD_REPO,
        "commit": PIFUHD_COMMIT,
    }


def _ensure_pifuhd_checkpoint(cache_root: Path, checkpoint_url: str) -> tuple[Path, dict]:
    checkpoint_path = cache_root / "checkpoints" / f"pifuhd_{PIFUHD_COMMIT}.pt"
    if checkpoint_path.is_file() and checkpoint_path.stat().st_size >= MIN_CHECKPOINT_BYTES:
        return checkpoint_path, {
            "cache_hit": True,
            "path": checkpoint_path.as_posix(),
            "bytes": int(checkpoint_path.stat().st_size),
        }

    download_info = _download_url(checkpoint_url, checkpoint_path, MIN_CHECKPOINT_BYTES)
    return checkpoint_path, {"cache_hit": False, **download_info}


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


def _prepare_pifuhd_input(
    image_path: Path,
    mask_path: Path | None,
    prepared_image_path: Path,
    rect_path: Path,
    prepared_mask_path: Path,
    rect_padding: float,
    background: int,
) -> dict:
    from PIL import Image
    import numpy as np

    image_obj = Image.open(image_path).convert("RGB")
    image_arr = np.asarray(image_obj, dtype=np.uint8)
    height, width = image_arr.shape[:2]
    mask_arr = None
    mask_pixels = 0

    if mask_path is not None:
        mask_obj = Image.open(mask_path).convert("L").resize((width, height), Image.Resampling.NEAREST)
        mask_arr = np.asarray(mask_obj, dtype=np.uint8) > 127
        mask_pixels = int(mask_arr.sum())
        background_arr = np.full_like(image_arr, int(max(0, min(255, background))))
        image_arr = np.where(mask_arr[..., None], image_arr, background_arr)
        Image.fromarray((mask_arr.astype(np.uint8) * 255), mode="L").save(prepared_mask_path)

    prepared_image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_arr, mode="RGB").save(prepared_image_path)

    if mask_arr is not None and mask_pixels > 0:
        ys, xs = np.where(mask_arr)
        x_min = float(xs.min())
        x_max = float(xs.max() + 1)
        y_min = float(ys.min())
        y_max = float(ys.max() + 1)
    else:
        x_min = 0.0
        x_max = float(width)
        y_min = 0.0
        y_max = float(height)

    bbox_width = max(1.0, x_max - x_min)
    bbox_height = max(1.0, y_max - y_min)
    side = max(bbox_width, bbox_height) * (1.0 + 2.0 * max(0.0, rect_padding))
    side = max(16.0, side)
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    rect = [
        int(round(cx - side * 0.5)),
        int(round(cy - side * 0.5)),
        int(round(side)),
        int(round(side)),
    ]
    rect_path.write_text("%d %d %d %d\n" % tuple(rect), encoding="utf-8")

    return {
        "source_image": image_path.name,
        "source_mask": mask_path.name if mask_path is not None else "",
        "prepared_image": prepared_image_path.name,
        "prepared_mask": prepared_mask_path.name if mask_path is not None else "",
        "size_wh": [int(width), int(height)],
        "mask_used": bool(mask_path is not None),
        "mask_pixels": mask_pixels,
        "rect_xywh": rect,
        "rect_padding": float(rect_padding),
        "background": int(max(0, min(255, background))),
    }


def _run_command(cmd: list[str], cwd: Path, env_extra: dict[str, str]) -> dict:
    env = os.environ.copy()
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
    elapsed = round(time.time() - start, 3)
    output = proc.stdout or ""
    result = {
        "cmd": cmd,
        "cwd": cwd.as_posix(),
        "returncode": int(proc.returncode),
        "seconds": elapsed,
        "stdout_tail": "\n".join(output.splitlines()[-160:]),
    }
    if proc.returncode != 0:
        raise RuntimeError(json.dumps(result, ensure_ascii=False, indent=2))
    return result


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
                parts = raw_line.strip().split()[1:]
                indices: list[int] = []
                for part in parts:
                    if not part:
                        continue
                    index = int(part.split("/")[0])
                    if index < 0:
                        index = len(vertices) + index
                    else:
                        index -= 1
                    indices.append(index)
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
def run_pifuhd_mesh_remote(
    scene_subdir: str,
    output_subdir: str,
    image_name: str = DEFAULT_IMAGE_NAME,
    mask_name: str = "",
    resolution: int = DEFAULT_RESOLUTION,
    rect_padding: float = 0.15,
    background: int = 127,
    source_zip_url: str = PIFUHD_SOURCE_ZIP_URL,
    checkpoint_url: str = PIFUHD_CHECKPOINT_URL,
) -> dict:
    out_root = _remote_output_path(output_subdir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "ok": False,
        "scene_subdir": _norm(scene_subdir),
        "output_subdir": _norm(output_subdir),
        "image_name": image_name,
        "mask_name": mask_name,
        "resolution": int(resolution),
        "pifuhd_repo": PIFUHD_REPO,
        "pifuhd_commit": PIFUHD_COMMIT,
        "checkpoint_url": checkpoint_url,
        "source_zip_url": source_zip_url,
        "started_at_unix": time.time(),
    }

    try:
        scene_root = _remote_data_path(scene_subdir)
        image_path, mask_path = _select_input_paths(scene_root, image_name, mask_name)
        stage_root = Path(tempfile.mkdtemp(prefix="pifuhd_mesh_teacher_"))
        input_dir = stage_root / "input"
        result_dir = stage_root / "results"
        input_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        prepared_image_path = input_dir / image_path.name
        prepared_mask_path = input_dir / f"{image_path.stem}_mask.png"
        rect_path = input_dir / f"{image_path.stem}_rect.txt"
        summary["input"] = _prepare_pifuhd_input(
            image_path=image_path,
            mask_path=mask_path,
            prepared_image_path=prepared_image_path,
            rect_path=rect_path,
            prepared_mask_path=prepared_mask_path,
            rect_padding=rect_padding,
            background=background,
        )

        shutil.copy2(prepared_image_path, out_root / prepared_image_path.name)
        shutil.copy2(rect_path, out_root / rect_path.name)
        if prepared_mask_path.is_file():
            shutil.copy2(prepared_mask_path, out_root / prepared_mask_path.name)

        cache_root = Path(str(REMOTE_CACHE_DIR))
        source_dir, source_info = _ensure_pifuhd_source(cache_root, source_zip_url)
        checkpoint_path, checkpoint_info = _ensure_pifuhd_checkpoint(cache_root, checkpoint_url)
        summary["pifuhd_source"] = source_info
        summary["checkpoint"] = checkpoint_info
        cache_volume.commit()

        cmd = [
            sys.executable,
            "-m",
            "apps.simple_test",
            "-i",
            input_dir.as_posix(),
            "-o",
            result_dir.as_posix(),
            "-c",
            checkpoint_path.as_posix(),
            "-r",
            str(int(resolution)),
            "--use_rect",
        ]
        command_result = _run_command(
            cmd,
            cwd=source_dir,
            env_extra={
                "PYTHONPATH": source_dir.as_posix(),
                "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1",
            },
        )
        (out_root / "pifuhd_command.log").write_text(command_result["stdout_tail"], encoding="utf-8")
        summary["command"] = command_result

        obj_paths = sorted(result_dir.rglob("*.obj"))
        if not obj_paths:
            raise RuntimeError(f"PIFuHD completed but produced no OBJ under {result_dir}")

        mesh_records = []
        for obj_path in obj_paths:
            dest_obj = out_root / obj_path.name
            shutil.copy2(obj_path, dest_obj)
            dest_ply = out_root / f"{obj_path.stem}.ply"
            mesh_record = _obj_to_ascii_ply(dest_obj, dest_ply)
            mesh_record["obj_bytes"] = int(dest_obj.stat().st_size)
            mesh_record["ply_bytes"] = int(dest_ply.stat().st_size)
            mesh_records.append(mesh_record)

        summary["ok"] = True
        summary["meshes"] = mesh_records
        summary["finished_at_unix"] = time.time()
        blocker_path = out_root / "pifuhd_blocker.txt"
        if blocker_path.exists():
            blocker_path.unlink()
    except Exception as exc:
        summary["ok"] = False
        summary["error"] = repr(exc)
        summary["traceback"] = traceback.format_exc()
        (out_root / "pifuhd_blocker.txt").write_text(
            "PIFuHD mesh teacher smoke did not complete.\n\n"
            f"Error: {repr(exc)}\n\n"
            f"Traceback:\n{summary['traceback']}\n",
            encoding="utf-8",
        )
    finally:
        (out_root / "pifuhd_mesh_teacher_summary.json").write_text(
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
    mask_name: str = "",
    resolution: int = DEFAULT_RESOLUTION,
    rect_padding: float = 0.15,
    background: int = 127,
    source_zip_url: str = PIFUHD_SOURCE_ZIP_URL,
    checkpoint_url: str = PIFUHD_CHECKPOINT_URL,
):
    remote = _upload_dir(Path(local_scene_dir), remote_scene_subdir)
    summary = run_pifuhd_mesh_remote.remote(
        remote,
        output_subdir,
        image_name=image_name,
        mask_name=mask_name,
        resolution=resolution,
        rect_padding=rect_padding,
        background=background,
        source_zip_url=source_zip_url,
        checkpoint_url=checkpoint_url,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if download_local_dir:
        _download_dir(summary["output_subdir"], Path(download_local_dir))
