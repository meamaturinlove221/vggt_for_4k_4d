from __future__ import annotations

import json
import os
import shutil
from pathlib import PurePosixPath
from pathlib import Path

import modal


APP_NAME = os.environ.get("VGGT_MODAL_PSHUMAN_SPACE_APP_NAME", "vggt-pshuman-space-probe")
REMOTE_DATA_DIR = PurePosixPath("/mnt/data")
REMOTE_OUTPUT_DIR = PurePosixPath("/mnt/out")
DATA_VOLUME_NAME = os.environ.get("VGGT_MODAL_DATA_VOLUME", "vggt-4k4d-data")
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_MODAL_OUTPUT_VOLUME", "vggt-4k4d-output")
GPU_SPEC = os.environ.get("VGGT_MODAL_PSHUMAN_SPACE_GPU", "A10G")
TIMEOUT_SEC = int(os.environ.get("VGGT_MODAL_PSHUMAN_SPACE_TIMEOUT_SEC", str(30 * 60)))

DEFAULT_LOCAL_SCENE_DIR = "output/4k4d_preprocessed_scene_variants/0012_11_frame0000_6views_sparseproto_headshoulder_crop"
DEFAULT_REMOTE_SCENE_SUBDIR = "pshuman_space/0012_11_frame0000_6views_sparseproto_headshoulder_crop"
DEFAULT_OUTPUT_SUBDIR = "detail_normal_refiner_20260426/pshuman_space_cam30"
DEFAULT_DOWNLOAD_LOCAL_DIR = f"output/{DEFAULT_OUTPUT_SUBDIR}"
DEFAULT_IMAGE_NAME = "30_src_cam30.png"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("gradio_client==2.5.0", "Pillow")
)

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)


def _norm(value: str) -> str:
    cleaned = (value or "").replace("\\", "/").strip("/")
    if not cleaned:
        raise ValueError("empty subpath")
    return cleaned


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


@app.function(
    image=image,
    gpu=GPU_SPEC,
    timeout=TIMEOUT_SEC,
    volumes={REMOTE_OUTPUT_DIR.as_posix(): output_volume},
)
def probe_space(space_id: str = "fffiloni/PSHuman", output_subdir: str = "pshuman_space_probe") -> dict:
    from gradio_client import Client

    out_root = REMOTE_OUTPUT_DIR / output_subdir.strip("/").replace("\\", "/")
    out_path = str(out_root)
    os.makedirs(out_path, exist_ok=True)
    summary = {"ok": False, "space_id": space_id, "output_subdir": output_subdir}
    try:
        client = Client(space_id)
        api = client.view_api(return_format="dict")
        summary["ok"] = True
        summary["api"] = api
    except Exception as exc:
        summary["error"] = repr(exc)
    with open(os.path.join(out_path, "pshuman_space_probe_summary.json"), "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)
    output_volume.commit()
    return summary


def _select_scene_image(scene_root: Path, image_name: str) -> tuple[Path, Path | None]:
    image_path = scene_root / "images" / image_name
    if not image_path.is_file():
        available = sorted(path.name for path in (scene_root / "images").glob("*") if path.is_file())
        raise FileNotFoundError(f"Image not found: {image_path}; available={available}")
    mask_path = scene_root / "masks" / image_name
    return image_path, mask_path if mask_path.is_file() else None


def _prepare_rgba_input(image_path: Path, mask_path: Path | None, out_path: Path) -> dict:
    from PIL import Image

    rgb = Image.open(image_path).convert("RGB")
    if mask_path is not None:
        alpha = Image.open(mask_path).convert("L").resize(rgb.size, Image.Resampling.NEAREST)
    else:
        alpha = Image.new("L", rgb.size, 255)
    rgba = Image.merge("RGBA", (*rgb.split(), alpha))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(out_path)
    return {
        "image": image_path.name,
        "mask": mask_path.name if mask_path is not None else None,
        "rgba": out_path.name,
        "size_wh": list(rgb.size),
    }


def _copy_returned_file(value, out_root: Path, label: str) -> dict:
    if isinstance(value, dict):
        if "video" in value:
            return _copy_returned_file(value["video"], out_root, label)
        path_value = value.get("path")
    else:
        path_value = value
    if not path_value:
        return {"label": label, "copied": False, "reason": "empty"}
    source = Path(str(path_value))
    if not source.is_file():
        return {"label": label, "copied": False, "source": str(source), "reason": "not_file"}
    suffix = source.suffix or ".bin"
    dest = out_root / f"{label}{suffix}"
    shutil.copy2(source, dest)
    return {"label": label, "copied": True, "source": str(source), "path": dest.name, "bytes": int(dest.stat().st_size)}


@app.function(
    image=image,
    gpu=GPU_SPEC,
    timeout=TIMEOUT_SEC,
    volumes={
        REMOTE_DATA_DIR.as_posix(): data_volume,
        REMOTE_OUTPUT_DIR.as_posix(): output_volume,
    },
)
def run_pshuman_space_remote(
    scene_subdir: str,
    output_subdir: str,
    image_name: str = DEFAULT_IMAGE_NAME,
    space_id: str = "fffiloni/PSHuman",
    remove_bg: bool = False,
) -> dict:
    from gradio_client import Client, handle_file

    out_root = Path(str(REMOTE_OUTPUT_DIR / _norm(output_subdir)))
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "ok": False,
        "space_id": space_id,
        "scene_subdir": _norm(scene_subdir),
        "output_subdir": _norm(output_subdir),
        "image_name": image_name,
        "remove_bg": bool(remove_bg),
    }
    try:
        scene_root = Path(str(REMOTE_DATA_DIR / _norm(scene_subdir)))
        image_path, mask_path = _select_scene_image(scene_root, image_name)
        input_path = out_root / f"{Path(image_name).stem}_pshuman_input_rgba.png"
        summary["input"] = _prepare_rgba_input(image_path, mask_path, input_path)

        client = Client(space_id)
        result = client.predict(
            input_pil=handle_file(str(input_path)),
            remove_bg=bool(remove_bg),
            api_name="/process_image",
        )
        summary["raw_result_type"] = type(result).__name__
        if not isinstance(result, (list, tuple)) or len(result) < 3:
            raise RuntimeError(f"Unexpected PSHuman Space result: {result!r}")
        summary["outputs"] = [
            _copy_returned_file(result[0], out_root, "output_video"),
            _copy_returned_file(result[1], out_root, "mesh_obj"),
            _copy_returned_file(result[2], out_root, "mesh_colored_obj"),
        ]
        summary["ok"] = any(item.get("copied") for item in summary["outputs"])
    except Exception as exc:
        summary["error"] = repr(exc)
    with (out_root / "pshuman_space_run_summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)
    output_volume.commit()
    return summary


@app.local_entrypoint()
def main(
    mode: str = "probe",
    local_scene_dir: str = DEFAULT_LOCAL_SCENE_DIR,
    remote_scene_subdir: str = DEFAULT_REMOTE_SCENE_SUBDIR,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    download_local_dir: str = DEFAULT_DOWNLOAD_LOCAL_DIR,
    image_name: str = DEFAULT_IMAGE_NAME,
    space_id: str = "fffiloni/PSHuman",
    remove_bg: bool = False,
):
    if mode == "probe":
        summary = probe_space.remote(space_id=space_id, output_subdir=output_subdir)
    elif mode == "run":
        remote = _upload_dir(Path(local_scene_dir), remote_scene_subdir)
        summary = run_pshuman_space_remote.remote(
            scene_subdir=remote,
            output_subdir=output_subdir,
            image_name=image_name,
            space_id=space_id,
            remove_bg=remove_bg,
        )
        if download_local_dir:
            _download_dir(summary["output_subdir"], Path(download_local_dir))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
