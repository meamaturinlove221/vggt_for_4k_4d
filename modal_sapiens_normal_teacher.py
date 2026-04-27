from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath

import modal


REMOTE_DATA_DIR = PurePosixPath("/mnt/data")
REMOTE_OUTPUT_DIR = PurePosixPath("/mnt/out")
APP_NAME = os.environ.get("VGGT_MODAL_SAPIENS_NORMAL_APP_NAME", "vggt-sapiens-normal-teacher")
DATA_VOLUME_NAME = os.environ.get("VGGT_MODAL_DATA_VOLUME", "vggt-4k4d-data")
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_MODAL_OUTPUT_VOLUME", "vggt-4k4d-output")


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "numpy==1.26.1",
        "Pillow",
        "opencv-python-headless",
        "huggingface_hub",
        "tqdm",
    )
)
app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)


def _norm(value: str) -> str:
    value = (value or "").replace("\\", "/").strip("/")
    if not value:
        raise ValueError("empty subpath")
    return value


def _upload_dir(local_dir: Path, remote_subdir: str) -> str:
    local_dir = local_dir.expanduser().resolve()
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
        rel = Path(entry.path)
        try:
            rel = rel.relative_to(prefix)
        except ValueError:
            pass
        dest = local_dir / rel
        if entry.type == modal.volume.FileEntryType.DIRECTORY:
            dest.mkdir(parents=True, exist_ok=True)
            continue
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as file_obj:
            output_volume.read_file_into_fileobj(entry.path, file_obj)


@app.function(
    image=image,
    gpu="A10G",
    cpu=8,
    memory=49152,
    timeout=4 * 60 * 60,
    volumes={str(REMOTE_DATA_DIR): data_volume, str(REMOTE_OUTPUT_DIR): output_volume},
)
def run_sapiens_normal_remote(
    scene_subdir: str,
    output_subdir: str,
    model_repo: str = "facebook/sapiens-normal-0.3b-torchscript",
    model_filename: str = "sapiens_0.3b_normal_render_people_epoch_66_torchscript.pt2",
    max_views: int = 0,
    input_height: int = 1024,
    input_width: int = 768,
) -> dict:
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from PIL import Image

    scene_root = Path(str(REMOTE_DATA_DIR / _norm(scene_subdir)))
    out_root = Path(str(REMOTE_OUTPUT_DIR / _norm(output_subdir)))
    out_root.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(path for path in (scene_root / "images").iterdir() if path.is_file())
    mask_paths = sorted(path for path in (scene_root / "masks").iterdir() if path.is_file())
    if max_views and max_views > 0:
        image_paths = image_paths[:max_views]
        mask_paths = mask_paths[:max_views]
    if len(image_paths) != len(mask_paths):
        raise ValueError(f"image/mask count mismatch: {len(image_paths)} vs {len(mask_paths)}")

    checkpoint = hf_hub_download(repo_id=model_repo, filename=model_filename)
    model = torch.jit.load(checkpoint, map_location="cuda").eval().cuda()
    mean = torch.tensor([123.5, 116.5, 103.5], device="cuda", dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([58.5, 57.0, 57.5], device="cuda", dtype=torch.float32).view(1, 3, 1, 1)

    normal_maps = []
    mask_maps = []
    records = []
    with torch.no_grad():
        for idx, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(image_path)
            orig_h, orig_w = bgr.shape[:2]
            resized = cv2.resize(bgr, (int(input_width), int(input_height)), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(resized).cuda().float().permute(2, 0, 1)[None]
            tensor = tensor[:, [2, 1, 0], :, :]
            tensor = (tensor - mean) / std
            result = model(tensor)[0]
            if isinstance(result, (tuple, list)):
                result = result[0]
            result = F.interpolate(result[None], size=(orig_h, orig_w), mode="bilinear", align_corners=False)[0]
            normal = result.float().detach().cpu().numpy().transpose(1, 2, 0)
            normal = normal / np.clip(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6, None)
            mask = np.asarray(Image.open(mask_path).convert("L").resize((orig_w, orig_h), Image.Resampling.NEAREST)) > 127
            normal[~mask] = 0.0
            rgb = np.clip((normal + 1.0) * 0.5, 0.0, 1.0)
            rgb[~mask] = 1.0
            rgb_u8 = (rgb * 255.0).astype(np.uint8)
            Image.fromarray(rgb_u8).save(out_root / f"{idx:02d}_{image_path.stem}_sapiens_normal.png")
            np.save(out_root / f"{idx:02d}_{image_path.stem}_sapiens_normal.npy", normal.astype(np.float32))
            normal_maps.append(normal.astype(np.float32))
            mask_maps.append(mask.astype(bool))
            records.append(
                {
                    "index": idx,
                    "image": image_path.name,
                    "mask": mask_path.name,
                    "normal_png": f"{idx:02d}_{image_path.stem}_sapiens_normal.png",
                    "shape": [int(orig_h), int(orig_w), 3],
                    "mask_pixels": int(mask.sum()),
                }
            )

    if normal_maps:
        np.savez_compressed(
            out_root / "sapiens_normals.npz",
            normal=np.stack(normal_maps, axis=0).astype(np.float32),
            mask=np.stack(mask_maps, axis=0).astype(bool),
            image_names=np.asarray([record["image"] for record in records]),
        )
    summary = {
        "scene_subdir": _norm(scene_subdir),
        "output_subdir": _norm(output_subdir),
        "num_views": len(records),
        "model_repo": model_repo,
        "model_filename": model_filename,
        "input_shape_hw": [int(input_height), int(input_width)],
        "records": records,
    }
    (out_root / "external_sapiens_normal_teacher_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    output_volume.commit()
    return summary


@app.local_entrypoint()
def run_from_local(
    local_scene_dir: str,
    remote_scene_subdir: str,
    output_subdir: str,
    download_local_dir: str = "",
    max_views: int = 0,
    model_repo: str = "facebook/sapiens-normal-0.3b-torchscript",
    model_filename: str = "sapiens_0.3b_normal_render_people_epoch_66_torchscript.pt2",
):
    remote = _upload_dir(Path(local_scene_dir), remote_scene_subdir)
    summary = run_sapiens_normal_remote.remote(
        remote,
        output_subdir,
        model_repo=model_repo,
        model_filename=model_filename,
        max_views=max_views,
    )
    print(json.dumps(summary, indent=2))
    if download_local_dir:
        _download_dir(summary["output_subdir"], Path(download_local_dir))
