from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath

import modal

REPO_ROOT = Path(__file__).resolve().parent
REMOTE_DATA_DIR = PurePosixPath('/mnt/data')
REMOTE_OUTPUT_DIR = PurePosixPath('/mnt/out')
APP_NAME = os.environ.get('VGGT_MODAL_EXTERNAL_NORMAL_APP_NAME', 'vggt-external-normal-teacher')
DATA_VOLUME_NAME = os.environ.get('VGGT_MODAL_DATA_VOLUME', 'vggt-4k4d-data')
OUTPUT_VOLUME_NAME = os.environ.get('VGGT_MODAL_OUTPUT_VOLUME', 'vggt-4k4d-output')

image = (
    modal.Image.debian_slim(python_version='3.10')
    .apt_install('git', 'libgl1', 'libglib2.0-0')
    .pip_install(
        'torch==2.3.1',
        'torchvision==0.18.1',
        'numpy==1.26.1',
        'Pillow',
        'opencv-python-headless',
        'controlnet-aux==0.0.10',
        'transformers==4.41.2',
        'accelerate',
        'safetensors',
        'huggingface_hub',
    )
)
app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)

def _norm(v: str) -> str:
    v=(v or '').replace('\\','/').strip('/')
    if not v:
        raise ValueError('empty subpath')
    return v

def _upload_dir(local_dir: Path, remote_subdir: str) -> str:
    local_dir=local_dir.expanduser().resolve()
    remote_subdir=_norm(remote_subdir)
    with data_volume.batch_upload(force=True) as batch:
        for p in local_dir.rglob('*'):
            if p.is_file():
                batch.put_file(str(p), f'{remote_subdir}/{p.relative_to(local_dir).as_posix()}')
    return remote_subdir

def _download_dir(remote_subdir: str, local_dir: Path) -> None:
    remote_subdir=_norm(remote_subdir)
    local_dir=local_dir.expanduser().resolve(); local_dir.mkdir(parents=True, exist_ok=True)
    prefix=Path(remote_subdir)
    for entry in output_volume.listdir(remote_subdir, recursive=True):
        rel=Path(entry.path)
        try: rel=rel.relative_to(prefix)
        except ValueError: pass
        dest=local_dir/rel
        if entry.type == modal.volume.FileEntryType.DIRECTORY:
            dest.mkdir(parents=True, exist_ok=True); continue
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('wb') as f:
            output_volume.read_file_into_fileobj(entry.path, f)

@app.function(image=image, gpu='A10G', cpu=8, memory=32768, timeout=3*60*60, volumes={str(REMOTE_DATA_DIR): data_volume, str(REMOTE_OUTPUT_DIR): output_volume})
def run_normalbae_remote(scene_subdir: str, output_subdir: str, max_views: int = 0) -> dict:
    from PIL import Image
    import numpy as np
    from controlnet_aux import NormalBaeDetector

    scene_root = Path(str(REMOTE_DATA_DIR / _norm(scene_subdir)))
    out_root = Path(str(REMOTE_OUTPUT_DIR / _norm(output_subdir)))
    out_root.mkdir(parents=True, exist_ok=True)
    image_dir = scene_root / 'images'
    paths = sorted([p for p in image_dir.iterdir() if p.is_file()])
    if max_views and max_views > 0:
        paths = paths[:max_views]
    detector = NormalBaeDetector.from_pretrained('lllyasviel/Annotators')
    records=[]
    normals=[]
    for idx,p in enumerate(paths):
        img=Image.open(p).convert('RGB')
        normal=detector(img)
        normal=normal.resize((518,518))
        out_path=out_root / f'{idx:02d}_{p.stem}_normalbae.png'
        normal.save(out_path)
        arr=np.asarray(normal, dtype=np.uint8)
        normals.append(arr)
        records.append({'index': idx, 'image': p.name, 'normal_png': out_path.name, 'shape': list(arr.shape)})
    if normals:
        np.savez_compressed(out_root/'normalbae_normals.npz', normal_rgb=np.stack(normals,0), image_names=np.asarray([r['image'] for r in records]))
    summary={'scene_subdir': _norm(scene_subdir), 'output_subdir': _norm(output_subdir), 'num_views': len(records), 'records': records, 'model':'lllyasviel/Annotators NormalBaeDetector'}
    (out_root/'external_normal_teacher_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    output_volume.commit()
    return summary

@app.local_entrypoint()
def run_from_local(local_scene_dir: str, remote_scene_subdir: str, output_subdir: str, download_local_dir: str = '', max_views: int = 0):
    remote=_upload_dir(Path(local_scene_dir), remote_scene_subdir)
    summary=run_normalbae_remote.remote(remote, output_subdir, max_views=max_views)
    print(json.dumps(summary, indent=2))
    if download_local_dir:
        _download_dir(summary['output_subdir'], Path(download_local_dir))
