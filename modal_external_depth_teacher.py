from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath

import modal

REMOTE_DATA_DIR = PurePosixPath('/mnt/data')
REMOTE_OUTPUT_DIR = PurePosixPath('/mnt/out')
APP_NAME = os.environ.get('VGGT_MODAL_EXTERNAL_DEPTH_APP_NAME', 'vggt-external-depth-teacher')
DATA_VOLUME_NAME = os.environ.get('VGGT_MODAL_DATA_VOLUME', 'vggt-4k4d-data')
OUTPUT_VOLUME_NAME = os.environ.get('VGGT_MODAL_OUTPUT_VOLUME', 'vggt-4k4d-output')

image = (
    modal.Image.debian_slim(python_version='3.10')
    .apt_install('git', 'libgl1', 'libglib2.0-0')
    .pip_install(
        'torch==2.3.1', 'torchvision==0.18.1', 'numpy==1.26.1', 'Pillow',
        'opencv-python-headless', 'transformers==4.48.0', 'accelerate', 'safetensors', 'huggingface_hub'
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
    local_dir=local_dir.expanduser().resolve(); remote_subdir=_norm(remote_subdir)
    with data_volume.batch_upload(force=True) as batch:
        for p in local_dir.rglob('*'):
            if p.is_file():
                batch.put_file(str(p), f'{remote_subdir}/{p.relative_to(local_dir).as_posix()}')
    return remote_subdir

def _download_dir(remote_subdir: str, local_dir: Path) -> None:
    remote_subdir=_norm(remote_subdir); local_dir=local_dir.expanduser().resolve(); local_dir.mkdir(parents=True, exist_ok=True)
    prefix=Path(remote_subdir)
    for entry in output_volume.listdir(remote_subdir, recursive=True):
        rel=Path(entry.path)
        try: rel=rel.relative_to(prefix)
        except ValueError: pass
        dest=local_dir/rel
        if entry.type == modal.volume.FileEntryType.DIRECTORY:
            dest.mkdir(parents=True, exist_ok=True); continue
        if entry.type != modal.volume.FileEntryType.FILE: continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('wb') as f:
            output_volume.read_file_into_fileobj(entry.path, f)

@app.function(image=image, gpu='A10G', cpu=8, memory=32768, timeout=3*60*60, volumes={str(REMOTE_DATA_DIR):data_volume, str(REMOTE_OUTPUT_DIR):output_volume})
def run_depth_anything_remote(scene_subdir: str, output_subdir: str, max_views: int = 0) -> dict:
    from PIL import Image
    import numpy as np
    import torch
    from transformers import pipeline
    scene_root=Path(str(REMOTE_DATA_DIR / _norm(scene_subdir)))
    out_root=Path(str(REMOTE_OUTPUT_DIR / _norm(output_subdir))); out_root.mkdir(parents=True, exist_ok=True)
    paths=sorted([p for p in (scene_root/'images').iterdir() if p.is_file()])
    if max_views and max_views>0: paths=paths[:max_views]
    device=0 if torch.cuda.is_available() else -1
    pipe=pipeline(task='depth-estimation', model='depth-anything/Depth-Anything-V2-Small-hf', device=device)
    depths=[]; records=[]
    for idx,p in enumerate(paths):
        img=Image.open(p).convert('RGB')
        result=pipe(img)
        depth=result['depth'].resize((518,518))
        arr=np.asarray(depth, dtype=np.float32)
        # normalize for PNG preview only; npz keeps raw model relative scale
        lo,hi=np.percentile(arr[np.isfinite(arr)],[1,99])
        vis=np.clip((arr-lo)/max(hi-lo,1e-6),0,1)
        Image.fromarray((vis*255).astype(np.uint8)).save(out_root/f'{idx:02d}_{p.stem}_depthanything_v2.png')
        depths.append(arr.astype(np.float32)); records.append({'index':idx,'image':p.name,'shape':list(arr.shape),'min':float(np.nanmin(arr)),'max':float(np.nanmax(arr))})
    if depths:
        np.savez_compressed(out_root/'depthanything_v2_depths.npz', depth=np.stack(depths,0), image_names=np.asarray([r['image'] for r in records]))
    summary={'scene_subdir':_norm(scene_subdir),'output_subdir':_norm(output_subdir),'num_views':len(records),'records':records,'model':'depth-anything/Depth-Anything-V2-Small-hf'}
    (out_root/'external_depth_teacher_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    output_volume.commit(); return summary

@app.local_entrypoint()
def run_from_local(local_scene_dir: str, remote_scene_subdir: str, output_subdir: str, download_local_dir: str='', max_views: int=0):
    remote=_upload_dir(Path(local_scene_dir), remote_scene_subdir)
    summary=run_depth_anything_remote.remote(remote, output_subdir, max_views=max_views)
    print(json.dumps(summary,indent=2))
    if download_local_dir:
        _download_dir(summary['output_subdir'], Path(download_local_dir))
