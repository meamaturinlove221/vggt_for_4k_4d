from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
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
]


def _resolve_requirements() -> list[str]:
    candidate = REPO_ROOT / "requirements.txt"
    if candidate.exists():
        return _load_requirements(candidate)
    return list(DEFAULT_REQUIREMENTS)


APP_NAME = os.environ.get("VGGT_MODAL_DETAIL_REFINER_APP_NAME", "vggt-detail-normal-refiner")
DATA_VOLUME_NAME = os.environ.get("VGGT_MODAL_DATA_VOLUME", "vggt-4k4d-data")
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_MODAL_OUTPUT_VOLUME", "vggt-4k4d-output")
GPU_SPEC = os.environ.get("VGGT_MODAL_GPU", "A100-40GB")
CPU_COUNT = float(os.environ.get("VGGT_MODAL_CPU", "8"))
MEMORY_MB = int(os.environ.get("VGGT_MODAL_MEMORY_MB", str(64 * 1024)))
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

TRAIN_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(*_resolve_requirements())
    .add_local_dir(str(REPO_ROOT / "vggt"), remote_path=(REMOTE_CODE_DIR / "vggt").as_posix(), ignore=CODE_SYNC_IGNORE)
    .add_local_dir(str(REPO_ROOT / "training"), remote_path=(REMOTE_CODE_DIR / "training").as_posix(), ignore=CODE_SYNC_IGNORE)
    .add_local_dir(str(REPO_ROOT / "tools"), remote_path=(REMOTE_CODE_DIR / "tools").as_posix(), ignore=CODE_SYNC_IGNORE)
)

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)


@dataclass
class RefinerTrainConfig:
    train_dataset_subpath: str
    output_subdir: str
    val_dataset_subpath: str = ""
    epochs: int = 30
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 7
    base_dim: int = 32
    residual_scale: float = 0.35
    max_train_samples: int = 0
    max_val_samples: int = 0
    visualize_count: int = 8
    num_workers: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(blob: str) -> "RefinerTrainConfig":
        return RefinerTrainConfig(**json.loads(blob))


def _normalize_subpath(value: str) -> str:
    cleaned = (value or "").strip().replace("\\", "/").strip("/")
    if not cleaned:
        raise ValueError("Expected a non-empty volume-relative path.")
    return cleaned


def _remote_data_path(subpath: str) -> Path:
    return Path(str(REMOTE_DATA_DIR / _normalize_subpath(subpath)))


def _remote_output_path(subpath: str) -> Path:
    return Path(str(REMOTE_OUTPUT_DIR / _normalize_subpath(subpath)))


def _upload_single_file(local_path: Path, remote_subpath: str) -> str:
    local_path = local_path.expanduser().resolve()
    if not local_path.is_file():
        raise FileNotFoundError(f"File not found: {local_path}")

    remote_subpath = _normalize_subpath(remote_subpath)
    print(f"[modal-detail] upload file: {local_path} -> {DATA_VOLUME_NAME}:{remote_subpath}")
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(local_path), remote_subpath)
    return remote_subpath


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


@app.function(
    image=TRAIN_IMAGE,
    gpu=GPU_SPEC,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SEC,
    volumes={
        REMOTE_DATA_DIR.as_posix(): data_volume,
        REMOTE_OUTPUT_DIR.as_posix(): output_volume,
    },
)
def run_remote_detail_normal_refiner(cfg_json: str) -> dict:
    cfg = RefinerTrainConfig.from_json(cfg_json)
    remote_code_dir = Path(str(REMOTE_CODE_DIR))
    output_root = _remote_output_path(cfg.output_subdir)
    output_root.mkdir(parents=True, exist_ok=True)
    train_dataset = _remote_data_path(cfg.train_dataset_subpath)
    if not train_dataset.is_file():
        raise FileNotFoundError(f"Remote train dataset not found: {train_dataset}")
    val_dataset = _remote_data_path(cfg.val_dataset_subpath) if cfg.val_dataset_subpath.strip() else train_dataset
    if not val_dataset.is_file():
        raise FileNotFoundError(f"Remote val dataset not found: {val_dataset}")

    cmd = [
        sys.executable,
        str(remote_code_dir / "tools" / "train_detail_normal_refiner.py"),
        "--train-dataset-npz",
        train_dataset.as_posix(),
        "--val-dataset-npz",
        val_dataset.as_posix(),
        "--output-dir",
        output_root.as_posix(),
        "--epochs",
        str(cfg.epochs),
        "--batch-size",
        str(cfg.batch_size),
        "--lr",
        str(cfg.lr),
        "--weight-decay",
        str(cfg.weight_decay),
        "--seed",
        str(cfg.seed),
        "--device",
        "cuda",
        "--base-dim",
        str(cfg.base_dim),
        "--residual-scale",
        str(cfg.residual_scale),
        "--max-train-samples",
        str(cfg.max_train_samples),
        "--max-val-samples",
        str(cfg.max_val_samples),
        "--visualize-count",
        str(cfg.visualize_count),
        "--num-workers",
        str(cfg.num_workers),
    ]

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    repo_pythonpath = str(remote_code_dir)
    env["PYTHONPATH"] = repo_pythonpath if not existing_pythonpath else repo_pythonpath + os.pathsep + existing_pythonpath
    env["PYTHONUNBUFFERED"] = "1"

    print("[modal-detail] data_volume =", DATA_VOLUME_NAME, flush=True)
    print("[modal-detail] output_volume =", OUTPUT_VOLUME_NAME, flush=True)
    print("[modal-detail] gpu =", GPU_SPEC, flush=True)
    print("[modal-detail] output_root =", output_root.as_posix(), flush=True)
    print("[modal-detail] command =", shlex.join(cmd), flush=True)

    subprocess.run(cmd, cwd=str(remote_code_dir), env=env, check=True)
    summary_path = output_root / "run_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Expected run summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["output_subdir"] = output_root.relative_to(Path(str(REMOTE_OUTPUT_DIR))).as_posix()
    output_volume.commit()
    return summary


@app.local_entrypoint()
def run_from_local(
    train_dataset_path: str,
    val_dataset_path: str = "",
    remote_dataset_root: str = "detail_normal_refiner_datasets",
    remote_train_name: str = "",
    remote_val_name: str = "",
    output_subdir: str = "detail_normal_refiner/remote_run",
    download_local_dir: str = "",
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 7,
    base_dim: int = 32,
    residual_scale: float = 0.35,
    max_train_samples: int = 0,
    max_val_samples: int = 0,
    visualize_count: int = 8,
    num_workers: int = 0,
) -> None:
    remote_dataset_root = _normalize_subpath(remote_dataset_root)
    train_path = Path(train_dataset_path).expanduser().resolve()
    train_name = remote_train_name.strip() or train_path.name
    train_subpath = _upload_single_file(train_path, f"{remote_dataset_root}/{train_name}")

    val_subpath = ""
    if val_dataset_path.strip():
        val_path = Path(val_dataset_path).expanduser().resolve()
        val_name = remote_val_name.strip() or val_path.name
        val_subpath = _upload_single_file(val_path, f"{remote_dataset_root}/{val_name}")

    cfg = RefinerTrainConfig(
        train_dataset_subpath=train_subpath,
        val_dataset_subpath=val_subpath,
        output_subdir=output_subdir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        base_dim=base_dim,
        residual_scale=residual_scale,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        visualize_count=visualize_count,
        num_workers=num_workers,
    )
    print("[modal-detail] launch config:")
    print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
    summary = run_remote_detail_normal_refiner.remote(cfg.to_json())
    print("[modal-detail] remote summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if download_local_dir.strip():
        _download_volume_dir(summary["output_subdir"], Path(download_local_dir))
