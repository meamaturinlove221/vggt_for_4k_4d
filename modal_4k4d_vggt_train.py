from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
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
    "huggingface_hub",
    "einops",
    "safetensors",
]


def _resolve_base_requirements() -> list[str]:
    candidate = REPO_ROOT / "requirements.txt"
    if candidate.exists():
        return _load_requirements(candidate)
    return list(DEFAULT_REQUIREMENTS)


APP_NAME = os.environ.get("VGGT_MODAL_TRAIN_APP_NAME", "vggt-4k4d-train")
DATA_VOLUME_NAME = os.environ.get("VGGT_MODAL_DATA_VOLUME", "vggt-4k4d-data")
OUTPUT_VOLUME_NAME = os.environ.get("VGGT_MODAL_OUTPUT_VOLUME", "vggt-4k4d-output")
GPU_SPEC = os.environ.get("VGGT_MODAL_GPU", "A100-40GB")
CPU_COUNT = float(os.environ.get("VGGT_MODAL_CPU", "8"))
MEMORY_MB = int(os.environ.get("VGGT_MODAL_MEMORY_MB", str(96 * 1024)))
TIMEOUT_SEC = int(os.environ.get("VGGT_MODAL_TIMEOUT_SEC", str(12 * 60 * 60)))

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
    .apt_install(
        "git",
        "build-essential",
        "ffmpeg",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
    )
    .pip_install(
        *_resolve_base_requirements(),
        "hydra-core",
        "omegaconf",
        "iopath",
        "fvcore",
        "wcmatch",
        "tensorboard",
        "opencv-python-headless",
    )
    .add_local_dir(
        str(REPO_ROOT / "vggt"),
        remote_path=(REMOTE_CODE_DIR / "vggt").as_posix(),
        ignore=CODE_SYNC_IGNORE,
    )
    .add_local_dir(
        str(REPO_ROOT / "training"),
        remote_path=(REMOTE_CODE_DIR / "training").as_posix(),
        ignore=CODE_SYNC_IGNORE,
    )
)

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)


@dataclass
class TrainingConfig:
    case_subdirs: list[str] = field(default_factory=list)
    output_subdir: str = ""
    config_name: str = "4k4d_prior_case"
    exp_name: str = "4k4d_prior_case"
    pretrained_repo: str = "facebook/VGGT-1B"
    pretrained_filename: str = "model.pt"
    max_epochs: int = 5
    limit_train_batches: int = 100
    limit_val_batches: int = 10
    val_epoch_freq: int = 1
    learning_rate: float = 1e-5
    max_img_per_gpu: int = 13
    fix_img_num: int = -1
    img_nums_min: int = 7
    img_nums_max: int = 13
    len_train: int = 200
    len_test: int = 20
    seed_value: int = 42

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(blob: str) -> "TrainingConfig":
        payload = json.loads(blob)
        payload["case_subdirs"] = list(payload.get("case_subdirs", []))
        return TrainingConfig(**payload)


def _normalize_subpath(value: str) -> str:
    cleaned = (value or "").strip().replace("\\", "/").strip("/")
    if not cleaned:
        raise ValueError("Expected a non-empty volume-relative path.")
    return cleaned


def _remote_data_path(subpath: str) -> Path:
    return Path(str(REMOTE_DATA_DIR / _normalize_subpath(subpath)))


def _resolve_output_root(case_subdirs: list[str], output_subdir: str) -> Path:
    if output_subdir.strip():
        return Path(str(REMOTE_OUTPUT_DIR / _normalize_subpath(output_subdir)))

    first_case = Path(case_subdirs[0]).name if case_subdirs else "case"
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    return Path(str(REMOTE_OUTPUT_DIR / "vggt_4k4d_train" / f"{run_tag}_{first_case}"))


def _split_csv_paths(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _upload_case_dir(local_dir: Path, remote_subdir: str) -> str:
    local_dir = local_dir.expanduser().resolve()
    if not local_dir.is_dir():
        raise NotADirectoryError(f"Case directory not found: {local_dir}")

    remote_subdir = _normalize_subpath(remote_subdir)
    print(f"[modal-train] upload case: {local_dir} -> {DATA_VOLUME_NAME}:{remote_subdir}")
    with data_volume.batch_upload(force=True) as batch:
        for path in local_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(local_dir).as_posix()
            batch.put_file(str(path), f"{remote_subdir}/{rel}")
    return remote_subdir


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


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def _extract_model_state_dict(payload):
    if isinstance(payload, dict):
        if "model" in payload and isinstance(payload["model"], dict):
            return payload["model"]
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]
        return payload
    raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")


def _infer_model_kwargs_from_state_dict(state_dict: dict) -> dict:
    camera_token = state_dict.get("aggregator.camera_token")
    embed_dim = int(camera_token.shape[-1]) if camera_token is not None else 1024
    proj0 = state_dict.get("aggregator.human_prior_adapter.proj.0.weight")
    summary_proj0 = state_dict.get("aggregator.human_prior_adapter.summary_proj.0.weight")
    gate = state_dict.get("aggregator.human_prior_adapter.input_fusion.gate")
    return {
        "img_size": 518,
        "patch_size": 14,
        "embed_dim": embed_dim,
        "enable_camera": any(key.startswith("camera_head.") for key in state_dict),
        "enable_point": any(key.startswith("point_head.") for key in state_dict),
        "enable_depth": any(key.startswith("depth_head.") for key in state_dict),
        "enable_track": any(key.startswith("track_head.") for key in state_dict),
        "human_prior_channels": int(proj0.shape[1]) if proj0 is not None else 0,
        "human_prior_summary_channels": int(summary_proj0.shape[1]) if summary_proj0 is not None else 0,
        "human_prior_hidden_dim": int(proj0.shape[0]) if proj0 is not None else 64,
        "human_prior_gate_init": float(gate.item()) if gate is not None else 0.0,
    }


def _find_latest_checkpoint(ckpt_dir: Path) -> Path:
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    checkpoint_paths = sorted(ckpt_dir.glob("*.pt"), key=lambda path: path.stat().st_mtime)
    if not checkpoint_paths:
        raise FileNotFoundError(f"No .pt checkpoints found under {ckpt_dir}")
    return checkpoint_paths[-1]


def _build_hydra_overrides(cfg: TrainingConfig, case_roots: list[Path], log_dir: Path, ckpt_path: Path) -> list[str]:
    quoted_case_roots = ",".join(f"'{root.as_posix()}'" for root in case_roots)
    img_num_override = f"[{cfg.img_nums_min},{cfg.img_nums_max}]"

    return [
        f"exp_name={cfg.exp_name}",
        f"seed_value={cfg.seed_value}",
        f"max_epochs={cfg.max_epochs}",
        f"val_epoch_freq={cfg.val_epoch_freq}",
        f"limit_train_batches={cfg.limit_train_batches}",
        f"limit_val_batches={cfg.limit_val_batches}",
        f"max_img_per_gpu={cfg.max_img_per_gpu}",
        f"logging.log_dir={log_dir.as_posix()}",
        f"checkpoint.save_dir={(log_dir / 'ckpts').as_posix()}",
        f"checkpoint.resume_checkpoint_path={ckpt_path.as_posix()}",
        "checkpoint.strict=False",
        f"optim.optimizer.lr={cfg.learning_rate}",
        f"data.train.max_img_per_gpu={cfg.max_img_per_gpu}",
        f"data.val.max_img_per_gpu={cfg.max_img_per_gpu}",
        f"data.train.common_config.max_img_per_gpu={cfg.max_img_per_gpu}",
        f"data.train.common_config.fix_img_num={cfg.fix_img_num}",
        f"data.val.common_config.fix_img_num={cfg.fix_img_num}",
        f"data.train.common_config.img_nums={img_num_override}",
        f"data.val.common_config.img_nums={img_num_override}",
        f"data.train.dataset.dataset_configs.0.len_train={cfg.len_train}",
        f"data.train.dataset.dataset_configs.0.len_test={cfg.len_test}",
        f"data.val.dataset.dataset_configs.0.len_train={cfg.len_train}",
        f"data.val.dataset.dataset_configs.0.len_test={cfg.len_test}",
        f"data.train.dataset.dataset_configs.0.case_roots=[{quoted_case_roots}]",
        f"data.val.dataset.dataset_configs.0.case_roots=[{quoted_case_roots}]",
    ]


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
def run_remote_vggt_training(cfg_json: str) -> dict:
    cfg = TrainingConfig.from_json(cfg_json)
    if not cfg.case_subdirs:
        raise ValueError("TrainingConfig.case_subdirs must not be empty.")

    remote_code_dir = Path(str(REMOTE_CODE_DIR))
    from huggingface_hub import hf_hub_download

    case_roots = [_remote_data_path(subdir) for subdir in cfg.case_subdirs]
    for case_root in case_roots:
        if not case_root.is_dir():
            raise FileNotFoundError(f"Remote case dir not found: {case_root}")

    output_root = _resolve_output_root(cfg.case_subdirs, cfg.output_subdir)
    log_dir = output_root / "logs"
    output_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    pretrained_dir = Path(str(REMOTE_DATA_DIR / "pretrained" / cfg.pretrained_repo.replace("/", "__")))
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = pretrained_dir / cfg.pretrained_filename
    if not ckpt_path.exists():
        downloaded = hf_hub_download(
            repo_id=cfg.pretrained_repo,
            filename=cfg.pretrained_filename,
            local_dir=pretrained_dir,
            local_dir_use_symlinks=False,
        )
        ckpt_path = Path(downloaded)
        data_volume.commit()

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(_find_free_port())
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    overrides = _build_hydra_overrides(cfg, case_roots, log_dir, ckpt_path)
    started = time.time()
    launch_path = remote_code_dir / "training" / "launch.py"
    cmd = [sys.executable, str(launch_path), "--config", cfg.config_name, *overrides]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    repo_pythonpath = str(remote_code_dir)
    env["PYTHONPATH"] = repo_pythonpath if not existing_pythonpath else repo_pythonpath + os.pathsep + existing_pythonpath
    env["PYTHONUNBUFFERED"] = "1"

    print("[modal-train] data_volume =", DATA_VOLUME_NAME, flush=True)
    print("[modal-train] output_volume =", OUTPUT_VOLUME_NAME, flush=True)
    print("[modal-train] gpu =", GPU_SPEC, flush=True)
    print("[modal-train] output_root =", output_root.as_posix(), flush=True)
    print("[modal-train] checkpoint =", ckpt_path.as_posix(), flush=True)
    print("[modal-train] command =", shlex.join(cmd), flush=True)

    try:
        subprocess.run(
            cmd,
            cwd=str(remote_code_dir),
            env=env,
            check=True,
        )

        import torch

        latest_checkpoint = _find_latest_checkpoint(log_dir / "ckpts")
        checkpoint_payload = torch.load(latest_checkpoint, map_location="cpu")
        state_dict = _extract_model_state_dict(checkpoint_payload)
        inference_checkpoint = output_root / "inference_model.pt"
        torch.save(
            {
                "model": state_dict,
                "model_kwargs": _infer_model_kwargs_from_state_dict(state_dict),
                "source_checkpoint": latest_checkpoint.as_posix(),
            },
            inference_checkpoint,
        )

        summary = {
            "status": "completed",
            "config_name": cfg.config_name,
            "exp_name": cfg.exp_name,
            "case_subdirs": cfg.case_subdirs,
            "case_roots": [root.as_posix() for root in case_roots],
            "pretrained_repo": cfg.pretrained_repo,
            "pretrained_checkpoint": ckpt_path.as_posix(),
            "output_root": output_root.as_posix(),
            "output_subdir": output_root.relative_to(Path(str(REMOTE_OUTPUT_DIR))).as_posix(),
            "log_dir": log_dir.as_posix(),
            "checkpoint_dir": (log_dir / "ckpts").as_posix(),
            "latest_checkpoint": latest_checkpoint.as_posix(),
            "inference_checkpoint": inference_checkpoint.as_posix(),
            "inference_checkpoint_relpath": inference_checkpoint.relative_to(Path(str(REMOTE_OUTPUT_DIR))).as_posix(),
            "elapsed_seconds": round(time.time() - started, 3),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "max_epochs": cfg.max_epochs,
            "limit_train_batches": cfg.limit_train_batches,
            "limit_val_batches": cfg.limit_val_batches,
        }
        (output_root / "run_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        output_volume.commit()
        return summary
    finally:
        try:
            output_volume.commit()
        except Exception as exc:
            print(f"[modal-train] final commit warning: {exc}", flush=True)


@app.local_entrypoint()
def upload_cases(
    local_case_dirs: str,
    remote_case_root: str = "training_cases",
) -> None:
    local_dirs = [Path(item) for item in _split_csv_paths(local_case_dirs)]
    if not local_dirs:
        raise ValueError("Please provide at least one local case dir.")

    remote_case_root = _normalize_subpath(remote_case_root)
    remote_subdirs = []
    for local_dir in local_dirs:
        remote_subdirs.append(_upload_case_dir(local_dir, f"{remote_case_root}/{local_dir.name}"))
    print("[modal-train] uploaded cases:")
    print(json.dumps(remote_subdirs, indent=2, ensure_ascii=False))


@app.local_entrypoint()
def run_cases(
    case_subdirs: str,
    output_subdir: str = "",
    download_local_dir: str = "",
    config_name: str = "4k4d_prior_case",
    exp_name: str = "4k4d_prior_case",
    pretrained_repo: str = "facebook/VGGT-1B",
    pretrained_filename: str = "model.pt",
    max_epochs: int = 5,
    limit_train_batches: int = 100,
    limit_val_batches: int = 10,
    val_epoch_freq: int = 1,
    learning_rate: float = 1e-5,
    max_img_per_gpu: int = 13,
    fix_img_num: int = -1,
    img_nums_min: int = 7,
    img_nums_max: int = 13,
    len_train: int = 200,
    len_test: int = 20,
) -> None:
    cfg = TrainingConfig(
        case_subdirs=_split_csv_paths(case_subdirs),
        output_subdir=output_subdir,
        config_name=config_name,
        exp_name=exp_name,
        pretrained_repo=pretrained_repo,
        pretrained_filename=pretrained_filename,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        val_epoch_freq=val_epoch_freq,
        learning_rate=learning_rate,
        max_img_per_gpu=max_img_per_gpu,
        fix_img_num=fix_img_num,
        img_nums_min=img_nums_min,
        img_nums_max=img_nums_max,
        len_train=len_train,
        len_test=len_test,
    )
    print("[modal-train] launch config:")
    print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
    summary = run_remote_vggt_training.remote(cfg.to_json())
    print("[modal-train] remote summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if download_local_dir.strip():
        local_dir = Path(download_local_dir).expanduser().resolve()
        _download_volume_dir(summary["output_subdir"], local_dir)
        print(f"[modal-train] downloaded artifacts to {local_dir}")


@app.local_entrypoint()
def run_cases_from_local(
    local_case_dirs: str,
    remote_case_root: str = "training_cases",
    output_subdir: str = "",
    download_local_dir: str = "",
    config_name: str = "4k4d_prior_case",
    exp_name: str = "4k4d_prior_case",
    pretrained_repo: str = "facebook/VGGT-1B",
    pretrained_filename: str = "model.pt",
    max_epochs: int = 5,
    limit_train_batches: int = 100,
    limit_val_batches: int = 10,
    val_epoch_freq: int = 1,
    learning_rate: float = 1e-5,
    max_img_per_gpu: int = 13,
    fix_img_num: int = -1,
    img_nums_min: int = 7,
    img_nums_max: int = 13,
    len_train: int = 200,
    len_test: int = 20,
) -> None:
    local_dirs = [Path(item) for item in _split_csv_paths(local_case_dirs)]
    if not local_dirs:
        raise ValueError("Please provide at least one local case dir.")

    remote_case_root = _normalize_subpath(remote_case_root)
    remote_subdirs = []
    for local_dir in local_dirs:
        remote_subdirs.append(_upload_case_dir(local_dir, f"{remote_case_root}/{local_dir.name}"))

    cfg = TrainingConfig(
        case_subdirs=remote_subdirs,
        output_subdir=output_subdir,
        config_name=config_name,
        exp_name=exp_name,
        pretrained_repo=pretrained_repo,
        pretrained_filename=pretrained_filename,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        val_epoch_freq=val_epoch_freq,
        learning_rate=learning_rate,
        max_img_per_gpu=max_img_per_gpu,
        fix_img_num=fix_img_num,
        img_nums_min=img_nums_min,
        img_nums_max=img_nums_max,
        len_train=len_train,
        len_test=len_test,
    )
    print("[modal-train] upload+run config:")
    print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
    summary = run_remote_vggt_training.remote(cfg.to_json())
    print("[modal-train] remote summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if download_local_dir.strip():
        local_dir = Path(download_local_dir).expanduser().resolve()
    else:
        local_dir = REPO_ROOT / "output" / "modal_training_results" / Path(summary["output_subdir"]).name
    _download_volume_dir(summary["output_subdir"], local_dir)
    print(f"[modal-train] downloaded artifacts to {local_dir}")


@app.local_entrypoint()
def download_run(
    remote_output_subdir: str,
    local_output_dir: str,
) -> None:
    local_dir = Path(local_output_dir).expanduser().resolve()
    _download_volume_dir(remote_output_subdir, local_dir)
    print(f"[modal-train] downloaded artifacts to {local_dir}")
