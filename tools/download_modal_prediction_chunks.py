from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reliably download Modal prediction chunks via `modal volume get`, "
            "validate each chunk locally, and reassemble predictions.npz."
        )
    )
    parser.add_argument("--remote-output-subdir", required=True, help="Remote output subdir under the Modal output volume.")
    parser.add_argument("--local-output-dir", required=True, help="Local directory to store summary, chunks, and predictions.npz.")
    parser.add_argument("--volume-name", default="vggt-4k4d-output", help="Modal volume name. Default: vggt-4k4d-output")
    parser.add_argument("--modal-exe", default="modal", help="Path to modal executable. Default: modal")
    parser.add_argument("--max-retries", type=int, default=8, help="Retries per downloaded file before giving up.")
    parser.add_argument("--retry-sleep-sec", type=float, default=2.0, help="Sleep between retries in seconds.")
    parser.add_argument("--overwrite", action="store_true", help="Delete the local output dir before downloading.")
    return parser.parse_args()


def run_modal_get(modal_exe: str, volume_name: str, remote_path: str, local_path: Path) -> tuple[bool, str]:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [modal_exe, "volume", "get", volume_name, remote_path, str(local_path), "--force"]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    output = "\n".join(part for part in [proc.stdout.strip(), proc.stderr.strip()] if part)
    return proc.returncode == 0, output


def validate_json(path: Path) -> None:
    json.loads(path.read_text(encoding="utf-8"))


def validate_npz(path: Path) -> None:
    with np.load(path, allow_pickle=False) as payload:
        for key in payload.files:
            _ = payload[key].shape


def download_with_retry(
    *,
    modal_exe: str,
    volume_name: str,
    remote_path: str,
    local_path: Path,
    validator,
    max_retries: int,
    retry_sleep_sec: float,
) -> dict[str, object]:
    attempts: list[dict[str, object]] = []
    for attempt_idx in range(1, max_retries + 1):
        if local_path.exists():
            local_path.unlink()
        ok, output = run_modal_get(modal_exe, volume_name, remote_path, local_path)
        record = {"attempt": attempt_idx, "success": bool(ok), "output": output}
        if ok:
            try:
                validator(local_path)
                record["validated"] = True
                attempts.append(record)
                return {
                    "path": str(local_path),
                    "attempts": attempts,
                    "success": True,
                }
            except Exception as exc:  # pragma: no cover - depends on flaky remote transfer
                record["validated"] = False
                record["validation_error"] = f"{type(exc).__name__}: {exc}"
        attempts.append(record)
        time.sleep(retry_sleep_sec)
    raise RuntimeError(
        f"Failed to download a valid file after {max_retries} retries: remote={remote_path} local={local_path}\n"
        f"attempts={json.dumps(attempts, ensure_ascii=False, indent=2)}"
    )


def reassemble_predictions(chunk_dir: Path, output_path: Path) -> dict[str, object]:
    manifest_path = chunk_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    arrays: dict[str, list[np.ndarray]] = {}
    for chunk_meta in manifest["chunks"]:
        chunk_path = chunk_dir / chunk_meta["file"]
        with np.load(chunk_path, allow_pickle=False) as loaded:
            for key in loaded.files:
                arrays.setdefault(key, []).append(np.asarray(loaded[key]))

    merged = {key: np.concatenate(values, axis=0) for key, values in arrays.items()}
    np.savez_compressed(output_path, **merged)
    with np.load(output_path, allow_pickle=False) as check:
        shapes = {key: list(np.asarray(check[key]).shape) for key in check.files}

    return {
        "output_path": str(output_path),
        "keys": sorted(merged.keys()),
        "shapes": shapes,
        "num_chunks": len(manifest["chunks"]),
    }


def main() -> int:
    args = parse_args()
    remote_output_subdir = args.remote_output_subdir.strip().strip("/").replace("\\", "/")
    local_output_dir = Path(args.local_output_dir).expanduser().resolve()

    if args.overwrite and local_output_dir.exists():
        shutil.rmtree(local_output_dir)
    local_output_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = local_output_dir / "predictions_chunks_v1"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    download_log: dict[str, object] = {
        "remote_output_subdir": remote_output_subdir,
        "local_output_dir": str(local_output_dir),
        "volume_name": args.volume_name,
        "files": [],
    }

    for remote_name, validator in [("summary.json", validate_json), ("predictions_chunks_v1/manifest.json", validate_json)]:
        remote_path = f"{remote_output_subdir}/{remote_name}"
        local_path = local_output_dir / remote_name
        result = download_with_retry(
            modal_exe=args.modal_exe,
            volume_name=args.volume_name,
            remote_path=remote_path,
            local_path=local_path,
            validator=validator,
            max_retries=int(args.max_retries),
            retry_sleep_sec=float(args.retry_sleep_sec),
        )
        download_log["files"].append({"remote_path": remote_path, **result})

    manifest = json.loads((chunk_dir / "manifest.json").read_text(encoding="utf-8"))
    for chunk_meta in manifest["chunks"]:
        remote_path = f"{remote_output_subdir}/predictions_chunks_v1/{chunk_meta['file']}"
        local_path = chunk_dir / chunk_meta["file"]
        result = download_with_retry(
            modal_exe=args.modal_exe,
            volume_name=args.volume_name,
            remote_path=remote_path,
            local_path=local_path,
            validator=validate_npz,
            max_retries=int(args.max_retries),
            retry_sleep_sec=float(args.retry_sleep_sec),
        )
        download_log["files"].append({"remote_path": remote_path, **result})

    assembly = reassemble_predictions(chunk_dir, local_output_dir / "predictions.npz")
    download_log["assembly"] = assembly

    log_path = local_output_dir / "download_log.json"
    log_path.write_text(json.dumps(download_log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(download_log, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
