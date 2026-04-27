from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modal_4k4d_vggt_infer import app, export_prediction_chunk_bytes_remote  # noqa: E402
from tools.subset_predictions_npz import load_subset_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download only the selected sparse-view prediction slices from a remote Modal "
            "predictions output and assemble them into a local subset predictions.npz."
        )
    )
    parser.add_argument("--remote-output-subdir", required=True)
    parser.add_argument("--subset-scene-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_or_fetch_view_chunk(
    *,
    remote_output_subdir: str,
    view_index: int,
    cache_dir: Path | None,
) -> dict[str, np.ndarray]:
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"view_{int(view_index):03d}.npz"
        if cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as payload:
                return {key: np.array(payload[key]) for key in payload.files}

    chunk_bytes = export_prediction_chunk_bytes_remote.remote(
        output_subdir=remote_output_subdir,
        start=int(view_index),
        end=int(view_index) + 1,
    )
    with np.load(BytesIO(chunk_bytes), allow_pickle=False) as payload:
        arrays = {key: np.array(payload[key]) for key in payload.files}
    if cache_path is not None:
        np.savez_compressed(cache_path, **arrays)
    return arrays


def main() -> int:
    args = parse_args()
    subset_scene_dir = Path(args.subset_scene_dir).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Re-run with --overwrite.")
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else None

    manifest, selected_indices, source_view_count = load_subset_manifest(subset_scene_dir)
    assembled: dict[str, list[np.ndarray] | np.ndarray] = {}
    subset_keys: list[str] = []
    passthrough_keys: list[str] = []

    with app.run():
        for offset, view_index in enumerate(selected_indices):
            chunk_arrays = _load_or_fetch_view_chunk(
                remote_output_subdir=str(args.remote_output_subdir),
                view_index=int(view_index),
                cache_dir=cache_dir,
            )
            for key, value in chunk_arrays.items():
                value = np.asarray(value)
                if value.ndim > 0 and int(value.shape[0]) == 1:
                    if key not in assembled:
                        assembled[key] = []
                        if key not in subset_keys:
                            subset_keys.append(key)
                    assert isinstance(assembled[key], list)
                    assembled[key].append(value)
                elif offset == 0:
                    assembled[key] = value
                    if key not in passthrough_keys:
                        passthrough_keys.append(key)

    final_arrays: dict[str, np.ndarray] = {}
    for key, value in assembled.items():
        if isinstance(value, list):
            final_arrays[key] = np.concatenate(value, axis=0)
        else:
            final_arrays[key] = value

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **final_arrays)

    summary = {
        "remote_output_subdir": str(args.remote_output_subdir),
        "subset_scene_dir": str(subset_scene_dir),
        "subset_camera_ids": [item.get("camera_id") for item in manifest.get("exported_views", [])],
        "selected_indices": selected_indices,
        "source_view_count": int(source_view_count),
        "subset_view_count": int(len(selected_indices)),
        "subset_keys": subset_keys,
        "passthrough_keys": passthrough_keys,
        "output_shapes": {key: list(np.asarray(value).shape) for key, value in final_arrays.items()},
        "output_path": str(output_path),
        "cache_dir": None if cache_dir is None else str(cache_dir),
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
