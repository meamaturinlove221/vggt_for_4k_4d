from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subset a predictions.npz payload to match a sparse-view scene manifest."
    )
    parser.add_argument(
        "--predictions-npz",
        required=True,
        help="Teacher predictions.npz aligned with the source scene view order.",
    )
    parser.add_argument(
        "--subset-scene-dir",
        required=True,
        help="Sparse-view scene directory containing scene_manifest.json with subset_selected_indices.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output path for the subset predictions .npz file.",
    )
    parser.add_argument(
        "--summary-path",
        default="",
        help="Optional JSON summary output path. Defaults to <output-path>.summary.json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output file.",
    )
    return parser.parse_args()


def load_subset_manifest(subset_scene_dir: Path) -> tuple[dict, list[int], int]:
    manifest_path = subset_scene_dir / "scene_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"scene_manifest.json not found under {subset_scene_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_indices = list(manifest.get("subset_selected_indices", []))
    if not selected_indices:
        raise ValueError(f"{manifest_path} does not contain subset_selected_indices")

    source_view_count = int(
        manifest.get("source_view_count")
        or manifest.get("full_view_count")
        or manifest.get("subset_full_view_count")
        or 0
    )

    def _scene_view_count(scene_dir_value: str) -> int:
        if not scene_dir_value:
            return 0
        source_manifest_path = Path(scene_dir_value) / "scene_manifest.json"
        if not source_manifest_path.is_file():
            return 0
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        return int(len(source_manifest.get("exported_views", [])))

    if source_view_count <= 0:
        source_view_count = _scene_view_count(str(manifest.get("source_scene_dir") or ""))
    if max(selected_indices) >= source_view_count:
        source_view_count = max(
            source_view_count,
            _scene_view_count(str(manifest.get("subset_from_scene") or "")),
        )
    if source_view_count <= 0:
        raise ValueError(f"Could not infer source view count for {subset_scene_dir}")

    if max(selected_indices) >= source_view_count:
        raise ValueError(
            f"subset_selected_indices contains {max(selected_indices)} but source view count is {source_view_count}"
        )
    return manifest, selected_indices, source_view_count


def main() -> int:
    args = parse_args()
    predictions_path = Path(args.predictions_npz).expanduser().resolve()
    subset_scene_dir = Path(args.subset_scene_dir).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    summary_path = (
        Path(args.summary_path).expanduser().resolve()
        if args.summary_path
        else output_path.with_suffix(output_path.suffix + ".summary.json")
    )

    if not predictions_path.is_file():
        raise FileNotFoundError(f"predictions.npz not found: {predictions_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Re-run with --overwrite.")

    manifest, selected_indices, source_view_count = load_subset_manifest(subset_scene_dir)
    selected_indices_array = np.asarray(selected_indices, dtype=np.int64)

    subset_arrays: dict[str, np.ndarray] = {}
    subset_keys: list[str] = []
    passthrough_keys: list[str] = []
    input_shapes: dict[str, list[int]] = {}
    output_shapes: dict[str, list[int]] = {}

    with np.load(predictions_path, allow_pickle=False) as payload:
        for key in payload.files:
            array = np.asarray(payload[key])
            input_shapes[key] = list(array.shape)
            if array.ndim > 0 and int(array.shape[0]) == source_view_count:
                subset_arrays[key] = array[selected_indices_array]
                subset_keys.append(key)
            else:
                subset_arrays[key] = array
                passthrough_keys.append(key)
            output_shapes[key] = list(np.asarray(subset_arrays[key]).shape)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **subset_arrays)

    summary = {
        "predictions_npz": str(predictions_path),
        "subset_scene_dir": str(subset_scene_dir),
        "subset_camera_ids": [item.get("camera_id") for item in manifest.get("exported_views", [])],
        "selected_indices": selected_indices,
        "source_view_count": source_view_count,
        "subset_view_count": int(len(selected_indices)),
        "subset_keys": subset_keys,
        "passthrough_keys": passthrough_keys,
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
        "output_path": str(output_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
