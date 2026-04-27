from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create sparse-view sub-scenes by subsetting an existing exported 4K4D scene."
    )
    parser.add_argument("--source-scene-dir", required=True, help="Existing exported scene directory, e.g. 60-view scene")
    parser.add_argument("--output-root", required=True, help="Output root for subset scenes")
    parser.add_argument(
        "--view-counts",
        nargs="+",
        type=int,
        default=[6, 8, 12],
        help="Total view counts including the target view",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("even", "coverage_desc"),
        default="even",
        help="How to choose source views from the existing scene",
    )
    parser.add_argument(
        "--subset-tag",
        default="localsubset",
        help="Tag appended to generated scene directory names",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing subset scene directories")
    return parser.parse_args()


def load_manifest(scene_dir: Path) -> dict:
    manifest_path = scene_dir / "scene_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"scene_manifest.json not found under {scene_dir}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_prior_payload(scene_dir: Path) -> dict[str, np.ndarray] | None:
    prior_path = scene_dir / "prior_maps.npz"
    if not prior_path.is_file():
        return None
    with np.load(prior_path, allow_pickle=False) as data:
        return {key: np.array(data[key]) for key in data.files}


def pick_subset_indices(exported_views: list[dict], total_views: int, selection_mode: str) -> list[int]:
    if total_views < 1:
        raise ValueError("total_views must be >= 1")
    if total_views > len(exported_views):
        raise ValueError(f"Requested {total_views} views from only {len(exported_views)} exported views")

    target_idx = None
    source_items: list[tuple[int, dict]] = []
    for idx, item in enumerate(exported_views):
        if item.get("role") == "tgt" and target_idx is None:
            target_idx = idx
        else:
            source_items.append((idx, item))
    if target_idx is None:
        raise ValueError("Could not find target view in scene manifest")

    num_sources = max(0, total_views - 1)
    if num_sources == 0:
        return [target_idx]
    if num_sources > len(source_items):
        raise ValueError(f"Requested {num_sources} source views but only {len(source_items)} are available")

    if selection_mode == "coverage_desc":
        ordered = sorted(
            source_items,
            key=lambda pair: float(pair[1].get("mask_coverage", 0.0)),
            reverse=True,
        )
        chosen = [idx for idx, _ in ordered[:num_sources]]
        return [target_idx] + chosen

    if num_sources == 1:
        return [target_idx, source_items[0][0]]

    positions = np.linspace(0, len(source_items) - 1, num=num_sources)
    chosen_indices: list[int] = []
    used: set[int] = set()
    for pos in positions:
        candidate_rank = int(round(float(pos)))
        candidate_rank = max(0, min(candidate_rank, len(source_items) - 1))
        if candidate_rank not in used:
            used.add(candidate_rank)
            chosen_indices.append(source_items[candidate_rank][0])
            continue
        for delta in range(1, len(source_items)):
            for direction in (-1, 1):
                alt_rank = candidate_rank + direction * delta
                if 0 <= alt_rank < len(source_items) and alt_rank not in used:
                    used.add(alt_rank)
                    chosen_indices.append(source_items[alt_rank][0])
                    break
            else:
                continue
            break
    chosen_indices = chosen_indices[:num_sources]
    return [target_idx] + sorted(chosen_indices)


def build_scene_name(scene_dir: Path, total_views: int, subset_tag: str) -> str:
    parts = scene_dir.name.split("_")
    if len(parts) >= 3 and parts[-1].endswith("views"):
        return "_".join(parts[:-1] + [f"{int(total_views)}views_{subset_tag}"])
    return f"{scene_dir.name}_{int(total_views)}views_{subset_tag}"


def copy_selected_images(dst_dir: Path, selected_views: list[dict]) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in selected_views:
        src_path = Path(item["image_path"])
        shutil.copy2(src_path, dst_dir / src_path.name)


def copy_selected_masks(dst_dir: Path, selected_views: list[dict]) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in selected_views:
        src_path = Path(item["mask_path"])
        shutil.copy2(src_path, dst_dir / src_path.name)


def save_contact_sheet(image_paths: list[Path], output_path: Path, thumb_size: int = 256) -> None:
    if not image_paths:
        return
    images = [Image.open(path).convert("RGB") for path in image_paths]
    cols = min(4, len(images))
    rows = math.ceil(len(images) / cols)
    canvas = Image.new("RGB", (cols * thumb_size, rows * thumb_size), color=(255, 255, 255))
    for idx, image in enumerate(images):
        tile = ImageOps.fit(image, (thumb_size, thumb_size), method=Image.Resampling.BILINEAR)
        canvas.paste(tile, ((idx % cols) * thumb_size, (idx // cols) * thumb_size))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_subset_manifest(
    dst_scene_dir: Path,
    source_scene_dir: Path,
    src_manifest: dict,
    selected_indices: list[int],
    selected_views: list[dict],
    selection_mode: str,
) -> None:
    manifest = dict(src_manifest)
    manifest["source_scene_dir"] = str(source_scene_dir)
    manifest["subset_from_scene"] = str(source_scene_dir)
    manifest["subset_selection_mode"] = selection_mode
    manifest["subset_selected_indices"] = selected_indices
    manifest["exported_views"] = selected_views
    manifest["source_cameras"] = [item["camera_id"] for item in selected_views if item.get("role") == "src"]
    manifest["camera_summary"] = {
        **dict(src_manifest.get("camera_summary", {})),
        "camera_ids": [item["camera_id"] for item in selected_views],
        "camera_count": len(selected_views),
    }
    manifest["subset_view_count"] = len(selected_views)
    if "prior_maps_file" in manifest:
        manifest["prior_maps_file"] = str((dst_scene_dir / "prior_maps.npz").resolve())

    prior_input_meta = manifest.get("prior_input_meta")
    if isinstance(prior_input_meta, dict):
        smplx_vertex_feature_meta = prior_input_meta.get("smplx_vertex_feature_meta")
        if isinstance(smplx_vertex_feature_meta, dict):
            src_views = smplx_vertex_feature_meta.get("views")
            if isinstance(src_views, list):
                smplx_vertex_feature_meta["views"] = [src_views[idx] for idx in selected_indices if idx < len(src_views)]

    (dst_scene_dir / "scene_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def slice_prior_payload(prior_payload: dict[str, np.ndarray], selected_indices: list[int]) -> dict[str, np.ndarray]:
    subset: dict[str, np.ndarray] = {}
    for key, value in prior_payload.items():
        array = np.asarray(value)
        if array.ndim > 0 and array.shape[0] >= max(selected_indices) + 1:
            subset[key] = array[selected_indices]
        else:
            subset[key] = array
    return subset


def main() -> int:
    args = parse_args()
    source_scene_dir = Path(args.source_scene_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    src_manifest = load_manifest(source_scene_dir)
    exported_views = list(src_manifest["exported_views"])
    prior_payload = load_prior_payload(source_scene_dir)

    summary_entries: list[dict[str, object]] = []
    for total_views in args.view_counts:
        selected_indices = pick_subset_indices(exported_views, total_views=total_views, selection_mode=args.selection_mode)
        selected_views = [dict(exported_views[idx]) for idx in selected_indices]

        dst_scene_dir = output_root / build_scene_name(source_scene_dir, total_views=total_views, subset_tag=args.subset_tag)
        if dst_scene_dir.exists():
            if not args.overwrite:
                raise FileExistsError(f"{dst_scene_dir} already exists. Re-run with --overwrite.")
            shutil.rmtree(dst_scene_dir)
        dst_scene_dir.mkdir(parents=True, exist_ok=True)

        copy_selected_images(dst_scene_dir / "images", selected_views)
        copy_selected_masks(dst_scene_dir / "masks", selected_views)

        image_paths = [dst_scene_dir / "images" / Path(item["image_path"]).name for item in selected_views]
        mask_paths = [dst_scene_dir / "masks" / Path(item["mask_path"]).name for item in selected_views]
        save_contact_sheet(image_paths, dst_scene_dir / "rgb_contact_sheet.png")
        save_contact_sheet(mask_paths, dst_scene_dir / "mask_contact_sheet.png")

        if prior_payload is not None:
            subset_prior = slice_prior_payload(prior_payload, selected_indices)
            np.savez_compressed(dst_scene_dir / "prior_maps.npz", **subset_prior)

        for local_rank, item in enumerate(selected_views):
            image_name = Path(item["image_path"]).name
            mask_name = Path(item["mask_path"]).name
            item["subset_rank"] = local_rank
            item["image_path"] = str((dst_scene_dir / "images" / image_name).resolve())
            item["mask_path"] = str((dst_scene_dir / "masks" / mask_name).resolve())

        write_subset_manifest(
            dst_scene_dir=dst_scene_dir,
            source_scene_dir=source_scene_dir,
            src_manifest=src_manifest,
            selected_indices=selected_indices,
            selected_views=selected_views,
            selection_mode=args.selection_mode,
        )

        summary_entries.append(
            {
                "view_count": int(total_views),
                "scene_dir": str(dst_scene_dir),
                "selected_indices": selected_indices,
                "selected_camera_ids": [item["camera_id"] for item in selected_views],
                "selection_mode": args.selection_mode,
            }
        )

    summary = {
        "source_scene_dir": str(source_scene_dir),
        "output_root": str(output_root),
        "view_counts": [int(v) for v in args.view_counts],
        "selection_mode": args.selection_mode,
        "entries": summary_entries,
    }
    summary_path = output_root / f"{source_scene_dir.name}_subset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
