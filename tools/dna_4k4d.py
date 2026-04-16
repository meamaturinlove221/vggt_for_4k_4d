from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import sys
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


SUBSET_NAME = "data_used_in_4K4D"
FILE_GID_MEMBER = f"{SUBSET_NAME}/data_used_in_4K4D_file_gid.json"
INNER_RGB_CAMS_MEMBER = f"{SUBSET_NAME}/data_used_in_4K4D_rgb_cams.zip"


@dataclass
class DatasetContext:
    dataset_path: Path
    dataset_dir: Path
    subset_name: str
    subset_roots: list[Path]
    outer_zips: list[Path]
    zip_index: dict[str, list[Path]]


def canonical_path(path: str | Path) -> str:
    return str(path).replace("\\", "/").lstrip("/")


def normalize_camera_id(camera_id: str | int) -> str:
    return f"{int(camera_id):02d}"


def sort_numeric(values) -> list[str]:
    def key_fn(value: str):
        if value.isdigit():
            return (0, int(value), value)
        match = re.search(r"(\d+)", value)
        if match:
            return (0, int(match.group(1)), value)
        return (1, 0, value)

    return sorted(list(values), key=key_fn)


def find_outer_zips(dataset_dir: Path, subset_name: str) -> list[Path]:
    return sorted(dataset_dir.glob(f"{subset_name}-*.zip"))


def detect_zip_part_gaps(zip_paths: list[Path]) -> list[int]:
    parts = []
    for zip_path in zip_paths:
        match = re.search(r"-(\d{3})\.zip$", zip_path.name)
        if match:
            parts.append(int(match.group(1)))
    if not parts:
        return []
    present = set(parts)
    return [idx for idx in range(min(parts), max(parts) + 1) if idx not in present]


def find_subset_roots(dataset_path: Path, subset_name: str) -> list[Path]:
    roots: list[Path] = []
    if dataset_path.name == subset_name and dataset_path.is_dir():
        roots.append(dataset_path.resolve())
    direct_child = dataset_path / subset_name
    if direct_child.is_dir():
        roots.append(direct_child.resolve())
    for candidate in dataset_path.rglob(subset_name):
        if candidate.is_dir():
            resolved = candidate.resolve()
            if resolved not in roots:
                roots.append(resolved)
    return roots


def build_zip_index(zip_paths: list[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                index[canonical_path(info.filename)].append(zip_path)
    return dict(index)


def build_context(dataset_path: Path, subset_name: str) -> DatasetContext:
    dataset_path = dataset_path.resolve()
    dataset_dir = dataset_path.parent if dataset_path.name == subset_name else dataset_path
    subset_roots = find_subset_roots(dataset_path if dataset_path.name == subset_name else dataset_dir, subset_name)
    return DatasetContext(
        dataset_path=dataset_path,
        dataset_dir=dataset_dir,
        subset_name=subset_name,
        subset_roots=subset_roots,
        outer_zips=find_outer_zips(dataset_dir, subset_name),
        zip_index=build_zip_index(find_outer_zips(dataset_dir, subset_name)),
    )


def locate_extracted_file(context: DatasetContext, canonical: str) -> Path | None:
    relative = Path(canonical).relative_to(context.subset_name)
    for subset_root in context.subset_roots:
        candidate = (subset_root / relative).resolve()
        if candidate.is_file():
            return candidate
    return None


def describe_expected_file(context: DatasetContext, canonical: str) -> dict[str, object]:
    extracted = locate_extracted_file(context, canonical)
    if extracted is not None:
        return {"status": "extracted", "path": str(extracted), "archives": []}
    archives = [str(path) for path in context.zip_index.get(canonical, [])]
    if archives:
        return {"status": "archived", "path": None, "archives": archives}
    return {"status": "missing", "path": None, "archives": []}


def load_file_gid_map(context: DatasetContext) -> tuple[dict[str, str], str]:
    for subset_root in context.subset_roots:
        candidate = subset_root / "data_used_in_4K4D_file_gid.json"
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as handle:
                return json.load(handle), str(candidate)
    for zip_path in context.outer_zips:
        with zipfile.ZipFile(zip_path) as archive:
            try:
                with archive.open(FILE_GID_MEMBER) as handle:
                    return json.load(io.TextIOWrapper(handle, encoding="utf-8")), f"{zip_path}!/{FILE_GID_MEMBER}"
            except KeyError:
                continue
    raise FileNotFoundError(f"Could not find {FILE_GID_MEMBER}.")


def category_for(canonical: str, subset_name: str) -> str:
    relative = Path(canonical).relative_to(subset_name)
    parts = relative.parts
    if not parts:
        return "root"
    if parts[0] == "apose" and len(parts) >= 2:
        return f"apose/{parts[1]}"
    return parts[0]


def download_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?id={file_id}&export=download"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_member(archive: zipfile.ZipFile, member_name: str, output_path: Path) -> None:
    ensure_parent(output_path)
    with archive.open(member_name) as src, output_path.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)


def materialize_archived_member(context: DatasetContext, canonical: str, temp_dir: Path) -> Path | None:
    archives = context.zip_index.get(canonical, [])
    if not archives:
        return None
    archive_path = archives[0]
    output_path = temp_dir / Path(canonical).name
    with zipfile.ZipFile(archive_path) as archive:
        copy_member(archive, canonical, output_path)
    return output_path


def materialize_rgb_cams_smc(context: DatasetContext, sequence_id: str, temp_dir: Path) -> tuple[Path | None, str]:
    extracted = locate_extracted_file(context, f"{context.subset_name}/rgb_cams/{sequence_id}_rgb_cams.smc")
    if extracted is not None:
        return extracted, "extracted"
    for subset_root in context.subset_roots:
        inner_zip = subset_root / "data_used_in_4K4D_rgb_cams.zip"
        if inner_zip.is_file():
            with zipfile.ZipFile(inner_zip) as archive:
                member = f"{sequence_id}_rgb_cams.smc"
                if member in archive.namelist():
                    output_path = temp_dir / member
                    copy_member(archive, member, output_path)
                    return output_path, f"{inner_zip}!/{member}"
    for zip_path in context.zip_index.get(INNER_RGB_CAMS_MEMBER, []):
        with zipfile.ZipFile(zip_path) as outer:
            with outer.open(INNER_RGB_CAMS_MEMBER) as handle:
                with zipfile.ZipFile(io.BytesIO(handle.read())) as inner:
                    member = f"{sequence_id}_rgb_cams.smc"
                    if member in inner.namelist():
                        output_path = temp_dir / member
                        copy_member(inner, member, output_path)
                        return output_path, f"{zip_path}!/{INNER_RGB_CAMS_MEMBER}!/{member}"
    return None, "missing"


def require_h5py():
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise SystemExit("h5py is required for the manifest command.") from exc
    return h5py


def load_camera_summary(smc_path: Path) -> dict[str, object]:
    h5py = require_h5py()
    with h5py.File(smc_path, "r") as handle:
        if "Camera_Parameter" not in handle:
            raise ValueError(f"{smc_path} does not contain Camera_Parameter.")
        camera_ids = [normalize_camera_id(camera_id) for camera_id in sort_numeric(handle["Camera_Parameter"].keys())]
        sample = handle["Camera_Parameter"][sort_numeric(handle["Camera_Parameter"].keys())[0]]
        matrix_shapes = {}
        for key in ("K", "D", "RT"):
            if key in sample:
                matrix_shapes[key] = list(sample[key][()].shape)
    return {
        "camera_ids": camera_ids,
        "camera_count": len(camera_ids),
        "matrix_shapes": matrix_shapes,
    }


def probe_smc_root(smc_path: Path) -> dict[str, object]:
    h5py = require_h5py()
    summary: dict[str, object] = {"root_keys": [], "groups": {}}
    with h5py.File(smc_path, "r") as handle:
        summary["root_keys"] = sort_numeric(handle.keys())
        for key in summary["root_keys"]:
            if hasattr(handle[key], "keys"):
                summary["groups"][key] = sort_numeric(handle[key].keys())[:20]
    return summary


def auto_pick_sources(cameras: list[str], target_camera: str, count: int) -> list[str]:
    candidates = [camera for camera in cameras if camera != target_camera]
    if count <= 0:
        return []
    if count >= len(candidates):
        return candidates
    step = max(1, len(candidates) // count)
    picked = candidates[::step][:count]
    for candidate in candidates:
        if len(picked) >= count:
            break
        if candidate not in picked:
            picked.append(candidate)
    return picked


def command_inventory(args: argparse.Namespace) -> int:
    context = build_context(Path(args.dataset_path), args.subset_name)
    file_gid_map, source = load_file_gid_map(context)
    expected = sort_numeric(file_gid_map.keys())
    categories: dict[str, dict[str, int]] = defaultdict(lambda: {"expected": 0, "local": 0, "missing": 0})
    missing_files = []
    local_count = 0
    for canonical in expected:
        category = category_for(canonical, args.subset_name)
        categories[category]["expected"] += 1
        status = describe_expected_file(context, canonical)
        if status["status"] == "missing":
            categories[category]["missing"] += 1
            missing_files.append(
                {
                    "canonical_path": canonical,
                    "category": category,
                    "google_drive_id": file_gid_map[canonical],
                    "download_url": download_url(file_gid_map[canonical]),
                }
            )
        else:
            categories[category]["local"] += 1
            local_count += 1
    report = {
        "dataset_path": str(context.dataset_path),
        "subset_roots": [str(root) for root in context.subset_roots],
        "outer_zip_parts": [str(path) for path in context.outer_zips],
        "heuristic_missing_zip_parts": detect_zip_part_gaps(context.outer_zips),
        "file_gid_source": source,
        "expected_file_count": len(expected),
        "local_expected_file_count": local_count,
        "missing_expected_file_count": len(missing_files),
        "categories": dict(categories),
        "missing_files": missing_files,
    }
    print(f"Dataset path: {report['dataset_path']}")
    print(f"Subset roots found: {len(report['subset_roots'])}")
    for root in report["subset_roots"]:
        print(f"  - {root}")
    print(f"Outer zip parts found: {len(report['outer_zip_parts'])}")
    for zip_path in report["outer_zip_parts"]:
        print(f"  - {zip_path}")
    gaps = report["heuristic_missing_zip_parts"]
    if gaps:
        print("Heuristic missing zip parts:", ", ".join(f"{gap:03d}" for gap in gaps))
    print(f"Expected canonical files: {report['expected_file_count']}")
    print(f"Local canonical files: {report['local_expected_file_count']}")
    print(f"Missing canonical files: {report['missing_expected_file_count']}")
    print("Category summary:")
    for category, counts in sorted(report["categories"].items()):
        print(f"  - {category}: expected={counts['expected']} local={counts['local']} missing={counts['missing']}")
    if missing_files:
        print("Missing files:")
        for item in missing_files:
            print(f"  - {item['canonical_path']}")
            print(f"    id={item['google_drive_id']}")
            print(f"    url={item['download_url']}")
    if args.json_out:
        output_path = Path(args.json_out).resolve()
        ensure_parent(output_path)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        print(f"Report written to {output_path}")
    return 0


def command_assemble(args: argparse.Namespace) -> int:
    context = build_context(Path(args.dataset_path), args.subset_name)
    if not context.outer_zips:
        raise SystemExit("No outer zip parts were found.")
    target_root = Path(args.target_root).resolve()
    extracted = 0
    skipped = 0
    for zip_path in context.outer_zips:
        print(f"Extracting {zip_path.name} -> {target_root}")
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                output_path = target_root / canonical_path(info.filename)
                if args.skip_existing and output_path.exists() and output_path.stat().st_size == info.file_size:
                    skipped += 1
                    continue
                copy_member(archive, info.filename, output_path)
                extracted += 1
    if args.extract_inner_rgb_cams:
        inner_zip = target_root / args.subset_name / "data_used_in_4K4D_rgb_cams.zip"
        if inner_zip.is_file():
            with zipfile.ZipFile(inner_zip) as archive:
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    output_path = target_root / args.subset_name / "rgb_cams" / Path(info.filename).name
                    if args.skip_existing and output_path.exists() and output_path.stat().st_size == info.file_size:
                        skipped += 1
                        continue
                    copy_member(archive, info.filename, output_path)
                    extracted += 1
    print(f"Assemble complete. extracted={extracted}, skipped={skipped}")
    return 0


def command_manifest(args: argparse.Namespace) -> int:
    context = build_context(Path(args.dataset_path), args.subset_name)
    temp_handle = tempfile.TemporaryDirectory(prefix="dna_4k4d_")
    temp_dir = Path(temp_handle.name)
    try:
        seq = args.seq
        main_canonical = f"{args.subset_name}/main/{seq}.smc"
        ann_canonical = f"{args.subset_name}/annotations/{seq}_annots.smc"
        kinect_canonical = f"{args.subset_name}/kinect/{seq}_kinect.smc"
        preview_canonical = f"{args.subset_name}/preview/{seq}.mp4"
        files = {
            "main_smc": describe_expected_file(context, main_canonical),
            "annotations_smc": describe_expected_file(context, ann_canonical),
            "kinect_smc": describe_expected_file(context, kinect_canonical),
            "preview_mp4": describe_expected_file(context, preview_canonical),
        }
        rgb_cams_path, rgb_cams_source = materialize_rgb_cams_smc(context, seq, temp_dir)
        camera_summary = load_camera_summary(rgb_cams_path) if rgb_cams_path is not None else None
        available_cameras = list(camera_summary["camera_ids"]) if camera_summary else []
        target_camera = normalize_camera_id(args.target_camera) if args.target_camera else (available_cameras[0] if available_cameras else None)
        source_cameras = [normalize_camera_id(camera) for camera in args.source_cameras] if args.source_cameras else []
        if not source_cameras and target_camera and available_cameras:
            source_cameras = auto_pick_sources(available_cameras, target_camera, args.auto_sources)
        main_probe = None
        ann_probe = None
        if args.materialize_archived and files["main_smc"]["status"] == "archived":
            materialized = materialize_archived_member(context, main_canonical, temp_dir)
            if materialized is not None:
                files["main_smc"]["status"] = "materialized"
                files["main_smc"]["path"] = str(materialized)
                main_probe = probe_smc_root(materialized)
        elif files["main_smc"]["status"] == "extracted":
            main_probe = probe_smc_root(Path(files["main_smc"]["path"]))
        if args.materialize_archived and files["annotations_smc"]["status"] == "archived":
            materialized = materialize_archived_member(context, ann_canonical, temp_dir)
            if materialized is not None:
                files["annotations_smc"]["status"] = "materialized"
                files["annotations_smc"]["path"] = str(materialized)
                ann_probe = probe_smc_root(materialized)
        elif files["annotations_smc"]["status"] == "extracted":
            ann_probe = probe_smc_root(Path(files["annotations_smc"]["path"]))
        blocking = []
        if files["main_smc"]["status"] == "missing":
            blocking.append(f"Missing {main_canonical}; RGB bridge is blocked.")
        if files["annotations_smc"]["status"] == "missing":
            blocking.append(f"Missing {ann_canonical}; mask bridge is blocked.")
        if camera_summary is None:
            blocking.append("Missing rgb_cams camera parameter SMC.")
        invalid_sources = [camera for camera in source_cameras if camera not in available_cameras]
        if invalid_sources:
            blocking.append(f"Invalid source cameras: {', '.join(invalid_sources)}")
        manifest = {
            "dataset_path": str(context.dataset_path),
            "subset_name": args.subset_name,
            "seq_id": seq,
            "frame_id": str(int(args.frame)),
            "target_camera": target_camera,
            "source_cameras": source_cameras,
            "files": {
                **files,
                "rgb_cams_smc": {
                    "status": "materialized" if rgb_cams_path is not None else "missing",
                    "path": str(rgb_cams_path) if rgb_cams_path is not None else None,
                    "source": rgb_cams_source,
                },
            },
            "camera_summary": camera_summary,
            "probes": {
                "main_smc": main_probe,
                "annotations_smc": ann_probe,
            },
            "status": {
                "ready_for_rgb_bridge": files["main_smc"]["status"] != "missing" and camera_summary is not None,
                "ready_for_mask_bridge": files["annotations_smc"]["status"] != "missing" and camera_summary is not None,
                "needs_main_download": files["main_smc"]["status"] == "missing",
            },
            "blocking_issues": blocking,
        }
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / f"{seq}_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        print(f"Manifest written to {manifest_path}")
        if blocking and not args.allow_partial:
            raise SystemExit("Manifest generated, but the case is still blocked. Re-run with --allow-partial to keep this as a successful preflight.")
        return 0
    finally:
        temp_handle.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inventory and bridge helper for DNA-Rendering 4K4D subset.")
    parser.add_argument("--subset-name", default=SUBSET_NAME, help="Subset name. Default: data_used_in_4K4D")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="Scan raw zip parts and extracted roots.")
    inventory.add_argument("--dataset-path", required=True, help="Folder containing the raw dataset zips or an extracted subset root.")
    inventory.add_argument("--json-out", help="Optional JSON report output path.")
    inventory.set_defaults(func=command_inventory)

    assemble = subparsers.add_parser("assemble", help="Merge all currently available outer zip parts.")
    assemble.add_argument("--dataset-path", required=True, help="Folder containing the raw dataset zip parts.")
    assemble.add_argument("--target-root", required=True, help="Destination parent folder for extracted files.")
    assemble.add_argument("--skip-existing", action="store_true", help="Skip files that already exist with the same size.")
    assemble.add_argument("--extract-inner-rgb-cams", action="store_true", help="Also extract the embedded rgb_cams zip into rgb_cams/*.smc.")
    assemble.set_defaults(func=command_assemble)

    manifest = subparsers.add_parser("manifest", help="Build one-sequence bridge manifest.")
    manifest.add_argument("--dataset-path", required=True, help="Folder containing the raw dataset zips or an extracted subset root.")
    manifest.add_argument("--seq", required=True, help="Sequence id such as 0012_11.")
    manifest.add_argument("--frame", default="0", help="Frame id to record in the manifest. Default: 0")
    manifest.add_argument("--target-camera", help="Target camera id. Example: 00")
    manifest.add_argument("--source-cameras", nargs="*", default=[], help="Explicit source camera ids.")
    manifest.add_argument("--auto-sources", type=int, default=6, help="Auto-pick N source cameras if --source-cameras is omitted.")
    manifest.add_argument("--materialize-archived", action="store_true", help="Temporarily extract archived main/annotation SMCs for one-case probing.")
    manifest.add_argument("--allow-partial", action="store_true", help="Exit successfully even if the case is still blocked.")
    manifest.add_argument("--output-dir", required=True, help="Output directory for the manifest JSON.")
    manifest.set_defaults(func=command_manifest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
