from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.import_external_smplx_params import (  # noqa: E402
    CAMERA_KEY_ALIASES,
    SMPLX_KEY_ALIASES,
    _align_camera_order,
    _load_npz_or_json,
    _normalize_camera_id,
    _normalize_smplx_params,
    _parse_camera_payload,
    _save_camera_npz,
    _save_smplx_npz,
)


MANIFEST_NAME = "external_prior_bundle_manifest.json"
MODE_CHOICES = ("external-regressor-json-npz", "fitting-result", "estimator-command")
PAYLOAD_SUFFIXES = {".json", ".npz"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

SMPLX_CONTAINER_KEYS = (
    "smplx",
    "smplx_params",
    "body_params",
    "body",
    "fit",
    "fit_result",
    "fitting_result",
    "result",
)
CAMERA_CONTAINER_KEYS = (
    "cameras",
    "camera",
    "camera_params",
    "cam_params",
    "camera_result",
    "calibration",
)
VIEW_CONTAINER_KEYS = ("views", "frames", "observations")
MASK_CONTAINER_KEYS = ("masks", "mask_paths", "mask_files", "silhouettes", "segmentation")
IMAGE_CONTAINER_KEYS = ("images", "image_paths", "image_files", "rgbs", "rgb_paths")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Real-data SMPL-X adapter driver. "
            "It can run an external estimator command, imports external regressor/fitting outputs, "
            "validates cameras/images/masks, "
            "and writes an import_external_smplx_params-compatible prior bundle manifest."
        )
    )
    parser.add_argument("--mode", required=True, choices=MODE_CHOICES, help="Import mode.")
    parser.add_argument("--input-path", required=True, help="External output directory or combined result file.")
    parser.add_argument("--scene-dir", help="Scene directory containing scene_manifest.json.")
    parser.add_argument("--scene-manifest", help="Explicit path to scene_manifest.json.")
    parser.add_argument("--smplx-input", help="Explicit SMPL-X json/npz input.")
    parser.add_argument("--camera-input", help="Explicit camera json/npz input.")
    parser.add_argument("--mask-input", help="Optional external mask directory/json/npz manifest.")
    parser.add_argument("--image-input", help="Optional external image directory/json/npz manifest.")
    parser.add_argument(
        "--estimator-command",
        help=(
            "Command to run when --mode estimator-command is selected. "
            "Placeholders are available: {scene_dir}, {scene_manifest}, {input_path}, {estimator_output_dir}."
        ),
    )
    parser.add_argument(
        "--estimator-cwd",
        help="Optional working directory for --estimator-command. Defaults to the repository root.",
    )
    parser.add_argument(
        "--estimator-output-dir",
        help="Directory where the estimator command writes SMPL-X/camera outputs. Defaults to --input-path.",
    )
    parser.add_argument(
        "--estimator-timeout-sec",
        type=int,
        default=7200,
        help="Timeout for --estimator-command.",
    )
    parser.add_argument(
        "--skip-estimator-run",
        action="store_true",
        help="With --mode estimator-command, skip running the command and only import existing outputs.",
    )
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame index for time-major payloads.")
    parser.add_argument("--output-dir", required=True, help="Output directory for the normalized bundle.")
    parser.add_argument("--strict", action="store_true", help="Fail on missing or ambiguous inputs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output files if they already exist.")
    return parser.parse_args()


def _to_python(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype == object and value.ndim == 0:
            return _to_python(value.item())
        return value
    if isinstance(value, dict):
        return {str(key): _to_python(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_python(item) for item in value]
    return value


def _load_payload(path: Path) -> Any:
    return _to_python(_load_npz_or_json(path))


def _is_mapping(value: Any) -> bool:
    return isinstance(value, dict)


def _has_smplx_signal(payload: Any) -> bool:
    if not _is_mapping(payload):
        return False
    keys = set(payload.keys())
    for canonical, aliases in SMPLX_KEY_ALIASES.items():
        if canonical in keys or any(alias in keys for alias in aliases):
            return True
    return False


def _has_camera_signal(payload: Any) -> bool:
    if not _is_mapping(payload):
        return False
    if "cameras" in payload and isinstance(payload["cameras"], dict):
        return True
    keys = set(payload.keys())
    for canonical, aliases in CAMERA_KEY_ALIASES.items():
        if canonical in keys or any(alias in keys for alias in aliases):
            return True
    return False


def _resolve_path_like(value: Any, *, base_dir: Path) -> Path:
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str):
        path = Path(value)
    else:
        raise TypeError(f"Unsupported path-like value: {type(value)!r}")
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def _probe_image_hw(path: Path) -> list[int]:
    with Image.open(path) as image:
        width, height = image.size
    return [int(height), int(width)]


def _build_source_ref(path: Path | None, *, nested_key: str | None = None) -> str | None:
    if path is None:
        return None
    if nested_key:
        return f"{path.resolve()}#{nested_key}"
    return str(path.resolve())


def _payload_stats(payload: dict[str, np.ndarray]) -> dict[str, list[int]]:
    stats: dict[str, list[int]] = {}
    for key, value in payload.items():
        stats[key] = list(np.asarray(value).shape)
    return stats


def _resolve_scene_manifest(scene_dir: Path | None, scene_manifest: Path | None) -> tuple[dict[str, Any] | None, Path | None, Path | None]:
    if scene_manifest is None and scene_dir is not None:
        scene_manifest = scene_dir / "scene_manifest.json"
    if scene_manifest is None:
        return None, None, scene_dir

    manifest_path = scene_manifest.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"scene_manifest.json not found at {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved_scene_dir = scene_dir.resolve() if scene_dir is not None else manifest_path.parent
    return manifest, manifest_path, resolved_scene_dir


def _collect_scene_views(scene_manifest: dict[str, Any], *, base_dir: Path) -> list[dict[str, Any]]:
    exported_views = scene_manifest.get("exported_views", [])
    if not isinstance(exported_views, list):
        raise ValueError("scene_manifest.exported_views must be a list.")

    views: list[dict[str, Any]] = []
    for index, view in enumerate(exported_views):
        if not isinstance(view, dict):
            raise ValueError(f"scene_manifest.exported_views[{index}] is not a mapping.")
        if "camera_id" not in view:
            raise ValueError(f"scene_manifest.exported_views[{index}] is missing camera_id.")
        views.append(
            {
                "index": index,
                "camera_id": _normalize_camera_id(view["camera_id"]),
                "role": view.get("role"),
                "image_path": _resolve_path_like(view["image_path"], base_dir=base_dir),
                "mask_path": _resolve_path_like(view["mask_path"], base_dir=base_dir),
                "declared_image_size": view.get("image_size"),
            }
        )
    return views


def _validate_scene_view_assets(scene_views: list[dict[str, Any]], *, strict: bool) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for view in scene_views:
        image_path = Path(view["image_path"])
        mask_path = Path(view["mask_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Scene image not found: {image_path}")
        if not mask_path.is_file():
            raise FileNotFoundError(f"Scene mask not found: {mask_path}")

        image_hw = _probe_image_hw(image_path)
        mask_hw = _probe_image_hw(mask_path)
        if mask_hw != image_hw:
            raise ValueError(f"Scene mask/image size mismatch for camera {view['camera_id']}: {mask_hw} vs {image_hw}")

        declared_size = view.get("declared_image_size")
        if declared_size is not None:
            declared_hw = [int(declared_size[0]), int(declared_size[1])]
            declared_wh = [declared_hw[1], declared_hw[0]]
            if declared_hw != image_hw and declared_wh != image_hw and strict:
                raise ValueError(
                    f"scene_manifest image_size mismatch for camera {view['camera_id']}: "
                    f"declared={declared_hw}, actual={image_hw}"
                )

        validated.append(
            {
                "camera_id": view["camera_id"],
                "role": view.get("role"),
                "scene_image_path": str(image_path),
                "scene_image_hw": image_hw,
                "scene_mask_path": str(mask_path),
                "scene_mask_hw": mask_hw,
                "scene_declared_image_hw": declared_size,
            }
        )
    return validated


def _camera_id_token_candidates(camera_id: str) -> tuple[str, ...]:
    return (
        f"cam{camera_id}",
        f"camera{camera_id}",
        f"_{camera_id}_",
        f"-{camera_id}-",
        f"_{camera_id}",
        f"-{camera_id}",
    )


def _infer_camera_id_from_text(text: str, expected_ids: set[str] | None = None) -> str | None:
    lowered = text.lower()
    if expected_ids:
        matches: list[str] = []
        for camera_id in sorted(expected_ids):
            if any(token in lowered for token in _camera_id_token_candidates(camera_id)):
                matches.append(camera_id)
        if len(matches) == 1:
            return matches[0]

    for pattern in (
        r"(?:cam|camera)[_\- ]*(\d{1,3})",
        r"(?:^|[_\- ])(\d{2,3})(?:$|[_\- ])",
    ):
        match = re.search(pattern, lowered)
        if match:
            return _normalize_camera_id(match.group(1))
    return None


def _collect_paths(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if not path.name.startswith(".")]


def _score_candidate(
    path: Path,
    *,
    include_tokens: tuple[str, ...],
    prefer_tokens: tuple[str, ...] = (),
    exclude_tokens: tuple[str, ...] = (),
) -> int:
    lowered = path.as_posix().lower()
    score = 0
    for token in include_tokens:
        if token in lowered:
            score += 10
    for token in prefer_tokens:
        if token in lowered:
            score += 4
    for token in exclude_tokens:
        if token in lowered:
            score -= 7
    return score


def _pick_best_candidate(
    candidates: list[Path],
    *,
    include_tokens: tuple[str, ...],
    prefer_tokens: tuple[str, ...] = (),
    exclude_tokens: tuple[str, ...] = (),
    strict: bool,
    label: str,
) -> Path | None:
    scored: list[tuple[int, int, int, str, Path]] = []
    for candidate in candidates:
        score = _score_candidate(
            candidate,
            include_tokens=include_tokens,
            prefer_tokens=prefer_tokens,
            exclude_tokens=exclude_tokens,
        )
        if score <= 0:
            continue
        scored.append((score, len(candidate.parts), len(str(candidate)), candidate.as_posix().lower(), candidate))

    if not scored:
        return None

    scored.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    best = scored[0]
    tied = [item for item in scored if item[0] == best[0]]
    if strict and len(tied) > 1:
        tied_paths = [str(item[4]) for item in tied[:5]]
        raise ValueError(f"Ambiguous {label} discovery: {tied_paths}")
    return best[4]


def _discover_payload_file(input_root: Path, *, kind: str, mode: str, strict: bool) -> Path | None:
    candidates = [path for path in _collect_paths(input_root) if path.is_file() and path.suffix.lower() in PAYLOAD_SUFFIXES]
    if kind == "combined":
        include = ("fit", "fitting", "result", "smplx")
        prefer = ("json", "combined", "output")
        exclude = ("camera", "mask", "image")
    elif kind == "smplx":
        include = ("smplx", "pose", "body")
        prefer = ("param", "params", "result", "results", "regressor", "fit" if mode == "fitting-result" else "pred")
        exclude = ("camera", "cam", "mask", "image", "manifest")
    elif kind == "camera":
        include = ("camera", "cam", "intrinsic", "extrinsic", "calib")
        prefer = ("param", "params", "result", "results")
        exclude = ("smplx", "mask", "image", "manifest")
    else:
        raise ValueError(f"Unsupported discovery kind: {kind}")

    return _pick_best_candidate(
        candidates,
        include_tokens=include,
        prefer_tokens=prefer,
        exclude_tokens=exclude,
        strict=strict,
        label=kind,
    )


def _discover_asset_input(input_root: Path, *, kind: str, strict: bool) -> Path | None:
    if kind == "mask":
        include = ("mask", "masks", "silhouette", "seg", "matte")
        prefer = ("json", "npz", "png")
        exclude = ("image", "rgb", "camera", "smplx", "manifest", "summary")
    elif kind == "image":
        include = ("image", "images", "rgb", "frame")
        prefer = ("json", "npz")
        exclude = ("mask", "camera", "smplx", "manifest", "summary")
    else:
        raise ValueError(f"Unsupported asset kind: {kind}")

    candidates = [
        path
        for path in _collect_paths(input_root)
        if path.is_dir() or (path.is_file() and path.suffix.lower() in PAYLOAD_SUFFIXES)
    ]
    return _pick_best_candidate(
        candidates,
        include_tokens=include,
        prefer_tokens=prefer,
        exclude_tokens=exclude,
        strict=strict,
        label=f"{kind}_input",
    )


def _extract_nested_payload(payload: Any, keys: tuple[str, ...]) -> tuple[Any | None, str | None]:
    if not _is_mapping(payload):
        return None, None
    for key in keys:
        if key in payload:
            return _to_python(payload[key]), key
    return None, None


def _extract_view_asset_mapping(payload: Any, *, kind: str, base_dir: Path) -> dict[str, Path]:
    if not _is_mapping(payload):
        return {}

    path_field = "mask_path" if kind == "mask" else "image_path"
    container_keys = MASK_CONTAINER_KEYS if kind == "mask" else IMAGE_CONTAINER_KEYS
    path_array_keys = ("mask_paths", "mask_files", "paths") if kind == "mask" else ("image_paths", "image_files", "paths")

    if "views" in payload and isinstance(payload["views"], list):
        mapping: dict[str, Path] = {}
        for item in payload["views"]:
            if not isinstance(item, dict):
                continue
            camera_id = item.get("camera_id")
            path_value = item.get(path_field)
            if camera_id is None or path_value is None:
                continue
            mapping[_normalize_camera_id(camera_id)] = _resolve_path_like(path_value, base_dir=base_dir)
        if mapping:
            return mapping

    for key in VIEW_CONTAINER_KEYS + container_keys:
        nested = payload.get(key)
        if isinstance(nested, dict):
            mapping = _extract_view_asset_mapping(nested, kind=kind, base_dir=base_dir)
            if mapping:
                return mapping

    if "camera_ids" in payload:
        path_values = None
        for key in path_array_keys:
            if key in payload:
                path_values = payload[key]
                break
        if path_values is not None:
            camera_ids = [_normalize_camera_id(item) for item in np.asarray(payload["camera_ids"]).tolist()]
            path_list = np.asarray(path_values).tolist()
            return {
                camera_id: _resolve_path_like(path_value, base_dir=base_dir)
                for camera_id, path_value in zip(camera_ids, path_list)
            }

    maybe_paths = payload.get("paths")
    if isinstance(maybe_paths, dict):
        return {
            _normalize_camera_id(camera_id): _resolve_path_like(path_value, base_dir=base_dir)
            for camera_id, path_value in maybe_paths.items()
        }

    if all(isinstance(key, str) for key in payload.keys()):
        mapping = {}
        for camera_id, path_value in payload.items():
            if isinstance(path_value, (str, Path)):
                mapping[_normalize_camera_id(camera_id)] = _resolve_path_like(path_value, base_dir=base_dir)
        if mapping:
            return mapping

    return {}


def _resolve_asset_mapping(
    input_ref: Path | None,
    *,
    kind: str,
    expected_camera_ids: list[str] | None,
    strict: bool,
) -> tuple[dict[str, Path], str | None]:
    if input_ref is None:
        return {}, None

    expected_set = set(expected_camera_ids or [])
    if input_ref.is_dir():
        grouped: dict[str, list[Path]] = {}
        for path in input_ref.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            camera_id = _infer_camera_id_from_text(path.stem, expected_ids=expected_set if expected_set else None)
            if camera_id is None:
                continue
            grouped.setdefault(camera_id, []).append(path.resolve())

        resolved: dict[str, Path] = {}
        duplicates: dict[str, list[str]] = {}
        for camera_id, candidates in grouped.items():
            if len(candidates) == 1:
                resolved[camera_id] = candidates[0]
                continue
            exact_matches = [
                candidate
                for candidate in candidates
                if any(token in candidate.stem.lower() for token in _camera_id_token_candidates(camera_id))
            ]
            if len(exact_matches) == 1:
                resolved[camera_id] = exact_matches[0]
            else:
                duplicates[camera_id] = [str(candidate) for candidate in candidates]

        if duplicates and strict:
            raise ValueError(f"Ambiguous {kind} files under {input_ref}: {duplicates}")
        return resolved, str(input_ref.resolve())

    if input_ref.suffix.lower() not in PAYLOAD_SUFFIXES:
        raise ValueError(f"Unsupported {kind} input extension: {input_ref.suffix}")

    payload = _load_payload(input_ref)
    mapping = _extract_view_asset_mapping(payload, kind=kind, base_dir=input_ref.parent.resolve())
    if not mapping and strict:
        raise ValueError(f"Could not parse any {kind} paths from {input_ref}")
    return mapping, str(input_ref.resolve())


def _merge_asset_validation(
    *,
    camera_ids_ordered: list[str],
    scene_entries: list[dict[str, Any]],
    image_mapping: dict[str, Path],
    mask_mapping: dict[str, Path],
    strict: bool,
) -> dict[str, Any]:
    ordered_camera_ids = list(camera_ids_ordered)
    if not ordered_camera_ids:
        ordered_camera_ids = sorted(set(image_mapping.keys()) | set(mask_mapping.keys()))

    entries_by_camera = {camera_id: {"camera_id": camera_id} for camera_id in ordered_camera_ids}
    for scene_entry in scene_entries:
        entries_by_camera.setdefault(scene_entry["camera_id"], {"camera_id": scene_entry["camera_id"]}).update(scene_entry)

    missing_external_images: list[str] = []
    missing_external_masks: list[str] = []
    size_mismatches: list[dict[str, Any]] = []

    for camera_id in ordered_camera_ids:
        entry = entries_by_camera[camera_id]

        image_path = image_mapping.get(camera_id)
        if image_mapping:
            if image_path is None:
                missing_external_images.append(camera_id)
            else:
                if not image_path.is_file():
                    raise FileNotFoundError(f"External image not found for camera {camera_id}: {image_path}")
                image_hw = _probe_image_hw(image_path)
                entry["external_image_path"] = str(image_path)
                entry["external_image_hw"] = image_hw
                scene_image_hw = entry.get("scene_image_hw")
                if scene_image_hw is not None and image_hw != scene_image_hw:
                    size_mismatches.append(
                        {
                            "camera_id": camera_id,
                            "type": "external_image_vs_scene_image",
                            "scene_image_hw": scene_image_hw,
                            "external_image_hw": image_hw,
                        }
                    )

        mask_path = mask_mapping.get(camera_id)
        if mask_mapping:
            if mask_path is None:
                missing_external_masks.append(camera_id)
            else:
                if not mask_path.is_file():
                    raise FileNotFoundError(f"External mask not found for camera {camera_id}: {mask_path}")
                mask_hw = _probe_image_hw(mask_path)
                entry["external_mask_path"] = str(mask_path)
                entry["external_mask_hw"] = mask_hw
                reference_hw = entry.get("external_image_hw") or entry.get("scene_image_hw")
                if reference_hw is not None and mask_hw != reference_hw:
                    size_mismatches.append(
                        {
                            "camera_id": camera_id,
                            "type": "external_mask_vs_reference_image",
                            "reference_hw": reference_hw,
                            "external_mask_hw": mask_hw,
                        }
                    )

    extra_external_images = sorted(set(image_mapping.keys()) - set(ordered_camera_ids))
    extra_external_masks = sorted(set(mask_mapping.keys()) - set(ordered_camera_ids))

    if strict:
        if missing_external_images:
            raise ValueError(f"External image mapping is missing cameras: {missing_external_images}")
        if missing_external_masks:
            raise ValueError(f"External mask mapping is missing cameras: {missing_external_masks}")
        if size_mismatches:
            raise ValueError(f"Image/mask size validation failed: {size_mismatches}")

    return {
        "camera_ids_ordered": ordered_camera_ids,
        "scene_view_count": len(scene_entries),
        "external_image_count": len(image_mapping),
        "external_mask_count": len(mask_mapping),
        "missing_external_images": missing_external_images,
        "missing_external_masks": missing_external_masks,
        "extra_external_images": extra_external_images,
        "extra_external_masks": extra_external_masks,
        "size_mismatches": size_mismatches,
        "views": [entries_by_camera[camera_id] for camera_id in ordered_camera_ids],
    }


def _ensure_output_paths(output_dir: Path, *, has_camera: bool, overwrite: bool) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    smplx_output = output_dir / "normalized_smplx_params.npz"
    camera_output = output_dir / "normalized_camera_params.npz"
    manifest_output = output_dir / MANIFEST_NAME

    planned = [smplx_output, manifest_output]
    if has_camera:
        planned.append(camera_output)
    if not overwrite:
        for path in planned:
            if path.exists():
                raise FileExistsError(f"{path} already exists. Re-run with --overwrite.")
    return smplx_output, camera_output, manifest_output


def _resolve_explicit_path(path_value: str | None, *, input_path: Path) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (input_path.parent if input_path.is_file() else input_path) / path
    return path.resolve()


def _load_combined_payload(input_path: Path, *, strict: bool) -> tuple[Any | None, Path | None]:
    if not input_path.is_file() or input_path.suffix.lower() not in PAYLOAD_SUFFIXES:
        return None, None
    payload = _load_payload(input_path)
    if not _is_mapping(payload):
        if strict:
            raise ValueError(f"Combined fitting payload must be a mapping: {input_path}")
        return None, None
    return payload, input_path


def _format_estimator_command(
    command: str,
    *,
    scene_dir: Path | None,
    scene_manifest: Path | None,
    input_path: Path,
    estimator_output_dir: Path,
) -> str:
    values = {
        "scene_dir": "" if scene_dir is None else str(scene_dir),
        "scene_manifest": "" if scene_manifest is None else str(scene_manifest),
        "input_path": str(input_path),
        "estimator_output_dir": str(estimator_output_dir),
    }
    return command.format(**values)


def _run_estimator_command(
    args: argparse.Namespace,
    *,
    scene_dir: Path | None,
    scene_manifest_path: Path | None,
) -> dict[str, Any] | None:
    if args.mode != "estimator-command":
        return None

    input_path = Path(args.input_path).expanduser().resolve()
    estimator_output_dir = (
        Path(args.estimator_output_dir).expanduser().resolve()
        if args.estimator_output_dir
        else input_path
    )
    estimator_output_dir.mkdir(parents=True, exist_ok=True)

    driver_log_dir = Path(args.output_dir).expanduser().resolve()
    driver_log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = driver_log_dir / "estimator_command_summary.json"
    if args.skip_estimator_run:
        summary = {
            "mode": "estimator-command",
            "skipped": True,
            "input_path": str(input_path),
            "estimator_output_dir": str(estimator_output_dir),
            "note": "Estimator run skipped; importing existing files from estimator_output_dir/input_path.",
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary

    if not args.estimator_command:
        raise ValueError("--estimator-command is required for --mode estimator-command unless --skip-estimator-run is set.")

    command = _format_estimator_command(
        args.estimator_command,
        scene_dir=scene_dir,
        scene_manifest=scene_manifest_path,
        input_path=input_path,
        estimator_output_dir=estimator_output_dir,
    )
    cwd = Path(args.estimator_cwd).expanduser().resolve() if args.estimator_cwd else REPO_ROOT
    if not cwd.is_dir():
        raise FileNotFoundError(f"Estimator cwd not found: {cwd}")

    completed = subprocess.run(
        command,
        cwd=str(cwd),
        shell=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=int(args.estimator_timeout_sec),
        check=False,
    )
    summary = {
        "mode": "estimator-command",
        "skipped": False,
        "command": command,
        "cwd": str(cwd),
        "input_path": str(input_path),
        "estimator_output_dir": str(estimator_output_dir),
        "timeout_sec": int(args.estimator_timeout_sec),
        "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout[-8000:],
        "stderr_tail": completed.stderr[-8000:],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Estimator command failed with returncode={completed.returncode}; see {summary_path}")
    return summary


def _resolve_inputs(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.estimator_output_dir or args.input_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    discovery_log: list[str] = []
    combined_payload: Any | None = None
    combined_payload_path: Path | None = None
    if args.mode == "fitting-result":
        payload, payload_path = _load_combined_payload(input_path, strict=args.strict)
        if payload is not None:
            combined_payload = payload
            combined_payload_path = payload_path
            discovery_log.append(f"Loaded combined fitting payload from {payload_path}")
        elif input_path.is_dir():
            discovered = _discover_payload_file(input_path, kind="combined", mode=args.mode, strict=args.strict)
            if discovered is not None:
                combined_payload = _load_payload(discovered)
                combined_payload_path = discovered.resolve()
                discovery_log.append(f"Discovered combined fitting payload at {combined_payload_path}")

    smplx_path = _resolve_explicit_path(args.smplx_input, input_path=input_path)
    camera_path = _resolve_explicit_path(args.camera_input, input_path=input_path)
    mask_input = _resolve_explicit_path(args.mask_input, input_path=input_path)
    image_input = _resolve_explicit_path(args.image_input, input_path=input_path)

    if smplx_path is None and input_path.is_dir():
        smplx_path = _discover_payload_file(input_path, kind="smplx", mode=args.mode, strict=args.strict)
        if smplx_path is not None:
            discovery_log.append(f"Discovered SMPL-X payload at {smplx_path}")
    if camera_path is None and input_path.is_dir():
        camera_path = _discover_payload_file(input_path, kind="camera", mode=args.mode, strict=args.strict)
        if camera_path is not None:
            discovery_log.append(f"Discovered camera payload at {camera_path}")
    if mask_input is None and input_path.is_dir():
        mask_input = _discover_asset_input(input_path, kind="mask", strict=args.strict)
        if mask_input is not None:
            discovery_log.append(f"Discovered external mask input at {mask_input}")
    if image_input is None and input_path.is_dir():
        image_input = _discover_asset_input(input_path, kind="image", strict=args.strict)
        if image_input is not None:
            discovery_log.append(f"Discovered external image input at {image_input}")

    smplx_payload = _load_payload(smplx_path) if smplx_path is not None else None
    camera_payload = _load_payload(camera_path) if camera_path is not None else None
    smplx_nested_key = None
    camera_nested_key = None
    mask_nested_source = None
    image_nested_source = None

    if smplx_payload is None and combined_payload is not None:
        if _has_smplx_signal(combined_payload):
            smplx_payload = combined_payload
        else:
            smplx_payload, smplx_nested_key = _extract_nested_payload(combined_payload, SMPLX_CONTAINER_KEYS)
        if smplx_payload is not None:
            discovery_log.append(f"Resolved SMPL-X payload from combined fitting payload key={smplx_nested_key or '<root>'}")

    if camera_payload is None and combined_payload is not None:
        if _has_camera_signal(combined_payload):
            camera_payload = combined_payload
        else:
            camera_payload, camera_nested_key = _extract_nested_payload(combined_payload, CAMERA_CONTAINER_KEYS)
        if camera_payload is not None:
            discovery_log.append(f"Resolved camera payload from combined fitting payload key={camera_nested_key or '<root>'}")

    if mask_input is None and combined_payload_path is not None:
        if _extract_view_asset_mapping(combined_payload, kind="mask", base_dir=combined_payload_path.parent):
            mask_input = combined_payload_path
            mask_nested_source = "combined_payload"
            discovery_log.append("Resolved external masks from combined fitting payload.")

    if image_input is None and combined_payload_path is not None:
        if _extract_view_asset_mapping(combined_payload, kind="image", base_dir=combined_payload_path.parent):
            image_input = combined_payload_path
            image_nested_source = "combined_payload"
            discovery_log.append("Resolved external images from combined fitting payload.")

    if smplx_payload is None:
        raise FileNotFoundError(
            "Could not resolve any SMPL-X payload. Pass --smplx-input or provide an input directory/payload with SMPL-X params."
        )
    if not _is_mapping(smplx_payload):
        raise ValueError("Resolved SMPL-X payload must be a mapping.")
    if camera_payload is not None and not _is_mapping(camera_payload):
        raise ValueError("Resolved camera payload must be a mapping.")

    if args.strict and not _has_smplx_signal(smplx_payload):
        raise ValueError("Resolved SMPL-X payload does not expose recognizable SMPL-X keys.")
    if camera_payload is not None and args.strict and not _has_camera_signal(camera_payload):
        raise ValueError("Resolved camera payload does not expose recognizable camera keys.")

    return {
        "input_path": input_path,
        "discovery_log": discovery_log,
        "smplx_payload": smplx_payload,
        "camera_payload": camera_payload,
        "smplx_source": _build_source_ref(smplx_path or combined_payload_path, nested_key=smplx_nested_key),
        "camera_source": _build_source_ref(camera_path or combined_payload_path, nested_key=camera_nested_key),
        "mask_input": mask_input,
        "image_input": image_input,
        "mask_nested_source": mask_nested_source,
        "image_nested_source": image_nested_source,
    }


def main() -> int:
    args = parse_args()

    scene_dir = Path(args.scene_dir).expanduser().resolve() if args.scene_dir else None
    scene_manifest_path = Path(args.scene_manifest).expanduser().resolve() if args.scene_manifest else None
    scene_manifest, resolved_scene_manifest_path, resolved_scene_dir = _resolve_scene_manifest(
        scene_dir=scene_dir,
        scene_manifest=scene_manifest_path,
    )
    estimator_run_summary = _run_estimator_command(
        args,
        scene_dir=resolved_scene_dir,
        scene_manifest_path=resolved_scene_manifest_path,
    )
    scene_views = [] if scene_manifest is None else _collect_scene_views(scene_manifest, base_dir=resolved_scene_dir or REPO_ROOT)
    scene_entries = _validate_scene_view_assets(scene_views, strict=args.strict) if scene_views else []
    scene_camera_ids = [entry["camera_id"] for entry in scene_entries]

    resolved_inputs = _resolve_inputs(args)
    smplx_params = _normalize_smplx_params(
        resolved_inputs["smplx_payload"],
        frame_idx=args.frame_idx,
        strict=args.strict,
    )

    smplx_output, camera_output, manifest_output = _ensure_output_paths(
        Path(args.output_dir).expanduser().resolve(),
        has_camera=resolved_inputs["camera_payload"] is not None,
        overwrite=args.overwrite,
    )
    _save_smplx_npz(smplx_output.parent, smplx_params)

    camera_ids_ordered: list[str] = []
    missing_scene_cameras_in_camera_file: list[str] = []
    camera_output_written = False
    if resolved_inputs["camera_payload"] is not None:
        camera_params = _parse_camera_payload(resolved_inputs["camera_payload"], strict=args.strict)
        if scene_manifest is not None:
            camera_ids_ordered, missing_scene_cameras_in_camera_file = _align_camera_order(
                camera_params,
                scene_manifest,
                strict=args.strict,
            )
        else:
            camera_ids_ordered = sorted(camera_params.keys())
        if camera_ids_ordered:
            _save_camera_npz(camera_output.parent, camera_ids_ordered, camera_params)
            camera_output_written = True
        elif args.strict:
            raise ValueError("No aligned camera ids were produced after camera parsing.")
    else:
        camera_params = {}

    if not camera_ids_ordered:
        camera_ids_ordered = scene_camera_ids

    image_mapping, image_source = _resolve_asset_mapping(
        resolved_inputs["image_input"],
        kind="image",
        expected_camera_ids=camera_ids_ordered or scene_camera_ids,
        strict=args.strict,
    )
    mask_mapping, mask_source = _resolve_asset_mapping(
        resolved_inputs["mask_input"],
        kind="mask",
        expected_camera_ids=camera_ids_ordered or scene_camera_ids,
        strict=args.strict,
    )
    asset_validation = _merge_asset_validation(
        camera_ids_ordered=camera_ids_ordered,
        scene_entries=scene_entries,
        image_mapping=image_mapping,
        mask_mapping=mask_mapping,
        strict=args.strict,
    )
    if not camera_ids_ordered:
        camera_ids_ordered = list(asset_validation["camera_ids_ordered"])

    bundle_manifest = {
        "format_version": 1,
        "frame_idx": int(args.frame_idx),
        "smplx_source": resolved_inputs["smplx_source"],
        "camera_source": resolved_inputs["camera_source"],
        "scene_dir": None if resolved_scene_dir is None else str(resolved_scene_dir),
        "scene_manifest": None if resolved_scene_manifest_path is None else str(resolved_scene_manifest_path),
        "scene_manifest_loaded": scene_manifest is not None,
        "smplx_output": smplx_output.name,
        "camera_output": camera_output.name if camera_output_written else None,
        "camera_ids_ordered": camera_ids_ordered,
        "missing_scene_cameras_in_camera_file": missing_scene_cameras_in_camera_file,
        "smplx_keys": sorted(smplx_params.keys()),
        "notes": [
            "This bundle stays compatible with tools/import_external_smplx_params.py output naming and manifest keys.",
            "The real-data driver only adapts/imports external SMPL-X pose sources. It does not train or download a large regressor.",
            "Current downstream prior generation can consume this bundle through --external-prior-bundle.",
        ],
        "driver": {
            "name": "run_realdata_smplx_driver",
            "mode": args.mode,
            "input_path": str(resolved_inputs["input_path"]),
            "discovery_log": resolved_inputs["discovery_log"],
            "strict": bool(args.strict),
            "estimator_run_summary": estimator_run_summary,
            "pose_source_answer": (
                "When real captures do not have SMPL-X labels, pose comes from an external "
                "SMPL-X regressor output, an estimator command launched by this driver, or an external "
                "fitting result. This driver launches/adapts the estimator boundary, then imports, validates, "
                "and normalizes that result into the current prior-bundle format."
            ),
        },
        "raw_inputs": {
            "smplx_source": resolved_inputs["smplx_source"],
            "camera_source": resolved_inputs["camera_source"],
            "image_source": image_source,
            "mask_source": mask_source,
            "image_nested_source": resolved_inputs["image_nested_source"],
            "mask_nested_source": resolved_inputs["mask_nested_source"],
        },
        "validation": {
            "smplx_value_shapes": _payload_stats(smplx_params),
            "camera_count": len(camera_ids_ordered),
            "camera_payload_present": bool(camera_params),
            "scene_view_count": len(scene_entries),
        },
        "view_asset_validation": asset_validation,
    }
    manifest_output.write_text(json.dumps(bundle_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[realdata-smplx-driver] mode={args.mode}")
    print(f"[realdata-smplx-driver] wrote: {smplx_output}")
    if camera_output_written:
        print(f"[realdata-smplx-driver] wrote: {camera_output} ({len(camera_ids_ordered)} cameras)")
    else:
        print("[realdata-smplx-driver] camera output skipped (no external camera payload resolved).")
    print(f"[realdata-smplx-driver] wrote: {manifest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
