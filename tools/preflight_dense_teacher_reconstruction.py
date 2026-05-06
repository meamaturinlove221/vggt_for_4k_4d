from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from preflight_differentiable_renderer_backend import parse_view_indices  # noqa: E402
from prepare_4k4d_prior_training_case import (  # noqa: E402
    align_intrinsics_for_scene_view,
    load_scene_manifest,
    recover_legacy_crop_source_sizes,
    resolve_scene_camera_params,
)


METHODS = {
    "A1_neural_sdf": {
        "goal": "known-camera neural SDF / NeuS-like dense same-frame surface",
        "requires": ["rgb", "mask", "known_camera", "same_frame_views"],
        "forbidden_as_teacher_without_gate": True,
        "stop_condition": "freeze if extracted mesh is shell/slab/template face/detached hands in Open3D",
    },
    "A2_gaussian_surface": {
        "goal": "Gaussian/2DGS surface extraction route",
        "requires": ["rgb", "mask", "known_camera", "same_frame_views"],
        "forbidden_as_teacher_without_gate": True,
        "stop_condition": "freeze if extracted surface is floating fragments or cannot rasterize to original 6v",
    },
    "A3_visual_hull_init": {
        "goal": "mask-constrained visual hull used only as initialization",
        "requires": ["mask", "known_camera", "same_frame_views"],
        "forbidden_as_teacher_without_gate": True,
        "stop_condition": "freeze if used as final teacher or if visual hull remains coarse template shell",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "A-line dense teacher reconstruction readiness preflight. This does not run NeuS/GS, "
            "does not export a teacher, and does not write any strict pass."
        )
    )
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--view-indices", default="0,10,20,30,40,50")
    parser.add_argument("--target-size", type=int, default=96)
    parser.add_argument("--methods", default="A1_neural_sdf,A2_gaussian_surface,A3_visual_hull_init")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_rgb_mask(view: dict[str, Any], target_size: int) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(Path(str(view["image_path"]))).convert("RGB")
    mask = Image.open(Path(str(view["mask_path"]))).convert("L")
    if image.size != (target_size, target_size):
        image = image.resize((target_size, target_size), Image.Resampling.BICUBIC)
    if mask.size != (target_size, target_size):
        mask = mask.resize((target_size, target_size), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8), np.asarray(mask, dtype=np.uint8) > 127


def mask_bbox(mask: np.ndarray) -> dict[str, Any]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return {"valid": False, "x0": 0, "y0": 0, "x1": 0, "y1": 0, "area": 0, "coverage": 0.0}
    return {
        "valid": True,
        "x0": int(xs.min()),
        "y0": int(ys.min()),
        "x1": int(xs.max()) + 1,
        "y1": int(ys.max()) + 1,
        "area": int(mask.sum()),
        "coverage": float(mask.mean()),
    }


def camera_sanity(intrinsic: np.ndarray, world_to_cam: np.ndarray, target_size: int) -> dict[str, Any]:
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    world_to_cam = np.asarray(world_to_cam, dtype=np.float32)
    rot = world_to_cam[:3, :3]
    trans = world_to_cam[:3, 3]
    det = float(np.linalg.det(rot))
    ortho_err = float(np.linalg.norm(rot.T @ rot - np.eye(3, dtype=np.float32)))
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "rotation_det": det,
        "rotation_orthogonality_error": ortho_err,
        "translation_norm": float(np.linalg.norm(trans)),
        "principal_point_inside_target": bool(0 <= cx <= target_size and 0 <= cy <= target_size),
        "positive_focal": bool(fx > 0 and fy > 0),
        "rotation_ok": bool(abs(abs(det) - 1.0) < 0.08 and ortho_err < 0.15),
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Dense Teacher Reconstruction Readiness",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This is A-line research readiness only. It is not a teacher, not a candidate, and not a strict pass.",
        "",
        "## Gate Truth",
        "",
        "```text",
        "strict_candidate_passes = 0",
        "strict_teacher_passes = 0",
        "no_teacher_export = true",
        "research_only = true",
        "```",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary["summary"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Method Matrix",
        "",
        "```json",
        json.dumps(summary["method_matrix"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Decision",
        "",
        "```text",
        summary["decision"],
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_dir = args.scene_dir.resolve()
    manifest = recover_legacy_crop_source_sizes(scene_dir, load_scene_manifest(scene_dir))
    views = manifest["exported_views"]
    view_indices = parse_view_indices(args.view_indices, len(views))
    dataset_root = args.dataset_root or Path(str(manifest.get("dataset_root", "")))
    cameras, camera_source = resolve_scene_camera_params(manifest, dataset_root, args.subset_name)
    target_size = int(args.target_size)

    rows = []
    mask_coverages = []
    bbox_areas = []
    camera_rows = []
    for view_index in view_indices:
        view = views[view_index]
        camera_id = str(view["camera_id"])
        rgb, mask = load_rgb_mask(view, target_size)
        bbox = mask_bbox(mask)
        mask_coverages.append(float(bbox["coverage"]))
        bbox_areas.append(int(bbox["area"]))
        params = cameras.get(camera_id)
        if params is None:
            cam = {"available": False}
        else:
            intrinsic = align_intrinsics_for_scene_view(np.asarray(params["intrinsic"], dtype=np.float32), view, target_size)
            cam = {"available": True, **camera_sanity(intrinsic, np.asarray(params["world_to_cam"], dtype=np.float32), target_size)}
        camera_rows.append(cam)
        rows.append(
            {
                "view_index": int(view_index),
                "camera_id": camera_id,
                "image_path": str(Path(str(view["image_path"])).resolve()),
                "mask_path": str(Path(str(view["mask_path"])).resolve()),
                "rgb_shape": list(rgb.shape),
                "mask_bbox": bbox,
                "camera": cam,
            }
        )

    requested_methods = [item.strip() for item in str(args.methods).split(",") if item.strip()]
    unknown = [item for item in requested_methods if item not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}; choices={sorted(METHODS)}")
    cameras_available = all(bool(row.get("available")) for row in camera_rows)
    camera_numeric_ok = cameras_available and all(
        bool(row.get("positive_focal")) and bool(row.get("rotation_ok")) for row in camera_rows
    )
    mask_ready = len(view_indices) >= 3 and min(mask_coverages or [0.0]) > 0.02
    coverage_spread = float(max(mask_coverages) - min(mask_coverages)) if mask_coverages else 0.0
    method_matrix = []
    for name in requested_methods:
        base = dict(METHODS[name])
        ready = bool(camera_numeric_ok and mask_ready)
        method_matrix.append(
            {
                "method": name,
                **base,
                "asset_ready": ready,
                "readiness_reasons": [
                    "known cameras available and sane" if camera_numeric_ok else "camera sanity failed or camera missing",
                    "multi-view masks available" if mask_ready else "insufficient mask/view coverage",
                    "must still pass Open3D and strict teacher gate after reconstruction",
                ],
            }
        )

    summary = {
        "status": "dense_teacher_reconstruction_readiness_complete",
        "summary": {
            "research_only": True,
            "no_teacher_export": True,
            "no_candidate_export": True,
            "strict_candidate_passes": 0,
            "strict_teacher_passes": 0,
            "scene_dir": str(scene_dir),
            "camera_source": camera_source,
            "view_count_total": int(len(views)),
            "selected_views": view_indices,
            "target_size": target_size,
            "mask_foreground": {
                "min_coverage": float(min(mask_coverages) if mask_coverages else 0.0),
                "max_coverage": float(max(mask_coverages) if mask_coverages else 0.0),
                "mean_coverage": float(np.mean(mask_coverages)) if mask_coverages else 0.0,
                "coverage_spread": coverage_spread,
                "min_pixels": int(min(bbox_areas) if bbox_areas else 0),
                "max_pixels": int(max(bbox_areas) if bbox_areas else 0),
            },
            "camera_availability": {
                "all_selected_cameras_available": bool(cameras_available),
                "camera_numeric_ok": bool(camera_numeric_ok),
            },
            "asset_ready_for_research_preflight": bool(camera_numeric_ok and mask_ready),
            "views": rows,
        },
        "method_matrix": method_matrix,
        "decision": (
            "Assets are ready for A-line research-preflight only if camera and mask readiness are true. "
            "Any reconstructed surface must still pass Open3D full/head/face/hairline/hands review and "
            "strict teacher gate before it can become a teacher."
        ),
    }
    (output_dir / "dense_teacher_reconstruction_readiness.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(output_dir / "dense_teacher_reconstruction_readiness.md", summary)
    print(json.dumps(summary["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
