from __future__ import annotations

import argparse
import importlib.util
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

from normal_line_multiview_eval import build_roi_masks, load_scene_view  # noqa: E402
from prepare_4k4d_prior_training_case import (  # noqa: E402
    align_intrinsics_for_scene_view,
    load_optional_annotation_payload,
    load_scene_manifest,
    recover_legacy_crop_source_sizes,
    resolve_scene_camera_params,
    resolve_smplx_model_dir,
)
from tools.smplx_numpy import (  # noqa: E402
    forward_smplx_mesh,
    load_smplx_model,
    rasterize_world_mesh,
    resolve_smplx_model_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Raw-image human surface upper-bound preflight. This intentionally "
            "does not use VGGT depth/point/normal observations. It audits whether "
            "raw RGB/masks/cameras/SMPL-X are available for a true differentiable "
            "surface optimization stage, and measures the starting SMPL-X mask fit."
        )
    )
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--smplx-model-dir", type=Path)
    parser.add_argument("--smplx-gender", choices=("neutral", "female", "male"), default="neutral")
    parser.add_argument("--target-size", type=int, default=518)
    parser.add_argument(
        "--max-views",
        type=int,
        default=60,
        help="Maximum manifest views to audit. Use 60 for the dense-view upper-bound preflight.",
    )
    parser.add_argument(
        "--view-stride",
        type=int,
        default=1,
        help="Audit every Nth selected view after manifest ordering.",
    )
    parser.add_argument("--min-mean-iou", type=float, default=0.45)
    parser.add_argument("--min-mean-target-recall", type=float, default=0.60)
    parser.add_argument("--overlay-limit", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def dependency_preflight() -> dict[str, Any]:
    optional_modules = {
        "torch": module_available("torch"),
        "smplx": module_available("smplx"),
        "cv2": module_available("cv2"),
        "open3d": module_available("open3d"),
        "trimesh": module_available("trimesh"),
        "pytorch3d": module_available("pytorch3d"),
        "nvdiffrast": module_available("nvdiffrast"),
        "kaolin": module_available("kaolin"),
    }
    differentiable_renderer_available = any(
        optional_modules[name] for name in ("pytorch3d", "nvdiffrast", "kaolin")
    )
    return {
        "optional_modules": optional_modules,
        "differentiable_renderer_available": bool(differentiable_renderer_available),
        "true_gradient_surface_optimization_available": bool(differentiable_renderer_available),
        "preflight_note": (
            "This script can audit raw assets and SMPL-X starting silhouette fit without "
            "a differentiable renderer. A true raw-image surface upper-bound requires "
            "pytorch3d, nvdiffrast, kaolin, or an equivalent differentiable renderer."
        ),
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value).replace("\\", "/")
    if isinstance(value, str):
        return value.replace("\\", "/")
    return value


def homogeneous(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :4] = matrix
        return out
    raise ValueError(f"Expected 3x4 or 4x4 camera matrix, got {matrix.shape}")


def compute_mask_metrics(rendered: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    rendered = np.asarray(rendered, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = rendered & target
    union = rendered | target
    render_pixels = int(rendered.sum())
    target_pixels = int(target.sum())
    intersection_pixels = int(intersection.sum())
    union_pixels = int(union.sum())
    false_positive = int((rendered & ~target).sum())
    false_negative = int((target & ~rendered).sum())
    return {
        "render_pixels": render_pixels,
        "target_pixels": target_pixels,
        "intersection_pixels": intersection_pixels,
        "union_pixels": union_pixels,
        "iou": float(intersection_pixels / union_pixels) if union_pixels else None,
        "target_recall": float(intersection_pixels / target_pixels) if target_pixels else None,
        "render_precision": float(intersection_pixels / render_pixels) if render_pixels else None,
        "false_positive_pixels": false_positive,
        "false_negative_pixels": false_negative,
    }


def summarize_numeric(values: list[float | None]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float32)
    if arr.size == 0:
        return {"count": 0, "mean": None, "p10": None, "p50": None, "p90": None, "min": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def overlay_masks(rgb: np.ndarray, target: np.ndarray, rendered: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.max() <= 1.5:
        rgb = rgb * 255.0
    out = np.clip(rgb, 0, 255).astype(np.float32)
    target = np.asarray(target, dtype=bool)
    rendered = np.asarray(rendered, dtype=bool)
    both = target & rendered
    target_only = target & ~rendered
    rendered_only = rendered & ~target
    out[target_only] = 0.45 * out[target_only] + 0.55 * np.array([0, 220, 0], dtype=np.float32)
    out[rendered_only] = 0.45 * out[rendered_only] + 0.55 * np.array([240, 0, 0], dtype=np.float32)
    out[both] = 0.55 * out[both] + 0.45 * np.array([255, 220, 0], dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def save_contact_sheet(image_paths: list[Path], output_path: Path, columns: int = 4) -> None:
    if not image_paths:
        return
    images = [Image.open(path).convert("RGB") for path in image_paths]
    width, height = images[0].size
    columns = max(1, int(columns))
    rows = int(np.ceil(len(images) / columns))
    sheet = Image.new("RGB", (columns * width, rows * height), (255, 255, 255))
    for idx, image in enumerate(images):
        sheet.paste(image, ((idx % columns) * width, (idx // columns) * height))
    sheet.save(output_path)


def render_smplx_mask(
    vertices: np.ndarray,
    faces: np.ndarray,
    world_to_cam: np.ndarray,
    intrinsic: np.ndarray,
    image_hw: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    result = rasterize_world_mesh(
        world_vertices=vertices,
        faces=faces,
        world_to_cam=homogeneous(world_to_cam),
        intrinsic=intrinsic,
        image_hw=image_hw,
        silhouette_mask=None,
        fill_knn=0,
        return_raster_mask=True,
    )
    _, _, _, raster_mask, meta = result
    return raster_mask.astype(bool), meta


def main() -> int:
    args = parse_args()
    scene_dir = args.scene_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} already exists and is not empty. Use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    dep = dependency_preflight()
    manifest = recover_legacy_crop_source_sizes(scene_dir, load_scene_manifest(scene_dir))
    dataset_root = Path(args.dataset_root or manifest["dataset_root"]).expanduser()
    views = list(manifest.get("exported_views", []))
    if not views:
        raise ValueError(f"No exported_views in scene manifest: {scene_dir / 'scene_manifest.json'}")
    selected_indices = list(range(0, len(views), max(1, int(args.view_stride))))[: max(1, int(args.max_views))]

    smplx_model_dir = resolve_smplx_model_dir(None if args.smplx_model_dir is None else str(args.smplx_model_dir))
    if smplx_model_dir is None:
        raise FileNotFoundError("Could not resolve SMPL-X model dir; pass --smplx-model-dir.")
    model_path = resolve_smplx_model_path(smplx_model_dir, args.smplx_gender)
    smplx_params, keypoints3d = load_optional_annotation_payload(manifest, dataset_root, args.subset_name)
    if not smplx_params:
        raise ValueError("Scene annotations do not provide SMPL-X parameters.")
    mesh = forward_smplx_mesh(
        model_path=model_path,
        betas=smplx_params["betas"],
        expression=smplx_params.get("expression"),
        fullpose=smplx_params["fullpose"],
        transl=smplx_params.get("transl"),
        scale=smplx_params.get("scale", 1.0),
    )
    vertices = np.asarray(mesh["vertices"], dtype=np.float32)
    faces = np.asarray(mesh["faces"], dtype=np.int32)
    smplx_model = load_smplx_model(model_path)
    dominant_joint = np.asarray(smplx_model["weights"], dtype=np.float32).argmax(axis=1)

    camera_params, camera_source = resolve_scene_camera_params(manifest, dataset_root, args.subset_name)

    per_view: list[dict[str, Any]] = []
    overlay_paths: list[Path] = []
    roi_metric_rows: dict[str, list[dict[str, Any]]] = {"full": [], "head": [], "face": []}
    for view_idx in selected_indices:
        view = views[view_idx]
        camera_id = str(view["camera_id"]).zfill(2)
        scene = load_scene_view(scene_dir, view_idx, (args.target_size, args.target_size))
        target_mask = np.asarray(scene.mask, dtype=bool)
        intrinsic = align_intrinsics_for_scene_view(
            np.asarray(camera_params[camera_id]["intrinsic"], dtype=np.float32),
            view,
            target_size=args.target_size,
        )
        rendered_mask, raster_meta = render_smplx_mask(
            vertices=vertices,
            faces=faces,
            world_to_cam=np.asarray(camera_params[camera_id]["world_to_cam"], dtype=np.float32),
            intrinsic=intrinsic,
            image_hw=(args.target_size, args.target_size),
        )
        full_metrics = compute_mask_metrics(rendered_mask, target_mask)
        roi_masks = build_roi_masks(target_mask)
        roi_metrics = {
            roi_name: compute_mask_metrics(rendered_mask & roi_mask, roi_mask)
            for roi_name, roi_mask in roi_masks.items()
        }
        for roi_name, metrics in roi_metrics.items():
            roi_metric_rows.setdefault(roi_name, []).append(metrics)

        overlay_path = overlay_dir / f"view_{view_idx:02d}_cam{camera_id}_smplx_mask_overlay.png"
        if len(overlay_paths) < max(0, int(args.overlay_limit)):
            Image.fromarray(overlay_masks(scene.rgb, target_mask, rendered_mask)).save(overlay_path)
            overlay_paths.append(overlay_path)
        per_view.append(
            {
                "view_index": int(view_idx),
                "camera_id": camera_id,
                "image_path": str(scene.image_path),
                "mask_path": str(scene.mask_path),
                "full": full_metrics,
                "roi": roi_metrics,
                "raster_meta": json_ready(raster_meta),
            }
        )

    save_contact_sheet(overlay_paths, output_dir / "smplx_mask_overlay_contact_sheet.png")

    full_ious = [row["full"]["iou"] for row in per_view]
    full_recalls = [row["full"]["target_recall"] for row in per_view]
    roi_summary = {
        roi_name: {
            "iou": summarize_numeric([row.get("iou") for row in rows]),
            "target_recall": summarize_numeric([row.get("target_recall") for row in rows]),
            "render_precision": summarize_numeric([row.get("render_precision") for row in rows]),
        }
        for roi_name, rows in roi_metric_rows.items()
    }
    mean_iou = summarize_numeric(full_ious)["mean"]
    mean_recall = summarize_numeric(full_recalls)["mean"]
    asset_preflight_pass = (
        len(per_view) > 0
        and mean_iou is not None
        and mean_recall is not None
        and mean_iou >= float(args.min_mean_iou)
        and mean_recall >= float(args.min_mean_target_recall)
    )
    true_stage_a_ready = bool(dep["true_gradient_surface_optimization_available"] and asset_preflight_pass)
    if true_stage_a_ready:
        truthful_status = "ready_for_true_raw_image_differentiable_surface_optimization"
    elif asset_preflight_pass:
        truthful_status = "assets_ok_but_blocked_missing_differentiable_renderer"
    else:
        truthful_status = "blocked_raw_smplx_silhouette_starting_fit_not_sufficient"

    summary = {
        "task": "raw_image_surface_upperbound_preflight_v0",
        "truthful_status": truthful_status,
        "scene_dir": str(scene_dir),
        "output_dir": str(output_dir),
        "dataset_root": str(dataset_root),
        "scene": {
            "seq_id": manifest.get("seq_id"),
            "frame_id": manifest.get("frame_id"),
            "preprocess_variant": manifest.get("preprocess_variant"),
            "view_count": len(views),
            "selected_view_count": len(selected_indices),
            "selected_indices": selected_indices,
            "camera_source": camera_source,
        },
        "dependency_preflight": dep,
        "smplx": {
            "model_path": str(model_path),
            "vertices": int(vertices.shape[0]),
            "faces": int(faces.shape[0]),
            "dominant_joint_count": int(np.unique(dominant_joint).size),
            "has_keypoints3d": keypoints3d is not None,
            "params_keys": sorted(str(k) for k in smplx_params.keys()),
        },
        "metrics": {
            "mean_full_iou": mean_iou,
            "mean_full_target_recall": mean_recall,
            "full_iou": summarize_numeric(full_ious),
            "full_target_recall": summarize_numeric(full_recalls),
            "roi": roi_summary,
            "asset_preflight_pass": bool(asset_preflight_pass),
            "true_stage_a_ready": bool(true_stage_a_ready),
        },
        "outputs": {
            "overlay_dir": str(overlay_dir),
            "overlay_contact_sheet": str(output_dir / "smplx_mask_overlay_contact_sheet.png"),
            "summary_json": str(output_dir / "raw_image_surface_upperbound_preflight_summary.json"),
            "report_md": str(output_dir / "report.md"),
        },
        "per_view": per_view,
        "strict_truthfulness": {
            "uses_vggt_depth_point_normal": False,
            "creates_candidate_predictions": False,
            "allows_cloud": False,
            "can_claim_mentor_pass": False,
            "note": (
                "This is a raw-image/SMPL-X/camera preflight only. It is not a candidate "
                "and cannot pass mentor requirements without true surface optimization and "
                "the existing full strict candidate gate."
            ),
        },
        "next_required_action": (
            "Install/provide a differentiable renderer or implement an equivalent differentiable "
            "render-and-optimize module, then optimize the shared human surface from raw RGB/masks. "
            "Do not continue VGGT-observation recycling or threshold/support loops."
        ),
    }
    summary_path = output_dir / "raw_image_surface_upperbound_preflight_summary.json"
    summary_path.write_text(json.dumps(json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8")

    report = [
        "# Raw Image Surface Upper-Bound Preflight v0",
        "",
        f"Status: `{truthful_status}`",
        "",
        "## Inputs",
        "",
        f"- scene: `{scene_dir}`",
        f"- selected views: `{len(selected_indices)}` / `{len(views)}`",
        f"- camera source: `{camera_source}`",
        f"- SMPL-X model: `{model_path}`",
        "",
        "## Dependency Preflight",
        "",
        f"- differentiable renderer available: `{dep['differentiable_renderer_available']}`",
        f"- true gradient surface optimization available: `{dep['true_gradient_surface_optimization_available']}`",
        "",
        "## Starting SMPL-X Silhouette Fit",
        "",
        f"- mean full IoU: `{mean_iou}`",
        f"- mean full target recall: `{mean_recall}`",
        f"- asset preflight pass: `{asset_preflight_pass}`",
        f"- true Stage A ready: `{true_stage_a_ready}`",
        "",
        "## Truthful Interpretation",
        "",
        "This preflight does not use VGGT depth, point, normal, or confidence as geometry evidence.",
        "It only audits whether raw images, masks, calibrated cameras, and SMPL-X scaffold can",
        "serve as the starting point for a true raw-image differentiable surface upper-bound.",
        "",
        "It does not create candidate predictions, does not unblock cloud, and cannot be reported",
        "as mentor success.",
        "",
        "## Next Required Action",
        "",
        summary["next_required_action"],
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(json_ready({k: summary[k] for k in ("truthful_status", "metrics", "outputs")}), indent=2))
    return 0 if true_stage_a_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
