from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_SIZE = 518
BACKGROUND_DARK = np.array([16, 19, 26], dtype=np.uint8)
PRIOR_NORMAL_CHANNELS = ("smplx_cam_nx", "smplx_cam_ny", "smplx_cam_nz")
PRIOR_NORMAL_INDEXES = (26, 27, 28)


@dataclass(frozen=True)
class CaseSpec:
    slug: str
    case_id: str
    view_tag: str
    scene_dir: Path
    predictions_npz: Path
    prior_maps_npz: Path
    variant: str
    source_label: str
    point_source: str = "depth_unprojection"


@dataclass(frozen=True)
class ProbeSpec:
    case_id: str
    view_tag: str
    inputs_npz: Path
    predictions_npz: Path
    summary_json: Path
    source_label: str = "pred_normal_frozen_probe"


def default_output_dir() -> Path:
    return REPO_ROOT / "output" / f"coarse_prior_normal_pack_{date.today().strftime('%Y%m%d')}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a mentor-facing coarse prior normal advisor pack.")
    parser.add_argument("--output-dir", default=str(default_output_dir()), help="Output directory root")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for point subsampling")
    parser.add_argument(
        "--conf-percentile",
        type=float,
        default=40.0,
        help="Point confidence percentile used before point-cloud rendering",
    )
    parser.add_argument(
        "--max-full-points",
        type=int,
        default=150000,
        help="Maximum points per full-body point-cloud render",
    )
    parser.add_argument(
        "--max-roi-points",
        type=int,
        default=65000,
        help="Maximum points per head/face ROI point-cloud render",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def open_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def open_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0


def preprocess_to_square(img: Image.Image, target_size: int, resample: Image.Resampling, fill: tuple[int, ...]) -> np.ndarray:
    width, height = img.size
    if width >= height:
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
    else:
        new_height = target_size
        new_width = round(width * (new_height / height) / 14) * 14
    resized = img.resize((new_width, new_height), resample)
    arr = np.asarray(resized)
    if arr.ndim == 2:
        canvas = np.full((target_size, target_size), fill[0], dtype=arr.dtype)
        top = (target_size - new_height) // 2
        left = (target_size - new_width) // 2
        canvas[top : top + new_height, left : left + new_width] = arr
        return canvas
    channels = arr.shape[2]
    canvas = np.full((target_size, target_size, channels), fill[:channels], dtype=arr.dtype)
    top = (target_size - new_height) // 2
    left = (target_size - new_width) // 2
    canvas[top : top + new_height, left : left + new_width] = arr
    return canvas


def scene_manifest_view_paths(scene_dir: Path, subdir: str) -> list[Path]:
    manifest = load_json(scene_dir / "scene_manifest.json")
    key = "image_path" if subdir == "images" else "mask_path"
    return [Path(item[key]) for item in manifest["exported_views"]]


def load_preprocessed_rgb_stack(scene_dir: Path) -> np.ndarray:
    image_paths = scene_manifest_view_paths(scene_dir, "images")
    images = [
        preprocess_to_square(Image.open(path).convert("RGB"), TARGET_SIZE, Image.Resampling.BILINEAR, (255, 255, 255))
        for path in image_paths
    ]
    return np.stack(images, axis=0).astype(np.uint8)


def load_preprocessed_mask_stack(scene_dir: Path) -> np.ndarray:
    mask_paths = scene_manifest_view_paths(scene_dir, "masks")
    masks = [
        preprocess_to_square(Image.open(path).convert("L"), TARGET_SIZE, Image.Resampling.NEAREST, (0,)).astype(bool)
        for path in mask_paths
    ]
    return np.stack(masks, axis=0)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def expand_box(
    box: tuple[int, int, int, int] | None,
    image_shape: tuple[int, int],
    pad_x_ratio: float,
    pad_y_ratio: float,
) -> tuple[int, int, int, int] | None:
    if box is None:
        return None
    height, width = image_shape
    x0, y0, x1, y1 = box
    box_w = x1 - x0
    box_h = y1 - y0
    pad_x = int(round(box_w * pad_x_ratio))
    pad_y = int(round(box_h * pad_y_ratio))
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(width, x1 + pad_x),
        min(height, y1 + pad_y),
    )


def head_and_face_boxes(mask: np.ndarray) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
    bbox = mask_bbox(mask)
    if bbox is None:
        return None, None
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0

    head_height = max(24, int(round(height * 0.44)))
    head_box = (x0, y0, x1, min(y1, y0 + head_height))
    head_box = expand_box(head_box, mask.shape, pad_x_ratio=0.10, pad_y_ratio=0.06)

    face_width = max(24, int(round(width * 0.56)))
    face_height = max(24, int(round(head_height * 0.66)))
    face_cx = (x0 + x1) // 2
    face_x0 = max(x0, face_cx - face_width // 2)
    face_x1 = min(x1, face_x0 + face_width)
    face_y0 = y0 + int(round(head_height * 0.11))
    face_y1 = min(y1, face_y0 + face_height)
    face_box = (face_x0, face_y0, face_x1, face_y1)
    face_box = expand_box(face_box, mask.shape, pad_x_ratio=0.22, pad_y_ratio=0.18)
    return head_box, face_box


def crop_box(arr: np.ndarray, box: tuple[int, int, int, int] | None) -> np.ndarray:
    if box is None:
        return arr
    x0, y0, x1, y1 = box
    return arr[y0:y1, x0:x1]


def map_box_between_resolutions(
    box: tuple[int, int, int, int] | None,
    src_shape: tuple[int, int],
    dst_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if box is None:
        return None
    src_h, src_w = src_shape
    dst_h, dst_w = dst_shape
    x_scale = dst_w / float(src_w)
    y_scale = dst_h / float(src_h)
    x0, y0, x1, y1 = box
    return (
        int(round(x0 * x_scale)),
        int(round(y0 * y_scale)),
        int(round(x1 * x_scale)),
        int(round(y1 * y_scale)),
    )


def normal_to_rgb(normals: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    rgb = np.clip((normals.astype(np.float32) + 1.0) * 0.5, 0.0, 1.0)
    rgb = (rgb * 255.0).astype(np.uint8)
    if mask is not None:
        rgb = rgb.copy()
        rgb[~mask] = 255
    return rgb


def select_overview_indices(num_views: int) -> list[int]:
    if num_views <= 6:
        return list(range(num_views))
    count = 6 if num_views >= 18 else 4
    indices = {0, num_views - 1}
    for idx in range(count):
        indices.add(int(round(idx * (num_views - 1) / max(1, count - 1))))
    return sorted(indices)


def save_panel_grid(
    images: list[np.ndarray],
    titles: list[str],
    output_path: Path,
    suptitle: str,
    ncols: int = 3,
    figsize_per_panel: tuple[float, float] = (4.2, 4.8),
) -> None:
    n = len(images)
    ncols = min(ncols, n)
    nrows = int(math.ceil(n / float(ncols)))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    fig.patch.set_facecolor("white")
    for idx, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if idx >= n:
            continue
        ax.imshow(images[idx])
        ax.set_title(titles[idx], fontsize=11)
    fig.suptitle(suptitle, fontsize=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_roi_triptych(
    rgb_crop: np.ndarray,
    normal_crop: np.ndarray,
    output_path: Path,
    suptitle: str,
    normal_title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 5.2))
    fig.patch.set_facecolor("white")
    for ax, image, title in zip(axes, [rgb_crop, normal_crop], ["RGB ROI", normal_title]):
        ax.imshow(image)
        ax.set_title(title, fontsize=12)
        ax.set_axis_off()
    fig.suptitle(suptitle, fontsize=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_fullbody_pair(
    rgb_image: np.ndarray,
    normal_image: np.ndarray,
    output_path: Path,
    suptitle: str,
    normal_title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.8))
    fig.patch.set_facecolor("white")
    for ax, image, title in zip(axes, [rgb_image, normal_image], ["RGB full body", normal_title]):
        ax.imshow(image)
        ax.set_title(title, fontsize=12)
        ax.set_axis_off()
    fig.suptitle(suptitle, fontsize=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_single_image_panel(
    image: np.ndarray,
    output_path: Path,
    suptitle: str,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 6.8))
    fig.patch.set_facecolor("white")
    ax.imshow(image)
    ax.set_axis_off()
    fig.suptitle(suptitle, fontsize=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_prior_bundle(prior_maps_npz: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    prior_payload = np.load(prior_maps_npz, allow_pickle=False)
    channels = prior_payload["prior_channels"].tolist()
    if list(channels[idx] for idx in PRIOR_NORMAL_INDEXES) != list(PRIOR_NORMAL_CHANNELS):
        raise ValueError(f"Unexpected prior normal channels in {prior_maps_npz}")
    prior_normals = np.stack([prior_payload["prior_maps"][:, idx] for idx in PRIOR_NORMAL_INDEXES], axis=-1)
    prior_mask = prior_payload["prior_mask"].astype(bool)
    return prior_normals.astype(np.float32), prior_mask, channels


def save_prior_normal_outputs(case: CaseSpec, output_dir: Path, inventory: list[dict[str, Any]]) -> dict[str, Path]:
    normals_dir = output_dir / "coarse_prior_normals"
    manifest = load_json(case.scene_dir / "scene_manifest.json")
    image_paths = scene_manifest_view_paths(case.scene_dir, "images")
    mask_paths = scene_manifest_view_paths(case.scene_dir, "masks")
    target_rgb = open_rgb(image_paths[0])
    target_mask = open_mask(mask_paths[0])
    prior_normals, prior_mask, _ = load_prior_bundle(case.prior_maps_npz)
    prior_rgb = normal_to_rgb(prior_normals, mask=prior_mask)

    selected = select_overview_indices(prior_rgb.shape[0])
    overview_images = [prior_rgb[idx] for idx in selected]
    overview_titles = [
        f"view {idx:02d} cam {manifest['exported_views'][idx]['camera_id']}"
        for idx in selected
    ]
    overview_path = normals_dir / f"{case.case_id}_coarse_prior_normal_overview_{case.view_tag}.png"
    full_body_normal_path = normals_dir / f"{case.case_id}_targetcam00_fullbody_coarse_prior_normal_{case.view_tag}.png"
    full_body_pair_path = normals_dir / f"{case.case_id}_targetcam00_fullbody_rgb_vs_coarse_prior_normal_{case.view_tag}.png"
    save_panel_grid(
        overview_images,
        overview_titles,
        overview_path,
        suptitle=f"{case.case_id} {case.view_tag} coarse prior normal overview",
        ncols=3,
    )
    save_single_image_panel(
        prior_rgb[0],
        full_body_normal_path,
        suptitle=f"{case.case_id} target cam00 full-body coarse prior normal ({case.view_tag})",
    )
    save_fullbody_pair(
        target_rgb,
        prior_rgb[0],
        full_body_pair_path,
        suptitle=f"{case.case_id} target cam00 RGB vs full-body coarse prior normal ({case.view_tag})",
        normal_title="SMPL-X coarse prior normal full body",
    )

    head_box_small, face_box_small = head_and_face_boxes(prior_mask[0])
    head_box_rgb = map_box_between_resolutions(head_box_small, prior_mask[0].shape, target_mask.shape)
    face_box_rgb = map_box_between_resolutions(face_box_small, prior_mask[0].shape, target_mask.shape)

    head_rgb_crop = crop_box(target_rgb, head_box_rgb)
    face_rgb_crop = crop_box(target_rgb, face_box_rgb)
    head_normal_crop = crop_box(prior_rgb[0], head_box_small)
    face_normal_crop = crop_box(prior_rgb[0], face_box_small)

    head_normal_path = normals_dir / f"{case.case_id}_targetcam00_head_roi_coarse_prior_normal_{case.view_tag}.png"
    face_normal_path = normals_dir / f"{case.case_id}_targetcam00_face_roi_coarse_prior_normal_{case.view_tag}.png"
    head_path = normals_dir / f"{case.case_id}_targetcam00_head_roi_rgb_vs_coarse_prior_normal_{case.view_tag}.png"
    face_path = normals_dir / f"{case.case_id}_targetcam00_face_roi_rgb_vs_coarse_prior_normal_{case.view_tag}.png"
    save_single_image_panel(
        head_normal_crop,
        head_normal_path,
        suptitle=f"{case.case_id} target cam00 head ROI coarse prior normal ({case.view_tag})",
    )
    save_single_image_panel(
        face_normal_crop,
        face_normal_path,
        suptitle=f"{case.case_id} target cam00 face ROI coarse prior normal ({case.view_tag})",
    )
    save_roi_triptych(
        head_rgb_crop,
        head_normal_crop,
        head_path,
        suptitle=f"{case.case_id} head ROI RGB vs coarse prior normal ({case.view_tag})",
        normal_title="SMPL-X coarse prior normal ROI",
    )
    save_roi_triptych(
        face_rgb_crop,
        face_normal_crop,
        face_path,
        suptitle=f"{case.case_id} face ROI RGB vs coarse prior normal ({case.view_tag})",
        normal_title="SMPL-X coarse prior normal ROI",
    )

    outputs = {
        "overview": overview_path,
        "full_body_normal": full_body_normal_path,
        "full_body_pair": full_body_pair_path,
        "head_normal": head_normal_path,
        "face_normal": face_normal_path,
        "head_roi": head_path,
        "face_roi": face_path,
    }
    inventory.extend(
        [
            {
                "category": "coarse_prior_normals",
                "path": str(overview_path.relative_to(output_dir)),
                "description": f"{case.view_tag} coarse prior normal overview contact sheet",
            },
            {
                "category": "coarse_prior_normals",
                "path": str(full_body_normal_path.relative_to(output_dir)),
                "description": f"{case.view_tag} target full-body coarse prior normal single image",
            },
            {
                "category": "coarse_prior_normals",
                "path": str(full_body_pair_path.relative_to(output_dir)),
                "description": f"{case.view_tag} target full-body RGB vs coarse prior normal",
            },
            {
                "category": "coarse_prior_normals",
                "path": str(head_normal_path.relative_to(output_dir)),
                "description": f"{case.view_tag} target head ROI coarse prior normal single image",
            },
            {
                "category": "coarse_prior_normals",
                "path": str(head_path.relative_to(output_dir)),
                "description": f"{case.view_tag} target head ROI RGB vs coarse prior normal",
            },
            {
                "category": "coarse_prior_normals",
                "path": str(face_normal_path.relative_to(output_dir)),
                "description": f"{case.view_tag} target face ROI coarse prior normal single image",
            },
            {
                "category": "coarse_prior_normals",
                "path": str(face_path.relative_to(output_dir)),
                "description": f"{case.view_tag} target face ROI RGB vs coarse prior normal",
            },
        ]
    )
    return outputs


def save_probe_outputs(probe: ProbeSpec, output_dir: Path, inventory: list[dict[str, Any]]) -> dict[str, Path]:
    normals_dir = output_dir / "failed_predicted_normal_probe"
    inputs = np.load(probe.inputs_npz, allow_pickle=False)
    preds = np.load(probe.predictions_npz, allow_pickle=False)
    pred_normals = preds["normal"].astype(np.float32)
    pred_mask = inputs["prior_mask"].astype(bool)
    pred_rgb = normal_to_rgb(pred_normals, mask=pred_mask)
    rgb_inputs = inputs["images"].astype(np.uint8)

    selected = select_overview_indices(pred_rgb.shape[0])
    overview_images = [pred_rgb[idx] for idx in selected]
    overview_titles = [f"probe view {idx:02d}" for idx in selected]
    overview_path = normals_dir / f"{probe.case_id}_failed_predicted_normal_probe_overview_{probe.view_tag}.png"
    full_body_normal_path = normals_dir / f"{probe.case_id}_silhouette_only_collapse_fullbody_pred_normal_{probe.view_tag}.png"
    full_body_pair_path = normals_dir / f"{probe.case_id}_failed_predicted_normal_probe_rgb_vs_pred_normal_{probe.view_tag}.png"
    save_panel_grid(
        overview_images,
        overview_titles,
        overview_path,
        suptitle=f"{probe.case_id} failed predicted normal probe overview ({probe.view_tag})",
        ncols=2,
        figsize_per_panel=(4.4, 4.6),
    )
    save_single_image_panel(
        pred_rgb[0],
        full_body_normal_path,
        suptitle=f"{probe.case_id} silhouette-only collapse full-body predicted normal ({probe.view_tag})",
    )
    save_fullbody_pair(
        rgb_inputs[0],
        pred_rgb[0],
        full_body_pair_path,
        suptitle=f"{probe.case_id} failed predicted normal probe RGB vs full-body prediction ({probe.view_tag})",
        normal_title="Failed predicted normal probe",
    )

    head_box_small, face_box_small = head_and_face_boxes(pred_mask[0])
    head_path = normals_dir / f"{probe.case_id}_silhouette_only_collapse_head_roi_{probe.view_tag}.png"
    face_path = normals_dir / f"{probe.case_id}_silhouette_only_collapse_face_roi_{probe.view_tag}.png"
    save_roi_triptych(
        crop_box(rgb_inputs[0], head_box_small),
        crop_box(pred_rgb[0], head_box_small),
        head_path,
        suptitle=f"{probe.case_id} failed predicted normal probe: silhouette-only collapse head ROI ({probe.view_tag})",
        normal_title="Silhouette-only collapse",
    )
    save_roi_triptych(
        crop_box(rgb_inputs[0], face_box_small),
        crop_box(pred_rgb[0], face_box_small),
        face_path,
        suptitle=f"{probe.case_id} failed predicted normal probe: silhouette-only collapse face ROI ({probe.view_tag})",
        normal_title="Silhouette-only collapse",
    )

    outputs = {
        "overview": overview_path,
        "full_body_normal": full_body_normal_path,
        "full_body_pair": full_body_pair_path,
        "head_roi": head_path,
        "face_roi": face_path,
    }
    inventory.extend(
        [
            {
                "category": "failed_predicted_normal_probe",
                "path": str(overview_path.relative_to(output_dir)),
                "description": "4-view failed predicted-normal probe overview",
            },
            {
                "category": "failed_predicted_normal_probe",
                "path": str(full_body_normal_path.relative_to(output_dir)),
                "description": "4-view silhouette-only collapse full-body predicted normal single image",
            },
            {
                "category": "failed_predicted_normal_probe",
                "path": str(full_body_pair_path.relative_to(output_dir)),
                "description": "4-view failed predicted-normal probe RGB vs predicted normal",
            },
            {
                "category": "failed_predicted_normal_probe",
                "path": str(head_path.relative_to(output_dir)),
                "description": "4-view silhouette-only collapse head ROI",
            },
            {
                "category": "failed_predicted_normal_probe",
                "path": str(face_path.relative_to(output_dir)),
                "description": "4-view silhouette-only collapse face ROI",
            },
        ]
    )
    return outputs


def closed_form_inverse_se3_numpy(se3: np.ndarray) -> np.ndarray:
    rotation = se3[:, :3, :3]
    translation = se3[:, :3, 3:]
    rotation_t = np.transpose(rotation, (0, 2, 1))
    top_right = -np.matmul(rotation_t, translation)
    inverted = np.tile(np.eye(4, dtype=se3.dtype), (len(rotation), 1, 1))
    inverted[:, :3, :3] = rotation_t
    inverted[:, :3, 3:] = top_right
    return inverted


def unproject_depth_map_to_point_map_numpy(
    depth_map: np.ndarray,
    extrinsics_cam: np.ndarray,
    intrinsics_cam: np.ndarray,
) -> np.ndarray:
    cam_to_world = closed_form_inverse_se3_numpy(extrinsics_cam)
    world_points = []
    for frame_idx in range(depth_map.shape[0]):
        depth = depth_map[frame_idx].squeeze(-1).astype(np.float32)
        intrinsic = intrinsics_cam[frame_idx].astype(np.float32)
        height, width = depth.shape
        u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))

        fu, fv = intrinsic[0, 0], intrinsic[1, 1]
        cu, cv = intrinsic[0, 2], intrinsic[1, 2]
        x_cam = (u - cu) * depth / fu
        y_cam = (v - cv) * depth / fv
        z_cam = depth
        cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1)

        rotation = cam_to_world[frame_idx, :3, :3]
        translation = cam_to_world[frame_idx, :3, 3]
        world = np.dot(cam_coords, rotation.T) + translation
        world_points.append(world.astype(np.float32))
    return np.stack(world_points, axis=0)


def resolve_point_source(payload: np.lib.npyio.NpzFile, point_source: str) -> tuple[np.ndarray, np.ndarray]:
    if point_source == "world_points":
        return payload["world_points"], payload["world_points_conf"]
    if point_source == "depth_unprojection":
        world_points = unproject_depth_map_to_point_map_numpy(payload["depth"], payload["extrinsic"], payload["intrinsic"])
        return world_points, payload["depth_conf"]
    raise ValueError(f"Unsupported point source: {point_source}")


def load_case_point_data(case: CaseSpec) -> dict[str, Any]:
    payload = np.load(case.predictions_npz, allow_pickle=False)
    world_points, point_conf = resolve_point_source(payload, case.point_source)
    rgb_stack = load_preprocessed_rgb_stack(case.scene_dir)
    mask_stack = load_preprocessed_mask_stack(case.scene_dir)
    prior_normals, prior_mask, _ = load_prior_bundle(case.prior_maps_npz)
    return {
        "world_points": world_points.astype(np.float32),
        "point_conf": point_conf.astype(np.float32),
        "rgb_stack": rgb_stack.astype(np.uint8),
        "mask_stack": mask_stack.astype(bool),
        "prior_normals": prior_normals,
        "prior_mask": prior_mask.astype(bool),
        "scene_manifest": load_json(case.scene_dir / "scene_manifest.json"),
    }


def flatten_masked_points(
    world_points: np.ndarray,
    point_conf: np.ndarray,
    rgb_stack: np.ndarray,
    mask_stack: np.ndarray,
    conf_percentile: float,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    points = world_points.reshape(-1, 3)
    conf = point_conf.reshape(-1)
    colors = rgb_stack.reshape(-1, 3)
    mask = mask_stack.reshape(-1)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(conf) & (conf > 0) & mask
    if not np.any(valid):
        raise RuntimeError("No valid masked points available")
    conf_threshold = float(np.percentile(conf[valid], conf_percentile))
    keep = valid & (conf >= conf_threshold)
    indices = np.flatnonzero(keep)
    if len(indices) > max_points:
        indices = rng.choice(indices, size=max_points, replace=False)
    return points[indices], colors[indices]


def sample_view_roi_points(
    world_points: np.ndarray,
    point_conf: np.ndarray,
    mask: np.ndarray,
    box: tuple[int, int, int, int] | None,
) -> np.ndarray:
    if box is None:
        return np.empty((0, 3), dtype=np.float32)
    crop_mask = np.zeros(mask.shape, dtype=bool)
    x0, y0, x1, y1 = box
    crop_mask[y0:y1, x0:x1] = True
    valid = crop_mask & mask & np.isfinite(point_conf) & (point_conf > 0) & np.isfinite(world_points).all(axis=-1)
    return world_points[valid].astype(np.float32)


def robust_aabb(points: np.ndarray, margin_ratio: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        raise RuntimeError("ROI point set is empty; cannot compute AABB")
    lower = np.percentile(points, 5.0, axis=0)
    upper = np.percentile(points, 95.0, axis=0)
    span = np.maximum(upper - lower, 1e-4)
    margin = span * margin_ratio
    return lower - margin, upper + margin


def make_box_mask(points: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.all(points >= lower[None, :], axis=1) & np.all(points <= upper[None, :], axis=1)


def build_reference_boxes(reference_case: CaseSpec) -> dict[str, Any]:
    data = load_case_point_data(reference_case)
    prior_mask = data["prior_mask"][0]
    head_box, face_box = head_and_face_boxes(prior_mask)
    body_points, _ = flatten_masked_points(
        data["world_points"],
        data["point_conf"],
        data["rgb_stack"],
        data["mask_stack"],
        conf_percentile=40.0,
        max_points=220000,
        rng=np.random.default_rng(0),
    )
    head_points = sample_view_roi_points(data["world_points"][0], data["point_conf"][0], data["mask_stack"][0], head_box)
    face_points = sample_view_roi_points(data["world_points"][0], data["point_conf"][0], data["mask_stack"][0], face_box)
    return {
        "head_box": head_box,
        "face_box": face_box,
        "head_aabb": robust_aabb(head_points, margin_ratio=0.16),
        "face_aabb": robust_aabb(face_points, margin_ratio=0.18),
        "body_aabb": robust_aabb(body_points, margin_ratio=0.08),
    }


def rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cx, sx = np.cos(pitch), np.sin(pitch)
    rot_y = np.array(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ],
        dtype=np.float32,
    )
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cx, -sx],
            [0.0, sx, cx],
        ],
        dtype=np.float32,
    )
    return rot_x @ rot_y


def render_projected_points(
    ax: plt.Axes,
    points: np.ndarray,
    colors: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    title: str,
    fixed_aabb: tuple[np.ndarray, np.ndarray] | None,
    point_size: float,
) -> None:
    if len(points) == 0:
        ax.set_facecolor(BACKGROUND_DARK / 255.0)
        ax.set_title(title, fontsize=11, color="white")
        ax.set_axis_off()
        return
    rot = rotation_matrix(yaw_deg, pitch_deg)
    rotated = points @ rot.T
    order = np.argsort(rotated[:, 2])
    projected = rotated[order]
    projected_colors = colors[order].astype(np.float32) / 255.0
    ax.scatter(
        projected[:, 0],
        -projected[:, 1],
        c=projected_colors,
        s=point_size,
        linewidths=0,
        marker="o",
        alpha=0.95,
    )
    ax.set_title(title, fontsize=11, color="white")
    ax.set_axis_off()
    ax.set_facecolor(BACKGROUND_DARK / 255.0)
    ax.set_aspect("equal", adjustable="box")
    if fixed_aabb is not None:
        lower, upper = fixed_aabb
        corners = np.array(
            [
                [lower[0], lower[1], lower[2]],
                [lower[0], lower[1], upper[2]],
                [lower[0], upper[1], lower[2]],
                [lower[0], upper[1], upper[2]],
                [upper[0], lower[1], lower[2]],
                [upper[0], lower[1], upper[2]],
                [upper[0], upper[1], lower[2]],
                [upper[0], upper[1], upper[2]],
            ],
            dtype=np.float32,
        )
        rotated_corners = corners @ rot.T
        x_coords = rotated_corners[:, 0]
        y_coords = -rotated_corners[:, 1]
    else:
        x_coords = projected[:, 0]
        y_coords = -projected[:, 1]
    if len(x_coords) == 0:
        x_coords = np.array([-1.0, 1.0], dtype=np.float32)
        y_coords = np.array([-1.0, 1.0], dtype=np.float32)
    x_margin = max(1e-3, (x_coords.max() - x_coords.min()) * 0.08)
    y_margin = max(1e-3, (y_coords.max() - y_coords.min()) * 0.08)
    ax.set_xlim(float(x_coords.min() - x_margin), float(x_coords.max() + x_margin))
    ax.set_ylim(float(y_coords.min() - y_margin), float(y_coords.max() + y_margin))


def render_triptych(
    points: np.ndarray,
    colors: np.ndarray,
    output_path: Path,
    suptitle: str,
    fixed_aabb: tuple[np.ndarray, np.ndarray] | None,
    point_size: float,
    figure_size: tuple[float, float] = (13.6, 4.8),
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=figure_size)
    fig.patch.set_facecolor(BACKGROUND_DARK / 255.0)
    views = [
        ("Front", 0.0, 0.0),
        ("Three-quarter", 35.0, -10.0),
        ("Side", 90.0, 0.0),
    ]
    for ax, (title, yaw_deg, pitch_deg) in zip(axes, views):
        render_projected_points(ax, points, colors, yaw_deg, pitch_deg, title, fixed_aabb, point_size=point_size)
    fig.suptitle(suptitle, color="white", fontsize=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_pointcloud_outputs(
    case: CaseSpec,
    output_dir: Path,
    inventory: list[dict[str, Any]],
    reference_boxes: dict[str, Any],
    conf_percentile: float,
    max_full_points: int,
    max_roi_points: int,
    rng_seed: int,
) -> dict[str, Path]:
    point_dir = output_dir / "pointcloud_open3d"
    data = load_case_point_data(case)
    rng = np.random.default_rng(rng_seed)
    full_points, full_colors = flatten_masked_points(
        data["world_points"],
        data["point_conf"],
        data["rgb_stack"],
        data["mask_stack"],
        conf_percentile=conf_percentile,
        max_points=max_full_points,
        rng=rng,
    )
    head_mask = make_box_mask(full_points, *reference_boxes["head_aabb"])
    face_mask = make_box_mask(full_points, *reference_boxes["face_aabb"])
    head_points = full_points[head_mask]
    head_colors = full_colors[head_mask]
    face_points = full_points[face_mask]
    face_colors = full_colors[face_mask]

    if len(head_points) > max_roi_points:
        head_idx = rng.choice(len(head_points), size=max_roi_points, replace=False)
        head_points = head_points[head_idx]
        head_colors = head_colors[head_idx]
    if len(face_points) > max_roi_points:
        face_idx = rng.choice(len(face_points), size=max_roi_points, replace=False)
        face_points = face_points[face_idx]
        face_colors = face_colors[face_idx]

    body_path = point_dir / f"{case.case_id}_open3d_human_full_{case.view_tag}_{case.variant}.png"
    head_path = point_dir / f"{case.case_id}_open3d_head_closeup_{case.view_tag}_{case.variant}.png"
    face_path = point_dir / f"{case.case_id}_open3d_face_closeup_{case.view_tag}_{case.variant}.png"

    render_triptych(
        full_points,
        full_colors,
        body_path,
        suptitle=f"{case.case_id} human full point cloud ({case.view_tag}, {case.variant})",
        fixed_aabb=reference_boxes["body_aabb"],
        point_size=0.35,
        figure_size=(13.8, 4.9),
    )
    render_triptych(
        head_points,
        head_colors,
        head_path,
        suptitle=f"{case.case_id} head close-up ({case.view_tag}, {case.variant})",
        fixed_aabb=reference_boxes["head_aabb"],
        point_size=1.8,
        figure_size=(13.8, 4.8),
    )
    render_triptych(
        face_points,
        face_colors,
        face_path,
        suptitle=f"{case.case_id} face close-up ({case.view_tag}, {case.variant})",
        fixed_aabb=reference_boxes["face_aabb"],
        point_size=2.1,
        figure_size=(13.8, 4.8),
    )

    outputs = {"body": body_path, "head": head_path, "face": face_path}
    inventory.extend(
        [
            {
                "category": "pointcloud_open3d",
                "path": str(body_path.relative_to(output_dir)),
                "description": f"{case.view_tag} {case.variant} full-body point cloud triptych",
            },
            {
                "category": "pointcloud_open3d",
                "path": str(head_path.relative_to(output_dir)),
                "description": f"{case.view_tag} {case.variant} head close-up point cloud triptych",
            },
            {
                "category": "pointcloud_open3d",
                "path": str(face_path.relative_to(output_dir)),
                "description": f"{case.view_tag} {case.variant} face close-up point cloud triptych",
            },
        ]
    )
    return outputs


def compose_image_strip(
    image_paths: list[Path],
    titles: list[str],
    output_path: Path,
    suptitle: str,
    figsize_per_panel: tuple[float, float] = (4.8, 4.6),
) -> None:
    images = [open_rgb(path) for path in image_paths]
    fig, axes = plt.subplots(1, len(images), figsize=(figsize_per_panel[0] * len(images), figsize_per_panel[1]))
    if len(images) == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image)
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()
    fig.suptitle(suptitle, fontsize=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def compose_image_grid_from_paths(
    image_paths: list[Path],
    titles: list[str],
    output_path: Path,
    suptitle: str,
    ncols: int = 2,
    figsize_per_panel: tuple[float, float] = (5.0, 4.8),
) -> None:
    images = [open_rgb(path) for path in image_paths]
    n = len(images)
    ncols = min(ncols, n)
    nrows = int(math.ceil(n / float(ncols)))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    fig.patch.set_facecolor("white")
    for idx, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if idx >= n:
            continue
        ax.imshow(images[idx])
        ax.set_title(titles[idx], fontsize=11)
    fig.suptitle(suptitle, fontsize=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_refined_pending_panel(output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f4f4f4")
    ax.set_axis_off()
    ax.text(
        0.5,
        0.56,
        title,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#333333",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.42,
        "pending real refined normal output\nnot fabricated",
        ha="center",
        va="center",
        fontsize=11,
        color="#555555",
        transform=ax.transAxes,
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_refined_overview_templates(
    output_dir: Path,
    prior_60: dict[str, Path],
    prior_13: dict[str, Path],
    prior_7: dict[str, Path],
    inventory: list[dict[str, Any]],
) -> tuple[Path, Path]:
    overview_dir = output_dir / "overview_layouts"
    placeholder_dir = overview_dir / "_template_placeholders"
    pending_full = placeholder_dir / "refined_normal_pending_fullbody.png"
    pending_head = placeholder_dir / "refined_normal_pending_head.png"
    make_refined_pending_panel(pending_full, "Refined normal")
    make_refined_pending_panel(pending_head, "Refined head normal")

    full_template = overview_dir / "0012_11_compare_coarse_prior_fullbody_60_13_7_refined_template.png"
    head_template = overview_dir / "0012_11_compare_coarse_prior_head_60_13_7_refined_template.png"
    compose_image_strip(
        [prior_60["full_body_normal"], prior_13["full_body_normal"], prior_7["full_body_normal"], pending_full],
        ["60v coarse prior", "13v coarse prior", "7v coarse prior", "refined normal pending"],
        full_template,
        suptitle="0012_11 fixed overview layout: 60v / 13v / 7v / refined",
        figsize_per_panel=(4.8, 4.8),
    )
    compose_image_strip(
        [prior_60["head_normal"], prior_13["head_normal"], prior_7["head_normal"], pending_head],
        ["60v coarse prior head", "13v coarse prior head", "7v coarse prior head", "refined head pending"],
        head_template,
        suptitle="0012_11 fixed head ROI layout: 60v / 13v / 7v / refined",
        figsize_per_panel=(4.8, 4.8),
    )
    inventory.extend(
        [
            {
                "category": "overview_layouts",
                "path": str(full_template.relative_to(output_dir)),
                "description": "Fixed 60v/13v/7v/refined full-body overview template with refined slot marked pending",
            },
            {
                "category": "overview_layouts",
                "path": str(head_template.relative_to(output_dir)),
                "description": "Fixed 60v/13v/7v/refined head ROI overview template with refined slot marked pending",
            },
        ]
    )
    return full_template, head_template


def build_final_coarse_prior_normal_pass_pack(
    output_dir: Path,
    outputs: dict[str, dict[str, Path]],
    inventory: list[dict[str, Any]],
) -> tuple[Path, Path, Path]:
    final_dir = output_dir / "final_coarse_prior_normal_pass_pack"
    figures_dir = final_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    selected = [
        (
            outputs["normals_baseline_60v"]["full_body_pair"],
            figures_dir / "01_fullbody_rgb_vs_coarse_prior_normal_60v.png",
            "60v full-body RGB vs coarse prior normal",
        ),
        (
            outputs["normals_baseline_60v"]["head_roi"],
            figures_dir / "02_head_roi_rgb_vs_coarse_prior_normal_60v.png",
            "60v head ROI RGB vs coarse prior normal",
        ),
        (
            outputs["normals_baseline_60v"]["face_roi"],
            figures_dir / "03_face_roi_rgb_vs_coarse_prior_normal_60v.png",
            "60v face ROI RGB vs coarse prior normal",
        ),
        (
            outputs["normals_baseline_60v"]["overview"],
            figures_dir / "04_coarse_prior_overview_60v.png",
            "60v coarse prior normal overview",
        ),
    ]

    for src, dst, _ in selected:
        shutil.copy2(src, dst)

    storyboard_path = figures_dir / "00_coarse_prior_normal_storyboard_60v.png"
    compose_image_grid_from_paths(
        image_paths=[item[1] for item in selected],
        titles=[
            "Full body RGB vs coarse prior normal",
            "Head ROI RGB vs coarse prior normal",
            "Face ROI RGB vs coarse prior normal",
            "60v coarse prior overview",
        ],
        output_path=storyboard_path,
        suptitle="0012_11 coarse prior normal accepted route storyboard (60v)",
        ncols=2,
        figsize_per_panel=(5.4, 4.8),
    )

    readme_path = final_dir / "README.md"
    send_text_path = final_dir / "send_to_advisor.md"

    readme_text = "\n".join(
        [
            "# Coarse Prior Normal Pass Pack",
            "",
            "- Assumption for this pack: the mentor has accepted the `coarse prior normal` route as the current stage conclusion.",
            "- Goal: keep the mentor-facing packet focused on `60v` coarse prior alignment, not on the failed `4v probe` branch.",
            "",
            "## Recommended Viewing Order",
            "",
            "- `figures/00_coarse_prior_normal_storyboard_60v.png`: one-page summary for the current accepted coarse-prior route.",
            "- `figures/04_coarse_prior_overview_60v.png`: compact multi-view evidence that the coarse prior is view-aligned and stable.",
            "",
            "## Mentor Main 4",
            "",
            "- `figures/01_fullbody_rgb_vs_coarse_prior_normal_60v.png`",
            "- `figures/02_head_roi_rgb_vs_coarse_prior_normal_60v.png`",
            "- `figures/03_face_roi_rgb_vs_coarse_prior_normal_60v.png`",
            "- `figures/04_coarse_prior_overview_60v.png`",
            "",
            "## Notes",
            "",
            "- Current normal evidence is `SMPL-X view-aligned coarse prior normal`; the next step is high-resolution local detail refinement.",
            "- `7v / 13v` figures are supplementary only and should not replace the `60v` main four.",
            "- The failed `4v probe` branch has been archived separately as an internal debugging branch.",
        ]
    )
    write_text(readme_path, readme_text)

    send_text = "\n".join(
        [
            "这版我按 prior normal 这一路整理成了可直接汇报的版本。",
            "",
            "1. 现在已经能稳定输出人体 prior normal，大图、head ROI 和 face ROI 都补齐了。",
            "2. 这一路重点放在把 pose-aligned prior normal 接成几何约束，并观察它和 face/head 几何质量的对应关系。",
            "3. 当前最直观的图是 `00_prior_normal_storyboard_60v.png`，另外补了 `baseline vs surfacepose` 的 face 对比，以及 `7v/13v/60v` 的 sparse-view face 对比。",
            "",
            "这版里我不再把 frozen probe 当主结论，而是以 prior normal 作为当前认可的方向来汇报。",
        ]
    )
    write_text(send_text_path, send_text)

    inventory.extend(
        [
            {
                "category": "final_prior_normal_pass_pack",
                "path": str(storyboard_path.relative_to(output_dir)),
                "description": "One-page storyboard for the mentor-approved prior-normal route",
            },
            {
                "category": "final_prior_normal_pass_pack",
                "path": str(readme_path.relative_to(output_dir)),
                "description": "README for the prior-normal pass pack",
            },
            {
                "category": "final_prior_normal_pass_pack",
                "path": str(send_text_path.relative_to(output_dir)),
                "description": "Suggested Chinese message for the mentor",
            },
        ]
    )
    return final_dir, readme_path, send_text_path


def build_docs(
    output_dir: Path,
    inventory: list[dict[str, Any]],
    files: dict[str, dict[str, Path]],
) -> tuple[Path, Path, Path]:
    readme_path = output_dir / "README.md"
    assessment_path = output_dir / "docs" / "normal_pack_assessment.md"
    mentor_pack_path = output_dir / "docs" / "mentor_minimal_pack.md"

    inventory_lines = [
        f"- `{item['category']}`: `{item['path']}` - {item['description']}"
        for item in inventory
    ]

    readme_text = "\n".join(
        [
            "# Normal Advisor Pack",
            "",
            f"- Generated on: {date.today().isoformat()}",
            "- Case: `0012_11`, frame `0000`",
            "- Available sparse-view ladders in the current workspace: `7v`, `13v`, `60v`",
            "- Extra predicted-normal evidence: `4v` frozen-backbone normal-head probe from `2026-04-20`",
            "",
            "## Directory Layout",
            "",
            "- `normals/`: prior-normal overviews and large target-view head/face ROI panels",
            "- `pointcloud_open3d/`: point-cloud triptychs for full body, head, and face",
            "- `comparisons/`: baseline vs surfacepose and normal-to-point summary figures",
            "- `sparse_views/`: fixed-region 7/13/60 sparse-view comparisons",
            "- `docs/`: assessment note",
            "",
            "## File Inventory",
            "",
            *inventory_lines,
            "",
            "## Key Notes",
            "",
            "- `7v` and `13v` currently have baseline geometry plus prior-normal observation only.",
            "- Direct baseline vs `+surfacepose` geometry comparison is available for `60v` in this pack.",
            "- The `4v` frozen probe figures show predicted normals, but they are not end-to-end geometry training curves.",
        ]
    )
    write_text(readme_path, readme_text)

    assessment_text = "\n".join(
        [
            "# normal_pack_assessment",
            "",
            "## What This Pack Can Prove",
            "",
            "- The normal visualization chain is stable: the pack now contains readable full-body overviews and enlarged head/face ROI figures instead of tiny crops.",
            "- The workspace does support a direct geometry comparison for `60v`: baseline VGGT versus the `60v surfacepose` run.",
            "- The sparse-view geometry trend is now easier to inspect because the `7v / 13v / 60v` head and face comparisons use a fixed 3D ROI window.",
            "",
            "## What It Still Cannot Fully Prove",
            "",
            "- The current workspace does not contain end-to-end `+normal` geometry outputs for `7v` or `13v` as of 2026-04-21, so the sparse-view figures are still baseline-only at the geometry level.",
            "- The `4v` frozen normal-head probe demonstrates predicted normal output, but it is still a frozen-backbone probe rather than a full end-to-end training result.",
            "- Because the sparse-view ladder available locally is `7v / 13v / 60v`, this pack does not claim a true `6v / 12v / 60v` result.",
            "",
            "## Current Bottleneck",
            "",
            "- The main missing evidence is not visualization any more; it is the absence of matching `+normal` end-to-end geometry runs for sparse-view settings.",
            "- That means the highest-value next step is to produce `7v` or `13v` end-to-end surfacepose/normal-conditioned inference so the new figure templates can be filled with a true baseline-vs-normal sparse-view ablation.",
            "",
            "## Priority Recommendation",
            "",
            "- First priority: run sparse-view end-to-end `+normal` inference or short training for `7v` and `13v`.",
            "- Second priority: if those runs are still unstable, strengthen the normal supervision branch before spending more time on additional visualization variants.",
            "- Third priority: only after sparse-view `+normal` geometry is available should we refine the pack toward a smaller mentor-facing subset.",
        ]
    )
    write_text(assessment_path, assessment_text)

    mentor_pack_text = "\n".join(
        [
            "# mentor_minimal_pack",
            "",
            "## Must Send 4",
            "",
            "- `normals/0012_11_targetcam00_fullbody_rgb_vs_prior_normal_60v.png`: use this as the first figure because it directly shows the accepted prior-normal route on the target view.",
            "- `normals/0012_11_targetcam00_head_roi_rgb_vs_prior_normal_60v.png`: this keeps the focus on the head region the mentor has been asking about.",
            "- `normals/0012_11_targetcam00_face_roi_rgb_vs_prior_normal_60v.png`: this is the cleanest close-up for face detail under the accepted route.",
            "- `comparisons/0012_11_normal_to_point_face_demo_60v.png`: this is the bridge figure that connects normal context to geometry inspection.",
            "",
            "## Optional 4",
            "",
            "- `comparisons/0012_11_compare_baseline_vs_surfacepose_face_60v.png`: use this when the discussion shifts from the prior itself to whether the geometric result is improving.",
            "- `comparisons/0012_11_compare_baseline_vs_surfacepose_head_60v.png`: same comparison for the broader head region.",
            "- `sparse_views/0012_11_compare_views_face_7_13_60.png`: use this for the current sparse-view trend under the available cases.",
            "- `sparse_views/0012_11_compare_views_head_7_13_60.png`: same trend plot for the head region.",
            "",
            "## Not Recommended",
            "",
            "- `normals/0012_11_pred_normal_overview_4v_probe.png`: this is no longer the preferred main message if the prior-normal route is already accepted.",
            "- `normals/0012_11_face_roi_rgb_vs_pred_normal_4v_probe.png`: same reason as above.",
            "- `pointcloud_open3d/0012_11_open3d_human_full_60v_baseline.png`: too broad for a fast mentor-facing packet.",
            "- `normals/0012_11_prior_normal_overview_7v.png`: useful internally, but weaker than the enlarged target-view figures.",
        ]
    )
    write_text(mentor_pack_path, mentor_pack_text)
    return readme_path, assessment_path, mentor_pack_path


def build_docs_v2(
    output_dir: Path,
    inventory: list[dict[str, Any]],
) -> tuple[Path, Path, Path]:
    readme_path = output_dir / "README.md"
    assessment_path = output_dir / "docs" / "normal_pack_assessment.md"
    mentor_pack_path = output_dir / "docs" / "mentor_minimal_pack.md"

    inventory_lines = [
        f"- `{item['category']}`: `{item['path']}` - {item['description']}"
        for item in inventory
    ]

    readme_text = "\n".join(
        [
            "# Normal Advisor Pack",
            "",
            f"- Generated on: {date.today().isoformat()}",
            "- Case: `0012_11`, frame `0000`",
            "- Available sparse-view ladders in the current workspace: `7v`, `13v`, `60v`",
            "- Current main message: `SMPL-X view-aligned coarse prior normal`",
            "",
            "## Directory Layout",
            "",
            "- `coarse_prior_normals/`: all accepted coarse prior normal assets",
            "- `failed_predicted_normal_probe/`: archived `4v probe` failure assets only",
            "- `pointcloud_open3d/`: Open3D point-cloud close-up renders",
            "- `comparisons/`: geometry-support figures such as baseline vs surfacepose and normal-to-point context",
            "- `overview_layouts/`: 60v / 13v / 7v coarse prior normal comparison layouts",
            "- `docs/`: assessment, captions, and next-branch design spec",
            "",
            "## File Inventory",
            "",
            *inventory_lines,
            "",
            "## Key Notes",
            "",
            "- Current normal evidence is `SMPL-X view-aligned coarse prior normal`; the next step is high-resolution local detail refinement.",
            "- `7v` and `13v` are supplementary evidence that the coarse prior normal remains stable under sparser view counts.",
            "- Direct baseline vs `+surfacepose` geometry comparison is available for `60v` in this pack.",
            "- The `4v probe` has been moved out of the main normal result directory and archived as `failed predicted normal probe / silhouette-only collapse`.",
            "- Source view count increases produce only limited visible change in the current coarse prior normal, so the main bottleneck is prior expressiveness, not simply view count.",
        ]
    )
    write_text(readme_path, readme_text)

    assessment_text = "\n".join(
        [
            "# normal_pack_assessment",
            "",
            "## What This Pack Can Prove",
            "",
            "- The coarse prior normal visualization chain is stable: the pack now contains readable full-body overviews and enlarged head/face ROI figures instead of tiny crops.",
            "- The workspace does support a direct geometry comparison for `60v`: baseline VGGT versus the `60v surfacepose` run.",
            "- The `60v / 13v / 7v` coarse prior normal full-body and head-ROI comparison pages are now explicit and readable.",
            "",
            "## What It Still Cannot Fully Prove",
            "",
            "- The current workspace does not contain end-to-end `+normal` geometry outputs for `7v` or `13v` as of 2026-04-21, so the sparse-view figures are still baseline-only at the geometry level.",
            "- The `4v probe` does not demonstrate a usable predicted-normal result; it has been archived as a silhouette-only collapse branch.",
            "- Because the sparse-view ladder available locally is `7v / 13v / 60v`, this pack does not claim a true `6v / 12v / 60v` result.",
            "",
            "## Current Bottleneck",
            "",
            "- Source view count increases produce only limited visible changes in the current coarse prior normal.",
            "- That means the main bottleneck is the current prior expression ceiling, not simply insufficient input views.",
            "- The next high-value step is an image-aligned detail normal refinement branch, not another large sparse-view end-to-end retrain before the refinement branch is stable.",
            "",
            "## Priority Recommendation",
            "",
            "- First priority: keep the current mentor message on `coarse prior normal` and stop mixing it with failed predicted-normal probe material.",
            "- Second priority: implement `detail_normal_refiner` as image-aligned residual refinement on `head / neck` and `shoulder line` ROI crops.",
            "- Third priority: only after `60v` ROI refinement becomes stable should the branch be tested on `13v`, then `7v`.",
        ]
    )
    write_text(assessment_path, assessment_text)

    mentor_pack_text = "\n".join(
        [
            "# mentor_minimal_pack",
            "",
            "## Must Send 4",
            "",
            "- `coarse_prior_normals/0012_11_targetcam00_fullbody_rgb_vs_coarse_prior_normal_60v.png`: torso orientation and major limb orientation are aligned, but head boundary and hairline remain coarse.",
            "- `coarse_prior_normals/0012_11_targetcam00_head_roi_rgb_vs_coarse_prior_normal_60v.png`: head pose and coarse silhouette are correct, but ear contour and hair boundary are still blunt.",
            "- `coarse_prior_normals/0012_11_targetcam00_face_roi_rgb_vs_coarse_prior_normal_60v.png`: face orientation is stable, but eyes, nose bridge, lips, and hairline still lack high-frequency detail.",
            "- `coarse_prior_normals/0012_11_coarse_prior_normal_overview_60v.png`: multi-view alignment is stable, but the result should still be described as coarse prior normal rather than refined normal.",
            "",
            "## Supplementary",
            "",
            "- `overview_layouts/0012_11_compare_coarse_prior_fullbody_60_13_7.png`: source view count increases bring only limited visible change to the current full-body coarse prior normal.",
            "- `overview_layouts/0012_11_compare_coarse_prior_head_60_13_7.png`: source view count increases bring only limited visible change to the current head ROI coarse prior normal.",
            "- `comparisons/0012_11_compare_baseline_vs_surfacepose_face_60v.png`: use only when the discussion shifts from the prior itself to geometry improvement evidence.",
            "",
            "## Archived Internal Debug",
            "",
            "- `failed_predicted_normal_probe/0012_11_failed_predicted_normal_probe_overview_4v_probe.png`: this branch is now explicitly labeled `failed predicted normal probe`.",
            "- `failed_predicted_normal_probe/0012_11_silhouette_only_collapse_head_roi_4v_probe.png`: use only for internal discussion of the silhouette-only collapse failure mode.",
        ]
    )
    write_text(mentor_pack_path, mentor_pack_text)
    return readme_path, assessment_path, mentor_pack_path


def build_detail_normal_refiner_spec(output_dir: Path) -> Path:
    spec_path = output_dir / "docs" / "detail_normal_refiner_spec.md"
    spec_text = "\n".join(
        [
            "# detail_normal_refiner",
            "",
            "## Branch Name",
            "",
            "- Primary name: `detail_normal_refiner`",
            "- Alternative alias: `pifuhd_style_normal_refine`",
            "",
            "## Fixed Positioning",
            "",
            "- This branch does not replace VGGT.",
            "- This branch does not replace coarse prior normal.",
            "- This branch performs image-aligned residual refinement on top of coarse prior normal.",
            "",
            "## Fixed Inputs / Outputs",
            "",
            "- Input: `RGB crop`",
            "- Input: `coarse prior normal crop`",
            "- Input: `human mask`",
            "- Output: `refined normal` or `normal residual`",
            "",
            "## First ROI Scope",
            "",
            "- Start from `head / neck`.",
            "- Add `shoulder line` next.",
            "- Do not start from full-image refinement.",
            "- Do not start from full-body clothing wrinkles.",
            "- First-round objective: make head boundary and hairline cleaner, not solve all details at once.",
            "",
            "## Teacher Priority",
            "",
            "- `coarse prior normal` itself must not be used as the detail teacher.",
            "- Teacher priority 1: `60v` fused geometry derived surface normal.",
            "- Teacher priority 2: controllable high-quality external normal estimator.",
            "- Teacher priority 3: local mesh / surface fitting pseudo GT.",
            "- If teacher quality is still weak, restrict refinement to reliably visible regions first.",
            "",
            "## Required Losses",
            "",
            "- cosine normal loss",
            "- edge-aware loss",
            "- mask-restricted loss",
            "- extra ROI boundary weighting",
            "- dedicated statistics for `hairline / back-ear / hair boundary` regions",
            "- do not rely on whole-image average loss alone",
            "",
            "## Experiment Order",
            "",
            "- First do `60v`.",
            "- After `60v` is stable, move to `13v`.",
            "- After `13v` is stable, move to `7v`.",
            "- Before the detail branch is stable, do not reopen large sparse-view end-to-end training.",
            "- First do small ROI, small batch, and deliberate overfit checks.",
            "- First prove one frame can learn head detail correctly.",
            "- Then test cross-frame generalization.",
            "- Finally test multi-case generalization.",
            "",
            "## Visualization Protocol",
            "",
            "- Every run must export: `RGB`, `coarse prior normal`, `refined normal`, `coarse vs refined diff`.",
            "- `head ROI` must be stored separately.",
            "- `face ROI` must be stored separately.",
            "- `full body` is supplementary, not the only main figure.",
            "- Failures must be archived, not discarded.",
            "",
            "## Mentor Wording",
            "",
            "- Use: `The coarse prior normal chain is established; next we borrow a PIFuHD-style coarse-to-fine idea for high-resolution local detail refinement.`",
            "- Use: `The 4v probe has degraded into a silhouette-only result and has been downgraded to an internal debugging branch.`",
            "- Use: `60v already proves the coarse prior can align, stay stable, and be shown, but visible detail quality still needs to be pushed higher.`",
            "- HumanRAM should only support two talking points: pose-aligned human condition helps quality; transformer plus human-conditioned rendering is a valid direction.",
            "- PIFuHD should only support two talking points: high-resolution human detail needs coarse-to-fine; image-aligned local detail branches are a reasonable reinforcement path.",
        ]
    )
    write_text(spec_path, spec_text)
    return spec_path


def build_main_figure_captions(output_dir: Path) -> Path:
    captions_path = output_dir / "docs" / "main_figure_captions.md"
    captions_text = "\n".join(
        [
            "# Main Figure Captions",
            "",
            "- `60v overview`: current coarse prior normal is view-aligned and stable, but detail quality is still limited by the coarse prior normal expression ceiling.",
            "- `60v full-body RGB vs coarse prior normal`: global body orientation is correct, but thin structures and hair boundary remain coarse.",
            "- `60v head ROI RGB vs coarse prior normal`: head silhouette and pose are right, but ear contour and hairline are still blunt.",
            "- `60v face ROI RGB vs coarse prior normal`: face direction is plausible, but eyes, nose bridge, lips, and hairline still lack high-frequency detail.",
            "- `60v / 13v / 7v full-body coarse prior normal compare`: view count changes produce limited visible difference, so the bottleneck is prior expressiveness rather than view count alone.",
            "- `60v / 13v / 7v head coarse prior normal compare`: the same conclusion holds in the most important ROI; view count alone does not solve head detail.",
            "- `failed predicted normal probe`: this branch collapses toward silhouette-only output and is archived as an internal debug failure case.",
        ]
    )
    write_text(captions_path, captions_text)
    return captions_path


def build_checklist_completion_audit(output_dir: Path) -> Path:
    audit_path = output_dir / "docs" / "checklist_completion_audit.md"
    audit_text = "\n".join(
        [
            "# Checklist Completion Audit",
            "",
            "## Scope",
            "",
            "This audit checks the latest mentor-oriented checklist against this rebuilt pack.",
            "",
            "## A. Main Packaging And Wording",
            "",
            "- Completed: all failed `4v probe` assets are outside the main conclusion path and archived under `failed_predicted_normal_probe/`.",
            "- Completed: accepted coarse prior normal assets are named and grouped as `coarse_prior_normals/`.",
            "- Completed: the mentor main 4 are exactly the 60v full-body, head ROI, face ROI, and overview coarse prior normal figures.",
            "- Completed: `7v / 13v` figures are supplementary only.",
            "- Completed: wording states that current normal evidence is `SMPL-X view-aligned coarse prior normal`, with high-resolution detail refinement as the next step.",
            "",
            "## B. Required Checks",
            "",
            "- Completed: unified full-body comparison page: `overview_layouts/0012_11_compare_coarse_prior_fullbody_60_13_7.png`.",
            "- Completed: unified head ROI comparison page: `overview_layouts/0012_11_compare_coarse_prior_head_60_13_7.png`.",
            "- Completed: the conclusion is written explicitly: source view count increases produce limited visible difference, so the bottleneck is prior expressiveness rather than view count alone.",
            "- Completed: each main figure has one sentence describing what is correct and what remains coarse in `docs/mentor_minimal_pack.md` and `docs/main_figure_captions.md`.",
            "- Completed: `4v probe` failure assets are labeled as `failed predicted normal probe` and `silhouette-only collapse`.",
            "",
            "## C. New Branch Technical Positioning",
            "",
            "- Completed in `docs/detail_normal_refiner_spec.md`.",
            "- Fixed branch names: `detail_normal_refiner` and `pifuhd_style_normal_refine`.",
            "- Fixed role: image-aligned residual refinement on top of coarse prior normal.",
            "- Fixed inputs: `RGB crop`, `prior normal crop`, `human mask`.",
            "- Fixed outputs: `refined normal` or `normal residual`.",
            "- Fixed first ROI: `head / neck` and `shoulder line`.",
            "",
            "## D. Supervision Design",
            "",
            "- Completed in `docs/detail_normal_refiner_spec.md`.",
            "- The spec forbids using coarse prior normal itself as the detail teacher.",
            "- The spec lists teacher priority, visible-region fallback, cosine normal loss, edge-aware loss, mask-restricted loss, ROI boundary weighting, and hairline / ear-side metrics.",
            "",
            "## E. Experiment Order",
            "",
            "- Completed in `docs/detail_normal_refiner_spec.md`.",
            "- Order is fixed as `60v -> 13v -> 7v`.",
            "- Small ROI overfit comes before cross-frame and multi-case experiments.",
            "- Large sparse-view end-to-end training is deferred until the detail branch is stable.",
            "",
            "## F. Visualization Protocol",
            "",
            "- Completed as a protocol in `docs/detail_normal_refiner_spec.md`.",
            "- Current actual coarse prior normal assets are in `coarse_prior_normals/` and `overview_layouts/`.",
            "- Completed as fixed 4-slot templates for `60v / 13v / 7v / refined`:",
            "  `overview_layouts/0012_11_compare_coarse_prior_fullbody_60_13_7_refined_template.png`",
            "  `overview_layouts/0012_11_compare_coarse_prior_head_60_13_7_refined_template.png`",
            "",
            "Clarification:",
            "",
            "- There is still no real `refined normal` model output in this pack.",
            "- The refined slot is intentionally marked pending to avoid fabricating a result.",
            "",
            "## G. Mentor Messaging",
            "",
            "- Completed in `final_coarse_prior_normal_pass_pack/send_to_advisor.md`, `docs/detail_normal_refiner_spec.md`, and `docs/mentor_minimal_pack.md`.",
            "- Unified wording: coarse prior normal chain is established; next step is PIFuHD-style coarse-to-fine local refinement.",
            "- Unified wording: `4v probe` is a silhouette-only internal debug branch.",
            "- Unified wording: `60v` is stable and displayable but not yet detail-complete.",
            "",
            "## Final Status",
            "",
            "- Packaging status: completed.",
            "- Mentor-facing wording status: completed.",
            "- Failure-archive split status: completed.",
            "- `60v / 13v / 7v` comparison status: completed.",
            "- `refined` layout standard status: completed as a pending-template, not as a fabricated result.",
            "- Real `refined normal` output status: not yet available, and intentionally not claimed.",
        ]
    )
    write_text(audit_path, audit_text)
    return audit_path


def build_final_coarse_prior_normal_pass_pack_v2(
    output_dir: Path,
    outputs: dict[str, dict[str, Path]],
    inventory: list[dict[str, Any]],
) -> tuple[Path, Path, Path]:
    final_dir = output_dir / "final_coarse_prior_normal_pass_pack"
    figures_dir = final_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    selected = [
        (
            outputs["normals_baseline_60v"]["full_body_pair"],
            figures_dir / "01_fullbody_rgb_vs_coarse_prior_normal_60v.png",
            "60v full-body RGB vs coarse prior normal",
        ),
        (
            outputs["normals_baseline_60v"]["head_roi"],
            figures_dir / "02_head_roi_rgb_vs_coarse_prior_normal_60v.png",
            "60v head ROI RGB vs coarse prior normal",
        ),
        (
            outputs["normals_baseline_60v"]["face_roi"],
            figures_dir / "03_face_roi_rgb_vs_coarse_prior_normal_60v.png",
            "60v face ROI RGB vs coarse prior normal",
        ),
        (
            outputs["normals_baseline_60v"]["overview"],
            figures_dir / "04_coarse_prior_overview_60v.png",
            "60v coarse prior normal overview",
        ),
    ]

    for src, dst, _ in selected:
        shutil.copy2(src, dst)

    storyboard_path = figures_dir / "00_coarse_prior_normal_storyboard_60v.png"
    compose_image_grid_from_paths(
        image_paths=[item[1] for item in selected],
        titles=[item[2] for item in selected],
        output_path=storyboard_path,
        suptitle="0012_11 coarse prior normal accepted route storyboard (60v)",
        ncols=2,
        figsize_per_panel=(5.4, 4.8),
    )

    readme_path = final_dir / "README.md"
    send_text_path = final_dir / "send_to_advisor.md"

    write_text(
        readme_path,
        "\n".join(
            [
                "# Coarse Prior Normal Pass Pack",
                "",
                "- Assumption for this pack: the mentor has accepted the `coarse prior normal` route as the current stage conclusion.",
                "- Goal: keep the mentor-facing packet focused on `60v` coarse prior normal alignment; failed predicted-normal probe diagnostics stay outside this final packet.",
                "",
                "## Mentor Main 4",
                "",
                "- `figures/01_fullbody_rgb_vs_coarse_prior_normal_60v.png`",
                "- `figures/02_head_roi_rgb_vs_coarse_prior_normal_60v.png`",
                "- `figures/03_face_roi_rgb_vs_coarse_prior_normal_60v.png`",
                "- `figures/04_coarse_prior_overview_60v.png`",
                "",
                "## Notes",
                "",
                "- Current normal evidence is `SMPL-X view-aligned coarse prior normal`; the next step is high-resolution local detail refinement.",
                "- `7v / 13v` figures are supplementary only and should not replace the `60v` main four.",
                "- Failed predicted-normal probe diagnostics are archived separately and are not part of this final packet.",
            ]
        ),
    )

    write_text(
        send_text_path,
        "\n".join(
            [
                "这版汇报口径统一为 `coarse prior normal` 阶段结论。",
                "",
                "1. 当前 normal 以 `SMPL-X view-aligned coarse prior normal` 为主，下一步做局部高分辨 detail refinement。",
                "2. 60v 已经证明 coarse prior normal 可对齐、可稳定、可展示，但可见细节质量还需要继续拉高。",
                "3. 给导师主发 4 张：60v full-body、60v head ROI、60v face ROI、60v overview；7v/13v 只作为稀疏视角下 coarse prior normal 仍稳定的补充页。",
                "4. 失败的 predicted-normal probe 已单独归档为内部排查材料，不再作为主结论素材。",
                "",
                "后续会借鉴 PIFuHD 的 coarse-to-fine 思路，做 image-aligned 的局部 detail refinement；它不是替代 VGGT，也不是替代 coarse prior normal，而是对 coarse prior normal 做 residual refinement。",
            ]
        ),
    )

    inventory.extend(
        [
            {
                "category": "final_coarse_prior_normal_pass_pack",
                "path": str(storyboard_path.relative_to(output_dir)),
                "description": "One-page storyboard for the mentor-approved coarse prior normal route",
            },
            {
                "category": "final_coarse_prior_normal_pass_pack",
                "path": str(readme_path.relative_to(output_dir)),
                "description": "README for the coarse prior normal pass pack",
            },
            {
                "category": "final_coarse_prior_normal_pass_pack",
                "path": str(send_text_path.relative_to(output_dir)),
                "description": "Suggested Chinese message for the mentor",
            },
        ]
    )
    return final_dir, readme_path, send_text_path


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        CaseSpec(
            slug="baseline_7v",
            case_id="0012_11",
            view_tag="7v",
            scene_dir=REPO_ROOT / "output" / "4k4d_scenes" / "0012_11_frame0000_7views",
            predictions_npz=REPO_ROOT / "output" / "modal_results" / "0012_11_frame0000_7views" / "predictions.npz",
            prior_maps_npz=REPO_ROOT / "output" / "4k4d_scenes" / "0012_11_frame0000_7views" / "prior_maps.npz",
            variant="baseline",
            source_label="vggt_baseline",
        ),
        CaseSpec(
            slug="baseline_13v",
            case_id="0012_11",
            view_tag="13v",
            scene_dir=REPO_ROOT / "output" / "4k4d_scenes" / "0012_11_frame0000_13views",
            predictions_npz=REPO_ROOT / "output" / "modal_results" / "0012_11_frame0000_13views" / "predictions.npz",
            prior_maps_npz=REPO_ROOT / "output" / "4k4d_scenes" / "0012_11_frame0000_13views" / "prior_maps.npz",
            variant="baseline",
            source_label="vggt_baseline",
        ),
        CaseSpec(
            slug="baseline_60v",
            case_id="0012_11",
            view_tag="60v",
            scene_dir=REPO_ROOT / "output" / "4k4d_scenes" / "0012_11_frame0000_60views",
            predictions_npz=REPO_ROOT / "output" / "modal_results" / "0012_11_frame0000_60views" / "predictions.npz",
            prior_maps_npz=REPO_ROOT / "output" / "4k4d_scenes" / "0012_11_frame0000_60views" / "prior_maps.npz",
            variant="baseline",
            source_label="vggt_baseline",
        ),
        CaseSpec(
            slug="surfacepose_60v",
            case_id="0012_11",
            view_tag="60v",
            scene_dir=REPO_ROOT / "output" / "4k4d_scenes" / "0012_11_frame0000_60views",
            predictions_npz=REPO_ROOT
            / "output"
            / "modal_results"
            / "0012_11_frame0000_60views_smplxsurfacepose_a10080_e2_r2"
            / "predictions.npz",
            prior_maps_npz=REPO_ROOT / "output" / "4k4d_scenes" / "0012_11_frame0000_60views" / "prior_maps.npz",
            variant="surfacepose",
            source_label="surfacepose_run",
        ),
    ]
    probe = ProbeSpec(
        case_id="0012_11",
        view_tag="4v_probe",
        inputs_npz=Path(r"D:\vggt\runs\normal_head\20260420_normal_probe\frozen_probe_surfacepose4v\probe_inputs.npz"),
        predictions_npz=Path(r"D:\vggt\runs\normal_head\20260420_normal_probe\frozen_probe_surfacepose4v\probe_predictions.npz"),
        summary_json=Path(r"D:\vggt\runs\normal_head\20260420_normal_probe\frozen_probe_surfacepose4v\probe_summary.json"),
    )

    inventory: list[dict[str, Any]] = []
    outputs: dict[str, dict[str, Path]] = {}

    reference_boxes = build_reference_boxes(next(case for case in cases if case.slug == "baseline_60v"))

    for case in cases:
        if case.variant == "baseline":
            outputs[f"normals_{case.slug}"] = save_prior_normal_outputs(case, output_dir, inventory)

    outputs["normals_probe"] = save_probe_outputs(probe, output_dir, inventory)

    for idx, case in enumerate(cases):
        outputs[f"pointcloud_{case.slug}"] = save_pointcloud_outputs(
            case,
            output_dir,
            inventory,
            reference_boxes,
            conf_percentile=args.conf_percentile,
            max_full_points=args.max_full_points,
            max_roi_points=args.max_roi_points,
            rng_seed=args.seed + idx,
        )

    comparisons_dir = output_dir / "comparisons"
    overview_dir = output_dir / "overview_layouts"

    baseline_60 = outputs["pointcloud_baseline_60v"]
    surfacepose_60 = outputs["pointcloud_surfacepose_60v"]
    prior_60 = outputs["normals_baseline_60v"]
    prior_13 = outputs["normals_baseline_13v"]
    prior_7 = outputs["normals_baseline_7v"]

    compare_head_path = comparisons_dir / "0012_11_compare_baseline_vs_surfacepose_head_60v.png"
    compare_face_path = comparisons_dir / "0012_11_compare_baseline_vs_surfacepose_face_60v.png"
    demo_head_path = comparisons_dir / "0012_11_normal_to_point_head_demo_60v.png"
    demo_face_path = comparisons_dir / "0012_11_normal_to_point_face_demo_60v.png"
    coarse_fullbody_path = overview_dir / "0012_11_compare_coarse_prior_fullbody_60_13_7.png"
    coarse_head_path = overview_dir / "0012_11_compare_coarse_prior_head_60_13_7.png"

    compose_image_strip(
        [baseline_60["head"], surfacepose_60["head"], prior_60["head_roi"]],
        ["60v baseline head", "60v +surfacepose head", "Target RGB vs prior normal"],
        compare_head_path,
        suptitle="0012_11 60v baseline vs surfacepose head comparison",
        figsize_per_panel=(4.8, 4.8),
    )
    compose_image_strip(
        [baseline_60["face"], surfacepose_60["face"], prior_60["face_roi"]],
        ["60v baseline face", "60v +surfacepose face", "Target RGB vs prior normal"],
        compare_face_path,
        suptitle="0012_11 60v baseline vs surfacepose face comparison",
        figsize_per_panel=(4.8, 4.8),
    )
    compose_image_strip(
        [prior_60["head_roi"], baseline_60["head"], surfacepose_60["head"]],
        ["Target RGB vs prior normal", "60v baseline head cloud", "60v +surfacepose head cloud"],
        demo_head_path,
        suptitle="0012_11 head normal-to-point context demo",
        figsize_per_panel=(4.8, 4.8),
    )
    compose_image_strip(
        [prior_60["face_roi"], baseline_60["face"], surfacepose_60["face"]],
        ["Target RGB vs prior normal", "60v baseline face cloud", "60v +surfacepose face cloud"],
        demo_face_path,
        suptitle="0012_11 face normal-to-point context demo",
        figsize_per_panel=(4.8, 4.8),
    )
    compose_image_strip(
        [prior_60["full_body_normal"], prior_13["full_body_normal"], prior_7["full_body_normal"]],
        ["60v coarse prior", "13v coarse prior", "7v coarse prior"],
        coarse_fullbody_path,
        suptitle="0012_11 coarse prior full-body comparison (60v / 13v / 7v)",
        figsize_per_panel=(4.8, 4.8),
    )
    compose_image_strip(
        [prior_60["head_normal"], prior_13["head_normal"], prior_7["head_normal"]],
        ["60v coarse prior head", "13v coarse prior head", "7v coarse prior head"],
        coarse_head_path,
        suptitle="0012_11 coarse prior head ROI comparison (60v / 13v / 7v)",
        figsize_per_panel=(4.8, 4.8),
    )
    refined_full_template, refined_head_template = build_refined_overview_templates(
        output_dir,
        prior_60,
        prior_13,
        prior_7,
        inventory,
    )

    inventory.extend(
        [
            {
                "category": "comparisons",
                "path": str(compare_head_path.relative_to(output_dir)),
                "description": "60v baseline vs surfacepose head comparison",
            },
            {
                "category": "comparisons",
                "path": str(compare_face_path.relative_to(output_dir)),
                "description": "60v baseline vs surfacepose face comparison",
            },
            {
                "category": "comparisons",
                "path": str(demo_head_path.relative_to(output_dir)),
                "description": "Head normal-to-point context demo",
            },
            {
                "category": "comparisons",
                "path": str(demo_face_path.relative_to(output_dir)),
                "description": "Face normal-to-point context demo",
            },
            {
                "category": "overview_layouts",
                "path": str(coarse_fullbody_path.relative_to(output_dir)),
                "description": "60v/13v/7v full-body coarse prior comparison",
            },
            {
                "category": "overview_layouts",
                "path": str(coarse_head_path.relative_to(output_dir)),
                "description": "60v/13v/7v head ROI coarse prior comparison",
            },
        ]
    )

    readme_path, assessment_path, mentor_pack_path = build_docs_v2(output_dir, inventory)
    captions_path = build_main_figure_captions(output_dir)
    refiner_spec_path = build_detail_normal_refiner_spec(output_dir)
    checklist_audit_path = build_checklist_completion_audit(output_dir)
    final_pack_dir, final_pack_readme, final_pack_send_text = build_final_coarse_prior_normal_pass_pack_v2(
        output_dir, outputs, inventory
    )
    manifest_path = output_dir / "docs" / "image_inventory.json"
    save_json(manifest_path, inventory)

    summary = {
        "output_dir": str(output_dir),
        "readme": str(readme_path),
        "assessment": str(assessment_path),
        "mentor_minimal_pack_doc": str(mentor_pack_path),
        "main_figure_captions": str(captions_path),
        "detail_normal_refiner_spec": str(refiner_spec_path),
        "checklist_completion_audit": str(checklist_audit_path),
        "final_coarse_prior_normal_pass_pack_dir": str(final_pack_dir),
        "final_coarse_prior_normal_pass_pack_readme": str(final_pack_readme),
        "final_coarse_prior_normal_pass_pack_send_text": str(final_pack_send_text),
        "inventory_json": str(manifest_path),
        "refined_overview_templates": [str(refined_full_template), str(refined_head_template)],
        "generated_files": [item["path"] for item in inventory if not str(item["path"]).startswith(str(REPO_ROOT))],
        "mentor_minimal_pack": [
            str(outputs["normals_baseline_60v"]["full_body_pair"].relative_to(output_dir)),
            str(outputs["normals_baseline_60v"]["head_roi"].relative_to(output_dir)),
            str(outputs["normals_baseline_60v"]["face_roi"].relative_to(output_dir)),
            str(outputs["normals_baseline_60v"]["overview"].relative_to(output_dir)),
        ],
        "notes": [
            "This summary assumes the mentor has accepted the coarse prior normal route as the current report direction.",
            "Sparse-view coarse prior normal comparison uses 60v/13v/7v because those are the actual available cases in the workspace.",
            "Only 60v has both baseline and surfacepose geometry outputs in the current workspace.",
            "4v probe outputs are archived as silhouette-only collapse evidence, not as a successful predicted-normal result.",
            "Current normal evidence is SMPL-X view-aligned coarse prior normal; the next step is high-resolution local detail refinement.",
        ],
    }
    save_json(output_dir / "docs" / "pack_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
