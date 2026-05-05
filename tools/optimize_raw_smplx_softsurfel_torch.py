from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from normal_line_multiview_eval import load_scene_view  # noqa: E402
from prepare_4k4d_prior_training_case import (  # noqa: E402
    align_intrinsics_for_scene_view,
    load_optional_annotation_payload,
    load_scene_manifest,
    recover_legacy_crop_source_sizes,
    resolve_scene_camera_params,
    resolve_smplx_model_dir,
)
from tools.smplx_numpy import (  # noqa: E402
    build_smplx_vertex_features,
    compute_vertex_normals,
    forward_smplx_mesh,
    rasterize_world_mesh,
    resolve_smplx_model_path,
)


PART_NAMES = {
    0: "torso_limbs",
    1: "left_hand",
    2: "right_hand",
    3: "head_face",
    4: "head_top_hairline_proxy",
    5: "lower_clothing_proxy",
}

PART_ALIASES = {
    "body": 0,
    "torso": 0,
    "limbs": 0,
    "left_hand": 1,
    "lh": 1,
    "right_hand": 2,
    "rh": 2,
    "hands": (1, 2),
    "hand": (1, 2),
    "face": 3,
    "head": 3,
    "head_face": 3,
    "hair": 4,
    "hairline": 4,
    "head_top": 4,
    "clothing": 5,
    "skirt": 5,
    "lower": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Raw-image soft-surface upper-bound v1 smoke. This script uses raw RGB/masks/"
            "calibrated cameras/SMPL-X only. It intentionally does not use VGGT depth, "
            "point maps, normals, confidence, or r-candidate outputs."
        )
    )
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-name", default="data_used_in_4K4D")
    parser.add_argument("--smplx-model-dir", type=Path)
    parser.add_argument("--smplx-gender", choices=("neutral", "female", "male"), default="neutral")
    parser.add_argument("--target-size", type=int, default=128)
    parser.add_argument("--max-views", type=int, default=6)
    parser.add_argument("--view-stride", type=int, default=1)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--surfel-samples", type=int, default=1200)
    parser.add_argument(
        "--balanced-part-surfels",
        action="store_true",
        help=(
            "Sample a minimum number of surfels from key body parts before area-filling the rest. "
            "This is a representation diagnostic for tiny head/hand/hairline support, not a pass gate."
        ),
    )
    parser.add_argument("--min-surfel-hand", type=int, default=0)
    parser.add_argument("--min-surfel-head", type=int, default=0)
    parser.add_argument("--min-surfel-hairline", type=int, default=0)
    parser.add_argument("--min-surfel-torso", type=int, default=0)
    parser.add_argument("--min-surfel-clothing", type=int, default=0)
    parser.add_argument("--surface-samples-for-sdf", type=int, default=2500)
    parser.add_argument("--boundary-samples", type=int, default=192)
    parser.add_argument("--render-pixel-chunk", type=int, default=4096)
    parser.add_argument("--gaussian-sigma", type=float, default=1.7)
    parser.add_argument(
        "--renderer",
        choices=("surfel", "triangle"),
        default="surfel",
        help=(
            "Differentiable mask/depth renderer for the optimization loss. "
            "triangle is a CPU smoke for connected mesh visibility; surfel remains the default."
        ),
    )
    parser.add_argument(
        "--triangle-inside-softness",
        type=float,
        default=70.0,
        help="Sigmoid sharpness for the soft barycentric inside test when --renderer triangle.",
    )
    parser.add_argument(
        "--triangle-render-face-budget",
        type=int,
        default=0,
        help=(
            "Face budget for --renderer triangle. 0 reuses sampled surfel faces; negative renders all "
            "mesh faces; positive uses a deterministic subset of that many mesh faces."
        ),
    )
    parser.add_argument(
        "--triangle-face-chunk",
        type=int,
        default=256,
        help="Number of faces processed per inner loop for --renderer triangle.",
    )
    parser.add_argument(
        "--depth-softness",
        type=float,
        default=0.0,
        help=(
            "Optional soft z ordering for surfel rendering. When >0, depth/normal maps use "
            "spatial Gaussian weights multiplied by exp(-z/depth_softness), stabilized per pixel. "
            "The alpha mask still uses spatial support."
        ),
    )
    parser.add_argument(
        "--connected-template-payload",
        type=Path,
        help=(
            "Optional v2 raw-surface carrier payload from build_connected_human_surface_template.py. "
            "When set, the optimizer uses the connected hybrid mesh instead of the plain SMPL-X mesh."
        ),
    )
    parser.add_argument("--mask-weight", type=float, default=1.0)
    parser.add_argument(
        "--recall-weight",
        type=float,
        default=0.45,
        help="Soft target-coverage guard. Prevents the optimizer from improving IoU by shrinking the rendered body.",
    )
    parser.add_argument("--outside-weight", type=float, default=0.20)
    parser.add_argument("--boundary-weight", type=float, default=0.05)
    parser.add_argument(
        "--part-recall-weight",
        type=float,
        default=0.0,
        help=(
            "Optional part-aware coverage guard for the connected surface smoke. "
            "It uses coarse raw-mask regions for upper head/hairline/hands so IoU "
            "cannot improve only by shrinking the full-body shell. Disabled by default."
        ),
    )
    parser.add_argument("--head-target-frac", type=float, default=0.35)
    parser.add_argument("--hairline-target-frac", type=float, default=0.18)
    parser.add_argument("--hand-side-target-frac", type=float, default=0.18)
    parser.add_argument("--part-target-min-pixels", type=int, default=16)
    parser.add_argument("--photo-weight", type=float, default=0.08)
    parser.add_argument(
        "--photo-depth-tolerance",
        type=float,
        default=0.0,
        help=(
            "Optional visibility filter for the photometric loss. When >0, a surfel must be "
            "near the current rendered front depth in that view before its sampled RGB counts."
        ),
    )
    parser.add_argument(
        "--image-edge-weight",
        type=float,
        default=0.0,
        help=(
            "Optional raw-RGB edge distance loss for connected face/hair/hand/clothing vertices. "
            "This uses Canny edges from the input images, not VGGT depth/point/normal, and is "
            "disabled by default."
        ),
    )
    parser.add_argument(
        "--image-edge-part-ids",
        default="1,2,3,4,5",
        help=(
            "Comma-separated part ids or aliases for --image-edge-weight. "
            "Examples: 'face,hair,hands,clothing' or '1,2,3,4,5'."
        ),
    )
    parser.add_argument("--image-edge-canny-low", type=float, default=40.0)
    parser.add_argument("--image-edge-canny-high", type=float, default=120.0)
    parser.add_argument("--image-edge-mask-dilate", type=int, default=3)
    parser.add_argument(
        "--image-edge-max-distance",
        type=float,
        default=0.08,
        help="Clamp normalized edge distance in the optional image-edge loss; <=0 disables clamp.",
    )
    parser.add_argument(
        "--freeze-global-transform",
        action="store_true",
        help=(
            "Keep the SMPL-X/global carrier scale and translation fixed. This is a local "
            "upper-bound diagnostic to prevent the optimizer from improving IoU by shrinking "
            "or sliding the entire template shell."
        ),
    )
    parser.add_argument("--translation-reg", type=float, default=0.05)
    parser.add_argument("--scale-reg", type=float, default=0.05)
    parser.add_argument("--offset-reg", type=float, default=0.35)
    parser.add_argument("--offset-smooth-reg", type=float, default=0.08)
    parser.add_argument("--normal-offset-limit-body", type=float, default=0.015)
    parser.add_argument("--normal-offset-limit-hands", type=float, default=0.030)
    parser.add_argument("--normal-offset-limit-head", type=float, default=0.022)
    parser.add_argument("--normal-offset-limit-hairline", type=float, default=0.040)
    parser.add_argument("--normal-offset-limit-clothing", type=float, default=0.035)
    parser.add_argument(
        "--hairline-free-offset-limit",
        type=float,
        default=0.0,
        help=(
            "Optional connected mesh diagnostic: allow only head-top/hairline vertices to move in 3D, "
            "bounded by this world-space limit. Disabled by default."
        ),
    )
    parser.add_argument("--hairline-free-offset-reg", type=float, default=0.50)
    parser.add_argument("--hairline-free-smooth-reg", type=float, default=0.10)
    parser.add_argument(
        "--part-free-offset-limit-face",
        type=float,
        default=0.0,
        help="Optional bounded 3D residual for connected face/head vertices. Disabled by default.",
    )
    parser.add_argument(
        "--part-free-offset-limit-hands",
        type=float,
        default=0.0,
        help="Optional bounded 3D residual for connected hand vertices. Disabled by default.",
    )
    parser.add_argument(
        "--part-free-offset-limit-hairline",
        type=float,
        default=0.0,
        help="Optional bounded 3D residual for connected hair/head-top vertices. Disabled by default.",
    )
    parser.add_argument(
        "--part-free-offset-limit-clothing",
        type=float,
        default=0.0,
        help="Optional bounded 3D residual for connected clothing/lower-body vertices. Disabled by default.",
    )
    parser.add_argument("--part-free-offset-reg", type=float, default=0.35)
    parser.add_argument("--part-free-smooth-reg", type=float, default=0.08)
    parser.add_argument(
        "--hair-boundary-weight",
        type=float,
        default=0.0,
        help=(
            "Optional connected-cap diagnostic. Pulls the hair/head cap outer ring toward "
            "raw-mask silhouette boundaries using image SDF; disabled by default."
        ),
    )
    parser.add_argument(
        "--face-landmarker-task",
        type=Path,
        help=(
            "Optional MediaPipe FaceLandmarker task. When combined with --face-landmark-weight, "
            "2D landmarks weakly constrain only the connected SMPL-X face vertices. This is a "
            "raw-image diagnostic, not a floating face-patch teacher."
        ),
    )
    parser.add_argument(
        "--face-landmark-weight",
        type=float,
        default=0.0,
        help="Weight for the connected-face projected landmark Chamfer loss. Disabled by default.",
    )
    parser.add_argument(
        "--face-landmark-bidir-weight",
        type=float,
        default=0.10,
        help="Small projected-face-to-landmark term to avoid a one-way nearest-vertex shortcut.",
    )
    parser.add_argument(
        "--face-landmark-pad",
        type=int,
        default=-1,
        help="Head crop pad in target-size pixels for landmark detection; negative uses 8 percent of target size.",
    )
    parser.add_argument("--face-landmark-min-points", type=int, default=80)
    parser.add_argument("--face-landmark-min-confidence", type=float, default=0.02)
    parser.add_argument(
        "--hand-landmarker-task",
        type=Path,
        help=(
            "Optional MediaPipe HandLandmarker task. When combined with --hand-landmark-weight, "
            "2D hand landmarks weakly constrain only connected SMPL-X hand vertices."
        ),
    )
    parser.add_argument(
        "--hand-landmark-weight",
        type=float,
        default=0.0,
        help="Weight for connected-hand projected landmark Chamfer loss. Disabled by default.",
    )
    parser.add_argument("--hand-landmark-bidir-weight", type=float, default=0.10)
    parser.add_argument("--hand-landmark-min-points", type=int, default=12)
    parser.add_argument("--hand-landmark-min-confidence", type=float, default=0.02)
    parser.add_argument(
        "--extra-hairline-surfels",
        type=int,
        default=0,
        help=(
            "Optional diagnostic: build a shared 3D head-top/hairline support set from raw mask pixels "
            "that are missing from the optimized SMPL-X raster. Disabled by default."
        ),
    )
    parser.add_argument("--extra-hairline-max-per-view", type=int, default=256)
    parser.add_argument("--extra-hairline-min-support", type=int, default=2)
    parser.add_argument("--extra-hairline-top-frac", type=float, default=0.40)
    parser.add_argument("--extra-hairline-nearest-radius", type=float, default=10.0)
    parser.add_argument("--extra-hairline-depth-offset", type=float, default=0.0)
    parser.add_argument("--extra-hairline-voxel", type=float, default=0.004)
    parser.add_argument("--extra-hairline-export-radius", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overlay-limit", type=int, default=8)
    parser.add_argument("--export-raster-targets", action="store_true")
    parser.add_argument(
        "--export-scene-dir",
        type=Path,
        help="Optional scene whose protocol should receive hard-rasterized depth/world/normal targets.",
    )
    parser.add_argument("--export-target-size", type=int, default=518)
    parser.add_argument("--export-target-views", default="all", help="Comma-separated view indices or 'all'.")
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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
    raise ValueError(f"Expected 3x4 or 4x4 matrix, got {matrix.shape}")


def align_intrinsics_for_loaded_scene_view(intrinsic: np.ndarray, view: dict[str, Any], target_size: int) -> np.ndarray:
    """Align intrinsics to the already-exported crop PNG then optional smoke resize.

    `crop_pad_to_square` manifests store crop bboxes in the native exported image
    frame (normally 518x518). The shared helper is correct for the native training
    size, but if we pass a smaller CPU-smoke size directly it applies those native
    bbox coordinates to the smaller frame. For this raw-image smoke we first align
    at the manifest image size and then scale the exported square view down.
    """
    image_size = view.get("image_size") or [target_size, target_size]
    native_size = int(image_size[0]) if len(image_size) >= 1 else int(target_size)
    meta = view.get("preprocess_meta") or {}
    if meta.get("transform") == "crop_pad_to_square" and native_size != int(target_size):
        native = align_intrinsics_for_scene_view(intrinsic, view, target_size=native_size)
        scale = float(target_size) / float(max(1, native_size))
        out = native.astype(np.float32).copy()
        out[0, :] *= scale
        out[1, :] *= scale
        return out
    return align_intrinsics_for_scene_view(intrinsic, view, target_size=target_size)


def mask_sdf(mask: np.ndarray) -> np.ndarray:
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    inside = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
    outside = cv2.distanceTransform((1 - mask_u8).astype(np.uint8), cv2.DIST_L2, 3)
    sdf = outside - inside
    return (sdf / float(max(mask.shape))).astype(np.float32)


def parse_part_id_spec(spec: str) -> list[int]:
    out: list[int] = []
    for raw_item in str(spec).replace(";", ",").split(","):
        item = raw_item.strip().lower()
        if not item:
            continue
        if item.isdigit() or (item.startswith("-") and item[1:].isdigit()):
            value = int(item)
            if value not in PART_NAMES:
                raise ValueError(f"Unknown part id {value!r} in --image-edge-part-ids")
            out.append(value)
            continue
        if item not in PART_ALIASES:
            raise ValueError(f"Unknown part alias {item!r} in --image-edge-part-ids")
        value = PART_ALIASES[item]
        if isinstance(value, tuple):
            out.extend(int(v) for v in value)
        else:
            out.append(int(value))
    return sorted(set(out))


def image_edge_distance(rgb: np.ndarray, mask: np.ndarray, low: float, high: float, mask_dilate: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Distance to raw RGB Canny edges, normalized by image size.

    This deliberately uses only raw image/mask evidence. It is a local surface
    objective, not a teacher surface and not a VGGT shell recycling path.
    """

    rgb_u8 = np.clip(normalize_rgb(rgb) * 255.0, 0.0, 255.0).astype(np.uint8)
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, float(low), float(high)).astype(np.uint8)
    mask_u8 = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if int(mask_dilate) > 0:
        k = max(1, int(mask_dilate))
        kernel = np.ones((k, k), dtype=np.uint8)
        mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
    edges = (edges > 0).astype(np.uint8) * mask_u8
    edge_pixels = int(edges.sum())
    if edge_pixels == 0:
        dist = np.ones(mask_u8.shape, dtype=np.float32)
    else:
        dist = cv2.distanceTransform((1 - edges).astype(np.uint8), cv2.DIST_L2, 3)
        dist = (dist / float(max(mask_u8.shape))).astype(np.float32)
    return dist.astype(np.float32), {
        "edge_pixels": edge_pixels,
        "low": float(low),
        "high": float(high),
        "mask_dilate": int(mask_dilate),
    }


def boundary_points(mask: np.ndarray, max_points: int) -> np.ndarray:
    mask_u8 = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    grad = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, kernel)
    ys, xs = np.nonzero(grad)
    if xs.size == 0:
        ys, xs = np.nonzero(mask_u8)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    count = min(int(max_points), xs.size)
    indices = np.linspace(0, xs.size - 1, count).round().astype(np.int64)
    return np.stack([xs[indices], ys[indices]], axis=1).astype(np.float32)


def resolve_scene_path(scene_dir: Path, raw: str | Path) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path
    candidate = scene_dir / path
    if candidate.exists():
        return candidate
    return path


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def clamp_image_box(
    box: tuple[int, int, int, int],
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    x0 = max(0, min(int(width), int(x0)))
    y0 = max(0, min(int(height), int(y0)))
    x1 = max(x0 + 1, min(int(width), int(x1)))
    y1 = max(y0 + 1, min(int(height), int(y1)))
    return x0, y0, x1, y1


def head_box_from_mask_simple(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    bbox = mask_bbox(mask)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    body_h = max(1, y1 - y0)
    body_w = max(1, x1 - x0)
    head_h = max(18, int(round(body_h * 0.45)))
    raw = (
        x0 - max(3, int(round(body_w * 0.04))),
        y0 - max(3, int(round(body_h * 0.02))),
        x1 + max(3, int(round(body_w * 0.04))),
        min(y1, y0 + head_h) + max(3, int(round(body_h * 0.02))),
    )
    return clamp_image_box(raw, mask.shape[0], mask.shape[1])


def load_image_mask_for_detection(image_path: Path, mask_path: Path, target_size: int) -> tuple[Image.Image, np.ndarray]:
    image = Image.open(image_path).convert("RGB")
    if image.size != (target_size, target_size):
        image = image.resize((target_size, target_size), Image.Resampling.BICUBIC)
    mask_image = Image.open(mask_path).convert("L")
    if mask_image.size != (target_size, target_size):
        mask_image = mask_image.resize((target_size, target_size), Image.Resampling.NEAREST)
    mask = (np.asarray(mask_image, dtype=np.uint8) > 127)
    return image, mask


def create_face_landmarker(task_path: Path, min_confidence: float) -> tuple[Any | None, Any | None, dict[str, Any]]:
    task_path = task_path.expanduser().resolve()
    meta: dict[str, Any] = {
        "requested": True,
        "task_path": str(task_path),
        "available": False,
        "reason": None,
    }
    if not task_path.is_file():
        meta["reason"] = "task_file_missing"
        return None, None, meta
    try:
        import mediapipe as mp  # type: ignore
        from mediapipe.tasks import python as mp_python  # type: ignore
        from mediapipe.tasks.python import vision as mp_vision  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local optional package
        meta["reason"] = f"mediapipe_import_failed: {type(exc).__name__}: {exc}"
        return None, None, meta
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(task_path)),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=float(min_confidence),
        min_face_presence_confidence=float(min_confidence),
        min_tracking_confidence=float(min_confidence),
    )
    detector = mp_vision.FaceLandmarker.create_from_options(options)
    meta.update({"available": True, "reason": None})
    return mp, detector, meta


def create_hand_landmarker(task_path: Path, min_confidence: float) -> tuple[Any | None, Any | None, dict[str, Any]]:
    task_path = task_path.expanduser().resolve()
    meta: dict[str, Any] = {
        "requested": True,
        "task_path": str(task_path),
        "available": False,
        "reason": None,
    }
    if not task_path.is_file():
        meta["reason"] = "task_file_missing"
        return None, None, meta
    try:
        import mediapipe as mp  # type: ignore
        from mediapipe.tasks import python as mp_python  # type: ignore
        from mediapipe.tasks.python import vision as mp_vision  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local optional package
        meta["reason"] = f"mediapipe_import_failed: {type(exc).__name__}: {exc}"
        return None, None, meta
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(task_path)),
        num_hands=2,
        min_hand_detection_confidence=float(min_confidence),
        min_hand_presence_confidence=float(min_confidence),
        min_tracking_confidence=float(min_confidence),
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)
    meta.update({"available": True, "reason": None})
    return mp, detector, meta


def detect_face_landmarks_2d(
    *,
    mp_module: Any,
    detector: Any,
    image_path: Path,
    mask_path: Path,
    target_size: int,
    pad: int,
) -> tuple[np.ndarray | None, dict[str, Any], Image.Image]:
    image, mask = load_image_mask_for_detection(image_path, mask_path, target_size)
    head_box = head_box_from_mask_simple(mask)
    meta: dict[str, Any] = {"detected": False, "head_box": None, "landmarks": 0, "inside_mask_ratio": 0.0}
    if head_box is None:
        meta["reason"] = "no_head_box"
        return None, meta, image
    x0, y0, x1, y1 = head_box
    x0, y0, x1, y1 = clamp_image_box((x0 - pad, y0 - pad, x1 + pad, y1 + pad), target_size, target_size)
    if x1 <= x0 + 12 or y1 <= y0 + 12:
        meta.update({"reason": "tiny_head_box", "head_box": [x0, y0, x1, y1]})
        return None, meta, image
    crop = image.crop((x0, y0, x1, y1)).resize((512, 512), Image.Resampling.BICUBIC)
    result = detector.detect(mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=np.asarray(crop)))
    if not result.face_landmarks:
        meta.update({"reason": "no_facemesh", "head_box": [x0, y0, x1, y1]})
        return None, meta, image
    coords = []
    for lm in result.face_landmarks[0]:
        coords.append([x0 + float(lm.x) * (x1 - x0), y0 + float(lm.y) * (y1 - y0), float(lm.z)])
    coords_np = np.asarray(coords, dtype=np.float32)
    xi = np.clip(np.rint(coords_np[:, 0]).astype(np.int32), 0, target_size - 1)
    yi = np.clip(np.rint(coords_np[:, 1]).astype(np.int32), 0, target_size - 1)
    inside = mask[yi, xi]
    meta.update(
        {
            "detected": True,
            "head_box": [x0, y0, x1, y1],
            "landmarks": int(coords_np.shape[0]),
            "inside_mask": int(inside.sum()),
            "inside_mask_ratio": float(inside.mean()) if inside.size else 0.0,
        }
    )
    return coords_np, meta, image


def detect_hand_landmarks_2d(
    *,
    mp_module: Any,
    detector: Any,
    image_path: Path,
    mask_path: Path,
    target_size: int,
) -> tuple[list[np.ndarray], dict[str, Any], Image.Image]:
    image, mask = load_image_mask_for_detection(image_path, mask_path, target_size)
    result = detector.detect(mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=np.asarray(image)))
    landmark_sets: list[np.ndarray] = []
    inside_ratios: list[float] = []
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            coords = np.asarray(
                [[float(lm.x) * target_size, float(lm.y) * target_size, float(lm.z)] for lm in hand],
                dtype=np.float32,
            )
            xi = np.clip(np.rint(coords[:, 0]).astype(np.int32), 0, target_size - 1)
            yi = np.clip(np.rint(coords[:, 1]).astype(np.int32), 0, target_size - 1)
            inside = mask[yi, xi]
            landmark_sets.append(coords)
            inside_ratios.append(float(inside.mean()) if inside.size else 0.0)
    meta: dict[str, Any] = {
        "detected": bool(landmark_sets),
        "hands": int(len(landmark_sets)),
        "landmarks": int(sum(item.shape[0] for item in landmark_sets)),
        "inside_mask_ratio": float(np.mean(inside_ratios)) if inside_ratios else 0.0,
    }
    if not landmark_sets:
        meta["reason"] = "no_hands"
    return landmark_sets, meta, image


def save_face_landmark_overlay(image: Image.Image, landmarks: np.ndarray, output_path: Path) -> None:
    draw = image.copy()
    drawer = ImageDraw.Draw(draw)
    for x, y, _ in np.asarray(landmarks, dtype=np.float32):
        drawer.ellipse((float(x) - 1.5, float(y) - 1.5, float(x) + 1.5, float(y) + 1.5), fill=(255, 32, 32))
    draw.save(output_path)


def save_hand_landmark_overlay(image: Image.Image, landmark_sets: list[np.ndarray], output_path: Path) -> None:
    draw = image.copy()
    drawer = ImageDraw.Draw(draw)
    colors = [(32, 96, 255), (255, 160, 32)]
    for hand_idx, landmarks in enumerate(landmark_sets):
        color = colors[hand_idx % len(colors)]
        for x, y, _ in np.asarray(landmarks, dtype=np.float32):
            drawer.ellipse((float(x) - 1.6, float(y) - 1.6, float(x) + 1.6, float(y) + 1.6), fill=color)
    draw.save(output_path)


def coarse_part_target_masks(
    mask: np.ndarray,
    *,
    head_frac: float,
    hairline_frac: float,
    hand_side_frac: float,
    hand_y_min_frac: float = 0.20,
    hand_y_max_frac: float = 0.88,
) -> dict[str, np.ndarray]:
    """Build raw-mask part proxies for a coverage guard.

    These are intentionally coarse image-space guards, not part annotations.
    Their job is to catch the recurring failure where the connected surface gets
    a better global IoU by losing upper-head/hairline/hand coverage.
    """

    mask_bool = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(mask_bool)
    empty = np.zeros_like(mask_bool, dtype=bool)
    if xs.size == 0:
        return {
            "head_upper": empty.copy(),
            "hairline_top": empty.copy(),
            "hands_side": empty.copy(),
        }

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    width = max(1, x1 - x0 + 1)
    height = max(1, y1 - y0 + 1)

    yy, xx = np.indices(mask_bool.shape)
    head_cut = y0 + int(round(np.clip(float(head_frac), 0.02, 0.80) * height))
    hair_cut = y0 + int(round(np.clip(float(hairline_frac), 0.02, 0.60) * height))
    side = max(1, int(round(np.clip(float(hand_side_frac), 0.02, 0.45) * width)))
    hand_y0 = y0 + int(round(np.clip(float(hand_y_min_frac), 0.0, 0.95) * height))
    hand_y1 = y0 + int(round(np.clip(float(hand_y_max_frac), 0.05, 1.0) * height))

    head_upper = mask_bool & (yy <= head_cut)
    hairline_top = mask_bool & (yy <= hair_cut)
    side_band = (xx <= x0 + side) | (xx >= x1 - side)
    hand_y_band = (yy >= hand_y0) & (yy <= hand_y1)
    hands_side = mask_bool & side_band & hand_y_band
    return {
        "head_upper": head_upper,
        "hairline_top": hairline_top,
        "hands_side": hands_side,
    }


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.max() > 1.5:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def project_points(points: torch.Tensor, world_to_cam: torch.Tensor, intrinsic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rotation = world_to_cam[:3, :3]
    translation = world_to_cam[:3, 3]
    cam = points @ rotation.T + translation[None, :]
    z = cam[:, 2]
    uvw = cam @ intrinsic.T
    uv = uvw[:, :2] / uvw[:, 2:3].clamp_min(1e-6)
    return uv, z, cam


def sample_grid_values(image: torch.Tensor, uv: torch.Tensor, height: int, width: int) -> torch.Tensor:
    x = uv[:, 0] / float(max(1, width - 1)) * 2.0 - 1.0
    y = uv[:, 1] / float(max(1, height - 1)) * 2.0 - 1.0
    grid = torch.stack([x, y], dim=-1).view(1, 1, -1, 2)
    sampled = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    # grid_sample returns [1, C, 1, N]; expose samples as [N, C].
    return sampled[0, :, 0, :].transpose(0, 1)


def sample_sdf(sdf: torch.Tensor, uv: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return sample_grid_values(sdf, uv, height, width).reshape(-1)


def classify_vertex_parts(canonical_positions: np.ndarray) -> np.ndarray:
    canonical = np.asarray(canonical_positions, dtype=np.float32)
    x = canonical[:, 0]
    y = canonical[:, 1]
    z = canonical[:, 2]
    center_x = float(np.median(x))
    abs_x = np.abs(x - np.median(x))
    y20, y82, y88, y94, y96 = np.percentile(y, [20, 82, 88, 94, 96])
    abs_x88 = np.percentile(abs_x, 88)
    abs_x94 = np.percentile(abs_x, 94)
    z_head_median = float(np.median(z[y > y82])) if np.any(y > y82) else float(np.median(z))

    parts = np.zeros((canonical.shape[0],), dtype=np.int64)
    parts[y < y20] = 5
    parts[y > y82] = 3
    parts[y > y96] = 4
    hands = (abs_x > abs_x88) & (y > y20) & (y < y94)
    far_hands = (abs_x > abs_x94) & (y > y20) & (y < y96)
    left_hand = (hands | far_hands) & (x < center_x)
    right_hand = (hands | far_hands) & (x >= center_x)
    parts[left_hand] = 1
    parts[right_hand] = 2
    # Keep front-face vertices in the head bucket; the hairline bucket stays
    # reserved for top/head cap freedom.
    _ = z_head_median
    return parts.astype(np.int64)


def make_part_limits(parts: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    limits = np.full(parts.shape, float(args.normal_offset_limit_body), dtype=np.float32)
    limits[(parts == 1) | (parts == 2)] = float(args.normal_offset_limit_hands)
    limits[parts == 3] = float(args.normal_offset_limit_head)
    limits[parts == 4] = float(args.normal_offset_limit_hairline)
    limits[parts == 5] = float(args.normal_offset_limit_clothing)

    reg_weights = np.full(parts.shape, 1.0, dtype=np.float32)
    reg_weights[(parts == 1) | (parts == 2)] = 0.55
    reg_weights[parts == 3] = 0.65
    reg_weights[parts == 4] = 0.35
    reg_weights[parts == 5] = 0.45
    return limits, reg_weights


def make_part_free_limits(parts: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    limits = np.zeros(parts.shape, dtype=np.float32)
    limits[(parts == 1) | (parts == 2)] = float(args.part_free_offset_limit_hands)
    limits[parts == 3] = float(args.part_free_offset_limit_face)
    limits[parts == 4] = float(args.part_free_offset_limit_hairline)
    limits[parts == 5] = float(args.part_free_offset_limit_clothing)
    return limits.astype(np.float32)


def unique_edges(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    edges = np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0).astype(np.int64)


def sample_surface_plan(
    base_vertices: np.ndarray,
    faces: np.ndarray,
    vertex_parts: np.ndarray,
    sample_count: int,
    seed: int,
    min_part_samples: dict[int, int] | None = None,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    triangles = np.asarray(base_vertices, dtype=np.float32)[np.asarray(faces, dtype=np.int64)]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    face_vertex_parts_all = vertex_parts[np.asarray(faces, dtype=np.int64)]
    face_parts = np.asarray(
        [np.bincount(row.astype(np.int64), minlength=len(PART_NAMES)).argmax() for row in face_vertex_parts_all],
        dtype=np.int64,
    )

    def choose_from_pool(pool: np.ndarray, count: int) -> np.ndarray:
        if count <= 0 or pool.size == 0:
            return np.zeros((0,), dtype=np.int64)
        local_areas = areas[pool]
        probs = local_areas / np.clip(local_areas.sum(), 1e-8, None)
        return rng.choice(pool, size=int(count), replace=True, p=probs).astype(np.int64)

    total = max(1, int(sample_count))
    chosen: list[np.ndarray] = []
    used = 0
    if min_part_samples:
        for part_id, requested in min_part_samples.items():
            remaining = total - used
            if remaining <= 0:
                break
            count = min(max(0, int(requested)), remaining)
            pool = np.nonzero(face_parts == int(part_id))[0].astype(np.int64)
            selected = choose_from_pool(pool, count)
            if selected.size:
                chosen.append(selected)
                used += int(selected.size)
    remaining = total - used
    if remaining > 0:
        all_pool = np.arange(len(faces), dtype=np.int64)
        chosen.append(choose_from_pool(all_pool, remaining))
    face_indices = np.concatenate(chosen, axis=0) if chosen else choose_from_pool(np.arange(len(faces)), total)
    if face_indices.shape[0] > total:
        face_indices = face_indices[:total]
    rng.shuffle(face_indices)

    u = rng.random(face_indices.shape[0]).astype(np.float32)
    v = rng.random(face_indices.shape[0]).astype(np.float32)
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    bary = np.stack([1.0 - u - v, u, v], axis=1).astype(np.float32)
    surfel_vertex_ids = np.asarray(faces, dtype=np.int64)[face_indices]
    surfel_vertex_parts = vertex_parts[surfel_vertex_ids]
    surfel_parts = np.asarray(
        [np.bincount(row.astype(np.int64), minlength=len(PART_NAMES)).argmax() for row in surfel_vertex_parts],
        dtype=np.int64,
    )
    return {
        "face_indices": face_indices.astype(np.int64),
        "vertex_ids": surfel_vertex_ids.astype(np.int64),
        "barycentric": bary.astype(np.float32),
        "part_ids": surfel_parts.astype(np.int64),
    }


def choose_triangle_render_faces(
    faces: np.ndarray,
    surfel_face_indices: np.ndarray,
    budget: int,
) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    if int(budget) < 0:
        return np.arange(faces.shape[0], dtype=np.int64)
    if int(budget) == 0:
        return np.unique(np.asarray(surfel_face_indices, dtype=np.int64)).astype(np.int64)
    count = min(int(budget), faces.shape[0])
    return np.linspace(0, faces.shape[0] - 1, count).round().astype(np.int64)


def compute_surfels(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    face_indices: torch.Tensor,
    barycentric: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sampled_faces = faces.index_select(0, face_indices)
    tri = vertices[sampled_faces]
    surfels = (tri * barycentric[:, :, None]).sum(dim=1)
    normals = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    normals = F.normalize(normals, dim=1, eps=1e-6)
    return surfels, normals


def render_soft_surfel_maps(
    surfels: torch.Tensor,
    normals: torch.Tensor,
    world_to_cam: torch.Tensor,
    intrinsic: torch.Tensor,
    height: int,
    width: int,
    sigma: float,
    pixel_chunk: int,
    depth_softness: float = 0.0,
) -> dict[str, torch.Tensor]:
    uv, z, cam = project_points(surfels, world_to_cam, intrinsic)
    valid = (
        torch.isfinite(uv).all(dim=1)
        & torch.isfinite(z)
        & (z > 1e-5)
        & (uv[:, 0] >= -3.0 * float(sigma))
        & (uv[:, 0] <= float(width - 1) + 3.0 * float(sigma))
        & (uv[:, 1] >= -3.0 * float(sigma))
        & (uv[:, 1] <= float(height - 1) + 3.0 * float(sigma))
    )
    uv_valid = uv[valid]
    z_valid = z[valid]
    normals_valid = normals[valid]
    if uv_valid.shape[0] == 0:
        zeros = torch.zeros((height, width), dtype=surfels.dtype, device=surfels.device)
        return {
            "mask": zeros,
            "depth": zeros,
            "normal": torch.zeros((height, width, 3), dtype=surfels.dtype, device=surfels.device),
            "visibility": zeros,
            "valid_count": torch.zeros((), dtype=surfels.dtype, device=surfels.device),
        }

    ys = torch.arange(height, dtype=surfels.dtype, device=surfels.device) + 0.5
    xs = torch.arange(width, dtype=surfels.dtype, device=surfels.device) + 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    pixels = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

    masks = []
    depths = []
    normal_maps = []
    vis_maps = []
    sigma2 = max(1e-6, float(sigma) ** 2)
    chunk = max(1, int(pixel_chunk))
    for start in range(0, pixels.shape[0], chunk):
        pixel_chunk_xy = pixels[start : start + chunk]
        d2 = (pixel_chunk_xy[:, None, :] - uv_valid[None, :, :]).square().sum(dim=2)
        spatial_logits = -0.5 * d2 / sigma2
        spatial_weights = torch.exp(spatial_logits)
        sumw_spatial = spatial_weights.sum(dim=1).clamp_min(1e-8)
        # Saturating alpha keeps the mask differentiable without pretending to be a z-buffer.
        alpha = 1.0 - torch.exp(-sumw_spatial)
        if float(depth_softness) > 0.0:
            depth_logits = spatial_logits - z_valid[None, :] / max(1e-6, float(depth_softness))
            depth_logits = depth_logits - depth_logits.max(dim=1, keepdim=True).values
            weights = torch.exp(depth_logits) * (spatial_weights > 1e-7).to(spatial_weights.dtype)
        else:
            weights = spatial_weights
        sumw = weights.sum(dim=1).clamp_min(1e-8)
        depth = (weights * z_valid[None, :]).sum(dim=1) / sumw
        normal = (weights @ normals_valid) / sumw[:, None]
        normal = F.normalize(normal, dim=1, eps=1e-6)
        masks.append(alpha)
        depths.append(depth)
        normal_maps.append(normal)
        vis_maps.append(sumw_spatial)

    mask = torch.cat(masks, dim=0).reshape(height, width).clamp(0.0, 1.0)
    depth = torch.cat(depths, dim=0).reshape(height, width)
    normal = torch.cat(normal_maps, dim=0).reshape(height, width, 3)
    visibility = torch.cat(vis_maps, dim=0).reshape(height, width)
    return {
        "mask": mask,
        "depth": depth,
        "normal": normal,
        "visibility": visibility,
        "valid_count": valid.float().sum(),
    }


def render_soft_triangle_maps(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    face_indices: torch.Tensor,
    world_to_cam: torch.Tensor,
    intrinsic: torch.Tensor,
    height: int,
    width: int,
    pixel_chunk: int,
    inside_softness: float,
    face_chunk: int = 256,
) -> dict[str, torch.Tensor]:
    """CPU-friendly soft triangle smoke for connected surface visibility.

    This is intentionally modest: it renders a sampled set of connected mesh
    triangles with a soft barycentric inside test. It is a diagnostic renderer,
    not a production rasterizer or a strict teacher gate.
    """

    if face_indices.numel() == 0:
        zeros = torch.zeros((height, width), dtype=vertices.dtype, device=vertices.device)
        return {
            "mask": zeros,
            "depth": zeros,
            "normal": torch.zeros((height, width, 3), dtype=vertices.dtype, device=vertices.device),
            "visibility": zeros,
            "valid_count": torch.zeros((), dtype=vertices.dtype, device=vertices.device),
        }
    selected_faces = faces.index_select(0, face_indices)
    tri_world = vertices[selected_faces]
    tri_flat = tri_world.reshape(-1, 3)
    uv_flat, z_flat, _ = project_points(tri_flat, world_to_cam, intrinsic)
    uv_tri = uv_flat.reshape(-1, 3, 2)
    z_tri = z_flat.reshape(-1, 3)
    face_normals = torch.cross(tri_world[:, 1] - tri_world[:, 0], tri_world[:, 2] - tri_world[:, 0], dim=1)
    face_normals = F.normalize(face_normals, dim=1, eps=1e-6)

    finite = torch.isfinite(uv_tri).all(dim=(1, 2)) & torch.isfinite(z_tri).all(dim=1) & (z_tri > 1e-5).all(dim=1)
    min_xy = uv_tri.min(dim=1).values
    max_xy = uv_tri.max(dim=1).values
    intersects = (
        (max_xy[:, 0] >= -2.0)
        & (min_xy[:, 0] <= float(width + 1))
        & (max_xy[:, 1] >= -2.0)
        & (min_xy[:, 1] <= float(height + 1))
    )
    valid_faces = finite & intersects
    if not valid_faces.any():
        zeros = torch.zeros((height, width), dtype=vertices.dtype, device=vertices.device)
        return {
            "mask": zeros,
            "depth": zeros,
            "normal": torch.zeros((height, width, 3), dtype=vertices.dtype, device=vertices.device),
            "visibility": zeros,
            "valid_count": torch.zeros((), dtype=vertices.dtype, device=vertices.device),
        }
    uv_tri = uv_tri[valid_faces]
    z_tri = z_tri[valid_faces]
    face_normals = face_normals[valid_faces]

    ys = torch.arange(height, dtype=vertices.dtype, device=vertices.device) + 0.5
    xs = torch.arange(width, dtype=vertices.dtype, device=vertices.device) + 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    pixels = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

    masks = []
    depths = []
    normal_maps = []
    vis_maps = []
    pchunk = max(1, int(pixel_chunk))
    fchunk = max(1, int(face_chunk))
    sharpness = max(1e-3, float(inside_softness))
    for start in range(0, pixels.shape[0], pchunk):
        pixel_xy = pixels[start : start + pchunk]
        sumw = torch.zeros((pixel_xy.shape[0],), dtype=vertices.dtype, device=vertices.device)
        depth_num = torch.zeros_like(sumw)
        normal_num = torch.zeros((pixel_xy.shape[0], 3), dtype=vertices.dtype, device=vertices.device)
        for face_start in range(0, uv_tri.shape[0], fchunk):
            uv = uv_tri[face_start : face_start + fchunk]
            z = z_tri[face_start : face_start + fchunk]
            normals = face_normals[face_start : face_start + fchunk]
            v0 = uv[:, 0, :]
            v1 = uv[:, 1, :]
            v2 = uv[:, 2, :]
            denom = (v1[:, 1] - v2[:, 1]) * (v0[:, 0] - v2[:, 0]) + (v2[:, 0] - v1[:, 0]) * (v0[:, 1] - v2[:, 1])
            good = denom.abs() > 1e-6
            if not good.any():
                continue
            v0 = v0[good]
            v1 = v1[good]
            v2 = v2[good]
            z = z[good]
            normals = normals[good]
            denom = denom[good]
            px = pixel_xy[:, 0:1]
            py = pixel_xy[:, 1:2]
            w0 = ((v1[:, 1] - v2[:, 1])[None, :] * (px - v2[:, 0][None, :]) + (v2[:, 0] - v1[:, 0])[None, :] * (py - v2[:, 1][None, :])) / denom[None, :]
            w1 = ((v2[:, 1] - v0[:, 1])[None, :] * (px - v2[:, 0][None, :]) + (v0[:, 0] - v2[:, 0])[None, :] * (py - v2[:, 1][None, :])) / denom[None, :]
            w2 = 1.0 - w0 - w1
            inside = torch.sigmoid(sharpness * w0) * torch.sigmoid(sharpness * w1) * torch.sigmoid(sharpness * w2)
            tri_z = w0 * z[:, 0][None, :] + w1 * z[:, 1][None, :] + w2 * z[:, 2][None, :]
            positive = tri_z > 1e-5
            weights = inside * positive.to(inside.dtype)
            local_sumw = weights.sum(dim=1)
            sumw = sumw + local_sumw
            depth_num = depth_num + (weights * tri_z).sum(dim=1)
            normal_num = normal_num + weights @ normals
        alpha = 1.0 - torch.exp(-sumw.clamp_min(0.0))
        depth = depth_num / sumw.clamp_min(1e-8)
        normal = F.normalize(normal_num / sumw.clamp_min(1e-8)[:, None], dim=1, eps=1e-6)
        masks.append(alpha)
        depths.append(depth)
        normal_maps.append(normal)
        vis_maps.append(sumw)

    mask = torch.cat(masks, dim=0).reshape(height, width).clamp(0.0, 1.0)
    depth = torch.cat(depths, dim=0).reshape(height, width)
    normal = torch.cat(normal_maps, dim=0).reshape(height, width, 3)
    visibility = torch.cat(vis_maps, dim=0).reshape(height, width)
    return {
        "mask": mask,
        "depth": depth,
        "normal": normal,
        "visibility": visibility,
        "valid_count": valid_faces.float().sum(),
    }


def photometric_consistency_loss(
    surfels: torch.Tensor,
    view_payloads: list[dict[str, Any]],
    height: int,
    width: int,
    *,
    visibility_depths: list[torch.Tensor] | None = None,
    depth_tolerance: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    colors = []
    weights = []
    for view_i, payload in enumerate(view_payloads):
        uv, z, _ = project_points(surfels, payload["world_to_cam"], payload["intrinsic"])
        in_image = (
            (z > 1e-5)
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] <= width - 1)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] <= height - 1)
        )
        if visibility_depths is not None and depth_tolerance > 0.0 and view_i < len(visibility_depths):
            sampled_depth = sample_grid_values(visibility_depths[view_i][None, None], uv, height, width).reshape(-1)
            visible = torch.isfinite(sampled_depth) & (sampled_depth > 1e-5)
            visible = visible & ((z - sampled_depth).abs() <= float(depth_tolerance))
            in_image = in_image & visible
        sampled_rgb = sample_grid_values(payload["rgb_t"], uv, height, width)
        sampled_mask = sample_grid_values(payload["mask_t"], uv, height, width).reshape(-1).clamp(0.0, 1.0)
        weight = sampled_mask * in_image.float()
        colors.append(sampled_rgb)
        weights.append(weight)

    color_stack = torch.stack(colors, dim=0)
    weight_stack = torch.stack(weights, dim=0)
    support = (weight_stack > 0.25).float().sum(dim=0)
    valid = support >= 2.0
    if not valid.any():
        zero = surfels.sum() * 0.0
        return zero, {"valid_surfels": 0.0, "mean_support": 0.0}
    local_colors = color_stack[:, valid, :]
    local_weights = weight_stack[:, valid].clamp_min(0.0)
    norm = local_weights.sum(dim=0).clamp_min(1e-6)
    mean = (local_colors * local_weights[:, :, None]).sum(dim=0) / norm[:, None]
    residual = torch.sqrt((local_colors - mean[None, :, :]).square().sum(dim=2) + 1e-6)
    loss = (residual * local_weights).sum() / local_weights.sum().clamp_min(1e-6)
    return loss, {
        "valid_surfels": float(valid.float().sum().detach().cpu()),
        "mean_support": float(support[valid].mean().detach().cpu()),
    }


def part_recall_loss(
    surfels: torch.Tensor,
    normals: torch.Tensor,
    surfel_part_ids: torch.Tensor,
    view_payloads: list[dict[str, Any]],
    height: int,
    width: int,
    *,
    sigma: float,
    pixel_chunk: int,
    depth_softness: float,
    min_pixels: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    specs = {
        "head_upper": (3, 4),
        "hairline_top": (4,),
        "hands_side": (1, 2),
    }
    losses: list[torch.Tensor] = []
    rows: dict[str, dict[str, float]] = {}
    for name, part_ids in specs.items():
        part_mask = torch.zeros_like(surfel_part_ids, dtype=torch.bool)
        for part_id in part_ids:
            part_mask = part_mask | (surfel_part_ids == int(part_id))
        if not part_mask.any():
            continue
        part_surfels = surfels[part_mask]
        part_normals = normals[part_mask]
        per_view_losses: list[torch.Tensor] = []
        pixel_counts: list[float] = []
        for payload in view_payloads:
            target = payload.get("part_target_tensors", {}).get(name)
            if target is None:
                continue
            target_2d = target.reshape(height, width)
            target_area = target_2d.sum()
            if float(target_area.detach().cpu()) < int(min_pixels):
                continue
            render = render_soft_surfel_maps(
                surfels=part_surfels,
                normals=part_normals,
                world_to_cam=payload["world_to_cam"],
                intrinsic=payload["intrinsic"],
                height=height,
                width=width,
                sigma=sigma,
                pixel_chunk=pixel_chunk,
                depth_softness=depth_softness,
            )
            rendered_mask = render["mask"].clamp(0.0, 1.0)
            per_view_losses.append((target_2d * (1.0 - rendered_mask)).sum() / target_area.clamp_min(1.0))
            pixel_counts.append(float(target_area.detach().cpu()))
        if per_view_losses:
            stacked = torch.stack(per_view_losses)
            losses.append(stacked.mean())
            rows[name] = {
                "views": float(len(per_view_losses)),
                "mean_loss": float(stacked.mean().detach().cpu()),
                "mean_target_pixels": float(np.mean(pixel_counts)) if pixel_counts else 0.0,
            }
    if not losses:
        zero = surfels.sum() * 0.0
        return zero, {"enabled_terms": 0.0, "terms": rows}
    loss = torch.stack(losses).mean()
    return loss, {"enabled_terms": float(len(losses)), "terms": rows}


def vertex_silhouette_boundary_loss(
    vertices: torch.Tensor,
    vertex_ids: torch.Tensor,
    view_payloads: list[dict[str, Any]],
    height: int,
    width: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    if vertex_ids.numel() == 0:
        zero = vertices.sum() * 0.0
        return zero, {"visible_vertices": 0.0, "views": 0.0}
    selected = vertices.index_select(0, vertex_ids)
    losses: list[torch.Tensor] = []
    visible_counts: list[float] = []
    for payload in view_payloads:
        uv, z, _ = project_points(selected, payload["world_to_cam"], payload["intrinsic"])
        valid = (
            (z > 1e-5)
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] <= width - 1)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] <= height - 1)
        )
        if not valid.any():
            continue
        sdf_values = sample_sdf(payload["sdf"], uv[valid], height, width)
        losses.append(sdf_values.abs().mean())
        visible_counts.append(float(valid.float().sum().detach().cpu()))
    if not losses:
        zero = vertices.sum() * 0.0
        return zero, {"visible_vertices": 0.0, "views": 0.0}
    stacked = torch.stack(losses)
    return stacked.mean(), {
        "visible_vertices": float(np.mean(visible_counts)) if visible_counts else 0.0,
        "views": float(len(losses)),
    }


def vertex_image_edge_loss(
    vertices: torch.Tensor,
    vertex_ids: torch.Tensor,
    view_payloads: list[dict[str, Any]],
    height: int,
    width: int,
    *,
    max_distance: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if vertex_ids.numel() == 0:
        zero = vertices.sum() * 0.0
        return zero, {"visible_vertices": 0.0, "views": 0.0, "edge_pixels": 0.0}
    selected = vertices.index_select(0, vertex_ids)
    losses: list[torch.Tensor] = []
    visible_counts: list[float] = []
    edge_counts: list[float] = []
    for payload in view_payloads:
        edge_meta = payload.get("image_edge_meta", {})
        if int(edge_meta.get("edge_pixels", 0)) <= 0:
            continue
        edge_sdf = payload.get("image_edge_sdf")
        if edge_sdf is None:
            continue
        uv, z, _ = project_points(selected, payload["world_to_cam"], payload["intrinsic"])
        valid = (
            (z > 1e-5)
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] <= width - 1)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] <= height - 1)
        )
        if not valid.any():
            continue
        distances = sample_sdf(edge_sdf, uv[valid], height, width)
        if float(max_distance) > 0.0:
            distances = distances.clamp(max=float(max_distance))
        losses.append(distances.mean())
        visible_counts.append(float(valid.float().sum().detach().cpu()))
        edge_counts.append(float(edge_meta.get("edge_pixels", 0)))
    if not losses:
        zero = vertices.sum() * 0.0
        return zero, {"visible_vertices": 0.0, "views": 0.0, "edge_pixels": 0.0}
    stacked = torch.stack(losses)
    return stacked.mean(), {
        "visible_vertices": float(np.mean(visible_counts)) if visible_counts else 0.0,
        "views": float(len(losses)),
        "edge_pixels": float(np.mean(edge_counts)) if edge_counts else 0.0,
    }


def face_landmark_projection_loss(
    vertices: torch.Tensor,
    vertex_ids: torch.Tensor,
    view_payloads: list[dict[str, Any]],
    height: int,
    width: int,
    *,
    bidirectional_weight: float,
    min_points: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Weak 2D landmark constraint attached to the connected face mesh.

    This intentionally does not triangulate or create a floating 3D face patch.
    The only optimized geometry remains the connected surface vertices.
    """

    if vertex_ids.numel() == 0:
        zero = vertices.sum() * 0.0
        return zero, {"views": 0.0, "face_vertices": 0.0, "landmarks": 0.0, "reason": "no_face_vertices"}
    selected = vertices.index_select(0, vertex_ids)
    losses: list[torch.Tensor] = []
    landmark_counts: list[float] = []
    vertex_counts: list[float] = []
    scale = float(max(height, width))
    for payload in view_payloads:
        landmarks = payload.get("face_landmarks_t")
        if landmarks is None or landmarks.numel() < int(min_points) * 2:
            continue
        uv, z, _ = project_points(selected, payload["world_to_cam"], payload["intrinsic"])
        valid = (
            (z > 1e-5)
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] <= width - 1)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] <= height - 1)
        )
        if int(valid.float().sum().detach().cpu()) < 8:
            continue
        uv_valid = uv[valid] / scale
        landmarks_xy = landmarks[:, :2] / scale
        dists = torch.cdist(landmarks_xy, uv_valid)
        lm_to_mesh = dists.min(dim=1).values.mean()
        mesh_to_lm = dists.min(dim=0).values.mean()
        losses.append(lm_to_mesh + float(bidirectional_weight) * mesh_to_lm)
        landmark_counts.append(float(landmarks_xy.shape[0]))
        vertex_counts.append(float(uv_valid.shape[0]))
    if not losses:
        zero = vertices.sum() * 0.0
        return zero, {"views": 0.0, "face_vertices": 0.0, "landmarks": 0.0, "reason": "no_usable_landmark_views"}
    stacked = torch.stack(losses)
    return stacked.mean(), {
        "views": float(len(losses)),
        "mean_face_vertices": float(np.mean(vertex_counts)) if vertex_counts else 0.0,
        "mean_landmarks": float(np.mean(landmark_counts)) if landmark_counts else 0.0,
        "reason": None,
    }


def hand_landmark_projection_loss(
    vertices: torch.Tensor,
    left_vertex_ids: torch.Tensor,
    right_vertex_ids: torch.Tensor,
    view_payloads: list[dict[str, Any]],
    height: int,
    width: int,
    *,
    bidirectional_weight: float,
    min_points: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Weak 2D hand constraint attached to connected SMPL-X hand vertices."""

    if left_vertex_ids.numel() == 0 and right_vertex_ids.numel() == 0:
        zero = vertices.sum() * 0.0
        return zero, {"views": 0.0, "hands": 0.0, "reason": "no_hand_vertices"}

    def projected_vertices(vertex_ids: torch.Tensor, payload: dict[str, Any]) -> torch.Tensor | None:
        if vertex_ids.numel() == 0:
            return None
        selected = vertices.index_select(0, vertex_ids)
        uv, z, _ = project_points(selected, payload["world_to_cam"], payload["intrinsic"])
        valid = (
            (z > 1e-5)
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] <= width - 1)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] <= height - 1)
        )
        if int(valid.float().sum().detach().cpu()) < 6:
            return None
        return uv[valid] / float(max(height, width))

    def chamfer(landmarks_xy: torch.Tensor, uv_valid: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(landmarks_xy, uv_valid)
        return dists.min(dim=1).values.mean() + float(bidirectional_weight) * dists.min(dim=0).values.mean()

    losses: list[torch.Tensor] = []
    used_hands = 0
    used_views = 0
    for payload in view_payloads:
        hand_sets = payload.get("hand_landmarks_t") or []
        if not hand_sets:
            continue
        left_uv = projected_vertices(left_vertex_ids, payload)
        right_uv = projected_vertices(right_vertex_ids, payload)
        if left_uv is None and right_uv is None:
            continue
        view_used = False
        for landmarks in hand_sets:
            if landmarks.numel() < int(min_points) * 2:
                continue
            landmarks_xy = landmarks[:, :2] / float(max(height, width))
            candidates: list[torch.Tensor] = []
            if left_uv is not None:
                candidates.append(chamfer(landmarks_xy, left_uv))
            if right_uv is not None:
                candidates.append(chamfer(landmarks_xy, right_uv))
            if candidates:
                losses.append(torch.stack(candidates).min())
                used_hands += 1
                view_used = True
        if view_used:
            used_views += 1
    if not losses:
        zero = vertices.sum() * 0.0
        return zero, {"views": 0.0, "hands": 0.0, "reason": "no_usable_hand_landmark_views"}
    return torch.stack(losses).mean(), {"views": float(used_views), "hands": float(used_hands), "reason": None}


def compute_mask_metrics(rendered: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    rendered = np.asarray(rendered, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = rendered & target
    union = rendered | target
    render_pixels = int(rendered.sum())
    target_pixels = int(target.sum())
    intersection_pixels = int(intersection.sum())
    union_pixels = int(union.sum())
    return {
        "render_pixels": render_pixels,
        "target_pixels": target_pixels,
        "intersection_pixels": intersection_pixels,
        "union_pixels": union_pixels,
        "iou": float(intersection_pixels / union_pixels) if union_pixels else None,
        "target_recall": float(intersection_pixels / target_pixels) if target_pixels else None,
        "render_precision": float(intersection_pixels / render_pixels) if render_pixels else None,
    }


def summarize(values: list[float | None]) -> dict[str, Any]:
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
    out = np.asarray(rgb, dtype=np.float32)
    if out.max() <= 1.5:
        out *= 255.0
    target = np.asarray(target, dtype=bool)
    rendered = np.asarray(rendered, dtype=bool)
    both = target & rendered
    target_only = target & ~rendered
    rendered_only = rendered & ~target
    out[target_only] = 0.45 * out[target_only] + 0.55 * np.array([0, 220, 0], dtype=np.float32)
    out[rendered_only] = 0.45 * out[rendered_only] + 0.55 * np.array([240, 0, 0], dtype=np.float32)
    out[both] = 0.55 * out[both] + 0.45 * np.array([255, 220, 0], dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def save_contact_sheet(paths: list[Path], output_path: Path, columns: int = 4) -> None:
    if not paths:
        return
    images = [Image.open(path).convert("RGB") for path in paths]
    width, height = images[0].size
    rows = int(np.ceil(len(images) / max(1, columns)))
    sheet = Image.new("RGB", (columns * width, rows * height), (255, 255, 255))
    for idx, image in enumerate(images):
        sheet.paste(image, ((idx % columns) * width, (idx // columns) * height))
    sheet.save(output_path)


def save_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {vertices.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write(f"element face {faces.shape[0]}\n")
        handle.write("property list uchar int vertex_indices\nend_header\n")
        for vertex in vertices:
            handle.write(f"{float(vertex[0])} {float(vertex[1])} {float(vertex[2])}\n")
        for face in faces:
            handle.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def save_point_ply(path: Path, points: np.ndarray, normals: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    normals = None if normals is None else np.asarray(normals, dtype=np.float32)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        has_normals = normals is not None and normals.shape == points.shape
        if has_normals:
            handle.write("property float nx\nproperty float ny\nproperty float nz\n")
        handle.write("end_header\n")
        if has_normals:
            for point, normal in zip(points, normals):
                handle.write(
                    f"{float(point[0])} {float(point[1])} {float(point[2])} "
                    f"{float(normal[0])} {float(normal[1])} {float(normal[2])}\n"
                )
        else:
            for point in points:
                handle.write(f"{float(point[0])} {float(point[1])} {float(point[2])}\n")


def save_depth_image(path: Path, depth: np.ndarray, mask: np.ndarray) -> None:
    depth = np.asarray(depth, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if mask.any():
        vals = depth[mask]
        lo, hi = np.percentile(vals[np.isfinite(vals)], [2, 98])
        denom = max(1e-6, float(hi - lo))
        img = np.clip((depth - lo) / denom, 0.0, 1.0)
    else:
        img = np.zeros_like(depth, dtype=np.float32)
    Image.fromarray((img * 255.0).astype(np.uint8)).save(path)


def save_normal_image(path: Path, normal: np.ndarray, mask: np.ndarray) -> None:
    normal = np.asarray(normal, dtype=np.float32)
    img = np.clip((normal + 1.0) * 0.5, 0.0, 1.0)
    img[~np.asarray(mask, dtype=bool)] = 0.0
    Image.fromarray((img * 255.0).astype(np.uint8)).save(path)


def parse_view_spec(spec: str, count: int) -> list[int]:
    text = str(spec).strip().lower()
    if text == "all":
        return list(range(count))
    out: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0 or idx >= count:
            raise IndexError(f"view index {idx} outside [0,{count})")
        out.append(idx)
    return sorted(set(out))


def backproject_pixels_to_world(
    pixels_xy: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    world_to_cam: np.ndarray,
) -> np.ndarray:
    pixels_xy = np.asarray(pixels_xy, dtype=np.float32)
    depth = np.asarray(depth, dtype=np.float32).reshape(-1)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    world_to_cam = homogeneous(world_to_cam)
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    cam = np.stack(
        [
            (pixels_xy[:, 0] + 0.5 - cx) * depth / max(1e-8, fx),
            (pixels_xy[:, 1] + 0.5 - cy) * depth / max(1e-8, fy),
            depth,
        ],
        axis=1,
    ).astype(np.float32)
    cam_to_world = np.linalg.inv(world_to_cam).astype(np.float32)
    return (cam @ cam_to_world[:3, :3].T + cam_to_world[:3, 3][None, :]).astype(np.float32)


def nearest_rendered_depths(
    query_xy: np.ndarray,
    source_xy: np.ndarray,
    source_depth: np.ndarray,
    max_radius: float,
    *,
    chunk: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query_xy = np.asarray(query_xy, dtype=np.float32)
    source_xy = np.asarray(source_xy, dtype=np.float32)
    source_depth = np.asarray(source_depth, dtype=np.float32).reshape(-1)
    if query_xy.size == 0 or source_xy.size == 0:
        return (
            np.zeros((0,), dtype=bool),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    keep = np.zeros((query_xy.shape[0],), dtype=bool)
    depth = np.zeros((query_xy.shape[0],), dtype=np.float32)
    distance = np.full((query_xy.shape[0],), np.inf, dtype=np.float32)
    max_r2 = float(max_radius) ** 2
    for start in range(0, query_xy.shape[0], max(1, int(chunk))):
        q = query_xy[start : start + chunk]
        d2 = ((q[:, None, :] - source_xy[None, :, :]) ** 2).sum(axis=2)
        nearest = d2.argmin(axis=1)
        best = d2[np.arange(q.shape[0]), nearest]
        local_keep = np.isfinite(best) & (best <= max_r2)
        keep[start : start + q.shape[0]] = local_keep
        depth[start : start + q.shape[0]] = source_depth[nearest]
        distance[start : start + q.shape[0]] = np.sqrt(np.maximum(best, 0.0)).astype(np.float32)
    return keep, depth, distance


def mask_support_for_points(
    points_world: np.ndarray,
    view_payloads: list[dict[str, Any]],
    height: int,
    width: int,
) -> np.ndarray:
    points_world = np.asarray(points_world, dtype=np.float32)
    support = np.zeros((points_world.shape[0],), dtype=np.int32)
    if points_world.size == 0:
        return support
    points_t = torch.from_numpy(points_world)
    for payload in view_payloads:
        uv, z, _ = project_points(
            points_t,
            torch.from_numpy(payload["world_to_cam_np"].astype(np.float32)),
            torch.from_numpy(payload["intrinsic_np"].astype(np.float32)),
        )
        uv_np = uv.numpy()
        z_np = z.numpy()
        px = np.rint(uv_np[:, 0]).astype(np.int32)
        py = np.rint(uv_np[:, 1]).astype(np.int32)
        valid = (
            np.isfinite(uv_np).all(axis=1)
            & np.isfinite(z_np)
            & (z_np > 1e-5)
            & (px >= 0)
            & (px < width)
            & (py >= 0)
            & (py < height)
        )
        if valid.any():
            mask = np.asarray(payload["mask"], dtype=bool)
            hit = np.zeros_like(valid)
            hit[valid] = mask[py[valid], px[valid]]
            support += hit.astype(np.int32)
    return support


def voxel_downsample_points(
    points: np.ndarray,
    normals: np.ndarray,
    support: np.ndarray,
    voxel: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    support = np.asarray(support, dtype=np.int32)
    if points.shape[0] == 0:
        return points, normals, support
    if voxel > 0:
        keys = np.round(points / float(voxel)).astype(np.int64)
        _, unique_idx = np.unique(keys, axis=0, return_index=True)
        points = points[unique_idx]
        normals = normals[unique_idx]
        support = support[unique_idx]
    order = np.lexsort((np.arange(points.shape[0]), -support))
    if max_points > 0:
        order = order[: int(max_points)]
    points = points[order].astype(np.float32)
    normals = normals[order].astype(np.float32)
    normals = normals / np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8, None)
    return points, normals, support[order].astype(np.int32)


def build_extra_hairline_surfels(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    view_payloads: list[dict[str, Any]],
    max_points: int,
    max_per_view: int,
    min_support: int,
    top_frac: float,
    nearest_radius: float,
    depth_offset: float,
    voxel: float,
    output_dir: Path,
) -> dict[str, Any]:
    if int(max_points) <= 0:
        return {
            "enabled": False,
            "points": np.zeros((0, 3), dtype=np.float32),
            "normals": np.zeros((0, 3), dtype=np.float32),
            "support": np.zeros((0,), dtype=np.int32),
            "summary": {"candidate_points": 0, "kept_points": 0},
        }

    height, width = view_payloads[0]["mask"].shape
    head_center = np.asarray(vertices, dtype=np.float32)[np.asarray(vertices[:, 1]).argsort()[-max(16, vertices.shape[0] // 12) :]].mean(axis=0)
    candidates: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    per_view_rows: list[dict[str, Any]] = []
    for payload in view_payloads:
        depth_map, _, _, raster_mask, _ = rasterize_world_mesh(
            world_vertices=vertices,
            faces=faces,
            world_to_cam=payload["world_to_cam_np"],
            intrinsic=payload["intrinsic_np"],
            image_hw=payload["mask"].shape,
            silhouette_mask=None,
            fill_knn=0,
            return_raster_mask=True,
        )
        target = np.asarray(payload["mask"], dtype=bool)
        ys_mask, xs_mask = np.nonzero(target)
        if ys_mask.size == 0:
            per_view_rows.append({"view_index": payload["view_index"], "candidate_pixels": 0, "kept_pixels": 0})
            continue
        top = int(ys_mask.min())
        bottom = int(ys_mask.max())
        top_limit = int(round(top + np.clip(float(top_frac), 0.05, 0.80) * max(1, bottom - top + 1)))
        yy = np.arange(height)[:, None]
        head_top_band = yy <= top_limit
        missing = target & ~raster_mask & head_top_band
        source = raster_mask & target & (yy <= min(height - 1, top_limit + int(max(2.0, nearest_radius * 1.5))))
        my, mx = np.nonzero(missing)
        sy, sx = np.nonzero(source)
        if my.size == 0 or sy.size == 0:
            per_view_rows.append(
                {
                    "view_index": payload["view_index"],
                    "top_limit": top_limit,
                    "candidate_pixels": int(my.size),
                    "kept_pixels": 0,
                }
            )
            continue
        if my.size > int(max_per_view):
            pick = np.linspace(0, my.size - 1, int(max_per_view)).round().astype(np.int64)
            my = my[pick]
            mx = mx[pick]
        query_xy = np.stack([mx, my], axis=1).astype(np.float32)
        source_xy = np.stack([sx, sy], axis=1).astype(np.float32)
        source_depth = depth_map[sy, sx].astype(np.float32)
        keep, nearest_depth, nearest_distance = nearest_rendered_depths(
            query_xy,
            source_xy,
            source_depth,
            max_radius=float(nearest_radius),
        )
        valid_depth = np.isfinite(nearest_depth) & ((nearest_depth + float(depth_offset)) > 1e-5)
        keep &= valid_depth
        world = backproject_pixels_to_world(
            query_xy[keep],
            nearest_depth[keep] + float(depth_offset),
            payload["intrinsic_np"],
            payload["world_to_cam_np"],
        )
        if world.shape[0] > 0:
            candidates.append(world)
            distances.append(nearest_distance[keep].astype(np.float32))
        per_view_rows.append(
            {
                "view_index": payload["view_index"],
                "camera_id": payload["camera_id"],
                "top_limit": top_limit,
                "candidate_pixels": int(query_xy.shape[0]),
                "kept_pixels": int(world.shape[0]),
                "nearest_distance_p50": None if not keep.any() else float(np.percentile(nearest_distance[keep], 50)),
                "nearest_distance_p90": None if not keep.any() else float(np.percentile(nearest_distance[keep], 90)),
            }
        )

    if not candidates:
        return {
            "enabled": True,
            "points": np.zeros((0, 3), dtype=np.float32),
            "normals": np.zeros((0, 3), dtype=np.float32),
            "support": np.zeros((0,), dtype=np.int32),
            "summary": {
                "candidate_points": 0,
                "kept_points": 0,
                "min_support": int(min_support),
                "per_view": per_view_rows,
            },
        }
    points = np.concatenate(candidates, axis=0).astype(np.float32)
    support = mask_support_for_points(points, view_payloads, height, width)
    keep = support >= int(max(1, min_support))
    points = points[keep]
    support = support[keep]
    normals = points - head_center[None, :]
    normals = normals / np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8, None)
    points, normals, support = voxel_downsample_points(points, normals, support, float(voxel), int(max_points))
    ply_path = output_dir / "optimized_softsurfel_extra_hairline_points.ply"
    save_point_ply(ply_path, points, normals)
    summary = {
        "enabled": True,
        "candidate_points": int(sum(row.get("kept_pixels", 0) for row in per_view_rows)),
        "kept_points": int(points.shape[0]),
        "min_support": int(min_support),
        "support": summarize(support.astype(np.float32).tolist()),
        "top_frac": float(top_frac),
        "nearest_radius": float(nearest_radius),
        "depth_offset": float(depth_offset),
        "voxel": float(voxel),
        "ply_path": ply_path,
        "per_view": per_view_rows,
        "note": (
            "Extra hairline surfels are shared 3D diagnostic support built from raw mask missing pixels "
            "and SMPL-X depth anchors. They are not an external teacher and do not pass any gate by themselves."
        ),
    }
    return {"enabled": True, "points": points, "normals": normals, "support": support, "summary": summary}


def splat_points_into_maps(
    *,
    depth_map: np.ndarray,
    point_map: np.ndarray,
    mask_map: np.ndarray,
    normal_map: np.ndarray,
    points: np.ndarray,
    normals_world: np.ndarray,
    world_to_cam: np.ndarray,
    intrinsic: np.ndarray,
    radius: int,
) -> dict[str, int]:
    points = np.asarray(points, dtype=np.float32)
    normals_world = np.asarray(normals_world, dtype=np.float32)
    if points.shape[0] == 0:
        return {"candidate_points": 0, "projected_points": 0, "updated_pixels": 0}
    world_to_cam = homogeneous(world_to_cam)
    rotation = world_to_cam[:3, :3].astype(np.float32)
    translation = world_to_cam[:3, 3].astype(np.float32)
    cam = points @ rotation.T + translation[None, :]
    z = cam[:, 2]
    uvw = cam @ np.asarray(intrinsic, dtype=np.float32).T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-8, None)
    height, width = depth_map.shape
    px = np.rint(uv[:, 0]).astype(np.int32)
    py = np.rint(uv[:, 1]).astype(np.int32)
    valid = (
        np.isfinite(uv).all(axis=1)
        & np.isfinite(z)
        & (z > 1e-5)
        & (px >= -int(radius))
        & (px < width + int(radius))
        & (py >= -int(radius))
        & (py < height + int(radius))
    )
    cam_normals = normals_world @ rotation.T
    cam_normals = cam_normals / np.clip(np.linalg.norm(cam_normals, axis=1, keepdims=True), 1e-8, None)
    updated = 0
    projected = int(valid.sum())
    rad = max(0, int(radius))
    for idx in np.nonzero(valid)[0].tolist():
        for dy in range(-rad, rad + 1):
            yy = int(py[idx] + dy)
            if yy < 0 or yy >= height:
                continue
            for dx in range(-rad, rad + 1):
                xx = int(px[idx] + dx)
                if xx < 0 or xx >= width:
                    continue
                if rad > 0 and dx * dx + dy * dy > rad * rad:
                    continue
                if (not mask_map[yy, xx]) or float(z[idx]) < float(depth_map[yy, xx]):
                    depth_map[yy, xx] = float(z[idx])
                    point_map[yy, xx] = points[idx]
                    normal_map[yy, xx] = cam_normals[idx]
                    mask_map[yy, xx] = True
                    updated += 1
    return {"candidate_points": int(points.shape[0]), "projected_points": projected, "updated_pixels": int(updated)}


def export_raster_targets(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_normals: np.ndarray,
    extra_points: np.ndarray | None = None,
    extra_normals: np.ndarray | None = None,
    extra_splat_radius: int = 1,
    scene_dir: Path,
    dataset_root: Path,
    subset_name: str,
    target_size: int,
    view_spec: str,
    output_dir: Path,
) -> dict[str, Any]:
    export_dir = output_dir / "rasterized_surface_targets"
    image_dir = export_dir / "debug_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    manifest = recover_legacy_crop_source_sizes(scene_dir, load_scene_manifest(scene_dir))
    camera_params, camera_source = resolve_scene_camera_params(manifest, dataset_root, subset_name)
    views = list(manifest["exported_views"])
    view_indices = parse_view_spec(view_spec, len(views))

    depths = []
    world_points = []
    normal_maps = []
    masks = []
    intrinsics = []
    extrinsics = []
    camera_ids = []
    rows = []
    extra_points_np = np.zeros((0, 3), dtype=np.float32) if extra_points is None else np.asarray(extra_points, dtype=np.float32)
    extra_normals_np = (
        np.zeros_like(extra_points_np, dtype=np.float32)
        if extra_normals is None
        else np.asarray(extra_normals, dtype=np.float32)
    )
    if extra_normals_np.shape != extra_points_np.shape:
        extra_normals_np = np.zeros_like(extra_points_np, dtype=np.float32)
    extra_rows = []
    for view_idx in view_indices:
        view = views[view_idx]
        camera_id = str(view["camera_id"]).zfill(2)
        intrinsic_np = align_intrinsics_for_loaded_scene_view(
            np.asarray(camera_params[camera_id]["intrinsic"], dtype=np.float32),
            view,
            target_size=int(target_size),
        )
        world_to_cam_np = homogeneous(np.asarray(camera_params[camera_id]["world_to_cam"], dtype=np.float32))
        cam_normals = vertex_normals @ world_to_cam_np[:3, :3].T
        cam_normals = cam_normals / np.clip(np.linalg.norm(cam_normals, axis=1, keepdims=True), 1e-8, None)
        depth_map, point_map, mask_map, normal_map, raster_mask, meta = rasterize_world_mesh(
            world_vertices=vertices,
            faces=faces,
            world_to_cam=world_to_cam_np,
            intrinsic=intrinsic_np,
            image_hw=(int(target_size), int(target_size)),
            silhouette_mask=None,
            fill_knn=0,
            vertex_features=cam_normals.astype(np.float32),
            return_vertex_features=True,
            return_raster_mask=True,
        )
        depth_map = depth_map.astype(np.float32)
        point_map = point_map.astype(np.float32)
        normal_map = normal_map.astype(np.float32)
        mask_map = raster_mask.astype(bool)
        extra_meta = splat_points_into_maps(
            depth_map=depth_map,
            point_map=point_map,
            mask_map=mask_map,
            normal_map=normal_map,
            points=extra_points_np,
            normals_world=extra_normals_np,
            world_to_cam=world_to_cam_np,
            intrinsic=intrinsic_np,
            radius=int(extra_splat_radius),
        )
        depths.append(depth_map)
        world_points.append(point_map)
        normal_maps.append(normal_map)
        masks.append(mask_map)
        intrinsics.append(intrinsic_np.astype(np.float32))
        extrinsics.append(world_to_cam_np.astype(np.float32))
        camera_ids.append(camera_id)
        rows.append(
            {
                "view_index": int(view_idx),
                "camera_id": camera_id,
                "rasterized_pixels": int(mask_map.sum()),
                "extra_hairline_splat": extra_meta,
                "meta": meta,
            }
        )
        extra_rows.append({"view_index": int(view_idx), "camera_id": camera_id, **extra_meta})
        prefix = image_dir / f"view_{view_idx:02d}_cam{camera_id}"
        Image.fromarray(mask_map.astype(np.uint8) * 255).save(prefix.with_name(prefix.name + "_mask.png"))
        save_depth_image(prefix.with_name(prefix.name + "_depth.png"), depth_map, mask_map)
        save_normal_image(prefix.with_name(prefix.name + "_normal.png"), normal_map, mask_map)

    depths_np = np.stack(depths, axis=0).astype(np.float32)
    worlds_np = np.stack(world_points, axis=0).astype(np.float32)
    normals_np = np.stack(normal_maps, axis=0).astype(np.float32)
    masks_np = np.stack(masks, axis=0).astype(bool)
    conf_np = masks_np.astype(np.float32)
    intrinsics_np = np.stack(intrinsics, axis=0).astype(np.float32)
    extrinsics_np = np.stack(extrinsics, axis=0).astype(np.float32)
    npz_path = export_dir / "rasterized_surface_targets.npz"
    np.savez_compressed(
        npz_path,
        depths=depths_np,
        depth=depths_np[..., None],
        world_points=worlds_np,
        normals=normals_np,
        normal=normals_np,
        teacher_mask=masks_np,
        world_points_conf=conf_np,
        point_conf=conf_np,
        depth_conf=conf_np,
        normal_conf=conf_np,
        conf=conf_np,
        intrinsic=intrinsics_np,
        extrinsic=extrinsics_np,
        camera_ids=np.asarray(camera_ids),
    )
    summary = {
        "npz_path": npz_path,
        "scene_dir": scene_dir,
        "camera_source": camera_source,
        "target_size": int(target_size),
        "view_indices": view_indices,
        "camera_ids": camera_ids,
        "extra_hairline": {
            "points": int(extra_points_np.shape[0]),
            "splat_radius": int(extra_splat_radius),
            "rows": extra_rows,
        },
        "rows": rows,
        "debug_image_dir": image_dir,
        "note": (
            "These are hard-rasterized debug targets from the raw-image optimized mesh. "
            "They are not a strict-passing teacher unless the external teacher gate and "
            "explicit Open3D visual review pass."
        ),
    }
    (export_dir / "rasterized_surface_targets_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def evaluate_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    view_payloads: list[dict[str, Any]],
    output_dir: Path,
    overlay_limit: int,
) -> tuple[list[dict[str, Any]], list[Path]]:
    rows: list[dict[str, Any]] = []
    overlay_paths: list[Path] = []
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for payload in view_payloads:
        rendered = rasterize_world_mesh(
            world_vertices=vertices,
            faces=faces,
            world_to_cam=payload["world_to_cam_np"],
            intrinsic=payload["intrinsic_np"],
            image_hw=payload["mask"].shape,
            silhouette_mask=None,
            fill_knn=0,
            return_raster_mask=True,
        )[3]
        metrics = compute_mask_metrics(rendered, payload["mask"])
        rows.append({"view_index": payload["view_index"], "camera_id": payload["camera_id"], "metrics": metrics})
        if len(overlay_paths) < overlay_limit:
            overlay_path = overlay_dir / f"view_{payload['view_index']:02d}_cam{payload['camera_id']}_overlay.png"
            Image.fromarray(overlay_masks(payload["rgb"], payload["mask"], rendered)).save(overlay_path)
            overlay_paths.append(overlay_path)
    return rows, overlay_paths


def save_soft_render_debug(
    vertices_t: torch.Tensor,
    faces_t: torch.Tensor,
    face_indices_t: torch.Tensor,
    barycentric_t: torch.Tensor,
    view_payloads: list[dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
) -> list[Path]:
    debug_dir = output_dir / "soft_render_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with torch.no_grad():
        surfels, normals = compute_surfels(vertices_t, faces_t, face_indices_t, barycentric_t)
        if int(args.triangle_render_face_budget) < 0:
            render_face_indices_t = torch.arange(faces_t.shape[0], dtype=torch.long, device=faces_t.device)
        elif int(args.triangle_render_face_budget) > 0:
            count = min(int(args.triangle_render_face_budget), int(faces_t.shape[0]))
            render_face_indices_t = torch.linspace(0, int(faces_t.shape[0]) - 1, count, device=faces_t.device).round().long()
        else:
            render_face_indices_t = torch.unique(face_indices_t)
        for payload in view_payloads[: max(1, int(args.overlay_limit))]:
            if str(args.renderer) == "triangle":
                render = render_soft_triangle_maps(
                    vertices=vertices_t,
                    faces=faces_t,
                    face_indices=render_face_indices_t,
                    world_to_cam=payload["world_to_cam"],
                    intrinsic=payload["intrinsic"],
                    height=int(args.target_size),
                    width=int(args.target_size),
                    pixel_chunk=int(args.render_pixel_chunk),
                    inside_softness=float(args.triangle_inside_softness),
                    face_chunk=int(args.triangle_face_chunk),
                )
            else:
                render = render_soft_surfel_maps(
                    surfels=surfels,
                    normals=normals,
                    world_to_cam=payload["world_to_cam"],
                    intrinsic=payload["intrinsic"],
                    height=int(args.target_size),
                    width=int(args.target_size),
                    sigma=float(args.gaussian_sigma),
                    pixel_chunk=int(args.render_pixel_chunk),
                    depth_softness=float(args.depth_softness),
                )
            mask_np = render["mask"].detach().cpu().numpy()
            hard_mask = mask_np > 0.30
            prefix = debug_dir / f"view_{payload['view_index']:02d}_cam{payload['camera_id']}"
            mask_path = prefix.with_name(prefix.name + "_soft_mask.png")
            depth_path = prefix.with_name(prefix.name + "_depth.png")
            normal_path = prefix.with_name(prefix.name + "_normal.png")
            overlay_path = prefix.with_name(prefix.name + "_soft_overlay.png")
            Image.fromarray((np.clip(mask_np, 0, 1) * 255.0).astype(np.uint8)).save(mask_path)
            save_depth_image(depth_path, render["depth"].detach().cpu().numpy(), hard_mask)
            save_normal_image(normal_path, render["normal"].detach().cpu().numpy(), hard_mask)
            Image.fromarray(overlay_masks(payload["rgb"], payload["mask"], hard_mask)).save(overlay_path)
            paths.append(overlay_path)
    return paths


def main() -> int:
    args = parse_args()
    scene_dir = args.scene_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} already exists and is not empty. Use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = recover_legacy_crop_source_sizes(scene_dir, load_scene_manifest(scene_dir))
    dataset_root = Path(args.dataset_root or manifest["dataset_root"]).expanduser()
    smplx_model_dir = resolve_smplx_model_dir(None if args.smplx_model_dir is None else str(args.smplx_model_dir))
    if smplx_model_dir is None:
        raise FileNotFoundError("Could not resolve SMPL-X model dir; pass --smplx-model-dir.")
    model_path = resolve_smplx_model_path(smplx_model_dir, args.smplx_gender)
    smplx_params, _ = load_optional_annotation_payload(manifest, dataset_root, args.subset_name)
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
    static_features = build_smplx_vertex_features(
        model_path=model_path,
        betas=smplx_params["betas"],
        expression=smplx_params.get("expression"),
    )

    base_vertices_np = np.asarray(mesh["vertices"], dtype=np.float32)
    faces_np = np.asarray(mesh["faces"], dtype=np.int32)
    normals_np = compute_vertex_normals(base_vertices_np, faces_np).astype(np.float32)
    vertex_parts_np = classify_vertex_parts(np.asarray(static_features["canonical_positions"], dtype=np.float32))
    face_landmark_vertex_mask_np = vertex_parts_np == 3
    connected_template_summary: dict[str, Any] | None = None
    hair_outer_vertex_ids_np = np.zeros((0,), dtype=np.int64)
    if args.connected_template_payload:
        template_payload = args.connected_template_payload.expanduser().resolve()
        with np.load(template_payload, allow_pickle=False) as payload:
            template_base_count = int(np.asarray(payload["vertices"]).shape[0])
            base_vertices_np = np.asarray(payload["hybrid_vertices"], dtype=np.float32)
            faces_np = np.asarray(payload["hybrid_faces"], dtype=np.int32)
            base_part_ids = np.asarray(payload["part_ids"], dtype=np.int64)
            hair_ring_count = int(np.asarray(payload["hair_ring_vertex_ids"], dtype=np.int64).shape[0])
            if base_part_ids.shape[0] > base_vertices_np.shape[0]:
                raise ValueError(
                    f"Template part ids ({base_part_ids.shape[0]}) exceed vertex count ({base_vertices_np.shape[0]})."
                )
            vertex_parts_np = np.full((base_vertices_np.shape[0],), 4, dtype=np.int64)
            vertex_parts_np[: base_part_ids.shape[0]] = base_part_ids
            hair_outer_vertex_ids_np = np.arange(
                template_base_count + hair_ring_count,
                template_base_count + 2 * hair_ring_count,
                dtype=np.int64,
            )
            if "face_front_vertex_mask" in payload.files:
                template_face_mask = np.asarray(payload["face_front_vertex_mask"], dtype=bool)
                face_landmark_vertex_mask_np = np.zeros((base_vertices_np.shape[0],), dtype=bool)
                copy_count = min(template_face_mask.shape[0], face_landmark_vertex_mask_np.shape[0])
                face_landmark_vertex_mask_np[:copy_count] = template_face_mask[:copy_count]
            else:
                face_landmark_vertex_mask_np = vertex_parts_np == 3
        normals_np = compute_vertex_normals(base_vertices_np, faces_np).astype(np.float32)
        connected_template_summary = {
            "payload": template_payload,
            "vertices": int(base_vertices_np.shape[0]),
            "faces": int(faces_np.shape[0]),
            "new_vertices": int(base_vertices_np.shape[0] - mesh["vertices"].shape[0]),
            "hair_outer_ring_vertices": int(hair_outer_vertex_ids_np.shape[0]),
            "note": (
                "Connected raw-surface v2 carrier is used for the local upper-bound smoke. "
                "It is not a teacher and does not permit cloud."
            ),
        }
    if face_landmark_vertex_mask_np.shape[0] != base_vertices_np.shape[0] or int(face_landmark_vertex_mask_np.sum()) < 16:
        face_landmark_vertex_mask_np = vertex_parts_np == 3
    part_limits_np, part_reg_weights_np = make_part_limits(vertex_parts_np, args)
    part_free_limits_np = make_part_free_limits(vertex_parts_np, args)
    edges_np = unique_edges(faces_np)
    image_edge_part_ids = parse_part_id_spec(str(args.image_edge_part_ids))
    image_edge_vertex_mask_np = np.isin(vertex_parts_np, np.asarray(image_edge_part_ids, dtype=np.int64))

    min_part_samples = None
    if bool(args.balanced_part_surfels):
        min_part_samples = {
            0: int(args.min_surfel_torso),
            1: int(args.min_surfel_hand),
            2: int(args.min_surfel_hand),
            3: int(args.min_surfel_head),
            4: int(args.min_surfel_hairline),
            5: int(args.min_surfel_clothing),
        }
    surfel_plan = sample_surface_plan(
        base_vertices=base_vertices_np,
        faces=faces_np,
        vertex_parts=vertex_parts_np,
        sample_count=int(args.surfel_samples),
        seed=int(args.seed),
        min_part_samples=min_part_samples,
    )
    sdf_sample_count = min(int(args.surface_samples_for_sdf), base_vertices_np.shape[0])
    sdf_indices_np = np.linspace(0, base_vertices_np.shape[0] - 1, sdf_sample_count).round().astype(np.int64)

    camera_params, camera_source = resolve_scene_camera_params(manifest, dataset_root, args.subset_name)
    views = list(manifest["exported_views"])
    selected_indices = list(range(0, len(views), max(1, int(args.view_stride))))[: max(1, int(args.max_views))]

    requested_device = str(args.device).strip().lower()
    if requested_device != "cpu" and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    height = width = int(args.target_size)
    view_payloads: list[dict[str, Any]] = []
    face_mp_module = None
    face_detector = None
    face_landmarker_meta: dict[str, Any] = {"requested": bool(args.face_landmarker_task), "available": False}
    face_landmark_overlay_paths: list[Path] = []
    face_landmark_rows: list[dict[str, Any]] = []
    hand_mp_module = None
    hand_detector = None
    hand_landmarker_meta: dict[str, Any] = {"requested": bool(args.hand_landmarker_task), "available": False}
    hand_landmark_overlay_paths: list[Path] = []
    hand_landmark_rows: list[dict[str, Any]] = []
    image_edge_rows: list[dict[str, Any]] = []
    if args.face_landmarker_task is not None:
        face_mp_module, face_detector, face_landmarker_meta = create_face_landmarker(
            args.face_landmarker_task,
            min_confidence=float(args.face_landmark_min_confidence),
        )
        if float(args.face_landmark_weight) > 0.0 and face_detector is None:
            raise RuntimeError(
                "Face landmark loss was requested, but the detector is unavailable: "
                f"{face_landmarker_meta.get('reason')}"
            )
    if args.hand_landmarker_task is not None:
        hand_mp_module, hand_detector, hand_landmarker_meta = create_hand_landmarker(
            args.hand_landmarker_task,
            min_confidence=float(args.hand_landmark_min_confidence),
        )
        if float(args.hand_landmark_weight) > 0.0 and hand_detector is None:
            raise RuntimeError(
                "Hand landmark loss was requested, but the detector is unavailable: "
                f"{hand_landmarker_meta.get('reason')}"
            )
    face_landmark_pad = (
        int(args.face_landmark_pad)
        if int(args.face_landmark_pad) >= 0
        else max(4, int(round(float(height) * 0.08)))
    )
    face_landmark_overlay_dir = output_dir / "face_landmark_overlays"
    if face_detector is not None:
        face_landmark_overlay_dir.mkdir(parents=True, exist_ok=True)
    hand_landmark_overlay_dir = output_dir / "hand_landmark_overlays"
    if hand_detector is not None:
        hand_landmark_overlay_dir.mkdir(parents=True, exist_ok=True)
    for view_idx in selected_indices:
        view = views[view_idx]
        camera_id = str(view["camera_id"]).zfill(2)
        scene = load_scene_view(scene_dir, view_idx, (height, width))
        image_path = resolve_scene_path(scene_dir, view["image_path"])
        mask_path = resolve_scene_path(scene_dir, view["mask_path"])
        rgb_np = normalize_rgb(scene.rgb)
        mask_np = np.asarray(scene.mask, dtype=bool)
        intrinsic_np = align_intrinsics_for_loaded_scene_view(
            np.asarray(camera_params[camera_id]["intrinsic"], dtype=np.float32),
            view,
            target_size=height,
        )
        world_to_cam_np = homogeneous(np.asarray(camera_params[camera_id]["world_to_cam"], dtype=np.float32))
        boundary_np = boundary_points(mask_np, args.boundary_samples)
        part_targets_np = coarse_part_target_masks(
            mask_np,
            head_frac=float(args.head_target_frac),
            hairline_frac=float(args.hairline_target_frac),
            hand_side_frac=float(args.hand_side_target_frac),
        )
        image_edge_sdf_np, image_edge_meta = image_edge_distance(
            rgb_np,
            mask_np,
            low=float(args.image_edge_canny_low),
            high=float(args.image_edge_canny_high),
            mask_dilate=int(args.image_edge_mask_dilate),
        )
        image_edge_meta.update({"view_index": int(view_idx), "camera_id": camera_id})
        image_edge_rows.append(json_ready(image_edge_meta))
        face_landmarks_np: np.ndarray | None = None
        face_meta: dict[str, Any] = {"detected": False, "reason": "face_landmarker_disabled"}
        if face_detector is not None and face_mp_module is not None:
            face_landmarks_np, face_meta, face_image = detect_face_landmarks_2d(
                mp_module=face_mp_module,
                detector=face_detector,
                image_path=image_path,
                mask_path=mask_path,
                target_size=height,
                pad=face_landmark_pad,
            )
            if face_landmarks_np is not None and len(face_landmark_overlay_paths) < int(args.overlay_limit):
                overlay_path = face_landmark_overlay_dir / f"view_{view_idx:02d}_cam{camera_id}_landmarks.png"
                save_face_landmark_overlay(face_image, face_landmarks_np, overlay_path)
                face_landmark_overlay_paths.append(overlay_path)
        face_meta.update({"view_index": int(view_idx), "camera_id": camera_id})
        face_landmark_rows.append(json_ready(face_meta))
        hand_landmarks_np: list[np.ndarray] = []
        hand_meta: dict[str, Any] = {"detected": False, "reason": "hand_landmarker_disabled", "hands": 0, "landmarks": 0}
        if hand_detector is not None and hand_mp_module is not None:
            hand_landmarks_np, hand_meta, hand_image = detect_hand_landmarks_2d(
                mp_module=hand_mp_module,
                detector=hand_detector,
                image_path=image_path,
                mask_path=mask_path,
                target_size=height,
            )
            if hand_landmarks_np and len(hand_landmark_overlay_paths) < int(args.overlay_limit):
                overlay_path = hand_landmark_overlay_dir / f"view_{view_idx:02d}_cam{camera_id}_hands.png"
                save_hand_landmark_overlay(hand_image, hand_landmarks_np, overlay_path)
                hand_landmark_overlay_paths.append(overlay_path)
        hand_meta.update({"view_index": int(view_idx), "camera_id": camera_id})
        hand_landmark_rows.append(json_ready(hand_meta))
        view_payloads.append(
            {
                "view_index": int(view_idx),
                "camera_id": camera_id,
                "image_path": image_path,
                "mask_path": mask_path,
                "rgb": rgb_np,
                "mask": mask_np,
                "rgb_t": torch.from_numpy(rgb_np).permute(2, 0, 1)[None].to(device=device),
                "mask_t": torch.from_numpy(mask_np.astype(np.float32))[None, None].to(device=device),
                "part_targets": part_targets_np,
                "part_target_tensors": {
                    name: torch.from_numpy(value.astype(np.float32))[None, None].to(device=device)
                    for name, value in part_targets_np.items()
                },
                "sdf": torch.from_numpy(mask_sdf(mask_np))[None, None].to(device=device),
                "image_edge_sdf": torch.from_numpy(image_edge_sdf_np)[None, None].to(device=device),
                "image_edge_meta": image_edge_meta,
                "boundary": torch.from_numpy(boundary_np).to(device=device),
                "intrinsic": torch.from_numpy(intrinsic_np).to(device=device),
                "world_to_cam": torch.from_numpy(world_to_cam_np).to(device=device),
                "intrinsic_np": intrinsic_np,
                "world_to_cam_np": world_to_cam_np,
                "face_landmarks": face_landmarks_np,
                "face_landmarks_t": (
                    None
                    if face_landmarks_np is None
                    else torch.from_numpy(face_landmarks_np.astype(np.float32)).to(device=device)
                ),
                "face_landmark_meta": face_meta,
                "hand_landmarks": hand_landmarks_np,
                "hand_landmarks_t": [
                    torch.from_numpy(item.astype(np.float32)).to(device=device) for item in hand_landmarks_np
                ],
                "hand_landmark_meta": hand_meta,
            }
        )
    if face_detector is not None and hasattr(face_detector, "close"):
        face_detector.close()
    if hand_detector is not None and hasattr(hand_detector, "close"):
        hand_detector.close()

    base_vertices = torch.from_numpy(base_vertices_np).to(device=device)
    base_normals = torch.from_numpy(normals_np).to(device=device)
    faces_t = torch.from_numpy(faces_np.astype(np.int64)).to(device=device)
    face_indices_t = torch.from_numpy(surfel_plan["face_indices"]).to(device=device)
    render_face_indices_np = choose_triangle_render_faces(
        faces_np,
        surfel_plan["face_indices"],
        budget=int(args.triangle_render_face_budget),
    )
    render_face_indices_t = torch.from_numpy(render_face_indices_np.astype(np.int64)).to(device=device)
    barycentric_t = torch.from_numpy(surfel_plan["barycentric"]).to(device=device)
    surfel_part_ids_t = torch.from_numpy(surfel_plan["part_ids"]).to(device=device)
    sdf_indices_t = torch.from_numpy(sdf_indices_np).to(device=device)
    part_limits_t = torch.from_numpy(part_limits_np).to(device=device)
    part_reg_weights_t = torch.from_numpy(part_reg_weights_np).to(device=device)
    part_free_limits_t = torch.from_numpy(part_free_limits_np).to(device=device)[:, None]
    part_free_vertex_mask_t = (part_free_limits_t.reshape(-1) > 0.0).float()
    edges_t = torch.from_numpy(edges_np).to(device=device)
    center = torch.from_numpy(base_vertices_np.mean(axis=0, keepdims=True).astype(np.float32)).to(device=device)
    hairline_vertex_mask_t = torch.from_numpy((vertex_parts_np == 4).astype(np.float32))[:, None].to(device=device)
    hair_outer_vertex_ids_t = torch.from_numpy(hair_outer_vertex_ids_np.astype(np.int64)).to(device=device)
    face_landmark_vertex_ids_t = torch.from_numpy(np.nonzero(face_landmark_vertex_mask_np)[0].astype(np.int64)).to(device=device)
    left_hand_vertex_ids_t = torch.from_numpy(np.nonzero(vertex_parts_np == 1)[0].astype(np.int64)).to(device=device)
    right_hand_vertex_ids_t = torch.from_numpy(np.nonzero(vertex_parts_np == 2)[0].astype(np.int64)).to(device=device)
    image_edge_vertex_ids_t = torch.from_numpy(np.nonzero(image_edge_vertex_mask_np)[0].astype(np.int64)).to(device=device)

    delta_t = torch.zeros(3, device=device, requires_grad=True)
    log_scale = torch.zeros(1, device=device, requires_grad=True)
    normal_offsets = torch.zeros(base_vertices_np.shape[0], device=device, requires_grad=True)
    hairline_free_offsets = torch.zeros((base_vertices_np.shape[0], 3), device=device, requires_grad=True)
    part_free_offsets = torch.zeros((base_vertices_np.shape[0], 3), device=device, requires_grad=True)
    optimizer_params = [normal_offsets] if bool(args.freeze_global_transform) else [delta_t, log_scale, normal_offsets]
    if float(args.hairline_free_offset_limit) > 0.0:
        optimizer_params.append(hairline_free_offsets)
    if float(part_free_limits_np.max(initial=0.0)) > 0.0:
        optimizer_params.append(part_free_offsets)
    optimizer = torch.optim.Adam(optimizer_params, lr=float(args.lr))

    history: list[dict[str, Any]] = []
    for step in range(int(args.steps)):
        optimizer.zero_grad(set_to_none=True)
        bounded_offsets = torch.tanh(normal_offsets) * part_limits_t
        hairline_free = torch.tanh(hairline_free_offsets) * float(args.hairline_free_offset_limit)
        hairline_free = hairline_free * hairline_vertex_mask_t
        part_free = torch.tanh(part_free_offsets) * part_free_limits_t
        if bool(args.freeze_global_transform):
            vertices = base_vertices
        else:
            vertices = center + torch.exp(log_scale).clamp(0.85, 1.15) * (base_vertices - center) + delta_t[None, :]
        vertices = vertices + base_normals * bounded_offsets[:, None]
        vertices = vertices + hairline_free
        vertices = vertices + part_free
        surfels, surfel_normals = compute_surfels(vertices, faces_t, face_indices_t, barycentric_t)

        mask_losses = []
        recall_losses = []
        outside_losses = []
        boundary_losses = []
        visibility_depths = []
        sampled_vertices = vertices.index_select(0, sdf_indices_t)
        for payload in view_payloads:
            if str(args.renderer) == "triangle":
                render = render_soft_triangle_maps(
                    vertices=vertices,
                    faces=faces_t,
                    face_indices=render_face_indices_t,
                    world_to_cam=payload["world_to_cam"],
                    intrinsic=payload["intrinsic"],
                    height=height,
                    width=width,
                    pixel_chunk=int(args.render_pixel_chunk),
                    inside_softness=float(args.triangle_inside_softness),
                    face_chunk=int(args.triangle_face_chunk),
                )
            else:
                render = render_soft_surfel_maps(
                    surfels=surfels,
                    normals=surfel_normals,
                    world_to_cam=payload["world_to_cam"],
                    intrinsic=payload["intrinsic"],
                    height=height,
                    width=width,
                    sigma=float(args.gaussian_sigma),
                    pixel_chunk=int(args.render_pixel_chunk),
                    depth_softness=float(args.depth_softness),
                )
            visibility_depths.append(render["depth"])
            rendered_mask = render["mask"].clamp(1e-4, 1.0 - 1e-4)
            target_mask = payload["mask_t"].reshape(height, width)
            mask_losses.append(F.binary_cross_entropy(rendered_mask, target_mask))
            target_area = target_mask.sum().clamp_min(1.0)
            recall_losses.append((target_mask * (1.0 - rendered_mask)).sum() / target_area)

            uv, z, _ = project_points(sampled_vertices, payload["world_to_cam"], payload["intrinsic"])
            sdf_values = sample_sdf(payload["sdf"], uv, height, width)
            in_front = z > 1e-5
            outside_losses.append(torch.relu(sdf_values[in_front]).mean() if in_front.any() else sdf_values.mean() * 0.0)

            boundary = payload["boundary"]
            if boundary.numel() > 0:
                uv_surfel, z_surfel, _ = project_points(surfels, payload["world_to_cam"], payload["intrinsic"])
                valid = (
                    (z_surfel > 1e-5)
                    & (uv_surfel[:, 0] >= 0.0)
                    & (uv_surfel[:, 0] <= width - 1)
                    & (uv_surfel[:, 1] >= 0.0)
                    & (uv_surfel[:, 1] <= height - 1)
                )
                uv_valid = uv_surfel[valid]
                if uv_valid.shape[0] > 0:
                    uv_norm = uv_valid / float(max(height, width))
                    boundary_norm = boundary / float(max(height, width))
                    dists = torch.cdist(boundary_norm, uv_norm)
                    boundary_losses.append(dists.min(dim=1).values.mean())

        mask_loss = torch.stack(mask_losses).mean() if mask_losses else torch.zeros((), device=device)
        recall_loss = torch.stack(recall_losses).mean() if recall_losses else torch.zeros((), device=device)
        outside_loss = torch.stack(outside_losses).mean() if outside_losses else torch.zeros((), device=device)
        boundary_loss = torch.stack(boundary_losses).mean() if boundary_losses else torch.zeros((), device=device)
        photo_loss, photo_meta = photometric_consistency_loss(
            surfels,
            view_payloads,
            height,
            width,
            visibility_depths=visibility_depths,
            depth_tolerance=float(args.photo_depth_tolerance),
        )
        part_loss, part_meta = part_recall_loss(
            surfels=surfels,
            normals=surfel_normals,
            surfel_part_ids=surfel_part_ids_t,
            view_payloads=view_payloads,
            height=height,
            width=width,
            sigma=float(args.gaussian_sigma),
            pixel_chunk=int(args.render_pixel_chunk),
            depth_softness=float(args.depth_softness),
            min_pixels=int(args.part_target_min_pixels),
        )
        hair_boundary_loss, hair_boundary_meta = vertex_silhouette_boundary_loss(
            vertices=vertices,
            vertex_ids=hair_outer_vertex_ids_t,
            view_payloads=view_payloads,
            height=height,
            width=width,
        )
        face_landmark_loss, face_landmark_meta = face_landmark_projection_loss(
            vertices=vertices,
            vertex_ids=face_landmark_vertex_ids_t,
            view_payloads=view_payloads,
            height=height,
            width=width,
            bidirectional_weight=float(args.face_landmark_bidir_weight),
            min_points=int(args.face_landmark_min_points),
        )
        hand_landmark_loss, hand_landmark_meta = hand_landmark_projection_loss(
            vertices=vertices,
            left_vertex_ids=left_hand_vertex_ids_t,
            right_vertex_ids=right_hand_vertex_ids_t,
            view_payloads=view_payloads,
            height=height,
            width=width,
            bidirectional_weight=float(args.hand_landmark_bidir_weight),
            min_points=int(args.hand_landmark_min_points),
        )
        image_edge_loss, image_edge_meta = vertex_image_edge_loss(
            vertices=vertices,
            vertex_ids=image_edge_vertex_ids_t,
            view_payloads=view_payloads,
            height=height,
            width=width,
            max_distance=float(args.image_edge_max_distance),
        )

        if bool(args.freeze_global_transform):
            global_reg = delta_t.sum() * 0.0 + log_scale.sum() * 0.0
        else:
            global_reg = float(args.translation_reg) * delta_t.square().sum() + float(args.scale_reg) * log_scale.square().sum()
        offset_values = bounded_offsets
        offset_reg = (part_reg_weights_t * offset_values.square()).mean()
        smooth_reg = (offset_values[edges_t[:, 0]] - offset_values[edges_t[:, 1]]).square().mean()
        hairline_free_reg = hairline_free.square().sum(dim=1)
        hairline_count = hairline_vertex_mask_t.reshape(-1).sum().clamp_min(1.0)
        hairline_free_reg = (hairline_free_reg * hairline_vertex_mask_t.reshape(-1)).sum() / hairline_count
        hairline_free_smooth = (hairline_free[edges_t[:, 0]] - hairline_free[edges_t[:, 1]]).square().sum(dim=1).mean()
        part_free_count = part_free_vertex_mask_t.sum().clamp_min(1.0)
        part_free_reg = (part_free.square().sum(dim=1) * part_free_vertex_mask_t).sum() / part_free_count
        edge_free_mask = ((part_free_limits_t[edges_t[:, 0], 0] > 0.0) | (part_free_limits_t[edges_t[:, 1], 0] > 0.0)).float()
        edge_free_count = edge_free_mask.sum().clamp_min(1.0)
        part_free_smooth = (
            (part_free[edges_t[:, 0]] - part_free[edges_t[:, 1]]).square().sum(dim=1) * edge_free_mask
        ).sum() / edge_free_count
        loss = (
            float(args.mask_weight) * mask_loss
            + float(args.recall_weight) * recall_loss
            + float(args.outside_weight) * outside_loss
            + float(args.boundary_weight) * boundary_loss
            + float(args.part_recall_weight) * part_loss
            + float(args.hair_boundary_weight) * hair_boundary_loss
            + float(args.face_landmark_weight) * face_landmark_loss
            + float(args.hand_landmark_weight) * hand_landmark_loss
            + float(args.image_edge_weight) * image_edge_loss
            + float(args.photo_weight) * photo_loss
            + global_reg
            + float(args.offset_reg) * offset_reg
            + float(args.offset_smooth_reg) * smooth_reg
            + float(args.hairline_free_offset_reg) * hairline_free_reg
            + float(args.hairline_free_smooth_reg) * hairline_free_smooth
            + float(args.part_free_offset_reg) * part_free_reg
            + float(args.part_free_smooth_reg) * part_free_smooth
        )
        loss.backward()
        optimizer.step()

        if step == 0 or step == int(args.steps) - 1 or (step + 1) % max(1, int(args.steps) // 5) == 0:
            history.append(
                {
                    "step": int(step),
                    "loss": float(loss.detach().cpu()),
                    "mask_loss": float(mask_loss.detach().cpu()),
                    "soft_recall_loss": float(recall_loss.detach().cpu()),
                    "outside_loss": float(outside_loss.detach().cpu()),
                    "boundary_loss": float(boundary_loss.detach().cpu()),
                    "part_recall_loss": float(part_loss.detach().cpu()),
                    "part_recall_meta": part_meta,
                    "hair_boundary_loss": float(hair_boundary_loss.detach().cpu()),
                    "hair_boundary_meta": hair_boundary_meta,
                    "face_landmark_loss": float(face_landmark_loss.detach().cpu()),
                    "face_landmark_meta": face_landmark_meta,
                    "hand_landmark_loss": float(hand_landmark_loss.detach().cpu()),
                    "hand_landmark_meta": hand_landmark_meta,
                    "image_edge_loss": float(image_edge_loss.detach().cpu()),
                    "image_edge_meta": image_edge_meta,
                    "photometric_consistency_loss": float(photo_loss.detach().cpu()),
                    "offset_reg": float(offset_reg.detach().cpu()),
                    "offset_smooth_reg": float(smooth_reg.detach().cpu()),
                    "hairline_free_offset_reg": float(hairline_free_reg.detach().cpu()),
                    "hairline_free_smooth_reg": float(hairline_free_smooth.detach().cpu()),
                    "part_free_offset_reg": float(part_free_reg.detach().cpu()),
                    "part_free_smooth_reg": float(part_free_smooth.detach().cpu()),
                    "photo_valid_surfels": photo_meta["valid_surfels"],
                    "photo_mean_support": photo_meta["mean_support"],
                    "translation": [float(v) for v in delta_t.detach().cpu().numpy().reshape(-1)],
                    "scale": float(torch.exp(log_scale.detach()).cpu().item()),
                }
            )

    with torch.no_grad():
        final_offsets = torch.tanh(normal_offsets) * part_limits_t
        final_hairline_free = torch.tanh(hairline_free_offsets) * float(args.hairline_free_offset_limit)
        final_hairline_free = final_hairline_free * hairline_vertex_mask_t
        final_part_free = torch.tanh(part_free_offsets) * part_free_limits_t
        if bool(args.freeze_global_transform):
            optimized = base_vertices
        else:
            optimized = center + torch.exp(log_scale).clamp(0.85, 1.15) * (base_vertices - center) + delta_t[None, :]
        optimized = optimized + base_normals * final_offsets[:, None]
        optimized = optimized + final_hairline_free
        optimized = optimized + final_part_free
        optimized_np = optimized.detach().cpu().numpy().astype(np.float32)
        final_offsets_np = final_offsets.detach().cpu().numpy().astype(np.float32)
        final_hairline_free_np = final_hairline_free.detach().cpu().numpy().astype(np.float32)
        final_part_free_np = final_part_free.detach().cpu().numpy().astype(np.float32)

    initial_rows, initial_overlays = evaluate_mesh(base_vertices_np, faces_np, view_payloads, output_dir / "initial", args.overlay_limit)
    optimized_rows, optimized_overlays = evaluate_mesh(optimized_np, faces_np, view_payloads, output_dir / "optimized", args.overlay_limit)
    save_contact_sheet(initial_overlays, output_dir / "initial_overlay_contact_sheet.png")
    save_contact_sheet(optimized_overlays, output_dir / "optimized_overlay_contact_sheet.png")
    save_ply(output_dir / "optimized_softsurfel_surface_mesh.ply", optimized_np, faces_np)

    optimized_t = torch.from_numpy(optimized_np).to(device=device)
    soft_overlay_paths = save_soft_render_debug(
        vertices_t=optimized_t,
        faces_t=faces_t,
        face_indices_t=face_indices_t,
        barycentric_t=barycentric_t,
        view_payloads=view_payloads,
        args=args,
        output_dir=output_dir,
    )
    save_contact_sheet(soft_overlay_paths, output_dir / "soft_render_overlay_contact_sheet.png")

    extra_hairline = build_extra_hairline_surfels(
        vertices=optimized_np,
        faces=faces_np,
        view_payloads=view_payloads,
        max_points=int(args.extra_hairline_surfels),
        max_per_view=int(args.extra_hairline_max_per_view),
        min_support=int(args.extra_hairline_min_support),
        top_frac=float(args.extra_hairline_top_frac),
        nearest_radius=float(args.extra_hairline_nearest_radius),
        depth_offset=float(args.extra_hairline_depth_offset),
        voxel=float(args.extra_hairline_voxel),
        output_dir=output_dir,
    )
    extra_hairline_summary = extra_hairline["summary"]

    initial_iou = summarize([row["metrics"]["iou"] for row in initial_rows])
    optimized_iou = summarize([row["metrics"]["iou"] for row in optimized_rows])
    initial_recall = summarize([row["metrics"]["target_recall"] for row in initial_rows])
    optimized_recall = summarize([row["metrics"]["target_recall"] for row in optimized_rows])
    iou_delta = (
        float(optimized_iou["mean"] - initial_iou["mean"])
        if optimized_iou["mean"] is not None and initial_iou["mean"] is not None
        else None
    )
    recall_delta = (
        float(optimized_recall["mean"] - initial_recall["mean"])
        if optimized_recall["mean"] is not None and initial_recall["mean"] is not None
        else None
    )

    part_stats = {}
    for part_id, part_name in PART_NAMES.items():
        mask = vertex_parts_np == part_id
        values = final_offsets_np[mask]
        part_stats[part_name] = {
            "vertices": int(mask.sum()),
            "mean_abs_offset": float(np.mean(np.abs(values))) if values.size else 0.0,
            "p90_abs_offset": float(np.percentile(np.abs(values), 90)) if values.size else 0.0,
            "limit": float(np.max(part_limits_np[mask])) if mask.any() else 0.0,
        }
    hairline_free_norm = np.linalg.norm(final_hairline_free_np, axis=1)
    hairline_mask_np = vertex_parts_np == 4
    hairline_free_stats = {
        "enabled": bool(float(args.hairline_free_offset_limit) > 0.0),
        "limit": float(args.hairline_free_offset_limit),
        "vertices": int(hairline_mask_np.sum()),
        "mean_norm": float(hairline_free_norm[hairline_mask_np].mean()) if hairline_mask_np.any() else 0.0,
        "p90_norm": float(np.percentile(hairline_free_norm[hairline_mask_np], 90)) if hairline_mask_np.any() else 0.0,
        "max_norm": float(hairline_free_norm[hairline_mask_np].max()) if hairline_mask_np.any() else 0.0,
        "note": (
            "This is connected head-top mesh residual only. It is not a hair teacher and must pass "
            "the same strict visual gates before being useful."
        ),
    }
    part_free_norm = np.linalg.norm(final_part_free_np, axis=1)
    part_free_stats: dict[str, Any] = {
        "enabled": bool(float(part_free_limits_np.max(initial=0.0)) > 0.0),
        "limits": {
            "face": float(args.part_free_offset_limit_face),
            "hands": float(args.part_free_offset_limit_hands),
            "hairline": float(args.part_free_offset_limit_hairline),
            "clothing": float(args.part_free_offset_limit_clothing),
        },
        "offset_reg": float(args.part_free_offset_reg),
        "smooth_reg": float(args.part_free_smooth_reg),
        "note": (
            "Part-free offsets are bounded connected 3D residuals. They increase surface capacity "
            "without adding floating face/hand/hair patches and still require strict visual gates."
        ),
    }
    for part_id, part_name in PART_NAMES.items():
        mask = (vertex_parts_np == int(part_id)) & (part_free_limits_np > 0.0)
        part_free_stats[part_name] = {
            "enabled_vertices": int(mask.sum()),
            "limit": float(part_free_limits_np[mask].max()) if mask.any() else 0.0,
            "mean_norm": float(part_free_norm[mask].mean()) if mask.any() else 0.0,
            "p90_norm": float(np.percentile(part_free_norm[mask], 90)) if mask.any() else 0.0,
            "max_norm": float(part_free_norm[mask].max()) if mask.any() else 0.0,
        }
    part_target_stats: dict[str, Any] = {}
    for name in ("head_upper", "hairline_top", "hands_side"):
        counts = [
            int(np.asarray(payload.get("part_targets", {}).get(name, np.zeros_like(payload["mask"]))).sum())
            for payload in view_payloads
        ]
        part_target_stats[name] = summarize([float(v) for v in counts])
    surfel_part_stats = {}
    surfel_part_ids_np = np.asarray(surfel_plan["part_ids"], dtype=np.int64)
    for part_id, part_name in PART_NAMES.items():
        surfel_part_stats[part_name] = int((surfel_part_ids_np == int(part_id)).sum())
    face_landmark_detected = [row for row in face_landmark_rows if bool(row.get("detected"))]
    face_landmark_stats = {
        "detector": face_landmarker_meta,
        "weight": float(args.face_landmark_weight),
        "bidirectional_weight": float(args.face_landmark_bidir_weight),
        "pad": int(face_landmark_pad),
        "face_vertex_count": int(face_landmark_vertex_mask_np.sum()),
        "detected_views": int(len(face_landmark_detected)),
        "selected_views": int(len(view_payloads)),
        "mean_landmarks": (
            float(np.mean([float(row.get("landmarks", 0)) for row in face_landmark_detected]))
            if face_landmark_detected
            else 0.0
        ),
        "mean_inside_mask_ratio": (
            float(np.mean([float(row.get("inside_mask_ratio", 0.0)) for row in face_landmark_detected]))
            if face_landmark_detected
            else 0.0
        ),
        "rows": face_landmark_rows,
        "overlay_paths": face_landmark_overlay_paths,
        "note": (
            "These landmarks are used only as optional 2D weak constraints on the connected face mesh. "
            "They are not triangulated and do not create a face teacher patch."
        ),
    }
    hand_landmark_detected = [row for row in hand_landmark_rows if bool(row.get("detected"))]
    hand_landmark_stats = {
        "detector": hand_landmarker_meta,
        "weight": float(args.hand_landmark_weight),
        "bidirectional_weight": float(args.hand_landmark_bidir_weight),
        "left_hand_vertex_count": int((vertex_parts_np == 1).sum()),
        "right_hand_vertex_count": int((vertex_parts_np == 2).sum()),
        "detected_views": int(len(hand_landmark_detected)),
        "selected_views": int(len(view_payloads)),
        "detected_hands": int(sum(int(row.get("hands", 0)) for row in hand_landmark_detected)),
        "mean_inside_mask_ratio": (
            float(np.mean([float(row.get("inside_mask_ratio", 0.0)) for row in hand_landmark_detected]))
            if hand_landmark_detected
            else 0.0
        ),
        "rows": hand_landmark_rows,
        "overlay_paths": hand_landmark_overlay_paths,
        "note": (
            "These landmarks are used only as optional 2D weak constraints on connected hand vertices. "
            "They are not triangulated and do not create a hand teacher patch."
        ),
    }
    image_edge_stats = {
        "weight": float(args.image_edge_weight),
        "part_ids": [int(v) for v in image_edge_part_ids],
        "part_names": [PART_NAMES[int(v)] for v in image_edge_part_ids],
        "vertex_count": int(image_edge_vertex_mask_np.sum()),
        "canny_low": float(args.image_edge_canny_low),
        "canny_high": float(args.image_edge_canny_high),
        "mask_dilate": int(args.image_edge_mask_dilate),
        "max_distance": float(args.image_edge_max_distance),
        "mean_edge_pixels": (
            float(np.mean([float(row.get("edge_pixels", 0)) for row in image_edge_rows]))
            if image_edge_rows
            else 0.0
        ),
        "usable_views": int(sum(1 for row in image_edge_rows if int(row.get("edge_pixels", 0)) > 0)),
        "selected_views": int(len(view_payloads)),
        "rows": image_edge_rows,
        "note": (
            "Image-edge loss samples raw RGB Canny edge distance for connected mesh vertices only. "
            "It uses no VGGT depth/point/normal and creates no floating patch or teacher."
        ),
    }

    truthful_status = "raw_softsurfel_surface_smoke_complete_not_teacher_or_candidate"
    export_summary = None
    if args.export_raster_targets:
        export_scene = args.export_scene_dir.expanduser().resolve() if args.export_scene_dir else scene_dir
        export_summary = export_raster_targets(
            vertices=optimized_np,
            faces=faces_np,
            vertex_normals=compute_vertex_normals(optimized_np, faces_np).astype(np.float32),
            extra_points=extra_hairline["points"],
            extra_normals=extra_hairline["normals"],
            extra_splat_radius=int(args.extra_hairline_export_radius),
            scene_dir=export_scene,
            dataset_root=dataset_root,
            subset_name=str(args.subset_name),
            target_size=int(args.export_target_size),
            view_spec=str(args.export_target_views),
            output_dir=output_dir,
        )

    summary = {
        "task": "raw_image_softsurfel_surface_upperbound_v1_smoke",
        "truthful_status": truthful_status,
        "scene_dir": scene_dir,
        "output_dir": output_dir,
        "uses_vggt_depth_point_normal": False,
        "creates_candidate_predictions": False,
        "creates_teacher_targets": bool(args.export_raster_targets),
        "teacher_targets_strict_pass": False,
        "allows_cloud": False,
        "scene": {
            "selected_view_count": len(selected_indices),
            "selected_indices": selected_indices,
            "camera_source": camera_source,
            "target_size": int(args.target_size),
        },
        "global_transform": {
            "freeze_global_transform": bool(args.freeze_global_transform),
            "final_scale": float(torch.exp(log_scale.detach()).cpu().item()),
            "final_translation": [float(v) for v in delta_t.detach().cpu().numpy().reshape(-1)],
        },
        "config": vars(args),
        "part_names": PART_NAMES,
        "connected_template": connected_template_summary,
        "renderer": {
            "mode": str(args.renderer),
            "triangle_inside_softness": float(args.triangle_inside_softness),
            "triangle_render_face_budget": int(args.triangle_render_face_budget),
            "triangle_face_chunk": int(args.triangle_face_chunk),
            "sampled_render_faces": int(render_face_indices_t.numel()),
            "note": (
                "triangle mode is a connected-mesh visibility diagnostic. It is still not a production "
                "rasterizer and does not create a teacher or cloud unblocker."
            ),
        },
        "part_stats": part_stats,
        "surfel_part_stats": surfel_part_stats,
        "part_target_stats": part_target_stats,
        "face_landmarks": face_landmark_stats,
        "hand_landmarks": hand_landmark_stats,
        "image_edges": image_edge_stats,
        "hairline_free_offset": hairline_free_stats,
        "part_free_offset": part_free_stats,
        "extra_hairline": extra_hairline_summary,
        "optimization_history": history,
        "metrics": {
            "initial_iou": initial_iou,
            "optimized_iou": optimized_iou,
            "iou_delta": iou_delta,
            "initial_target_recall": initial_recall,
            "optimized_target_recall": optimized_recall,
            "target_recall_delta": recall_delta,
        },
        "outputs": {
            "optimized_mesh": output_dir / "optimized_softsurfel_surface_mesh.ply",
            "initial_contact_sheet": output_dir / "initial_overlay_contact_sheet.png",
            "optimized_contact_sheet": output_dir / "optimized_overlay_contact_sheet.png",
            "soft_render_contact_sheet": output_dir / "soft_render_overlay_contact_sheet.png",
            "extra_hairline_points": output_dir / "optimized_softsurfel_extra_hairline_points.ply",
            "summary_json": output_dir / "raw_softsurfel_surface_summary.json",
            "report_md": output_dir / "report.md",
            "rasterized_surface_targets": None if export_summary is None else export_summary.get("npz_path"),
        },
        "rasterized_surface_targets": export_summary,
        "current_blocker": (
            f"This is a CPU small-resolution raw surface smoke using the {str(args.renderer)} renderer. "
            "It adds differentiable soft mask rendering, multi-view RGB consistency, and part-aware "
            "residual limits, but it is not a production connected-surface backend, not a "
            "strict-passing teacher, and not a mentor candidate. It must not unblock cloud."
        ),
        "next_required_action": (
            "Scale the renderer carefully, add true visibility/depth ordering and surface-to-view "
            "depth/world/normal target export, then run strict teacher/candidate visual gates. "
            "Do not return to r-candidate threshold/confidence loops."
        ),
    }
    (output_dir / "raw_softsurfel_surface_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report = [
        "# Raw-Image Soft Surfel Surface Upper-Bound v1 Smoke",
        "",
        f"Status: `{truthful_status}`",
        "",
        f"- selected views: `{len(selected_indices)}`",
        f"- target size: `{int(args.target_size)}`",
        f"- surfel samples: `{int(args.surfel_samples)}`",
        f"- uses VGGT depth/point/normal: `False`",
        f"- creates teacher targets: `{bool(args.export_raster_targets)}`",
        f"- creates candidate predictions: `False`",
        f"- freeze global transform: `{bool(args.freeze_global_transform)}`",
        f"- renderer: `{str(args.renderer)}`",
        f"- sampled render faces: `{int(render_face_indices_t.numel())}`",
        f"- hairline free-offset enabled: `{hairline_free_stats['enabled']}`",
        f"- hairline free-offset p90: `{hairline_free_stats['p90_norm']}`",
        f"- balanced part surfels: `{bool(args.balanced_part_surfels)}`",
        f"- surfel part stats: `{surfel_part_stats}`",
        f"- part recall weight: `{float(args.part_recall_weight)}`",
        f"- part target stats: `{json_ready(part_target_stats)}`",
        f"- hair boundary weight: `{float(args.hair_boundary_weight)}`",
        f"- face landmark weight: `{float(args.face_landmark_weight)}`",
        f"- face landmark detected views: `{face_landmark_stats['detected_views']}/{face_landmark_stats['selected_views']}`",
        f"- face landmark vertex count: `{face_landmark_stats['face_vertex_count']}`",
        f"- hand landmark weight: `{float(args.hand_landmark_weight)}`",
        f"- hand landmark detected views: `{hand_landmark_stats['detected_views']}/{hand_landmark_stats['selected_views']}`",
        f"- hand landmark detected hands: `{hand_landmark_stats['detected_hands']}`",
        f"- image edge weight: `{float(args.image_edge_weight)}`",
        f"- image edge part ids: `{image_edge_stats['part_ids']}`",
        f"- image edge usable views: `{image_edge_stats['usable_views']}/{image_edge_stats['selected_views']}`",
        f"- image edge mean pixels: `{image_edge_stats['mean_edge_pixels']}`",
        f"- part-free offsets enabled: `{part_free_stats['enabled']}`",
        f"- part-free limits: `{json_ready(part_free_stats['limits'])}`",
        f"- extra hairline surfels: `{extra_hairline_summary.get('kept_points', 0)}`",
        f"- initial mean IoU: `{initial_iou['mean']}`",
        f"- optimized mean IoU: `{optimized_iou['mean']}`",
        f"- IoU delta: `{iou_delta}`",
        f"- initial target recall: `{initial_recall['mean']}`",
        f"- optimized target recall: `{optimized_recall['mean']}`",
        f"- target recall delta: `{recall_delta}`",
        "",
        "This is the first raw-image v1 soft-surface smoke: it adds a pure-Torch CPU soft",
        "surfel renderer, multi-view photometric consistency, and part-aware residual limits.",
        "It is still not a full human surface backend, not a strict teacher, and not a cloud",
        "unblocker.",
        "",
        "Current blocker:",
        "",
        str(summary["current_blocker"]),
        "",
        "Next required action:",
        "",
        str(summary["next_required_action"]),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(json_ready({k: summary[k] for k in ("truthful_status", "metrics", "outputs")}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
