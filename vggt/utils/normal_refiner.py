from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


COARSE_NORMAL_CHANNELS = ("smplx_cam_nx", "smplx_cam_ny", "smplx_cam_nz")
COARSE_VISIBLE_MASK_CHANNEL = "smplx_visible_mask"


def preprocess_rgb_image(path: str | Path, target_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    if width >= height:
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
    else:
        new_height = target_size
        new_width = round(width * (new_height / height) / 14) * 14

    new_width = max(14, int(new_width))
    new_height = max(14, int(new_height))
    image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    pad_left = (target_size - new_width) // 2
    pad_top = (target_size - new_height) // 2
    canvas.paste(image, (pad_left, pad_top))
    return np.asarray(canvas, dtype=np.uint8)


def preprocess_mask_image(path: str | Path, target_size: int) -> np.ndarray:
    image = Image.open(path).convert("L")
    width, height = image.size
    if width >= height:
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
    else:
        new_height = target_size
        new_width = round(width * (new_height / height) / 14) * 14

    new_width = max(14, int(new_width))
    new_height = max(14, int(new_height))
    image = image.resize((new_width, new_height), Image.Resampling.NEAREST)
    canvas = Image.new("L", (target_size, target_size), 0)
    pad_left = (target_size - new_width) // 2
    pad_top = (target_size - new_height) // 2
    canvas.paste(image, (pad_left, pad_top))
    return (np.asarray(canvas, dtype=np.uint8) > 127).astype(bool)


def normal_to_rgb(normals: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float32)
    rgb = np.clip((normals + 1.0) * 0.5, 0.0, 1.0)
    rgb = (rgb * 255.0).astype(np.uint8)
    if mask is not None:
        rgb = rgb.copy()
        rgb[~np.asarray(mask, dtype=bool)] = 255
    return rgb


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def clamp_box(
    box: tuple[int, int, int, int],
    image_hw: tuple[int, int],
) -> tuple[int, int, int, int]:
    height, width = int(image_hw[0]), int(image_hw[1])
    x0, y0, x1, y1 = box
    x0 = max(0, min(width, x0))
    y0 = max(0, min(height, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    return x0, y0, x1, y1


def expand_box(
    box: tuple[int, int, int, int],
    image_hw: tuple[int, int],
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    pad_x: int = 0,
    pad_y: int = 0,
    min_size: int = 16,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    half_w = max(min_size / 2.0, 0.5 * (x1 - x0) * scale_x + pad_x)
    half_h = max(min_size / 2.0, 0.5 * (y1 - y0) * scale_y + pad_y)
    expanded = (
        int(round(cx - half_w)),
        int(round(cy - half_h)),
        int(round(cx + half_w)),
        int(round(cy + half_h)),
    )
    return clamp_box(expanded, image_hw)


def head_box_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    bbox = mask_bbox(mask)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    body_h = y1 - y0
    head_h = max(24, int(round(body_h * 0.45)))
    raw = (x0, y0, x1, min(y1, y0 + head_h))
    return expand_box(
        raw,
        mask.shape,
        scale_x=1.15,
        scale_y=1.08,
        pad_x=max(4, int(round((x1 - x0) * 0.03))),
        pad_y=max(4, int(round(body_h * 0.02))),
    )


def face_box_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    head_box = head_box_from_mask(mask)
    if head_box is None:
        return None
    x0, y0, x1, y1 = head_box
    w = x1 - x0
    h = y1 - y0
    face_w = max(24, int(round(w * 0.62)))
    face_h = max(24, int(round(h * 0.62)))
    cx = int(round(0.5 * (x0 + x1)))
    cy = y0 + int(round(h * 0.42))
    raw = (
        cx - face_w // 2,
        cy - face_h // 2,
        cx + face_w // 2,
        cy + face_h // 2,
    )
    return clamp_box(raw, mask.shape)


def shoulder_box_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    bbox = mask_bbox(mask)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    body_h = y1 - y0
    shoulder_y0 = y0 + int(round(body_h * 0.20))
    shoulder_y1 = y0 + int(round(body_h * 0.60))
    raw = (x0, shoulder_y0, x1, min(y1, shoulder_y1))
    return expand_box(
        raw,
        mask.shape,
        scale_x=1.18,
        scale_y=1.05,
        pad_x=max(6, int(round((x1 - x0) * 0.05))),
        pad_y=max(4, int(round(body_h * 0.01))),
    )


def crop_array(array: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    if array.ndim == 2:
        return array[y0:y1, x0:x1]
    return array[y0:y1, x0:x1, ...]


def resize_array(array: np.ndarray, target_size: int, is_mask: bool = False) -> np.ndarray:
    if array.ndim == 2:
        pil = Image.fromarray(array.astype(np.uint8) * 255 if is_mask else array.astype(np.uint8))
        pil = pil.resize((target_size, target_size), Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR)
        arr = np.asarray(pil)
        return (arr > 127) if is_mask else arr

    if array.ndim == 3 and array.shape[-1] == 3:
        if array.dtype != np.uint8:
            if is_mask:
                pil = Image.fromarray((array[..., 0] > 0.5).astype(np.uint8) * 255)
                pil = pil.resize((target_size, target_size), Image.Resampling.NEAREST)
                arr = (np.asarray(pil) > 127).astype(array.dtype)
                return np.repeat(arr[..., None], 3, axis=-1)
            pil = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
            pil = pil.resize((target_size, target_size), Image.Resampling.BILINEAR)
            return np.asarray(pil)
        pil = Image.fromarray(array)
        pil = pil.resize((target_size, target_size), Image.Resampling.BILINEAR)
        return np.asarray(pil)

    raise ValueError(f"Unsupported array shape for resize: {array.shape}")


def resize_float_array(array: np.ndarray, target_size: int, is_mask: bool = False) -> np.ndarray:
    arr = np.asarray(array)
    mode = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR

    if arr.ndim == 2:
        pil = Image.fromarray(arr.astype(np.float32), mode="F")
        pil = pil.resize((target_size, target_size), mode)
        resized = np.asarray(pil, dtype=np.float32)
        return (resized > 0.5) if is_mask else resized

    if arr.ndim == 3 and arr.shape[-1] in {1, 3}:
        channels = []
        for channel_idx in range(arr.shape[-1]):
            pil = Image.fromarray(arr[..., channel_idx].astype(np.float32), mode="F")
            pil = pil.resize((target_size, target_size), mode)
            channel = np.asarray(pil, dtype=np.float32)
            if is_mask:
                channel = (channel > 0.5).astype(np.float32)
            channels.append(channel)
        stacked = np.stack(channels, axis=-1)
        if is_mask:
            return stacked > 0.5
        return stacked.astype(np.float32)

    raise ValueError(f"Unsupported float array shape for resize: {arr.shape}")


def points_world_to_camera(points_world: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    rotation = np.asarray(extrinsic[:, :3], dtype=np.float32)
    translation = np.asarray(extrinsic[:, 3], dtype=np.float32)
    return np.einsum("hwc,rc->hwr", np.asarray(points_world, dtype=np.float32), rotation) + translation


def point_map_to_normal_torch(
    point_map: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if point_map.ndim != 4 or point_map.shape[-1] != 3:
        raise ValueError(f"Expected point_map [B, H, W, 3], got {tuple(point_map.shape)}")
    if mask.ndim != 3:
        raise ValueError(f"Expected mask [B, H, W], got {tuple(mask.shape)}")

    padded_mask = F.pad(mask.bool(), (1, 1, 1, 1), mode="constant", value=0)
    padded_pts = F.pad(
        point_map.permute(0, 3, 1, 2),
        (1, 1, 1, 1),
        mode="constant",
        value=0.0,
    ).permute(0, 2, 3, 1)

    center = padded_pts[:, 1:-1, 1:-1, :]
    up = padded_pts[:, :-2, 1:-1, :]
    left = padded_pts[:, 1:-1, :-2, :]
    down = padded_pts[:, 2:, 1:-1, :]
    right = padded_pts[:, 1:-1, 2:, :]

    up_dir = up - center
    left_dir = left - center
    down_dir = down - center
    right_dir = right - center

    normals = torch.stack(
        [
            torch.cross(up_dir, left_dir, dim=-1),
            torch.cross(left_dir, down_dir, dim=-1),
            torch.cross(down_dir, right_dir, dim=-1),
            torch.cross(right_dir, up_dir, dim=-1),
        ],
        dim=0,
    )
    valids = torch.stack(
        [
            padded_mask[:, :-2, 1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2],
            padded_mask[:, 1:-1, :-2] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:, 1:-1],
            padded_mask[:, 2:, 1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:],
            padded_mask[:, 1:-1, 2:] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2, 1:-1],
        ],
        dim=0,
    )
    normals = F.normalize(normals, p=2, dim=-1, eps=eps)
    return normals, valids


def collapse_point_normals_torch(
    normals: torch.Tensor,
    valids: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = valids.float()[..., None]
    summed = (normals * weights).sum(dim=0)
    counts = weights.sum(dim=0)
    collapsed = summed / torch.clamp(counts, min=eps)
    collapsed = F.normalize(collapsed, p=2, dim=-1, eps=eps)
    valid = counts[..., 0] > 0
    return collapsed, valid


def point_map_to_normal_numpy(point_map: np.ndarray, mask: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    points = torch.from_numpy(np.asarray(point_map, dtype=np.float32))[None]
    valid_mask = torch.from_numpy(np.asarray(mask, dtype=bool))[None]
    normals4, valids4 = point_map_to_normal_torch(points, valid_mask, eps=eps)
    collapsed, valid = collapse_point_normals_torch(normals4, valids4, eps=eps)
    normal_map = collapsed[0].cpu().numpy().astype(np.float32)
    valid_map = valid[0].cpu().numpy().astype(bool)
    normal_map[~valid_map] = 0.0
    return normal_map, valid_map


def extract_coarse_prior_normal(
    prior_maps: np.ndarray,
    prior_channels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    channel_names = [str(name) for name in np.asarray(prior_channels).tolist()]
    lookup = {name: idx for idx, name in enumerate(channel_names)}
    normal_idx = [lookup[name] for name in COARSE_NORMAL_CHANNELS]
    visible_idx = lookup[COARSE_VISIBLE_MASK_CHANNEL]

    normal = np.asarray(prior_maps[:, normal_idx, :, :], dtype=np.float32).transpose(0, 2, 3, 1)
    visible = np.asarray(prior_maps[:, visible_idx, :, :], dtype=np.float32) > 0.5
    norms = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / np.clip(norms, 1e-6, None)
    normal[~visible] = 0.0
    return normal.astype(np.float32), visible.astype(bool)
