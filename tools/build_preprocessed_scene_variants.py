from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.export_4k4d_human_prior import pad_resize_map  # noqa: E402
from tools.prepare_4k4d_prior_training_case import load_and_preprocess_images_numpy, preprocess_scene_masks  # noqa: E402


TARGET_SIZE = 518
VARIANT_CHOICES = (
    "full",
    "human_crop",
    "human_crop_hardmask",
    "human_crop_softmatte",
    "headshoulder_crop",
    "headface_crop",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build sparse-view preprocessing variants: full image, human crop, "
            "human crop + hard mask, a first-pass soft-matting crop, a "
            "head/neck/shoulder ROI crop, and a tighter head/face crop."
        )
    )
    parser.add_argument("--scene-dir", required=True, help="Source scene directory")
    parser.add_argument("--output-root", required=True, help="Output root for derived scene variants")
    parser.add_argument("--bbox-scale", type=float, default=1.20, help="Scale factor applied to the human bbox before crop")
    parser.add_argument("--bbox-pad", type=int, default=8, help="Extra pixel padding around the aligned bbox before crop")
    parser.add_argument("--mask-background", choices=("black", "white", "gray"), default="white")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANT_CHOICES,
        default=["full", "human_crop", "human_crop_hardmask"],
    )
    parser.add_argument(
        "--softmatte-radius",
        type=float,
        default=4.0,
        help="Gaussian blur radius used to turn the binary human mask into a soft alpha matte.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    return parser.parse_args()


def _load_scene_manifest(scene_dir: Path) -> dict:
    manifest_path = scene_dir / "scene_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing scene manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_aligned_scene(scene_dir: Path, manifest: dict) -> tuple[np.ndarray, np.ndarray]:
    image_paths = [str(Path(view["image_path"])) for view in manifest["exported_views"]]
    images = load_and_preprocess_images_numpy(image_paths, target_size=TARGET_SIZE)
    masks = preprocess_scene_masks(manifest, target_size=TARGET_SIZE).astype(bool)
    return images, masks


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask.astype(bool))
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    image_hw: tuple[int, int],
    *,
    scale: float,
    pad: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    h, w = image_hw
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    half_w = max(8.0, 0.5 * (x1 - x0) * scale + pad)
    half_h = max(8.0, 0.5 * (y1 - y0) * scale + pad)
    new_x0 = max(0, int(round(cx - half_w)))
    new_y0 = max(0, int(round(cy - half_h)))
    new_x1 = min(w, int(round(cx + half_w)))
    new_y1 = min(h, int(round(cy + half_h)))
    new_x1 = max(new_x1, new_x0 + 1)
    new_y1 = max(new_y1, new_y0 + 1)
    return new_x0, new_y0, new_x1, new_y1


def _clip_bbox_to_image(
    bbox: tuple[float, float, float, float],
    image_hw: tuple[int, int],
) -> tuple[int, int, int, int]:
    h, w = image_hw
    x0, y0, x1, y1 = bbox
    out_x0 = max(0, int(round(x0)))
    out_y0 = max(0, int(round(y0)))
    out_x1 = min(w, int(round(x1)))
    out_y1 = min(h, int(round(y1)))
    out_x1 = max(out_x1, out_x0 + 1)
    out_y1 = max(out_y1, out_y0 + 1)
    return out_x0, out_y0, out_x1, out_y1


def _headshoulder_bbox(
    mask: np.ndarray,
    image_hw: tuple[int, int],
    *,
    scale: float,
    pad: int,
) -> tuple[int, int, int, int] | None:
    body_bbox = _mask_bbox(mask)
    if body_bbox is None:
        return None

    x0, y0, x1, y1 = body_bbox
    body_h = max(1.0, float(y1 - y0))
    upper_limit = min(mask.shape[0], int(round(y0 + body_h * 0.55)))
    upper_mask = np.zeros_like(mask, dtype=bool)
    upper_mask[:upper_limit] = mask[:upper_limit]
    upper_bbox = _mask_bbox(upper_mask)
    if upper_bbox is None:
        upper_bbox = body_bbox

    ux0, uy0, ux1, uy1 = upper_bbox
    upper_w = max(1.0, float(ux1 - ux0))
    upper_h = max(1.0, float(uy1 - uy0))
    cx = 0.5 * (ux0 + ux1)
    # Bias the crop slightly downward so the shoulder line stays visible.
    cy = uy0 + upper_h * 0.58
    crop_side = max(upper_w * 1.18, upper_h * 1.28) * scale + 2.0 * float(pad)
    half = max(18.0, 0.5 * crop_side)
    return _clip_bbox_to_image((cx - half, cy - half, cx + half, cy + half), image_hw)


def _headface_bbox(
    mask: np.ndarray,
    image_hw: tuple[int, int],
    *,
    scale: float,
    pad: int,
) -> tuple[int, int, int, int] | None:
    body_bbox = _mask_bbox(mask)
    if body_bbox is None:
        return None

    x0, y0, x1, y1 = body_bbox
    body_h = max(1.0, float(y1 - y0))
    upper_limit = min(mask.shape[0], int(round(y0 + body_h * 0.42)))
    upper_mask = np.zeros_like(mask, dtype=bool)
    upper_mask[:upper_limit] = mask[:upper_limit]
    upper_bbox = _mask_bbox(upper_mask)
    if upper_bbox is None:
        upper_bbox = _headshoulder_bbox(mask, image_hw, scale=1.0, pad=pad)
    if upper_bbox is None:
        return None

    ux0, uy0, ux1, uy1 = upper_bbox
    upper_w = max(1.0, float(ux1 - ux0))
    upper_h = max(1.0, float(uy1 - uy0))
    cx = 0.5 * (ux0 + ux1)
    cy = uy0 + upper_h * 0.55
    crop_side = max(upper_w * 1.05, upper_h * 1.12) * scale + 2.0 * float(pad)
    half = max(16.0, 0.5 * crop_side)
    return _clip_bbox_to_image((cx - half, cy - half, cx + half, cy + half), image_hw)


def _roi_focus_label(variant: str) -> str:
    if variant == "headshoulder_crop":
        return "headshoulder"
    if variant == "headface_crop":
        return "headface"
    return "fullbody"


def _resolve_variant_bbox(
    mask: np.ndarray,
    image_hw: tuple[int, int],
    *,
    variant: str,
    scale: float,
    pad: int,
) -> tuple[int, int, int, int] | None:
    bbox = _mask_bbox(mask)
    if bbox is None:
        return None
    if variant == "headshoulder_crop":
        return _headshoulder_bbox(mask, image_hw, scale=scale, pad=pad)
    if variant == "headface_crop":
        return _headface_bbox(mask, image_hw, scale=scale, pad=pad)
    return _expand_bbox(bbox, image_hw, scale=scale, pad=pad)


def _background_rgb(mode: str) -> tuple[int, int, int]:
    if mode == "black":
        return (0, 0, 0)
    if mode == "gray":
        return (127, 127, 127)
    return (255, 255, 255)


def _fit_crop_to_square(
    arr: np.ndarray,
    *,
    mode: str,
    background: tuple[int, int, int] | int,
) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        resized = pad_resize_map(arr.astype(np.float32), TARGET_SIZE, mode=mode)
        return resized
    if arr.ndim == 3 and arr.shape[-1] == 3:
        image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        width, height = image.size
        if width >= height:
            new_width = TARGET_SIZE
            new_height = round(height * (new_width / width) / 14) * 14
        else:
            new_height = TARGET_SIZE
            new_width = round(width * (new_height / height) / 14) * 14
        new_width = max(14, int(new_width))
        new_height = max(14, int(new_height))
        image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), background if isinstance(background, tuple) else (int(background),) * 3)
        top = (TARGET_SIZE - new_height) // 2
        left = (TARGET_SIZE - new_width) // 2
        canvas.paste(image, (left, top))
        return np.asarray(canvas, dtype=np.float32)
    channels = []
    for channel_idx in range(arr.shape[-1]):
        channels.append(pad_resize_map(arr[..., channel_idx].astype(np.float32), TARGET_SIZE, mode=mode))
    return np.stack(channels, axis=-1)


def _build_soft_matte(mask: np.ndarray, *, blur_radius: float) -> np.ndarray:
    mask_uint8 = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0) * 255.0
    mask_img = Image.fromarray(mask_uint8.astype(np.uint8), mode="L")
    alpha = np.asarray(mask_img.filter(ImageFilter.GaussianBlur(radius=max(0.0, float(blur_radius)))), dtype=np.float32) / 255.0
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def _soft_matte_composite(
    rgb: np.ndarray,
    alpha: np.ndarray,
    *,
    bg_rgb: tuple[int, int, int],
) -> np.ndarray:
    alpha = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)[..., None]
    foreground = np.asarray(rgb, dtype=np.float32)
    background = np.asarray(bg_rgb, dtype=np.float32).reshape(1, 1, 3)
    composited = foreground * alpha + background * (1.0 - alpha)
    return np.clip(composited, 0.0, 255.0).astype(np.uint8)


def _transform_variant(
    *,
    rgb: np.ndarray,
    mask: np.ndarray,
    prior_maps: np.ndarray,
    prior_mask: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int] | None,
    variant: str,
    bg_rgb: tuple[int, int, int],
    softmatte_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    if variant == "full" or bbox_xyxy is None:
        if variant == "human_crop_hardmask":
            masked_rgb = rgb.copy()
            masked_rgb[~mask] = np.asarray(bg_rgb, dtype=np.uint8)
            return masked_rgb, mask.astype(np.uint8) * 255, prior_maps, prior_mask, {
                "variant": variant,
                "transform": "identity_masked",
                "source_aligned_size": [int(TARGET_SIZE), int(TARGET_SIZE)],
            }
        if variant == "human_crop_softmatte":
            matte = _build_soft_matte(mask.astype(np.float32), blur_radius=softmatte_radius)
            matte_rgb = _soft_matte_composite(rgb, matte, bg_rgb=bg_rgb)
            matte_uint8 = np.clip(np.rint(matte * 255.0), 0, 255).astype(np.uint8)
            matte_prior_maps = np.asarray(prior_maps, dtype=np.float32).copy()
            if matte_prior_maps.shape[0] > 0:
                matte_prior_maps[0] = matte
            return matte_rgb, matte_uint8, matte_prior_maps, prior_mask, {
                "variant": variant,
                "transform": "identity_softmatte",
                "softmatte_radius": float(softmatte_radius),
                "source_aligned_size": [int(TARGET_SIZE), int(TARGET_SIZE)],
            }
        return rgb, mask.astype(np.uint8) * 255, prior_maps, prior_mask, {
            "variant": variant,
            "transform": "identity",
            "source_aligned_size": [int(TARGET_SIZE), int(TARGET_SIZE)],
        }

    x0, y0, x1, y1 = bbox_xyxy
    crop_rgb = rgb[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]
    crop_prior_mask = prior_mask[y0:y1, x0:x1]
    crop_prior_maps = prior_maps[:, y0:y1, x0:x1]

    if variant == "human_crop_hardmask":
        crop_rgb = crop_rgb.copy()
        crop_rgb[~crop_mask] = np.asarray(bg_rgb, dtype=np.uint8)

    out_rgb = _fit_crop_to_square(crop_rgb, mode="bicubic", background=bg_rgb).clip(0, 255).astype(np.uint8)
    out_mask_float = (_fit_crop_to_square(crop_mask.astype(np.float32), mode="nearest", background=0) > 0.5).astype(np.float32)
    out_prior_mask = (_fit_crop_to_square(crop_prior_mask.astype(np.float32), mode="nearest", background=0) > 0.5)
    if variant == "human_crop_softmatte":
        out_mask_float = _build_soft_matte(out_mask_float, blur_radius=softmatte_radius)
        out_rgb = _soft_matte_composite(out_rgb, out_mask_float, bg_rgb=bg_rgb)
    out_mask = np.clip(np.rint(out_mask_float * 255.0), 0, 255).astype(np.uint8)

    transformed_channels = []
    for channel_idx in range(crop_prior_maps.shape[0]):
        channel = crop_prior_maps[channel_idx]
        channel_name = str(channel_idx)
        mode = "nearest" if channel_name in {"0"} else "bilinear"
        transformed = _fit_crop_to_square(channel.astype(np.float32), mode=mode, background=0)
        if transformed.ndim == 3:
            transformed = transformed[..., 0]
        transformed_channels.append(transformed.astype(np.float32))
    out_prior_maps = np.stack(transformed_channels, axis=0)
    if variant == "human_crop_softmatte" and out_prior_maps.shape[0] > 0:
        out_prior_maps[0] = out_mask_float.astype(np.float32)

    return out_rgb, out_mask, out_prior_maps, out_prior_mask.astype(bool), {
        "variant": variant,
        "crop_bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "transform": "crop_pad_to_square",
        "softmatte_radius": float(softmatte_radius) if variant == "human_crop_softmatte" else None,
        "roi_focus": _roi_focus_label(variant),
        "source_aligned_size": [int(TARGET_SIZE), int(TARGET_SIZE)],
    }


def _render_contact_sheet(images: list[Image.Image], labels: list[str], output_path: Path) -> None:
    cell_w = 220
    cols = 4
    rows = (len(images) + cols - 1) // cols
    resized = []
    heights = []
    for image in images:
        height = int(round(image.height * (cell_w / image.width)))
        resized_img = image.resize((cell_w, max(1, height)), Image.Resampling.BILINEAR)
        resized.append(resized_img)
        heights.append(resized_img.height)
    cell_h = max(heights) if heights else cell_w
    label_h = 20
    canvas = Image.new("RGB", (cols * cell_w, rows * (cell_h + label_h)), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for idx, (image, label) in enumerate(zip(resized, labels)):
        row = idx // cols
        col = idx % cols
        x = col * cell_w
        y = row * (cell_h + label_h)
        canvas.paste(image, (x, y))
        draw.text((x + 4, y + cell_h + 2), label, fill=(240, 240, 240), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _write_variant_scene(
    *,
    scene_dir: Path,
    output_root: Path,
    manifest: dict,
    images: np.ndarray,
    masks: np.ndarray,
    prior_payload: dict[str, np.ndarray],
    variant: str,
    bbox_scale: float,
    bbox_pad: int,
    bg_rgb: tuple[int, int, int],
    softmatte_radius: float,
    overwrite: bool,
) -> Path:
    out_dir = output_root / f"{scene_dir.name}_{variant}"
    if out_dir.exists() and not overwrite:
        raise FileExistsError(f"{out_dir} already exists. Re-run with --overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    prior_maps = np.asarray(prior_payload["prior_maps"]).astype(np.float32)
    prior_mask = np.asarray(prior_payload["prior_mask"]).astype(bool)
    prior_channels = np.asarray(prior_payload["prior_channels"])
    prior_summary_tokens = np.asarray(prior_payload["prior_summary_tokens"]) if "prior_summary_tokens" in prior_payload else None
    prior_summary_channels = np.asarray(prior_payload["prior_summary_channels"]) if "prior_summary_channels" in prior_payload else None

    written_images: list[Image.Image] = []
    written_masks: list[Image.Image] = []
    variant_views = []
    variant_prior_maps = []
    variant_prior_masks = []

    for view_idx, view in enumerate(manifest["exported_views"]):
        expanded_bbox = _resolve_variant_bbox(
            masks[view_idx],
            masks[view_idx].shape,
            variant=variant,
            scale=bbox_scale,
            pad=bbox_pad,
        )
        out_rgb, out_mask, out_prior_map, out_prior_mask, meta = _transform_variant(
            rgb=images[view_idx],
            mask=masks[view_idx],
            prior_maps=prior_maps[view_idx],
            prior_mask=prior_mask[view_idx],
            bbox_xyxy=expanded_bbox,
            variant=variant,
            bg_rgb=bg_rgb,
            softmatte_radius=softmatte_radius,
        )

        stem = Path(view["image_path"]).stem
        image_path = out_dir / "images" / f"{stem}.png"
        mask_path = out_dir / "masks" / f"{stem}.png"
        Image.fromarray(out_rgb).save(image_path)
        Image.fromarray(out_mask).save(mask_path)

        written_images.append(Image.fromarray(out_rgb))
        written_masks.append(Image.fromarray(np.repeat(out_mask[..., None], 3, axis=-1)))
        variant_prior_maps.append(out_prior_map.astype(np.float16))
        variant_prior_masks.append(out_prior_mask.astype(bool))

        updated_view = dict(view)
        updated_view["image_path"] = str(image_path.resolve())
        updated_view["mask_path"] = str(mask_path.resolve())
        updated_view["image_size"] = [TARGET_SIZE, TARGET_SIZE]
        updated_view["source_image_size"] = list(view.get("image_size", [TARGET_SIZE, TARGET_SIZE]))
        updated_view["preprocess_variant"] = variant
        updated_view["preprocess_meta"] = meta
        variant_views.append(updated_view)

    np.savez_compressed(
        out_dir / "prior_maps.npz",
        prior_maps=np.stack(variant_prior_maps, axis=0),
        prior_summary_tokens=prior_summary_tokens.astype(np.float16) if prior_summary_tokens is not None else np.zeros((len(variant_views), 0, 0), dtype=np.float16),
        prior_mask=np.stack(variant_prior_masks, axis=0),
        prior_channels=prior_channels,
        prior_summary_channels=prior_summary_channels if prior_summary_channels is not None else np.asarray([], dtype=object),
    )

    labels = [f"{view['camera_id']} ({view['role']})" for view in variant_views]
    _render_contact_sheet(written_images, labels, out_dir / "rgb_contact_sheet.png")
    _render_contact_sheet(written_masks, labels, out_dir / "mask_contact_sheet.png")

    updated_manifest = dict(manifest)
    updated_manifest["exported_views"] = variant_views
    updated_manifest["preprocess_variant"] = variant
    updated_manifest["source_scene_dir"] = str(scene_dir.resolve())
    updated_manifest["prior_maps_file"] = str((out_dir / "prior_maps.npz").resolve())
    updated_manifest["preprocess_variant_summary"] = {
        "variant": variant,
        "bbox_scale": float(bbox_scale),
        "bbox_pad": int(bbox_pad),
        "mask_background_rgb": list(bg_rgb),
        "roi_focus": _roi_focus_label(variant),
        "softmatte_radius": float(softmatte_radius) if variant == "human_crop_softmatte" else None,
    }
    (out_dir / "scene_manifest.json").write_text(json.dumps(updated_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir


def main() -> int:
    args = parse_args()
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = _load_scene_manifest(scene_dir)
    images, masks = _load_aligned_scene(scene_dir, manifest)
    with np.load(scene_dir / "prior_maps.npz", allow_pickle=False) as prior_payload:
        prior_data = {key: np.array(prior_payload[key]) for key in prior_payload.files}

    bg_rgb = _background_rgb(args.mask_background)
    written_dirs = []
    for variant in args.variants:
        written_dirs.append(
            str(
                _write_variant_scene(
                    scene_dir=scene_dir,
                    output_root=output_root,
                    manifest=manifest,
                    images=images,
                    masks=masks,
                    prior_payload=prior_data,
                    variant=variant,
                    bbox_scale=float(args.bbox_scale),
                    bbox_pad=int(args.bbox_pad),
                    bg_rgb=bg_rgb,
                    softmatte_radius=float(args.softmatte_radius),
                    overwrite=bool(args.overwrite),
                )
            )
        )

    summary = {
        "source_scene_dir": str(scene_dir),
        "output_root": str(output_root),
        "variants": list(args.variants),
        "bbox_scale": float(args.bbox_scale),
        "bbox_pad": int(args.bbox_pad),
        "mask_background": args.mask_background,
        "softmatte_radius": float(args.softmatte_radius),
        "written_variant_dirs": written_dirs,
    }
    (output_root / f"{scene_dir.name}_preprocess_variants_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
