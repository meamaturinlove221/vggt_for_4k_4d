from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_preprocessed_scene_variants import (  # noqa: E402
    TARGET_SIZE,
    _background_rgb,
    _expand_bbox,
    _headface_bbox,
    _headshoulder_bbox,
    _mask_bbox,
    _render_contact_sheet,
)
from tools.prepare_4k4d_prior_training_case import (  # noqa: E402
    DEFAULT_BODY_PART_COUNT,
    DEFAULT_BODY_PART_EMBED_DIM,
    DEFAULT_VERTEX_ID_EMBED_DIM,
    SUBSET_NAME,
    build_prior_stack,
    build_external_prior_stack,
    load_external_prior_bundle,
    resolve_smplx_model_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a true high-resolution human/head crop scene from the original RGB/mask pixels. "
            "Unlike build_preprocessed_scene_variants.py, this crops before the final 518 pad-resize."
        )
    )
    parser.add_argument("--scene-dir", required=True, help="Source scene directory with high-res images/masks")
    parser.add_argument("--output-scene-dir", required=True, help="Output prior-enabled cropped scene directory")
    parser.add_argument("--roi", choices=("human", "headface", "headshoulder"), default="headface")
    parser.add_argument("--bbox-scale", type=float, default=1.0, help="Scale factor on the raw-image ROI bbox")
    parser.add_argument("--bbox-pad", type=int, default=24, help="Raw-pixel padding around the ROI bbox")
    parser.add_argument("--target-size", type=int, default=TARGET_SIZE, help="Final square tensor size")
    parser.add_argument("--mask-background", choices=("white", "gray", "black"), default="white")
    parser.add_argument("--hardmask", action="store_true", help="Replace non-human crop pixels with mask background")
    parser.add_argument(
        "--skip-prior",
        action="store_true",
        help="Write only cropped images/masks/manifest. Use this for external teacher inputs such as PSHuman.",
    )
    parser.add_argument("--external-prior-bundle", default="", help="Optional external prior bundle to rebuild prior maps")
    parser.add_argument(
        "--rebuild-scene-prior",
        action="store_true",
        help="Rebuild scene-local SMPL-X prior after raw crop using annotations and transformed camera intrinsics",
    )
    parser.add_argument("--dataset-root", default="", help="4K4D dataset root; defaults to scene_manifest.json")
    parser.add_argument("--subset-name", default=SUBSET_NAME)
    parser.add_argument("--smplx-model-dir", default="", help="Optional SMPL-X model directory for bundle reprojection")
    parser.add_argument("--smplx-gender", default="neutral", choices=("neutral", "male", "female"))
    parser.add_argument("--mesh-fill-knn", type=int, default=4)
    parser.add_argument("--summary-token-count", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=6.0)
    parser.add_argument("--min-conf", type=float, default=0.05)
    parser.add_argument("--vertex-id-dim", type=int, default=DEFAULT_VERTEX_ID_EMBED_DIM)
    parser.add_argument("--body-part-dim", type=int, default=DEFAULT_BODY_PART_EMBED_DIM)
    parser.add_argument("--body-part-count", type=int, default=DEFAULT_BODY_PART_COUNT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve_raw_roi_bbox(
    mask: np.ndarray,
    image_hw: tuple[int, int],
    *,
    roi: str,
    bbox_scale: float,
    bbox_pad: int,
) -> tuple[int, int, int, int] | None:
    if roi == "human":
        bbox = _mask_bbox(mask)
        if bbox is None:
            return None
        return _expand_bbox(bbox, image_hw, scale=float(bbox_scale), pad=int(bbox_pad))
    if roi == "headface":
        return _headface_bbox(mask, image_hw, scale=float(bbox_scale), pad=int(bbox_pad))
    if roi == "headshoulder":
        return _headshoulder_bbox(mask, image_hw, scale=float(bbox_scale), pad=int(bbox_pad))
    raise ValueError(f"Unsupported ROI: {roi}")


def _load_manifest(scene_dir: Path) -> dict:
    manifest_path = scene_dir / "scene_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing scene_manifest.json: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_mask(mask_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    if mask.size != image_size:
        mask = mask.resize(image_size, Image.Resampling.NEAREST)
    return np.asarray(mask, dtype=np.uint8) > 127


def _raw_bbox_to_aligned_bbox(
    raw_bbox: tuple[int, int, int, int],
    raw_size_wh: list[int] | tuple[int, int],
    target_size: int,
) -> tuple[int, int, int, int]:
    width, height = int(raw_size_wh[0]), int(raw_size_wh[1])
    if width <= 0 or height <= 0:
        return 0, 0, int(target_size), int(target_size)
    if width >= height:
        new_width = int(target_size)
        new_height = round(height * (new_width / width) / 14) * 14
    else:
        new_height = int(target_size)
        new_width = round(width * (new_height / height) / 14) * 14
    new_width = max(14, int(new_width))
    new_height = max(14, int(new_height))
    scale_x = new_width / float(width)
    scale_y = new_height / float(height)
    pad_left = (int(target_size) - new_width) // 2
    pad_top = (int(target_size) - new_height) // 2
    x0, y0, x1, y1 = raw_bbox
    return (
        max(0, min(int(target_size) - 1, int(np.floor(x0 * scale_x + pad_left)))),
        max(0, min(int(target_size) - 1, int(np.floor(y0 * scale_y + pad_top)))),
        max(1, min(int(target_size), int(np.ceil(x1 * scale_x + pad_left)))),
        max(1, min(int(target_size), int(np.ceil(y1 * scale_y + pad_top)))),
    )


def _resample_mode(mode: str) -> int:
    if mode == "nearest":
        return Image.Resampling.NEAREST
    if mode == "bicubic":
        return Image.Resampling.BICUBIC
    return Image.Resampling.BILINEAR


def _fit_crop_to_square_size(
    arr: np.ndarray,
    *,
    target_size: int,
    mode: str,
    background: tuple[int, int, int] | int,
) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        height, width = arr.shape
        if width >= height:
            new_width = int(target_size)
            new_height = round(height * (new_width / max(1, width)) / 14) * 14
        else:
            new_height = int(target_size)
            new_width = round(width * (new_height / max(1, height)) / 14) * 14
        new_width = max(14, int(new_width))
        new_height = max(14, int(new_height))
        image = Image.fromarray(arr.astype(np.float32), mode="F")
        image = image.resize((new_width, new_height), _resample_mode(mode))
        bg_value = float(background if isinstance(background, int) else 0)
        canvas = np.full((int(target_size), int(target_size)), bg_value, dtype=np.float32)
        top = (int(target_size) - new_height) // 2
        left = (int(target_size) - new_width) // 2
        canvas[top : top + new_height, left : left + new_width] = np.asarray(image, dtype=np.float32)
        return canvas
    if arr.ndim == 3 and arr.shape[-1] == 3:
        image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        width, height = image.size
        if width >= height:
            new_width = int(target_size)
            new_height = round(height * (new_width / max(1, width)) / 14) * 14
        else:
            new_height = int(target_size)
            new_width = round(width * (new_height / max(1, height)) / 14) * 14
        new_width = max(14, int(new_width))
        new_height = max(14, int(new_height))
        image = image.resize((new_width, new_height), _resample_mode(mode))
        bg_rgb = background if isinstance(background, tuple) else (int(background),) * 3
        canvas = Image.new("RGB", (int(target_size), int(target_size)), bg_rgb)
        top = (int(target_size) - new_height) // 2
        left = (int(target_size) - new_width) // 2
        canvas.paste(image, (left, top))
        return np.asarray(canvas, dtype=np.float32)
    channels = []
    for channel_idx in range(arr.shape[-1]):
        channels.append(
            _fit_crop_to_square_size(
                arr[..., channel_idx],
                target_size=int(target_size),
                mode=mode,
                background=background,
            )
        )
    return np.stack(channels, axis=-1)


def _resize_prior_like_image(
    prior_maps: np.ndarray,
    prior_mask: np.ndarray,
    aligned_bbox: tuple[int, int, int, int],
    target_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = aligned_bbox
    crop_maps = prior_maps[:, y0:y1, x0:x1]
    crop_mask = prior_mask[y0:y1, x0:x1]
    transformed = []
    for channel_idx, channel in enumerate(crop_maps):
        mode = "nearest" if channel_idx == 0 else "bilinear"
        transformed.append(
            _fit_crop_to_square_size(channel.astype(np.float32), target_size=int(target_size), mode=mode, background=0)
        )
    out_maps = np.stack(transformed, axis=0).astype(np.float32)
    out_mask = _fit_crop_to_square_size(
        crop_mask.astype(np.float32), target_size=int(target_size), mode="nearest", background=0
    ) > 0.5
    return out_maps, out_mask


def _render_debug_overlay(image: Image.Image, mask: np.ndarray, bbox: tuple[int, int, int, int], output_path: Path) -> None:
    preview = image.copy().convert("RGB")
    preview.thumbnail((420, 420), Image.Resampling.BICUBIC)
    sx = preview.size[0] / image.size[0]
    sy = preview.size[1] / image.size[1]
    draw = ImageDraw.Draw(preview)
    x0, y0, x1, y1 = bbox
    draw.rectangle((x0 * sx, y0 * sy, x1 * sx, y1 * sy), outline=(255, 0, 0), width=3)
    draw.text((8, 8), f"raw crop {x1-x0}x{y1-y0}", fill=(255, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(output_path)


def _write_scene(
    *,
    scene_dir: Path,
    output_scene_dir: Path,
    manifest: dict,
    roi: str,
    bbox_scale: float,
    bbox_pad: int,
    target_size: int,
    bg_rgb: tuple[int, int, int],
    hardmask: bool,
    external_prior_bundle: Path | None,
    rebuild_scene_prior: bool,
    skip_prior: bool,
    dataset_root: Path,
    subset_name: str,
    smplx_model_dir: Path | None,
    smplx_gender: str,
    mesh_fill_knn: int,
    summary_token_count: int,
    sigma: float,
    min_conf: float,
    vertex_id_dim: int,
    body_part_dim: int,
    body_part_count: int,
    overwrite: bool,
) -> dict:
    if output_scene_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output scene exists; pass --overwrite: {output_scene_dir}")
        shutil.rmtree(output_scene_dir)
    (output_scene_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_scene_dir / "masks").mkdir(parents=True, exist_ok=True)
    (output_scene_dir / "debug_overlays").mkdir(parents=True, exist_ok=True)

    source_prior_payload = None
    if skip_prior and (external_prior_bundle is not None or rebuild_scene_prior):
        raise ValueError("--skip-prior cannot be combined with --external-prior-bundle or --rebuild-scene-prior")

    if external_prior_bundle is None and not rebuild_scene_prior and not skip_prior:
        prior_path = scene_dir / "prior_maps.npz"
        if not prior_path.is_file():
            raise FileNotFoundError(f"Need source prior_maps.npz or --external-prior-bundle: {prior_path}")
        source_prior_payload = np.load(prior_path, allow_pickle=False)

    written_views = []
    written_images: list[Image.Image] = []
    written_masks: list[Image.Image] = []
    transformed_prior_maps = []
    transformed_prior_masks = []
    crop_summaries = []

    try:
        for view_idx, view in enumerate(manifest["exported_views"]):
            image_path = Path(view["image_path"])
            mask_path = Path(view["mask_path"])
            image = Image.open(image_path).convert("RGB")
            mask = _load_mask(mask_path, image.size)
            image_hw = (image.size[1], image.size[0])
            bbox = _resolve_raw_roi_bbox(
                mask,
                image_hw,
                roi=roi,
                bbox_scale=float(bbox_scale),
                bbox_pad=int(bbox_pad),
            )
            if bbox is None:
                bbox = (0, 0, image.size[0], image.size[1])
            x0, y0, x1, y1 = bbox

            crop_rgb = np.asarray(image, dtype=np.uint8)[y0:y1, x0:x1]
            crop_mask = mask[y0:y1, x0:x1]
            if hardmask:
                crop_rgb = crop_rgb.copy()
                crop_rgb[~crop_mask] = np.asarray(bg_rgb, dtype=np.uint8)

            out_rgb = (
                _fit_crop_to_square_size(crop_rgb, target_size=int(target_size), mode="bicubic", background=bg_rgb)
                .clip(0, 255)
                .astype(np.uint8)
            )
            out_mask = (
                _fit_crop_to_square_size(
                    crop_mask.astype(np.float32),
                    target_size=int(target_size),
                    mode="nearest",
                    background=0,
                )
                > 0.5
            ).astype(np.uint8) * 255

            stem = Path(view["image_path"]).stem
            out_image_path = output_scene_dir / "images" / f"{stem}.png"
            out_mask_path = output_scene_dir / "masks" / f"{stem}.png"
            Image.fromarray(out_rgb).save(out_image_path)
            Image.fromarray(out_mask).save(out_mask_path)
            _render_debug_overlay(image, mask, bbox, output_scene_dir / "debug_overlays" / f"{stem}_raw_bbox.png")
            written_images.append(Image.fromarray(out_rgb))
            written_masks.append(Image.fromarray(np.repeat(out_mask[..., None], 3, axis=-1)))

            updated_view = dict(view)
            updated_view["image_path"] = str(out_image_path.resolve())
            updated_view["mask_path"] = str(out_mask_path.resolve())
            updated_view["original_source_image_size"] = list(view.get("image_size", [image.size[0], image.size[1]]))
            updated_view["source_image_size"] = [target_size, target_size]
            updated_view["image_size"] = [target_size, target_size]
            updated_view["preprocess_variant"] = f"raw_{roi}_crop"
            updated_view["preprocess_meta"] = {
                "variant": f"raw_{roi}_crop",
                "transform": "raw_crop_pad_to_square",
                "crop_bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
                "raw_image_size_wh": [int(image.size[0]), int(image.size[1])],
                "aligned_source_size": [int(target_size), int(target_size)],
                "source_pixel_stage": "original_rgb_before_518_resize",
                "hardmask": bool(hardmask),
                "mask_background_rgb": list(bg_rgb),
            }
            written_views.append(updated_view)
            crop_summaries.append(
                {
                    "camera_id": view.get("camera_id"),
                    "bbox_xyxy_raw": [int(x0), int(y0), int(x1), int(y1)],
                    "bbox_size_raw": [int(x1 - x0), int(y1 - y0)],
                    "raw_image_size_wh": [int(image.size[0]), int(image.size[1])],
                    "mask_pixels_raw_crop": int(crop_mask.sum()),
                }
            )

        out_manifest = dict(manifest)
        out_manifest["exported_views"] = written_views
        out_manifest["source_scene_dir"] = str(scene_dir.resolve())
        out_manifest["preprocess_variant"] = f"raw_{roi}_crop"
        out_manifest["preprocess_variant_summary"] = {
            "variant": f"raw_{roi}_crop",
            "source_pixel_stage": "original_rgb_before_518_resize",
            "target_size": int(target_size),
            "bbox_scale": float(bbox_scale),
            "bbox_pad": int(bbox_pad),
            "hardmask": bool(hardmask),
            "mask_background_rgb": list(bg_rgb),
            "crop_summaries": crop_summaries,
        }

        if skip_prior:
            out_manifest.pop("prior_maps_file", None)
            out_manifest.pop("prior_channels", None)
            out_manifest.pop("prior_summary_channels", None)
            out_manifest.pop("prior_summary_token_count", None)
            out_manifest["prior_source"] = "skipped_for_external_teacher_input"
        elif external_prior_bundle is not None:
            external_bundle = load_external_prior_bundle(external_prior_bundle, out_manifest)
            prior_maps, prior_mask, prior_summary_tokens, _, _, prior_meta = build_external_prior_stack(
                out_manifest,
                target_size=int(target_size),
                mask_only=False,
                smplx_params=external_bundle["smplx_params"],
                camera_params=external_bundle["camera_params"],
                external_bundle_meta=external_bundle["resolved_meta"],
                smplx_model_dir=smplx_model_dir,
                smplx_gender=smplx_gender,
                mesh_fill_knn=int(mesh_fill_knn),
                summary_token_count=int(summary_token_count),
            )
            prior_channels = np.asarray(prior_meta["channel_names"])
            prior_summary_channels = np.asarray(prior_meta["summary_channel_names"])
            out_manifest["prior_input_meta"] = prior_meta
            out_manifest["external_prior_bundle"] = external_bundle["resolved_meta"]
            out_manifest["prior_source"] = "external_prior_bundle_raw_crop_reprojected"
        elif rebuild_scene_prior:
            prior_maps, prior_mask, prior_summary_tokens, _, _, prior_meta = build_prior_stack(
                out_manifest,
                dataset_root=dataset_root,
                subset_name=subset_name,
                target_size=int(target_size),
                mask_only=False,
                sigma=float(sigma),
                min_conf=float(min_conf),
                smplx_model_dir=smplx_model_dir,
                smplx_gender=smplx_gender,
                mesh_fill_knn=int(mesh_fill_knn),
                summary_token_count=int(summary_token_count),
                vertex_id_dim=int(vertex_id_dim),
                body_part_dim=int(body_part_dim),
                body_part_count=int(body_part_count),
            )
            prior_channels = np.asarray(prior_meta["channel_names"])
            prior_summary_channels = np.asarray(prior_meta["summary_channel_names"])
            out_manifest["prior_input_meta"] = prior_meta
            out_manifest["prior_source"] = "scene_annotation_raw_crop_reprojected"
        else:
            assert source_prior_payload is not None
            source_prior_maps = np.asarray(source_prior_payload["prior_maps"], dtype=np.float32)
            source_prior_masks = np.asarray(source_prior_payload["prior_mask"], dtype=bool)
            if source_prior_maps.shape[2:] != (target_size, target_size):
                raise ValueError(f"Source prior maps must already be {target_size}x{target_size}, got {source_prior_maps.shape}")
            for view_idx, view in enumerate(written_views):
                raw_bbox = tuple(int(v) for v in view["preprocess_meta"]["crop_bbox_xyxy"])
                raw_size = view["preprocess_meta"]["raw_image_size_wh"]
                aligned_bbox = _raw_bbox_to_aligned_bbox(raw_bbox, raw_size, target_size)
                if aligned_bbox[2] <= aligned_bbox[0] or aligned_bbox[3] <= aligned_bbox[1]:
                    aligned_bbox = (0, 0, target_size, target_size)
                out_map, out_prior_mask = _resize_prior_like_image(
                    source_prior_maps[view_idx],
                    source_prior_masks[view_idx],
                    aligned_bbox,
                    int(target_size),
                )
                transformed_prior_maps.append(out_map.astype(np.float16))
                transformed_prior_masks.append(out_prior_mask.astype(bool))
                view["preprocess_meta"]["source_prior_crop_bbox_xyxy_518"] = [int(v) for v in aligned_bbox]

            prior_maps = np.stack(transformed_prior_maps, axis=0)
            prior_mask = np.stack(transformed_prior_masks, axis=0)
            prior_summary_tokens = np.asarray(source_prior_payload["prior_summary_tokens"], dtype=np.float16)
            prior_channels = np.asarray(source_prior_payload["prior_channels"])
            prior_summary_channels = np.asarray(source_prior_payload["prior_summary_channels"])
            out_manifest["prior_input_meta"] = dict(manifest.get("prior_input_meta", {}))
            out_manifest["preprocess_variant_summary"]["prior_alignment_note"] = (
                "Dense prior maps were cropped from the existing 518-aligned prior using a scaled raw bbox. "
                "Use --external-prior-bundle for exact cropped-camera SMPL-X reprojection."
            )

        if not skip_prior:
            np.savez_compressed(
                output_scene_dir / "prior_maps.npz",
                prior_maps=np.asarray(prior_maps, dtype=np.float16),
                prior_summary_tokens=np.asarray(prior_summary_tokens, dtype=np.float16)
                if prior_summary_tokens is not None
                else np.zeros((len(written_views), 0, 0), dtype=np.float16),
                prior_mask=np.asarray(prior_mask, dtype=bool),
                prior_channels=prior_channels,
                prior_summary_channels=prior_summary_channels,
            )
            out_manifest["prior_maps_file"] = str((output_scene_dir / "prior_maps.npz").resolve())
            out_manifest["prior_channels"] = [str(v) for v in prior_channels.tolist()]
            out_manifest["prior_summary_channels"] = [str(v) for v in prior_summary_channels.tolist()]
            out_manifest["prior_summary_token_count"] = int(np.asarray(prior_summary_tokens).shape[1]) if prior_summary_tokens is not None and np.asarray(prior_summary_tokens).ndim >= 2 else 0

        _render_contact_sheet(written_images, [f"{v['camera_id']} ({v['role']})" for v in written_views], output_scene_dir / "rgb_contact_sheet.png")
        _render_contact_sheet(written_masks, [f"{v['camera_id']} ({v['role']})" for v in written_views], output_scene_dir / "mask_contact_sheet.png")
        (output_scene_dir / "scene_manifest.json").write_text(json.dumps(out_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        summary = {
            "source_scene_dir": str(scene_dir.resolve()),
            "output_scene_dir": str(output_scene_dir.resolve()),
            "variant": f"raw_{roi}_crop",
            "target_size": int(target_size),
            "hardmask": bool(hardmask),
            "skip_prior": bool(skip_prior),
            "prior_shape": [] if skip_prior else list(np.asarray(prior_maps).shape),
            "prior_summary_shape": []
            if skip_prior
            else (list(np.asarray(prior_summary_tokens).shape) if prior_summary_tokens is not None else [len(written_views), 0, 0]),
            "crop_summaries": crop_summaries,
        }
        (output_scene_dir / f"highres_{roi}_scene_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return summary
    finally:
        if source_prior_payload is not None:
            source_prior_payload.close()


def main() -> int:
    args = parse_args()
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    output_scene_dir = Path(args.output_scene_dir).expanduser().resolve()
    manifest = _load_manifest(scene_dir)
    external_prior_bundle = Path(args.external_prior_bundle).expanduser().resolve() if args.external_prior_bundle.strip() else None
    smplx_model_dir = resolve_smplx_model_dir(args.smplx_model_dir if args.smplx_model_dir.strip() else None)
    dataset_root = Path(args.dataset_root).expanduser().resolve() if args.dataset_root.strip() else Path(manifest["dataset_root"]).expanduser().resolve()
    summary = _write_scene(
        scene_dir=scene_dir,
        output_scene_dir=output_scene_dir,
        manifest=manifest,
        roi=args.roi,
        bbox_scale=float(args.bbox_scale),
        bbox_pad=int(args.bbox_pad),
        target_size=int(args.target_size),
        bg_rgb=_background_rgb(args.mask_background),
        hardmask=bool(args.hardmask),
        external_prior_bundle=external_prior_bundle,
        rebuild_scene_prior=bool(args.rebuild_scene_prior),
        skip_prior=bool(args.skip_prior),
        dataset_root=dataset_root,
        subset_name=args.subset_name,
        smplx_model_dir=smplx_model_dir,
        smplx_gender=args.smplx_gender,
        mesh_fill_knn=int(args.mesh_fill_knn),
        summary_token_count=int(args.summary_token_count),
        sigma=float(args.sigma),
        min_conf=float(args.min_conf),
        vertex_id_dim=int(args.vertex_id_dim),
        body_part_dim=int(args.body_part_dim),
        body_part_count=int(args.body_part_count),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
