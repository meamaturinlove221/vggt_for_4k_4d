from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import dna_4k4d as dna  # noqa: E402
from prepare_4k4d_prior_training_case import build_prior_stack, resolve_smplx_model_dir  # noqa: E402


def decode_encoded_image(buffer: np.ndarray) -> Image.Image:
    decoded = cv2.imdecode(np.asarray(buffer), cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("Failed to decode encoded image bytes.")
    rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def load_rgb_frame(main_smc: Path, camera_id: str, frame_id: str) -> Image.Image:
    cam_num = int(camera_id)
    group_name = "Camera_5mp" if cam_num < 48 else "Camera_12mp"
    cam_key = str(cam_num)
    frame_key = str(int(frame_id))
    with h5py.File(main_smc, "r") as handle:
        buffer = handle[group_name][cam_key]["color"][frame_key][()]
    return decode_encoded_image(buffer)


def load_mask_frame(ann_smc: Path, camera_id: str, frame_id: str) -> Image.Image:
    cam_key = str(int(camera_id))
    frame_key = str(int(frame_id))
    with h5py.File(ann_smc, "r") as handle:
        buffer = handle["Mask"][cam_key]["mask"][frame_key][()]
    rgb = decode_encoded_image(buffer)
    gray = np.max(np.asarray(rgb), axis=2).astype(np.uint8)
    return Image.fromarray(gray, mode="L")


def make_contact_sheet(images: list[Image.Image], labels: list[str], output_path: Path) -> None:
    cell_w = 320
    cols = 4
    rows = (len(images) + cols - 1) // cols
    resized = []
    heights = []
    for image in images:
        height = int(image.height * (cell_w / image.width))
        resized.append(image.resize((cell_w, height)))
        heights.append(height)
    cell_h = max(heights)
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
        draw.text((x + 6, y + cell_h + 2), label, fill=(240, 240, 240), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def resolve_smc(context: dna.DatasetContext, canonical_path: str, temp_dir: Path) -> Path:
    status = dna.describe_expected_file(context, canonical_path)
    if status["status"] == "extracted":
        return Path(status["path"])
    if status["status"] == "archived":
        materialized = dna.materialize_archived_member(context, canonical_path, temp_dir)
        if materialized is not None:
            return materialized
    raise FileNotFoundError(f"Could not resolve required file: {canonical_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export one 4K4D frame as a plain image scene for official VGGT inference.")
    parser.add_argument("--dataset-root", required=True, help="Extracted data_used_in_4K4D root or its parent folder.")
    parser.add_argument("--seq", required=True, help="Sequence id such as 0012_11")
    parser.add_argument("--frame", default="0", help="Frame id. Default: 0")
    parser.add_argument("--target-camera", default="00", help="Target camera id. Default: 00")
    parser.add_argument("--source-cameras", nargs="*", default=[], help="Explicit source camera ids.")
    parser.add_argument("--auto-sources", type=int, default=6, help="Auto-pick N source cameras if none are provided.")
    parser.add_argument("--output-dir", required=True, help="Output scene directory.")
    parser.add_argument("--target-size", type=int, default=518, help="Target square size for exported prior maps.")
    parser.add_argument("--mask-only", action="store_true", help="Use silhouette-only non-SMPL channels.")
    parser.add_argument("--sigma", type=float, default=6.0, help="Gaussian sigma for 2D keypoint heatmaps.")
    parser.add_argument("--min-conf", type=float, default=0.05, help="Minimum confidence for 2D keypoint heatmaps.")
    parser.add_argument("--smplx-model-dir", help="Directory containing SMPL-X model files such as SMPLX_NEUTRAL.npz.")
    parser.add_argument("--smplx-gender", choices=("neutral", "female", "male"), default="neutral")
    parser.add_argument("--mesh-fill-knn", type=int, default=4, help="KNN used to densify SMPL-X feature priors inside the silhouette.")
    parser.add_argument("--summary-token-count", type=int, default=16, help="Number of pooled SMPL-X summary tokens per view.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing exported images.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    context = dna.build_context(Path(args.dataset_root), dna.SUBSET_NAME)
    frame_id = str(int(args.frame))
    target_camera = dna.normalize_camera_id(args.target_camera)
    output_dir = Path(args.output_dir).resolve()
    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dna_4k4d_export_") as temp_name:
        temp_dir = Path(temp_name)
        main_smc = resolve_smc(context, f"{dna.SUBSET_NAME}/main/{args.seq}.smc", temp_dir)
        ann_smc = resolve_smc(context, f"{dna.SUBSET_NAME}/annotations/{args.seq}_annots.smc", temp_dir)
        rgb_cams_smc, rgb_cams_source = dna.materialize_rgb_cams_smc(context, args.seq, temp_dir)
        if rgb_cams_smc is None:
            raise FileNotFoundError(f"Could not resolve rgb_cams SMC for {args.seq}")
        camera_summary = dna.load_camera_summary(rgb_cams_smc)
        available_cameras = list(camera_summary["camera_ids"])

        source_cameras = [dna.normalize_camera_id(camera) for camera in args.source_cameras]
        if not source_cameras:
            source_cameras = dna.auto_pick_sources(available_cameras, target_camera, args.auto_sources)
        selected_cameras = [target_camera, *source_cameras]

        labels = []
        rgb_images = []
        mask_images = []
        exported = []
        for idx, camera_id in enumerate(selected_cameras):
            role = "tgt" if idx == 0 else "src"
            prefix = f"{idx:02d}_{role}_cam{camera_id}"
            rgb_image = load_rgb_frame(main_smc, camera_id, frame_id)
            mask_image = load_mask_frame(ann_smc, camera_id, frame_id)

            rgb_path = image_dir / f"{prefix}.png"
            mask_path = mask_dir / f"{prefix}.png"
            if not args.overwrite and (rgb_path.exists() or mask_path.exists()):
                raise FileExistsError(f"{rgb_path} already exists. Re-run with --overwrite.")

            rgb_image.save(rgb_path)
            mask_image.save(mask_path)

            labels.append(f"{camera_id} ({role})")
            rgb_images.append(rgb_image)
            mask_images.append(mask_image.convert("RGB"))
            exported.append(
                {
                    "camera_id": camera_id,
                    "role": role,
                    "image_path": str(rgb_path),
                    "mask_path": str(mask_path),
                    "image_size": list(rgb_image.size),
                    "mask_coverage": float((np.asarray(mask_image) > 0).mean()),
                }
            )

        make_contact_sheet(rgb_images, labels, output_dir / "rgb_contact_sheet.png")
        make_contact_sheet(mask_images, labels, output_dir / "mask_contact_sheet.png")

        scene_manifest = {
            "dataset_root": str(context.subset_roots[0] if context.subset_roots else context.dataset_path),
            "seq_id": args.seq,
            "frame_id": frame_id,
            "target_camera": target_camera,
            "source_cameras": source_cameras,
            "camera_summary": camera_summary,
            "rgb_cams_source": rgb_cams_source,
            "main_smc": str(main_smc),
            "annotations_smc": str(ann_smc),
            "exported_views": exported,
        }

        smplx_model_dir = resolve_smplx_model_dir(args.smplx_model_dir)
        if smplx_model_dir is None:
            raise FileNotFoundError(
                "SMPL-X prior export now requires model files. Pass --smplx-model-dir or set VGGT_SMPLX_MODEL_DIR."
            )

        prior_maps, prior_mask, prior_summary_tokens, _, _, prior_input_meta = build_prior_stack(
            scene_manifest=scene_manifest,
            dataset_root=Path(scene_manifest["dataset_root"]),
            subset_name=dna.SUBSET_NAME,
            target_size=int(args.target_size),
            mask_only=bool(args.mask_only),
            sigma=float(args.sigma),
            min_conf=float(args.min_conf),
            smplx_model_dir=smplx_model_dir,
            smplx_gender=args.smplx_gender,
            mesh_fill_knn=int(args.mesh_fill_knn),
            summary_token_count=int(args.summary_token_count),
        )
        vertex_feature_meta = prior_input_meta.get("smplx_vertex_feature_meta", {})
        if not bool(vertex_feature_meta.get("enabled")):
            raise RuntimeError(f"Failed to export dense SMPL-X vertex priors: {vertex_feature_meta}")

        prior_npz_path = output_dir / "prior_maps.npz"
        np.savez_compressed(
            prior_npz_path,
            prior_maps=prior_maps.astype(np.float16),
            prior_summary_tokens=prior_summary_tokens.astype(np.float16),
            prior_mask=prior_mask.astype(bool),
            prior_channels=np.asarray(prior_input_meta["channel_names"]),
            prior_summary_channels=np.asarray(prior_input_meta.get("summary_channel_names", [])),
        )
        scene_manifest["prior_maps_file"] = prior_npz_path.name
        scene_manifest["prior_channels"] = prior_input_meta["channel_names"]
        scene_manifest["prior_summary_channels"] = prior_input_meta.get("summary_channel_names", [])
        scene_manifest["prior_summary_token_count"] = 0 if prior_summary_tokens is None else int(prior_summary_tokens.shape[1])
        scene_manifest["prior_input_meta"] = prior_input_meta

        with (output_dir / "scene_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(scene_manifest, handle, indent=2, ensure_ascii=False)

    print(f"Exported {len(selected_cameras)} views to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
