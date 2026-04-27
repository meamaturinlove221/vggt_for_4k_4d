from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt.utils.normal_refiner import normal_to_rgb  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RGB / normal / diff visual packs from predictions.npz.")
    parser.add_argument("--predictions-npz", required=True, help="npz containing rgb and normal maps")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rgb-key", default="rgb")
    parser.add_argument("--coarse-key", default="coarse_prior_normal")
    parser.add_argument("--refined-key", default="refined_normal")
    parser.add_argument("--teacher-key", default="teacher_normal")
    parser.add_argument("--human-mask-key", default="human_mask")
    parser.add_argument("--teacher-mask-key", default="teacher_mask")
    return parser.parse_args()


def save_contact_sheet(images: list[np.ndarray], output_path: Path, thumb_size: int = 256) -> None:
    if not images:
        return
    cols = min(4, len(images))
    rows = math.ceil(len(images) / cols)
    canvas = Image.new("RGB", (cols * thumb_size, rows * thumb_size), color=(255, 255, 255))
    for idx, image in enumerate(images):
        tile = Image.fromarray(image).resize((thumb_size, thumb_size), Image.Resampling.BILINEAR)
        x = (idx % cols) * thumb_size
        y = (idx // cols) * thumb_size
        canvas.paste(tile, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def diff_to_rgb(refined: np.ndarray, coarse: np.ndarray) -> np.ndarray:
    diff = np.abs(refined - coarse).mean(axis=-1)
    diff = np.clip(diff / max(1e-6, float(diff.max())), 0.0, 1.0)
    return np.stack(
        [
            (diff * 255.0).astype(np.uint8),
            np.zeros_like(diff, dtype=np.uint8),
            ((1.0 - diff) * 255.0).astype(np.uint8),
        ],
        axis=-1,
    )


def main() -> int:
    args = parse_args()
    payload = np.load(args.predictions_npz, allow_pickle=False)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb = payload[args.rgb_key]
    coarse = payload[args.coarse_key]
    refined = payload[args.refined_key] if args.refined_key in payload.files else None
    teacher = payload[args.teacher_key] if args.teacher_key in payload.files else None
    human_mask = payload[args.human_mask_key]
    teacher_mask = payload[args.teacher_mask_key] if args.teacher_mask_key in payload.files else human_mask

    rgb_images: list[np.ndarray] = []
    coarse_images: list[np.ndarray] = []
    refined_images: list[np.ndarray] = []
    teacher_images: list[np.ndarray] = []
    diff_images: list[np.ndarray] = []

    count = rgb.shape[0]
    for idx in range(count):
        rgb_img = rgb[idx].astype(np.uint8)
        coarse_img = normal_to_rgb(coarse[idx].astype(np.float32), np.asarray(human_mask[idx]).squeeze() > 0.5)
        rgb_images.append(rgb_img)
        coarse_images.append(coarse_img)
        Image.fromarray(rgb_img).save(output_dir / f"{idx:02d}_rgb.png")
        Image.fromarray(coarse_img).save(output_dir / f"{idx:02d}_coarse_prior_normal.png")

        strip = [rgb_img, coarse_img]
        if refined is not None:
            refined_img = normal_to_rgb(refined[idx].astype(np.float32), np.asarray(human_mask[idx]).squeeze() > 0.5)
            refined_images.append(refined_img)
            Image.fromarray(refined_img).save(output_dir / f"{idx:02d}_refined_normal.png")
            strip.append(refined_img)

            diff_img = diff_to_rgb(refined[idx].astype(np.float32), coarse[idx].astype(np.float32))
            diff_images.append(diff_img)
            Image.fromarray(diff_img).save(output_dir / f"{idx:02d}_coarse_vs_refined_diff.png")
            strip.append(diff_img)

        if teacher is not None:
            teacher_img = normal_to_rgb(teacher[idx].astype(np.float32), np.asarray(teacher_mask[idx]).squeeze() > 0.5)
            teacher_images.append(teacher_img)
            Image.fromarray(teacher_img).save(output_dir / f"{idx:02d}_teacher_normal.png")
            strip.append(teacher_img)

        Image.fromarray(np.concatenate(strip, axis=1)).save(output_dir / f"{idx:02d}_summary_strip.png")

    save_contact_sheet(rgb_images, output_dir / "00_rgb_contact_sheet.png")
    save_contact_sheet(coarse_images, output_dir / "01_coarse_prior_normal_contact_sheet.png")
    if refined_images:
        save_contact_sheet(refined_images, output_dir / "02_refined_normal_contact_sheet.png")
    if diff_images:
        save_contact_sheet(diff_images, output_dir / "03_coarse_vs_refined_diff_contact_sheet.png")
    if teacher_images:
        save_contact_sheet(teacher_images, output_dir / "04_teacher_normal_contact_sheet.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
