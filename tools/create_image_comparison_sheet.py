from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a labeled image comparison sheet.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--title", default="", help="Optional sheet title.")
    parser.add_argument("--cols", type=int, default=2, help="Number of columns.")
    parser.add_argument(
        "--item",
        action="append",
        default=[],
        help="Comparison item in the form 'Label::/path/to/image.png'. Repeatable.",
    )
    parser.add_argument("--padding", type=int, default=48, help="Outer and inner padding in pixels.")
    parser.add_argument("--caption-height", type=int, default=110, help="Caption strip height per tile.")
    parser.add_argument("--title-height", type=int, default=140, help="Title strip height.")
    parser.add_argument("--bg", default="#ffffff", help="Background color.")
    parser.add_argument("--fg", default="#111111", help="Foreground/text color.")
    parser.add_argument("--border", default="#d0d0d0", help="Tile border color.")
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def parse_items(raw_items: list[str]) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for raw in raw_items:
        if "::" not in raw:
            raise ValueError(f"Invalid --item format: {raw}")
        label, path_str = raw.split("::", 1)
        path = Path(path_str).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        items.append((label.strip(), path))
    if not items:
        raise ValueError("Please provide at least one --item.")
    return items


def main() -> int:
    args = parse_args()
    items = parse_items(args.item)

    images = [(label, Image.open(path).convert("RGB")) for label, path in items]
    max_width = max(image.width for _, image in images)
    max_height = max(image.height for _, image in images)

    cols = max(1, int(args.cols))
    rows = (len(images) + cols - 1) // cols
    padding = int(args.padding)
    caption_height = int(args.caption_height)
    title_height = int(args.title_height) if args.title else 0

    tile_width = max_width
    tile_height = caption_height + max_height
    canvas_width = padding + cols * tile_width + (cols - 1) * padding + padding
    canvas_height = padding + title_height + rows * tile_height + (rows - 1) * padding + padding

    canvas = Image.new("RGB", (canvas_width, canvas_height), color=args.bg)
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(max(28, title_height // 3 if title_height else 28))
    caption_font = load_font(max(22, caption_height // 3))

    if args.title:
        bbox = draw.textbbox((0, 0), args.title, font=title_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        title_x = (canvas_width - text_w) // 2
        title_y = padding + (title_height - text_h) // 2 - 8
        draw.text((title_x, title_y), args.title, fill=args.fg, font=title_font)

    top_offset = padding + title_height
    for idx, (label, image) in enumerate(images):
        row = idx // cols
        col = idx % cols
        tile_x = padding + col * (tile_width + padding)
        tile_y = top_offset + row * (tile_height + padding)
        image_x = tile_x + (tile_width - image.width) // 2
        image_y = tile_y + caption_height + (max_height - image.height) // 2

        draw.rectangle(
            (tile_x, tile_y, tile_x + tile_width - 1, tile_y + tile_height - 1),
            outline=args.border,
            width=2,
        )
        bbox = draw.textbbox((0, 0), label, font=caption_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = tile_x + (tile_width - text_w) // 2
        text_y = tile_y + (caption_height - text_h) // 2 - 4
        draw.text((text_x, text_y), label, fill=args.fg, font=caption_font)
        canvas.paste(image, (image_x, image_y))

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"Saved comparison sheet to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
