#!/usr/bin/env python3
"""Save before/after PPE preprocessing figures with correctly transformed boxes."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps


NAMES = ["Person", "helmet", "no_helmet", "vest", "no_vest"]
COLORS = ["#00A9FF", "#00C853", "#FF3D00", "#9C27B0", "#FFB300"]
PANEL = 550
PADDING = 22


def read_boxes(path):
    """Load normalized YOLO boxes as (class, cx, cy, width, height)."""
    boxes = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Invalid label in {path}: {line}")
        category = int(fields[0])
        coords = tuple(map(float, fields[1:]))
        if not 0 <= category < len(NAMES):
            raise ValueError(f"Invalid class in {path}: {line}")
        boxes.append((category, *coords))
    return boxes


def letterbox(image, boxes, size=640):
    """Resize without distortion and adjust boxes for the padding."""
    width, height = image.size
    scale = min(size / width, size / height)
    new_width, new_height = round(width * scale), round(height * scale)
    left, top = (size - new_width) // 2, (size - new_height) // 2
    output = Image.new("RGB", (size, size), "#555555")
    output.paste(image.resize((new_width, new_height), Image.Resampling.LANCZOS), (left, top))
    adjusted = [(c, (x * new_width + left) / size, (y * new_height + top) / size,
                 w * new_width / size, h * new_height / size) for c, x, y, w, h in boxes]
    return output, adjusted


def flip(image, boxes):
    """Mirror the image and every box's horizontal center."""
    return ImageOps.mirror(image), [(c, 1 - x, y, w, h) for c, x, y, w, h in boxes]


def draw_panel(image, boxes, heading, detail, font, small_font):
    panel = Image.new("RGB", (PANEL, PANEL + 92), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((PADDING, 15), heading, font=font, fill="#182433")
    view_height = PANEL - 50
    scale = min((PANEL - 2 * PADDING) / image.width, view_height / image.height)
    width, height = round(image.width * scale), round(image.height * scale)
    left = (PANEL - width) // 2
    top = 52 + (view_height - height) // 2
    panel.paste(image.resize((width, height), Image.Resampling.LANCZOS), (left, top))
    for category, x, y, w, h in boxes:
        x1, y1 = left + (x - w / 2) * width, top + (y - h / 2) * height
        x2, y2 = left + (x + w / 2) * width, top + (y + h / 2) * height
        color = COLORS[category]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = NAMES[category]
        text_box = draw.textbbox((0, 0), label, font=small_font)
        label_w, label_h = text_box[2] + 8, text_box[3] + 5
        label_y = max(top, y1 - label_h)
        draw.rectangle((x1, label_y, x1 + label_w, label_y + label_h), fill=color)
        draw.text((x1 + 4, label_y + 1), label, font=small_font, fill="white")
    draw.text((PADDING, PANEL + 44), detail, font=small_font, fill="#506070")
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data" / "construction-ppe-5class")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "figures")
    parser.add_argument("--images", nargs="*", default=[
        "ultralytics-ppe-nov2026-009.jpg",
        "source-8c4bd5367bc76d76cd877bc769fbd70a.jpg",
    ])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    font = ImageFont.truetype(font_path, 27)
    small_font = ImageFont.truetype(font_path, 17)
    for filename in args.images:
        source = args.data / "images" / "train" / filename
        label = args.data / "labels" / "train" / (Path(filename).stem + ".txt")
        if not source.is_file() or not label.is_file():
            raise FileNotFoundError(source if not source.is_file() else label)
        image = Image.open(source).convert("RGB")
        boxes = read_boxes(label)
        resized, resized_boxes = letterbox(image, boxes)
        flipped, flipped_boxes = flip(image, boxes)
        dark = ImageEnhance.Contrast(ImageEnhance.Brightness(image).enhance(0.65)).enhance(1.15)
        panels = [
            draw_panel(image, boxes, "Original", f"{image.width} x {image.height}; {len(boxes)} boxes", font, small_font),
            draw_panel(resized, resized_boxes, "Resize + padding", "640 x 640; boxes adjusted", font, small_font),
            draw_panel(flipped, flipped_boxes, "Horizontal flip", "x center becomes 1 - x", font, small_font),
            draw_panel(dark, boxes, "Brightness + contrast", "brightness 0.65; contrast 1.15", font, small_font),
        ]
        canvas = Image.new("RGB", (2 * PANEL + 18, 2 * (PANEL + 92) + 18), "#DDE4EB")
        for i, panel in enumerate(panels):
            canvas.paste(panel, ((i % 2) * (PANEL + 18), (i // 2) * (PANEL + 92 + 18)))
        output = args.output / f"preprocessing_{Path(filename).stem}.jpg"
        canvas.save(output, quality=93)
        print(output)


if __name__ == "__main__":
    main()
