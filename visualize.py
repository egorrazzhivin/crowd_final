"""Draw one photo's normalized boxes from a predictions JSONL. Requires Pillow."""

import argparse
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from label import IMAGE_CACHE, load_image, open_text, read_dataset


def render_prediction(row, image):
    boxes = row.get("boxes")
    if row.get("error") or not isinstance(boxes, list):
        raise ValueError(f"Photo {row['photo_id']} has no successful prediction: {row.get('error')}")
    width, height = image.size
    canvas = Image.new("RGB", (width, height + 48), "#17212b")
    canvas.paste(image, (0, 48))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    draw.text((12, 14), f"Photo {row['photo_id']} | {len(boxes)} detected boxes", fill="white", font=font)
    line_width = max(2, round(min(width, height) / 250))
    for index, box in enumerate(boxes, 1):
        left, top, bw, bh = (box[key] for key in ("left", "top", "width", "height"))
        if not all(type(v) in (int, float) and math.isfinite(v) for v in (left, top, bw, bh)):
            raise ValueError(f"Invalid box {index}: {box}")
        if not (0 <= left < 1 and 0 <= top < 1 and bw > 0 and bh > 0
                and left + bw <= 1 + 1e-6 and top + bh <= 1 + 1e-6):
            raise ValueError(f"Box {index} is outside normalized image coordinates: {box}")
        x1, y1 = min(width - 1, round(left * width)), 48 + min(height - 1, round(top * height))
        x2 = min(width - 1, round((left + bw) * width))
        y2 = 48 + min(height - 1, round((top + bh) * height))
        draw.rectangle((x1, y1, x2, y2), outline="#00ff88", width=line_width)
        label = f"#{index}"
        label_width = draw.textbbox((0, 0), label, font=font)[2] + 10
        label_y = max(48, y1 - 24)
        label_x = min(x1, max(0, width - label_width))
        draw.rectangle((label_x, label_y, label_x + label_width, label_y + 24), fill="#00ff88")
        draw.text((label_x + 5, label_y + 3), label, fill="#082218", font=font)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="Predictions JSONL or a run directory")
    parser.add_argument("--photo-id", type=int, help="Defaults to the first prediction")
    parser.add_argument("--dataset", type=Path, help="Optional URL source if predictions omit photo_url")
    parser.add_argument("--output", type=Path, help="Output PNG path")
    parser.add_argument("--max-side", type=int, default=1600)
    args = parser.parse_args()
    if args.max_side <= 0:
        parser.error("--max-side must be positive")
    try:
        if args.predictions.is_dir():
            args.predictions = args.predictions / "predictions.jsonl"
        with open_text(args.predictions) as file:
            matches = [r for line in file if line.strip()
                       for r in [json.loads(line)] if args.photo_id is None or r["photo_id"] == args.photo_id]
        if not matches:
            raise ValueError("No matching photo found")
        if args.photo_id is not None and len(matches) != 1:
            raise ValueError("Duplicate photo_id in predictions")
        row = matches[0]
        url = row.get("photo_url")
        if not url and args.dataset:
            url = next((r["photo_url"] for r in read_dataset(args.dataset) if r["photo_id"] == row["photo_id"]), None)
        if not url:
            raise ValueError("photo_url is missing; pass --dataset")
        if row.get("error") or not isinstance(row.get("boxes"), list):
            raise ValueError(f"Photo has no successful prediction: {row.get('error')}")
        image = load_image(url, IMAGE_CACHE, args.max_side)
        canvas = render_prediction(row, image)
        output = args.output or args.predictions.with_name(f"{args.predictions.stem}_{row['photo_id']}.png")
        inputs = [args.predictions] + ([args.dataset] if args.dataset else [])
        if output.exists() or output.resolve() in {path.resolve() for path in inputs}:
            raise FileExistsError(f"Refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output, format="PNG")
        print(output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"{type(error).__name__}: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
