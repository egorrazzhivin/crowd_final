"""Local bbox inference. Dependencies are listed in requirements.txt.

Input JSONL: {"photo_id": 123, "photo_url": "https://.../original.jpg"}
Detector prompts: one query per line, e.g. "a butterfly" and "a moth".
Qwen prompts: a full instruction; the JSON/coordinate contract is appended below.
Inputs can be paths or filenames in data/ and prompts/.
Outputs: data/predictions/<dataset>_<model>_threshold-<value>/{predictions.jsonl,log.jsonl,run.json}.
GroundingDINO directories also include _nms-<IoU> or _nms-off.
"""

import argparse
import gzip
import hashlib
import io
import json
import math
import sys
import time
from importlib.metadata import version
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PROMPT_DIR = ROOT / "prompts"
IMAGE_CACHE = ROOT / ".image_cache"

MODELS = {
    "grounding-dino": "IDEA-Research/grounding-dino-tiny",
    "owlv2": "google/owlv2-base-patch16",
    "qwen": "Qwen/Qwen3-VL-2B-Instruct",
}
QWEN_FORMAT = """
Return only a JSON array of objects with a bbox_2d field, for example:
[{"bbox_2d": [100, 200, 400, 600]}].
Each box is [x_min, y_min, x_max, y_max], with coordinates relative to the
image width and height on a scale from 0 to 1000. Do not return pixel coordinates.
Return [] if there are no matching objects. Do not include Markdown or explanations.
""".strip()


def resolve_input(path, directory):
    if path.is_file():
        return path.resolve()
    if not path.is_absolute() and path.parent == Path("."):
        candidate = directory / path
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Input file not found: {path}")


def open_text(path):
    path = Path(path)
    if not path.exists() and path.suffix != ".gz":
        path = path.with_name(path.name + ".gz")
    return gzip.open(path, "rt", encoding="utf-8-sig") if path.suffix == ".gz" else path.open(encoding="utf-8-sig")


def result_directory(args):
    threshold = "none" if args.threshold is None else str(args.threshold)
    dataset_stem = Path(args.dataset.name.removesuffix(".gz")).stem
    name = f"{dataset_stem}_{args.model}_threshold-{threshold}"
    if args.model == "grounding-dino":
        name += f"_nms-{args.nms_iou if args.nms_iou is not None else 'off'}"
    tag = getattr(args, "tag", None)
    if tag:
        name += f"__{tag}"
    return DATA_DIR / "predictions" / name


def read_dataset(path):
    rows, ids = [], set()
    with open_text(path) as file:
        for number, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            photo_id, url = row["photo_id"], row["photo_url"]
            if type(photo_id) is not int or photo_id in ids:
                raise ValueError(f"Invalid or duplicate photo_id on line {number}")
            if not isinstance(url, str) or not url.startswith(("https://", "http://")):
                raise ValueError(f"Invalid photo_url on line {number}")
            ids.add(photo_id)
            rows.append({"photo_id": photo_id, "photo_url": url})
    if not rows:
        raise ValueError("Dataset is empty")
    return rows


def load_image(url, cache, max_side):
    from PIL import Image, ImageOps

    cache.mkdir(parents=True, exist_ok=True)
    path = cache / hashlib.sha256(url.encode()).hexdigest()
    if not path.exists():
        request = Request(url, headers={"User-Agent": "ButterflyCourseProject/1.0"})
        for attempt in range(4):
            try:
                with urlopen(request, timeout=30) as response:
                    content = response.read()
                break
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
        path.write_bytes(content)
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def normalize_boxes(boxes, width, height):
    result = []
    for box in boxes:
        if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
            raise ValueError(f"Invalid box: {box}")
        x1, y1, x2, y2 = map(float, box)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Inverted or empty box: {box}")
        x1, x2 = (min(1.0, max(0.0, x / width)) for x in (x1, x2))
        y1, y2 = (min(1.0, max(0.0, y / height)) for y in (y1, y2))
        if x2 > x1 and y2 > y1:
            result.append({"left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1})
    return result


def nms_indices(boxes, scores, iou_threshold):
    import torch
    from torchvision.ops import nms

    if len(boxes) != len(scores):
        raise ValueError("Boxes and scores have different lengths")
    if not boxes:
        return []
    # CPU NMS avoids depending on an MPS implementation of torchvision's custom op.
    return nms(torch.tensor(boxes, dtype=torch.float32),
               torch.tensor(scores, dtype=torch.float32), iou_threshold).tolist()


def parse_qwen(text):
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text[text.index("\n") + 1:-3].strip()
    objects = json.loads(text)
    if not isinstance(objects, list):
        raise ValueError("Qwen must return a JSON array")
    boxes = []
    for obj in objects:
        box = obj["bbox_2d"]
        if len(box) != 4 or not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1000 for v in box):
            raise ValueError(f"Invalid Qwen coordinates: {box}")
        boxes.append(box)
    return normalize_boxes(boxes, 1000, 1000)


def load_model(args, prompt):
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    model_id = MODELS[args.model]
    pretrained = {"revision": args.revision, "cache_dir": str(ROOT / ".model_cache" / "hub")}
    processor = AutoProcessor.from_pretrained(model_id, **pretrained)
    effective_prompt = prompt + "\n\n" + QWEN_FORMAT if args.model == "qwen" else prompt
    if args.model == "qwen":
        from transformers import Qwen3VLForConditionalGeneration

        dtype = torch.float32 if device == "cpu" else torch.float16
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, **pretrained, dtype=dtype, attn_implementation="sdpa"
        )
    else:
        options = {"disable_custom_kernels": True} if args.model == "grounding-dino" else {}
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, **pretrained, **options)
    model = model.to(device).eval()

    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    def predict(image, retry=False):
        with torch.inference_mode():
            if args.model == "qwen":
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": effective_prompt},
                ]}]
                inputs = processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt",
                ).to(device)
                options = {"do_sample": True, "temperature": 0.2, "top_p": 0.9} if retry else {"do_sample": False}
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, **options)
                raw = processor.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                return {"raw": raw}
            queries = [line.strip().rstrip(".") for line in prompt.splitlines() if line.strip()]
            text = ". ".join(queries) + "." if args.model == "grounding-dino" else [queries]
            inputs = processor(images=image, text=text, return_tensors="pt").to(device)
            outputs = model(**inputs)
            # OWLv2 pads on the bottom/right to a square; normalize back to the actual image.
            size = (max(image.size),) * 2 if args.model == "owlv2" else image.size[::-1]
            options = {"input_ids": inputs.input_ids} if args.model == "grounding-dino" else {}
            detected = processor.post_process_grounded_object_detection(
                outputs, threshold=args.threshold, target_sizes=[size], **options,
            )[0]
            return {"xyxy": detected["boxes"].detach().cpu().tolist(), "scores": detected["scores"].detach().cpu().tolist()}

    metadata = {
        "model_id": model_id, "resolved_revision": model.config._commit_hash,
        "device": device, "dtype": str(model.dtype), "effective_prompt": effective_prompt,
        "versions": {name: version(name) for name in ("torch", "torchvision", "transformers", "pillow", "scipy")},
    }
    if args.model == "qwen":
        metadata["json_retry_policy"] = {"max_attempts": 3, "temperature": 0.2, "top_p": 0.9}
    return predict, synchronize, metadata


def run(args):
    rows = read_dataset(args.dataset)
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("Prompt is empty")
    if args.limit:
        rows = rows[:args.limit]
    directory = result_directory(args)
    output, log_path, meta_path = (directory / name for name in ("predictions.jsonl", "log.jsonl", "run.json"))
    for path in (output, log_path, meta_path):
        if path.resolve() in (args.dataset.resolve(), args.prompt.resolve()) or path.exists() or path.with_name(path.name + ".gz").exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    predict, synchronize, metadata = load_model(args, prompt)
    metadata.update({
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "prompt": prompt, "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "photo_count": len(rows), "seed": 0, "exif_transpose": True,
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    errors = 0
    with output.open("x", encoding="utf-8") as predictions, log_path.open("x", encoding="utf-8") as log:
        for index, row in enumerate(rows, 1):
            started, raw, inference_seconds = time.perf_counter(), None, None
            image_size = None
            attempts = []
            result = {**row, "boxes": None, "error": None}
            try:
                image = load_image(row["photo_url"], IMAGE_CACHE, args.max_image_side)
                image_size = list(image.size)
                synchronize()
                inference_started = time.perf_counter()
                if args.model == "qwen":
                    for attempt in range(3):
                        raw = predict(image, retry=attempt > 0)
                        synchronize()
                        inference_seconds = time.perf_counter() - inference_started
                        try:
                            result["boxes"] = parse_qwen(raw["raw"])
                        except (ValueError, TypeError, KeyError) as error:
                            attempts.append({"raw": raw["raw"], "error": f"{type(error).__name__}: {error}"})
                            if attempt == 2:
                                raise
                            print(f"Photo {row['photo_id']}: invalid Qwen JSON, retry {attempt + 1}/2", file=sys.stderr)
                        else:
                            break
                else:
                    raw = predict(image)
                    synchronize()
                    inference_seconds = time.perf_counter() - inference_started
                    boxes = raw["xyxy"]
                    if args.model == "grounding-dino" and args.nms_iou is not None:
                        raw["nms_kept_indices"] = nms_indices(boxes, raw["scores"], args.nms_iou)
                        boxes = [boxes[i] for i in raw["nms_kept_indices"]]
                    result["boxes"] = normalize_boxes(boxes, *image.size)
            except Exception as error:
                errors += 1
                result["error"] = f"{type(error).__name__}: {error}"
            predictions.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            predictions.flush()
            log.write(json.dumps({
                "photo_id": row["photo_id"], "seconds": time.perf_counter() - started,
                "inference_seconds": inference_seconds, "image_size": image_size,
                "prediction": raw, "error": result["error"],
                "failed_json_attempts": attempts,
            }, ensure_ascii=False) + "\n")
            log.flush()
            print(f"[{index}/{len(rows)}] {row['photo_id']}: {result['error'] or str(len(result['boxes'])) + ' boxes'}", file=sys.stderr)
    print(f"Saved {output}; errors: {errors}/{len(rows)}", file=sys.stderr)
    return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, help="Dataset path or filename in data/")
    parser.add_argument("prompt", type=Path, help="Prompt path or filename in prompts/")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--revision", default="main", help="Hugging Face revision or commit hash")
    parser.add_argument("--threshold", type=float, help="Detector confidence threshold; not applicable to Qwen")
    parser.add_argument("--nms-iou", type=float, help="Enable GroundingDINO NMS with this IoU threshold; disabled by default")
    parser.add_argument("--max-new-tokens", type=int, help="Qwen only; defaults to 2048")
    parser.add_argument("--max-image-side", type=int, default=1280)
    parser.add_argument("--limit", type=int, help="Process only the first N photos for a pilot")
    parser.add_argument("--tag", default=None, help="Optional suffix appended to the output directory name")
    args = parser.parse_args()
    if args.model == "grounding-dino":
        if args.nms_iou is not None and not 0 <= args.nms_iou <= 1:
            parser.error("--nms-iou must be between 0 and 1")
    elif args.nms_iou is not None:
        parser.error("NMS options are only used by GroundingDINO")
    if args.model == "qwen":
        if args.threshold is not None:
            parser.error("Qwen has no confidence threshold; omit --threshold")
        if args.max_new_tokens is None:
            args.max_new_tokens = 2048
    elif args.max_new_tokens is not None:
        parser.error("--max-new-tokens is only used by Qwen")
    if args.threshold is None and args.model != "qwen":
        args.threshold = 0.1 if args.model == "owlv2" else 0.25
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    if args.max_image_side <= 0 or (args.max_new_tokens is not None and args.max_new_tokens <= 0) or (args.limit is not None and args.limit <= 0):
        parser.error("Image size, token limit and photo limit must be positive")
    try:
        args.dataset = resolve_input(args.dataset, DATA_DIR)
        args.prompt = resolve_input(args.prompt, PROMPT_DIR)
        return run(args)
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(1, f"{type(error).__name__}: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
