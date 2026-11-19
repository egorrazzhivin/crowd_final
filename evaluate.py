"""Evaluate local predictions against ours_gold; never reads contest gold."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from label import DATA_DIR, normalize_boxes, open_text, read_dataset


def read_rows(path):
    rows = {}
    with open_text(path) as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            photo_id = row["photo_id"]
            if type(photo_id) is not int or photo_id in rows:
                raise ValueError(f"Invalid or duplicate photo_id in {path}")
            if row.get("error"):
                raise ValueError(f"Photo {photo_id} has an error in {path}")
            rows[photo_id] = row
    return rows


def coordinates(boxes):
    result = []
    for box in boxes:
        x, y, width, height = (box[key] for key in ("left", "top", "width", "height"))
        if not all(math.isfinite(v) for v in (x, y, width, height)):
            raise ValueError("Nonfinite box coordinates")
        if min(x, y) < 0 or min(width, height) <= 0 or x + width > 1 + 1e-6 or y + height > 1 + 1e-6:
            raise ValueError("Invalid normalized box")
        result.append((x, y, x + width, y + height))
    return np.asarray(result, dtype=float).reshape(-1, 4)


def counts(predicted, gold, threshold):
    predicted, gold = coordinates(predicted), coordinates(gold)
    if not len(predicted) or not len(gold):
        return 0, len(predicted), len(gold)
    low = np.maximum(predicted[:, None, :2], gold[None, :, :2])
    high = np.minimum(predicted[:, None, 2:], gold[None, :, 2:])
    intersection = np.maximum(high - low, 0).prod(axis=2)
    area_p = (predicted[:, 2:] - predicted[:, :2]).prod(axis=1)
    area_g = (gold[:, 2:] - gold[:, :2]).prod(axis=1)
    iou = intersection / (area_p[:, None] + area_g[None, :] - intersection)
    # Maximize valid one-to-one matches first, then their total IoU.
    valid = iou >= threshold
    weights = valid * (min(iou.shape) + 1 + iou)
    p, g = linear_sum_assignment(weights, maximize=True)
    tp = int(valid[p, g].sum())
    return tp, len(predicted) - tp, len(gold) - tp


def scores(tp, fp, fn):
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def evaluate(gold, predicted):
    missing = gold.keys() - predicted.keys()
    if missing:
        raise ValueError(f"Missing predictions for {len(missing)} gold photos")
    metrics, per_photo = {}, []
    for threshold in (0.5, 0.75):
        total = np.zeros(3, dtype=int)
        for photo_id, row in gold.items():
            values = counts(predicted[photo_id], row["boxes"], threshold)
            total += values
            per_photo.append({"photo_id": photo_id, "iou": threshold, **scores(*values)})
        metrics[str(threshold)] = scores(*(int(v) for v in total))
    negatives = [photo_id for photo_id, row in gold.items() if not row["boxes"]]
    metrics["negative_photos"] = {
        "count": len(negatives),
        "with_false_positives": sum(bool(predicted[i]) for i in negatives),
        "false_positive_boxes": sum(len(predicted[i]) for i in negatives),
    }
    return {"metrics": metrics, "per_photo": per_photo}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dino-threshold", type=float, default=0.25)
    parser.add_argument("--owl-threshold", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "metrics/ours_baseline.json")
    args = parser.parse_args()
    if not all(0 <= v <= 1 for v in (args.dino_threshold, args.owl_threshold)):
        parser.error("Thresholds must be between 0 and 1")
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    gold_path = DATA_DIR / "gold/ours_gold.jsonl"
    gold = read_rows(gold_path)
    dataset = read_dataset(DATA_DIR / "dataset_ours.jsonl")
    source_ids = {row["photo_id"] for row in dataset}
    if not gold or gold.keys() - source_ids:
        raise ValueError("Gold is empty or contains unknown photo IDs")
    report = {
        "gold": str(gold_path), "gold_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
        "evaluated_photos": len(gold), "dataset_photos": len(dataset),
        "unannotated_photo_ids": sorted(source_ids - gold.keys()),
        "gold_boxes": sum(len(row["boxes"]) for row in gold.values()),
        "matching": "maximum-cardinality one-to-one IoU matching; micro metrics; IoU >= cutoff",
        "nms": False, "models": {},
    }
    for model, suffix, threshold in (
        ("grounding-dino", "grounding-dino_threshold-0.0_nms-off", args.dino_threshold),
        ("owlv2", "owlv2_threshold-0.0", args.owl_threshold),
        ("qwen", "qwen_threshold-none", None),
    ):
        folder = DATA_DIR / "predictions" / f"dataset_ours_{suffix}"
        predicted = {}
        if model == "qwen":
            predicted = {i: row["boxes"] for i, row in read_rows(folder / "predictions.jsonl").items()}
        else:
            # Stream raw candidate logs instead of retaining millions of boxes in memory.
            with open_text(folder / "log.jsonl") as file:
                for line in file:
                    row = json.loads(line)
                    photo_id = row["photo_id"]
                    if photo_id not in gold:
                        continue
                    if row.get("error") or photo_id in predicted:
                        raise ValueError(f"Invalid detector record: {photo_id}")
                    raw = row["prediction"]
                    if len(raw["xyxy"]) != len(raw["scores"]):
                        raise ValueError("Boxes and scores have different lengths")
                    boxes = [box for box, score in zip(raw["xyxy"], raw["scores"]) if score > threshold]
                    predicted[photo_id] = normalize_boxes(boxes, *row["image_size"])
        report["models"][model] = {
            "threshold": threshold, "source": str(folder), **evaluate(gold, predicted),
        }
        print(model, json.dumps(report["models"][model]["metrics"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
