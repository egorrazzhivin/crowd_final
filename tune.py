"""Tune score threshold and NMS for DINO/OWLv2 on ours_gold.

Reads the saved raw candidate logs (xyxy + scores + image_size) from
data/predictions/dataset_ours_<model>_threshold-0.0..., applies the score
threshold and optional greedy NMS on raw pixel boxes, normalizes via
label.normalize_boxes, and evaluates with evaluate.evaluate (same matching
code as baseline). No model inference is performed.
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate import evaluate, read_rows
from label import DATA_DIR, normalize_boxes, open_text


MODEL_FOLDERS = {
    "grounding-dino": "dataset_ours_grounding-dino_threshold-0.0_nms-off",
    "owlv2": "dataset_ours_owlv2_threshold-0.0",
}


def load_candidates(folder: Path) -> dict[int, tuple[list, list, list]]:
    result: dict[int, tuple[list, list, list]] = {}
    with open_text(folder / "log.jsonl") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            photo_id = row["photo_id"]
            if row.get("error"):
                raise ValueError(f"Photo {photo_id} has an error in {folder}")
            raw = row["prediction"]
            if len(raw["xyxy"]) != len(raw["scores"]):
                raise ValueError(f"Boxes and scores differ for {photo_id}")
            result[photo_id] = (raw["xyxy"], raw["scores"], row["image_size"])
    return result


def nms_numpy(boxes, scores, iou_threshold: float) -> np.ndarray:
    """Greedy NMS; matches torchvision.ops.nms semantics (suppress IoU > threshold)."""
    if len(boxes) == 0:
        return np.empty(0, dtype=int)
    boxes = np.asarray(boxes, dtype=float)
    scores = np.asarray(scores, dtype=float)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=int)


def build_predicted(candidates, gold, threshold: float, nms_iou: float | None) -> dict:
    predicted = {}
    for photo_id in gold:
        if photo_id not in candidates:
            raise ValueError(f"Missing candidates for {photo_id}")
        xyxy, scores, image_size = candidates[photo_id]
        xyxy_arr = np.asarray(xyxy, dtype=float).reshape(-1, 4)
        scores_arr = np.asarray(scores, dtype=float)
        mask = scores_arr > threshold
        boxes = xyxy_arr[mask]
        box_scores = scores_arr[mask]
        if nms_iou is not None and len(boxes):
            keep = nms_numpy(boxes, box_scores, nms_iou)
            boxes = boxes[keep]
        width, height = image_size
        predicted[photo_id] = normalize_boxes(boxes.tolist(), width, height)
    return predicted


def parse_nms(spec: str | None) -> list[float | None]:
    if spec is None:
        return [None]
    return [None if v.lower() == "none" else float(v) for v in spec.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_FOLDERS), required=True)
    parser.add_argument("--thresholds", required=True, help="e.g. 0.05,0.1,0.2,...")
    parser.add_argument("--nms-ious", default=None, help="e.g. none,0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folder", type=Path, default=None, help="Override prediction directory with log.jsonl")
    args = parser.parse_args()

    gold = read_rows(DATA_DIR / "gold/ours_gold.jsonl")
    if args.folder is not None:
        folder = args.folder
    else:
        folder = DATA_DIR / "predictions" / MODEL_FOLDERS[args.model]
    print(f"Loading candidates from {folder}")
    candidates = load_candidates(folder)
    missing = gold.keys() - candidates.keys()
    if missing:
        raise ValueError(f"{len(missing)} gold photos have no candidates")

    thresholds = [float(v) for v in args.thresholds.split(",")]
    nms_ious = parse_nms(args.nms_ious)

    rows: list[dict] = []
    for threshold in thresholds:
        for nms_iou in nms_ious:
            predicted = build_predicted(candidates, gold, threshold, nms_iou)
            metrics = evaluate(gold, predicted)["metrics"]
            rows.append({
                "model": args.model,
                "threshold": threshold,
                "nms_iou": nms_iou,
                "f1_0.5": metrics["0.5"]["f1"],
                "precision_0.5": metrics["0.5"]["precision"],
                "recall_0.5": metrics["0.5"]["recall"],
                "tp_0.5": metrics["0.5"]["tp"],
                "fp_0.5": metrics["0.5"]["fp"],
                "fn_0.5": metrics["0.5"]["fn"],
                "f1_0.75": metrics["0.75"]["f1"],
                "precision_0.75": metrics["0.75"]["precision"],
                "recall_0.75": metrics["0.75"]["recall"],
                "negative_fp_boxes": metrics["negative_photos"]["false_positive_boxes"],
            })
            print(
                f"thr={threshold:<4} nms={str(nms_iou):<5} "
                f"f1@0.5={metrics['0.5']['f1']:.4f} "
                f"f1@0.75={metrics['0.75']['f1']:.4f} "
                f"neg_fp={metrics['negative_photos']['false_positive_boxes']}"
            )

    rows.sort(key=lambda r: (r["f1_0.5"], r["f1_0.75"]), reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {args.output}")
    print("Top 5 by F1@0.5:")
    for r in rows[:5]:
        print(
            f"  thr={r['threshold']:<4} nms={str(r['nms_iou']):<5} "
            f"F1@0.5={r['f1_0.5']:.4f} F1@0.75={r['f1_0.75']:.4f} "
            f"P={r['precision_0.5']:.4f} R={r['recall_0.5']:.4f}"
        )


if __name__ == "__main__":
    main()