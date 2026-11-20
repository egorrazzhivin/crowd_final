#!/usr/bin/env bash
# usage: ./run_prompts.sh grounding-dino v2 v3 v4 v5
set -euo pipefail

MODEL="$1"; shift
DATASET="data/dataset_ours.jsonl"
LOGS="data/experiments/prompt_tuning"
PREDS="data/predictions/experiments"
mkdir -p "$LOGS" "$PREDS"

for v in "$@"; do
  TAG="detector_${v}_${MODEL}"
  RUN_LOG="${LOGS}/${TAG}.run.log"
  GRID_CSV="${LOGS}/${TAG}.grid.csv"

  if [ -f "$GRID_CSV" ]; then
    echo "[$MODEL/$v] skip: $GRID_CSV exists"
    continue
  fi

  echo "=== [$MODEL/$v] inference"
  python label.py "$DATASET" "prompts/detector_${v}.txt" \
    --model "$MODEL" --threshold 0.01 --device mps \
    --tag "prompt_${v}" \
    >> "$RUN_LOG" 2>&1

  SRC="data/predictions/dataset_ours_${MODEL}_threshold-0.01"
  [ "$MODEL" = "grounding-dino" ] && SRC="${SRC}_nms-off"
  SRC="${SRC}__prompt_${v}"
  mv "$SRC" "${PREDS}/${TAG}"

  echo "=== [$MODEL/$v] tune"
  python tune.py --model "$MODEL" --folder "${PREDS}/${TAG}" \
    --thresholds 0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50 \
    --nms-ious none,0.3,0.4,0.5 \
    --output "$GRID_CSV" >> "$RUN_LOG" 2>&1

  echo "[$MODEL/$v] done"
done