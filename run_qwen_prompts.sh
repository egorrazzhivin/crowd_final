#!/usr/bin/env bash
set -euo pipefail

DATASET="data/dataset_ours.jsonl"
PREDS="data/predictions/experiments"
mkdir -p "$PREDS"

for v in v2 v3 v4; do
    TAG="qwen_${v}"
    OUT="${PREDS}/${TAG}"
    LOG="${PREDS}/${TAG}.run.log"

    if [ -d "$OUT" ]; then
        echo "[$TAG] already exists, skipping"
        continue
    fi

    echo "=== [$TAG] inference ==="

    python label.py "$DATASET" "prompts/qwen_${v}.txt" \
        --model qwen --device mps --tag "$TAG" \
        >> "$LOG" 2>&1

    SRC="data/predictions/dataset_ours_qwen_threshold-none__${TAG}"

    if [ -d "$SRC" ]; then
        mv "$SRC" "$OUT"
    else
        echo "ERROR: output folder not found: $SRC"
        exit 1
    fi

    echo "[$TAG] done"
done
