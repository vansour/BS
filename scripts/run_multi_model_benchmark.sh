#!/usr/bin/env bash
set -euo pipefail

# Run the same benchmark video set across multiple fog-model baselines.
#
# Usage:
#   bash scripts/run_multi_model_benchmark.sh
#
# Optional overrides:
#   BS_BENCHMARK_CONFIG=/abs/path/to/benchmark_videos.json \
#   BS_MODEL_CONFIG=/abs/path/to/benchmark_models.json \
#   BS_MULTI_BENCHMARK_OUTPUT_DIR=/abs/path/to/output \
#   BS_BENCHMARK_YOLO_WEIGHTS=/abs/path/to/yolo11n.pt \
#   BS_BENCHMARK_SAMPLE_STRIDE=5 \
#   BS_BENCHMARK_MAX_FRAMES=300 \
#   bash scripts/run_multi_model_benchmark.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BENCHMARK_CONFIG="${BS_BENCHMARK_CONFIG:-$ROOT_DIR/configs/benchmark_videos.json}"
MODEL_CONFIG="${BS_MODEL_CONFIG:-$ROOT_DIR/configs/benchmark_models.json}"
OUTPUT_DIR="${BS_MULTI_BENCHMARK_OUTPUT_DIR:-$ROOT_DIR/outputs/Benchmark_Model_Compare_v1}"
YOLO_WEIGHTS="${BS_BENCHMARK_YOLO_WEIGHTS:-$ROOT_DIR/yolo11n.pt}"
SAMPLE_STRIDE="${BS_BENCHMARK_SAMPLE_STRIDE:-}"
MAX_FRAMES="${BS_BENCHMARK_MAX_FRAMES:-}"
UNIFIED_CONF="${BS_BENCHMARK_UNIFIED_CONF:-}"
HYBRID_CONF="${BS_BENCHMARK_HYBRID_CONF:-}"
IMGSZ="${BS_BENCHMARK_IMGSZ:-}"
TOPK_PREVIEW="${BS_BENCHMARK_TOPK_PREVIEW:-}"
REUSE_EXISTING="${BS_MULTI_BENCHMARK_REUSE_EXISTING:-0}"

mkdir -p "$OUTPUT_DIR"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$OUTPUT_DIR/multi_model_benchmark_${RUN_STAMP}.log"

CMD=(
  python
  scripts/run_multi_model_benchmark.py
  --benchmark-config "$BENCHMARK_CONFIG"
  --model-config "$MODEL_CONFIG"
  --output-dir "$OUTPUT_DIR"
  --yolo-weights "$YOLO_WEIGHTS"
)

if [[ -n "$SAMPLE_STRIDE" ]]; then
  CMD+=(--sample-stride "$SAMPLE_STRIDE")
fi
if [[ -n "$MAX_FRAMES" ]]; then
  CMD+=(--max-frames "$MAX_FRAMES")
fi
if [[ -n "$UNIFIED_CONF" ]]; then
  CMD+=(--unified-conf "$UNIFIED_CONF")
fi
if [[ -n "$HYBRID_CONF" ]]; then
  CMD+=(--hybrid-conf "$HYBRID_CONF")
fi
if [[ -n "$IMGSZ" ]]; then
  CMD+=(--imgsz "$IMGSZ")
fi
if [[ -n "$TOPK_PREVIEW" ]]; then
  CMD+=(--topk-preview "$TOPK_PREVIEW")
fi
if [[ "$REUSE_EXISTING" == "1" ]]; then
  CMD+=(--reuse-existing)
fi

echo "Multi-model benchmark run"
echo "ROOT_DIR=$ROOT_DIR"
echo "BENCHMARK_CONFIG=$BENCHMARK_CONFIG"
echo "MODEL_CONFIG=$MODEL_CONFIG"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "YOLO_WEIGHTS=$YOLO_WEIGHTS"
echo "RUN_LOG=$RUN_LOG"
printf 'COMMAND='
printf '%q ' "${CMD[@]}"
printf '\n'

"${CMD[@]}" 2>&1 | tee "$RUN_LOG"
