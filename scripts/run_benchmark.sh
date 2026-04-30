#!/usr/bin/env bash
set -euo pipefail

# Run the route-evaluation benchmark with a config-driven video list.
#
# Usage:
#   bash scripts/run_benchmark.sh
#
# Optional overrides:
#   BS_BENCHMARK_CONFIG=/abs/path/to/benchmark_videos.json \
#   BS_BENCHMARK_OUTPUT_DIR=/abs/path/to/output \
#   BS_BENCHMARK_FOG_WEIGHTS=/abs/path/to/unified_model_best.pt \
#   BS_BENCHMARK_YOLO_WEIGHTS=/abs/path/to/yolo11n.pt \
#   BS_BENCHMARK_SAMPLE_STRIDE=5 \
#   BS_BENCHMARK_MAX_FRAMES=300 \
#   bash scripts/run_benchmark.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BENCHMARK_CONFIG="${BS_BENCHMARK_CONFIG:-$ROOT_DIR/configs/benchmark_videos.json}"
OUTPUT_DIR="${BS_BENCHMARK_OUTPUT_DIR:-$ROOT_DIR/outputs/Benchmark_v1}"
FOG_WEIGHTS="${BS_BENCHMARK_FOG_WEIGHTS:-}"
YOLO_WEIGHTS="${BS_BENCHMARK_YOLO_WEIGHTS:-$ROOT_DIR/yolo11n.pt}"
SAMPLE_STRIDE="${BS_BENCHMARK_SAMPLE_STRIDE:-}"
MAX_FRAMES="${BS_BENCHMARK_MAX_FRAMES:-}"
UNIFIED_CONF="${BS_BENCHMARK_UNIFIED_CONF:-}"
HYBRID_CONF="${BS_BENCHMARK_HYBRID_CONF:-}"
IMGSZ="${BS_BENCHMARK_IMGSZ:-}"
TOPK_PREVIEW="${BS_BENCHMARK_TOPK_PREVIEW:-}"

mkdir -p "$OUTPUT_DIR"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$OUTPUT_DIR/benchmark_${RUN_STAMP}.log"

CMD=(
  python
  scripts/evaluate_inference_routes.py
  --benchmark-config "$BENCHMARK_CONFIG"
  --output-dir "$OUTPUT_DIR"
  --yolo-weights "$YOLO_WEIGHTS"
)

if [[ -n "$FOG_WEIGHTS" ]]; then
  CMD+=(--fog-weights "$FOG_WEIGHTS")
fi
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

echo "Benchmark run"
echo "ROOT_DIR=$ROOT_DIR"
echo "BENCHMARK_CONFIG=$BENCHMARK_CONFIG"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "FOG_WEIGHTS=${FOG_WEIGHTS:-auto-resolve}"
echo "YOLO_WEIGHTS=$YOLO_WEIGHTS"
echo "RUN_LOG=$RUN_LOG"
printf 'COMMAND='
printf '%q ' "${CMD[@]}"
printf '\n'

"${CMD[@]}" 2>&1 | tee "$RUN_LOG"
