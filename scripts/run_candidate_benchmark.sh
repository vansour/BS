#!/usr/bin/env bash
set -euo pipefail

# Benchmark one candidate fog-model checkpoint against the baseline model set.
#
# Usage:
#   BS_CANDIDATE_WEIGHTS=/abs/path/to/unified_model_best.pt \
#   BS_CANDIDATE_LABEL=my_candidate \
#   bash scripts/run_candidate_benchmark.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CANDIDATE_WEIGHTS="${BS_CANDIDATE_WEIGHTS:-}"
CANDIDATE_LABEL="${BS_CANDIDATE_LABEL:-}"
CANDIDATE_SLUG="${BS_CANDIDATE_SLUG:-}"
CANDIDATE_NOTES="${BS_CANDIDATE_NOTES:-}"

if [[ -z "$CANDIDATE_WEIGHTS" ]]; then
  echo "BS_CANDIDATE_WEIGHTS is required." >&2
  exit 1
fi
if [[ -z "$CANDIDATE_LABEL" ]]; then
  echo "BS_CANDIDATE_LABEL is required." >&2
  exit 1
fi

BENCHMARK_CONFIG="${BS_BENCHMARK_CONFIG:-$ROOT_DIR/configs/benchmark_videos.json}"
BASE_MODEL_CONFIG="${BS_BASE_MODEL_CONFIG:-$ROOT_DIR/configs/benchmark_models.json}"
OUTPUT_DIR="${BS_CANDIDATE_BENCHMARK_OUTPUT_DIR:-$ROOT_DIR/outputs/Candidate_Benchmark_v1}"
YOLO_WEIGHTS="${BS_BENCHMARK_YOLO_WEIGHTS:-$ROOT_DIR/yolo11n.pt}"
SAMPLE_STRIDE="${BS_BENCHMARK_SAMPLE_STRIDE:-}"
MAX_FRAMES="${BS_BENCHMARK_MAX_FRAMES:-}"
UNIFIED_CONF="${BS_BENCHMARK_UNIFIED_CONF:-}"
HYBRID_CONF="${BS_BENCHMARK_HYBRID_CONF:-}"
IMGSZ="${BS_BENCHMARK_IMGSZ:-}"
TOPK_PREVIEW="${BS_BENCHMARK_TOPK_PREVIEW:-}"
REUSE_EXISTING="${BS_CANDIDATE_BENCHMARK_REUSE_EXISTING:-0}"

mkdir -p "$OUTPUT_DIR"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$OUTPUT_DIR/candidate_benchmark_${RUN_STAMP}.log"

CMD=(
  python
  scripts/benchmark_candidate_model.py
  --candidate-weights "$CANDIDATE_WEIGHTS"
  --candidate-label "$CANDIDATE_LABEL"
  --benchmark-config "$BENCHMARK_CONFIG"
  --base-model-config "$BASE_MODEL_CONFIG"
  --output-dir "$OUTPUT_DIR"
  --yolo-weights "$YOLO_WEIGHTS"
)

if [[ -n "$CANDIDATE_SLUG" ]]; then
  CMD+=(--candidate-slug "$CANDIDATE_SLUG")
fi
if [[ -n "$CANDIDATE_NOTES" ]]; then
  CMD+=(--candidate-notes "$CANDIDATE_NOTES")
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
if [[ "$REUSE_EXISTING" == "1" ]]; then
  CMD+=(--reuse-existing)
fi

echo "Candidate benchmark run"
echo "ROOT_DIR=$ROOT_DIR"
echo "CANDIDATE_WEIGHTS=$CANDIDATE_WEIGHTS"
echo "CANDIDATE_LABEL=$CANDIDATE_LABEL"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "RUN_LOG=$RUN_LOG"
printf 'COMMAND='
printf '%q ' "${CMD[@]}"
printf '\n'

"${CMD[@]}" 2>&1 | tee "$RUN_LOG"
