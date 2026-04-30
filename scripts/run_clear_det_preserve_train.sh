#!/usr/bin/env bash
set -euo pipefail

# Clear-weather detection preservation run.
#
# Goal:
# - Start from the current formal fog-focused checkpoint
# - Unfreeze the detector again
# - Keep fog heads active
# - Add an auxiliary clear-image detection loss to reduce the control-clip gap
#
# Usage:
#   bash scripts/run_clear_det_preserve_train.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_OUTPUT_DIR="$ROOT_DIR/outputs/Fog_Detection_Project_clearpreserve"
DEFAULT_CHECKPOINT_DIR="$DEFAULT_OUTPUT_DIR/checkpoints"
DEFAULT_RESUME_CHECKPOINT="$ROOT_DIR/outputs/Fog_Detection_Project_fogfocus_full/checkpoints/checkpoint_epoch_0012.pt"

export BS_OUTPUT_DIR="${BS_OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
export BS_CHECKPOINT_DIR="${BS_CHECKPOINT_DIR:-$DEFAULT_CHECKPOINT_DIR}"
export BS_RESUME_CHECKPOINT="${BS_RESUME_CHECKPOINT:-$DEFAULT_RESUME_CHECKPOINT}"

export BS_YOLO_BASE_MODEL="${BS_YOLO_BASE_MODEL:-yolo11n.pt}"
export BS_DEVICE="${BS_DEVICE:-cuda}"

export BS_BATCH_SIZE="${BS_BATCH_SIZE:-16}"
export BS_EPOCHS="${BS_EPOCHS:-14}"
export BS_LR="${BS_LR:-5e-6}"
export BS_QAT_EPOCHS="${BS_QAT_EPOCHS:-0}"
export BS_SKIP_QAT="${BS_SKIP_QAT:-1}"

export BS_PRECOMPUTE_DEPTH_CACHE="${BS_PRECOMPUTE_DEPTH_CACHE:-0}"
export BS_NUM_WORKERS="${BS_NUM_WORKERS:-8}"
export BS_PREFETCH_FACTOR="${BS_PREFETCH_FACTOR:-2}"
export BS_PERSISTENT_WORKERS="${BS_PERSISTENT_WORKERS:-1}"

export BS_MAX_TRAIN_BATCHES="${BS_MAX_TRAIN_BATCHES:-0}"
export BS_MAX_VAL_BATCHES="${BS_MAX_VAL_BATCHES:-0}"

export BS_RESUME_MODEL_ONLY="${BS_RESUME_MODEL_ONLY:-1}"
export BS_FREEZE_YOLO_FOR_FOG="${BS_FREEZE_YOLO_FOR_FOG:-0}"
export BS_DET_LOSS_WEIGHT="${BS_DET_LOSS_WEIGHT:-0.35}"
export BS_CLEAR_DET_LOSS_WEIGHT="${BS_CLEAR_DET_LOSS_WEIGHT:-0.75}"
export BS_FOG_CLS_LOSS_WEIGHT="${BS_FOG_CLS_LOSS_WEIGHT:-1.5}"
export BS_FOG_REG_LOSS_WEIGHT="${BS_FOG_REG_LOSS_WEIGHT:-1.15}"

export BS_FOG_CLEAR_PROB="${BS_FOG_CLEAR_PROB:-0.15}"
export BS_FOG_UNIFORM_PROB="${BS_FOG_UNIFORM_PROB:-0.35}"
export BS_FOG_PATCHY_PROB="${BS_FOG_PATCHY_PROB:-0.50}"
export BS_FOG_BETA_MIN="${BS_FOG_BETA_MIN:-0.04}"
export BS_UNIFORM_DEPTH_SCALE="${BS_UNIFORM_DEPTH_SCALE:-7.0}"
export BS_PATCHY_DEPTH_BASE="${BS_PATCHY_DEPTH_BASE:-2.0}"
export BS_PATCHY_DEPTH_NOISE_SCALE="${BS_PATCHY_DEPTH_NOISE_SCALE:-8.0}"

export BS_FOG_LABEL_SMOOTHING="${BS_FOG_LABEL_SMOOTHING:-0.05}"
export BS_FOG_CLS_CLEAR_WEIGHT="${BS_FOG_CLS_CLEAR_WEIGHT:-0.75}"
export BS_FOG_CLS_UNIFORM_WEIGHT="${BS_FOG_CLS_UNIFORM_WEIGHT:-1.0}"
export BS_FOG_CLS_PATCHY_WEIGHT="${BS_FOG_CLS_PATCHY_WEIGHT:-1.1}"

export BS_SEED="${BS_SEED:-42}"

mkdir -p "$BS_OUTPUT_DIR" "$BS_CHECKPOINT_DIR"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$BS_OUTPUT_DIR/clear_det_preserve_train_${RUN_STAMP}.log"

echo "Clear-weather detection preservation run"
echo "ROOT_DIR=$ROOT_DIR"
echo "BS_OUTPUT_DIR=$BS_OUTPUT_DIR"
echo "BS_CHECKPOINT_DIR=$BS_CHECKPOINT_DIR"
echo "BS_RESUME_CHECKPOINT=$BS_RESUME_CHECKPOINT"
echo "BS_EPOCHS=$BS_EPOCHS"
echo "BS_LR=$BS_LR"
echo "BS_DET_LOSS_WEIGHT=$BS_DET_LOSS_WEIGHT"
echo "BS_CLEAR_DET_LOSS_WEIGHT=$BS_CLEAR_DET_LOSS_WEIGHT"
echo "BS_FREEZE_YOLO_FOR_FOG=$BS_FREEZE_YOLO_FOR_FOG"
echo "RUN_LOG=$RUN_LOG"

python src/train.py 2>&1 | tee "$RUN_LOG"
