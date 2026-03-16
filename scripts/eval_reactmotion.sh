#!/usr/bin/env bash
set -euo pipefail

# =========================
# Paths (edit these)
# =========================
DATASET_DIR="/path/to/dataset/A2R"
PAIRS_CSV="./data/test.csv"
AUDIO_CODE_DIR="/path/to/audio_codes"
GEN_CKPT="/path/to/generator/checkpoint"
OUT_DIR="./eval_output/gen_dump"

# =========================
# Eval settings
# =========================
COND_MODE="${1:-t+a+e}"    # t | a | t+e | a+e | t+a | t+a+e
SPLIT="test"
NUM_GEN=3
SRC_LEN=512
GEN_MAX_LEN=256
KEY_BY="group_id"
SEED=42

# Optional caption model (uncomment to enable)
# CAPTION_CKPT="/path/to/caption/model"
# CAPTION_FLAGS="--caption_ckpt ${CAPTION_CKPT}"
CAPTION_FLAGS="--no_caption"

# =========================
# Auto-infer flags from COND_MODE
# =========================
USE_EMO_FLAG=""
if [[ "${COND_MODE}" == *"+e"* ]]; then
  USE_EMO_FLAG="--use_emotion"
fi

AUDIO_FLAGS="--audio_mode none"
if [[ "${COND_MODE}" == *"a"* ]]; then
  AUDIO_FLAGS="--audio_mode code --audio_code_dir ${AUDIO_CODE_DIR} --audio_token_level base"
fi

RUN_TAG="${COND_MODE//+/p}"
OUT_DIR="${OUT_DIR}_${RUN_TAG}"
mkdir -p "${OUT_DIR}"

# =========================
# Run
# =========================
echo "======================================================="
echo "[Eval ReactMotion - Generate]"
echo "  cond_mode = ${COND_MODE}"
echo "  gen_ckpt  = ${GEN_CKPT}"
echo "  out_dir   = ${OUT_DIR}"
echo "======================================================="

python -m reactmotion.eval.eval_reactmotion \
  --pairs_csv "${PAIRS_CSV}" \
  --dataset_dir "${DATASET_DIR}" \
  --gen_ckpt "${GEN_CKPT}" \
  ${CAPTION_FLAGS} \
  --cond_mode "${COND_MODE}" \
  ${USE_EMO_FLAG} \
  --only_split "${SPLIT}" \
  --key_by "${KEY_BY}" \
  --num_gen "${NUM_GEN}" \
  --source_len "${SRC_LEN}" \
  --gen_max_len_codes "${GEN_MAX_LEN}" \
  --out_dir "${OUT_DIR}" \
  ${AUDIO_FLAGS} \
  --seed "${SEED}"

echo "======================================================="
echo "[Done] ${COND_MODE} -> ${OUT_DIR}"
echo "======================================================="
