#!/usr/bin/env bash
set -euo pipefail

# =========================
# Unified evaluation script
# Runs win-rate/Gen@3 + FID/Diversity
# =========================

# Paths (edit these)
DATASET_DIR="/path/to/dataset/A2R"
PAIRS_CSV="./data"
AUDIO_CODE_DIR="/path/to/audio_codes"
GEN_CKPT="/path/to/generator/checkpoint"
JUDGE_CKPT="/path/to/judge/best.pt"
OUT_DIR="./eval_output"

# FID-specific paths (only needed for --pipeline fid or all)
T2M_OPT="./checkpoints/t2m/Comp_v6_KLD005/opt.txt"
VQVAE_CKPT="/path/to/motion_VQVAE/net_last.pth"
MEAN_PATH="./dataset/HumanML3D/Mean.npy"
STD_PATH="./dataset/HumanML3D/Std.npy"

# Settings
COND_MODE="${1:-t+a+e}"         # t | a | t+e | a+e | t+a | t+a+e
PIPELINE="${2:-all}"             # all | winrate | fid
SPLIT="test"
NUM_GEN=3
SEED=42

# Auto-infer
USE_EMO_FLAG=""
if [[ "${COND_MODE}" == *"+e"* ]]; then
  USE_EMO_FLAG="--use_emotion"
fi

AUDIO_FLAGS="--audio_mode none"
if [[ "${COND_MODE}" == *"a"* ]]; then
  AUDIO_FLAGS="--audio_mode code --audio_code_dir ${AUDIO_CODE_DIR}"
fi

echo "======================================================="
echo "[Evaluate] pipeline=${PIPELINE}  cond_mode=${COND_MODE}"
echo "======================================================="

python -m reactmotion.eval.evaluate \
  --pipeline "${PIPELINE}" \
  --dataset_dir "${DATASET_DIR}" \
  --pairs_csv "${PAIRS_CSV}" \
  --audio_code_dir "${AUDIO_CODE_DIR}" \
  --gen_ckpt "${GEN_CKPT}" \
  --judge_ckpt "${JUDGE_CKPT}" \
  --cond_mode "${COND_MODE}" \
  ${USE_EMO_FLAG} \
  ${AUDIO_FLAGS} \
  --num_gen "${NUM_GEN}" \
  --eval_split "${SPLIT}" \
  --out_dir "${OUT_DIR}" \
  --t2m_opt "${T2M_OPT}" \
  --vqvae_ckpt "${VQVAE_CKPT}" \
  --mean_path "${MEAN_PATH}" \
  --std_path "${STD_PATH}" \
  --seed "${SEED}"

echo "======================================================="
echo "[Done] Results in ${OUT_DIR}"
echo "======================================================="
