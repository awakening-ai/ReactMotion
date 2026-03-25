#!/usr/bin/env bash
set -euo pipefail

# =========================
# Paths (edit these)
# =========================
DATASET_DIR="/path/to/dataset"
PAIRS_DIR="./reactmotion/data"                       # dir with train.csv / val.csv / test.csv
DATASET_DIR="/path/to/audio_code"
SAVE_DIR="./output/judge"
KEY_BY="group_id"

# =========================
# Train knobs
# =========================
BS=16
NW=4
EPOCHS=50
LR=5e-5
WD=0.01
EVAL_EVERY=200
SEED=465

# Force single-modality ratio
FORCE_SINGLE=0.10

# Loss weights
W_FUSED=1.0
W_TEXT=0.50
W_AUDIO=0.50
W_EMO=0.20
W_ALIGN=0.0001

# Ordering margin
W_ORD=0.50
M_GS=0.20
M_SN=0.20

# Optional flags (uncomment to enable)
# USE_SILVER="--use_silver_as_pos"
# FREEZE_TEXT="--freeze_text"
# REQUIRE_AUDIO="--require_audio"
# DISABLE_TEXT="--disable_text"
# DISABLE_AUDIO="--disable_audio"
# DISABLE_EMO="--disable_emo"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "${SAVE_DIR}"

# =========================
# Run
# =========================
echo "======================================================="
echo "[Train JudgeNetwork]"
echo "  save_dir = ${SAVE_DIR}"
echo "======================================================="

python -m reactmotion.train.train_judge \
  --dataset_dir "${DATASET_DIR}" \
  --pairs_csv "${PAIRS_DIR}" \
  --audio_code_dir "${AUDIO_CODE_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --key_by "${KEY_BY}" \
  --batch_size "${BS}" \
  --num_workers "${NW}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --wd "${WD}" \
  --eval_every "${EVAL_EVERY}" \
  --seed "${SEED}" \
  --force_single_ratio "${FORCE_SINGLE}" \
  --w_fused "${W_FUSED}" \
  --w_text "${W_TEXT}" \
  --w_audio "${W_AUDIO}" \
  --w_emo "${W_EMO}" \
  --w_align "${W_ALIGN}" \
  --w_ord "${W_ORD}" \
  --m_gs "${M_GS}" \
  --m_sn "${M_SN}" \
  ${USE_SILVER:-} \
  ${FREEZE_TEXT:-} \
  ${REQUIRE_AUDIO:-} \
  ${DISABLE_TEXT:-} \
  ${DISABLE_AUDIO:-} \
  ${DISABLE_EMO:-}

echo "======================================================="
echo "[Done] -> ${SAVE_DIR}"
echo "======================================================="
