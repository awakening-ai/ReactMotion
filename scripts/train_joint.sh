#!/usr/bin/env bash
set -euo pipefail

# =========================
# Joint ReactMotion + T2M Training
# =========================
# This script trains ReactMotion (speaker utterance → listener motion)
# jointly with Text-to-Motion (HumanML3D caption → motion).
# T2M samples are reformulated as dialogue prompts and share the
# same PROMPT_V3 template. Ratio: ~33% T2M, ~67% ReactMotion.

# =========================
# Paths (edit these)
# =========================
DATASET_DIR="/path/to/dataset"
PAIRS_DIR="./reactmotion/data"                       # dir with train.csv / val.csv / test.csv
AUDIO_CODE_DIR="/path/to/audio_code"
MODEL_NAME="google-t5/t5-base"
OUTPUT_DIR="./output/joint_reactmotion_t2m"

# HumanML3D directory (default: ${DATASET_DIR}/HumanML3D)
HUMANML3D_DIR="${DATASET_DIR}/HumanML3D"

# =========================
# Conditioning mode (pick one)
# =========================
COND_MODE="${1:-t+a+e}"    # t | a | t+e | a+e | t+a | t+a+e

# =========================
# Loss type
# =========================
LOSS_TYPE="${2:-multi_ce_rank}"  # multi_ce_rank | ce | multi_ce | rank

# =========================
# T2M co-training settings
# =========================
T2M_RATIO=0.33             # 33% T2M, 67% ReactMotion
T2M_LOSS_WEIGHT=1.0        # Weight for T2M loss

# =========================
# Train knobs
# =========================
BATCH_SIZE=8
GRAD_ACCUM=2
LR=5e-5
MAX_STEPS=100000
SAVE_STEPS=5000
SAVE_TOTAL_LIMIT=10
EVAL_STEPS=5000
NUM_WORKERS=4

SEED=42
SRC_LEN=512
TGT_LEN=256
KEY_BY="group_id"

# gold sampling
K_GOLD=2
SAMPLE_GOLD="random"
NORMALIZE_LSE="--normalize_logsumexp"

# rank loss hyperparams
RANK_MARGIN=0.5
W_RANK=0.25
W_GN=0.25

# modality dropout
COND_DROPOUT=0.30
DROP_E_ONLY="--drop_e_only_when_multi"

# group weight
GROUP_W_MODE="from_csv"    # none | from_csv | constant
GROUP_W_COL="group_w"
GROUP_W_AGG="mean"         # mean | max | first
GROUP_W_CLIP_MIN=0.2
GROUP_W_CLIP_MAX=5.0

# anti-template (0 = disabled)
DIVERSITY_W=0.0
BATCH_TEMPLATE_W=0.0

# inverse-frequency reweighting (uncomment to enable)
# IFW_FLAGS="--use_inverse_freq_reweight --freq_alpha 1.0"
IFW_FLAGS=""

# wandb
WANDB_MODE="disabled"    # online | offline | disabled
WANDB_PROJECT="ReactMotion"
WANDB_TAGS="ReactMotion,JointT2M,${LOSS_TYPE},cond=${COND_MODE}"

# =========================
# Auto-infer flags from COND_MODE
# =========================
USE_EMO_FLAG=""
if [[ "${COND_MODE}" == *"+e"* ]]; then
  USE_EMO_FLAG="--use_emotion"
fi

AUDIO_FLAGS="--audio_mode none"
if [[ "${COND_MODE}" == *"a"* ]]; then
  AUDIO_FLAGS="--audio_mode code --audio_code_dir ${AUDIO_CODE_DIR}"
fi

RUN_TAG="${COND_MODE//+/p}"
OUTPUT_DIR="${OUTPUT_DIR}/cond_${RUN_TAG}_${LOSS_TYPE}"
mkdir -p "${OUTPUT_DIR}"

# =========================
# Run
# =========================
echo "======================================================="
echo "[Train Joint ReactMotion + T2M]"
echo "  cond_mode    = ${COND_MODE}"
echo "  loss_type    = ${LOSS_TYPE}"
echo "  t2m_ratio    = ${T2M_RATIO}"
echo "  t2m_weight   = ${T2M_LOSS_WEIGHT}"
echo "  humanml3d    = ${HUMANML3D_DIR}"
echo "  output_dir   = ${OUTPUT_DIR}"
echo "======================================================="

python -m reactmotion.train.train_reactmotion \
  --model_name "${MODEL_NAME}" \
  --dataset_dir "${DATASET_DIR}" \
  --pairs_csv "${PAIRS_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --grad_accum "${GRAD_ACCUM}" \
  --learning_rate "${LR}" \
  --max_steps "${MAX_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --seed "${SEED}" \
  --source_len "${SRC_LEN}" \
  --target_len "${TGT_LEN}" \
  --cond_mode "${COND_MODE}" \
  ${USE_EMO_FLAG} \
  --key_by "${KEY_BY}" \
  --cond_dropout "${COND_DROPOUT}" \
  ${DROP_E_ONLY} \
  --loss_type "${LOSS_TYPE}" \
  --rank_margin "${RANK_MARGIN}" \
  --w_rank "${W_RANK}" \
  --w_gn "${W_GN}" \
  --k_gold "${K_GOLD}" \
  --sample_gold "${SAMPLE_GOLD}" \
  ${NORMALIZE_LSE} \
  --diversity_w "${DIVERSITY_W}" \
  --batch_template_w "${BATCH_TEMPLATE_W}" \
  ${IFW_FLAGS} \
  ${AUDIO_FLAGS} \
  --group_w_mode "${GROUP_W_MODE}" \
  --group_w_col "${GROUP_W_COL}" \
  --group_w_agg "${GROUP_W_AGG}" \
  --group_w_clip_min "${GROUP_W_CLIP_MIN}" \
  --group_w_clip_max "${GROUP_W_CLIP_MAX}" \
  --enable_t2m \
  --t2m_ratio "${T2M_RATIO}" \
  --t2m_loss_weight "${T2M_LOSS_WEIGHT}" \
  --humanml3d_dir "${HUMANML3D_DIR}" \
  --auto_resume \
  --do_eval \
  --eval_steps "${EVAL_STEPS}" \
  --num_workers "${NUM_WORKERS}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_mode "${WANDB_MODE}" \
  --wandb_tags "${WANDB_TAGS}"

echo "======================================================="
echo "[Done] Joint ${COND_MODE} (${LOSS_TYPE}) -> ${OUTPUT_DIR}"
echo "======================================================="
