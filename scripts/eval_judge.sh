#!/usr/bin/env bash
set -euo pipefail

# =========================
# Paths (edit these)
# =========================
CKPT="/path/to/judge/best.pt"
DATASET_DIR="/path/to/dataset/A2R"
PAIRS_CSV="./data"                        # dir with train.csv / val.csv / test.csv
AUDIO_CODE_DIR="/path/to/audio_codes"
SAVE_DIR="./eval_output/judge"
KEY_BY="group_id"

# =========================
# Eval settings
# =========================
COND_MODE="${1:-t+a+e}"    # t | a | t+e | a+e | t+a | t+a+e
BS=32
NW=4
BOOT=2000
DEVICE="cuda"

# =========================
# Auto-infer cond_head from mode
# =========================
case "${COND_MODE}" in
  "t")     COND_HEAD="text"  ;;
  "a")     COND_HEAD="audio" ;;
  *)       COND_HEAD="fused" ;;
esac

RUN_TAG="${COND_MODE//+/p}"
OUT_DIR="${SAVE_DIR}/eval_${RUN_TAG}_head_${COND_HEAD}"
mkdir -p "${OUT_DIR}"

# =========================
# Run
# =========================
echo "======================================================="
echo "[Eval JudgeNetwork]"
echo "  cond_mode  = ${COND_MODE}"
echo "  cond_head  = ${COND_HEAD}"
echo "  ckpt       = ${CKPT}"
echo "  out_dir    = ${OUT_DIR}"
echo "======================================================="

python -m reactmotion.eval.eval_judge \
  --ckpt "${CKPT}" \
  --dataset_dir "${DATASET_DIR}" \
  --pairs_csv "${PAIRS_CSV}" \
  --audio_code_dir "${AUDIO_CODE_DIR}" \
  --key_by "${KEY_BY}" \
  --batch_size "${BS}" \
  --num_workers "${NW}" \
  --bootstrap "${BOOT}" \
  --device "${DEVICE}" \
  --save_dir "${OUT_DIR}" \
  --eval_splits val test \
  --modes "${COND_MODE}" \
  --cond_head "${COND_HEAD}"

echo "======================================================="
echo "[Done] mode=${COND_MODE} -> ${OUT_DIR}"
echo "======================================================="
