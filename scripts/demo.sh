#!/bin/bash
# ============================================================
# ReactMotion Demo Script
# ============================================================
# Usage:
#   bash scripts/demo.sh                   # CLI demo (text-only)
#   bash scripts/demo.sh --gradio          # Launch Gradio web UI
#   bash scripts/demo.sh --audio           # CLI demo with audio
# ============================================================

set -e

# ── Paths (MODIFY THESE) ──
GEN_CKPT="logs/checkpoints/checkpoint-XXXXX"           # ReactMotion T5 checkpoint
VQVAE_CKPT="external/pretrained_vqvae/t2m.pth"           # Motion VQ-VAE checkpoint
MEAN_PATH="external/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy"
STD_PATH="external/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy"

# ── Parse mode ──
MODE="text"
for arg in "$@"; do
    case $arg in
        --gradio)  MODE="gradio" ;;
        --audio)   MODE="audio" ;;
    esac
done

if [ "$MODE" = "gradio" ]; then
    echo "=========================================="
    echo "  ReactMotion Gradio Demo"
    echo "=========================================="
    python demo_gradio.py \
        --gen_ckpt  "$GEN_CKPT" \
        --vqvae_ckpt "$VQVAE_CKPT" \
        --mean_path "$MEAN_PATH" \
        --std_path  "$STD_PATH" \
        --port 7860 \
        --share

elif [ "$MODE" = "audio" ]; then
    echo "=========================================="
    echo "  ReactMotion CLI Demo (Audio + Text)"
    echo "=========================================="
    python demo_inference.py \
        --gen_ckpt  "$GEN_CKPT" \
        --vqvae_ckpt "$VQVAE_CKPT" \
        --mean_path "$MEAN_PATH" \
        --std_path  "$STD_PATH" \
        --text "I just saw a shooting star streak across the sky! Did you see it too?" \
        --audio "samples/speaker.wav" \
        --emotion "excited" \
        --cond_mode "t+a+e" \
        --num_gen 3 \
        --out_path outputs/demo_audio.mp4

else
    echo "=========================================="
    echo "  ReactMotion CLI Demo (Text-only)"
    echo "=========================================="

    # Example 1: Excited speaker
    python demo_inference.py \
        --gen_ckpt  "$GEN_CKPT" \
        --vqvae_ckpt "$VQVAE_CKPT" \
        --mean_path "$MEAN_PATH" \
        --std_path  "$STD_PATH" \
        --text "I just saw a shooting star streak across the sky! Did you see it too?" \
        --emotion "excited" \
        --cond_mode "t+e" \
        --num_gen 3 \
        --out_path outputs/demo_excited.mp4

    # Example 2: Sad speaker
    python demo_inference.py \
        --gen_ckpt  "$GEN_CKPT" \
        --vqvae_ckpt "$VQVAE_CKPT" \
        --mean_path "$MEAN_PATH" \
        --std_path  "$STD_PATH" \
        --text "That's really unfortunate. I'm sorry to hear about your loss." \
        --emotion "sad" \
        --cond_mode "t+e" \
        --num_gen 3 \
        --out_path outputs/demo_sad.mp4

    # Example 3: Warning
    python demo_inference.py \
        --gen_ckpt  "$GEN_CKPT" \
        --vqvae_ckpt "$VQVAE_CKPT" \
        --mean_path "$MEAN_PATH" \
        --std_path  "$STD_PATH" \
        --text "Watch out! There's a car coming from the left!" \
        --emotion "fearful" \
        --cond_mode "t+e" \
        --num_gen 3 \
        --out_path outputs/demo_warning.mp4

    echo ""
    echo "Done! Videos saved to outputs/"
fi
