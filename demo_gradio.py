#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReactMotion Gradio Demo
========================

Interactive web UI for generating reactive listener motions.

Usage:
  python demo_gradio.py \
    --gen_ckpt  logs/checkpoints/checkpoint-XXXXX \
    --vqvae_ckpt models/VQVAE/net_best_fid.bin \
    --mean_path  data/HumanML3D/mean.npy \
    --std_path   data/HumanML3D/std.npy \
    --port 7860

Then open http://localhost:7860 in your browser.
"""

import os
import re
import argparse
import random
import tempfile
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchaudio
import imageio
import gradio as gr

from transformers import T5Tokenizer, T5ForConditionalGeneration

from reactmotion.dataset.prompt_builder import build_prompt
from reactmotion.models.vqvae import HumanVQVAE
from reactmotion.utils.motion_process import recover_from_ric
from reactmotion.visualization.plot_3d_global import plot_3d_motion

# ── Reuse core functions from demo_inference ──
from demo_inference import (
    parse_motion_tokens,
    encode_audio_to_text,
    build_vqvae,
    decode_motion_codes,
    render_motion_video,
    generate_motion,
    _VQVAEArgs,
)


# ─────────────────────────────────────────
# Global model holders (loaded once at startup)
# ─────────────────────────────────────────
_TOKENIZER: Optional[T5Tokenizer] = None
_MODEL: Optional[T5ForConditionalGeneration] = None
_VAE: Optional[HumanVQVAE] = None
_MEAN_NP: Optional[np.ndarray] = None
_STD_NP: Optional[np.ndarray] = None
_DEVICE: str = "cuda"


def load_models(gen_ckpt: str, vqvae_ckpt: str, mean_path: str, std_path: str):
    global _TOKENIZER, _MODEL, _VAE, _MEAN_NP, _STD_NP, _DEVICE

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Gradio] Loading generator from {gen_ckpt} ...")
    _TOKENIZER = T5Tokenizer.from_pretrained(gen_ckpt)
    _MODEL = T5ForConditionalGeneration.from_pretrained(gen_ckpt).to(_DEVICE).eval()
    print(f"[Gradio] Generator loaded: vocab={len(_TOKENIZER)}, "
          f"params={sum(p.numel() for p in _MODEL.parameters()) / 1e6:.1f}M")

    print(f"[Gradio] Loading VQ-VAE from {vqvae_ckpt} ...")
    _VAE = build_vqvae(vqvae_ckpt, device=_DEVICE)

    _MEAN_NP = np.load(mean_path).astype(np.float32)
    _STD_NP = np.load(std_path).astype(np.float32)
    print(f"[Gradio] All models loaded on {_DEVICE}")


# ─────────────────────────────────────────
# Gradio inference function
# ─────────────────────────────────────────
def infer(
    text: str,
    audio_path: Optional[str],
    emotion: str,
    cond_mode: str,
    num_gen: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
) -> List[str]:
    """
    Main Gradio inference callback.
    Returns a list of video file paths.
    """
    if _TOKENIZER is None or _MODEL is None:
        return ["Error: Models not loaded."]

    # seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # encode audio if needed
    audio_text = ""
    if audio_path and "a" in cond_mode:
        try:
            audio_text = encode_audio_to_text(
                wav_path=audio_path,
                device=_DEVICE,
                audio_sr=24000,
                mimi_codebooks=8,
                mimi_chunk_frames=32,
                mimi_cardinality=2048,
                audio_level="base",
            )
        except Exception as e:
            print(f"[Audio Encode Error] {e}")
            audio_text = ""

    # generate motion tokens
    num_gen = max(1, min(num_gen, 5))
    all_codes = generate_motion(
        tokenizer=_TOKENIZER,
        model=_MODEL,
        text=text or "",
        audio_text=audio_text,
        emotion=emotion or "",
        cond_mode=cond_mode,
        source_len=512,
        max_length=256,
        temperature=temperature,
        top_p=top_p,
        top_k=int(top_k),
        num_gen=num_gen,
        device=_DEVICE,
    )

    # decode & render each
    video_paths = []
    for i, codes in enumerate(all_codes):
        if len(codes) == 0:
            continue

        joints = decode_motion_codes(codes, _VAE, _MEAN_NP, _STD_NP, device=_DEVICE)

        # save to temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=tempfile.gettempdir())
        tmp.close()

        title = (text or "")[:60] if text else "Generated Motion"
        render_motion_video(joints, tmp.name, title=title, fps=20)
        video_paths.append(tmp.name)

    return video_paths


def gradio_infer(
    text: str,
    audio,            # Gradio audio component returns (sr, np_array) or filepath
    emotion: str,
    cond_mode: str,
    num_gen: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
):
    """Wrapper that handles Gradio's audio format."""
    audio_path = None
    if audio is not None:
        if isinstance(audio, str):
            # filepath mode
            audio_path = audio
        elif isinstance(audio, tuple) and len(audio) == 2:
            # (sample_rate, numpy_array) mode
            sr, wav_np = audio
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            wav_t = torch.from_numpy(wav_np).float()
            if wav_t.dim() == 1:
                wav_t = wav_t.unsqueeze(0)
            if wav_t.dim() == 2 and wav_t.shape[0] > wav_t.shape[1]:
                wav_t = wav_t.T  # [channels, samples]
            torchaudio.save(tmp.name, wav_t, sr)
            audio_path = tmp.name

    videos = infer(
        text=text,
        audio_path=audio_path,
        emotion=emotion,
        cond_mode=cond_mode,
        num_gen=int(num_gen),
        temperature=temperature,
        top_p=top_p,
        top_k=int(top_k),
        seed=int(seed),
    )

    # Return results: up to 3 videos in the gallery
    results = []
    for vp in videos[:3]:
        results.append(vp)

    # pad with None if fewer
    while len(results) < 3:
        results.append(None)

    return results[0], results[1], results[2]


# ─────────────────────────────────────────
# Build Gradio UI
# ─────────────────────────────────────────
def build_demo():
    with gr.Blocks(
        title="ReactMotion Demo",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # ReactMotion: Reactive Listener Motion Generation
            Generate naturalistic listener body motions from speaker utterance (text and/or audio).
            """
        )

        with gr.Row():
            # ── Left column: Inputs ──
            with gr.Column(scale=1):
                gr.Markdown("### Speaker Input")

                text_input = gr.Textbox(
                    label="Speaker Transcription",
                    placeholder="e.g., I just saw a shooting star! Did you see it too?",
                    lines=3,
                )
                audio_input = gr.Audio(
                    label="Speaker Audio (optional)",
                    type="filepath",
                )
                emotion_input = gr.Dropdown(
                    label="Speaker Emotion",
                    choices=["", "happy", "sad", "excited", "angry", "surprised",
                             "fearful", "disgusted", "neutral", "amused", "curious"],
                    value="",
                    allow_custom_value=True,
                )
                cond_mode_input = gr.Radio(
                    label="Conditioning Mode",
                    choices=["t", "t+e", "a", "a+e", "t+a", "t+a+e"],
                    value="t+e",
                )

                gr.Markdown("### Generation Settings")
                with gr.Row():
                    num_gen_input = gr.Slider(
                        label="Number of Generations", minimum=1, maximum=5, value=3, step=1,
                    )
                    seed_input = gr.Number(label="Seed", value=42, precision=0)

                with gr.Row():
                    temp_input = gr.Slider(
                        label="Temperature", minimum=0.1, maximum=2.0, value=0.85, step=0.05,
                    )
                    top_p_input = gr.Slider(
                        label="Top-p", minimum=0.1, maximum=1.0, value=0.95, step=0.05,
                    )
                    top_k_input = gr.Slider(
                        label="Top-k", minimum=10, maximum=500, value=200, step=10,
                    )

                gen_btn = gr.Button("Generate Motion", variant="primary", size="lg")

            # ── Right column: Outputs ──
            with gr.Column(scale=2):
                gr.Markdown("### Generated Listener Motions")
                with gr.Row():
                    video_out_0 = gr.Video(label="Generation 1")
                    video_out_1 = gr.Video(label="Generation 2")
                    video_out_2 = gr.Video(label="Generation 3")

        # ── Examples ──
        gr.Markdown("### Examples")
        gr.Examples(
            examples=[
                ["I just saw a shooting star streak across the sky!", None, "excited", "t+e", 3, 0.85, 0.95, 200, 42],
                ["That's really unfortunate. I'm sorry to hear about your loss.", None, "sad", "t+e", 3, 0.85, 0.95, 200, 42],
                ["Hey, guess what? I got the promotion!", None, "happy", "t+e", 3, 0.85, 0.95, 200, 42],
                ["Watch out! There's a car coming from the left!", None, "fearful", "t+e", 3, 1.05, 0.90, 100, 42],
            ],
            inputs=[text_input, audio_input, emotion_input, cond_mode_input,
                    num_gen_input, temp_input, top_p_input, top_k_input, seed_input],
        )

        # ── Connect ──
        gen_btn.click(
            fn=gradio_infer,
            inputs=[
                text_input, audio_input, emotion_input, cond_mode_input,
                num_gen_input, temp_input, top_p_input, top_k_input, seed_input,
            ],
            outputs=[video_out_0, video_out_1, video_out_2],
        )

    return demo


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="ReactMotion Gradio Demo")
    ap.add_argument("--gen_ckpt", type=str, required=True,
                    help="Path to ReactMotion generator checkpoint")
    ap.add_argument("--vqvae_ckpt", type=str, required=True,
                    help="Path to Motion VQ-VAE checkpoint")
    ap.add_argument("--mean_path", type=str, required=True,
                    help="Path to HumanML3D mean.npy")
    ap.add_argument("--std_path", type=str, required=True,
                    help="Path to HumanML3D std.npy")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true",
                    help="Create a public Gradio share link")
    args = ap.parse_args()

    # Load models
    load_models(args.gen_ckpt, args.vqvae_ckpt, args.mean_path, args.std_path)

    # Build and launch
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
