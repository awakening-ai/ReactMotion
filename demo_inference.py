#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReactMotion Inference Demo
==========================

Generate listener motion from speaker utterance (text / audio / both),
decode via Motion VQ-VAE, and render a stick-figure video.

Usage examples:
  # Text-only (speaker transcription)
  python demo_inference.py \
    --gen_ckpt  logs/checkpoints/checkpoint-XXXXX \
    --vqvae_ckpt external/pretrained_vqvae/t2m.pth \
    --mean_path  external/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy \
    --std_path   external/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy \
    --text "I just saw a shooting star streak across the sky! Did you see it too?" \
    --emotion "excited" \
    --out_path outputs/demo_text.mp4

  # Audio input (wav file, encoded with Mimi)
  python demo_inference.py \
    --gen_ckpt  logs/checkpoints/checkpoint-XXXXX \
    --vqvae_ckpt external/pretrained_vqvae/t2m.pth \
    --mean_path  external/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy \
    --std_path   external/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy \
    --audio  samples/speaker.wav \
    --cond_mode a+e --emotion "happy" \
    --out_path outputs/demo_audio.mp4

  # Text + Audio
  python demo_inference.py \
    --gen_ckpt  logs/checkpoints/checkpoint-XXXXX \
    --vqvae_ckpt external/pretrained_vqvae/t2m.pth \
    --mean_path  external/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy \
    --std_path   external/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy \
    --text "Let's celebrate!" --audio samples/speaker.wav \
    --cond_mode t+a+e --emotion "happy" \
    --out_path outputs/demo_fused.mp4
"""

import os
import re
import argparse
import random
from typing import List, Optional

import numpy as np
import torch
import torchaudio
import imageio

from transformers import T5Tokenizer, T5ForConditionalGeneration

from reactmotion.dataset.prompt_builder import build_prompt
from reactmotion.models.vqvae import HumanVQVAE
from reactmotion.utils.motion_process import recover_from_ric
from reactmotion.visualization.plot_3d_global import plot_3d_motion


# ─────────────────────────────────────────
# Motion token parsing
# ─────────────────────────────────────────
_MOTION_SPAN_RE = re.compile(r"<Motion Tokens>(.*?)</Motion Tokens>", re.DOTALL)
_MOTION_TOKEN_RE = re.compile(r"<Motion Token\s+(\d+)>")
_MOTION_TOKEN_SHORT_RE = re.compile(r"<(\d+)>")


def parse_motion_tokens(text: str, max_len: int = 200, codebook_size: int = 512) -> List[int]:
    if text is None:
        return []
    s = str(text)
    m = _MOTION_SPAN_RE.search(s)
    span = m.group(1) if m else s
    codes = [int(x) for x in _MOTION_TOKEN_RE.findall(span)]
    if len(codes) == 0:
        codes = [int(x) for x in _MOTION_TOKEN_SHORT_RE.findall(span)]
    out = []
    for c in codes:
        if 0 <= c < codebook_size:
            out.append(c)
        else:
            break
    return out[:max_len]


# ─────────────────────────────────────────
# Audio encoding (wav → Mimi tokens → text)
# ─────────────────────────────────────────
def encode_audio_to_text(
    wav_path: str,
    device: str = "cuda",
    audio_sr: int = 24000,
    mimi_codebooks: int = 8,
    mimi_chunk_frames: int = 32,
    mimi_cardinality: int = 2048,
    audio_level: str = "base",
) -> str:
    """Encode a wav file into <Audio Tokens> text using Mimi."""
    from reactmotion.dataset.mimi_encoder import MimiStreamingEncoder

    mimi = MimiStreamingEncoder(device=device, codebooks=mimi_codebooks)

    wav, sr = torchaudio.load(wav_path)
    if sr != audio_sr:
        wav = torchaudio.functional.resample(wav, sr, audio_sr)
    # ensure mono [1, T]
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.to(torch.float32)

    codes_list, _ = mimi.encode_many_concat(
        [wav.to(device)],
        chunk_frames=mimi_chunk_frames,
        return_latent=False,
    )
    codes = codes_list[0].detach().cpu()  # [K, T]

    # format tokens
    L = min(codes.shape[0], mimi_codebooks)
    parts = ["<Audio Tokens>"]
    if audio_level == "base":
        for tok in codes[0].reshape(-1):
            t = int(tok.item())
            if 0 <= t < mimi_cardinality:
                parts.append(f"<Audio Level 0 Token {t}>")
    elif audio_level == "all":
        for lv in range(L):
            for tok in codes[lv].reshape(-1):
                t = int(tok.item())
                if 0 <= t < mimi_cardinality:
                    parts.append(f"<Audio Level {lv} Token {t}>")
    parts.append("</Audio Tokens>")
    return " ".join(parts)


# ─────────────────────────────────────────
# VQ-VAE builder
# ─────────────────────────────────────────
class _VQVAEArgs:
    """Minimal args for HumanVQVAE (t2m config)."""
    dataname = "t2m"
    code_dim = 512
    nb_code = 512
    output_emb_width = 512
    down_t = 2
    stride_t = 2
    width = 512
    depth = 3
    dilation_growth_rate = 3
    vq_act = "relu"
    vq_norm = None
    quantizer = "ema_reset"
    beta = 1.0
    mu = 0.99


def build_vqvae(ckpt_path: str, device: str = "cuda") -> HumanVQVAE:
    args = _VQVAEArgs()
    vae = HumanVQVAE(
        args,
        nb_code=args.nb_code,
        code_dim=args.code_dim,
        output_emb_width=args.output_emb_width,
        down_t=args.down_t,
        stride_t=args.stride_t,
        width=args.width,
        depth=args.depth,
        dilation_growth_rate=args.dilation_growth_rate,
        activation=args.vq_act,
        norm=args.vq_norm,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = ckpt.get("net", ckpt)  # handle both {net: ...} and raw state_dict
    vae.load_state_dict(state, strict=True)
    vae = vae.to(device).eval()
    print(f"[VQ-VAE] loaded from {ckpt_path}")
    return vae


# ─────────────────────────────────────────
# Decode motion codes → joint positions
# ─────────────────────────────────────────
@torch.no_grad()
def decode_motion_codes(
    codes: List[int],
    vae: HumanVQVAE,
    mean_np: np.ndarray,
    std_np: np.ndarray,
    device: str = "cuda",
) -> np.ndarray:
    """
    Decode discrete motion codes → (T, 22, 3) joint positions.

    Pipeline:
      codes → VQ-VAE decoder → [T, 263] normalized features
      → denormalize with mean/std
      → recover_from_ric → [T, 22, 3] joint positions
    """
    idx = torch.from_numpy(np.array(codes, dtype=np.int64)).unsqueeze(0).to(device)
    pose_norm = vae.forward_decoder(idx)[0]  # [T, 263]

    # denormalize
    mean_t = torch.from_numpy(mean_np).to(device)
    std_t = torch.from_numpy(std_np).to(device)
    pose = pose_norm * std_t + mean_t  # [T, 263]

    # recover 3D joint positions from HumanML3D ric representation
    joints = recover_from_ric(pose.unsqueeze(0), joints_num=22)  # [1, T, 22, 3]
    joints = joints.squeeze(0).cpu().numpy()  # [T, 22, 3]
    return joints


# ─────────────────────────────────────────
# Render joints → video
# ─────────────────────────────────────────
def render_motion_video(
    joints: np.ndarray,
    out_path: str,
    title: Optional[str] = None,
    fps: int = 20,
) -> str:
    """Render (T, 22, 3) joints to an mp4 stick-figure video."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    title_arg = [title, ""] if title else None
    frames = plot_3d_motion(
        [joints, None, title_arg],
        figsize=(6, 6),
        fps=fps,
        fixed_camera=True,
        draw_traj=True,
        draw_ground=False,
    )  # (T, H, W, 3) uint8

    imageio.mimsave(out_path, frames, fps=fps)
    print(f"[Video] saved to {out_path}  ({frames.shape[0]} frames, {fps} fps)")
    return out_path


# ─────────────────────────────────────────
# Full inference pipeline
# ─────────────────────────────────────────
@torch.no_grad()
def generate_motion(
    tokenizer: T5Tokenizer,
    model: T5ForConditionalGeneration,
    text: str = "",
    audio_text: str = "",
    emotion: str = "",
    cond_mode: str = "t+e",
    source_len: int = 512,
    max_length: int = 256,
    temperature: float = 0.85,
    top_p: float = 0.95,
    top_k: int = 200,
    num_gen: int = 1,
    device: str = "cuda",
) -> List[List[int]]:
    """Generate motion token sequences from speaker input."""
    use_transcription = "t" in cond_mode
    use_audio = "a" in cond_mode
    use_emotion = "+e" in cond_mode or cond_mode == "e"

    prompt = build_prompt(
        speaker_transcription=text,
        speaker_audio=audio_text if use_audio else "",
        speaker_emotion=emotion,
        use_transcription=use_transcription,
        use_audio=use_audio,
        use_emotion=use_emotion,
    )

    enc = tokenizer(
        prompt,
        padding=False,
        truncation=True,
        max_length=source_len,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device, dtype=torch.long)
    attention_mask = enc["attention_mask"].to(device, dtype=torch.long)

    all_codes = []
    for i in range(num_gen):
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=1,
        )
        out_text = tokenizer.decode(out[0], skip_special_tokens=False)
        out_text = out_text.replace("<pad>", "").replace("</s>", "").strip()

        codes = parse_motion_tokens(out_text, max_len=200, codebook_size=512)
        if len(codes) == 0:
            print(f"[WARN] generation {i} produced no valid motion tokens, using fallback")
            codes = [0] * 50

        all_codes.append(codes)
        print(f"  [Gen {i}] {len(codes)} motion tokens")

    return all_codes


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="ReactMotion Inference Demo")

    # Model checkpoints
    ap.add_argument("--gen_ckpt", type=str, required=True,
                    help="Path to ReactMotion generator checkpoint (T5 dir)")
    ap.add_argument("--vqvae_ckpt", type=str, required=True,
                    help="Path to Motion VQ-VAE checkpoint (.bin or .pt)")
    ap.add_argument("--mean_path", type=str, required=True,
                    help="Path to HumanML3D mean.npy")
    ap.add_argument("--std_path", type=str, required=True,
                    help="Path to HumanML3D std.npy")

    # Input
    ap.add_argument("--text", type=str, default="",
                    help="Speaker transcription text")
    ap.add_argument("--audio", type=str, default="",
                    help="Path to speaker audio wav file")
    ap.add_argument("--emotion", type=str, default="",
                    help="Speaker emotion label (e.g., happy, sad, excited)")
    ap.add_argument("--cond_mode", type=str, default="t+e",
                    choices=["t", "t+e", "a", "a+e", "t+a", "t+a+e"],
                    help="Conditioning modalities")

    # Audio encoding
    ap.add_argument("--audio_sr", type=int, default=24000)
    ap.add_argument("--mimi_codebooks", type=int, default=8)
    ap.add_argument("--mimi_cardinality", type=int, default=2048)
    ap.add_argument("--mimi_chunk_frames", type=int, default=32)
    ap.add_argument("--audio_token_level", type=str, default="base",
                    choices=["base", "all"])

    # Generation
    ap.add_argument("--num_gen", type=int, default=3,
                    help="Number of diverse motions to generate")
    ap.add_argument("--source_len", type=int, default=512)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)

    # Output
    ap.add_argument("--out_path", type=str, default="outputs/demo.mp4",
                    help="Output video path (for single gen) or prefix (for multi gen)")
    ap.add_argument("--fps", type=int, default=20)

    args = ap.parse_args()

    # ── Seed ──
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # ── Load generator ──
    print(f"[Generator] loading from {args.gen_ckpt}")
    tokenizer = T5Tokenizer.from_pretrained(args.gen_ckpt)
    model = T5ForConditionalGeneration.from_pretrained(args.gen_ckpt).to(device).eval()
    print(f"[Generator] vocab_size={len(tokenizer)}, params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # ── Load VQ-VAE ──
    vae = build_vqvae(args.vqvae_ckpt, device=device)
    mean_np = np.load(args.mean_path).astype(np.float32)
    std_np = np.load(args.std_path).astype(np.float32)
    print(f"[VQ-VAE] mean shape={mean_np.shape}, std shape={std_np.shape}")

    # ── Encode audio if provided ──
    audio_text = ""
    if args.audio and "a" in args.cond_mode:
        print(f"[Audio] encoding {args.audio} with Mimi ...")
        audio_text = encode_audio_to_text(
            wav_path=args.audio,
            device=device,
            audio_sr=args.audio_sr,
            mimi_codebooks=args.mimi_codebooks,
            mimi_chunk_frames=args.mimi_chunk_frames,
            mimi_cardinality=args.mimi_cardinality,
            audio_level=args.audio_token_level,
        )
        print(f"[Audio] encoded to {len(audio_text.split())} tokens")

    # ── Generate ──
    print(f"\n{'='*60}")
    print(f"Speaker Text:    {args.text or '(none)'}")
    print(f"Speaker Audio:   {args.audio or '(none)'}")
    print(f"Speaker Emotion: {args.emotion or '(none)'}")
    print(f"Cond Mode:       {args.cond_mode}")
    print(f"Num Generations: {args.num_gen}")
    print(f"{'='*60}\n")

    all_codes = generate_motion(
        tokenizer=tokenizer,
        model=model,
        text=args.text,
        audio_text=audio_text,
        emotion=args.emotion,
        cond_mode=args.cond_mode,
        source_len=args.source_len,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_gen=args.num_gen,
        device=device,
    )

    # ── Decode & Render ──
    base, ext = os.path.splitext(args.out_path)
    if not ext:
        ext = ".mp4"

    for i, codes in enumerate(all_codes):
        print(f"\n[Decode {i}] {len(codes)} codes → joints ...")
        joints = decode_motion_codes(codes, vae, mean_np, std_np, device=device)
        print(f"  joints shape: {joints.shape}")

        # save motion codes
        code_path = f"{base}_gen{i}.motion_codes.npy"
        np.save(code_path, np.array(codes, dtype=np.int32))

        # render video
        if args.num_gen == 1:
            vid_path = f"{base}{ext}"
        else:
            vid_path = f"{base}_gen{i}{ext}"

        title = args.text[:80] if args.text else (args.audio or "Generated Motion")
        render_motion_video(joints, vid_path, title=title, fps=args.fps)

    print(f"\n[Done] All outputs saved to {os.path.dirname(args.out_path) or '.'}")


if __name__ == "__main__":
    main()
