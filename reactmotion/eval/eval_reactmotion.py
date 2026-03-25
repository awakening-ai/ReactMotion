#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval/eval_reactmotion.py

Generation-only exporter that MATCHES your training pipeline:

- Uses ReactMotionNet to iterate groups (same filtering/key_by/audio_mode).
- Builds prompts with dataset.prompt_builder.build_prompt (same as Trainer).
- Supports audio_mode: none | code | wav
  - code: loads codes from npy/npz and formats tokens (same as dataset)
  - wav : encodes wav via MimiStreamingEncoder and formats tokens (same as Trainer)
- Loads tokenizer+model from --gen_ckpt (checkpoint dir).
- Generates num_gen motion sequences per group, saves:
  .motion_codes.npy, .raw_model_output.txt, .meta.json
- (Optional) caption each generated sequence via a T5 caption model.

NOTE:
- For wav mode, this script replicates Trainer-side wav->Mimi->text conversion.
"""

import os, re, json, math, argparse, random, hashlib, csv
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from tqdm import tqdm

import torch
import torchaudio
from transformers import T5Tokenizer, T5ForConditionalGeneration


# -----------------------------
# MUST MATCH YOUR PROJECT IMPORTS
# -----------------------------
from reactmotion.dataset.reactmotionnet_dataset import ReactMotionNet, ensure_2d_mono
from reactmotion.dataset.prompt_builder import build_prompt
from reactmotion.dataset.mimi_encoder import MimiStreamingEncoder
from reactmotion.dataset.audio_aug import MixedAcousticAug


# -----------------------------
# Motion token parsing (A2RM output)
# -----------------------------
_MOTION_SPAN_RE = re.compile(r"<Motion Tokens>(.*?)</Motion Tokens>", re.DOTALL)
_MOTION_TOKEN_RE = re.compile(r"<Motion Token\s+(\d+)>")
_MOTION_TOKEN_SHORT_RE = re.compile(r"<(\d+)>")  # <123>

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


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Caption model helpers (optional)
# -----------------------------
def build_caption_prompt(motion_codes: List[int]) -> str:
    motion_string = "<Motion Tokens>" + "".join([f"<{c}>" for c in motion_codes]) + "</Motion Tokens>"
    return "Generate text: " + motion_string


# -----------------------------
# misc
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def stable_hash_str(s: str) -> str:
    return hashlib.md5(str(s).encode("utf-8")).hexdigest()[:10]

def group_hash(group_key: str) -> str:
    return hashlib.md5(group_key.encode("utf-8")).hexdigest()[:12]

def save_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -----------------------------
# Wav -> Mimi -> Audio Tokens (MATCH Trainer)
# -----------------------------
def codes_to_audio_text(
    codes: torch.Tensor,
    audio_level: str,
    mimi_codebooks: int,
    mimi_cardinality: int,
) -> str:
    """
    codes: [L, T] (L=codebooks)
    MUST MATCH ReactMotionTrainer._codes_to_audio_text
    """
    arr = codes
    L = min(arr.shape[0], mimi_codebooks)
    parts = ["<Audio Tokens>"]

    def emit_level(lv: int):
        for tok in arr[lv].reshape(-1):
            t = int(tok.item())
            if 0 <= t < mimi_cardinality:
                parts.append(f"<Audio Level {lv} Token {t}>")

    if audio_level == "base":
        emit_level(0)
    elif audio_level == "all":
        for lv in range(L):
            emit_level(lv)
    elif audio_level == "rand":
        lv_num = int(torch.randint(1, L + 1, (1,)).item())
        for lv in range(lv_num):
            emit_level(lv)
    else:
        raise ValueError("audio_token_level must be base/all/rand")

    parts.append("</Audio Tokens>")
    return " ".join(parts)


@torch.no_grad()
def encode_wav_to_audio_texts(
    wav_paths: List[str],
    device: str,
    audio_sr: int,
    mimi_codebooks: int,
    mimi_chunk_frames: int,
    mimi_cardinality: int,
    audio_level: str,
    enable_audio_aug: bool,
) -> List[str]:
    """
    Batch encode wavs into <Audio Tokens> text, like Trainer._encode_wav_paths
    """
    mimi = MimiStreamingEncoder(device=device, codebooks=mimi_codebooks)
    aug = MixedAcousticAug(sr=audio_sr, device="cpu")

    wavs_cpu = []
    for p in wav_paths:
        w, sr = torchaudio.load(p)
        if sr != audio_sr:
            w = torchaudio.functional.resample(w, sr, audio_sr)
        w = ensure_2d_mono(w.to(torch.float32).cpu())
        if enable_audio_aug:
            w = aug(w)
        wavs_cpu.append(w)

    codes_list, _ = mimi.encode_many_concat(
        [w.to(device, non_blocking=True) for w in wavs_cpu],
        chunk_frames=mimi_chunk_frames,
        return_latent=False,
    )

    texts = []
    for codes in codes_list:
        txt = codes_to_audio_text(
            codes=codes.detach().cpu(),
            audio_level=audio_level,
            mimi_codebooks=mimi_codebooks,
            mimi_cardinality=mimi_cardinality,
        )
        texts.append(txt)
    return texts


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    # data (match train)
    ap.add_argument("--dataset_dir", type=str, required=True)
    ap.add_argument("--pairs_csv", type=str, required=True)
    ap.add_argument("--only_split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--key_by", type=str, default="group_id", choices=["group_id", "sayings_emotion", "sayings_only"])

    # conditioning (match train)
    ap.add_argument("--cond_mode", type=str, default="t+e",
                    choices=["t", "t+e", "a", "a+e", "t+a", "t+a+e"])
    ap.add_argument("--use_emotion", action="store_true")

    # audio (match train)
    ap.add_argument("--audio_mode", type=str, default="none", choices=["none", "code", "wav"])
    ap.add_argument("--audio_code_dir", type=str, default=None)
    ap.add_argument("--wav_dir", type=str, default=None)
    ap.add_argument("--audio_token_level", type=str, default="base", choices=["base", "all", "rand"])

    # wav->mimi (must match train args if used)
    ap.add_argument("--audio_sr", type=int, default=24000)
    ap.add_argument("--mimi_codebooks", type=int, default=8)
    ap.add_argument("--mimi_cardinality", type=int, default=2048)
    ap.add_argument("--mimi_chunk_frames", type=int, default=32)
    ap.add_argument("--enable_audio_aug", action="store_true")

    # generator ckpt (single)
    ap.add_argument("--gen_ckpt", type=str, required=True)

    # caption (optional)
    ap.add_argument("--caption_ckpt", type=str, default="")
    ap.add_argument("--no_caption", action="store_true")

    # generation knobs
    ap.add_argument("--num_gen", type=int, default=3)
    ap.add_argument("--source_len", type=int, default=512)      # match train source_len
    ap.add_argument("--gen_max_length", type=int, default=256)  # model.generate max_length
    ap.add_argument("--gen_max_len_codes", type=int, default=256)
    ap.add_argument("--gen_top_k", type=int, default=200)

    ap.add_argument("--out_dir", type=str, default="./a2rm_gen_dump_match_train")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Device]", device)
    ensure_dir(args.out_dir)

    # dataset flags inferred exactly like train_gsn_rank_flex.py
    use_transcription = ("t" in args.cond_mode)
    use_emotion = bool(args.use_emotion) and ("+e" in args.cond_mode)
    use_audio = ("a" in args.cond_mode)

    if use_audio and args.audio_mode == "none":
        print("[WARN] cond_mode includes 'a' but audio_mode=none -> audio will be empty (NOT matching train if you trained with audio).")

    # build dataset (same grouping/filtering as train)
    ds = ReactMotionNet(
        split=args.only_split,
        dataset_dir=args.dataset_dir,
        pairs_csv=args.pairs_csv,
        use_transcription=use_transcription,
        use_emotion=use_emotion,
        key_by=args.key_by,
        audio_mode=args.audio_mode,
        audio_token_level=args.audio_token_level,
        audio_code_dir=args.audio_code_dir,
        wav_dir=args.wav_dir,
        min_gold=1, min_silver=1, min_neg=1,
        min_audio=1 if args.audio_mode != "none" else 0,
    )
    print("[Groups]", len(ds))

    # load generator from ckpt (avoid vocab mismatch)
    tok = T5Tokenizer.from_pretrained(str(args.gen_ckpt))
    model = T5ForConditionalGeneration.from_pretrained(args.gen_ckpt).to(device).eval()
    ckpt_hash = stable_hash_str(args.gen_ckpt)
    print(f"[Load GEN] ckpt={args.gen_ckpt} hash={ckpt_hash}")

    # optional caption model
    cap_tok = cap_model = None
    do_caption = (not args.no_caption) and bool(args.caption_ckpt)
    if do_caption:
        cap_tok = T5Tokenizer.from_pretrained(args.caption_ckpt)
        cap_model = T5ForConditionalGeneration.from_pretrained(args.caption_ckpt).to(device).eval()
        print("[Load CAPTION]", args.caption_ckpt)
    else:
        print("[Caption] disabled")

    @torch.no_grad()
    def caption_motion_codes(codes: List[int]) -> str:
        assert cap_tok is not None and cap_model is not None
        prompt = build_caption_prompt(codes)
        inp = cap_tok(prompt, return_tensors="pt").input_ids.to(device, dtype=torch.long)
        out = cap_model.generate(inp, max_length=200, num_beams=1, do_sample=False)
        txt = cap_tok.decode(out[0], skip_special_tokens=True).strip().strip('"')
        return txt

    # diversity presets (same idea as the original script)
    diversity_presets = [
        {"temperature": 0.85, "top_p": 0.95, "top_k": args.gen_top_k},
        {"temperature": 1.05, "top_p": 0.90, "top_k": max(50, args.gen_top_k // 2)},
        {"temperature": 1.25, "top_p": 0.85, "top_k": max(20, args.gen_top_k // 4)},
    ]

    # index jsonl (group-level) + flat jsonl/csv (sample-level)
    index_path = os.path.join(args.out_dir, f"index_{args.only_split}.jsonl")
    idx_f = open(index_path, "a", encoding="utf-8")

    flat_jsonl_path = os.path.join(args.out_dir, f"index_{args.only_split}.flat.jsonl")
    flat_f = open(flat_jsonl_path, "a", encoding="utf-8")

    flat_csv_path = os.path.join(args.out_dir, f"index_{args.only_split}.flat.csv")
    csv_exists = os.path.exists(flat_csv_path) and os.path.getsize(flat_csv_path) > 0
    csv_f = open(flat_csv_path, "a", encoding="utf-8", newline="")
    csv_writer = csv.writer(csv_f)
    if not csv_exists:
        csv_writer.writerow([
            "idx",
            "group_key",
            "mode",
            "cond_head",
            "sayings",
            "caption_txt",
            "emotion",
            "audio_code_path",
            "motion_codes_npy",
            "raw_output_txt",
            "ckpt_hash",
            "group_hash",
            "split",
        ])

    mode_dir = os.path.join(args.out_dir, f"cond={args.cond_mode}", f"ckpt={ckpt_hash}")
    ensure_dir(mode_dir)

    # wav encoding cache (optional, speed)
    wav_cache: Dict[str, str] = {}
    global_idx = 0  # flat sample idx

    for qi in tqdm(range(len(ds)), desc=f"Gen[{args.cond_mode}]"):
        item = ds[qi]
        key = str(item["key"])
        gh = group_hash(f"{key}|||{args.cond_mode}")
        gdir = os.path.join(mode_dir, f"group={gh}")
        ensure_dir(gdir)

        transcription = str(item.get("transcription", ""))
        emotion = str(item.get("emotion", ""))

        # resolve audio text
        audio_text = ""
        audio_path = ""

        if use_audio:
            if args.audio_mode == "code":
                audio_text = str(item.get("audio_text", ""))
                audio_path = str(item.get("audio_code_path", ""))

            elif args.audio_mode == "wav":
                wav_path = str(item.get("wav_path", ""))
                audio_path = wav_path
                if wav_path:
                    if wav_path in wav_cache:
                        audio_text = wav_cache[wav_path]
                    else:
                        txts = encode_wav_to_audio_texts(
                            wav_paths=[wav_path],
                            device=device,
                            audio_sr=args.audio_sr,
                            mimi_codebooks=args.mimi_codebooks,
                            mimi_chunk_frames=args.mimi_chunk_frames,
                            mimi_cardinality=args.mimi_cardinality,
                            audio_level=args.audio_token_level,
                            enable_audio_aug=args.enable_audio_aug,
                        )
                        audio_text = txts[0]
                        wav_cache[wav_path] = audio_text

        # prompt MUST match trainer usage
        prompt = build_prompt(
            speaker_transcription=transcription,
            speaker_audio=audio_text if use_audio else "",
            speaker_emotion=emotion,
            use_transcription=use_transcription,
            use_audio=use_audio,
            use_emotion=use_emotion,
        )

        enc = tok(
            prompt,
            padding=False,
            truncation=True,
            max_length=int(args.source_len),
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device, dtype=torch.long)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device, dtype=torch.long)

        # deterministic base seed per group
        base_seed = int(hashlib.md5(f"{key}|||{args.cond_mode}".encode("utf-8")).hexdigest()[:8], 16) ^ int(args.seed)

        saved_items = []

        for gi in range(int(args.num_gen)):
            preset = diversity_presets[gi % len(diversity_presets)]
            gen_seed = (base_seed + 1009 * (gi + 1)) & 0x7FFFFFFF
            seed_everything(gen_seed)

            gen_kwargs = dict(
                max_length=int(args.gen_max_length),
                do_sample=True,
                temperature=float(preset["temperature"]),
                top_k=int(preset["top_k"]),
                top_p=float(preset["top_p"]),
                num_return_sequences=1,
            )

            if attention_mask is not None:
                out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
            else:
                out = model.generate(input_ids=input_ids, **gen_kwargs)

            out_text = tok.decode(out[0], skip_special_tokens=False)
            out_text = out_text.replace("<pad>", "").replace("</s>", "").strip()

            codes = parse_motion_tokens(out_text, max_len=int(args.gen_max_len_codes), codebook_size=512)
            if len(codes) == 0:
                codes = [1] * min(int(args.gen_max_len_codes), 196)

            cap = ""
            if do_caption:
                cap = caption_motion_codes(codes)

            stem = f"{args.cond_mode}__{ckpt_hash}__{gh}__gen{gi:02d}"
            code_path = os.path.join(gdir, stem + ".motion_codes.npy")
            raw_path = os.path.join(gdir, stem + ".raw_model_output.txt")
            meta_path = os.path.join(gdir, stem + ".meta.json")
            cap_path = os.path.join(gdir, stem + ".caption.txt")

            np.save(code_path, np.asarray(codes, dtype=np.int32))
            save_text(raw_path, out_text + "\n")
            if do_caption:
                save_text(cap_path, cap + "\n")

            meta = {
                "cond_mode": args.cond_mode,
                "gen_ckpt": args.gen_ckpt,
                "ckpt_hash": ckpt_hash,
                "split": args.only_split,
                "key_by": args.key_by,
                "key": key,
                "group_hash": gh,
                "use_transcription": use_transcription,
                "use_audio": use_audio,
                "use_emotion": use_emotion,
                "transcription": transcription,
                "emotion": emotion,
                "audio_mode": args.audio_mode,
                "audio_path": audio_path,
                "audio_token_level": args.audio_token_level if use_audio else None,
                "group_w": float(item.get("group_w", 1.0)),
                "gen_idx": gi,
                "gen_seed": int(gen_seed),
                "sampling": preset,
                "paths": {
                    "motion_codes_npy": code_path,
                    "raw_output_txt": raw_path,
                    "caption_txt": cap_path if do_caption else "",
                },
            }
            save_json(meta_path, meta)
            saved_items.append(meta)

            # --- flat jsonl + csv (per-sample; convenient for scorer etc.) ---
            flat_rec = {
                "idx": global_idx,
                "group_key": key,
                "mode": args.cond_mode,
                "cond_head": "fused",
                "sayings": transcription,
                "caption_txt": cap if do_caption else "",
                "emotion": emotion,
                "audio_code_path": audio_path if use_audio else "",
                "motion_codes_npy": code_path,
                "raw_output_txt": raw_path,
                "ckpt_hash": ckpt_hash,
                "group_hash": gh,
                "split": args.only_split,
            }
            flat_f.write(json.dumps(flat_rec, ensure_ascii=False) + "\n")
            csv_writer.writerow([
                flat_rec["idx"],
                flat_rec["group_key"],
                flat_rec["mode"],
                flat_rec["cond_head"],
                flat_rec["sayings"],
                flat_rec["caption_txt"],
                flat_rec["emotion"],
                flat_rec["audio_code_path"],
                flat_rec["motion_codes_npy"],
                flat_rec["raw_output_txt"],
                flat_rec["ckpt_hash"],
                flat_rec["group_hash"],
                flat_rec["split"],
            ])
            global_idx += 1

        idx_f.write(json.dumps({
            "cond_mode": args.cond_mode,
            "gen_ckpt": args.gen_ckpt,
            "ckpt_hash": ckpt_hash,
            "split": args.only_split,
            "key": key,
            "group_hash": gh,
            "num_gen": len(saved_items),
            "items": [it["paths"] for it in saved_items],
        }, ensure_ascii=False) + "\n")
        idx_f.flush()
        os.fsync(idx_f.fileno())

    idx_f.close()
    flat_f.close()
    csv_f.close()
    print("[Done] Saved to:", args.out_dir)
    print("[Index group-level]", index_path)
    print("[Index flat jsonl]", flat_jsonl_path)
    print("[Index flat csv]", flat_csv_path)


if __name__ == "__main__":
    main()
