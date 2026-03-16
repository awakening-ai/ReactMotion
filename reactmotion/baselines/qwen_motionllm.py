#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_to_motion_mg_motionllm.py

用 external/MG-MotionLLM 从 Qwen 生成的 captions 生成 motion tokens。

流程:
  1. 读取 qwen_meta.jsonl（每行: key, sayings, emotion, captions, ...）
  2. 对每条 caption，用 MG-MotionLLM T5 模型生成 motion tokens
  3. 保存 tokens 到 out_dir/tokens/<key>/m{idx}.npy，并写 meta.jsonl

依赖:
  - external/MG-MotionLLM 的 t2m 模型（如 t2m-ft-from-GSPretrained-base）
  - transformers, torch

Usage:
  python -m eval.qwen_to_motion_mg_motionllm \
    --qwen_meta ./out_qwen/qwen_meta.jsonl \
    --mg_model ./t2m-ft-from-GSPretrained-base/checkpoint-300000 \
    --out_dir ./out_qwen_mg_tokens \
    --instruction "Generate motion: " \
    --max_new_tokens 200 \
    --temperature 0.8 \
    --seed 42
"""

import os
import re
import json
import argparse
from os.path import join as pjoin
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration


def parse_motion_tokens_from_output(output_text: str, max_token: int = 511) -> List[int]:
    """从 MG-MotionLLM 输出解析 motion token IDs。"""
    output = re.findall(r"\d+", output_text)
    tokens = []
    for num in output:
        n = int(num)
        if n > max_token:
            break
        tokens.append(n)
    return tokens


def save_motion_tokens(path: str, tokens: np.ndarray, meta: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, tokens.astype(np.int32))
    meta_path = path + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen_meta", type=str, required=True,
                    help="Qwen 输出的 qwen_meta.jsonl 路径")
    ap.add_argument("--mg_model", type=str, required=True,
                    help="MG-MotionLLM t2m 模型路径 (HuggingFace 或本地 checkpoint 目录)")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="输出目录，会创建 tokens/ 和 meta.jsonl")
    ap.add_argument("--instruction", type=str, default="Generate motion: ",
                    help="Text-to-motion 的 prompt 前缀")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=200)
    ap.add_argument("--do_sample", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_token", type=int, default=511,
                    help="Motion token 最大值，超过则截断")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    tokens_root = pjoin(args.out_dir, "tokens")
    meta_path = pjoin(args.out_dir, "meta.jsonl")

    # Load MG-MotionLLM T5 model
    print(f"[Load] MG-MotionLLM from {args.mg_model}")
    tokenizer = T5Tokenizer.from_pretrained(args.mg_model)
    model = T5ForConditionalGeneration.from_pretrained(args.mg_model)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Read qwen_meta
    with open(args.qwen_meta, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    print(f"[Run] {len(lines)} groups, generating motion tokens ...")
    with open(meta_path, "w", encoding="utf-8") as fw:
        for i, ln in enumerate(tqdm(lines, desc="MG-MotionLLM")):
            obj = json.loads(ln)
            key = str(obj["key"])
            sayings = str(obj.get("sayings", ""))
            emotion = str(obj.get("emotion", ""))
            cond_mode = str(obj.get("cond_mode", "t+e"))
            captions: List[str] = [str(x) for x in obj.get("captions", [])]

            base_seed = args.seed + i * 1000
            token_paths: List[str] = []

            for j, caption in enumerate(captions):
                m_seed = base_seed + j
                torch.manual_seed(m_seed)
                prompt = args.instruction + caption

                input_ids = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                ).input_ids.to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        input_ids,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature if args.do_sample else 1.0,
                        top_k=args.top_k if args.do_sample else 50,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )

                output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                tokens = parse_motion_tokens_from_output(output_text, max_token=args.max_token)

                if len(tokens) == 0:
                    tokens = [0]  # fallback

                out_path = pjoin(tokens_root, str(key), f"m{j}.npy")
                meta = dict(
                    key=key,
                    sayings=sayings,
                    emotion=emotion,
                    cond_mode=cond_mode,
                    caption=caption,
                    seed=m_seed,
                    split=obj.get("split", "test"),
                    idx=i,
                    sample=j,
                )
                save_motion_tokens(out_path, np.array(tokens), meta)
                token_paths.append(out_path)

            rec = dict(
                key=key,
                sayings=sayings,
                emotion=emotion,
                cond_mode=cond_mode,
                captions=captions,
                tokens=token_paths,
                split=obj.get("split", "test"),
            )
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[Done] meta: {meta_path}")
    print(f"[Done] tokens: {tokens_root}")


if __name__ == "__main__":
    main()
