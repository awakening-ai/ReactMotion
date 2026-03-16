#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finetune_qwen_lora.py

LoRA fine-tune Qwen3-30B-A3B on the listener motion caption task.

Data  : new_data/train.csv  (rows where label == 'gold')
Input : speaker transcription  (+optional emotion when --cond_mode t+e)
Target: listener motion caption  -> JSON {"actions": [...]}

Prompt content mirrors eval/qwen_t2m_pipeline.py::build_prompt() exactly,
but is delivered via the Qwen3 chat template (system + user + assistant)
so the instruct model is fine-tuned in its native format.

Usage:
    python train/finetune_qwen_lora.py \
        --model_path /ibex/project/c2191/luoc/LLM_checkpoints/qwen3-30b-a3b-thinking-2507 \
        --train_csv  new_data/train.csv \
        --val_csv    new_data/val.csv   \
        --cond_mode  t+e                \
        --output_dir checkpoints/qwen_lora_t2e
"""

import os
import re
import json
import math
import random
import logging
import argparse
import functools
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt helpers  (mirrors eval/qwen_t2m_pipeline.py::build_prompt exactly)
# ─────────────────────────────────────────────────────────────────────────────

# The first three lines of build_prompt become the system message so the model
# receives the same semantic content in its native chat format.
SYSTEM_PROMPT = (
    "You are an expert animator for dyadic conversation.\n"
    "Given the speaker utterance (and optional emotion), propose natural "
    "LISTENER nonverbal reactions.\n"
    "Output STRICT JSON only. No markdown."
)


def normalize_text(x) -> str:
    s = "" if pd.isna(x) else str(x)
    return re.sub(r"\s+", " ", s).strip()


def build_user_message(sayings: str, emotion: str, n: int, cond_mode: str) -> str:
    """
    Mirrors the body of qwen_t2m_pipeline.py::build_prompt() (after the
    system-level preamble), so prompt content is identical at train & eval time.
    """
    sayings = sayings.strip()
    emo = (emotion or "").strip()
    use_emo = (
        cond_mode == "t+e"
        and bool(emo)
        and emo.lower() not in ("nan", "none", "")
    )
    emoline = f"Speaker emotion: {emo}\n" if use_emo else ""

    return (
        f"Speaker utterance: {sayings}\n"
        f"{emoline}"
        f"Return JSON with key `actions`, an array of exactly {n} concise action plans.\n"
        "Each action plan should be 1 sentence describing listener motion style "
        "(head, gaze, torso, hands).\n"
        "Do NOT mention camera or scene.\n"
        "Example:\n"
        '{"actions": ["...", "...", "..."]}'
    )


def build_assistant_response(captions: List[str]) -> str:
    return json.dumps({"actions": captions}, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MotionCaptionDataset(Dataset):
    """
    Reads a CSV, keeps only label=='gold' rows, groups by (sayings, emotion),
    and builds one training example per unique (sayings, emotion) pair.
    Pre-tokenizes all examples in __init__ so __getitem__ is O(1) (no tokenizer
    calls during training), which greatly speeds up training.
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer,
        cond_mode: str = "t+e",
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.cond_mode = cond_mode
        self.max_length = max_length

        df = pd.read_csv(csv_path, encoding="utf-8")
        gold = df[df["label"] == "gold"].copy()
        gold["sayings"] = gold["sayings"].map(normalize_text)
        gold["emotion"] = gold["emotion"].astype(str).map(normalize_text)
        gold["motion_caption"] = gold["motion_caption"].astype(str).map(normalize_text)

        groups = (
            gold.groupby(["sayings", "emotion"])["motion_caption"]
            .apply(list)
            .reset_index()
        )

        samples_raw: List[dict] = []
        for _, row in groups.iterrows():
            sayings = row["sayings"]
            emotion = row["emotion"]
            captions = [c for c in row["motion_caption"] if c]
            if not captions:
                continue
            n = len(captions)
            user_msg = build_user_message(sayings, emotion, n, cond_mode)
            response = build_assistant_response(captions)
            samples_raw.append(dict(user_msg=user_msg, response=response))

        # Pre-tokenize once so training loop never calls tokenizer
        self.cache: List[dict] = []
        for s in samples_raw:
            messages_full = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": s["user_msg"]},
                {"role": "assistant", "content": s["response"]},
            ]
            messages_prompt = messages_full[:-1]

            full_text = tokenizer.apply_chat_template(
                messages_full, tokenize=False, add_generation_prompt=False
            )
            prompt_text = tokenizer.apply_chat_template(
                messages_prompt, tokenize=False, add_generation_prompt=True
            )

            full_ids = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )["input_ids"]
            prompt_ids = tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )["input_ids"]

            prompt_len = len(prompt_ids)
            seq_len = len(full_ids)
            labels = [-100] * prompt_len + full_ids[prompt_len:]
            if len(labels) < seq_len:
                labels += [-100] * (seq_len - len(labels))
            labels = labels[:seq_len]

            self.cache.append({
                "input_ids": full_ids,
                "labels": labels,
                "attention_mask": [1] * seq_len,
            })

        logger.info(
            f"Loaded & pre-tokenized {len(self.cache)} examples "
            f"(from {len(gold)} gold rows in {csv_path})"
        )

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, idx: int) -> dict:
        c = self.cache[idx]
        return {
            "input_ids":      torch.tensor(c["input_ids"],      dtype=torch.long),
            "labels":         torch.tensor(c["labels"],         dtype=torch.long),
            "attention_mask": torch.tensor(c["attention_mask"], dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Collator  (left-pad to batch max length)
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: List[dict], pad_token_id: int) -> dict:
    max_len = max(item["input_ids"].size(0) for item in batch)

    def _pad(tensors, pad_val):
        return torch.stack(
            [
                torch.cat(
                    [t, torch.full((max_len - t.size(0),), pad_val, dtype=t.dtype)]
                )
                for t in tensors
            ]
        )

    return {
        "input_ids":      _pad([x["input_ids"]      for x in batch], pad_token_id),
        "labels":         _pad([x["labels"]          for x in batch], -100),
        "attention_mask": _pad([x["attention_mask"]  for x in batch], 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen for listener motion captions")

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument("--train_csv",  type=str, default="./new_data/train.csv")
    parser.add_argument("--val_csv",    type=str, default="./new_data/val.csv")
    parser.add_argument(
        "--cond_mode", type=str, default="t+e", choices=["t", "t+e"],
        help="t: speaker text only;  t+e: text + emotion",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model_path", type=str,
        default="/ibex/project/c2191/luoc/LLM_checkpoints/qwen3-30b-a3b-thinking-2507",
    )
    parser.add_argument("--output_dir", type=str, default="./checkpoints/qwen_lora")
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument(
        "--attn_implementation", type=str, default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="sdpa=fast built-in; flash_attention_2=faster but needs pip install flash-attn",
    )

    # ── LoRA ──────────────────────────────────────────────────────────────────
    parser.add_argument("--lora_r",       type=int,   default=16)
    parser.add_argument("--lora_alpha",   type=int,   default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules", type=str,
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated list of linear module names to apply LoRA to",
    )

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument("--num_epochs",                    type=int,   default=3)
    parser.add_argument("--per_device_train_batch_size",   type=int,   default=1)
    parser.add_argument("--per_device_eval_batch_size",    type=int,   default=1)
    parser.add_argument("--gradient_accumulation_steps",   type=int,   default=16)
    parser.add_argument("--learning_rate",                 type=float, default=2e-4)
    parser.add_argument("--max_length",                    type=int,   default=512)
    parser.add_argument("--warmup_ratio",                  type=float, default=0.05)
    parser.add_argument("--lr_scheduler_type",             type=str,   default="cosine")
    parser.add_argument("--save_steps",                    type=int,   default=100)
    parser.add_argument("--eval_steps",                    type=int,   default=100)
    parser.add_argument("--logging_steps",                 type=int,   default=10)
    parser.add_argument("--seed",                          type=int,   default=42)
    parser.add_argument(
        "--load_best_model_at_end", action="store_true", default=False,
        help="Save & reload the checkpoint with lowest eval loss",
    )

    args = parser.parse_args()

    # ── Reproducibility ───────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    logger.info(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ─────────────────────────────────────────────────────────────────
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    logger.info(
        f"Loading model from {args.model_path}  (dtype={args.dtype}, "
        f"attn={args.attn_implementation})"
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto",
        attn_implementation=args.attn_implementation,
    )
    # Required when using gradient checkpointing with PEFT
    model.enable_input_require_grads()

    # ── LoRA ──────────────────────────────────────────────────────────────────
    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = MotionCaptionDataset(
        args.train_csv,
        tokenizer,
        cond_mode=args.cond_mode,
        max_length=args.max_length,
    )

    val_dataset = None
    has_val = os.path.isfile(args.val_csv)
    if has_val:
        val_dataset = MotionCaptionDataset(
            args.val_csv,
            tokenizer,
            cond_mode=args.cond_mode,
            max_length=args.max_length,
        )

    # ── Training arguments ────────────────────────────────────────────────────
    # Only keep 2 checkpoints on disk (current + one previous); best/current saved at end
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if has_val else "no",
        eval_steps=args.eval_steps if has_val else None,
        save_strategy="steps",
        fp16=(args.dtype == "float16"),
        bf16=(args.dtype == "bfloat16"),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        load_best_model_at_end=args.load_best_model_at_end and has_val,
        metric_for_best_model="eval_loss" if has_val else None,
        greater_is_better=False,
        ddp_find_unused_parameters=False,
    )

    _collate = functools.partial(collate_fn, pad_token_id=tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=_collate,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Starting training ...")
    trainer.train()

    # ── Save only best and current LoRA ───────────────────────────────────────
    import glob
    import shutil

    best_dir = os.path.join(args.output_dir, "best")
    current_dir = os.path.join(args.output_dir, "current")

    # Model in memory is the best one (if load_best_model_at_end and has_val)
    logger.info(f"Saving best LoRA to {best_dir}")
    os.makedirs(best_dir, exist_ok=True)
    trainer.model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)

    # Copy the latest checkpoint-* to current/ (last step's LoRA)
    checkpoints = glob.glob(os.path.join(args.output_dir, "checkpoint-*"))
    if checkpoints:
        def _step(p):
            name = os.path.basename(p.rstrip("/"))
            m = re.search(r"checkpoint-(\d+)$", name)
            return int(m.group(1)) if m else 0
        latest_ckpt = max(checkpoints, key=_step)
        logger.info(f"Copying current (latest) LoRA from {latest_ckpt} to {current_dir}")
        if os.path.isdir(current_dir):
            shutil.rmtree(current_dir)
        shutil.copytree(latest_ckpt, current_dir)
        # Remove intermediate checkpoint dirs so only best/ and current/ remain
        for d in checkpoints:
            shutil.rmtree(d, ignore_errors=True)
    else:
        # No step checkpoints (e.g. very short run): best is also current
        if not os.path.samefile(best_dir, current_dir):
            if os.path.isdir(current_dir):
                shutil.rmtree(current_dir)
            shutil.copytree(best_dir, current_dir)

    logger.info("Done. LoRA saved under: best/, current/")


if __name__ == "__main__":
    main()
