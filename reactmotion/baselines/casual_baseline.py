#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_qwen_t2m.py

Use Qwen to sample 3 plausible listener reaction action plans from:
  (speaker sayings + emotion)
Then feed each plan into a Text-to-Human model to generate 3 motions.

Outputs:
  out_dir/
    motions/{group_key}/m0.npz, m1.npz, m2.npz
    meta.jsonl
"""

import os, re, json, time, random, argparse
from os.path import join as pjoin
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# -------------------------
# utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def normalize_text(x: Any) -> str:
    s = "" if pd.isna(x) else str(x)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def read_split_csv(pairs_csv_or_dir: str, split: str) -> pd.DataFrame:
    if os.path.isdir(pairs_csv_or_dir):
        path = pjoin(pairs_csv_or_dir, f"{split}.csv")
        if not os.path.isfile(path):
            raise RuntimeError(f"Missing split csv: {path}")
        return pd.read_csv(path, encoding="utf-8")
    df = pd.read_csv(pairs_csv_or_dir, encoding="utf-8")
    if "split" not in df.columns:
        raise RuntimeError("pairs_csv is a file but has no `split` column. Provide dir with train/val/test.")
    sp = df["split"].astype(str).str.lower().str.strip()
    return df[sp == split].copy()

def group_key_from_row(row: pd.Series, key_by: str) -> str:
    if key_by == "group_id":
        return str(row["group_id"])
    if key_by == "sayings_only":
        return str(row["sayings"])
    # sayings_emotion
    return f"{row['sayings']}|||{row['emotion']}"

# -------------------------
# Qwen: vLLM (preferred) or HF fallback
# -------------------------
class QwenSampler:
    def __init__(
        self,
        model_path: str,
        backend: str = "vllm",  # vllm | hf
        tp: int = 1,
        dtype: str = "bfloat16",
        max_model_len: int = 4096,
        device: str = "cuda",
    ):
        self.model_path = model_path
        self.backend = backend
        self.tp = int(tp)
        self.dtype = dtype
        self.max_model_len = int(max_model_len)
        self.device = device

        self._init_ok = False
        self.llm = None
        self.tok = None
        self.model = None

        if backend == "vllm":
            try:
                from vllm import LLM
                self.llm = LLM(
                    model=model_path,
                    tensor_parallel_size=self.tp,
                    dtype=self.dtype,
                    max_model_len=self.max_model_len,
                    trust_remote_code=True,
                )
                self._init_ok = True
            except Exception as e:
                print(f"[WARN] vLLM init failed -> fallback to HF. err={e}")
                self.backend = "hf"

        if self.backend == "hf":
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16,
                device_map="auto",   # let HF shard across GPUs if possible
            )
            self.model.eval()
            self._init_ok = True

        if not self._init_ok:
            raise RuntimeError("Failed to init QwenSampler.")

    def build_prompt(self, sayings: str, emotion: str, n: int = 3) -> str:
        # 你可以按你数据的“listener”风格更细化，比如加入动作库、约束（不能走动、不能大幅度等）
        emo = (emotion or "").strip()
        emoline = f"Speaker emotion: {emo}\n" if emo else ""
        return (
            "You are an expert animator for dyadic conversation.\n"
            "Given the speaker utterance (and optional emotion), propose natural LISTENER nonverbal reactions.\n"
            "Output STRICT JSON only. No markdown.\n\n"
            f"Speaker utterance: {sayings.strip()}\n"
            f"{emoline}"
            f"Return JSON with key `actions`, an array of exactly {n} concise action plans.\n"
            "Each action plan should be 1 sentence describing listener motion style (head, gaze, torso, hands).\n"
            "Do NOT mention camera or scene.\n"
            "Example:\n"
            "{\"actions\": [\"...\", \"...\", \"...\"]}\n"
        )

    def sample_actions(
        self,
        sayings: str,
        emotion: str,
        n: int = 3,
        temperature: float = 0.9,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
        seed: int = 0,
    ) -> List[str]:
        prompt = self.build_prompt(sayings, emotion, n=n)
        if self.backend == "vllm":
            from vllm import SamplingParams
            sp = SamplingParams(
                n=1,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
                seed=seed,
            )
            out = self.llm.generate([prompt], sp)[0].outputs[0].text
        else:
            # HF
            set_seed(seed)
            inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
            gen = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tok.eos_token_id,
            )
            out = self.tok.decode(gen[0], skip_special_tokens=True)
            # 有些模型会把 prompt 也一起 decode 出来：做个截断
            if out.startswith(prompt):
                out = out[len(prompt):].strip()

        # parse JSON robustly
        actions = self._parse_actions_json(out, n=n)
        return actions

    def _parse_actions_json(self, text: str, n: int) -> List[str]:
        s = text.strip()
        # 尝试提取第一个 { ... } 块
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            s = m.group(0).strip()
        try:
            obj = json.loads(s)
            acts = obj.get("actions", [])
            acts = [str(x).strip() for x in acts if str(x).strip()]
        except Exception:
            # fallback：用行切分
            acts = [ln.strip("-• \t\r") for ln in s.splitlines() if ln.strip()]
        # pad/trim to exactly n
        if len(acts) >= n:
            return acts[:n]
        while len(acts) < n:
            acts.append(acts[-1] if len(acts) else "listener nods gently and keeps attentive gaze")
        return acts

# -------------------------
# Text-to-Human adapter (你把内部换成你的模型)
# -------------------------
class Text2HumanGenerator:
    def __init__(self, ckpt: str, device: str = "cuda", fp16: bool = True):
        self.ckpt = ckpt
        self.device = device
        self.fp16 = fp16
        # TODO: load your T2M model here
        # e.g., self.model = ...
        # self.model.eval()

    @torch.no_grad()
    def generate(self, text: str, seed: int = 0, **kwargs) -> np.ndarray:
        """
        Return a motion representation.
        Replace with your actual inference code.

        For example, return joints positions [T, J, 3] or [T, D] or discrete tokens.
        """
        # TODO: call your model
        # motion = self.model.sample(text, seed=seed, ...)
        # return motion

        # placeholder (so script runs)
        rng = np.random.RandomState(seed)
        T, D = 196, 263
        return rng.randn(T, D).astype(np.float32)

def save_motion_npz(path: str, motion: np.ndarray, meta: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, motion=motion, meta=json.dumps(meta, ensure_ascii=False))

# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", type=str, required=True, help="dir(train.csv/val.csv/test.csv) or a csv with split col")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--key_by", type=str, default="group_id", choices=["group_id", "sayings_emotion", "sayings_only"])
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--qwen_path", type=str, default="/ibex/project/c2191/luoc/LLM_checkpoints/qwen3-30b-a3b-thinking-2507")
    ap.add_argument("--qwen_backend", type=str, default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size for vLLM")
    ap.add_argument("--qwen_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--qwen_max_len", type=int, default=4096)

    ap.add_argument("--n_samples", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=256)

    ap.add_argument("--t2m_ckpt", type=str, required=True, help="your text-to-human checkpoint")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_groups", type=int, default=0, help="0 means all")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    motions_root = pjoin(args.out_dir, "motions")
    meta_path = pjoin(args.out_dir, "meta.jsonl")

    print("[Load CSV] ...")
    df = read_split_csv(args.pairs_csv, args.split)
    need_cols = ["sayings", "emotion"]
    if args.key_by == "group_id":
        need_cols.append("group_id")
    for c in need_cols:
        if c not in df.columns:
            raise RuntimeError(f"Missing column `{c}` in csv. Found: {list(df.columns)}")

    df["sayings"] = df["sayings"].map(normalize_text)
    df["emotion"] = df["emotion"].astype(str)

    # group one row per group (你也可以做更复杂聚合)
    print("[Group] ...")
    # 用第一条记录代表该组的 sayings/emotion
    groups = []
    seen = set()
    for _, r in df.iterrows():
        k = group_key_from_row(r, args.key_by)
        if k in seen:
            continue
        seen.add(k)
        groups.append((k, str(r["sayings"]), str(r["emotion"])))

    if args.max_groups and args.max_groups > 0:
        groups = groups[: args.max_groups]
    print(f"[Groups] count={len(groups)}")

    print("[Init Qwen] ...")
    qwen = QwenSampler(
        model_path=args.qwen_path,
        backend=args.qwen_backend,
        tp=args.tp,
        dtype=args.qwen_dtype,
        max_model_len=args.qwen_max_len,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("[Init T2M] ...")
    t2m = Text2HumanGenerator(args.t2m_ckpt, device="cuda" if torch.cuda.is_available() else "cpu")

    print("[Generate] ...")
    with open(meta_path, "w", encoding="utf-8") as fw:
        for i, (key, sayings, emotion) in enumerate(groups):
            base_seed = args.seed + i * 1000
            actions = qwen.sample_actions(
                sayings=sayings,
                emotion=emotion,
                n=args.n_samples,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                seed=base_seed,
            )

            motion_paths = []
            for j, act in enumerate(actions):
                m_seed = base_seed + j
                motion = t2m.generate(act, seed=m_seed)
                out_path = pjoin(motions_root, str(key), f"m{j}.npz")
                meta = dict(
                    key=key,
                    sayings=sayings,
                    emotion=emotion,
                    action_text=act,
                    seed=m_seed,
                    split=args.split,
                    idx=i,
                    sample=j,
                )
                save_motion_npz(out_path, motion, meta)
                motion_paths.append(out_path)

            rec = dict(
                key=key,
                sayings=sayings,
                emotion=emotion,
                actions=actions,
                motions=motion_paths,
                split=args.split,
            )
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if (i + 1) % 20 == 0:
                print(f"  done {i+1}/{len(groups)}")

    print("[Done] meta:", meta_path)
    print("[Done] motions root:", motions_root)

if __name__ == "__main__":
    main()