#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_t2m_pipeline.py

Pipeline:
  - Read speaker sayings (+ optional emotion) from new_data/test.csv
  - Mode `t`   : use text only
  - Mode `t+e` : use text + emotion
  - Use Qwen (Qwen3-30B) to sample listener reaction motion captions
  - Feed each caption into external T2M-GPT to generate motion tokens
  - Save motion tokens under out_dir/tokens/<group_key>/m{idx}.npy and a meta.jsonl

参考:
  - eval/casualed_bsaeline.py (Qwen + Text2Human skeleton)
  - dataset/dataset_A2RM_gsn_flex.py (对 CSV / grouping 的约定)
"""

import os
import re
import json
import time
import random
import argparse
from os.path import join as pjoin
from typing import Any, Dict, List, Tuple

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
    """
    兼容:
      - 传目录: 包含 train.csv/val.csv/test.csv
      - 传单个 csv: 需要有 split 列
    """
    if os.path.isdir(pairs_csv_or_dir):
        path = pjoin(pairs_csv_or_dir, f"{split}.csv")
        if not os.path.isfile(path):
            raise RuntimeError(f"Missing split csv: {path}")
        return pd.read_csv(path, encoding="utf-8")

    df = pd.read_csv(pairs_csv_or_dir, encoding="utf-8")
    if "split" not in df.columns:
        raise RuntimeError(
            "pairs_csv is a file but has no `split` column. "
            "Provide dir with train/val/test, or add split column."
        )
    sp = df["split"].astype(str).str.lower().str.strip()
    return df[sp == split].copy()


def group_key_from_row(row: pd.Series, key_by: str) -> str:
    if key_by == "group_id":
        return str(row["group_id"])
    if key_by == "sayings_only":
        return str(row["sayings"])
    # sayings_emotion
    return f"{row['sayings']}|||{row['emotion']}"


# ─────────────────────────────────────────────────────────────────────────────
# Prompt constants  (shared with train/finetune_qwen_lora.py)
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are an expert animator for dyadic conversation.\n"
    "Given the speaker utterance (and optional emotion), propose natural "
    "LISTENER nonverbal reactions.\n"
    "Output STRICT JSON only. No markdown."
)


def _build_user_message(sayings: str, emotion: str, n: int, cond_mode: str) -> str:
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


# -------------------------
# Qwen wrapper (vLLM / HF)
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
        lora_path: str = "",    # path to LoRA adapter (optional)
    ):
        self.model_path = model_path
        self.backend = backend
        self.tp = int(tp)
        self.dtype = dtype
        self.max_model_len = int(max_model_len)
        self.device = device
        self.lora_path = lora_path

        self._init_ok = False
        self.llm = None
        self.tok = None
        self.model = None

        if backend == "vllm":
            try:
                from vllm import LLM

                lora_kwargs = {}
                if lora_path and os.path.isdir(lora_path):
                    lora_kwargs["enable_lora"] = True

                self.llm = LLM(
                    model=model_path,
                    tensor_parallel_size=self.tp,
                    dtype=self.dtype,
                    max_model_len=self.max_model_len,
                    trust_remote_code=True,
                    **lora_kwargs,
                )
                self._lora_request = None
                if lora_path and os.path.isdir(lora_path):
                    from vllm.lora.request import LoRARequest
                    self._lora_request = LoRARequest("motion_lora", 1, lora_path)

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
                device_map="auto",
            )
            # Load LoRA adapter if provided
            if lora_path and os.path.isdir(lora_path):
                from peft import PeftModel
                print(f"[INFO] Loading LoRA adapter from {lora_path}")
                self.model = PeftModel.from_pretrained(self.model, lora_path)
                self.model = self.model.merge_and_unload()
            self.model.eval()
            self._init_ok = True

        if not self._init_ok:
            raise RuntimeError("Failed to init QwenSampler.")

    def build_prompt(self, sayings: str, emotion: str, n: int = 3, cond_mode: str = "t+e") -> str:
        """
        Returns the plain-text version of the prompt (kept for reference).
        sample_actions() now uses the chat-template path instead.
        """
        sayings = sayings.strip()
        emo = (emotion or "").strip()
        use_emo = (cond_mode == "t+e") and bool(emo)
        emoline = f"Speaker emotion: {emo}\n" if use_emo else ""

        return (
            "You are an expert animator for dyadic conversation.\n"
            "Given the speaker utterance (and optional emotion), propose natural LISTENER nonverbal reactions.\n"
            "Output STRICT JSON only. No markdown.\n\n"
            f"Speaker utterance: {sayings}\n"
            f"{emoline}"
            f"Return JSON with key `actions`, an array of exactly {n} concise action plans.\n"
            "Each action plan should be 1 sentence describing listener motion style (head, gaze, torso, hands).\n"
            "Do NOT mention camera or scene.\n"
            "Example:\n"
            "{\"actions\": [\"...\", \"...\", \"...\"]}\n"
        )

    def _build_messages(self, sayings: str, emotion: str, n: int, cond_mode: str) -> List[Dict]:
        """Chat-template messages matching the SFT training format."""
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_message(sayings, emotion, n, cond_mode)},
        ]

    def sample_actions(
        self,
        sayings: str,
        emotion: str,
        n: int = 3,
        temperature: float = 0.9,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
        seed: int = 0,
        cond_mode: str = "t+e",
    ) -> List[str]:
        messages = self._build_messages(sayings, emotion, n=n, cond_mode=cond_mode)

        if self.backend == "vllm":
            from vllm import SamplingParams

            sp = SamplingParams(
                n=1,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
                seed=seed,
            )
            generate_kwargs: Dict[str, Any] = {}
            if self._lora_request is not None:
                generate_kwargs["lora_request"] = self._lora_request
            # chat_template_kwargs suppresses Qwen3 thinking tokens when supported
            try:
                out = self.llm.chat(
                    messages, sp,
                    chat_template_kwargs={"enable_thinking": False},
                    **generate_kwargs,
                )[0].outputs[0].text
            except TypeError:
                out = self.llm.chat(messages, sp, **generate_kwargs)[0].outputs[0].text
        else:
            # HF: apply chat template then generate
            set_seed(seed)
            # enable_thinking=False suppresses <think>...</think> for Qwen3 thinking models
            chat_kwargs: Dict[str, Any] = dict(
                tokenize=False,
                add_generation_prompt=True,
            )
            try:
                prompt_text = self.tok.apply_chat_template(
                    messages, enable_thinking=False, **chat_kwargs
                )
            except TypeError:
                prompt_text = self.tok.apply_chat_template(messages, **chat_kwargs)

            inputs = self.tok(prompt_text, return_tensors="pt").to(self.model.device)
            prompt_len = inputs["input_ids"].shape[1]
            gen = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tok.eos_token_id,
            )
            # Decode only the newly generated tokens
            out = self.tok.decode(
                gen[0][prompt_len:], skip_special_tokens=True
            ).strip()

        actions = self._parse_actions_json(out, n=n)
        return actions

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove Qwen3 <think>...</think> reasoning block if present."""
        s = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        return s.strip()

    def _parse_actions_json(self, text: str, n: int) -> List[str]:
        # Strip Qwen3 thinking block before parsing JSON
        s = self._strip_thinking(text).strip()

        # Extract the first {...} JSON object
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            s = m.group(0).strip()
        try:
            obj = json.loads(s)
            acts = obj.get("actions", [])
            acts = [str(x).strip() for x in acts if str(x).strip()]
        except Exception:
            acts = [ln.strip("-• \t\r") for ln in s.splitlines() if ln.strip()]

        if len(acts) >= n:
            return acts[:n]
        while len(acts) < n:
            acts.append(acts[-1] if len(acts) else "listener nods gently and keeps attentive gaze")
        return acts


# -------------------------
# T2M-GPT adapter
# -------------------------
class T2MGPTGenerator:
    """
    Wrap external/T2M-GPT to generate motion tokens from a text caption.
    Uses CLIP (ViT-B/32) as text encoder + Text2Motion_Transformer for autoregressive decoding.

    Required checkpoints:
      vqvae_ckpt  : HumanVQVAE checkpoint  (e.g. checkpoints/t2m/VQVAEV3_CB1024_.../net_last.pth)
      trans_ckpt  : GPT transformer ckpt   (e.g. checkpoints/t2m/T2M_Seq2Seq_.../net_best_fid.pth)
    """

    def __init__(
        self,
        t2m_root: str,
        ckpt: str,                    # transformer checkpoint path (--t2m_ckpt)
        device: str = "cuda",
        fp16: bool = False,
        # VQ-VAE config (must match the VQ-VAE checkpoint)
        vqvae_ckpt: str = "",
        nb_code: int = 512,
        code_dim: int = 512,
        output_emb_width: int = 512,
        down_t: int = 2,
        stride_t: int = 2,
        width: int = 512,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        vq_act: str = "relu",
        quantizer: str = "ema_reset",
        # GPT config
        block_size: int = 25,
        embed_dim_gpt: int = 512,
        clip_dim: int = 512,
        num_layers: int = 2,
        n_head_gpt: int = 8,
        ff_rate: int = 4,
        drop_out_rate: float = 0.1,
        if_categorial: bool = True,   # True = sample, False = greedy (argmax)
    ):
        import sys
        sys.path.insert(0, t2m_root)

        try:
            import clip as clip_lib
        except ImportError:
            raise ImportError("pip install git+https://github.com/openai/CLIP.git")

        import models.vqvae as vqvae_mod
        import models.t2m_trans as trans_mod

        self.device = torch.device(device)
        self.nb_code = nb_code
        self.if_categorial = if_categorial

        # ── CLIP ──────────────────────────────────────────────────
        print("[T2M] loading CLIP ViT-B/32 ...")
        self.clip_model, _ = clip_lib.load("ViT-B/32", device=self.device, jit=False)
        clip_lib.model.convert_weights(self.clip_model)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_tokenize = clip_lib.tokenize

        # ── VQ-VAE (for decode only; same config as training) ──────
        class _Args:
            dataname = "t2m"
            beta = 1.0
            mu = 0.99

        _a = _Args()
        _a.quantizer = quantizer
        if vqvae_ckpt and os.path.isfile(vqvae_ckpt):
            # Auto-detect nb_code from the codebook weight shape
            _vq_sd = torch.load(vqvae_ckpt, map_location="cpu")
            _vq_net = _vq_sd.get("net", _vq_sd)
            # codebook key is typically "vqvae.quantizer.codebook" or similar
            _cb_key = next(
                (k for k in _vq_net if "codebook" in k and k.endswith(".weight")), None
            )
            if _cb_key is None:
                _cb_key = next((k for k in _vq_net if "codebook" in k), None)
            if _cb_key is not None:
                nb_code = int(_vq_net[_cb_key].shape[0])
                code_dim = int(_vq_net[_cb_key].shape[1])
                print(f"[T2M] VQ-VAE detected: nb_code={nb_code}, code_dim={code_dim}")

        self.vqvae = vqvae_mod.HumanVQVAE(
            _a, nb_code=nb_code, code_dim=code_dim,
            output_emb_width=output_emb_width,
            down_t=down_t, stride_t=stride_t,
            width=width, depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation=vq_act, norm=None,
        )
        if vqvae_ckpt and os.path.isfile(vqvae_ckpt):
            print(f"[T2M] loading VQ-VAE from {vqvae_ckpt}")
            self.vqvae.load_state_dict(_vq_net, strict=True)
        self.vqvae = self.vqvae.to(self.device).eval()

        # ── GPT transformer: auto-detect arch from checkpoint ─────
        if not ckpt or not os.path.isfile(ckpt):
            raise RuntimeError(f"[T2M] GPT checkpoint not found: {ckpt!r}")

        print(f"[T2M] loading GPT from {ckpt}")
        trans_raw = torch.load(ckpt, map_location="cpu")
        sd = trans_raw.get("trans", trans_raw)

        # Infer hyperparameters from state dict shapes so we never mismatch
        # tok_emb.weight: [num_vq+2, embed_dim]
        tok_w = sd["trans_base.tok_emb.weight"]
        _num_vq   = int(tok_w.shape[0]) - 2          # +2 special tokens
        _embed    = int(tok_w.shape[1])
        # pos_embedding.weight: [block_size, embed_dim]
        pos_w     = sd["trans_base.pos_embedding.weight"]
        _bsz      = int(pos_w.shape[0])
        # count layers by scanning block keys
        _nlayers  = 0
        while f"trans_base.blocks.{_nlayers}.ln1.weight" in sd:
            _nlayers += 1
        # n_head: not directly readable; derive from attn weight shape
        # key.weight: [embed_dim, embed_dim] — we just keep the user default
        # (all standard T2M-GPT configs use n_head divisible by embed_dim/64)
        _n_head   = max(1, _embed // 64)   # 512→8, 1024→16

        print(f"[T2M] detected arch: num_vq={_num_vq}, embed={_embed}, "
              f"block_size={_bsz}, num_layers={_nlayers}, n_head={_n_head}")

        self.nb_code = _num_vq   # override for generate()
        self.trans = trans_mod.Text2Motion_Transformer(
            num_vq=_num_vq,
            embed_dim=_embed,
            clip_dim=clip_dim,
            block_size=_bsz,
            num_layers=_nlayers,
            n_head=_n_head,
            drop_out_rate=drop_out_rate,
            fc_rate=ff_rate,
        )
        self.trans.load_state_dict(sd, strict=True)
        self.trans = self.trans.to(self.device).eval()

        print("[T2M] T2MGPTGenerator ready.")

    @torch.no_grad()
    def generate(self, text: str, seed: int = 0, **kwargs) -> np.ndarray:
        """
        Encode text with CLIP, autoregressively sample motion tokens.
        Returns shape [T] int32  (VQ-code indices).
        """
        torch.manual_seed(seed)
        # CLIP text encoding
        tokens = self.clip_tokenize([text], truncate=True).to(self.device)  # [1, 77]
        clip_feat = self.clip_model.encode_text(tokens).float()              # [1, 512]

        # Autoregressive sampling
        gen = self.trans.sample(clip_feat, if_categorial=self.if_categorial)  # [1, T]
        motion_tokens = gen[0].cpu().numpy().astype(np.int32)                 # [T]
        return motion_tokens


def save_motion_tokens(path: str, tokens: np.ndarray, meta: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, tokens)
    meta_path = path + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)


def _run_stage_qwen(args) -> str:
    """
    Stage 1: 只用 Qwen 生成 captions, 不跑 T2M-GPT.
    输出:
      out_dir/qwen_meta.jsonl  每行: {key, sayings, emotion, cond_mode, captions, split}
    """
    os.makedirs(args.out_dir, exist_ok=True)
    qwen_meta_path = pjoin(args.out_dir, "qwen_meta.jsonl")

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

    print("[Group] ...")
    groups: List[Tuple[str, str, str]] = []
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
        lora_path=getattr(args, "lora_path", ""),
    )

    print("[Stage=QWEN] Generate captions ...")
    with open(qwen_meta_path, "w", encoding="utf-8") as fw:
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
                cond_mode=args.cond_mode,
            )
            rec = dict(
                key=key,
                sayings=sayings,
                emotion=emotion,
                cond_mode=args.cond_mode,
                captions=actions,
                split=args.split,
            )
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if (i + 1) % 20 == 0:
                print(f"  [Qwen] done {i+1}/{len(groups)}")

    print("[Stage=QWEN] Done. meta:", qwen_meta_path)
    return qwen_meta_path


def _run_stage_t2m(args, qwen_meta_path: str):
    """
    Stage 2: 只用 T2M-GPT 消费 qwen_meta.jsonl, 逐条生成 motion tokens.
    随时被 kill 也不会丢已完成样本.
    """
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_root = pjoin(args.out_dir, "tokens")
    meta_path = pjoin(args.out_dir, "meta.jsonl")

    print("[Init T2M-GPT] ...")
    t2m = T2MGPTGenerator(
        t2m_root=args.t2m_root,
        ckpt=args.t2m_ckpt,
        device="cuda" if torch.cuda.is_available() else "cpu",
        vqvae_ckpt=args.t2m_vqvae_ckpt,
        nb_code=args.t2m_nb_code,
        code_dim=args.t2m_code_dim,
        output_emb_width=args.t2m_output_emb_width,
        down_t=args.t2m_down_t,
        stride_t=args.t2m_stride_t,
        width=args.t2m_width,
        depth=args.t2m_depth,
        dilation_growth_rate=args.t2m_dilation_growth_rate,
        block_size=args.t2m_block_size,
        embed_dim_gpt=args.t2m_embed_dim_gpt,
        num_layers=args.t2m_num_layers,
        n_head_gpt=args.t2m_n_head_gpt,
        if_categorial=args.t2m_if_categorial,
    )

    print("[Stage=T2M] Generate motion tokens from captions ...")
    # 逐行读, 每个 group/prompt 生成完立刻落盘
    with open(qwen_meta_path, "r", encoding="utf-8") as fr, open(meta_path, "w", encoding="utf-8") as fw:
        for i, ln in enumerate(fr):
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            key = str(obj["key"])
            sayings = str(obj.get("sayings", ""))
            emotion = str(obj.get("emotion", ""))
            cond_mode = str(obj.get("cond_mode", args.cond_mode))
            captions: List[str] = [str(x) for x in obj.get("captions", [])]

            base_seed = args.seed + i * 1000
            token_paths: List[str] = []
            for j, act in enumerate(captions):
                m_seed = base_seed + j
                tokens = t2m.generate(act, seed=m_seed)
                out_path = pjoin(tokens_root, str(key), f"m{j}.npy")
                meta = dict(
                    key=key,
                    sayings=sayings,
                    emotion=emotion,
                    cond_mode=cond_mode,
                    caption=act,
                    seed=m_seed,
                    split=obj.get("split", args.split),
                    idx=i,
                    sample=j,
                )
                save_motion_tokens(out_path, tokens, meta)
                token_paths.append(out_path)

            rec = dict(
                key=key,
                sayings=sayings,
                emotion=emotion,
                cond_mode=cond_mode,
                captions=captions,
                tokens=token_paths,
                split=obj.get("split", args.split),
            )
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if (i + 1) % 20 == 0:
                print(f"  [T2M] done {i+1}")

    print("[Stage=T2M] Done. meta:", meta_path)
    print("[Stage=T2M] tokens root:", tokens_root)


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pairs_csv",
        type=str,
        default="./new_data",
        help="dir(train.csv/val.csv/test.csv) or a csv with split column",
    )
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--key_by", type=str, default="group_id", choices=["group_id", "sayings_emotion", "sayings_only"])
    ap.add_argument("--out_dir", type=str, required=True, help="输出根目录, 会创建 tokens/ / meta.jsonl / qwen_meta.jsonl")

    # stage: qwen / t2m / both
    ap.add_argument(
        "--stage",
        type=str,
        default="both",
        choices=["qwen", "t2m", "both"],
        help="qwen: 只生成 captions; t2m: 只消费已有 captions; both: 一条龙",
    )
    ap.add_argument(
        "--qwen_meta",
        type=str,
        default="",
        help="当 stage=t2m 时, 指定已有的 qwen_meta.jsonl; 为空则默认用 out_dir/qwen_meta.jsonl",
    )

    # cond modes: t / t+e
    ap.add_argument("--cond_mode", type=str, default="t+e", choices=["t", "t+e"], help="t: only text; t+e: text+emotion")

    # Qwen
    ap.add_argument(
        "--qwen_path",
        type=str,
        default="/ibex/project/c2191/luoc/LLM_checkpoints/qwen3-30b-a3b-thinking-2507",
    )
    ap.add_argument(
        "--lora_path", type=str, default="",
        help="Path to LoRA adapter dir (checkpoints/qwen_lora/final). Empty = base model only.",
    )
    ap.add_argument("--qwen_backend", type=str, default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size for vLLM")
    ap.add_argument("--qwen_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--qwen_max_len", type=int, default=4096)

    ap.add_argument("--n_samples", type=int, default=3, help="每个 group 采样多少条反应 caption")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=256)

    # T2M-GPT
    ap.add_argument(
        "--t2m_root",
        type=str,
        default="external/T2M-GPT",
        help="T2M-GPT 源码根目录 (用于你在 T2MGPTGenerator 里 import)",
    )
    ap.add_argument("--t2m_ckpt", type=str, default="",
                    help="T2M-GPT GPT transformer checkpoint (stage=t2m/both 需要)")
    ap.add_argument("--t2m_vqvae_ckpt", type=str, default="",
                    help="T2M VQ-VAE checkpoint (HumanVQVAE), 用于 T2MGPTGenerator 内部解码校验")
    # VQ-VAE / GPT architecture (must match checkpoints)
    ap.add_argument("--t2m_nb_code", type=int, default=512)
    ap.add_argument("--t2m_code_dim", type=int, default=512)
    ap.add_argument("--t2m_output_emb_width", type=int, default=512)
    ap.add_argument("--t2m_down_t", type=int, default=2)
    ap.add_argument("--t2m_stride_t", type=int, default=2)
    ap.add_argument("--t2m_width", type=int, default=512)
    ap.add_argument("--t2m_depth", type=int, default=3)
    ap.add_argument("--t2m_dilation_growth_rate", type=int, default=3)
    ap.add_argument("--t2m_block_size", type=int, default=25,
                    help="GPT block size (max generated token length)")
    ap.add_argument("--t2m_embed_dim_gpt", type=int, default=512)
    ap.add_argument("--t2m_num_layers", type=int, default=2)
    ap.add_argument("--t2m_n_head_gpt", type=int, default=8)
    ap.add_argument("--t2m_if_categorial", action="store_true", default=True,
                    help="sample (True) vs greedy (False) decoding")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_groups", type=int, default=0, help="0 表示全部 group (仅 stage 包含 qwen 时生效)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Stage 1: Qwen
    qwen_meta_path = ""
    if args.stage in ["qwen", "both"]:
        qwen_meta_path = _run_stage_qwen(args)

    # Stage 2: T2M
    if args.stage in ["t2m", "both"]:
        if not qwen_meta_path:
            qwen_meta_path = args.qwen_meta or pjoin(args.out_dir, "qwen_meta.jsonl")
        if not qwen_meta_path or not os.path.isfile(qwen_meta_path):
            raise RuntimeError(f"Stage=T2M 需要 qwen_meta.jsonl, 找不到: {qwen_meta_path}")
        if not args.t2m_ckpt:
            raise RuntimeError("Stage=T2M/BOTH 需要指定 --t2m_ckpt")
        _run_stage_t2m(args, qwen_meta_path)


if __name__ == "__main__":
    main()

