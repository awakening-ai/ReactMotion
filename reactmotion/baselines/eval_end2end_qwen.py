#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import math
import glob
import argparse
import random
import csv
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
)

PROMPT_V2 = (
    "You are modeling a speaker-listener dyadic interaction.\n\n"
    "Input:\n"
    "- SPEAKER_TRANSCRIPTION: <Transcription_Placeholder>\n"
    "- LISTENER_EMOTION (optional): <Emotion_Placeholder>\n\n"
    "Output:\n"
    "Return ONLY a sequence of listener motion tokens in the exact format:\n"
    "<Motion Tokens> <Motion Token i> ... </Motion Tokens>\n"
    "Do NOT output any other words.\n"
)

def format_emotion(emotion: str) -> str:
    emotion = (emotion or "").strip()
    if not emotion:
        return ""
    return f"<Emotion> {emotion} </Emotion>"

def build_source_text(transcription: str, emotion: str) -> str:
    trans = (transcription or "").strip()
    emo = format_emotion(emotion)
    return (PROMPT_V2
            .replace("<Transcription_Placeholder>", trans)
            .replace("<Emotion_Placeholder>", emo)
            .strip())


_MOTION_SPAN_RE = re.compile(r"<Motion Tokens>(.*?)</Motion Tokens>", re.DOTALL)
_MOTION_TOKEN_RE = re.compile(r"<Motion Token\s+(\d+)>")
_MOTION_TOKEN_SHORT_RE = re.compile(r"<(\d+)>")  # ✅ 只抓 <123> 这种，不是裸数字

def parse_motion_tokens_v2(text: str, max_len: int = 200, codebook_size: int = 512) -> List[int]:
    """
    结构化解析（不使用裸数字 re.findall）：
    1) 先找 <Motion Tokens>...</Motion Tokens> span（如果存在）
    2) 在 span 内优先解析 <Motion Token i>
    3) 若为空，再解析 <i>（短格式）
    4) 过滤 0..511，截断 max_len
    """
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

# ----------------------------
# Utils: split by sayings (avoid leakage)
# ----------------------------
def split_by_sayings(df: pd.DataFrame, seed: int = 42,
                     ratios=(0.9, 0.05, 0.05)) -> pd.DataFrame:
    """
    如果 df 没有 split 列：按 sayings 做 group-level 随机划分 train/val/test
    """
    if "split" in df.columns:
        return df

    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = random.Random(seed)

    sayings_list = df["sayings"].astype(str).fillna("").tolist()
    uniq = list(dict.fromkeys(sayings_list))
    rng.shuffle(uniq)

    n = len(uniq)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_set = set(uniq[:n_train])
    val_set = set(uniq[n_train:n_train + n_val])
    test_set = set(uniq[n_train + n_val:])

    def _assign(s: str) -> str:
        if s in train_set:
            return "train"
        if s in val_set:
            return "val"
        return "test"

    df = df.copy()
    df["split"] = df["sayings"].astype(str).map(_assign)
    return df


# ----------------------------
# Parse motion tokens robustly
# ----------------------------
def parse_motion_tokens(text: str, max_len: Optional[int] = None, codebook_size: int = 512) -> List[int]:
    """
    只解析 <Motion Token i>，优先取 <Motion Tokens>...</Motion Tokens> 内的内容。
    """
    m = _MOTION_SPAN_RE.search(text)
    span = m.group(1) if m else text

    codes = [int(x) for x in _MOTION_TOKEN_RE.findall(span)]
    # 过滤非法
    codes = [c for c in codes if 0 <= c < codebook_size]

    if max_len is not None:
        codes = codes[:max_len]
    return codes


def parse_motion_tokens_any(text: str, max_len: int = 200, codebook_size: int = 512) -> List[int]:
    """
    优先解析 <Motion Token i>；否则 fallback 到提取所有数字
    """
    if text is None:
        return []
    text = str(text)

    m = _MOTION_SPAN_RE.search(text)
    span = m.group(1) if m else text

    codes = [int(x) for x in _MOTION_TOKEN_RE.findall(span)]
    if len(codes) == 0:
        # fallback: digits
        nums = re.findall(r"\d+", text)
        codes = [int(x) for x in nums]

    # clip + filter
    out = []
    for c in codes:
        if 0 <= c < codebook_size:
            out.append(c)
        else:
            # 一旦出现 >511，常见是模型跑飞/混入其它数字，直接截断更稳
            break

    return out[:max_len]


# ----------------------------
# Token histogram embedding + cosine winrate
# ----------------------------
def token_hist_emb(codes: List[int], codebook_size: int = 512) -> np.ndarray:
    h = np.zeros((codebook_size,), dtype=np.float32)
    if not codes:
        return h
    for c in codes:
        if 0 <= c < codebook_size:
            h[c] += 1.0
    # L2 normalize
    n = np.linalg.norm(h) + 1e-8
    return h / n

def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def gen_vs_neg_winrate(gen_emb: np.ndarray, pos_embs: List[np.ndarray], neg_embs: List[np.ndarray]) -> float:
    """
    win = sim(gen, best_pos) > sim(gen, neg)
    """
    if len(pos_embs) == 0 or len(neg_embs) == 0:
        return float("nan")
    sims_pos = [cosine_np(gen_emb, p) for p in pos_embs]
    best_pos = max(sims_pos)
    wins = 0
    for n in neg_embs:
        if best_pos > cosine_np(gen_emb, n):
            wins += 1
    return wins / max(1, len(neg_embs))


# ----------------------------
# Find audio code file for a group
# ----------------------------
def find_audio_code_file(audio_dir: str, raw_file_name: str) -> Optional[str]:
    """
    raw_file_name 形如 000267_1
    优先匹配 audio_dir/raw_file_name(.npz/.npy)，否则用 glob 找前缀
    """
    stem = str(raw_file_name).strip()
    if not stem:
        return None
    for ext in [".npz", ".npy"]:
        p = os.path.join(audio_dir, stem + ext)
        if os.path.isfile(p):
            return p

    # fallback: any file startswith stem
    cand = glob.glob(os.path.join(audio_dir, stem + ".*"))
    if cand:
        return cand[0]

    # 再 fallback：如果 audio 文件名是 motionid_xxx 形式，你也可以按 stem 前缀搜
    cand = glob.glob(os.path.join(audio_dir, stem + "*"))
    if cand:
        return cand[0]
    return None

def load_audio_tokens(audio_code_path: str) -> np.ndarray:
    """
    期望 npz 里有 'codes'，否则 npy 直接是 codes
    return: codes ndarray
    """
    if audio_code_path.endswith(".npz"):
        d = np.load(audio_code_path)
        if "codes" not in d:
            raise KeyError(f"Missing 'codes' in {audio_code_path}, keys={list(d.keys())}")
        return d["codes"]
    else:
        return np.load(audio_code_path)


# ----------------------------
# HumanML3D VQVAE token lookup
# ----------------------------
def build_vq_index(vq_dir: str) -> Dict[str, str]:
    m = {}
    for fn in os.listdir(vq_dir):
        if fn.endswith(".npy"):
            stem = os.path.splitext(fn)[0]
            m[stem] = os.path.join(vq_dir, fn)
    return m

def vqvae_lookup(vq_by_stem: Dict[str, str], motion_id: str) -> Optional[str]:
    base = str(motion_id)
    if base in vq_by_stem:
        return vq_by_stem[base]
    if base.isdigit() and ("M" + base) in vq_by_stem:
        return vq_by_stem["M" + base]
    if base.startswith("M") and base[1:].isdigit() and (base[1:] in vq_by_stem):
        return vq_by_stem[base[1:]]
    return None

def motion_id_from_raw(raw_file_name: str) -> str:
    s = str(raw_file_name)
    mid = s.split("_", 1)[0]
    return str(mid).zfill(6)

def load_motion_codes_from_vq(vq_path: str) -> List[int]:
    arr = np.load(vq_path)
    arr = np.asarray(arr).reshape(-1).tolist()
    # 有些 VQ 文件可能含 padding/特殊值，这里只做基本过滤
    out = []
    for x in arr:
        try:
            c = int(x)
        except Exception:
            continue
        if 0 <= c < 512:
            out.append(c)
    return out


# ----------------------------
# Build A2RM input prompt (audio tokens -> text)
# ----------------------------
def build_a2rm_prompt(a_codes: np.ndarray, level_audio: int = 0) -> str:
    gen_rm_instruction = "Generate reactive motion for the audio: <Audio_Tokens_Placeholder>"
    audio_text = "<Audio Tokens>"
    if level_audio == 0:
        a0 = a_codes[0]
        for t in a0.reshape(-1):
            audio_text += f"<Audio Level 0 Token {int(t)}>"
    else:
        # expect [K,T]
        for i in range(a_codes.shape[0]):
            for j in range(a_codes.shape[1]):
                audio_text += f"<Audio Level {i} Token {int(a_codes[i,j])}>"
    audio_text += "</Audio Tokens>"
    return gen_rm_instruction.replace("<Audio_Tokens_Placeholder>", audio_text)


# ----------------------------
# Caption prompt (motion tokens -> caption)
# ----------------------------
def build_caption_prompt(motion_codes: List[int]) -> str:
    cap_instruction = "Generate text: "
    motion_string = "<Motion Tokens>" + "".join([f"<Motion Token {c}>" for c in motion_codes]) + "</Motion Tokens>"
    return cap_instruction + motion_string


def build_uniform_candidates(
    gen_items: List[Dict[str, Any]],
    pos_items: List[Dict[str, Any]],
    neg_items: List[Dict[str, Any]],
    seed: int,
    max_total: int = 24,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    把 gen/pos/neg 打散混合 -> 统一命名成 C01..Cxx
    返回：
      - candidates_for_llm: [{"cid":"C01","caption":"..."}...]
      - cid2orig: {"C01":"gen_0", "C02":"pos_3", ...}
    """
    all_items = []
    for it in gen_items:
        all_items.append({"orig_id": str(it["id"]), "caption": str(it["caption"])})
    for it in pos_items:
        all_items.append({"orig_id": str(it["id"]), "caption": str(it["caption"])})
    for it in neg_items:
        all_items.append({"orig_id": str(it["id"]), "caption": str(it["caption"])})

    # 打乱，避免固定顺序泄露类别
    rng = random.Random(int(seed))
    rng.shuffle(all_items)

    # 截断总数，控制 prompt 长度
    all_items = all_items[:max_total]

    cid2orig = {}
    candidates = []
    for i, it in enumerate(all_items, start=1):
        cid = f"C{i:02d}"
        cid2orig[cid] = it["orig_id"]
        candidates.append({"cid": cid, "caption": it["caption"]})
    return candidates, cid2orig


def normalize_topk_ids(topk: Any) -> List[str]:
    """
    LLM 输出可能是 ["C01","C02"] 或 "C01, C02" 或混杂其它文本
    这里统一提取 C\d\d
    """
    if topk is None:
        return []
    if isinstance(topk, list):
        s = " ".join([str(x) for x in topk])
    else:
        s = str(topk)
    return re.findall(r"\bC\d{2}\b", s)


# ----------------------------
# Qwen judge: choose top-3 among (gen + pos + neg)
# ----------------------------
def build_qwen_rank_prompt_uniform(
    sayings: str,
    emotion: str,
    candidates: List[Dict[str, Any]],  # [{"cid","caption"}]
    top_k: int = 3,
) -> str:
    payload = {
        "task": "Rank motions by how well they match a dyadic conversation query (speaker sayings + emotion).",
        "query": {"sayings": sayings, "emotion": emotion},
        "candidates": candidates,
        "instructions": [
            "Each candidate has a neutral id like C01, C02, ... The id contains NO label information.",
            "Select the best TOP-3 candidates that are most plausible LISTENER reactions to the query.",
            "Use ONLY the candidate captions as evidence; do not assume extra info.",
            "Return STRICT JSON only, with field top3 as a list of candidate ids.",
            "Do NOT output any text outside JSON."
        ],
        "output_schema": {"top3": ["C01", "C02", "C03"], "reason": "short explanation"},
        "top_k": top_k,
    }

    return (
        "You are an expert evaluator for listener reactive motions.\n"
        "Return STRICT JSON only (no markdown, no extra text).\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n"
    )


def extract_json_strict(text: str) -> Optional[Dict[str, Any]]:
    if text is None:
        return None
    s = text.strip()
    # 尝试直接解析
    try:
        return json.loads(s)
    except Exception:
        pass
    # 尝试截取第一个 {...} 块
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# ----------------------------
# Qwen runner (vLLM optional, fallback transformers)
# ----------------------------
class QwenJudge:
    def __init__(self, model_path: str, use_vllm: bool = True,
                 tp: int = 1, gpu_mem_util: float = 0.90,
                 max_new_tokens: int = 512, temperature: float = 0.0):
        self.model_path = model_path
        self.use_vllm = use_vllm
        self.tp = tp
        self.gpu_mem_util = gpu_mem_util
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._mode = None

        if use_vllm:
            try:
                from vllm import LLM, SamplingParams  # type: ignore
                self._vllm_LLM = LLM(
                    model=model_path,
                    tensor_parallel_size=tp,
                    gpu_memory_utilization=gpu_mem_util,
                    trust_remote_code=True,
                )
                self._vllm_SamplingParams = SamplingParams
                self._mode = "vllm"
            except Exception as e:
                print(f"[WARN] vLLM not available or failed to init, fallback to transformers. err={e}")
                self._mode = "hf"
        else:
            self._mode = "hf"

        if self._mode == "hf":
            self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            ).eval()

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        if self._mode == "vllm":
            sp = self._vllm_SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                top_p=1.0,
            )
            outs = self._vllm_LLM.generate([prompt], sp)
            return outs[0].outputs[0].text

        # hf
        inputs = self.tok(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature > 0),
            temperature=max(1e-6, self.temperature),
            top_p=1.0,
        )
        txt = self.tok.decode(out[0], skip_special_tokens=True)
        # 有时会把 prompt 也 decode 出来，简单去掉前缀
        if txt.startswith(prompt):
            txt = txt[len(prompt):]
        return txt.strip()


def query_key(sayings: str, emotion: str) -> str:
    s = (sayings or "") + "|||" + (emotion or "")
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def ensure_parent(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

def append_row(writer, f, row: dict):
    writer.writerow(row)
    f.flush()
    os.fsync(f.fileno())

def load_done_keys_from_csv(path: str, key_col: str = "query_key") -> set:
    if not os.path.isfile(path):
        return set()
    done = set()
    try:
        with open(path, "r", encoding="utf-8") as rf:
            r = csv.DictReader(rf)
            for row in r:
                k = row.get(key_col, "")
                if k:
                    done.add(k)
    except Exception as e:
        print(f"[WARN] failed to read done keys from {path}: {e}")
    return done


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--pairs_csv", type=str, required=True, help="pairs_annotated.csv (must contain sayings, emotion, label, raw_file_name, motion_caption)")
    ap.add_argument("--dataset_dir", type=str, required=True, help="A2R dataset root, containing HumanML3D/VQVAE etc.")
    ap.add_argument("--audio_dir", type=str, required=True, help="audio code dir containing *.npz with key 'codes'")
    ap.add_argument("--split_seed", type=int, default=42)

    ap.add_argument("--a2rm_ckpt", type=str, required=True, help="A2RM T5 checkpoint dir")
    ap.add_argument("--caption_ckpt", type=str, required=True, help="motion2text T5 checkpoint dir")

    ap.add_argument("--level_audio", type=int, default=0)
    ap.add_argument("--num_candidates", type=int, default=8)
    ap.add_argument("--gen_max_len", type=int, default=200)
    ap.add_argument("--gen_temperature", type=float, default=0.8)
    ap.add_argument("--gen_top_k", type=int, default=200)

    ap.add_argument("--max_pos_eval", type=int, default=32)
    ap.add_argument("--max_neg_eval", type=int, default=64)

    ap.add_argument("--out_dir", type=str, default="./eval_out")
    ap.add_argument("--only_test", action="store_true", help="evaluate only split=test (recommended)")

    # Qwen
    ap.add_argument("--qwen_path", type=str, default="/ibex/project/c2191/luoc/LLM_checkpoints/qwen3-30b-a3b-thinking-2507")
    ap.add_argument("--use_qwen", action="store_true")
    ap.add_argument("--qwen_use_vllm", action="store_true")
    ap.add_argument("--qwen_tp", type=int, default=1)
    ap.add_argument("--qwen_gpu_mem_util", type=float, default=0.90)
    ap.add_argument("--qwen_max_new_tokens", type=int, default=512)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "tokens"), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Device]", device)

    # load pairs
    df = pd.read_csv(args.pairs_csv, encoding="utf-8")
    for col in ["sayings", "emotion", "label", "raw_file_name"]:
        if col not in df.columns:
            raise RuntimeError(f"Missing column `{col}` in {args.pairs_csv}")

    # ensure split (by sayings)
    df = pd.read_csv(args.pairs_csv, encoding="utf-8")
    required = ["sayings", "emotion", "label", "raw_file_name", "split"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {args.pairs_csv}: {missing}")


    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["sayings"] = df["sayings"].astype(str).fillna("")
    df["emotion"] = df["emotion"].astype(str).fillna("")
    df["raw_file_name"] = df["raw_file_name"].astype(str).fillna("")

    df_eval = df[df["split"] == "test"].copy()

    print("[Pairs] total=", len(df), " eval=", len(df_eval), " splits=", df["split"].value_counts().to_dict())

    # vq index
    motion_vq_dir = os.path.join(args.dataset_dir, "HumanML3D", "VQVAE")
    if not os.path.isdir(motion_vq_dir):
        raise RuntimeError(f"Missing motion_vq_dir: {motion_vq_dir}")
    vq_by_stem = build_vq_index(motion_vq_dir)
    print("[VQ] indexed:", len(vq_by_stem))

    # load A2RM + caption models
    a2rm_tok = T5Tokenizer.from_pretrained(args.a2rm_ckpt)
    a2rm_model = T5ForConditionalGeneration.from_pretrained(args.a2rm_ckpt).to(device).eval()

    cap_tok = T5Tokenizer.from_pretrained(args.caption_ckpt)
    cap_model = T5ForConditionalGeneration.from_pretrained(args.caption_ckpt).to(device).eval()

    # Qwen judge optional
    qwen = None
    if args.use_qwen:
        qwen = QwenJudge(
            model_path=args.qwen_path,
            use_vllm=args.qwen_use_vllm,
            tp=args.qwen_tp,
            gpu_mem_util=args.qwen_gpu_mem_util,
            max_new_tokens=args.qwen_max_new_tokens,
            temperature=0.0,
        )
        print("[Qwen] enabled:", args.qwen_path, "mode=", ("vllm" if args.qwen_use_vllm else "hf"))

    # group by query = (sayings, emotion)
    groups = list(df_eval.groupby(["sayings", "emotion"], dropna=False))
    print("[Groups]", len(groups))

    cand_rows = []
    qwen_rows = []


    cand_csv = os.path.join(args.out_dir, "candidates.csv")
    qwen_csv = os.path.join(args.out_dir, "qwen_top3.csv")  # 旧：保持原样（可选）
    qwen_dbg_csv = os.path.join(args.out_dir, "qwen_top3_debug.csv")  # 新：专门存调试信息

    ensure_parent(cand_csv)
    ensure_parent(qwen_csv)
    ensure_parent(qwen_dbg_csv)

    # ✅ 恢复：如果 qwen_top3.csv 已经有这个 query_key，就认为该 group 完整评测过（含 Qwen），直接跳过
    done_keys = load_done_keys_from_csv(qwen_csv, key_col="query_key") if args.use_qwen else set()
    print(f"[Resume] done groups from {qwen_csv}: {len(done_keys)}")

    # candidates.csv 的字段
    cand_fieldnames = [
        "query_key","split","sayings","emotion","audio_code_path",
        "cand_id","cand_tokens_path","cand_caption","cand_token_len",
        "tokenhist_winrate_vs_neg",
    ]

    # qwen_top3.csv 的字段
    qwen_fieldnames = [
        "query_key","split","sayings","emotion","audio_code_path",
        "top3","gen_in_top3","reason","raw_qwen_text",
    ]
    # qwen_top3_debug.csv 的字段（更全，便于调试）
    qwen_dbg_fieldnames = [
        "query_key", "split", "sayings", "emotion", "audio_code_path",

        # Qwen 输出
        "top3_cids",  # ["C07","C09","C15"]
        "top3_orig",  # ["pos_1","pos_4","gen_2"]
        "top3_captions",  # 对应 caption 列表
        "gen_in_top3",
        "reason",
        "raw_qwen_text",

        # Qwen 实际看到的 candidates（完整列表）
        "candidates_json",  # list[ {cid, orig_id, type, caption} ]

        # 另外把三类 caption 也单独存一份（更好 grep）
        "gen_caps_json",
        "pos_caps_json",
        "neg_caps_json",
    ]

    # 打开文件（append）
    cand_exists = os.path.isfile(cand_csv)
    qwen_exists = os.path.isfile(qwen_csv)

    cand_f = open(cand_csv, "a", encoding="utf-8", newline="")
    cand_w = csv.DictWriter(cand_f, fieldnames=cand_fieldnames)
    if not cand_exists:
        cand_w.writeheader()
        cand_f.flush(); os.fsync(cand_f.fileno())

    qwen_f = None
    qwen_w = None
    if args.use_qwen:
        qwen_f = open(qwen_csv, "a", encoding="utf-8", newline="")
        qwen_w = csv.DictWriter(qwen_f, fieldnames=qwen_fieldnames)
        if not qwen_exists:
            qwen_w.writeheader()
            qwen_f.flush(); os.fsync(qwen_f.fileno())

    qwen_dbg_exists = os.path.isfile(qwen_dbg_csv)
    qwen_dbg_f = open(qwen_dbg_csv, "a", encoding="utf-8", newline="")
    qwen_dbg_w = csv.DictWriter(qwen_dbg_f, fieldnames=qwen_dbg_fieldnames)
    if not qwen_dbg_exists:
        qwen_dbg_w.writeheader()
        qwen_dbg_f.flush();
        os.fsync(qwen_dbg_f.fileno())

    for (sayings, emotion), g in tqdm(groups, desc="Eval groups"):
        qk = query_key(sayings, emotion)
        if args.use_qwen and (qk in done_keys):
            continue

        # pos/neg motion ids (unique)
        g = g.copy()
        g["motion_id"] = g["raw_file_name"].apply(motion_id_from_raw)

        pos_ids = list(dict.fromkeys(g[g["label"] == "pos"]["motion_id"].tolist()))
        neg_ids = list(dict.fromkeys(g[g["label"] == "neg"]["motion_id"].tolist()))
        pos_ids = pos_ids[: args.max_pos_eval]
        neg_ids = neg_ids[: args.max_neg_eval]

        # build pos/neg embeddings (token hist)
        pos_embs = []
        pos_caps = []
        for mid in pos_ids:
            p = vqvae_lookup(vq_by_stem, mid)
            if p is None:
                continue
            codes = load_motion_codes_from_vq(p)
            pos_embs.append(token_hist_emb(codes))
        # captions (from csv, if exists)
        if "motion_caption" in g.columns:
            pos_caps = g[g["label"] == "pos"]["motion_caption"].astype(str).tolist()[:8]
            neg_caps = g[g["label"] == "neg"]["motion_caption"].astype(str).tolist()[:8]
        else:
            neg_caps = []

        neg_embs = []
        for mid in neg_ids:
            p = vqvae_lookup(vq_by_stem, mid)
            if p is None:
                continue
            codes = load_motion_codes_from_vq(p)
            neg_embs.append(token_hist_emb(codes))


        pos_rows = g[g["label"] == "pos"]

        # build prompt
        input_text = build_source_text(sayings, emotion)
        input_ids = a2rm_tok(input_text, return_tensors="pt").input_ids.to("cuda", dtype=torch.long)

        # generate N candidates
        gen_items_for_qwen = []
        cand_caps_for_qwen = []

        for ci in range(args.num_candidates):
            with torch.no_grad():
                out = a2rm_model.generate(
                    input_ids,
                    max_length=256,
                    do_sample=True,
                    temperature=0.8,
                    top_k=200
                )
#             out_text = a2rm_tok.decode(out[0], skip_special_tokens=True)
#             print('out_text:', out_text)
            out_text = a2rm_tok.decode(out[0], skip_special_tokens=False)
            out_text = out_text.replace("<pad>", "").replace("</s>", "").strip()

            codes = parse_motion_tokens_v2(out_text, max_len=args.gen_max_len, codebook_size=512)
            if len(codes) == 0:
                codes = [1] * 196  # 你原来的 fallback

            motion_string = "<Motion Tokens>" + "".join([f"<{t}>" for t in codes]) + "</Motion Tokens>"
            cap_prompt = "Generate text: " + motion_string


            # save tokens
            key_hash = abs(hash((sayings, emotion))) % (10**12)
            tok_dir = os.path.join(args.out_dir, "tokens", f"q{key_hash}")
            os.makedirs(tok_dir, exist_ok=True)
            tok_path = os.path.join(tok_dir, f"cand_{ci:03d}.npy")
            np.save(tok_path, np.array(codes, dtype=np.int16))


            print('Motion Tokens to be captioned:', motion_string)

            with torch.no_grad():
                cap_inp = cap_tok(cap_prompt, return_tensors="pt").input_ids.to(device, dtype=torch.long)
                cap_out = cap_model.generate(
                    cap_inp,
                    max_length=200,
                    num_beams=1,
                    do_sample=False,
                )
            cap_text = cap_tok.decode(cap_out[0], skip_special_tokens=True).strip().strip('"')
            print("cap_text:",cap_text)
            # winrate vs neg
            emb = token_hist_emb(codes)
            wr = gen_vs_neg_winrate(emb, pos_embs=pos_embs, neg_embs=neg_embs)

            row = {
                "query_key": qk,
                "split": g["split"].iloc[0] if "split" in g.columns else "unknown",
                "sayings": sayings,
                "emotion": emotion,
                "cand_id": f"gen_{ci}",
                "cand_tokens_path": tok_path,
                "cand_caption": cap_text,
                "cand_token_len": len(codes),
                "tokenhist_winrate_vs_neg": wr,
            }
            append_row(cand_w, cand_f, row)

            gen_items_for_qwen.append({"id": f"gen_{ci}", "caption": cap_text})
            cand_caps_for_qwen.append(cap_text)

        # Qwen top-3 selection among (gen+pos+neg)
        if qwen is not None:
            # 原始 pos/neg（仍然用 motion_caption）
            pos_items = [{"id": f"pos_{i}", "caption": c} for i, c in enumerate(pos_caps)]
            neg_items = [{"id": f"neg_{i}", "caption": c} for i, c in enumerate(neg_caps)]

            # ✅ 统一命名 + 打散
            # 用 query_key 做 seed，保证可复现（也可以用 hash(sayings+emotion)）
            seed_int = int(qk[:8], 16)  # md5 前 8 位转 int
            candidates_llm, cid2orig = build_uniform_candidates(
                gen_items=gen_items_for_qwen,
                pos_items=pos_items,
                neg_items=neg_items,
                seed=seed_int,
                max_total=24,   # 你可以调：8 gen + 8 pos + 8 neg = 24
            )

            prompt = build_qwen_rank_prompt_uniform(
                sayings=sayings,
                emotion=emotion,
                candidates=candidates_llm,
                top_k=3,
            )

            qtxt = qwen.generate(prompt)
            jobj = extract_json_strict(qtxt)

            top3_cids = []
            reason = ""
            if isinstance(jobj, dict):
                top3_cids = normalize_topk_ids(jobj.get("top3", []))
                reason = str(jobj.get("reason", ""))

            # ✅ 映射回原 id（gen_*, pos_*, neg_*）
            top3_orig = [cid2orig[c] for c in top3_cids if c in cid2orig]

            gen_in_top3 = sum(1 for x in top3_orig if isinstance(x, str) and x.startswith("gen_"))

            done_keys.add(qk)

            # 构建 cid->caption
            cid2caption = {c["cid"]: c["caption"] for c in candidates_llm}

            # 构建 orig_id -> caption / type（gen/pos/neg）
            orig2caption = {}
            orig2type = {}
            for c in candidates_llm:
                cid = c["cid"]
                orig = cid2orig.get(cid, "")
                cap = c["caption"]
                orig2caption[orig] = cap
                if isinstance(orig, str):
                    if orig.startswith("gen_"):
                        orig2type[orig] = "gen"
                    elif orig.startswith("pos_"):
                        orig2type[orig] = "pos"
                    elif orig.startswith("neg_"):
                        orig2type[orig] = "neg"
                    else:
                        orig2type[orig] = "unk"
                else:
                    orig2type[orig] = "unk"

            # top3 captions（既存 cid，也存 orig 对应 caption）
            top3_captions = [cid2caption.get(cid, "") for cid in top3_cids]

            # candidates_json（Qwen 看到的完整候选，带类型、映射）
            candidates_json = []
            for c in candidates_llm:
                cid = c["cid"]
                orig = cid2orig.get(cid, "")
                candidates_json.append({
                    "cid": cid,
                    "orig_id": orig,
                    "type": orig2type.get(orig, "unk"),
                    "caption": c["caption"],
                })

            # 三类 caption 单独存（更容易 debug）
            gen_caps_json = json.dumps([it["caption"] for it in gen_items_for_qwen], ensure_ascii=False)
            pos_caps_json = json.dumps(pos_caps, ensure_ascii=False)
            neg_caps_json = json.dumps(neg_caps, ensure_ascii=False)

            # 原本你写 qwen_top3.csv 的 qrow 可保留
            qrow = {
                "query_key": qk,
                "split": g["split"].iloc[0] if "split" in g.columns else "unknown",
                "sayings": sayings,
                "emotion": emotion,
                "top3": json.dumps(top3_orig, ensure_ascii=False),
                "gen_in_top3": gen_in_top3,
                "reason": reason[:400],
                "raw_qwen_text": qtxt[:800],
            }
            append_row(qwen_w, qwen_f, qrow)

            # ✅ 新增：写 qwen_top3_debug.csv
            qdbg_row = {
                "query_key": qk,
                "split": g["split"].iloc[0] if "split" in g.columns else "unknown",
                "sayings": sayings,
                "emotion": emotion,
                "audio_code_path": "",  # 你如果后面补了 audio_code_path，就写真实值

                "top3_cids": json.dumps(top3_cids, ensure_ascii=False),
                "top3_orig": json.dumps(top3_orig, ensure_ascii=False),
                "top3_captions": json.dumps(top3_captions, ensure_ascii=False),
                "gen_in_top3": gen_in_top3,
                "reason": reason[:400],
                "raw_qwen_text": qtxt[:800],

                "candidates_json": json.dumps(candidates_json, ensure_ascii=False),

                "gen_caps_json": gen_caps_json,
                "pos_caps_json": pos_caps_json,
                "neg_caps_json": neg_caps_json,
            }
            append_row(qwen_dbg_w, qwen_dbg_f, qdbg_row)

    # ----------------------------
    # save outputs (streaming mode)
    # ----------------------------
    cand_csv = os.path.join(args.out_dir, "candidates.csv")
    print("[Saved]", cand_csv, "(streaming append mode)")

    if args.use_qwen:
        qcsv = os.path.join(args.out_dir, "qwen_top3.csv")
        print("[Saved]", qcsv, "(streaming append mode)")

        # 汇总：从 CSV 读回算（不依赖内存里的 qwen_rows）
        try:
            qdf = pd.read_csv(qcsv, encoding="utf-8")
            if len(qdf) > 0 and "gen_in_top3" in qdf.columns:
                print(
                    "[Qwen Summary] groups=", len(qdf),
                    "avg_gen_in_top3=", float(qdf["gen_in_top3"].mean()),
                    "pct_gen_in_top3>=1=", float((qdf["gen_in_top3"] >= 1).mean()),
                    "pct_gen_in_top3==3=", float((qdf["gen_in_top3"] == 3).mean()),
                )
            else:
                print("[Qwen Summary] empty or missing gen_in_top3.")
        except Exception as e:
            print("[WARN] failed to load qwen summary from csv:", e)

    cand_f.close()
    if qwen_f is not None:
        qwen_f.close()
    if qwen_dbg_f is not None:
        qwen_dbg_f.close()


if __name__ == "__main__":
    main()
