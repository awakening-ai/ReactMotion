#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate A2RM gen_dump outputs with a trained unified scorer + winrates + template-bias analysis.

What this script does:
1) Score every generated sample in gen_dump -> scores.csv / scores.jsonl
   - adds motion_hash / motion_len per generated sample
2) Pick best gen per (mode, group_key) -> group_summary.csv
   - includes best_gen_motion_hash / best_gen_len
3) Load reference (gold/silver/neg) from pairs_csv + VQ files under dataset_dir/HumanML3D/VQVAE
   - compute group winrates:
        win_gen_vs_neg, win_gen_vs_silver, win_gen_vs_gold,
        gen_at3 (best gen in top3 among {gold+silver+neg+best_gen}),
        gen_any_at3 (any of 3 gens in top3 among {gold+silver+neg+all_gens})
   -> group_metrics.csv
4) Bucket analysis by motion_hash frequency (default threshold >=50 = template)
   - outputs:
        aggregate_by_bucket.csv
        aggregate_by_mode_bucket.csv

Usage example:
python -m eval.eval_scorer_on_gen_dump \
  --scorer_ckpt /ibex/project/c2191/luoc/checkpoints/scorer-optimized/best.pt \
  --gen_dump_dir /ibex/project/c2191/luoc/results/reactmotion/dump_tpe/cond=t+a \
  --index_jsonl /ibex/project/c2191/luoc/results/reactmotion/dump_tpe/index_test.jsonl \
  --pairs_csv ./new_data/test.csv \
  --dataset_dir /ibex/project/c2191/luoc/dataset/A2R \
  --split test \
  --pairs_key_by group_id \
  --out_dir /ibex/project/c2191/luoc/results/eval_results/reactmotion-conda-tpa-10000_scorer_win \
  --batch_size 256 --num_workers 4 \
  --max_groups 100

Notes:
- motion_hash is md5 over raw int32 bytes of the (unpadded) motion token sequence.
- For bucket freq, we use frequency of motion_hash over *all generated samples* (scores.csv),
  then assign each group by best_gen_motion_hash's global frequency.
"""

import os, re, json, math, argparse, hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import T5Tokenizer, T5EncoderModel


# -----------------------------
# constants
# -----------------------------
MODES = ["a", "a+e", "t", "t+e", "t+a", "t+a+e"]
MODE2ID = {m: i for i, m in enumerate(MODES)}

DEFAULT_AUDIO_VOCAB = 2048
DEFAULT_AUDIO_PAD = 2048
DEFAULT_MOTION_VOCAB = 512


# -----------------------------
# helper: io
# -----------------------------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def infer_meta_path_from_motion_path(p: str) -> str:
    if p.endswith(".motion_codes.npy"):
        return p.replace(".motion_codes.npy", ".meta.json")
    return p + ".meta.json"


# -----------------------------
# helper: hashing
# -----------------------------
def motion_hash_from_codes(codes_1d: np.ndarray) -> str:
    arr = np.asarray(codes_1d, dtype=np.int32).reshape(-1)
    return hashlib.md5(arr.tobytes()).hexdigest()


# -----------------------------
# helper: load audio codes (npz/npy)
# -----------------------------
def load_audio_codes_any(path: str) -> np.ndarray:
    obj = np.load(path, allow_pickle=False)
    if isinstance(obj, np.lib.npyio.NpzFile):
        if "codes" in obj.files:
            arr = obj["codes"]
        else:
            arr = obj[obj.files[0]]
        obj.close()
        return np.array(arr)
    return np.array(obj)

def normalize_audio_codes(arr: np.ndarray, codebooks: int = 8) -> np.ndarray:
    a = np.array(arr)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    elif a.ndim == 2:
        pass
    else:
        a = a.reshape(a.shape[0], -1)

    # try align to [T, K]
    if a.shape[0] == codebooks and a.shape[1] != codebooks:
        a = a.transpose(1, 0)
    elif a.shape[1] == codebooks:
        pass
    elif a.shape[1] > a.shape[0] and a.shape[0] < codebooks:
        a = a.transpose(1, 0)

    T, K = a.shape
    if K < codebooks:
        pad = np.zeros((T, codebooks - K), dtype=a.dtype)
        a = np.concatenate([a, pad], axis=1)
    elif K > codebooks:
        a = a[:, :codebooks]
    return a.astype(np.int64)

def load_motion_codes(path: str, codebook_size: int = 512) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    arr = np.asarray(arr).reshape(-1)
    arr = np.clip(arr, 0, codebook_size - 1).astype(np.int64)
    return arr


# -----------------------------
# batches
# -----------------------------
@dataclass
class CondBatch:
    has_t: torch.Tensor
    has_a: torch.Tensor
    has_e: torch.Tensor
    mode_ids: torch.Tensor
    text_input_ids: torch.Tensor
    text_attn_mask: torch.Tensor
    emotion_ids: torch.Tensor
    audio_codes: torch.Tensor
    audio_pad_mask: torch.Tensor

def move_cb_to(cb: CondBatch, device: torch.device) -> CondBatch:
    return CondBatch(
        has_t=cb.has_t.to(device, non_blocking=True),
        has_a=cb.has_a.to(device, non_blocking=True),
        has_e=cb.has_e.to(device, non_blocking=True),
        mode_ids=cb.mode_ids.to(device, non_blocking=True),
        text_input_ids=cb.text_input_ids.to(device, non_blocking=True),
        text_attn_mask=cb.text_attn_mask.to(device, non_blocking=True),
        emotion_ids=cb.emotion_ids.to(device, non_blocking=True),
        audio_codes=cb.audio_codes.to(device, non_blocking=True),
        audio_pad_mask=cb.audio_pad_mask.to(device, non_blocking=True),
    )


# -----------------------------
# model (must match training)
# -----------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.norm(self.enc(x, src_key_padding_mask=src_key_padding_mask))

class AttentionPooling(nn.Module):
    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.proj = nn.Linear(d_model, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(x, self.query)     # [B,T]
        scores = scores.masked_fill(mask, -1e9)  # True=pad
        w = F.softmax(scores, dim=-1)
        pooled = torch.bmm(w.unsqueeze(1), x).squeeze(1)
        out = self.norm(self.proj(pooled))
        return F.normalize(out, p=2, dim=-1)

class AudioTokenProcessorMulti(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, codebooks: int, d_model: int, max_len: int):
        super().__init__()
        self.pad_id = int(pad_id)
        self.codebooks = int(codebooks)
        self.token_emb = nn.Embedding(vocab_size + 1, d_model, padding_idx=pad_id)
        self.level_emb = nn.Embedding(codebooks, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.token_emb.weight, mean=0, std=0.02)

    def forward(self, audio_codes: torch.Tensor, audio_pad_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, Ta, K = audio_codes.shape
        x = self.token_emb(audio_codes)  # [B,Ta,K,D]
        lv = torch.arange(K, device=audio_codes.device).view(1, 1, K)
        x = x + self.level_emb(lv)
        valid_k = (audio_codes != self.pad_id).float().unsqueeze(-1)
        denom = valid_k.sum(dim=2).clamp(min=1.0)
        x = (x * valid_k).sum(dim=2) / denom  # [B,Ta,D]
        pos = torch.arange(Ta, device=audio_codes.device).unsqueeze(0)
        x = x + self.pos_emb(pos)
        return x, audio_pad_mask

class MotionTokenProcessor(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.emb.weight, mean=0, std=0.02)

    def forward(self, motion_codes: torch.Tensor, motion_pad_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = motion_codes.shape
        x = self.emb(motion_codes)
        pos = torch.arange(T, device=motion_codes.device).unsqueeze(0)
        x = x + self.pos(pos)
        return x, motion_pad_mask

class UnifiedScorerBig(nn.Module):
    def __init__(
        self,
        t5_name_or_path: str,
        num_emotions: int,
        d_model: int,
        output_dim: int,
        nhead: int,
        enc_layers: int,
        ff_dim: int,
        dropout: float,
        audio_vocab: int,
        audio_pad_id: int,
        audio_codebooks: int,
        max_audio_len: int,
        motion_vocab: int,
        max_motion_len: int,
        temperature: float = 0.07,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model({d_model}) must be divisible by nhead({nhead})")

        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))

        self.text_enc = T5EncoderModel.from_pretrained(t5_name_or_path)
        self.text_proj = nn.Linear(self.text_enc.config.d_model, d_model)

        self.audio_proc = AudioTokenProcessorMulti(
            vocab_size=audio_vocab, pad_id=audio_pad_id, codebooks=audio_codebooks,
            d_model=d_model, max_len=max_audio_len,
        )
        self.audio_enc = TransformerEncoder(d_model=d_model, nhead=nhead, num_layers=enc_layers,
                                            dim_feedforward=ff_dim, dropout=dropout)

        self.emo_emb = nn.Embedding(num_emotions, d_model)
        self.emo_ln = nn.LayerNorm(d_model)

        self.fuse_type_emb = nn.Embedding(3, d_model)   # 0=text,1=audio,2=emo
        self.mode_emb = nn.Embedding(len(MODES), d_model)

        self.cond_fuse = TransformerEncoder(d_model=d_model, nhead=nhead, num_layers=enc_layers,
                                            dim_feedforward=ff_dim, dropout=dropout)
        self.cond_pool = AttentionPooling(d_model=d_model, output_dim=output_dim)

        self.motion_proc = MotionTokenProcessor(vocab_size=motion_vocab, d_model=d_model, max_len=max_motion_len)
        self.motion_enc = TransformerEncoder(d_model=d_model, nhead=nhead, num_layers=enc_layers,
                                             dim_feedforward=ff_dim, dropout=dropout)
        self.motion_pool = AttentionPooling(d_model=d_model, output_dim=output_dim)

    def scale(self) -> torch.Tensor:
        x = self.logit_scale.float().clamp(math.log(1e-3), math.log(20.0))
        return x.exp()

    def encode_condition(self, cb: CondBatch) -> torch.Tensor:
        def _repair_all_mask(x: torch.Tensor, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            all_mask = m.all(dim=1)
            if all_mask.any():
                idx = all_mask.nonzero(as_tuple=False).view(-1)
                m[idx, 0] = False
                x[idx, 0] = 0.0
            return x, m

        device = cb.text_input_ids.device
        chunks, masks = [], []

        out = self.text_enc(input_ids=cb.text_input_ids, attention_mask=cb.text_attn_mask)
        xt = self.text_proj(out.last_hidden_state)
        mt = (cb.text_attn_mask == 0) | (~cb.has_t).unsqueeze(1)
        xt, mt = _repair_all_mask(xt, mt)
        xt = xt + self.fuse_type_emb(torch.zeros(1, device=device, dtype=torch.long)).view(1, 1, -1)
        chunks.append(xt); masks.append(mt)

        xa, ma = self.audio_proc(cb.audio_codes, cb.audio_pad_mask)
        ma = ma | (~cb.has_a).unsqueeze(1)
        xa, ma = _repair_all_mask(xa, ma)
        xa = self.audio_enc(xa, src_key_padding_mask=ma)
        xa = xa + self.fuse_type_emb(torch.ones(1, device=device, dtype=torch.long)).view(1, 1, -1)
        chunks.append(xa); masks.append(ma)

        xe = self.emo_ln(self.emo_emb(cb.emotion_ids)).unsqueeze(1)
        me = (~cb.has_e).unsqueeze(1).to(device=device)
        xe, me = _repair_all_mask(xe, me)
        xe = xe + self.fuse_type_emb(torch.full((1,), 2, device=device, dtype=torch.long)).view(1, 1, -1)
        chunks.append(xe); masks.append(me)

        x = torch.cat(chunks, dim=1)
        m = torch.cat(masks, dim=1)
        x, m = _repair_all_mask(x, m)
        x = x + self.mode_emb(cb.mode_ids).unsqueeze(1)
        x = self.cond_fuse(x, src_key_padding_mask=m)
        return self.cond_pool(x, m)

    def encode_motion(self, motion_codes: torch.Tensor, motion_pad_mask: torch.Tensor) -> torch.Tensor:
        x, m = self.motion_proc(motion_codes, motion_pad_mask)
        x = self.motion_enc(x, src_key_padding_mask=m)
        return self.motion_pool(x, m)


# -----------------------------
# dataset: read gen dump (via index jsonl OR walk)
# -----------------------------
class GenDumpDataset(Dataset):
    def __init__(self, gen_dump_dir: str, index_jsonl: Optional[str] = None):
        self.items: List[Dict[str, Any]] = []

        if index_jsonl and os.path.isfile(index_jsonl):
            entries = read_jsonl(index_jsonl)
            for e in entries:
                mode = e.get("mode", e.get("cond_mode", ""))
                group_key = e.get("group_key", e.get("key", ""))
                split = e.get("split", "")
                for it in e.get("items", []):
                    mp = it.get("motion_codes_npy", "")
                    if not mp:
                        continue
                    meta_p = infer_meta_path_from_motion_path(mp)
                    self.items.append({
                        "mode": mode,
                        "group_key": group_key,
                        "split": split,
                        "motion_codes_path": mp,
                        "meta_path": meta_p,
                    })
        else:
            for root, _, files in os.walk(gen_dump_dir):
                for fn in files:
                    if fn.endswith(".motion_codes.npy"):
                        mp = os.path.join(root, fn)
                        meta_p = infer_meta_path_from_motion_path(mp)
                        self.items.append({
                            "mode": "",
                            "group_key": "",
                            "split": "",
                            "motion_codes_path": mp,
                            "meta_path": meta_p,
                        })

        loaded = []
        for it0 in self.items:
            meta_path = it0["meta_path"]
            if not os.path.isfile(meta_path):
                continue
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            mode = meta.get("mode", meta.get("cond_mode", it0.get("mode", "")))
            group_key = meta.get("group_key", meta.get("key", it0.get("group_key", "")))
            group_key = "" if group_key is None else str(group_key)
            split = meta.get("split", it0.get("split", ""))

            loaded.append({
                "mode": mode,
                "group_key": group_key,
                "group_hash": meta.get("group_hash", ""),
                "ckpt_hash": meta.get("ckpt_hash", ""),
                "sayings": meta.get("sayings", ""),
                "emotion": meta.get("emotion", ""),
                "audio_code_path": meta.get("audio_code_path", ""),
                "audio_token_level": meta.get("audio_token_level", None),
                "gen_idx": meta.get("gen_idx", -1),
                "gen_seed": meta.get("gen_seed", -1),
                "sampling": meta.get("sampling", {}),
                "split": split,
                "motion_codes_path": it0["motion_codes_path"],
                "meta_path": meta_path,
            })

        self.items = loaded
        print(f"[GenDumpDataset] items={len(self.items)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


# -----------------------------
# collator for scorer (also computes motion_hash/len)
# -----------------------------
class ScorerCollator:
    def __init__(
        self,
        t5_tokenizer: T5Tokenizer,
        emo2id: Dict[str, int],
        max_text_len: int = 128,
        max_audio_len: int = 512,
        max_motion_len: int = 196,
        audio_codebooks: int = 8,
        audio_pad_id: int = DEFAULT_AUDIO_PAD,
        motion_codebook_size: int = DEFAULT_MOTION_VOCAB,
    ):
        self.tok = t5_tokenizer
        self.emo2id = emo2id
        self.max_text_len = int(max_text_len)
        self.max_audio_len = int(max_audio_len)
        self.max_motion_len = int(max_motion_len)
        self.audio_codebooks = int(audio_codebooks)
        self.audio_pad_id = int(audio_pad_id)
        self.motion_codebook_size = int(motion_codebook_size)

    def _emo_id(self, emo: str) -> int:
        s = (emo or "").strip().lower()
        return int(self.emo2id.get(s, self.emo2id.get("<unk>", 0)))

    def __call__(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        B = len(items)
        modes = [str(x.get("mode", "")).strip().lower() for x in items]

        # fallback infer from path segment "mode=xxx"
        for i, m in enumerate(modes):
            if m in MODE2ID:
                continue
            p = str(items[i].get("motion_codes_path", ""))
            m2 = ""
            if "mode=" in p:
                seg = p.split("mode=")[-1]
                m2 = seg.split("/")[0].strip()
            elif "cond=" in p:
                seg = p.split("cond=")[-1]
                m2 = seg.split("/")[0].strip()

            modes[i] = m2 if m2 in MODE2ID else "t"

        has_t = torch.tensor([("t" in m) for m in modes], dtype=torch.bool)
        has_a = torch.tensor([("a" in m) for m in modes], dtype=torch.bool)
        has_e = torch.tensor([m.endswith("+e") for m in modes], dtype=torch.bool)
        mode_ids = torch.tensor([MODE2ID.get(m, MODE2ID["t"]) for m in modes], dtype=torch.long)

        texts = [x.get("sayings", "") or "" for x in items]
        enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_text_len, return_tensors="pt")
        text_input_ids = enc["input_ids"]
        text_attn_mask = enc["attention_mask"]

        emotion_ids = torch.tensor([self._emo_id(x.get("emotion", "")) for x in items], dtype=torch.long)

        audio_list, audio_pad_mask_list = [], []
        for x in items:
            p = str(x.get("audio_code_path", "")).strip()
            if p and os.path.exists(p):
                a = normalize_audio_codes(load_audio_codes_any(p), codebooks=self.audio_codebooks)
                a = a[: self.max_audio_len]
                T = a.shape[0]
                pad_T = self.max_audio_len - T
                if pad_T > 0:
                    pad = np.full((pad_T, self.audio_codebooks), self.audio_pad_id, dtype=np.int64)
                    a = np.concatenate([a, pad], axis=0)
                m = np.zeros((self.max_audio_len,), dtype=np.bool_)
                if T < self.max_audio_len:
                    m[T:] = True
            else:
                a = np.full((self.max_audio_len, self.audio_codebooks), self.audio_pad_id, dtype=np.int64)
                m = np.ones((self.max_audio_len,), dtype=np.bool_)
            audio_list.append(torch.from_numpy(a))
            audio_pad_mask_list.append(torch.from_numpy(m))

        audio_codes = torch.stack(audio_list, dim=0)              # [B,Ta,K]
        audio_pad_mask = torch.stack(audio_pad_mask_list, dim=0)  # [B,Ta]

        cb = CondBatch(
            has_t=has_t, has_a=has_a, has_e=has_e, mode_ids=mode_ids,
            text_input_ids=text_input_ids, text_attn_mask=text_attn_mask,
            emotion_ids=emotion_ids,
            audio_codes=audio_codes, audio_pad_mask=audio_pad_mask,
        )

        # motion (one per item)
        Tm = self.max_motion_len
        motion_codes = torch.zeros((B, Tm), dtype=torch.long)
        motion_pad = torch.ones((B, Tm), dtype=torch.bool)

        motion_hashes: List[str] = []
        motion_lens: List[int] = []

        for b, x in enumerate(items):
            mp = x["motion_codes_path"]
            m = load_motion_codes(mp, codebook_size=self.motion_codebook_size)[:Tm]
            T = int(len(m))
            motion_codes[b, :T] = torch.from_numpy(m)
            motion_pad[b, :T] = False
            motion_hashes.append(motion_hash_from_codes(m))
            motion_lens.append(T)

        meta = {
            "mode": modes,
            "group_key": [x.get("group_key", "") for x in items],
            "group_hash": [x.get("group_hash", "") for x in items],
            "ckpt_hash": [x.get("ckpt_hash", "") for x in items],
            "split": [x.get("split", "") for x in items],
            "gen_idx": [x.get("gen_idx", -1) for x in items],
            "gen_seed": [x.get("gen_seed", -1) for x in items],
            "motion_codes_path": [x.get("motion_codes_path", "") for x in items],
            "meta_path": [x.get("meta_path", "") for x in items],
            "motion_hash": motion_hashes,
            "motion_len": motion_lens,
        }

        return {"cb": cb, "motion_codes": motion_codes, "motion_pad": motion_pad, "meta": meta}


# -----------------------------
# Pairs CSV -> reference map (gold/silver/neg) with VQ paths
# -----------------------------
def _read_split_csv(pairs_csv: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(pairs_csv)
    if "split" not in df.columns:
        raise RuntimeError(f"pairs_csv missing split col. cols={list(df.columns)}")
    df["split"] = df["split"].astype(str)
    return df[df["split"] == split].copy()

def _to_motion_id(raw_file_name: str) -> str:
    s = str(raw_file_name)
    m = re.search(r"(\d{6})", s)
    if m:
        return m.group(1)
    # fallback: take leading token
    return os.path.basename(s).split("_")[0]

def _clean_audio_stem(x: str) -> str:
    s = str(x).strip()
    s = s.replace(".wav", "").replace(".mp3", "").replace(".flac", "")
    return s

def _index_vq_by_motion_id(vq_dir: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for root, _, files in os.walk(vq_dir):
        for fn in files:
            if not fn.endswith(".npy"):
                continue
            p = os.path.join(root, fn)
            m = re.search(r"(\d{6})", fn)
            if not m:
                continue
            mid = m.group(1)
            # keep first occurrence (stable)
            if mid not in mapping:
                mapping[mid] = p
    return mapping

def _pick_code_from_stem(audio_code_dir: str, stem: str) -> Optional[str]:
    # fast path
    p = os.path.join(audio_code_dir, f"{stem}.npz")
    if os.path.exists(p):
        return p
    # fallback glob
    try:
        import glob
        cand = sorted(glob.glob(os.path.join(audio_code_dir, f"{stem}*.npz")))
        return cand[0] if cand else None
    except Exception:
        return None

def load_pairs_as_ref_map(
    pairs_csv: str,
    split: str,
    dataset_dir: str,
    key_by: str,
    audio_code_dir: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """
    Returns:
      ref_map[key] = {
         "sayings": str,
         "emotion": str,
         "audio_code_path": str,
         "gold_vq_paths": [...],
         "silver_vq_paths": [...],
         "neg_vq_paths": [...],
      }
      drops dict
    """
    assert key_by in ["group_id", "sayings_emotion", "sayings_only"]

    vq_dir = os.path.join(dataset_dir, "HumanML3D", "VQVAE")
    vq_by_mid = _index_vq_by_motion_id(vq_dir)
    print(f"[VQ] indexed {len(vq_by_mid)} motion_ids from {vq_dir}")

    df = _read_split_csv(pairs_csv, split).copy()

    need_cols = ["label", "sayings", "emotion", "raw_file_name"]
    for c in need_cols:
        if c not in df.columns:
            raise RuntimeError(f"pairs_csv missing col={c}. cols={list(df.columns)}")
    if key_by == "group_id" and "group_id" not in df.columns:
        raise RuntimeError("pairs_key_by=group_id but pairs_csv missing group_id column")

    if "generated_wav_name" not in df.columns:
        # allow, but then audio_code_path empty
        df["generated_wav_name"] = ""

    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["sayings"] = df["sayings"].astype(str)
    df["emotion"] = df["emotion"].astype(str)
    df["motion_id"] = df["raw_file_name"].apply(_to_motion_id)
    df["generated_wav_name"] = df["generated_wav_name"].apply(_clean_audio_stem)

    drops = dict(bad_label=0, vq_missing=0)

    if key_by == "group_id":
        group_cols = ["group_id"]
    elif key_by == "sayings_only":
        group_cols = ["sayings"]
    else:
        group_cols = ["sayings", "emotion"]

    ref_map: Dict[str, Dict[str, Any]] = {}

    for keys, g in df.groupby(group_cols, dropna=False):
        if key_by == "group_id":
            key = str(keys) if not isinstance(keys, tuple) else str(keys[0])
        else:
            key = "|||".join(map(str, keys)) if isinstance(keys, tuple) else str(keys)

        sayings = str(g["sayings"].iloc[0])
        emotion = str(g["emotion"].iloc[0])

        gold, silver, neg = [], [], []
        audio_stems = []

        ok = True
        for r in g.itertuples(index=False):
            lab = str(getattr(r, "label")).lower().strip()
            if lab not in {"gold", "silver", "neg"}:
                drops["bad_label"] += 1
                ok = False
                break
            mid = str(getattr(r, "motion_id"))
            vq = vq_by_mid.get(mid, None)
            if vq is None:
                drops["vq_missing"] += 1
                continue

            if lab == "gold":
                gold.append(vq)
            elif lab == "silver":
                silver.append(vq)
            else:
                neg.append(vq)

            stem = str(getattr(r, "generated_wav_name")).strip()
            if stem:
                audio_stems.append(stem)

        if not ok:
            continue

        # unique keep order
        def uniq(xs):
            return list(dict.fromkeys(xs))

        gold = uniq(gold)
        silver = uniq(silver)
        neg = uniq(neg)
        audio_stems = uniq(audio_stems)

        audio_code_path = ""
        if audio_code_dir and audio_stems:
            for st in audio_stems:
                p = _pick_code_from_stem(audio_code_dir, st)
                if p is not None:
                    audio_code_path = p
                    break

        if len(gold) == 0 or len(silver) == 0 or len(neg) == 0:
            # keep it anyway? you likely expect all exist; but safe to skip
            continue

        ref_map[key] = dict(
            sayings=sayings,
            emotion=emotion,
            audio_code_path=audio_code_path,
            gold_vq_paths=gold,
            silver_vq_paths=silver,
            neg_vq_paths=neg,
        )

    print(f"[Pairs] ref groups={len(ref_map)} drop={drops}")
    return ref_map, drops


# -----------------------------
# Build CondBatch for one group (reusing scorer inputs)
# -----------------------------
def build_condbatch_single(
    tok: T5Tokenizer,
    emo2id: Dict[str, int],
    mode: str,
    sayings: str,
    emotion: str,
    audio_code_path: str,
    max_text_len: int,
    max_audio_len: int,
    audio_codebooks: int = 8,
    audio_pad_id: int = DEFAULT_AUDIO_PAD,
) -> CondBatch:
    mode = mode.strip().lower()
    if mode not in MODE2ID:
        mode = "t"

    has_t = torch.tensor([("t" in mode)], dtype=torch.bool)
    has_a = torch.tensor([("a" in mode)], dtype=torch.bool)
    has_e = torch.tensor([mode.endswith("+e")], dtype=torch.bool)
    mode_ids = torch.tensor([MODE2ID[mode]], dtype=torch.long)

    enc = tok([sayings or ""], padding=True, truncation=True, max_length=max_text_len, return_tensors="pt")
    text_input_ids = enc["input_ids"]
    text_attn_mask = enc["attention_mask"]

    emo_key = (emotion or "").strip().lower()
    emo_id = int(emo2id.get(emo_key, emo2id.get("<unk>", 0)))
    emotion_ids = torch.tensor([emo_id], dtype=torch.long)

    # audio codes
    if ("a" in mode) and audio_code_path and os.path.exists(audio_code_path):
        a = normalize_audio_codes(load_audio_codes_any(audio_code_path), codebooks=audio_codebooks)
        a = a[:max_audio_len]
        T = a.shape[0]
        pad_T = max_audio_len - T
        if pad_T > 0:
            pad = np.full((pad_T, audio_codebooks), audio_pad_id, dtype=np.int64)
            a = np.concatenate([a, pad], axis=0)
        m = np.zeros((max_audio_len,), dtype=np.bool_)
        if T < max_audio_len:
            m[T:] = True
    else:
        a = np.full((max_audio_len, audio_codebooks), audio_pad_id, dtype=np.int64)
        m = np.ones((max_audio_len,), dtype=np.bool_)

    audio_codes = torch.from_numpy(a).unsqueeze(0)         # [1,Ta,K]
    audio_pad_mask = torch.from_numpy(m).unsqueeze(0)      # [1,Ta]

    return CondBatch(
        has_t=has_t, has_a=has_a, has_e=has_e, mode_ids=mode_ids,
        text_input_ids=text_input_ids, text_attn_mask=text_attn_mask,
        emotion_ids=emotion_ids,
        audio_codes=audio_codes, audio_pad_mask=audio_pad_mask,
    )


def load_motion_as_tensor(
    path: str,
    max_motion_len: int,
    motion_codebook_size: int = DEFAULT_MOTION_VOCAB,
) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
    m = load_motion_codes(path, codebook_size=motion_codebook_size)[:max_motion_len]
    T = int(len(m))
    mh = motion_hash_from_codes(m)
    motion = torch.zeros((1, max_motion_len), dtype=torch.long)
    pad = torch.ones((1, max_motion_len), dtype=torch.bool)
    motion[0, :T] = torch.from_numpy(m)
    pad[0, :T] = False
    return motion, pad, T, mh


# -----------------------------
# ranking helpers
# -----------------------------
def gen_at_k(best_gen_score: float, other_scores: List[float], k: int = 3) -> int:
    # rank among [best_gen] + other_scores, higher is better
    scores = [best_gen_score] + list(other_scores)
    # compute rank of index 0
    sorted_idx = np.argsort(-np.asarray(scores))
    rank = int(np.where(sorted_idx == 0)[0][0]) + 1
    return int(rank <= k)

def any_gen_at_k(gen_scores: List[float], other_scores: List[float], k: int = 3) -> int:
    if len(gen_scores) == 0:
        return 0
    scores = list(gen_scores) + list(other_scores)
    sorted_idx = np.argsort(-np.asarray(scores))
    topk = set(sorted_idx[:k].tolist())
    # any gen index (0..len(gen_scores)-1) in topk
    return int(any(i in topk for i in range(len(gen_scores))))


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer_ckpt", type=str, required=True, help="path to scorer checkpoint (cur.pt / best.pt)")
    ap.add_argument("--gen_dump_dir", type=str, required=True)
    ap.add_argument("--index_jsonl", type=str, default="", help="index_test.jsonl (recommended)")
    ap.add_argument("--out_dir", type=str, required=True)

    # winrate refs
    ap.add_argument("--pairs_csv", type=str, required=True)
    ap.add_argument("--dataset_dir", type=str, required=True)
    ap.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    ap.add_argument("--pairs_key_by", type=str, default="group_id", choices=["group_id", "sayings_emotion", "sayings_only"])
    ap.add_argument("--audio_code_dir", type=str, default="", help="optional override; default uses dataset_dir/audio-raws-09-01-2026-code")
    ap.add_argument("--max_groups", type=int, default=0, help="if >0, only evaluate first N gen groups for winrates")

    # bucket
    ap.add_argument("--template_freq_th", type=int, default=50, help="motion_hash freq >= th is template bucket")

    # dataloader
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)

    # lengths
    ap.add_argument("--max_text_len", type=int, default=128)
    ap.add_argument("--max_audio_len", type=int, default=512)
    ap.add_argument("--max_motion_len", type=int, default=196)

    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print("[Device]", device)

    # load ckpt
    ckpt = torch.load(args.scorer_ckpt, map_location="cpu")
    emo2id = ckpt.get("emo2id", {"<unk>": 0})
    train_args = ckpt.get("args", {})
    model_sd = ckpt["model"]

    # build model from ckpt args
    t5_encoder = train_args.get("t5_encoder", train_args.get("t5_name_or_path", "google-t5/t5-base"))
    d_model = int(train_args.get("d_model", 768))
    output_dim = int(train_args.get("output_dim", 512))
    nhead = int(train_args.get("nhead", 12))
    enc_layers = int(train_args.get("enc_layers", 6))
    ff_dim = int(train_args.get("ff_dim", 3072))
    dropout = float(train_args.get("dropout", 0.1))
    temperature = float(train_args.get("temperature", 0.07))

    print("[CKPT] Loaded args from checkpoint.")
    print("  t5_encoder:", t5_encoder)
    print("  d_model/output_dim:", d_model, output_dim)
    print("  layers/nhead/ff:", enc_layers, nhead, ff_dim)
    print("  emo_vocab:", len(emo2id))

    tok = T5Tokenizer.from_pretrained(t5_encoder)

    model = UnifiedScorerBig(
        t5_name_or_path=t5_encoder,
        num_emotions=len(emo2id),
        d_model=d_model,
        output_dim=output_dim,
        nhead=nhead,
        enc_layers=enc_layers,
        ff_dim=ff_dim,
        dropout=dropout,
        audio_vocab=DEFAULT_AUDIO_VOCAB,
        audio_pad_id=DEFAULT_AUDIO_PAD,
        audio_codebooks=8,
        max_audio_len=int(args.max_audio_len),
        motion_vocab=DEFAULT_MOTION_VOCAB,
        max_motion_len=int(args.max_motion_len),
        temperature=temperature,
    )
    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    if missing:
        print("[WARN] missing keys:", len(missing))
    if unexpected:
        print("[WARN] unexpected keys:", len(unexpected))

    model.to(device).eval()

    # -----------------------------
    # 1) Score all generated samples
    # -----------------------------
    ds = GenDumpDataset(args.gen_dump_dir, index_jsonl=args.index_jsonl if args.index_jsonl else None)
    collate = ScorerCollator(
        tok, emo2id,
        max_text_len=args.max_text_len,
        max_audio_len=args.max_audio_len,
        max_motion_len=args.max_motion_len,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate,
    )

    out_jsonl = os.path.join(args.out_dir, "scores.jsonl")
    out_csv = os.path.join(args.out_dir, "scores.csv")
    out_group_csv = os.path.join(args.out_dir, "group_summary.csv")
    out_group_metrics = os.path.join(args.out_dir, "group_metrics.csv")
    out_bucket_csv = os.path.join(args.out_dir, "aggregate_by_bucket.csv")
    out_mode_bucket_csv = os.path.join(args.out_dir, "aggregate_by_mode_bucket.csv")

    rows = []
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for batch in tqdm(loader, desc="Score gen samples"):
            cb = move_cb_to(batch["cb"], device)
            motion_codes = batch["motion_codes"].to(device, non_blocking=True)
            motion_pad = batch["motion_pad"].to(device, non_blocking=True)

            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.bfloat16):
                    zc = model.encode_condition(cb)                    # [B,D]
                    zm = model.encode_motion(motion_codes, motion_pad) # [B,D]
                    score = (model.scale().float() * (zc.float() * zm.float()).sum(dim=1)).detach().cpu().numpy()

            meta = batch["meta"]
            B = len(score)
            for i in range(B):
                rec = {
                    "mode": meta["mode"][i],
                    "split": meta["split"][i],
                    "group_key": meta["group_key"][i],
                    "group_hash": meta["group_hash"][i],
                    "ckpt_hash": meta["ckpt_hash"][i],
                    "gen_idx": int(meta["gen_idx"][i]),
                    "gen_seed": int(meta["gen_seed"][i]),
                    "score": float(score[i]),
                    "motion_hash": meta["motion_hash"][i],
                    "motion_len": int(meta["motion_len"][i]),
                    "motion_codes_path": meta["motion_codes_path"][i],
                    "meta_path": meta["meta_path"][i],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("[Saved]", out_jsonl)
    print("[Saved]", out_csv)

    if len(df) == 0:
        print("[Done] empty gen scores.")
        return

    # best gen per (mode, group_key)
    df2 = df.sort_values(["mode", "group_key", "score"], ascending=[True, True, False])
    best = df2.groupby(["mode", "group_key"], dropna=False).head(1).copy()
    best = best.rename(columns={
        "motion_hash": "best_gen_motion_hash",
        "motion_len": "best_gen_len",
    })
    best.to_csv(out_group_csv, index=False)
    print("[Saved]", out_group_csv)

    print("\n[Top best samples]")
    print(best.sort_values(["mode", "score"], ascending=[True, False]).head(10)[
        ["mode", "group_key", "gen_idx", "score", "best_gen_len", "best_gen_motion_hash"]
    ].to_string(index=False))

    # global hash freq over ALL gen samples
    hash_freq = df["motion_hash"].value_counts().to_dict()

    # -----------------------------
    # 2) Winrates against refs
    # -----------------------------
    audio_code_dir = args.audio_code_dir.strip()
    if not audio_code_dir:
        audio_code_dir = os.path.join(args.dataset_dir, "audio-raws-09-01-2026-code")

    ref_map, _ = load_pairs_as_ref_map(
        pairs_csv=args.pairs_csv,
        split=args.split,
        dataset_dir=args.dataset_dir,
        key_by=args.pairs_key_by,
        audio_code_dir=audio_code_dir,
    )

    # collect gen groups (mode, group_key) present in best
    gen_groups = list(best[["mode", "group_key"]].itertuples(index=False, name=None))
    print(f"[Gen] groups={len(gen_groups)}")

    # optional head
    if args.max_groups and args.max_groups > 0:
        gen_groups = gen_groups[: int(args.max_groups)]
        print(f"[Gen] trunc to first {len(gen_groups)} groups via --max_groups")

    # build quick access: all gen scores per group
    # group_df: rows for (mode, group_key)
    grouped_gens: Dict[Tuple[str, str], pd.DataFrame] = {}
    for (m, gk), sub in df.groupby(["mode", "group_key"], dropna=False):
        grouped_gens[(str(m), str(gk))] = sub

    # compute group metrics
    metrics_rows = []
    for mode, group_key in tqdm(gen_groups, desc="Group winrates"):
        mode = str(mode)
        group_key = str(group_key)

        # only evaluate groups that exist in ref_map
        # NOTE: ref_map is keyed by pairs_key_by. For group_id => group_key should be same numeric string.
        if group_key not in ref_map:
            continue

        ref = ref_map[group_key]

        # best gen row
        sub = grouped_gens.get((mode, group_key), None)
        if sub is None or len(sub) == 0:
            continue
        sub_sorted = sub.sort_values("score", ascending=False)
        best_row = sub_sorted.iloc[0]
        best_gen_score = float(best_row["score"])
        best_gen_hash = str(best_row["motion_hash"])
        best_gen_len = int(best_row["motion_len"])
        best_gen_path = str(best_row["motion_codes_path"])

        # all gen scores (typically 3)
        gen_scores_all = sub_sorted["score"].astype(float).tolist()

        # build condition batch (single)
        cb1 = build_condbatch_single(
            tok=tok,
            emo2id=emo2id,
            mode=mode,
            sayings=ref["sayings"],
            emotion=ref["emotion"],
            audio_code_path=ref.get("audio_code_path", ""),
            max_text_len=int(args.max_text_len),
            max_audio_len=int(args.max_audio_len),
        )
        cb1 = move_cb_to(cb1, device)

        # build motion candidates: refs
        gold_paths = ref["gold_vq_paths"]
        silver_paths = ref["silver_vq_paths"]
        neg_paths = ref["neg_vq_paths"]

        # load motions into one big batch
        cand_paths = gold_paths + silver_paths + neg_paths
        cand_types = (["gold"] * len(gold_paths)) + (["silver"] * len(silver_paths)) + (["neg"] * len(neg_paths))

        motion_list = []
        pad_list = []
        for p in cand_paths:
            m_codes = load_motion_codes(p, codebook_size=DEFAULT_MOTION_VOCAB)[: int(args.max_motion_len)]
            T = len(m_codes)
            mm = torch.zeros((int(args.max_motion_len),), dtype=torch.long)
            mpad = torch.ones((int(args.max_motion_len),), dtype=torch.bool)
            mm[:T] = torch.from_numpy(m_codes)
            mpad[:T] = False
            motion_list.append(mm)
            pad_list.append(mpad)

        motion_codes = torch.stack(motion_list, dim=0).to(device, non_blocking=True)   # [N,T]
        motion_pad = torch.stack(pad_list, dim=0).to(device, non_blocking=True)       # [N,T]

        # score refs
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.bfloat16):
                zc = model.encode_condition(cb1)  # [1,D]
                zm = model.encode_motion(motion_codes, motion_pad)  # [N,D]
                ref_scores = (model.scale().float() * (zc.float() * zm.float()).sum(dim=1)).detach().cpu().numpy().tolist()

        gold_scores = [s for s, t in zip(ref_scores, cand_types) if t == "gold"]
        silver_scores = [s for s, t in zip(ref_scores, cand_types) if t == "silver"]
        neg_scores = [s for s, t in zip(ref_scores, cand_types) if t == "neg"]

        # win: best gen vs best of each category (1=win, 0.5=tie, 0=loss)
        def win_vs_best(best_s: float, others: List[float]) -> float:
            if not others:
                return float("nan")
            best_other = float(np.max(others))
            if best_s > best_other:
                return 1.0
            if best_s == best_other:
                return 0.5
            return 0.0

        best_gold_score = float(np.max(gold_scores)) if len(gold_scores) > 0 else float("nan")
        best_silver_score = float(np.max(silver_scores)) if len(silver_scores) > 0 else float("nan")
        best_neg_score = float(np.max(neg_scores)) if len(neg_scores) > 0 else float("nan")

        win_gn = win_vs_best(best_gen_score, neg_scores)
        win_gs = win_vs_best(best_gen_score, silver_scores)
        win_gg = win_vs_best(best_gen_score, gold_scores)

        margin_vs_best_gold = float(best_gen_score - best_gold_score) if np.isfinite(best_gold_score) else float("nan")



        # gen_at3: best gen among (refs + best gen)
        gen_at3 = gen_at_k(best_gen_score, gold_scores + silver_scores + neg_scores, k=3)

        # gen_any_at3: any gen among (refs + all gens)
        gen_any_at3 = any_gen_at_k(gen_scores_all, gold_scores + silver_scores + neg_scores, k=3)

        # bucket by best_gen_motion_hash global freq
        freq = int(hash_freq.get(best_gen_hash, 0))

        # coarse bucket (保留你现有的)
        bucket = f">={int(args.template_freq_th)}" if freq >= int(args.template_freq_th) else f"<{int(args.template_freq_th)}"

        # fine bucket: 你可以改这些边界
        def bucket_fine_from_freq(f: int) -> str:
            if f <= 0:
                return "0"
            if f == 1:
                return "1"
            if f <= 3:
                return "2-3"
            if f <= 9:
                return "4-9"
            if f <= 19:
                return "10-19"
            if f <= 49:
                return "20-49"
            if f <= 99:
                return "50-99"
            if f <= 199:
                return "100-199"
            return "200+"

        bucket_fine = bucket_fine_from_freq(freq)

        metrics_rows.append({
            "mode": mode,
            "split": args.split,
            "group_key": group_key,
            "best_gen_score": best_gen_score,
            "best_gold_score": best_gold_score,              # NEW (debug)
            "win_gen_vs_neg": win_gn,
            "win_gen_vs_silver": win_gs,
            "win_gen_vs_gold": win_gg,
            "margin_vs_best_gold": margin_vs_best_gold,
            "gen_at3": int(gen_at3),
            "gen_any_at3": int(gen_any_at3),

            "best_gen_motion_hash": best_gen_hash,
            "best_gen_len": best_gen_len,
            "best_gen_motion_freq": freq,
            "bucket": bucket,
            "bucket_fine": bucket_fine,
            "best_gen_motion_codes_path": best_gen_path,

            "num_gold": len(gold_scores),
            "num_silver": len(silver_scores),
            "num_neg": len(neg_scores),
            "num_gen": len(gen_scores_all),
        })

    mdf = pd.DataFrame(metrics_rows)
    mdf.to_csv(out_group_metrics, index=False)
    out_overall_csv = os.path.join(args.out_dir, "aggregate_overall.csv")
    out_by_mode_csv = os.path.join(args.out_dir, "aggregate_by_mode.csv")
    out_by_fine_bucket_csv = os.path.join(args.out_dir, "aggregate_by_fine_bucket.csv")
    out_by_mode_fine_bucket_csv = os.path.join(args.out_dir, "aggregate_by_mode_fine_bucket.csv")

    print("[Saved]", out_group_metrics)

    if len(mdf) == 0:
        print("[Done] no groups matched ref_map (check pairs_key_by / split / gen group_key).")
        return

    # -----------------------------
    # 3) Aggregate helpers
    # -----------------------------
    def safe_mean(s: pd.Series) -> float:
        s2 = pd.to_numeric(s, errors="coerce")
        return float(s2.mean())


    METRIC_COLS = [
        "win_gen_vs_neg",
        "win_gen_vs_silver",
        "win_gen_vs_gold",
        "margin_vs_best_gold",
        "gen_at3",
        "gen_any_at3",
    ]

    def agg_df(sub: pd.DataFrame) -> Dict[str, Any]:
        out = {"groups": int(sub["group_key"].nunique())}
        for c in METRIC_COLS:
            out[f"mean {c}"] = safe_mean(sub[c])
        return out

    # pretty order for fine buckets
    FINE_BUCKET_ORDER = ["0","1","2-3","4-9","10-19","20-49","50-99","100-199","200+"]
    def bucket_fine_sort_key(s: Any) -> int:
        s = str(s)
        return FINE_BUCKET_ORDER.index(s) if s in FINE_BUCKET_ORDER else 999

    # -----------------------------
    # 3.1) overall (no bucket)
    # -----------------------------
    overall = agg_df(mdf)
    overall["split"] = args.split
    overall["template_th"] = int(args.template_freq_th)
    pd.DataFrame([overall]).to_csv(out_overall_csv, index=False)
    print("[Saved]", out_overall_csv)

    print("\n[Aggregate]")
    for k, v in overall.items():
        print(f"{k:>20s} = {v}")

    # -----------------------------
    # 3.2) by mode (no bucket)
    # -----------------------------
    rows_mode = []
    for mode, sub in mdf.groupby("mode", dropna=False):
        r = agg_df(sub)
        r["mode"] = mode
        r["split"] = args.split
        rows_mode.append(r)
    pd.DataFrame(rows_mode).sort_values(["mode"]).to_csv(out_by_mode_csv, index=False)
    print("[Saved]", out_by_mode_csv)

    # -----------------------------
    # 4) coarse bucket (keep existing)
    # -----------------------------
    g_bucket = mdf.groupby(["bucket"], dropna=False)
    out_rows = []
    for b, sub in g_bucket:
        r = agg_df(sub)
        r["bucket"] = b
        out_rows.append(r)
    pd.DataFrame(out_rows).sort_values(["bucket"]).to_csv(out_bucket_csv, index=False)
    print("[Saved]", out_bucket_csv)

    g_mode_bucket = mdf.groupby(["mode", "bucket"], dropna=False)
    out_rows2 = []
    for (mode, b), sub in g_mode_bucket:
        r = agg_df(sub)
        r["mode"] = mode
        r["bucket"] = b
        out_rows2.append(r)
    pd.DataFrame(out_rows2).sort_values(["mode", "bucket"]).to_csv(out_mode_bucket_csv, index=False)
    print("[Saved]", out_mode_bucket_csv)

    # -----------------------------
    # 5) fine bucket (new)
    # -----------------------------
    rows_fb = []
    for b, sub in mdf.groupby("bucket_fine", dropna=False):
        r = agg_df(sub)
        r["bucket_fine"] = b
        rows_fb.append(r)
    df_fb = pd.DataFrame(rows_fb)
    df_fb["__k"] = df_fb["bucket_fine"].map(bucket_fine_sort_key)
    df_fb.sort_values(["__k"]).drop(columns="__k").to_csv(out_by_fine_bucket_csv, index=False)
    print("[Saved]", out_by_fine_bucket_csv)

    rows_mfb = []
    for (mode, b), sub in mdf.groupby(["mode", "bucket_fine"], dropna=False):
        r = agg_df(sub)
        r["mode"] = mode
        r["bucket_fine"] = b
        rows_mfb.append(r)
    df_mfb = pd.DataFrame(rows_mfb)
    df_mfb["__k"] = df_mfb["bucket_fine"].map(bucket_fine_sort_key)
    df_mfb.sort_values(["mode", "__k"]).drop(columns="__k").to_csv(out_by_mode_fine_bucket_csv, index=False)
    print("[Saved]", out_by_mode_fine_bucket_csv)


    print("[Done]")


if __name__ == "__main__":
    main()
