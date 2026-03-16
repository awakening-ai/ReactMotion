#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reactmotion/models/judge_network.py

JudgeNetwork: Multi-modal scorer/ranker for best-of-K motion selection.

Contains:
- JudgeNetwork model (condition + motion encoders with contrastive scoring)
- JudgeGroupDataset, GroupCollator (data loading for judge training/eval)
- Loss functions: group_infonce_loss, in_group_order_margin_loss, alignment_reg
- Evaluation helpers: acc_at_k_any_gold, ndcg_at_k, run_eval
- Utility dataclasses: CondBatch, GroupBatch
"""

import os, re, json, math, random
from dataclasses import dataclass, field
from os.path import join as pjoin
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import T5Tokenizer, T5EncoderModel


# =========================================================
# Constants / Utils
# =========================================================

MODES_FULL = ["a", "a+e", "t", "t+e", "t+a", "t+a+e"]
MODE2ID = {m: i for i, m in enumerate(MODES_FULL)}

GOLD_ALIASES = {"gold", "pos", "positive", "gt", "true", "1"}
SILVER_ALIASES = {"silver"}
NEG_ALIASES = {"neg", "negative", "0"}

DEFAULT_AUDIO_VOCAB = 2048
DEFAULT_AUDIO_PAD = 2048
DEFAULT_MOTION_VOCAB = 512


def safe_l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def canon_label(x: Any) -> str:
    s = "" if pd.isna(x) else str(x).strip().lower()
    if s in GOLD_ALIASES:
        return "gold"
    if s in SILVER_ALIASES:
        return "silver"
    if s in NEG_ALIASES:
        return "neg"
    return s


def normalize_text(x: Any) -> str:
    s = "" if pd.isna(x) else str(x)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def clean_audio_stem(x: Any) -> str:
    s = "" if pd.isna(x) else str(x).strip()
    if s.lower().endswith(".wav"):
        s = s[:-4]
    return s


def motion_id_from_raw(raw_file_name: Any) -> str:
    x = "" if pd.isna(raw_file_name) else str(raw_file_name).strip()
    mid = x.split("_", 1)[0] if x else ""
    return str(mid).zfill(6) if mid else ""


def read_split_csv(pairs_csv_or_dir: str, split: str) -> pd.DataFrame:
    if os.path.isdir(pairs_csv_or_dir):
        path = pjoin(pairs_csv_or_dir, f"{split}.csv")
        if not os.path.isfile(path):
            raise RuntimeError(f"Missing split csv: {path}")
        return pd.read_csv(path, encoding="utf-8")
    df = pd.read_csv(pairs_csv_or_dir, encoding="utf-8")
    if "split" not in df.columns:
        raise RuntimeError("pairs_csv is file but no `split` column; use dir train/val/test or add split col.")
    sp = df["split"].astype(str).str.lower().str.strip()
    return df[sp == split].copy()


def index_vq_dir(vq_dir: str) -> Dict[str, str]:
    if not os.path.isdir(vq_dir):
        raise RuntimeError(f"Missing VQVAE dir: {vq_dir}")
    mp: Dict[str, str] = {}
    for fn in os.listdir(vq_dir):
        if fn.endswith(".npy"):
            stem = os.path.splitext(fn)[0]
            mp[stem] = pjoin(vq_dir, fn)
    return mp


def pick_code_from_stem(code_dir: str, stem: str) -> Optional[str]:
    p_npz = pjoin(code_dir, stem + ".npz")
    if os.path.exists(p_npz):
        return p_npz
    p_npy = pjoin(code_dir, stem + ".npy")
    if os.path.exists(p_npy):
        return p_npy
    return None


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

    # heuristic transpose to [T,K]
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


def load_motion_codes(vq_path: str, codebook_size: int = 512) -> np.ndarray:
    arr = np.load(vq_path, allow_pickle=False)
    arr = np.asarray(arr).reshape(-1)
    arr = np.clip(arr, 0, codebook_size - 1).astype(np.int64)
    return arr


# =========================================================
# Batches
# =========================================================

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
    debug_modes: List[str] = field(default_factory=list)


@dataclass
class GroupBatch:
    cb: CondBatch
    motion_codes: torch.Tensor
    motion_pad: torch.Tensor
    label: torch.Tensor
    cand_mask: torch.Tensor
    group_ids: List[Any]
    cand_paths: List[List[str]]
    cand_item_w: torch.Tensor     # [B,C]
    group_w: torch.Tensor         # [B]


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
        debug_modes=getattr(cb, "debug_modes", []),
    )

def fuse_mean_masked(
    z_t: torch.Tensor,
    z_a: torch.Tensor,
    z_e: torch.Tensor,
    has_t: torch.Tensor,
    has_a: torch.Tensor,
    has_e: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Mean-fuse condition embeddings with per-sample modality masks.

    Inputs:
      z_*   : [B,D]
      has_* : [B] bool  (True means modality is present)

    Output:
      z_f   : [B,D] L2-normalized
    """
    # masks: [B,1]
    m_t = has_t.to(dtype=z_t.dtype).unsqueeze(-1)
    m_a = has_a.to(dtype=z_a.dtype).unsqueeze(-1)
    m_e = has_e.to(dtype=z_e.dtype).unsqueeze(-1)

    # sanitize (avoid NaN/Inf blowing up downstream)
    z_t = torch.nan_to_num(z_t, nan=0.0, posinf=0.0, neginf=0.0)
    z_a = torch.nan_to_num(z_a, nan=0.0, posinf=0.0, neginf=0.0)
    z_e = torch.nan_to_num(z_e, nan=0.0, posinf=0.0, neginf=0.0)

    num = z_t * m_t + z_a * m_a + z_e * m_e
    den = (m_t + m_a + m_e).clamp_min(1.0)  # avoid divide-by-zero
    z_f = num / den
    return safe_l2norm(z_f, eps=eps)

# =========================================================
# Dataset (group-wise)
# =========================================================

class JudgeGroupDataset(Dataset):
    """
    Each __getitem__ returns:
      - condition: sayings, emotion, audio_code_path (optional)
      - candidate motions: gold/silver/neg motion vq paths
    """
    def __init__(
        self,
        split: str,
        pairs_csv: str,
        dataset_dir: str,
        audio_code_dir: str,
        key_by: str = "group_id",
        seed: int = 42,
        k_gold: int = 3,
        k_silver: int = 2,
        k_neg: int = 5,
        require_audio: bool = False,
    ):
        assert split in ["train", "val", "test"]
        assert key_by in ["group_id", "sayings_emotion"]

        df = read_split_csv(pairs_csv, split).copy()
        need_cols = ["label", "sayings", "emotion", "raw_file_name", "generated_wav_name"]
        missing = [c for c in need_cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"Missing columns in csv: {missing}. Found: {list(df.columns)}")
        if key_by == "group_id" and ("group_id" not in df.columns):
            raise RuntimeError("key_by='group_id' but csv missing `group_id` column")

        df["label_c"] = df["label"].apply(canon_label)
        df["sayings"] = df["sayings"].map(normalize_text)
        df["emotion"] = df["emotion"].astype(str).fillna("").str.strip()
        df["motion_id"] = df["raw_file_name"].apply(motion_id_from_raw)
        df["audio_stem"] = df["generated_wav_name"].map(clean_audio_stem)

        if "item_w" in df.columns:
            df["item_w"] = pd.to_numeric(df["item_w"], errors="coerce").fillna(1.0).astype(np.float32)
        else:
            df["item_w"] = 1.0

        if "group_w" in df.columns:
            df["group_w"] = pd.to_numeric(df["group_w"], errors="coerce").fillna(1.0).astype(np.float32)
        else:
            df["group_w"] = 1.0

        vq_dir = pjoin(dataset_dir, "HumanML3D", "VQVAE")
        vq_by_mid = index_vq_dir(vq_dir)

        self.audio_code_dir = audio_code_dir
        self.seed = int(seed)
        self.k_gold = int(k_gold)
        self.k_silver = int(k_silver)
        self.k_neg = int(k_neg)
        self.epoch = 0

        gb = df.groupby(["group_id"], dropna=False) if key_by == "group_id" else df.groupby(["sayings", "emotion"], dropna=False)

        # motion-id -> max item weight
        mid2w = {}
        for mid, w in zip(df["motion_id"].tolist(), df["item_w"].tolist()):
            if not mid:
                continue
            mid2w[mid] = max(float(mid2w.get(mid, 0.0)), float(w))

        drops = {"no_gold": 0, "no_neg": 0, "no_audio": 0, "vq_missing": 0}
        groups = []

        def mids_to_vq_with_w(mids):
            out = []
            for mid in mids:
                p = vq_by_mid.get(mid)
                if p is None:
                    drops["vq_missing"] += 1
                    continue
                out.append((p, float(mid2w.get(mid, 1.0))))
            # unique by path keep max w
            mp = {}
            for p, w in out:
                mp[p] = max(mp.get(p, 0.0), float(w))
            return [(p, mp[p]) for p in mp.keys()]

        for key, g in gb:
            sayings = str(g["sayings"].iloc[0])
            emotion = str(g["emotion"].iloc[0])
            gid = (key if not isinstance(key, tuple) else key[0]) if key_by == "group_id" else f"{sayings}__{emotion}"

            # audio codes
            stems = [s for s in g["audio_stem"].tolist() if str(s).strip()]
            stems = list(dict.fromkeys([str(s).strip() for s in stems]))
            audio_paths = []
            for st in stems:
                p = pick_code_from_stem(audio_code_dir, st)
                if p is not None:
                    audio_paths.append(p)
            audio_paths = list(dict.fromkeys(audio_paths))

            if require_audio and len(audio_paths) == 0:
                drops["no_audio"] += 1
                continue

            gold = mids_to_vq_with_w(g[g["label_c"] == "gold"]["motion_id"].tolist())
            silver = mids_to_vq_with_w(g[g["label_c"] == "silver"]["motion_id"].tolist())
            neg = mids_to_vq_with_w(g[g["label_c"] == "neg"]["motion_id"].tolist())

            if len(gold) == 0:
                drops["no_gold"] += 1
                continue
            if len(neg) == 0:
                drops["no_neg"] += 1
                continue

            group_w = float(g["group_w"].iloc[0]) if "group_w" in g.columns else 1.0

            groups.append(dict(
                group_id=gid,
                group_w=group_w,
                sayings=sayings,
                emotion=emotion,
                audio_paths=audio_paths,
                gold=gold,
                silver=silver,
                neg=neg,
            ))

        self.groups = groups
        print(f"[JudgeGroupDataset] split={split} groups={len(self.groups)} key_by={key_by}")
        print("[JudgeGroupDataset] drops:", drops)
        if len(self.groups) == 0:
            raise RuntimeError("0 groups after filtering. Check csv / vq_dir / audio_code_dir / require_audio.")

    def __len__(self):
        return len(self.groups)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    @staticmethod
    def _sample_k(rng: random.Random, arr: List[Any], k: int) -> List[Any]:
        if len(arr) == 0 or k <= 0:
            return []
        if len(arr) >= k:
            return rng.sample(arr, k)
        return [rng.choice(arr) for _ in range(k)]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        g = self.groups[idx]
        rng = random.Random(self.seed ^ (idx * 1000003) ^ (self.epoch * 9176))

        audio_path = ""
        if len(g["audio_paths"]) > 0:
            audio_path = rng.choice(g["audio_paths"])

        gold_s = self._sample_k(rng, g["gold"], self.k_gold)
        silv_s = self._sample_k(rng, g["silver"], self.k_silver) if len(g["silver"]) else []
        neg_s  = self._sample_k(rng, g["neg"], self.k_neg)

        cand_paths, cand_labels, cand_w = [], [], []
        for p, w in gold_s:
            cand_paths.append(p); cand_labels.append(2); cand_w.append(w)
        for p, w in silv_s:
            cand_paths.append(p); cand_labels.append(1); cand_w.append(w)
        for p, w in neg_s:
            cand_paths.append(p); cand_labels.append(0); cand_w.append(w)

        return dict(
            group_id=g["group_id"],
            group_w=float(g.get("group_w", 1.0)),
            sayings=g["sayings"],
            emotion=g["emotion"],
            audio_code_path=audio_path,
            cand_motion_paths=cand_paths,
            cand_labels=np.array(cand_labels, dtype=np.int64),
            cand_item_w=np.array(cand_w, dtype=np.float32),
        )


# =========================================================
# Collator (force single-modality ratio + ablation disables)
# =========================================================

class GroupCollator:
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
        seed: int = 42,
        deterministic_mode: bool = False,
        # forced single modality ratio
        force_single_ratio: float = 0.0,   # probability of forcing t-only or a-only
        disable_text: bool = False,
        disable_audio: bool = False,
        disable_emo: bool = False,
        fixed_mode: str = ""
    ):
        self.tok = t5_tokenizer
        self.emo2id = emo2id
        self.max_text_len = int(max_text_len)
        self.max_audio_len = int(max_audio_len)
        self.max_motion_len = int(max_motion_len)
        self.audio_codebooks = int(audio_codebooks)
        self.audio_pad_id = int(audio_pad_id)
        self.motion_codebook_size = int(motion_codebook_size)

        self.seed = int(seed)
        self.deterministic_mode = bool(deterministic_mode)
        self._call_idx = 0

        self.force_single_ratio = float(force_single_ratio)
        self.disable_text = bool(disable_text)
        self.disable_audio = bool(disable_audio)
        self.disable_emo = bool(disable_emo)

        self.fixed_mode = (fixed_mode or "").strip()
        if self.fixed_mode and self.fixed_mode not in MODES_FULL:
            raise ValueError(f"fixed_mode must be one of {MODES_FULL}, got {self.fixed_mode}")

        if self.disable_text and self.disable_audio and self.disable_emo:
            raise ValueError("All modalities disabled. At least one must remain enabled.")

    def _emo_id(self, emo: str) -> int:
        s = (emo or "").strip().lower()
        return int(self.emo2id.get(s, self.emo2id.get("<unk>", 0)))

    def _sample_modes(self, B: int) -> List[str]:
        if self.fixed_mode:
            return [self.fixed_mode] * B

        rng = random.Random(self.seed ^ (self._call_idx * 1337))
        self._call_idx += 1

        if self.deterministic_mode:
            base = self._call_idx
            return [MODES_FULL[(base + i) % len(MODES_FULL)] for i in range(B)]

        modes = []
        for _ in range(B):
            if self.force_single_ratio > 0 and rng.random() < self.force_single_ratio:
                # force single modality: t-only or a-only
                modes.append(rng.choice(["t", "a"]))
            else:
                modes.append(rng.choice(MODES_FULL))
        return modes

    @staticmethod
    def _repair_invalid(has_t, has_a, has_e):
        # forbid only-e -> turn on text
        only_e = has_e & (~has_t) & (~has_a)
        if only_e.any():
            has_t = has_t | only_e
        # forbid all-false -> turn on text
        none = (~has_t) & (~has_a) & (~has_e)
        if none.any():
            has_t = has_t | none
        return has_t, has_a, has_e

    def __call__(self, items: List[Dict[str, Any]]) -> GroupBatch:
        B = len(items)
        modes = self._sample_modes(B)

        has_t = torch.tensor([("t" in m) for m in modes], dtype=torch.bool)
        has_a = torch.tensor([("a" in m) for m in modes], dtype=torch.bool)
        has_e = torch.tensor([m.endswith("+e") for m in modes], dtype=torch.bool)
        mode_ids = torch.tensor([MODE2ID[m] for m in modes], dtype=torch.long)

        # global disables
        if self.disable_text:
            has_t[:] = False
        if self.disable_audio:
            has_a[:] = False
        if self.disable_emo:
            has_e[:] = False
        has_t, has_a, has_e = self._repair_invalid(has_t, has_a, has_e)

        # text
        texts = [it.get("sayings", "") or "" for it in items]
        enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_text_len, return_tensors="pt")
        text_input_ids = enc["input_ids"]
        text_attn_mask = enc["attention_mask"]

        # emotion ids
        emotion_ids = torch.tensor([self._emo_id(it.get("emotion", "")) for it in items], dtype=torch.long)

        # audio codes
        audio_list, audio_pad_mask_list = [], []
        for it in items:
            p = str(it.get("audio_code_path", "")).strip()
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

        # if sample wants audio but audio missing => disable audio for that sample
        valid_audio = ~audio_pad_mask.all(dim=1)
        bad_a = has_a & (~valid_audio)
        if bad_a.any():
            has_a[bad_a] = False
            has_t, has_a, has_e = self._repair_invalid(has_t, has_a, has_e)

        cb = CondBatch(
            has_t=has_t, has_a=has_a, has_e=has_e, mode_ids=mode_ids,
            text_input_ids=text_input_ids, text_attn_mask=text_attn_mask,
            emotion_ids=emotion_ids,
            audio_codes=audio_codes, audio_pad_mask=audio_pad_mask,
            debug_modes=modes,
        )

        # candidates pack
        C = max(len(it["cand_motion_paths"]) for it in items)
        Tm = self.max_motion_len
        motion_codes = torch.zeros((B, C, Tm), dtype=torch.long)
        motion_pad   = torch.ones((B, C, Tm), dtype=torch.bool)
        label        = torch.full((B, C), -1, dtype=torch.long)
        cand_mask    = torch.zeros((B, C), dtype=torch.bool)

        group_ids = [it["group_id"] for it in items]
        cand_paths_meta = [it["cand_motion_paths"] for it in items]

        cand_item_w = torch.ones((B, C), dtype=torch.float32)
        group_w = torch.tensor([float(it.get("group_w", 1.0)) for it in items], dtype=torch.float32)

        for b, it in enumerate(items):
            paths = it["cand_motion_paths"]
            labs = it["cand_labels"]
            ws = it.get("cand_item_w", None)
            if ws is None:
                ws = np.ones((len(paths),), dtype=np.float32)
            for j, p in enumerate(paths):
                m = load_motion_codes(p, codebook_size=self.motion_codebook_size)[:Tm]
                T = len(m)
                motion_codes[b, j, :T] = torch.from_numpy(m)
                motion_pad[b, j, :T] = False
                label[b, j] = int(labs[j])
                cand_mask[b, j] = True
                cand_item_w[b, j] = float(ws[j])

        return GroupBatch(
            cb=cb,
            motion_codes=motion_codes, motion_pad=motion_pad,
            label=label, cand_mask=cand_mask,
            group_ids=group_ids, cand_paths=cand_paths_meta,
            cand_item_w=cand_item_w,
            group_w=group_w,
        )


# =========================================================
# Model components
# =========================================================

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
        # mask True=pad/ignore
        B, T, D = x.shape
        scores = torch.matmul(x, self.query)  # [B,T]

        # handle all-masked rows
        all_mask = mask.all(dim=1)  # [B]
        if all_mask.any():
            mask = mask.clone()
            mask[all_mask, 0] = False
            x = x.clone()
            x[all_mask, 0, :] = 0.0
            scores = torch.matmul(x, self.query)

        scores = scores.masked_fill(mask, -1e9)
        w = F.softmax(scores, dim=-1)
        pooled = torch.bmm(w.unsqueeze(1), x).squeeze(1)  # [B,D]
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
        # audio_codes: [B,Ta,K]
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
        # motion_codes: [B,T]
        B, T = motion_codes.shape
        x = self.emb(motion_codes)
        pos = torch.arange(T, device=motion_codes.device).unsqueeze(0)
        x = x + self.pos(pos)
        return x, motion_pad_mask


class JudgeNetwork(nn.Module):
    """
    Encode:
      condition -> z_t, z_a, z_e, z_fused
      motion    -> z_m
    Score:
      s = scale * dot(z_cond, z_motion)
    """
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

        # Text
        self.text_enc = T5EncoderModel.from_pretrained(t5_name_or_path)
        self.text_proj = nn.Linear(self.text_enc.config.d_model, d_model)
        self.text_pool = AttentionPooling(d_model=d_model, output_dim=output_dim)

        # Audio
        self.audio_proc = AudioTokenProcessorMulti(
            vocab_size=audio_vocab, pad_id=audio_pad_id, codebooks=audio_codebooks,
            d_model=d_model, max_len=max_audio_len,
        )
        self.audio_enc = TransformerEncoder(d_model=d_model, nhead=nhead, num_layers=enc_layers,
                                            dim_feedforward=ff_dim, dropout=dropout)
        self.audio_pool = AttentionPooling(d_model=d_model, output_dim=output_dim)

        # Emotion
        self.emo_emb = nn.Embedding(num_emotions, d_model)
        self.emo_ln = nn.LayerNorm(d_model)
        self.emo_proj = nn.Linear(d_model, output_dim)

        # Fused condition encoder (token-level fuse)
        self.fuse_type_emb = nn.Embedding(3, d_model)   # 0=text,1=audio,2=emo
        self.mode_emb = nn.Embedding(len(MODES_FULL), d_model)
        self.cond_fuse = TransformerEncoder(d_model=d_model, nhead=nhead, num_layers=enc_layers,
                                            dim_feedforward=ff_dim, dropout=dropout)
        self.cond_pool = AttentionPooling(d_model=d_model, output_dim=output_dim)

        # Motion
        self.motion_proc = MotionTokenProcessor(vocab_size=motion_vocab, d_model=d_model, max_len=max_motion_len)
        self.motion_enc = TransformerEncoder(d_model=d_model, nhead=nhead, num_layers=enc_layers,
                                             dim_feedforward=ff_dim, dropout=dropout)
        self.motion_pool = AttentionPooling(d_model=d_model, output_dim=output_dim)

    def scale(self) -> torch.Tensor:
        x = self.logit_scale.float().clamp(math.log(1e-3), math.log(20.0))
        return x.exp()

    def encode_condition(self, cb: CondBatch) -> Dict[str, torch.Tensor]:
        device = cb.text_input_ids.device
        B = cb.text_input_ids.size(0)

        # =========================
        # ---- text (SAFE) ----
        # =========================
        has_t = cb.has_t.bool()

        # allocate deterministic "no-info" outputs
        Tt = cb.text_input_ids.size(1)
        d_model = self.text_proj.out_features
        out_dim = self.text_pool.proj.out_features

        xt = torch.zeros((B, Tt, d_model), device=device, dtype=torch.float32)  # [B,Tt,d_model]
        z_t = torch.zeros((B, out_dim), device=device, dtype=torch.float32)  # [B,output_dim]
        mt = torch.ones((B, Tt), device=device, dtype=torch.bool)  # True=pad

        if has_t.any():
            ids_t = cb.text_input_ids[has_t]
            att_t = cb.text_attn_mask[has_t]

            out_t = self.text_enc(input_ids=ids_t, attention_mask=att_t)
            xt_t = self.text_proj(out_t.last_hidden_state)  # [Bt,Tt,d_model]
            mt_t = (att_t == 0)  # True=pad
            zt_t = self.text_pool(xt_t, mt_t)  # [Bt,output_dim]

            xt[has_t] = xt_t.to(torch.float32)
            mt[has_t] = mt_t
            z_t[has_t] = zt_t.to(torch.float32)

        # =========================
        # ---- audio (SAFE) ----
        # =========================
        has_a = cb.has_a.bool()
        Ta = cb.audio_codes.size(1)

        xa = torch.zeros((B, Ta, d_model), device=device, dtype=torch.float32)  # [B,Ta,d_model]
        ma = torch.ones((B, Ta), device=device, dtype=torch.bool)  # True=pad
        z_a = torch.zeros((B, out_dim), device=device, dtype=torch.float32)  # [B,output_dim]

        if has_a.any():
            ac = cb.audio_codes[has_a]
            am = cb.audio_pad_mask[has_a]

            xa_a, ma_a = self.audio_proc(ac, am)  # [Ba,Ta,d_model], [Ba,Ta]
            xa_a = self.audio_enc(xa_a, src_key_padding_mask=ma_a)
            za_a = self.audio_pool(xa_a, ma_a)  # [Ba,output_dim]

            xa[has_a] = xa_a.to(torch.float32)
            ma[has_a] = ma_a
            z_a[has_a] = za_a.to(torch.float32)

        # =========================
        # ---- emotion ----
        # =========================
        ze0 = self.emo_ln(self.emo_emb(cb.emotion_ids))  # [B,d_model]
        z_e = F.normalize(self.emo_proj(ze0), p=2, dim=-1)  # [B,output_dim]

        # =========================
        # ---- fused ----
        # =========================
        # text token in d_model: mean over non-pad
        mt_nonpad = (~mt).float()  # 1 for non-pad
        denom_t = mt_nonpad.sum(dim=1, keepdim=True).clamp(min=1.0)
        t_tok = (xt * mt_nonpad.unsqueeze(-1)).sum(dim=1) / denom_t  # [B,d_model]

        # audio token in d_model: mean over non-pad
        ma_nonpad = (~ma).float()
        denom_a = ma_nonpad.sum(dim=1, keepdim=True).clamp(min=1.0)
        a_tok = (xa * ma_nonpad.unsqueeze(-1)).sum(dim=1) / denom_a  # [B,d_model]

        e_tok = ze0  # [B,d_model]

        t_tok = t_tok + self.fuse_type_emb(torch.tensor([0], device=device)).view(1, -1)
        a_tok = a_tok + self.fuse_type_emb(torch.tensor([1], device=device)).view(1, -1)
        e_tok = e_tok + self.fuse_type_emb(torch.tensor([2], device=device)).view(1, -1)

        x = torch.stack([t_tok, a_tok, e_tok], dim=1)  # [B,3,d_model]
        m = torch.stack([~cb.has_t, ~cb.has_a, ~cb.has_e], dim=1)  # True=mask

        # forbid all-masked rows (extra safety)
        all_mask = m.all(dim=1)
        if all_mask.any():
            m = m.clone()
            x = x.clone()
            m[all_mask, 0] = False
            x[all_mask, 0] = 0.0

        x = x + self.mode_emb(cb.mode_ids).unsqueeze(1)
        x = self.cond_fuse(x, src_key_padding_mask=m)
        z_f = self.cond_pool(x, m)

        # keep dtype consistent with the rest of the model (optional)
        # (If you prefer bf16/fp16 under autocast, you can cast back here.)
        return {"z_t": z_t, "z_a": z_a, "z_e": z_e, "z_f": z_f}

    def encode_motion(self, motion_codes: torch.Tensor, motion_pad_mask: torch.Tensor) -> torch.Tensor:
        x, m = self.motion_proc(motion_codes, motion_pad_mask)
        x = self.motion_enc(x, src_key_padding_mask=m)
        return self.motion_pool(x, m)


# =========================================================
# Losses / Metrics
# =========================================================

def in_group_order_margin_loss(
    logits: torch.Tensor,       # [B,C] masked -inf for invalid
    label: torch.Tensor,        # [B,C], 2 gold, 1 silver, 0 neg
    cand_mask: torch.Tensor,    # [B,C]
    m_gs: float = 0.2,
    m_sn: float = 0.2,
    sample_w: Optional[torch.Tensor] = None,  # [B]
) -> torch.Tensor:
    device = logits.device

    gold_m = (label == 2) & cand_mask
    silv_m = (label == 1) & cand_mask
    neg_m  = (label == 0) & cand_mask

    neg_inf = torch.tensor(float("-inf"), device=device, dtype=logits.dtype)

    best_g = torch.where(gold_m.any(dim=1), logits.masked_fill(~gold_m, neg_inf).max(dim=1).values, neg_inf)
    best_s = torch.where(silv_m.any(dim=1), logits.masked_fill(~silv_m, neg_inf).max(dim=1).values, neg_inf)
    best_n = torch.where(neg_m.any(dim=1),  logits.masked_fill(~neg_m,  neg_inf).max(dim=1).values, neg_inf)

    keep = gold_m.any(dim=1) & neg_m.any(dim=1) & torch.isfinite(best_g) & torch.isfinite(best_n)
    if sample_w is not None:
        keep = keep & (sample_w > 0)

    if not keep.any():
        return logits.sum() * 0.0

    # gold > neg
    l_gn = F.relu(m_gs - (best_g - best_n))

    # gold > silver, silver > neg if silver exists; otherwise fallback to gold>neg
    has_s = silv_m.any(dim=1) & torch.isfinite(best_s)
    l_gs = F.relu(m_gs - (best_g - best_s))
    l_sn = F.relu(m_sn - (best_s - best_n))
    l_pair = torch.where(has_s, l_gs + l_sn, l_gn)

    l = l_pair[keep]
    if sample_w is not None:
        w = sample_w[keep].float()
        return (l * w).sum() / w.sum().clamp_min(1e-6)
    return l.mean()


def group_infonce_loss(
    zc: torch.Tensor,                 # [B,D]
    zm: torch.Tensor,                 # [B,C,D]
    label: torch.Tensor,              # [B,C]
    cand_mask: torch.Tensor,          # [B,C]
    logit_scale: torch.Tensor,        # scalar
    use_silver_as_pos: bool = False,
    sample_w: Optional[torch.Tensor] = None,     # [B]
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = logit_scale.float() * torch.einsum("bd,bcd->bc", zc.float(), zm.float())
    logits = logits.masked_fill(~cand_mask, float("-inf"))

    if use_silver_as_pos:
        pos = (label >= 1) & cand_mask
    else:
        pos = (label == 2) & cand_mask

    keep = cand_mask.any(dim=1) & pos.any(dim=1)
    if sample_w is not None:
        keep = keep & (sample_w > 0)

    if not keep.any():
        return logits.sum() * 0.0, logits

    logits_k = logits[keep]
    pos_k = pos[keep]

    log_den = torch.logsumexp(logits_k, dim=1)
    log_num = torch.logsumexp(logits_k.masked_fill(~pos_k, float("-inf")), dim=1)
    loss_vec = (log_den - log_num)

    if sample_w is not None:
        w = sample_w[keep].float()
        loss = (loss_vec * w).sum() / w.sum().clamp_min(1e-6)
    else:
        loss = loss_vec.mean()
    return loss, logits


def group_infonce_loss_with_bank(
    zc: torch.Tensor,               # [B,D]
    zm: torch.Tensor,               # [B,C,D]
    label: torch.Tensor,            # [B,C]
    cand_mask: torch.Tensor,        # [B,C]
    logit_scale: torch.Tensor,      # scalar
    z_bank: Optional[torch.Tensor], # [M,D] or None
    bank_alpha: float = 1.0,        # strength on bank logits only (NOT on loss)
    use_silver_as_pos: bool = False,
    sample_w: Optional[torch.Tensor] = None,   # [B]
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits_main = logit_scale.float() * torch.einsum("bd,bcd->bc", zc.float(), zm.float())
    logits_main = logits_main.masked_fill(~cand_mask, float("-inf"))

    if use_silver_as_pos:
        pos = (label >= 1) & cand_mask
    else:
        pos = (label == 2) & cand_mask

    keep = cand_mask.any(dim=1) & pos.any(dim=1)
    if sample_w is not None:
        keep = keep & (sample_w > 0)
    if not keep.any():
        return logits_main.sum() * 0.0, logits_main

    lm = logits_main[keep]  # [K,C]
    pm = pos[keep]          # [K,C]

    if z_bank is not None and z_bank.numel() > 0:
        lb = (logit_scale.float() * bank_alpha) * (zc[keep].float() @ z_bank.float().T)  # [K,M]
        log_den = torch.logsumexp(torch.cat([lm, lb], dim=1), dim=1)
    else:
        log_den = torch.logsumexp(lm, dim=1)

    log_num = torch.logsumexp(lm.masked_fill(~pm, float("-inf")), dim=1)
    loss_vec = (log_den - log_num)

    if sample_w is not None:
        w = sample_w[keep].float()
        loss = (loss_vec * w).sum() / w.sum().clamp_min(1e-6)
    else:
        loss = loss_vec.mean()
    return loss, logits_main


def alignment_reg(zt, za, ze, has_t, has_a, has_e) -> torch.Tensor:
    """
    Encourage modality embeddings to agree (high cosine similarity) on samples where both modalities exist.
    This improves scoring consistency between t-only / a-only / fused modes.
    """
    zt = F.normalize(zt, p=2, dim=-1)
    za = F.normalize(za, p=2, dim=-1)
    ze = F.normalize(ze, p=2, dim=-1)

    reg_sum = 0.0
    cnt = 0

    def add_pair(z1, z2, m1, m2):
        nonlocal reg_sum, cnt
        m = (m1 & m2).float()
        if m.sum().item() < 1:
            return
        cos = (z1 * z2).sum(dim=-1)       # [-1,1]
        # penalty = 1 - cos (minimize)
        reg = ((1.0 - cos) * m).sum() / m.sum().clamp_min(1.0)
        reg_sum = reg_sum + reg
        cnt += 1

    add_pair(zt, za, has_t, has_a)
    add_pair(zt, ze, has_t, has_e)
    add_pair(za, ze, has_a, has_e)

    if cnt == 0:
        return zt.sum() * 0.0
    return reg_sum / cnt


@torch.no_grad()
def acc_at_k_any_gold(logits: torch.Tensor, label: torch.Tensor, cand_mask: torch.Tensor, k: int) -> float:
    logits = logits.masked_fill(~cand_mask, float("-inf"))
    B, C = logits.shape
    kk = min(k, C)
    topk = torch.topk(logits, k=kk, dim=1).indices
    gold = (label == 2)
    hits = [float(gold[b, topk[b]].any().item()) for b in range(B)]
    return float(np.mean(hits)) if hits else float("nan")


@torch.no_grad()
def ndcg_at_k(logits: torch.Tensor, label: torch.Tensor, cand_mask: torch.Tensor, k: int = 5) -> float:
    logits = logits.masked_fill(~cand_mask, float("-inf"))
    B, C = logits.shape
    kk = min(k, C)

    gain = torch.zeros_like(label, dtype=torch.float32)
    gain[label == 2] = 2.0
    gain[label == 1] = 1.0
    gain = gain * cand_mask.float()

    order = torch.argsort(logits, dim=1, descending=True)
    scores = []
    for b in range(B):
        idx = order[b, :kk]
        g = gain[b, idx]
        denom = torch.log2(torch.arange(kk, device=logits.device, dtype=torch.float32) + 2.0)
        dcg = (g / denom).sum().item()
        ideal = torch.sort(gain[b, cand_mask[b]], descending=True).values[:kk]
        idcg = (ideal / denom[:ideal.numel()]).sum().item() if ideal.numel() > 0 else 0.0
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else float("nan")


@torch.no_grad()
def run_eval(model: JudgeNetwork, loader, device, use_silver_as_pos: bool):
    model.eval()
    losses, a1, a3, a5, ndcgs = [], [], [], [], []

    for gb in loader:
        cb = move_cb_to(gb.cb, device)
        B, C, Tm = gb.motion_codes.shape
        mc = gb.motion_codes.view(B*C, Tm).to(device, non_blocking=True)
        mp = gb.motion_pad.view(B*C, Tm).to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.bfloat16):
            zs = model.encode_condition(cb)
            zf = zs["z_f"]
            zm = model.encode_motion(mc, mp).view(B, C, -1)

            loss, logits = group_infonce_loss(
                zf, zm,
                gb.label.to(device, non_blocking=True),
                gb.cand_mask.to(device, non_blocking=True),
                model.scale(),
                use_silver_as_pos=use_silver_as_pos,
                sample_w=None,
            )

        losses.append(float(loss.item()))
        lab = gb.label.to(device)
        msk = gb.cand_mask.to(device)
        a1.append(acc_at_k_any_gold(logits, lab, msk, 1))
        a3.append(acc_at_k_any_gold(logits, lab, msk, 3))
        a5.append(acc_at_k_any_gold(logits, lab, msk, 5))
        ndcgs.append(ndcg_at_k(logits, lab, msk, 5))

    def mean(xs):
        xs = [x for x in xs if x == x]
        return float(np.mean(xs)) if xs else float("nan")

    return dict(
        val_loss=mean(losses),
        acc1=mean(a1), acc3=mean(a3), acc5=mean(a5),
        ndcg5=mean(ndcgs),
    )
