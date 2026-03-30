#!/usr/bin/env python3
# reactmotion.dataset.reactmotionnet_dataset
# -*- coding: utf-8 -*-

import os
import re
import random
from os.path import join as pjoin
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset



def ensure_2d_mono(w: torch.Tensor) -> torch.Tensor:
    if w.dim() == 3:
        w = w.squeeze(0)
    if w.dim() == 1:
        w = w.unsqueeze(0)
    if w.dim() != 2:
        raise ValueError(f"Wave shape {tuple(w.shape)} invalid")
    if w.size(0) > 1:
        w = w.mean(dim=0, keepdim=True)
    return w

# ============================================================
# Utilities
# ============================================================

def normalize_text(x: Any) -> str:
    s = "" if pd.isna(x) else str(x)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _clean_audio_stem(x: Any) -> str:
    s = "" if pd.isna(x) else str(x).strip()
    if s.lower().endswith(".wav"):
        s = s[:-4]
    return s

def _read_split_csv(pairs_csv_or_dir: str, split: str) -> pd.DataFrame:
    if os.path.isdir(pairs_csv_or_dir):
        path = pjoin(pairs_csv_or_dir, f"{split}.csv")
        if not os.path.isfile(path):
            raise RuntimeError(f"Missing split csv: {path}")
        return pd.read_csv(path, encoding="utf-8")

    df = pd.read_csv(pairs_csv_or_dir, encoding="utf-8")
    if "split" not in df.columns:
        raise RuntimeError(
            "pairs_csv is a file but has no column `split`. "
            "Provide a directory with train.csv/val.csv/test.csv, or add split column."
        )
    sp = df["split"].astype(str).str.lower().str.strip()
    return df[sp == split].copy()

def _to_motion_id(raw_file_name: Any) -> str:
    x = "" if pd.isna(raw_file_name) else str(raw_file_name).strip()
    mid = x.split("_", 1)[0] if x else ""
    return str(mid).zfill(6) if mid else ""

def _pick_code_from_stem(code_dir: str, stem: str) -> Optional[str]:
    p_npz = pjoin(code_dir, stem + ".npz")
    if os.path.exists(p_npz):
        return p_npz
    p_npy = pjoin(code_dir, stem + ".npy")
    if os.path.exists(p_npy):
        return p_npy
    return None

def _pick_wav_from_stem(wav_dir: str, stem: str) -> Optional[str]:
    for ext in [".wav", ".flac", ".mp3", ".m4a"]:
        p = pjoin(wav_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None

def load_audio_tokens_any(path: str) -> np.ndarray:
    obj = np.load(path, allow_pickle=False)
    if isinstance(obj, np.lib.npyio.NpzFile):
        if "codes" in obj.files:
            arr = obj["codes"]
        else:
            arr = obj[obj.files[0]]
        obj.close()
        return arr
    return obj

def _format_audio_tokens(a_tokens: np.ndarray, level: str = "base") -> str:
    level = str(level)
    arr = np.array(a_tokens)

    if arr.ndim == 1:
        parts = ["<Audio Tokens>"]
        for t in arr.reshape(-1):
            parts.append(f"<Audio Token {int(t)}>")
        parts.append("</Audio Tokens>")
        return " ".join(parts)

    L = int(arr.shape[0])
    parts = ["<Audio Tokens>"]

    if level == "base":
        for t in arr[0].reshape(-1):
            parts.append(f"<Audio Level 0 Token {int(t)}>")
    elif level == "all":
        for i in range(L):
            for t in arr[i].reshape(-1):
                parts.append(f"<Audio Level {i} Token {int(t)}>")
    elif level == "rand":
        k = int(np.random.choice(np.arange(1, L + 1)))
        for i in range(k):
            for t in arr[i].reshape(-1):
                parts.append(f"<Audio Level {i} Token {int(t)}>")
    else:
        raise ValueError(f"Unknown audio_token_level={level}")

    parts.append("</Audio Tokens>")
    return " ".join(parts)

# ============================================================
# VQ resolver
# ============================================================

def resolve_vq_path(vq_dir: str, motion_id: str) -> Optional[str]:
    s = str(motion_id).strip()

    if s.endswith(".npy"):
        s = s[:-4].strip()

    m = re.match(r"^([Mm]\d+|\d+)", s)
    if m:
        s = m.group(1)

    if s.lower().startswith("m") and s[1:].isdigit():
        cand = [s, s.upper(), s.lower()]
        for k in cand:
            p = os.path.join(vq_dir, f"{k}.npy")
            if os.path.isfile(p):
                return p
        return None

    if s.isdigit():
        v = int(s)
        cand = [
            str(v).zfill(6),
            str(v).zfill(5),
            str(v),
            s,
        ]
        for k in cand:
            p = os.path.join(vq_dir, f"{k}.npy")
            if os.path.isfile(p):
                return p
        return None

    return None

def normalize_mid(mid: str) -> str:
    s = str(mid).strip()
    if s.endswith(".npy"):
        s = s[:-4]
    return s

# ============================================================
# Dataset
# ============================================================

class ReactMotionNet(Dataset):
    """
    Each item = one GROUP.
    Returns per item:
      key, transcription, emotion
      gold_vq_paths / silver_vq_paths / neg_vq_paths
      audio:
        - none: nothing
        - code: audio_text + audio_code_path
        - wav : wav_path
      group_w: float  ✅ NEW
    """

    def __init__(
        self,
        split: str,
        dataset_dir: str,
        pairs_csv: str,

        use_transcription: bool = True,
        use_emotion: bool = True,

        key_by: str = "group_id",  # group_id | sayings_emotion | sayings_only

        audio_mode: str = "none",        # none | code | wav
        audio_token_level: str = "base", # base | all | rand (only for code)
        audio_code_dir: Optional[str] = None,
        wav_dir: Optional[str] = None,

        min_gold: int = 1,
        min_silver: int = 2,
        min_neg: int = 5,
        min_audio: int = 1,

        max_gold_store: int = 64,
        max_silver_store: int = 64,
        max_neg_store: int = 256,

        debug_print_k: int = 2,

        # ✅ NEW: group weight from csv
        group_w_mode: str = "none",      # none | from_csv | constant
        group_w_col: str = "group_w",     # group_w | score | item_w ...
        group_w_agg: str = "mean",       # mean | max | first
        group_w_const: float = 1.0,
        group_w_clip_min: float = 0.2,
        group_w_clip_max: float = 5.0,

    ):
        assert split in ["train", "val", "test"]
        assert key_by in ["group_id", "sayings_emotion", "sayings_only"]
        assert audio_mode in ["none", "code", "wav"]
        assert audio_token_level in ["base", "all", "rand"]
        assert group_w_mode in ["none", "from_csv", "constant"]
        assert group_w_agg in ["mean", "max", "first"]

        self.split = split
        self.dataset_dir = dataset_dir
        self.pairs_csv = pairs_csv

        self.use_transcription = bool(use_transcription)
        self.use_emotion = bool(use_emotion)

        self.key_by = str(key_by)

        self.audio_mode = str(audio_mode)
        self.audio_token_level = str(audio_token_level)

        self.audio_code_dir = audio_code_dir or pjoin(self.dataset_dir, "audio_code")
        self.wav_dir = wav_dir or pjoin(self.dataset_dir, "audio_wav")

        self.min_gold = int(min_gold)
        self.min_silver = int(min_silver)
        self.min_neg = int(min_neg)
        self.min_audio = int(min_audio)

        self.max_gold_store = int(max_gold_store)
        self.max_silver_store = int(max_silver_store)
        self.max_neg_store = int(max_neg_store)

        self.group_w_mode = str(group_w_mode)
        self.group_w_const = float(group_w_const)
        self.group_w_clip_min = float(group_w_clip_min)
        self.group_w_clip_max = float(group_w_clip_max)
        self.group_w_col = str(group_w_col)
        self.group_w_agg = str(group_w_agg)

        # missing vq log
        self.missing_vq_log_path = split + "_missing.txt"
        self.missing_vq_log_unique = True

        # motion vq
        self.motion_root = pjoin(self.dataset_dir, "HumanML3D")
        self.motion_vqvae_dir = pjoin(self.motion_root, "VQVAE")

        if self.audio_mode == "code" and (not os.path.isdir(self.audio_code_dir)):
            raise RuntimeError(f"audio_mode=code but missing audio_code_dir: {self.audio_code_dir}")
        if self.audio_mode == "wav" and (not os.path.isdir(self.wav_dir)):
            raise RuntimeError(f"audio_mode=wav but missing wav_dir: {self.wav_dir}")

        df = _read_split_csv(self.pairs_csv, split).copy()

        need_cols = ["tier_label", "speaker_transcript", "speaker_emotion", "motion_id"]
        if self.audio_mode != "none":
            need_cols.append("speaker_audio_wav")

        for c in need_cols:
            if c not in df.columns:
                raise RuntimeError(f"Missing column `{c}` in csv. Found: {list(df.columns)}")

        if self.group_w_mode == "from_csv" and self.group_w_col:
            if self.group_w_col not in df.columns:
                print(
                    f"[WARN][ReactMotionNet] group_w_mode='from_csv' but "
                    f"column '{self.group_w_col}' not found in csv (found: {list(df.columns)}). "
                    f"Falling back to w=1.0 for all groups."
                )
                self.group_w_mode = "none"

        if self.key_by == "group_id" and ("group_id" not in df.columns):
            raise RuntimeError("key_by='group_id' but csv missing `group_id` column")

        df["tier_label"] = df["tier_label"].astype(str).str.lower().str.strip()
        df["speaker_transcript"] = df["speaker_transcript"].map(normalize_text)
        df["speaker_emotion"] = df["speaker_emotion"].astype(str)
        df["motion_id"] = df["motion_id"].apply(_to_motion_id)

        if self.audio_mode != "none":
            df["speaker_audio_wav"] = df["speaker_audio_wav"].map(_clean_audio_stem)

        drops = dict(
            bad_label=0,
            vq_missing=0,
            gold_insufficient=0,
            silver_insufficient=0,
            neg_insufficient=0,
            audio_missing=0,
            audio_stem_empty=0,
        )
        missing_vq_ids = set()
        grouped: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []

        if self.key_by == "group_id":
            group_cols = ["group_id"]
        elif self.key_by == "sayings_only":
            group_cols = ["speaker_transcript"]
        else:
            group_cols = ["speaker_transcript", "speaker_emotion"]

        def _safe_float(x, default=1.0):
            try:
                return float(x)
            except Exception:
                return float(default)

        for keys, g in df.groupby(group_cols, dropna=False):
            sayings = str(g["speaker_transcript"].iloc[0])
            emotion = str(g["speaker_emotion"].iloc[0])
            key = str(keys) if not isinstance(keys, tuple) else "|||".join(map(str, keys))

            gold_mids: List[str] = []
            silver_mids: List[str] = []
            neg_mids: List[str] = []
            audio_stems: List[str] = []

            ok = True
            for r in g.itertuples(index=False):
                lab = str(getattr(r, "tier_label")).lower().strip()
                if lab not in {"gold", "silver", "neg"}:
                    drops["bad_label"] += 1
                    ok = False
                    break

                mid = normalize_mid(getattr(r, "motion_id"))
                if lab == "gold":
                    gold_mids.append(mid)
                elif lab == "silver":
                    silver_mids.append(mid)
                else:
                    neg_mids.append(mid)

                if self.audio_mode != "none":
                    stem = str(getattr(r, "speaker_audio_wav")).strip()
                    if not stem:
                        drops["audio_stem_empty"] += 1
                    else:
                        audio_stems.append(stem)

            if not ok:
                continue

            gold_mids = list(dict.fromkeys(gold_mids))[: self.max_gold_store]
            silver_mids = list(dict.fromkeys(silver_mids))[: self.max_silver_store]
            neg_mids = list(dict.fromkeys(neg_mids))[: self.max_neg_store]
            audio_stems = list(dict.fromkeys(audio_stems))

            def mids_to_vq(mids):
                out = []
                for mid_ in mids:
                    p = resolve_vq_path(self.motion_vqvae_dir, mid_)
                    if p is not None:
                        out.append(p)
                    else:
                        drops["vq_missing"] += 1
                        if mid_:
                            missing_vq_ids.add(str(mid_))
                return out

            gold_vq = mids_to_vq(gold_mids)
            silver_vq = mids_to_vq(silver_mids)
            neg_vq = mids_to_vq(neg_mids)

            if len(gold_vq) < self.min_gold:
                drops["gold_insufficient"] += 1
                continue
            if len(silver_vq) < self.min_silver:
                drops["silver_insufficient"] += 1
                continue
            if len(neg_vq) < self.min_neg:
                drops["neg_insufficient"] += 1
                continue

            audio_paths: List[str] = []
            if self.audio_mode == "code":
                for stem in audio_stems:
                    p = _pick_code_from_stem(self.audio_code_dir, stem)
                    if p is not None:
                        audio_paths.append(p)
                audio_paths = sorted(set(audio_paths))
                if len(audio_paths) < self.min_audio:
                    drops["audio_missing"] += 1
                    continue

            if self.audio_mode == "wav":
                for stem in audio_stems:
                    p = _pick_wav_from_stem(self.wav_dir, stem)
                    if p is not None:
                        audio_paths.append(p)
                audio_paths = sorted(set(audio_paths))
                if len(audio_paths) < self.min_audio:
                    drops["audio_missing"] += 1
                    continue

            # compute group weight
            if self.group_w_mode == "constant":
                w = self.group_w_const

            elif self.group_w_mode == "from_csv":
                # group_w_col: column name; group_w_agg: mean | max | first
                col = self.group_w_col
                vals = [_safe_float(v, 1.0) for v in g[col].tolist()] if col in g.columns else [1.0]
                if not vals:
                    vals = [1.0]
                if self.group_w_agg == "mean":
                    w = float(np.mean(vals))
                elif self.group_w_agg == "max":
                    w = float(np.max(vals))
                else:  # "first"
                    w = vals[0]

            else:
                # "none" or fallback
                w = 1.0

            w = float(np.clip(w, self.group_w_clip_min, self.group_w_clip_max))

            grouped[key] = dict(
                transcription=sayings if self.use_transcription else "",
                emotion=emotion if self.use_emotion else "",
                gold_vq_paths=gold_vq,
                silver_vq_paths=silver_vq,
                neg_vq_paths=neg_vq,
                audio_paths=audio_paths,
                group_w=w,  # ✅ NEW
            )
            order.append(key)

        self.grouped = grouped
        self.keys = order

        print(
            f"[ReactMotionNet] split={split} queries={len(self.keys)} "
            f"key_by={self.key_by} audio_mode={self.audio_mode} "
            f"vq_dir={self.motion_vqvae_dir}"
        )
        print("[ReactMotionNet] drop_reasons:", drops)

        # summary
        n_groups = len(self.keys)
        gold_cnt = silver_cnt = neg_cnt = audio_cnt = 0
        uniq_vq, uniq_gold, uniq_silver, uniq_neg, uniq_audio = set(), set(), set(), set(), set()

        for k in self.keys:
            it = self.grouped[k]
            g_ = it["gold_vq_paths"]
            s_ = it["silver_vq_paths"]
            n_ = it["neg_vq_paths"]

            gold_cnt += len(g_)
            silver_cnt += len(s_)
            neg_cnt += len(n_)

            for p in g_:
                uniq_vq.add(p); uniq_gold.add(p)
            for p in s_:
                uniq_vq.add(p); uniq_silver.add(p)
            for p in n_:
                uniq_vq.add(p); uniq_neg.add(p)

            if self.audio_mode != "none":
                a = it["audio_paths"]
                audio_cnt += len(a)
                for ap in a:
                    uniq_audio.add(ap)

        total_vq_cnt = gold_cnt + silver_cnt + neg_cnt
        avg_vq_per_group = (total_vq_cnt / n_groups) if n_groups > 0 else 0.0

        print(
            f"[ReactMotionNet][SUMMARY] kept_groups={n_groups} | "
            f"vq_total={total_vq_cnt} (gold={gold_cnt}, silver={silver_cnt}, neg={neg_cnt}) | "
            f"vq_unique={len(uniq_vq)} (gold={len(uniq_gold)}, silver={len(uniq_silver)}, neg={len(uniq_neg)}) | "
            f"avg_vq_per_group={avg_vq_per_group:.2f}"
        )
        if self.audio_mode != "none":
            print(
                f"[ReactMotionNet][SUMMARY] audio_total={audio_cnt} | "
                f"audio_unique={len(uniq_audio)} | audio_mode={self.audio_mode}"
            )

        for i, k in enumerate(self.keys[:debug_print_k]):
            it = self.grouped[k]
            print(
                " ", i, "key=", k,
                "gold=", len(it["gold_vq_paths"]),
                "silver=", len(it["silver_vq_paths"]),
                "neg=", len(it["neg_vq_paths"]),
                "audio=", len(it["audio_paths"]) if self.audio_mode != "none" else 0,
                "w=", it.get("group_w", 1.0),
                "sayings_head=", it["transcription"][:60],
            )

        if self.missing_vq_log_path and len(missing_vq_ids) > 0:
            all_ids = set(missing_vq_ids)
            if self.missing_vq_log_unique and os.path.isfile(self.missing_vq_log_path):
                try:
                    with open(self.missing_vq_log_path, "r", encoding="utf-8") as f:
                        for ln in f:
                            ln = ln.strip()
                            if ln:
                                all_ids.add(ln)
                except Exception:
                    pass
            with open(self.missing_vq_log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(sorted(all_ids)) + "\n")
            print(f"[MissingVQ] saved {len(all_ids)} ids -> {self.missing_vq_log_path}")

        if len(self.keys) == 0:
            raise RuntimeError(
                "[ReactMotionNet] got 0 queries after filtering.\n"
                "See drop_reasons above. Common causes:\n"
                "  - key_by wrong (try group_id)\n"
                "  - speaker_audio_wav missing/empty\n"
                "  - audio_code_dir/wav_dir wrong\n"
                "  - VQVAE dir missing / motion_id mismatch\n"
            )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        key = self.keys[idx]
        it = self.grouped[key]

        out: Dict[str, Any] = dict(
            key=key,
            transcription=it["transcription"],
            emotion=it["emotion"],
            gold_vq_paths=it["gold_vq_paths"],
            silver_vq_paths=it["silver_vq_paths"],
            neg_vq_paths=it["neg_vq_paths"],
            group_w=float(it.get("group_w", 1.0)),  # ✅ NEW
        )

        if self.audio_mode == "code":
            code_path = random.choice(it["audio_paths"])
            codes = load_audio_tokens_any(code_path)
            out["audio_text"] = _format_audio_tokens(codes, level=self.audio_token_level)
            out["audio_code_path"] = code_path

        if self.audio_mode == "wav":
            out["wav_path"] = random.choice(it["audio_paths"])

        return out
