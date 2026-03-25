#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval/eval_reactmotion_with_judge.py

Aligned with the scorer (JudgeNetwork) from train/train_scorer.py, reduces NaN/Inf
Strict-L2 missing-modality injection (consistent with eval_unified_scorer_strictL2):
    - no text  -> text_input_ids all pad, attention_mask all 0
    - no audio -> audio_codes all pad_id, audio_pad_mask all True
    - no emo   -> emotion_ids = <unk>
Motion all-pad / empty sequence fix: forces at least 1 token to avoid AttentionPooling softmax all-mask -> NaN
Supports cond_head: fused/text/audio/emo
Output:
  1) scores.jsonl / scores.csv (per generated sample scores)
  2) best_gen_per_group.csv (best gen per group)
  3) group_metrics.csv (gen vs {gold,silver,neg} win / gen@k / ndcg@k etc.)
Win unified as mean(gen) > mean(ref): mean(gen)>mean(gold/silver/neg)

[IMPORTANT]
- gen_dump formats vary widely; this script handles multiple field name variants:
    At minimum must be able to extract:
      - group key (group_id or sayings+emotion)
      - mode (a/t/t+a/... can be absent, defaults to args.fixed_mode or "t")
      - motion token (motion_codes(list/int) or motion_npy_path / vq_path)
    Condition info:
      - sayings (text), emotion (emo), audio_code_path (npz/npy) are optional
- If your gen_dump field names differ: modify the mapping in `parse_gen_item()`.

Usage:
python eval/eval_gen_dump_using_scorer.py \
  --ckpt /path/to/best.pt \
  --gen_dump /path/to/gen_dump.jsonl \
  --pairs_csv /path/to/pairs.csv \
  --dataset_dir /path/to/dataset \
  --audio_code_dir /path/to/dataset/audio_codes \
  --out_dir /path/to/out \
  --cond_head fused \
  --fixed_mode "" \
  --k_gold 3 --k_silver 2 --k_neg 5
"""

import os, re, json, math, argparse, hashlib
from os.path import join as pjoin
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import T5Tokenizer

# ===== import EXACT training definitions (aligned) =====
from reactmotion.models.judge_network import (
    JudgeNetwork,
    DEFAULT_AUDIO_VOCAB, DEFAULT_AUDIO_PAD, DEFAULT_MOTION_VOCAB,
    MODES_FULL, MODE2ID,
    canon_label, normalize_text, clean_audio_stem,
    read_split_csv, index_vq_dir, pick_code_from_stem,
    load_audio_codes_any, normalize_audio_codes, load_motion_codes,
    CondBatch, move_cb_to,
    group_infonce_loss,  # reuse for group metrics
)

# -------------------------
# utils
# -------------------------

def seed_everything(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_1token_motion(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m).reshape(-1).astype(np.int64)
    if m.size == 0:
        return np.array([0], dtype=np.int64)
    return m

def motion_hash(m: np.ndarray) -> str:
    m = np.asarray(m, dtype=np.int64).reshape(-1)
    h = hashlib.md5(m.tobytes()).hexdigest()
    return h

def nan_guard(name: str, x: torch.Tensor) -> torch.Tensor:
    if torch.isnan(x).any() or torch.isinf(x).any():
        nan_rate = torch.isnan(x).float().mean().item()
        inf_rate = torch.isinf(x).float().mean().item()
        print(f"[ERR] {name} has nan/inf: nan_rate={nan_rate:.6f} inf_rate={inf_rate:.6f}")
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x

def safe_get(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def normalize_mode(m: str) -> str:
    m = (m or "").strip().lower()
    m = m.replace(" ", "")
    m = m.replace("tp", "t+p")  # just in case
    # allow "t+a+e" etc
    if m in MODES_FULL:
        return m
    # common aliases
    aliases = {
        "ta": "t+a",
        "tae": "t+a+e",
        "te": "t+e",
        "ae": "a+e",
    }
    if m in aliases:
        return aliases[m]
    # if someone uses "t+a+p" nonsense, just fallback to "t"
    return ""

def build_group_key(item: Dict[str, Any], key_by: str) -> str:
    if key_by == "group_id":
        gid = safe_get(item, ["group_id", "key", "gid", "groupKey", "group"], "")
        return str(gid)

    # sayings_emotion
    s = normalize_text(safe_get(item, ["sayings", "text", "utterance", "prompt"], ""))
    e = str(safe_get(item, ["emotion", "emo"], "") or "").strip().lower()
    return f"{s}__{e}"


# -------------------------
# gen_dump parsing
# -------------------------

def parse_gen_item(raw: Dict[str, Any], args) -> Dict[str, Any]:
    """
    Map arbitrary gen_dump json -> internal fields
    Required outputs:
      group_key, mode, sayings, emotion, audio_code_path, motion_codes (np.int64 1D)
    """
    # group key
    group_key = build_group_key(raw, args.key_by)
    if not group_key:
        # last resort: use index later; but we prefer not
        group_key = str(safe_get(raw, ["id", "idx", "sample_id"], ""))

    # mode
    mode = normalize_mode(safe_get(raw, ["mode", "cond_mode", "condition_mode"], ""))
    if not mode:
        mode = (args.fixed_mode or "").strip()
    if not mode:
        mode = "t"  # default fallback

    # condition
    sayings = normalize_text(safe_get(raw, ["sayings", "text", "utterance", "prompt"], ""))
    emotion = str(safe_get(raw, ["emotion", "emo"], "") or "").strip()

    # audio code path (optional)
    audio_code_path = str(safe_get(raw, ["audio_code_path", "audio_codes_path", "audio_path"], "") or "").strip()


    # backfill from pairs_csv if missing
    gid = str(safe_get(raw, ["group_id", "key"], "") or "").strip()
    m = getattr(args, "_gid2cond", {}).get(gid, None)
    if m is not None:
        if not sayings:
            sayings = m["sayings"]
        if not emotion:
            emotion = m["emotion"]
        if not audio_code_path:
            st = m["audio_stem"]
            p = pick_code_from_stem(args.audio_code_dir, st)
            audio_code_path = p or ""


    # motion codes:
    # 1) direct list: "motion_codes" or "codes"
    mc = safe_get(raw, ["motion_codes", "codes", "motion"], None)
    motion_codes = None
    if mc is not None and isinstance(mc, (list, tuple)):
        motion_codes = np.asarray(mc, dtype=np.int64).reshape(-1)
    # 2) npy path: "motion_code_path" or "vq_path" or "motion_npy"
    if motion_codes is None:
        mp = str(safe_get(raw, ["motion_codes_npy", "motion_code_path", "vq_path", "motion_npy", "motion_path"], "") or "").strip()

        if mp and os.path.exists(mp):
            motion_codes = load_motion_codes(mp, codebook_size=args.motion_vocab)  # clip inside
    if motion_codes is None:
        # if missing, set empty -> later repaired
        motion_codes = np.asarray([], dtype=np.int64)

    motion_codes = np.clip(motion_codes, 0, args.motion_vocab - 1).astype(np.int64)
    motion_codes = motion_codes[: args.max_motion_len]
    motion_codes = ensure_1token_motion(motion_codes)

    return dict(
        group_key=group_key,
        mode=mode,
        sayings=sayings,
        emotion=emotion,
        audio_code_path=audio_code_path,
        motion_codes=motion_codes,
        # keep original for debugging
        raw=raw,
    )

# -------------------------
# Dataset for gen samples (single-motion scoring)
# -------------------------

class GenDumpDataset(Dataset):
    def __init__(self, gen_dump_path: str, args):
        self.args = args
        self.items: List[Dict[str, Any]] = []
        assert os.path.isfile(gen_dump_path), f"Missing gen_dump: {gen_dump_path}"
        with open(gen_dump_path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)

                # group-level
                g_mode = normalize_mode(obj.get("cond_mode", "")) or (args.fixed_mode or "t")
                g_key  = str(obj.get("key", "") or obj.get("group_id", "") or "").strip()  # group_id
                if not g_key:
                    g_key = str(obj.get("group_hash", "")).strip()

                items = obj.get("items", [])
                if not isinstance(items, list) or len(items) == 0:
                    continue

                for j, it0 in enumerate(items):
                    raw = {}
                    raw.update(obj)   # include group fields
                    raw.update(it0)   # override with item fields
                    raw["mode"] = g_mode
                    raw["group_id"] = g_key
                    raw["idx_in_group"] = j

                    it = parse_gen_item(raw, args)
                    it["idx"] = len(self.items)
                    self.items.append(it)

        if len(self.items) == 0:
            raise RuntimeError("0 items parsed from gen_dump.")
        print(f"[GenDumpDataset] items={len(self.items)} from {gen_dump_path}")

    def __len__(self): return len(self.items)
    def __getitem__(self, i: int) -> Dict[str, Any]:
        return self.items[i]

# -------------------------
# Strict-L2 collator for gen samples
# -------------------------

class GenDumpCollatorStrictL2:
    """
    Build CondBatch + motion batch for single-motion scoring.
    Each batch item has exactly ONE motion sequence.

    Strict-L2:
      - no text -> pad + attn=0
      - no audio -> all pad + all padmask True
      - no emo -> <unk>
    """
    def __init__(
        self,
        tok: T5Tokenizer,
        emo2id: Dict[str, int],
        max_text_len: int,
        max_audio_len: int,
        max_motion_len: int,
        audio_codebooks: int,
        audio_pad_id: int,
        motion_vocab: int,
        fixed_mode: str = "",
        disable_text: bool = False,
        disable_audio: bool = False,
        disable_emo: bool = False,
        strict_l2: bool = True,
    ):
        self.tok = tok
        self.emo2id = emo2id
        self.max_text_len = int(max_text_len)
        self.max_audio_len = int(max_audio_len)
        self.max_motion_len = int(max_motion_len)
        self.audio_codebooks = int(audio_codebooks)
        self.audio_pad_id = int(audio_pad_id)
        self.motion_vocab = int(motion_vocab)

        self.fixed_mode = (fixed_mode or "").strip()
        self.disable_text = bool(disable_text)
        self.disable_audio = bool(disable_audio)
        self.disable_emo = bool(disable_emo)
        self.strict_l2 = bool(strict_l2)

        if self.disable_text and self.disable_audio and self.disable_emo:
            raise ValueError("All modalities disabled.")

    def _emo_id(self, emo: str) -> int:
        s = (emo or "").strip().lower()
        return int(self.emo2id.get(s, self.emo2id.get("<unk>", 0)))

    @staticmethod
    def _repair_only_e(has_t, has_a, has_e):
        only_e = has_e & (~has_t) & (~has_a)
        if only_e.any():
            has_t = has_t | only_e
        none = (~has_t) & (~has_a) & (~has_e)
        if none.any():
            has_t = has_t | none
        return has_t, has_a, has_e

    def __call__(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        B = len(items)

        modes = []
        for it in items:
            m = self.fixed_mode if self.fixed_mode else (it.get("mode", "t") or "t")
            m = normalize_mode(m) or "t"
            modes.append(m)

        has_t = torch.tensor([("t" in m) for m in modes], dtype=torch.bool)
        has_a = torch.tensor([("a" in m) for m in modes], dtype=torch.bool)
        has_e = torch.tensor([m.endswith("+e") for m in modes], dtype=torch.bool)
        mode_ids = torch.tensor([MODE2ID[m] for m in modes], dtype=torch.long)

        # global disable
        if self.disable_text: has_t[:] = False
        if self.disable_audio: has_a[:] = False
        if self.disable_emo: has_e[:] = False
        has_t, has_a, has_e = self._repair_only_e(has_t, has_a, has_e)

        # text
        texts = [it.get("sayings", "") or "" for it in items]
        enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_text_len, return_tensors="pt")
        text_input_ids = enc["input_ids"]
        text_attn_mask = enc["attention_mask"]

        # emo
        emotion_ids = torch.tensor([self._emo_id(it.get("emotion", "")) for it in items], dtype=torch.long)

        # audio
        audio_list, audio_mask_list = [], []
        for it in items:
            p = str(it.get("audio_code_path", "") or "").strip()
            if p and os.path.exists(p):
                a = normalize_audio_codes(load_audio_codes_any(p), codebooks=self.audio_codebooks)
                a = a[: self.max_audio_len]
                T = a.shape[0]
                pad_T = self.max_audio_len - T
                if pad_T > 0:
                    pad = np.full((pad_T, self.audio_codebooks), self.audio_pad_id, dtype=np.int64)
                    a = np.concatenate([a, pad], axis=0)
                m = np.zeros((self.max_audio_len,), dtype=np.bool_)
                if T < self.max_audio_len: m[T:] = True
            else:
                a = np.full((self.max_audio_len, self.audio_codebooks), self.audio_pad_id, dtype=np.int64)
                m = np.ones((self.max_audio_len,), dtype=np.bool_)
            audio_list.append(torch.from_numpy(a))
            audio_mask_list.append(torch.from_numpy(m))
        audio_codes = torch.stack(audio_list, dim=0)             # [B,Ta,K]
        audio_pad_mask = torch.stack(audio_mask_list, dim=0)     # [B,Ta]

        # Strict-L2 injection (MATCH strict eval)
        if self.strict_l2:
            no_t = ~has_t
            if no_t.any():
                pad_id = int(self.tok.pad_token_id)
                text_input_ids[no_t] = pad_id
                text_attn_mask[no_t] = 0

            no_a = ~has_a
            if no_a.any():
                audio_codes[no_a] = int(self.audio_pad_id)
                audio_pad_mask[no_a] = True

            no_e = ~has_e
            if no_e.any():
                unk = int(self.emo2id.get("<unk>", 0))
                emotion_ids[no_e] = unk

        cb = CondBatch(
            has_t=has_t, has_a=has_a, has_e=has_e, mode_ids=mode_ids,
            text_input_ids=text_input_ids, text_attn_mask=text_attn_mask,
            emotion_ids=emotion_ids,
            audio_codes=audio_codes, audio_pad_mask=audio_pad_mask,
            debug_modes=modes,
        )

        # motion: each item has one sequence -> pack as [B,T]
        Tm = self.max_motion_len
        motion_codes = torch.zeros((B, Tm), dtype=torch.long)
        motion_pad = torch.ones((B, Tm), dtype=torch.bool)

        for b, it in enumerate(items):
            m = np.asarray(it["motion_codes"], dtype=np.int64).reshape(-1)
            m = np.clip(m, 0, self.motion_vocab - 1).astype(np.int64)
            m = m[:Tm]
            m = ensure_1token_motion(m)
            T = int(len(m))
            motion_codes[b, :T] = torch.from_numpy(m)
            motion_pad[b, :T] = False

        # motion all-mask repair (extra safety)
        all_mask = motion_pad.all(dim=1)
        if all_mask.any():
            idx = all_mask.nonzero(as_tuple=False).view(-1)
            motion_pad[idx, 0] = False
            motion_codes[idx, 0] = 0

        meta = {
            "group_key": [it["group_key"] for it in items],
            "mode": [it["mode"] for it in items],
            "idx": [it.get("idx", -1) for it in items],
            "sayings": [it.get("sayings", "") for it in items],
            "emotion": [it.get("emotion", "") for it in items],
            "audio_code_path": [it.get("audio_code_path", "") for it in items],
        }
        return {"cb": cb, "motion_codes": motion_codes, "motion_pad": motion_pad, "meta": meta}

# -------------------------
# score gen_dump
# -------------------------

@torch.no_grad()
def score_gen_dump(
    model: JudgeNetwork,
    loader: DataLoader,
    device: torch.device,
    cond_head: str = "fused",
) -> List[Dict[str, Any]]:
    model.eval()
    rows: List[Dict[str, Any]] = []

    for batch in tqdm(loader, desc="score_gen", leave=False):
        cb = move_cb_to(batch["cb"], device)
        mc = batch["motion_codes"].to(device, non_blocking=True)
        mp = batch["motion_pad"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.bfloat16):
            zs = model.encode_condition(cb)
            if cond_head == "text":
                zc = zs["z_t"]
            elif cond_head == "audio":
                zc = zs["z_a"]
            elif cond_head == "emo":
                zc = zs["z_e"]
            else:
                zc = zs["z_f"]

            zm = model.encode_motion(mc, mp)

            zc = nan_guard(f"zc({cond_head})", zc)
            zm = nan_guard("zm", zm)
            scale = nan_guard("scale", model.scale())

            score = (scale.float() * (zc.float() * zm.float()).sum(dim=1))  # [B]

        meta = batch["meta"]
        for i in range(score.numel()):
            # motion meta: hash/len from CPU motion_codes
            m_np = batch["motion_codes"][i].cpu().numpy()
            # trim pad (pad positions where motion_pad True)
            pad_np = batch["motion_pad"][i].cpu().numpy().astype(bool)
            m_trim = m_np[~pad_np]
            m_trim = ensure_1token_motion(m_trim)

            rows.append(dict(
                idx=int(meta["idx"][i]),
                group_key=str(meta["group_key"][i]),
                mode=str(meta["mode"][i]),
                cond_head=str(cond_head),
                score=float(score[i].item()),
                motion_len=int(len(m_trim)),
                motion_hash=motion_hash(m_trim),
                sayings=str(meta["sayings"][i]),
                emotion=str(meta["emotion"][i]),
                audio_code_path=str(meta["audio_code_path"][i]),
            ))

    return rows

# -------------------------
# build reference candidates per group from pairs_csv
# -------------------------

def build_ref_groups(
    split: str,
    pairs_csv: str,
    dataset_dir: str,
    audio_code_dir: str,
    key_by: str,
    k_gold: int,
    k_silver: int,
    k_neg: int,
    require_audio: bool,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Similar to JudgeGroupDataset groups, but returns a compact list:
      {
        group_key, group_w, sayings, emotion, audio_code_path(optional),
        cand_paths(list), cand_labels(np.int64)
      }
    """
    import random
    df = read_split_csv(pairs_csv, split).copy()
    need_cols = ["tier_label", "speaker_transcript", "speaker_emotion", "motion_id", "speaker_audio_wav"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in csv: {missing}. Found: {list(df.columns)}")

    df["label_c"] = df["tier_label"].apply(canon_label)
    df["sayings"] = df["speaker_transcript"].map(normalize_text)
    df["emotion"] = df["speaker_emotion"].astype(str).fillna("").str.strip()
    df["audio_stem"] = df["speaker_audio_wav"].map(clean_audio_stem)

    vq_dir = pjoin(dataset_dir, "HumanML3D", "VQVAE")
    vq_by_mid = index_vq_dir(vq_dir)

    def motion_id_from_raw(raw_file_name: Any) -> str:
        x = "" if pd.isna(raw_file_name) else str(raw_file_name).strip()
        mid = x.split("_", 1)[0] if x else ""
        return str(mid).zfill(6) if mid else ""

    df["motion_id"] = df["motion_id"].apply(motion_id_from_raw)

    gb = df.groupby(["group_id"], dropna=False) if key_by == "group_id" and "group_id" in df.columns \
        else df.groupby(["sayings", "emotion"], dropna=False)

    groups: List[Dict[str, Any]] = []
    drops = {"no_gold": 0, "no_neg": 0, "no_audio": 0, "vq_missing": 0}

    for key, g in gb:
        sayings = str(g["sayings"].iloc[0])
        emotion = str(g["emotion"].iloc[0])
        if key_by == "group_id" and "group_id" in g.columns:
            group_key = str(g["group_id"].iloc[0])
        else:
            group_key = f"{sayings}__{emotion.lower()}"

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

        def mids_to_paths(mids: List[str]) -> List[str]:
            out = []
            for mid in mids:
                p = vq_by_mid.get(mid)
                if p is None:
                    drops["vq_missing"] += 1
                    continue
                out.append(p)
            out = list(dict.fromkeys(out))
            return out

        gold = mids_to_paths(g[g["label_c"] == "gold"]["motion_id"].tolist())
        silver = mids_to_paths(g[g["label_c"] == "silver"]["motion_id"].tolist())
        neg = mids_to_paths(g[g["label_c"] == "neg"]["motion_id"].tolist())

        if len(gold) == 0:
            drops["no_gold"] += 1
            continue
        if len(neg) == 0:
            drops["no_neg"] += 1
            continue

        rng = random.Random(seed ^ (hash(group_key) & 0xFFFFFFFF))

        def sample_k(arr: List[str], k: int) -> List[str]:
            if len(arr) == 0 or k <= 0: return []
            if len(arr) >= k: return rng.sample(arr, k)
            return [rng.choice(arr) for _ in range(k)]

        gold_s = sample_k(gold, k_gold)
        silv_s = sample_k(silver, k_silver) if len(silver) else []
        neg_s  = sample_k(neg, k_neg)

        cand_paths = gold_s + silv_s + neg_s
        cand_labels = np.array([2]*len(gold_s) + [1]*len(silv_s) + [0]*len(neg_s), dtype=np.int64)

        audio_code_path = rng.choice(audio_paths) if len(audio_paths) else ""

        groups.append(dict(
            group_key=group_key,
            sayings=sayings,
            emotion=emotion,
            audio_code_path=audio_code_path,
            cand_paths=cand_paths,
            cand_labels=cand_labels,
        ))

    print(f"[RefGroups] split={split} groups={len(groups)} drops={drops}")
    return groups

# -------------------------
# group metrics: best gen vs refs
# -------------------------

@torch.no_grad()
def compute_group_metrics(
    model: JudgeNetwork,
    device: torch.device,
    tok: T5Tokenizer,
    emo2id: Dict[str, int],
    best_gen_df: pd.DataFrame,
    ref_groups: List[Dict[str, Any]],
    args,
    all_gens_by_group: Optional[Dict[Tuple[str, str], List[np.ndarray]]] = None,
) -> pd.DataFrame:
    """
    For each ref group:
      candidates = [gen_motions] + gold/silver/neg from csv sampling
      score by same condition embedding (mode fixed per row or args.fixed_mode)
      metrics:
        - win_mode=mean: win(mean(gen) vs mean(ref)) [default]
        - win_mode=best: win(best_gen vs mean(ref))
        - win_mode=worst: win(worst_gen vs mean(ref))
      gen@3, ndcg@5 (gain g=2,s=1,n=0)
    """
    # map group_key -> best gen motion_codes (load from gen_dump scores csv/jsonl not available here)
    # We'll store best_gen motion in best_gen_df as "best_motion_codes_path"?? Not.
    # So: we recompute by reading from a saved "scores.jsonl"? Too heavy.
    # Practical approach:
    # - during score_gen_dump we already wrote motion_hash/len but not codes.
    # - To compute strict group metrics, we need actual best gen codes.
    # Therefore this script saves best gen codes to a small npy file per row.
    #
    # Implementation: expect best_gen_df has "best_motion_npy" pointing to npy saved during scoring.
    if "best_motion_npy" not in best_gen_df.columns:
        raise RuntimeError("best_gen_df missing 'best_motion_npy'. This script writes it; if you modified, keep it.")

    # index ref groups by group_key
    rg_map = {g["group_key"]: g for g in ref_groups}

    rows = []

    for _, r in tqdm(best_gen_df.iterrows(), total=len(best_gen_df), desc="group_metrics", leave=False):
        gk = str(r["group_key"])
        if gk not in rg_map:
            continue
        g = rg_map[gk]

        mode = normalize_mode(str(r.get("mode", ""))) or (args.fixed_mode or "t")
        if mode not in MODES_FULL:
            mode = "t"

        # build a single-sample CondBatch for this group
        has_t = torch.tensor([("t" in mode)], dtype=torch.bool)
        has_a = torch.tensor([("a" in mode)], dtype=torch.bool)
        has_e = torch.tensor([mode.endswith("+e")], dtype=torch.bool)
        if args.disable_text: has_t[:] = False
        if args.disable_audio: has_a[:] = False
        if args.disable_emo: has_e[:] = False
        # repair only-e/none
        only_e = has_e & (~has_t) & (~has_a)
        if only_e.any(): has_t = has_t | only_e
        none = (~has_t) & (~has_a) & (~has_e)
        if none.any(): has_t = has_t | none

        # text
        enc = tok([g["sayings"]], padding=True, truncation=True, max_length=args.max_text_len, return_tensors="pt")
        text_input_ids = enc["input_ids"]
        text_attn_mask = enc["attention_mask"]

        # emo
        emo = str(g["emotion"] or "").strip().lower()
        emo_id = int(emo2id.get(emo, emo2id.get("<unk>", 0)))
        emotion_ids = torch.tensor([emo_id], dtype=torch.long)

        # audio
        p = str(g.get("audio_code_path", "") or "").strip()
        if p and os.path.exists(p):
            a = normalize_audio_codes(load_audio_codes_any(p), codebooks=8)[: args.max_audio_len]
            T = a.shape[0]
            pad_T = args.max_audio_len - T
            if pad_T > 0:
                pad = np.full((pad_T, 8), DEFAULT_AUDIO_PAD, dtype=np.int64)
                a = np.concatenate([a, pad], axis=0)
            am = np.zeros((args.max_audio_len,), dtype=np.bool_)
            if T < args.max_audio_len: am[T:] = True
        else:
            a = np.full((args.max_audio_len, 8), DEFAULT_AUDIO_PAD, dtype=np.int64)
            am = np.ones((args.max_audio_len,), dtype=np.bool_)

        audio_codes = torch.from_numpy(a).unsqueeze(0)           # [1,Ta,K]
        audio_pad_mask = torch.from_numpy(am).unsqueeze(0)       # [1,Ta]

        # Strict-L2 injection
        if True:
            no_t = ~has_t
            if no_t.any():
                pad_id = int(tok.pad_token_id)
                text_input_ids[no_t] = pad_id
                text_attn_mask[no_t] = 0
            no_a = ~has_a
            if no_a.any():
                audio_codes[no_a] = int(DEFAULT_AUDIO_PAD)
                audio_pad_mask[no_a] = True
            no_e = ~has_e
            if no_e.any():
                unk = int(emo2id.get("<unk>", 0))
                emotion_ids[no_e] = unk

        cb = CondBatch(
            has_t=has_t, has_a=has_a, has_e=has_e,
            mode_ids=torch.tensor([MODE2ID[mode]], dtype=torch.long),
            text_input_ids=text_input_ids,
            text_attn_mask=text_attn_mask,
            emotion_ids=emotion_ids,
            audio_codes=audio_codes,
            audio_pad_mask=audio_pad_mask,
            debug_modes=[mode],
        )
        cb = move_cb_to(cb, device)

        # candidates: gen(s) + refs
        # When win_mode in (worst, mean) and all_gens available: use all gens
        win_mode_val = getattr(args, "win_mode", "mean")
        use_all_gens = (
            win_mode_val in ("worst", "mean")
            and all_gens_by_group is not None
            and (mode, gk) in all_gens_by_group
            and len(all_gens_by_group[(mode, gk)]) >= 1
        )
        if use_all_gens:
            gen_codes_list = all_gens_by_group[(mode, gk)]
            ref_paths = list(g["cand_paths"])
            K = len(gen_codes_list)
        else:
            best_path = str(r["best_motion_npy"])
            best_codes = load_motion_codes(best_path, codebook_size=args.motion_vocab)[: args.max_motion_len]
            gen_codes_list = [ensure_1token_motion(best_codes)]
            ref_paths = list(g["cand_paths"])
            K = 1

        cand_labels = np.array([-1] * K + list(g["cand_labels"]), dtype=np.int64)  # -1 for gen
        gain_labels = np.array([0] * K + list(g["cand_labels"]), dtype=np.int64)

        # pack motions [C,T]
        C = K + len(ref_paths)
        Tm = args.max_motion_len
        mc = torch.zeros((C, Tm), dtype=torch.long)
        mp = torch.ones((C, Tm), dtype=torch.bool)

        for j in range(K):
            m = np.asarray(gen_codes_list[j], dtype=np.int64).reshape(-1)[:Tm]
            m = ensure_1token_motion(m)
            T = len(m)
            mc[j, :T] = torch.from_numpy(m)
            mp[j, :T] = False
        for j, pth in enumerate(ref_paths):
            m = load_motion_codes(pth, codebook_size=args.motion_vocab)[:Tm]
            m = ensure_1token_motion(m)
            T = len(m)
            mc[K + j, :T] = torch.from_numpy(m)
            mp[K + j, :T] = False

        # all-mask repair
        all_mask = mp.all(dim=1)
        if all_mask.any():
            idx = all_mask.nonzero(as_tuple=False).view(-1)
            mp[idx, 0] = False
            mc[idx, 0] = 0

        mc = mc.to(device, non_blocking=True)
        mp = mp.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.bfloat16):
            zs = model.encode_condition(cb)
            if args.cond_head == "text":
                zc = zs["z_t"]
            elif args.cond_head == "audio":
                zc = zs["z_a"]
            elif args.cond_head == "emo":
                zc = zs["z_e"]
            else:
                zc = zs["z_f"]

            zm = model.encode_motion(mc, mp)  # [C,D]

            zc = nan_guard("zc_group", zc)
            zm = nan_guard("zm_group", zm)
            scale = nan_guard("scale_group", model.scale())

            logits = (scale.float() * (zc.float() @ zm.float().t())).view(1, C)  # [1,C]
            logits = logits.float()

        # metrics
        # indices: 0..K-1 are gens, K..C-1 are refs
        gen_scores = logits[0, :K].detach().float().cpu().numpy().astype(np.float64)
        ref_scores = logits[0, K:].detach().float().cpu().numpy().astype(np.float64)
        ref_lab = gain_labels[K:]

        best_gen_score = float(np.max(gen_scores))
        worst_gen_score = float(np.min(gen_scores)) if K > 1 else best_gen_score
        mean_gen_score = float(np.mean(gen_scores))

        gold_scores = ref_scores[ref_lab == 2]
        silv_scores = ref_scores[ref_lab == 1]
        neg_scores  = ref_scores[ref_lab == 0]

        # win_mode: mean -> mean(gen) vs mean(ref); best/worst
        win_mode = getattr(args, "win_mode", "mean")
        if win_mode == "mean":
            gen_for_win = mean_gen_score
        elif win_mode == "worst":
            gen_for_win = worst_gen_score
        else:
            gen_for_win = best_gen_score

        def win_mean_vs_mean(gen_s: float, arr: np.ndarray) -> float:
            if arr.size == 0:
                return float("nan")
            mean_other = float(np.mean(arr))
            if gen_s > mean_other:
                return 1.0
            if gen_s == mean_other:
                return 0.5
            return 0.0

        win_g = win_mean_vs_mean(gen_for_win, gold_scores)
        win_s = win_mean_vs_mean(gen_for_win, silv_scores)
        win_n = win_mean_vs_mean(gen_for_win, neg_scores)

        # gen@3: best gen in top3
        order = np.argsort(-logits.detach().cpu().numpy()[0])  # desc
        top3 = order[: min(3, C)]
        best_gen_idx = int(np.argmax(gen_scores))
        gen_at3 = float((best_gen_idx in top3))

        # ndcg@5 gain: gold=2 silver=1 neg/gen=0
        kk = min(5, C)
        gains = np.zeros((C,), dtype=np.float64)
        gains[:K] = 0.0
        gains[K:] = np.where(ref_lab == 2, 2.0, np.where(ref_lab == 1, 1.0, 0.0))
        denom = np.log2(np.arange(kk) + 2.0)
        dcg = float((gains[order[:kk]] / denom).sum())
        ideal = np.sort(gains)[::-1][:kk]
        idcg = float((ideal / denom[:ideal.size]).sum()) if ideal.size else 0.0
        ndcg5 = float(dcg / idcg) if idcg > 0 else 0.0

        rows.append(dict(
            group_key=gk,
            mode=mode,
            best_gen_score=best_gen_score,
            mean_gen_score=mean_gen_score,
            worst_gen_score=worst_gen_score if K > 1 else float("nan"),
            win_gen_vs_gold=win_g,
            win_gen_vs_silver=win_s,
            win_gen_vs_neg=win_n,
            gen_at3=gen_at3,
            ndcg5=ndcg5,
            best_motion_hash=str(r.get("best_motion_hash", "")),
            best_motion_len=int(r.get("best_motion_len", -1)),
        ))

    out = pd.DataFrame(rows)
    return out

# -------------------------
# main
# -------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--gen_dump", type=str, required=True)

    ap.add_argument("--pairs_csv", type=str, required=True)
    ap.add_argument("--dataset_dir", type=str, required=True)
    ap.add_argument("--audio_code_dir", type=str, required=True)

    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--key_by", type=str, default="group_id", choices=["group_id", "sayings_emotion"])
    ap.add_argument("--eval_split", type=str, default="test", choices=["val", "test"])

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--fixed_mode", type=str, default="", help="if set, overwrite gen item mode.")
    ap.add_argument("--disable_text", action="store_true")
    ap.add_argument("--disable_audio", action="store_true")
    ap.add_argument("--disable_emo", action="store_true")

    ap.add_argument("--cond_head", type=str, default="fused", choices=["fused", "text", "audio", "emo"])

    # fallback model args (may be overwritten by ckpt args)
    ap.add_argument("--t5_encoder", type=str, default="google-t5/t5-base")
    ap.add_argument("--max_text_len", type=int, default=128)
    ap.add_argument("--max_audio_len", type=int, default=512)
    ap.add_argument("--max_motion_len", type=int, default=196)

    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--output_dim", type=int, default=512)
    ap.add_argument("--nhead", type=int, default=12)
    ap.add_argument("--enc_layers", type=int, default=6)
    ap.add_argument("--ff_dim", type=int, default=3072)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=0.07)

    ap.add_argument("--motion_vocab", type=int, default=DEFAULT_MOTION_VOCAB)

    # ref sampling
    ap.add_argument("--k_gold", type=int, default=3)
    ap.add_argument("--k_silver", type=int, default=2)
    ap.add_argument("--k_neg", type=int, default=5)
    ap.add_argument("--win_mode", type=str, default="mean", choices=["mean", "best", "worst"],
                    help="mean: mean(gen) vs mean(ref). best: best_gen vs mean(ref). worst: worst_gen vs mean(ref).")
    ap.add_argument("--require_audio", action="store_true")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="")

    args = ap.parse_args()
    seed_everything(args.seed)

    df_split = read_split_csv(args.pairs_csv, args.eval_split).copy()
    df_split["group_id"] = df_split["group_id"].astype(str).str.strip()
    gid2 = df_split.groupby("group_id").head(1).set_index("group_id")
    args._gid2cond = {
        gid: {
            "sayings": normalize_text(row["speaker_transcript"]),
            "emotion": str(row["speaker_emotion"]).strip(),
            "audio_stem": clean_audio_stem(row["speaker_audio_wav"]),
        }
        for gid, row in gid2.iterrows()
    }

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    ckpt = torch.load(args.ckpt, map_location="cpu")

    # Load emo2id
    emo2id = ckpt.get("emo2id", None)
    if emo2id is None:
        df_tr = read_split_csv(args.pairs_csv, "train")
        df_va = read_split_csv(args.pairs_csv, "val")
        emos = sorted(set([str(x).strip().lower() for x in list(df_tr["speaker_emotion"]) + list(df_va["speaker_emotion"]) if str(x).strip()]))
        emo2id = {"<unk>": 0}
        for e in emos:
            if e not in emo2id:
                emo2id[e] = len(emo2id)
        print("[WARN] ckpt has no emo2id; rebuilt from train+val.")
    print("[Emotion] size =", len(emo2id))

    # Sync model-shape args from ckpt if exists
    ckpt_args = ckpt.get("args", None)
    if ckpt_args is not None:
        model_keys = {"t5_encoder","max_text_len","max_audio_len","max_motion_len","d_model","output_dim","nhead","enc_layers","ff_dim","dropout","temperature"}
        for k in model_keys:
            if k in ckpt_args and hasattr(args, k):
                setattr(args, k, ckpt_args[k])
        print("[CKPT] Loaded model-config args from checkpoint (model-only).")

    # Build aligned model
    model = JudgeNetwork(
        t5_name_or_path=args.t5_encoder,
        num_emotions=len(emo2id),
        d_model=args.d_model,
        output_dim=args.output_dim,
        nhead=args.nhead,
        enc_layers=args.enc_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        audio_vocab=DEFAULT_AUDIO_VOCAB,
        audio_pad_id=DEFAULT_AUDIO_PAD,
        audio_codebooks=8,
        max_audio_len=args.max_audio_len,
        motion_vocab=args.motion_vocab,
        max_motion_len=args.max_motion_len,
        temperature=args.temperature,
    ).to(device)

    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[WARN] Missing keys:", missing[:20], ("..." if len(missing) > 20 else ""))
    if unexpected:
        print("[WARN] Unexpected keys:", unexpected[:20], ("..." if len(unexpected) > 20 else ""))

    tok = T5Tokenizer.from_pretrained(args.t5_encoder)

    # -------- score gen dump --------
    ds = GenDumpDataset(args.gen_dump, args)
    collate = GenDumpCollatorStrictL2(
        tok=tok,
        emo2id=emo2id,
        max_text_len=args.max_text_len,
        max_audio_len=args.max_audio_len,
        max_motion_len=args.max_motion_len,
        audio_codebooks=8,
        audio_pad_id=DEFAULT_AUDIO_PAD,
        motion_vocab=args.motion_vocab,
        fixed_mode=args.fixed_mode,
        disable_text=args.disable_text,
        disable_audio=args.disable_audio,
        disable_emo=args.disable_emo,
        strict_l2=True,
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

    rows = score_gen_dump(model, loader, device, cond_head=args.cond_head)

    scores_jsonl = pjoin(args.out_dir, "scores.jsonl")
    with open(scores_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("[Write]", scores_jsonl)

    df_scores = pd.DataFrame(rows)
    scores_csv = pjoin(args.out_dir, "scores.csv")
    df_scores.to_csv(scores_csv, index=False)
    print("[Write]", scores_csv)

    # -------- pick best gen per (mode, group_key) --------
    # Save best motion codes to npy for later group-metrics:
    # We'll reconstruct by re-reading gen_dump items (fast enough) and write per best row.
    # Build idx->motion_codes mapping from dataset
    idx2codes = {it["idx"]: it["motion_codes"] for it in ds.items}

    df_scores["mode_norm"] = df_scores["mode"].map(lambda x: normalize_mode(str(x)) or "t")
    grp_cols = ["mode_norm", "group_key"]
    df_best = df_scores.sort_values("score", ascending=False).groupby(grp_cols, as_index=False).head(1).copy()
    df_best = df_best.rename(columns={"mode_norm": "mode"})

    best_dir = pjoin(args.out_dir, "best_motion_npy")
    os.makedirs(best_dir, exist_ok=True)

    best_npy_paths = []
    for _, r in df_best.iterrows():
        idx = int(r["idx"])
        m = idx2codes.get(idx, np.asarray([], dtype=np.int64))
        m = ensure_1token_motion(np.asarray(m, dtype=np.int64)[: args.max_motion_len])
        npy_path = pjoin(best_dir, f"best_idx{idx}_g{hashlib.md5(str(r['group_key']).encode()).hexdigest()[:8]}_m{r['mode']}.npy")
        np.save(npy_path, m.astype(np.int64))
        best_npy_paths.append(npy_path)

    df_best["best_motion_npy"] = best_npy_paths
    df_best = df_best.rename(columns={
        "motion_hash": "best_motion_hash",
        "motion_len": "best_motion_len",
        "score": "best_gen_score",
    })

    best_csv = pjoin(args.out_dir, "best_gen_per_group.csv")
    df_best[["group_key","mode","best_gen_score","best_motion_hash","best_motion_len","best_motion_npy"]].to_csv(best_csv, index=False)
    print("[Write]", best_csv)

    # Build all_gens_by_group for win_mode in (mean, worst)
    all_gens_by_group: Dict[Tuple[str, str], List[np.ndarray]] = {}
    for (m, gk), grp in df_scores.groupby(["mode_norm", "group_key"]):
        codes_list = []
        for idx in grp["idx"].tolist():
            c = idx2codes.get(idx, np.asarray([], dtype=np.int64))
            c = ensure_1token_motion(np.asarray(c, dtype=np.int64)[: args.max_motion_len])
            codes_list.append(c)
        if codes_list:
            all_gens_by_group[(str(m), str(gk))] = codes_list
    print(f"[Win mode] {args.win_mode} (all_gens groups={len(all_gens_by_group)})")

    # -------- group metrics vs refs (gold/silver/neg) --------
    ref_groups = build_ref_groups(
        split=args.eval_split,
        pairs_csv=args.pairs_csv,
        dataset_dir=args.dataset_dir,
        audio_code_dir=args.audio_code_dir,
        key_by=args.key_by,
        k_gold=args.k_gold,
        k_silver=args.k_silver,
        k_neg=args.k_neg,
        require_audio=bool(args.require_audio),
        seed=int(args.seed) + 123,
    )
    df_metrics = compute_group_metrics(
        model=model,
        device=device,
        tok=tok,
        emo2id=emo2id,
        best_gen_df=df_best,
        ref_groups=ref_groups,
        args=args,
        all_gens_by_group=all_gens_by_group,
    )

    out_metrics = pjoin(args.out_dir, "group_metrics.csv")
    df_metrics.to_csv(out_metrics, index=False)
    print("[Write]", out_metrics)

    # quick summary
    if len(df_metrics):
        def m(x):
            x = pd.to_numeric(x, errors="coerce")
            return float(x.mean()) if x.notna().any() else float("nan")
        print("[Summary]",
              f"win_vs_neg={m(df_metrics['win_gen_vs_neg']):.3f}",
              f"win_vs_silver={m(df_metrics['win_gen_vs_silver']):.3f}",
              f"win_vs_gold={m(df_metrics['win_gen_vs_gold']):.3f}",
              f"gen@3={m(df_metrics['gen_at3']):.3f}",
              f"ndcg5={m(df_metrics['ndcg5']):.3f}",
              sep=" | ")

    print("[Done]")

if __name__ == "__main__":
    main()
