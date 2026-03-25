#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval/eval_retrieval.py

Retrieval-based listener motion generation baseline evaluation.
For each test sample, retrieves the most similar motion from the full train set as the prediction,
then computes FID & Diversity using EvaluatorModelWrapper.

Method 1: eval_wrapper
  - Uses TextEncoderBiGRUCo (GloVe+POS features) from EvaluatorModelWrapper
    to encode speaker sayings into the same co-embedding space as the motion encoder
  - Builds train motion embedding bank
  - Cosine similarity -> top-1
  Requires: --glove_dir (default ./glove) and spacy en_core_web_sm

Method 2: scorer
  - Loads a trained JudgeNetwork
  - Supports multiple conditioning modes: t | t+e | a | a+e | t+a+e
  - Pre-encodes all train motion codes into a motion embedding bank
  - For each test sample: encode condition -> cosine sim -> top-1
  Requires: --scorer_ckpt

FID & Diversity: retrieved motion -> VQ-VAE decode -> EvaluatorModelWrapper embedding -> FID vs real

Examples:

# Method 1
python -m eval.eval_retrieval \\
  --method eval_wrapper \\
  --dataset_dir /path/to/dataset \\
  --pairs_csv ./new_data \\
  --t2m_opt ./checkpoints/t2m/Comp_v6_KLD005/opt.txt \\
  --vqvae_ckpt /path/to/motion_VQVAE/net_last.pth \\
  --glove_dir ./glove \\
  --mean_path /path/mean.npy --std_path /path/std.npy

# Method 2
python -m eval.eval_retrieval \\
  --method scorer \\
  --cond_modes t t+e a a+e t+a+e \\
  --dataset_dir /path/to/dataset \\
  --pairs_csv ./new_data \\
  --t2m_opt ./checkpoints/t2m/Comp_v6_KLD005/opt.txt \\
  --vqvae_ckpt /path/to/motion_VQVAE/net_last.pth \\
  --scorer_ckpt /path/to/checkpoints/scorer/best.pt \\
  --audio_code_dir /path/to/dataset/audio-codes \\
  --mean_path /path/mean.npy --std_path /path/std.npy
"""

import os
import sys
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import linalg
from scipy.spatial.distance import pdist
from tqdm import tqdm

from reactmotion.options.get_eval_option import get_opt
from reactmotion.models.evaluator_wrapper import EvaluatorModelWrapper
import reactmotion.models.vqvae as vqvae_module



# ─────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────

def frechet_distance(mu1: np.ndarray, s1: np.ndarray,
                     mu2: np.ndarray, s2: np.ndarray, eps: float = 1e-6) -> float:
    diff = mu1 - mu2
    cm, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(cm).all():
        cm = linalg.sqrtm(
            (s1 + np.eye(s1.shape[0]) * eps).dot(s2 + np.eye(s2.shape[0]) * eps))
    if np.iscomplexobj(cm):
        cm = cm.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(cm))


def all_pair_diversity(act: np.ndarray) -> float:
    if act.shape[0] < 2:
        return 0.0
    return float(pdist(act, metric="euclidean").mean())


# ─────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────

def normalize_text_field(x: Any) -> str:
    s = "" if pd.isna(x) else str(x).strip()
    return " ".join(s.split())


def motion_id_from_raw(raw: Any) -> str:
    x = "" if pd.isna(raw) else str(raw).strip()
    mid = x.split("_", 1)[0] if x else ""
    return str(mid).zfill(6) if mid else ""


def read_csv_split(pairs_csv: str, split: str) -> pd.DataFrame:
    if os.path.isdir(pairs_csv):
        p = os.path.join(pairs_csv, f"{split}.csv")
        return pd.read_csv(p, encoding="utf-8")
    df = pd.read_csv(pairs_csv, encoding="utf-8")
    if "split" in df.columns:
        df = df[df["split"].str.lower().str.strip() == split].copy()
    return df.reset_index(drop=True)


def unique_motion_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per unique motion_id (dedup listener motions)."""
    df = df.copy()
    df["_mid"] = df["motion_id"].apply(motion_id_from_raw)
    df = df[df["_mid"] != "000000"].copy()
    return df.drop_duplicates("_mid").reset_index(drop=True)


class AudioCodeIndex:
    """
    One-time scan of audio_code_dir, then fast stem → path lookup.
    Tries exact match first; falls back to prefix/suffix fuzzy match.
    """
    def __init__(self, audio_code_dir: str):
        self.audio_code_dir = audio_code_dir
        self._exact: Dict[str, str] = {}   # stem (no ext) → full path
        self._diagnosed = False

        if not audio_code_dir or not os.path.isdir(audio_code_dir):
            return

        for fname in os.listdir(audio_code_dir):
            if fname.endswith((".npy", ".npz")):
                key = fname
                for ext in (".npy", ".npz"):
                    if key.endswith(ext):
                        key = key[:-len(ext)]
                        break
                self._exact[key] = os.path.join(audio_code_dir, fname)

    def _clean(self, stem: str) -> str:
        s = str(stem).strip()
        if s.lower().endswith(".wav"):
            s = s[:-4]
        return s

    def find(self, raw_stem: str) -> Optional[str]:
        if not raw_stem:
            return None
        stem = self._clean(raw_stem)

        # 1. exact match
        if stem in self._exact:
            return self._exact[stem]

        # 2. the CSV stem might have a leading path component → use basename
        base = os.path.basename(stem)
        if base in self._exact:
            return self._exact[base]

        # 3. files named "{base}_codes", "codes_{base}", etc.
        for candidate in (f"{base}_codes", f"codes_{base}",
                          f"{base}_code",  f"code_{base}"):
            if candidate in self._exact:
                return self._exact[candidate]

        # 4. prefix match: file starts with stem (e.g. stem="001585_1_..." → "001585_1_..._codes")
        for key, path in self._exact.items():
            if key.startswith(base) or base.startswith(key):
                return path

        return None

    def diagnose(self, test_stems: List[str], n: int = 5) -> None:
        """Print a diagnostic showing what was searched vs what exists."""
        if self._diagnosed:
            return
        self._diagnosed = True

        disk_samples = sorted(self._exact.keys())[:n]
        search_samples = [self._clean(s) for s in test_stems if s][:n]

        print("\n[AudioDiag] ──────────────────────────────────────────")
        print(f"  audio_code_dir : {self.audio_code_dir}")
        print(f"  total files    : {len(self._exact)}")
        print(f"  disk samples   : {disk_samples}")
        print(f"  search stems   : {search_samples}")
        if disk_samples and search_samples:
            d, s = disk_samples[0], search_samples[0]
            print(f"  → Check if '{s}' matches any of '{d}' etc.")
        print("[AudioDiag] ──────────────────────────────────────────\n")


# Module-level cache so we only scan the dir once per process
_audio_index_cache: Dict[str, AudioCodeIndex] = {}


def find_audio_code_path(audio_code_dir: str, stem: str) -> Optional[str]:
    """Cached, fuzzy lookup: stem → audio code .npy/.npz path."""
    if not audio_code_dir or not stem:
        return None
    if audio_code_dir not in _audio_index_cache:
        _audio_index_cache[audio_code_dir] = AudioCodeIndex(audio_code_dir)
    return _audio_index_cache[audio_code_dir].find(stem)


# ─────────────────────────────────────────────────────────────
# VQ-VAE
# ─────────────────────────────────────────────────────────────

def build_vqvae(ckpt_path: str, device) -> torch.nn.Module:
    class _A:
        dataname = "t2m"; quantizer = "ema_reset"
        beta = 1.0; mu = 0.99

    ckpt = torch.load(ckpt_path, map_location="cpu")
    net_sd = ckpt.get("net", ckpt)
    cb_key = next((k for k in net_sd if "codebook" in k and k.endswith(".weight")), None)
    nb_code, code_dim = 512, 512
    if cb_key:
        nb_code = int(net_sd[cb_key].shape[0])
        code_dim = int(net_sd[cb_key].shape[1])

    net = vqvae_module.HumanVQVAE(
        _A(), nb_code=nb_code, code_dim=code_dim,
        output_emb_width=512, down_t=2, stride_t=2,
        width=512, depth=3, dilation_growth_rate=3,
        activation="relu", norm=None,
    )
    net.load_state_dict(net_sd, strict=True)
    return net.to(device).eval()


# ─────────────────────────────────────────────────────────────
# Real motion embeddings (reference set)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_real_embeddings(
    motion_ids: List[str],
    joint_vecs_dir: str,
    eval_wrapper: EvaluatorModelWrapper,
    device,
    max_frames: int = 196,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    batch_size: int = 64,
) -> np.ndarray:
    motions, lens, good_ids = [], [], []
    for mid in tqdm(motion_ids, desc="Loading real joint vecs"):
        p = os.path.join(joint_vecs_dir, f"{mid}.npy")
        if not os.path.isfile(p):
            continue
        arr = np.load(p, allow_pickle=False).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != 263:
            continue
        T = min(arr.shape[0], max_frames)
        arr = arr[:T].copy()
        if mean is not None and std is not None:
            arr = (arr - mean) / (std + 1e-8)
        motions.append(arr)
        lens.append(T)
        good_ids.append(mid)

    print(f"  real: {len(good_ids)} motions found")
    feats = []
    for i in range(0, len(motions), batch_size):
        bm, bl = motions[i:i + batch_size], lens[i:i + batch_size]
        max_T = max(bl)
        B = len(bm)
        x = np.zeros((B, max_T, 263), dtype=np.float32)
        for j, (m, L) in enumerate(zip(bm, bl)):
            x[j, :L] = m
        xt = torch.from_numpy(x).to(device)
        lt = torch.tensor(bl, dtype=torch.long, device=device)
        feats.append(eval_wrapper.get_motion_embeddings(xt, lt).cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0)


# ─────────────────────────────────────────────────────────────
# Retrieved motion: decode VQ codes → embed for FID + save outputs
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_retrieved_embeddings(
    retrieved_ids: List[str],          # train motion_id for each test query
    query_ids: List[str],              # test motion_id (used as output filename key)
    vqvae_dir: str,
    vae: torch.nn.Module,
    eval_wrapper: EvaluatorModelWrapper,
    device,
    max_frames: int = 196,
    batch_size: int = 64,
    # optional save
    save_joint_vecs_dir: Optional[str] = None,   # save denormalized joint vecs here
    save_motion_tokens_dir: Optional[str] = None, # save VQ code .npy here (copy)
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Decode VQ codes one-by-one (VQ-VAE decoder doesn't support true batching).
    Decoded joints are in the T2M *normalized* space (mean=0, std=1 per dim).

    If save dirs are given:
      • joint_vecs_dir/{query_id}.npy   → denormalized joints (× std + mean)
      • motion_tokens_dir/{query_id}.npy → VQ code tokens (copied from vqvae_dir)
    """
    if save_joint_vecs_dir:
        os.makedirs(save_joint_vecs_dir,   exist_ok=True)
    if save_motion_tokens_dir:
        os.makedirs(save_motion_tokens_dir, exist_ok=True)

    all_joints, all_lens = [], []
    for qid, mid in tqdm(zip(query_ids, retrieved_ids),
                         total=len(retrieved_ids), desc="Decoding retrieved VQ codes"):
        p = os.path.join(vqvae_dir, f"{mid}.npy")
        if not os.path.isfile(p):
            dummy = torch.zeros(4, 263)
            all_joints.append(dummy)
            all_lens.append(4)
            # save fallback zeros
            if save_joint_vecs_dir:
                np.save(os.path.join(save_joint_vecs_dir, f"{qid}.npy"),
                        dummy.numpy().astype(np.float32))
            if save_motion_tokens_dir:
                np.save(os.path.join(save_motion_tokens_dir, f"{qid}.npy"),
                        np.zeros(1, dtype=np.int32))
            continue

        codes = np.load(p, allow_pickle=False).astype(np.int64).reshape(-1)
        idx   = torch.from_numpy(codes).unsqueeze(0).to(device)
        pose  = vae.forward_decoder(idx)[0]          # [T, 263] normalized space
        T     = min(pose.shape[0], max_frames)
        pose_t = pose[:T].cpu()
        all_joints.append(pose_t)
        all_lens.append(T)

        # ── Save denormalized joint vectors ───────────────────
        if save_joint_vecs_dir:
            j_np = pose_t.numpy().astype(np.float32)
            if mean is not None and std is not None:
                j_np = j_np * std + mean               # denormalize → original space
            np.save(os.path.join(save_joint_vecs_dir, f"{qid}.npy"), j_np)

        # ── Save motion token codes ────────────────────────────
        if save_motion_tokens_dir:
            np.save(os.path.join(save_motion_tokens_dir, f"{qid}.npy"),
                    codes.astype(np.int32))

    # ── Embed for FID (normalized space is correct for EvaluatorModelWrapper) ──
    feats = []
    for i in range(0, len(all_joints), batch_size):
        bj, bl = all_joints[i:i + batch_size], all_lens[i:i + batch_size]
        max_T = max(bl)
        B = len(bj)
        x = torch.zeros(B, max_T, 263)
        for k, (j, L) in enumerate(zip(bj, bl)):
            x[k, :L] = j
        lt = torch.tensor(bl, dtype=torch.long, device=device)
        feats.append(eval_wrapper.get_motion_embeddings(x.to(device), lt).cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0)


# ─────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────

def cosine_retrieve(
    query_embs: np.ndarray,   # [Nq, D]  L2-normalized
    bank_embs:  np.ndarray,   # [Nb, D]  L2-normalized
    bank_ids:   List[str],
    force_unique: bool = False,
) -> List[str]:
    """
    Return top-1 motion_id for each query.

    force_unique=True: greedy assignment — each train motion is used at most once.
      Queries are processed in descending order of their maximum similarity
      (most confident first). When Nq > Nb, later queries fall back to the
      globally best remaining motion (with repetition only after exhaustion).
    """
    sim = query_embs @ bank_embs.T   # [Nq, Nb]

    if not force_unique:
        idx = np.argmax(sim, axis=1)
        return [bank_ids[i] for i in idx]

    # ── Greedy unique assignment ──────────────────────────────
    Nq, Nb = sim.shape
    assigned: List[Optional[str]] = [None] * Nq
    used: set = set()

    # Process queries most-confident first (highest max-sim gets first pick)
    best_sim_per_query = sim.max(axis=1)          # [Nq]
    query_order = np.argsort(-best_sim_per_query)  # descending

    for qi in query_order:
        ranked = np.argsort(-sim[qi])             # [Nb] desc similarity
        chosen = None
        for bi in ranked:
            if bi not in used:
                chosen = bi
                used.add(bi)
                break
        if chosen is None:
            # All train motions exhausted (Nq > Nb): pick globally best (allows repeat)
            chosen = int(np.argmax(sim[qi]))
        assigned[qi] = bank_ids[chosen]

    return assigned  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────
# Method 1: EvaluatorWrapper retrieval
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def build_motion_bank_eval_wrapper(
    motion_ids: List[str],
    joint_vecs_dir: str,
    eval_wrapper: EvaluatorModelWrapper,
    device,
    max_frames: int = 196,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    batch_size: int = 64,
) -> Tuple[np.ndarray, List[str]]:
    """Build L2-normed motion embedding bank in EvaluatorWrapper space."""
    motions, lens, valid_ids = [], [], []
    for mid in tqdm(motion_ids, desc="Building motion bank (eval_wrapper)"):
        p = os.path.join(joint_vecs_dir, f"{mid}.npy")
        if not os.path.isfile(p):
            continue
        arr = np.load(p, allow_pickle=False).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != 263:
            continue
        T = min(arr.shape[0], max_frames)
        arr = arr[:T].copy()
        if mean is not None and std is not None:
            arr = (arr - mean) / (std + 1e-8)
        motions.append(arr)
        lens.append(T)
        valid_ids.append(mid)

    feats = []
    for i in tqdm(range(0, len(motions), batch_size), desc="Embedding motion bank"):
        bm, bl = motions[i:i + batch_size], lens[i:i + batch_size]
        max_T = max(bl)
        B = len(bm)
        x = np.zeros((B, max_T, 263), dtype=np.float32)
        for j, (m, L) in enumerate(zip(bm, bl)):
            x[j, :L] = m
        xt = torch.from_numpy(x).to(device)
        lt = torch.tensor(bl, dtype=torch.long, device=device)
        emb = eval_wrapper.get_motion_embeddings(xt, lt)      # [B, 512]
        feats.append(F.normalize(emb, dim=-1).cpu().numpy().astype(np.float32))

    bank = np.concatenate(feats, axis=0)
    return bank, valid_ids


def encode_texts_glove(
    sayings: List[str],
    text_encoder,
    w_vectorizer,
    device,
    max_text_len: int = 20,
) -> np.ndarray:
    """Encode texts with T2M GloVe text encoder → L2-normed [N, 512]."""
    import spacy
    from reactmotion.utils.word_vectorizer import POS_enumerator
    nlp = spacy.load("en_core_web_sm")
    n_pos = len(POS_enumerator)

    word_embs_list, pos_ohot_list, cap_lens = [], [], []
    for sentence in sayings:
        sentence = sentence.replace("-", "")
        doc = nlp(sentence)
        words = []
        for token in doc:
            w = token.text.lower()
            if not w.isalpha():
                continue
            words.append(f"{w}/{token.pos_}")

        we_list, po_list = [], []
        for wpos in words[:max_text_len]:
            we, pe = w_vectorizer[wpos]
            we_list.append(we.astype(np.float32))
            po_list.append(pe.astype(np.float32))

        cap_lens.append(max(1, len(we_list)))
        while len(we_list) < max_text_len:
            we_list.append(np.zeros(300, dtype=np.float32))
            po_list.append(np.zeros(n_pos, dtype=np.float32))

        word_embs_list.append(np.stack(we_list[:max_text_len]))
        pos_ohot_list.append(np.stack(po_list[:max_text_len]))

    word_embs_np = np.stack(word_embs_list)   # [N, L, 300]
    pos_ohot_np  = np.stack(pos_ohot_list)    # [N, L, n_pos]
    cap_lens_np  = np.array(cap_lens, dtype=np.int64)  # [N]

    # TextEncoderBiGRUCo uses pack_padded_sequence(enforce_sorted=True) →
    # must sort by length descending, then restore original order
    sort_idx   = np.argsort(-cap_lens_np)          # descending
    unsort_idx = np.argsort(sort_idx)

    word_embs_t = torch.from_numpy(word_embs_np[sort_idx]).float().to(device)
    pos_ohot_t  = torch.from_numpy(pos_ohot_np[sort_idx]).float().to(device)
    cap_lens_t  = torch.from_numpy(cap_lens_np[sort_idx]).to(device)

    with torch.no_grad():
        text_emb = text_encoder(word_embs_t, pos_ohot_t, cap_lens_t)  # [N, 512]

    # Restore original query order
    text_emb = text_emb[unsort_idx]
    return F.normalize(text_emb, dim=-1).cpu().numpy().astype(np.float32)


@torch.no_grad()
def _encode_texts_clip(sayings: List[str], device, out_dim: int) -> np.ndarray:
    """
    CLIP ViT-B/32 text encoder fallback.
    Output is projected/padded to out_dim via zero-padding or truncation.
    """
    import clip as clip_module  # openai/clip
    model, _ = clip_module.load("ViT-B/32", device=device)
    model.eval()

    feats = []
    BATCH = 256
    for i in range(0, len(sayings), BATCH):
        batch = sayings[i:i + BATCH]
        toks  = clip_module.tokenize(batch, truncate=True).to(device)
        emb   = model.encode_text(toks).float()            # [B, 512]
        feats.append(F.normalize(emb, dim=-1).cpu().numpy().astype(np.float32))

    clip_emb = np.concatenate(feats, axis=0)               # [N, clip_dim]
    if clip_emb.shape[1] == out_dim:
        return clip_emb
    # pad/truncate to out_dim (CLIP ViT-B/32 = 512, same as T2M → usually fine)
    if clip_emb.shape[1] < out_dim:
        pad = np.zeros((clip_emb.shape[0], out_dim - clip_emb.shape[1]), dtype=np.float32)
        return np.concatenate([clip_emb, pad], axis=1)
    return clip_emb[:, :out_dim]


def _encode_test_texts(
    sayings: List[str],
    eval_wrapper: EvaluatorModelWrapper,
    glove_dir: str,
    out_dim: int,
    device,
) -> np.ndarray:
    """
    Encode test speaker sayings to query embeddings for retrieval.
    Priority:
      1. T2M GloVe text encoder (same space as motion encoder — best for retrieval)
      2. CLIP ViT-B/32 (good semantic encoding, different space but works as proxy)
      3. Random unit vectors (baseline / debug)
    """
    # ── Try T2M GloVe encoder ────────────────────────────────
    try:
        from reactmotion.utils.word_vectorizer import WordVectorizer
        w_vec = WordVectorizer(glove_dir, "our_vab")
        print(f"[TextEnc] GloVe loaded → T2M text encoder, {len(sayings)} queries ...")
        return encode_texts_glove(sayings, eval_wrapper.text_encoder, w_vec, device)
    except Exception as e:
        print(f"[TextEnc] GloVe/spacy unavailable ({e})")

    # ── Try CLIP ─────────────────────────────────────────────
    try:
        print(f"[TextEnc] falling back to CLIP ViT-B/32, {len(sayings)} queries ...")
        return _encode_texts_clip(sayings, device, out_dim)
    except Exception as e:
        print(f"[TextEnc] CLIP unavailable ({e})")

    # ── Last resort: random (gives upper-bound diversity / random baseline) ──
    print("[TextEnc] WARNING: all text encoders failed — using random embeddings.")
    print("          Install spacy + en_core_web_sm for proper text retrieval:")
    print("          python -m spacy download en_core_web_sm")
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((len(sayings), out_dim)).astype(np.float32)
    return raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8)


# ─────────────────────────────────────────────────────────────
# Method 2: Scorer-based retrieval
# ─────────────────────────────────────────────────────────────

def load_scorer(ckpt_path: str, device):
    """Load JudgeNetwork from checkpoint. Returns (model, emo2id, args_dict)."""
    from reactmotion.models.judge_network import JudgeNetwork, DEFAULT_MOTION_VOCAB, DEFAULT_AUDIO_PAD, DEFAULT_AUDIO_VOCAB
    ckpt = torch.load(ckpt_path, map_location="cpu")
    emo2id   = ckpt.get("emo2id", {"<unk>": 0})
    cfg      = ckpt.get("args", {})
    model_sd = ckpt.get("model", ckpt)

    model = JudgeNetwork(
        t5_name_or_path =cfg.get("t5_encoder", "google-t5/t5-base"),
        num_emotions    =len(emo2id),
        d_model         =cfg.get("d_model", 512),
        output_dim      =cfg.get("output_dim", 256),
        nhead           =cfg.get("nhead", 8),
        enc_layers      =cfg.get("enc_layers", 4),
        ff_dim          =cfg.get("ff_dim", 2048),
        dropout         =cfg.get("dropout", 0.1),
        audio_vocab     =cfg.get("audio_vocab", DEFAULT_AUDIO_VOCAB),
        audio_pad_id    =cfg.get("audio_pad_id", DEFAULT_AUDIO_PAD),
        audio_codebooks =cfg.get("audio_codebooks", 8),
        max_audio_len   =cfg.get("max_audio_len", 512),
        motion_vocab    =cfg.get("motion_vocab", DEFAULT_MOTION_VOCAB),
        max_motion_len  =cfg.get("max_motion_len", 196),
    )
    model.load_state_dict(model_sd, strict=False)
    model.eval()
    print(f"[Scorer] loaded from {ckpt_path}  emo_vocab={len(emo2id)}")
    return model.to(device), emo2id, cfg


@torch.no_grad()
def build_motion_bank_scorer(
    motion_ids: List[str],
    vqvae_dir: str,
    scorer,
    device,
    motion_vocab: int = 512,
    batch_size: int = 64,
) -> Tuple[np.ndarray, List[str]]:
    """Pre-encode all train motion codes → L2-normed bank [N, D]."""
    from reactmotion.models.judge_network import load_motion_codes

    codes_list, valid_ids = [], []
    for mid in motion_ids:
        p = os.path.join(vqvae_dir, f"{mid}.npy")
        if not os.path.isfile(p):
            continue
        codes = load_motion_codes(p, motion_vocab)
        codes_list.append(codes)
        valid_ids.append(mid)

    print(f"[ScorerBank] {len(valid_ids)} motions to encode")
    feats = []
    for i in tqdm(range(0, len(codes_list), batch_size), desc="Encoding motion bank (scorer)"):
        batch = codes_list[i:i + batch_size]
        max_L = max(c.shape[0] for c in batch)
        B = len(batch)
        mc = torch.zeros(B, max_L, dtype=torch.long, device=device)
        mp = torch.ones(B, max_L, dtype=torch.bool, device=device)  # True = pad
        for k, c in enumerate(batch):
            L = c.shape[0]
            mc[k, :L] = torch.from_numpy(c)
            mp[k, :L] = False
        zm = scorer.encode_motion(mc, mp)                             # [B, D]
        feats.append(F.normalize(zm, dim=-1).cpu().float().numpy())

    return np.concatenate(feats, axis=0), valid_ids


def _build_cond_batch(
    rows: List[Dict],
    cond_mode: str,
    t5_tokenizer,
    emo2id: Dict[str, int],
    audio_code_dir: str,
    audio_max_len: int,
    audio_codebooks: int,
    device,
):
    """Build CondBatch dataclass for scorer encode_condition."""
    from reactmotion.models.judge_network import (
        CondBatch, MODE2ID, MODES_FULL,
        DEFAULT_AUDIO_PAD, load_audio_codes_any, normalize_audio_codes,
    )

    B    = len(rows)
    mode = cond_mode.lower().replace(" ", "")
    has_t = "t" in mode
    has_a = "a" in mode
    has_e = "e" in mode

    # ---- Text ----
    sayings = [normalize_text_field(r.get("sayings", "")) for r in rows]
    enc = t5_tokenizer(sayings, return_tensors="pt", padding=True,
                       truncation=True, max_length=128)
    text_ids  = enc["input_ids"].to(device)
    text_mask = enc["attention_mask"].to(device)

    # ---- Emotion ----
    emo_ids = []
    for r in rows:
        e = str(r.get("emotion", "")).strip().lower()
        emo_ids.append(int(emo2id.get(e, emo2id.get("<unk>", 0))))
    emo_ids_t = torch.tensor(emo_ids, dtype=torch.long, device=device)

    # ---- Audio ----
    # AudioTokenProcessorMulti expects [B, Ta, K] — keep K codebooks as last dim
    wav_batch = np.full((B, audio_max_len, audio_codebooks), DEFAULT_AUDIO_PAD, dtype=np.int64)
    wav_mask  = np.ones((B, audio_max_len), dtype=bool)          # True = pad
    n_audio_found = 0
    stems_tried: List[str] = []
    if has_a and audio_code_dir:
        for k, r in enumerate(rows):
            stem = str(r.get("generated_wav_name", "") or r.get("audio_stem", "")).strip()
            stems_tried.append(stem)
            p = find_audio_code_path(audio_code_dir, stem)
            if p:
                ac = normalize_audio_codes(load_audio_codes_any(p),
                                           codebooks=audio_codebooks)  # [T, K]
                ac = ac[:audio_max_len]                                 # truncate time
                T  = ac.shape[0]
                pad_rows = np.full((audio_max_len - T, audio_codebooks),
                                   DEFAULT_AUDIO_PAD, dtype=np.int64)
                wav_batch[k] = np.concatenate([ac, pad_rows], axis=0)
                wav_mask[k, :T] = False
                n_audio_found += 1
    if has_a and n_audio_found == 0 and audio_code_dir:
        # Show a one-time diagnostic so user can see the mismatch
        if audio_code_dir not in _audio_index_cache:
            _audio_index_cache[audio_code_dir] = AudioCodeIndex(audio_code_dir)
        _audio_index_cache[audio_code_dir].diagnose(stems_tried)
    wav_batch = torch.from_numpy(wav_batch).to(device)   # [B, Ta, K]
    wav_mask  = torch.from_numpy(wav_mask).to(device)    # [B, Ta]

    # Disable has_a per-sample when no real audio was loaded (all-PAD → identical embeddings)
    valid_audio = ~wav_mask.all(dim=1)                    # [B] True if any real frame

    # ---- Mode ----
    mode_id = MODE2ID.get(mode, 0)
    mode_ids_t = torch.full((B,), mode_id, dtype=torch.long, device=device)

    # ---- Boolean flags ----
    has_t_t = torch.full((B,), has_t, dtype=torch.bool, device=device)
    has_a_t = torch.full((B,), has_a, dtype=torch.bool, device=device) & valid_audio
    has_e_t = torch.full((B,), has_e, dtype=torch.bool, device=device)

    return CondBatch(
        has_t=has_t_t, has_a=has_a_t, has_e=has_e_t,
        mode_ids=mode_ids_t,
        text_input_ids=text_ids, text_attn_mask=text_mask,
        emotion_ids=emo_ids_t,
        audio_codes=wav_batch, audio_pad_mask=wav_mask,
    )


HEAD_KEY = {"fused": "z_f", "text": "z_t", "audio": "z_a", "emo": "z_e"}


@torch.no_grad()
def encode_test_queries_scorer(
    test_rows: List[Dict],
    cond_mode: str,
    scorer,
    t5_tokenizer,
    emo2id: Dict[str, int],
    audio_code_dir: str,
    audio_max_len: int,
    audio_codebooks: int,
    cond_head: str,
    device,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode all test condition queries → L2-normed [N, D]."""
    head_key = HEAD_KEY.get(cond_head, "z_f")
    feats = []
    total_audio_found = 0
    needs_audio = "a" in cond_mode.lower()

    for i in tqdm(range(0, len(test_rows), batch_size),
                  desc=f"Encoding test queries ({cond_mode}/{cond_head})"):
        batch = test_rows[i:i + batch_size]
        cb = _build_cond_batch(
            batch, cond_mode, t5_tokenizer, emo2id,
            audio_code_dir, audio_max_len, audio_codebooks, device,
        )
        if needs_audio:
            total_audio_found += int(cb.has_a.sum().item())
        zs = scorer.encode_condition(cb)   # {"z_t","z_a","z_e","z_f"}
        z  = zs[head_key]                  # [B, D]
        feats.append(F.normalize(z, dim=-1).cpu().float().numpy())

    if needs_audio:
        total = len(test_rows)
        print(f"  [Audio] {total_audio_found}/{total} test audio code files found"
              + ("  ✓" if total_audio_found == total else
                 "  ⚠ some missing → those samples treated as no-audio"))

    return np.concatenate(feats, axis=0)


# ─────────────────────────────────────────────────────────────
# Eval helpers
# ─────────────────────────────────────────────────────────────

def compute_fid_diversity(real_emb: np.ndarray, gen_emb: np.ndarray) -> Dict:
    real_mu  = np.mean(real_emb, axis=0)
    real_cov = np.cov(real_emb, rowvar=False)
    gen_mu   = np.mean(gen_emb, axis=0)
    gen_cov  = np.cov(gen_emb, rowvar=False)
    fid      = frechet_distance(real_mu, real_cov, gen_mu, gen_cov)
    div_real = all_pair_diversity(real_emb)
    div_gen  = all_pair_diversity(gen_emb)
    return dict(fid=fid, div_real=div_real, div_gen=div_gen)


# ─────────────────────────────────────────────────────────────
# Save helpers
# ─────────────────────────────────────────────────────────────

def _save_mode_metrics(out_dir: str, mode_tag: str, metrics: Dict) -> None:
    mode_dir = os.path.join(out_dir, mode_tag)
    os.makedirs(mode_dir, exist_ok=True)
    with open(os.path.join(mode_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Retrieval-based listener motion eval")

    # Data
    ap.add_argument("--dataset_dir",  required=True)
    ap.add_argument("--pairs_csv",    required=True,
                    help="dir (with train/test.csv) or single csv with split column")
    ap.add_argument("--train_split",  default="train")
    ap.add_argument("--test_split",   default="test")

    # Method
    ap.add_argument("--method", choices=["eval_wrapper", "scorer", "random"], required=True)
    ap.add_argument("--cond_modes", nargs="+",
                    default=["t", "t+e", "a", "a+e", "t+a", "t+a+e"],
                    help="[scorer only] conditioning modes to evaluate")
    ap.add_argument("--cond_head", default="fused",
                    choices=["fused", "text", "audio", "emo"],
                    help="[scorer only] which head embedding to use for similarity")

    # Models
    ap.add_argument("--t2m_opt",     required=True,
                    help="T2M opt.txt, e.g. ./checkpoints/t2m/Comp_v6_KLD005/opt.txt")
    ap.add_argument("--vqvae_ckpt",  required=True,
                    help="VQ-VAE checkpoint for decoding motion codes")
    ap.add_argument("--scorer_ckpt", default="",
                    help="[scorer only] scorer checkpoint (.pt)")

    # EvalWrapper / GloVe text encoder
    ap.add_argument("--glove_dir",   default="./glove",
                    help="[eval_wrapper] GloVe dir with our_vab_*.npy/pkl")

    # Normalization
    ap.add_argument("--mean_path", default='/path/to/T2M_GPT/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy',
                    help="mean.npy for joint vector normalization")
    ap.add_argument("--std_path",  default='/path/to/T2M_GPT/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy',
                    help="std.npy for joint vector normalization")

    # Audio (scorer method)
    ap.add_argument("--audio_code_dir",   default="",
                    help="[scorer] dir with audio code .npy/.npz files")
    ap.add_argument("--audio_max_len",    type=int, default=512)
    ap.add_argument("--audio_codebooks",  type=int, default=8)

    # Output
    ap.add_argument("--out_dir",    default="",
                    help="Root output dir. Per-mode subdirs are created automatically:\n"
                         "  {out_dir}/{mode}/joint_vecs/{query_id}.npy  — denormalized joints\n"
                         "  {out_dir}/{mode}/motion_tokens/{query_id}.npy — VQ token codes\n"
                         "  {out_dir}/{mode}/metrics.json\n"
                         "  {out_dir}/summary.json")
    ap.add_argument("--out_json",   default="", help="(legacy) save result summary to JSON")

    # Misc
    ap.add_argument("--batch_size",   type=int, default=64)
    ap.add_argument("--max_frames",   type=int, default=196)
    ap.add_argument("--force_unique", action="store_true",
                    help="Greedy assignment: each train motion used at most once. "
                         "Fixes mode collapse when embeddings cluster (e.g. audio-only mode).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (used by --method random)")

    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    joint_vecs_dir = os.path.join(args.dataset_dir, "HumanML3D", "new_joint_vecs")
    vqvae_dir      = os.path.join(args.dataset_dir, "HumanML3D", "VQVAE")

    # Normalization stats
    mean_np = std_np = None
    if args.mean_path and args.std_path:
        if os.path.isfile(args.mean_path) and os.path.isfile(args.std_path):
            mean_np = np.load(args.mean_path).astype(np.float32)
            std_np  = np.load(args.std_path).astype(np.float32)
            print(f"[Norm] loaded Mean/Std from {args.mean_path}")

    # EvaluatorModelWrapper (used by both methods for FID eval)
    eval_opt     = get_opt(args.t2m_opt, device)
    eval_wrapper = EvaluatorModelWrapper(eval_opt)

    # VQ-VAE (for decoding retrieved motion codes)
    print("[VQ-VAE] loading ...")
    vae = build_vqvae(args.vqvae_ckpt, device)

    # Load CSVs
    train_df   = read_csv_split(args.pairs_csv, args.train_split)
    test_df    = read_csv_split(args.pairs_csv, args.test_split)
    train_uniq = unique_motion_rows(train_df)
    test_uniq  = unique_motion_rows(test_df)

    train_ids = train_uniq["_mid"].tolist()
    test_ids  = test_uniq["_mid"].tolist()
    print(f"[Data] train unique motions={len(train_ids)}  test unique motions={len(test_ids)}")

    # ── Real test embeddings (reference for FID) ─────────────
    print("\n[Real] building test motion embeddings ...")
    real_emb = collect_real_embeddings(
        test_ids, joint_vecs_dir, eval_wrapper, device,
        max_frames=args.max_frames, mean=mean_np, std=std_np,
        batch_size=args.batch_size,
    )
    print(f"[Real] {real_emb.shape[0]} samples, dim={real_emb.shape[1]}")

    results: Dict[str, Dict] = {}

    # ── helper: save dirs for a given tag ────────────────────
    def mode_dirs(tag: str) -> Tuple[Optional[str], Optional[str]]:
        if not args.out_dir:
            return None, None
        base = os.path.join(args.out_dir, tag)
        return os.path.join(base, "joint_vecs"), os.path.join(base, "motion_tokens")

    # ══════════════════════════════════════════════════════════
    # Method 1: EvaluatorWrapper retrieval
    # ══════════════════════════════════════════════════════════
    if args.method == "eval_wrapper":
        print("\n=== Method 1: EvaluatorWrapper retrieval ===")

        # Build motion embedding bank
        bank_emb, bank_ids = build_motion_bank_eval_wrapper(
            train_ids, joint_vecs_dir, eval_wrapper, device,
            max_frames=args.max_frames, mean=mean_np, std=std_np,
            batch_size=args.batch_size,
        )
        print(f"[MotionBank] {len(bank_ids)} motions, dim={bank_emb.shape[1]}")

        # Encode test sayings with T2M GloVe text encoder
        # Falls back to CLIP if spacy/GloVe unavailable
        test_sayings = test_uniq["speaker_transcript"].apply(normalize_text_field).tolist()
        query_emb = _encode_test_texts(
            test_sayings, eval_wrapper, args.glove_dir, bank_emb.shape[1], device
        )

        # Retrieve top-1 (or unique greedy)
        top1_ids = cosine_retrieve(query_emb, bank_emb, bank_ids,
                                   force_unique=args.force_unique)
        print(f"[Retrieve] retrieved {len(top1_ids)} motions  "
              f"(unique={len(set(top1_ids))})")

        jv_dir, mt_dir = mode_dirs("eval_wrapper")
        print("[Eval] decoding + embedding retrieved motions ...")
        gen_emb = collect_retrieved_embeddings(
            top1_ids, test_ids, vqvae_dir, vae, eval_wrapper, device,
            max_frames=args.max_frames, batch_size=args.batch_size,
            save_joint_vecs_dir=jv_dir, save_motion_tokens_dir=mt_dir,
            mean=mean_np, std=std_np,
        )
        metrics = compute_fid_diversity(real_emb, gen_emb)
        print("\n========== EvalWrapper Retrieval ==========")
        print(f"  FID (retrieved vs real):   {metrics['fid']:.4f}")
        print(f"  Diversity (real):           {metrics['div_real']:.4f}")
        print(f"  Diversity (retrieved):      {metrics['div_gen']:.4f}")
        results["eval_wrapper"] = metrics

        if jv_dir:
            _save_mode_metrics(args.out_dir, "eval_wrapper", metrics)

    # ══════════════════════════════════════════════════════════
    # Method 2: Scorer-based retrieval
    # ══════════════════════════════════════════════════════════
    elif args.method == "scorer":
        if not args.scorer_ckpt:
            raise RuntimeError("--scorer_ckpt is required for --method scorer")

        print("\n=== Method 2: Scorer-based retrieval ===")
        scorer, emo2id, cfg = load_scorer(args.scorer_ckpt, device)

        motion_vocab = cfg.get("motion_vocab", 512)

        from transformers import T5Tokenizer
        t5_name = cfg.get("t5_encoder", "google-t5/t5-base")
        t5_tok  = T5Tokenizer.from_pretrained(t5_name)
        print(f"[T5] tokenizer loaded from '{t5_name}'")

        # Build scorer motion bank (encode all train motion codes)
        bank_emb, bank_ids = build_motion_bank_scorer(
            train_ids, vqvae_dir, scorer, device,
            motion_vocab=motion_vocab, batch_size=args.batch_size,
        )
        print(f"[MotionBank] {len(bank_ids)} motions, dim={bank_emb.shape[1]}")

        # Build test row dicts
        test_rows = []
        for _, r in test_uniq.iterrows():
            test_rows.append({
                "sayings":            normalize_text_field(r.get("speaker_transcript", "")),
                "emotion":            str(r.get("speaker_emotion", "")),
                "generated_wav_name": str(r.get("speaker_audio_wav", "")
                                          or r.get("audio_stem", "")),
                "_mid":               r["_mid"],
            })
        query_ids = [r["_mid"] for r in test_rows]

        # Evaluate each conditioning mode
        print("\n========== Scorer Retrieval Results ==========")
        for cond_mode in args.cond_modes:
            # sanitize mode for filesystem (replace + with _)
            mode_tag = f"scorer_{cond_mode.replace('+', '_')}"
            print(f"\n--- Mode: {cond_mode} (head={args.cond_head}, tag={mode_tag}) ---")

            query_emb = encode_test_queries_scorer(
                test_rows, cond_mode, scorer, t5_tok, emo2id,
                args.audio_code_dir, args.audio_max_len, args.audio_codebooks,
                args.cond_head, device, batch_size=args.batch_size,
            )

            top1_ids = cosine_retrieve(query_emb, bank_emb, bank_ids,
                                       force_unique=args.force_unique)
            n_unique = len(set(top1_ids))
            print(f"  [Retrieve] {len(top1_ids)} motions  "
                  f"(unique={n_unique}"
                  + ("  ⚠ collapsed" if n_unique <= 5 else "") + ")")

            jv_dir, mt_dir = mode_dirs(mode_tag)
            gen_emb = collect_retrieved_embeddings(
                top1_ids, query_ids, vqvae_dir, vae, eval_wrapper, device,
                max_frames=args.max_frames, batch_size=args.batch_size,
                save_joint_vecs_dir=jv_dir, save_motion_tokens_dir=mt_dir,
                mean=mean_np, std=std_np,
            )
            metrics = compute_fid_diversity(real_emb, gen_emb)
            tag = f"scorer_{cond_mode}"
            results[tag] = metrics

            print(f"  FID (retrieved vs real):   {metrics['fid']:.4f}")
            print(f"  Diversity (real):           {metrics['div_real']:.4f}")
            print(f"  Diversity (retrieved):      {metrics['div_gen']:.4f}")

            if jv_dir:
                _save_mode_metrics(args.out_dir, mode_tag, metrics)
                print(f"  [Saved] joint_vecs → {jv_dir}")
                print(f"  [Saved] motion_tokens → {mt_dir}")

    # ══════════════════════════════════════════════════════════
    # Method 3: Random baseline
    # ══════════════════════════════════════════════════════════
    elif args.method == "random":
        print("\n=== Method 3: Random retrieval baseline ===")
        rng = np.random.default_rng(args.seed)

        # For each test query pick a random train motion (with replacement allowed)
        # Run multiple seeds for stability, then average → or single run is fine
        random_ids = rng.choice(train_ids, size=len(test_ids), replace=True).tolist()
        n_unique = len(set(random_ids))
        print(f"[Random] {len(random_ids)} samples, {n_unique} distinct train motions  "
              f"(seed={args.seed})")

        jv_dir, mt_dir = mode_dirs("random")
        gen_emb = collect_retrieved_embeddings(
            random_ids, test_ids, vqvae_dir, vae, eval_wrapper, device,
            max_frames=args.max_frames, batch_size=args.batch_size,
            save_joint_vecs_dir=jv_dir, save_motion_tokens_dir=mt_dir,
            mean=mean_np, std=std_np,
        )
        metrics = compute_fid_diversity(real_emb, gen_emb)
        results["random"] = metrics

        print(f"\n========== Random Baseline ==========")
        print(f"  FID (random vs real):      {metrics['fid']:.4f}")
        print(f"  Diversity (real):           {metrics['div_real']:.4f}")
        print(f"  Diversity (random):         {metrics['div_gen']:.4f}")

        if jv_dir:
            _save_mode_metrics(args.out_dir, "random", metrics)
            print(f"  [Saved] joint_vecs → {jv_dir}")
            print(f"  [Saved] motion_tokens → {mt_dir}")

    # ── Summary ──────────────────────────────────────────────
    print("\n\n========== Summary ==========")
    print(f"{'Method':<25}  {'FID':>8}  {'Div(gen)':>10}  {'Div(real)':>10}")
    print("-" * 58)
    for tag, v in results.items():
        print(f"  {tag:<23}  {v['fid']:>8.4f}  {v['div_gen']:>10.4f}  {v['div_real']:>10.4f}")

    # Save summary
    if args.out_dir:
        summary_path = os.path.join(args.out_dir, "summary.json")
        os.makedirs(args.out_dir, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Saved] summary → {summary_path}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Saved] {args.out_json}")


if __name__ == "__main__":
    main()
