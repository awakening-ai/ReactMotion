#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval/eval_judge.py

Strict-L2 “strict missing modality” eval:
- For modalities not enabled for a sample, replace inputs with “no information” at collate time:
    * no text  -> text_input_ids all pad, attention_mask all 0
    * no audio -> audio_codes all pad_id, audio_pad_mask all True
    * no emo   -> emotion_ids = <unk>

Win metric unified as mean(A) > mean(B): g>n: mean(gold)>mean(neg); g>s: mean(gold)>mean(silver); s>n: mean(silver)>mean(neg)
Supports: --fixed_mode, --cond_head, bootstrap CI

Dependencies:
- Requires the following defined in train/train_scorer.py (or your own path):
  DEFAULT_AUDIO_VOCAB, DEFAULT_AUDIO_PAD, DEFAULT_MOTION_VOCAB,
  read_split_csv, JudgeGroupDataset, JudgeNetwork, move_cb_to,
  group_infonce_loss
"""

import os, json, argparse, random
from os.path import join as pjoin
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import T5Tokenizer

# ---- import from your training file ----
from reactmotion.models.judge_network import (
    DEFAULT_AUDIO_VOCAB, DEFAULT_AUDIO_PAD, DEFAULT_MOTION_VOCAB,
    read_split_csv, JudgeGroupDataset, JudgeNetwork, move_cb_to,
    group_infonce_loss,
    MODES_FULL, MODE2ID,
    canon_label, normalize_text, clean_audio_stem,
    load_audio_codes_any, normalize_audio_codes,
    load_motion_codes,
    CondBatch, GroupBatch
)
try:
    from reactmotion.models.judge_network import fuse_mean_masked
except ImportError:
    def fuse_mean_masked(z_t, z_a, z_e, has_t, has_a, has_e, eps=1e-6):
        m_t = has_t.float().unsqueeze(-1)
        m_a = has_a.float().unsqueeze(-1)
        m_e = has_e.float().unsqueeze(-1)
        z_t = torch.nan_to_num(z_t, nan=0.0, posinf=0.0, neginf=0.0)
        z_a = torch.nan_to_num(z_a, nan=0.0, posinf=0.0, neginf=0.0)
        z_e = torch.nan_to_num(z_e, nan=0.0, posinf=0.0, neginf=0.0)
        num = z_t * m_t + z_a * m_a + z_e * m_e
        den = (m_t + m_a + m_e).clamp_min(1.0)
        z_f = num / den
        z_f = z_f / z_f.norm(dim=-1, keepdim=True).clamp_min(eps)
        return z_f

# -------------------------
# utils
# -------------------------

def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def bootstrap_mean_ci(values, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    mean = float(v.mean())
    rng = np.random.default_rng(seed)
    n = v.size
    boots = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        boots.append(v[idx].mean())
    boots = np.asarray(boots)
    lo = float(np.quantile(boots, alpha/2))
    hi = float(np.quantile(boots, 1 - alpha/2))
    return mean, (lo, hi)

@torch.no_grad()
def ndcg_at_k_gain_2_1_0(logits: torch.Tensor, label: torch.Tensor, cand_mask: torch.Tensor, k: int) -> float:
    """
    gain: gold=2, silver=1, neg=0 (label itself is 0/1/2)
    """
    logits = logits.masked_fill(~cand_mask, float("-inf"))
    B, C = logits.shape
    kk = min(k, C)

    gain = label.float() * cand_mask.float()
    order = torch.argsort(logits, dim=1, descending=True)
    denom = torch.log2(torch.arange(kk, device=logits.device, dtype=torch.float32) + 2.0)

    scores = []
    for b in range(B):
        idx = order[b, :kk]
        g = gain[b, idx]
        dcg = (g / denom).sum().item()

        valid_g = gain[b, cand_mask[b]]
        ideal = torch.sort(valid_g, descending=True).values[:kk]
        if ideal.numel() == 0:
            scores.append(0.0)
            continue
        idcg = (ideal / denom[:ideal.numel()]).sum().item()
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else float("nan")

@torch.no_grad()
def win_rates_groupwise(logits: torch.Tensor, label: torch.Tensor, cand_mask: torch.Tensor, tie_value: float = 0.5) -> dict:
    """
    Win = mean(A) > mean(B) (unified style).
    g>n: mean(gold) > mean(neg); g>s: mean(gold) > mean(silver); s>n: mean(silver) > mean(neg).
    """
    logits = logits.masked_fill(~cand_mask, float("-inf"))
    B, C = logits.shape

    out_gn, out_gs, out_sn = [], [], []
    for b in range(B):
        s = logits[b]
        lb = label[b]
        mb = cand_mask[b]
        gold_idx = torch.nonzero((lb == 2) & mb, as_tuple=False).view(-1)
        silv_idx = torch.nonzero((lb == 1) & mb, as_tuple=False).view(-1)
        neg_idx  = torch.nonzero((lb == 0) & mb, as_tuple=False).view(-1)

        def _mean_beat_mean(idx_a, idx_b):
            if idx_a.numel() == 0 or idx_b.numel() == 0:
                return float("nan")
            mean_a = s[idx_a].float().mean().item()
            mean_b = s[idx_b].float().mean().item()
            if mean_a > mean_b:
                return 1.0
            if mean_a == mean_b:
                return tie_value
            return 0.0

        out_gn.append(_mean_beat_mean(gold_idx, neg_idx))
        out_gs.append(_mean_beat_mean(gold_idx, silv_idx))
        out_sn.append(_mean_beat_mean(silv_idx, neg_idx))

    return {"win_g_gt_n": out_gn, "win_g_gt_s": out_gs, "win_s_gt_n": out_sn}

# -------------------------
# Strict-L2 Collator
# -------------------------

class JudgeEvalCollator:
    """
    Based on the GroupCollator approach from training, with added Strict-L2:
    - no text  -> pad/zero attn
    - no audio -> all pad + all padmask True
    - no emo   -> <unk>

    Also supports:
    - fixed_mode: force the entire batch to use a specific mode
    - disable_text/audio/emo: globally disable modalities (for ablation)
    """

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
        deterministic_mode: bool = True,
        fixed_mode: str = "",
        disable_text: bool = False,
        disable_audio: bool = False,
        disable_emo: bool = False,
        strict_l2: bool = True,   # Always True for this script
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
        self.fixed_mode = (fixed_mode or "").strip()
        if self.fixed_mode and self.fixed_mode not in MODES_FULL:
            raise ValueError(f"fixed_mode must be one of {MODES_FULL}, got {self.fixed_mode}")

        self.disable_text = bool(disable_text)
        self.disable_audio = bool(disable_audio)
        self.disable_emo = bool(disable_emo)
        self.strict_l2 = bool(strict_l2)

        if self.disable_text and self.disable_audio and self.disable_emo:
            raise ValueError("All modalities disabled. At least one must remain enabled.")

        self._call_idx = 0

    def _emo_id(self, emo: str) -> int:
        s = (emo or "").strip().lower()
        return int(self.emo2id.get(s, self.emo2id.get("<unk>", 0)))

    def _sample_modes(self, B: int) -> List[str]:
        if self.fixed_mode:
            return [self.fixed_mode] * B
        # eval default is deterministic: cycle in order
        if self.deterministic_mode:
            base = self._call_idx
            self._call_idx += 1
            return [MODES_FULL[(base + i) % len(MODES_FULL)] for i in range(B)]
        # if truly random sampling is needed:
        rng = random.Random(self.seed ^ (self._call_idx * 1337))
        self._call_idx += 1
        return [rng.choice(MODES_FULL) for _ in range(B)]

    @staticmethod
    def _repair_only_e(has_t, has_a, has_e):
        only_e = has_e & (~has_t) & (~has_a)
        if only_e.any():
            has_t = has_t | only_e
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
        if self.disable_text: has_t[:] = False
        if self.disable_audio: has_a[:] = False
        if self.disable_emo: has_e[:] = False
        has_t, has_a, has_e = self._repair_only_e(has_t, has_a, has_e)

        # -------- text tokenize (always run tokenizer, then strict-L2 blank out) --------
        texts = [it.get("sayings", "") or "" for it in items]
        enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_text_len, return_tensors="pt")
        text_input_ids = enc["input_ids"]
        text_attn_mask = enc["attention_mask"]

        # -------- emotion ids --------
        emotion_ids = torch.tensor([self._emo_id(it.get("emotion", "")) for it in items], dtype=torch.long)

        # -------- audio codes --------
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
                if T < self.max_audio_len: m[T:] = True
            else:
                a = np.full((self.max_audio_len, self.audio_codebooks), self.audio_pad_id, dtype=np.int64)
                m = np.ones((self.max_audio_len,), dtype=np.bool_)
            audio_list.append(torch.from_numpy(a))
            audio_pad_mask_list.append(torch.from_numpy(m))

        audio_codes = torch.stack(audio_list, dim=0)              # [B,Ta,K]
        audio_pad_mask = torch.stack(audio_pad_mask_list, dim=0)  # [B,Ta]

        # -------- Strict-L2 injection --------
        if self.strict_l2:
            # no text: pad + attn=0
            no_t = ~has_t
            if no_t.any():
                pad_id = int(self.tok.pad_token_id)
                text_input_ids[no_t] = pad_id
                text_attn_mask[no_t] = 0

            # no audio: all pad + all padmask True
            no_a = ~has_a
            if no_a.any():
                audio_codes[no_a] = int(self.audio_pad_id)
                audio_pad_mask[no_a] = True

            # no emo: set to <unk>
            no_e = ~has_e
            if no_e.any():
                unk = int(self.emo2id.get("<unk>", 0))
                emotion_ids[no_e] = unk

        cb = CondBatch(
            has_t=has_t, has_a=has_a, has_e=has_e, mode_ids=mode_ids,
            text_input_ids=text_input_ids, text_attn_mask=text_attn_mask,
            emotion_ids=emotion_ids,
            audio_codes=audio_codes, audio_pad_mask=audio_pad_mask,
            debug_modes=modes,   # added: list[str]
        )


        # -------- candidates pack --------
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

def _nan_guard(name, x):
    if torch.isnan(x).any() or torch.isinf(x).any():
        nan_rate = torch.isnan(x).float().mean().item()
        inf_rate = torch.isinf(x).float().mean().item()
        print(f"[ERR] {name} has nan/inf: nan_rate={nan_rate:.6f} inf_rate={inf_rate:.6f}")
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x

# -------------------------
# eval loop
# -------------------------

@torch.no_grad()
def run_eval_ranker_reliability(
    model: JudgeNetwork,
    loader,
    device,
    use_silver_as_pos: bool,
    bootstrap: int,
    seed: int,
    cond_head: str,
):
    model.eval()

    all_win_gn, all_win_gs, all_win_sn = [], [], []
    all_mrr = []
    all_ndcg3, all_ndcg5, all_ndcg10 = [], [], []
    all_losses = []

    for gb in tqdm(loader, desc="eval", leave=False):
        cb = move_cb_to(gb.cb, device)
        B, C, Tm = gb.motion_codes.shape
        mc = gb.motion_codes.view(B*C, Tm).to(device, non_blocking=True)
        mp = gb.motion_pad.view(B*C, Tm).to(device, non_blocking=True)

        lab = gb.label.to(device, non_blocking=True)
        msk = gb.cand_mask.to(device, non_blocking=True)

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
                # Fallback: fused head can produce NaN for a+e (audio+emo only) due to cond_fuse/cond_pool
                if (torch.isnan(zc).any() or torch.isinf(zc).any()) and cond_head == "fused":
                    zc = fuse_mean_masked(zs["z_t"], zs["z_a"], zs["z_e"], cb.has_t, cb.has_a, cb.has_e)

            zc = _nan_guard(f"zc({cond_head})", zc)

            zm = model.encode_motion(mc, mp).view(B, C, -1)
            zm = _nan_guard("zm", zm)

            loss, logits = group_infonce_loss(
                zc, zm, lab, msk, model.scale(),
                use_silver_as_pos=use_silver_as_pos,
                sample_w=None,
            )

            # logits guard (prevent all win/ndcg from failing downstream)
            logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
            
        all_losses.append(float(loss.item()))

        wr = win_rates_groupwise(logits, lab, msk, tie_value=0.5)
        all_win_gn.extend(wr["win_g_gt_n"])
        all_win_gs.extend(wr["win_g_gt_s"])
        all_win_sn.extend(wr["win_s_gt_n"])

        # per-group MRR(gold)
        logits_masked = logits.masked_fill(~msk, float("-inf"))
        order = torch.argsort(logits_masked, dim=1, descending=True)
        gold = (lab == 2) & msk
        for b in range(B):
            ranks = torch.nonzero(gold[b, order[b]], as_tuple=False).view(-1)
            all_mrr.append(0.0 if ranks.numel() == 0 else 1.0 / (int(ranks[0].item()) + 1))

        # per-group nDCG@K
        for K, store in [(3, all_ndcg3), (5, all_ndcg5), (10, all_ndcg10)]:
            for b in range(B):
                store.append(ndcg_at_k_gain_2_1_0(
                    logits_masked[b:b+1], lab[b:b+1], msk[b:b+1], k=K
                ))

    loss_mean = float(np.mean(all_losses)) if all_losses else float("nan")

    win_gn_mean, win_gn_ci = bootstrap_mean_ci(all_win_gn, n_boot=bootstrap, seed=seed+11)
    win_gs_mean, win_gs_ci = bootstrap_mean_ci(all_win_gs, n_boot=bootstrap, seed=seed+13)
    win_sn_mean, win_sn_ci = bootstrap_mean_ci(all_win_sn, n_boot=bootstrap, seed=seed+17)

    mrr_mean, mrr_ci = bootstrap_mean_ci(all_mrr, n_boot=bootstrap, seed=seed+19)

    ndcg3_mean, ndcg3_ci = bootstrap_mean_ci(all_ndcg3, n_boot=bootstrap, seed=seed+23)
    ndcg5_mean, ndcg5_ci = bootstrap_mean_ci(all_ndcg5, n_boot=bootstrap, seed=seed+29)
    ndcg10_mean, ndcg10_ci = bootstrap_mean_ci(all_ndcg10, n_boot=bootstrap, seed=seed+31)

    return {
        "loss": loss_mean,

        "win_g_gt_n": win_gn_mean, "win_g_gt_n_ci95": list(win_gn_ci),
        "win_g_gt_s": win_gs_mean, "win_g_gt_s_ci95": list(win_gs_ci),
        "win_s_gt_n": win_sn_mean, "win_s_gt_n_ci95": list(win_sn_ci),

        "mrr_gold": mrr_mean, "mrr_gold_ci95": list(mrr_ci),

        "ndcg3": ndcg3_mean, "ndcg3_ci95": list(ndcg3_ci),
        "ndcg5": ndcg5_mean, "ndcg5_ci95": list(ndcg5_ci),
        "ndcg10": ndcg10_mean, "ndcg10_ci95": list(ndcg10_ci),
    }

# -------------------------
# model builder
# -------------------------

def build_model_from_ckpt(args, emo2id, device, ckpt_obj):
    # use structural params saved in checkpoint (without overriding eval-only params)
    ckpt_args = ckpt_obj.get("args", None)
    if ckpt_args is not None:
        # only sync model-structure-related keys
        model_keys = {
            "t5_encoder", "max_text_len", "max_audio_len", "max_motion_len",
            "d_model", "output_dim", "nhead", "enc_layers", "ff_dim", "dropout", "temperature"
        }
        for k in model_keys:
            if k in ckpt_args and hasattr(args, k):
                setattr(args, k, ckpt_args[k])
        print("[CKPT] Loaded model-config args from checkpoint (model-only).")

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
        motion_vocab=DEFAULT_MOTION_VOCAB,
        max_motion_len=args.max_motion_len,
        temperature=args.temperature,
    ).to(device)

    state = ckpt_obj.get("model", ckpt_obj)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[WARN] Missing keys:", missing[:20], ("..." if len(missing) > 20 else ""))
    if unexpected:
        print("[WARN] Unexpected keys:", unexpected[:20], ("..." if len(unexpected) > 20 else ""))
    return model

# -------------------------
# main
# -------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--dataset_dir", type=str, required=True)
    ap.add_argument("--pairs_csv", type=str, required=True)
    ap.add_argument("--audio_code_dir", type=str, required=True)
    ap.add_argument("--key_by", type=str, default="group_id", choices=["group_id", "sayings_emotion"])

    ap.add_argument("--save_dir", type=str, default="")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--eval_splits", nargs="+", default=["val", "test"], choices=["val", "test"])

    # modes to eval (default all 6)
    ap.add_argument("--modes", nargs="+", default=["a", "a+e", "t", "t+e", "t+a", "t+a+e"])
    ap.add_argument("--fixed_mode", type=str, default="", help="If set, only evaluate this mode.")

    # strict-L2 is always ON in this script
    ap.add_argument("--disable_text", action="store_true")
    ap.add_argument("--disable_audio", action="store_true")
    ap.add_argument("--disable_emo", action="store_true")

    ap.add_argument("--cond_head", type=str, default="fused",
                    choices=["fused", "text", "audio", "emo"],
                    help="Which condition head to use for logits.")

    # fallback model args (may be overwritten by ckpt model-only args)
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

    ap.add_argument("--use_silver_as_pos", action="store_true")
    ap.add_argument("--require_audio", action="store_true")

    ap.add_argument("--k_gold", type=int, default=3)
    ap.add_argument("--k_silver", type=int, default=2)
    ap.add_argument("--k_neg", type=int, default=5)

    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="")

    args = ap.parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    ckpt = torch.load(args.ckpt, map_location="cpu")

    # emo2id
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

    # save_dir
    save_dir = args.save_dir.strip() if args.save_dir else os.path.dirname(os.path.abspath(args.ckpt))
    os.makedirs(save_dir, exist_ok=True)

    # model
    model = build_model_from_ckpt(args, emo2id, device, ckpt)

    tok = T5Tokenizer.from_pretrained(args.t5_encoder)

    modes_to_eval = [args.fixed_mode] if args.fixed_mode else list(args.modes)
    for m in modes_to_eval:
        if m not in MODES_FULL:
            raise ValueError(f"Bad mode {m}. Must be in {MODES_FULL}")

    tag_disable = f"disableT{int(args.disable_text)}_disableA{int(args.disable_audio)}_disableE{int(args.disable_emo)}"
    tag_head = f"head_{args.cond_head}"
    tag_strict = "strictL2"

    results = {}

    for split in args.eval_splits:
        ds = JudgeGroupDataset(
            split, args.pairs_csv, args.dataset_dir, args.audio_code_dir,
            key_by=args.key_by,
            seed=args.seed + (0 if split == "val" else 12345),
            k_gold=args.k_gold, k_silver=args.k_silver, k_neg=args.k_neg,
            require_audio=bool(args.require_audio),
        )

        results[split] = {}

        for mode in modes_to_eval:
            collate = JudgeEvalCollator(
                tok, emo2id,
                max_text_len=args.max_text_len,
                max_audio_len=args.max_audio_len,
                max_motion_len=args.max_motion_len,
                deterministic_mode=True,
                seed=args.seed + 999,
                fixed_mode=mode,
                disable_text=args.disable_text,
                disable_audio=args.disable_audio,
                disable_emo=args.disable_emo,
                strict_l2=True,
            )

            loader = DataLoader(
                ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,
                drop_last=False, collate_fn=collate,
            )

            metrics = run_eval_ranker_reliability(
                model, loader, device,
                use_silver_as_pos=bool(args.use_silver_as_pos),
                bootstrap=int(args.bootstrap),
                seed=int(args.seed) + (0 if split == "val" else 9999) + (MODE2ID[mode] * 101),
                cond_head=args.cond_head,
            )

            results[split][mode] = metrics

            out_path = pjoin(save_dir, f"eval_{split}_{mode.replace('+','p')}_{tag_head}_{tag_strict}_{tag_disable}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "ckpt": os.path.abspath(args.ckpt),
                        "split": split,
                        "mode": mode,
                        "cond_head": args.cond_head,
                        "strict": "L2",
                        "disable": {"text": bool(args.disable_text), "audio": bool(args.disable_audio), "emo": bool(args.disable_emo)},
                        "metrics": metrics,
                        "bootstrap": int(args.bootstrap),
                        "use_silver_as_pos": bool(args.use_silver_as_pos),
                        "key_by": args.key_by,
                        "k_gold": args.k_gold, "k_silver": args.k_silver, "k_neg": args.k_neg,
                        "batch_size": args.batch_size,
                    },
                    f, indent=2
                )

            print(
                f"[{split.upper()} | {mode} | {tag_head} | {tag_strict} | {tag_disable}] "
                f"loss={metrics['loss']:.4f} | "
                f"Win(g>n)={metrics['win_g_gt_n']:.3f} [{metrics['win_g_gt_n_ci95'][0]:.3f},{metrics['win_g_gt_n_ci95'][1]:.3f}] | "
                f"Win(g>s)={metrics['win_g_gt_s']:.3f} [{metrics['win_g_gt_s_ci95'][0]:.3f},{metrics['win_g_gt_s_ci95'][1]:.3f}] | "
                f"Win(s>n)={metrics['win_s_gt_n']:.3f} [{metrics['win_s_gt_n_ci95'][0]:.3f},{metrics['win_s_gt_n_ci95'][1]:.3f}] | "
                f"MRR(g)={metrics['mrr_gold']:.3f} | "
                f"nDCG@3={metrics['ndcg3']:.3f} nDCG@5={metrics['ndcg5']:.3f} nDCG@10={metrics['ndcg10']:.3f}"
            )

    out_all = pjoin(save_dir, f"eval_all_by_mode_{tag_head}_{tag_strict}_{tag_disable}.json")
    with open(out_all, "w", encoding="utf-8") as f:
        json.dump({"ckpt": os.path.abspath(args.ckpt), "results": results}, f, indent=2)
    print("[Done] wrote", out_all)

if __name__ == "__main__":
    main()
