# -*- coding: utf-8 -*-
"""
ReactMotionTrainer

✅ NEW:
1) group-wise weighted loss via batch["group_weights"] (shape [Ng])
2) in-batch template suppression:
   - compute signature per group (from a representative labels row)
   - if same signature repeats inside this batch, add penalty
"""

import math
import hashlib
import random
from collections import OrderedDict, Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer
import torchaudio

from reactmotion.dataset.reactmotionnet_dataset import ensure_2d_mono  # if you have it there; else remove
from reactmotion.dataset.audio_aug import MixedAcousticAug
from reactmotion.dataset.mimi_encoder import MimiStreamingEncoder
from reactmotion.dataset.prompt_builder import build_prompt


# =========================
# Diversity Eval (optional)
# =========================
@dataclass
class DiversityEvalConfig:
    enabled: bool = True
    eval_batches: int = 20
    max_new_tokens: int = 256
    cond_variants: Optional[List[str]] = None
    sample_times: int = 4
    temperature: float = 1.0
    top_p: float = 0.95
    distinct_n: Tuple[int, int] = (1, 2)
    query_only: bool = True
    seed: int = 42


def _decode_to_token_lists(gen_ids: torch.Tensor, pad_id: int, eos_id: Optional[int]) -> List[List[int]]:
    out = []
    for seq in gen_ids.tolist():
        if eos_id is not None and eos_id in seq:
            seq = seq[: seq.index(eos_id) + 1]
        while len(seq) > 0 and seq[-1] == pad_id:
            seq.pop()
        out.append(seq)
    return out

def _distinct_n(seqs: List[List[int]], n: int) -> float:
    grams = []
    for s in seqs:
        if len(s) < n:
            continue
        grams.extend([tuple(s[i : i + n]) for i in range(len(s) - n + 1)])
    if len(grams) == 0:
        return 0.0
    return len(set(grams)) / float(len(grams))

def _unique_ratio(seqs: List[List[int]]) -> float:
    if len(seqs) == 0:
        return 0.0
    return len({tuple(s) for s in seqs}) / float(len(seqs))

def _top1_freq(seqs: List[List[int]]) -> float:
    if len(seqs) == 0:
        return 0.0
    from collections import Counter as _C
    c = _C(tuple(s) for s in seqs)
    most = c.most_common(1)[0][1]
    return most / float(len(seqs))


class ReactMotionTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        *args,
        loss_type: str = "multi_ce_rank",  # multi_ce_rank | ce | multi_ce | rank
        normalize_logsumexp: bool = True,
        rank_margin: float = 1.0,
        w_rank: float = 0.5,
        w_gn: float = 0.5,


        source_len: int = 256,
        cond_mode: str = "t+a+e",

        cond_dropout: float = 0.30,
        drop_e_only_when_multi: bool = False,

        # global anti-template (LRU, optional)
        diversity_w: float = 0.0,
        diversity_cache_size: int = 50000,
        diversity_use_first_gold: bool = True,

        # ✅ NEW: in-batch template suppression
        batch_template_w: float = 0.0,
        batch_template_power: float = 1.0,  # penalty ~ (repeat_count-1)^power

        # inverse-frequency reweighting: upweight rare motion tokens in loss
        use_inverse_freq_reweight: bool = False,
        freq_alpha: float = 1.0,           # exponent: w(t) = 1 / freq(t)^alpha
        motion_codebook_size: int = 512,

        # wav->mimi (only needed when audio_mode="wav"; pass enable_wav=True to activate)
        enable_wav: bool = False,
        audio_sr: int = 24000,
        mimi_codebooks: int = 8,
        mimi_cardinality: int = 2048,
        mimi_chunk_frames: int = 32,
        audio_level: str = "base",
        enable_audio_aug: bool = False,
        audio_cache_size: int = 4096,

        # T2M co-training loss weight
        t2m_loss_weight: float = 1.0,

        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.loss_type = str(loss_type)
        _valid_loss = ["multi_ce_rank", "ce", "multi_ce", "rank"]
        if self.loss_type not in _valid_loss:
            raise ValueError(f"loss_type must be one of {_valid_loss}, got {self.loss_type}")

        self.normalize_logsumexp = bool(normalize_logsumexp)
        self.rank_margin = float(rank_margin)
        self.w_rank = float(w_rank)
        self.w_gn = float(w_gn)

        # inverse-frequency reweighting
        self.use_inverse_freq_reweight = bool(use_inverse_freq_reweight)
        self.freq_alpha = float(freq_alpha)
        self.motion_codebook_size = int(motion_codebook_size)
        self._token_counts: Optional[torch.Tensor] = None  # lazy init

        self.source_len = int(source_len)
        self.cond_mode = str(cond_mode)

        self.cond_dropout = float(cond_dropout)
        self.drop_e_only_when_multi = bool(drop_e_only_when_multi)

        self.diversity_w = float(diversity_w)
        self.diversity_use_first_gold = bool(diversity_use_first_gold)
        self._sig_counts = OrderedDict()
        self._sig_cache_size = int(diversity_cache_size)

        self.batch_template_w = float(batch_template_w)
        self.batch_template_power = float(batch_template_power)

        self.enable_wav = bool(enable_wav)
        self.audio_sr = int(audio_sr)
        self.mimi_codebooks = int(mimi_codebooks)
        self.mimi_cardinality = int(mimi_cardinality)
        self.mimi_chunk_frames = int(mimi_chunk_frames)
        self.audio_level = str(audio_level)
        self.enable_audio_aug = bool(enable_audio_aug)

        self._audio_cache: Dict[str, str] = {}
        self._audio_cache_order: List[str] = []
        self._audio_cache_size = int(audio_cache_size)

        self.t2m_loss_weight = float(t2m_loss_weight)

        if self.enable_wav:
            self.mimi = MimiStreamingEncoder(device=str(self.args.device), codebooks=self.mimi_codebooks)
            self.aug = MixedAcousticAug(sr=self.audio_sr, device="cpu")

    # -------------------------
    # inverse-frequency token weights
    # -------------------------
    def _get_token_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Build per-token weights from inverse frequency of motion tokens.
        Counts are accumulated across training steps (online).
        Returns weight tensor of shape [V] on the same device as labels.
        """
        V = labels.max().item() + 1
        V = max(V, self.motion_codebook_size)

        if self._token_counts is None:
            self._token_counts = torch.ones(V, dtype=torch.float64)

        # expand if needed
        if self._token_counts.numel() < V:
            new = torch.ones(V, dtype=torch.float64)
            new[: self._token_counts.numel()] = self._token_counts
            self._token_counts = new

        # update counts from this batch (valid tokens only)
        valid = labels[labels >= 0].detach().cpu().long()
        if valid.numel() > 0:
            self._token_counts.scatter_add_(
                0, valid.clamp(max=V - 1),
                torch.ones_like(valid, dtype=torch.float64),
            )

        freq = self._token_counts[:V].float()
        freq = freq / freq.sum().clamp_min(1.0)                     # normalize to prob
        w = 1.0 / (freq.clamp_min(1e-8) ** self.freq_alpha)         # inverse freq
        w = w / w.mean()                                              # normalize so mean=1
        return w.to(labels.device)

    # -------------------------
    # seq logp (length-normalized), with optional inverse-freq reweight
    # -------------------------
    def _seq_logp(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Return length-normalized log-prob for each sequence:
          score = sum(w(t) * log p(y_t)) / sum(w(t))
        where w(t)=1 by default, or inverse-frequency weight if enabled.
        """
        log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]
        mask = (labels != -100)                    # [B, T]
        safe = labels.clone()
        safe[~mask] = 0

        tok = log_probs.gather(-1, safe.unsqueeze(-1)).squeeze(-1)  # [B, T]

        if self.use_inverse_freq_reweight:
            tw = self._get_token_weights(labels)     # [V]
            per_tok_w = tw[safe.clamp(min=0)]         # [B, T]
            per_tok_w = per_tok_w * mask.float()
            tok = tok * per_tok_w
            denom = per_tok_w.sum(dim=1).clamp_min(1e-6)  # [B]
        else:
            tok = tok * mask.float()
            denom = mask.float().sum(dim=1).clamp_min(1.0)  # [B]

        return tok.sum(dim=1) / denom                               # [B]

    # -------------------------
    # signature
    # -------------------------
    @staticmethod
    def _label_signature(label_seq: torch.Tensor) -> str:
        ids = label_seq[label_seq != -100].detach().to("cpu", non_blocking=True).tolist()
        if len(ids) == 0:
            return "EMPTY"
        h = hashlib.sha1((" ".join(map(str, ids))).encode("utf-8")).hexdigest()
        return h

    def _sig_count(self, sig: str) -> int:
        if sig in self._sig_counts:
            c = self._sig_counts.pop(sig)
            self._sig_counts[sig] = c
            return c
        return 0

    def _sig_inc(self, sig: str):
        if sig in self._sig_counts:
            c = self._sig_counts.pop(sig) + 1
            self._sig_counts[sig] = c
        else:
            self._sig_counts[sig] = 1
        while len(self._sig_counts) > self._sig_cache_size:
            self._sig_counts.popitem(last=False)

    # -------------------------
    # wav cache helpers
    # -------------------------
    def _cache_get(self, key: str) -> Optional[str]:
        return self._audio_cache.get(key, None)

    def _cache_put(self, key: str, val: str):
        if key in self._audio_cache:
            return
        self._audio_cache[key] = val
        self._audio_cache_order.append(key)
        if len(self._audio_cache_order) > self._audio_cache_size:
            old = self._audio_cache_order.pop(0)
            self._audio_cache.pop(old, None)

    def _codes_to_audio_text(self, codes: torch.Tensor) -> str:
        arr = codes
        L = min(arr.shape[0], self.mimi_codebooks)
        parts = ["<Audio Tokens>"]

        def emit_level(lv: int):
            for tok in arr[lv].reshape(-1):
                t = int(tok.item())
                if 0 <= t < self.mimi_cardinality:
                    parts.append(f"<Audio Level {lv} Token {t}>")

        if self.audio_level == "base":
            emit_level(0)
        elif self.audio_level == "all":
            for lv in range(L):
                emit_level(lv)
        elif self.audio_level == "rand":
            lv_num = int(torch.randint(1, L + 1, (1,)).item())
            for lv in range(lv_num):
                emit_level(lv)
        else:
            raise ValueError("audio_level must be base/all/rand")

        parts.append("</Audio Tokens>")
        return " ".join(parts)

    @torch.no_grad()
    def _encode_wav_paths(self, wav_paths: List[str]) -> List[str]:
        if not self.enable_wav:
            raise RuntimeError("enable_wav=False but wav mode requested")

        audio_texts: List[Optional[str]] = [None] * len(wav_paths)
        need: List[int] = []

        for i, p in enumerate(wav_paths):
            if not p:
                raise RuntimeError("Empty wav_path encountered in group_meta.")
            cached = self._cache_get(p)
            if cached is not None:
                audio_texts[i] = cached
            else:
                need.append(i)

        if len(need) == 0:
            return [t for t in audio_texts if t is not None]  # type: ignore

        wavs_cpu = []
        for i in need:
            p = wav_paths[i]
            w, sr = torchaudio.load(p)
            if sr != self.audio_sr:
                w = torchaudio.functional.resample(w, sr, self.audio_sr)
            w = ensure_2d_mono(w.to(torch.float32).cpu())
            if self.enable_audio_aug:
                w = self.aug(w)
            wavs_cpu.append(w)

        codes_list, _ = self.mimi.encode_many_concat(
            [w.to(self.args.device, non_blocking=True) for w in wavs_cpu],
            chunk_frames=self.mimi_chunk_frames,
            return_latent=False,
        )

        for j, idx in enumerate(need):
            p = wav_paths[idx]
            txt = self._codes_to_audio_text(codes_list[j].cpu())
            audio_texts[idx] = txt
            self._cache_put(p, txt)

        return [t for t in audio_texts if t is not None]  # type: ignore

    def _cond_str_to_mask(self, cond: str) -> Tuple[bool, bool, bool]:
        use_t = ("t" in cond)
        use_a = ("a" in cond)
        use_e = cond.endswith("+e")
        if (not use_t) and (not use_a):
            use_t = True
        return (use_t, use_a, use_e)

    def _prepare_enc_for_loss(self, inputs: Dict) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
        group_sizes_gold = inputs.pop("group_sizes_gold")
        labels = inputs["labels"]

        # code/none mode
        if "input_ids" in inputs and "attention_mask" in inputs:
            enc = {
                "input_ids": inputs["input_ids"].to(self.args.device),
                "attention_mask": inputs["attention_mask"].to(self.args.device),
                "labels": labels.to(self.args.device),
            }
            return enc, enc["labels"], group_sizes_gold

        # wav mode
        group_meta = inputs.pop("group_meta")
        wav_paths = [m.get("wav_path", "") for m in group_meta]
        audio_texts = self._encode_wav_paths(wav_paths)

        sources = []
        for gi, meta in enumerate(group_meta):
            use_t, use_a, use_e = self._cond_str_to_mask(self.cond_mode)
            src = build_prompt(
                speaker_transcription=meta.get("transcription", ""),
                speaker_audio=audio_texts[gi] if use_a else "",
                speaker_emotion=meta.get("emotion", ""),
                use_transcription=use_t,
                use_audio=use_a,
                use_emotion=use_e,
            )
            K = int(group_sizes_gold[gi].item())
            sources.extend([src] * (K + 2))

        tok = self.tokenizer(
            sources,
            padding=True,
            truncation=True,
            max_length=self.source_len,
            return_tensors="pt",
        )
        enc = {k: v.to(self.args.device) for k, v in tok.items()}
        enc["labels"] = labels.to(self.args.device)
        return enc, enc["labels"], group_sizes_gold

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        group_sizes_gold = inputs.get("group_sizes_gold", None)
        if group_sizes_gold is None:
            raise RuntimeError("Batch missing 'group_sizes_gold'.")

        group_weights = inputs.get("group_weights", None)  # [Ng]
        if group_weights is None:
            group_weights = torch.ones((int(group_sizes_gold.numel()),), dtype=torch.float32)
        group_weights = group_weights.to(self.args.device, dtype=torch.float32)

        # T2M mask: which groups are text-to-motion (CE-only, no ranking)
        is_t2m_mask = inputs.pop("is_t2m_mask", None)  # [Ng] bool or None

        enc, labels, group_sizes_gold = self._prepare_enc_for_loss(inputs)
        outputs = model(**enc)
        logits = outputs.logits
        seq_logp = self._seq_logp(logits, labels)

        # -------------------------
        # layout sanity check:  [gold_1..gold_K, silver, neg] per group
        # -------------------------
        B = int(seq_logp.numel())
        Ng = int(group_sizes_gold.numel())
        expected = int(group_sizes_gold.sum().item() + 2 * Ng)
        if expected != B:
            raise RuntimeError(f"expected expanded={expected} but got B={B}.")

        # -------------------------
        # signature (for anti-template / batch-template)
        # -------------------------
        sigs: List[str] = []
        offset = 0
        for K in group_sizes_gold.tolist():
            K = int(K)
            sigs.append(self._label_signature(labels[offset]))
            offset += (K + 2)
        sig_cnt = Counter(sigs)

        # -------------------------
        # per-group loss
        # -------------------------
        losses = []
        weights = []
        offset = 0
        m = self.rank_margin

        for gi, K in enumerate(group_sizes_gold.tolist()):
            K = int(K)
            gold_vals = seq_logp[offset : offset + K]
            silver_val = seq_logp[offset + K]
            neg_val = seq_logp[offset + K + 1]

            # Check if this group is a T2M sample
            gi_is_t2m = bool(is_t2m_mask[gi]) if is_t2m_mask is not None else False

            if gi_is_t2m:
                # T2M: simple CE loss only (K=1, silver/neg are dummies)
                loss_ce = -gold_vals[0]
                group_loss = loss_ce
                # Apply T2M loss weight scaling
                weights.append(group_weights[gi] * self.t2m_loss_weight)
            else:
                # ReactMotion: full ranking + CE + penalties
                # --- gold score: LogSumExp over K golds ---
                gmax = gold_vals.max()
                lse = gmax + torch.log(torch.exp(gold_vals - gmax).sum())
                if self.normalize_logsumexp:
                    lse = lse - math.log(K)
                gold_score = lse

                # --- CE component ---
                if self.loss_type == "ce":
                    loss_ce = -gold_vals[0]
                elif self.loss_type in ("multi_ce", "multi_ce_rank"):
                    loss_ce = -gold_score
                else:
                    loss_ce = torch.tensor(0.0, device=gold_vals.device)

                # --- Rank component ---
                if self.loss_type in ("rank", "multi_ce_rank"):
                    loss_rank = (
                        F.softplus(m - (gold_score - silver_val)) +
                        F.softplus(m - (silver_val - neg_val)) +
                        self.w_gn * F.softplus(m - (gold_score - neg_val))
                    )
                else:
                    loss_rank = torch.tensor(0.0, device=gold_vals.device)

                # --- optional penalties ---
                diversity_pen = 0.0
                if self.diversity_w > 0:
                    sig = sigs[gi]
                    c = self._sig_count(sig)
                    diversity_pen = self.diversity_w * math.log(1.0 + float(c))
                    self._sig_inc(sig)

                batch_pen = 0.0
                if self.batch_template_w > 0:
                    rep = int(sig_cnt[sigs[gi]])
                    if rep > 1:
                        batch_pen = self.batch_template_w * float((rep - 1) ** self.batch_template_power)

                group_loss = loss_ce + self.w_rank * loss_rank + diversity_pen + batch_pen
                weights.append(group_weights[gi])

            losses.append(group_loss)
            offset += (K + 2)

        losses_t = torch.stack(losses)          # [Ng]
        weights_t = torch.stack(weights)        # [Ng]

        denom = torch.clamp(weights_t.sum(), min=1e-6)
        loss = (losses_t * weights_t).sum() / denom

        return (loss, outputs) if return_outputs else loss

    def evaluate_diversity(self, cfg):
        # minimal implementation: no generation evaluation yet, return empty metrics
        # fill in the actual generation + statistics logic later
        return {}
