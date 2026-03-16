# train/callback_diversity_early_stop.py
# -*- coding: utf-8 -*-
import hashlib
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch
from transformers import TrainerCallback


def _seq_signature(token_ids: List[int]) -> str:
    """
    Make a stable signature for a generated token sequence.
    We use md5 of bytes for compactness.
    """
    b = bytes(int(x) % 256 for x in token_ids)
    return hashlib.md5(b).hexdigest()


def _strip_special(ids: torch.Tensor, pad_id: int, eos_id: Optional[int]) -> List[int]:
    """
    Remove pad and anything after eos (inclusive).
    """
    x = ids.tolist()
    # cut at eos
    if eos_id is not None and eos_id in x:
        x = x[: x.index(eos_id)]
    # remove pads
    x = [t for t in x if t != pad_id]
    return x


def _distinct_n(seqs: List[List[int]], n: int) -> float:
    """
    distinct-n over tokens: unique n-grams / total n-grams
    """
    total = 0
    uniq = set()
    for s in seqs:
        if len(s) < n:
            continue
        for i in range(len(s) - n + 1):
            total += 1
            uniq.add(tuple(s[i:i+n]))
    return float(len(uniq) / max(total, 1))


@dataclass
class DiversityEarlyStopConfig:
    # stop if collapse persists for `patience` evals
    patience: int = 3

    # collapse rules (you can tune)
    min_unique_ratio: float = 0.35     # too low -> template
    max_top1_freq: float = 0.08        # too high -> template

    # eval sampling
    eval_batches: int = 20             # how many val batches to generate each eval
    max_new_tokens: int = 256
    num_beams: int = 1
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0

    # if True, also compute distinct-1/2 and include in logs
    log_distinct: bool = True


class DiversityEarlyStopCallback(TrainerCallback):
    """
    Early-stop training when diversity collapses (mode collapse / template actions).
    Triggered on each evaluation.

    Works even if trainer.predict_with_generate=False, because we run generate() here.
    """
    def __init__(self, cfg: DiversityEarlyStopConfig):
        self.cfg = cfg
        self.bad_count = 0
        self.last_metrics = {}

    @torch.no_grad()
    def _compute_diversity_metrics(self, trainer) -> Dict[str, float]:
        model = trainer.model
        tokenizer = trainer.tokenizer
        dl = trainer.get_eval_dataloader()

        model.eval()
        device = trainer.args.device

        pad_id = tokenizer.pad_token_id
        eos_id = tokenizer.eos_token_id

        sig_counts = {}
        seqs_tokens: List[List[int]] = []

        n_batches = 0
        total = 0

        for batch in dl:
            n_batches += 1
            if n_batches > self.cfg.eval_batches:
                break

            # move needed fields to device
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)

            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=self.cfg.max_new_tokens,
                num_beams=self.cfg.num_beams,
                do_sample=self.cfg.do_sample,
                temperature=self.cfg.temperature if self.cfg.do_sample else None,
                top_p=self.cfg.top_p if self.cfg.do_sample else None,
            )

            for i in range(gen.size(0)):
                toks = _strip_special(gen[i], pad_id=pad_id, eos_id=eos_id)
                seqs_tokens.append(toks)
                sig = _seq_signature(toks)
                sig_counts[sig] = sig_counts.get(sig, 0) + 1
                total += 1

        unique = len(sig_counts)
        top1 = max(sig_counts.values()) if sig_counts else 0

        unique_ratio = float(unique / max(total, 1))
        top1_freq = float(top1 / max(total, 1))

        metrics = {
            "div/unique_ratio": unique_ratio,
            "div/top1_freq": top1_freq,
            "div/num_samples": float(total),
            "div/num_unique": float(unique),
        }

        if self.cfg.log_distinct:
            metrics["div/distinct1"] = _distinct_n(seqs_tokens, 1)
            metrics["div/distinct2"] = _distinct_n(seqs_tokens, 2)

        return metrics

    def on_evaluate(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer", None)
        if trainer is None:
            return control

        # 只在主进程做（避免多卡重复 generate）
        if hasattr(trainer, "is_world_process_zero") and (not trainer.is_world_process_zero()):
            return control

        metrics = self._compute_diversity_metrics(trainer)
        self.last_metrics = metrics

        # 记录到 log / wandb
        trainer.log(metrics)

        # 判定是否坍塌
        collapse = (
            metrics["div/unique_ratio"] < self.cfg.min_unique_ratio
            or metrics["div/top1_freq"] > self.cfg.max_top1_freq
        )

        if collapse:
            self.bad_count += 1
            trainer.log({"div/collapse": 1.0, "div/bad_count": float(self.bad_count)})
        else:
            self.bad_count = 0
            trainer.log({"div/collapse": 0.0, "div/bad_count": 0.0})

        if self.bad_count >= self.cfg.patience:
            trainer.log({"div/early_stop": 1.0})
            control.should_training_stop = True

        return control
