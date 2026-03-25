# train/callback_diversity_eval.py
import numpy as np
import torch
from transformers import TrainerCallback

def _seq_signature(ids, ignore_pad_id=-100, eos_id=None, max_len=None):
    # ids: 1D list[int]
    out = []
    for x in ids:
        if x == ignore_pad_id:
            continue
        if eos_id is not None and x == eos_id:
            break
        out.append(int(x))
        if max_len is not None and len(out) >= max_len:
            break
    return tuple(out)

def _distinct_1_2(seqs):
    # seqs: list[list[int]]
    unigrams = set()
    bigrams = set()
    total_uni = 0
    total_bi = 0
    for s in seqs:
        s = [int(x) for x in s]
        total_uni += max(len(s), 1)
        total_bi += max(len(s) - 1, 1)
        for x in s:
            unigrams.add((x,))
        for i in range(len(s) - 1):
            bigrams.add((s[i], s[i+1]))
    d1 = len(unigrams) / max(total_uni, 1)
    d2 = len(bigrams) / max(total_bi, 1)
    return float(d1), float(d2)

class DiversitySimpleCallback(TrainerCallback):
    def __init__(self, eval_batches=20, num_return_sequences=1, max_new_tokens=128,
                 do_sample=False, temperature=1.0, top_p=1.0, top_k=0,
                 sig_max_len=64, prefix="div"):
        self.eval_batches = int(eval_batches)
        self.num_return_sequences = int(num_return_sequences)
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.sig_max_len = int(sig_max_len)
        self.prefix = str(prefix)

    @torch.no_grad()
    def on_evaluate(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer", None)
        if trainer is None:
            trainer = getattr(self, "trainer", None)
        if trainer is None:
            return

        model = trainer.model
        model.eval()

        dl = trainer.get_eval_dataloader()
        it = iter(dl)

        # tokenizer pad/eos
        tok = getattr(trainer, "tokenizer", None)
        eos_id = tok.eos_token_id if tok is not None else None
        pad_id = tok.pad_token_id if tok is not None else None

        all_sigs = []          # across
        all_token_seqs = []    # across distinct

        within_uniq_list = []
        within_top1_list = []

        for _ in range(self.eval_batches):
            try:
                batch = next(it)
            except StopIteration:
                break

            if "input_ids" not in batch:
                continue

            input_ids = batch["input_ids"].to(model.device)
            attn = batch.get("attention_mask", None)
            if attn is not None:
                attn = attn.to(model.device)

            B = int(input_ids.size(0))
            R = int(self.num_return_sequences)

            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=self.max_new_tokens,
                num_beams=1,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else None,
                top_p=self.top_p if self.do_sample else None,
                top_k=self.top_k if self.do_sample else None,
                num_return_sequences=R,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
            # gen: [B*R, T]
            gen = gen.detach().cpu().tolist()

            # convert each generated sequence to a signature
            sigs = []
            for ids in gen:
                sig = _seq_signature(
                    ids,
                    ignore_pad_id=pad_id if pad_id is not None else -100,
                    eos_id=eos_id,
                    max_len=self.sig_max_len,
                )
                sigs.append(sig)
                all_sigs.append(sig)
                all_token_seqs.append(list(sig))

            # ---- within-query stats ----
            # HF generate groups R results for the same input together (default grouping order)
            # shape: [B, R]
            if len(sigs) == B * R:
                for b in range(B):
                    chunk = sigs[b * R : (b + 1) * R]
                    c = {}
                    for s in chunk:
                        c[s] = c.get(s, 0) + 1
                    within_uniq = len(c) / R
                    within_top1 = max(c.values()) / R
                    within_uniq_list.append(within_uniq)
                    within_top1_list.append(within_top1)

        if len(all_sigs) == 0:
            return

        # ---- across metrics ----
        counts = {}
        for s in all_sigs:
            counts[s] = counts.get(s, 0) + 1
        across_top1 = max(counts.values()) / len(all_sigs)
        across_uniq = len(counts) / len(all_sigs)
        d1, d2 = _distinct_1_2(all_token_seqs)

        metrics = {
            f"{self.prefix}/across_unique_ratio": float(across_uniq),
            f"{self.prefix}/across_top1_freq": float(across_top1),
            f"{self.prefix}/across_distinct1": float(d1),
            f"{self.prefix}/across_distinct2": float(d2),
            f"{self.prefix}/n_gen": int(len(all_sigs)),
        }

        # ---- within metrics ----
        if len(within_uniq_list) > 0:
            metrics[f"{self.prefix}/within_unique_ratio"] = float(np.mean(within_uniq_list))
            metrics[f"{self.prefix}/within_top1_freq"] = float(np.mean(within_top1_list))
            metrics[f"{self.prefix}/n_within_queries"] = int(len(within_uniq_list))

        trainer.log(metrics)
