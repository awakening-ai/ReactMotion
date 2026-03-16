# # # # # #!/usr/bin/env python3
# # # # # # -*- coding: utf-8 -*-
# # # # # """
# # # # # Eval 6 modes with:
# # # # # - Generate N (=num_gen, default 3) gen candidates per query group
# # # # # - Build labeled pools: gold / silver / neg (captions from CSV motion_caption if exists, else VQ -> caption model)
# # # # # - Ask Qwen to rank TOP-k among (gen + labeled)
# # # # # - Metrics:
# # # # #     (Ranker ability) Oracle nDCG@5 (gain: gold=2, silver=1, others=0)
# # # # #         - base: rank only labeled (no gen), compute nDCG@5 on FULL ranking
# # # # #         - withgen: rank gen + labeled, compute nDCG@5 on FULL ranking
# # # # #         - drop = withgen - base
# # # # #     (Gen quality)
# # # # #         - Gen-vs-Neg WinRate
# # # # #         - Gen-vs-Silver WinRate
# # # # #         - Gen-vs-Gold WinRate
# # # # #         - Gen@3 rate
# # # # #
# # # # # Notes / fixes vs your pasted version:
# # # # # - Prompt builder EXACTLY matches dataset/prompt_builder.py style (prints empty fields when disabled).
# # # # # - cond_mode parsing is explicit (no heuristic endswith).
# # # # # - Qwen output parsing is strict-first; if invalid/insufficient, auto-fix via build_fix_json_prompt.
# # # # # - nDCG@5 is computed from FULL ranking (topk + append missing in candidate order), not only top5 slice.
# # # # # - Removed unused PROMPT_V3 + build_source_text_v3 (avoid mismatch with training).
# # # # # - Output CSV keeps only essential columns (as you requested).
# # # # #
# # # # # Usage example:
# # # # # python eval_6modes.py \
# # # # #   --pairs_csv pairs_annotated.split.csv \
# # # # #   --dataset_dir /ibex/project/c2191/luoc/dataset/A2R \
# # # # #   --ckpt_map_json ckpt_map.json \
# # # # #   --caption_ckpt /path/to/caption_t5_ckpt \
# # # # #   --qwen_path /path/to/qwen_judge \
# # # # #   --audio_code_dir /ibex/project/c2191/luoc/dataset/A2R/audio-raws-09-01-2026-code \
# # # # #   --out_dir ./eval_6modes_out \
# # # # #   --k_eval 10 --num_gen 3 --max_total_candidates 30
# # # # # """
# # # # #
# # # # # import os
# # # # # import re
# # # # # import json
# # # # # import csv
# # # # # import math
# # # # # import argparse
# # # # # import random
# # # # # import hashlib
# # # # # from typing import Dict, List, Tuple, Optional, Any
# # # # #
# # # # # import numpy as np
# # # # # import pandas as pd
# # # # # from tqdm import tqdm
# # # # #
# # # # # import torch
# # # # # from transformers import (
# # # # #     T5Tokenizer,
# # # # #     T5ForConditionalGeneration,
# # # # #     AutoTokenizer,
# # # # #     AutoModelForCausalLM,
# # # # # )
# # # # #
# # # # # # =========================================================
# # # # # # Motion token parsing
# # # # # # =========================================================
# # # # # _MOTION_SPAN_RE = re.compile(r"<Motion Tokens>(.*?)</Motion Tokens>", re.DOTALL)
# # # # # _MOTION_TOKEN_RE = re.compile(r"<Motion Token\s+(\d+)>")
# # # # # _MOTION_TOKEN_SHORT_RE = re.compile(r"<(\d+)>")  # <123> short form
# # # # #
# # # # #
# # # # # def parse_motion_tokens(text: str, max_len: int = 200, codebook_size: int = 512) -> List[int]:
# # # # #     """Parse <Motion Tokens> ... </Motion Tokens> or short form <123>."""
# # # # #     if text is None:
# # # # #         return []
# # # # #     s = str(text)
# # # # #
# # # # #     m = _MOTION_SPAN_RE.search(s)
# # # # #     span = m.group(1) if m else s
# # # # #
# # # # #     codes = [int(x) for x in _MOTION_TOKEN_RE.findall(span)]
# # # # #     if len(codes) == 0:
# # # # #         codes = [int(x) for x in _MOTION_TOKEN_SHORT_RE.findall(span)]
# # # # #
# # # # #     out: List[int] = []
# # # # #     for c in codes:
# # # # #         if 0 <= c < codebook_size:
# # # # #             out.append(c)
# # # # #         else:
# # # # #             break
# # # # #     return out[:max_len]
# # # # #
# # # # #
# # # # # # =========================================================
# # # # # # Prompt builder (MATCH TRAINING EXACTLY)
# # # # # # reactmotion/dataset/prompt_builder.py::build_prompt
# # # # # # =========================================================
# # # # # def _parse_cond_mode(cond_mode: str) -> Tuple[bool, bool, bool]:
# # # # #     """
# # # # #     Modes:
# # # # #       - a       : audio only
# # # # #       - a+e     : audio + emotion
# # # # #       - t       : transcription only
# # # # #       - t+e     : transcription + emotion
# # # # #       - t+a     : transcription + audio
# # # # #       - t+a+e   : transcription + audio + emotion
# # # # #     """
# # # # #     cm = (cond_mode or "").strip().lower()
# # # # #     if cm not in {"a", "a+e", "t", "t+e", "t+a", "t+a+e"}:
# # # # #         raise ValueError(f"Unknown cond_mode={cond_mode}")
# # # # #     use_t = ("t" in cm)
# # # # #     use_a = ("a" in cm)
# # # # #     use_e = ("+e" in cm)
# # # # #     return use_t, use_a, use_e
# # # # #
# # # # #
# # # # # def build_prompt_condmode(
# # # # #     speaker_transcription: str,
# # # # #     speaker_audio: str,
# # # # #     speaker_emotion: str,
# # # # #     cond_mode: str,
# # # # # ) -> str:
# # # # #     """
# # # # #     EXACT formatting matches training prompt builder:
# # # # #       - always prints SPEAKER_TRANSCRIPTION line (empty if disabled)
# # # # #       - always prints SPEAKER_AUDIO line (empty if disabled)
# # # # #       - prints SPEAKER_EMOTION line only if use_emotion AND emotion is non-empty
# # # # #     """
# # # # #     use_transcription, use_audio, use_emotion = _parse_cond_mode(cond_mode)
# # # # #
# # # # #     t = (speaker_transcription or "").strip()
# # # # #     a = (speaker_audio or "").strip()
# # # # #     e = (speaker_emotion or "").strip()
# # # # #
# # # # #     lines = []
# # # # #     lines.append("You are modeling a speaker-listener dyadic interaction.\n\n")
# # # # #     lines.append("Input:\n")
# # # # #     lines.append(f"- SPEAKER_TRANSCRIPTION: {t if use_transcription else ''}\n")
# # # # #     lines.append(f"- SPEAKER_AUDIO: {a if use_audio else ''}\n")
# # # # #     if use_emotion and e:
# # # # #         lines.append(f"- SPEAKER_EMOTION: <Emotion> {e} </Emotion>\n")
# # # # #     lines.append("\nOutput:\n")
# # # # #     lines.append("Return ONLY a sequence of listener motion tokens in the exact format:\n")
# # # # #     lines.append("<Motion Tokens> <Motion Token i> ... </Motion Tokens>\n")
# # # # #     lines.append("Do NOT output any other words.\n")
# # # # #     return "".join(lines).strip()
# # # # #
# # # # #
# # # # # # =========================================================
# # # # # # JSON parsing / repair
# # # # # # =========================================================
# # # # # def extract_last_json_object(text: str) -> Optional[Dict[str, Any]]:
# # # # #     """Extract the last complete JSON object from text by brace matching."""
# # # # #     if text is None:
# # # # #         return None
# # # # #     s = str(text)
# # # # #     starts = [i for i, ch in enumerate(s) if ch == "{"]  # all
# # # # #     if not starts:
# # # # #         return None
# # # # #
# # # # #     last_obj = None
# # # # #     for st in starts:
# # # # #         depth = 0
# # # # #         in_str = False
# # # # #         esc = False
# # # # #         for i in range(st, len(s)):
# # # # #             c = s[i]
# # # # #             if in_str:
# # # # #                 if esc:
# # # # #                     esc = False
# # # # #                 elif c == "\\":
# # # # #                     esc = True
# # # # #                 elif c == '"':
# # # # #                     in_str = False
# # # # #             else:
# # # # #                 if c == '"':
# # # # #                     in_str = True
# # # # #                 elif c == "{":
# # # # #                     depth += 1
# # # # #                 elif c == "}":
# # # # #                     depth -= 1
# # # # #                     if depth == 0:
# # # # #                         blob = s[st : i + 1]
# # # # #                         try:
# # # # #                             obj = json.loads(blob)
# # # # #                             last_obj = obj
# # # # #                         except Exception:
# # # # #                             pass
# # # # #                         break
# # # # #     return last_obj
# # # # #
# # # # #
# # # # # def normalize_topk_ids(topk: Any) -> List[str]:
# # # # #     """
# # # # #     Extract Cxx ids from any object/list/string.
# # # # #     This is intentionally tolerant to Qwen including extra stuff.
# # # # #     """
# # # # #     if topk is None:
# # # # #         return []
# # # # #     if isinstance(topk, list):
# # # # #         s = " ".join([str(x) for x in topk])
# # # # #     else:
# # # # #         s = str(topk)
# # # # #     return re.findall(r"\bC\d{2}\b", s)
# # # # #
# # # # #
# # # # # def build_fix_json_prompt(raw_text: str, top_k: int) -> str:
# # # # #     key = f"top{int(top_k)}"
# # # # #     return (
# # # # #         "You are a strict JSON formatter.\n"
# # # # #         "Fix the output below to be a SINGLE valid JSON object and NOTHING ELSE.\n"
# # # # #         f"Return ONLY JSON with exactly one key: {key}\n"
# # # # #         f"- {key} must be an array of exactly {int(top_k)} unique ids like C01, C02, ...\n"
# # # # #         "No extra keys. No explanation. No markdown.\n\n"
# # # # #         "BAD OUTPUT:\n"
# # # # #         f"{raw_text}\n"
# # # # #     )
# # # # #
# # # # #
# # # # # # =========================================================
# # # # # # Qwen rank prompt
# # # # # # =========================================================
# # # # # def build_qwen_rank_prompt_uniform(
# # # # #     query_payload: Dict[str, Any],
# # # # #     candidates: List[Dict[str, Any]],
# # # # #     top_k: int,
# # # # # ) -> str:
# # # # #     k = int(top_k)
# # # # #     key = f"top{k}"
# # # # #     payload = {
# # # # #         "task": "Dyadic SPEAKER→LISTENER: rank LISTENER motion candidates as plausible non-verbal reactions to the SPEAKER utterance.",
# # # # #         "query": query_payload,
# # # # #         "candidates": candidates,
# # # # #         "rules": [
# # # # #             "Rank LISTENER reactions only (NOT speaker actions).",
# # # # #             "Use ONLY candidate captions as evidence; ids (C01..) contain no label info.",
# # # # #             f"Output ONLY JSON: {{{key}: [..]}} with exactly {k} unique ids, best first.",
# # # # #             "No extra text/keys/markdown.",
# # # # #         ],
# # # # #     }
# # # # #     return (
# # # # #         "You are an expert evaluator for listener reactive motions in dyadic conversations.\n"
# # # # #         "IMPORTANT: Output MUST be a single valid JSON object and NOTHING ELSE.\n"
# # # # #         "If you output any extra text outside JSON, the output is considered INVALID.\n"
# # # # #         f"{json.dumps(payload, ensure_ascii=False)}"
# # # # #     )
# # # # #
# # # # #
# # # # # # =========================================================
# # # # # # Candidate mixing
# # # # # # =========================================================
# # # # # def build_uniform_candidates(
# # # # #     gen_items: List[Dict[str, Any]],
# # # # #     gold_items: List[Dict[str, Any]],
# # # # #     silver_items: List[Dict[str, Any]],
# # # # #     neg_items: List[Dict[str, Any]],
# # # # #     seed: int,
# # # # #     max_total: int = 30,
# # # # # ) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, str]]:
# # # # #     """
# # # # #     Mix + rename to C01..; return:
# # # # #       candidates_for_llm: [{"cid":"C01","caption":"..."}...]
# # # # #       cid2orig: {"C01":"gen_0", ...}
# # # # #       orig2type: {"gen_0":"gen", "gold_0":"gold", "silver_2":"silver", "neg_5":"neg"}
# # # # #     """
# # # # #     all_items: List[Dict[str, str]] = []
# # # # #     orig2type: Dict[str, str] = {}
# # # # #
# # # # #     def _push(items: List[Dict[str, Any]], t: str):
# # # # #         for it in items:
# # # # #             oid = str(it["id"])
# # # # #             all_items.append({"orig_id": oid, "caption": str(it["caption"])})
# # # # #             orig2type[oid] = t
# # # # #
# # # # #     _push(gen_items, "gen")
# # # # #     _push(gold_items, "gold")
# # # # #     _push(silver_items, "silver")
# # # # #     _push(neg_items, "neg")
# # # # #
# # # # #     rng = random.Random(int(seed))
# # # # #     rng.shuffle(all_items)
# # # # #     all_items = all_items[: int(max_total)]
# # # # #
# # # # #     cid2orig: Dict[str, str] = {}
# # # # #     candidates: List[Dict[str, Any]] = []
# # # # #     for i, it in enumerate(all_items, start=1):
# # # # #         cid = f"C{i:02d}"
# # # # #         cid2orig[cid] = it["orig_id"]
# # # # #         candidates.append({"cid": cid, "caption": it["caption"]})
# # # # #     return candidates, cid2orig, orig2type
# # # # #
# # # # #
# # # # # # =========================================================
# # # # # # Qwen wrapper
# # # # # # =========================================================
# # # # # class QwenJudge:
# # # # #     def __init__(
# # # # #         self,
# # # # #         model_path: str,
# # # # #         use_vllm: bool = True,
# # # # #         tp: int = 1,
# # # # #         gpu_mem_util: float = 0.90,
# # # # #         max_new_tokens: int = 512,
# # # # #         temperature: float = 0.0,
# # # # #         max_model_len: int = 8192,
# # # # #     ):
# # # # #         self.model_path = model_path
# # # # #         self.use_vllm = use_vllm
# # # # #         self.tp = tp
# # # # #         self.gpu_mem_util = gpu_mem_util
# # # # #         self.max_new_tokens = max_new_tokens
# # # # #         self.temperature = temperature
# # # # #         self.max_model_len = max_model_len
# # # # #         self._mode = None
# # # # #
# # # # #         if use_vllm:
# # # # #             try:
# # # # #                 from vllm import LLM, SamplingParams  # type: ignore
# # # # #
# # # # #                 self._vllm_LLM = LLM(
# # # # #                     model=model_path,
# # # # #                     tensor_parallel_size=tp,
# # # # #                     gpu_memory_utilization=gpu_mem_util,
# # # # #                     trust_remote_code=True,
# # # # #                     max_model_len=max_model_len,
# # # # #                     enforce_eager=False,
# # # # #                 )
# # # # #                 self._vllm_SamplingParams = SamplingParams
# # # # #                 self._mode = "vllm"
# # # # #             except Exception as e:
# # # # #                 print(f"[WARN] vLLM init failed, fallback to HF. err={e}")
# # # # #                 self._mode = "hf"
# # # # #         else:
# # # # #             self._mode = "hf"
# # # # #
# # # # #         if self._mode == "hf":
# # # # #             self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# # # # #             self.model = AutoModelForCausalLM.from_pretrained(
# # # # #                 model_path,
# # # # #                 trust_remote_code=True,
# # # # #                 torch_dtype=torch.bfloat16,
# # # # #                 device_map="auto",
# # # # #             ).eval()
# # # # #
# # # # #     @torch.no_grad()
# # # # #     def generate(self, prompt: str) -> str:
# # # # #         if self._mode == "vllm":
# # # # #             sp = self._vllm_SamplingParams(
# # # # #                 temperature=self.temperature,
# # # # #                 max_tokens=self.max_new_tokens,
# # # # #                 top_p=1.0,
# # # # #             )
# # # # #             outs = self._vllm_LLM.generate([prompt], sp)
# # # # #             return outs[0].outputs[0].text.strip()
# # # # #
# # # # #         inputs = self.tok(prompt, return_tensors="pt")
# # # # #         inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
# # # # #         out = self.model.generate(
# # # # #             **inputs,
# # # # #             max_new_tokens=self.max_new_tokens,
# # # # #             do_sample=(self.temperature > 0),
# # # # #             temperature=max(1e-6, self.temperature),
# # # # #             top_p=1.0,
# # # # #         )
# # # # #         txt = self.tok.decode(out[0], skip_special_tokens=True)
# # # # #         if txt.startswith(prompt):
# # # # #             txt = txt[len(prompt) :]
# # # # #         return txt.strip()
# # # # #
# # # # #
# # # # # # =========================================================
# # # # # # Metrics
# # # # # # =========================================================
# # # # # def ndcg_at_k(relevances: List[int], k: int) -> float:
# # # # #     rel = relevances[:k]
# # # # #     dcg = 0.0
# # # # #     for i, r in enumerate(rel, start=1):
# # # # #         if r > 0:
# # # # #             dcg += (2.0**r - 1.0) / math.log2(i + 1.0)
# # # # #
# # # # #     ideal = sorted(relevances, reverse=True)[:k]
# # # # #     idcg = 0.0
# # # # #     for i, r in enumerate(ideal, start=1):
# # # # #         if r > 0:
# # # # #             idcg += (2.0**r - 1.0) / math.log2(i + 1.0)
# # # # #
# # # # #     if idcg <= 1e-12:
# # # # #         return 0.0
# # # # #     return float(dcg / idcg)
# # # # #
# # # # #
# # # # # def gain_by_type(t: str) -> int:
# # # # #     if t == "gold":
# # # # #         return 2
# # # # #     if t == "silver":
# # # # #         return 1
# # # # #     return 0  # neg/gen/unk
# # # # #
# # # # #
# # # # # def build_all_orig_from_candidates(candidates: List[Dict[str, Any]], cid2orig: Dict[str, str]) -> List[str]:
# # # # #     out: List[str] = []
# # # # #     for it in candidates:
# # # # #         cid = it["cid"]
# # # # #         if cid in cid2orig:
# # # # #             out.append(cid2orig[cid])
# # # # #     return out
# # # # #
# # # # #
# # # # # def build_full_ranking(topk_orig: List[str], all_orig: List[str]) -> List[str]:
# # # # #     ranked = list(topk_orig)
# # # # #     for oid in all_orig:
# # # # #         if oid not in ranked:
# # # # #             ranked.append(oid)
# # # # #     return ranked
# # # # #
# # # # #
# # # # # def oracle_ndcg_at_k_from_full_ranking(ranked_full: List[str], orig2type: Dict[str, str], k: int) -> float:
# # # # #     rel = [gain_by_type(orig2type.get(oid, "unk")) for oid in ranked_full]
# # # # #     return ndcg_at_k(rel, k=k)
# # # # #
# # # # #
# # # # # def winrate_best_gen_vs_type(ranked_orig_ids: List[str], orig2type: Dict[str, str], comp_type: str) -> float:
# # # # #     rank = {oid: i for i, oid in enumerate(ranked_orig_ids)}
# # # # #     gen_ids = [oid for oid in ranked_orig_ids if orig2type.get(oid) == "gen"]
# # # # #     comp_ids = [oid for oid in ranked_orig_ids if orig2type.get(oid) == comp_type]
# # # # #     if len(gen_ids) == 0 or len(comp_ids) == 0:
# # # # #         return float("nan")
# # # # #     best_gen_rank = min(rank[oid] for oid in gen_ids)
# # # # #     wins = sum(1 for oid in comp_ids if best_gen_rank < rank[oid])
# # # # #     return float(wins / len(comp_ids))
# # # # #
# # # # #
# # # # # def gen_at_k_rate(ranked_orig_ids: List[str], orig2type: Dict[str, str], k: int = 3) -> float:
# # # # #     topk = ranked_orig_ids[:k]
# # # # #     return 1.0 if any(orig2type.get(oid) == "gen" for oid in topk) else 0.0
# # # # #
# # # # #
# # # # # # =========================================================
# # # # # # IO / utilities
# # # # # # =========================================================
# # # # # def ensure_parent(path: str):
# # # # #     os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
# # # # #
# # # # #
# # # # # def hash_file(path: str) -> str:
# # # # #     h = hashlib.md5()
# # # # #     with open(path, "rb") as f:
# # # # #         while True:
# # # # #             b = f.read(1024 * 1024)
# # # # #             if not b:
# # # # #                 break
# # # # #             h.update(b)
# # # # #     return h.hexdigest()[:10]
# # # # #
# # # # #
# # # # # def stable_hash_str(s: str) -> str:
# # # # #     return hashlib.md5(str(s).encode("utf-8")).hexdigest()[:10]
# # # # #
# # # # #
# # # # # def make_run_tag(args: argparse.Namespace) -> str:
# # # # #     parts = [
# # # # #         f"numgen{args.num_gen}",
# # # # #         f"keval{args.k_eval}",
# # # # #         f"maxtotal{args.max_total_candidates}",
# # # # #         f"cap{stable_hash_str(args.caption_ckpt)}",
# # # # #         f"qwen{stable_hash_str(args.qwen_path)}",
# # # # #         f"ckmap{hash_file(args.ckpt_map_json)}",
# # # # #     ]
# # # # #     return "_".join(parts)
# # # # #
# # # # #
# # # # # def motion_id_from_raw(raw_file_name: str) -> str:
# # # # #     # "000267_1_..." -> "000267"
# # # # #     s = str(raw_file_name)
# # # # #     mid = s.split("_", 1)[0]
# # # # #     return str(mid).zfill(6)
# # # # #
# # # # #
# # # # # def build_vq_index(vq_dir: str) -> Dict[str, str]:
# # # # #     m: Dict[str, str] = {}
# # # # #     for fn in os.listdir(vq_dir):
# # # # #         if fn.endswith(".npy"):
# # # # #             stem = os.path.splitext(fn)[0]
# # # # #             m[stem] = os.path.join(vq_dir, fn)
# # # # #     return m
# # # # #
# # # # #
# # # # # def vqvae_lookup(vq_by_stem: Dict[str, str], motion_id: str) -> Optional[str]:
# # # # #     base = str(motion_id)
# # # # #     if base in vq_by_stem:
# # # # #         return vq_by_stem[base]
# # # # #     if base.isdigit() and ("M" + base) in vq_by_stem:
# # # # #         return vq_by_stem["M" + base]
# # # # #     if base.startswith("M") and base[1:].isdigit() and (base[1:] in vq_by_stem):
# # # # #         return vq_by_stem[base[1:]]
# # # # #     return None
# # # # #
# # # # #
# # # # # def load_motion_codes_from_vq(vq_path: str, codebook_size: int = 512) -> List[int]:
# # # # #     arr = np.load(vq_path, allow_pickle=False)
# # # # #     arr = np.asarray(arr).reshape(-1).tolist()
# # # # #     out: List[int] = []
# # # # #     for x in arr:
# # # # #         try:
# # # # #             c = int(x)
# # # # #         except Exception:
# # # # #             continue
# # # # #         if 0 <= c < codebook_size:
# # # # #             out.append(c)
# # # # #     return out
# # # # #
# # # # #
# # # # # def build_caption_prompt(motion_codes: List[int]) -> str:
# # # # #     motion_string = "<Motion Tokens>" + "".join([f"<{c}>" for c in motion_codes]) + "</Motion Tokens>"
# # # # #     return "Generate text: " + motion_string
# # # # #
# # # # #
# # # # # def load_audio_tokens_any(path: str) -> np.ndarray:
# # # # #     obj = np.load(path, allow_pickle=False)
# # # # #     if isinstance(obj, np.lib.npyio.NpzFile):
# # # # #         if "codes" in obj.files:
# # # # #             arr = obj["codes"]
# # # # #         else:
# # # # #             arr = obj[obj.files[0]]
# # # # #         obj.close()
# # # # #         return arr
# # # # #     return obj
# # # # #
# # # # #
# # # # # def format_audio_tokens(a_tokens: np.ndarray, level: str = "base") -> str:
# # # # #     """
# # # # #     a_tokens: [L,T] or [T]
# # # # #     level: base/all/rand
# # # # #     """
# # # # #     level = str(level)
# # # # #     arr = np.array(a_tokens)
# # # # #
# # # # #     if arr.ndim == 1:
# # # # #         parts = ["<Audio Tokens>"]
# # # # #         for t in arr.reshape(-1):
# # # # #             parts.append(f"<Audio Token {int(t)}>")
# # # # #         parts.append("</Audio Tokens>")
# # # # #         return " ".join(parts)
# # # # #
# # # # #     L = int(arr.shape[0])
# # # # #     parts = ["<Audio Tokens>"]
# # # # #
# # # # #     if level == "base":
# # # # #         for t in arr[0].reshape(-1):
# # # # #             parts.append(f"<Audio Level 0 Token {int(t)}>")
# # # # #     elif level == "all":
# # # # #         for i in range(L):
# # # # #             for t in arr[i].reshape(-1):
# # # # #                 parts.append(f"<Audio Level {i} Token {int(t)}>")
# # # # #     elif level == "rand":
# # # # #         k = int(np.random.choice(np.arange(1, L + 1)))
# # # # #         for i in range(k):
# # # # #             for t in arr[i].reshape(-1):
# # # # #                 parts.append(f"<Audio Level {i} Token {int(t)}>")
# # # # #     else:
# # # # #         raise ValueError(f"Unknown audio_token_level={level}")
# # # # #
# # # # #     parts.append("</Audio Tokens>")
# # # # #     return " ".join(parts)
# # # # #
# # # # #
# # # # # def pick_code_from_stem(code_dir: str, stem: str) -> Optional[str]:
# # # # #     stem = str(stem).strip()
# # # # #     if not stem:
# # # # #         return None
# # # # #     p_npz = os.path.join(code_dir, stem + ".npz")
# # # # #     if os.path.exists(p_npz):
# # # # #         return p_npz
# # # # #     p_npy = os.path.join(code_dir, stem + ".npy")
# # # # #     if os.path.exists(p_npy):
# # # # #         return p_npy
# # # # #     return None
# # # # #
# # # # #
# # # # # # =========================================================
# # # # # # Main
# # # # # # =========================================================
# # # # # def main():
# # # # #     ap = argparse.ArgumentParser()
# # # # #
# # # # #     ap.add_argument("--pairs_csv", type=str, required=True)
# # # # #     ap.add_argument("--dataset_dir", type=str, required=True)
# # # # #     ap.add_argument("--ckpt_map_json", type=str, required=True, help="json mapping cond_mode -> a2rm_ckpt_dir")
# # # # #
# # # # #     # audio (for modes containing 'a')
# # # # #     ap.add_argument("--audio_code_dir", type=str, default=None)
# # # # #     ap.add_argument("--audio_token_level", type=str, default="base", choices=["base", "all", "rand"])
# # # # #
# # # # #     # caption model
# # # # #     ap.add_argument("--caption_ckpt", type=str, required=True)
# # # # #
# # # # #     # generation
# # # # #     ap.add_argument("--num_gen", type=int, default=3)
# # # # #     ap.add_argument("--gen_max_len", type=int, default=200)
# # # # #     ap.add_argument("--gen_temperature", type=float, default=0.8)
# # # # #     ap.add_argument("--gen_top_k", type=int, default=200)
# # # # #
# # # # #     # candidates cap
# # # # #     ap.add_argument("--max_gold", type=int, default=1)
# # # # #     ap.add_argument("--max_silver", type=int, default=8)
# # # # #     ap.add_argument("--max_neg", type=int, default=12)
# # # # #     ap.add_argument("--max_total_candidates", type=int, default=30)
# # # # #
# # # # #     # eval
# # # # #     ap.add_argument("--only_test", action="store_true", default=True)
# # # # #     ap.add_argument("--split_name", type=str, default="test")
# # # # #     ap.add_argument("--group_by", type=str, default="group_id", choices=["group_id", "sayings_emotion"])
# # # # #     ap.add_argument("--k_eval", type=int, default=10)
# # # # #
# # # # #     # qwen
# # # # #     ap.add_argument("--qwen_path", type=str, required=True)
# # # # #     ap.add_argument("--qwen_use_vllm", action="store_true")
# # # # #     ap.add_argument("--qwen_tp", type=int, default=1)
# # # # #     ap.add_argument("--qwen_gpu_mem_util", type=float, default=0.90)
# # # # #     ap.add_argument("--qwen_max_new_tokens", type=int, default=512)
# # # # #     ap.add_argument("--qwen_max_model_len", type=int, default=8192)
# # # # #
# # # # #     # out
# # # # #     ap.add_argument("--out_dir", type=str, default="./eval_6modes_out")
# # # # #     ap.add_argument("--seed", type=int, default=42)
# # # # #
# # # # #     args = ap.parse_args()
# # # # #
# # # # #     random.seed(args.seed)
# # # # #     np.random.seed(args.seed)
# # # # #     torch.manual_seed(args.seed)
# # # # #
# # # # #     os.makedirs(args.out_dir, exist_ok=True)
# # # # #     device = "cuda" if torch.cuda.is_available() else "cpu"
# # # # #     print("[Device]", device)
# # # # #
# # # # #     # load ckpt map
# # # # #     with open(args.ckpt_map_json, "r", encoding="utf-8") as f:
# # # # #         ckpt_map = json.load(f)
# # # # #
# # # # #     modes = ["a", "a+e", "t", "t+e", "t+a", "t+a+e"]
# # # # #     for m in modes:
# # # # #         if m not in ckpt_map:
# # # # #             raise RuntimeError(f"ckpt_map_json missing mode: {m}")
# # # # #
# # # # #     # load pairs
# # # # #     df = pd.read_csv(args.pairs_csv, encoding="utf-8")
# # # # #     required_cols = ["sayings", "emotion", "label", "raw_file_name", "split", "generated_wav_name"]
# # # # #     missing = [c for c in required_cols if c not in df.columns]
# # # # #     if missing:
# # # # #         raise RuntimeError(f"Missing columns in pairs_csv: {missing}")
# # # # #
# # # # #     df["label"] = df["label"].astype(str).str.lower().str.strip()
# # # # #     df["sayings"] = df["sayings"].astype(str).fillna("")
# # # # #     df["emotion"] = df["emotion"].astype(str).fillna("")
# # # # #     df["raw_file_name"] = df["raw_file_name"].astype(str).fillna("")
# # # # #     df["generated_wav_name"] = df["generated_wav_name"].astype(str).fillna("")
# # # # #     df["split"] = df["split"].astype(str).str.lower().str.strip()
# # # # #
# # # # #     if args.only_test:
# # # # #         df_eval = df[df["split"] == args.split_name].copy()
# # # # #     else:
# # # # #         df_eval = df.copy()
# # # # #
# # # # #     if len(df_eval) == 0:
# # # # #         raise RuntimeError(f"No rows for split={args.split_name} in pairs_csv")
# # # # #
# # # # #     # grouping
# # # # #     if args.group_by == "group_id":
# # # # #         if "group_id" not in df_eval.columns:
# # # # #             print("[WARN] group_by=group_id but no group_id column; fallback to sayings_emotion")
# # # # #             args.group_by = "sayings_emotion"
# # # # #
# # # # #     if args.group_by == "group_id":
# # # # #         groups = list(df_eval.groupby(["group_id"], dropna=False))
# # # # #     else:
# # # # #         groups = list(df_eval.groupby(["sayings", "emotion"], dropna=False))
# # # # #
# # # # #     print("[Groups]", len(groups), "group_by=", args.group_by)
# # # # #
# # # # #     # vq index
# # # # #     motion_vq_dir = os.path.join(args.dataset_dir, "HumanML3D", "VQVAE")
# # # # #     if not os.path.isdir(motion_vq_dir):
# # # # #         raise RuntimeError(f"Missing motion_vq_dir: {motion_vq_dir}")
# # # # #     vq_by_stem = build_vq_index(motion_vq_dir)
# # # # #     print("[VQ] indexed:", len(vq_by_stem))
# # # # #
# # # # #     # audio code dir
# # # # #     audio_code_dir = args.audio_code_dir or os.path.join(args.dataset_dir, "audio-raws-09-01-2026-code")
# # # # #     if not os.path.isdir(audio_code_dir):
# # # # #         print(f"[WARN] audio_code_dir not found: {audio_code_dir} (a-modes may have empty audio)")
# # # # #
# # # # #     # caption model
# # # # #     cap_tok = T5Tokenizer.from_pretrained(args.caption_ckpt)
# # # # #     cap_model = T5ForConditionalGeneration.from_pretrained(args.caption_ckpt).to(device).eval()
# # # # #
# # # # #     @torch.no_grad()
# # # # #     def caption_motion_codes(codes: List[int]) -> str:
# # # # #         prompt = build_caption_prompt(codes)
# # # # #         inp = cap_tok(prompt, return_tensors="pt").input_ids.to(device, dtype=torch.long)
# # # # #         out = cap_model.generate(inp, max_length=200, num_beams=1, do_sample=False)
# # # # #         txt = cap_tok.decode(out[0], skip_special_tokens=True).strip().strip('"')
# # # # #         return txt
# # # # #
# # # # #     # qwen
# # # # #     qwen = QwenJudge(
# # # # #         model_path=args.qwen_path,
# # # # #         use_vllm=args.qwen_use_vllm,
# # # # #         tp=args.qwen_tp,
# # # # #         gpu_mem_util=args.qwen_gpu_mem_util,
# # # # #         max_new_tokens=args.qwen_max_new_tokens,
# # # # #         temperature=0.0,
# # # # #         max_model_len=args.qwen_max_model_len,
# # # # #     )
# # # # #     print("[Qwen] enabled:", args.qwen_path, "mode=", ("vllm" if args.qwen_use_vllm else "hf"))
# # # # #
# # # # #     # output csv unique name per run
# # # # #     run_tag = make_run_tag(args)
# # # # #     out_csv = os.path.join(args.out_dir, f"eval_6modes_{run_tag}.csv")
# # # # #     ensure_parent(out_csv)
# # # # #
# # # # #     # minimal header only
# # # # #     fieldnames = [
# # # # #         "eval_key",
# # # # #         "mode",
# # # # #         "split",
# # # # #         "group_key",
# # # # #         "sayings",
# # # # #         "emotion",
# # # # #         "audio_code_path",
# # # # #         "num_gen",
# # # # #         "num_gold",
# # # # #         "num_silver",
# # # # #         "num_neg",
# # # # #         "k_eval",
# # # # #         "oracle_ndcg5_base",
# # # # #         "oracle_ndcg5_withgen",
# # # # #         "oracle_drop_ndcg5",
# # # # #         "win_gen_vs_neg",
# # # # #         "win_gen_vs_silver",
# # # # #         "win_gen_vs_gold",
# # # # #         "gen_at3",
# # # # #         "topk_orig_json",
# # # # #         "topk_types_json",
# # # # #     ]
# # # # #
# # # # #     exists = os.path.isfile(out_csv)
# # # # #     fcsv = open(out_csv, "a", encoding="utf-8", newline="")
# # # # #     writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
# # # # #     if not exists:
# # # # #         writer.writeheader()
# # # # #         fcsv.flush()
# # # # #         os.fsync(fcsv.fileno())
# # # # #
# # # # #     # cache a2rm per mode
# # # # #     a2rm_models: Dict[str, Any] = {}
# # # # #     a2rm_toks: Dict[str, Any] = {}
# # # # #
# # # # #     def get_a2rm_model(mode: str):
# # # # #         if mode in a2rm_models:
# # # # #             return a2rm_toks[mode], a2rm_models[mode]
# # # # #         ckpt = ckpt_map[mode]
# # # # #         tok = T5Tokenizer.from_pretrained(ckpt)
# # # # #         model = T5ForConditionalGeneration.from_pretrained(ckpt).to(device).eval()
# # # # #         a2rm_toks[mode] = tok
# # # # #         a2rm_models[mode] = model
# # # # #         print(f"[Load A2RM] mode={mode} ckpt={ckpt}")
# # # # #         return tok, model
# # # # #
# # # # #     def canon_label(x: str) -> str:
# # # # #         s = (x or "").strip().lower()
# # # # #         if s in {"gold", "pos", "positive", "gt", "true", "1"}:
# # # # #             return "gold"
# # # # #         if s in {"silver"}:
# # # # #             return "silver"
# # # # #         if s in {"neg", "negative", "0"}:
# # # # #             return "neg"
# # # # #         return s
# # # # #
# # # # #     def make_eval_key(mode: str, group_key: str, audio_code_path: str) -> str:
# # # # #         s = f"{mode}|||{group_key}|||{audio_code_path}"
# # # # #         return hashlib.md5(s.encode("utf-8")).hexdigest()
# # # # #
# # # # #     def resolve_audio_for_group(g: pd.DataFrame) -> Tuple[str, str]:
# # # # #         stems = [str(x).strip() for x in g["generated_wav_name"].tolist() if str(x).strip()]
# # # # #         stems = list(dict.fromkeys(stems))
# # # # #         if (not stems) or (not os.path.isdir(audio_code_dir)):
# # # # #             return "", ""
# # # # #         stem = random.choice(stems)
# # # # #         p = pick_code_from_stem(audio_code_dir, stem)
# # # # #         if p is None:
# # # # #             return "", ""
# # # # #         codes = load_audio_tokens_any(p)
# # # # #         return format_audio_tokens(codes, level=args.audio_token_level), p
# # # # #
# # # # #     def qwen_rank_topk(
# # # # #         query_payload: Dict[str, Any],
# # # # #         candidates: List[Dict[str, Any]],
# # # # #         cid2orig: Dict[str, str],
# # # # #         top_k: int,
# # # # #     ) -> List[str]:
# # # # #         """
# # # # #         Returns list of Cxx length=top_k (filled if missing), all in cid2orig.
# # # # #         Uses strict JSON parse, else fix prompt, else fallback extraction.
# # # # #         """
# # # # #         k = int(top_k)
# # # # #         key = f"top{k}"
# # # # #         prompt = build_qwen_rank_prompt_uniform(query_payload, candidates, top_k=k)
# # # # #         raw = qwen.generate(prompt)
# # # # #
# # # # #         obj = extract_last_json_object(raw)
# # # # #         if obj is None or key not in obj:
# # # # #             # try fix-json once
# # # # #             fix = build_fix_json_prompt(raw, top_k=k)
# # # # #             raw2 = qwen.generate(fix)
# # # # #             obj = extract_last_json_object(raw2)
# # # # #
# # # # #         top_ids = []
# # # # #         if isinstance(obj, dict) and (key in obj):
# # # # #             top_ids = normalize_topk_ids(obj.get(key))
# # # # #         if len(top_ids) < k:
# # # # #             # fallback: extract from raw text
# # # # #             top_ids = normalize_topk_ids(raw)
# # # # #
# # # # #         # keep only valid + unique
# # # # #         seen = set()
# # # # #         out = []
# # # # #         for c in top_ids:
# # # # #             if c in cid2orig and c not in seen:
# # # # #                 seen.add(c)
# # # # #                 out.append(c)
# # # # #             if len(out) >= k:
# # # # #                 break
# # # # #
# # # # #         # force fill to k by candidate order
# # # # #         if len(out) < k:
# # # # #             for it in candidates:
# # # # #                 c = it["cid"]
# # # # #                 if c in cid2orig and c not in seen:
# # # # #                     seen.add(c)
# # # # #                     out.append(c)
# # # # #                 if len(out) >= k:
# # # # #                     break
# # # # #
# # # # #         return out[:k]
# # # # #
# # # # #     # main loop
# # # # #     for mode in modes:
# # # # #         tok, model = get_a2rm_model(mode)
# # # # #
# # # # #         for keys, g in tqdm(groups, desc=f"Eval[{mode}]"):
# # # # #             if args.group_by == "group_id":
# # # # #                 group_key = str(keys) if not isinstance(keys, tuple) else str(keys[0])
# # # # #                 sayings = str(g["sayings"].iloc[0])
# # # # #                 emotion = str(g["emotion"].iloc[0])
# # # # #             else:
# # # # #                 sayings, emotion = keys
# # # # #                 sayings = str(sayings)
# # # # #                 emotion = str(emotion)
# # # # #                 group_key = f"{sayings}|||{emotion}"
# # # # #
# # # # #             split = str(g["split"].iloc[0])
# # # # #
# # # # #             gg_all = g.copy()
# # # # #             gg_all["label_canon"] = gg_all["label"].apply(canon_label)
# # # # #             gg_all["motion_id"] = gg_all["raw_file_name"].apply(motion_id_from_raw)
# # # # #
# # # # #             use_csv_caption = ("motion_caption" in gg_all.columns)
# # # # #
# # # # #             def build_items(lbl: str, max_n: int) -> List[Dict[str, Any]]:
# # # # #                 sub = gg_all[gg_all["label_canon"] == lbl].copy()
# # # # #                 if len(sub) == 0:
# # # # #                     return []
# # # # #                 mids = list(dict.fromkeys(sub["motion_id"].tolist()))[: int(max_n)]
# # # # #
# # # # #                 items: List[Dict[str, Any]] = []
# # # # #                 for i, mid in enumerate(mids):
# # # # #                     cap = ""
# # # # #                     if use_csv_caption:
# # # # #                         ss = sub[sub["motion_id"] == mid]
# # # # #                         if "motion_caption" in ss.columns and len(ss) > 0:
# # # # #                             cap = str(ss["motion_caption"].iloc[0]).strip()
# # # # #
# # # # #                     if not cap:
# # # # #                         p = vqvae_lookup(vq_by_stem, mid)
# # # # #                         if p is None:
# # # # #                             continue
# # # # #                         codes = load_motion_codes_from_vq(p)
# # # # #                         cap = caption_motion_codes(codes)
# # # # #
# # # # #                     if cap:
# # # # #                         items.append({"id": f"{lbl}_{i}", "caption": cap})
# # # # #                 return items
# # # # #
# # # # #             gold_items = build_items("gold", args.max_gold)
# # # # #             silver_items = build_items("silver", args.max_silver)
# # # # #             neg_items = build_items("neg", args.max_neg)
# # # # #
# # # # #             # audio for a-modes
# # # # #             audio_text = ""
# # # # #             audio_code_path = ""
# # # # #             if "a" in mode:
# # # # #                 audio_text, audio_code_path = resolve_audio_for_group(g)
# # # # #
# # # # #             eval_key = make_eval_key(mode, group_key, audio_code_path)
# # # # #
# # # # #             # if no labeled at all, write NaN row
# # # # #             if (len(gold_items) + len(silver_items) + len(neg_items)) == 0:
# # # # #                 writer.writerow(
# # # # #                     dict(
# # # # #                         eval_key=eval_key,
# # # # #                         mode=mode,
# # # # #                         split=split,
# # # # #                         group_key=group_key,
# # # # #                         sayings=sayings,
# # # # #                         emotion=emotion,
# # # # #                         audio_code_path=audio_code_path,
# # # # #                         num_gen=0,
# # # # #                         num_gold=0,
# # # # #                         num_silver=0,
# # # # #                         num_neg=0,
# # # # #                         k_eval=min(int(args.k_eval), 0),
# # # # #                         oracle_ndcg5_base=float("nan"),
# # # # #                         oracle_ndcg5_withgen=float("nan"),
# # # # #                         oracle_drop_ndcg5=float("nan"),
# # # # #                         win_gen_vs_neg=float("nan"),
# # # # #                         win_gen_vs_silver=float("nan"),
# # # # #                         win_gen_vs_gold=float("nan"),
# # # # #                         gen_at3=float("nan"),
# # # # #                         topk_orig_json="[]",
# # # # #                         topk_types_json="[]",
# # # # #                     )
# # # # #                 )
# # # # #                 fcsv.flush()
# # # # #                 os.fsync(fcsv.fileno())
# # # # #                 continue
# # # # #
# # # # #             query_payload = {
# # # # #                 "role_definition": {
# # # # #                     "speaker": "the person who said the utterance below",
# # # # #                     "listener": "the person whose motion candidates we are ranking",
# # # # #                 },
# # # # #                 "speaker_sayings": sayings,
# # # # #                 "speaker_emotion": emotion,
# # # # #                 "cond_mode": mode,
# # # # #             }
# # # # #
# # # # #             seed_int = int(hashlib.md5(f"{group_key}|||{mode}".encode("utf-8")).hexdigest()[:8], 16)
# # # # #
# # # # #             # ----------------------------
# # # # #             # 1) ORACLE BASE: labeled-only, TOP-5, compute FULL-ranking nDCG@5
# # # # #             # ----------------------------
# # # # #             cand_base, cid2orig_base, orig2type_base = build_uniform_candidates(
# # # # #                 gen_items=[],
# # # # #                 gold_items=gold_items,
# # # # #                 silver_items=silver_items,
# # # # #                 neg_items=neg_items,
# # # # #                 seed=seed_int ^ 0xBADC0DE,
# # # # #                 max_total=args.max_total_candidates,
# # # # #             )
# # # # #
# # # # #             top5_cids_base = qwen_rank_topk(query_payload, cand_base, cid2orig_base, top_k=5)
# # # # #             top5_orig_base = [cid2orig_base[c] for c in top5_cids_base if c in cid2orig_base]
# # # # #
# # # # #             all_orig_base = build_all_orig_from_candidates(cand_base, cid2orig_base)
# # # # #             ranked_base_full = build_full_ranking(top5_orig_base, all_orig_base)
# # # # #             oracle_ndcg5_base = oracle_ndcg_at_k_from_full_ranking(ranked_base_full, orig2type_base, k=5)
# # # # #
# # # # #             # ----------------------------
# # # # #             # 2) GENERATE gen candidates
# # # # #             # ----------------------------
# # # # #             input_text = build_prompt_condmode(
# # # # #                 speaker_transcription=sayings,
# # # # #                 speaker_audio=audio_text,
# # # # #                 speaker_emotion=emotion,
# # # # #                 cond_mode=mode,
# # # # #             )
# # # # #             input_ids = tok(input_text, return_tensors="pt").input_ids.to(device, dtype=torch.long)
# # # # #
# # # # #             gen_items: List[Dict[str, Any]] = []
# # # # #             for ci in range(int(args.num_gen)):
# # # # #                 out = model.generate(
# # # # #                     input_ids,
# # # # #                     max_length=256,
# # # # #                     do_sample=True,
# # # # #                     temperature=args.gen_temperature,
# # # # #                     top_k=args.gen_top_k,
# # # # #                 )
# # # # #                 out_text = tok.decode(out[0], skip_special_tokens=False)
# # # # #                 out_text = out_text.replace("<pad>", "").replace("</s>", "").strip()
# # # # #                 codes = parse_motion_tokens(out_text, max_len=args.gen_max_len, codebook_size=512)
# # # # #                 if len(codes) == 0:
# # # # #                     codes = [1] * min(int(args.gen_max_len), 196)
# # # # #                 cap = caption_motion_codes(codes)
# # # # #                 gen_items.append({"id": f"gen_{ci}", "caption": cap})
# # # # #
# # # # #             # ----------------------------
# # # # #             # 3) WITH-GEN ranking: TOP-k_eval, compute FULL-ranking nDCG@5 + gen metrics
# # # # #             # ----------------------------
# # # # #             cand_all, cid2orig, orig2type = build_uniform_candidates(
# # # # #                 gen_items=gen_items,
# # # # #                 gold_items=gold_items,
# # # # #                 silver_items=silver_items,
# # # # #                 neg_items=neg_items,
# # # # #                 seed=seed_int,
# # # # #                 max_total=args.max_total_candidates,
# # # # #             )
# # # # #
# # # # #             k_eval = int(args.k_eval)
# # # # #             k_eval = min(k_eval, len(cand_all))
# # # # #             if k_eval <= 0:
# # # # #                 k_eval = min(5, len(cand_all))
# # # # #
# # # # #             topk_cids = qwen_rank_topk(query_payload, cand_all, cid2orig, top_k=k_eval)
# # # # #             topk_orig = [cid2orig[c] for c in topk_cids if c in cid2orig]
# # # # #             topk_types = [orig2type.get(oid, "unk") for oid in topk_orig]
# # # # #
# # # # #             all_orig_all = build_all_orig_from_candidates(cand_all, cid2orig)
# # # # #             ranked_all_full = build_full_ranking(topk_orig, all_orig_all)
# # # # #
# # # # #             oracle_ndcg5_withgen = oracle_ndcg_at_k_from_full_ranking(ranked_all_full, orig2type, k=5)
# # # # #             oracle_drop_ndcg5 = oracle_ndcg5_withgen - oracle_ndcg5_base
# # # # #
# # # # #             # gen quality
# # # # #             win_gen_vs_neg = winrate_best_gen_vs_type(ranked_all_full, orig2type, "neg")
# # # # #             win_gen_vs_silver = winrate_best_gen_vs_type(ranked_all_full, orig2type, "silver")
# # # # #             win_gen_vs_gold = winrate_best_gen_vs_type(ranked_all_full, orig2type, "gold")
# # # # #             gen_at3 = gen_at_k_rate(ranked_all_full, orig2type, k=3)
# # # # #
# # # # #             writer.writerow(
# # # # #                 dict(
# # # # #                     eval_key=eval_key,
# # # # #                     mode=mode,
# # # # #                     split=split,
# # # # #                     group_key=group_key,
# # # # #                     sayings=sayings,
# # # # #                     emotion=emotion,
# # # # #                     audio_code_path=audio_code_path,
# # # # #                     num_gen=len(gen_items),
# # # # #                     num_gold=len(gold_items),
# # # # #                     num_silver=len(silver_items),
# # # # #                     num_neg=len(neg_items),
# # # # #                     k_eval=k_eval,
# # # # #                     oracle_ndcg5_base=oracle_ndcg5_base,
# # # # #                     oracle_ndcg5_withgen=oracle_ndcg5_withgen,
# # # # #                     oracle_drop_ndcg5=oracle_drop_ndcg5,
# # # # #                     win_gen_vs_neg=win_gen_vs_neg,
# # # # #                     win_gen_vs_silver=win_gen_vs_silver,
# # # # #                     win_gen_vs_gold=win_gen_vs_gold,
# # # # #                     gen_at3=gen_at3,
# # # # #                     topk_orig_json=json.dumps(topk_orig, ensure_ascii=False),
# # # # #                     topk_types_json=json.dumps(topk_types, ensure_ascii=False),
# # # # #                 )
# # # # #             )
# # # # #             fcsv.flush()
# # # # #             os.fsync(fcsv.fileno())
# # # # #
# # # # #     fcsv.close()
# # # # #     print("[Saved]", out_csv)
# # # # #
# # # # #
# # # # # if __name__ == "__main__":
# # # # #     main()
# # # #
# # # #
# # # # #!/usr/bin/env python3
# # # # # -*- coding: utf-8 -*-
# # # # """
# # # # Pairwise Eval (6 modes) with Bradley–Terry (BTL)
# # # #
# # # # What it does
# # # # ============
# # # # For each (mode, group):
# # # #   1) Build labeled pool: gold / silver / neg captions
# # # #      - Use CSV column `motion_caption` if exists and non-empty
# # # #      - Else: load VQ tokens -> caption model (T5) to get caption
# # # #   2) Generate N (=num_gen) gen candidates using A2RM ckpt for this mode
# # # #      - parse motion tokens -> caption via caption model
# # # #   3) Rank with pair-wise Qwen judge:
# # # #      - Compare pairs of candidates by caption only (dyadic speaker->listener relevance)
# # # #      - Collect pairwise outcomes
# # # #      - Fit Bradley–Terry scores -> global ranking order
# # # #   4) Metrics:
# # # #      (Ranker ability) Oracle nDCG@5 (gold-only gain=2, others=0)
# # # #         - base: BTL ranking on labeled-only pool
# # # #         - withgen: BTL ranking on full pool (gen+labeled)
# # # #         - drop = withgen - base
# # # #      (Gen quality) computed on FULL BTL ranking:
# # # #         - Gen-vs-Neg WinRate (best gen beats each neg)
# # # #         - Gen-vs-Silver WinRate
# # # #         - Gen-vs-Gold WinRate
# # # #         - Gen@3 rate (top3 contains any gen)
# # # #
# # # # Input assumptions
# # # # =================
# # # # pairs_csv must have columns:
# # # #   - sayings, emotion, label, raw_file_name, split, generated_wav_name
# # # # Optional:
# # # #   - motion_caption
# # # #
# # # # Audio codes (for modes containing 'a') are resolved via generated_wav_name stem:
# # # #   audio_code_dir/{stem}.npz or .npy
# # # #
# # # # ckpt_map_json maps cond_mode -> a2rm checkpoint dir (HF format):
# # # # {
# # # #   "a": ".../checkpoint-155000",
# # # #   "a+e": "...",
# # # #   "t": "...",
# # # #   "t+e": "...",
# # # #   "t+a": "...",
# # # #   "t+a+e": "..."
# # # # }
# # # #
# # # # Output CSV columns are minimal (no raw LLM text).
# # # # """
# # # #
# # # # import os
# # # # import re
# # # # import json
# # # # import csv
# # # # import math
# # # # import argparse
# # # # import random
# # # # import hashlib
# # # # from typing import Dict, List, Tuple, Optional, Any
# # # #
# # # # import numpy as np
# # # # import pandas as pd
# # # # from tqdm import tqdm
# # # #
# # # # import torch
# # # # from transformers import (
# # # #     T5Tokenizer,
# # # #     T5ForConditionalGeneration,
# # # #     AutoTokenizer,
# # # #     AutoModelForCausalLM,
# # # # )
# # # #
# # # # # -----------------------------
# # # # # Motion token parsing (A2RM output)
# # # # # -----------------------------
# # # # _MOTION_SPAN_RE = re.compile(r"<Motion Tokens>(.*?)</Motion Tokens>", re.DOTALL)
# # # # _MOTION_TOKEN_RE = re.compile(r"<Motion Token\s+(\d+)>")
# # # # _MOTION_TOKEN_SHORT_RE = re.compile(r"<(\d+)>")  # <123>
# # # #
# # # # def parse_motion_tokens(text: str, max_len: int = 200, codebook_size: int = 512) -> List[int]:
# # # #     if text is None:
# # # #         return []
# # # #     s = str(text)
# # # #     m = _MOTION_SPAN_RE.search(s)
# # # #     span = m.group(1) if m else s
# # # #
# # # #     codes = [int(x) for x in _MOTION_TOKEN_RE.findall(span)]
# # # #     if len(codes) == 0:
# # # #         codes = [int(x) for x in _MOTION_TOKEN_SHORT_RE.findall(span)]
# # # #
# # # #     out = []
# # # #     for c in codes:
# # # #         if 0 <= c < codebook_size:
# # # #             out.append(c)
# # # #         else:
# # # #             break
# # # #     return out[:max_len]
# # # #
# # # # # -----------------------------
# # # # # Prompt builder (MATCH TRAINING)
# # # # # -----------------------------
# # # # def build_prompt_condmode(
# # # #     speaker_transcription: str,
# # # #     speaker_audio: str,
# # # #     speaker_emotion: str,
# # # #     cond_mode: str,
# # # # ) -> str:
# # # #     """
# # # #     EXACTLY like your dataset/prompt_builder.py::build_prompt,
# # # #     but driven by cond_mode:
# # # #       - a       : audio only
# # # #       - a+e     : audio + emotion
# # # #       - t       : transcription only
# # # #       - t+e     : transcription + emotion
# # # #       - t+a     : transcription + audio
# # # #       - t+a+e   : transcription + audio + emotion
# # # #     """
# # # #     cm = (cond_mode or "").strip().lower()
# # # #     use_transcription = ("t" in cm)
# # # #     use_audio = ("a" in cm)
# # # #     use_emotion = (cm.endswith("+e") or cm in ("a+e", "t+e", "t+a+e"))
# # # #
# # # #     t = (speaker_transcription or "").strip()
# # # #     a = (speaker_audio or "").strip()
# # # #     e = (speaker_emotion or "").strip()
# # # #
# # # #     lines = []
# # # #     lines.append("You are modeling a speaker-listener dyadic interaction.\n\n")
# # # #     lines.append("Input:\n")
# # # #     lines.append(f"- SPEAKER_TRANSCRIPTION: {t if use_transcription else ''}\n")
# # # #     lines.append(f"- SPEAKER_AUDIO: {a if use_audio else ''}\n")
# # # #     if use_emotion and e:
# # # #         lines.append(f"- SPEAKER_EMOTION: <Emotion> {e} </Emotion>\n")
# # # #     lines.append("\nOutput:\n")
# # # #     lines.append("Return ONLY a sequence of listener motion tokens in the exact format:\n")
# # # #     lines.append("<Motion Tokens> <Motion Token i> ... </Motion Tokens>\n")
# # # #     lines.append("Do NOT output any other words.\n")
# # # #     return "".join(lines).strip()
# # # #
# # # # # -----------------------------
# # # # # Caption model helpers (T5)
# # # # # -----------------------------
# # # # def build_caption_prompt(motion_codes: List[int]) -> str:
# # # #     motion_string = "<Motion Tokens>" + "".join([f"<{c}>" for c in motion_codes]) + "</Motion Tokens>"
# # # #     return "Generate text: " + motion_string
# # # #
# # # # # -----------------------------
# # # # # Audio token formatting
# # # # # -----------------------------
# # # # def load_audio_tokens_any(path: str) -> np.ndarray:
# # # #     obj = np.load(path, allow_pickle=False)
# # # #     if isinstance(obj, np.lib.npyio.NpzFile):
# # # #         if "codes" in obj.files:
# # # #             arr = obj["codes"]
# # # #         else:
# # # #             arr = obj[obj.files[0]]
# # # #         obj.close()
# # # #         return arr
# # # #     return obj
# # # #
# # # # def format_audio_tokens(a_tokens: np.ndarray, level: str = "base") -> str:
# # # #     """
# # # #     a_tokens: [L,T] or [T]
# # # #     level: base/all/rand
# # # #     """
# # # #     arr = np.array(a_tokens)
# # # #     level = str(level)
# # # #
# # # #     if arr.ndim == 1:
# # # #         parts = ["<Audio Tokens>"]
# # # #         for t in arr.reshape(-1):
# # # #             parts.append(f"<Audio Token {int(t)}>")
# # # #         parts.append("</Audio Tokens>")
# # # #         return " ".join(parts)
# # # #
# # # #     L = int(arr.shape[0])
# # # #     parts = ["<Audio Tokens>"]
# # # #
# # # #     if level == "base":
# # # #         for t in arr[0].reshape(-1):
# # # #             parts.append(f"<Audio Level 0 Token {int(t)}>")
# # # #     elif level == "all":
# # # #         for i in range(L):
# # # #             for t in arr[i].reshape(-1):
# # # #                 parts.append(f"<Audio Level {i} Token {int(t)}>")
# # # #     elif level == "rand":
# # # #         k = int(np.random.choice(np.arange(1, L + 1)))
# # # #         for i in range(k):
# # # #             for t in arr[i].reshape(-1):
# # # #                 parts.append(f"<Audio Level {i} Token {int(t)}>")
# # # #     else:
# # # #         raise ValueError(f"Unknown audio_token_level={level}")
# # # #
# # # #     parts.append("</Audio Tokens>")
# # # #     return " ".join(parts)
# # # #
# # # # def pick_code_from_stem(code_dir: str, stem: str) -> Optional[str]:
# # # #     stem = str(stem).strip()
# # # #     if not stem:
# # # #         return None
# # # #     p_npz = os.path.join(code_dir, stem + ".npz")
# # # #     if os.path.exists(p_npz):
# # # #         return p_npz
# # # #     p_npy = os.path.join(code_dir, stem + ".npy")
# # # #     if os.path.exists(p_npy):
# # # #         return p_npy
# # # #     return None
# # # #
# # # # # -----------------------------
# # # # # VQ lookup for labeled motions
# # # # # -----------------------------
# # # # def motion_id_from_raw(raw_file_name: str) -> str:
# # # #     s = str(raw_file_name)
# # # #     mid = s.split("_", 1)[0]
# # # #     return str(mid).zfill(6)
# # # #
# # # # def build_vq_index(vq_dir: str) -> Dict[str, str]:
# # # #     m = {}
# # # #     for fn in os.listdir(vq_dir):
# # # #         if fn.endswith(".npy"):
# # # #             stem = os.path.splitext(fn)[0]
# # # #             m[stem] = os.path.join(vq_dir, fn)
# # # #     return m
# # # #
# # # # def vqvae_lookup(vq_by_stem: Dict[str, str], motion_id: str) -> Optional[str]:
# # # #     base = str(motion_id)
# # # #     if base in vq_by_stem:
# # # #         return vq_by_stem[base]
# # # #     if base.isdigit() and ("M" + base) in vq_by_stem:
# # # #         return vq_by_stem["M" + base]
# # # #     if base.startswith("M") and base[1:].isdigit() and (base[1:] in vq_by_stem):
# # # #         return vq_by_stem[base[1:]]
# # # #     return None
# # # #
# # # # def load_motion_codes_from_vq(vq_path: str, codebook_size: int = 512) -> List[int]:
# # # #     arr = np.load(vq_path, allow_pickle=False)
# # # #     arr = np.asarray(arr).reshape(-1).tolist()
# # # #     out = []
# # # #     for x in arr:
# # # #         try:
# # # #             c = int(x)
# # # #         except Exception:
# # # #             continue
# # # #         if 0 <= c < codebook_size:
# # # #             out.append(c)
# # # #     return out
# # # #
# # # # # -----------------------------
# # # # # Candidate packing (rename to C01..)
# # # # # -----------------------------
# # # # def build_uniform_candidates(
# # # #     gen_items: List[Dict[str, Any]],
# # # #     gold_items: List[Dict[str, Any]],
# # # #     silver_items: List[Dict[str, Any]],
# # # #     neg_items: List[Dict[str, Any]],
# # # #     seed: int,
# # # #     max_total: int,
# # # # ) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, str], List[str]]:
# # # #     """
# # # #     Returns:
# # # #       candidates: [{"cid":"C01","caption":"..."}, ...]
# # # #       cid2orig: {"C01":"gen_0", ...}
# # # #       orig2type: {"gen_0":"gen","gold_0":"gold"...}
# # # #       orig_ids_in_order: list of orig_id aligned with candidates list
# # # #     """
# # # #     all_items = []
# # # #     orig2type: Dict[str, str] = {}
# # # #
# # # #     def add(items, t):
# # # #         for it in items:
# # # #             oid = str(it["id"])
# # # #             all_items.append({"orig_id": oid, "caption": str(it["caption"])})
# # # #             orig2type[oid] = t
# # # #
# # # #     add(gen_items, "gen")
# # # #     add(gold_items, "gold")
# # # #     add(silver_items, "silver")
# # # #     add(neg_items, "neg")
# # # #
# # # #     rng = random.Random(int(seed))
# # # #     rng.shuffle(all_items)
# # # #     all_items = all_items[: max_total]
# # # #
# # # #     cid2orig: Dict[str, str] = {}
# # # #     candidates: List[Dict[str, Any]] = []
# # # #     orig_ids_in_order: List[str] = []
# # # #
# # # #     for i, it in enumerate(all_items, start=1):
# # # #         cid = f"C{i:02d}"
# # # #         cid2orig[cid] = it["orig_id"]
# # # #         candidates.append({"cid": cid, "caption": it["caption"]})
# # # #         orig_ids_in_order.append(it["orig_id"])
# # # #
# # # #     return candidates, cid2orig, orig2type, orig_ids_in_order
# # # #
# # # # # -----------------------------
# # # # # Pairwise Qwen judge
# # # # # -----------------------------
# # # # def extract_last_json_object(text: str) -> Optional[Dict[str, Any]]:
# # # #     if text is None:
# # # #         return None
# # # #     s = str(text)
# # # #     starts = [i for i, ch in enumerate(s) if ch == "{"]  # keep all
# # # #     if not starts:
# # # #         return None
# # # #
# # # #     last_obj = None
# # # #     for st in starts:
# # # #         depth = 0
# # # #         in_str = False
# # # #         esc = False
# # # #         for i in range(st, len(s)):
# # # #             c = s[i]
# # # #             if in_str:
# # # #                 if esc:
# # # #                     esc = False
# # # #                 elif c == "\\":
# # # #                     esc = True
# # # #                 elif c == '"':
# # # #                     in_str = False
# # # #             else:
# # # #                 if c == '"':
# # # #                     in_str = True
# # # #                 elif c == "{":
# # # #                     depth += 1
# # # #                 elif c == "}":
# # # #                     depth -= 1
# # # #                     if depth == 0:
# # # #                         blob = s[st : i + 1]
# # # #                         try:
# # # #                             obj = json.loads(blob)
# # # #                             last_obj = obj
# # # #                         except Exception:
# # # #                             pass
# # # #                         break
# # # #     return last_obj
# # # #
# # # # _CID_RE = re.compile(r"\bC\d{2}\b")
# # # #
# # # # def parse_winner_cid(raw: str, allowed: set) -> Optional[str]:
# # # #     """
# # # #     Accept:
# # # #       {"winner":"C01"} OR {"better":"C01"} OR {"choice":"C01"}
# # # #     Fallback: find any Cxx in text.
# # # #     """
# # # #     obj = extract_last_json_object(raw) or {}
# # # #     for k in ["winner", "better", "choice", "answer", "selected"]:
# # # #         v = obj.get(k, None)
# # # #         if isinstance(v, str) and v in allowed:
# # # #             return v
# # # #
# # # #     m = _CID_RE.findall(str(raw))
# # # #     for cid in m:
# # # #         if cid in allowed:
# # # #             return cid
# # # #     return None
# # # #
# # # # def build_qwen_pairwise_prompt(
# # # #     query_payload: Dict[str, Any],
# # # #     cand_a: Dict[str, Any],
# # # #     cand_b: Dict[str, Any],
# # # # ) -> str:
# # # #     payload = {
# # # #         "task": "Dyadic SPEAKER→LISTENER pairwise preference.",
# # # #         "query": query_payload,
# # # #         "candidates": [
# # # #             {"cid": cand_a["cid"], "caption": cand_a["caption"]},
# # # #             {"cid": cand_b["cid"], "caption": cand_b["caption"]},
# # # #         ],
# # # #         "rules": [
# # # #             "You are comparing LISTENER reactions only (NOT speaker actions).",
# # # #             "Pick the ONE candidate that is the more plausible listener non-verbal response to the speaker sayings/emotion.",
# # # #             "Use ONLY the two captions as evidence; ids contain no label information.",
# # # #             "Return ONLY JSON: {\"winner\": \"Cxx\"}. No other keys/text.",
# # # #         ],
# # # #     }
# # # #     return (
# # # #         "You are an expert evaluator for listener reactive motions.\n"
# # # #         "IMPORTANT: Output MUST be a single valid JSON object and NOTHING ELSE.\n"
# # # #         f"{json.dumps(payload, ensure_ascii=False)}"
# # # #     )
# # # #
# # # # class QwenJudge:
# # # #     def __init__(
# # # #         self,
# # # #         model_path: str,
# # # #         use_vllm: bool = True,
# # # #         tp: int = 1,
# # # #         gpu_mem_util: float = 0.90,
# # # #         max_new_tokens: int = 256,
# # # #         temperature: float = 0.0,
# # # #         max_model_len: int = 8192,
# # # #     ):
# # # #         self.model_path = model_path
# # # #         self.use_vllm = use_vllm
# # # #         self.tp = tp
# # # #         self.gpu_mem_util = gpu_mem_util
# # # #         self.max_new_tokens = max_new_tokens
# # # #         self.temperature = temperature
# # # #         self.max_model_len = max_model_len
# # # #         self._mode = None
# # # #
# # # #         if use_vllm:
# # # #             try:
# # # #                 from vllm import LLM, SamplingParams  # type: ignore
# # # #                 self._vllm_LLM = LLM(
# # # #                     model=model_path,
# # # #                     tensor_parallel_size=tp,
# # # #                     gpu_memory_utilization=gpu_mem_util,
# # # #                     trust_remote_code=True,
# # # #                     max_model_len=max_model_len,
# # # #                     enforce_eager=False,
# # # #                 )
# # # #                 self._vllm_SamplingParams = SamplingParams
# # # #                 self._mode = "vllm"
# # # #             except Exception as e:
# # # #                 print(f"[WARN] vLLM init failed, fallback to HF. err={e}")
# # # #                 self._mode = "hf"
# # # #         else:
# # # #             self._mode = "hf"
# # # #
# # # #         if self._mode == "hf":
# # # #             self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# # # #             self.model = AutoModelForCausalLM.from_pretrained(
# # # #                 model_path,
# # # #                 trust_remote_code=True,
# # # #                 torch_dtype=torch.bfloat16,
# # # #                 device_map="auto",
# # # #             ).eval()
# # # #
# # # #     @torch.no_grad()
# # # #     def generate(self, prompt: str) -> str:
# # # #         if self._mode == "vllm":
# # # #             sp = self._vllm_SamplingParams(
# # # #                 temperature=self.temperature,
# # # #                 max_tokens=self.max_new_tokens,
# # # #                 top_p=1.0,
# # # #             )
# # # #             outs = self._vllm_LLM.generate([prompt], sp)
# # # #             return outs[0].outputs[0].text.strip()
# # # #
# # # #         inputs = self.tok(prompt, return_tensors="pt")
# # # #         inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
# # # #         out = self.model.generate(
# # # #             **inputs,
# # # #             max_new_tokens=self.max_new_tokens,
# # # #             do_sample=(self.temperature > 0),
# # # #             temperature=max(1e-6, self.temperature),
# # # #             top_p=1.0,
# # # #         )
# # # #         txt = self.tok.decode(out[0], skip_special_tokens=True)
# # # #         if txt.startswith(prompt):
# # # #             txt = txt[len(prompt):]
# # # #         return txt.strip()
# # # #
# # # # # -----------------------------
# # # # # Bradley–Terry fitting + ranking
# # # # # -----------------------------
# # # # def fit_btl_scores(
# # # #     n_items: int,
# # # #     matches: List[Tuple[int, int, int]],  # (i, j, y) y=1 => i wins else j wins
# # # #     l2: float = 1e-2,
# # # #     lr: float = 0.2,
# # # #     steps: int = 600,
# # # # ) -> np.ndarray:
# # # #     s = np.zeros(n_items, dtype=np.float64)
# # # #     if len(matches) == 0:
# # # #         return s
# # # #
# # # #     for _ in range(steps):
# # # #         grad = np.zeros_like(s)
# # # #         for i, j, y in matches:
# # # #             d = s[i] - s[j]
# # # #             p = 1.0 / (1.0 + math.exp(-d))
# # # #             g = (p - y)  # dL/dd
# # # #             grad[i] += g
# # # #             grad[j] -= g
# # # #         grad += l2 * s
# # # #         s -= lr * grad / max(1, len(matches))
# # # #         s -= s.mean()  # remove shift ambiguity
# # # #     return s
# # # #
# # # # def btl_order(scores: np.ndarray) -> List[int]:
# # # #     return np.argsort(-scores).tolist()
# # # #
# # # # # -----------------------------
# # # # # Metrics
# # # # # -----------------------------
# # # # def type_to_gain_gold_only(t: str) -> int:
# # # #     return 2 if t == "gold" else 0
# # # #
# # # # def ndcg_at_k(relevances: List[int], k: int) -> float:
# # # #     rel = relevances[:k]
# # # #     dcg = 0.0
# # # #     for i, r in enumerate(rel, start=1):
# # # #         if r > 0:
# # # #             dcg += (2.0 ** r - 1.0) / math.log2(i + 1.0)
# # # #
# # # #     ideal = sorted(relevances, reverse=True)[:k]
# # # #     idcg = 0.0
# # # #     for i, r in enumerate(ideal, start=1):
# # # #         if r > 0:
# # # #             idcg += (2.0 ** r - 1.0) / math.log2(i + 1.0)
# # # #
# # # #     if idcg <= 1e-12:
# # # #         return 0.0
# # # #     return float(dcg / idcg)
# # # #
# # # # def winrate_best_gen_vs_type_from_order(
# # # #     order: List[int],
# # # #     idx2type: List[str],
# # # #     comp_type: str,
# # # # ) -> float:
# # # #     rank = {idx: r for r, idx in enumerate(order)}
# # # #     gen_idxs = [i for i, t in enumerate(idx2type) if t == "gen"]
# # # #     comp_idxs = [i for i, t in enumerate(idx2type) if t == comp_type]
# # # #     if len(gen_idxs) == 0 or len(comp_idxs) == 0:
# # # #         return float("nan")
# # # #     best_gen_rank = min(rank[i] for i in gen_idxs)
# # # #     wins = sum(1 for j in comp_idxs if best_gen_rank < rank[j])
# # # #     return float(wins / len(comp_idxs))
# # # #
# # # # def gen_at_k_from_order(order: List[int], idx2type: List[str], k: int = 3) -> float:
# # # #     topk = order[:k]
# # # #     return 1.0 if any(idx2type[i] == "gen" for i in topk) else 0.0
# # # #
# # # # # -----------------------------
# # # # # Pair sampling strategy
# # # # # -----------------------------
# # # # def sample_pairs(
# # # #     n: int,
# # # #     max_pairs: int,
# # # #     seed: int,
# # # # ) -> List[Tuple[int, int]]:
# # # #     """
# # # #     Sample unordered pairs (i<j). If n small, you can set max_pairs large to cover all.
# # # #     """
# # # #     rng = random.Random(seed)
# # # #     all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
# # # #     if len(all_pairs) <= max_pairs:
# # # #         rng.shuffle(all_pairs)
# # # #         return all_pairs
# # # #     return rng.sample(all_pairs, k=max_pairs)
# # # #
# # # # # -----------------------------
# # # # # misc
# # # # # -----------------------------
# # # # def ensure_parent(path: str):
# # # #     os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
# # # #
# # # # def hash_file(path: str) -> str:
# # # #     h = hashlib.md5()
# # # #     with open(path, "rb") as f:
# # # #         while True:
# # # #             b = f.read(1024 * 1024)
# # # #             if not b:
# # # #                 break
# # # #             h.update(b)
# # # #     return h.hexdigest()[:10]
# # # #
# # # # def stable_hash_str(s: str) -> str:
# # # #     return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]
# # # #
# # # # def make_run_tag(args: argparse.Namespace) -> str:
# # # #     parts = [
# # # #         f"numgen{args.num_gen}",
# # # #         f"keval{args.k_eval}",
# # # #         f"maxtotal{args.max_total_candidates}",
# # # #         f"pairs{args.max_pairs}",
# # # #         f"cap{stable_hash_str(args.caption_ckpt)}",
# # # #         f"qwen{stable_hash_str(args.qwen_path)}",
# # # #         f"ckmap{hash_file(args.ckpt_map_json)}",
# # # #     ]
# # # #     return "_".join(parts)
# # # #
# # # # def canon_label(x: str) -> str:
# # # #     s = (x or "").strip().lower()
# # # #     if s in {"gold", "pos", "positive", "gt", "true", "1"}:
# # # #         return "gold"
# # # #     if s in {"silver"}:
# # # #         return "silver"
# # # #     if s in {"neg", "negative", "0"}:
# # # #         return "neg"
# # # #     return s
# # # #
# # # # # -----------------------------
# # # # # Main
# # # # # -----------------------------
# # # # def main():
# # # #     ap = argparse.ArgumentParser()
# # # #
# # # #     ap.add_argument("--pairs_csv", type=str, required=True)
# # # #     ap.add_argument("--dataset_dir", type=str, required=True)
# # # #     ap.add_argument("--ckpt_map_json", type=str, required=True)
# # # #
# # # #     ap.add_argument("--audio_code_dir", type=str, default=None)
# # # #     ap.add_argument("--audio_token_level", type=str, default="base", choices=["base", "all", "rand"])
# # # #
# # # #     ap.add_argument("--caption_ckpt", type=str, required=True)
# # # #
# # # #     ap.add_argument("--num_gen", type=int, default=3)
# # # #     ap.add_argument("--gen_max_len", type=int, default=200)
# # # #     ap.add_argument("--gen_temperature", type=float, default=0.8)
# # # #     ap.add_argument("--gen_top_k", type=int, default=200)
# # # #
# # # #     ap.add_argument("--max_gold", type=int, default=1)
# # # #     ap.add_argument("--max_silver", type=int, default=8)
# # # #     ap.add_argument("--max_neg", type=int, default=12)
# # # #     ap.add_argument("--max_total_candidates", type=int, default=30)
# # # #
# # # #     ap.add_argument("--only_split", type=str, default="test")  # "test"/"val"/"train"/"all"
# # # #     ap.add_argument("--group_by", type=str, default="group_id", choices=["group_id", "sayings_emotion"])
# # # #
# # # #     # output ranking list length
# # # #     ap.add_argument("--k_eval", type=int, default=10)
# # # #
# # # #     # pairwise budget
# # # #     ap.add_argument("--max_pairs", type=int, default=120)  # per (mode, group) on FULL pool
# # # #     ap.add_argument("--max_pairs_base", type=int, default=80)  # per (mode, group) on LABELED-only pool
# # # #
# # # #     # qwen
# # # #     ap.add_argument("--qwen_path", type=str, required=True)
# # # #     ap.add_argument("--qwen_use_vllm", action="store_true")
# # # #     ap.add_argument("--qwen_tp", type=int, default=1)
# # # #     ap.add_argument("--qwen_gpu_mem_util", type=float, default=0.90)
# # # #     ap.add_argument("--qwen_max_new_tokens", type=int, default=256)
# # # #     ap.add_argument("--qwen_max_model_len", type=int, default=8192)
# # # #
# # # #     ap.add_argument("--out_dir", type=str, default="./eval_pairwise_btl_out")
# # # #     ap.add_argument("--seed", type=int, default=42)
# # # #
# # # #     args = ap.parse_args()
# # # #     random.seed(args.seed)
# # # #     np.random.seed(args.seed)
# # # #     torch.manual_seed(args.seed)
# # # #
# # # #     os.makedirs(args.out_dir, exist_ok=True)
# # # #     device = "cuda" if torch.cuda.is_available() else "cpu"
# # # #     print("[Device]", device)
# # # #
# # # #     # ckpt map
# # # #     with open(args.ckpt_map_json, "r", encoding="utf-8") as f:
# # # #         ckpt_map = json.load(f)
# # # #
# # # #     modes = ["a", "a+e", "t", "t+e", "t+a", "t+a+e"]
# # # #     for m in modes:
# # # #         if m not in ckpt_map:
# # # #             raise RuntimeError(f"ckpt_map_json missing mode: {m}")
# # # #
# # # #     # read pairs
# # # #     df = pd.read_csv(args.pairs_csv, encoding="utf-8")
# # # #     required_cols = ["sayings", "emotion", "label", "raw_file_name", "split", "generated_wav_name"]
# # # #     missing = [c for c in required_cols if c not in df.columns]
# # # #     if missing:
# # # #         raise RuntimeError(f"Missing columns in pairs_csv: {missing}")
# # # #
# # # #     df["label"] = df["label"].astype(str).str.lower().str.strip()
# # # #     df["sayings"] = df["sayings"].astype(str).fillna("")
# # # #     df["emotion"] = df["emotion"].astype(str).fillna("")
# # # #     df["raw_file_name"] = df["raw_file_name"].astype(str).fillna("")
# # # #     df["generated_wav_name"] = df["generated_wav_name"].astype(str).fillna("")
# # # #     df["split"] = df["split"].astype(str).str.lower().str.strip()
# # # #
# # # #     if args.only_split != "all":
# # # #         df = df[df["split"] == args.only_split].copy()
# # # #     if len(df) == 0:
# # # #         raise RuntimeError(f"No rows for split={args.only_split}")
# # # #
# # # #     # group
# # # #     if args.group_by == "group_id":
# # # #         if "group_id" not in df.columns:
# # # #             print("[WARN] group_by=group_id but no group_id column; fallback to sayings_emotion")
# # # #             args.group_by = "sayings_emotion"
# # # #
# # # #     if args.group_by == "group_id":
# # # #         groups = list(df.groupby(["group_id"], dropna=False))
# # # #     else:
# # # #         groups = list(df.groupby(["sayings", "emotion"], dropna=False))
# # # #     print("[Groups]", len(groups), "group_by=", args.group_by)
# # # #
# # # #     # VQ index
# # # #     motion_vq_dir = os.path.join(args.dataset_dir, "HumanML3D", "VQVAE")
# # # #     if not os.path.isdir(motion_vq_dir):
# # # #         raise RuntimeError(f"Missing motion_vq_dir: {motion_vq_dir}")
# # # #     vq_by_stem = build_vq_index(motion_vq_dir)
# # # #     print("[VQ] indexed:", len(vq_by_stem))
# # # #
# # # #     # audio code dir
# # # #     audio_code_dir = args.audio_code_dir or os.path.join(args.dataset_dir, "audio-raws-09-01-2026-code")
# # # #     if not os.path.isdir(audio_code_dir):
# # # #         print(f"[WARN] audio_code_dir not found: {audio_code_dir} (a-modes will use empty audio)")
# # # #
# # # #     # caption model
# # # #     cap_tok = T5Tokenizer.from_pretrained(args.caption_ckpt)
# # # #     cap_model = T5ForConditionalGeneration.from_pretrained(args.caption_ckpt).to(device).eval()
# # # #
# # # #     @torch.no_grad()
# # # #     def caption_motion_codes(codes: List[int]) -> str:
# # # #         prompt = build_caption_prompt(codes)
# # # #         inp = cap_tok(prompt, return_tensors="pt").input_ids.to(device, dtype=torch.long)
# # # #         out = cap_model.generate(inp, max_length=200, num_beams=1, do_sample=False)
# # # #         txt = cap_tok.decode(out[0], skip_special_tokens=True).strip().strip('"')
# # # #         return txt
# # # #
# # # #     # qwen
# # # #     qwen = QwenJudge(
# # # #         model_path=args.qwen_path,
# # # #         use_vllm=args.qwen_use_vllm,
# # # #         tp=args.qwen_tp,
# # # #         gpu_mem_util=args.qwen_gpu_mem_util,
# # # #         max_new_tokens=args.qwen_max_new_tokens,
# # # #         temperature=0.0,
# # # #         max_model_len=args.qwen_max_model_len,
# # # #     )
# # # #     print("[Qwen] model=", args.qwen_path, "backend=", ("vllm" if args.qwen_use_vllm else "hf"))
# # # #
# # # #     # output file
# # # #     run_tag = make_run_tag(args)
# # # #     out_csv = os.path.join(args.out_dir, f"eval_pairwise_btl_{run_tag}.csv")
# # # #     ensure_parent(out_csv)
# # # #
# # # #     fieldnames = [
# # # #         "eval_key",
# # # #         "mode",
# # # #         "split",
# # # #         "group_key",
# # # #         "sayings",
# # # #         "emotion",
# # # #         "audio_code_path",
# # # #         "num_gen",
# # # #         "num_gold",
# # # #         "num_silver",
# # # #         "num_neg",
# # # #         "k_eval",
# # # #         "oracle_ndcg5_base",
# # # #         "oracle_ndcg5_withgen",
# # # #         "oracle_drop_ndcg5",
# # # #         "win_gen_vs_neg",
# # # #         "win_gen_vs_silver",
# # # #         "win_gen_vs_gold",
# # # #         "gen_at3",
# # # #         "topk_orig_json",
# # # #         "topk_types_json",
# # # #     ]
# # # #
# # # #     exists = os.path.isfile(out_csv)
# # # #     fcsv = open(out_csv, "a", encoding="utf-8", newline="")
# # # #     writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
# # # #     if not exists:
# # # #         writer.writeheader()
# # # #         fcsv.flush()
# # # #         os.fsync(fcsv.fileno())
# # # #
# # # #     # cache a2rm models per mode
# # # #     a2rm_models: Dict[str, Any] = {}
# # # #     a2rm_toks: Dict[str, Any] = {}
# # # #
# # # #     def get_a2rm_model(mode: str):
# # # #         if mode in a2rm_models:
# # # #             return a2rm_toks[mode], a2rm_models[mode]
# # # #         ckpt = ckpt_map[mode]
# # # #         tok = T5Tokenizer.from_pretrained(ckpt)
# # # #         model = T5ForConditionalGeneration.from_pretrained(ckpt).to(device).eval()
# # # #         a2rm_toks[mode] = tok
# # # #         a2rm_models[mode] = model
# # # #         print(f"[Load A2RM] mode={mode} ckpt={ckpt}")
# # # #         return tok, model
# # # #
# # # #     # resolve audio for group (a-modes) using generated_wav_name stem
# # # #     def resolve_audio_for_group(g: pd.DataFrame) -> Tuple[str, str]:
# # # #         stems = [str(x).strip() for x in g["generated_wav_name"].tolist() if str(x).strip()]
# # # #         stems = list(dict.fromkeys(stems))
# # # #         if (not stems) or (not os.path.isdir(audio_code_dir)):
# # # #             return "", ""
# # # #         stem = random.choice(stems)
# # # #         p = pick_code_from_stem(audio_code_dir, stem)
# # # #         if p is None:
# # # #             return "", ""
# # # #         codes = load_audio_tokens_any(p)
# # # #         return format_audio_tokens(codes, level=args.audio_token_level), p
# # # #
# # # #     def make_eval_key(mode: str, group_key: str, audio_code_path: str) -> str:
# # # #         s = f"{mode}|||{group_key}|||{audio_code_path}"
# # # #         return hashlib.md5(s.encode("utf-8")).hexdigest()
# # # #
# # # #     # build label items for a group
# # # #     def build_items(
# # # #         gg_all: pd.DataFrame,
# # # #         lbl: str,
# # # #         max_n: int,
# # # #         use_csv_caption: bool,
# # # #     ) -> List[Dict[str, Any]]:
# # # #         sub = gg_all[gg_all["label_canon"] == lbl].copy()
# # # #         if len(sub) == 0:
# # # #             return []
# # # #         mids = list(dict.fromkeys(sub["motion_id"].tolist()))[:max_n]
# # # #         items = []
# # # #         for i, mid in enumerate(mids):
# # # #             cap = ""
# # # #             if use_csv_caption:
# # # #                 ss = sub[sub["motion_id"] == mid]
# # # #                 if "motion_caption" in ss.columns and len(ss) > 0:
# # # #                     cap = str(ss["motion_caption"].iloc[0]).strip()
# # # #             if not cap:
# # # #                 p = vqvae_lookup(vq_by_stem, mid)
# # # #                 if p is None:
# # # #                     continue
# # # #                 codes = load_motion_codes_from_vq(p)
# # # #                 cap = caption_motion_codes(codes)
# # # #             items.append({"id": f"{lbl}_{i}", "caption": cap})
# # # #         return items
# # # #
# # # #     def run_pairwise_btl(
# # # #         query_payload: Dict[str, Any],
# # # #         candidates: List[Dict[str, Any]],  # [{"cid","caption"}]
# # # #         max_pairs: int,
# # # #         seed_int: int,
# # # #     ) -> List[int]:
# # # #         """
# # # #         returns order over indices 0..N-1 (descending best first)
# # # #         """
# # # #         N = len(candidates)
# # # #         if N <= 1:
# # # #             return list(range(N))
# # # #
# # # #         pairs = sample_pairs(N, max_pairs=max_pairs, seed=seed_int ^ 0x13579BDF)
# # # #         allowed = {c["cid"] for c in candidates}
# # # #         matches: List[Tuple[int, int, int]] = []
# # # #
# # # #         for (i, j) in pairs:
# # # #             ca, cb = candidates[i], candidates[j]
# # # #             prompt = build_qwen_pairwise_prompt(query_payload, ca, cb)
# # # #             out = qwen.generate(prompt)
# # # #             winner = parse_winner_cid(out, allowed=allowed)
# # # #
# # # #             # if parse failed, do a deterministic fallback: prefer i (keeps pipeline running)
# # # #             if winner is None:
# # # #                 y = 1
# # # #             else:
# # # #                 y = 1 if winner == ca["cid"] else 0
# # # #
# # # #             matches.append((i, j, y))
# # # #
# # # #         scores = fit_btl_scores(N, matches)
# # # #         return btl_order(scores)
# # # #
# # # #     # ----------------------------
# # # #     # main loop
# # # #     # ----------------------------
# # # #     for mode in modes:
# # # #         tok, model = get_a2rm_model(mode)
# # # #
# # # #         for keys, g in tqdm(groups, desc=f"Eval[{mode}]"):
# # # #             if args.group_by == "group_id":
# # # #                 group_key = str(keys) if not isinstance(keys, tuple) else str(keys[0])
# # # #                 sayings = str(g["sayings"].iloc[0])
# # # #                 emotion = str(g["emotion"].iloc[0])
# # # #             else:
# # # #                 sayings, emotion = keys
# # # #                 sayings = str(sayings)
# # # #                 emotion = str(emotion)
# # # #                 group_key = f"{sayings}|||{emotion}"
# # # #
# # # #             split = str(g["split"].iloc[0])
# # # #
# # # #             gg_all = g.copy()
# # # #             gg_all["label_canon"] = gg_all["label"].apply(canon_label)
# # # #             gg_all["motion_id"] = gg_all["raw_file_name"].apply(motion_id_from_raw)
# # # #             use_csv_caption = ("motion_caption" in gg_all.columns)
# # # #
# # # #             gold_items = build_items(gg_all, "gold", args.max_gold, use_csv_caption)
# # # #             silver_items = build_items(gg_all, "silver", args.max_silver, use_csv_caption)
# # # #             neg_items = build_items(gg_all, "neg", args.max_neg, use_csv_caption)
# # # #
# # # #             # audio for a-modes
# # # #             audio_text = ""
# # # #             audio_code_path = ""
# # # #             if "a" in mode:
# # # #                 audio_text, audio_code_path = resolve_audio_for_group(g)
# # # #
# # # #             eval_key = make_eval_key(mode, group_key, audio_code_path)
# # # #
# # # #             # query payload for judge
# # # #             query_payload = {
# # # #                 "speaker_sayings": sayings,
# # # #                 "speaker_emotion": emotion,
# # # #                 "cond_mode": mode,
# # # #                 "note": "In a dyadic conversation, rank LISTENER non-verbal responses to the SPEAKER's utterance.",
# # # #             }
# # # #
# # # #             # seed per group-mode
# # # #             seed_int = int(hashlib.md5(f"{group_key}|||{mode}".encode("utf-8")).hexdigest()[:8], 16)
# # # #
# # # #             # ----------------------------
# # # #             # Base (labeled-only) BTL ranking -> oracle nDCG@5
# # # #             # ----------------------------
# # # #             cand_base, cid2orig_base, orig2type_base, orig_ids_base = build_uniform_candidates(
# # # #                 gen_items=[],
# # # #                 gold_items=gold_items,
# # # #                 silver_items=silver_items,
# # # #                 neg_items=neg_items,
# # # #                 seed=seed_int ^ 0xBADC0DE,
# # # #                 max_total=args.max_total_candidates,
# # # #             )
# # # #
# # # #             if len(cand_base) == 0:
# # # #                 # no candidates at all -> write row with NaNs
# # # #                 row = dict(
# # # #                     eval_key=eval_key,
# # # #                     mode=mode,
# # # #                     split=split,
# # # #                     group_key=group_key,
# # # #                     sayings=sayings,
# # # #                     emotion=emotion,
# # # #                     audio_code_path=audio_code_path,
# # # #                     num_gen=0,
# # # #                     num_gold=len(gold_items),
# # # #                     num_silver=len(silver_items),
# # # #                     num_neg=len(neg_items),
# # # #                     k_eval=min(args.k_eval, 0),
# # # #                     oracle_ndcg5_base=float("nan"),
# # # #                     oracle_ndcg5_withgen=float("nan"),
# # # #                     oracle_drop_ndcg5=float("nan"),
# # # #                     win_gen_vs_neg=float("nan"),
# # # #                     win_gen_vs_silver=float("nan"),
# # # #                     win_gen_vs_gold=float("nan"),
# # # #                     gen_at3=float("nan"),
# # # #                     topk_orig_json="[]",
# # # #                     topk_types_json="[]",
# # # #                 )
# # # #                 writer.writerow(row)
# # # #                 fcsv.flush()
# # # #                 os.fsync(fcsv.fileno())
# # # #                 continue
# # # #
# # # #             order_base = run_pairwise_btl(
# # # #                 query_payload=query_payload,
# # # #                 candidates=cand_base,
# # # #                 max_pairs=args.max_pairs_base,
# # # #                 seed_int=seed_int ^ 0x2468ACE,
# # # #             )
# # # #             # build base top5 types for oracle gold-only nDCG
# # # #             base_ranked_orig = [cid2orig_base[cand_base[i]["cid"]] for i in order_base]
# # # #             base_top5_types = [orig2type_base.get(oid, "unk") for oid in base_ranked_orig[:5]]
# # # #             oracle_ndcg5_base = ndcg_at_k([type_to_gain_gold_only(t) for t in base_top5_types], k=5)
# # # #
# # # #             # ----------------------------
# # # #             # Generate N gen candidates
# # # #             # ----------------------------
# # # #             input_text = build_prompt_condmode(
# # # #                 speaker_transcription=sayings,
# # # #                 speaker_audio=audio_text,
# # # #                 speaker_emotion=emotion,
# # # #                 cond_mode=mode,
# # # #             )
# # # #             input_ids = tok(input_text, return_tensors="pt").input_ids.to(device, dtype=torch.long)
# # # #
# # # #             gen_items: List[Dict[str, Any]] = []
# # # #             for ci in range(int(args.num_gen)):
# # # #                 out = model.generate(
# # # #                     input_ids,
# # # #                     max_length=256,
# # # #                     do_sample=True,
# # # #                     temperature=args.gen_temperature,
# # # #                     top_k=args.gen_top_k,
# # # #                 )
# # # #                 out_text = tok.decode(out[0], skip_special_tokens=False)
# # # #                 out_text = out_text.replace("<pad>", "").replace("</s>", "").strip()
# # # #                 codes = parse_motion_tokens(out_text, max_len=args.gen_max_len, codebook_size=512)
# # # #                 if len(codes) == 0:
# # # #                     codes = [1] * min(args.gen_max_len, 196)
# # # #                 cap = caption_motion_codes(codes)
# # # #                 gen_items.append({"id": f"gen_{ci}", "caption": cap})
# # # #
# # # #             # ----------------------------
# # # #             # Full pool (gen + labeled) BTL ranking
# # # #             # ----------------------------
# # # #             cand_all, cid2orig, orig2type, orig_ids_all = build_uniform_candidates(
# # # #                 gen_items=gen_items,
# # # #                 gold_items=gold_items,
# # # #                 silver_items=silver_items,
# # # #                 neg_items=neg_items,
# # # #                 seed=seed_int,
# # # #                 max_total=args.max_total_candidates,
# # # #             )
# # # #
# # # #             # cap k_eval
# # # #             k_eval = min(int(args.k_eval), len(cand_all))
# # # #             if k_eval <= 0:
# # # #                 k_eval = min(5, len(cand_all))
# # # #
# # # #             order_all = run_pairwise_btl(
# # # #                 query_payload=query_payload,
# # # #                 candidates=cand_all,
# # # #                 max_pairs=args.max_pairs,
# # # #                 seed_int=seed_int,
# # # #             )
# # # #
# # # #             ranked_orig = [cid2orig[cand_all[i]["cid"]] for i in order_all]
# # # #             ranked_types = [orig2type.get(oid, "unk") for oid in ranked_orig]
# # # #             ranked_types = ["neg" if t == "unk" else t for t in ranked_types]
# # # #
# # # #             # oracle with-gen nDCG@5 (gold-only)
# # # #             top5_types_withgen = ranked_types[:5]
# # # #             oracle_ndcg5_withgen = ndcg_at_k([type_to_gain_gold_only(t) for t in top5_types_withgen], k=5)
# # # #             oracle_drop_ndcg5 = oracle_ndcg5_withgen - oracle_ndcg5_base
# # # #
# # # #             # gen quality on full order
# # # #             win_gen_vs_neg = winrate_best_gen_vs_type_from_order(order_all, ranked_types, "neg")
# # # #             win_gen_vs_silver = winrate_best_gen_vs_type_from_order(order_all, ranked_types, "silver")
# # # #             win_gen_vs_gold = winrate_best_gen_vs_type_from_order(order_all, ranked_types, "gold")
# # # #             gen_at3 = gen_at_k_from_order(order_all, ranked_types, k=3)
# # # #
# # # #             # top-k outputs in orig/type space
# # # #             topk_orig = ranked_orig[:k_eval]
# # # #             topk_types = ranked_types[:k_eval]
# # # #
# # # #             row = dict(
# # # #                 eval_key=eval_key,
# # # #                 mode=mode,
# # # #                 split=split,
# # # #                 group_key=group_key,
# # # #                 sayings=sayings,
# # # #                 emotion=emotion,
# # # #                 audio_code_path=audio_code_path,
# # # #                 num_gen=len(gen_items),
# # # #                 num_gold=len(gold_items),
# # # #                 num_silver=len(silver_items),
# # # #                 num_neg=len(neg_items),
# # # #                 k_eval=k_eval,
# # # #                 oracle_ndcg5_base=oracle_ndcg5_base,
# # # #                 oracle_ndcg5_withgen=oracle_ndcg5_withgen,
# # # #                 oracle_drop_ndcg5=oracle_drop_ndcg5,
# # # #                 win_gen_vs_neg=win_gen_vs_neg,
# # # #                 win_gen_vs_silver=win_gen_vs_silver,
# # # #                 win_gen_vs_gold=win_gen_vs_gold,
# # # #                 gen_at3=gen_at3,
# # # #                 topk_orig_json=json.dumps(topk_orig, ensure_ascii=False),
# # # #                 topk_types_json=json.dumps(topk_types, ensure_ascii=False),
# # # #             )
# # # #             writer.writerow(row)
# # # #             fcsv.flush()
# # # #             os.fsync(fcsv.fileno())
# # # #
# # # #     fcsv.close()
# # # #     print("[Saved]", out_csv)
# # # #
# # # #
# # # # if __name__ == "__main__":
# # # #     main()
# # #
# # #
# # # #!/usr/bin/env python3
# # # # -*- coding: utf-8 -*-
# # #
# # # import os
# # # import re
# # # import json
# # # import csv
# # # import math
# # # import argparse
# # # import random
# # # import hashlib
# # # from typing import Dict, List, Tuple, Optional, Any
# # #
# # # import numpy as np
# # # import pandas as pd
# # # from tqdm import tqdm
# # #
# # # import torch
# # # from transformers import (
# # #     T5Tokenizer,
# # #     T5ForConditionalGeneration,
# # #     AutoTokenizer,
# # #     AutoModelForCausalLM,
# # # )
# # #
# # # # -----------------------------
# # # # Caption model helpers (T5)
# # # # -----------------------------
# # # def build_caption_prompt(motion_codes: List[int]) -> str:
# # #     motion_string = "<Motion Tokens>" + "".join([f"<{c}>" for c in motion_codes]) + "</Motion Tokens>"
# # #     return "Generate text: " + motion_string
# # #
# # # # -----------------------------
# # # # VQ lookup for labeled motions
# # # # -----------------------------
# # # def motion_id_from_raw(raw_file_name: str) -> str:
# # #     s = str(raw_file_name)
# # #     mid = s.split("_", 1)[0]
# # #     return str(mid).zfill(6)
# # #
# # # def build_vq_index(vq_dir: str) -> Dict[str, str]:
# # #     m = {}
# # #     for fn in os.listdir(vq_dir):
# # #         if fn.endswith(".npy"):
# # #             stem = os.path.splitext(fn)[0]
# # #             m[stem] = os.path.join(vq_dir, fn)
# # #     return m
# # #
# # # def vqvae_lookup(vq_by_stem: Dict[str, str], motion_id: str) -> Optional[str]:
# # #     base = str(motion_id)
# # #     if base in vq_by_stem:
# # #         return vq_by_stem[base]
# # #     if base.isdigit() and ("M" + base) in vq_by_stem:
# # #         return vq_by_stem["M" + base]
# # #     if base.startswith("M") and base[1:].isdigit() and (base[1:] in vq_by_stem):
# # #         return vq_by_stem[base[1:]]
# # #     return None
# # #
# # # def load_motion_codes_from_vq(vq_path: str, codebook_size: int = 512) -> List[int]:
# # #     arr = np.load(vq_path, allow_pickle=False)
# # #     arr = np.asarray(arr).reshape(-1).tolist()
# # #     out = []
# # #     for x in arr:
# # #         try:
# # #             c = int(x)
# # #         except Exception:
# # #             continue
# # #         if 0 <= c < codebook_size:
# # #             out.append(c)
# # #     return out
# # #
# # # # -----------------------------
# # # # Candidate packing (rename to C01..)
# # # # -----------------------------
# # # def build_uniform_candidates_labeled_only(
# # #     gold_items: List[Dict[str, Any]],
# # #     silver_items: List[Dict[str, Any]],
# # #     neg_items: List[Dict[str, Any]],
# # #     seed: int,
# # #     max_total: int,
# # # ) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, str]]:
# # #     """
# # #     Returns:
# # #       candidates: [{"cid":"C01","caption":"..."}, ...]
# # #       cid2orig: {"C01":"gold_0", ...}
# # #       orig2type: {"gold_0":"gold","silver_0":"silver","neg_0":"neg"...}
# # #     """
# # #     all_items = []
# # #     orig2type: Dict[str, str] = {}
# # #
# # #     def add(items, t):
# # #         for it in items:
# # #             oid = str(it["id"])
# # #             all_items.append({"orig_id": oid, "caption": str(it["caption"])})
# # #             orig2type[oid] = t
# # #
# # #     add(gold_items, "gold")
# # #     add(silver_items, "silver")
# # #     add(neg_items, "neg")
# # #
# # #     rng = random.Random(int(seed))
# # #     rng.shuffle(all_items)
# # #     all_items = all_items[: max_total]
# # #
# # #     cid2orig: Dict[str, str] = {}
# # #     candidates: List[Dict[str, Any]] = []
# # #     for i, it in enumerate(all_items, start=1):
# # #         cid = f"C{i:02d}"
# # #         cid2orig[cid] = it["orig_id"]
# # #         candidates.append({"cid": cid, "caption": it["caption"]})
# # #     return candidates, cid2orig, orig2type
# # #
# # # # -----------------------------
# # # # Pairwise Qwen judge
# # # # -----------------------------
# # # def extract_last_json_object(text: str) -> Optional[Dict[str, Any]]:
# # #     if text is None:
# # #         return None
# # #     s = str(text)
# # #     starts = [i for i, ch in enumerate(s) if ch == "{"]  # keep all
# # #     if not starts:
# # #         return None
# # #
# # #     last_obj = None
# # #     for st in starts:
# # #         depth = 0
# # #         in_str = False
# # #         esc = False
# # #         for i in range(st, len(s)):
# # #             c = s[i]
# # #             if in_str:
# # #                 if esc:
# # #                     esc = False
# # #                 elif c == "\\":
# # #                     esc = True
# # #                 elif c == '"':
# # #                     in_str = False
# # #             else:
# # #                 if c == '"':
# # #                     in_str = True
# # #                 elif c == "{":
# # #                     depth += 1
# # #                 elif c == "}":
# # #                     depth -= 1
# # #                     if depth == 0:
# # #                         blob = s[st : i + 1]
# # #                         try:
# # #                             obj = json.loads(blob)
# # #                             last_obj = obj
# # #                         except Exception:
# # #                             pass
# # #                         break
# # #     return last_obj
# # #
# # # _CID_RE = re.compile(r"\bC\d{2}\b")
# # #
# # # def parse_winner_cid(raw: str, allowed: set) -> Optional[str]:
# # #     obj = extract_last_json_object(raw) or {}
# # #     for k in ["winner", "better", "choice", "answer", "selected"]:
# # #         v = obj.get(k, None)
# # #         if isinstance(v, str) and v in allowed:
# # #             return v
# # #
# # #     m = _CID_RE.findall(str(raw))
# # #     for cid in m:
# # #         if cid in allowed:
# # #             return cid
# # #     return None
# # #
# # # def build_qwen_pairwise_prompt(
# # #     query_payload: Dict[str, Any],
# # #     cand_a: Dict[str, Any],
# # #     cand_b: Dict[str, Any],
# # # ) -> str:
# # #     payload = {
# # #         "task": "Dyadic SPEAKER→LISTENER pairwise preference.",
# # #         "query": query_payload,
# # #         "candidates": [
# # #             {"cid": cand_a["cid"], "caption": cand_a["caption"]},
# # #             {"cid": cand_b["cid"], "caption": cand_b["caption"]},
# # #         ],
# # #         "rules": [
# # #             "You are comparing LISTENER reactions only (NOT speaker actions).",
# # #             "Pick the ONE candidate that is the more plausible listener non-verbal response to the speaker sayings/emotion.",
# # #             "Use ONLY the two captions as evidence; ids contain no label information.",
# # #             "Return ONLY JSON: {\"winner\": \"Cxx\"}. No other keys/text.",
# # #         ],
# # #     }
# # #     return (
# # #         "You are an expert evaluator for listener reactive motions.\n"
# # #         "IMPORTANT: Output MUST be a single valid JSON object and NOTHING ELSE.\n"
# # #         f"{json.dumps(payload, ensure_ascii=False)}"
# # #     )
# # #
# # # class QwenJudge:
# # #     def __init__(
# # #         self,
# # #         model_path: str,
# # #         use_vllm: bool = True,
# # #         tp: int = 1,
# # #         gpu_mem_util: float = 0.90,
# # #         max_new_tokens: int = 256,
# # #         temperature: float = 0.0,
# # #         max_model_len: int = 8192,
# # #     ):
# # #         self.model_path = model_path
# # #         self.use_vllm = use_vllm
# # #         self.tp = tp
# # #         self.gpu_mem_util = gpu_mem_util
# # #         self.max_new_tokens = max_new_tokens
# # #         self.temperature = temperature
# # #         self.max_model_len = max_model_len
# # #         self._mode = None
# # #
# # #         if use_vllm:
# # #             try:
# # #                 from vllm import LLM, SamplingParams  # type: ignore
# # #                 self._vllm_LLM = LLM(
# # #                     model=model_path,
# # #                     tensor_parallel_size=tp,
# # #                     gpu_memory_utilization=gpu_mem_util,
# # #                     trust_remote_code=True,
# # #                     max_model_len=max_model_len,
# # #                     enforce_eager=False,
# # #                 )
# # #                 self._vllm_SamplingParams = SamplingParams
# # #                 self._mode = "vllm"
# # #             except Exception as e:
# # #                 print(f"[WARN] vLLM init failed, fallback to HF. err={e}")
# # #                 self._mode = "hf"
# # #         else:
# # #             self._mode = "hf"
# # #
# # #         if self._mode == "hf":
# # #             self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# # #             self.model = AutoModelForCausalLM.from_pretrained(
# # #                 model_path,
# # #                 trust_remote_code=True,
# # #                 torch_dtype=torch.bfloat16,
# # #                 device_map="auto",
# # #             ).eval()
# # #
# # #     @torch.no_grad()
# # #     def generate(self, prompt: str) -> str:
# # #         if self._mode == "vllm":
# # #             sp = self._vllm_SamplingParams(
# # #                 temperature=self.temperature,
# # #                 max_tokens=self.max_new_tokens,
# # #                 top_p=1.0,
# # #             )
# # #             outs = self._vllm_LLM.generate([prompt], sp)
# # #             return outs[0].outputs[0].text.strip()
# # #
# # #         inputs = self.tok(prompt, return_tensors="pt")
# # #         inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
# # #         out = self.model.generate(
# # #             **inputs,
# # #             max_new_tokens=self.max_new_tokens,
# # #             do_sample=(self.temperature > 0),
# # #             temperature=max(1e-6, self.temperature),
# # #             top_p=1.0,
# # #         )
# # #         txt = self.tok.decode(out[0], skip_special_tokens=True)
# # #         if txt.startswith(prompt):
# # #             txt = txt[len(prompt):]
# # #         return txt.strip()
# # #
# # # # -----------------------------
# # # # Bradley–Terry fitting + ranking
# # # # -----------------------------
# # # def fit_btl_scores(
# # #     n_items: int,
# # #     matches: List[Tuple[int, int, int]],  # (i, j, y) y=1 => i wins else j wins
# # #     l2: float = 1e-2,
# # #     lr: float = 0.2,
# # #     steps: int = 600,
# # # ) -> np.ndarray:
# # #     s = np.zeros(n_items, dtype=np.float64)
# # #     if len(matches) == 0:
# # #         return s
# # #
# # #     for _ in range(steps):
# # #         grad = np.zeros_like(s)
# # #         for i, j, y in matches:
# # #             d = s[i] - s[j]
# # #             p = 1.0 / (1.0 + math.exp(-d))
# # #             g = (p - y)  # dL/dd
# # #             grad[i] += g
# # #             grad[j] -= g
# # #         grad += l2 * s
# # #         s -= lr * grad / max(1, len(matches))
# # #         s -= s.mean()  # remove shift ambiguity
# # #     return s
# # #
# # # def btl_order(scores: np.ndarray) -> List[int]:
# # #     return np.argsort(-scores).tolist()
# # #
# # # # -----------------------------
# # # # Metrics
# # # # -----------------------------
# # # def type_to_gain_graded(t: str) -> int:
# # #     # As requested: gold=2, silver=1, neg=0
# # #     if t == "gold":
# # #         return 2
# # #     if t == "silver":
# # #         return 1
# # #     return 0
# # #
# # # def ndcg_at_k(relevances: List[int], k: int) -> float:
# # #     rel = relevances[:k]
# # #     dcg = 0.0
# # #     for i, r in enumerate(rel, start=1):
# # #         if r > 0:
# # #             dcg += (2.0 ** r - 1.0) / math.log2(i + 1.0)
# # #
# # #     ideal = sorted(relevances, reverse=True)[:k]
# # #     idcg = 0.0
# # #     for i, r in enumerate(ideal, start=1):
# # #         if r > 0:
# # #             idcg += (2.0 ** r - 1.0) / math.log2(i + 1.0)
# # #
# # #     if idcg <= 1e-12:
# # #         return 0.0
# # #     return float(dcg / idcg)
# # #
# # # def winrate_best_typeA_vs_typeB_from_order(
# # #     order: List[int],          # permutation over indices 0..N-1
# # #     idx2type: List[str],       # type aligned with candidate index
# # #     typeA: str,
# # #     typeB: str,
# # # ) -> float:
# # #     rank = {idx: r for r, idx in enumerate(order)}
# # #     A = [i for i, t in enumerate(idx2type) if t == typeA]
# # #     B = [i for i, t in enumerate(idx2type) if t == typeB]
# # #     if len(A) == 0 or len(B) == 0:
# # #         return float("nan")
# # #     bestA = min(rank[i] for i in A)
# # #     wins = sum(1 for j in B if bestA < rank[j])
# # #     return float(wins / len(B))
# # #
# # # def type_at_k_from_order(order: List[int], idx2type: List[str], t: str, k: int) -> float:
# # #     topk = order[:k]
# # #     return 1.0 if any(idx2type[i] == t for i in topk) else 0.0
# # #
# # # # -----------------------------
# # # # Pair sampling strategy
# # # # -----------------------------
# # # def sample_pairs(n: int, max_pairs: int, seed: int) -> List[Tuple[int, int]]:
# # #     rng = random.Random(seed)
# # #     all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
# # #     if len(all_pairs) <= max_pairs:
# # #         rng.shuffle(all_pairs)
# # #         return all_pairs
# # #     return rng.sample(all_pairs, k=max_pairs)
# # #
# # # # -----------------------------
# # # # misc
# # # # -----------------------------
# # # def ensure_parent(path: str):
# # #     os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
# # #
# # # def hash_file(path: str) -> str:
# # #     h = hashlib.md5()
# # #     with open(path, "rb") as f:
# # #         while True:
# # #             b = f.read(1024 * 1024)
# # #             if not b:
# # #                 break
# # #             h.update(b)
# # #     return h.hexdigest()[:10]
# # #
# # # def stable_hash_str(s: str) -> str:
# # #     return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]
# # #
# # # def make_run_tag(args: argparse.Namespace) -> str:
# # #     parts = [
# # #         f"keval{args.k_eval}",
# # #         f"maxtotal{args.max_total_candidates}",
# # #         f"pairs{args.max_pairs}",
# # #         f"cap{stable_hash_str(args.caption_ckpt)}",
# # #         f"qwen{stable_hash_str(args.qwen_path)}",
# # #         f"ckmap{hash_file(args.ckpt_map_json)}",
# # #     ]
# # #     return "_".join(parts)
# # #
# # # def canon_label(x: str) -> str:
# # #     s = (x or "").strip().lower()
# # #     if s in {"gold", "pos", "positive", "gt", "true", "1"}:
# # #         return "gold"
# # #     if s in {"silver"}:
# # #         return "silver"
# # #     if s in {"neg", "negative", "0"}:
# # #         return "neg"
# # #     return s
# # #
# # # # -----------------------------
# # # # Main
# # # # -----------------------------
# # # def main():
# # #     ap = argparse.ArgumentParser()
# # #
# # #     ap.add_argument("--pairs_csv", type=str, required=True)
# # #     ap.add_argument("--dataset_dir", type=str, required=True)
# # #     ap.add_argument("--ckpt_map_json", type=str, required=True)
# # #
# # #     # captioner for labeled motions (if motion_caption not provided in CSV)
# # #     ap.add_argument("--caption_ckpt", type=str, required=True)
# # #
# # #     ap.add_argument("--max_gold", type=int, default=2)
# # #     ap.add_argument("--max_silver", type=int, default=8)
# # #     ap.add_argument("--max_neg", type=int, default=12)
# # #     ap.add_argument("--max_total_candidates", type=int, default=30)
# # #
# # #     ap.add_argument("--only_split", type=str, default="test")  # test/val/train/all
# # #     ap.add_argument("--group_by", type=str, default="group_id", choices=["group_id", "sayings_emotion"])
# # #
# # #     ap.add_argument("--k_eval", type=int, default=10)
# # #
# # #     # pairwise budget
# # #     ap.add_argument("--max_pairs", type=int, default=120)
# # #
# # #     # qwen
# # #     ap.add_argument("--qwen_path", type=str, required=True)
# # #     ap.add_argument("--qwen_use_vllm", action="store_true")
# # #     ap.add_argument("--qwen_tp", type=int, default=1)
# # #     ap.add_argument("--qwen_gpu_mem_util", type=float, default=0.90)
# # #     ap.add_argument("--qwen_max_new_tokens", type=int, default=256)
# # #     ap.add_argument("--qwen_max_model_len", type=int, default=8192)
# # #
# # #     ap.add_argument("--out_dir", type=str, default="./eval_ranker_only_out")
# # #     ap.add_argument("--seed", type=int, default=42)
# # #
# # #     args = ap.parse_args()
# # #     random.seed(args.seed)
# # #     np.random.seed(args.seed)
# # #     torch.manual_seed(args.seed)
# # #
# # #     os.makedirs(args.out_dir, exist_ok=True)
# # #     device = "cuda" if torch.cuda.is_available() else "cpu"
# # #     print("[Device]", device)
# # #
# # #     # ckpt map exists only for logging / run tag; ranker-only doesn't load A2RM
# # #     with open(args.ckpt_map_json, "r", encoding="utf-8") as f:
# # #         ckpt_map = json.load(f)
# # #
# # #     modes = ["a", "a+e", "t", "t+e", "t+a", "t+a+e"]
# # #     for m in modes:
# # #         if m not in ckpt_map:
# # #             raise RuntimeError(f"ckpt_map_json missing mode: {m}")
# # #
# # #     # read pairs
# # #     df = pd.read_csv(args.pairs_csv, encoding="utf-8")
# # #     required_cols = ["sayings", "emotion", "label", "raw_file_name", "split"]
# # #     missing = [c for c in required_cols if c not in df.columns]
# # #     if missing:
# # #         raise RuntimeError(f"Missing columns in pairs_csv: {missing}")
# # #
# # #     df["label"] = df["label"].astype(str).str.lower().str.strip()
# # #     df["sayings"] = df["sayings"].astype(str).fillna("")
# # #     df["emotion"] = df["emotion"].astype(str).fillna("")
# # #     df["raw_file_name"] = df["raw_file_name"].astype(str).fillna("")
# # #     df["split"] = df["split"].astype(str).str.lower().str.strip()
# # #
# # #     if args.only_split != "all":
# # #         df = df[df["split"] == args.only_split].copy()
# # #     if len(df) == 0:
# # #         raise RuntimeError(f"No rows for split={args.only_split}")
# # #
# # #     # group
# # #     if args.group_by == "group_id":
# # #         if "group_id" not in df.columns:
# # #             print("[WARN] group_by=group_id but no group_id column; fallback to sayings_emotion")
# # #             args.group_by = "sayings_emotion"
# # #
# # #     if args.group_by == "group_id":
# # #         groups = list(df.groupby(["group_id"], dropna=False))
# # #     else:
# # #         groups = list(df.groupby(["sayings", "emotion"], dropna=False))
# # #     print("[Groups]", len(groups), "group_by=", args.group_by)
# # #
# # #     # VQ index
# # #     motion_vq_dir = os.path.join(args.dataset_dir, "HumanML3D", "VQVAE")
# # #     if not os.path.isdir(motion_vq_dir):
# # #         raise RuntimeError(f"Missing motion_vq_dir: {motion_vq_dir}")
# # #     vq_by_stem = build_vq_index(motion_vq_dir)
# # #     print("[VQ] indexed:", len(vq_by_stem))
# # #
# # #     # caption model
# # #     cap_tok = T5Tokenizer.from_pretrained(args.caption_ckpt)
# # #     cap_model = T5ForConditionalGeneration.from_pretrained(args.caption_ckpt).to(device).eval()
# # #
# # #     @torch.no_grad()
# # #     def caption_motion_codes(codes: List[int]) -> str:
# # #         prompt = build_caption_prompt(codes)
# # #         inp = cap_tok(prompt, return_tensors="pt").input_ids.to(device, dtype=torch.long)
# # #         out = cap_model.generate(inp, max_length=200, num_beams=1, do_sample=False)
# # #         txt = cap_tok.decode(out[0], skip_special_tokens=True).strip().strip('"')
# # #         return txt
# # #
# # #     # qwen
# # #     qwen = QwenJudge(
# # #         model_path=args.qwen_path,
# # #         use_vllm=args.qwen_use_vllm,
# # #         tp=args.qwen_tp,
# # #         gpu_mem_util=args.qwen_gpu_mem_util,
# # #         max_new_tokens=args.qwen_max_new_tokens,
# # #         temperature=0.0,
# # #         max_model_len=args.qwen_max_model_len,
# # #     )
# # #     print("[Qwen] model=", args.qwen_path, "backend=", ("vllm" if args.qwen_use_vllm else "hf"))
# # #
# # #     # output file
# # #     run_tag = make_run_tag(args)
# # #     out_csv = os.path.join(args.out_dir, f"eval_ranker_only_{run_tag}.csv")
# # #     ensure_parent(out_csv)
# # #
# # #     fieldnames = [
# # #         "eval_key",
# # #         "mode",
# # #         "split",
# # #         "group_key",
# # #         "sayings",
# # #         "emotion",
# # #         "num_gold",
# # #         "num_silver",
# # #         "num_neg",
# # #         "k_eval",
# # #         "ndcgK_graded",          # gold=2, silver=1, neg=0
# # #         "win_gen_vs_neg",        # == win_gold_vs_neg
# # #         "win_gen_vs_silver",     # == win_gold_vs_silver
# # #         "win_gen_vs_gold",       # == win_silver_vs_gold (lower is better)
# # #         "win_gold_vs_neg",
# # #         "win_gold_vs_silver",
# # #         "win_silver_vs_gold",
# # #         "gold_at3",
# # #         "topk_orig_json",
# # #         "topk_types_json",
# # #     ]
# # #
# # #     exists = os.path.isfile(out_csv)
# # #     fcsv = open(out_csv, "a", encoding="utf-8", newline="")
# # #     writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
# # #     if not exists:
# # #         writer.writeheader()
# # #         fcsv.flush()
# # #         os.fsync(fcsv.fileno())
# # #
# # #     def make_eval_key(mode: str, group_key: str) -> str:
# # #         s = f"{mode}|||{group_key}"
# # #         return hashlib.md5(s.encode("utf-8")).hexdigest()
# # #
# # #     # build labeled items for a group
# # #     def build_items(
# # #         gg_all: pd.DataFrame,
# # #         lbl: str,
# # #         max_n: int,
# # #     ) -> List[Dict[str, Any]]:
# # #         sub = gg_all[gg_all["label_canon"] == lbl].copy()
# # #         if len(sub) == 0:
# # #             return []
# # #         mids = list(dict.fromkeys(sub["motion_id"].tolist()))[:max_n]
# # #
# # #         use_csv_caption = ("motion_caption" in gg_all.columns)
# # #         items = []
# # #         for i, mid in enumerate(mids):
# # #             cap = ""
# # #             if use_csv_caption:
# # #                 ss = sub[sub["motion_id"] == mid]
# # #                 if "motion_caption" in ss.columns and len(ss) > 0:
# # #                     cap = str(ss["motion_caption"].iloc[0]).strip()
# # #
# # #             if not cap:
# # #                 p = vqvae_lookup(vq_by_stem, mid)
# # #                 if p is None:
# # #                     continue
# # #                 codes = load_motion_codes_from_vq(p)
# # #                 cap = caption_motion_codes(codes)
# # #
# # #             items.append({"id": f"{lbl}_{i}", "caption": cap})
# # #         return items
# # #
# # #     def run_pairwise_btl(
# # #         query_payload: Dict[str, Any],
# # #         candidates: List[Dict[str, Any]],  # [{"cid","caption"}]
# # #         max_pairs: int,
# # #         seed_int: int,
# # #     ) -> List[int]:
# # #         N = len(candidates)
# # #         if N <= 1:
# # #             return list(range(N))
# # #
# # #         pairs = sample_pairs(N, max_pairs=max_pairs, seed=seed_int ^ 0x13579BDF)
# # #         allowed = {c["cid"] for c in candidates}
# # #         matches: List[Tuple[int, int, int]] = []
# # #
# # #         for (i, j) in pairs:
# # #             ca, cb = candidates[i], candidates[j]
# # #             prompt = build_qwen_pairwise_prompt(query_payload, ca, cb)
# # #             out = qwen.generate(prompt)
# # #             winner = parse_winner_cid(out, allowed=allowed)
# # #
# # #             # safer fallback: if parse failed, SKIP this match (avoid systematic bias)
# # #             if winner is None:
# # #                 continue
# # #
# # #             y = 1 if winner == ca["cid"] else 0
# # #             matches.append((i, j, y))
# # #
# # #         scores = fit_btl_scores(N, matches)
# # #         return btl_order(scores)
# # #
# # #     # ----------------------------
# # #     # main loop
# # #     # ----------------------------
# # #     for mode in modes:
# # #         for keys, g in tqdm(groups, desc=f"EvalRankerOnly[{mode}]"):
# # #             if args.group_by == "group_id":
# # #                 group_key = str(keys) if not isinstance(keys, tuple) else str(keys[0])
# # #                 sayings = str(g["sayings"].iloc[0])
# # #                 emotion = str(g["emotion"].iloc[0])
# # #             else:
# # #                 sayings, emotion = keys
# # #                 sayings = str(sayings)
# # #                 emotion = str(emotion)
# # #                 group_key = f"{sayings}|||{emotion}"
# # #
# # #             split = str(g["split"].iloc[0])
# # #
# # #             gg_all = g.copy()
# # #             gg_all["label_canon"] = gg_all["label"].apply(canon_label)
# # #             gg_all["motion_id"] = gg_all["raw_file_name"].apply(motion_id_from_raw)
# # #
# # #             gold_items = build_items(gg_all, "gold", args.max_gold)
# # #             silver_items = build_items(gg_all, "silver", args.max_silver)
# # #             neg_items = build_items(gg_all, "neg", args.max_neg)
# # #
# # #             eval_key = make_eval_key(mode, group_key)
# # #
# # #             query_payload = {
# # #                 "speaker_sayings": sayings,
# # #                 "speaker_emotion": emotion,
# # #                 "cond_mode": mode,
# # #                 "note": "Rank LISTENER non-verbal responses to the SPEAKER's utterance.",
# # #             }
# # #
# # #             seed_int = int(hashlib.md5(f"{group_key}|||{mode}".encode("utf-8")).hexdigest()[:8], 16)
# # #
# # #             cand, cid2orig, orig2type = build_uniform_candidates_labeled_only(
# # #                 gold_items=gold_items,
# # #                 silver_items=silver_items,
# # #                 neg_items=neg_items,
# # #                 seed=seed_int ^ 0xBADC0DE,
# # #                 max_total=args.max_total_candidates,
# # #             )
# # #
# # #             if len(cand) == 0:
# # #                 row = dict(
# # #                     eval_key=eval_key,
# # #                     mode=mode,
# # #                     split=split,
# # #                     group_key=group_key,
# # #                     sayings=sayings,
# # #                     emotion=emotion,
# # #                     num_gold=len(gold_items),
# # #                     num_silver=len(silver_items),
# # #                     num_neg=len(neg_items),
# # #                     k_eval=0,
# # #                     ndcgK_graded=float("nan"),
# # #                     win_gen_vs_neg=float("nan"),
# # #                     win_gen_vs_silver=float("nan"),
# # #                     win_gen_vs_gold=float("nan"),
# # #                     win_gold_vs_neg=float("nan"),
# # #                     win_gold_vs_silver=float("nan"),
# # #                     win_silver_vs_gold=float("nan"),
# # #                     gold_at3=float("nan"),
# # #                     topk_orig_json="[]",
# # #                     topk_types_json="[]",
# # #                 )
# # #                 writer.writerow(row)
# # #                 fcsv.flush()
# # #                 os.fsync(fcsv.fileno())
# # #                 continue
# # #
# # #             order = run_pairwise_btl(
# # #                 query_payload=query_payload,
# # #                 candidates=cand,
# # #                 max_pairs=args.max_pairs,
# # #                 seed_int=seed_int,
# # #             )
# # #
# # #             # ranked lists
# # #             ranked_orig = [cid2orig[cand[i]["cid"]] for i in order]
# # #             ranked_types = [orig2type.get(oid, "neg") for oid in ranked_orig]  # default neg
# # #             k_eval = min(int(args.k_eval), len(ranked_types))
# # #             if k_eval <= 0:
# # #                 k_eval = min(5, len(ranked_types))
# # #
# # #             # nDCG graded (gold=2,silver=1,neg=0) on the *ranker order*
# # #             rels = [type_to_gain_graded(t) for t in ranked_types]
# # #             ndcgK_graded = ndcg_at_k(rels, k=k_eval)
# # #
# # #             # idx2type aligned with candidate index (IMPORTANT!)
# # #             idx2type = [orig2type.get(cid2orig[c["cid"]], "neg") for c in cand]
# # #
# # #             # winrates among labeled types
# # #             win_gold_vs_neg = winrate_best_typeA_vs_typeB_from_order(order, idx2type, "gold", "neg")
# # #             win_gold_vs_silver = winrate_best_typeA_vs_typeB_from_order(order, idx2type, "gold", "silver")
# # #             win_silver_vs_gold = winrate_best_typeA_vs_typeB_from_order(order, idx2type, "silver", "gold")
# # #
# # #             # keep requested legacy names (mapping described at top)
# # #             win_gen_vs_neg = win_gold_vs_neg
# # #             win_gen_vs_silver = win_gold_vs_silver
# # #             win_gen_vs_gold = win_silver_vs_gold  # lower is better
# # #
# # #             gold_at3 = type_at_k_from_order(order, idx2type, "gold", k=3)
# # #
# # #             row = dict(
# # #                 eval_key=eval_key,
# # #                 mode=mode,
# # #                 split=split,
# # #                 group_key=group_key,
# # #                 sayings=sayings,
# # #                 emotion=emotion,
# # #                 num_gold=len(gold_items),
# # #                 num_silver=len(silver_items),
# # #                 num_neg=len(neg_items),
# # #                 k_eval=k_eval,
# # #                 ndcgK_graded=ndcgK_graded,
# # #                 win_gen_vs_neg=win_gen_vs_neg,
# # #                 win_gen_vs_silver=win_gen_vs_silver,
# # #                 win_gen_vs_gold=win_gen_vs_gold,
# # #                 win_gold_vs_neg=win_gold_vs_neg,
# # #                 win_gold_vs_silver=win_gold_vs_silver,
# # #                 win_silver_vs_gold=win_silver_vs_gold,
# # #                 gold_at3=gold_at3,
# # #                 topk_orig_json=json.dumps(ranked_orig[:k_eval], ensure_ascii=False),
# # #                 topk_types_json=json.dumps(ranked_types[:k_eval], ensure_ascii=False),
# # #             )
# # #             writer.writerow(row)
# # #             fcsv.flush()
# # #             os.fsync(fcsv.fileno())
# # #
# # #     fcsv.close()
# # #     print("[Saved]", out_csv)
# # #
# # #
# # # if __name__ == "__main__":
# # #     main()
# #
# # #!/usr/bin/env python3
# # # -*- coding: utf-8 -*-
# #
# # import os, re, json, csv, math, argparse, random, hashlib
# # from typing import Dict, List, Tuple, Optional, Any
# #
# # import numpy as np
# # import pandas as pd
# # from tqdm import tqdm
# #
# # import torch
# # from transformers import (
# #     T5Tokenizer,
# #     T5ForConditionalGeneration,
# #     AutoTokenizer,
# #     AutoModelForCausalLM,
# # )
# #
# # # -----------------------------
# # # Motion token parsing (A2RM output)
# # # -----------------------------
# # _MOTION_SPAN_RE = re.compile(r"<Motion Tokens>(.*?)</Motion Tokens>", re.DOTALL)
# # _MOTION_TOKEN_RE = re.compile(r"<Motion Token\s+(\d+)>")
# # _MOTION_TOKEN_SHORT_RE = re.compile(r"<(\d+)>")  # <123>
# #
# # def parse_motion_tokens(text: str, max_len: int = 200, codebook_size: int = 512) -> List[int]:
# #     if text is None:
# #         return []
# #     s = str(text)
# #     m = _MOTION_SPAN_RE.search(s)
# #     span = m.group(1) if m else s
# #
# #     codes = [int(x) for x in _MOTION_TOKEN_RE.findall(span)]
# #     if len(codes) == 0:
# #         codes = [int(x) for x in _MOTION_TOKEN_SHORT_RE.findall(span)]
# #
# #     out = []
# #     for c in codes:
# #         if 0 <= c < codebook_size:
# #             out.append(c)
# #         else:
# #             break
# #     return out[:max_len]
# #
# # # -----------------------------
# # # Prompt builder (MATCH TRAINING)
# # # -----------------------------
# # def build_prompt_condmode(
# #     speaker_transcription: str,
# #     speaker_audio: str,
# #     speaker_emotion: str,
# #     cond_mode: str,
# # ) -> str:
# #     cm = (cond_mode or "").strip().lower()
# #     use_transcription = ("t" in cm)
# #     use_audio = ("a" in cm)
# #     use_emotion = (cm.endswith("+e") or cm in ("a+e", "t+e", "t+a+e"))
# #
# #     t = (speaker_transcription or "").strip()
# #     a = (speaker_audio or "").strip()
# #     e = (speaker_emotion or "").strip()
# #
# #     lines = []
# #     lines.append("You are modeling a speaker-listener dyadic interaction.\n\n")
# #     lines.append("Input:\n")
# #     lines.append(f"- SPEAKER_TRANSCRIPTION: {t if use_transcription else ''}\n")
# #     lines.append(f"- SPEAKER_AUDIO: {a if use_audio else ''}\n")
# #     if use_emotion and e:
# #         lines.append(f"- SPEAKER_EMOTION: <Emotion> {e} </Emotion>\n")
# #     lines.append("\nOutput:\n")
# #     lines.append("Return ONLY a sequence of listener motion tokens in the exact format:\n")
# #     lines.append("<Motion Tokens> <Motion Token i> ... </Motion Tokens>\n")
# #     lines.append("Do NOT output any other words.\n")
# #     return "".join(lines).strip()
# #
# # # -----------------------------
# # # Caption model helpers (T5)
# # # -----------------------------
# # def build_caption_prompt(motion_codes: List[int]) -> str:
# #     motion_string = "<Motion Tokens>" + "".join([f"<{c}>" for c in motion_codes]) + "</Motion Tokens>"
# #     return "Generate text: " + motion_string
# #
# # # -----------------------------
# # # Audio token formatting
# # # -----------------------------
# # def load_audio_tokens_any(path: str) -> np.ndarray:
# #     obj = np.load(path, allow_pickle=False)
# #     if isinstance(obj, np.lib.npyio.NpzFile):
# #         if "codes" in obj.files:
# #             arr = obj["codes"]
# #         else:
# #             arr = obj[obj.files[0]]
# #         obj.close()
# #         return arr
# #     return obj
# #
# # def format_audio_tokens(a_tokens: np.ndarray, level: str = "base") -> str:
# #     arr = np.array(a_tokens)
# #     level = str(level)
# #
# #     if arr.ndim == 1:
# #         parts = ["<Audio Tokens>"]
# #         for t in arr.reshape(-1):
# #             parts.append(f"<Audio Token {int(t)}>")
# #         parts.append("</Audio Tokens>")
# #         return " ".join(parts)
# #
# #     L = int(arr.shape[0])
# #     parts = ["<Audio Tokens>"]
# #
# #     if level == "base":
# #         for t in arr[0].reshape(-1):
# #             parts.append(f"<Audio Level 0 Token {int(t)}>")
# #     elif level == "all":
# #         for i in range(L):
# #             for t in arr[i].reshape(-1):
# #                 parts.append(f"<Audio Level {i} Token {int(t)}>")
# #     elif level == "rand":
# #         k = int(np.random.choice(np.arange(1, L + 1)))
# #         for i in range(k):
# #             for t in arr[i].reshape(-1):
# #                 parts.append(f"<Audio Level {i} Token {int(t)}>")
# #     else:
# #         raise ValueError(f"Unknown audio_token_level={level}")
# #
# #     parts.append("</Audio Tokens>")
# #     return " ".join(parts)
# #
# # def pick_code_from_stem(code_dir: str, stem: str) -> Optional[str]:
# #     stem = str(stem).strip()
# #     if not stem:
# #         return None
# #     p_npz = os.path.join(code_dir, stem + ".npz")
# #     if os.path.exists(p_npz):
# #         return p_npz
# #     p_npy = os.path.join(code_dir, stem + ".npy")
# #     if os.path.exists(p_npy):
# #         return p_npy
# #     return None
# #
# # # -----------------------------
# # # VQ lookup for labeled motions
# # # -----------------------------
# # def motion_id_from_raw(raw_file_name: str) -> str:
# #     s = str(raw_file_name)
# #     mid = s.split("_", 1)[0]
# #     return str(mid).zfill(6)
# #
# # def build_vq_index(vq_dir: str) -> Dict[str, str]:
# #     m = {}
# #     for fn in os.listdir(vq_dir):
# #         if fn.endswith(".npy"):
# #             stem = os.path.splitext(fn)[0]
# #             m[stem] = os.path.join(vq_dir, fn)
# #     return m
# #
# # def vqvae_lookup(vq_by_stem: Dict[str, str], motion_id: str) -> Optional[str]:
# #     base = str(motion_id)
# #     if base in vq_by_stem:
# #         return vq_by_stem[base]
# #     if base.isdigit() and ("M" + base) in vq_by_stem:
# #         return vq_by_stem["M" + base]
# #     if base.startswith("M") and base[1:].isdigit() and (base[1:] in vq_by_stem):
# #         return vq_by_stem[base[1:]]
# #     return None
# #
# # def load_motion_codes_from_vq(vq_path: str, codebook_size: int = 512) -> List[int]:
# #     arr = np.load(vq_path, allow_pickle=False)
# #     arr = np.asarray(arr).reshape(-1).tolist()
# #     out = []
# #     for x in arr:
# #         try:
# #             c = int(x)
# #         except Exception:
# #             continue
# #         if 0 <= c < codebook_size:
# #             out.append(c)
# #     return out
# #
# # # -----------------------------
# # # Candidate packing (rename to C01..)
# # # -----------------------------
# # def build_uniform_candidates(
# #     gen_items: List[Dict[str, Any]],
# #     gold_items: List[Dict[str, Any]],
# #     silver_items: List[Dict[str, Any]],
# #     neg_items: List[Dict[str, Any]],
# #     seed: int,
# #     max_total: int,
# # ) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, str]]:
# #     """
# #     Returns:
# #       candidates: [{"cid":"C01","caption":"..."}, ...] (shuffled)
# #       cid2orig: {"C01":"gen_0", ...}
# #       orig2type: {"gen_0":"gen","gold_0":"gold","silver_0":"silver","neg_0":"neg"...}
# #     """
# #     all_items = []
# #     orig2type: Dict[str, str] = {}
# #
# #     def add(items, t):
# #         for it in items:
# #             oid = str(it["id"])
# #             all_items.append({"orig_id": oid, "caption": str(it["caption"])})
# #             orig2type[oid] = t
# #
# #     add(gen_items, "gen")
# #     add(gold_items, "gold")
# #     add(silver_items, "silver")
# #     add(neg_items, "neg")
# #
# #     rng = random.Random(int(seed))
# #     rng.shuffle(all_items)
# #     all_items = all_items[: max_total]
# #
# #     cid2orig: Dict[str, str] = {}
# #     candidates: List[Dict[str, Any]] = []
# #
# #     for i, it in enumerate(all_items, start=1):
# #         cid = f"C{i:02d}"
# #         cid2orig[cid] = it["orig_id"]
# #         candidates.append({"cid": cid, "caption": it["caption"]})
# #     return candidates, cid2orig, orig2type
# #
# # # -----------------------------
# # # Pairwise Qwen judge
# # # -----------------------------
# # def extract_last_json_object(text: str) -> Optional[Dict[str, Any]]:
# #     if text is None:
# #         return None
# #     s = str(text)
# #     starts = [i for i, ch in enumerate(s) if ch == "{"]  # keep all
# #     if not starts:
# #         return None
# #
# #     last_obj = None
# #     for st in starts:
# #         depth = 0
# #         in_str = False
# #         esc = False
# #         for i in range(st, len(s)):
# #             c = s[i]
# #             if in_str:
# #                 if esc:
# #                     esc = False
# #                 elif c == "\\":
# #                     esc = True
# #                 elif c == '"':
# #                     in_str = False
# #             else:
# #                 if c == '"':
# #                     in_str = True
# #                 elif c == "{":
# #                     depth += 1
# #                 elif c == "}":
# #                     depth -= 1
# #                     if depth == 0:
# #                         blob = s[st : i + 1]
# #                         try:
# #                             obj = json.loads(blob)
# #                             last_obj = obj
# #                         except Exception:
# #                             pass
# #                         break
# #     return last_obj
# #
# # _CID_RE = re.compile(r"\bC\d{2}\b")
# #
# # def parse_winner_cid(raw: str, allowed: set) -> Optional[str]:
# #     obj = extract_last_json_object(raw) or {}
# #     for k in ["winner", "better", "choice", "answer", "selected"]:
# #         v = obj.get(k, None)
# #         if isinstance(v, str) and v in allowed:
# #             return v
# #
# #     m = _CID_RE.findall(str(raw))
# #     for cid in m:
# #         if cid in allowed:
# #             return cid
# #     return None
# #
# # def build_qwen_pairwise_prompt(
# #     query_payload: Dict[str, Any],
# #     cand_a: Dict[str, Any],
# #     cand_b: Dict[str, Any],
# # ) -> str:
# #     payload = {
# #         "task": "Dyadic SPEAKER→LISTENER pairwise preference.",
# #         "query": query_payload,
# #         "candidates": [
# #             {"cid": cand_a["cid"], "caption": cand_a["caption"]},
# #             {"cid": cand_b["cid"], "caption": cand_b["caption"]},
# #         ],
# #         "rules": [
# #             "You are comparing LISTENER reactions only (NOT speaker actions).",
# #             "Pick the ONE candidate that is the more plausible listener non-verbal response to the speaker sayings/emotion.",
# #             "Use ONLY the two captions as evidence; ids contain no label information.",
# #             "Return ONLY JSON: {\"winner\": \"Cxx\"}. No other keys/text.",
# #         ],
# #     }
# #     return (
# #         "You are an expert evaluator for listener reactive motions.\n"
# #         "IMPORTANT: Output MUST be a single valid JSON object and NOTHING ELSE.\n"
# #         f"{json.dumps(payload, ensure_ascii=False)}"
# #     )
# #
# # class QwenJudge:
# #     def __init__(
# #         self,
# #         model_path: str,
# #         use_vllm: bool = True,
# #         tp: int = 1,
# #         gpu_mem_util: float = 0.90,
# #         max_new_tokens: int = 256,
# #         temperature: float = 0.0,
# #         max_model_len: int = 8192,
# #     ):
# #         self.model_path = model_path
# #         self.use_vllm = use_vllm
# #         self.tp = tp
# #         self.gpu_mem_util = gpu_mem_util
# #         self.max_new_tokens = max_new_tokens
# #         self.temperature = temperature
# #         self.max_model_len = max_model_len
# #         self._mode = None
# #
# #         if use_vllm:
# #             try:
# #                 from vllm import LLM, SamplingParams  # type: ignore
# #                 self._vllm_LLM = LLM(
# #                     model=model_path,
# #                     tensor_parallel_size=tp,
# #                     gpu_memory_utilization=gpu_mem_util,
# #                     trust_remote_code=True,
# #                     max_model_len=max_model_len,
# #                     enforce_eager=False,
# #                 )
# #                 self._vllm_SamplingParams = SamplingParams
# #                 self._mode = "vllm"
# #             except Exception as e:
# #                 print(f"[WARN] vLLM init failed, fallback to HF. err={e}")
# #                 self._mode = "hf"
# #         else:
# #             self._mode = "hf"
# #
# #         if self._mode == "hf":
# #             self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# #             self.model = AutoModelForCausalLM.from_pretrained(
# #                 model_path,
# #                 trust_remote_code=True,
# #                 torch_dtype=torch.bfloat16,
# #                 device_map="auto",
# #             ).eval()
# #
# #     @torch.no_grad()
# #     def generate(self, prompt: str) -> str:
# #         if self._mode == "vllm":
# #             sp = self._vllm_SamplingParams(
# #                 temperature=self.temperature,
# #                 max_tokens=self.max_new_tokens,
# #                 top_p=1.0,
# #             )
# #             outs = self._vllm_LLM.generate([prompt], sp)
# #             return outs[0].outputs[0].text.strip()
# #
# #         inputs = self.tok(prompt, return_tensors="pt")
# #         inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
# #         out = self.model.generate(
# #             **inputs,
# #             max_new_tokens=self.max_new_tokens,
# #             do_sample=(self.temperature > 0),
# #             temperature=max(1e-6, self.temperature),
# #             top_p=1.0,
# #         )
# #         txt = self.tok.decode(out[0], skip_special_tokens=True)
# #         if txt.startswith(prompt):
# #             txt = txt[len(prompt):]
# #         return txt.strip()
# #
# # # -----------------------------
# # # Bradley–Terry fitting + ranking
# # # -----------------------------
# # def fit_btl_scores(
# #     n_items: int,
# #     matches: List[Tuple[int, int, int]],  # (i, j, y) y=1 => i wins else j wins
# #     l2: float = 1e-2,
# #     lr: float = 0.2,
# #     steps: int = 600,
# # ) -> np.ndarray:
# #     s = np.zeros(n_items, dtype=np.float64)
# #     if len(matches) == 0:
# #         return s
# #
# #     for _ in range(steps):
# #         grad = np.zeros_like(s)
# #         for i, j, y in matches:
# #             d = s[i] - s[j]
# #             p = 1.0 / (1.0 + math.exp(-d))
# #             g = (p - y)  # dL/dd
# #             grad[i] += g
# #             grad[j] -= g
# #         grad += l2 * s
# #         s -= lr * grad / max(1, len(matches))
# #         s -= s.mean()
# #     return s
# #
# # def btl_order(scores: np.ndarray) -> List[int]:
# #     return np.argsort(-scores).tolist()
# #
# # # -----------------------------
# # # Metrics
# # # -----------------------------
# # def type_to_gain_graded_labeled(t: str) -> int:
# #     # For ranker ability only: gold=2, silver=1, neg=0
# #     if t == "gold":
# #         return 2
# #     if t == "silver":
# #         return 1
# #     return 0
# #
# # def ndcg_at_k(relevances: List[int], k: int) -> float:
# #     rel = relevances[:k]
# #     dcg = 0.0
# #     for i, r in enumerate(rel, start=1):
# #         if r > 0:
# #             dcg += (2.0 ** r - 1.0) / math.log2(i + 1.0)
# #
# #     ideal = sorted(relevances, reverse=True)[:k]
# #     idcg = 0.0
# #     for i, r in enumerate(ideal, start=1):
# #         if r > 0:
# #             idcg += (2.0 ** r - 1.0) / math.log2(i + 1.0)
# #
# #     if idcg <= 1e-12:
# #         return 0.0
# #     return float(dcg / idcg)
# #
# # def winrate_best_gen_vs_type_from_order(order: List[int], idx2type: List[str], comp_type: str) -> float:
# #     # IMPORTANT: idx2type MUST align with candidate indices (0..N-1), not ranked order
# #     rank = {idx: r for r, idx in enumerate(order)}
# #     gen_idxs = [i for i, t in enumerate(idx2type) if t == "gen"]
# #     comp_idxs = [i for i, t in enumerate(idx2type) if t == comp_type]
# #     if len(gen_idxs) == 0 or len(comp_idxs) == 0:
# #         return float("nan")
# #     best_gen_rank = min(rank[i] for i in gen_idxs)
# #     wins = sum(1 for j in comp_idxs if best_gen_rank < rank[j])
# #     return float(wins / len(comp_idxs))
# #
# # def gen_at_k_from_order(order: List[int], idx2type: List[str], k: int = 3) -> float:
# #     topk = order[:k]
# #     return 1.0 if any(idx2type[i] == "gen" for i in topk) else 0.0
# #
# # # -----------------------------
# # # Pair sampling
# # # -----------------------------
# # def sample_pairs(n: int, max_pairs: int, seed: int) -> List[Tuple[int, int]]:
# #     rng = random.Random(seed)
# #     all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
# #     if len(all_pairs) <= max_pairs:
# #         rng.shuffle(all_pairs)
# #         return all_pairs
# #     return rng.sample(all_pairs, k=max_pairs)
# #
# # # -----------------------------
# # # misc
# # # -----------------------------
# # def ensure_parent(path: str):
# #     os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
# #
# # def hash_file(path: str) -> str:
# #     h = hashlib.md5()
# #     with open(path, "rb") as f:
# #         while True:
# #             b = f.read(1024 * 1024)
# #             if not b:
# #                 break
# #             h.update(b)
# #     return h.hexdigest()[:10]
# #
# # def stable_hash_str(s: str) -> str:
# #     return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]
# #
# # def canon_label(x: str) -> str:
# #     s = (x or "").strip().lower()
# #     if s in {"gold", "pos", "positive", "gt", "true", "1"}:
# #         return "gold"
# #     if s in {"silver"}:
# #         return "silver"
# #     if s in {"neg", "negative", "0"}:
# #         return "neg"
# #     return s
# #
# # def make_run_tag(args: argparse.Namespace) -> str:
# #     parts = [
# #         f"numgen{args.num_gen}",
# #         f"keval{args.k_eval}",
# #         f"maxtotal{args.max_total_candidates}",
# #         f"pairs{args.max_pairs}",
# #         f"pairsL{args.max_pairs_labeled}",
# #         f"cap{stable_hash_str(args.caption_ckpt)}",
# #         f"qwen{stable_hash_str(args.qwen_path)}",
# #         f"ckmap{hash_file(args.ckpt_map_json)}",
# #     ]
# #     return "_".join(parts)
# #
# # # -----------------------------
# # # Main
# # # -----------------------------
# # def main():
# #     ap = argparse.ArgumentParser()
# #
# #     ap.add_argument("--pairs_csv", type=str, required=True)
# #     ap.add_argument("--dataset_dir", type=str, required=True)
# #     ap.add_argument("--ckpt_map_json", type=str, required=True)
# #
# #     ap.add_argument("--audio_code_dir", type=str, default=None)
# #     ap.add_argument("--audio_token_level", type=str, default="base", choices=["base", "all", "rand"])
# #
# #     ap.add_argument("--caption_ckpt", type=str, required=True)
# #
# #     # generation (model ability)
# #     ap.add_argument("--num_gen", type=int, default=3)
# #     ap.add_argument("--gen_max_len", type=int, default=200)
# #     ap.add_argument("--gen_temperature", type=float, default=0.8)
# #     ap.add_argument("--gen_top_k", type=int, default=200)
# #
# #     # candidate budgets
# #     ap.add_argument("--max_gold", type=int, default=1)
# #     ap.add_argument("--max_silver", type=int, default=8)
# #     ap.add_argument("--max_neg", type=int, default=12)
# #     ap.add_argument("--max_total_candidates", type=int, default=30)
# #
# #     ap.add_argument("--only_split", type=str, default="test")
# #     ap.add_argument("--group_by", type=str, default="group_id", choices=["group_id", "sayings_emotion"])
# #
# #     ap.add_argument("--k_eval", type=int, default=10)
# #
# #     # pairwise budgets
# #     ap.add_argument("--max_pairs", type=int, default=120)         # full pool (gen+labeled)
# #     ap.add_argument("--max_pairs_labeled", type=int, default=80)  # labeled-only pool (ranker ability)
# #
# #     # qwen
# #     ap.add_argument("--qwen_path", type=str, required=True)
# #     ap.add_argument("--qwen_use_vllm", action="store_true")
# #     ap.add_argument("--qwen_tp", type=int, default=1)
# #     ap.add_argument("--qwen_gpu_mem_util", type=float, default=0.90)
# #     ap.add_argument("--qwen_max_new_tokens", type=int, default=256)
# #     ap.add_argument("--qwen_max_model_len", type=int, default=8192)
# #
# #     ap.add_argument("--out_dir", type=str, default="./eval_gen_and_ranker_out")
# #     ap.add_argument("--seed", type=int, default=42)
# #
# #     args = ap.parse_args()
# #     random.seed(args.seed)
# #     np.random.seed(args.seed)
# #     torch.manual_seed(args.seed)
# #
# #     os.makedirs(args.out_dir, exist_ok=True)
# #     device = "cuda" if torch.cuda.is_available() else "cpu"
# #     print("[Device]", device)
# #
# #     # ckpt map
# #     with open(args.ckpt_map_json, "r", encoding="utf-8") as f:
# #         ckpt_map = json.load(f)
# #
# #     modes = ["a", "a+e", "t", "t+e", "t+a", "t+a+e"]
# #     for m in modes:
# #         if m not in ckpt_map:
# #             raise RuntimeError(f"ckpt_map_json missing mode: {m}")
# #
# #     # read pairs
# #     df = pd.read_csv(args.pairs_csv, encoding="utf-8")
# #     required_cols = ["sayings", "emotion", "label", "raw_file_name", "split", "generated_wav_name"]
# #     missing = [c for c in required_cols if c not in df.columns]
# #     if missing:
# #         raise RuntimeError(f"Missing columns in pairs_csv: {missing}")
# #
# #     df["label"] = df["label"].astype(str).str.lower().str.strip()
# #     df["sayings"] = df["sayings"].astype(str).fillna("")
# #     df["emotion"] = df["emotion"].astype(str).fillna("")
# #     df["raw_file_name"] = df["raw_file_name"].astype(str).fillna("")
# #     df["generated_wav_name"] = df["generated_wav_name"].astype(str).fillna("")
# #     df["split"] = df["split"].astype(str).str.lower().str.strip()
# #
# #     if args.only_split != "all":
# #         df = df[df["split"] == args.only_split].copy()
# #     if len(df) == 0:
# #         raise RuntimeError(f"No rows for split={args.only_split}")
# #
# #     # group
# #     if args.group_by == "group_id":
# #         if "group_id" not in df.columns:
# #             print("[WARN] group_by=group_id but no group_id column; fallback to sayings_emotion")
# #             args.group_by = "sayings_emotion"
# #
# #     if args.group_by == "group_id":
# #         groups = list(df.groupby(["group_id"], dropna=False))
# #     else:
# #         groups = list(df.groupby(["sayings", "emotion"], dropna=False))
# #     print("[Groups]", len(groups), "group_by=", args.group_by)
# #
# #     # VQ index
# #     motion_vq_dir = os.path.join(args.dataset_dir, "HumanML3D", "VQVAE")
# #     if not os.path.isdir(motion_vq_dir):
# #         raise RuntimeError(f"Missing motion_vq_dir: {motion_vq_dir}")
# #     vq_by_stem = build_vq_index(motion_vq_dir)
# #     print("[VQ] indexed:", len(vq_by_stem))
# #
# #     # audio code dir
# #     audio_code_dir = args.audio_code_dir or os.path.join(args.dataset_dir, "audio-raws-09-01-2026-code")
# #     if not os.path.isdir(audio_code_dir):
# #         print(f"[WARN] audio_code_dir not found: {audio_code_dir} (a-modes will use empty audio)")
# #
# #     # caption model
# #     cap_tok = T5Tokenizer.from_pretrained(args.caption_ckpt)
# #     cap_model = T5ForConditionalGeneration.from_pretrained(args.caption_ckpt).to(device).eval()
# #
# #     @torch.no_grad()
# #     def caption_motion_codes(codes: List[int]) -> str:
# #         prompt = build_caption_prompt(codes)
# #         inp = cap_tok(prompt, return_tensors="pt").input_ids.to(device, dtype=torch.long)
# #         out = cap_model.generate(inp, max_length=200, num_beams=1, do_sample=False)
# #         txt = cap_tok.decode(out[0], skip_special_tokens=True).strip().strip('"')
# #         return txt
# #
# #     # qwen judge
# #     qwen = QwenJudge(
# #         model_path=args.qwen_path,
# #         use_vllm=args.qwen_use_vllm,
# #         tp=args.qwen_tp,
# #         gpu_mem_util=args.qwen_gpu_mem_util,
# #         max_new_tokens=args.qwen_max_new_tokens,
# #         temperature=0.0,
# #         max_model_len=args.qwen_max_model_len,
# #     )
# #     print("[Qwen] model=", args.qwen_path, "backend=", ("vllm" if args.qwen_use_vllm else "hf"))
# #
# #     # cache A2RM models per mode
# #     a2rm_models: Dict[str, Any] = {}
# #     a2rm_toks: Dict[str, Any] = {}
# #
# #     def get_a2rm_model(mode: str):
# #         if mode in a2rm_models:
# #             return a2rm_toks[mode], a2rm_models[mode]
# #         ckpt = ckpt_map[mode]
# #         tok = T5Tokenizer.from_pretrained(ckpt)
# #         model = T5ForConditionalGeneration.from_pretrained(ckpt).to(device).eval()
# #         a2rm_toks[mode] = tok
# #         a2rm_models[mode] = model
# #         print(f"[Load A2RM] mode={mode} ckpt={ckpt}")
# #         return tok, model
# #
# #     # resolve audio for group (a-modes)
# #     def resolve_audio_for_group(g: pd.DataFrame) -> Tuple[str, str]:
# #         stems = [str(x).strip() for x in g["generated_wav_name"].tolist() if str(x).strip()]
# #         stems = list(dict.fromkeys(stems))
# #         if (not stems) or (not os.path.isdir(audio_code_dir)):
# #             return "", ""
# #         stem = random.choice(stems)
# #         p = pick_code_from_stem(audio_code_dir, stem)
# #         if p is None:
# #             return "", ""
# #         codes = load_audio_tokens_any(p)
# #         return format_audio_tokens(codes, level=args.audio_token_level), p
# #
# #     def make_eval_key(mode: str, group_key: str, audio_code_path: str) -> str:
# #         s = f"{mode}|||{group_key}|||{audio_code_path}"
# #         return hashlib.md5(s.encode("utf-8")).hexdigest()
# #
# #     def run_pairwise_btl(
# #         query_payload: Dict[str, Any],
# #         candidates: List[Dict[str, Any]],  # [{"cid","caption"}]
# #         max_pairs: int,
# #         seed_int: int,
# #     ) -> List[int]:
# #         N = len(candidates)
# #         if N <= 1:
# #             return list(range(N))
# #
# #         pairs = sample_pairs(N, max_pairs=max_pairs, seed=seed_int ^ 0x13579BDF)
# #         allowed = {c["cid"] for c in candidates}
# #         matches: List[Tuple[int, int, int]] = []
# #
# #         for (i, j) in pairs:
# #             ca, cb = candidates[i], candidates[j]
# #             prompt = build_qwen_pairwise_prompt(query_payload, ca, cb)
# #             out = qwen.generate(prompt)
# #             winner = parse_winner_cid(out, allowed=allowed)
# #
# #             # safer fallback: if parse failed, skip (avoid bias)
# #             if winner is None:
# #                 continue
# #
# #             y = 1 if winner == ca["cid"] else 0
# #             matches.append((i, j, y))
# #
# #         scores = fit_btl_scores(N, matches)
# #         return btl_order(scores)
# #
# #     # build labeled items for a group
# #     def build_items(gg_all: pd.DataFrame, lbl: str, max_n: int) -> List[Dict[str, Any]]:
# #         sub = gg_all[gg_all["label_canon"] == lbl].copy()
# #         if len(sub) == 0:
# #             return []
# #         mids = list(dict.fromkeys(sub["motion_id"].tolist()))[:max_n]
# #         use_csv_caption = ("motion_caption" in gg_all.columns)
# #         items = []
# #         for i, mid in enumerate(mids):
# #             cap = ""
# #             if use_csv_caption:
# #                 ss = sub[sub["motion_id"] == mid]
# #                 if "motion_caption" in ss.columns and len(ss) > 0:
# #                     cap = str(ss["motion_caption"].iloc[0]).strip()
# #             if not cap:
# #                 p = vqvae_lookup(vq_by_stem, mid)
# #                 if p is None:
# #                     continue
# #                 codes = load_motion_codes_from_vq(p)
# #                 cap = caption_motion_codes(codes)
# #             items.append({"id": f"{lbl}_{i}", "caption": cap})
# #         return items
# #
# #     # output
# #     run_tag = make_run_tag(args)
# #     out_csv = os.path.join(args.out_dir, f"eval_gen_and_ranker_{run_tag}.csv")
# #     ensure_parent(out_csv)
# #
# #     # 你要的：win_gen_vs_* + gen_at3（测生成）
# #     # 以及：ranker_ndcg@K（只用 labeled 测 ranker）
# #     fieldnames = [
# #         "eval_key",
# #         "mode",
# #         "split",
# #         "group_key",
# #         "sayings",
# #         "emotion",
# #         "audio_code_path",
# #
# #         "num_gen",
# #         "num_gold",
# #         "num_silver",
# #         "num_neg",
# #
# #         "k_eval",
# #
# #         # ---- generation ability (full pool ranker -> compare best gen) ----
# #         "win_gen_vs_neg",
# #         "win_gen_vs_silver",
# #         "win_gen_vs_gold",
# #         "gen_at3",
# #
# #         # ---- ranker ability (labeled-only, graded gain) ----
# #         "ranker_ndcg_labeled",   # gold=2 silver=1 neg=0 on labeled-only ranking
# #
# #         # diagnostics
# #         "topk_orig_json",
# #         "topk_types_json",
# #     ]
# #
# #     exists = os.path.isfile(out_csv)
# #     fcsv = open(out_csv, "a", encoding="utf-8", newline="")
# #     writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
# #     if not exists:
# #         writer.writeheader()
# #         fcsv.flush()
# #         os.fsync(fcsv.fileno())
# #
# #     # ----------------------------
# #     # main loop
# #     # ----------------------------
# #     for mode in modes:
# #         tok, model = get_a2rm_model(mode)
# #
# #         for keys, g in tqdm(groups, desc=f"Eval[{mode}]"):
# #             if args.group_by == "group_id":
# #                 group_key = str(keys) if not isinstance(keys, tuple) else str(keys[0])
# #                 sayings = str(g["sayings"].iloc[0])
# #                 emotion = str(g["emotion"].iloc[0])
# #             else:
# #                 sayings, emotion = keys
# #                 sayings = str(sayings)
# #                 emotion = str(emotion)
# #                 group_key = f"{sayings}|||{emotion}"
# #
# #             split = str(g["split"].iloc[0])
# #
# #             gg_all = g.copy()
# #             gg_all["label_canon"] = gg_all["label"].apply(canon_label)
# #             gg_all["motion_id"] = gg_all["raw_file_name"].apply(motion_id_from_raw)
# #
# #             gold_items = build_items(gg_all, "gold", args.max_gold)
# #             silver_items = build_items(gg_all, "silver", args.max_silver)
# #             neg_items = build_items(gg_all, "neg", args.max_neg)
# #
# #             # audio for a-modes
# #             audio_text = ""
# #             audio_code_path = ""
# #             if "a" in mode:
# #                 audio_text, audio_code_path = resolve_audio_for_group(g)
# #
# #             eval_key = make_eval_key(mode, group_key, audio_code_path)
# #
# #             query_payload = {
# #                 "speaker_sayings": sayings,
# #                 "speaker_emotion": emotion,
# #                 "cond_mode": mode,
# #                 "note": "In a dyadic conversation, rank LISTENER non-verbal responses to the SPEAKER's utterance.",
# #             }
# #
# #             seed_int = int(hashlib.md5(f"{group_key}|||{mode}".encode("utf-8")).hexdigest()[:8], 16)
# #
# #             # ----------------------------
# #             # (B) Ranker ability: labeled-only BTL ranking, graded nDCG@K
# #             # ----------------------------
# #             cand_labeled, cid2orig_L, orig2type_L = build_uniform_candidates(
# #                 gen_items=[],
# #                 gold_items=gold_items,
# #                 silver_items=silver_items,
# #                 neg_items=neg_items,
# #                 seed=seed_int ^ 0xBADC0DE,
# #                 max_total=args.max_total_candidates,
# #             )
# #
# #             ranker_ndcg_labeled = float("nan")
# #             if len(cand_labeled) > 0:
# #                 order_L = run_pairwise_btl(
# #                     query_payload=query_payload,
# #                     candidates=cand_labeled,
# #                     max_pairs=args.max_pairs_labeled,
# #                     seed_int=seed_int ^ 0x2468ACE,
# #                 )
# #                 ranked_orig_L = [cid2orig_L[cand_labeled[i]["cid"]] for i in order_L]
# #                 ranked_types_L = [orig2type_L.get(oid, "neg") for oid in ranked_orig_L]
# #                 kL = min(int(args.k_eval), len(ranked_types_L))
# #                 if kL <= 0:
# #                     kL = min(5, len(ranked_types_L))
# #                 rels_L = [type_to_gain_graded_labeled(t) for t in ranked_types_L]
# #                 ranker_ndcg_labeled = ndcg_at_k(rels_L, k=kL)
# #
# #             # ----------------------------
# #             # Generate gen candidates (A2RM model ability)
# #             # ----------------------------
# #             input_text = build_prompt_condmode(
# #                 speaker_transcription=sayings,
# #                 speaker_audio=audio_text,
# #                 speaker_emotion=emotion,
# #                 cond_mode=mode,
# #             )
# #             input_ids = tok(input_text, return_tensors="pt").input_ids.to(device, dtype=torch.long)
# #
# #             gen_items: List[Dict[str, Any]] = []
# #             for ci in range(int(args.num_gen)):
# #                 out = model.generate(
# #                     input_ids,
# #                     max_length=256,
# #                     do_sample=True,
# #                     temperature=args.gen_temperature,
# #                     top_k=args.gen_top_k,
# #                 )
# #                 out_text = tok.decode(out[0], skip_special_tokens=False)
# #                 out_text = out_text.replace("<pad>", "").replace("</s>", "").strip()
# #                 codes = parse_motion_tokens(out_text, max_len=args.gen_max_len, codebook_size=512)
# #                 if len(codes) == 0:
# #                     codes = [1] * min(args.gen_max_len, 196)
# #                 cap = caption_motion_codes(codes)
# #                 gen_items.append({"id": f"gen_{ci}", "caption": cap})
# #
# #             # ----------------------------
# #             # (A) Generation ability: full pool ranking -> win_gen_vs_* + gen_at3
# #             # ----------------------------
# #             cand_all, cid2orig, orig2type = build_uniform_candidates(
# #                 gen_items=gen_items,
# #                 gold_items=gold_items,
# #                 silver_items=silver_items,
# #                 neg_items=neg_items,
# #                 seed=seed_int,
# #                 max_total=args.max_total_candidates,
# #             )
# #
# #             k_eval = min(int(args.k_eval), len(cand_all))
# #             if k_eval <= 0:
# #                 k_eval = min(5, len(cand_all))
# #
# #             # If no candidates, write NaNs
# #             if len(cand_all) == 0:
# #                 row = dict(
# #                     eval_key=eval_key,
# #                     mode=mode,
# #                     split=split,
# #                     group_key=group_key,
# #                     sayings=sayings,
# #                     emotion=emotion,
# #                     audio_code_path=audio_code_path,
# #                     num_gen=len(gen_items),
# #                     num_gold=len(gold_items),
# #                     num_silver=len(silver_items),
# #                     num_neg=len(neg_items),
# #                     k_eval=0,
# #                     win_gen_vs_neg=float("nan"),
# #                     win_gen_vs_silver=float("nan"),
# #                     win_gen_vs_gold=float("nan"),
# #                     gen_at3=float("nan"),
# #                     ranker_ndcg_labeled=ranker_ndcg_labeled,
# #                     topk_orig_json="[]",
# #                     topk_types_json="[]",
# #                 )
# #                 writer.writerow(row)
# #                 fcsv.flush()
# #                 os.fsync(fcsv.fileno())
# #                 continue
# #
# #             order_all = run_pairwise_btl(
# #                 query_payload=query_payload,
# #                 candidates=cand_all,
# #                 max_pairs=args.max_pairs,
# #                 seed_int=seed_int,
# #             )
# #
# #             # IMPORTANT: idx2type aligned with candidate indices
# #             idx2type_all = [orig2type.get(cid2orig[c["cid"]], "neg") for c in cand_all]
# #
# #             win_gen_vs_neg = winrate_best_gen_vs_type_from_order(order_all, idx2type_all, "neg")
# #             win_gen_vs_silver = winrate_best_gen_vs_type_from_order(order_all, idx2type_all, "silver")
# #             win_gen_vs_gold = winrate_best_gen_vs_type_from_order(order_all, idx2type_all, "gold")
# #             gen_at3 = gen_at_k_from_order(order_all, idx2type_all, k=3)
# #
# #             # top-k outputs (for debugging / inspection)
# #             ranked_orig = [cid2orig[cand_all[i]["cid"]] for i in order_all]
# #             ranked_types = [orig2type.get(oid, "neg") for oid in ranked_orig]
# #             topk_orig = ranked_orig[:k_eval]
# #             topk_types = ranked_types[:k_eval]
# #
# #             row = dict(
# #                 eval_key=eval_key,
# #                 mode=mode,
# #                 split=split,
# #                 group_key=group_key,
# #                 sayings=sayings,
# #                 emotion=emotion,
# #                 audio_code_path=audio_code_path,
# #                 num_gen=len(gen_items),
# #                 num_gold=len(gold_items),
# #                 num_silver=len(silver_items),
# #                 num_neg=len(neg_items),
# #                 k_eval=k_eval,
# #                 win_gen_vs_neg=win_gen_vs_neg,
# #                 win_gen_vs_silver=win_gen_vs_silver,
# #                 win_gen_vs_gold=win_gen_vs_gold,
# #                 gen_at3=gen_at3,
# #                 ranker_ndcg_labeled=ranker_ndcg_labeled,
# #                 topk_orig_json=json.dumps(topk_orig, ensure_ascii=False),
# #                 topk_types_json=json.dumps(topk_types, ensure_ascii=False),
# #             )
# #             writer.writerow(row)
# #             fcsv.flush()
# #             os.fsync(fcsv.fileno())
# #
# #     fcsv.close()
# #     print("[Saved]", out_csv)
# #
# #
# # if __name__ == "__main__":
# #     main()
#
#
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
#
# """
# Eval script (A2RM):
#
# Goal A (generation ability, NO BTL):
# - full pool = gen + gold + silver + neg
# - pairwise judge ONLY on (gen vs others) (optionally quota by type)
# - aggregate by Copeland score (wins-losses) -> global order
# - report:
#   win_gen_vs_neg / win_gen_vs_silver / win_gen_vs_gold (best-gen empirical winrate)
#   gen_at3 (top-3 contains any gen)
#
# Goal B (ranker ability, WITH BTL):
# - labeled-only pool = gold + silver + neg (NO gen)
# - random pair sampling on labeled-only pool
# - fit Bradley–Terry (BTL) -> order
# - report:
#   ranker_ndcg_labeled (graded gain: gold=2, silver=1, neg=0)
#
# Speed:
# - vLLM batched generation for Qwen judge (recommended)
# - HF fallback still works (but slower)
#
# Usage example:
# python eval_gen_copeland_ranker_btl.py \
#   --pairs_csv ./new_data/test.csv \
#   --dataset_dir /ibex/project/c2191/luoc/dataset/A2R \
#   --ckpt_map_json ./ckpt_map.json \
#   --caption_ckpt /path/to/t5_caption_ckpt \
#   --qwen_path /path/to/Qwen \
#   --qwen_use_vllm \
#   --qwen_tp 1 \
#   --qwen_batch_size 128 \
#   --max_pairs 120 \
#   --max_pairs_labeled 120
# """
#
# import os, re, json, csv, math, argparse, random, hashlib
# from typing import Dict, List, Tuple, Optional, Any
#
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
#
# import torch
# from transformers import (
#     T5Tokenizer,
#     T5ForConditionalGeneration,
#     AutoTokenizer,
#     AutoModelForCausalLM,
# )
#
# # -----------------------------
# # Motion token parsing (A2RM output)
# # -----------------------------
# _MOTION_SPAN_RE = re.compile(r"<Motion Tokens>(.*?)</Motion Tokens>", re.DOTALL)
# _MOTION_TOKEN_RE = re.compile(r"<Motion Token\s+(\d+)>")
# _MOTION_TOKEN_SHORT_RE = re.compile(r"<(\d+)>")  # <123>
#
# def parse_motion_tokens(text: str, max_len: int = 200, codebook_size: int = 512) -> List[int]:
#     if text is None:
#         return []
#     s = str(text)
#     m = _MOTION_SPAN_RE.search(s)
#     span = m.group(1) if m else s
#
#     codes = [int(x) for x in _MOTION_TOKEN_RE.findall(span)]
#     if len(codes) == 0:
#         codes = [int(x) for x in _MOTION_TOKEN_SHORT_RE.findall(span)]
#
#     out = []
#     for c in codes:
#         if 0 <= c < codebook_size:
#             out.append(c)
#         else:
#             break
#     return out[:max_len]
#
# # -----------------------------
# # Prompt builder (MATCH TRAINING)
# # -----------------------------
# def build_prompt_condmode(
#     speaker_transcription: str,
#     speaker_audio: str,
#     speaker_emotion: str,
#     cond_mode: str,
# ) -> str:
#     """
#     a       : audio only
#     a+e     : audio + emotion
#     t       : transcription only
#     t+e     : transcription + emotion
#     t+a     : transcription + audio
#     t+a+e   : transcription + audio + emotion
#     """
#     cm = (cond_mode or "").strip().lower()
#     use_transcription = ("t" in cm)
#     use_audio = ("a" in cm)
#     use_emotion = (cm.endswith("+e") or cm in ("a+e", "t+e", "t+a+e"))
#
#     t = (speaker_transcription or "").strip()
#     a = (speaker_audio or "").strip()
#     e = (speaker_emotion or "").strip()
#
#     lines = []
#     lines.append("You are modeling a speaker-listener dyadic interaction.\n\n")
#     lines.append("Input:\n")
#     lines.append(f"- SPEAKER_TRANSCRIPTION: {t if use_transcription else ''}\n")
#     lines.append(f"- SPEAKER_AUDIO: {a if use_audio else ''}\n")
#     if use_emotion and e:
#         lines.append(f"- SPEAKER_EMOTION: <Emotion> {e} </Emotion>\n")
#     lines.append("\nOutput:\n")
#     lines.append("Return ONLY a sequence of listener motion tokens in the exact format:\n")
#     lines.append("<Motion Tokens> <Motion Token i> ... </Motion Tokens>\n")
#     lines.append("Do NOT output any other words.\n")
#     return "".join(lines).strip()
#
# # -----------------------------
# # Caption model helpers (T5)
# # -----------------------------
# def build_caption_prompt(motion_codes: List[int]) -> str:
#     motion_string = "<Motion Tokens>" + "".join([f"<{c}>" for c in motion_codes]) + "</Motion Tokens>"
#     return "Generate text: " + motion_string
#
# # -----------------------------
# # Audio token formatting
# # -----------------------------
# def load_audio_tokens_any(path: str) -> np.ndarray:
#     obj = np.load(path, allow_pickle=False)
#     if isinstance(obj, np.lib.npyio.NpzFile):
#         if "codes" in obj.files:
#             arr = obj["codes"]
#         else:
#             arr = obj[obj.files[0]]
#         obj.close()
#         return arr
#     return obj
#
# def format_audio_tokens(a_tokens: np.ndarray, level: str = "base") -> str:
#     arr = np.array(a_tokens)
#     level = str(level)
#
#     if arr.ndim == 1:
#         parts = ["<Audio Tokens>"]
#         for t in arr.reshape(-1):
#             parts.append(f"<Audio Token {int(t)}>")
#         parts.append("</Audio Tokens>")
#         return " ".join(parts)
#
#     L = int(arr.shape[0])
#     parts = ["<Audio Tokens>"]
#
#     if level == "base":
#         for t in arr[0].reshape(-1):
#             parts.append(f"<Audio Level 0 Token {int(t)}>")
#     elif level == "all":
#         for i in range(L):
#             for t in arr[i].reshape(-1):
#                 parts.append(f"<Audio Level {i} Token {int(t)}>")
#     elif level == "rand":
#         k = int(np.random.choice(np.arange(1, L + 1)))
#         for i in range(k):
#             for t in arr[i].reshape(-1):
#                 parts.append(f"<Audio Level {i} Token {int(t)}>")
#     else:
#         raise ValueError(f"Unknown audio_token_level={level}")
#
#     parts.append("</Audio Tokens>")
#     return " ".join(parts)
#
# def pick_code_from_stem(code_dir: str, stem: str) -> Optional[str]:
#     stem = str(stem).strip()
#     if not stem:
#         return None
#     p_npz = os.path.join(code_dir, stem + ".npz")
#     if os.path.exists(p_npz):
#         return p_npz
#     p_npy = os.path.join(code_dir, stem + ".npy")
#     if os.path.exists(p_npy):
#         return p_npy
#     return None
#
# # -----------------------------
# # VQ lookup for labeled motions
# # -----------------------------
# def motion_id_from_raw(raw_file_name: str) -> str:
#     s = str(raw_file_name)
#     mid = s.split("_", 1)[0]
#     return str(mid).zfill(6)
#
# def build_vq_index(vq_dir: str) -> Dict[str, str]:
#     m = {}
#     for fn in os.listdir(vq_dir):
#         if fn.endswith(".npy"):
#             stem = os.path.splitext(fn)[0]
#             m[stem] = os.path.join(vq_dir, fn)
#     return m
#
# def vqvae_lookup(vq_by_stem: Dict[str, str], motion_id: str) -> Optional[str]:
#     base = str(motion_id)
#     if base in vq_by_stem:
#         return vq_by_stem[base]
#     if base.isdigit() and ("M" + base) in vq_by_stem:
#         return vq_by_stem["M" + base]
#     if base.startswith("M") and base[1:].isdigit() and (base[1:] in vq_by_stem):
#         return vq_by_stem[base[1:]]
#     return None
#
# def load_motion_codes_from_vq(vq_path: str, codebook_size: int = 512) -> List[int]:
#     arr = np.load(vq_path, allow_pickle=False)
#     arr = np.asarray(arr).reshape(-1).tolist()
#     out = []
#     for x in arr:
#         try:
#             c = int(x)
#         except Exception:
#             continue
#         if 0 <= c < codebook_size:
#             out.append(c)
#     return out
#
# # -----------------------------
# # Candidate packing (rename to C01..)
# # -----------------------------
# def build_uniform_candidates(
#     gen_items: List[Dict[str, Any]],
#     gold_items: List[Dict[str, Any]],
#     silver_items: List[Dict[str, Any]],
#     neg_items: List[Dict[str, Any]],
#     seed: int,
#     max_total: int,
# ) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, str]]:
#     """
#     Returns:
#       candidates: [{"cid":"C01","caption":"..."}, ...] (shuffled)
#       cid2orig: {"C01":"gen_0", ...}
#       orig2type: {"gen_0":"gen","gold_0":"gold","silver_0":"silver","neg_0":"neg"...}
#     """
#     all_items = []
#     orig2type: Dict[str, str] = {}
#
#     def add(items, t):
#         for it in items:
#             oid = str(it["id"])
#             all_items.append({"orig_id": oid, "caption": str(it["caption"])})
#             orig2type[oid] = t
#
#     add(gen_items, "gen")
#     add(gold_items, "gold")
#     add(silver_items, "silver")
#     add(neg_items, "neg")
#
#     rng = random.Random(int(seed))
#     rng.shuffle(all_items)
#     all_items = all_items[: max_total]
#
#     cid2orig: Dict[str, str] = {}
#     candidates: List[Dict[str, Any]] = []
#
#     for i, it in enumerate(all_items, start=1):
#         cid = f"C{i:02d}"
#         cid2orig[cid] = it["orig_id"]
#         candidates.append({"cid": cid, "caption": it["caption"]})
#     return candidates, cid2orig, orig2type
#
# # -----------------------------
# # Pairwise Qwen judge parsing
# # -----------------------------
# def extract_last_json_object(text: str) -> Optional[Dict[str, Any]]:
#     if text is None:
#         return None
#     s = str(text)
#     starts = [i for i, ch in enumerate(s) if ch == "{"]  # keep all
#     if not starts:
#         return None
#
#     last_obj = None
#     for st in starts:
#         depth = 0
#         in_str = False
#         esc = False
#         for i in range(st, len(s)):
#             c = s[i]
#             if in_str:
#                 if esc:
#                     esc = False
#                 elif c == "\\":
#                     esc = True
#                 elif c == '"':
#                     in_str = False
#             else:
#                 if c == '"':
#                     in_str = True
#                 elif c == "{":
#                     depth += 1
#                 elif c == "}":
#                     depth -= 1
#                     if depth == 0:
#                         blob = s[st : i + 1]
#                         try:
#                             obj = json.loads(blob)
#                             last_obj = obj
#                         except Exception:
#                             pass
#                         break
#     return last_obj
#
# _CID_RE = re.compile(r"\bC\d{2}\b")
#
# def parse_winner_cid(raw: str, allowed: set) -> Optional[str]:
#     """
#     Accept:
#       {"winner":"C01"} OR {"better":"C01"} OR {"choice":"C01"}
#     Fallback: find any Cxx in text.
#     """
#     obj = extract_last_json_object(raw) or {}
#     for k in ["winner", "better", "choice", "answer", "selected"]:
#         v = obj.get(k, None)
#         if isinstance(v, str) and v in allowed:
#             return v
#
#     m = _CID_RE.findall(str(raw))
#     for cid in m:
#         if cid in allowed:
#             return cid
#     return None
#
# def build_qwen_pairwise_prompt(
#     query_payload: Dict[str, Any],
#     cand_a: Dict[str, Any],
#     cand_b: Dict[str, Any],
# ) -> str:
#     payload = {
#         "task": "Dyadic SPEAKER→LISTENER pairwise preference.",
#         "query": query_payload,
#         "candidates": [
#             {"cid": cand_a["cid"], "caption": cand_a["caption"]},
#             {"cid": cand_b["cid"], "caption": cand_b["caption"]},
#         ],
#         "rules": [
#             "You are comparing LISTENER reactions only (NOT speaker actions).",
#             "Pick the ONE candidate that is the more plausible listener non-verbal response to the speaker sayings/emotion.",
#             "Use ONLY the two captions as evidence; ids contain no label information.",
#             "Return ONLY JSON: {\"winner\": \"Cxx\"}. No other keys/text.",
#         ],
#     }
#     return (
#         "You are an expert evaluator for listener reactive motions.\n"
#         "IMPORTANT: Output MUST be a single valid JSON object and NOTHING ELSE.\n"
#         f"{json.dumps(payload, ensure_ascii=False)}"
#     )
#
# # -----------------------------
# # Qwen Judge (vLLM batch + HF fallback)
# # -----------------------------
# class QwenJudge:
#     def __init__(
#         self,
#         model_path: str,
#         use_vllm: bool = True,
#         tp: int = 1,
#         gpu_mem_util: float = 0.90,
#         max_new_tokens: int = 256,
#         temperature: float = 0.0,
#         max_model_len: int = 8192,
#     ):
#         self.model_path = model_path
#         self.use_vllm = use_vllm
#         self.tp = tp
#         self.gpu_mem_util = gpu_mem_util
#         self.max_new_tokens = max_new_tokens
#         self.temperature = temperature
#         self.max_model_len = max_model_len
#         self._mode = None
#
#         if use_vllm:
#             try:
#                 from vllm import LLM, SamplingParams  # type: ignore
#                 self._vllm_LLM = LLM(
#                     model=model_path,
#                     tensor_parallel_size=tp,
#                     gpu_memory_utilization=gpu_mem_util,
#                     trust_remote_code=True,
#                     max_model_len=max_model_len,
#                     enforce_eager=False,
#                 )
#                 self._vllm_SamplingParams = SamplingParams
#                 self._mode = "vllm"
#             except Exception as e:
#                 print(f"[WARN] vLLM init failed, fallback to HF. err={e}")
#                 self._mode = "hf"
#         else:
#             self._mode = "hf"
#
#         if self._mode == "hf":
#             self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 model_path,
#                 trust_remote_code=True,
#                 torch_dtype=torch.bfloat16,
#                 device_map="auto",
#             ).eval()
#
#     @torch.no_grad()
#     def generate(self, prompt: str) -> str:
#         if self._mode == "vllm":
#             sp = self._vllm_SamplingParams(
#                 temperature=self.temperature,
#                 max_tokens=self.max_new_tokens,
#                 top_p=1.0,
#             )
#             outs = self._vllm_LLM.generate([prompt], sp)
#             return outs[0].outputs[0].text.strip()
#
#         inputs = self.tok(prompt, return_tensors="pt")
#         inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
#         out = self.model.generate(
#             **inputs,
#             max_new_tokens=self.max_new_tokens,
#             do_sample=(self.temperature > 0),
#             temperature=max(1e-6, self.temperature),
#             top_p=1.0,
#         )
#         txt = self.tok.decode(out[0], skip_special_tokens=True)
#         if txt.startswith(prompt):
#             txt = txt[len(prompt):]
#         return txt.strip()
#
#     @torch.no_grad()
#     def generate_batch(self, prompts: List[str]) -> List[str]:
#         if len(prompts) == 0:
#             return []
#         if self._mode == "vllm":
#             sp = self._vllm_SamplingParams(
#                 temperature=self.temperature,
#                 max_tokens=self.max_new_tokens,
#                 top_p=1.0,
#             )
#             outs = self._vllm_LLM.generate(prompts, sp)
#             return [o.outputs[0].text.strip() for o in outs]
#
#         # HF fallback: micro-batch loop
#         return [self.generate(p) for p in prompts]
#
# # -----------------------------
# # Bradley–Terry fitting + ranking (for labeled-only ranker ability)
# # -----------------------------
# def fit_btl_scores(
#     n_items: int,
#     matches: List[Tuple[int, int, int]],  # (i, j, y) y=1 => i wins else j wins
#     l2: float = 1e-2,
#     lr: float = 0.2,
#     steps: int = 600,
# ) -> np.ndarray:
#     s = np.zeros(n_items, dtype=np.float64)
#     if len(matches) == 0:
#         return s
#
#     for _ in range(steps):
#         grad = np.zeros_like(s)
#         for i, j, y in matches:
#             d = s[i] - s[j]
#             p = 1.0 / (1.0 + math.exp(-d))
#             g = (p - y)  # dL/dd
#             grad[i] += g
#             grad[j] -= g
#         grad += l2 * s
#         s -= lr * grad / max(1, len(matches))
#         s -= s.mean()
#     return s
#
# def btl_order(scores: np.ndarray) -> List[int]:
#     return np.argsort(-scores).tolist()
#
# # -----------------------------
# # Metrics
# # -----------------------------
# def type_to_gain_graded_labeled(t: str) -> int:
#     # For ranker ability only: gold=2, silver=1, neg=0
#     if t == "gold":
#         return 2
#     if t == "silver":
#         return 1
#     return 0
#
# def ndcg_at_k(relevances: List[int], k: int) -> float:
#     rel = relevances[:k]
#     dcg = 0.0
#     for i, r in enumerate(rel, start=1):
#         if r > 0:
#             dcg += (2.0 ** r - 1.0) / math.log2(i + 1.0)
#
#     ideal = sorted(relevances, reverse=True)[:k]
#     idcg = 0.0
#     for i, r in enumerate(ideal, start=1):
#         if r > 0:
#             idcg += (2.0 ** r - 1.0) / math.log2(i + 1.0)
#
#     if idcg <= 1e-12:
#         return 0.0
#     return float(dcg / idcg)
#
# def gen_at_k_from_order(order: List[int], idx2type: List[str], k: int = 3) -> float:
#     topk = order[:k]
#     return 1.0 if any(idx2type[i] == "gen" for i in topk) else 0.0
#
# # -----------------------------
# # Pair sampling
# # -----------------------------
# def sample_pairs_all(n: int, max_pairs: int, seed: int) -> List[Tuple[int, int]]:
#     """Sample unordered pairs (i<j) from all pairs."""
#     rng = random.Random(seed)
#     all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
#     if len(all_pairs) <= max_pairs:
#         rng.shuffle(all_pairs)
#         return all_pairs
#     return rng.sample(all_pairs, k=max_pairs)
#
# def sample_pairs_gen_only(
#     idx2type: List[str],
#     max_pairs: int,
#     seed: int,
#     quota_neg: float = 0.5,
#     quota_silver: float = 0.3,
#     quota_gold: float = 0.2,
# ) -> List[Tuple[int, int]]:
#     """
#     Only sample pairs where one side is gen and the other is non-gen.
#     Allocate budgets by type quota (neg/silver/gold). Will top-up if some type missing.
#     """
#     rng = random.Random(seed)
#     gen_idxs = [i for i, t in enumerate(idx2type) if t == "gen"]
#     if len(gen_idxs) == 0:
#         return []
#
#     other_by_type: Dict[str, List[int]] = {"neg": [], "silver": [], "gold": []}
#     for i, t in enumerate(idx2type):
#         if t in other_by_type:
#             other_by_type[t].append(i)
#
#     quotas = {
#         "neg": int(round(max_pairs * quota_neg)),
#         "silver": int(round(max_pairs * quota_silver)),
#         "gold": int(round(max_pairs * quota_gold)),
#     }
#     # fix rounding drift
#     diff = max_pairs - sum(quotas.values())
#     if diff != 0:
#         quotas["neg"] += diff
#
#     pairs: List[Tuple[int, int]] = []
#     used = set()
#
#     def add_from_pool(pool: List[Tuple[int, int]], k: int):
#         nonlocal pairs, used
#         if k <= 0 or len(pool) == 0:
#             return
#         pool = list(dict.fromkeys(pool))
#         if len(pool) <= k:
#             rng.shuffle(pool)
#             for p in pool:
#                 if p not in used:
#                     pairs.append(p); used.add(p)
#         else:
#             for p in rng.sample(pool, k=k):
#                 if p not in used:
#                     pairs.append(p); used.add(p)
#
#     # per-type sampling
#     for t in ["neg", "silver", "gold"]:
#         pool = []
#         for g in gen_idxs:
#             for o in other_by_type[t]:
#                 if o == g:
#                     continue
#                 i, j = (g, o) if g < o else (o, g)
#                 pool.append((i, j))
#         add_from_pool(pool, quotas[t])
#
#     # top-up if underfilled
#     if len(pairs) < max_pairs:
#         all_pool = []
#         for g in gen_idxs:
#             for t, lst in other_by_type.items():
#                 for o in lst:
#                     if o == g:
#                         continue
#                     i, j = (g, o) if g < o else (o, g)
#                     all_pool.append((i, j))
#         all_pool = list(dict.fromkeys(all_pool))
#         rng.shuffle(all_pool)
#         for p in all_pool:
#             if len(pairs) >= max_pairs:
#                 break
#             if p not in used:
#                 pairs.append(p); used.add(p)
#
#     rng.shuffle(pairs)
#     return pairs[:max_pairs]
#
# # -----------------------------
# # Batched pairwise judging
# # -----------------------------
# def judge_pairs_batched(
#     qwen: QwenJudge,
#     query_payload: Dict[str, Any],
#     candidates: List[Dict[str, Any]],
#     pairs: List[Tuple[int, int]],
#     batch_size: int = 128,
# ) -> List[Optional[int]]:
#     """
#     For each pair (i,j), return winner index: i or j. If parse failed => None.
#     """
#     if len(pairs) == 0:
#         return []
#     allowed = {c["cid"] for c in candidates}
#     winners: List[Optional[int]] = [None] * len(pairs)
#
#     for st in range(0, len(pairs), batch_size):
#         ed = min(st + batch_size, len(pairs))
#         chunk = pairs[st:ed]
#         prompts = [build_qwen_pairwise_prompt(query_payload, candidates[i], candidates[j]) for (i, j) in chunk]
#         outs = qwen.generate_batch(prompts)
#
#         for k, raw in enumerate(outs):
#             i, j = chunk[k]
#             wcid = parse_winner_cid(raw, allowed=allowed)
#             if wcid is None:
#                 winners[st + k] = None
#             else:
#                 winners[st + k] = i if wcid == candidates[i]["cid"] else j
#
#     return winners
#
# # -----------------------------
# # Full pool aggregation: Copeland (NO BTL)
# # -----------------------------
# def copeland_from_batched_outcomes(
#     N: int,
#     pairs: List[Tuple[int, int]],
#     winners: List[Optional[int]],
# ) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
#     """
#     Returns:
#       scores: int wins-losses
#       outcomes: (i,j,y) y=1 => i wins else j wins (aligned with pair direction)
#     """
#     scores = np.zeros(N, dtype=np.int32)
#     outcomes: List[Tuple[int, int, int]] = []
#     for (i, j), w in zip(pairs, winners):
#         if w is None:
#             continue
#         if w == i:
#             scores[i] += 1; scores[j] -= 1
#             outcomes.append((i, j, 1))
#         else:
#             scores[i] -= 1; scores[j] += 1
#             outcomes.append((i, j, 0))
#     return scores, outcomes
#
# def copeland_order(scores: np.ndarray, seed_int: int) -> List[int]:
#     # stable tie-break
#     rng = np.random.RandomState(seed_int & 0xFFFFFFFF)
#     tie_noise = rng.uniform(low=-1e-3, high=1e-3, size=len(scores))
#     return np.argsort(-(scores.astype(np.float32) + tie_noise)).tolist()
#
# def best_gen_winrates_from_outcomes(
#     idx2type: List[str],
#     scores: np.ndarray,
#     outcomes: List[Tuple[int, int, int]],
# ) -> Tuple[Optional[int], float, float, float]:
#     """
#     Pick best gen by highest Copeland score, then compute empirical winrate vs neg/silver/gold
#     using only outcomes involving that best gen.
#     """
#     gen_idxs = [i for i, t in enumerate(idx2type) if t == "gen"]
#     if len(gen_idxs) == 0:
#         return None, float("nan"), float("nan"), float("nan")
#
#     best_gen = sorted(gen_idxs, key=lambda i: int(scores[i]), reverse=True)[0]
#
#     win = {"neg": 0, "silver": 0, "gold": 0}
#     tot = {"neg": 0, "silver": 0, "gold": 0}
#
#     for i, j, y in outcomes:
#         if best_gen not in (i, j):
#             continue
#         other = j if best_gen == i else i
#         other_type = idx2type[other]
#         if other_type not in tot:
#             continue
#
#         tot[other_type] += 1
#         best_wins = (y == 1 and best_gen == i) or (y == 0 and best_gen == j)
#         if best_wins:
#             win[other_type] += 1
#
#     def rate(t):
#         return float(win[t] / tot[t]) if tot[t] > 0 else float("nan")
#
#     return best_gen, rate("neg"), rate("silver"), rate("gold")
#
# # -----------------------------
# # Labeled-only: BTL (WITH BTL)
# # -----------------------------
# def btl_order_from_pairs_batched(
#     qwen: QwenJudge,
#     query_payload: Dict[str, Any],
#     candidates: List[Dict[str, Any]],
#     pairs: List[Tuple[int, int]],
#     qwen_batch_size: int,
#     btl_l2: float = 1e-2,
#     btl_lr: float = 0.2,
#     btl_steps: int = 600,
# ) -> List[int]:
#     N = len(candidates)
#     if N <= 1:
#         return list(range(N))
#     winners = judge_pairs_batched(
#         qwen=qwen,
#         query_payload=query_payload,
#         candidates=candidates,
#         pairs=pairs,
#         batch_size=qwen_batch_size,
#     )
#     matches: List[Tuple[int, int, int]] = []
#     for (i, j), w in zip(pairs, winners):
#         if w is None:
#             continue
#         y = 1 if w == i else 0
#         matches.append((i, j, y))
#
#     scores = fit_btl_scores(N, matches, l2=btl_l2, lr=btl_lr, steps=btl_steps)
#     return btl_order(scores)
#
# # -----------------------------
# # misc
# # -----------------------------
# def ensure_parent(path: str):
#     os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
#
# def hash_file(path: str) -> str:
#     h = hashlib.md5()
#     with open(path, "rb") as f:
#         while True:
#             b = f.read(1024 * 1024)
#             if not b:
#                 break
#             h.update(b)
#     return h.hexdigest()[:10]
#
# def stable_hash_str(s: str) -> str:
#     return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]
#
# def make_run_tag(args: argparse.Namespace) -> str:
#     parts = [
#         f"numgen{args.num_gen}",
#         f"keval{args.k_eval}",
#         f"maxtotal{args.max_total_candidates}",
#         f"pairsG{args.max_pairs}",
#         f"pairsL{args.max_pairs_labeled}",
#         f"qbs{args.qwen_batch_size}",
#         f"cap{stable_hash_str(args.caption_ckpt)}",
#         f"qwen{stable_hash_str(args.qwen_path)}",
#         f"ckmap{hash_file(args.ckpt_map_json)}",
#     ]
#     return "_".join(parts)
#
# def canon_label(x: str) -> str:
#     s = (x or "").strip().lower()
#     if s in {"gold", "pos", "positive", "gt", "true", "1"}:
#         return "gold"
#     if s in {"silver"}:
#         return "silver"
#     if s in {"neg", "negative", "0"}:
#         return "neg"
#     return s
#
# # -----------------------------
# # Main
# # -----------------------------
# def main():
#     ap = argparse.ArgumentParser()
#
#     ap.add_argument("--pairs_csv", type=str, required=True)
#     ap.add_argument("--dataset_dir", type=str, required=True)
#     ap.add_argument("--ckpt_map_json", type=str, required=True)
#
#     ap.add_argument("--audio_code_dir", type=str, default=None)
#     ap.add_argument("--audio_token_level", type=str, default="base", choices=["base", "all", "rand"])
#
#     ap.add_argument("--caption_ckpt", type=str, required=True)
#
#     # generation (model ability)
#     ap.add_argument("--num_gen", type=int, default=3)
#     ap.add_argument("--gen_max_len", type=int, default=200)
#     ap.add_argument("--gen_temperature", type=float, default=0.8)
#     ap.add_argument("--gen_top_k", type=int, default=200)
#
#     # candidate budgets
#     ap.add_argument("--max_gold", type=int, default=1)
#     ap.add_argument("--max_silver", type=int, default=8)
#     ap.add_argument("--max_neg", type=int, default=12)
#     ap.add_argument("--max_total_candidates", type=int, default=30)
#
#     ap.add_argument("--only_split", type=str, default="test")  # test/val/train/all
#     ap.add_argument("--group_by", type=str, default="group_id", choices=["group_id", "sayings_emotion"])
#
#     ap.add_argument("--k_eval", type=int, default=10)
#
#     # pairwise budgets
#     ap.add_argument("--max_pairs", type=int, default=120)           # full pool (gen vs others)
#     ap.add_argument("--max_pairs_labeled", type=int, default=120)   # labeled-only (random all-pairs sampled)
#
#     # gen-vs-others quota (full pool)
#     ap.add_argument("--quota_neg", type=float, default=0.5)
#     ap.add_argument("--quota_silver", type=float, default=0.3)
#     ap.add_argument("--quota_gold", type=float, default=0.2)
#
#     # qwen
#     ap.add_argument("--qwen_path", type=str, required=True)
#     ap.add_argument("--qwen_use_vllm", action="store_true")
#     ap.add_argument("--qwen_tp", type=int, default=1)
#     ap.add_argument("--qwen_gpu_mem_util", type=float, default=0.90)
#     ap.add_argument("--qwen_max_new_tokens", type=int, default=256)
#     ap.add_argument("--qwen_max_model_len", type=int, default=8192)
#     ap.add_argument("--qwen_batch_size", type=int, default=128)
#
#     # BTL hyperparams (labeled-only)
#     ap.add_argument("--btl_steps", type=int, default=600)
#     ap.add_argument("--btl_lr", type=float, default=0.2)
#     ap.add_argument("--btl_l2", type=float, default=1e-2)
#
#     ap.add_argument("--out_dir", type=str, default="./eval_gen_copeland_ranker_btl_out")
#     ap.add_argument("--seed", type=int, default=42)
#
#     args = ap.parse_args()
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#
#     os.makedirs(args.out_dir, exist_ok=True)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print("[Device]", device)
#
#     # ckpt map
#     with open(args.ckpt_map_json, "r", encoding="utf-8") as f:
#         ckpt_map = json.load(f)
#
#     modes = ["a", "a+e", "t", "t+e", "t+a", "t+a+e"]
#     for m in modes:
#         if m not in ckpt_map:
#             raise RuntimeError(f"ckpt_map_json missing mode: {m}")
#
#     # read pairs
#     df = pd.read_csv(args.pairs_csv, encoding="utf-8")
#     required_cols = ["sayings", "emotion", "label", "raw_file_name", "split", "generated_wav_name"]
#     missing = [c for c in required_cols if c not in df.columns]
#     if missing:
#         raise RuntimeError(f"Missing columns in pairs_csv: {missing}")
#
#     df["label"] = df["label"].astype(str).str.lower().str.strip()
#     df["sayings"] = df["sayings"].astype(str).fillna("")
#     df["emotion"] = df["emotion"].astype(str).fillna("")
#     df["raw_file_name"] = df["raw_file_name"].astype(str).fillna("")
#     df["generated_wav_name"] = df["generated_wav_name"].astype(str).fillna("")
#     df["split"] = df["split"].astype(str).str.lower().str.strip()
#
#     if args.only_split != "all":
#         df = df[df["split"] == args.only_split].copy()
#     if len(df) == 0:
#         raise RuntimeError(f"No rows for split={args.only_split}")
#
#     # group
#     if args.group_by == "group_id":
#         if "group_id" not in df.columns:
#             print("[WARN] group_by=group_id but no group_id column; fallback to sayings_emotion")
#             args.group_by = "sayings_emotion"
#
#     if args.group_by == "group_id":
#         groups = list(df.groupby(["group_id"], dropna=False))
#     else:
#         groups = list(df.groupby(["sayings", "emotion"], dropna=False))
#     print("[Groups]", len(groups), "group_by=", args.group_by)
#
#     # VQ index
#     motion_vq_dir = os.path.join(args.dataset_dir, "HumanML3D", "VQVAE")
#     if not os.path.isdir(motion_vq_dir):
#         raise RuntimeError(f"Missing motion_vq_dir: {motion_vq_dir}")
#     vq_by_stem = build_vq_index(motion_vq_dir)
#     print("[VQ] indexed:", len(vq_by_stem))
#
#     # audio code dir
#     audio_code_dir = args.audio_code_dir or os.path.join(args.dataset_dir, "audio-raws-09-01-2026-code")
#     if not os.path.isdir(audio_code_dir):
#         print(f"[WARN] audio_code_dir not found: {audio_code_dir} (a-modes will use empty audio)")
#
#     # caption model
#     cap_tok = T5Tokenizer.from_pretrained(args.caption_ckpt)
#     cap_model = T5ForConditionalGeneration.from_pretrained(args.caption_ckpt).to(device).eval()
#
#     @torch.no_grad()
#     def caption_motion_codes(codes: List[int]) -> str:
#         prompt = build_caption_prompt(codes)
#         inp = cap_tok(prompt, return_tensors="pt").input_ids.to(device, dtype=torch.long)
#         out = cap_model.generate(inp, max_length=200, num_beams=1, do_sample=False)
#         txt = cap_tok.decode(out[0], skip_special_tokens=True).strip().strip('"')
#         return txt
#
#     # qwen
#     qwen = QwenJudge(
#         model_path=args.qwen_path,
#         use_vllm=args.qwen_use_vllm,
#         tp=args.qwen_tp,
#         gpu_mem_util=args.qwen_gpu_mem_util,
#         max_new_tokens=args.qwen_max_new_tokens,
#         temperature=0.0,
#         max_model_len=args.qwen_max_model_len,
#     )
#     print("[Qwen] model=", args.qwen_path, "backend=", ("vllm" if args.qwen_use_vllm else "hf"), "batch=", args.qwen_batch_size)
#
#     # cache A2RM models per mode
#     a2rm_models: Dict[str, Any] = {}
#     a2rm_toks: Dict[str, Any] = {}
#
#     def get_a2rm_model(mode: str):
#         if mode in a2rm_models:
#             return a2rm_toks[mode], a2rm_models[mode]
#         ckpt = ckpt_map[mode]
#         tok = T5Tokenizer.from_pretrained(ckpt)
#         model = T5ForConditionalGeneration.from_pretrained(ckpt).to(device).eval()
#         a2rm_toks[mode] = tok
#         a2rm_models[mode] = model
#         print(f"[Load A2RM] mode={mode} ckpt={ckpt}")
#         return tok, model
#
#     # resolve audio for group (a-modes)
#     def resolve_audio_for_group(g: pd.DataFrame) -> Tuple[str, str]:
#         stems = [str(x).strip() for x in g["generated_wav_name"].tolist() if str(x).strip()]
#         stems = list(dict.fromkeys(stems))
#         if (not stems) or (not os.path.isdir(audio_code_dir)):
#             return "", ""
#         stem = random.choice(stems)
#         p = pick_code_from_stem(audio_code_dir, stem)
#         if p is None:
#             return "", ""
#         codes = load_audio_tokens_any(p)
#         return format_audio_tokens(codes, level=args.audio_token_level), p
#
#     def make_eval_key(mode: str, group_key: str, audio_code_path: str) -> str:
#         s = f"{mode}|||{group_key}|||{audio_code_path}"
#         return hashlib.md5(s.encode("utf-8")).hexdigest()
#
#     # labeled items
#     def build_items(gg_all: pd.DataFrame, lbl: str, max_n: int) -> List[Dict[str, Any]]:
#         sub = gg_all[gg_all["label_canon"] == lbl].copy()
#         if len(sub) == 0:
#             return []
#         mids = list(dict.fromkeys(sub["motion_id"].tolist()))[:max_n]
#         use_csv_caption = ("motion_caption" in gg_all.columns)
#         items = []
#         for i, mid in enumerate(mids):
#             cap = ""
#             if use_csv_caption:
#                 ss = sub[sub["motion_id"] == mid]
#                 if "motion_caption" in ss.columns and len(ss) > 0:
#                     cap = str(ss["motion_caption"].iloc[0]).strip()
#             if not cap:
#                 p = vqvae_lookup(vq_by_stem, mid)
#                 if p is None:
#                     continue
#                 codes = load_motion_codes_from_vq(p)
#                 cap = caption_motion_codes(codes)
#             items.append({"id": f"{lbl}_{i}", "caption": cap})
#         return items
#
#     # output
#     run_tag = make_run_tag(args)
#     out_csv = os.path.join(args.out_dir, f"eval_gen_copeland_ranker_btl_{run_tag}.csv")
#     ensure_parent(out_csv)
#
#     fieldnames = [
#         "eval_key",
#         "mode",
#         "split",
#         "group_key",
#         "sayings",
#         "emotion",
#         "audio_code_path",
#
#         "num_gen",
#         "num_gold",
#         "num_silver",
#         "num_neg",
#
#         "k_eval",
#
#         # generation ability (Copeland on gen-vs-others)
#         "win_gen_vs_neg",
#         "win_gen_vs_silver",
#         "win_gen_vs_gold",
#         "gen_at3",
#
#         # ranker ability (BTL on labeled-only)
#         "ranker_ndcg_labeled",
#
#         # debug
#         "topk_orig_json",
#         "topk_types_json",
#     ]
#
#     exists = os.path.isfile(out_csv)
#     fcsv = open(out_csv, "a", encoding="utf-8", newline="")
#     writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
#     if not exists:
#         writer.writeheader()
#         fcsv.flush()
#         os.fsync(fcsv.fileno())
#
#     # ----------------------------
#     # main loop
#     # ----------------------------
#     for mode in modes:
#         tok, model = get_a2rm_model(mode)
#
#         for keys, g in tqdm(groups, desc=f"Eval[{mode}]"):
#             if args.group_by == "group_id":
#                 group_key = str(keys) if not isinstance(keys, tuple) else str(keys[0])
#                 sayings = str(g["sayings"].iloc[0])
#                 emotion = str(g["emotion"].iloc[0])
#             else:
#                 sayings, emotion = keys
#                 sayings = str(sayings)
#                 emotion = str(emotion)
#                 group_key = f"{sayings}|||{emotion}"
#
#             split = str(g["split"].iloc[0])
#
#             gg_all = g.copy()
#             gg_all["label_canon"] = gg_all["label"].apply(canon_label)
#             gg_all["motion_id"] = gg_all["raw_file_name"].apply(motion_id_from_raw)
#
#             gold_items = build_items(gg_all, "gold", args.max_gold)
#             silver_items = build_items(gg_all, "silver", args.max_silver)
#             neg_items = build_items(gg_all, "neg", args.max_neg)
#
#             # audio for a-modes
#             audio_text = ""
#             audio_code_path = ""
#             if "a" in mode:
#                 audio_text, audio_code_path = resolve_audio_for_group(g)
#
#             eval_key = make_eval_key(mode, group_key, audio_code_path)
#
#             query_payload = {
#                 "speaker_sayings": sayings,
#                 "speaker_emotion": emotion,
#                 "cond_mode": mode,
#                 "note": "In a dyadic conversation, rank LISTENER non-verbal responses to the SPEAKER's utterance.",
#             }
#
#             seed_int = int(hashlib.md5(f"{group_key}|||{mode}".encode("utf-8")).hexdigest()[:8], 16)
#
#             # ----------------------------
#             # (B) Ranker ability: labeled-only BTL -> nDCG(2/1/0)
#             # ----------------------------
#             cand_L, cid2orig_L, orig2type_L = build_uniform_candidates(
#                 gen_items=[],
#                 gold_items=gold_items,
#                 silver_items=silver_items,
#                 neg_items=neg_items,
#                 seed=seed_int ^ 0xBADC0DE,
#                 max_total=args.max_total_candidates,
#             )
#
#             ranker_ndcg_labeled = float("nan")
#             if len(cand_L) > 0:
#                 Nl = len(cand_L)
#                 pairs_L = sample_pairs_all(Nl, max_pairs=min(args.max_pairs_labeled, Nl * (Nl - 1) // 2), seed=seed_int ^ 0x2468ACE)
#                 order_L = btl_order_from_pairs_batched(
#                     qwen=qwen,
#                     query_payload=query_payload,
#                     candidates=cand_L,
#                     pairs=pairs_L,
#                     qwen_batch_size=args.qwen_batch_size,
#                     btl_l2=args.btl_l2,
#                     btl_lr=args.btl_lr,
#                     btl_steps=args.btl_steps,
#                 )
#                 ranked_orig_L = [cid2orig_L[cand_L[i]["cid"]] for i in order_L]
#                 ranked_types_L = [orig2type_L.get(oid, "neg") for oid in ranked_orig_L]
#                 kL = min(int(args.k_eval), len(ranked_types_L))
#                 if kL <= 0:
#                     kL = min(5, len(ranked_types_L))
#                 rels_L = [type_to_gain_graded_labeled(t) for t in ranked_types_L]
#                 ranker_ndcg_labeled = ndcg_at_k(rels_L, k=kL)
#
#             # ----------------------------
#             # Generate gen candidates
#             # ----------------------------
#             input_text = build_prompt_condmode(
#                 speaker_transcription=sayings,
#                 speaker_audio=audio_text,
#                 speaker_emotion=emotion,
#                 cond_mode=mode,
#             )
#             input_ids = tok(input_text, return_tensors="pt").input_ids.to(device, dtype=torch.long)
#
#             gen_items: List[Dict[str, Any]] = []
#             for ci in range(int(args.num_gen)):
#                 out = model.generate(
#                     input_ids,
#                     max_length=256,
#                     do_sample=True,
#                     temperature=args.gen_temperature,
#                     top_k=args.gen_top_k,
#                 )
#                 out_text = tok.decode(out[0], skip_special_tokens=False)
#                 out_text = out_text.replace("<pad>", "").replace("</s>", "").strip()
#                 codes = parse_motion_tokens(out_text, max_len=args.gen_max_len, codebook_size=512)
#                 if len(codes) == 0:
#                     codes = [1] * min(args.gen_max_len, 196)
#                 cap = caption_motion_codes(codes)
#                 gen_items.append({"id": f"gen_{ci}", "caption": cap})
#
#             # ----------------------------
#             # (A) Generation ability: full pool, gen-vs-others only, Copeland order
#             # ----------------------------
#             cand_all, cid2orig, orig2type = build_uniform_candidates(
#                 gen_items=gen_items,
#                 gold_items=gold_items,
#                 silver_items=silver_items,
#                 neg_items=neg_items,
#                 seed=seed_int,
#                 max_total=args.max_total_candidates,
#             )
#
#             k_eval = min(int(args.k_eval), len(cand_all))
#             if k_eval <= 0:
#                 k_eval = min(5, len(cand_all))
#
#             if len(cand_all) == 0:
#                 row = dict(
#                     eval_key=eval_key,
#                     mode=mode,
#                     split=split,
#                     group_key=group_key,
#                     sayings=sayings,
#                     emotion=emotion,
#                     audio_code_path=audio_code_path,
#                     num_gen=len(gen_items),
#                     num_gold=len(gold_items),
#                     num_silver=len(silver_items),
#                     num_neg=len(neg_items),
#                     k_eval=0,
#                     win_gen_vs_neg=float("nan"),
#                     win_gen_vs_silver=float("nan"),
#                     win_gen_vs_gold=float("nan"),
#                     gen_at3=float("nan"),
#                     ranker_ndcg_labeled=ranker_ndcg_labeled,
#                     topk_orig_json="[]",
#                     topk_types_json="[]",
#                 )
#                 writer.writerow(row)
#                 fcsv.flush()
#                 os.fsync(fcsv.fileno())
#                 continue
#
#             idx2type_all = [orig2type.get(cid2orig[c["cid"]], "neg") for c in cand_all]
#             pairs_full = sample_pairs_gen_only(
#                 idx2type=idx2type_all,
#                 max_pairs=min(args.max_pairs, len(cand_all) * (len(cand_all) - 1) // 2),
#                 seed=seed_int ^ 0xABCDEF01,
#                 quota_neg=args.quota_neg,
#                 quota_silver=args.quota_silver,
#                 quota_gold=args.quota_gold,
#             )
#
#             winners_full = judge_pairs_batched(
#                 qwen=qwen,
#                 query_payload=query_payload,
#                 candidates=cand_all,
#                 pairs=pairs_full,
#                 batch_size=args.qwen_batch_size,
#             )
#
#             scores_full, outcomes_full = copeland_from_batched_outcomes(len(cand_all), pairs_full, winners_full)
#             order_all = copeland_order(scores_full, seed_int=seed_int ^ 0x10203040)
#
#             _, win_gen_vs_neg, win_gen_vs_silver, win_gen_vs_gold = best_gen_winrates_from_outcomes(
#                 idx2type=idx2type_all,
#                 scores=scores_full,
#                 outcomes=outcomes_full,
#             )
#
#             gen_at3 = gen_at_k_from_order(order_all, idx2type_all, k=3)
#
#             ranked_orig = [cid2orig[cand_all[i]["cid"]] for i in order_all]
#             ranked_types = [orig2type.get(oid, "neg") for oid in ranked_orig]
#             topk_orig = ranked_orig[:k_eval]
#             topk_types = ranked_types[:k_eval]
#
#             row = dict(
#                 eval_key=eval_key,
#                 mode=mode,
#                 split=split,
#                 group_key=group_key,
#                 sayings=sayings,
#                 emotion=emotion,
#                 audio_code_path=audio_code_path,
#                 num_gen=len(gen_items),
#                 num_gold=len(gold_items),
#                 num_silver=len(silver_items),
#                 num_neg=len(neg_items),
#                 k_eval=k_eval,
#                 win_gen_vs_neg=win_gen_vs_neg,
#                 win_gen_vs_silver=win_gen_vs_silver,
#                 win_gen_vs_gold=win_gen_vs_gold,
#                 gen_at3=gen_at3,
#                 ranker_ndcg_labeled=ranker_ndcg_labeled,
#                 topk_orig_json=json.dumps(topk_orig, ensure_ascii=False),
#                 topk_types_json=json.dumps(topk_types, ensure_ascii=False),
#             )
#             writer.writerow(row)
#             fcsv.flush()
#             os.fsync(fcsv.fileno())
#
#     fcsv.close()
#     print("[Saved]", out_csv)
#
#
# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Eval script (A2RM) — revised per your requirements.

Changes vs previous:
1) Remove topk_orig_json column (also remove topk_types_json by default; keep if you want).
2) Generation ability (NO BTL): stable forced comparisons per gen
   - each gen compares vs ALL gold (usually 1)
   - each gen compares vs at least 2 silver (or all if <2)
   - each gen compares vs 3~5 neg (or all if <3)
   Outputs:
     - best_gen_win_vs_neg/silver/gold  (kept as win_gen_vs_* column names)
     - avg_gen_win_vs_neg/silver/gold
     - best_gen_idx (debug)
     - gen_at3 computed by Copeland on gen-vs-others outcomes (still NO BTL)

3) Ranker ability (WITH BTL): more efficient pair sampling on labeled-only pool
   - gold vs (silver + neg): cover all pairs
   - silver vs neg: cover all pairs
   - NO neg vs neg
   Then fit BTL as before and report ranker_ndcg_labeled (gain: gold=2, silver=1, neg=0)

vLLM batching:
- Uses qwen.generate_batch() with --qwen_batch_size

Notes:
- If any type is missing in a group, corresponding winrate becomes NaN (no comparisons).
- Forced gen-vs-others comparisons can be large if you have many neg; we cap neg per gen with --gen_vs_neg_k (default 5).

Example:
python eval_gen_forced_ranker_btl.py \
  --pairs_csv ./new_data/test.csv \
  --dataset_dir /ibex/project/c2191/luoc/dataset/A2R \
  --ckpt_map_json ./ckpt_map.json \
  --caption_ckpt /path/to/t5_caption_ckpt \
  --qwen_path /path/to/Qwen \
  --qwen_use_vllm \
  --qwen_batch_size 128
"""

import os, re, json, csv, math, argparse, random, hashlib
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
)

# -----------------------------
# Motion token parsing (A2RM output)
# -----------------------------
_MOTION_SPAN_RE = re.compile(r"<Motion Tokens>(.*?)</Motion Tokens>", re.DOTALL)
_MOTION_TOKEN_RE = re.compile(r"<Motion Token\s+(\d+)>")
_MOTION_TOKEN_SHORT_RE = re.compile(r"<(\d+)>")  # <123>

def parse_motion_tokens(text: str, max_len: int = 200, codebook_size: int = 512) -> List[int]:
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

# -----------------------------
# Prompt builder (MATCH TRAINING)
# -----------------------------
def build_prompt_condmode(
    speaker_transcription: str,
    speaker_audio: str,
    speaker_emotion: str,
    cond_mode: str,
) -> str:
    cm = (cond_mode or "").strip().lower()
    use_transcription = ("t" in cm)
    use_audio = ("a" in cm)
    use_emotion = (cm.endswith("+e") or cm in ("a+e", "t+e", "t+a+e"))

    t = (speaker_transcription or "").strip()
    a = (speaker_audio or "").strip()
    e = (speaker_emotion or "").strip()

    lines = []
    lines.append("You are modeling a speaker-listener dyadic interaction.\n\n")
    lines.append("Input:\n")
    lines.append(f"- SPEAKER_TRANSCRIPTION: {t if use_transcription else ''}\n")
    lines.append(f"- SPEAKER_AUDIO: {a if use_audio else ''}\n")
    if use_emotion and e:
        lines.append(f"- SPEAKER_EMOTION: <Emotion> {e} </Emotion>\n")
    lines.append("\nOutput:\n")
    lines.append("Return ONLY a sequence of listener motion tokens in the exact format:\n")
    lines.append("<Motion Tokens> <Motion Token i> ... </Motion Tokens>\n")
    lines.append("Do NOT output any other words.\n")
    return "".join(lines).strip()

# -----------------------------
# Caption model helpers (T5)
# -----------------------------
def build_caption_prompt(motion_codes: List[int]) -> str:
    motion_string = "<Motion Tokens>" + "".join([f"<{c}>" for c in motion_codes]) + "</Motion Tokens>"
    return "Generate text: " + motion_string

# -----------------------------
# Audio token formatting
# -----------------------------
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

def format_audio_tokens(a_tokens: np.ndarray, level: str = "base") -> str:
    arr = np.array(a_tokens)
    level = str(level)

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

def pick_code_from_stem(code_dir: str, stem: str) -> Optional[str]:
    stem = str(stem).strip()
    if not stem:
        return None
    p_npz = os.path.join(code_dir, stem + ".npz")
    if os.path.exists(p_npz):
        return p_npz
    p_npy = os.path.join(code_dir, stem + ".npy")
    if os.path.exists(p_npy):
        return p_npy
    return None

# -----------------------------
# VQ lookup for labeled motions
# -----------------------------
def motion_id_from_raw(raw_file_name: str) -> str:
    s = str(raw_file_name)
    mid = s.split("_", 1)[0]
    return str(mid).zfill(6)

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

def load_motion_codes_from_vq(vq_path: str, codebook_size: int = 512) -> List[int]:
    arr = np.load(vq_path, allow_pickle=False)
    arr = np.asarray(arr).reshape(-1).tolist()
    out = []
    for x in arr:
        try:
            c = int(x)
        except Exception:
            continue
        if 0 <= c < codebook_size:
            out.append(c)
    return out

# -----------------------------
# Candidate packing (C01..)
# -----------------------------
def build_uniform_candidates(
    gen_items: List[Dict[str, Any]],
    gold_items: List[Dict[str, Any]],
    silver_items: List[Dict[str, Any]],
    neg_items: List[Dict[str, Any]],
    seed: int,
    max_total: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, str], Dict[str, int]]:
    """
    Returns:
      candidates: [{"cid":"C01","caption":"..."}, ...]
      cid2orig: {"C01":"gen_0", ...}
      orig2type: {"gen_0":"gen","gold_0":"gold"...}
      orig2idx: {"gen_0": idx_in_candidates, ...} (after shuffle)
    """
    all_items = []
    orig2type: Dict[str, str] = {}

    def add(items, t):
        for it in items:
            oid = str(it["id"])
            all_items.append({"orig_id": oid, "caption": str(it["caption"])})
            orig2type[oid] = t

    add(gen_items, "gen")
    add(gold_items, "gold")
    add(silver_items, "silver")
    add(neg_items, "neg")

    rng = random.Random(int(seed))
    rng.shuffle(all_items)
    all_items = all_items[: max_total]

    cid2orig: Dict[str, str] = {}
    candidates: List[Dict[str, Any]] = []
    orig2idx: Dict[str, int] = {}

    for i, it in enumerate(all_items, start=1):
        cid = f"C{i:02d}"
        cid2orig[cid] = it["orig_id"]
        candidates.append({"cid": cid, "caption": it["caption"]})
        orig2idx[it["orig_id"]] = len(candidates) - 1

    return candidates, cid2orig, orig2type, orig2idx

# -----------------------------
# Pairwise Qwen judge parsing
# -----------------------------
def extract_last_json_object(text: str) -> Optional[Dict[str, Any]]:
    if text is None:
        return None
    s = str(text)
    starts = [i for i, ch in enumerate(s) if ch == "{"]  # keep all
    if not starts:
        return None

    last_obj = None
    for st in starts:
        depth = 0
        in_str = False
        esc = False
        for i in range(st, len(s)):
            c = s[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        blob = s[st : i + 1]
                        try:
                            obj = json.loads(blob)
                            last_obj = obj
                        except Exception:
                            pass
                        break
    return last_obj

_CID_RE = re.compile(r"\bC\d{2}\b")

def parse_winner_cid(raw: str, allowed: set) -> Optional[str]:
    obj = extract_last_json_object(raw) or {}
    for k in ["winner", "better", "choice", "answer", "selected"]:
        v = obj.get(k, None)
        if isinstance(v, str) and v in allowed:
            return v
    m = _CID_RE.findall(str(raw))
    for cid in m:
        if cid in allowed:
            return cid
    return None
#
# def build_qwen_pairwise_prompt(
#     query_payload: Dict[str, Any],
#     cand_a: Dict[str, Any],
#     cand_b: Dict[str, Any],
# ) -> str:
#     payload = {
#         "task": "Dyadic SPEAKER→LISTENER pairwise preference.",
#         "query": query_payload,
#         "candidates": [
#             {"cid": cand_a["cid"], "caption": cand_a["caption"]},
#             {"cid": cand_b["cid"], "caption": cand_b["caption"]},
#         ],
#         "rules": [
#             "You are comparing LISTENER reactions only (NOT speaker actions).",
#             "Pick the ONE candidate that is the more plausible listener non-verbal response to the speaker sayings/emotion.",
#             "Use ONLY the two captions as evidence; ids contain no label information.",
#             "Return ONLY JSON: {\"winner\": \"Cxx\"}. No other keys/text.",
#         ],
#     }
#     return (
#         "You are an expert evaluator for listener reactive motions.\n"
#         "IMPORTANT: Output MUST be a single valid JSON object and NOTHING ELSE.\n"
#         f"{json.dumps(payload, ensure_ascii=False)}"
#     )
#
import json
from typing import Dict, Any

def build_qwen_pairwise_prompt(
    query_payload: Dict[str, Any],
    cand_a: Dict[str, Any],
    cand_b: Dict[str, Any],
) -> str:
    system = (
        "You are an expert evaluator for LISTENER reactive motions in dyadic conversation.\n"
        "Your job: given the SPEAKER's sayings and optional emotion, pick which LISTENER motion caption is the more plausible non-verbal response.\n"
        "IMPORTANT: Output MUST be a single valid JSON object and NOTHING ELSE.\n"
    )

    few_shots = [
        {
            "task": "pairwise_preference",
            "query": {
                "sayings": "A call came through saying you’ve won a vacation for two. I was so not expecting that!",
                "emotion": "surprised (positive)"
            },
            "candidates": [
                {"cid": "C01", "caption": "a person moves their hand to their mouth then lowers it"},
                {"cid": "C02", "caption": "we have someone acting like a duck"},
            ],
            "decision": {"winner": "C01"},
            "why_brief": "Hand-to-mouth is a common surprised reaction; 'acting like a duck' is unrelated."
        },
        {
            "task": "pairwise_preference",
            "query": {
                "sayings": "A call came through saying you’ve won a vacation for two. I was so not expecting that!",
                "emotion": "surprised (positive)"
            },
            # swapped order here:
            "candidates": [
                {"cid": "C01", "caption": "a man bounces on his feet while waving enthusiastically with his right arm."},
                {"cid": "C02", "caption": "a person moves their hand to their mouth then lowers it"},
            ],
            # winner must switch to C02:
            "decision": {"winner": "C02"},
            "why_brief": "Bouncing/waving could be excitement, but hand-to-mouth is more directly aligned with sudden surprise."
        },
    ]

    rubric = {
        "rules": [
            "Compare LISTENER reactions only (NOT speaker actions).",
            "Pick the ONE candidate that is the more plausible listener non-verbal response to the speaker sayings/emotion.",
            "Use ONLY the two captions as evidence; ids contain no label information.",
            "Prefer reactions that match emotion + social context (e.g., surprise: hand-to-mouth, step back, cover face, brief freeze).",
            "Penalize unrelated/odd actions (animals, intoxicated fighting, random phone acting) unless clearly justified by the speaker context.",
            "If both are plausible, choose the one that is more specific, natural, and directly aligned with the utterance/emotion.",
            "Return ONLY JSON: {\"winner\": \"Cxx\"}. No other keys/text.",
        ]
    }

    payload = {
        "task": "Dyadic SPEAKER→LISTENER pairwise preference.",
        "few_shots": [
            {
                "query": ex["query"],
                "candidates": ex["candidates"],
                "output": ex["decision"],
                "note": ex["why_brief"],
            }
            for ex in few_shots
        ],
        "query": query_payload,
        "candidates": [
            {"cid": cand_a["cid"], "caption": cand_a["caption"]},
            {"cid": cand_b["cid"], "caption": cand_b["caption"]},
        ],
        **rubric,
    }

    return system + json.dumps(payload, ensure_ascii=False)


# -----------------------------
# Qwen Judge (vLLM batch + HF fallback)
# -----------------------------
class QwenJudge:
    def __init__(
        self,
        model_path: str,
        use_vllm: bool = True,
        tp: int = 1,
        gpu_mem_util: float = 0.90,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        max_model_len: int = 8192,
    ):
        self.model_path = model_path
        self.use_vllm = use_vllm
        self.tp = tp
        self.gpu_mem_util = gpu_mem_util
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_model_len = max_model_len
        self._mode = None

        if use_vllm:
            try:
                from vllm import LLM, SamplingParams  # type: ignore
                self._vllm_LLM = LLM(
                    model=model_path,
                    tensor_parallel_size=tp,
                    gpu_memory_utilization=gpu_mem_util,
                    trust_remote_code=True,
                    max_model_len=max_model_len,
                    enforce_eager=False,
                )
                self._vllm_SamplingParams = SamplingParams
                self._mode = "vllm"
            except Exception as e:
                print(f"[WARN] vLLM init failed, fallback to HF. err={e}")
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
    def generate_batch(self, prompts: List[str]) -> List[str]:
        if len(prompts) == 0:
            return []
        if self._mode == "vllm":
            sp = self._vllm_SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                top_p=1.0,
            )
            outs = self._vllm_LLM.generate(prompts, sp)
            return [o.outputs[0].text.strip() for o in outs]

        # HF fallback: loop
        outs = []
        for p in prompts:
            inputs = self.tok(p, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1e-6,
                top_p=1.0,
            )
            txt = self.tok.decode(out[0], skip_special_tokens=True).strip()
            outs.append(txt)
        return outs

# -----------------------------
# Batched pairwise judging
# -----------------------------
def judge_pairs_batched(
    qwen: QwenJudge,
    query_payload: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    pairs: List[Tuple[int, int]],
    batch_size: int = 128,
) -> List[Optional[int]]:
    """
    For each pair (i,j), return winner index: i or j. If parse failed => None.
    """
    if len(pairs) == 0:
        return []
    allowed = {c["cid"] for c in candidates}
    winners: List[Optional[int]] = [None] * len(pairs)

    for st in range(0, len(pairs), batch_size):
        ed = min(st + batch_size, len(pairs))
        chunk = pairs[st:ed]
        prompts = [build_qwen_pairwise_prompt(query_payload, candidates[i], candidates[j]) for (i, j) in chunk]
        outs = qwen.generate_batch(prompts)
        for k, raw in enumerate(outs):
            i, j = chunk[k]
            wcid = parse_winner_cid(raw, allowed=allowed)
            if wcid is None:
                winners[st + k] = None
            else:
                winners[st + k] = i if wcid == candidates[i]["cid"] else j
    return winners

# -----------------------------
# Generation ability (NO BTL): forced comparisons per gen
# -----------------------------
def build_forced_gen_pairs(
    idx2type: List[str],
    gen_idxs: List[int],
    gold_idxs: List[int],
    silver_idxs: List[int],
    neg_idxs: List[int],
    seed: int,
    silver_k: int = 2,
    neg_k_low: int = 3,
    neg_k_high: int = 5,
) -> List[Tuple[int, int]]:
    """
    For each gen:
      - vs ALL gold
      - vs at least silver_k silver (or all if fewer)
      - vs random K neg where K ~ Uniform[neg_k_low, neg_k_high] (clipped by available)
    Return unique unordered pairs (i<j).
    """
    rng = random.Random(seed)
    pairs = []
    used = set()

    for g in gen_idxs:
        # all gold
        for o in gold_idxs:
            i, j = (g, o) if g < o else (o, g)
            if i != j and (i, j) not in used:
                pairs.append((i, j)); used.add((i, j))

        # silver: at least silver_k
        if len(silver_idxs) > 0:
            if len(silver_idxs) <= silver_k:
                chosen_s = list(silver_idxs)
            else:
                chosen_s = rng.sample(silver_idxs, k=silver_k)
            for o in chosen_s:
                i, j = (g, o) if g < o else (o, g)
                if i != j and (i, j) not in used:
                    pairs.append((i, j)); used.add((i, j))

        # neg: 3~5
        if len(neg_idxs) > 0:
            kk = rng.randint(int(neg_k_low), int(neg_k_high))
            kk = min(kk, len(neg_idxs))
            chosen_n = list(neg_idxs) if kk == len(neg_idxs) else rng.sample(neg_idxs, k=kk)
            for o in chosen_n:
                i, j = (g, o) if g < o else (o, g)
                if i != j and (i, j) not in used:
                    pairs.append((i, j)); used.add((i, j))

    rng.shuffle(pairs)
    return pairs

def copeland_scores_from_outcomes(
    N: int,
    pairs: List[Tuple[int, int]],
    winners: List[Optional[int]],
) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """
    Copeland score (wins-losses). Also returns outcomes (i,j,y) for later stats.
    """
    scores = np.zeros(N, dtype=np.int32)
    outcomes: List[Tuple[int, int, int]] = []
    for (i, j), w in zip(pairs, winners):
        if w is None:
            continue
        if w == i:
            scores[i] += 1; scores[j] -= 1
            outcomes.append((i, j, 1))
        else:
            scores[i] -= 1; scores[j] += 1
            outcomes.append((i, j, 0))
    return scores, outcomes

def order_by_copeland(scores: np.ndarray, seed: int) -> List[int]:
    rng = np.random.RandomState(seed & 0xFFFFFFFF)
    noise = rng.uniform(-1e-3, 1e-3, size=len(scores))
    return np.argsort(-(scores.astype(np.float32) + noise)).tolist()

def winrate_for_gen(
    gen_idx: int,
    idx2type: List[str],
    outcomes: List[Tuple[int, int, int]],
) -> Dict[str, float]:
    """
    Empirical winrate for a specific gen against each type (neg/silver/gold),
    computed only from outcomes involving that gen.
    """
    win = {"neg": 0, "silver": 0, "gold": 0}
    tot = {"neg": 0, "silver": 0, "gold": 0}
    for i, j, y in outcomes:
        if gen_idx not in (i, j):
            continue
        other = j if gen_idx == i else i
        t = idx2type[other]
        if t not in tot:
            continue
        tot[t] += 1
        gen_wins = (y == 1 and gen_idx == i) or (y == 0 and gen_idx == j)
        if gen_wins:
            win[t] += 1

    def rate(t):
        return float(win[t] / tot[t]) if tot[t] > 0 else float("nan")

    return {f"vs_{t}": rate(t) for t in ["neg", "silver", "gold"]}

def avg_winrates_over_gens(
    gen_idxs: List[int],
    idx2type: List[str],
    outcomes: List[Tuple[int, int, int]],
) -> Dict[str, float]:
    """
    Average gen winrate vs each type: mean over gens (ignoring NaNs).
    """
    per = [winrate_for_gen(g, idx2type, outcomes) for g in gen_idxs]
    out = {}
    for t in ["neg", "silver", "gold"]:
        vals = [d[f"vs_{t}"] for d in per if not math.isnan(d[f"vs_{t}"])]
        out[f"avg_vs_{t}"] = float(np.mean(vals)) if len(vals) > 0 else float("nan")
    return out

def best_gen_by_copeland(
    gen_idxs: List[int],
    scores: np.ndarray,
) -> Optional[int]:
    if len(gen_idxs) == 0:
        return None
    return sorted(gen_idxs, key=lambda i: int(scores[i]), reverse=True)[0]

def gen_at3_from_order(order: List[int], idx2type: List[str]) -> float:
    top3 = order[:3]
    return 1.0 if any(idx2type[i] == "gen" for i in top3) else 0.0

# -----------------------------
# Ranker ability (BTL): efficient labeled pair set (no neg-neg)
# -----------------------------
def build_labeled_pairs_no_negneg(
    idx2type: List[str],
) -> List[Tuple[int, int]]:
    """
    Build all labeled pairs:
      - gold vs (silver+neg)
      - silver vs neg
      - NO neg vs neg
      - NO gold vs gold, NO silver vs silver (optional, usually unnecessary)
    Return all unique unordered pairs (i<j).
    """
    gold = [i for i, t in enumerate(idx2type) if t == "gold"]
    silver = [i for i, t in enumerate(idx2type) if t == "silver"]
    neg = [i for i, t in enumerate(idx2type) if t == "neg"]

    pairs = []
    used = set()

    # gold vs silver+neg
    for g in gold:
        for o in silver + neg:
            i, j = (g, o) if g < o else (o, g)
            if i != j and (i, j) not in used:
                pairs.append((i, j)); used.add((i, j))

    # silver vs neg
    for s in silver:
        for o in neg:
            i, j = (s, o) if s < o else (o, s)
            if i != j and (i, j) not in used:
                pairs.append((i, j)); used.add((i, j))

    return pairs

def fit_btl_scores(
    n_items: int,
    matches: List[Tuple[int, int, int]],  # (i, j, y) y=1 => i wins else j wins
    l2: float = 1e-2,
    lr: float = 0.2,
    steps: int = 600,
) -> np.ndarray:
    s = np.zeros(n_items, dtype=np.float64)
    if len(matches) == 0:
        return s

    for _ in range(steps):
        grad = np.zeros_like(s)
        for i, j, y in matches:
            d = s[i] - s[j]
            p = 1.0 / (1.0 + math.exp(-d))
            g = (p - y)
            grad[i] += g
            grad[j] -= g
        grad += l2 * s
        s -= lr * grad / max(1, len(matches))
        s -= s.mean()
    return s

def btl_order(scores: np.ndarray) -> List[int]:
    return np.argsort(-scores).tolist()

def type_to_gain_graded_labeled(t: str) -> int:
    if t == "gold":
        return 2
    if t == "silver":
        return 1
    return 0

def ndcg_at_k(relevances: List[int], k: int) -> float:
    rel = relevances[:k]
    dcg = 0.0
    for i, r in enumerate(rel, start=1):
        if r > 0:
            dcg += (2.0 ** r - 1.0) / math.log2(i + 1.0)

    ideal = sorted(relevances, reverse=True)[:k]
    idcg = 0.0
    for i, r in enumerate(ideal, start=1):
        if r > 0:
            idcg += (2.0 ** r - 1.0) / math.log2(i + 1.0)

    if idcg <= 1e-12:
        return 0.0
    return float(dcg / idcg)

# -----------------------------
# misc
# -----------------------------
def ensure_parent(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

def hash_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:10]

def stable_hash_str(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]

def make_run_tag(args: argparse.Namespace) -> str:
    parts = [
        f"numgen{args.num_gen}",
        f"keval{args.k_eval}",
        f"maxtotal{args.max_total_candidates}",
        f"negK{args.gen_vs_neg_k}",
        f"cap{stable_hash_str(args.caption_ckpt)}",
        f"qwen{stable_hash_str(args.qwen_path)}",
        f"ckmap{hash_file(args.ckpt_map_json)}",
    ]
    return "_".join(parts)

def canon_label(x: str) -> str:
    s = (x or "").strip().lower()
    if s in {"gold", "pos", "positive", "gt", "true", "1"}:
        return "gold"
    if s in {"silver"}:
        return "silver"
    if s in {"neg", "negative", "0"}:
        return "neg"
    return s

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--pairs_csv", type=str, required=True)
    ap.add_argument("--dataset_dir", type=str, required=True)
    ap.add_argument("--ckpt_map_json", type=str, required=True)

    ap.add_argument("--audio_code_dir", type=str, default=None)
    ap.add_argument("--audio_token_level", type=str, default="base", choices=["base", "all", "rand"])

    ap.add_argument("--caption_ckpt", type=str, required=True)

    # generation
    ap.add_argument("--num_gen", type=int, default=3)
    ap.add_argument("--gen_max_len", type=int, default=200)
    ap.add_argument("--gen_temperature", type=float, default=0.8)
    ap.add_argument("--gen_top_k", type=int, default=200)

    # forced gen-vs-neg cap
    ap.add_argument("--gen_vs_silver_k", type=int, default=2)
    ap.add_argument("--gen_vs_neg_k", type=int, default=5)   # we will use [3..gen_vs_neg_k]
    ap.add_argument("--gen_vs_neg_k_low", type=int, default=3)

    # candidate budgets
    ap.add_argument("--max_gold", type=int, default=1)
    ap.add_argument("--max_silver", type=int, default=8)
    ap.add_argument("--max_neg", type=int, default=12)
    ap.add_argument("--max_total_candidates", type=int, default=30)

    ap.add_argument("--only_split", type=str, default="test")  # test/val/train/all
    ap.add_argument("--group_by", type=str, default="group_id", choices=["group_id", "sayings_emotion"])
    ap.add_argument("--k_eval", type=int, default=10)

    # qwen
    ap.add_argument("--qwen_path", type=str, required=True)
    ap.add_argument("--qwen_use_vllm", action="store_true")
    ap.add_argument("--qwen_tp", type=int, default=1)
    ap.add_argument("--qwen_gpu_mem_util", type=float, default=0.90)
    ap.add_argument("--qwen_max_new_tokens", type=int, default=256)
    ap.add_argument("--qwen_max_model_len", type=int, default=8192)
    ap.add_argument("--qwen_batch_size", type=int, default=32)

    # BTL hyperparams
    ap.add_argument("--btl_steps", type=int, default=600)
    ap.add_argument("--btl_lr", type=float, default=0.2)
    ap.add_argument("--btl_l2", type=float, default=1e-2)

    ap.add_argument("--out_dir", type=str, default="./eval_gen_forced_ranker_btl_out")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Device]", device)

    # ckpt map
    with open(args.ckpt_map_json, "r", encoding="utf-8") as f:
        ckpt_map = json.load(f)

    modes = ["a", "a+e", "t", "t+e", "t+a", "t+a+e"]
    for m in modes:
        if m not in ckpt_map:
            raise RuntimeError(f"ckpt_map_json missing mode: {m}")

    # read pairs
    df = pd.read_csv(args.pairs_csv, encoding="utf-8")
    required_cols = ["sayings", "emotion", "label", "raw_file_name", "split", "generated_wav_name"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in pairs_csv: {missing}")

    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["sayings"] = df["sayings"].astype(str).fillna("")
    df["emotion"] = df["emotion"].astype(str).fillna("")
    df["raw_file_name"] = df["raw_file_name"].astype(str).fillna("")
    df["generated_wav_name"] = df["generated_wav_name"].astype(str).fillna("")
    df["split"] = df["split"].astype(str).str.lower().str.strip()

    if args.only_split != "all":
        df = df[df["split"] == args.only_split].copy()
    if len(df) == 0:
        raise RuntimeError(f"No rows for split={args.only_split}")

    # group
    if args.group_by == "group_id":
        if "group_id" not in df.columns:
            print("[WARN] group_by=group_id but no group_id column; fallback to sayings_emotion")
            args.group_by = "sayings_emotion"

    if args.group_by == "group_id":
        groups = list(df.groupby(["group_id"], dropna=False))
    else:
        groups = list(df.groupby(["sayings", "emotion"], dropna=False))
    print("[Groups]", len(groups), "group_by=", args.group_by)

    # VQ index
    motion_vq_dir = os.path.join(args.dataset_dir, "HumanML3D", "VQVAE")
    if not os.path.isdir(motion_vq_dir):
        raise RuntimeError(f"Missing motion_vq_dir: {motion_vq_dir}")
    vq_by_stem = build_vq_index(motion_vq_dir)
    print("[VQ] indexed:", len(vq_by_stem))

    # audio code dir
    audio_code_dir = args.audio_code_dir or os.path.join(args.dataset_dir, "audio-raws-09-01-2026-code")
    if not os.path.isdir(audio_code_dir):
        print(f"[WARN] audio_code_dir not found: {audio_code_dir} (a-modes will use empty audio)")

    # caption model
    cap_tok = T5Tokenizer.from_pretrained(args.caption_ckpt)
    cap_model = T5ForConditionalGeneration.from_pretrained(args.caption_ckpt).to(device).eval()

    @torch.no_grad()
    def caption_motion_codes(codes: List[int]) -> str:
        prompt = build_caption_prompt(codes)
        inp = cap_tok(prompt, return_tensors="pt").input_ids.to(device, dtype=torch.long)
        out = cap_model.generate(inp, max_length=200, num_beams=1, do_sample=False)
        txt = cap_tok.decode(out[0], skip_special_tokens=True).strip().strip('"')
        return txt

    # qwen
    qwen = QwenJudge(
        model_path=args.qwen_path,
        use_vllm=args.qwen_use_vllm,
        tp=args.qwen_tp,
        gpu_mem_util=args.qwen_gpu_mem_util,
        max_new_tokens=args.qwen_max_new_tokens,
        temperature=0.0,
        max_model_len=args.qwen_max_model_len,
    )
    print("[Qwen] backend=", ("vllm" if args.qwen_use_vllm else "hf"), "batch=", args.qwen_batch_size)

    # cache A2RM models per mode
    a2rm_models: Dict[str, Any] = {}
    a2rm_toks: Dict[str, Any] = {}

    def get_a2rm_model(mode: str):
        if mode in a2rm_models:
            return a2rm_toks[mode], a2rm_models[mode]
        ckpt = ckpt_map[mode]
        tok = T5Tokenizer.from_pretrained(ckpt)
        model = T5ForConditionalGeneration.from_pretrained(ckpt).to(device).eval()
        a2rm_toks[mode] = tok
        a2rm_models[mode] = model
        print(f"[Load A2RM] mode={mode} ckpt={ckpt}")
        return tok, model

    # resolve audio for group (a-modes)
    def resolve_audio_for_group(g: pd.DataFrame) -> Tuple[str, str]:
        stems = [str(x).strip() for x in g["generated_wav_name"].tolist() if str(x).strip()]
        stems = list(dict.fromkeys(stems))
        if (not stems) or (not os.path.isdir(audio_code_dir)):
            return "", ""
        stem = random.choice(stems)
        p = pick_code_from_stem(audio_code_dir, stem)
        if p is None:
            return "", ""
        codes = load_audio_tokens_any(p)
        return format_audio_tokens(codes, level=args.audio_token_level), p

    def make_eval_key(mode: str, group_key: str, audio_code_path: str) -> str:
        s = f"{mode}|||{group_key}|||{audio_code_path}"
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    # labeled items builder
    def build_items(gg_all: pd.DataFrame, lbl: str, max_n: int) -> List[Dict[str, Any]]:
        sub = gg_all[gg_all["label_canon"] == lbl].copy()
        if len(sub) == 0:
            return []
        mids = list(dict.fromkeys(sub["motion_id"].tolist()))[:max_n]
        use_csv_caption = ("motion_caption" in gg_all.columns)
        items = []
        for i, mid in enumerate(mids):
            cap = ""
            if use_csv_caption:
                ss = sub[sub["motion_id"] == mid]
                if "motion_caption" in ss.columns and len(ss) > 0:
                    cap = str(ss["motion_caption"].iloc[0]).strip()
            if not cap:
                p = vqvae_lookup(vq_by_stem, mid)
                if p is None:
                    continue
                codes = load_motion_codes_from_vq(p)
                cap = caption_motion_codes(codes)
            items.append({"id": f"{lbl}_{i}", "caption": cap})
        return items

    # output
    run_tag = make_run_tag(args)
    out_csv = os.path.join(args.out_dir, f"eval_gen_forced_ranker_btl_{run_tag}.csv")
    ensure_parent(out_csv)

    fieldnames = [
        "eval_key",
        "mode",
        "split",
        "group_key",
        "sayings",
        "emotion",
        "audio_code_path",

        "num_gen",
        "num_gold",
        "num_silver",
        "num_neg",

        "k_eval",

        # generation ability (NO BTL)
        "win_gen_vs_neg",
        "win_gen_vs_silver",
        "win_gen_vs_gold",
        "avg_gen_win_vs_neg",
        "avg_gen_win_vs_silver",
        "avg_gen_win_vs_gold",
        "best_gen_idx",
        "gen_at3",

        # ranker ability (BTL labeled-only)
        "ranker_ndcg_labeled",
    ]

    exists = os.path.isfile(out_csv)
    fcsv = open(out_csv, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
    if not exists:
        writer.writeheader()
        fcsv.flush()
        os.fsync(fcsv.fileno())

    # ----------------------------
    # main loop
    # ----------------------------
    for mode in modes:
        tok, model = get_a2rm_model(mode)

        for keys, g in tqdm(groups, desc=f"Eval[{mode}]"):
            if args.group_by == "group_id":
                group_key = str(keys) if not isinstance(keys, tuple) else str(keys[0])
                sayings = str(g["sayings"].iloc[0])
                emotion = str(g["emotion"].iloc[0])
            else:
                sayings, emotion = keys
                sayings = str(sayings)
                emotion = str(emotion)
                group_key = f"{sayings}|||{emotion}"

            split = str(g["split"].iloc[0])

            gg_all = g.copy()
            gg_all["label_canon"] = gg_all["label"].apply(canon_label)
            gg_all["motion_id"] = gg_all["raw_file_name"].apply(motion_id_from_raw)

            gold_items = build_items(gg_all, "gold", args.max_gold)
            silver_items = build_items(gg_all, "silver", args.max_silver)
            neg_items = build_items(gg_all, "neg", args.max_neg)

            # audio for a-modes
            audio_text = ""
            audio_code_path = ""
            if "a" in mode:
                audio_text, audio_code_path = resolve_audio_for_group(g)

            eval_key = make_eval_key(mode, group_key, audio_code_path)

            query_payload = {
                "speaker_sayings": sayings,
                "speaker_emotion": emotion,
                "cond_mode": mode,
                "note": "In a dyadic conversation, compare LISTENER non-verbal responses to the SPEAKER.",
            }

            seed_int = int(hashlib.md5(f"{group_key}|||{mode}".encode("utf-8")).hexdigest()[:8], 16)

            # ----------------------------
            # Ranker ability: labeled-only BTL with no neg-neg
            # ----------------------------
            cand_L, cid2orig_L, orig2type_L, _ = build_uniform_candidates(
                gen_items=[],
                gold_items=gold_items,
                silver_items=silver_items,
                neg_items=neg_items,
                seed=seed_int ^ 0xBADC0DE,
                max_total=args.max_total_candidates,
            )
            ranker_ndcg_labeled = float("nan")
            if len(cand_L) > 0:
                idx2type_L = [orig2type_L.get(cid2orig_L[c["cid"]], "neg") for c in cand_L]
                pairs_L = build_labeled_pairs_no_negneg(idx2type_L)

                winners_L = judge_pairs_batched(
                    qwen=qwen,
                    query_payload=query_payload,
                    candidates=cand_L,
                    pairs=pairs_L,
                    batch_size=args.qwen_batch_size,
                )

                matches = []
                for (i, j), w in zip(pairs_L, winners_L):
                    if w is None:
                        continue
                    y = 1 if w == i else 0
                    matches.append((i, j, y))

                scores_L = fit_btl_scores(
                    n_items=len(cand_L),
                    matches=matches,
                    l2=args.btl_l2,
                    lr=args.btl_lr,
                    steps=args.btl_steps,
                )
                order_L = btl_order(scores_L)
                ranked_types_L = [idx2type_L[i] for i in order_L]
                kL = min(int(args.k_eval), len(ranked_types_L))
                if kL <= 0:
                    kL = min(5, len(ranked_types_L))
                rels = [type_to_gain_graded_labeled(t) for t in ranked_types_L]
                ranker_ndcg_labeled = ndcg_at_k(rels, k=kL)

            # ----------------------------
            # Generate N gen candidates
            # ----------------------------
            input_text = build_prompt_condmode(
                speaker_transcription=sayings,
                speaker_audio=audio_text,
                speaker_emotion=emotion,
                cond_mode=mode,
            )
            input_ids = tok(input_text, return_tensors="pt").input_ids.to(device, dtype=torch.long)

            gen_items: List[Dict[str, Any]] = []
            for ci in range(int(args.num_gen)):
                out = model.generate(
                    input_ids,
                    max_length=256,
                    do_sample=True,
                    temperature=args.gen_temperature,
                    top_k=args.gen_top_k,
                )
                out_text = tok.decode(out[0], skip_special_tokens=False)
                out_text = out_text.replace("<pad>", "").replace("</s>", "").strip()
                codes = parse_motion_tokens(out_text, max_len=args.gen_max_len, codebook_size=512)
                if len(codes) == 0:
                    codes = [1] * min(args.gen_max_len, 196)
                cap = caption_motion_codes(codes)
                gen_items.append({"id": f"gen_{ci}", "caption": cap})

            # ----------------------------
            # Generation ability: forced comparisons per gen (NO BTL)
            # ----------------------------
            cand_all, cid2orig, orig2type, orig2idx = build_uniform_candidates(
                gen_items=gen_items,
                gold_items=gold_items,
                silver_items=silver_items,
                neg_items=neg_items,
                seed=seed_int ^ 0x13579BDF,
                max_total=args.max_total_candidates,
            )

            idx2type_all = [orig2type.get(cid2orig[c["cid"]], "neg") for c in cand_all]
            gen_idxs = [i for i, t in enumerate(idx2type_all) if t == "gen"]
            gold_idxs = [i for i, t in enumerate(idx2type_all) if t == "gold"]
            silver_idxs = [i for i, t in enumerate(idx2type_all) if t == "silver"]
            neg_idxs = [i for i, t in enumerate(idx2type_all) if t == "neg"]

            # handle empty pool
            if len(cand_all) == 0 or len(gen_idxs) == 0:
                row = dict(
                    eval_key=eval_key,
                    mode=mode,
                    split=split,
                    group_key=group_key,
                    sayings=sayings,
                    emotion=emotion,
                    audio_code_path=audio_code_path,
                    num_gen=len(gen_items),
                    num_gold=len(gold_items),
                    num_silver=len(silver_items),
                    num_neg=len(neg_items),
                    k_eval=min(int(args.k_eval), len(cand_all)) if len(cand_all) > 0 else 0,
                    win_gen_vs_neg=float("nan"),
                    win_gen_vs_silver=float("nan"),
                    win_gen_vs_gold=float("nan"),
                    avg_gen_win_vs_neg=float("nan"),
                    avg_gen_win_vs_silver=float("nan"),
                    avg_gen_win_vs_gold=float("nan"),
                    best_gen_idx=-1,
                    gen_at3=float("nan"),
                    ranker_ndcg_labeled=ranker_ndcg_labeled,
                )
                writer.writerow(row)
                fcsv.flush()
                os.fsync(fcsv.fileno())
                continue

            # forced pairs
            pairs_forced = build_forced_gen_pairs(
                idx2type=idx2type_all,
                gen_idxs=gen_idxs,
                gold_idxs=gold_idxs,
                silver_idxs=silver_idxs,
                neg_idxs=neg_idxs,
                seed=seed_int ^ 0xABCDEF01,
                silver_k=args.gen_vs_silver_k,
                neg_k_low=args.gen_vs_neg_k_low,
                neg_k_high=max(args.gen_vs_neg_k_low, args.gen_vs_neg_k),
            )

            winners_forced = judge_pairs_batched(
                qwen=qwen,
                query_payload=query_payload,
                candidates=cand_all,
                pairs=pairs_forced,
                batch_size=args.qwen_batch_size,
            )
            copeland_scores, outcomes = copeland_scores_from_outcomes(
                N=len(cand_all),
                pairs=pairs_forced,
                winners=winners_forced,
            )
            order_all = order_by_copeland(copeland_scores, seed=seed_int ^ 0x10203040)
            gen_at3 = gen_at3_from_order(order_all, idx2type_all)

            # avg winrates across gens
            avg_rates = avg_winrates_over_gens(gen_idxs, idx2type_all, outcomes)

            # best gen by Copeland
            best_gen = best_gen_by_copeland(gen_idxs, copeland_scores)
            if best_gen is None:
                best_idx = -1
                best_vs = {"vs_neg": float("nan"), "vs_silver": float("nan"), "vs_gold": float("nan")}
            else:
                best_idx = int(best_gen)  # index in candidate list
                best_vs = winrate_for_gen(best_gen, idx2type_all, outcomes)

            # output columns: keep your original names for best-gen
            row = dict(
                eval_key=eval_key,
                mode=mode,
                split=split,
                group_key=group_key,
                sayings=sayings,
                emotion=emotion,
                audio_code_path=audio_code_path,
                num_gen=len(gen_items),
                num_gold=len(gold_items),
                num_silver=len(silver_items),
                num_neg=len(neg_items),
                k_eval=min(int(args.k_eval), len(cand_all)),
                win_gen_vs_neg=best_vs["vs_neg"],
                win_gen_vs_silver=best_vs["vs_silver"],
                win_gen_vs_gold=best_vs["vs_gold"],
                avg_gen_win_vs_neg=avg_rates["avg_vs_neg"],
                avg_gen_win_vs_silver=avg_rates["avg_vs_silver"],
                avg_gen_win_vs_gold=avg_rates["avg_vs_gold"],
                best_gen_idx=best_idx,
                gen_at3=gen_at3,
                ranker_ndcg_labeled=ranker_ndcg_labeled,
            )
            writer.writerow(row)
            fcsv.flush()
            os.fsync(fcsv.fileno())

    fcsv.close()
    print("[Saved]", out_csv)


if __name__ == "__main__":
    main()
