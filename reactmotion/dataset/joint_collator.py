#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JointCollator: extends ReactMotionCollator to handle mixed batches
of ReactMotion and Text-to-Motion (T2M) samples.

T2M samples:
  - Use K=1 gold, dummy silver/neg (same motion code)
  - No audio, no modality dropout
  - Prompt: reformulated caption as SPEAKER_TRANSCRIPTION (already done in dataset)
  - Flagged via is_t2m for loss routing in trainer
"""

import random
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch

from reactmotion.dataset.collator import ReactMotionCollator, PROMPT_V3


class JointCollator(ReactMotionCollator):
    """
    Handles mixed batches containing both ReactMotion and T2M samples.

    T2M samples are identified by `is_t2m=True` in the feature dict.
    For T2M: always K=1 gold, same motion as silver/neg (dummy), text-only.
    """

    def __call__(self, features: List[Dict[str, Any]]):
        # Separate into T2M and ReactMotion features, preserving original indices
        rm_features = []
        t2m_features = []
        is_t2m_flags = []

        for f in features:
            if f.get("is_t2m", False):
                t2m_features.append(f)
                is_t2m_flags.append(True)
            else:
                rm_features.append(f)
                is_t2m_flags.append(False)

        # Process in order: ReactMotion first, then T2M
        # We'll reconstruct the original order mapping later

        # --- Audio text for ReactMotion features ---
        if self.audio_mode == "none":
            rm_audio_text_list = ["" for _ in rm_features]
        elif self.audio_mode == "code":
            rm_audio_text_list = []
            for f in rm_features:
                at = str(f.get("audio_text", "")).strip()
                if not at:
                    raise RuntimeError("audio_mode=code but feature missing audio_text")
                rm_audio_text_list.append(at)
        else:  # wav
            wav_paths = []
            for f in rm_features:
                wp = str(f.get("wav_path", "")).strip()
                if not wp:
                    raise RuntimeError("audio_mode=wav but feature missing wav_path")
                wav_paths.append(wp)
            rm_audio_text_list = self._encode_wavpaths_to_audio_text(wav_paths)

        expanded_sources: List[str] = []
        expanded_targets: List[str] = []
        group_sizes_gold: List[int] = []
        group_weights: List[float] = []
        group_is_t2m: List[bool] = []

        # --- Process ReactMotion features ---
        for gi, f in enumerate(rm_features):
            gold_paths = list(f.get("gold_vq_paths", []))
            silver_paths = list(f.get("silver_vq_paths", []))
            neg_paths = list(f.get("neg_vq_paths", []))

            if len(gold_paths) == 0:
                raise RuntimeError("Group has 0 gold.")
            if len(silver_paths) == 0:
                silver_paths = gold_paths
            if len(neg_paths) == 0:
                neg_paths = gold_paths

            chosen_gold = self._choose_gold_paths(gold_paths)
            K = len(chosen_gold)
            group_sizes_gold.append(K)
            group_weights.append(float(f.get("group_w", 1.0)))
            group_is_t2m.append(False)

            use_t, use_a, use_e = self._sample_cond_mask()
            src = self._build_source_text(
                transcription=f.get("transcription", ""),
                emotion=f.get("emotion", ""),
                audio_text=rm_audio_text_list[gi],
                use_t=use_t,
                use_a=use_a,
                use_e=use_e,
            )

            for p in chosen_gold:
                m = self._load_motion_tokens(p)
                expanded_sources.append(src)
                expanded_targets.append(self._motion_tokens_to_text(m))

            s_path = random.choice(silver_paths)
            m = self._load_motion_tokens(s_path)
            expanded_sources.append(src)
            expanded_targets.append(self._motion_tokens_to_text(m))

            n_path = random.choice(neg_paths)
            m = self._load_motion_tokens(n_path)
            expanded_sources.append(src)
            expanded_targets.append(self._motion_tokens_to_text(m))

        # --- Process T2M features ---
        for f in t2m_features:
            gold_paths = list(f.get("gold_vq_paths", []))
            if len(gold_paths) == 0:
                raise RuntimeError("T2M group has 0 gold.")

            vq_path = gold_paths[0]
            K = 1
            group_sizes_gold.append(K)
            group_weights.append(float(f.get("group_w", 1.0)))
            group_is_t2m.append(True)

            # T2M: text-only, no audio, no emotion, no modality dropout
            src = self._build_source_text(
                transcription=f.get("transcription", ""),
                emotion="",
                audio_text="",
                use_t=True,
                use_a=False,
                use_e=False,
            )

            # Gold
            m = self._load_motion_tokens(vq_path)
            expanded_sources.append(src)
            expanded_targets.append(self._motion_tokens_to_text(m))

            # Dummy silver (same motion)
            expanded_sources.append(src)
            expanded_targets.append(self._motion_tokens_to_text(m))

            # Dummy neg (same motion)
            expanded_sources.append(src)
            expanded_targets.append(self._motion_tokens_to_text(m))

        # --- Tokenize ---
        enc = self.tokenizer(
            expanded_sources,
            padding=True,
            truncation=True,
            max_length=self.source_len,
            return_tensors="pt",
        )
        lab = self.tokenizer(
            expanded_targets,
            padding=True,
            truncation=True,
            max_length=self.target_len,
            return_tensors="pt",
        )["input_ids"]

        lab = lab.clone()
        lab[lab == self.tokenizer.pad_token_id] = -100

        enc["labels"] = lab
        enc["group_sizes_gold"] = torch.tensor(group_sizes_gold, dtype=torch.long)
        enc["group_weights"] = torch.tensor(group_weights, dtype=torch.float32)
        enc["is_t2m_mask"] = torch.tensor(group_is_t2m, dtype=torch.bool)
        return enc
