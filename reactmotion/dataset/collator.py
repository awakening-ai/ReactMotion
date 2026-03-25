# -*- coding: utf-8 -*-
# reactmotion.dataset.collator
import random
import numpy as np
import torch
from functools import lru_cache
from typing import Dict, Any, List, Optional, Tuple

from reactmotion.dataset.mimi_encoder import MimiStreamingEncoder
from reactmotion.dataset.audio_aug import MixedAcousticAug


def ensure_2d_mono(w: torch.Tensor) -> torch.Tensor:
    if w.dim() == 3:
        w = w.squeeze(0)
    if w.dim() == 1:
        w = w.unsqueeze(0)
    if w.dim() != 2:
        raise ValueError(f"Wave shape {tuple(w.shape)} not 1/2/3D")
    if w.size(0) > 1:
        w = w.mean(dim=0, keepdim=True)
    return w


PROMPT_V3 = (
    "You are modeling a speaker-listener dyadic interaction.\n\n"
    "Input:\n"
    "- SPEAKER_TRANSCRIPTION (optional): <Transcription_Placeholder>\n"
    "- SPEAKER_AUDIO (optional): <Audio_Placeholder>\n"
    "- SPEAKER_EMOTION (optional): <Emotion_Placeholder>\n\n"
    "Output:\n"
    "Return ONLY a sequence of listener motion tokens in the exact format:\n"
    "<Motion Tokens> <Motion Token i> ... </Motion Tokens>\n"
    "Do NOT output any other words.\n"
)


class ReactMotionCollator:
    """
    Per group -> expanded:
      [gold_1..gold_K, silver_1, neg_1]

    ✅ NEW:
      - batch returns `group_weights`: [Ng] float32 (one weight per group)
    """

    def __init__(
        self,
        tokenizer,
        source_len: int,
        target_len: int,
        fixed_k_gold: int = 2,
        sample_gold: str = "random",   # random / first
        motion_codebook_size: int = 512,
        force_first_motion: bool = True,
        one_gold: bool = True,

        cond_mode: str = "t+e",
        audio_mode: str = "none",

        cond_dropout: float = 0.30,
        drop_e_only_when_multi: bool = False,

        audio_sr: int = 24000,
        mimi_codebooks: int = 8,
        mimi_cardinality: int = 2048,
        mimi_chunk_frames: int = 32,
        audio_token_level: str = "base",  # base | all | rand
        enable_audio_aug: bool = False,

        device: str = "cuda",
    ):
        self.tokenizer = tokenizer
        self.source_len = int(source_len)
        self.target_len = int(target_len)

        self.fixed_k_gold = int(fixed_k_gold)
        self.sample_gold = str(sample_gold)
        self.motion_codebook_size = int(motion_codebook_size)
        self.force_first_motion = bool(force_first_motion)
        self.one_gold = bool(one_gold)

        self.cond_mode = str(cond_mode)
        self.audio_mode = str(audio_mode)

        valid = {"t", "t+e", "a", "a+e", "t+a", "t+a+e"}
        if self.cond_mode not in valid:
            raise ValueError(f"cond_mode must be one of {sorted(valid)}, got {self.cond_mode}")
        if self.audio_mode not in ["none", "code", "wav"]:
            raise ValueError("audio_mode must be one of: none/code/wav")
        if self.sample_gold not in ["random", "first"]:
            raise ValueError("sample_gold must be random/first")
        if self.fixed_k_gold <= 0:
            raise ValueError("fixed_k_gold must be >=1")

        self.cond_dropout = float(cond_dropout)
        self.drop_e_only_when_multi = bool(drop_e_only_when_multi)

        self.audio_sr = int(audio_sr)
        self.mimi_codebooks = int(mimi_codebooks)
        self.mimi_cardinality = int(mimi_cardinality)
        self.mimi_chunk_frames = int(mimi_chunk_frames)
        assert audio_token_level in ("base", "all", "rand"), f"Invalid audio_token_level: {audio_token_level}"
        self.audio_token_level = str(audio_token_level)
        self.enable_audio_aug = bool(enable_audio_aug)
        self.device = str(device)

        self.mimi: Optional[MimiStreamingEncoder] = None
        self.aug: Optional[MixedAcousticAug] = None
        if self.audio_mode == "wav":
            self.mimi = MimiStreamingEncoder(device=self.device, codebooks=self.mimi_codebooks)
            self.aug = MixedAcousticAug(sr=self.audio_sr, device="cpu")

    def _sample_cond_mask(self) -> Tuple[bool, bool, bool]:
        avail_t = ("t" in self.cond_mode)
        avail_a = ("a" in self.cond_mode)
        avail_e = self.cond_mode.endswith("+e")

        use_t, use_a, use_e = avail_t, avail_a, avail_e
        n_avail = int(avail_t) + int(avail_a) + int(avail_e)

        if n_avail >= 2 and self.cond_dropout > 0:
            if avail_t and random.random() < self.cond_dropout:
                use_t = False
            if avail_a and random.random() < self.cond_dropout:
                use_a = False
            if avail_e:
                # drop_e_only_when_multi=True means: only allow dropping emotion when
                # there are >=3 total modalities (so after dropping e, >=2 remain).
                # For modes like t+e or a+e (n_avail=2), keep emotion.
                if (not self.drop_e_only_when_multi) or (n_avail >= 3):
                    if random.random() < self.cond_dropout:
                        use_e = False

        # forbid only-e
        if (not use_t) and (not use_a):
            if avail_t:
                use_t = True
            elif avail_a:
                use_a = True
            else:
                use_e = True

        if (not use_t) and (not use_a) and (not use_e):
            if avail_t:
                use_t = True
            elif avail_a:
                use_a = True
            else:
                use_e = True

        return use_t, use_a, use_e

    @lru_cache(maxsize=8192)
    def _load_motion_tokens_cached(self, vq_path: str) -> np.ndarray:
        arr = np.load(vq_path, allow_pickle=False, mmap_mode="r")
        if arr.ndim == 1:
            return np.array(arr, dtype=np.int64)
        return np.array(arr[0], dtype=np.int64)

    def _load_motion_tokens(self, vq_path: str) -> np.ndarray:
        if self.force_first_motion:
            return self._load_motion_tokens_cached(vq_path)
        arr = np.load(vq_path, allow_pickle=False)
        if arr.ndim == 1:
            return arr.astype(np.int64)
        return np.array(random.choice(arr), dtype=np.int64)

    def _motion_tokens_to_text(self, m_tokens: np.ndarray) -> str:
        parts = ["<Motion Tokens>"]
        for token in m_tokens.reshape(-1):
            t = int(token)
            if 0 <= t < self.motion_codebook_size:
                parts.append(f"<Motion Token {t}>")
        parts.append("</Motion Tokens>")
        return " ".join(parts)

    def _codes_to_audio_text(self, codes: np.ndarray) -> str:
        arr = np.array(codes)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[0] > arr.shape[-1] and arr.shape[-1] == self.mimi_codebooks:
            arr = arr.transpose(1, 0)

        L = min(arr.shape[0], self.mimi_codebooks)
        parts = ["<Audio Tokens>"]

        def _emit_level(lv: int):
            for tok in arr[lv].reshape(-1):
                t = int(tok)
                if 0 <= t < self.mimi_cardinality:
                    parts.append(f"<Audio Level {lv} Token {t}>")

        if self.audio_token_level == "base":
            _emit_level(0)
        elif self.audio_token_level == "all":
            for lv in range(L):
                _emit_level(lv)
        elif self.audio_token_level == "rand":
            lv_num = random.randint(1, L)
            for lv in range(lv_num):
                _emit_level(lv)

        parts.append("</Audio Tokens>")
        return " ".join(parts)

    @torch.no_grad()
    def _encode_wavpaths_to_audio_text(self, wav_paths: List[str]) -> List[str]:
        import torchaudio
        if self.mimi is None:
            raise RuntimeError("audio_mode=wav but mimi encoder is not initialized")

        wavs = []
        for wp in wav_paths:
            w, sr = torchaudio.load(wp)
            w = ensure_2d_mono(w.to(torch.float32).cpu())
            if sr != self.audio_sr:
                w = torchaudio.functional.resample(w, sr, self.audio_sr)
            if self.enable_audio_aug and (self.aug is not None) and (random.random() < 0.5):
                w = self.aug(w)
            wavs.append(w)

        codes_list, _ = self.mimi.encode_many_concat(
            [w.to(self.device, non_blocking=True) for w in wavs],
            chunk_frames=self.mimi_chunk_frames,
            return_latent=False,
        )
        return [self._codes_to_audio_text(c.cpu().numpy()) for c in codes_list]

    def _format_emotion(self, emotion: str) -> str:
        emotion = (emotion or "").strip()
        if not emotion:
            return ""
        return f"<Emotion> {emotion} </Emotion>"

    def _build_source_text(
        self,
        transcription: str,
        emotion: str,
        audio_text: str,
        use_t: bool,
        use_a: bool,
        use_e: bool,
    ) -> str:
        trans = (transcription or "").strip() if use_t else ""
        emo = self._format_emotion(emotion) if use_e else ""
        aud = (audio_text or "").strip() if use_a else ""

        s = PROMPT_V3
        s = s.replace("<Transcription_Placeholder>", trans)
        s = s.replace("<Emotion_Placeholder>", emo)
        s = s.replace("<Audio_Placeholder>", aud)
        return s.strip()

    def _choose_gold_paths(self, gold_paths: List[str]) -> List[str]:
        if len(gold_paths) == 0:
            raise RuntimeError("Group has 0 gold (filter in dataset min_gold>=1).")

        if self.one_gold:
            if self.sample_gold == "first":
                return [gold_paths[0]]
            return [random.choice(gold_paths)]

        K = self.fixed_k_gold
        if self.sample_gold == "first":
            return [gold_paths[0] for _ in range(K)]

        if len(gold_paths) >= K:
            tmp = gold_paths[:]
            random.shuffle(tmp)
            return tmp[:K]
        return [random.choice(gold_paths) for _ in range(K)]

    def __call__(self, features: List[Dict[str, Any]]):
        # 1) audio_text per group
        if self.audio_mode == "none":
            audio_text_list = ["" for _ in features]
        elif self.audio_mode == "code":
            audio_text_list = []
            for f in features:
                at = str(f.get("audio_text", "")).strip()
                if not at:
                    raise RuntimeError("audio_mode=code but feature missing audio_text")
                audio_text_list.append(at)
        else:  # wav
            wav_paths = []
            for f in features:
                wp = str(f.get("wav_path", "")).strip()
                if not wp:
                    raise RuntimeError("audio_mode=wav but feature missing wav_path")
                wav_paths.append(wp)
            audio_text_list = self._encode_wavpaths_to_audio_text(wav_paths)

        expanded_sources: List[str] = []
        expanded_targets: List[str] = []
        group_sizes_gold: List[int] = []
        group_weights: List[float] = []   # ✅ NEW

        # 2) expand each group
        for gi, f in enumerate(features):
            gold_paths = list(f.get("gold_vq_paths", []))
            silver_paths = list(f.get("silver_vq_paths", []))
            neg_paths = list(f.get("neg_vq_paths", []))

            if len(gold_paths) == 0:
                raise RuntimeError("Group has 0 gold (filter in dataset min_gold>=1).")
            if len(silver_paths) == 0:
                silver_paths = gold_paths
            if len(neg_paths) == 0:
                neg_paths = gold_paths

            chosen_gold = self._choose_gold_paths(gold_paths)
            K = len(chosen_gold)
            group_sizes_gold.append(K)

            # ✅ NEW: per-group weight
            w = float(f.get("group_w", 1.0))
            group_weights.append(w)

            use_t, use_a, use_e = self._sample_cond_mask()
            src = self._build_source_text(
                transcription=f.get("transcription", ""),
                emotion=f.get("emotion", ""),
                audio_text=audio_text_list[gi],
                use_t=use_t,
                use_a=use_a,
                use_e=use_e,
            )

            # gold
            for p in chosen_gold:
                m = self._load_motion_tokens(p)
                expanded_sources.append(src)
                expanded_targets.append(self._motion_tokens_to_text(m))

            # silver
            s_path = random.choice(silver_paths)
            m = self._load_motion_tokens(s_path)
            expanded_sources.append(src)
            expanded_targets.append(self._motion_tokens_to_text(m))

            # neg
            n_path = random.choice(neg_paths)
            m = self._load_motion_tokens(n_path)
            expanded_sources.append(src)
            expanded_targets.append(self._motion_tokens_to_text(m))

        # 3) tokenize
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
        enc["group_weights"] = torch.tensor(group_weights, dtype=torch.float32)  # ✅ NEW
        return enc
