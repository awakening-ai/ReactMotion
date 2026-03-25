import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import pandas as pd
import os

import torchaudio
import torchaudio.functional as AF
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download
from moshi.models import loaders  # pip install moshiko-pytorch
import soundfile as sf          # needed for sf.read
import tempfile                 # TemporaryDirectory
import subprocess               # subprocess.run
from pathlib import Path        # Path(td) / "tmp.wav"
from typing import Optional, Tuple  # Optional[int], Tuple[Tensor,int]

import math, tempfile, subprocess
from pathlib import Path
from typing import List, Optional, Tuple

os.environ["NO_CUDA_GRAPH"] = "1"  # disable Mimi CUDA graph capture
torch.backends.cudnn.benchmark = True  # allow auto-selection of faster algorithms
torch.set_float32_matmul_precision("high")  # helps some ops, negligible side effects


def ensure_2d_mono(w: torch.Tensor) -> torch.Tensor:
    """Force to [1, T]."""
    if w.dim() == 3:
        w = w.squeeze(0)  # [B,C,T] -> [C,T]
    if w.dim() == 1:
        w = w.unsqueeze(0)  # [T] -> [1,T]
    if w.dim() != 2:
        raise ValueError(f"Wave shape {tuple(w.shape)} is not 1/2/3D")
    if w.size(0) > 1:
        w = w.mean(dim=0, keepdim=True)
    return w


# ================= Mixed acoustic augmentation (CPU, dataset only) =================

class MixedAcousticAug:
    """
    Random mix of lightweight, semantics-preserving acoustic augmentations.
    """

    def __init__(self, sr: int,
                 p_noise=0.35, p_rir=0.25, p_band=0.25, p_codec=0.00,  # codec off by default (slow)
                 p_speed=0.20, p_f0=0.20, p_loud=0.30,
                 p_tilt=0.20, p_timbre=0.30, p_exciter=0.10,
                 device="cpu"):
        self.sr = sr
        self.device = device
        self.p_noise = p_noise
        self.p_rir = p_rir
        self.p_band = p_band
        self.p_codec = p_codec
        self.p_speed = p_speed
        self.p_f0 = p_f0
        self.p_loud = p_loud
        self.p_tilt = p_tilt
        self.p_timbre = p_timbre
        self.p_exciter = p_exciter

    # basic filters
    def _lp(self, w, cutoff):
        return AF.lowpass_biquad(w, self.sr, float(cutoff))

    def _hp(self, w, cutoff):
        return AF.highpass_biquad(w, self.sr, float(cutoff))

    # atoms
    def add_noise(self, w, snr=None):
        snr_db = snr if snr is not None else random.uniform(18, 28)
        noise = torch.randn_like(w)
        s_rms = w.pow(2).mean().sqrt()
        n_rms = noise.pow(2).mean().sqrt() + 1e-8
        noise = noise * (s_rms / (10 ** (snr_db / 20))) / n_rms
        return (w + noise).clamp(-1, 1)

    def rir_smallroom(self, w):
        cut = float(random.uniform(3500.0, 7000.0))
        w_lp = self._lp(w, cut)
        shift = int(round(0.02 * self.sr))
        early = 0.1 * torch.roll(w_lp, shifts=shift, dims=-1) if 0 < shift < w.size(-1) else torch.zeros_like(w_lp)
        return (0.9 * w_lp + early).clamp(-1, 1)

    def bandlimit(self, w):
        t = random.choice(["tel", "bright", "warm"])
        if t == "tel":   return self._lp(self._hp(w, 300), 3400)
        if t == "bright": return self._hp(w, 120)
        return self._lp(w, 5500)

    def time_stretch_small(self, w):
        if w.shape[-1] < 2400:  # <0.1s at 24k
            return w
        factor = random.uniform(0.98, 1.02)
        n_fft, hop = 1024, 256
        window = torch.hann_window(n_fft, device=w.device)

        # skip very short segments (avoid boundary effects)
        if w.shape[-1] < 2 * hop:
            return w

        # if too short, pad to at least one frame
        need_trim = False
        if w.shape[-1] < n_fft:
            w = torch.nn.functional.pad(w, (0, n_fft - w.shape[-1]))
            need_trim = True
        orig_len = w.shape[-1]

        spec = torch.stft(w, n_fft=n_fft, hop_length=hop, window=window,
                          center=True, return_complex=True)
        n_freq = spec.size(-2)
        phase_adv = torch.linspace(0, math.pi * hop, n_freq, device=w.device).unsqueeze(-1)
        stretched = AF.phase_vocoder(spec, factor, phase_adv)

        # omit length to avoid warnings; we manually align length afterwards
        out = torch.istft(stretched, n_fft=n_fft, hop_length=hop, window=window, center=True)

        # align to original length
        if out.shape[-1] >= orig_len:
            out = out[..., :orig_len]
        else:
            out = torch.nn.functional.pad(out, (0, orig_len - out.shape[-1]))

        # if we padded earlier to meet n_fft, ensure we don't exceed bounds
        if need_trim:
            out = out[..., :orig_len]

        return out.clamp(-1, 1)

    def f0_shift_small(self, w):
        if w.shape[-1] < 2400:
            return w
        st = random.uniform(-1.0, 1.0)  # ±1 semitone
        fac = 2 ** (st / 12)
        up = AF.resample(w, self.sr, int(self.sr * fac))
        out = AF.resample(up, int(self.sr * fac), self.sr)
        return out[..., :w.shape[-1]]

    def loudness(self, w):
        db = random.uniform(-2.0, 2.0)
        return (w * (10 ** (db / 20))).clamp(-1, 1)

    def spectral_tilt(self, w):
        return self._hp(w, 120) if random.random() < 0.5 else self._lp(w, 5500)

    def peq(self, w, cf, g, q=1.0):
        return AF.equalizer_biquad(w, self.sr, float(cf), float(g), float(q))

    def timbre_multi_eq(self, w):
        choice = random.choice(["bright", "warm", "nasal", "chesty"])
        if choice == "bright":
            w = self.peq(w, 4000, +3.0, 0.9);
            w = self.peq(w, 8000, +2.0, 0.9);
            w = self.peq(w, 300, -1.0, 1.2)
        elif choice == "warm":
            w = self.peq(w, 200, +2.0, 0.9);
            w = self.peq(w, 6000, -2.0, 0.9)
        elif choice == "nasal":
            w = self.peq(w, 1100, +2.0, 1.0);
            w = self.peq(w, 120, -1.0, 1.2)
        else:
            w = self.peq(w, 180, +2.0, 1.0);
            w = self.peq(w, 7000, -1.0, 1.2)
        return w

    def exciter(self, w):
        mix = random.uniform(0.1, 0.2)
        hp = self._hp(w, 2000)
        wet = torch.tanh(1.5 * hp)
        return ((1 - mix) * w + mix * wet).clamp(-1, 1)

    def __call__(self, wav_1t: torch.Tensor) -> torch.Tensor:
        w = wav_1t
        ops = []
        if random.random() < 0.35: ops.append(self.add_noise)
        if random.random() < 0.25: ops.append(self.rir_smallroom)
        if random.random() < 0.25: ops.append(self.bandlimit)
        if random.random() < 0.20: ops.append(self.time_stretch_small)
        if random.random() < 0.20: ops.append(self.f0_shift_small)
        if random.random() < 0.30: ops.append(self.loudness)
        if random.random() < 0.20: ops.append(self.spectral_tilt)
        if random.random() < 0.30: ops.append(self.timbre_multi_eq)
        if random.random() < 0.10: ops.append(self.exciter)

        if not ops:
            return w
        random.shuffle(ops)
        for f in ops[:3]:
            w = f(ensure_2d_mono(w))
        return ensure_2d_mono(w)